from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, utc_now  # noqa: E402
from nfl_ats.source_policy import require_acquisition  # noqa: E402

SOURCE_ID = "internet_archive_pfr_boxscores"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
ORIGINAL_URL_TEMPLATE = "https://www.pro-football-reference.com/boxscores/{pfr_id}.htm"
CDX_URL_TEMPLATE = (
    "https://web.archive.org/cdx/search/cdx"
    "?url={original}&output=json&filter=statuscode:200&from={not_before}"
)
REPLAY_URL_TEMPLATE = "https://web.archive.org/web/{ts}id_/{original}"

MIN_DELAY_SECONDS = 8.0
DEFAULT_INITIAL_BACKOFF_SECONDS = 60.0
DEFAULT_MAX_REQUEST_RETRIES = 5
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_FALLBACK_CAPTURES = 2
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

DEFAULT_RAW_ROOT = REPO / "data" / "raw" / "officials_pfr_wayback"
DEFAULT_PROCESSED_ROOT = REPO / "data" / "processed" / "officials_pfr_wayback"

OFFICIALS_PARQUET_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "effective_time",
    "home_team",
    "away_team",
    "pfr_id",
    "position",
    "official_name",
    "source",
    "wayback_capture_timestamp",
    "wayback_url",
    "fetched_at_utc",
]


@dataclass
class RateLimiter:
    delay_seconds: float
    sleep_fn: Callable[[float], None] = time.sleep
    _last_call: float | None = field(default=None, init=False, repr=False)

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = max(self.delay_seconds - elapsed, 0.0)
            if remaining > 0:
                self.sleep_fn(remaining)
        self._last_call = time.monotonic()


@dataclass(frozen=True)
class FetchResult:
    content: bytes | None
    status_code: int | None
    error: str | None


FetchFn = Callable[[str], FetchResult]


def build_fetch_fn(user_agent: str = USER_AGENT, timeout: float = 90.0) -> FetchFn:
    session = requests.Session()
    session.headers["User-Agent"] = user_agent

    def fetch(url: str) -> FetchResult:
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            return FetchResult(None, None, f"{type(exc).__name__}: {exc}")
        if response.status_code == 200:
            return FetchResult(response.content, 200, None)
        return FetchResult(None, response.status_code, f"http_{response.status_code}")

    return fetch


@dataclass(frozen=True)
class RequestOutcome:
    content: bytes | None
    status_code: int | None
    error: str | None
    attempts: int
    backoff_schedule_seconds: tuple[float, ...]
    gave_up_after_retries: bool


def fetch_with_backoff(
    url: str,
    fetch_fn: FetchFn,
    limiter: RateLimiter,
    *,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_attempts: int = DEFAULT_MAX_REQUEST_RETRIES,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RequestOutcome:

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    backoff_schedule: list[float] = []
    backoff = initial_backoff_seconds
    result = FetchResult(None, None, "unreached")
    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        result = fetch_fn(url)
        if result.status_code == 200:
            return RequestOutcome(
                content=result.content,
                status_code=200,
                error=None,
                attempts=attempt,
                backoff_schedule_seconds=tuple(backoff_schedule),
                gave_up_after_retries=False,
            )
        retryable = result.status_code in RETRYABLE_STATUS_CODES
        if not retryable or attempt == max_attempts:
            return RequestOutcome(
                content=None,
                status_code=result.status_code,
                error=result.error,
                attempts=attempt,
                backoff_schedule_seconds=tuple(backoff_schedule),
                gave_up_after_retries=retryable,
            )
        sleep_fn(backoff)
        backoff_schedule.append(backoff)
        backoff *= 2.0
    raise AssertionError("fetch_with_backoff exhausted its loop without returning")


def newest_schedule_snapshot(repo: Path = REPO) -> Path:
    candidates = sorted((repo / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise SystemExit(f"no data/raw/*/schedules.parquet snapshot found under {repo}")
    return candidates[-1]


def load_games(schedule_path: Path, *, season_start: int, season_end: int) -> pd.DataFrame:
    columns = ["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "pfr"]
    frame = pd.read_parquet(schedule_path, columns=columns)
    frame = frame.loc[
        (frame["season"] >= season_start)
        & (frame["season"] <= season_end)
        & (frame["game_type"] == "REG")
    ].copy()
    frame = frame.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    missing_pfr = frame["pfr"].isna().sum()
    if missing_pfr:
        frame = frame.loc[frame["pfr"].notna()].reset_index(drop=True)
    return frame


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_tags(fragment: str) -> str:
    text = _TAG_RE.sub(" ", fragment)
    text = text.replace("&amp;", "&").replace("&nbsp;", " ")
    return _WS_RE.sub(" ", text).strip()


_OFFICIALS_TABLE_RE = re.compile(
    r'<table[^>]*\bid=["\'](?:officials|ref_info)["\'][^>]*>(.*?)</table>',
    re.IGNORECASE | re.DOTALL,
)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)

_INLINE_OFFICIALS_RE = re.compile(
    r"Officials:\s*(.*?)(?:</div>|<br\s*/?>|$)", re.IGNORECASE | re.DOTALL
)
_INLINE_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z .]*?):\s*([^:,]+?)(?:,\s*(?=[A-Za-z][A-Za-z .]*?:)|$)"
)


def _parse_officials_table(html: str) -> list[tuple[str, str]]:
    table_match = _OFFICIALS_TABLE_RE.search(html)
    if table_match is None:
        return []
    rows: list[tuple[str, str]] = []
    for row_match in _ROW_RE.finditer(table_match.group(1)):
        cells = [strip_tags(c) for c in _CELL_RE.findall(row_match.group(1))]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        position, name = cells[0], cells[1]
        if position.lower() in {"position", "official"}:
            continue
        rows.append((position, name))
    return rows


def _parse_officials_inline(html: str) -> list[tuple[str, str]]:
    match = _INLINE_OFFICIALS_RE.search(html)
    if match is None:
        return []
    segment = strip_tags(match.group(1))
    rows: list[tuple[str, str]] = []
    for pair_match in _INLINE_PAIR_RE.finditer(segment):
        position = pair_match.group(1).strip()
        name = pair_match.group(2).strip().strip(",").strip()
        if position and name:
            rows.append((position, name))
    return rows


def parse_officials_block(html: str) -> tuple[list[tuple[str, str]], list[str]]:

    rows = _parse_officials_table(html)
    if rows:
        return rows, []
    rows = _parse_officials_inline(html)
    if rows:
        return rows, ["matched via inline Officials: line, not a table id=officials"]
    return [], ["no officials block found (neither table nor inline strategy matched)"]


def load_existing_manifest(snapshot_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        return {"games": []}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def already_fetched_pfr_ids(manifest: dict[str, Any], snapshot_dir: Path) -> set[str]:
    fetched: set[str] = set()
    for row in manifest.get("games", []):
        pfr_id = row.get("pfr_id")
        html_file = row.get("html_file")
        if pfr_id and html_file and (snapshot_dir / html_file).exists():
            fetched.add(str(pfr_id))
    return fetched


@dataclass
class SweepConfig:
    season_start: int
    season_end: int
    run_id: str
    raw_root: Path = DEFAULT_RAW_ROOT
    processed_root: Path = DEFAULT_PROCESSED_ROOT
    schedule_path: Path | None = None
    delay_seconds: float = MIN_DELAY_SECONDS
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    limit: int | None = None
    fallback_captures: int = DEFAULT_FALLBACK_CAPTURES
    retry_unparsed: bool = False


def run_sweep(
    config: SweepConfig,
    *,
    fetch_fn: FetchFn | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> dict[str, Any]:
    if config.delay_seconds < MIN_DELAY_SECONDS:
        raise SystemExit(
            f"--delay-seconds must be >= {MIN_DELAY_SECONDS} (politeness floor for this source)"
        )
    require_acquisition(SOURCE_ID)

    schedule_path = config.schedule_path or newest_schedule_snapshot()
    games = load_games(
        schedule_path, season_start=config.season_start, season_end=config.season_end
    )

    snapshot_dir = config.raw_root / config.run_id
    html_dir = snapshot_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_existing_manifest(snapshot_dir)
    manifest_rows: list[dict[str, Any]] = list(manifest.get("games", []))
    already_fetched = already_fetched_pfr_ids(manifest, snapshot_dir)

    fetch = fetch_fn if fetch_fn is not None else build_fetch_fn()
    limiter = RateLimiter(config.delay_seconds, sleep_fn=sleep_fn)

    officials_rows: list[dict[str, Any]] = []
    per_season_attempted: dict[int, int] = {}
    per_season_fetched: dict[int, int] = {}
    per_season_parsed_rows: dict[int, int] = {}
    per_season_html_on_disk: dict[int, int] = {}
    consecutive_failures = 0
    total_requests = 0
    new_fetch_count = 0
    fallback_replay_fetches = 0
    games_retried_unparsed = 0
    games_retry_no_new_capture = 0
    stopped_early = False
    stop_reason: str | None = None

    for game in games.itertuples(index=False):
        pfr_id = str(game.pfr)
        season = int(game.season)
        existing = _find_manifest_row(manifest_rows, pfr_id)
        previous_attempts: list[dict[str, Any]] = []

        if pfr_id in already_fetched and existing is not None:
            html_path = snapshot_dir / str(existing["html_file"])
            html_text = html_path.read_text(encoding="utf-8", errors="replace")
            rows, warnings = parse_officials_block(html_text)
            existing["officials_parsed"] = len(rows)
            existing["parse_warnings"] = warnings
            if rows or not config.retry_unparsed:
                per_season_parsed_rows[season] = per_season_parsed_rows.get(season, 0) + len(rows)
                per_season_html_on_disk[season] = per_season_html_on_disk.get(season, 0) + 1
                for position, name in rows:
                    officials_rows.append(_officials_row(game, position, name, source_row=existing))
                continue
            games_retried_unparsed += 1
            previous_attempts = _previous_attempts(existing)

        if config.limit is not None and new_fetch_count >= config.limit:
            stopped_early = True
            stop_reason = "limit_reached"
            continue

        moment = now or datetime.now(UTC)
        fetch_instant = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
        original_url = ORIGINAL_URL_TEMPLATE.format(pfr_id=pfr_id)
        not_before = capture_not_before(game.gameday)
        cdx_url = CDX_URL_TEMPLATE.format(original=original_url, not_before=not_before)

        per_season_attempted[season] = per_season_attempted.get(season, 0) + 1
        new_fetch_count += 1

        cdx_outcome = fetch_with_backoff(
            cdx_url,
            fetch,
            limiter,
            initial_backoff_seconds=config.initial_backoff_seconds,
            max_attempts=config.max_request_retries,
            sleep_fn=sleep_fn,
        )
        total_requests += cdx_outcome.attempts

        row: dict[str, Any] = {
            "game_id": game.game_id,
            "season": season,
            "week": int(game.week),
            "pfr_id": pfr_id,
            "gameday": str(game.gameday),
            "home_team": game.home_team,
            "away_team": game.away_team,
            "original_url": original_url,
            "cdx_url": cdx_url,
            "fetch_instant_utc": fetch_instant,
            "cdx_status_code": cdx_outcome.status_code,
            "cdx_attempts": cdx_outcome.attempts,
            "cdx_backoff_schedule_seconds": list(cdx_outcome.backoff_schedule_seconds),
            "cdx_error": cdx_outcome.error,
            "wayback_capture_timestamp": None,
            "wayback_url": None,
            "replay_status_code": None,
            "replay_attempts": 0,
            "replay_backoff_schedule_seconds": [],
            "replay_error": None,
            "html_file": None,
            "outcome": None,
            "officials_parsed": 0,
            "parse_warnings": [],
            "attempted_captures": list(previous_attempts),
            "fallback_fetches": 0,
            "retried_unparsed": bool(previous_attempts),
        }

        if cdx_outcome.content is None:
            if existing is not None and existing.get("html_file"):
                row = _keep_existing_page(existing, row)
            row["outcome"] = "cdx_fetch_failed"
            consecutive_failures += 1
            _upsert_manifest_row(manifest_rows, row)
            _write_manifest(snapshot_dir, manifest_rows, config)
            if consecutive_failures >= config.max_consecutive_failures:
                stopped_early = True
                stop_reason = (
                    f"hard_stop_after_{config.max_consecutive_failures}_consecutive_failures"
                )
                break
            continue

        already_tried = {str(a.get("wayback_capture_timestamp")) for a in previous_attempts}
        candidates = [
            ts
            for ts in _rank_capture_timestamps(cdx_outcome.content, not_before=not_before)
            if ts not in already_tried
        ]
        if not candidates:
            consecutive_failures = 0
            if existing is not None and existing.get("html_file"):
                row = _keep_existing_page(existing, row)
                row["outcome"] = "fetched"
                row["retry_note"] = "no post-game capture beyond those already fetched"
                games_retry_no_new_capture += 1
                per_season_html_on_disk[season] = per_season_html_on_disk.get(season, 0) + 1
            else:
                row["outcome"] = "no_capture_found"
            _upsert_manifest_row(manifest_rows, row)
            _write_manifest(snapshot_dir, manifest_rows, config)
            continue

        chosen: dict[str, Any] | None = None
        last_fetched: dict[str, Any] | None = None
        parsed_rows: list[tuple[str, str]] = []
        replay_failed = False
        for index, capture_ts in enumerate(candidates[: 1 + config.fallback_captures]):
            wayback_url = REPLAY_URL_TEMPLATE.format(ts=capture_ts, original=original_url)
            replay_outcome = fetch_with_backoff(
                wayback_url,
                fetch,
                limiter,
                initial_backoff_seconds=config.initial_backoff_seconds,
                max_attempts=config.max_request_retries,
                sleep_fn=sleep_fn,
            )
            total_requests += replay_outcome.attempts
            if index > 0:
                fallback_replay_fetches += 1
                row["fallback_fetches"] += 1
            attempt: dict[str, Any] = {
                "wayback_capture_timestamp": capture_ts,
                "wayback_url": wayback_url,
                "replay_status_code": replay_outcome.status_code,
                "replay_attempts": replay_outcome.attempts,
                "replay_backoff_schedule_seconds": list(replay_outcome.backoff_schedule_seconds),
                "replay_error": replay_outcome.error,
                "html_file": None,
                "officials_parsed": 0,
                "parse_warnings": [],
                "fetch_instant_utc": fetch_instant,
            }
            row["attempted_captures"].append(attempt)
            if replay_outcome.content is None:
                replay_failed = True
                break
            html_text = replay_outcome.content.decode("utf-8", errors="replace")
            html_path = html_dir / f"{pfr_id}__{capture_ts}.html"
            html_path.write_text(html_text, encoding="utf-8")
            attempt["html_file"] = f"html/{pfr_id}__{capture_ts}.html"
            rows, warnings = parse_officials_block(html_text)
            attempt["officials_parsed"] = len(rows)
            attempt["parse_warnings"] = warnings
            last_fetched = attempt
            if rows:
                chosen = attempt
                parsed_rows = rows
                break

        if replay_failed:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        page = chosen or last_fetched
        if page is None:
            if existing is not None and existing.get("html_file"):
                row = _keep_existing_page(existing, row)
                row["outcome"] = "fetched"
                row["retry_note"] = "newer capture fetch failed; kept the page already on disk"
                per_season_html_on_disk[season] = per_season_html_on_disk.get(season, 0) + 1
            else:
                failed = row["attempted_captures"][-1]
                for key in _REPLAY_KEYS:
                    row[key] = failed[key]
                row["outcome"] = "replay_fetch_failed"
        else:
            for key in (*_REPLAY_KEYS, "html_file", "officials_parsed", "parse_warnings"):
                row[key] = page[key]
            row["outcome"] = "fetched"
            per_season_fetched[season] = per_season_fetched.get(season, 0) + 1
            per_season_parsed_rows[season] = per_season_parsed_rows.get(season, 0) + len(
                parsed_rows
            )
            per_season_html_on_disk[season] = per_season_html_on_disk.get(season, 0) + 1

        _upsert_manifest_row(manifest_rows, row)
        _write_manifest(snapshot_dir, manifest_rows, config)

        if replay_failed and consecutive_failures >= config.max_consecutive_failures:
            stopped_early = True
            stop_reason = f"hard_stop_after_{config.max_consecutive_failures}_consecutive_failures"
            break

        for position, name in parsed_rows:
            officials_rows.append(_officials_row(game, position, name, source_row=row))

    officials_frame = pd.DataFrame(officials_rows, columns=OFFICIALS_PARQUET_COLUMNS)
    processed_dir = config.processed_root / config.run_id
    parquet_path = processed_dir / "officials_2009_2014.parquet"
    atomic_parquet(officials_frame, parquet_path)
    stamp_sidecar(
        parquet_path,
        {
            "row_count": len(officials_frame),
            "run_id": config.run_id,
            "season_start": config.season_start,
            "season_end": config.season_end,
            "source": SOURCE_ID,
        },
        project_root=REPO,
    )

    summary = {
        "run_id": config.run_id,
        "season_start": config.season_start,
        "season_end": config.season_end,
        "games_in_window": len(games),
        "games_already_on_disk": len(already_fetched),
        "new_fetch_attempts": new_fetch_count,
        "total_http_requests": total_requests,
        "games_fetched_ok": sum(per_season_fetched.values()),
        "games_cdx_or_replay_failed": sum(
            1
            for r in manifest_rows
            if r.get("outcome") in ("cdx_fetch_failed", "replay_fetch_failed")
        ),
        "games_no_capture_found": sum(
            1 for r in manifest_rows if r.get("outcome") == "no_capture_found"
        ),
        "officials_rows_parsed": len(officials_frame),
        "games_parsed_zero": sum(
            1
            for r in manifest_rows
            if r.get("outcome") == "fetched" and not r.get("officials_parsed")
        ),
        "fallback_replay_fetches": fallback_replay_fetches,
        "games_retried_unparsed": games_retried_unparsed,
        "games_retry_no_new_capture": games_retry_no_new_capture,
        "per_season_attempted": per_season_attempted,
        "per_season_fetched": per_season_fetched,
        "per_season_parsed_rows": per_season_parsed_rows,
        "per_season_html_on_disk": per_season_html_on_disk,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "raw_snapshot_dir": str(snapshot_dir),
        "processed_parquet": str(parquet_path),
    }
    _write_manifest(snapshot_dir, manifest_rows, config, summary=summary)
    return summary


_REPLAY_KEYS = (
    "wayback_capture_timestamp",
    "wayback_url",
    "replay_status_code",
    "replay_attempts",
    "replay_backoff_schedule_seconds",
    "replay_error",
)


def _find_manifest_row(manifest_rows: list[dict[str, Any]], pfr_id: str) -> dict[str, Any] | None:
    for row in manifest_rows:
        if row.get("pfr_id") == pfr_id:
            return row
    return None


def _upsert_manifest_row(manifest_rows: list[dict[str, Any]], row: dict[str, Any]) -> None:

    for index, existing in enumerate(manifest_rows):
        if existing.get("pfr_id") == row.get("pfr_id"):
            manifest_rows[index] = row
            return
    manifest_rows.append(row)


def _previous_attempts(existing: dict[str, Any]) -> list[dict[str, Any]]:

    attempts = list(existing.get("attempted_captures") or [])
    if not attempts and existing.get("wayback_capture_timestamp"):
        attempts = [
            {
                **{key: existing.get(key) for key in _REPLAY_KEYS},
                "html_file": existing.get("html_file"),
                "officials_parsed": existing.get("officials_parsed", 0),
                "parse_warnings": list(existing.get("parse_warnings") or []),
                "fetch_instant_utc": existing.get("fetch_instant_utc"),
            }
        ]
    return [{**attempt, "from_previous_run": True} for attempt in attempts]


def _keep_existing_page(existing: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:

    kept = dict(row)
    for key in (*_REPLAY_KEYS, "html_file", "officials_parsed", "parse_warnings"):
        kept[key] = existing.get(key)
    kept["officials_parsed"] = int(existing.get("officials_parsed") or 0)
    kept["parse_warnings"] = list(existing.get("parse_warnings") or [])
    return kept


def _officials_row(
    game: Any, position: str, name: str, *, source_row: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "season": int(game.season),
        "week": int(game.week),
        "game_id": game.game_id,
        "game_date": str(game.gameday),
        "effective_time": str(game.gameday),
        "home_team": game.home_team,
        "away_team": game.away_team,
        "pfr_id": str(game.pfr),
        "position": position,
        "official_name": name,
        "source": SOURCE_ID,
        "wayback_capture_timestamp": (source_row or {}).get("wayback_capture_timestamp"),
        "wayback_url": (source_row or {}).get("wayback_url"),
        "fetched_at_utc": (source_row or {}).get("fetch_instant_utc"),
    }


def capture_not_before(gameday: Any) -> str:

    day = pd.Timestamp(gameday).normalize() + pd.Timedelta(days=1)
    return day.strftime("%Y%m%d")


def _rank_capture_timestamps(cdx_json_bytes: bytes, *, not_before: str | None = None) -> list[str]:

    try:
        rows = json.loads(cdx_json_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return []
    if not isinstance(rows, list):
        return []
    data_rows = rows[1:] if rows and rows[0] and rows[0][0] == "urlkey" else rows
    candidates = {str(r[1]) for r in data_rows if isinstance(r, list) and len(r) > 1}
    if not_before is not None:
        candidates = {ts for ts in candidates if ts[:8] >= not_before}
    return sorted(candidates, reverse=True)


def _select_capture_timestamp(
    cdx_json_bytes: bytes, *, not_before: str | None = None
) -> str | None:

    ranked = _rank_capture_timestamps(cdx_json_bytes, not_before=not_before)
    return ranked[0] if ranked else None


def _write_manifest(
    snapshot_dir: Path,
    games: list[dict[str, Any]],
    config: SweepConfig,
    *,
    summary: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema": "officials_pfr_wayback_manifest/1",
        "run_id": config.run_id,
        "source": SOURCE_ID,
        "user_agent": USER_AGENT,
        "delay_seconds": config.delay_seconds,
        "initial_backoff_seconds": config.initial_backoff_seconds,
        "max_request_retries": config.max_request_retries,
        "max_consecutive_failures": config.max_consecutive_failures,
        "fallback_captures": config.fallback_captures,
        "capture_policy": "newest_post_game_capture_with_fallback",
        "season_start": config.season_start,
        "season_end": config.season_end,
        "updated_at_utc": utc_now(),
        "games": games,
    }
    if summary is not None:
        payload["summary"] = summary
    atomic_json(payload, snapshot_dir / "manifest.json")


def run_id_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season-start", type=int, default=2009)
    parser.add_argument("--season-end", type=int, default=2014)
    parser.add_argument("--run-id", default=None, metavar="YYYYMMDDTHHMMSSZ")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--schedule-snapshot", type=Path, default=None)
    parser.add_argument("--delay-seconds", type=float, default=MIN_DELAY_SECONDS)
    parser.add_argument(
        "--initial-backoff-seconds", type=float, default=DEFAULT_INITIAL_BACKOFF_SECONDS
    )
    parser.add_argument("--max-request-retries", type=int, default=DEFAULT_MAX_REQUEST_RETRIES)
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap NEW fetch attempts this invocation"
    )
    parser.add_argument(
        "--fallback-captures",
        type=int,
        default=DEFAULT_FALLBACK_CAPTURES,
        help=(
            "after the newest post-game capture parses 0 officials, try up to this many "
            "further captures (newest first) before recording the game as parsed-0"
        ),
    )
    parser.add_argument(
        "--retry-unparsed",
        action="store_true",
        help=(
            "on a resumed run, re-attempt games whose manifest row parsed 0 officials: "
            "re-parse the on-disk page first, then fetch newer captures only if still 0"
        ),
    )
    args = parser.parse_args(argv)
    if args.fallback_captures < 0:
        parser.error("--fallback-captures must be >= 0")

    config = SweepConfig(
        season_start=args.season_start,
        season_end=args.season_end,
        run_id=args.run_id or run_id_now(),
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        schedule_path=args.schedule_snapshot,
        delay_seconds=args.delay_seconds,
        initial_backoff_seconds=args.initial_backoff_seconds,
        max_request_retries=args.max_request_retries,
        max_consecutive_failures=args.max_consecutive_failures,
        limit=args.limit,
        fallback_captures=args.fallback_captures,
        retry_unparsed=args.retry_unparsed,
    )
    summary = run_sweep(config)
    print(
        f"run_id={summary['run_id']} window={summary['season_start']}-{summary['season_end']} "
        f"games_in_window={summary['games_in_window']} "
        f"already_on_disk={summary['games_already_on_disk']} "
        f"new_fetch_attempts={summary['new_fetch_attempts']} "
        f"fetched_ok={summary['games_fetched_ok']} "
        f"failed={summary['games_cdx_or_replay_failed']} "
        f"no_capture={summary['games_no_capture_found']} "
        f"total_http_requests={summary['total_http_requests']} "
        f"officials_rows={summary['officials_rows_parsed']} "
        f"parsed_zero={summary['games_parsed_zero']} "
        f"fallback_fetches={summary['fallback_replay_fetches']} "
        f"retried_unparsed={summary['games_retried_unparsed']} "
        f"stopped_early={summary['stopped_early']} stop_reason={summary['stop_reason']}"
    )
    return (
        1
        if summary["stopped_early"]
        and summary["stop_reason"]
        and "hard_stop" in summary["stop_reason"]
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
