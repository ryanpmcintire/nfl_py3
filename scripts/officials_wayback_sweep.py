"""Polite, resumable Wayback sweep for 2009-2014 PFR officiating-crew boxscores.

LEAD-59 (``ROADMAP.md``): the local nflverse officials feed
(``data/raw/officials/*/officials.parquet``) only covers 2015-2025 (measured,
``docs/referee_battery.md``). ``schedules.parquet`` already carries a single
head-referee name for every season back to 2009 (``referee`` column), but the
FULL seven-person crew (Umpire, Head Linesman/Down Judge, Line Judge, Field
Judge, Side Judge, Back Judge, Replay Official) for 2009-2014 exists only on
Pro-Football-Reference's own boxscore pages -- direct fetches of those pages
draw an HTTP 403 bot wall (measured 2026-09-03/04, ``docs/officials_archive_probe.md``),
and a single 2026-09-03/04 Wayback replay attempt of the same page drew an
HTTP 429 throttle from prior heavy crawling. This script is the "polite
Wayback pass" that probe document named as the unblock path: it never
touches pro-football-reference.com directly, only ``web.archive.org``.

Two-step fetch per game, both against ``web.archive.org`` only (the network
exception this lane was granted): a **CDX API lookup**
(``web.archive.org/cdx/search/cdx``, already used by
``scripts/pilot_vegasinsider_wayback.py`` for the same purpose) substitutes
for the "Wayback availability API" named in this lane's task brief -- the
literal ``archive.org/wayback/available`` endpoint lives on a different host
(``archive.org``, not ``web.archive.org``), which the granted exception does
not cover; CDX answers the identical question ("does an archived capture of
this URL exist, and at what timestamp") from a host the exception does
cover. The second step replays the chosen capture at
``web.archive.org/web/<ts>id_/<original>`` (the raw, unrewritten endpoint,
same convention as ``pilot_vegasinsider_wayback.py``).

Politeness contract (binding, not a default anyone should relax without a
new measurement): at least ``--delay-seconds`` (floor 8.0, enforced) between
every single HTTP request -- CDX and replay calls both, including retries;
exponential backoff starting at ``--initial-backoff-seconds`` (default 60s,
doubling each retry) on any 429 or 5xx response, up to
``--max-request-retries`` attempts per call; and a hard stop, mid-sweep, the
moment ``--max-consecutive-failures`` (default 5) GAMES in a row fail to
produce a usable HTTP 200 for either the CDX lookup or the replay fetch
after their own retries are exhausted. A "no capture exists for this game"
CDX result (a clean 200 response whose JSON simply lists zero rows) is a
content outcome, not a fetch failure, and does NOT count toward the
consecutive-failure counter -- only throttle/error responses do.

Resumable by construction: every fetched replay page is written to
``<raw-root>/<run-id>/html/<pfr_id>__<capture_ts>.html`` (runs before
2026-09-07 wrote ``html/<pfr_id>.html``; the manifest row's ``html_file``
is authoritative either way) and recorded in
``<raw-root>/<run-id>/manifest.json`` as it happens (not buffered to the
end), so a killed or `Ctrl+C`-interrupted run, or a fresh invocation given
the same ``--run-id``, skips every game whose manifest row names an HTML
file that exists on disk -- zero new network requests for already-fetched
games (unless ``--retry-unparsed`` is passed and the page parses 0, see
below). Passing a new ``--run-id`` (or omitting it) starts a fresh snapshot
directory instead.

Two directories, per this fleet's data-hygiene rule (raw captures never
mutated in place, derived tables never live under ``data/raw``):

- ``data/raw/officials_pfr_wayback/<run-id>/`` -- the immutable capture:
  every fetched HTML byte-for-byte, plus ``manifest.json`` (one row per
  game: original/wayback URL, CDX capture timestamp, fetch instant, HTTP
  status, retry/backoff counts, parse outcome).
- ``data/processed/officials_pfr_wayback/<run-id>/officials_2009_2014.parquet``
  -- the parsed crew-assignment rows, one per (game, position), with a
  ``stamp_sidecar()`` provenance JSON beside it (ENG-38 convention: a
  derived tabular write, not an experiment, so ``write_experiment_artifact``
  would misrepresent it).

Capture-selection policy (2026-09-07, lane N, after the 2009-2013 run
``20260907T175420Z`` parsed 0 officials on 418 of 418 fetched pages): the
EARLIEST post-game capture is the wrong one to take. PFR added the
officials block to its boxscore layout years after the 2009-2011 games were
played (the 2009-2010 pages captured 2009-2011 carry no ``game_info`` /
``ref_info`` block at all; see ``docs/officials_archive_probe.md``), and even
in 2014 a capture taken one or two days after kickoff can predate the
block. The sweep therefore prefers the NEWEST capture after the game
(CDX ``from=<day after game>`` keeps the pre-game placeholder pages out and
the query carries no ``limit``, so the full post-game list is ranked
client-side), and if the chosen capture parses zero
officials it falls back through up to ``--fallback-captures`` (default 2)
further post-game captures, newest first, before recording the game as
parsed-0. Capture recency does not affect pregame safety: an officiating
assignment is a fact fixed before kickoff, so a page archived in 2026 says
exactly what a page archived in 2009 would have said about who officiated
the game -- ``effective_time`` stays the game's own date regardless of
which capture was read. ``--retry-unparsed`` makes a resumed run re-attempt
games whose existing manifest row parsed 0 officials under the old policy
(re-parsing the on-disk page first, then fetching newer captures only if
that still yields nothing), so run ``20260907T175420Z`` can be re-swept
without refetching pages that already parsed. Every fetched page is its own
immutable file (``html/<pfr_id>__<capture_ts>.html``); the manifest row's
``html_file`` names the capture the crew rows were read from and
``attempted_captures`` lists every capture fetched for the game.

Pregame-safety: an officiating-crew assignment is a fact fixed BEFORE
kickoff (the same premise ``docs/referee_battery.md`` and
``src/nfl_ats/referee_assignments_capture.py`` already argue from for the
2015+ feed and the live weekly capture respectively). This module's
``effective_time`` column is each game's own ``gameday`` -- the assignment
is safe to use in any pregame feature keyed to that date or later, never
earlier.

**Parser structure**: originally (2026-09-05) written blind against two
plausible layouts while Wayback was 429-throttled; since checked against
live captures (2026-09-07). Measured layouts: 2013-2015-era captures carry
``<table id="ref_info">`` with bolded position labels; 2016+ captures
(including 2026 captures of 2009 and 2011 games) carry
``<table id="officials">`` inside an HTML comment (``setup_commented``),
which the regex reads through. The inline "Officials: Referee: X, ..."
fallback has not been seen on any live capture and stays as a defensive
third path. Pre-2013 captures of 2009-2011 games carry NO officials block
under any layout -- that is why capture selection prefers the newest.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/officials_wayback_sweep.py `
        --season-start 2014 --season-end 2014 --run-id 20260905T000000Z

    # continue the same run later (skips every game already on disk):
    .\\.tools\\uv.exe run --no-sync python scripts/officials_wayback_sweep.py `
        --season-start 2009 --season-end 2013 --run-id 20260905T000000Z

    # re-sweep a run whose pages parsed 0 officials under the old policy
    # (re-parses on-disk pages first; fetches newer captures only if needed):
    .\\.tools\\uv.exe run --no-sync python scripts/officials_wayback_sweep.py `
        --season-start 2009 --season-end 2013 --run-id 20260907T175420Z --retry-unparsed
"""

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
# ``from=`` (2026-09-07): captures BEFORE the game are pre-game placeholder
# pages with no officials block -- measured on the first live fetch of
# 2014_01_GB_SEA, whose earliest capture was 2014-05-30 for a 2014-09-04 game
# and parsed 0 officials. Only captures from the day after the game qualify.
# No ``limit`` (2026-09-07, lane N, measured): ``limit=5`` handed back the
# five EARLIEST post-game captures, which for 2009-2011 games predate PFR's
# officials block; ``limit=-3`` (the CDX "last N" form) drew an HTTP 504 --
# a negative limit forces a full index scan -- while the unbounded query for
# the same URLs returned 200 with 61-74 rows (a few KB). The selector ranks
# the full list client-side, newest first.
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


# ---------------------------------------------------------------------------
# Rate limiting and backoff-aware fetch
# ---------------------------------------------------------------------------


@dataclass
class RateLimiter:
    """Enforces at least ``delay_seconds`` between successive ``wait()`` calls."""

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
    """One URL, up to ``max_attempts`` tries, exponential backoff on 429/5xx only.

    Every attempt (including retries) goes through ``limiter.wait()`` first,
    so the >= 8s floor between requests holds even across retries of the
    SAME url, not just between distinct games.
    """

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
    # Unreachable: the loop above always returns by attempt == max_attempts.
    raise AssertionError("fetch_with_backoff exhausted its loop without returning")


# ---------------------------------------------------------------------------
# Schedule loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Officials-block parsing
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_tags(fragment: str) -> str:
    text = _TAG_RE.sub(" ", fragment)
    text = text.replace("&amp;", "&").replace("&nbsp;", " ")
    return _WS_RE.sub(" ", text).strip()


# Strategy A: a dedicated table, e.g. <table ... id="officials"> ... </table>,
# 2026-09-07 (measured on the first live post-game captures, 2014 season):
# that era's boxscore names the same table id="ref_info" and bolds each
# position label in <b>...</b>; strip_tags already handles the label.
# with one <tr> per crew position and two cells (position label, name).
_OFFICIALS_TABLE_RE = re.compile(
    r'<table[^>]*\bid=["\'](?:officials|ref_info)["\'][^>]*>(.*?)</table>',
    re.IGNORECASE | re.DOTALL,
)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)

# Strategy B: an inline "Officials: Referee: X, Umpire: Y, ..." line, e.g.
# inside a <div>...</div> in the page's scorebox_meta block.
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
    """Return ``(rows, warnings)`` where each row is ``(position, official_name)``.

    Tries the dedicated-table strategy first, then the inline-line strategy;
    returns an empty list with a warning if neither structure is present.
    See the module docstring's "Parser structure, honestly labelled" note.
    """

    rows = _parse_officials_table(html)
    if rows:
        return rows, []
    rows = _parse_officials_inline(html)
    if rows:
        return rows, ["matched via inline Officials: line, not a table id=officials"]
    return [], ["no officials block found (neither table nor inline strategy matched)"]


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


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
            # Keep the manifest honest about the CURRENT parser's read of the
            # on-disk page (the 2014 run's first five rows stayed at 0 after
            # the ref_info parser fix because this branch never wrote back).
            existing["officials_parsed"] = len(rows)
            existing["parse_warnings"] = warnings
            if rows or not config.retry_unparsed:
                per_season_parsed_rows[season] = per_season_parsed_rows.get(season, 0) + len(rows)
                per_season_html_on_disk[season] = per_season_html_on_disk.get(season, 0) + 1
                for position, name in rows:
                    officials_rows.append(_officials_row(game, position, name, source_row=existing))
                continue
            # --retry-unparsed: the on-disk page still parses 0 under the
            # current parser, so fall through and try NEWER captures. Every
            # capture already fetched for this game is excluded below.
            games_retried_unparsed += 1
            previous_attempts = _previous_attempts(existing)

        if config.limit is not None and new_fetch_count >= config.limit:
            # Cap NEW network work only: keep walking so every page already
            # on disk is still re-parsed into the parquet (2026-09-07).
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
                # A retry whose CDX lookup failed keeps the page it already has.
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
            consecutive_failures = 0  # a clean 200 with zero rows is not a throttle failure
            if existing is not None and existing.get("html_file"):
                # Nothing newer than the zero-parse page already on disk.
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

        # Newest-first, then up to ``fallback_captures`` further captures if
        # the page fetched parses zero officials.
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
    """Replace the row for ``row['pfr_id']`` in place, or append if absent.

    A resumed run that re-attempts a game (a failed CDX/replay row, or a
    ``--retry-unparsed`` row) must not leave two rows for one game.
    """

    for index, existing in enumerate(manifest_rows):
        if existing.get("pfr_id") == row.get("pfr_id"):
            manifest_rows[index] = row
            return
    manifest_rows.append(row)


def _previous_attempts(existing: dict[str, Any]) -> list[dict[str, Any]]:
    """Every capture already fetched for this game, from an existing manifest row.

    Rows written before the fallback policy carry a single capture in the
    top-level replay fields; rows written after it carry ``attempted_captures``.
    Each entry is tagged ``from_previous_run`` so the retry row keeps the
    old page's provenance without re-fetching it.
    """

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
    """Carry the previously fetched page's fields into a retry row that found nothing better."""

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
    """The CDX ``from=`` bound: the day AFTER the game, as ``YYYYMMDD``.

    A PFR boxscore URL exists before kickoff as a placeholder page; the
    officials block only appears on the post-game page. Measured 2026-09-07:
    the earliest capture of 2014_01_GB_SEA (game 2014-09-04) was dated
    2014-05-30 and parsed zero officials.
    """

    day = pd.Timestamp(gameday).normalize() + pd.Timedelta(days=1)
    return day.strftime("%Y%m%d")


def _rank_capture_timestamps(cdx_json_bytes: bytes, *, not_before: str | None = None) -> list[str]:
    """Post-game capture timestamps from a CDX JSON body, NEWEST first, deduplicated.

    2026-09-07 (lane N): the earliest post-game capture used to be taken,
    and it parsed 0 officials on 418 of 418 pages for 2009-2010 -- PFR added
    the officials block to that era's boxscores years after the games, so
    only later captures carry it. Newest-first is safe because an
    officiating assignment is fixed before kickoff; the archive date of the
    page that reports it changes nothing about when the fact was knowable.
    ``not_before`` re-applies the CDX ``from=`` bound client-side so a
    response that ignored it can never hand back a pre-game placeholder.
    """

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
    """The single preferred capture: the newest post-game one (see ``_rank_capture_timestamps``)."""

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
