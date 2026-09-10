from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.robotparser
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.provenance import sha256_bytes, utc_now  # noqa: E402
from scripts.ingest_nflcom_injuries import (  # noqa: E402
    CELL,
    NICKNAME_TO_CODE,
    ROW,
    SUBTITLE,
    TABLE,
    TEAM_ABBR,
    resolve_current_reg_week,
    strip_html,
)

PRIMARY_URL = "https://www.nfl.com/inactives/"
PRIMARY_ROBOTS_URL = "https://www.nfl.com/robots.txt"
FALLBACK_URL = "https://www.rotowire.com/football/inactives.php"
FALLBACK_ROBOTS_URL = "https://www.rotowire.com/robots.txt"
USER_AGENT = "nfl-ats-research-snapshot/1.0 (private research; contact: local repo owner)"
DEFAULT_DELAY_SECONDS = 2.5

PRIMARY_PLACEHOLDER_TEXT = "Please check back soon for NFL Inactive Reports for this Season"
FALLBACK_PLACEHOLDER_TEXT = "No teams have announced their inactives for this week yet"

SLOTS = (
    "sun_early",
    "sun_late",
    "thu_afternoon_early",
    "thu_afternoon_late",
    "thu_primetime",
    "wed_primetime",
    "sat_early",
    "sat_late",
)

EMPTY_REASON_OFFSEASON_PLACEHOLDER = "primary_offseason_placeholder"
EMPTY_REASON_NO_SCHEDULE = "no_schedule_snapshot"
EMPTY_REASON_SEASON_COMPLETE = "no_upcoming_reg_kickoff"
EMPTY_REASON_UNRECOGNIZED_STRUCTURE = "unrecognized_page_structure"
EMPTY_REASON_FETCH_FAILED = "primary_and_fallback_fetch_failed"

PARQUET_COLUMNS = [
    "captured_at_utc",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "team",
    "player_name",
    "position",
    "status",
    "source_url",
]

_SECTION_SPLIT_CANDIDATES = (
    re.compile(r'<section class="nfl-o-inactive-report__unit">'),
    re.compile(r'<section class="nfl-o-inactives-report__unit">'),
)

FetchFn = Callable[[str, str], tuple[str | None, int | None, str | None, bool]]


def robots_allows(session: requests.Session, url: str, robots_url: str) -> bool:
    response = session.get(robots_url, timeout=60)
    response.raise_for_status()
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch("*", url)


def fetch_page(
    session: requests.Session, url: str, delay: float
) -> tuple[str | None, int | None, str | None]:
    error: str | None = None
    status: int | None = None
    for attempt in range(2):
        try:
            response = session.get(url, timeout=90)
            status = response.status_code
            if status == 200 and response.text:
                return response.text, 200, None
            error = f"http_{status}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt == 0:
            time.sleep(delay)
    return None, status, error


def _default_fetch(
    url: str, robots_url: str, *, session: requests.Session, delay: float
) -> tuple[str | None, int | None, str | None, bool]:
    try:
        allowed = robots_allows(session, url, robots_url)
    except requests.RequestException as exc:
        return None, None, f"robots_fetch_error: {type(exc).__name__}: {exc}", False
    if not allowed:
        return None, None, "robots_disallowed", False
    html_text, status, error = fetch_page(session, url, delay)
    return html_text, status, error, True


def _build_fetch_fn(delay: float) -> FetchFn:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    def fetch_fn(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        return _default_fetch(url, robots_url, session=session, delay=delay)

    return fetch_fn


def _find_column(labels: tuple[str, ...], heads: list[str]) -> int:
    for pos, head in enumerate(heads):
        if any(label in head for label in labels):
            return pos
    return -1


def _cell_value(pos: int, row_cells: list[str]) -> str | None:
    if pos < 0 or pos >= len(row_cells) or not row_cells[pos]:
        return None
    return row_cells[pos]


def _parse_shared_design_system(
    html_text: str,
    *,
    season: int,
    week: int,
    source_url: str,
    fetched_at_utc: str,
) -> tuple[list[dict[str, Any]], list[str]]:

    warnings: list[str] = []
    sections: list[str] = []
    for section_split in _SECTION_SPLIT_CANDIDATES:
        candidate_sections = section_split.split(html_text)[1:]
        if candidate_sections:
            sections = candidate_sections
            break
    if not sections:
        return [], warnings

    rows: list[dict[str, Any]] = []
    abbrs_all = TEAM_ABBR.findall(html_text)
    for section in sections:
        section_abbrs = TEAM_ABBR.findall(section)
        subtitles = SUBTITLE.findall(section)
        tables = TABLE.findall(section)
        codes_by_order = [
            NICKNAME_TO_CODE.get(strip_html(name).casefold(), "") for name in subtitles
        ]
        if len(tables) != len(subtitles):
            warnings.append(f"table/subtitle count mismatch in a {source_url} section")
        fallback = section_abbrs or abbrs_all
        for idx, table_html in enumerate(tables):
            code = ""
            if idx < len(codes_by_order):
                code = codes_by_order[idx]
            elif len(fallback) == 2:
                code = fallback[min(idx, 1)]
            if not code:
                warnings.append(f"unresolved team in a {source_url} section")
                continue
            header: list[str] = []
            body_rows: list[list[str]] = []
            for row_html in ROW.findall(table_html):
                cells = [strip_html(cell) for cell in CELL.findall(row_html)]
                if not cells:
                    continue
                if not header and "player" in cells[0].casefold():
                    header = [cell.casefold() for cell in cells]
                    continue
                body_rows.append(cells)
            if not header:
                warnings.append(f"missing header row in {code} table on {source_url}")
                continue

            col_player = _find_column(("player",), header)
            col_position = _find_column(("position", "pos"), header)
            col_status = _find_column(("status", "reason", "inactive"), header)

            for cells in body_rows:
                player = _cell_value(col_player, cells)
                if player is None:
                    continue
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "team": code,
                        "player_name": player,
                        "position": _cell_value(col_position, cells),
                        "status": _cell_value(col_status, cells) or "Inactive",
                        "source_url": source_url,
                        "fetched_at_utc": fetched_at_utc,
                    }
                )
    return rows, warnings


def _schedule_lookup(repo: Path, season: int, week: int) -> dict[str, tuple[str, str, str]]:

    hits = sorted((repo / "data" / "raw").glob("*/schedules.parquet"))
    if not hits:
        return {}
    sched = pd.read_parquet(
        hits[-1], columns=["season", "week", "game_type", "game_id", "home_team", "away_team"]
    )
    sched = sched.loc[
        (sched["season"] == season) & (sched["week"] == week) & (sched["game_type"] == "REG")
    ]
    lookup: dict[str, tuple[str, str, str]] = {}
    for _, row in sched.iterrows():
        game = (str(row["game_id"]), str(row["home_team"]), str(row["away_team"]))
        lookup[str(row["home_team"])] = game
        lookup[str(row["away_team"])] = game
    return lookup


def run_capture(
    *,
    season: int | None,
    week: int | None,
    slot: str,
    out_root: Path,
    repo: Path = REPO,
    delay: float = DEFAULT_DELAY_SECONDS,
    fetch: FetchFn | None = None,
    now: datetime | None = None,
) -> tuple[Path, bool]:

    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r}, expected one of {SLOTS}")

    moment = now or datetime.now(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    fetched_at_utc = moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    snapshot = out_root / stamp
    snapshot.mkdir(parents=True, exist_ok=True)

    def _write_empty(
        reason: str, *, ok: bool, schedule_error: str | None = None
    ) -> tuple[Path, bool]:
        frame = pd.DataFrame(columns=PARQUET_COLUMNS)
        atomic_parquet(frame, snapshot / "inactives.parquet")
        manifest: dict[str, Any] = {
            "schema": "nflcom_inactives_snapshot/1",
            "snapshot_id": snapshot.name,
            "captured_at_utc": fetched_at_utc,
            "slot": slot,
            "season": season,
            "week": week,
            "source_used": "none",
            "row_count": 0,
            "teams_seen": [],
            "empty_reason": reason,
            "schedule_error": schedule_error,
            "warnings": [],
            "user_agent": USER_AGENT,
            "delay_seconds": delay,
            "ok": ok,
            "generated_at_utc": utc_now(),
        }
        atomic_json(manifest, snapshot / "manifest.json")
        return snapshot, ok

    resolved_season = season
    resolved_week = week
    if resolved_season is None or resolved_week is None:
        try:
            resolved_season, resolved_week = resolve_current_reg_week(repo, pd.Timestamp(moment))
        except SystemExit as exc:
            error_text = str(exc)
            reason = (
                EMPTY_REASON_NO_SCHEDULE
                if error_text.startswith("no data/raw")
                else EMPTY_REASON_SEASON_COMPLETE
            )
            return _write_empty(reason, ok=True, schedule_error=error_text)

    fetch_fn: FetchFn = fetch if fetch is not None else _build_fetch_fn(delay)

    primary_html, primary_status, primary_error, primary_robots_ok = fetch_fn(
        PRIMARY_URL, PRIMARY_ROBOTS_URL
    )
    primary_sha: str | None = None
    if primary_html is not None:
        primary_sha = sha256_bytes(primary_html.encode("utf-8"))
        (snapshot / "primary.html").write_text(primary_html, encoding="utf-8")

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    source_used = "none"
    empty_reason: str | None = None

    primary_showed_placeholder = bool(primary_html and PRIMARY_PLACEHOLDER_TEXT in primary_html)
    if primary_showed_placeholder:
        empty_reason = EMPTY_REASON_OFFSEASON_PLACEHOLDER
    elif primary_html is not None:
        rows, parse_warnings = _parse_shared_design_system(
            primary_html,
            season=resolved_season,
            week=resolved_week,
            source_url=PRIMARY_URL,
            fetched_at_utc=fetched_at_utc,
        )
        warnings.extend(parse_warnings)
        if rows:
            source_used = "primary"

    fallback_html: str | None = None
    fallback_status: int | None = None
    fallback_error: str | None = None
    fallback_robots_ok: bool | None = None
    fallback_sha: str | None = None
    fallback_showed_placeholder = False
    need_fallback = not rows and empty_reason != EMPTY_REASON_OFFSEASON_PLACEHOLDER

    if need_fallback:
        fallback_html, fallback_status, fallback_error, fallback_robots_ok = fetch_fn(
            FALLBACK_URL, FALLBACK_ROBOTS_URL
        )
        if fallback_html is not None:
            fallback_sha = sha256_bytes(fallback_html.encode("utf-8"))
            (snapshot / "fallback.html").write_text(fallback_html, encoding="utf-8")
            fallback_showed_placeholder = FALLBACK_PLACEHOLDER_TEXT in fallback_html
            if fallback_showed_placeholder:
                if empty_reason is None:
                    empty_reason = EMPTY_REASON_OFFSEASON_PLACEHOLDER
            else:
                fb_rows, fb_warnings = _parse_shared_design_system(
                    fallback_html,
                    season=resolved_season,
                    week=resolved_week,
                    source_url=FALLBACK_URL,
                    fetched_at_utc=fetched_at_utc,
                )
                warnings.extend(fb_warnings)
                if fb_rows:
                    rows = fb_rows
                    source_used = "fallback"
                    warnings.append(
                        "primary source parsed 0 rows without showing its known "
                        "placeholder text -- its guessed markup structure likely "
                        "needs fixing against real in-season data"
                    )

    ok = True
    if not rows and empty_reason is None:
        if primary_html is None and fallback_html is None:
            empty_reason = EMPTY_REASON_FETCH_FAILED
        else:
            empty_reason = EMPTY_REASON_UNRECOGNIZED_STRUCTURE
        ok = False

    schedule_map = _schedule_lookup(repo, resolved_season, resolved_week) if rows else {}
    for row in rows:
        row["captured_at_utc"] = row.pop("fetched_at_utc")
        game_id, home_team, away_team = schedule_map.get(row["team"], (None, None, None))
        row["game_id"] = game_id
        row["home_team"] = home_team
        row["away_team"] = away_team

    frame = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    atomic_parquet(frame, snapshot / "inactives.parquet")

    manifest = {
        "schema": "nflcom_inactives_snapshot/1",
        "snapshot_id": snapshot.name,
        "captured_at_utc": fetched_at_utc,
        "slot": slot,
        "season": resolved_season,
        "week": resolved_week,
        "source_used": source_used,
        "primary": {
            "url": PRIMARY_URL,
            "robots_check": {"url": PRIMARY_ROBOTS_URL, "allowed": primary_robots_ok},
            "http_status": primary_status,
            "error": primary_error,
            "sha256": primary_sha,
            "bytes": len(primary_html.encode("utf-8")) if primary_html else 0,
            "showed_known_placeholder": primary_showed_placeholder,
        },
        "fallback": (
            {
                "url": FALLBACK_URL,
                "robots_check": {"url": FALLBACK_ROBOTS_URL, "allowed": fallback_robots_ok},
                "http_status": fallback_status,
                "error": fallback_error,
                "sha256": fallback_sha,
                "bytes": len(fallback_html.encode("utf-8")) if fallback_html else 0,
                "showed_known_placeholder": fallback_showed_placeholder,
            }
            if need_fallback
            else None
        ),
        "row_count": len(frame),
        "teams_seen": sorted(str(t) for t in frame["team"].dropna().unique()) if len(frame) else [],
        "empty_reason": empty_reason,
        "warnings": warnings[:50],
        "user_agent": USER_AGENT,
        "delay_seconds": delay,
        "ok": ok,
        "generated_at_utc": utc_now(),
    }
    atomic_json(manifest, snapshot / "manifest.json")
    return snapshot, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "players" / "inactives")
    parser.add_argument(
        "--current",
        action="store_true",
        help="resolve the live (season, REG week) from the schedules snapshot",
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument(
        "--slot",
        type=str,
        required=True,
        choices=SLOTS,
        help="which scheduler window triggered this capture (recorded in the manifest)",
    )
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    args = parser.parse_args(argv)

    if args.delay < 2.0:
        raise SystemExit("delay must be >= 2 seconds (polite rate limiting)")
    if not args.current and (args.season is None or args.week is None):
        raise SystemExit("pass --current, or both --season and --week")

    season = None if args.current else args.season
    week = None if args.current else args.week

    snapshot, ok = run_capture(
        season=season, week=week, slot=args.slot, out_root=args.out, delay=args.delay
    )
    print(f"snapshot: {snapshot} (ok={ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
