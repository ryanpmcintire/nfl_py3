from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
UV = REPO / ".tools" / "uv.exe"
STATE_PATH = REPO / "data" / "scheduler_state.json"
LOG_PATH = REPO / "data" / "scheduler_log.txt"
HEARTBEAT_PATH = REPO / "data" / "scheduler_heartbeat.json"
POLL_SECONDS = 60
HEARTBEAT_STALE_AFTER_SECONDS = POLL_SECONDS * 3

READ_ONLY_SCRIPT = True
READ_ONLY_EXCEPTIONS: dict[int, str] = {
    1322: "STATE_PATH.parent.mkdir -- STATE_PATH == REPO / 'data' / 'scheduler_state.json'",
    1324: "tmp is STATE_PATH's own .tmp sibling (atomic replace), same tree",
    1351: "HEARTBEAT_PATH.parent.mkdir -- REPO / 'data' / 'scheduler_heartbeat.json'",
    1365: "tmp is HEARTBEAT_PATH's own .tmp sibling (atomic replace), same tree",
    1474: "LOG_PATH.parent.mkdir -- LOG_PATH == REPO / 'data' / 'scheduler_log.txt'",
}

DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from nfl_ats.capture_freshness import (  # noqa: E402
    SourceFreshness,
    any_unexpected_missing,
    compute_freshness,
    render_table,
)


@dataclass(frozen=True)
class Job:
    name: str
    day: str
    at: str
    grace_minutes: int
    command: list[str]
    enabled: bool
    why: str
    season_guarded: bool = True
    dedupe_dir: str = ""
    dedupe_minutes: int = 0
    added_on: str = ""
    catch_up: bool = False
    requires: tuple[str, ...] = ()
    retry_backoff_minutes: int = 0
    max_retries: int = 0


def _ps(script: str) -> list[str]:
    return [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(REPO / "scripts" / script),
    ]


def _cli(*args: str) -> list[str]:
    return [str(UV), "run", "--no-sync", "nfl-ats", *args]


INJURY_CAPTURE = [
    str(UV),
    "run",
    "--no-sync",
    "python",
    str(REPO / "scripts" / "ingest_nflcom_injuries.py"),
    "--current",
]

SPORTRADAR_INJURY_CAPTURE = [
    str(UV),
    "run",
    "--no-sync",
    "python",
    str(REPO / "scripts" / "capture_sportradar_injuries.py"),
]
SPORTRADAR_INJURY_CAPTURE_ENABLED = bool(os.environ.get("SPORTRADAR_API_KEY"))

LINEUP_CAPTURE = [
    str(UV),
    "run",
    "--no-sync",
    "python",
    str(REPO / "scripts" / "refresh_lineup_forecast.py"),
]


def _sportradar_injury_job(day: str, at: str, report: str) -> Job:
    return Job(
        f"sportradar_injuries_{day}",
        day,
        at,
        240,
        SPORTRADAR_INJURY_CAPTURE,
        SPORTRADAR_INJURY_CAPTURE_ENABLED,
        f"Sportradar {report}; enabled only with SPORTRADAR_API_KEY.",
        dedupe_dir="data/raw/sportradar_injuries",
        dedupe_minutes=300,
        added_on="2026-09-02",
    )


def _nflverse_injuries_job(day: str) -> Job:
    return Job(
        f"nflverse_injuries_{day}",
        day,
        "06:00",
        180,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "nflverse_injuries_ingest.py"),
        ],
        True,
        "2026-09-08: measured that no job in this file EVER captured injury data into "
        "the model pipeline -- the NFL.com jobs (injuries_wed/thu/fri/sat) are paused by "
        "MKT-09 source policy and sportradar_injuries_* are disabled without a paid key, "
        "so the active card's injury input was frozen at whatever snapshot a session "
        "built by hand. This bulk-ingests the nflverse injuries release (a public GitHub "
        "release asset, not NFL.com, so MKT-09 does not apply) that "
        "specialist_absence_fade_refresh_overlay.latest_nflverse_injuries_snapshot reads. "
        "06:00 ET clears every same-day consumer with hours to spare (earliest same-day "
        "deadline: refresh_sun 10:00 ET); catch_up=True because a late run is still a "
        "valid, un-mislabelled bulk snapshot, matching player_arrests_tue's own reasoning.",
        dedupe_dir="data/raw/nflverse_injuries",
        dedupe_minutes=240,
        added_on="2026-09-08",
        catch_up=True,
    )


def _nflverse_injuries_pm_job(day: str) -> Job:
    return Job(
        f"nflverse_injuries_{day}_pm",
        day,
        "16:30",
        120,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "nflverse_injuries_ingest.py"),
        ],
        True,
        "2026-09-09: the 06:00 capture runs BEFORE the league publishes that day's own "
        "report, so every same-day consumer reads yesterday's. Measured on Week 1 "
        "Wednesday: the 06:00 ET capture (data/raw/nflverse_injuries/20260909T100005Z) "
        "held 11 2026 rows -- NE and SEA only, published Tuesday for the Wednesday "
        "opener -- while the 16:10 ET capture (20260909T201039Z) held 29, adding SF and "
        "LA for the Thursday game. The 12:00 lineups_wed refit therefore scored SF at LA "
        "with an all-zero injury block on the very day both teams filed. This second "
        "same-day pass lands after the afternoon filings and before the evening refit. "
        "catch_up=True for the same reason the 06:00 job carries it: a late bulk "
        "snapshot is still a valid, un-mislabelled one.",
        dedupe_dir="data/raw/nflverse_injuries",
        dedupe_minutes=120,
        added_on="2026-09-09",
        catch_up=True,
    )


def _player_snapshot_pm_job(day: str) -> Job:
    return Job(
        f"player_snapshot_{day}_pm",
        day,
        "16:45",
        120,
        _cli(
            "player-ingest",
            "--injury-start-season",
            "2009",
            "--injury-end-season",
            "2026",
            "--roster-start-season",
            "2009",
            "--roster-end-season",
            "2026",
            "--snap-start-season",
            "2013",
            "--snap-end-season",
            "2025",
            "--include-postseason",
            "--timestamp-fallback",
            "week_proxy",
        ),
        True,
        "2026-09-09: the consumed half of nflverse_injuries_<day>_pm. Measured the same "
        "day with this exact argv: the 06:28 ET snapshot 20260909T102833Z recorded "
        "n_proxy_rows_per_season 2026=11, the 17:52 ET snapshot 20260909T215234Z "
        "recorded 29, and rebuilding the weak_stack table on the newer one moved SF at "
        "LA from an exactly-zero injury block to -0.140 points of predicted residual and "
        "the SF pick from 55.58% to 56.01%. 16:45 ET is 15m after its capture "
        "counterpart, matching the 06:00/06:15 spacing.",
        dedupe_dir="data/players/raw",
        dedupe_minutes=120,
        added_on="2026-09-09",
        catch_up=True,
    )


def _player_snapshot_job(day: str) -> Job:
    return Job(
        f"player_snapshot_{day}",
        day,
        "06:15",
        180,
        _cli(
            "player-ingest",
            "--injury-start-season",
            "2009",
            "--injury-end-season",
            "2026",
            "--roster-start-season",
            "2009",
            "--roster-end-season",
            "2026",
            "--snap-start-season",
            "2013",
            "--snap-end-season",
            "2025",
            "--include-postseason",
            "--timestamp-fallback",
            "week_proxy",
        ),
        True,
        "2026-09-08: the second half of closing the same injury-data gap its "
        "nflverse_injuries_<day> counterpart closes -- this is what weekly-run's "
        "--refresh-player-data (every lineups_* job, and implicitly every refresh-picks "
        "pass that reads the active card's feature table) actually consumes. "
        "--timestamp-fallback week_proxy (ENG-39) is required: nflverse's 2025+ injuries "
        "release omits date_modified entirely, so the plain default ('drop') zeroes the "
        "injury block for every 2025+ game -- this is the exact argv the 2026-09-08 "
        "manual snapshot (20260908T192720Z) used to put the season's first non-zero "
        "injury cells on the SEA/NE opener. KNOWN, BY-DESIGN LIMIT (not a scheduler "
        "defect): canonicalize_injuries's week_proxy timestamp is "
        "INJURY_PROXY_HOURS_BEFORE_KICKOFF (24h) before that team's own kickoff, and "
        "prediction_safety's lineage check correctly refuses a snapshot whose "
        "effective_timestamp is still in the future -- so a proxied row is not actually "
        "usable by weekly-run's margin-predict step until 24h before its own game's "
        "kickoff, regardless of how early this job runs (measured 2026-09-08: lineups_sat "
        "FAIL(2), 'lineage effective_timestamp is after the prediction timestamp', for "
        "exactly this reason on the Wednesday opener). 06:15 ET (5m after "
        "nflverse_injuries_<day>, avoiding same-tick contention rather than any real "
        "dependency) clears every same-day consumer with hours to spare. catch_up=True: "
        "a late run is still a valid snapshot, matching player_arrests_tue.",
        dedupe_dir="data/players/raw",
        dedupe_minutes=240,
        added_on="2026-09-08",
        catch_up=True,
    )


def _injury_news_job(day: str) -> Job:

    return Job(
        f"injury_news_{day}",
        day,
        "16:00",
        120,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "ingest_injury_news.py"),
            "--fresh-snapshot",
        ],
        True,
        "2026-09-09: docs/follow_news_gate.md's served F3p veto reads "
        "load_news_sources/follow_news_for_game (injury_signal_refresh_tilt.py), whose "
        "ProFootballTalk fallback is the ONLY reader for the live 2026 season -- the "
        "official nflverse injury rows carry no date_modified this season "
        "(_season_has_readable_official_rows is False) -- and this bulk PFT scrape "
        "(ingest_injury_news.py) had never been scheduled, so the veto reads no news "
        "and never fires live. --fresh-snapshot (added this session, mirrors "
        "ingest_transaction_news.py's ENG-32 pattern) copies every prior month's "
        "parquet forward with zero network requests and force-refetches only the "
        "current month; without it a scheduled run would fetch the current month "
        "once and then skip it forever. 16:00 ET clears before nflverse_injuries_"
        "<day>_pm (16:30) and lineups_thu_pm (17:00), the consumers the news reader "
        "sits ahead of. catch_up=False: a late bulk headline pull past this window is "
        "still useful but is no longer the timely same-day read this slot is for.",
        dedupe_dir="data/raw/injury_news",
        dedupe_minutes=120,
        added_on="2026-09-09",
        catch_up=False,
    )


def _inactives_capture(slot: str) -> list[str]:
    return [
        str(UV),
        "run",
        "--no-sync",
        "python",
        str(REPO / "scripts" / "capture_inactives.py"),
        "--current",
        "--slot",
        slot,
    ]


SCHEDULE: tuple[Job, ...] = (
    *(
        Job(
            f"lineups_{day}",
            day,
            "14:30" if day == "tue" else "12:00",
            180,
            LINEUP_CAPTURE,
            True,
            "Current depth-chart starters for the static This Week lineup panel.",
            dedupe_dir="artifacts/lineups",
            dedupe_minutes=180,
            added_on="2026-09-03",
        )
        for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    ),
    Job(
        "odds_tue_open",
        "tue",
        "12:05",
        180,
        _ps("odds_capture.ps1"),
        True,
        "Tuesday opener: the grade the pool settles on. The pool's spreads lock "
        "at 12:00 ET (owner, 2026-09-08: 'Spreads lock: Tue 12:00 PM'), so this "
        "is the first capture of the day and lands just after that lock -- it IS "
        "the line the pool grades on. Nothing captures odds earlier on a Tuesday: "
        "the week's opener is the EARLIEST Tuesday quote per book. Wide grace -- "
        "the locked line does not move, so a late capture is still the opener.",
        season_guarded=False,
        dedupe_dir="data/market/raw",
        dedupe_minutes=90,
    ),
    Job(
        "splash_board_tue",
        "tue",
        "12:05",
        15,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "check_splash_board.py"),
        ],
        True,
        "Does data/splash hold a validated capture of the board the pool grades "
        "on for the week the 12:20 lock is about to record? The board is a HAND "
        "read -- capture_splash_lines.py converts text someone copied off the "
        "page, nothing scrapes the site -- so the one manual step in the whole "
        "lock day had no job watching it. Measured 2026-09-09: with only Week 1 "
        "captured, a Week 2 lock spends ~8 minutes refitting and then fails "
        "closed on pool_line_source because three served games carry "
        "whole-number nflverse lines. This asks the same question in a second, "
        "at 12:05, and weekly_lock requires it. Retries every 3 minutes inside "
        "the 15-minute window so a board read that lands at 12:10 still clears "
        "the 12:20 lock; if it lands later, capture it and run "
        "--run-job weekly_lock by hand.",
        added_on="2026-09-09",
        catch_up=False,
        retry_backoff_minutes=3,
        max_retries=4,
    ),
    Job(
        "weekly_lock",
        "tue",
        "12:20",
        120,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "scheduled_weekly_lock.py"),
        ],
        True,
        "Lock-day paper forecast, formed on the line the pool locked at noon. "
        "Runs only for an actual scheduled game week, after the 12:05 opener "
        "and the 12:05 pool-board check both succeed, and closes at 14:20; "
        "picks are due at each game's own kickoff (Sunday 4 PM ET cap), so a "
        "lock after noon costs nothing.",
        added_on="2026-09-02",
        requires=("odds_tue_open", "splash_board_tue"),
    ),
    Job(
        "airnow_tue_checkpoint",
        "tue",
        "11:40",
        15,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "capture_airnow_hourly.py"),
        ],
        True,
        "No-auth EPA hourly AQI file capture immediately before the Tuesday-noon "
        "research checkpoint. AirNow normally publishes each UTC hour at ~:35; "
        "the capture intentionally selects the latest completed observation hour.",
        dedupe_dir="data/raw/airnow_hourly",
        dedupe_minutes=50,
        added_on="2026-09-02",
    ),
    Job(
        "odds_wed_opener",
        "wed",
        "18:00",
        90,
        _ps("odds_capture.ps1"),
        True,
        "Pre-Wednesday-opener line, ~2h before an 8:20 kickoff; also the "
        "first post-Tuesday line every ordinary week.",
        season_guarded=False,
        dedupe_dir="data/market/raw",
        dedupe_minutes=90,
        added_on="2026-09-07",
    ),
    Job(
        "odds_thu_tnf",
        "thu",
        "18:00",
        90,
        _ps("odds_capture.ps1"),
        True,
        "Pre-TNF, ~2h before an 8:15 kickoff.",
        season_guarded=False,
        dedupe_dir="data/market/raw",
        dedupe_minutes=90,
    ),
    Job(
        "odds_sat",
        "sat",
        "12:00",
        180,
        _ps("odds_capture.ps1"),
        True,
        "Saturday state of the board.",
        season_guarded=False,
        dedupe_dir="data/market/raw",
        dedupe_minutes=90,
    ),
    Job(
        "odds_sun_close",
        "sun",
        "12:30",
        25,
        _ps("odds_capture.ps1"),
        True,
        "CLOSING line for the 13:00 slate. Short grace on purpose: a capture "
        "after 13:00 is not a close, it is a live line, and mislabelling that "
        "would corrupt every CLV number computed from it.",
        season_guarded=False,
        dedupe_dir="data/market/raw",
        dedupe_minutes=90,
    ),
    Job(
        "odds_sun_late",
        "sun",
        "16:15",
        60,
        _ps("odds_capture.ps1"),
        True,
        "Late-window close.",
        season_guarded=False,
        dedupe_dir="data/market/raw",
        dedupe_minutes=90,
    ),
    Job(
        "odds_mon_mnf",
        "mon",
        "19:00",
        90,
        _ps("odds_capture.ps1"),
        True,
        "Pre-MNF.",
        season_guarded=False,
        dedupe_dir="data/market/raw",
        dedupe_minutes=90,
    ),
    Job(
        "odds_tue_open_halves",
        "tue",
        "12:05",
        180,
        _cli("odds-ingest-halves"),
        True,
        "LEAD-61 per-event half/quarter-game market capture riding the "
        "Tuesday opener window; requires=('odds_tue_open',) for its event "
        "ids and quota reading.",
        season_guarded=False,
        added_on="2026-09-05",
        requires=("odds_tue_open",),
    ),
    Job(
        "odds_sat_halves",
        "sat",
        "12:00",
        180,
        _cli("odds-ingest-halves"),
        True,
        "LEAD-61 per-event half/quarter-game market capture riding the "
        "Saturday board window; requires=('odds_sat',) for its event ids "
        "and quota reading.",
        season_guarded=False,
        added_on="2026-09-05",
        requires=("odds_sat",),
    ),
    Job(
        "public_betting_sat",
        "sat",
        "12:00",
        240,
        _ps("public_betting_capture.ps1"),
        True,
        "Saturday bet%/money% snapshot.",
        season_guarded=False,
        dedupe_dir="data/raw/public_betting_live",
        dedupe_minutes=90,
    ),
    Job(
        "public_betting_sun",
        "sun",
        "12:00",
        45,
        _ps("public_betting_capture.ps1"),
        True,
        "Sunday pre-kickoff bet%/money%; must precede the 13:00 slate.",
        season_guarded=False,
        dedupe_dir="data/raw/public_betting_live",
        dedupe_minutes=90,
    ),
    Job(
        "injuries_wed",
        "wed",
        "17:30",
        240,
        INJURY_CAPTURE,
        False,
        "PAUSED by MKT-09 source policy: NFL.com terms require express consent "
        "before systematic retrieval.",
        dedupe_dir="data/raw/nflcom_injuries",
        dedupe_minutes=300,
    ),
    Job(
        "injuries_thu",
        "thu",
        "17:30",
        240,
        INJURY_CAPTURE,
        False,
        "PAUSED by MKT-09 source policy: NFL.com terms require express consent "
        "before systematic retrieval.",
        dedupe_dir="data/raw/nflcom_injuries",
        dedupe_minutes=300,
    ),
    Job(
        "injuries_fri",
        "fri",
        "17:30",
        240,
        INJURY_CAPTURE,
        False,
        "PAUSED by MKT-09 source policy: NFL.com terms require express consent "
        "before systematic retrieval.",
        dedupe_dir="data/raw/nflcom_injuries",
        dedupe_minutes=300,
    ),
    Job(
        "injuries_sat",
        "sat",
        "10:00",
        240,
        INJURY_CAPTURE,
        False,
        "PAUSED by MKT-09 source policy: NFL.com terms require express consent "
        "before systematic retrieval.",
        dedupe_dir="data/raw/nflcom_injuries",
        dedupe_minutes=300,
    ),
    *(
        _sportradar_injury_job(day, at, report)
        for day, at, report in (
            ("wed", "17:30", "weekly practice report"),
            ("thu", "17:30", "weekly practice report"),
            ("fri", "17:30", "final game-status report"),
            ("sat", "10:00", "post-Friday fallback"),
        )
    ),
    Job(
        "refresh_wed",
        "wed",
        "18:15",
        90,
        _cli("refresh-picks", "--record-decisions", "--note", "wednesday_opener"),
        True,
        "Pre-Wednesday-opener pass (2026 Week 1 opens Wednesday): runs on the "
        "odds_wed_opener capture and closes before an 8:20 kickoff; a no-op "
        "in weeks with no Wednesday game.",
        added_on="2026-09-07",
    ),
    Job(
        "refresh_thu",
        "thu",
        "15:00",
        240,
        _cli("refresh-picks", "--record-decisions", "--note", "thursday_afternoon"),
        True,
        "Pre-TNF pass: finalizes Thursday picks on Tue-to-Thu information.",
    ),
    Job(
        "refresh_sat",
        "sat",
        "10:30",
        300,
        _cli("refresh-picks", "--record-decisions", "--note", "saturday_pass"),
        True,
        "Everything not yet locked gets another look.",
    ),
    Job(
        "refresh_sun",
        "sun",
        "10:00",
        300,
        _cli(
            "refresh-picks",
            "--record-decisions",
            "--note",
            "sunday_morning_final",
            "--publish-card",
        ),
        True,
        "FINAL pass; the only one that touches the card, additively.",
    ),
    Job(
        "refresh_trigger_log_sun",
        "sun",
        "18:00",
        240,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "refresh_trigger_log.py"),
            "--scan",
            "--current",
        ],
        True,
        "ENG-08 timing-policy instrumentation: reconstructs real non-clock "
        "refresh triggers plus the week's fired clock checkpoints and appends "
        "them, deadline-validated, to the read-only evidence log.",
        added_on="2026-09-04",
        catch_up=True,
    ),
    Job(
        "backup_data",
        "sun",
        "22:00",
        300,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "backup_data.py"),
            "--include-artifacts",
        ],
        True,
        "Weekly off-device mirror to E:. Runs AFTER the week's last capture "
        "(refresh_sun 10:00, odds_sun_late 16:15) so a week's point-in-time "
        "data or artifact ledger is never left unmirrored over the following "
        "week. Needs no "
        "dedupe guard: backup_data.py is idempotent by construction -- a "
        "second run finds every file size- and mtime-identical and copies "
        "nothing (measured 2026-08-27: 14.6s for a no-op pass over 42,839 "
        "files), so double-scheduling costs seconds, not a re-copy. If the "
        "mirror drive is absent the run fails loudly and the next one resumes; "
        "a partial copy is not a corrupt state.",
        season_guarded=False,
        added_on="2026-08-27",
        catch_up=True,
    ),
    Job(
        "player_arrests_tue",
        "tue",
        "07:00",
        90,
        _cli("ingest-player-arrests"),
        True,
        "Feeds the PROMOTED player-arrest policy component on the live card "
        "(HANDOFF.md). Runs at 07:00, closing by 08:30 -- a full two hours "
        "before odds_tue_open (09:00, the grade the pool settles on) so a "
        "fresh arrest snapshot is always on disk before the Tuesday publish "
        "reads it. Idempotent by construction: every run fetches the USA "
        "Today public arrests table (no auth, no paid API) into a fresh "
        "UTC-stamped snapshot dir and never mutates an old one, so running "
        "late or twice just adds a newer, equally valid snapshot -- catch_up "
        "is safe. Measured 2026-08-31: 56 pages, 1,116 rows, ~3-4 minutes "
        "including the 1.5s per-page delay, well inside the 90m grace and "
        "the 1800s subprocess timeout.",
        season_guarded=False,
        dedupe_dir="data/raw/player_arrests",
        dedupe_minutes=240,
        added_on="2026-09-01",
        catch_up=True,
        retry_backoff_minutes=15,
        max_retries=2,
    ),
    *(_nflverse_injuries_job(day) for day in ("wed", "thu", "fri", "sat", "sun")),
    *(_player_snapshot_job(day) for day in ("wed", "thu", "fri", "sat", "sun")),
    *(_nflverse_injuries_pm_job(day) for day in ("wed", "thu", "fri", "sat")),
    *(_player_snapshot_pm_job(day) for day in ("wed", "thu", "fri", "sat")),
    *(_injury_news_job(day) for day in ("wed", "thu", "fri", "sat", "sun")),
    Job(
        "injury_news_sun_early",
        "sun",
        "11:30",
        20,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "ingest_injury_news.py"),
            "--fresh-snapshot",
        ],
        True,
        "2026-09-09: a second same-day PFT pull ahead of refresh_sun_inactives_early "
        "(11:55, closing 12:50 before the 13:00 ET slate) so Sunday-morning injury "
        "news filed since the 16:00 Saturday pull can still veto a Sunday follow "
        "before that refresh reads it. 20m grace closes at 11:50, five minutes clear "
        "of 11:55.",
        dedupe_dir="data/raw/injury_news",
        dedupe_minutes=120,
        added_on="2026-09-09",
        catch_up=False,
    ),
    *(
        Job(
            f"lineups_{day}_pm",
            day,
            at,
            120,
            LINEUP_CAPTURE,
            True,
            "2026-09-09: a capture nothing rebuilds the feature table from is not on the "
            "card. lineups_<day> refits at 12:00 ET off the 06:15 snapshot, so a report "
            "filed that afternoon cannot reach a pick until the next day's noon refit -- "
            "and for a Thursday game there is no next day. This runs the same "
            "refresh_lineup_forecast.py argv the noon job runs, so it inherits that "
            "job's fail-closed behaviour unchanged. The TIME is set by the week_proxy "
            "horizon, not by the filing time: canonicalize_injuries stamps an undated "
            "report at its own game's kickoff minus 24h, and lineage.py refuses a card "
            "whose effective_timestamp is still in the future, so a refit before that "
            "horizon aborts rather than prices the report. Measured 2026-09-09: the "
            "Thursday SF at LA rows captured at 16:10 ET carry effective_observed_at "
            "2026-09-10T00:35Z (kickoff 2026-09-11T00:35Z minus 24h), so a 17:05 ET "
            "refit would have failed exactly the way lineups_tue failed on 2026-09-08. "
            "wed 20:45 clears a Thursday-night kickoff's horizon; sat 13:30 clears the "
            "Sunday 13:00 ET horizon that carries most of the slate.",
            dedupe_dir="artifacts/margin_predictions",
            dedupe_minutes=120,
            added_on="2026-09-09",
        )
        for day, at in (("wed", "20:45"), ("thu", "17:00"), ("sat", "13:30"))
    ),
    Job(
        "inactives_sun_early",
        "sun",
        "11:35",
        15,
        _inactives_capture("sun_early"),
        True,
        "Covers the 147 Sun-13:00-ET 2026 kickoffs (docs/inactives_channel.md "
        "Section 2). True T-90 for a 13:00 kickoff is 11:30 ET; 11:35 gives the "
        "source a moment to publish before the first fetch, same idea as "
        "odds_sun_close's short grace. This slot's deadline equals kickoff "
        "itself (pick_refresh.sunday_pick_lock does not bind until 16:00 ET), "
        "so the doc's measured +90m slack at T-90 is real runway before lock.",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
        added_on="2026-09-01",
        catch_up=False,
    ),
    Job(
        "inactives_sun_late",
        "sun",
        "14:40",
        15,
        _inactives_capture("sun_late"),
        True,
        "Covers the 58 Sun-16:05..17:00-ET 2026 kickoffs (docs/inactives_channel.md "
        "Section 2). True T-90 ranges 14:35-15:30 ET depending on the week's exact "
        "late slate, but the BINDING deadline for this slot is the week's fixed "
        "Sunday 16:00 ET pick lock, not each game's own kickoff -- 14:40 lands "
        "inside every week's T-90 window while leaving 65 minutes of grace-close "
        "margin (14:40 + 15m grace = 14:55) before that lock.",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
        added_on="2026-09-01",
        catch_up=False,
    ),
    Job(
        "inactives_thu_afternoon_early",
        "thu",
        "11:35",
        15,
        _inactives_capture("thu_afternoon_early"),
        True,
        "Thu variant, Option A of docs/inactives_channel.md Section 6: measured "
        "2026 Thu kickoffs are 13:00/16:30/20:15/20:20/20:35 ET (Thanksgiving and "
        "the season-opener week kick earlier than the usual TNF slot), so one "
        "fixed time cannot cover them all -- three jobs approximate T-90 for each "
        "historically observed cluster. This one covers a 13:00 ET kickoff "
        "(T-90=11:30 ET). Named _early/_primetime rather than reusing the doc's "
        "literal 'inactives_thu_afternoon' name for two separate times: Job.name "
        "doubles as the run-state key (f'{name}@{date}'), so two same-named Jobs "
        "landing on the same Thursday would collide and the later one would "
        "silently no-op against the earlier one's already-written state entry.",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
        added_on="2026-09-01",
        catch_up=False,
    ),
    Job(
        "inactives_thu_afternoon_late",
        "thu",
        "15:05",
        15,
        _inactives_capture("thu_afternoon_late"),
        True,
        "Second Thu cluster: covers a 16:30 ET kickoff (T-90=15:00 ET). See "
        "inactives_thu_afternoon_early's comment for why this needs its own "
        "job/name rather than sharing one.",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
        added_on="2026-09-01",
        catch_up=False,
    ),
    Job(
        "inactives_thu_primetime",
        "thu",
        "18:50",
        20,
        _inactives_capture("thu_primetime"),
        True,
        "Third Thu cluster: covers the regular TNF kickoff times, measured 2026 "
        "as 20:15/20:20/20:35 ET (T-90=18:45-19:05 ET). Wider 20m grace than the "
        "two afternoon jobs because this one window has to straddle three "
        "distinct historically observed kickoff times instead of one.",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
        added_on="2026-09-01",
        catch_up=False,
    ),
    Job(
        "inactives_wed_primetime",
        "wed",
        "18:50",
        20,
        _inactives_capture("wed_primetime"),
        True,
        "Wednesday primetime opener (2026 Week 1: 20:20 ET, T-90=18:50 ET); "
        "the inactives_thu_primetime window, one day earlier.",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
        added_on="2026-09-09",
        catch_up=False,
    ),
    Job(
        "inactives_sat_early",
        "sat",
        "15:30",
        15,
        _inactives_capture("sat_early"),
        True,
        "Sat variant, same Option-A gap as Thu (docs/inactives_channel.md "
        "Section 6): 2026 measured only 2 Sat games (17:00/20:20 ET), but a "
        "real December late-season Saturday slate can carry more games at more "
        "varied times. Covers a 17:00 ET kickoff (T-90=15:30 ET).",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
        added_on="2026-09-01",
        catch_up=False,
    ),
    Job(
        "inactives_sat_late",
        "sat",
        "18:50",
        20,
        _inactives_capture("sat_late"),
        True,
        "Covers a 20:20 ET Sat kickoff (T-90=18:50 ET). docs/inactives_channel.md "
        "Section 6 only formalized the 15:30 ET Sat job by name and noted in "
        "prose that a later kickoff would need a second job 'at sat 18:50 ET' "
        "without writing it up as its own proposal -- added here as its own row "
        "for the same reason the Thu cluster needs three, following the doc's "
        "own logic rather than leaving that second job unbuilt.",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
        added_on="2026-09-01",
        catch_up=False,
    ),
    Job(
        "refresh_thu_inactives_early",
        "thu",
        "11:55",
        55,
        _cli("refresh-picks", "--record-decisions", "--note", "thu_inactives_early"),
        True,
        "After inactives_thu_afternoon_early closes at 11:50; its 55m grace ends "
        "at 12:50, ten minutes before a 13:00 ET kickoff.",
        added_on="2026-09-02",
        catch_up=False,
    ),
    Job(
        "refresh_thu_inactives_late",
        "thu",
        "15:25",
        55,
        _cli("refresh-picks", "--record-decisions", "--note", "thu_inactives_late"),
        True,
        "After inactives_thu_afternoon_late closes at 15:20; its 55m grace ends "
        "at 16:20, ten minutes before a 16:30 ET kickoff.",
        added_on="2026-09-02",
        catch_up=False,
    ),
    Job(
        "refresh_thu_inactives_primetime",
        "thu",
        "19:15",
        50,
        _cli("refresh-picks", "--record-decisions", "--note", "thu_inactives_primetime"),
        True,
        "After inactives_thu_primetime closes at 19:10; its 50m grace ends at "
        "20:05, ten minutes before the earliest 20:15 ET primetime kickoff.",
        added_on="2026-09-02",
        catch_up=False,
    ),
    Job(
        "refresh_wed_inactives_primetime",
        "wed",
        "19:15",
        50,
        _cli("refresh-picks", "--record-decisions", "--note", "wed_inactives_primetime"),
        True,
        "After inactives_wed_primetime closes at 19:10; its 50m grace ends at "
        "20:05, fifteen minutes before the 20:20 ET Wednesday opener kickoff.",
        added_on="2026-09-09",
        catch_up=False,
    ),
    Job(
        "refresh_sat_inactives_early",
        "sat",
        "15:50",
        60,
        _cli("refresh-picks", "--record-decisions", "--note", "sat_inactives_early"),
        True,
        "After inactives_sat_early closes at 15:45; its 60m grace ends at 16:50, "
        "ten minutes before a 17:00 ET kickoff.",
        added_on="2026-09-02",
        catch_up=False,
    ),
    Job(
        "refresh_sat_inactives_late",
        "sat",
        "19:15",
        55,
        _cli("refresh-picks", "--record-decisions", "--note", "sat_inactives_late"),
        True,
        "After inactives_sat_late closes at 19:10; its 55m grace ends at 20:10, "
        "ten minutes before a 20:20 ET kickoff.",
        added_on="2026-09-02",
        catch_up=False,
    ),
    Job(
        "refresh_sun_inactives_early",
        "sun",
        "11:55",
        55,
        _cli("refresh-picks", "--record-decisions", "--note", "sun_inactives_early"),
        True,
        "After inactives_sun_early closes at 11:50; its 55m grace ends at 12:50, "
        "ten minutes before the 13:00 ET slate.",
        added_on="2026-09-02",
        catch_up=False,
    ),
    Job(
        "refresh_sun_inactives_late",
        "sun",
        "15:00",
        50,
        _cli("refresh-picks", "--record-decisions", "--note", "sun_inactives_late"),
        True,
        "After inactives_sun_late closes at 14:55; its 50m grace ends at 15:50, "
        "ten minutes before the fixed Sunday 16:00 ET lock.",
        added_on="2026-09-02",
        catch_up=False,
    ),
    Job(
        "referee_assignments_wed",
        "wed",
        "15:00",
        240,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "capture_referee_assignments.py"),
            "--current",
        ],
        True,
        "Weekly officiating-crew assignment capture (Football Zebras). Wed "
        "15:00 ET clears every 2025-measured publish timestamp (latest "
        "normal-week sample: Wed 12:42 ET) by 2+ hours. catch_up=True: a late "
        "run is still a valid snapshot, not a mislabelled one, matching "
        "player_arrests_tue/backup_data's reasoning rather than the odds "
        "captures' point-in-time-only one.",
        dedupe_dir="data/players/referee_assignments",
        dedupe_minutes=240,
        added_on="2026-09-01",
        catch_up=True,
    ),
    Job(
        "pfr_transactions_wed",
        "wed",
        "07:00",
        120,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "ingest_transaction_news.py"),
            "--fresh-snapshot",
        ],
        True,
        "Live PFR transaction-wire capture feeding late-week injury/roster "
        "awareness. Fresh dated snapshot per run (ENG-32); prior years "
        "copied forward with no re-fetch, bounded cost per run.",
        season_guarded=True,
        dedupe_dir="data/raw/pfr_transactions",
        dedupe_minutes=2000,
        added_on="2026-09-03",
        catch_up=True,
    ),
    Job(
        "pfr_transactions_sat",
        "sat",
        "07:00",
        120,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "ingest_transaction_news.py"),
            "--fresh-snapshot",
        ],
        True,
        "Live PFR transaction-wire capture feeding late-week injury/roster "
        "awareness. Fresh dated snapshot per run (ENG-32); prior years "
        "copied forward with no re-fetch, bounded cost per run.",
        season_guarded=True,
        dedupe_dir="data/raw/pfr_transactions",
        dedupe_minutes=2000,
        added_on="2026-09-03",
        catch_up=True,
    ),
    *(
        Job(
            f"settle_{day}",
            day,
            "23:59",
            120,
            _cli("settle", "--write-graded"),
            True,
            "2026-09-09: nothing turned a final score into a graded row until the NEXT "
            "Tuesday. weekly-run is the only thing that refreshes "
            "data/processed/game_features.parquet, and prospective-score reads results out "
            "of that table -- and it only ever covered two of the fifteen ledgers a lock "
            "and refresh week writes. This grades every ledger at its own frozen decision "
            "line against results fetched now, writes the graded rows BESIDE each ledger "
            "(the recorded picks are never touched) and refreshes "
            "artifacts/settlement/game_results.parquet, which the board's settled strip "
            "reads on top of the weekly feature table. 23:59 ET clears the last kickoff of "
            "each game day (TNF, SNF, MNF all end by ~23:30); catch_up=True because a late "
            "run is still a correct settlement, and the grace runs to 01:59 so a result "
            "that lands after midnight is still caught. --write-graded is stripped by "
            "--dry, so a manual rehearsal reports without writing.",
            season_guarded=True,
            added_on="2026-09-09",
            catch_up=True,
        )
        for day in ("wed", "thu", "sun", "mon")
    ),
    Job(
        "verify_full_weekly",
        "mon",
        "03:00",
        240,
        [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "verify_full.py")],
        False,
        "Full verification tier (ENG-11): the AGENTS.md release gate "
        "(ruff format --check, ruff check, mypy src, full pytest), run "
        "on a schedule separate from the fast PR loop. Disabled by "
        "default -- enable deliberately, this is a multi-minute CPU-bound "
        "run, not something to fire unattended by default.",
        season_guarded=False,
        added_on="2026-09-04",
        catch_up=True,
    ),
)


SNAPSHOT_NAME = re.compile(r"^(\d{8}T\d{6}Z)$")


def newest_snapshot_age_minutes(relative_dir: str, now: datetime) -> float | None:

    root = REPO / relative_dir
    if not root.is_dir():
        return None
    newest: datetime | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = SNAPSHOT_NAME.match(child.name)
        if not match:
            continue
        try:
            stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if newest is None or stamp > newest:
            newest = stamp
    if newest is None:
        return None
    return (now - newest).total_seconds() / 60.0


def already_captured(job: Job, now: datetime) -> tuple[bool, float | None]:
    if not job.dedupe_dir or job.dedupe_minutes <= 0:
        return False, None
    age = newest_snapshot_age_minutes(job.dedupe_dir, now)
    if age is None:
        return False, None
    return age < job.dedupe_minutes, age


def predates_job(job: Job, start: datetime) -> bool:

    if not job.added_on:
        return False
    return start.date() < date.fromisoformat(job.added_on)


def snapshot_in_window(job: Job, start: datetime) -> bool:

    if not job.dedupe_dir:
        return False
    root = REPO / job.dedupe_dir
    if not root.is_dir():
        return False
    end = start + timedelta(minutes=job.grace_minutes)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = SNAPSHOT_NAME.match(child.name)
        if not match:
            continue
        try:
            stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if start <= stamp <= end:
            return True
    return False


_SEASON_CACHE: list[Any] = []


def season_active(when: datetime) -> bool:

    if not _SEASON_CACHE:
        hits = sorted((REPO / "data" / "raw").glob("*/schedules.parquet"))
        if not hits:
            _SEASON_CACHE.append(None)
        else:
            import pandas as pd

            sched = pd.read_parquet(hits[-1], columns=["game_type", "gameday"])
            sched = sched.loc[
                sched["game_type"].astype(str).isin({"REG", "WC", "DIV", "CON", "SB"})
            ]
            days = pd.to_datetime(sched["gameday"], errors="coerce").dropna()
            _SEASON_CACHE.append(set(days.dt.date))
    known = _SEASON_CACHE[0]
    if known is None:
        return True
    day = when.date()
    return any((day + timedelta(days=offset)) in known for offset in range(-10, 4))


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"runs": {}}
    try:
        return cast("dict[str, Any]", json.loads(STATE_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {"runs": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_PATH)


def write_heartbeat(
    *,
    started_at: datetime,
    now: datetime,
    code_sha256: str | None = None,
    schedule_digest: str | None = None,
) -> None:

    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "started_at": started_at.isoformat(timespec="seconds"),
        "last_poll_at": now.isoformat(timespec="seconds"),
        "poll_seconds": POLL_SECONDS,
        "enabled_job_count": sum(1 for job in SCHEDULE if job.enabled),
        "code_sha256": code_sha256 if code_sha256 is not None else compute_code_sha256(),
        "schedule_digest": (
            schedule_digest if schedule_digest is not None else compute_schedule_digest()
        ),
    }
    tmp = HEARTBEAT_PATH.with_suffix(".tmp")
    with _HEARTBEAT_WRITE_LOCK:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(HEARTBEAT_PATH)


_HEARTBEAT_WRITE_LOCK = threading.Lock()


def start_heartbeat_keepalive(
    *,
    started_at: datetime,
    code_sha256: str,
    schedule_digest: str,
    stop: threading.Event,
) -> threading.Thread:

    def _beat() -> None:
        while not stop.wait(POLL_SECONDS):
            try:
                write_heartbeat(
                    started_at=started_at,
                    now=datetime.now(tz=ET),
                    code_sha256=code_sha256,
                    schedule_digest=schedule_digest,
                )
            except Exception as exc:  # pragma: no cover - the beat must never kill the loop
                log(f"HEARTBEAT-ERROR {type(exc).__name__}: {exc}")

    thread = threading.Thread(target=_beat, name="heartbeat-keepalive", daemon=True)
    thread.start()
    return thread


def read_heartbeat() -> dict[str, Any] | None:
    if not HEARTBEAT_PATH.is_file():
        return None
    try:
        payload = json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def compute_code_sha256() -> str:

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def compute_schedule_digest() -> str:

    canonical = [
        {
            "name": job.name,
            "weekday": job.day,
            "time": job.at,
            "grace": job.grace_minutes,
            "enabled": job.enabled,
            "command": job.command,
            "added_on": job.added_on,
        }
        for job in SCHEDULE
    ]
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _short_hash(value: str | None) -> str:

    return value[:12] if value else "unknown"


_DEFAULT_JOB_HEALTH: dict[str, Any] = {
    "last_success_at": None,
    "last_failure_at": None,
    "last_error": "",
    "consecutive_failures": 0,
    "missed_window_count": 0,
}


def _job_health_entry(state: dict[str, Any], job_name: str) -> dict[str, Any]:
    job_health: dict[str, Any] = state.setdefault("job_health", {})
    entry: dict[str, Any] = job_health.setdefault(job_name, dict(_DEFAULT_JOB_HEALTH))
    return entry


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=ET).isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def occurrence(job: Job, now: datetime) -> datetime:

    target_h, target_m = (int(part) for part in job.at.split(":"))
    days_back = (now.weekday() - DAYS[job.day]) % 7
    day = (now - timedelta(days=days_back)).date()
    stamp = datetime.combine(day, datetime.min.time(), tzinfo=ET).replace(
        hour=target_h, minute=target_m
    )
    if stamp > now:
        stamp -= timedelta(days=7)
    return stamp


def retry_eligible(job: Job, record: dict[str, Any], now: datetime) -> bool:

    if job.retry_backoff_minutes <= 0 or job.max_retries <= 0:
        return False
    if not str(record.get("status", "")).startswith("FAIL"):
        return False
    if int(record.get("retries", 0)) >= job.max_retries:
        return False
    ran_at_raw = record.get("ran_at")
    if not ran_at_raw:
        return False
    try:
        ran_at = datetime.fromisoformat(str(ran_at_raw))
    except ValueError:
        return False
    return now >= ran_at + timedelta(minutes=job.retry_backoff_minutes)


def due_jobs(now: datetime, state: dict[str, Any]) -> list[tuple[Job, datetime]]:
    due: list[tuple[Job, datetime]] = []
    for job in SCHEDULE:
        if not job.enabled:
            continue
        start = occurrence(job, now)
        key = f"{job.name}@{start.date().isoformat()}"
        record = state["runs"].get(key)
        if record is not None:
            if not retry_eligible(job, record, now):
                continue
        elif job.season_guarded and not season_active(start):
            continue
        if predates_job(job, start):
            continue
        if not start <= now <= start + timedelta(minutes=job.grace_minutes):
            continue
        blocked = unsatisfied_prerequisites(job, start, state)
        if blocked:
            note_blocked(job, start, blocked, state)
            continue
        due.append((job, start))
    return due


def unsatisfied_prerequisites(job: Job, start: datetime, state: dict[str, Any]) -> list[str]:

    accepted = {"OK", "ALREADY-CAPTURED"}
    blocked = []
    for required_name in job.requires:
        record = state["runs"].get(f"{required_name}@{start.date().isoformat()}", {})
        if record.get("status") not in accepted:
            blocked.append(f"{required_name}={record.get('status', 'no record')}")
    return blocked


def prerequisites_satisfied(job: Job, start: datetime, state: dict[str, Any]) -> bool:

    return not unsatisfied_prerequisites(job, start, state)


def note_blocked(job: Job, start: datetime, blocked: list[str], state: dict[str, Any]) -> None:

    entry = _job_health_entry(state, job.name)
    key = f"{job.name}@{start.date().isoformat()}"
    if entry.get("last_blocked_notice") == key:
        return
    entry["last_blocked_notice"] = key
    closes = (start + timedelta(minutes=job.grace_minutes)).strftime("%H:%M")
    log(
        f"BLOCKED {job.name} (window {start.isoformat()}, closes {closes}): "
        f"prerequisites not successful: {', '.join(blocked)}"
    )


def record_already_captured(job: Job, start: datetime, age: float, state: dict[str, Any]) -> None:

    state["runs"][f"{job.name}@{start.date().isoformat()}"] = {
        "status": "ALREADY-CAPTURED",
        "window_start": start.isoformat(),
        "newest_snapshot_age_minutes": round(age, 1),
    }
    entry = _job_health_entry(state, job.name)
    entry["last_success_at"] = datetime.now(tz=ET).isoformat(timespec="seconds")
    entry["consecutive_failures"] = 0
    log(f"ALREADY-CAPTURED {job.name}: a snapshot {age:.0f}m old already covers this window")


def sweep_missed(now: datetime, state: dict[str, Any]) -> None:

    for job in SCHEDULE:
        if not job.enabled:
            continue
        start = occurrence(job, now)
        key = f"{job.name}@{start.date().isoformat()}"
        if key in state["runs"] or (job.season_guarded and not season_active(start)):
            continue
        if predates_job(job, start):
            continue
        if now > start + timedelta(minutes=job.grace_minutes):
            if snapshot_in_window(job, start):
                state["runs"][key] = {
                    "status": "ALREADY-CAPTURED",
                    "window_start": start.isoformat(),
                    "note": "a snapshot exists inside this window (captured by another runner)",
                }
                continue
            if job.catch_up:
                run_job(job, start, state, catch_up=True)
                continue
            record: dict[str, Any] = {
                "status": "MISSED",
                "window_start": start.isoformat(),
            }
            if not prerequisites_satisfied(job, start, state):
                record["blocked_by"] = list(job.requires)
            state["runs"][key] = record
            entry = _job_health_entry(state, job.name)
            entry["missed_window_count"] = int(entry.get("missed_window_count", 0)) + 1
            blocked = (
                f"; prerequisites not successful: {', '.join(job.requires)}"
                if record.get("blocked_by")
                else ""
            )
            log(f"MISSED {job.name} (window {start.isoformat()} +{job.grace_minutes}m{blocked})")


def failure_detail(stderr: str | None, stdout_tail: str, *, limit: int = 300) -> str:

    text = (stderr or "").strip() or stdout_tail
    return text[-limit:]


def execute_job(command: list[str]) -> tuple[str, str]:

    try:
        no_window = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=1800,
            creationflags=no_window,
        )
        out = (proc.stdout or "").strip().splitlines()
        tail = out[-1][:300] if out else ""
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        detail = tail if proc.returncode == 0 else failure_detail(proc.stderr, tail)
    except subprocess.TimeoutExpired:
        status, detail = "FAIL(timeout)", "exceeded 1800s"
    except OSError as exc:
        status, detail = "FAIL(oserror)", str(exc)[:300]
    return status, detail


RECORDING_FLAGS: frozenset[str] = frozenset(
    {"--record-decisions", "--publish-card", "--write-graded"}
)


def dry_command(command: list[str]) -> list[str]:
    return [token for token in command if token not in RECORDING_FLAGS]


def has_ever_executed(state: dict[str, Any], job_name: str) -> bool:

    entry = state.get("job_health", {}).get(job_name) or {}
    if any(
        entry.get(field) for field in ("last_success_at", "last_failure_at", "last_manual_run_at")
    ):
        return True
    prefix = f"{job_name}@"
    return any(
        key.startswith(prefix)
        and str(record.get("status", "")) not in {"MISSED", "ALREADY-CAPTURED"}
        for key, record in state.get("runs", {}).items()
    )


def run_job_manually(job: Job, state: dict[str, Any], *, dry: bool = False) -> int:

    command = dry_command(list(job.command)) if dry else list(job.command)
    label = "MANUAL-DRY-RUN" if dry else "MANUAL-RUN"
    log(f"{label} {job.name}: {' '.join(command)}")
    status, detail = execute_job(command)
    log(f"{label} {status} {job.name}: {detail}")
    fresh = load_state()
    entry = _job_health_entry(fresh, job.name)
    entry["last_manual_run_at"] = datetime.now(tz=ET).isoformat(timespec="seconds")
    entry["last_manual_status"] = f"{status}{' (dry)' if dry else ''}"
    entry["last_manual_detail"] = detail[:300]
    if status != "OK":
        entry["last_error"] = detail[:300]
    save_state(fresh)
    _job_health_entry(state, job.name).update(entry)
    print(f"{label} {job.name}: {status}")
    if detail:
        print(detail)
    return 0 if status == "OK" else 1


DEFAULT_REHEARSE_SKIP_PREFIX = "lineups_,backup_data,verify_full_weekly"

REHEARSE_EXPECTED_FAILURES: frozenset[str] = frozenset({"weekly_lock"})


def _last_nonempty_line(text: str, *, limit: int = 120) -> str:

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:limit]
    return ""


def _rehearsal_jobs(*, skip_prefix: str, only_prefix: str) -> list[Job]:
    skip_prefixes = tuple(part for part in skip_prefix.split(",") if part)
    only_prefixes = tuple(part for part in only_prefix.split(",") if part)
    return [
        job
        for job in SCHEDULE
        if job.enabled
        and not job.name.startswith(skip_prefixes)
        and (not only_prefixes or job.name.startswith(only_prefixes))
    ]


def render_rehearsal_table(rows: list[dict[str, Any]]) -> str:
    header = f"{'job':<34} {'when':<11} {'seconds':>8}  {'verdict':<20} last line (<=120 chars)"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['job']:<34} {row['when']:<11} {row['seconds']:>7.1f}s  "
            f"{row['verdict']:<20} {row['last_line']}"
        )
    return "\n".join(lines)


def render_rehearsal_summary(rows: list[dict[str, Any]]) -> str:
    non_ok = [row for row in rows if row["code"] != 0]
    expected = [row for row in non_ok if row["job"] in REHEARSE_EXPECTED_FAILURES]
    unexpected = [row for row in non_ok if row["job"] not in REHEARSE_EXPECTED_FAILURES]

    lines = ["SUMMARY"]
    if expected:
        lines.append(
            "  expected on a non-lock day (weekly_lock's by-design refusal without "
            "--season/--week; still counted below):"
        )
        for row in expected:
            lines.append(f"    {row['job']}: {row['verdict']} -- {row['last_line']}")
    if unexpected:
        lines.append("  failed:")
        for row in unexpected:
            lines.append(f"    {row['job']}: {row['verdict']} -- {row['last_line']}")
    if not non_ok:
        lines.append("  (no failures)")
    lines.append(f"total {len(rows)} ok {len(rows) - len(non_ok)} failed {len(non_ok)}")
    return "\n".join(lines)


def rehearse_all(
    state: dict[str, Any],
    *,
    skip_prefix: str = DEFAULT_REHEARSE_SKIP_PREFIX,
    only_prefix: str = "",
    stop_on_fail: bool = False,
) -> int:

    jobs = _rehearsal_jobs(skip_prefix=skip_prefix, only_prefix=only_prefix)
    log(
        f"REHEARSE-ALL start ({len(jobs)} jobs; skip_prefix={skip_prefix!r} "
        f"only_prefix={only_prefix!r} stop_on_fail={stop_on_fail})"
    )

    rows: list[dict[str, Any]] = []
    for job in jobs:
        started = time.monotonic()
        code = run_job_manually(job, state, dry=True)
        elapsed = time.monotonic() - started
        entry = state.get("job_health", {}).get(job.name, {})
        verdict = str(entry.get("last_manual_status", "FAIL" if code else "OK"))
        last_line = _last_nonempty_line(str(entry.get("last_manual_detail", "")))
        rows.append(
            {
                "job": job.name,
                "when": f"{job.day} {job.at}",
                "seconds": elapsed,
                "code": code,
                "verdict": verdict,
                "last_line": last_line,
            }
        )
        if code != 0 and stop_on_fail:
            break

    print(render_rehearsal_table(rows))
    print()
    print(render_rehearsal_summary(rows))

    failed = sum(1 for row in rows if row["code"] != 0)
    log(f"REHEARSE-ALL end (total {len(rows)} ok {len(rows) - failed} failed {failed})")
    return 1 if failed else 0


def run_job(job: Job, start: datetime, state: dict[str, Any], *, catch_up: bool = False) -> None:
    key = f"{job.name}@{start.date().isoformat()}"
    previous = state["runs"].get(key)
    retries = int(previous.get("retries", 0)) + 1 if previous is not None else 0
    label = "RETRY" if previous is not None else ("CATCH-UP-RUN" if catch_up else "RUN")
    attempt_note = f", attempt {retries + 1}" if previous is not None else ""
    log(f"{label} {job.name} (window {start.isoformat()}{attempt_note})")
    status, detail = execute_job(list(job.command))
    if catch_up and status == "OK":
        status = "CAUGHT_UP"
    record: dict[str, Any] = {
        "status": status,
        "window_start": start.isoformat(),
        "ran_at": datetime.now(tz=ET).isoformat(timespec="seconds"),
    }
    if catch_up:
        record["caught_up"] = True
    if retries:
        record["retries"] = retries
    fresh = load_state()
    for target in (fresh, state):
        target.setdefault("runs", {})[key] = record
        entry = _job_health_entry(target, job.name)
        if status in {"OK", "CAUGHT_UP"}:
            entry["last_success_at"] = record["ran_at"]
            entry["consecutive_failures"] = 0
        else:
            entry["last_failure_at"] = record["ran_at"]
            entry["last_error"] = detail[:300]
            entry["consecutive_failures"] = int(entry.get("consecutive_failures", 0)) + 1
    save_state(fresh)
    _merge_health(state, fresh)
    log(f"{status} {job.name}: {detail}")


def _merge_health(state: dict[str, Any], fresh: dict[str, Any]) -> None:
    for name, entry in (fresh.get("job_health") or {}).items():
        _job_health_entry(state, name).update(entry)
    state.setdefault("runs", {}).update(fresh.get("runs") or {})


def prune(state: dict[str, Any], keep_days: int = 60) -> None:
    cutoff = (datetime.now(tz=ET) - timedelta(days=keep_days)).date().isoformat()
    state["runs"] = {
        key: value for key, value in state["runs"].items() if key.split("@")[-1] >= cutoff
    }


def build_health_report(now: datetime, state: dict[str, Any]) -> dict[str, Any]:

    heartbeat = read_heartbeat()
    heartbeat_age_seconds: float | None = None
    daemon_alive = False
    if heartbeat is not None:
        try:
            last_poll = datetime.fromisoformat(str(heartbeat.get("last_poll_at", "")))
        except ValueError:
            heartbeat = None
        else:
            heartbeat_age_seconds = (now - last_poll).total_seconds()
            daemon_alive = heartbeat_age_seconds <= HEARTBEAT_STALE_AFTER_SECONDS

    missed = [
        {"key": key, **record}
        for key, record in sorted(state.get("runs", {}).items())
        if record.get("status") == "MISSED"
    ]
    missed_unacknowledged = [row for row in missed if not row.get("acknowledged")]

    sources: list[SourceFreshness] = compute_freshness(
        SCHEDULE, repo_root=REPO, now=now, season_active=season_active
    )

    code_sha256_disk = compute_code_sha256()
    schedule_digest_disk = compute_schedule_digest()
    code_sha256_running = heartbeat.get("code_sha256") if heartbeat else None
    schedule_digest_running = heartbeat.get("schedule_digest") if heartbeat else None
    code_current = code_sha256_running is not None and code_sha256_running == code_sha256_disk
    schedule_current = (
        schedule_digest_running is not None and schedule_digest_running == schedule_digest_disk
    )

    ok = (
        daemon_alive
        and not missed_unacknowledged
        and not any_unexpected_missing(sources)
        and code_current
        and schedule_current
    )

    return {
        "checked_at": now.isoformat(timespec="seconds"),
        "heartbeat": {
            "path": str(HEARTBEAT_PATH),
            "exists": heartbeat is not None,
            "pid": heartbeat.get("pid") if heartbeat else None,
            "started_at": heartbeat.get("started_at") if heartbeat else None,
            "last_poll_at": heartbeat.get("last_poll_at") if heartbeat else None,
            "poll_seconds": heartbeat.get("poll_seconds") if heartbeat else None,
            "enabled_job_count": heartbeat.get("enabled_job_count") if heartbeat else None,
            "age_seconds": (
                round(heartbeat_age_seconds, 1) if heartbeat_age_seconds is not None else None
            ),
            "stale_after_seconds": HEARTBEAT_STALE_AFTER_SECONDS,
            "daemon_alive": daemon_alive,
        },
        "code_version": {
            "code_sha256_running": code_sha256_running,
            "code_sha256_disk": code_sha256_disk,
            "code_current": code_current,
            "schedule_digest_running": schedule_digest_running,
            "schedule_digest_disk": schedule_digest_disk,
            "schedule_current": schedule_current,
        },
        "missed": missed,
        "missed_unacknowledged": missed_unacknowledged,
        "job_health": state.get("job_health", {}),
        "sources": sources,
        "ok": ok,
    }


def _health_report_json(report: dict[str, Any]) -> dict[str, Any]:

    out = dict(report)
    out["sources"] = [source.as_dict() for source in report["sources"]]
    return out


def render_health(report: dict[str, Any]) -> str:
    hb = report["heartbeat"]
    lines = [
        f"capture scheduler health  ({report['checked_at']})",
        f"heartbeat: {hb['path']}",
    ]
    if hb["exists"]:
        lines.append(
            f"  pid {hb['pid']}  started {hb['started_at']}  last poll {hb['last_poll_at']} "
            f"({hb['age_seconds']}s ago)"
        )
        lines.append(
            f"  poll interval {hb['poll_seconds']}s  dead-after {hb['stale_after_seconds']}s  "
            f"verdict: {'ALIVE' if hb['daemon_alive'] else 'DEAD'}"
        )
        cv = report["code_version"]
        lines.append(
            "  code: "
            + (
                "current"
                if cv["code_current"]
                else (
                    f"STALE (daemon started with {_short_hash(cv['code_sha256_running'])}, "
                    f"disk is {_short_hash(cv['code_sha256_disk'])})"
                )
            )
        )
        lines.append(
            "  schedule: "
            + (
                "current"
                if cv["schedule_current"]
                else (
                    f"STALE (daemon started with {_short_hash(cv['schedule_digest_running'])}, "
                    f"disk is {_short_hash(cv['schedule_digest_disk'])})"
                )
            )
        )
        if not cv["code_current"] or not cv["schedule_current"]:
            lines.append(
                "  STALE -- restart to run the code/schedule on disk: "
                "scripts/stop_capture_scheduler.cmd then scripts/start_capture_scheduler.cmd"
            )
    else:
        lines.append(
            "  MISSING -- no heartbeat file. Either the daemon has never run under this "
            "code (restart it: scripts/stop_capture_scheduler.cmd then "
            "scripts/start_capture_scheduler.cmd), or it is dead."
        )
    lines.append("")
    lines.append(
        f"missed windows: {len(report['missed'])} "
        f"({len(report['missed_unacknowledged'])} unacknowledged)"
    )
    for row in report["missed"]:
        blocked = f"  blocked_by={row['blocked_by']}" if row.get("blocked_by") else ""
        acknowledged = row.get("acknowledged")
        if acknowledged:
            lines.append(
                f"     {row['key']}  window {row.get('window_start', '?')}  "
                f"MISSED (acknowledged: {acknowledged.get('reason', '')}){blocked}"
            )
        else:
            lines.append(f"  !! {row['key']}  window {row.get('window_start', '?')}{blocked}")
    lines.append("")
    lines.append("per-source freshness:")
    lines.append(render_table(report["sources"]))
    lines.append("")
    lines.append("OVERALL: " + ("OK" if report["ok"] else "FAIL"))
    return "\n".join(lines)


def describe_daemon(now: datetime) -> str:
    heartbeat = read_heartbeat()
    if heartbeat is None:
        return "NOT RUNNING (no heartbeat file) -- start scripts/start_capture_scheduler.cmd"
    try:
        last_poll = datetime.fromisoformat(str(heartbeat.get("last_poll_at", "")))
    except ValueError:
        return "NOT RUNNING (unreadable heartbeat) -- start scripts/start_capture_scheduler.cmd"
    age = int((now - last_poll).total_seconds())
    pid = heartbeat.get("pid")
    if age > HEARTBEAT_STALE_AFTER_SECONDS:
        return (
            f"NOT RUNNING (pid {pid}, last poll {age}s ago, stale after "
            f"{HEARTBEAT_STALE_AFTER_SECONDS}s) -- start scripts/start_capture_scheduler.cmd"
        )
    return f"RUNNING (pid {pid}, last poll {age}s ago, started {heartbeat.get('started_at')})"


def show_status(now: datetime, state: dict[str, Any]) -> None:
    print(f"capture scheduler status  ({now.isoformat(timespec='seconds')})")
    print(f"state: {STATE_PATH}")
    print(f"log:   {LOG_PATH}")
    print(f"daemon: {describe_daemon(now)}")
    print()
    print(f"{'job':<22} {'when':<14} {'grace':>6}  {'enabled':<8} last occurrence")
    for job in SCHEDULE:
        start = occurrence(job, now)
        key = f"{job.name}@{start.date().isoformat()}"
        record = state["runs"].get(key)
        if record:
            last = f"{record['status']} ({start.date()})"
            retries = record.get("retries")
            if retries:
                last += f" after {retries} {'retry' if retries == 1 else 'retries'}"
        elif predates_job(job, start):
            last = f"added {job.added_on} (window predates job)"
        elif job.season_guarded and not season_active(start):
            last = f"offseason ({start.date()})"
        elif now <= start + timedelta(minutes=job.grace_minutes) and not prerequisites_satisfied(
            job, start, state
        ):
            last = f"waiting for {', '.join(job.requires)}"
        elif now <= start + timedelta(minutes=job.grace_minutes):
            open_until = (start + timedelta(minutes=job.grace_minutes)).strftime("%H:%M")
            last = f"window OPEN until {open_until}"
        else:
            last = f"not run ({start.date()})"
        if job.enabled and not has_ever_executed(state, job.name):
            last += f" | NEVER RUN (exercise: --run-job {job.name})"
        print(
            f"{job.name:<22} {job.day} {job.at:<10} {job.grace_minutes:>5}m  "
            f"{'yes' if job.enabled else 'no':<8} {last}"
        )


def acknowledge_missed(key: str, reason: str, state: dict[str, Any]) -> int:

    record = state.get("runs", {}).get(key)
    if record is None:
        print(f"no state row for {key}; nothing to acknowledge", file=sys.stderr)
        return 1
    if record.get("status") != "MISSED":
        print(
            f"{key} is not MISSED (status={record.get('status')!r}); nothing to acknowledge",
            file=sys.stderr,
        )
        return 1
    record["acknowledged"] = {
        "reason": reason,
        "at": datetime.now(tz=ET).isoformat(timespec="seconds"),
    }
    save_state(state)
    print(f"acknowledged {key}: {reason}")
    return 0


def pid_is_alive(pid: int) -> bool:

    if sys.platform == "win32":
        import ctypes

        query_limited_information = 0x1000
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(query_limited_information, False, pid)
        if not handle:
            return False
        still_active = 259
        exit_code = ctypes.c_ulong()
        try:
            queried = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        finally:
            kernel32.CloseHandle(handle)
        return bool(queried) and exit_code.value == still_active
    try:  # type: ignore[unreachable]
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def daemon_is_running(now: datetime) -> tuple[bool, int | None]:

    heartbeat = read_heartbeat()
    if heartbeat is None:
        return False, None
    pid_raw = heartbeat.get("pid")
    pid = pid_raw if isinstance(pid_raw, int) else None
    try:
        last_poll = datetime.fromisoformat(str(heartbeat.get("last_poll_at", "")))
    except ValueError:
        return False, pid
    age = (now - last_poll).total_seconds()
    if age > HEARTBEAT_STALE_AFTER_SECONDS:
        return False, pid
    if pid is None:
        return False, pid
    return pid_is_alive(pid), pid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run what is due, then exit")
    parser.add_argument("--status", action="store_true", help="print schedule and exit")
    parser.add_argument(
        "--health",
        action="store_true",
        help=(
            "fail-visible summary (ENG-03): heartbeat age, daemon alive/dead, MISSED "
            "windows, per-source freshness. Exits non-zero on any failure; read-only."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="with --health, emit JSON instead of a table"
    )
    parser.add_argument(
        "--is-running",
        action="store_true",
        help=(
            "ENG-26: exit 0 (and print the pid) if a fresh heartbeat names a still-alive "
            "pid, exit 1 otherwise. Used by start_capture_scheduler.cmd to refuse a second "
            "daemon; read-only."
        ),
    )
    parser.add_argument(
        "--acknowledge-missed",
        metavar="JOB@DATE",
        help=(
            "ENG-26: record an acknowledgement for an existing MISSED state row (never "
            "deletes it; requires --reason). --health then reports it as "
            "'MISSED (acknowledged: <reason>)' and stops counting it toward the non-zero "
            "exit."
        ),
    )
    parser.add_argument(
        "--reason",
        help="required with --acknowledge-missed: why the MISSED window is acknowledged",
    )
    parser.add_argument(
        "--run-job",
        metavar="NAME",
        help=(
            "execute one SCHEDULE job now, ignoring its window, with exactly the argv "
            "the daemon runs; exit 0 on OK. Required for every job added or edited in a "
            "session (AGENTS.md). Records last_manual_run_at, never a dated runs row."
        ),
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help=(
            "with --run-job: drop only --record-decisions/--publish-card from the argv so "
            "a refresh/publish job can be exercised before its real week without writing "
            "a ledger or the card"
        ),
    )
    parser.add_argument(
        "--rehearse-all",
        action="store_true",
        help=(
            "execute every ENABLED job's real argv once via run_job_manually(dry=True), "
            "in SCHEDULE order (the one-shot replacement for hand-running --run-job NAME "
            "--dry job by job). Prints a table + SUMMARY; exit 1 if any job failed."
        ),
    )
    parser.add_argument(
        "--skip-prefix",
        default=DEFAULT_REHEARSE_SKIP_PREFIX,
        help=(
            "with --rehearse-all: comma-separated job-name prefixes to skip (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--only-prefix",
        default="",
        help="with --rehearse-all: comma-separated job-name prefixes to include (default: all)",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="with --rehearse-all: stop at the first non-OK job (default: keep going)",
    )
    args = parser.parse_args(argv)

    state = load_state()
    now = datetime.now(tz=ET)

    if args.is_running:
        alive, pid = daemon_is_running(now)
        if alive:
            print(f"running (pid {pid})")
            return 0
        print(f"not running (pid {pid})" if pid is not None else "not running (no heartbeat)")
        return 1

    if args.run_job:
        by_name = {job.name: job for job in SCHEDULE}
        if args.run_job not in by_name:
            print(f"unknown job {args.run_job!r}; see --status for names", file=sys.stderr)
            return 2
        return run_job_manually(by_name[args.run_job], state, dry=args.dry)
    if args.dry:
        print("--dry only applies with --run-job", file=sys.stderr)
        return 2

    if args.rehearse_all:
        return rehearse_all(
            state,
            skip_prefix=args.skip_prefix,
            only_prefix=args.only_prefix,
            stop_on_fail=args.stop_on_fail,
        )

    if args.acknowledge_missed:
        if not args.reason:
            print("--acknowledge-missed requires --reason", file=sys.stderr)
            return 2
        return acknowledge_missed(args.acknowledge_missed, args.reason, state)

    if args.health:
        report = build_health_report(now, state)
        if args.json:
            print(json.dumps(_health_report_json(report), indent=2, sort_keys=True))
        else:
            print(render_health(report))
        return 0 if report["ok"] else 1

    if args.status:
        show_status(now, state)
        return 0

    if args.once:
        for job, start in due_jobs(now, state):
            satisfied, age = already_captured(job, now)
            if satisfied and age is not None:
                record_already_captured(job, start, age, state)
                continue
            run_job(job, start, state)
        sweep_missed(now, state)
        prune(state)
        save_state(state)
        return 0

    log(
        f"scheduler started (poll {POLL_SECONDS}s, {sum(j.enabled for j in SCHEDULE)} enabled jobs)"
    )
    started_at = datetime.now(tz=ET)
    code_sha256 = compute_code_sha256()
    schedule_digest = compute_schedule_digest()
    keepalive_stop = threading.Event()
    start_heartbeat_keepalive(
        started_at=started_at,
        code_sha256=code_sha256,
        schedule_digest=schedule_digest,
        stop=keepalive_stop,
    )
    try:
        while True:
            try:
                now = datetime.now(tz=ET)
                write_heartbeat(
                    started_at=started_at,
                    now=now,
                    code_sha256=code_sha256,
                    schedule_digest=schedule_digest,
                )
                state = load_state()
                for job, start in due_jobs(now, state):
                    run_job(job, start, state)
                _merge_health(state, load_state())
                sweep_missed(now, state)
                prune(state)
                save_state(state)
            except Exception as exc:
                log(f"TICK-ERROR {type(exc).__name__}: {exc}")
            time.sleep(POLL_SECONDS)
    finally:
        keepalive_stop.set()


if __name__ == "__main__":
    sys.exit(main())
