"""In-repo scheduler for the recurring point-in-time captures.

Why this exists
---------------
The captures used to live in eight Windows Task Scheduler entries, all pointing
at the same two .ps1 files. Every property that matters -- when each runs, what
ran last, whether anything failed -- lived in opaque per-machine config outside
version control, and a failed run was silent. The owner ruled that mechanism
out. GitHub Actions was considered and ruled out too: this repository is public
and the odds feed is purchased, so captured quotes cannot land in it or in its
(publicly downloadable) workflow artifact store, and `odds-ingest` also needs
a local feature table that a fresh runner does not have.

So the schedule lives HERE, in `SCHEDULE` below: readable, diffable, reviewable,
and changed by editing code rather than by clicking through a GUI.

Windows, not instants
---------------------
Each job declares a target local time and a grace period, and the loop polls.
A job runs if now is inside [target, target + grace] and that occurrence has
not run yet. This is deliberately more forgiving than cron: a machine asleep at
the target instant still captures a few minutes late instead of losing the week,
and a job whose window closed unrun is recorded as **MISSED** rather than
vanishing. Cron cannot tell you what it failed to do; this can.

`grace` is therefore a real decision per job, not padding. `odds_sun_close`
targets 12:30 against 13:00 ET kickoffs, so its grace is short on purpose --
running it late would mislabel a post-kickoff quote as a closing line.

Usage
-----
    python scripts/capture_scheduler.py --status   # what is scheduled/ran/missed
    python scripts/capture_scheduler.py --once     # run whatever is due, exit
    python scripts/capture_scheduler.py            # poll forever (the daemon)

`--once` is what a Claude session or any ad-hoc invocation should call; it is
idempotent, so running it repeatedly costs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
UV = REPO / ".tools" / "uv.exe"
STATE_PATH = REPO / "data" / "scheduler_state.json"
LOG_PATH = REPO / "data" / "scheduler_log.txt"
POLL_SECONDS = 60

DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


@dataclass(frozen=True)
class Job:
    name: str
    day: str
    at: str  # "HH:MM" local (America/New_York)
    grace_minutes: int
    command: list[str]
    enabled: bool
    why: str
    # False = run year-round. The market captures do: books post week-1 lines
    # months early, and the existing Task Scheduler entries have been capturing
    # them all summer (rows=4544 on 2026-08-25, well before any kickoff).
    # Gating those on "is it the season" would silently drop early line
    # movement, so only the jobs that genuinely need a live week are guarded.
    season_guarded: bool = True
    # Directory of timestamped snapshots this job produces, plus how recent a
    # snapshot has to be for the job to consider its work already done. This is
    # what makes the job safe to double-schedule: during the migration the
    # retired Windows tasks and this scheduler both target the same captures,
    # and whichever fires first satisfies the other. It also protects against a
    # second scheduler copy, a manual run, or a `--once` invocation landing in
    # the same window. Each threshold must stay well under the gap to this
    # job's nearest sibling (odds: 225 min at Sun 12:30 -> 16:15; injuries:
    # 16.5 h at Fri 17:30 -> Sat 10:00) or a real capture would be skipped.
    dedupe_dir: str = ""
    dedupe_minutes: int = 0
    # ISO date this job was ADDED to the schedule. Windows that closed before
    # it are neither run nor branded MISSED -- the job did not exist to miss
    # them. Without this, adding any job on a Thursday immediately fabricates a
    # MISSED row for the Sunday window that closed before it was written, and a
    # MISSED row is the one signal that means captures are being LOST
    # PERMANENTLY (AGENTS.md tells every session to restart the scheduler on
    # seeing one). `snapshot_in_window` already protects the jobs that produce
    # timestamped snapshots; this protects the ones that do not, and every
    # future job added to SCHEDULE. Empty = no backfill guard, the behaviour
    # every pre-existing job was written under.
    added_on: str = ""
    # True only for a job whose command is safe to run late: idempotent, and
    # not a point-in-time capture that a delayed run would mislabel (a closing
    # line captured after kickoff is not a closing line; an off-device mirror
    # copied twelve hours late is still a correct mirror). When a catch_up
    # job's window closes with nothing run and no snapshot to show for it,
    # `sweep_missed` runs it right there instead of writing MISSED, and the
    # occurrence is recorded CAUGHT_UP -- a status that is honest about the
    # original miss (unlike OK, which would read as on time) without raising
    # the one alarm that is supposed to mean data is gone for good (MISSED).
    # Default False keeps every point-in-time job's behaviour byte-for-byte
    # unchanged: a missed window still writes MISSED and nothing runs late.
    catch_up: bool = False
    # Same-day jobs named here must have completed successfully before this
    # job becomes due. This is intentionally scheduler state (not wall-clock
    # inference): a paper forecast must never assume its opener capture landed.
    requires: tuple[str, ...] = ()


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
    str(REPO / "scripts" / "build_week_lineups.py"),
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


# The schedule. Times are America/New_York, matching the retired Task Scheduler
# entries exactly so the migration changes the mechanism and not the cadence.
SCHEDULE: tuple[Job, ...] = (
    # --- Projected lineup snapshots -----------------------------------------
    # Depth charts are mutable well before kickoff. One daily snapshot keeps
    # Monday/Tuesday planning useful and gives later injury/inactives feeds a
    # current-roster context without checking data into Git.
    *(
        Job(
            f"lineups_{day}",
            day,
            "12:00",
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
    # --- The Odds API point-in-time captures (3 requests each) ---------------
    Job(
        "odds_tue_open",
        "tue",
        "09:00",
        180,
        _ps("odds_capture.ps1"),
        True,
        "Tuesday opener: the grade the pool settles on. Wide grace -- the "
        "opener moves slowly and a late capture is still an opener.",
        season_guarded=False,
        dedupe_dir="data/market/raw",
        dedupe_minutes=90,
    ),
    Job(
        "weekly_lock",
        "tue",
        "09:15",
        120,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "scheduled_weekly_lock.py"),
        ],
        True,
        "Lock-day paper forecast. Runs only for an actual scheduled game week, "
        "after the Tuesday opener succeeds, and closes at 11:15 so the documented "
        "15-minute budget finishes before the 11:30 publication target.",
        added_on="2026-09-02",
        requires=("odds_tue_open",),
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
    # --- Action Network public betting percentages ---------------------------
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
    # --- NFL.com injury report revision stream -------------------------------
    # A living page teams overwrite Wed->Fri; the revisions cannot be recovered
    # retroactively. The FRIDAY run is the one the frozen challenger rule
    # consumes (it needs a page fetched at or after Friday 16:00 ET).
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
    # --- Licensed replacement injury report revision stream -----------------
    # Sportradar documents a four-hour endpoint cache. The command derives the
    # live REG week from the local schedule, requires every slate team, and
    # records the capture time as availability. Jobs remain dormant unless the
    # scheduler process has the provider credential; a missing secret must not
    # create recurring failures or imply that a capture occurred.
    *(
        _sportradar_injury_job(day, at, report)
        for day, at, report in (
            ("wed", "17:30", "weekly practice report"),
            ("thu", "17:30", "weekly practice report"),
            ("fri", "17:30", "final game-status report"),
            ("sat", "10:00", "post-Friday fallback"),
        )
    ),
    # --- Late-week pick refresh (docs/late_week_refresh.md cadence) ----------
    # These are the ONLY recording path for model_only_refresh_incumbent and
    # injury_signal_refresh_tilt. Over-running is explicitly harmless: "a pass
    # that finds nothing changed writes nothing".
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
    # --- Off-device data mirror ---------------------------------------------
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
    # --- Player-arrests point-in-time snapshot -------------------------------
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
        season_guarded=False,  # a running archive, not tied to a game week;
        # an offseason gap would show up as a stale snapshot once weekly
        # picks resume, and the cost is a few minutes once a week.
        dedupe_dir="data/raw/player_arrests",
        dedupe_minutes=240,
        added_on="2026-09-01",
        catch_up=True,
    ),
    # --- Official game-day inactives (WP17) -----------------------------------
    # docs/inactives_channel.md Section 2 (measured this session) computes T-90
    # ("official inactives instant" = kickoff - 90 minutes) against each slot's
    # own pick_refresh deadline for every 2026 REG game; Section 6 proposes the
    # capture windows below from that arithmetic. All seven are point-in-time
    # captures (catch_up=False -- a missed inactives window cannot be caught up
    # after the fact, unlike backup_data/player_arrests_tue) and season_guarded
    # (no REG game, no inactive list to capture). dedupe_dir points at
    # data/players/inactives, NOT the doc's originally proposed
    # data/raw/nflcom_inactives: the actual capture
    # (src/nfl_ats/inactives_capture.py) writes snapshots under
    # data/players/inactives/<UTC ts>/ per the WP17 task spec that built it, so
    # the dedupe target follows the real write location.
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
    # --- Post-inactives challenger refreshes (POL-11 / WP41) -----------------
    # Each pass begins five minutes after its capture window closes. The grace
    # ends ten minutes before the earliest deadline that capture can cover, so
    # an on-time scheduler run has a bounded decision-time path from snapshot
    # to refresh. These passes record the inactives challenger separately;
    # record_inactives_refresh_overlay consumes the RefreshResult read-only and
    # cannot change the played pick or its revision ledger.
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
    # --- Weekly officiating-crew assignments (WP22) --------------------------
    # Feeds a prospective join to the referee-battery/penalty-crew-tendencies
    # crew traits (docs/referee_battery.md, docs/penalty_crew_tendencies.md;
    # docs/referee_assignments_capture.md is this job's own predeclaration/
    # survey). Football Zebras' weekly post is the only public pregame source
    # found for the UPCOMING week's officiating assignment (no independent
    # second source exists: operations.nfl.com carries no weekly-assignments
    # page at all, measured), and its publish time is NOT fixed: measured
    # across 10 sampled 2025 weeks (docs/referee_assignments_capture.md
    # Section 2), timestamps range Mon 12:44 ET (Week 18's compressed finale
    # schedule) through Wed 12:42 ET (Weeks 8/9), with most weeks landing Tue
    # 16:53-21:27 ET -- NEVER measured before Tuesday afternoon, so this
    # capture cannot feed the Tuesday-lock/opener card, only a later-week
    # refresh (see that doc's Section 2 for the exact per-cell overlay this
    # would need to become a prospective challenger). Wed 15:00 ET clears
    # every measured 2025 publish time by >2h. catch_up=True for the same
    # reason as player_arrests_tue/backup_data (not the odds-close reasoning):
    # a late capture is still a valid, un-mislabelled snapshot -- assignments
    # do not go stale the way a post-kickoff "closing line" would -- and
    # src/nfl_ats/referee_assignments_capture.py is idempotent by
    # construction (every run writes a fresh UTC-stamped snapshot dir under
    # data/players/referee_assignments/ and never mutates an older one).
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
)


SNAPSHOT_NAME = re.compile(r"^(\d{8}T\d{6}Z)$")


def newest_snapshot_age_minutes(relative_dir: str, now: datetime) -> float | None:
    """Age of the newest UTC-stamped snapshot directory, in minutes.

    Reads the directory NAME rather than filesystem mtime: the name is the
    capture instant the rest of the project treats as authoritative, and mtime
    changes for reasons that have nothing to do with when data was captured
    (a backup restore, a file copy, an antivirus touch).
    """

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
    """Did this window close before the job was added to the schedule?

    Kept separate from `snapshot_in_window` because it answers a different
    question: not "did something else capture this?" but "was there anything
    here to capture it?". A window older than the job is not a gap in coverage,
    and must never be reported as one.
    """

    if not job.added_on:
        return False
    return start.date() < date.fromisoformat(job.added_on)


def snapshot_in_window(job: Job, start: datetime) -> bool:
    """Did a snapshot land inside this job's window, whoever produced it?

    Used before declaring a past window MISSED. Without it, the first run of
    this scheduler would brand every window that the (still-live) Windows tasks
    captured perfectly well as MISSED -- and since a MISSED row is the signal
    that captures are being LOST, a wall of false ones on day one would train
    every future reader to ignore the one alarm that matters.
    """

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
    """Is ``when`` inside the playing season?

    Every job here exists to capture something about a nearby game, so outside
    the season they must neither fire nor accumulate MISSED noise -- a
    scheduler that cries wolf all summer gets ignored in November.

    True when an in-contract game (REG/WC/DIV/CON/SB) falls anywhere in the
    span from ten days before to three days after ``when``. In season that is
    always satisfied by the
    PREVIOUS week's games (never more than seven days back), so mid-season
    jobs are unconditionally live; the three-day lookahead is what switches
    the scheduler on for the run-up to week 1, and the ten-day lookback is
    what keeps it on through the last week's aftermath before it goes quiet.
    Verified at the boundaries (measured 2026-08-25): off on 2026-09-02, on
    from 2026-09-08 through 2027-01-20, off again by 2027-02-15.
    """

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
        return True  # no schedule locally: never silently suppress a capture
    day = when.date()
    return any((day + timedelta(days=offset)) in known for offset in range(-10, 4))


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"runs": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"runs": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_PATH)


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=ET).isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    # The daemon runs headless (hidden console); nobody watches this stdout.
    # The file above is the record.
    print(line, flush=True)


def occurrence(job: Job, now: datetime) -> datetime:
    """The most recent scheduled instant for this job at or before ``now``."""

    target_h, target_m = (int(part) for part in job.at.split(":"))
    days_back = (now.weekday() - DAYS[job.day]) % 7
    day = (now - timedelta(days=days_back)).date()
    stamp = datetime.combine(day, datetime.min.time(), tzinfo=ET).replace(
        hour=target_h, minute=target_m
    )
    if stamp > now:
        stamp -= timedelta(days=7)
    return stamp


def due_jobs(now: datetime, state: dict[str, Any]) -> list[tuple[Job, datetime]]:
    due: list[tuple[Job, datetime]] = []
    for job in SCHEDULE:
        if not job.enabled:
            continue
        start = occurrence(job, now)
        key = f"{job.name}@{start.date().isoformat()}"
        if key in state["runs"] or (job.season_guarded and not season_active(start)):
            continue
        if predates_job(job, start):
            continue
        if not prerequisites_satisfied(job, start, state):
            continue
        if start <= now <= start + timedelta(minutes=job.grace_minutes):
            due.append((job, start))
    return due


def prerequisites_satisfied(job: Job, start: datetime, state: dict[str, Any]) -> bool:
    """Require successful same-date scheduler records for declared dependencies."""

    accepted = {"OK", "ALREADY-CAPTURED"}
    for required_name in job.requires:
        record = state["runs"].get(f"{required_name}@{start.date().isoformat()}", {})
        if record.get("status") not in accepted:
            return False
    return True


def record_already_captured(job: Job, start: datetime, age: float, state: dict[str, Any]) -> None:
    """Mark this occurrence satisfied by a capture something else already took."""

    state["runs"][f"{job.name}@{start.date().isoformat()}"] = {
        "status": "ALREADY-CAPTURED",
        "window_start": start.isoformat(),
        "newest_snapshot_age_minutes": round(age, 1),
    }
    log(f"ALREADY-CAPTURED {job.name}: a snapshot {age:.0f}m old already covers this window")


def sweep_missed(now: datetime, state: dict[str, Any]) -> None:
    """Record windows that closed without running, so a gap is visible.

    A `catch_up` job never gets the MISSED verdict: its window closing unrun
    triggers a run right here, on whichever tick first notices it (the next
    `--once` or the next poll of the daemon), and the occurrence is recorded
    CAUGHT_UP instead. `run_job` still writes the state key either way, so
    the `key in state["runs"]` check above makes a second sweep a no-op --
    the same mechanism that already stops every other status from re-firing
    guarantees this can never run twice for one occurrence.
    """

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
            blocked = (
                f"; prerequisites not successful: {', '.join(job.requires)}"
                if record.get("blocked_by")
                else ""
            )
            log(f"MISSED {job.name} (window {start.isoformat()} +{job.grace_minutes}m{blocked})")


def run_job(job: Job, start: datetime, state: dict[str, Any], *, catch_up: bool = False) -> None:
    log(f"{'CATCH-UP-RUN' if catch_up else 'RUN'} {job.name} (window {start.isoformat()})")
    try:
        # CREATE_NO_WINDOW: the daemon's console is hidden (or absent), and
        # without this flag a child could allocate and flash a visible one.
        no_window = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.run(
            job.command,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=1800,
            creationflags=no_window,
        )
        out = (proc.stdout or "").strip().splitlines()
        tail = out[-1][:200] if out else ""
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        detail = tail if proc.returncode == 0 else (proc.stderr or tail)[:300]
    except subprocess.TimeoutExpired:
        status, detail = "FAIL(timeout)", "exceeded 1800s"
    except OSError as exc:
        status, detail = "FAIL(oserror)", str(exc)[:300]
    if catch_up and status == "OK":
        # Honest about the original miss: not OK (which reads as on time),
        # not MISSED (data was not lost -- it just landed late).
        status = "CAUGHT_UP"
    record: dict[str, Any] = {
        "status": status,
        "window_start": start.isoformat(),
        "ran_at": datetime.now(tz=ET).isoformat(timespec="seconds"),
    }
    if catch_up:
        record["caught_up"] = True
    state["runs"][f"{job.name}@{start.date().isoformat()}"] = record
    save_state(state)
    log(f"{status} {job.name}: {detail}")


def prune(state: dict[str, Any], keep_days: int = 60) -> None:
    cutoff = (datetime.now(tz=ET) - timedelta(days=keep_days)).date().isoformat()
    state["runs"] = {
        key: value for key, value in state["runs"].items() if key.split("@")[-1] >= cutoff
    }


def show_status(now: datetime, state: dict[str, Any]) -> None:
    print(f"capture scheduler status  ({now.isoformat(timespec='seconds')})")
    print(f"state: {STATE_PATH}")
    print(f"log:   {LOG_PATH}")
    print()
    print(f"{'job':<22} {'when':<14} {'grace':>6}  {'enabled':<8} last occurrence")
    for job in SCHEDULE:
        start = occurrence(job, now)
        key = f"{job.name}@{start.date().isoformat()}"
        record = state["runs"].get(key)
        if record:
            last = f"{record['status']} ({start.date()})"
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
        print(
            f"{job.name:<22} {job.day} {job.at:<10} {job.grace_minutes:>5}m  "
            f"{'yes' if job.enabled else 'no':<8} {last}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run what is due, then exit")
    parser.add_argument("--status", action="store_true", help="print schedule and exit")
    args = parser.parse_args(argv)

    state = load_state()
    now = datetime.now(tz=ET)

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
    while True:
        try:
            now = datetime.now(tz=ET)
            state = load_state()
            for job, start in due_jobs(now, state):
                run_job(job, start, state)
            sweep_missed(now, state)
            prune(state)
            save_state(state)
        except Exception as exc:
            log(f"TICK-ERROR {type(exc).__name__}: {exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
