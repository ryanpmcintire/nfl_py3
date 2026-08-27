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


# The schedule. Times are America/New_York, matching the retired Task Scheduler
# entries exactly so the migration changes the mechanism and not the cadence.
SCHEDULE: tuple[Job, ...] = (
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
        True,
        "Wednesday practice report.",
        dedupe_dir="data/raw/nflcom_injuries",
        dedupe_minutes=300,
    ),
    Job(
        "injuries_thu",
        "thu",
        "17:30",
        240,
        INJURY_CAPTURE,
        True,
        "Thursday practice report.",
        dedupe_dir="data/raw/nflcom_injuries",
        dedupe_minutes=300,
    ),
    Job(
        "injuries_fri",
        "fri",
        "17:30",
        240,
        INJURY_CAPTURE,
        True,
        "FRIDAY FINAL -- the page nflcom_friday_refresh_out2_starters_v1 reads. "
        "17:30 clears the rule's Friday-16:00-ET floor by 90 minutes.",
        dedupe_dir="data/raw/nflcom_injuries",
        dedupe_minutes=300,
    ),
    Job(
        "injuries_sat",
        "sat",
        "10:00",
        240,
        INJURY_CAPTURE,
        True,
        "Final state before the Sunday slate.",
        dedupe_dir="data/raw/nflcom_injuries",
        dedupe_minutes=300,
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
        [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "backup_data.py")],
        True,
        "Weekly off-device mirror to E:. Runs AFTER the week's last capture "
        "(refresh_sun 10:00, odds_sun_late 16:15) so a week's point-in-time "
        "data is never left unmirrored over the following week. Needs no "
        "dedupe guard: backup_data.py is idempotent by construction -- a "
        "second run finds every file size- and mtime-identical and copies "
        "nothing (measured 2026-08-27: 14.6s for a no-op pass over 42,839 "
        "files), so double-scheduling costs seconds, not a re-copy. If the "
        "mirror drive is absent the run fails loudly and the next one resumes; "
        "a partial copy is not a corrupt state.",
        season_guarded=False,
        added_on="2026-08-27",
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

    True when a REG game falls anywhere in the span from ten days before to
    three days after ``when``. In season that is always satisfied by the
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
            sched = sched.loc[sched["game_type"].astype(str).eq("REG")]
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
        if start <= now <= start + timedelta(minutes=job.grace_minutes):
            due.append((job, start))
    return due


def record_already_captured(job: Job, start: datetime, age: float, state: dict[str, Any]) -> None:
    """Mark this occurrence satisfied by a capture something else already took."""

    state["runs"][f"{job.name}@{start.date().isoformat()}"] = {
        "status": "ALREADY-CAPTURED",
        "window_start": start.isoformat(),
        "newest_snapshot_age_minutes": round(age, 1),
    }
    log(f"ALREADY-CAPTURED {job.name}: a snapshot {age:.0f}m old already covers this window")


def sweep_missed(now: datetime, state: dict[str, Any]) -> None:
    """Record windows that closed without running, so a gap is visible."""

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
            state["runs"][key] = {"status": "MISSED", "window_start": start.isoformat()}
            log(f"MISSED {job.name} (window {start.isoformat()} +{job.grace_minutes}m)")


def run_job(job: Job, start: datetime, state: dict[str, Any]) -> None:
    log(f"RUN {job.name} (window {start.isoformat()})")
    try:
        proc = subprocess.run(job.command, cwd=REPO, capture_output=True, text=True, timeout=1800)
        out = (proc.stdout or "").strip().splitlines()
        tail = out[-1][:200] if out else ""
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        detail = tail if proc.returncode == 0 else (proc.stderr or tail)[:300]
    except subprocess.TimeoutExpired:
        status, detail = "FAIL(timeout)", "exceeded 1800s"
    except OSError as exc:
        status, detail = "FAIL(oserror)", str(exc)[:300]
    state["runs"][f"{job.name}@{start.date().isoformat()}"] = {
        "status": status,
        "window_start": start.isoformat(),
        "ran_at": datetime.now(tz=ET).isoformat(timespec="seconds"),
    }
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
