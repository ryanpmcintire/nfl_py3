"""Tests for the backfill guard on newly-added scheduler jobs.

A MISSED row is the scheduler's one alarm that captures are being LOST
PERMANENTLY, and AGENTS.md instructs every session to restart the scheduler on
seeing one. Adding a job to SCHEDULE used to fabricate exactly that alarm for
every window that closed before the job was written, which is the "wall of
false ones" the `snapshot_in_window` docstring already warns trains readers to
ignore the real thing. These tests pin the guard that prevents it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.capture_scheduler as capture_scheduler
from scripts.capture_scheduler import ET, Job


def make_job(**overrides: Any) -> Job:
    defaults: dict[str, Any] = {
        "name": "demo_job",
        "day": "sun",
        "at": "22:00",
        "grace_minutes": 300,
        "command": ["cmd.exe", "/c", "echo"],
        "enabled": True,
        "why": "demo",
        "season_guarded": False,
    }
    defaults.update(overrides)
    return Job(**defaults)


def empty_state() -> dict[str, Any]:
    return {"runs": {}}


# A Thursday, four days after the Sunday 22:00 window has closed.
THURSDAY = datetime(2026, 8, 27, 17, 0, tzinfo=ET)


def test_window_that_closed_before_the_job_existed_is_not_missed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(added_on="2026-08-27")
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    assert state["runs"] == {}


def test_the_same_window_is_missed_without_the_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard is load-bearing, not decorative -- drop it and the row appears."""
    job = make_job()  # no added_on
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    assert [record["status"] for record in state["runs"].values()] == ["MISSED"]


def test_a_window_on_the_day_the_job_was_added_still_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`added_on` suppresses windows BEFORE it, not the one it lands on."""
    job = make_job(added_on="2026-08-23")
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    assert [record["status"] for record in state["runs"].values()] == ["MISSED"]


def test_predates_job_is_inert_for_every_pre_existing_job() -> None:
    """Jobs written before this field must behave exactly as they did."""
    for job in capture_scheduler.SCHEDULE:
        if job.added_on:
            continue
        start = capture_scheduler.occurrence(job, THURSDAY)
        assert not capture_scheduler.predates_job(job, start)


def test_a_predating_window_is_never_due_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    job = make_job(added_on="2026-08-27", day="thu", at="16:00", grace_minutes=300)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    # Inside the grace window, so only `added_on` can hold it back.
    inside_window = datetime(2026, 8, 27, 17, 0, tzinfo=ET)
    assert capture_scheduler.due_jobs(inside_window, empty_state())

    later = make_job(added_on="2026-08-28", day="thu", at="16:00", grace_minutes=300)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (later,))
    assert not capture_scheduler.due_jobs(inside_window, empty_state())


def test_status_labels_a_predating_window_instead_of_calling_it_unrun(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job = make_job(added_on="2026-08-27")
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))

    capture_scheduler.show_status(THURSDAY, empty_state())

    out = capsys.readouterr().out
    assert "window predates job" in out
    assert "not run" not in out


def test_the_real_backup_job_is_guarded_and_runs_after_the_weeks_last_capture() -> None:
    """The job this guard was introduced for, pinned where it matters."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    backup = schedule["backup_data"]

    assert backup.added_on == "2026-08-27"
    assert not backup.season_guarded  # data accrues year-round
    sunday = datetime(2026, 8, 30, 23, 0, tzinfo=ET)
    latest_capture = max(
        capture_scheduler.occurrence(job, sunday)
        for job in capture_scheduler.SCHEDULE
        if job.name != "backup_data"
        and capture_scheduler.occurrence(job, sunday).date() == sunday.date()
    )
    assert capture_scheduler.occurrence(backup, sunday) > latest_capture


def test_backup_job_finishes_well_inside_the_subprocess_timeout() -> None:
    """run_job kills at 1800s. A no-op incremental pass measured 14.6s on
    2026-08-27 over 42,839 files; the margin is what makes this safe to run
    from a session's `--once`, so the command must stay the cheap one."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    command = schedule["backup_data"].command

    assert any(part.endswith("backup_data.py") for part in command)
    assert "--include-artifacts" in command
    # No --verify-all: that re-hashes 3.5 GB and would blow the timeout.
    assert "--verify-all" not in command


def test_a_missed_catch_up_job_runs_once_on_the_next_tick_and_shows_caught_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """catch_up=True turns a closed, unrun window into a late run, not a loss."""
    job = make_job(catch_up=True, command=["cmd.exe", "/c", "echo", "caught-up-ok"])
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    key = f"{job.name}@{capture_scheduler.occurrence(job, THURSDAY).date().isoformat()}"
    record = state["runs"][key]
    assert record["status"] == "CAUGHT_UP"
    assert record["caught_up"] is True
    # The state file run_job() wrote must be the tmp one, never the real one.
    assert (tmp_path / "state.json").is_file()


def test_a_non_catch_up_job_still_shows_missed_and_does_not_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default (catch_up=False) is byte-for-byte the pre-existing behaviour."""
    job = make_job(catch_up=False, command=["cmd.exe", "/c", "echo", "should-not-run"])
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    key = f"{job.name}@{capture_scheduler.occurrence(job, THURSDAY).date().isoformat()}"
    assert state["runs"][key]["status"] == "MISSED"
    # sweep_missed never runs a job -- confirm no state file was ever written
    # (run_job is the only thing in this module that calls save_state).
    assert not (tmp_path / "state.json").exists()


def test_a_catch_up_job_never_runs_twice_for_one_occurrence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(catch_up=True, command=["cmd.exe", "/c", "echo", "once-only"])
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)
    key = f"{job.name}@{capture_scheduler.occurrence(job, THURSDAY).date().isoformat()}"
    first_ran_at = state["runs"][key]["ran_at"]

    # A later tick, same occurrence still current -- must be a no-op.
    later = THURSDAY + timedelta(hours=6)
    capture_scheduler.sweep_missed(later, state)

    assert state["runs"][key]["ran_at"] == first_ran_at
    assert state["runs"][key]["status"] == "CAUGHT_UP"


def test_status_renders_caught_up_distinctly_from_ok_and_missed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job = make_job(catch_up=True)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    start = capture_scheduler.occurrence(job, THURSDAY)
    key = f"{job.name}@{start.date().isoformat()}"
    state = {
        "runs": {
            key: {
                "status": "CAUGHT_UP",
                "window_start": start.isoformat(),
                "ran_at": "2026-08-27T12:00:00-04:00",
                "caught_up": True,
            }
        }
    }

    capture_scheduler.show_status(THURSDAY, state)

    out = capsys.readouterr().out
    assert "CAUGHT_UP" in out
    assert "not run" not in out


def test_player_arrests_tue_is_idempotent_catch_up_and_added_on_guarded() -> None:
    """Pins the properties that justify catch_up=True on this specific job."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    arrests = schedule["player_arrests_tue"]

    assert arrests.catch_up is True
    assert arrests.added_on == "2026-09-01"
    assert arrests.command[-1] == "ingest-player-arrests"


def test_player_arrests_tue_window_closes_before_the_tuesday_opener() -> None:
    """Feeds the Tuesday publish; must finish well ahead of odds_tue_open."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    arrests = schedule["player_arrests_tue"]
    opener = schedule["odds_tue_open"]
    tuesday = datetime(2026, 9, 1, 12, 0, tzinfo=ET)

    arrests_close = capture_scheduler.occurrence(arrests, tuesday) + timedelta(
        minutes=arrests.grace_minutes
    )
    opener_start = capture_scheduler.occurrence(opener, tuesday)
    assert arrests_close < opener_start


def test_grace_window_does_not_collide_with_the_next_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backup running long must not still be inside another job's window."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    backup = schedule["backup_data"]
    sunday = datetime(2026, 8, 30, 23, 0, tzinfo=ET)
    close = capture_scheduler.occurrence(backup, sunday) + timedelta(minutes=backup.grace_minutes)

    monday_mnf = capture_scheduler.occurrence(schedule["odds_mon_mnf"], sunday + timedelta(days=2))
    assert close < monday_mnf


# --- Official inactives capture (WP17) --------------------------------------
# docs/inactives_channel.md Section 2 (measured 2026-09-01) derives T-90
# ("official inactives instant" = kickoff - 90 minutes) against each slot's
# own pick_refresh deadline; Section 6 proposes the capture windows these
# pins check. See scripts/capture_scheduler.py's inactives_* Job comments for
# the full per-row derivation.

_INACTIVES_JOB_NAMES = (
    "inactives_sun_early",
    "inactives_sun_late",
    "inactives_thu_afternoon_early",
    "inactives_thu_afternoon_late",
    "inactives_thu_primetime",
    "inactives_sat_early",
    "inactives_sat_late",
)


def test_inactives_jobs_are_point_in_time_and_added_this_session() -> None:
    """A missed inactives window cannot be caught up after the fact (unlike
    backup_data/player_arrests_tue -- an inactive list not captured before
    kickoff is simply gone), so every row must keep catch_up=False. Every row
    must also carry added_on so past Sun/Thu/Sat windows that closed before
    this session are never retroactively branded MISSED (see predates_job)."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    for name in _INACTIVES_JOB_NAMES:
        job = schedule[name]
        assert job.catch_up is False
        assert job.added_on == "2026-09-01"
        assert job.season_guarded is True
        # Actual write location is data/players/inactives (WP17 task spec),
        # NOT docs/inactives_channel.md Section 6's originally proposed
        # data/raw/nflcom_inactives -- the dedupe target must match reality.
        assert job.dedupe_dir == "data/players/inactives"
        assert job.dedupe_minutes == 60
        slot = name.removeprefix("inactives_")
        assert job.command[-2:] == ["--slot", slot]


def test_inactives_job_names_are_unique_across_the_whole_schedule() -> None:
    """Job.name doubles as the run-state key (f'{name}@{date}'); two same-named
    jobs landing on the same date would collide and the later one would
    silently no-op against the earlier one's already-written state entry --
    exactly the trap docs/inactives_channel.md Section 6's literal
    'inactives_thu_afternoon' (proposed for two different times) would have
    been."""
    names = [job.name for job in capture_scheduler.SCHEDULE]
    assert len(names) == len(set(names))


def test_inactives_sun_early_closes_well_before_the_thirteen_hundred_slate() -> None:
    """T-90 for a 13:00 ET kickoff is 11:30 ET; the job fires at 11:35 (five
    minutes after true T-90, giving the source a moment to publish) and its
    grace window must close with real margin before kickoff."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["inactives_sun_early"]
    sunday = datetime(2026, 9, 13, 20, 0, tzinfo=ET)
    close = capture_scheduler.occurrence(job, sunday) + timedelta(minutes=job.grace_minutes)
    assert close.strftime("%H:%M") == "11:50"
    kickoff = datetime(2026, 9, 13, 13, 0, tzinfo=ET)
    assert close < kickoff


def test_inactives_sun_late_closes_before_the_sunday_pick_lock() -> None:
    """This slot's binding deadline is the week's fixed Sunday 16:00 ET pick
    lock, not each game's own (16:05-17:00 ET) kickoff -- the job must close
    well before 16:00 regardless of that week's exact late-slate kickoffs."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["inactives_sun_late"]
    sunday = datetime(2026, 9, 13, 20, 0, tzinfo=ET)
    close = capture_scheduler.occurrence(job, sunday) + timedelta(minutes=job.grace_minutes)
    sunday_lock = datetime(2026, 9, 13, 16, 0, tzinfo=ET)
    assert close < sunday_lock
    assert (sunday_lock - close) == timedelta(minutes=65)


def test_inactives_thu_cluster_has_three_non_colliding_occurrences() -> None:
    """Three Thu jobs approximate T-90 for the historically observed 13:00 /
    16:30 / 20:15-20:35 ET kickoff clusters (one fixed time cannot cover all
    of them); their windows must not overlap each other."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    thursday = datetime(2026, 9, 10, 23, 0, tzinfo=ET)
    early = schedule["inactives_thu_afternoon_early"]
    late = schedule["inactives_thu_afternoon_late"]
    primetime = schedule["inactives_thu_primetime"]

    early_close = capture_scheduler.occurrence(early, thursday) + timedelta(
        minutes=early.grace_minutes
    )
    late_start = capture_scheduler.occurrence(late, thursday)
    late_close = capture_scheduler.occurrence(late, thursday) + timedelta(
        minutes=late.grace_minutes
    )
    primetime_start = capture_scheduler.occurrence(primetime, thursday)

    assert early_close < late_start
    assert late_close < primetime_start


def test_inactives_sat_cluster_has_two_non_colliding_occurrences() -> None:
    """Same Option-A gap as Thu, smaller in scope: a 17:00 ET and a 20:20 ET
    Sat kickoff each need their own T-90 job; their windows must not overlap."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    saturday = datetime(2026, 9, 12, 23, 0, tzinfo=ET)
    early = schedule["inactives_sat_early"]
    late = schedule["inactives_sat_late"]

    early_close = capture_scheduler.occurrence(early, saturday) + timedelta(
        minutes=early.grace_minutes
    )
    late_start = capture_scheduler.occurrence(late, saturday)
    assert early_close < late_start


def test_no_inactives_row_targets_snf_or_mnf() -> None:
    """docs/inactives_channel.md Section 2 measured SNF/MNF inactives as
    ALWAYS arriving after the Sunday 16:00 ET pick lock (0/17 playable in both
    slots), and Section 6 proposes no Sunday-evening or Monday capture row --
    only a grading-label use would justify one, and the doc does not derive
    it, so none should exist."""
    for job in capture_scheduler.SCHEDULE:
        if not job.name.startswith("inactives_"):
            continue
        assert job.day != "mon"
        if job.day == "sun":
            hour, _minute = (int(part) for part in job.at.split(":"))
            assert hour < 16  # strictly before the Sunday pick lock


_INACTIVES_REFRESH_WINDOWS = {
    "refresh_thu_inactives_early": ("inactives_thu_afternoon_early", "11:55", 55),
    "refresh_thu_inactives_late": ("inactives_thu_afternoon_late", "15:25", 55),
    "refresh_thu_inactives_primetime": ("inactives_thu_primetime", "19:15", 50),
    "refresh_sat_inactives_early": ("inactives_sat_early", "15:50", 60),
    "refresh_sat_inactives_late": ("inactives_sat_late", "19:15", 55),
    "refresh_sun_inactives_early": ("inactives_sun_early", "11:55", 55),
    "refresh_sun_inactives_late": ("inactives_sun_late", "15:00", 50),
}


def test_inactives_refreshes_begin_after_capture_and_stay_before_their_deadline() -> None:
    """The challenger gets only a valid capture-to-decision window, never a
    catch-up or publish-card path. Times are derived in docs/inactives_channel
    WP41 Section 7: capture close + five minutes, then ten minutes of deadline
    margin at the end of the refresh grace."""

    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    anchors = {
        "thu": datetime(2026, 11, 26, 23, 0, tzinfo=ET),
        "sat": datetime(2026, 12, 19, 23, 0, tzinfo=ET),
        "sun": datetime(2026, 9, 20, 23, 0, tzinfo=ET),
    }
    deadline_by_refresh = {
        "refresh_thu_inactives_early": datetime(2026, 11, 26, 13, 0, tzinfo=ET),
        "refresh_thu_inactives_late": datetime(2026, 11, 26, 16, 30, tzinfo=ET),
        "refresh_thu_inactives_primetime": datetime(2026, 11, 26, 20, 15, tzinfo=ET),
        "refresh_sat_inactives_early": datetime(2026, 12, 19, 17, 0, tzinfo=ET),
        "refresh_sat_inactives_late": datetime(2026, 12, 19, 20, 20, tzinfo=ET),
        "refresh_sun_inactives_early": datetime(2026, 9, 20, 13, 0, tzinfo=ET),
        "refresh_sun_inactives_late": datetime(2026, 9, 20, 16, 0, tzinfo=ET),
    }
    for refresh_name, (capture_name, target, grace) in _INACTIVES_REFRESH_WINDOWS.items():
        refresh = schedule[refresh_name]
        capture = schedule[capture_name]
        anchor = anchors[refresh.day]
        capture_close = capture_scheduler.occurrence(capture, anchor) + timedelta(
            minutes=capture.grace_minutes
        )
        refresh_start = capture_scheduler.occurrence(refresh, anchor)
        refresh_close = refresh_start + timedelta(minutes=refresh.grace_minutes)

        assert refresh.at == target
        assert refresh.grace_minutes == grace
        assert refresh_start == capture_close + timedelta(minutes=5)
        assert refresh_close == deadline_by_refresh[refresh_name] - timedelta(minutes=10)
        assert refresh.season_guarded is True
        assert refresh.catch_up is False
        assert refresh.added_on == "2026-09-02"
        assert refresh.command[-4:] == [
            "refresh-picks",
            "--record-decisions",
            "--note",
            refresh.command[-1],
        ]
        assert "--publish-card" not in refresh.command


def test_referee_assignments_wed_catch_up_and_added_on_guarded() -> None:
    """Pins the properties that justify catch_up=True on this specific job
    (same reasoning as player_arrests_tue/backup_data -- a late capture is
    still a valid, un-mislabelled snapshot, not a closing-line-style window
    a late run would corrupt)."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["referee_assignments_wed"]

    assert job.catch_up is True
    assert job.added_on == "2026-09-01"
    assert job.command[-1] == "--current"
    assert "capture_referee_assignments.py" in job.command[-2]
    assert job.dedupe_dir == "data/players/referee_assignments"
    assert job.dedupe_minutes == 240


def test_referee_assignments_wed_target_clears_the_latest_measured_publish_time() -> None:
    """docs/referee_assignments_capture.md Section 2 measured Football
    Zebras' own article:published_time across 10 sampled 2025 weeks: the
    latest within a normal (non-finale) week was Wed 12:42 ET (Weeks 8 and
    9). The job's own target time must clear that with real margin."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["referee_assignments_wed"]

    assert job.day == "wed"
    target_hour, target_minute = (int(part) for part in job.at.split(":"))
    latest_measured_publish_minutes = 12 * 60 + 42
    target_minutes = target_hour * 60 + target_minute
    assert target_minutes - latest_measured_publish_minutes >= 120  # >= 2h margin
