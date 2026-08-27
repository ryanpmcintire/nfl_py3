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

    assert command[-1].endswith("backup_data.py")
    # No --verify-all: that re-hashes 3.5 GB and would blow the timeout.
    assert "--verify-all" not in command


def test_grace_window_does_not_collide_with_the_next_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backup running long must not still be inside another job's window."""
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    backup = schedule["backup_data"]
    sunday = datetime(2026, 8, 30, 23, 0, tzinfo=ET)
    close = capture_scheduler.occurrence(backup, sunday) + timedelta(minutes=backup.grace_minutes)

    monday_mnf = capture_scheduler.occurrence(schedule["odds_mon_mnf"], sunday + timedelta(days=2))
    assert close < monday_mnf
