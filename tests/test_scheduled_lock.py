from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import nfl_ats.scheduled_lock as scheduled_lock
from nfl_ats.data import DataContractError
from nfl_ats.scheduled_lock import execute_scheduled_lock, resolve_lock_target

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.capture_scheduler as capture_scheduler

NOW = datetime.fromisoformat("2026-09-08T09:15:00-04:00")


def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_DAL_PHI", "2026_01_BUF_NYJ", "old"],
            "season": [2026, 2026, 2025],
            "week": [1, 1, 1],
            "game_type": ["REG", "REG", "REG"],
            "gameday": ["2026-09-10", "2026-09-13", "2025-09-04"],
        }
    )


def test_resolves_only_the_week_whose_games_share_today_as_lock_tuesday() -> None:
    target = resolve_lock_target(schedule(), now=NOW)

    assert (target.season, target.week) == (2026, 1)
    assert target.game_ids == frozenset({"2026_01_DAL_PHI", "2026_01_BUF_NYJ"})


def test_refuses_offseason_or_backdated_invocation() -> None:
    with pytest.raises(DataContractError, match="exactly one scheduled game week"):
        resolve_lock_target(schedule(), now=datetime.fromisoformat("2026-09-09T09:15:00-04:00"))


def test_complete_existing_week_is_idempotent_without_running_weekly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        scheduled_lock,
        "load_paper_decisions",
        lambda _: pd.DataFrame(
            {
                "season": [2026, 2026],
                "week": [1, 1],
                "game_id": ["2026_01_DAL_PHI", "2026_01_BUF_NYJ"],
            }
        ),
    )
    calls: list[tuple[int, int]] = []

    result = execute_scheduled_lock(
        schedule(),
        artifacts_root=tmp_path,
        now=NOW,
        weekly_runner=lambda season, week: calls.append((season, week)) or {},
        verifier=lambda season, week, summary: {},
    )

    assert result["status"] == "already_recorded"
    assert calls == []


def test_partial_existing_week_fails_closed_without_repair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        scheduled_lock,
        "load_paper_decisions",
        lambda _: pd.DataFrame({"season": [2026], "week": [1], "game_id": ["2026_01_DAL_PHI"]}),
    )
    with pytest.raises(DataContractError, match="partially recorded"):
        execute_scheduled_lock(
            schedule(),
            artifacts_root=tmp_path,
            now=NOW,
            weekly_runner=lambda season, week: {},
            verifier=lambda season, week, summary: {},
        )


def test_new_week_requires_safe_summary_and_full_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        scheduled_lock,
        "load_paper_decisions",
        lambda _: pd.DataFrame(columns=["season", "week", "game_id"]),
    )
    seen: list[tuple[int, int]] = []

    def run(season: int, week: int) -> dict[str, Any]:
        seen.append((season, week))
        return {
            "command": "weekly-run",
            "season": season,
            "week": week,
            "record_decisions": True,
            "dry_run": False,
            "published": True,
        }

    result = execute_scheduled_lock(
        schedule(),
        artifacts_root=tmp_path,
        now=NOW,
        weekly_runner=run,
        verifier=lambda season, week, summary: {
            "paper_ledger_rows": 2,
            "missing": [],
            "pending_wiring": [],
        },
    )

    assert seen == [(2026, 1)]
    assert result["status"] == "recorded_and_verified"
    assert (tmp_path / "scheduled_locks/2026-week-01/weekly_summary.json").is_file()


def test_scheduler_waits_for_successful_same_day_opener() -> None:
    job = {job.name: job for job in capture_scheduler.SCHEDULE}["weekly_lock"]
    start = capture_scheduler.occurrence(job, NOW)
    state = {"runs": {}}
    assert not capture_scheduler.prerequisites_satisfied(job, start, state)

    key = f"odds_tue_open@{start.date().isoformat()}"
    state["runs"][key] = {"status": "FAIL(1)"}
    assert not capture_scheduler.prerequisites_satisfied(job, start, state)
    state["runs"][key] = {"status": "OK"}
    assert capture_scheduler.prerequisites_satisfied(job, start, state)


def test_failed_opener_becomes_durable_missed_alarm_after_grace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = {job.name: job for job in capture_scheduler.SCHEDULE}["weekly_lock"]
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "scheduler.log")
    monkeypatch.setattr(capture_scheduler, "season_active", lambda _: True)
    start = capture_scheduler.occurrence(job, NOW)
    state = {"runs": {f"odds_tue_open@{start.date().isoformat()}": {"status": "FAIL(1)"}}}

    capture_scheduler.sweep_missed(datetime.fromisoformat("2026-09-08T11:16:00-04:00"), state)

    record = state["runs"][f"weekly_lock@{start.date().isoformat()}"]
    assert record["status"] == "MISSED"
    assert record["blocked_by"] == ["odds_tue_open"]
    assert "prerequisites not successful: odds_tue_open" in (tmp_path / "scheduler.log").read_text(
        encoding="utf-8"
    )


def test_real_job_has_no_backdate_flags_and_closes_by_1115() -> None:
    job = {job.name: job for job in capture_scheduler.SCHEDULE}["weekly_lock"]

    assert (job.day, job.at, job.grace_minutes) == ("tue", "09:15", 120)
    assert job.requires == ("odds_tue_open",)
    assert not job.catch_up
    assert "--season" not in job.command
    assert "--week" not in job.command
    assert "--record-decisions" not in job.command
