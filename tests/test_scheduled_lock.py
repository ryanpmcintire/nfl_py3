from __future__ import annotations

import json
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

NOW = datetime.fromisoformat("2026-09-08T12:20:00-04:00")


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

    capture_scheduler.sweep_missed(datetime.fromisoformat("2026-09-08T14:21:00-04:00"), state)

    record = state["runs"][f"weekly_lock@{start.date().isoformat()}"]
    assert record["status"] == "MISSED"
    assert record["blocked_by"] == ["odds_tue_open"]
    assert "prerequisites not successful: odds_tue_open" in (tmp_path / "scheduler.log").read_text(
        encoding="utf-8"
    )


def test_real_job_has_no_backdate_flags_and_closes_by_1420() -> None:
    job = {job.name: job for job in capture_scheduler.SCHEDULE}["weekly_lock"]

    assert (job.day, job.at, job.grace_minutes) == ("tue", "12:20", 120)
    assert job.requires == ("odds_tue_open",)
    assert not job.catch_up
    assert "--season" not in job.command
    assert "--week" not in job.command
    assert "--record-decisions" not in job.command


def test_lock_scripts_weekly_run_argv_parses_against_the_real_parser() -> None:
    """2026-09-07: the scheduled lock spawns `nfl-ats weekly-run ...` from a
    hard-coded argv that no scheduled run had ever exercised (the refresh
    jobs' identical gap took down every Sunday pass the day before). Pin
    that the argv the script builds is accepted by the real parser."""
    import subprocess
    from unittest import mock

    import scripts.scheduled_weekly_lock as lock_script
    from nfl_ats.cli import build_parser

    captured: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured.append([str(part) for part in command])
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    with mock.patch.object(lock_script.subprocess, "run", fake_run):
        lock_script._run_weekly(2026, 1)

    assert len(captured) == 1
    argv = captured[0]
    assert argv[argv.index("nfl-ats") + 1] == "weekly-run"
    args = build_parser().parse_args(argv[argv.index("nfl-ats") + 1 :])
    assert (args.season, args.week, args.record_decisions) == (2026, 1, True)


#: Verbatim shape of the 2026-09-08 lock's stderr: a step-4
#: BootstrapDegeneracyWarning (header plus its indented source echo), progress
#: banners, the decision-package line, and the real reason last. The old
#: `stderr[-500:]` kept the warning and threw the reason away.
_NOISY_STDERR = """weekly-run step 4 margin-backtest ...
F:\\Repos\\nfl_py3\\src\\nfl_ats\\cli_commands\\evaluation.py:442: \
BootstrapDegeneracyWarning: outcome_bootstrap_intervals(block=season): 8 bootstrap blocks \
(< 10) admit only 6,435 distinct resamples; measured coverage of a known truth at this block \
count is well under the nominal 95% (0.47 at 2 blocks, 0.76 at 4, 0.88 at 8), and the interval \
comes back with EXACTLY ZERO width in 25% of 2-block draws, 8% at 3 and 2% at 4. Report the \
estimate and probability_positive, not this interval.
  outcome_bootstrap_intervals(
weekly-run step 5 margin-predict ...
weekly-run step 8 publish-predictions ...
lock-day decision package: artifacts/lockday_packages/2026_wk01_20260908T163746Z
error: the reason the lock actually failed
"""


def test_failed_weekly_run_persists_its_whole_output_and_names_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-09-08: the lock aborted and the record kept was 200 characters of a
    step-4 warning. The child's stderr had been cut to its last 500 characters
    here, then the scheduler kept the first 200 of the JSON line carrying it,
    so the reason was gone twice over. Everything must now land on disk."""
    import subprocess
    from unittest import mock

    import scripts.scheduled_weekly_lock as lock_script

    monkeypatch.setattr(lock_script, "FAILURE_LOG_DIR", tmp_path / "lock_failures")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 2, stdout="partial stdout", stderr=_NOISY_STDERR
        )

    with (
        mock.patch.object(lock_script.subprocess, "run", fake_run),
        pytest.raises(lock_script.WeeklyRunFailure) as raised,
    ):
        lock_script._run_weekly(2026, 1)

    log_path = raised.value.log_path
    assert log_path.is_file()
    body = log_path.read_text(encoding="utf-8")
    assert _NOISY_STDERR.strip() in body, "the complete stderr must survive verbatim"
    assert "partial stdout" in body
    assert "weekly-run\n" in body or "weekly-run " in body

    message = str(raised.value)
    assert message.startswith("weekly-run failed (2): error: the reason the lock actually failed")
    assert log_path.name in message
    assert "BootstrapDegeneracyWarning" not in message, (
        "a warning from an earlier step must never stand in for the reason"
    )


def test_failed_lock_reports_the_log_path_inside_the_scheduler_200_char_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """capture_scheduler records only the first 200 characters of this line, so
    the path has to come before the message, and a failure raised before
    weekly-run ever starts still has to leave a full traceback behind."""

    import scripts.scheduled_weekly_lock as lock_script

    monkeypatch.setattr(lock_script, "FAILURE_LOG_DIR", tmp_path / "lock_failures")

    def boom(_: object) -> Any:
        raise DataContractError("snapshot is not verified")

    monkeypatch.setattr(lock_script, "latest_snapshot", lambda _: tmp_path)
    monkeypatch.setattr(lock_script, "load_verified_snapshot", boom)

    assert lock_script.main() == 1

    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["status"] == "failed_closed"
    assert payload["error"] == "snapshot is not verified"
    logged = Path(payload["error_log"])
    assert logged.is_file()
    assert "DataContractError: snapshot is not verified" in logged.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in logged.read_text(encoding="utf-8")
    assert line.index('"error_log"') < line.index('"error":'), (
        "the path must precede the message: capture_scheduler keeps only the "
        "first 200 characters of this line"
    )


def test_override_locks_a_named_week_on_any_day() -> None:
    """Owner, 2026-09-09: a missed lock is recorded late, not preserved."""
    wednesday = datetime.fromisoformat("2026-09-09T16:30:00-04:00")
    with pytest.raises(DataContractError, match="Pass --season/--week"):
        resolve_lock_target(schedule(), now=wednesday)
    target = resolve_lock_target(schedule(), now=wednesday, season=2026, week=1)
    assert (target.season, target.week) == (2026, 1)
    assert target.game_ids == frozenset({"2026_01_DAL_PHI", "2026_01_BUF_NYJ"})
    with pytest.raises(DataContractError, match="needs both"):
        resolve_lock_target(schedule(), now=wednesday, season=2026)
    # Only a week whose every game day has begun is refused.
    after_week = datetime.fromisoformat("2026-09-14T09:00:00-04:00")
    with pytest.raises(DataContractError, match="already begun"):
        resolve_lock_target(schedule(), now=after_week, season=2026, week=1)


def test_replace_reruns_a_recorded_week_and_tells_the_runner(
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
    monkeypatch.setattr(scheduled_lock, "write_stamped_artifact", lambda summary, path: None)
    calls: list[tuple[int, int, bool]] = []

    def runner(season: int, week: int, *, replace: bool = False) -> dict[str, Any]:
        calls.append((season, week, replace))
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
        now=datetime.fromisoformat("2026-09-09T16:30:00-04:00"),
        weekly_runner=runner,
        verifier=lambda season, week, summary: {"paper_ledger_rows": 2},
        season=2026,
        week=1,
        replace=True,
    )
    assert result["status"] == "recorded_and_verified"
    assert result["replaced"] is True
    assert calls == [(2026, 1, True)]
