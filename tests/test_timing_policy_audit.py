from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.timing_policy_audit import (
    audit_ledger,
    build_report,
    schedule_inventory,
    snapshot_inventory,
)


@dataclass(frozen=True)
class _TimingJob:
    name: str
    day: str
    at: str
    grace_minutes: int
    enabled: bool
    requires: tuple[str, ...]


def _job(name: str, *, requires: tuple[str, ...] = ()) -> _TimingJob:
    return _TimingJob(
        name=name,
        day="thu",
        at="15:00",
        grace_minutes=60,
        enabled=True,
        requires=requires,
    )


def test_schedule_inventory_does_not_mislabel_clock_jobs_as_event_triggered() -> None:
    report = schedule_inventory(
        (_job("weekly_lock", requires=("odds_tue_open",)), _job("refresh_thu"))
    )
    assert report["fixed_clock_jobs"] == 2
    assert report["event_triggered_jobs"] == 0
    assert report["dependency_gated_jobs"] == 1
    assert {row["dispatch_mode"] for row in report["jobs"]} == {"fixed_clock"}


def test_audit_ledger_checks_trigger_capture_and_decision_boundaries() -> None:
    decisions = pd.DataFrame({"game_id": ["g1"], "recorded_at_utc": ["2026-09-08T16:00:00Z"]})
    ledger = pd.DataFrame(
        {
            "revision_recorded_at_utc": ["2026-09-10T18:00:00Z"],
            "refresh_run_id": ["r1"],
            "game_id": ["g1"],
            "kickoff": ["2026-09-11T00:20:00Z"],
            "deadline": ["2026-09-11T00:20:00Z"],
            "injury_captured_at_utc": ["2026-09-10T17:30:00Z"],
            "trigger_type": ["injury_report"],
            "trigger_source": ["licensed_feed"],
            "trigger_observed_at_utc": ["2026-09-10T17:30:00Z"],
        }
    )
    report = audit_ledger("injuries", ledger, decisions)
    assert report["structured_trigger_provenance"] is True
    assert report["timing_violations"] == 0
    assert report["hours_after_tuesday_decision"]["median"] == 50.0

    ledger.loc[0, "injury_captured_at_utc"] = "2026-09-10T18:01:00Z"
    ledger.loc[0, "deadline"] = "2026-09-10T17:59:00Z"
    report = audit_ledger("injuries", ledger, decisions)
    assert report["source_captured_after_refresh"] == 1
    assert report["refresh_at_or_after_deadline"] == 1
    assert report["timing_violations"] == 2

    ledger.loc[0, "injury_captured_at_utc"] = "not-a-timestamp"
    report = audit_ledger("injuries", ledger, decisions)
    assert report["invalid_source_timestamps"] == 1
    assert report["timing_violations"] == 2


def test_snapshot_inventory_uses_immutable_utc_directory_names(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "injury_news"
    (root / "20260901T193026Z").mkdir(parents=True)
    (root / "not-a-snapshot").mkdir()
    report = snapshot_inventory(tmp_path)["injury_news"]
    assert report["timestamped_directories"] == 1
    assert report["latest_recorded_timestamp_utc"] == "2026-09-01T19:30:26+00:00"


def test_report_keeps_mkt08_open_without_observed_trigger_rows(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    evidence = repo_root / "registry" / "experiments" / "observed-movement-channel"
    evidence.mkdir(parents=True)
    (evidence / "run.json").write_text("{}", encoding="utf-8")

    report = build_report(
        repo_root=repo_root,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        schedule=(_job("weekly_lock", requires=("odds_tue_open",)), _job("refresh_thu")),
    )
    assert report["summary"]["fixed_timestamp_comparison_present"] is True
    assert report["summary"]["prospective_refresh_rows"] == 0
    assert report["summary"]["structured_news_trigger_rows"] == 0
    assert report["summary"]["mkt08_definition_satisfied"] is False
