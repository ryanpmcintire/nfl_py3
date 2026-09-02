"""Read-only audit of fixed and source-driven weekly decision timing.

This command never runs a capture or a forecast. It inventories the checked-in
scheduler policy, immutable source snapshots, and append-only decision ledgers,
then validates that every observed refresh follows its Tuesday decision and
source capture while preceding its decision deadline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

SNAPSHOT_STAMP = re.compile(r"^\d{8}T\d{6}Z$")
TRIGGER_COLUMNS = frozenset({"trigger_type", "trigger_source", "trigger_observed_at_utc"})


@dataclass(frozen=True)
class LedgerSpec:
    name: str
    relative_path: str


class TimingJob(Protocol):
    name: str
    day: str
    at: str
    grace_minutes: int
    enabled: bool
    requires: tuple[str, ...]


LEDGERS: tuple[LedgerSpec, ...] = (
    LedgerSpec("played_refresh", "prospective/pick_revisions.parquet"),
    LedgerSpec("injury_signal", "prospective/injury_signal_refresh_decisions.parquet"),
    LedgerSpec("injury_report", "prospective/nflcom_friday_refresh_decisions.parquet"),
    LedgerSpec("inactives", "prospective/inactives_refresh_decisions.parquet"),
    LedgerSpec("referee", "prospective/crew_tilt_refresh_decisions.parquet"),
)

CAPTURE_ROOTS: dict[str, str] = {
    "market": "market/raw",
    "injury_news": "raw/injury_news",
    "injury_reports": "raw/sportradar_injuries",
    "legacy_nflcom_injuries": "raw/nflcom_injuries",
    "weather_forecasts": "raw/forecast_archive",
    "referee_assignments": "players/referee_assignments",
    "inactives": "players/inactives",
}


def _timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column} contains missing or invalid timestamps")
    return values


def audit_ledger(name: str, frame: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, Any]:
    """Summarize one immutable refresh ledger and its timing violations."""

    if frame.empty:
        return {
            "name": name,
            "rows": 0,
            "runs": 0,
            "structured_trigger_provenance": TRIGGER_COLUMNS.issubset(frame.columns),
            "structured_trigger_rows": 0,
            "timing_violations": 0,
        }
    required = {"revision_recorded_at_utc", "refresh_run_id", "game_id", "kickoff"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} ledger is missing columns: {', '.join(missing)}")

    revision = _timestamps(frame, "revision_recorded_at_utc")
    kickoff = _timestamps(frame, "kickoff")
    deadline = _timestamps(frame, "deadline") if "deadline" in frame else kickoff
    if "original_recorded_at_utc" in frame:
        original = _timestamps(frame, "original_recorded_at_utc")
    else:
        decision_times = decisions.loc[:, ["game_id", "recorded_at_utc"]].drop_duplicates(
            "game_id", keep="last"
        )
        joined = frame.loc[:, ["game_id"]].merge(decision_times, on="game_id", how="left")
        original = pd.to_datetime(joined["recorded_at_utc"], utc=True, errors="coerce")

    missing_original = int(original.isna().sum())
    before_original = int((revision < original).fillna(False).sum())
    at_or_after_deadline = int((revision >= deadline).sum())
    capture_columns = sorted(
        column
        for column in frame.columns
        if column.endswith(("_captured_at_utc", "_fetched_at_utc"))
        or column == "trigger_observed_at_utc"
    )
    post_revision_captures = 0
    invalid_source_timestamps = 0
    capture_rows = 0
    for column in capture_columns:
        captured = pd.to_datetime(frame[column], utc=True, errors="coerce")
        supplied = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
        invalid_source_timestamps += int((supplied & captured.isna()).sum())
        present = supplied & captured.notna()
        capture_rows += int(present.sum())
        post_revision_captures += int((present & (captured > revision)).sum())

    violations = (
        missing_original
        + before_original
        + at_or_after_deadline
        + invalid_source_timestamps
        + post_revision_captures
    )
    latency_hours = (revision - original).dt.total_seconds().div(3600).dropna()
    if TRIGGER_COLUMNS.issubset(frame.columns):
        trigger_complete = frame.loc[:, sorted(TRIGGER_COLUMNS)].notna().all(axis=1)
        for column in ("trigger_type", "trigger_source"):
            trigger_complete &= frame[column].astype(str).str.strip().ne("")
        structured_trigger_rows = int(trigger_complete.sum())
    else:
        structured_trigger_rows = 0
    return {
        "name": name,
        "rows": len(frame),
        "runs": int(frame["refresh_run_id"].nunique()),
        "structured_trigger_provenance": structured_trigger_rows == len(frame),
        "structured_trigger_rows": structured_trigger_rows,
        "source_timestamp_columns": capture_columns,
        "source_timestamp_rows": capture_rows,
        "invalid_source_timestamps": invalid_source_timestamps,
        "missing_original_decision": missing_original,
        "refresh_before_original_decision": before_original,
        "refresh_at_or_after_deadline": at_or_after_deadline,
        "source_captured_after_refresh": post_revision_captures,
        "timing_violations": violations,
        "hours_after_tuesday_decision": {
            "minimum": None if latency_hours.empty else float(latency_hours.min()),
            "median": None if latency_hours.empty else float(latency_hours.median()),
            "maximum": None if latency_hours.empty else float(latency_hours.max()),
        },
    }


def snapshot_inventory(data_root: Path) -> dict[str, dict[str, Any]]:
    """Count UTC-stamped directories and capture/build timestamps in manifests."""

    inventory: dict[str, dict[str, Any]] = {}
    for source, relative in CAPTURE_ROOTS.items():
        root = data_root / relative
        timestamped_dirs = (
            sorted(
                (
                    path
                    for path in root.rglob("*")
                    if path.is_dir() and SNAPSHOT_STAMP.fullmatch(path.name)
                ),
                key=lambda path: path.name,
            )
            if root.is_dir()
            else []
        )
        directory_stamps = [path.name for path in timestamped_dirs]
        manifest_stamps: list[pd.Timestamp] = []
        manifest_fields: set[str] = set()
        if timestamped_dirs:
            latest_manifest = timestamped_dirs[-1] / "manifest.json"
            manifests = [latest_manifest] if latest_manifest.is_file() else []
        else:
            manifests = list(root.rglob("manifest.json")) if root.is_dir() else []
        for manifest_path in manifests:
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for field in (
                "captured_at_utc",
                "fetched_at_utc",
                "requested_at_utc",
                "snapshot_timestamp_utc",
                "built_at_utc",
                "observed_at_utc",
                "generated_at_utc",
                "fetched_at",
            ):
                stamp = pd.to_datetime(payload.get(field), utc=True, errors="coerce")
                if not pd.isna(stamp):
                    manifest_fields.add(field)
                    manifest_stamps.append(stamp)
        all_stamps = [
            *(
                pd.to_datetime(stamp, format="%Y%m%dT%H%M%SZ", utc=True)
                for stamp in directory_stamps
            ),
            *manifest_stamps,
        ]
        inventory[source] = {
            "root": str(root),
            "timestamped_directories": len(directory_stamps),
            "manifests_inspected": len(manifests),
            "manifest_timestamp_fields": sorted(manifest_fields),
            "latest_recorded_timestamp_utc": (
                None if not all_stamps else max(all_stamps).isoformat()
            ),
        }
    return inventory


def schedule_inventory(schedule: Sequence[TimingJob]) -> dict[str, Any]:
    """Expose which decision jobs are clock-driven versus dependency-gated."""

    decision_jobs = [
        job for job in schedule if job.name == "weekly_lock" or job.name.startswith("refresh_")
    ]
    rows = [
        {
            "name": job.name,
            "day": job.day,
            "at_et": job.at,
            "grace_minutes": job.grace_minutes,
            "enabled": job.enabled,
            "requires": list(job.requires),
            "dispatch_mode": "fixed_clock",
        }
        for job in decision_jobs
    ]
    return {
        "jobs": rows,
        "fixed_clock_jobs": len(rows),
        "event_triggered_jobs": 0,
        "dependency_gated_jobs": sum(bool(row["requires"]) for row in rows),
    }


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def scheduler_run_inventory(data_root: Path, decision_job_names: set[str]) -> dict[str, Any]:
    """Read actual scheduler outcomes for decision jobs without invoking them."""

    state_path = data_root / "scheduler_state.json"
    if not state_path.is_file():
        return {"state_path": str(state_path), "runs": [], "status_counts": {}}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("scheduler_state.json must contain an object-valued runs field")
    observed = []
    for occurrence, result in sorted(runs.items()):
        job_name = occurrence.rsplit("@", 1)[0]
        if job_name not in decision_job_names or not isinstance(result, dict):
            continue
        observed.append(
            {
                "occurrence": occurrence,
                "status": result.get("status"),
                "window_start": result.get("window_start"),
                "ran_at": result.get("ran_at"),
            }
        )
    status_counts = pd.Series([row["status"] for row in observed]).value_counts().to_dict()
    return {
        "state_path": str(state_path),
        "runs": observed,
        "status_counts": {str(status): int(count) for status, count in status_counts.items()},
    }


def build_report(
    *,
    repo_root: Path,
    data_root: Path,
    artifacts_root: Path,
    schedule: Sequence[TimingJob],
) -> dict[str, Any]:
    """Build the complete read-only timing-policy audit report."""

    decisions = _read_parquet(artifacts_root / "clv_ledger" / "decisions.parquet")
    if not decisions.empty and not {"game_id", "recorded_at_utc"}.issubset(decisions.columns):
        raise ValueError("paper-decision ledger lacks game_id or recorded_at_utc")
    if decisions.empty:
        decisions = pd.DataFrame(columns=["game_id", "recorded_at_utc"])

    ledger_reports = [
        audit_ledger(spec.name, _read_parquet(artifacts_root / spec.relative_path), decisions)
        for spec in LEDGERS
    ]
    experiment_root = repo_root / "registry" / "experiments"
    fixed_evidence = sorted(
        path.relative_to(repo_root).as_posix()
        for prefix in ("observed-movement-channel", "movement-attribution", "reliability-movement")
        for path in (experiment_root / prefix).glob("*.json")
    )
    observed_rows = sum(report["rows"] for report in ledger_reports)
    triggered_rows = sum(report["structured_trigger_rows"] for report in ledger_reports)
    violations = sum(report["timing_violations"] for report in ledger_reports)
    schedule_report = schedule_inventory(schedule)
    decision_job_names = {row["name"] for row in schedule_report["jobs"]}
    schedule_report["observed_runs"] = scheduler_run_inventory(data_root, decision_job_names)
    complete = bool(fixed_evidence) and observed_rows > 0 and triggered_rows > 0 and violations == 0
    return {
        "schema": "nfl_ats_timing_policy_audit/1",
        "fixed_timestamp_experiment_artifacts": fixed_evidence,
        "schedule": schedule_report,
        "capture_sources": snapshot_inventory(data_root),
        "paper_decision_rows": len(decisions),
        "refresh_ledgers": ledger_reports,
        "summary": {
            "fixed_timestamp_comparison_present": bool(fixed_evidence),
            "prospective_refresh_rows": observed_rows,
            "structured_news_trigger_rows": triggered_rows,
            "timing_violations": violations,
            "mkt08_definition_satisfied": complete,
            "remaining_gap": (
                None
                if complete
                else (
                    "No observed refresh row has structured trigger type, source, and capture time."
                )
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module_name = "scripts.capture_scheduler" if __package__ else "capture_scheduler"
    scheduler = importlib.import_module(module_name)
    report = build_report(
        repo_root=args.repo_root,
        data_root=args.data_root or args.repo_root / "data",
        artifacts_root=args.artifacts_root or args.repo_root / "artifacts",
        schedule=scheduler.SCHEDULE,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["summary"]["timing_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
