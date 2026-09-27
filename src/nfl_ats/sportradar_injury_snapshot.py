from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.provenance import sha256_file


class SportradarInjuryCaptureError(RuntimeError):
    pass


def load_for_decision(
    root: Path,
    decision_at: datetime,
    *,
    season: int,
    week: int,
    season_type: str,
) -> tuple[Path, pd.DataFrame]:
    cutoff = pd.Timestamp(decision_at)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise SportradarInjuryCaptureError("Decision time must carry a timezone")
    eligible: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        if (
            type(manifest.get("season")) is not int
            or manifest["season"] != season
            or type(manifest.get("week")) is not int
            or manifest["week"] != week
            or manifest.get("season_type") != season_type
        ):
            continue
        captured_at = pd.to_datetime(manifest.get("captured_at_utc"), utc=False, errors="coerce")
        if pd.isna(captured_at) or captured_at.tzinfo is None:
            raise SportradarInjuryCaptureError(f"Naive capture time in {manifest_path}")
        if captured_at <= cutoff.tz_convert("UTC"):
            eligible.append((captured_at, manifest_path.parent, manifest))
    if not eligible:
        raise SportradarInjuryCaptureError(
            f"No complete {season_type} {season} week {week} injury snapshot existed by {cutoff}"
        )
    captured_at, snapshot, manifest = max(eligible, key=lambda item: item[0])
    if manifest.get("schema") != "sportradar_nfl_injuries_snapshot/1":
        raise SportradarInjuryCaptureError(f"Wrong snapshot schema in {snapshot}")
    if manifest.get("snapshot_id") != snapshot.name:
        raise SportradarInjuryCaptureError(f"Snapshot identity mismatch in {snapshot}")
    if manifest.get("available_at_policy") != "capture_time_not_provider_status_date":
        raise SportradarInjuryCaptureError(f"Wrong availability policy in {snapshot}")
    entries = manifest.get("files")
    if (
        not isinstance(entries, list)
        or len(entries) != 2
        or any(not isinstance(entry, dict) for entry in entries)
        or {entry.get("path") for entry in entries} != {"source.json", "injuries.parquet"}
    ):
        raise SportradarInjuryCaptureError(f"Incomplete file manifest in {snapshot}")
    for entry in entries:
        path = snapshot / str(entry["path"])
        if (
            not path.is_file()
            or not isinstance(entry.get("bytes"), int)
            or path.stat().st_size != entry["bytes"]
            or not isinstance(entry.get("sha256"), str)
            or sha256_file(path) != entry["sha256"]
        ):
            raise SportradarInjuryCaptureError(f"Snapshot file failed SHA-256 verification: {path}")
    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        raise SportradarInjuryCaptureError(f"Missing coverage in {snapshot}")
    required = coverage.get("required_teams")
    reported = coverage.get("reported_teams")
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(team, str) or not team for team in required)
        or not isinstance(reported, list)
        or any(not isinstance(team, str) or not team for team in reported)
        or not set(required).issubset(reported)
    ):
        raise SportradarInjuryCaptureError(f"Incomplete team coverage in {snapshot}")
    frame = pd.read_parquet(snapshot / "injuries.parquet")
    canonical_columns = {
        "season",
        "week",
        "season_type",
        "team",
        "player",
        "player_id",
        "sr_id",
        "position",
        "injury",
        "practice_status",
        "game_status",
        "status_date",
        "source_generated_at_utc",
        "available_at_utc",
    }
    if not canonical_columns.issubset(frame.columns):
        raise SportradarInjuryCaptureError(f"Canonical rows are incomplete in {snapshot}")
    rows = coverage.get("rows")
    if not isinstance(rows, int) or rows != len(frame) or frame.empty:
        raise SportradarInjuryCaptureError(f"Row coverage mismatch in {snapshot}")
    frame_season = pd.to_numeric(frame["season"], errors="coerce")
    frame_week = pd.to_numeric(frame["week"], errors="coerce")
    if (
        frame_season.isna().any()
        or not frame_season.eq(season).all()
        or frame_week.isna().any()
        or not frame_week.eq(week).all()
        or not frame["season_type"].eq(season_type).all()
    ):
        raise SportradarInjuryCaptureError(
            f"Snapshot rows do not match requested target in {snapshot}"
        )
    available = pd.to_datetime(frame["available_at_utc"], utc=True, errors="coerce")
    if available.isna().any() or not available.eq(captured_at).all():
        raise SportradarInjuryCaptureError(
            "Snapshot availability does not match immutable capture time"
        )
    if (available > cutoff.tz_convert("UTC")).any():
        raise SportradarInjuryCaptureError(
            "Post-decision injury rows crossed the availability boundary"
        )
    return snapshot, frame
