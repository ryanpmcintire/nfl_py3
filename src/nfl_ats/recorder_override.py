"""Operator overrides shared by every lock-day recorder (owner, 2026-09-09)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path
from nfl_ats.io import atomic_parquet


def resolve_recording_forecast(
    artifacts_root: Path,
    active: dict[str, Any],
    *,
    forecast_artifact: str | None = None,
) -> tuple[Path, Any]:
    """Forecast a recorder reads, honouring ``--record-from-forecast``."""

    if forecast_artifact is not None:
        forecast = (artifacts_root / forecast_artifact).resolve()
        if not forecast.is_dir() or artifacts_root.resolve() not in forecast.parents:
            raise ValueError(f"Forecast artifact is not a directory under artifacts: {forecast}")
    else:
        linked = active_artifact_path(artifacts_root, active, "weekly_forecast")
        if linked is None:
            raise ValueError("Active ATS model has no linked weekly forecast")
        forecast = linked
    metadata_path = forecast / "metadata.json"
    if not metadata_path.is_file() or not (forecast / "recommendations.csv").is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if forecast_artifact is None and metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if not metadata.get("active_model_id"):
        raise ValueError("Weekly forecast metadata names no model id")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")
    return forecast, metadata


def replace_week_rows(
    existing: pd.DataFrame,
    ledger_path: Path,
    *,
    season: int,
    week: int,
    recorded_at: pd.Timestamp,
    columns: tuple[str, ...],
    challenger_id: str | None = None,
    kickoff_column: str = "kickoff",
) -> tuple[pd.DataFrame, int, int]:
    """Drop one week's still-pre-kickoff rows after a timestamped backup."""

    if existing.empty:
        return existing, 0, 0
    in_week = pd.to_numeric(existing["season"], errors="coerce").eq(season) & pd.to_numeric(
        existing["week"], errors="coerce"
    ).eq(week)
    if challenger_id is not None and "challenger_id" in existing.columns:
        in_week &= existing["challenger_id"].astype(str).eq(challenger_id)
    kickoffs = pd.to_datetime(existing[kickoff_column], errors="coerce", utc=True)
    dropping = in_week & kickoffs.gt(recorded_at)
    replaced = int(dropping.sum())
    left = int((in_week & ~dropping).sum())
    if not replaced:
        return existing, 0, left
    backup = ledger_path.with_name(
        f"{ledger_path.stem}.{recorded_at.strftime('%Y%m%dT%H%M%SZ')}.bak.parquet"
    )
    if not backup.exists():
        atomic_parquet(existing[list(columns)], backup)
    return existing.loc[~dropping].reset_index(drop=True), replaced, left
