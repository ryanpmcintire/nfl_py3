"""Immutable, pre-kickoff forecast records for prospective evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.prediction_safety import validate_prediction_card
from nfl_ats.provenance import sha256_file

FROZEN_PREDICTION_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "kickoff",
    "away_team",
    "home_team",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "home_cover_probability",
    "pick",
    "bet_side",
    "edge",
    "bet_odds",
    "break_even_probability",
    "market_home_no_vig_probability",
    "market_hold",
    "train_rows",
    "train_max_gameday",
)


@dataclass(frozen=True)
class FrozenForecast:
    forecast_id: str
    directory: Path
    games: int


def verify_frozen_forecast(directory: Path) -> dict[str, Any]:
    """Verify file integrity and re-run the prediction safety contract."""

    manifest_path = directory / "manifest.json"
    prediction_path = directory / "predictions.parquet"
    if not manifest_path.is_file() or not prediction_path.is_file():
        raise ValueError(f"Incomplete frozen forecast: {directory}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid frozen forecast manifest: {manifest_path}")
    if sha256_file(prediction_path) != payload.get("predictions_sha256"):
        raise ValueError(f"Frozen forecast digest mismatch: {directory}")
    predictions = pd.read_parquet(prediction_path)
    if len(predictions) != payload.get("games"):
        raise ValueError(f"Frozen forecast row-count mismatch: {directory}")
    forecast_id = payload.get("forecast_id")
    if forecast_id != directory.name or not predictions["forecast_id"].eq(forecast_id).all():
        raise ValueError(f"Frozen forecast identity mismatch: {directory}")
    model = payload.get("model", {})
    min_edge = float(model.get("min_edge", 0.02)) if isinstance(model, dict) else 0.02
    audit = validate_prediction_card(
        predictions,
        min_edge=min_edge,
        expected_season=int(payload["season"]),
        expected_week=int(payload["week"]),
        prospective=True,
        created_at=datetime.fromisoformat(str(payload["frozen_at_utc"])),
    )
    if int(payload.get("schema_version", 1)) >= 2:
        recorded = payload.get("prediction_safety")
        if not isinstance(recorded, dict):
            raise ValueError(f"Frozen forecast is missing its safety audit: {directory}")
        if recorded != audit.to_dict():
            raise ValueError(f"Frozen forecast safety audit mismatch: {directory}")
    return payload


def _utc(instant: datetime | None) -> datetime:
    value = instant or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def freeze_forecast(
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
    root: Path,
    *,
    created_at: datetime | None = None,
) -> FrozenForecast:
    """Write a new immutable forecast directory after enforcing pre-kickoff timing."""

    missing = sorted(set(FROZEN_PREDICTION_COLUMNS).difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions cannot be frozen; missing columns: {', '.join(missing)}")
    if predictions.empty:
        raise ValueError("Predictions cannot be frozen because there are no games")
    if predictions["game_id"].duplicated().any():
        raise ValueError("Predictions cannot be frozen with duplicate game IDs")
    outcome_columns = [
        column for column in ("home_cover", "result", "ats_margin") if column in predictions
    ]
    if outcome_columns and predictions[outcome_columns].notna().any(axis=None):
        raise ValueError("Prospective forecasts can only be frozen before outcomes are known")
    required_market_values = ("spread_line", "home_spread_odds", "away_spread_odds")
    missing_market = predictions.loc[
        predictions.loc[:, list(required_market_values)]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .any(axis=1),
        "game_id",
    ].astype(str)
    if not missing_market.empty:
        examples = ", ".join(missing_market.tolist()[:5])
        raise ValueError(f"Cannot freeze games with missing lines or prices: {examples}")

    frozen_at = _utc(created_at)
    kickoffs = pd.to_datetime(predictions["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        missing_games = predictions.loc[kickoffs.isna(), "game_id"].astype(str).tolist()
        examples = ", ".join(missing_games[:5])
        raise ValueError(f"Cannot prove pre-kickoff timing; kickoff is missing for: {examples}")
    if kickoffs.le(frozen_at).any():
        started = predictions.loc[kickoffs.le(frozen_at), "game_id"].astype(str).tolist()
        examples = ", ".join(started[:5])
        raise ValueError(f"Cannot freeze games at or after kickoff: {examples}")

    season = int(predictions["season"].iloc[0])
    week = int(predictions["week"].iloc[0])
    if predictions["season"].nunique() != 1 or predictions["week"].nunique() != 1:
        raise ValueError("A frozen forecast must contain exactly one season and week")
    min_edge = float(metadata.get("min_edge", 0.02))
    audit = validate_prediction_card(
        predictions,
        min_edge=min_edge,
        expected_season=season,
        expected_week=week,
        prospective=True,
        created_at=frozen_at,
    )
    forecast_id = f"{season}-week-{week:02d}-{run_id(frozen_at)}"
    destination = root / forecast_id
    if destination.exists():
        raise ValueError(f"Frozen forecast already exists: {destination}")

    frozen = predictions.loc[:, list(FROZEN_PREDICTION_COLUMNS)].copy()
    frozen["kickoff"] = kickoffs
    frozen.insert(0, "frozen_at_utc", frozen_at)
    frozen.insert(0, "forecast_id", forecast_id)
    prediction_path = destination / "predictions.parquet"
    atomic_parquet(frozen, prediction_path)
    manifest = {
        "schema_version": 2,
        "forecast_id": forecast_id,
        "frozen_at_utc": frozen_at.isoformat(),
        "season": season,
        "week": week,
        "games": len(frozen),
        "earliest_kickoff_utc": kickoffs.min().isoformat(),
        "latest_kickoff_utc": kickoffs.max().isoformat(),
        "predictions_sha256": sha256_file(prediction_path),
        "prediction_safety": audit.to_dict(),
        "model": metadata,
    }
    # Written last: its presence is the commit marker for a complete record.
    atomic_json(manifest, destination / "manifest.json")
    return FrozenForecast(forecast_id=forecast_id, directory=destination, games=len(frozen))
