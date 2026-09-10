from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION

DEFAULT_FEATURE_TABLE_PATH = "data/processed/game_features_weak_stack.parquet"


def write_challenger_registry(
    artifacts: Path,
    *,
    challenger_id: str,
    model_config: dict[str, object],
    status: str = "ACTIVE_PROSPECTIVE",
) -> None:
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {"challenger_id": challenger_id, "status": status, "model": dict(model_config)}
        ],
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_active_model_and_card(
    artifacts: Path,
    *,
    season: int,
    week: int,
    created_at_utc: str,
    recommendations: pd.DataFrame,
    ridge_alpha: float = 10.0,
    forecast_dir: str | None = None,
    feature_profile: str = "weak_stack",
    probability_method: str | None = None,
    min_edge: float = 0.02,
    min_train_games: int = 500,
    feature_table_path: str = DEFAULT_FEATURE_TABLE_PATH,
) -> None:
    if forecast_dir is None:
        forecast_dir = f"{season}-week-{week:02d}-forecast"
    forecast = artifacts / "margin_predictions" / forecast_dir
    forecast.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "season": season,
        "week": week,
        "created_at_utc": created_at_utc,
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": ridge_alpha,
        "calibration_method": "none",
    }
    if probability_method is not None:
        metadata["probability_method"] = probability_method
    metadata.update(
        {
            "feature_profile": feature_profile,
            "min_edge": min_edge,
            "min_train_games": min_train_games,
            "provenance": {
                "feature_table": {
                    "path": feature_table_path,
                    "sha256": "abc123",
                }
            },
        }
    )
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    recommendations.to_csv(forecast / "recommendations.csv", index=False)

    active: dict[str, object] = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "model_id": "model-xyz",
        "method": "market_residual",
        "feature_profile": feature_profile,
    }
    if probability_method is not None:
        active["probability_method"] = probability_method
    active.update(
        {
            "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
            "weekly_forecast": {
                "artifact": f"margin_predictions/{forecast_dir}",
                "season": season,
                "week": week,
            },
        }
    )
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")


def write_registry_root(tmp_path: Path, *, stadium_station_map_csv: str) -> Path:
    registry_root = tmp_path / "registry"
    (registry_root / "reference").mkdir(parents=True, exist_ok=True)
    (registry_root / "reference" / "stadium_station_map.csv").write_text(
        stadium_station_map_csv,
        encoding="utf-8",
    )
    return registry_root
