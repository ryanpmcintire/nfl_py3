"""Shared fixture-writing scaffold for overlay/tilt challenger-decision tests.

Extracted from the byte-identical copies of these helpers that were pasted
across ``tests/test_*overlay*.py`` / ``tests/test_*tilt*.py`` (wave-1
ref-tests-kit refactor; see reports/wave1/ref-tests-kit.md).

What lives here: the registry / active-model-card / stadium-registry-root
writers, which are identical across files except for a handful of constants
(challenger id, model config, season, week, created-at stamp).

What deliberately stays per-file: scenario data -- schedules, prediction
frames (``_recorder_predictions``), per-overlay behavioral asserts, and the
``*_is_leak_safe_*`` / ``refuses_*`` regression tests. Those are the leakage
and contract guards AGENTS.md mandates; they are NOT duplicated scaffolding.
"""

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
    """Write the ``prospective/challengers.json`` registry payload."""
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
    """Write the weekly forecast card (metadata + recommendations) and the
    ``active_ats_model.json`` snapshot the challenger recorders read."""
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
    """Write the reference stadium/station map under ``<tmp>/registry``."""
    registry_root = tmp_path / "registry"
    (registry_root / "reference").mkdir(parents=True, exist_ok=True)
    (registry_root / "reference" / "stadium_station_map.csv").write_text(
        stadium_station_map_csv,
        encoding="utf-8",
    )
    return registry_root
