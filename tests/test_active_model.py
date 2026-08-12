from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.active_model import (
    activate_matching_ats_model,
    active_artifact_path,
    load_active_ats_model,
)


def _metadata(feature_sha256: str) -> dict[str, object]:
    return {
        "created_at_utc": "2026-08-12T20:00:00+00:00",
        "season": 2026,
        "week": 1,
        "feature_profile": "player",
        "regressor": "ridge",
        "ats_method": "market_residual",
        "provenance": {
            "configuration_sha256": "config",
            "feature_table": {"sha256": feature_sha256},
        },
    }


def _evaluation(root: Path, feature_sha256: str = "features") -> Path:
    evaluation = root / "margins" / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "metadata.json").write_text(
        json.dumps(_metadata(feature_sha256)), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "method": ["market_residual"],
            "cover_accuracy": [0.5205],
            "cover_games": [2075],
        }
    ).to_csv(evaluation / "summary.csv", index=False)
    pd.DataFrame(
        {
            "method": ["market_residual", "market_residual"],
            "metric": ["cover_accuracy", "cover_accuracy"],
            "block": ["week", "season"],
            "lower": [0.4985, 0.5019],
            "upper": [0.5425, 0.5414],
        }
    ).to_csv(evaluation / "uncertainty.csv", index=False)
    return evaluation


def test_active_manifest_links_exact_evaluation_and_forecast(tmp_path: Path) -> None:
    evaluation = _evaluation(tmp_path)
    forecast = tmp_path / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)

    manifest = activate_matching_ats_model(tmp_path, forecast, _metadata("features"))

    assert manifest is not None
    assert manifest["status"] == "SYNCHRONIZED"
    assert manifest["historical_evaluation"]["accuracy"] == pytest.approx(0.5205)
    assert manifest["historical_evaluation"]["correct"] == 1080
    assert active_artifact_path(tmp_path, manifest, "historical_evaluation") == evaluation
    assert active_artifact_path(tmp_path, manifest, "weekly_forecast") == forecast
    assert load_active_ats_model(tmp_path) == manifest


def test_unmatched_forecast_does_not_replace_active_manifest(tmp_path: Path) -> None:
    _evaluation(tmp_path)
    forecast = tmp_path / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    active = activate_matching_ats_model(tmp_path, forecast, _metadata("features"))
    active_path = tmp_path / "active_ats_model.json"
    original = active_path.read_text(encoding="utf-8")

    unmatched = activate_matching_ats_model(tmp_path, forecast, _metadata("different-features"))

    assert active is not None
    assert unmatched is None
    assert active_path.read_text(encoding="utf-8") == original


def test_active_artifact_cannot_escape_artifacts_root(tmp_path: Path) -> None:
    manifest = {"historical_evaluation": {"artifact": "../outside"}}

    with pytest.raises(ValueError, match="escapes artifacts root"):
        active_artifact_path(tmp_path, manifest, "historical_evaluation")
