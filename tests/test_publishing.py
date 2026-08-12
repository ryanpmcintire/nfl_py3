from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.publishing import publish_active_predictions


def _write_active_publication_fixture(root: Path) -> tuple[Path, Path]:
    forecast = root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    metadata = {
        "active_model_id": "model-123",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        {
            "game_id": ["later", "earlier"],
            "gameday": ["2026-09-13", "2026-09-10"],
            "away_team": ["ARI", "SF"],
            "home_team": ["LAC", "LA"],
            "spread_line": [10.5, -3.5],
            "home_cover_probability": [0.38, 0.46],
            "method": ["market_residual", "market_residual"],
        }
    ).to_csv(forecast / "recommendations.csv", index=False)
    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "target": "ats_classification",
        "model_id": "model-123",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/evaluation",
            "accuracy": 0.5205,
            "correct": 1080,
            "games": 2075,
            "intervals": {"week": {"lower": 0.4985, "upper": 0.5425}},
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    readme = root / "README.md"
    readme.write_text("# Project\n\nDescription.\n\n## Details\n", encoding="utf-8")
    return forecast, readme


def test_publish_active_predictions_updates_github_markdown_idempotently(tmp_path: Path) -> None:
    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    instant = datetime(2026, 8, 12, tzinfo=UTC)

    result = publish_active_predictions(
        tmp_path, destination=destination, readme_path=readme, published_at=instant
    )
    first_readme = readme.read_text(encoding="utf-8")
    publish_active_predictions(
        tmp_path, destination=destination, readme_path=readme, published_at=instant
    )

    assert result["model_id"] == "model-123"
    assert result["games"] == 2
    assert readme.read_text(encoding="utf-8") == first_readme
    assert first_readme.count("<!-- CURRENT_PREDICTIONS:START -->") == 1
    assert "**1,080 of 2,075 non-push games correctly (52.05%)**" in first_readme
    assert first_readme.index("SF at LA") < first_readme.index("ARI at LAC")
    assert "SF -3.5" in first_readme
    assert "ARI +10.5" in first_readme
    assert "Published from synchronized model `model-123`" in destination.read_text(
        encoding="utf-8"
    )


def test_publish_rejects_weekly_model_id_mismatch(tmp_path: Path) -> None:
    forecast, readme = _write_active_publication_fixture(tmp_path)
    metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
    metadata["active_model_id"] = "wrong-model"
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model ID does not match"):
        publish_active_predictions(
            tmp_path,
            destination=tmp_path / "CURRENT_PREDICTIONS.md",
            readme_path=readme,
        )
