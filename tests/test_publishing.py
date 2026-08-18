from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.publishing import BEST_PICK_MARK, publish_active_predictions


def _write_line_sweep(forecast: Path, widths: dict[str, float]) -> None:
    """A sweep where each game's pick holds >= 0.50 across ``2 * width`` points.

    Both fixture games are AWAY picks (``home_cover_probability`` below 0.5), and
    the sweep always reports the HOME probability, so the run is written as the
    complement.
    """

    offsets = np.arange(-4.0, 4.5, 0.5)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "game_id": game_id,
                    "line_offset": offsets,
                    "home_cover_probability": np.where(np.abs(offsets) <= width, 0.4, 0.6),
                    "method": "market_residual",
                }
            )
            for game_id, width in widths.items()
        ],
        ignore_index=True,
    ).to_parquet(forecast / "line_sweep.parquet")


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


def test_published_card_marks_the_week_best_pick(tmp_path: Path) -> None:
    """POL-10: the card the user reads at pick time must name the Best Pick.

    The ledger answers "what did we choose?" months later; only the card
    answers "which one do I enter today?".
    """

    forecast, readme = _write_active_publication_fixture(tmp_path)
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 1.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)

    assert result["best_pick_game_id"] == "later"
    card = destination.read_text(encoding="utf-8")
    assert f"{BEST_PICK_MARK}ARI +10.5" in card
    assert "Best Pick of the week" in card
    assert "ARI +10.5 in ARI at LAC" in card
    # Exactly one row is marked (the other occurrence is the note's legend).
    assert card.count(BEST_PICK_MARK) == 1
    assert card.count(BEST_PICK_MARK.strip()) == 2


def test_published_card_discloses_a_tied_best_pick(tmp_path: Path) -> None:
    """POL-09/POL-10: an undisclosed tie is not a lean.

    The dashboard (nfl_ats.dashboard.app_pages.picks) already shows this
    disclosure; the published card must show the identical sentence via the
    same nfl_ats.best_pick.best_pick_tie_note the dashboard calls, so the two
    surfaces cannot silently disagree about whether a nomination is arbitrary.
    """

    forecast, readme = _write_active_publication_fixture(tmp_path)
    # "later" and "earlier" both hold to the same width -- a two-way tie.
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 3.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)

    assert result["best_pick_tied"] is True
    card = destination.read_text(encoding="utf-8")
    assert "2 games tie at the top of that signal" in card
    assert "reproducible, but not a lean" in card
    readme_text = readme.read_text(encoding="utf-8")
    assert "2 games tie at the top of that signal" in readme_text


def test_published_card_does_not_disclose_an_unambiguous_best_pick(tmp_path: Path) -> None:
    forecast, readme = _write_active_publication_fixture(tmp_path)
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 1.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)

    assert result["best_pick_tied"] is False
    assert "tie at the top" not in destination.read_text(encoding="utf-8")


def test_published_card_without_a_sweep_names_no_best_pick(tmp_path: Path) -> None:
    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)
    assert result["best_pick_game_id"] is None
    assert BEST_PICK_MARK not in destination.read_text(encoding="utf-8")


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
