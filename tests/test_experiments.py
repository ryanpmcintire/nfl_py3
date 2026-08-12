from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.experiments import paired_feature_comparisons, run_feature_set_experiment


def test_feature_set_experiment_compares_identical_windows(model_frame: pd.DataFrame) -> None:
    result = run_feature_set_experiment(
        model_frame,
        start_season=2020,
        feature_sets=("market", "market_graph", "full"),
        min_train_games=80,
    )
    assert set(result.summary["feature_set"]) == {"market", "market_graph", "full"}
    assert set(result.predictions["feature_set"]) == {"market", "market_graph", "full"}
    assert result.summary["games_evaluated"].nunique() == 1
    assert set(result.season_summary["feature_set"]) == {"market", "market_graph", "full"}
    assert result.season_summary["season"].nunique() == 1


def test_feature_set_experiment_validates_names(model_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown feature sets"):
        run_feature_set_experiment(model_frame, 2020, feature_sets=("mystery",))
    with pytest.raises(ValueError, match="must be unique"):
        run_feature_set_experiment(model_frame, 2020, feature_sets=("market", "market"))


def test_paired_feature_comparison_preserves_games_and_blocks() -> None:
    rows = []
    for game in range(20):
        actual = float(game % 2)
        for feature_set, probability in (
            ("baseline", 0.5),
            ("candidate", 0.9 if actual else 0.1),
        ):
            rows.append(
                {
                    "feature_set": feature_set,
                    "game_id": f"game_{game}",
                    "season": 2020 + game // 10,
                    "week": 1 + game % 10,
                    "home_cover": actual,
                    "home_cover_probability": probability,
                }
            )
    predictions = pd.DataFrame(rows)
    comparison = paired_feature_comparisons(
        predictions,
        baseline_feature_set="baseline",
        samples=20,
        block="season",
        seed=7,
    )
    assert comparison["paired_games"].eq(20).all()
    assert comparison["estimate"].gt(0).all()
    assert comparison["lower"].gt(0).all()


def test_paired_feature_comparison_guards() -> None:
    predictions = pd.DataFrame(
        {
            "feature_set": ["baseline"],
            "game_id": ["game"],
            "season": [2020],
            "week": [1],
            "home_cover": [1.0],
            "home_cover_probability": [0.5],
        }
    )
    with pytest.raises(ValueError, match="samples"):
        paired_feature_comparisons(
            predictions,
            baseline_feature_set="baseline",
            samples=9,
        )
    with pytest.raises(ValueError, match="Unknown paired baseline"):
        paired_feature_comparisons(
            predictions,
            baseline_feature_set="missing",
            samples=10,
        )
