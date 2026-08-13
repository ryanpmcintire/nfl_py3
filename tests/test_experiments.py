from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import PLAYER_STATE_METRICS
from nfl_ats.experiments import (
    nested_outcome_profile_selection,
    paired_feature_comparisons,
    run_feature_set_experiment,
    run_outcome_profile_experiment,
)


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


def test_player_profile_experiment_reuses_only_residual_method(
    model_frame: pd.DataFrame,
) -> None:
    enriched = model_frame.copy()
    index = np.arange(len(enriched))
    for metric_index, metric in enumerate(PLAYER_STATE_METRICS, start=1):
        enriched[f"diff_{metric}"] = np.cos(index / metric_index)
    result = run_outcome_profile_experiment(
        enriched,
        start_season=2020,
        profiles=("base", "player_injuries", "player"),
        min_train_games=80,
    )
    assert set(result.predictions["feature_profile"]) == {
        "base",
        "player_injuries",
        "player",
    }
    assert result.predictions["method"].eq("market_residual").all()
    assert len(result.predictions) == 60 * 3
    assert set(result.summary["feature_profile"]) == {
        "base",
        "player_injuries",
        "player",
    }
    synthetic = result.predictions.copy().sort_values(["feature_profile", "gameday", "game_id"])
    synthetic["season"] = (
        synthetic.groupby("feature_profile").cumcount().map(lambda value: 2018 + int(value) // 20)
    )
    nested = nested_outcome_profile_selection(
        synthetic,
        first_test_season=2020,
        validation_seasons=2,
    )
    assert len(nested.fold_summary) == 1
    assert len(nested.candidate_validation) == 3
    assert nested.predictions["season"].eq(2020).all()
    assert nested.predictions["selected_feature_profile"].nunique() == 1


def test_player_profile_experiment_guards(model_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="At least one player profile"):
        run_outcome_profile_experiment(model_frame, start_season=2020, profiles=())
    with pytest.raises(ValueError, match="must be unique"):
        run_outcome_profile_experiment(model_frame, start_season=2020, profiles=("base", "base"))
    with pytest.raises(ValueError, match="Unknown player profiles"):
        run_outcome_profile_experiment(model_frame, start_season=2020, profiles=("mystery",))
    with pytest.raises(ValueError, match="validation_seasons"):
        nested_outcome_profile_selection(
            pd.DataFrame(), first_test_season=2020, validation_seasons=0
        )
