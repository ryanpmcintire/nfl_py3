from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import nfl_ats.experiments as experiments_module
from nfl_ats.constants import PLAYER_STATE_METRICS
from nfl_ats.estimation_variance import BootstrapDegeneracyError, BootstrapDegeneracyWarning
from nfl_ats.experiments import (
    FROZEN_PLAYER_CALIBRATIONS,
    FROZEN_PLAYER_MODEL_PROFILES,
    FROZEN_PLAYER_RIDGE_ALPHAS,
    nested_outcome_configuration_selection,
    nested_outcome_profile_selection,
    paired_feature_comparisons,
    run_feature_set_experiment,
    run_frozen_player_model_selection,
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
    # A candidate this dominant should win essentially every blocked resample;
    # probability_positive is the continuous evidence statement, in [0, 1].
    assert comparison["probability_positive"].between(0.0, 1.0).all()
    assert comparison["probability_positive"].eq(1.0).all()


def test_paired_feature_comparison_flags_a_degenerate_block_count() -> None:
    """REGRESSION TEST for D4: a low-block interval must never leave this
    function looking like a normal one.

    ``registry/weak_signals.json``'s ``player_qb_continuity_matched_alpha``
    carries a terminal ``refuted_mechanism`` verdict next to a season-blocked
    interval of ``[0.0, 2.2177]`` computed on exactly 4 blocks -- a lower bound
    of exactly 0.0 being the fingerprint of a resampling distribution with 35
    achievable values pretending to be a smooth 95% quantile. Measured coverage
    at 4 blocks is ~0.80, not 0.95. The interval itself is unchanged (changing
    it would rewrite history silently); what changes is that ``blocks`` and
    ``degenerate_blocks`` now travel with it into the CSV a registry entry
    cites, and a warning fires at the point of computation.
    """

    rows = []
    for game in range(24):
        actual = float(game % 2)
        for feature_set, probability in (("baseline", 0.5), ("candidate", 0.9 if actual else 0.1)):
            rows.append(
                {
                    "feature_set": feature_set,
                    "game_id": f"game_{game}",
                    "season": 2020 + game // 6,
                    "week": 1 + game % 6,
                    "home_cover": actual,
                    "home_cover_probability": probability,
                }
            )
    predictions = pd.DataFrame(rows)

    with pytest.warns(BootstrapDegeneracyWarning, match="bootstrap blocks"):
        season_blocked = paired_feature_comparisons(
            predictions, baseline_feature_set="baseline", samples=50, block="season", seed=7
        )
    assert season_blocked["blocks"].eq(4).all()
    assert season_blocked["degenerate_blocks"].all(), (
        "a 4-block interval must be flagged; leaving it unflagged is the D4 defect"
    )

    # Same games, same estimate -- only the blocking choice differs. Week
    # blocking gives 24 blocks and is not flagged, which is why the companion
    # week-blocked row for those same 997 games was trustworthy while the
    # season-blocked one was not.
    week_blocked = paired_feature_comparisons(
        predictions, baseline_feature_set="baseline", samples=50, block="week", seed=7
    )
    assert week_blocked["blocks"].eq(24).all()
    assert not week_blocked["degenerate_blocks"].any()
    assert week_blocked["estimate"].to_numpy() == pytest.approx(
        season_blocked["estimate"].to_numpy()
    )

    with pytest.raises(BootstrapDegeneracyError):
        paired_feature_comparisons(
            predictions,
            baseline_feature_set="baseline",
            samples=50,
            block="season",
            seed=7,
            on_degenerate="raise",
        )


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

    configurations = synthetic.rename(columns={"feature_profile": "candidate_id"})
    configuration_selection = nested_outcome_configuration_selection(
        configurations,
        first_test_season=2020,
        validation_seasons=2,
    )
    assert configuration_selection.predictions["selected_candidate_id"].nunique() == 1
    assert (
        len(FROZEN_PLAYER_MODEL_PROFILES)
        * len(FROZEN_PLAYER_RIDGE_ALPHAS)
        * len(FROZEN_PLAYER_CALIBRATIONS)
        == 48
    )
    corrupted = configurations.copy()
    corrupted_row = corrupted.index[
        corrupted["candidate_id"].eq("player_injuries") & corrupted["season"].eq(2018)
    ][0]
    corrupted.loc[corrupted_row, "home_cover"] = 1.0 - float(
        corrupted.loc[corrupted_row, "home_cover"]
    )
    with pytest.raises(ValueError, match="different validation outcomes"):
        nested_outcome_configuration_selection(
            corrupted,
            first_test_season=2020,
            validation_seasons=2,
        )


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


@pytest.mark.full  # ENG-11: dominates --durations; frozen player model selection
def test_frozen_player_model_selection_reuses_raw_streams(
    model_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = experiments_module.walk_forward_outcomes(
        model_frame,
        start_season=2020,
        min_train_games=80,
        methods=("market_residual",),
    ).predictions.copy()
    raw["season"] = 2018 + np.arange(len(raw)) // 8
    raw["week"] = 1 + np.arange(len(raw)) % 8

    fit_calls: list[tuple[str, float]] = []

    def fake_walk_forward(*args: object, **kwargs: object) -> SimpleNamespace:
        fit_calls.append((str(kwargs["feature_profile"]), float(kwargs["ridge_alpha"])))
        return SimpleNamespace(predictions=raw.copy())

    def fake_calibrate(
        predictions: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        calibrated = predictions.loc[predictions["season"].ge(2018)].copy()
        calibrated["raw_home_cover_probability"] = calibrated["home_cover_probability"]
        calibrated["calibration_method"] = str(kwargs["method"])
        calibrated["calibration_rows"] = 0
        calibrated["calibration_max_gameday"] = pd.NaT
        return calibrated

    monkeypatch.setattr(experiments_module, "walk_forward_outcomes", fake_walk_forward)
    monkeypatch.setattr(
        experiments_module,
        "calibrate_cover_prediction_stream",
        fake_calibrate,
    )
    monkeypatch.setattr(
        experiments_module,
        "validate_outcome_prediction_card",
        lambda *args, **kwargs: None,
    )

    result = run_frozen_player_model_selection(model_frame, min_train_games=80)

    assert len(fit_calls) == 12
    assert len(set(fit_calls)) == 12
    assert result.raw_predictions["raw_candidate_id"].nunique() == 12
    assert result.predictions["candidate_id"].nunique() == 48
    assert set(result.summary["calibration_method"]) == set(FROZEN_PLAYER_CALIBRATIONS)
    assert set(result.nested.predictions["season"]) == set(range(2020, 2026))
