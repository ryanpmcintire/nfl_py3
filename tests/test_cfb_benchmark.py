from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.cfb_benchmark import (
    cfb_benchmark_uncertainty,
    cfb_evaluation_window,
    cfb_walk_forward_benchmark,
    fit_cfb_residual_model,
    summarize_cfb_benchmark,
)
from nfl_ats.data import DataContractError


def test_walk_forward_benchmark_contracts(cfb_features_frame: pd.DataFrame) -> None:
    result = cfb_walk_forward_benchmark(
        cfb_features_frame, start_season=2014, end_season=2014, min_train_games=50
    )
    predictions = result.predictions

    scored_2014 = cfb_features_frame.loc[cfb_features_frame["season"].eq(2014)]
    assert set(predictions["method"]) == {"market", "market_residual"}
    assert len(predictions) == 2 * len(scored_2014)
    assert predictions["evaluation_window"].eq("clean_core").all()
    assert predictions["bet_side"].eq("PASS").all()

    # Chronological integrity: training never reaches the scored game's date.
    cutoffs = pd.to_datetime(predictions["train_max_gameday"])
    assert (cutoffs < predictions["gameday"]).all()

    market = predictions.loc[predictions["method"].eq("market")]
    pd.testing.assert_series_equal(
        market["predicted_margin"],
        market["spread_line"],
        check_names=False,
    )
    assert market["predicted_market_residual"].eq(0.0).all()
    # Fixture odds are -105 home / -115 away, so the no-vig market baseline
    # always picks the away side: an explicit ~50% forced-pick control.
    assert market["home_cover_probability"].lt(0.5).all()

    residual = predictions.loc[predictions["method"].eq("market_residual")]
    assert residual["home_cover_probability"].between(0.0, 1.0).all()
    assert residual["predicted_margin"].notna().all()
    assert residual["train_rows"].ge(50).all()
    assert residual["distribution_rows"].ge(10).all()

    summary, season_summary = summarize_cfb_benchmark(predictions)
    clean = summary.loc[summary["evaluation_window"].eq("clean_core")]
    assert set(clean["method"]) == {"market", "market_residual"}
    assert clean["cover_games"].gt(0).all()
    assert {"cover_accuracy", "cover_brier_score", "margin_mae"}.issubset(summary.columns)
    assert season_summary["season"].eq(2014).all()


def test_min_train_floor_skips_early_weeks(cfb_features_frame: pd.DataFrame) -> None:
    result = cfb_walk_forward_benchmark(
        cfb_features_frame, start_season=2013, end_season=2014, min_train_games=50
    )
    first_scored = result.predictions["gameday"].min()
    training_pool = cfb_features_frame.loc[cfb_features_frame["gameday"].lt(first_scored)]
    assert len(training_pool) >= 50
    weeks = result.predictions.loc[result.predictions["season"].eq(2013), "week"]
    assert not weeks.empty
    assert weeks.min() > 1


def test_uncertainty_reuses_blocked_outcome_bootstrap(
    cfb_features_frame: pd.DataFrame,
) -> None:
    result = cfb_walk_forward_benchmark(
        cfb_features_frame, start_season=2014, end_season=2014, min_train_games=50
    )
    uncertainty = cfb_benchmark_uncertainty(result.predictions, samples=100, seed=7)
    assert set(uncertainty["block"]) == {"week", "season"}
    accuracy = uncertainty.loc[
        uncertainty["metric"].eq("cover_accuracy")
        & uncertainty["method"].eq("market_residual")
        & uncertainty["block"].eq("week")
    ]
    assert len(accuracy) == 1
    row = accuracy.iloc[0]
    assert row["lower"] <= row["estimate"] <= row["upper"]
    assert {"delta_vs_market", "delta_lower", "delta_upper"}.issubset(uncertainty.columns)
    assert pd.notna(row["delta_vs_market"])

    with pytest.raises(ValueError, match="clean-core"):
        cfb_benchmark_uncertainty(
            result.predictions.assign(evaluation_window="thin_2006_2011"), samples=100
        )


def test_residual_model_contracts(cfb_features_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="At least 50"):
        fit_cfb_residual_model(cfb_features_frame.head(10))
    with pytest.raises(DataContractError, match="missing columns"):
        fit_cfb_residual_model(cfb_features_frame.drop(columns=["spread_line"]))
    with pytest.raises(ValueError, match="distribution_fraction"):
        fit_cfb_residual_model(cfb_features_frame, distribution_fraction=0.05)

    model = fit_cfb_residual_model(cfb_features_frame)
    assert model.target == "market_residual"
    assert model.ridge_alpha == 10.0
    assert model.training_rows == len(cfb_features_frame)


def test_evaluation_window_labels() -> None:
    assert cfb_evaluation_window(2006) == "thin_2006_2011"
    assert cfb_evaluation_window(2012) == "clean_core"
    assert cfb_evaluation_window(2020) == "regime_2020"
    assert cfb_evaluation_window(2025) == "clean_core"
    with pytest.raises(ValueError, match="outside"):
        cfb_evaluation_window(2005)


def test_walk_forward_input_contracts(cfb_features_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="end_season"):
        cfb_walk_forward_benchmark(cfb_features_frame, start_season=2014, end_season=2013)
    with pytest.raises(DataContractError, match="missing columns"):
        cfb_walk_forward_benchmark(cfb_features_frame.drop(columns=["ats_margin"]))
    with pytest.raises(ValueError, match="No completed CFB games"):
        cfb_walk_forward_benchmark(
            cfb_features_frame, start_season=2024, end_season=2025, min_train_games=50
        )
    with pytest.raises(ValueError, match="enough prior training"):
        cfb_walk_forward_benchmark(
            cfb_features_frame, start_season=2013, end_season=2014, min_train_games=5_000
        )
