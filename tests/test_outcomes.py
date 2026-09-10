from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import nfl_ats.outcomes as outcomes_module
from nfl_ats.estimation_variance import BootstrapDegeneracyError, BootstrapDegeneracyWarning
from nfl_ats.key_numbers import summarize_key_number_calibration
from nfl_ats.outcomes import (
    MARGIN_DISTRIBUTION_METHODS,
    OUTCOME_METHODS,
    fit_margin_models_for_week,
    normalize_outcome_methods,
    outcome_bootstrap_intervals,
    score_outcome_week,
    score_outcome_week_line_sweep,
    summarize_outcome_method,
    walk_forward_key_number_mass,
    walk_forward_outcomes,
)


def test_walk_forward_outcomes_compares_common_weeks(model_frame: pd.DataFrame) -> None:
    result = walk_forward_outcomes(
        model_frame,
        start_season=2020,
        min_train_games=80,
        min_edge=0.0,
    )
    assert set(result.predictions["method"]) == set(OUTCOME_METHODS)
    assert len(result.predictions) == 60 * len(OUTCOME_METHODS)
    games_per_method = result.predictions.groupby("method")["game_id"].nunique()
    assert games_per_method.nunique() == 1
    assert set(result.summary["method"]) == set(OUTCOME_METHODS)
    fair = result.summary.loc[result.summary["method"].eq("fair_margin")].iloc[0]
    assert 0.0 <= fair["cover_brier_score"] <= 1.0
    assert fair["margin_mae"] >= 0.0
    assert 0.0 <= fair["win_brier_score"] <= 1.0
    direct = result.summary.loc[result.summary["method"].eq("direct_ats")].iloc[0]
    assert pd.isna(direct.get("margin_mae"))
    intervals = outcome_bootstrap_intervals(result.predictions, samples=20, seed=7)
    residual_brier = intervals.loc[
        intervals["method"].eq("market_residual") & intervals["metric"].eq("cover_brier_score")
    ].iloc[0]
    assert residual_brier["lower"] <= residual_brier["upper"]
    assert pd.notna(residual_brier["delta_vs_market"])


def test_walk_forward_outcomes_can_fit_only_requested_methods(model_frame: pd.DataFrame) -> None:
    result = walk_forward_outcomes(
        model_frame,
        start_season=2020,
        min_train_games=80,
        methods=("market_residual",),
    )
    assert result.predictions["method"].eq("market_residual").all()
    assert len(result.predictions) == 60
    intervals = outcome_bootstrap_intervals(result.predictions, samples=20, seed=7)
    assert intervals["method"].eq("market_residual").all()
    assert "delta_vs_market" not in intervals.columns
    assert normalize_outcome_methods(("direct_ats", "market")) == ("market", "direct_ats")
    with pytest.raises(ValueError, match="Unknown outcome methods"):
        normalize_outcome_methods(("mystery",))
    with pytest.raises(ValueError, match="must be unique"):
        normalize_outcome_methods(("market", "market"))


def test_outcome_bootstrap_intervals_flags_a_degenerate_block_count(
    model_frame: pd.DataFrame,
) -> None:

    predictions = walk_forward_outcomes(
        model_frame, start_season=2020, min_train_games=80, min_edge=0.0
    ).predictions

    with pytest.warns(BootstrapDegeneracyWarning, match="bootstrap blocks"):
        week_blocked = outcome_bootstrap_intervals(predictions, samples=20, seed=7)
    assert week_blocked["blocks"].eq(4).all()
    assert week_blocked["degenerate_blocks"].all(), (
        "a 4-block interval must be flagged; leaving it unflagged is the D4 defect"
    )

    with pytest.raises(BootstrapDegeneracyError):
        outcome_bootstrap_intervals(predictions, samples=20, seed=7, on_degenerate="raise")


def test_one_block_week_one_of_a_season_is_flagged_not_raised(
    model_frame: pd.DataFrame,
) -> None:

    predictions = walk_forward_outcomes(
        model_frame, start_season=2020, min_train_games=80, min_edge=0.0
    ).predictions
    first_week = predictions.loc[
        predictions["season"].eq(2020) & predictions["week"].eq(predictions["week"].min())
    ]
    assert first_week.groupby(["season", "week"]).ngroups == 1

    with pytest.warns(BootstrapDegeneracyWarning, match="collapses to a point"):
        intervals = outcome_bootstrap_intervals(first_week, samples=20, seed=7)

    assert intervals["blocks"].eq(1).all()
    assert intervals["degenerate_blocks"].all(), (
        "a 1-block 'interval' must be flagged; an unflagged one reads as real"
    )
    assert intervals["estimate"].notna().all(), (
        "the point estimate survives a degenerate block count -- it is the "
        "interval that is unavailable, never the signal"
    )
    assert (intervals["lower"] == intervals["upper"]).all(), (
        "one achievable resample collapses the interval onto the estimate"
    )


def test_score_outcome_week_outputs_fair_spreads(model_frame: pd.DataFrame) -> None:
    scored = score_outcome_week(
        model_frame,
        season=2020,
        week=1,
        min_train_games=80,
    )
    assert set(scored["method"]) == set(OUTCOME_METHODS)
    assert scored.loc[scored["method"].eq("fair_margin"), "fair_spread"].notna().all()
    assert (
        scored.loc[scored["method"].eq("market_residual"), "home_cover_probability"].notna().all()
    )


def test_outcome_guards_and_sparse_summary(model_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No completed games"):
        walk_forward_outcomes(model_frame, start_season=2030)
    missing = model_frame.copy()
    missing.loc[missing["season"].eq(2020) & missing["week"].eq(1), "spread_line"] = pd.NA
    with pytest.raises(ValueError, match="spread is missing"):
        score_outcome_week(missing, season=2020, week=1, min_train_games=80)
    summary = summarize_outcome_method(
        pd.DataFrame(
            {
                "method": ["straight_up"],
                "model_name": ["logistic"],
                "result": [3.0],
                "home_win_probability": [0.6],
                "home_cover": [pd.NA],
                "home_cover_probability": [pd.NA],
                "predicted_margin": [pd.NA],
            }
        )
    )
    assert summary["win_accuracy"] == 1.0
    assert "cover_accuracy" not in summary
    with pytest.raises(ValueError, match="samples"):
        outcome_bootstrap_intervals(model_frame, samples=9)


def test_fit_margin_models_for_week_matches_score_outcome_week(model_frame: pd.DataFrame) -> None:
    target, margin_models = fit_margin_models_for_week(
        model_frame, season=2020, week=1, min_train_games=80
    )
    assert set(margin_models) == set(MARGIN_DISTRIBUTION_METHODS)
    expected_games = model_frame.loc[
        model_frame["season"].eq(2020) & model_frame["week"].eq(1), "game_id"
    ]
    assert set(target["game_id"]) == set(expected_games)

    predictions = score_outcome_week(
        model_frame, season=2020, week=1, min_train_games=80, probability_method="ecdf"
    )
    fair_margin_rows = predictions.loc[predictions["method"].eq("fair_margin")].set_index("game_id")
    forecasts = margin_models["fair_margin"].predict(target)
    forecasts.index = target["game_id"].to_numpy()
    assert np.allclose(
        forecasts["home_cover_probability"],
        fair_margin_rows.loc[forecasts.index, "home_cover_probability"],
    )

    with pytest.raises(ValueError, match="Unknown margin-distribution methods"):
        fit_margin_models_for_week(
            model_frame, season=2020, week=1, min_train_games=80, methods=("direct_ats",)
        )


@pytest.mark.parametrize("probability_method", ["gaussian", "gaussian_median", "ecdf"])
def test_score_outcome_week_line_sweep_matches_score_outcome_week_at_zero_offset(
    model_frame: pd.DataFrame,
    probability_method: str,
) -> None:
    sweep = score_outcome_week_line_sweep(
        model_frame,
        season=2020,
        week=1,
        min_train_games=80,
        offsets=(0.0,),
        probability_method=probability_method,
    )
    predictions = score_outcome_week(
        model_frame,
        season=2020,
        week=1,
        min_train_games=80,
        probability_method=probability_method,
    )
    for method in MARGIN_DISTRIBUTION_METHODS:
        method_sweep = sweep.loc[sweep["method"].eq(method)].set_index("game_id")
        method_predictions = predictions.loc[predictions["method"].eq(method)].set_index("game_id")
        for column in (
            "home_cover_probability",
            "home_cover_probability_excluding_push",
            "push_probability",
            "home_loss_probability",
        ):
            if method == "market" and column == "home_cover_probability":
                continue
            assert np.allclose(
                method_sweep.loc[method_predictions.index, column],
                method_predictions[column],
            )

    with pytest.raises(ValueError, match="Unknown margin-distribution methods"):
        score_outcome_week_line_sweep(
            model_frame, season=2020, week=1, min_train_games=80, methods=("direct_ats",)
        )
    with pytest.raises(ValueError, match="At least one margin-distribution method"):
        score_outcome_week_line_sweep(
            model_frame, season=2020, week=1, min_train_games=80, methods=()
        )


def test_walk_forward_key_number_mass_produces_leak_safe_report(
    model_frame: pd.DataFrame,
) -> None:
    mass = walk_forward_key_number_mass(
        model_frame, start_season=2020, min_train_games=80, key_numbers=(3, 7)
    )
    assert set(mass["method"]) == set(MARGIN_DISTRIBUTION_METHODS)
    assert {"key_number_3", "key_number_7", "result", "spread_line"}.issubset(mass.columns)
    assert mass["key_number_3"].between(0.0, 1.0).all()
    summary = summarize_key_number_calibration(mass, key_numbers=(3, 7))
    assert set(summary["method"]) == set(MARGIN_DISTRIBUTION_METHODS)

    with pytest.raises(ValueError, match="No completed games"):
        walk_forward_key_number_mass(model_frame, start_season=2030)
    with pytest.raises(ValueError, match="Unknown margin-distribution methods"):
        walk_forward_key_number_mass(
            model_frame, start_season=2020, min_train_games=80, methods=("direct_ats",)
        )


def _with_extreme_future_weeks(
    frame: pd.DataFrame, *, season: int, after_week: int
) -> pd.DataFrame:

    future_mask = frame["season"].eq(season) & frame["week"].gt(after_week)
    assert int(future_mask.sum()) > 0, "fixture must contain rows after the target week"
    perturbed = frame.copy()
    sign = np.where(np.arange(int(future_mask.sum())) % 2 == 0, 1.0, -1.0)
    perturbed.loc[future_mask, "ats_margin"] = sign * 500.0
    perturbed.loc[future_mask, "result"] = (
        perturbed.loc[future_mask, "spread_line"] + perturbed.loc[future_mask, "ats_margin"]
    )
    return perturbed


def test_walk_forward_key_number_mass_ignores_games_after_the_target_week(
    model_frame: pd.DataFrame,
) -> None:

    baseline = walk_forward_key_number_mass(model_frame, start_season=2020, min_train_games=80)
    corrupted_frame = _with_extreme_future_weeks(model_frame, season=2020, after_week=1)
    corrupted = walk_forward_key_number_mass(corrupted_frame, start_season=2020, min_train_games=80)

    base_week1 = (
        baseline.loc[baseline["week"].eq(1)]
        .sort_values(["method", "game_id"])
        .reset_index(drop=True)
    )
    corrupted_week1 = (
        corrupted.loc[corrupted["week"].eq(1)]
        .sort_values(["method", "game_id"])
        .reset_index(drop=True)
    )
    assert not base_week1.empty
    pd.testing.assert_frame_equal(base_week1, corrupted_week1)


def test_walk_forward_outcomes_ignores_games_after_the_target_week(
    model_frame: pd.DataFrame,
) -> None:

    baseline = walk_forward_outcomes(
        model_frame, start_season=2020, min_train_games=80, min_edge=0.0
    ).predictions
    corrupted_frame = _with_extreme_future_weeks(model_frame, season=2020, after_week=1)
    corrupted = walk_forward_outcomes(
        corrupted_frame, start_season=2020, min_train_games=80, min_edge=0.0
    ).predictions

    base_week1 = (
        baseline.loc[baseline["week"].eq(1)]
        .sort_values(["method", "game_id"])
        .reset_index(drop=True)
    )
    corrupted_week1 = (
        corrupted.loc[corrupted["week"].eq(1)]
        .sort_values(["method", "game_id"])
        .reset_index(drop=True)
    )
    assert not base_week1.empty
    pd.testing.assert_frame_equal(base_week1, corrupted_week1)


def test_score_outcome_week_ignores_games_after_the_target_week(
    model_frame: pd.DataFrame,
) -> None:

    baseline = score_outcome_week(model_frame, season=2020, week=1, min_train_games=80)
    corrupted_frame = _with_extreme_future_weeks(model_frame, season=2020, after_week=1)
    corrupted = score_outcome_week(corrupted_frame, season=2020, week=1, min_train_games=80)

    pd.testing.assert_frame_equal(
        baseline.sort_values(["method", "game_id"]).reset_index(drop=True),
        corrupted.sort_values(["method", "game_id"]).reset_index(drop=True),
    )


def _reference_bootstrap(
    predictions: pd.DataFrame, samples: int, seed: int
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    groups = list(
        predictions.groupby(["season", "week"], sort=False, dropna=False).indices.values()
    )
    estimates = {
        str(method): summarize_outcome_method(group)
        for method, group in predictions.groupby("method", sort=True)
    }
    market = estimates["market"]
    keys = [
        (method, metric)
        for method, metrics in estimates.items()
        for metric in outcomes_module.OUTCOME_UNCERTAINTY_METRICS
        if metric in metrics
    ]
    draws = {key: np.empty(samples) for key in keys}
    deltas = {key: np.empty(samples) for key in keys if key[0] != "market" and key[1] in market}
    generator = np.random.default_rng(seed)
    for sample in range(samples):
        selected = generator.integers(0, len(groups), size=len(groups))
        positions = np.concatenate([groups[index] for index in selected])
        sampled = {
            str(method): summarize_outcome_method(group)
            for method, group in predictions.iloc[positions].groupby("method", sort=True)
        }
        for key in keys:
            method, metric = key
            draws[key][sample] = float(sampled[method][metric])
            if key in deltas:
                deltas[key][sample] = float(sampled[method][metric]) - float(
                    sampled["market"][metric]
                )
    return {
        key: (
            float(np.quantile(draw, 0.025)),
            float(np.quantile(draw, 0.975)),
            float(np.quantile(deltas[key], 0.025)) if key in deltas else np.nan,
            float(np.quantile(deltas[key], 0.975)) if key in deltas else np.nan,
        )
        for key, draw in draws.items()
    }


@pytest.mark.full
def test_vectorized_bootstrap_matches_reference(model_frame: pd.DataFrame) -> None:
    predictions = walk_forward_outcomes(
        model_frame, start_season=2020, min_train_games=80, min_edge=0.0
    ).predictions
    expected = _reference_bootstrap(predictions, samples=20, seed=11)
    actual = outcome_bootstrap_intervals(predictions, samples=20, seed=11).set_index(
        ["method", "metric"]
    )
    for key, (lower, upper, delta_lower, delta_upper) in expected.items():
        assert actual.loc[key, "lower"] == pytest.approx(lower, abs=1e-12)
        assert actual.loc[key, "upper"] == pytest.approx(upper, abs=1e-12)
        if np.isfinite(delta_lower):
            assert actual.loc[key, "delta_lower"] == pytest.approx(delta_lower, abs=1e-12)
            assert actual.loc[key, "delta_upper"] == pytest.approx(delta_upper, abs=1e-12)


def test_bootstrap_does_not_recompute_metrics_per_sample(
    model_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = walk_forward_outcomes(
        model_frame, start_season=2020, min_train_games=80, min_edge=0.0
    ).predictions
    calls = 0
    original = outcomes_module.summarize_outcome_method

    def counted(frame: pd.DataFrame) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(frame)

    monkeypatch.setattr(outcomes_module, "summarize_outcome_method", counted)
    outcome_bootstrap_intervals(predictions, samples=500, seed=7)
    assert calls == predictions["method"].nunique()
