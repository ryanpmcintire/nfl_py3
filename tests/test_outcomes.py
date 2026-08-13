from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import nfl_ats.outcomes as outcomes_module
from nfl_ats.outcomes import (
    OUTCOME_METHODS,
    outcome_bootstrap_intervals,
    score_outcome_week,
    summarize_outcome_method,
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
