from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.clv import week_blocked_bootstrap

SPEC = importlib.util.spec_from_file_location(
    "residual_offset_eval",
    Path(__file__).resolve().parents[1] / "scripts/residual_offset_opener_eval.py",
)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


@pytest.mark.parametrize("arm", study.ARMS)
def test_estimators_match_definitions(arm):
    values = np.arange(1.0, 258.0)
    values[-1] = 1000.0
    longer = np.array([-4.0, 2.0, 8.0])
    offsets = study.estimate_offsets(values, longer)
    expected = {
        "zero": 0,
        "production": values.sum() / len(values),
        "shrink_025": values.mean() / 4,
        "shrink_050": values.mean() / 2,
        "shrink_075": values.mean() * 3 / 4,
        "short_128": values[-128:].mean(),
        "long_40pct": 2,
        "median": 129,
    }
    for half in (64, 256):
        weights = np.array([2 ** (-(len(values) - 1 - i) / half) for i in range(len(values))])
        expected[f"ew_{half}"] = (values * weights).sum() / weights.sum()
    assert offsets[arm] == pytest.approx(expected[arm])


def test_short_history_and_invalid_residuals():
    assert study.estimate_offsets(np.array([1, 9]), np.array([0]))["short_128"] == 5
    for values in (np.array([]), np.array([np.nan]), np.array([np.inf])):
        with pytest.raises(ValueError):
            study.estimate_offsets(values, np.array([1]))


def test_every_offset_excludes_decision_and_future_outcomes(model_frame):
    features = model_frame.copy()
    features["gameday"] = pd.date_range("2010-01-01", periods=len(features))
    cutoff = features.iloc[120]["gameday"]
    config = {"regressor": "ridge", "ridge_alpha": 10.0, "feature_profile": "base"}
    model, offsets, training = study.weekly_models(features, cutoff, config)
    changed = features.copy()
    changed.loc[changed["gameday"].ge(cutoff), ["result", "ats_margin"]] = 999999
    second, after, _ = study.weekly_models(changed, cutoff, config)
    assert training["gameday"].max() < cutoff
    assert model.distribution_rows == 24
    assert offsets == after
    np.testing.assert_array_equal(model.residuals, second.residuals)
    # Input ordering cannot turn an in-week result into a prior residual.
    _, shuffled, _ = study.weekly_models(changed.sample(frac=1, random_state=4), cutoff, config)
    assert offsets == shuffled


def synthetic_predictions():
    frame = pd.DataFrame(
        {"season": [2020] * 6, "week": [1, 1, 2, 2, 2, 3], "margin_vs_open": [1, -1, 1, 1, -1, 0]}
    )
    for i, arm in enumerate(study.ARMS):
        frame[f"pick_{arm}"] = [(j + i) % 3 == 0 for j in range(6)]
        frame[f"correct_{arm}"] = (
            frame[f"pick_{arm}"]
            .eq(frame["margin_vs_open"].gt(0))
            .astype(float)
            .where(frame["margin_vs_open"].ne(0))
        )
    return frame


def test_fast_bootstrap_matches_repository_whole_week_draws():
    frame = synthetic_predictions().iloc[:5]
    estimate, draws = study.paired_bootstrap(frame, 200, 42)
    reference = week_blocked_bootstrap(
        frame,
        lambda f: {"delta": 100 * (f["correct_zero"] - f["correct_production"]).mean()},
        samples=200,
        seed=42,
    ).iloc[0]
    assert estimate[0] == pytest.approx(reference["estimate"])
    assert np.quantile(draws[:, 0], 0.025) == pytest.approx(reference["lower"])
    assert np.quantile(draws[:, 0], 0.975) == pytest.approx(reference["upper"])
    assert (draws[:, 0] > 0).mean() == reference["probability_positive"]


def test_null_freezes_picks_and_summary_excludes_pushes():
    frame = synthetic_predictions()
    before = frame.copy(deep=True)
    a = study.frozen_pick_null(frame.iloc[:5], 100, 7)
    b = study.frozen_pick_null(frame.iloc[:5], 100, 7)
    np.testing.assert_array_equal(a, b)
    assert (a[:, study.ARMS.index("production")] == 0).all()
    pd.testing.assert_frame_equal(frame, before)
    result = study.summarize(frame, samples=100, null_samples=100)
    assert result["archive_rows"] == 6 and result["graded_games"] == 5
    assert result["pushes"] == 1
    assert result["arms"]["production"]["delta"] == 0
    assert result["arms"]["production"]["probability_positive"] == 0


def test_prediction_frame_reproduces_probability_boundary_and_checks_drift(model_frame):
    features = model_frame.copy()
    features["gameday"] = pd.date_range("2010-01-01", periods=len(features))
    features["season"] = 2019
    features.loc[features.index[-6:], "season"] = 2020
    features.loc[features.index[-6:], "week"] = 1
    config = {"regressor": "ridge", "ridge_alpha": 10.0, "feature_profile": "base"}
    targets = features.tail(5).copy()
    targets["season"] = 2020
    targets["week"] = 1
    model, _, _ = study.weekly_models(features, targets["gameday"].min(), config)
    baseline = model.predict(targets, probability_method="gaussian")
    archive = targets[["game_id", "season", "week"]].reset_index(drop=True)
    archive["tue_open_home_spread"] = targets["spread_line"].to_numpy()
    archive["residual_at_open"] = baseline["predicted_market_residual"].to_numpy()
    archive["home_cover_probability_at_open"] = baseline["home_cover_probability"].to_numpy()
    archive["pick_home_at_open_probability_rule"] = (
        baseline["home_cover_probability"].ge(0.5).to_numpy()
    )
    archive["margin_vs_open"] = targets["result"].to_numpy() - targets["spread_line"].to_numpy()
    # Reverse archive order to test game-id alignment against the model's positional output.
    result = study.build_predictions(features, archive.iloc[::-1], config)
    assert result["pick_production"].equals(result["pick_home_at_open_probability_rule"])
    assert (result["training_max_gameday"] < result["decision_cutoff"]).all()
    assert result["decision_cutoff"].eq(features.iloc[-6]["gameday"]).all()
    assert result["archive_refit_cutoff"].eq(features.iloc[-5]["gameday"]).all()
    # An opening game absent from the quote archive cannot enter candidate offsets.
    clean_model, clean_offsets, _ = study.weekly_models(
        features, features.iloc[-6]["gameday"], config
    )
    assert len(clean_model.residuals) > 0
    for arm in study.ARMS:
        if arm != "production":
            np.testing.assert_allclose(result[f"offset_{arm}"], clean_offsets[arm])
    archive["residual_at_open"] += 1
    with pytest.raises(AssertionError):
        study.build_predictions(features, archive, config)
