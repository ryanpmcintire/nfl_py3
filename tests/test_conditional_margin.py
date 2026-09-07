"""Conditional margin atoms, support fallbacks, and future-week leakage contracts."""

import numpy as np
import pandas as pd
import pytest

from nfl_ats.conditional_margin import (
    CONDITIONAL_MARGIN_METHODS,
    fit_conditional_margin,
    predict_conditional_margin,
)


def history():
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(200)],
            "season": 2020,
            "week": 1,
            "gameday": pd.Timestamp("2020-09-10"),
            "predicted_margin": np.tile([3.0, 7.0], 100),
            "spread_line": np.tile([2.5, 7.5], 100),
            "result": np.tile([3.0, 7.0], 100),
        }
    )


def test_integer_key_mass_and_half_push_credit():
    lattice = fit_conditional_margin(history(), 3.0, 3.0)
    assert set(lattice.margins) == {3.0, 7.0}
    win, push, loss = lattice.probabilities(3.0)
    assert win + push + loss == pytest.approx(1.0)
    assert push > 0.7
    assert lattice.decision_probability(3.0) == pytest.approx(win + 0.5 * push)
    assert lattice.probabilities(3.5)[1] == 0
    assert lattice.decision_probability(2.5) > lattice.decision_probability(3.5)


def test_conditioning_preserves_absolute_atoms_changes_weights():
    low = fit_conditional_margin(history(), 3.0, 3.5)
    high = fit_conditional_margin(history(), 7.0, 3.5)
    assert np.array_equal(low.margins, high.margins)
    assert low.decision_probability(3.5) < high.decision_probability(3.5)


def test_keyshift_preserves_mass_and_matches_nearest_integer_median():
    original = fit_conditional_margin(history(), 5.2, 3.0)
    shifted = fit_conditional_margin(history(), 5.2, 3.0, method=CONDITIONAL_MARGIN_METHODS[1])
    assert np.array_equal(original.mass, shifted.mass)
    median = shifted.margins[np.searchsorted(np.cumsum(shifted.mass), 0.5)]
    assert abs(median - 5.2) <= 0.5


def test_keyside_support_and_fallback():
    supported = fit_conditional_margin(history(), 3.0, 2.5, method=CONDITIONAL_MARGIN_METHODS[2])
    assert supported.used_keyside
    assert supported.probabilities(3.0)[1] == pytest.approx(1.0)
    fallback = fit_conditional_margin(history(), 3.0, 10.5, method=CONDITIONAL_MARGIN_METHODS[2])
    base = fit_conditional_margin(history(), 3.0, 10.5)
    assert not fallback.used_keyside
    assert np.array_equal(base.mass, fallback.mass)


@pytest.mark.parametrize("method", CONDITIONAL_MARGIN_METHODS)
def test_future_and_same_week_cannot_move_earlier_probability(method):
    past = history()
    target = pd.DataFrame(
        {
            "game_id": ["target"],
            "season": [2020],
            "week": [2],
            "gameday": [pd.Timestamp("2020-09-17")],
            "predicted_margin": [3.0],
            "spread_line": [3.0],
        }
    )
    expected = predict_conditional_margin(past, target, method=method)
    future = past.copy()
    future["game_id"] += "future"
    future["week"] = 3
    future["gameday"] = pd.Timestamp("2020-09-24")
    future["result"] = -50.0
    same = future.copy()
    same["game_id"] += "same"
    same["week"] = 2
    same["gameday"] = pd.Timestamp("2020-09-16")
    actual = predict_conditional_margin(pd.concat([past, future, same]), target, method=method)
    pd.testing.assert_frame_equal(expected, actual)


def test_empty_history_fails_without_smooth_fallback():
    with pytest.raises(ValueError, match="prior completed"):
        fit_conditional_margin(history().iloc[:0], 3.0, 3.0)


def test_noninteger_actual_margins_rejected():
    h = history()
    h.loc[0, "result"] = 3.2
    with pytest.raises(ValueError, match="integers"):
        fit_conditional_margin(h, 3.0, 3.0)


@pytest.mark.parametrize("method", CONDITIONAL_MARGIN_METHODS)
def test_central_mapping_requires_and_uses_margin_pairs(method):
    from nfl_ats.calibration import (
        normalize_residual_smoothing_method,
        smoothed_home_cover_probability,
    )

    assert normalize_residual_smoothing_method(method) == method
    with pytest.raises(ValueError, match="prior completed"):
        smoothed_home_cover_probability(
            np.arange(20.0), np.array([3.0]), np.array([3.0]), method=method
        )
    value = smoothed_home_cover_probability(
        np.arange(20.0),
        np.array([3.0]),
        np.array([3.0]),
        method=method,
        conditional_history=history(),
    )
    assert value[0] == fit_conditional_margin(
        history(), 3.0, 3.0, method=method
    ).decision_probability(3.0)


@pytest.mark.parametrize("command", ["margin-predict", "margin-backtest", "opener-evaluation"])
@pytest.mark.parametrize("method", CONDITIONAL_MARGIN_METHODS)
def test_cli_accepts_additive_method(command, method):
    from nfl_ats.cli import build_parser

    args = [command, "--probability-method", method]
    if command == "margin-predict":
        args += ["--season", "2026", "--week", "1"]
    assert build_parser().parse_args(args).probability_method == method
