from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.hybrid_margin import (
    HYBRID_MARGIN_METHODS,
    HybridWeights,
    fit_hybrid_weights,
    key_distance,
    predict_hybrid_margin,
    size_cell,
)


def history() -> pd.DataFrame:
    n = 1200
    return pd.DataFrame(
        {
            "game_id": [str(i) for i in range(n)],
            "season": np.repeat([2020, 2021, 2022, 2023], 300),
            "week": np.tile(np.repeat(np.arange(1, 16), 20), 4),
            "gameday": pd.date_range("2020-01-01", periods=n),
            "spread_line": np.tile([3.0, 7.0, 10.0], 400),
            "result": np.tile([7.0, 14.0, 17.0], 400),
            "predicted_margin": 5.0,
            "p_smooth": 0.55,
            "p_lattice": 0.8,
        }
    )


def test_distances_and_monotone_weights() -> None:
    np.testing.assert_equal(key_distance([3, -3.5, 7, 10.5, 14, 0]), [0, 0.5, 0, 0.5, 0, 3])
    np.testing.assert_equal(size_cell([3, 3.5, -7, 7.5]), [0, 1, 1, 2])
    weights = HybridWeights((0.75, 2)).weights([3, 3.5, 4, 4.5])
    assert np.all(np.diff(weights) < 0)


def test_fit_and_support_fallback() -> None:
    h = history()
    assert fit_hybrid_weights(h).global_pair[0] == 1
    assert len(fit_hybrid_weights(h, method=HYBRID_MARGIN_METHODS[1]).size_pairs) == 3
    small = h.loc[h.spread_line.ne(10)]
    assert not fit_hybrid_weights(small, method=HYBRID_MARGIN_METHODS[1]).size_pairs
    assert fit_hybrid_weights(h.head(199)).global_pair[0] == 0


@pytest.mark.parametrize("method", HYBRID_MARGIN_METHODS)
def test_future_and_same_week_cannot_move_earlier_weights_or_probability(method: str) -> None:
    h = history()
    target = h.tail(20).copy()
    expected = predict_hybrid_margin(h, target, method=method)
    altered = h.copy()
    altered.loc[altered.game_id.isin(target.game_id), ["result", "p_smooth", "p_lattice"]] = [
        -70,
        0.99,
        0.01,
    ]
    future = h.head(30).assign(
        game_id=lambda x: "future" + x.game_id,
        season=2027,
        gameday=pd.Timestamp("2027-10-01"),
        result=-70,
    )
    got = predict_hybrid_margin(
        pd.concat([altered, future], ignore_index=True), target, method=method
    )
    cols = ["hybrid_weight", "hybrid_w0", "hybrid_tau", "home_cover_probability"]
    pd.testing.assert_frame_equal(expected[cols], got[cols])


@pytest.mark.parametrize("method", HYBRID_MARGIN_METHODS)
def test_central_mapper_requires_history_and_blends_components(method: str) -> None:
    residuals = np.linspace(-20, 20, 100)
    with pytest.raises(ValueError, match="history"):
        smoothed_home_cover_probability(residuals, np.array([5.0]), np.array([3.0]), method=method)
    h = history()
    result = smoothed_home_cover_probability(
        residuals, np.array([5.0]), np.array([3.0]), method=method, conditional_history=h
    )
    assert result[0] == pytest.approx(1.0)


def test_invalid_history_rejected() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        fit_hybrid_weights(history().assign(p_lattice=2.0))
