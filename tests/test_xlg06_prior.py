"""Release-blocking tests for the XLG-06 Stage-3 prior spec (no network)."""

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.xlg06_prior import (
    blend_prior,
    bootstrap_slope_ci,
    fit_rating_map,
    prepare_fit_frame,
    prior_mean,
    prior_weight,
    weight_curve,
)


def _frame() -> pd.DataFrame:
    rating = np.array([0.80, 0.85, 0.90, 0.95, 1.00])
    return pd.DataFrame(
        {
            "gsis_id": [f"P{i}" for i in range(5)],
            "position": ["WR"] * 5,
            "year": [2018] * 5,
            "recruit_year_num": [2018] * 5,
            "rookie_season": [2020] * 5,
            "rating_num": rating,
            "rookie_epa": 2.0 * rating - 1.0,
            "rookie_reg_weeks": [10.0] * 5,
        }
    )


def test_fit_recovers_a_synthetic_line() -> None:
    params = fit_rating_map(_frame())
    assert params["intercept"] == pytest.approx(-0.1, abs=1e-9)
    assert params["slope"] == pytest.approx(0.2, abs=1e-9)
    assert params["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert params["n"] == 5


def test_weight_curve_boundaries() -> None:
    assert prior_weight(0.0, n0=300.0) == pytest.approx(1.0)
    assert prior_weight(300.0, n0=300.0) == pytest.approx(0.5)
    assert prior_weight(1e9, n0=300.0) == pytest.approx(0.0, abs=1e-6)
    assert blend_prior(0.9, 5.0, 0.0, intercept=0.0, slope=1.0, n0=300.0) == pytest.approx(0.9)
    assert blend_prior(0.9, 5.0, 1e9, intercept=0.0, slope=1.0, n0=300.0) == pytest.approx(
        5.0, abs=1e-3
    )
    with pytest.raises(DataContractError):
        prior_weight(10.0, n0=0.0)
    with pytest.raises(DataContractError):
        prior_weight(-1.0, n0=300.0)


def test_unexpected_columns_fail_closed() -> None:
    frame = _frame()
    frame["future_epa"] = 1.0
    with pytest.raises(DataContractError, match="unexpected columns"):
        prepare_fit_frame(frame)


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_slope_ci(_frame(), seed=11, samples=50)
    second = bootstrap_slope_ci(_frame(), seed=11, samples=50)
    assert first["slope_ci95"] == pytest.approx(second["slope_ci95"])
    assert first["intercept_ci95"] == pytest.approx(second["intercept_ci95"])


def test_sensitivity_helper_selects_nothing() -> None:
    curves = weight_curve([150.0, 300.0, 600.0], [0.0, 300.0, 1200.0])
    assert set(curves) == {"N0=150", "N0=300", "N0=600"}
    assert curves["N0=300"][0] == pytest.approx(1.0)
    assert all(curves["N0=600"][i] >= curves["N0=150"][i] for i in range(3))
    assert prior_mean(0.9, intercept=1.0, slope=2.0) == pytest.approx(2.8)
