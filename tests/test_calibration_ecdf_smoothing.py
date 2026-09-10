from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from nfl_ats.calibration import (
    RESIDUAL_SMOOTHING_METHODS,
    ResidualSmoother,
    fit_residual_smoother,
    normalize_residual_smoothing_method,
    smoothed_home_cover_probability,
)
from nfl_ats.conditional_margin import CONDITIONAL_MARGIN_METHODS
from nfl_ats.discrete_margin_mapping import DISCRETE_MARGIN_METHODS
from nfl_ats.home_side_mapping import HOME_SIDE_MAPPING_METHODS
from nfl_ats.hybrid_margin import HYBRID_MARGIN_METHODS
from nfl_ats.margin import fit_margin_model


@pytest.mark.parametrize("target", ["margin", "market_residual"])
def test_ecdf_control_arm_reproduces_production_probabilities(
    model_frame: pd.DataFrame, target: str
) -> None:

    model = fit_margin_model(model_frame, target=target, model_name="ridge")  # type: ignore[arg-type]
    rows = model_frame.tail(15)
    residuals_before = model.residuals.copy()
    predictions = model.predict(rows)

    ecdf_probability = smoothed_home_cover_probability(
        model.residuals,
        predictions["predicted_margin"].to_numpy(dtype=float),
        rows["spread_line"].to_numpy(dtype=float),
        method="ecdf",
    )

    np.testing.assert_allclose(
        ecdf_probability,
        predictions["home_cover_probability"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(model.residuals, residuals_before)


def test_unknown_smoothing_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown residual smoothing method"):
        normalize_residual_smoothing_method("bogus")
    with pytest.raises(ValueError, match="Unknown residual smoothing method"):
        fit_residual_smoother(np.random.default_rng(0).normal(size=200), method="bogus")


def test_fit_residual_smoother_guards_small_and_degenerate_samples() -> None:
    with pytest.raises(ValueError, match="At least 10"):
        fit_residual_smoother(np.array([1.0, 2.0, 3.0]), method="ecdf")
    with pytest.raises(ValueError, match="degenerate"):
        fit_residual_smoother(np.full(50, 3.0), method="gaussian")


@pytest.mark.parametrize("method", RESIDUAL_SMOOTHING_METHODS)
def test_survival_is_monotone_and_bounded(method: str) -> None:
    if method in (
        *CONDITIONAL_MARGIN_METHODS,
        *HYBRID_MARGIN_METHODS,
        *HOME_SIDE_MAPPING_METHODS,
        *DISCRETE_MARGIN_METHODS,
    ):
        pytest.skip(
            "conditional-margin methods need prior margin pairs, not a residual sample; "
            "covered by tests/test_conditional_margin.py, tests/test_home_side_mapping.py "
            "and tests/test_discrete_margin_mapping.py"
        )
    rng = np.random.default_rng(20260817)
    residuals = rng.normal(loc=0.9, scale=13.1, size=600)
    smoother = fit_residual_smoother(residuals, method=method)
    thresholds = np.linspace(-40.0, 40.0, 41)
    survival = smoother.survival(thresholds)
    assert np.all(survival >= 0.0) and np.all(survival <= 1.0)
    assert np.all(np.diff(survival) <= 1e-12)
    assert survival[0] > 0.9
    assert survival[-1] < 0.1


@pytest.mark.parametrize("method", ["gaussian", "gaussian_kde", "skew_normal"])
def test_smoothed_methods_broadly_agree_with_ecdf_on_a_gaussian_sample(method: str) -> None:

    rng = np.random.default_rng(7)
    residuals = rng.normal(loc=0.0, scale=13.1, size=800)
    ecdf = fit_residual_smoother(residuals, method="ecdf")
    smoothed = fit_residual_smoother(residuals, method=method)
    thresholds = np.array([-10.0, -3.0, 0.0, 3.0, 10.0])
    ecdf_probability = ecdf.survival(thresholds)
    smoothed_probability = smoothed.survival(thresholds)
    assert np.max(np.abs(smoothed_probability - ecdf_probability)) < 0.08
    assert not np.allclose(smoothed_probability, ecdf_probability)


def test_smoothed_home_cover_probability_matches_manual_smoother_composition() -> None:
    rng = np.random.default_rng(11)
    residuals = rng.normal(loc=0.5, scale=13.0, size=500)
    centers = np.array([1.0, -2.5, 0.0])
    lines = np.array([-3.0, -3.0, 7.0])
    direct = smoothed_home_cover_probability(residuals, centers, lines, method="gaussian")
    smoother = fit_residual_smoother(residuals, method="gaussian")
    manual = smoother.survival(lines - centers)
    np.testing.assert_allclose(direct, manual)


def test_residual_smoother_is_immutable_dataclass() -> None:
    residuals = np.random.default_rng(3).normal(size=50)
    smoother = fit_residual_smoother(residuals, method="ecdf")
    assert isinstance(smoother, ResidualSmoother)
    with pytest.raises(FrozenInstanceError):
        smoother.n = 999  # type: ignore[misc]
