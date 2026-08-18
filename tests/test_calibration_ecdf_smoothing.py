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
from nfl_ats.margin import fit_margin_model


@pytest.mark.parametrize("target", ["margin", "market_residual"])
def test_ecdf_control_arm_reproduces_production_probabilities(
    model_frame: pd.DataFrame, target: str
) -> None:
    """The opt-in path never changes the frozen model unless a caller opts in.

    ``margin.py`` is untouched by this module (calibration.py owns none of
    it and does not import it). This test proves the ``method="ecdf"``
    control arm reads the SAME production probability
    (``MarginModel.predict``'s ``home_cover_probability``) from the SAME
    residual draws -- so every comparison against a smoothed method in
    ``docs/ecdf_smoothing.md`` is genuinely "smoothed vs production", not
    "smoothed vs a drifted reimplementation".
    """

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
    # The smoother is a pure reader: it must never mutate the model's residuals.
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
    rng = np.random.default_rng(20260817)
    residuals = rng.normal(loc=0.9, scale=13.1, size=600)
    smoother = fit_residual_smoother(residuals, method=method)
    thresholds = np.linspace(-40.0, 40.0, 41)
    survival = smoother.survival(thresholds)
    assert np.all(survival >= 0.0) and np.all(survival <= 1.0)
    # A survival function P(X > t) is non-increasing in t.
    assert np.all(np.diff(survival) <= 1e-12)
    # Deep in either tail, probability should be near its bound (loose check;
    # this is a sanity property of any of the four estimators, not a precise
    # calibration claim).
    assert survival[0] > 0.9
    assert survival[-1] < 0.1


@pytest.mark.parametrize("method", ["gaussian", "gaussian_kde", "skew_normal"])
def test_smoothed_methods_broadly_agree_with_ecdf_on_a_gaussian_sample(method: str) -> None:
    """Smoothing should not be a wild departure from the ECDF it replaces.

    Draws are genuinely Gaussian here (matching this project's own measured
    near-Gaussian ATS residual, sd ~13.1), so every method should land close
    to the analytic normal survival function -- while still being numerically
    distinct from the raw ECDF, since the whole point is denoising it.
    """

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
