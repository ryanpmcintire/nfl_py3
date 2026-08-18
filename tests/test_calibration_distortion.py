from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.calibration_distortion import (
    additive_plant_gamma,
    full_pipeline_home_pick,
    implied_pick_threshold,
    orthogonal_carrier,
    plant_additive_effect,
    sign_only_home_pick,
    standardized_carrier,
)
from nfl_ats.cfb_benchmark import fit_cfb_residual_model
from nfl_ats.data import DataContractError
from nfl_ats.margin import _smoothed_probability
from nfl_ats.purged_cv import synthetic_signal_accuracy, synthetic_signal_beta


def test_full_pipeline_pick_matches_margin_model_predict(cfb_features_frame: pd.DataFrame) -> None:
    """The vectorised pick must equal the production ``home_cover_probability >= 0.5``.

    This is the pin that lets the screen skip ``MarginModel.predict``'s
    per-row Python loop; if the two ever diverge the whole experiment is
    measuring something other than the production rule.
    """

    model = fit_cfb_residual_model(cfb_features_frame)
    forecasts = model.predict(cfb_features_frame)
    predicted_residual = forecasts["predicted_market_residual"].to_numpy(dtype=float)
    expected = forecasts["home_cover_probability"].to_numpy(dtype=float) >= 0.5
    np.testing.assert_array_equal(
        full_pipeline_home_pick(model.residuals, predicted_residual), expected
    )


def test_full_pipeline_pick_matches_smoothed_probability_directly() -> None:
    rng = np.random.default_rng(20260818)
    residuals = rng.normal(loc=0.9, scale=15.4, size=511)
    predicted_residual = rng.normal(loc=0.0, scale=3.0, size=250)
    expected = np.array(
        [_smoothed_probability(residuals + value, 0.0) >= 0.5 for value in predicted_residual]
    )
    np.testing.assert_array_equal(full_pipeline_home_pick(residuals, predicted_residual), expected)


@pytest.mark.parametrize("size", [1, 2, 3, 10, 11, 100, 101])
def test_implied_threshold_reproduces_full_pipeline_pick(size: int) -> None:
    """The full pipeline is exactly a threshold on the predicted residual."""

    rng = np.random.default_rng(size)
    residuals = rng.normal(loc=1.1, scale=12.0, size=size)
    threshold = implied_pick_threshold(residuals)
    grid = np.linspace(-25.0, 25.0, 2001)
    np.testing.assert_array_equal(full_pipeline_home_pick(residuals, grid), grid > threshold)


def test_zero_centred_residuals_make_the_two_estimators_agree() -> None:
    """With a symmetric, zero-median residual sample the location term vanishes."""

    residuals = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    predicted_residual = np.array([-2.0, -0.5, 0.5, 2.0])
    assert implied_pick_threshold(residuals) == pytest.approx(0.0)
    np.testing.assert_array_equal(
        full_pipeline_home_pick(residuals, predicted_residual),
        sign_only_home_pick(predicted_residual),
    )


def test_even_sized_residual_sample_uses_the_upper_median() -> None:
    """An even sample has no exact median, so the threshold lands on an order statistic.

    ``(successes + 0.5) / (n + 1) >= 0.5`` needs ``successes >= n / 2``, and
    with ``n`` even that is the ``n/2``-th largest draw -- the UPPER of the two
    middle values, not their average. On a perfectly symmetric four-draw sample
    this leaves a small nonzero threshold where the median is exactly zero. The
    effect is O(one order-statistic gap) and negligible at the 100-2,500 draw
    sizes ``fit_cfb_residual_model`` actually produces, but it is real and is
    pinned here so it is never mistaken for a location bias.
    """

    residuals = np.array([-3.0, -1.0, 1.0, 3.0])
    assert float(np.median(residuals)) == pytest.approx(0.0)
    assert implied_pick_threshold(residuals) == pytest.approx(-1.0)


def test_positive_residual_location_shifts_the_threshold_negative() -> None:
    """A residual sample sitting above zero buys home picks it would not otherwise make."""

    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert implied_pick_threshold(residuals) == pytest.approx(-3.0)
    picks = full_pipeline_home_pick(residuals, np.array([-2.0]))
    assert bool(picks[0]) is True
    assert bool(sign_only_home_pick(np.array([-2.0]))[0]) is False


def test_empty_residuals_rejected() -> None:
    with pytest.raises(ValueError, match="at least one residual"):
        implied_pick_threshold(np.array([]))
    with pytest.raises(ValueError, match="at least one residual"):
        full_pipeline_home_pick(np.array([]), np.array([1.0]))


def test_orthogonal_carrier_is_pm_one_and_seeded() -> None:
    carrier = orthogonal_carrier(500, seed=7)
    assert set(np.unique(carrier)) == {-1.0, 1.0}
    np.testing.assert_array_equal(carrier, orthogonal_carrier(500, seed=7))
    assert not np.array_equal(carrier, orthogonal_carrier(500, seed=8))
    with pytest.raises(ValueError, match="n must be positive"):
        orthogonal_carrier(0, seed=1)


def test_standardized_carrier_has_unit_scale_and_imputes(cfb_features_frame: pd.DataFrame) -> None:
    frame = cfb_features_frame.copy()
    frame.loc[frame.index[:3], "rest_diff"] = np.nan
    carrier = standardized_carrier(frame, "rest_diff")
    assert np.isfinite(carrier).all()
    assert float(np.mean(carrier)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.std(carrier)) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(KeyError):
        standardized_carrier(frame, "not_a_column")


def test_standardized_carrier_rejects_a_constant_column(cfb_features_frame: pd.DataFrame) -> None:
    frame = cfb_features_frame.copy()
    frame["flat"] = 2.0
    with pytest.raises(ValueError, match="constant"):
        standardized_carrier(frame, "flat")


def test_additive_plant_preserves_the_real_target_underneath() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "spread_line": [-3.0, 1.5, 0.0, 7.0],
            "ats_margin": [2.0, -4.0, 6.0, -1.0],
            "result": [-1.0, -2.5, 6.0, 6.0],
            "home_cover": [1.0, 0.0, 1.0, 0.0],
        }
    )
    carrier = np.array([1.0, -1.0, 1.0, -1.0])
    planted = plant_additive_effect(frame, carrier=carrier, gamma=3.0, column="planted")
    np.testing.assert_allclose(planted["ats_margin"].to_numpy(), [5.0, -7.0, 9.0, -4.0])
    np.testing.assert_allclose(planted["result"].to_numpy(), [2.0, -5.5, 9.0, 3.0])
    np.testing.assert_allclose(planted["home_cover"].to_numpy(), [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(planted["planted"].to_numpy(), carrier)
    # The source frame is untouched.
    np.testing.assert_allclose(frame["ats_margin"].to_numpy(), [2.0, -4.0, 6.0, -1.0])


def test_additive_plant_with_zero_gamma_is_the_identity() -> None:
    frame = pd.DataFrame(
        {
            "spread_line": [1.0, -2.0, 0.5],
            "ats_margin": [3.0, -1.0, 0.25],
            "result": [4.0, -3.0, 0.75],
            "home_cover": [1.0, 0.0, 1.0],
        }
    )
    planted = plant_additive_effect(frame, carrier=np.array([1.0, -1.0, 1.0]), gamma=0.0)
    np.testing.assert_allclose(planted["ats_margin"].to_numpy(), frame["ats_margin"].to_numpy())
    np.testing.assert_allclose(planted["home_cover"].to_numpy(), frame["home_cover"].to_numpy())


def test_additive_plant_validates_its_inputs() -> None:
    frame = pd.DataFrame({"spread_line": [1.0, 2.0], "ats_margin": [0.5, -0.5]})
    with pytest.raises(ValueError, match="carrier length"):
        plant_additive_effect(frame, carrier=np.array([1.0]), gamma=1.0)
    with pytest.raises(ValueError, match="carrier must be finite"):
        plant_additive_effect(frame, carrier=np.array([1.0, np.nan]), gamma=1.0)
    with pytest.raises(ValueError, match="gamma must be finite"):
        plant_additive_effect(frame, carrier=np.array([1.0, -1.0]), gamma=float("inf"))
    with pytest.raises(DataContractError):
        plant_additive_effect(
            frame.drop(columns=["ats_margin"]), carrier=np.array([1.0, -1.0]), gamma=1.0
        )


def test_additive_gamma_matches_the_overwrite_plants_own_scale() -> None:
    """An additive '1.3 point' plant carries the identical coefficient the overwrite plant uses."""

    noise_std = 15.43
    assert additive_plant_gamma(0.513, noise_std) == pytest.approx(
        synthetic_signal_beta(0.513, noise_std)
    )
    assert synthetic_signal_accuracy(
        additive_plant_gamma(0.53, noise_std), noise_std
    ) == pytest.approx(0.53)
