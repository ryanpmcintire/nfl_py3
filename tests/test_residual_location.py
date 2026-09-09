from __future__ import annotations

import numpy as np
import pytest

from nfl_ats.margin import _smoothed_probability
from nfl_ats.residual_location import (
    recency_weighted_home_cover_probability,
    recency_weighted_survival,
    recency_weights,
    shrunk_home_cover_probability,
    shrunk_survival,
)


def test_shrink_fraction_zero_reproduces_production_ecdf() -> None:
    """The shrinkage screen's control arm must match production exactly."""

    rng = np.random.default_rng(20260818)
    residuals = rng.normal(loc=0.9, scale=13.1, size=600)
    thresholds = np.linspace(-10.0, 10.0, 21)
    for threshold in thresholds:
        expected = _smoothed_probability(residuals, float(threshold))
        actual = shrunk_survival(residuals, float(threshold), shrink_fraction=0.0)
        assert actual == pytest.approx(expected, abs=1e-12)


def test_shrink_fraction_one_removes_the_location() -> None:
    """Full shrinkage re-centers the sample at zero before reading the ECDF."""

    rng = np.random.default_rng(1)
    residuals = rng.normal(loc=5.0, scale=10.0, size=500)
    shrunk_at_zero = shrunk_survival(residuals, 0.0, shrink_fraction=1.0)
    assert shrunk_at_zero == pytest.approx(0.5, abs=0.05)


def test_shrink_fraction_rejects_out_of_range_values() -> None:
    residuals = np.random.default_rng(2).normal(size=100)
    with pytest.raises(ValueError, match="shrink_fraction"):
        shrunk_survival(residuals, 0.0, shrink_fraction=1.5)
    with pytest.raises(ValueError, match="shrink_fraction"):
        shrunk_survival(residuals, 0.0, shrink_fraction=-0.1)


def test_shrunk_home_cover_probability_matches_survival() -> None:
    rng = np.random.default_rng(3)
    residuals = rng.normal(loc=1.2, scale=12.0, size=400)
    centers = np.array([3.0, -1.5, 0.0])
    lines = np.array([2.5, -1.0, 0.0])
    probability = shrunk_home_cover_probability(residuals, centers, lines, shrink_fraction=0.5)
    expected = np.array(
        [
            shrunk_survival(residuals, float(lines[i] - centers[i]), shrink_fraction=0.5)[0]
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(probability, expected)


def test_recency_weights_are_monotone_and_endpoint_correct() -> None:
    weights = recency_weights(10, half_life_games=3.0)
    assert len(weights) == 10
    assert np.all(np.diff(weights) > 0.0)
    assert weights[-1] == pytest.approx(1.0)
    assert weights[6] == pytest.approx(0.5, rel=1e-9)


def test_recency_weighted_survival_reweights_toward_recent_draws() -> None:
    """A short half-life should track only the most recent draws' sign mix."""

    residuals = np.concatenate([np.full(200, -5.0), np.full(50, 5.0)])
    long_half_life = recency_weighted_survival(residuals, 0.0, half_life_games=10_000.0)
    short_half_life = recency_weighted_survival(residuals, 0.0, half_life_games=5.0)
    assert long_half_life < 0.3
    assert short_half_life > 0.9


def test_recency_weighted_home_cover_probability_matches_survival() -> None:
    rng = np.random.default_rng(4)
    residuals = rng.normal(loc=0.5, scale=11.0, size=350)
    centers = np.array([1.0, 2.5])
    lines = np.array([0.5, -1.0])
    probability = recency_weighted_home_cover_probability(
        residuals, centers, lines, half_life_games=150.0
    )
    expected = np.array(
        [
            recency_weighted_survival(
                residuals, float(lines[i] - centers[i]), half_life_games=150.0
            )[0]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(probability, expected)


def test_recency_weights_reject_bad_half_life() -> None:
    with pytest.raises(ValueError, match="half_life_games"):
        recency_weights(10, half_life_games=0.0)
    with pytest.raises(ValueError, match="half_life_games"):
        recency_weights(10, half_life_games=float("nan"))
