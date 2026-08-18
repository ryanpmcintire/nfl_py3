from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.estimation_variance import (
    PairedInterval,
    bagged_values,
    bootstrap_row_indices,
    gate_by_disagreement,
    home_cover_probability_from_center,
    mde80,
    naive_block_bootstrap_interval,
    picks_differ_fraction,
    point_predicted_values,
    refit_aware_paired_interval,
    refit_pick_flip_rate,
    refit_predicted_values,
    refit_value_sd,
    shrink_predicted_margin,
)
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import _smoothed_probability


def _synthetic_frame(n: int, *, seed: int, noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = 3.0 * x1 - 2.0 * x2 + rng.normal(scale=noise, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


# ---------------------------------------------------------------------------
# bootstrap_row_indices
# ---------------------------------------------------------------------------


def test_bootstrap_row_indices_shape_and_range() -> None:
    indices = bootstrap_row_indices(25, n_boot=10, seed=1)
    assert indices.shape == (10, 25)
    assert indices.min() >= 0
    assert indices.max() < 25


def test_bootstrap_row_indices_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        bootstrap_row_indices(0, n_boot=5, seed=1)
    with pytest.raises(ValueError, match="n_boot must be at least 1"):
        bootstrap_row_indices(10, n_boot=0, seed=1)


# ---------------------------------------------------------------------------
# refit / point predictions
# ---------------------------------------------------------------------------


def test_refit_predicted_values_shape() -> None:
    train = _synthetic_frame(200, seed=10, noise=1.0)
    test = _synthetic_frame(30, seed=11, noise=1.0)
    refits = refit_predicted_values(
        train,
        test,
        feature_columns=["x1", "x2"],
        target_column="target",
        ridge_alpha=1.0,
        n_boot=15,
        seed=42,
    )
    assert refits.shape == (15, 30)
    assert np.all(np.isfinite(refits))


def test_near_noiseless_linear_fit_has_low_flip_rate() -> None:
    """With almost no noise and a light penalty, resampling training rows should
    barely move the fitted direction, so the model's own picks should almost
    never flip. This is the low-variance end of the phenomenon the module
    measures; the audit itself found the opposite end (15-22% flip) on real,
    noisy, thin-training CFB data."""

    train = _synthetic_frame(2_000, seed=20, noise=0.01)
    test = _synthetic_frame(100, seed=21, noise=0.01)
    point = point_predicted_values(
        train, test, feature_columns=["x1", "x2"], target_column="target", ridge_alpha=0.01
    )
    refits = refit_predicted_values(
        train,
        test,
        feature_columns=["x1", "x2"],
        target_column="target",
        ridge_alpha=0.01,
        n_boot=100,
        seed=7,
    )
    flip_rate = refit_pick_flip_rate(point, refits)
    assert flip_rate < 0.05


def test_refit_value_sd_and_bagged_values_shapes() -> None:
    refits = np.array([[1.0, 2.0, 3.0], [1.2, 1.8, 3.4], [0.8, 2.2, 2.6]])
    sd = refit_value_sd(refits)
    bagged = bagged_values(refits)
    assert sd.shape == (3,)
    assert bagged.shape == (3,)
    np.testing.assert_allclose(bagged, refits.mean(axis=0))
    assert np.all(sd >= 0.0)


# ---------------------------------------------------------------------------
# home_cover_probability_from_center pinned against margin._smoothed_probability
# ---------------------------------------------------------------------------


def test_home_cover_probability_matches_smoothed_probability() -> None:
    rng = np.random.default_rng(99)
    residuals = rng.normal(loc=0.6, scale=13.1, size=500)
    centers = rng.normal(scale=5.0, size=12)
    lines = rng.normal(scale=4.0, size=12)
    batched = home_cover_probability_from_center(centers, lines, residuals)
    expected = np.array(
        [
            _smoothed_probability(center + residuals, float(line))
            for center, line in zip(centers, lines, strict=True)
        ]
    )
    np.testing.assert_allclose(batched, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# shrink_predicted_margin
# ---------------------------------------------------------------------------


def test_shrink_fraction_bounds() -> None:
    spread = np.array([1.0, -2.0, 0.5])
    raw = np.array([3.0, 1.0, -0.5])
    np.testing.assert_allclose(
        shrink_predicted_margin(spread, raw, shrink_fraction=1.0), spread + raw
    )
    np.testing.assert_allclose(shrink_predicted_margin(spread, raw, shrink_fraction=0.0), spread)
    with pytest.raises(ValueError, match="shrink_fraction"):
        shrink_predicted_margin(spread, raw, shrink_fraction=1.5)
    with pytest.raises(ValueError, match="shrink_fraction"):
        shrink_predicted_margin(spread, raw, shrink_fraction=-0.1)


# ---------------------------------------------------------------------------
# naive_block_bootstrap_interval pinned against paired_feature_comparisons
# ---------------------------------------------------------------------------


def _paired_predictions_frame(n_games: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seasons = rng.integers(2018, 2021, size=n_games)
    weeks = rng.integers(1, 8, size=n_games)
    actual = rng.integers(0, 2, size=n_games).astype(float)
    baseline_prob = np.clip(rng.normal(loc=0.5, scale=0.15, size=n_games), 0.02, 0.98)
    candidate_prob = np.clip(baseline_prob + rng.normal(scale=0.1, size=n_games), 0.02, 0.98)
    rows = []
    for feature_set, probability in (("baseline", baseline_prob), ("candidate", candidate_prob)):
        for index in range(n_games):
            rows.append(
                {
                    "feature_set": feature_set,
                    "game_id": f"g{index}",
                    "season": int(seasons[index]),
                    "week": int(weeks[index]),
                    "home_cover": actual[index],
                    "home_cover_probability": probability[index],
                }
            )
    return pd.DataFrame(rows), actual, baseline_prob, candidate_prob, seasons, weeks


def test_naive_interval_matches_paired_feature_comparisons() -> None:
    predictions, actual, baseline_prob, candidate_prob, seasons, weeks = _paired_predictions_frame(
        120, seed=5
    )
    reference = paired_feature_comparisons(
        predictions,
        baseline_feature_set="baseline",
        samples=3_000,
        block="week",
        seed=20260812,
    )
    reference_row = reference.loc[reference["metric"].eq("accuracy_improvement")].iloc[0]

    block_ids = seasons.astype(np.int64) * 1000 + weeks.astype(np.int64)
    result = naive_block_bootstrap_interval(
        actual, baseline_prob, candidate_prob, block_ids, samples=3_000, seed=20260812
    )
    assert result.estimate == pytest.approx(float(reference_row["estimate"]), abs=1e-9)
    assert result.lower == pytest.approx(float(reference_row["lower"]), abs=0.02)
    assert result.upper == pytest.approx(float(reference_row["upper"]), abs=0.02)
    assert result.probability_positive == pytest.approx(
        float(reference_row["probability_positive"]), abs=0.03
    )


# ---------------------------------------------------------------------------
# refit_aware_paired_interval
# ---------------------------------------------------------------------------


def test_refit_aware_interval_with_zero_refit_variance_matches_naive_exactly() -> None:
    """With no refit variance (every draw's arrays identical), the honest
    interval degenerates to exactly the same algorithm as the naive one: one
    block resample per outer iteration, same rng sequence."""

    _, actual, baseline_prob, candidate_prob, seasons, weeks = _paired_predictions_frame(80, seed=6)
    block_ids = seasons.astype(np.int64) * 1000 + weeks.astype(np.int64)
    naive = naive_block_bootstrap_interval(
        actual, baseline_prob, candidate_prob, block_ids, samples=1_000, seed=123
    )
    constant_baseline = np.tile(baseline_prob, (1_000, 1))
    constant_candidate = np.tile(candidate_prob, (1_000, 1))
    honest = refit_aware_paired_interval(
        actual, constant_baseline, constant_candidate, block_ids, seed=123
    )
    assert honest.estimate == pytest.approx(naive.estimate, abs=1e-12)
    assert honest.lower == pytest.approx(naive.lower, abs=1e-12)
    assert honest.upper == pytest.approx(naive.upper, abs=1e-12)
    assert honest.probability_positive == pytest.approx(naive.probability_positive, abs=1e-12)


def test_refit_aware_interval_widens_with_real_refit_variance() -> None:
    """Adding genuine training-refit variance to the candidate arm must widen
    the interval relative to a naive interval built from the same point
    estimate -- the central empirical claim this module exists to check."""

    rng = np.random.default_rng(2026)
    n_games = 150
    actual = rng.integers(0, 2, size=n_games).astype(float)
    seasons = rng.integers(2018, 2021, size=n_games)
    weeks = rng.integers(1, 8, size=n_games)
    block_ids = seasons.astype(np.int64) * 1000 + weeks.astype(np.int64)
    baseline_point = np.clip(rng.normal(loc=0.5, scale=0.1, size=n_games), 0.05, 0.95)
    candidate_point = np.clip(baseline_point + 0.03, 0.05, 0.95)

    naive = naive_block_bootstrap_interval(
        actual, baseline_point, candidate_point, block_ids, samples=1_500, seed=1
    )

    n_boot = 300
    noisy_candidate = np.clip(
        candidate_point[np.newaxis, :] + rng.normal(scale=0.12, size=(n_boot, n_games)),
        0.01,
        0.99,
    )
    honest = refit_aware_paired_interval(
        actual, baseline_point[np.newaxis, :], noisy_candidate, block_ids, seed=1
    )
    naive_width = naive.upper - naive.lower
    honest_width = honest.upper - honest.lower
    assert honest_width > naive_width


def test_refit_aware_interval_requires_matching_draw_counts() -> None:
    actual = np.array([1.0, 0.0, 1.0])
    block_ids = np.array([1, 1, 2])
    baseline = np.random.default_rng(1).uniform(size=(20, 3))
    candidate = np.random.default_rng(2).uniform(size=(30, 3))
    with pytest.raises(ValueError, match="draw counts"):
        refit_aware_paired_interval(actual, baseline, candidate, block_ids)


# ---------------------------------------------------------------------------
# f lever: picks_differ_fraction / mde80 / gate_by_disagreement
# ---------------------------------------------------------------------------


def test_picks_differ_fraction_matches_manual_count() -> None:
    baseline = np.array([0.6, 0.4, 0.5, 0.51])
    candidate = np.array([0.6, 0.6, 0.5, 0.49])
    # game 0: both >=0.5 agree; game 1: baseline<0.5, candidate>=0.5 differ;
    # game 2: both exactly 0.5, agree; game 3: baseline>=0.5, candidate<0.5 differ.
    assert picks_differ_fraction(baseline, candidate) == pytest.approx(0.5)


def test_mde80_scales_with_sqrt_f_over_n() -> None:
    base = mde80(0.10, 1000)
    halved_f = mde80(0.05, 1000)
    assert halved_f == pytest.approx(base / np.sqrt(2.0))
    quadrupled_n = mde80(0.10, 4000)
    assert quadrupled_n == pytest.approx(base / 2.0)
    with pytest.raises(ValueError, match="f must be non-negative"):
        mde80(-0.1, 100)
    with pytest.raises(ValueError, match="n must be positive"):
        mde80(0.1, 0)


def test_gate_by_disagreement_endpoints_and_monotonic_f() -> None:
    rng = np.random.default_rng(3)
    baseline = np.clip(rng.normal(0.5, 0.1, size=200), 0.01, 0.99)
    candidate = np.clip(baseline + rng.normal(0.0, 0.2, size=200), 0.01, 0.99)

    untouched = gate_by_disagreement(baseline, candidate, threshold=0.0)
    np.testing.assert_allclose(untouched, candidate)

    fully_gated = gate_by_disagreement(baseline, candidate, threshold=2.0)
    np.testing.assert_allclose(fully_gated, baseline)

    f_ungated = picks_differ_fraction(baseline, candidate)
    f_partial = picks_differ_fraction(
        baseline, gate_by_disagreement(baseline, candidate, threshold=0.1)
    )
    assert f_partial <= f_ungated

    with pytest.raises(ValueError, match="threshold must be non-negative"):
        gate_by_disagreement(baseline, candidate, threshold=-0.1)


def test_paired_interval_is_frozen_dataclass_with_kind_tag() -> None:
    interval = PairedInterval(
        estimate=0.01, lower=-0.01, upper=0.03, probability_positive=0.7, samples=100, kind="naive"
    )
    assert interval.kind == "naive"
    with pytest.raises(AttributeError):
        interval.estimate = 0.02  # type: ignore[misc]
