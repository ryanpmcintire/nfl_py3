from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nfl_ats.estimation_variance import (
    MIN_BLOCKS_FOR_INTERVAL,
    RELIABLE_BLOCKS_FOR_INTERVAL,
    BootstrapDegeneracyError,
    BootstrapDegeneracyWarning,
    PairedInterval,
    _normal_cdf,
    _normal_quantile,
    bagged_values,
    block_bootstrap_means,
    block_count_verdict,
    bootstrap_row_indices,
    distinct_block_resamples,
    gate_by_disagreement,
    guard_block_count,
    home_cover_probability_from_center,
    inflate_recorded_interval,
    mde80,
    naive_block_bootstrap_interval,
    paired_refit_predicted_values,
    picks_differ_fraction,
    point_predicted_values,
    refit_aware_interval,
    refit_aware_paired_interval,
    refit_common_variance,
    refit_pick_flip_rate,
    refit_predicted_values,
    refit_value_sd,
    refit_variance_decomposition,
    shrink_predicted_margin,
)
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import _smoothed_probability, make_margin_estimator


def _synthetic_frame(n: int, *, seed: int, noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = 3.0 * x1 - 2.0 * x2 + rng.normal(scale=noise, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


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


def test_refit_aware_interval_with_zero_refit_variance_matches_naive_exactly() -> None:

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


def test_picks_differ_fraction_matches_manual_count() -> None:
    baseline = np.array([0.6, 0.4, 0.5, 0.51])
    candidate = np.array([0.6, 0.6, 0.5, 0.49])
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
    assert interval.degenerate is False
    assert interval.block_count is None
    with pytest.raises(AttributeError):
        interval.estimate = 0.02  # type: ignore[misc]


def test_distinct_block_resamples_counts_multisets_not_ordered_tuples() -> None:

    assert [distinct_block_resamples(k) for k in (1, 2, 3, 4, 5)] == [1, 3, 10, 35, 126]
    with pytest.raises(ValueError, match="block_count must be at least 1"):
        distinct_block_resamples(0)


def test_block_count_verdict_classifies_against_the_measured_floor() -> None:
    one = block_count_verdict(1)
    assert one.collapses_to_point and one.degenerate
    four = block_count_verdict(4)
    assert four.degenerate and not four.collapses_to_point
    marginal = block_count_verdict(MIN_BLOCKS_FOR_INTERVAL)
    assert not marginal.degenerate and marginal.marginal
    reliable = block_count_verdict(RELIABLE_BLOCKS_FOR_INTERVAL)
    assert not reliable.degenerate and not reliable.marginal


def test_guard_block_count_raises_warns_or_ignores() -> None:
    with pytest.raises(BootstrapDegeneracyError, match="bootstrap blocks"):
        guard_block_count(4, on_degenerate="raise")
    with pytest.warns(BootstrapDegeneracyWarning):
        verdict = guard_block_count(4, on_degenerate="warn")
    assert verdict.degenerate
    assert guard_block_count(4, on_degenerate="ignore").degenerate


def test_one_block_bootstrap_collapses_to_a_point_and_is_flagged() -> None:

    actual = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    baseline = np.array([0.4, 0.4, 0.6, 0.4, 0.6, 0.4])
    candidate = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
    blocks = np.zeros(len(actual), dtype=np.int64)
    with pytest.warns(BootstrapDegeneracyWarning):
        interval = naive_block_bootstrap_interval(
            actual, baseline, candidate, blocks, samples=200, seed=1
        )
    assert interval.upper == pytest.approx(interval.lower)
    assert interval.probability_positive == 1.0
    assert interval.degenerate is True
    assert interval.block_count == 1


@pytest.mark.parametrize("blocks", [1, 2, 3, 4, 5, 6, 8, 9])
def test_low_block_interval_is_never_reported_as_valid(blocks: int) -> None:

    rng = np.random.default_rng(4242)
    per_block = 12
    n = blocks * per_block
    actual = rng.integers(0, 2, size=n).astype(float)
    baseline = np.clip(rng.normal(0.5, 0.1, size=n), 0.05, 0.95)
    candidate = np.clip(baseline + rng.normal(0.02, 0.1, size=n), 0.05, 0.95)
    block_ids = np.repeat(np.arange(blocks), per_block)

    with pytest.warns(BootstrapDegeneracyWarning):
        naive = naive_block_bootstrap_interval(
            actual, baseline, candidate, block_ids, samples=300, seed=7
        )
    assert naive.degenerate is True, "a below-floor interval must be flagged, never silent"
    assert naive.block_count == blocks

    refits = np.tile(candidate, (12, 1)) + rng.normal(scale=0.02, size=(12, n))
    with pytest.raises(BootstrapDegeneracyError):
        refit_aware_interval(actual, baseline[np.newaxis, :], refits, block_ids, samples=300)
    with pytest.raises(BootstrapDegeneracyError):
        refit_variance_decomposition(
            actual, baseline[np.newaxis, :], refits, block_ids, samples=300
        )


def test_interval_at_the_floor_is_not_flagged_degenerate() -> None:
    rng = np.random.default_rng(99)
    per_block = 12
    blocks = MIN_BLOCKS_FOR_INTERVAL
    n = blocks * per_block
    actual = rng.integers(0, 2, size=n).astype(float)
    baseline = np.clip(rng.normal(0.5, 0.1, size=n), 0.05, 0.95)
    candidate = np.clip(baseline + rng.normal(0.02, 0.1, size=n), 0.05, 0.95)
    block_ids = np.repeat(np.arange(blocks), per_block)
    interval = naive_block_bootstrap_interval(
        actual, baseline, candidate, block_ids, samples=300, seed=7
    )
    assert interval.degenerate is False
    assert interval.block_count == blocks


@pytest.mark.parametrize("paired", [True, False])
def test_paired_refits_match_original_dataframe_algorithm_exactly(paired: bool) -> None:

    train = _synthetic_frame(60, seed=81, noise=3.0)
    test = _synthetic_frame(11, seed=82, noise=3.0)
    train.loc[[2, 17], "x2"] = np.nan
    test.loc[4, "x1"] = np.nan
    n_boot = 5
    seed = 29
    indices = bootstrap_row_indices(len(train), n_boot=n_boot, seed=seed)
    candidate_indices = (
        indices if paired else bootstrap_row_indices(len(train), n_boot=n_boot, seed=seed + 991)
    )
    target = train["target"].to_numpy(dtype=float)
    expected_baseline = np.empty((n_boot, len(test)))
    expected_candidate = np.empty((n_boot, len(test)))
    for draw in range(n_boot):
        baseline = make_margin_estimator("ridge", 42, ridge_alpha=0.5)
        baseline.fit(train.iloc[indices[draw]].loc[:, ["x1"]], target[indices[draw]])
        expected_baseline[draw] = baseline.predict(test.loc[:, ["x1"]])

        candidate = make_margin_estimator("ridge", 42, ridge_alpha=2.0)
        candidate.fit(
            train.iloc[candidate_indices[draw]].loc[:, ["x1", "x2"]],
            target[candidate_indices[draw]],
        )
        expected_candidate[draw] = candidate.predict(test.loc[:, ["x1", "x2"]])

    actual = paired_refit_predicted_values(
        train,
        test,
        baseline_feature_columns=["x1"],
        candidate_feature_columns=["x1", "x2"],
        target_column="target",
        ridge_alpha=1.0,
        baseline_ridge_alpha=0.5,
        candidate_ridge_alpha=2.0,
        n_boot=n_boot,
        seed=seed,
        paired=paired,
    )
    np.testing.assert_array_equal(actual.row_indices, indices)
    np.testing.assert_array_equal(actual.baseline, expected_baseline)
    np.testing.assert_array_equal(actual.candidate, expected_candidate)


def test_paired_refits_fit_both_arms_on_the_same_resampled_rows() -> None:

    train = _synthetic_frame(300, seed=1, noise=5.0)
    test = _synthetic_frame(40, seed=2, noise=5.0)
    refits = paired_refit_predicted_values(
        train,
        test,
        baseline_feature_columns=["x1"],
        candidate_feature_columns=["x1", "x2"],
        target_column="target",
        ridge_alpha=1.0,
        n_boot=8,
        seed=5,
    )
    assert refits.paired is True
    assert refits.row_indices.shape == (8, 300)
    np.testing.assert_array_equal(refits.row_indices, bootstrap_row_indices(300, n_boot=8, seed=5))
    np.testing.assert_allclose(
        refits.baseline,
        refit_predicted_values(
            train,
            test,
            feature_columns=["x1"],
            target_column="target",
            ridge_alpha=1.0,
            n_boot=8,
            seed=5,
        ),
    )
    np.testing.assert_allclose(
        refits.candidate,
        refit_predicted_values(
            train,
            test,
            feature_columns=["x1", "x2"],
            target_column="target",
            ridge_alpha=1.0,
            n_boot=8,
            seed=5,
        ),
    )


@pytest.mark.full
def test_unpaired_refits_overstate_the_refit_variance() -> None:

    train = _synthetic_frame(400, seed=11, noise=8.0)
    test = _synthetic_frame(240, seed=12, noise=8.0)
    actual = (test["target"].to_numpy() > 0.0).astype(float)
    block_ids = np.repeat(np.arange(20), 12)

    common = {
        "baseline_feature_columns": ["x1"],
        "candidate_feature_columns": ["x1", "x2"],
        "target_column": "target",
        "ridge_alpha": 1.0,
        "n_boot": 120,
        "seed": 3,
    }
    paired = paired_refit_predicted_values(train, test, **common)  # type: ignore[arg-type]
    unpaired = paired_refit_predicted_values(train, test, paired=False, **common)  # type: ignore[arg-type]
    assert paired.paired is True
    assert unpaired.paired is False

    def spread(candidate: np.ndarray) -> float:
        return refit_variance_decomposition(
            actual,
            1.0 / (1.0 + np.exp(-paired.baseline)),
            1.0 / (1.0 + np.exp(-candidate)),
            block_ids,
            samples=2_000,
            seed=17,
        ).refit_fixed_games_sd

    assert spread(unpaired.candidate) > spread(paired.candidate)


def test_block_bootstrap_means_matches_the_explicit_loop_distributionally() -> None:

    rng = np.random.default_rng(5)
    values = rng.normal(size=600)
    block_ids = np.repeat(np.arange(30), 20)
    fast = block_bootstrap_means(values, block_ids, samples=40_000, seed=1)

    grouped = [np.flatnonzero(block_ids == group) for group in range(30)]
    loop_rng = np.random.default_rng(2)
    slow = np.array(
        [
            float(
                np.mean(
                    values[
                        np.concatenate(
                            [grouped[index] for index in loop_rng.integers(0, 30, size=30)]
                        )
                    ]
                )
            )
            for _ in range(40_000)
        ]
    )
    assert float(np.mean(fast)) == pytest.approx(float(np.mean(slow)), abs=0.003)
    assert float(np.std(fast)) == pytest.approx(float(np.std(slow)), rel=0.03)
    assert float(np.quantile(fast, 0.025)) == pytest.approx(
        float(np.quantile(slow, 0.025)), abs=0.01
    )
    assert float(np.quantile(fast, 0.975)) == pytest.approx(
        float(np.quantile(slow, 0.975)), abs=0.01
    )


def test_refit_common_variance_recovers_a_planted_common_component() -> None:

    rng = np.random.default_rng(7)
    n_boot, n_games = 600, 400
    common_sd, interaction_sd = 0.02, 0.80
    common = rng.normal(scale=common_sd, size=(n_boot, 1))
    interaction = rng.normal(scale=interaction_sd, size=(n_boot, n_games))
    components = refit_common_variance(common + interaction, splits=40, seed=3)

    assert components.common == pytest.approx(common_sd**2, rel=0.35)
    assert components.fixed_games == pytest.approx(
        common_sd**2 + interaction_sd**2 / n_games, rel=0.25
    )
    assert components.fixed_games > 3.0 * components.common
    assert components.common_se > 0.0
    assert components.draws == n_boot


def test_refit_common_variance_returns_near_zero_for_pure_interaction() -> None:

    rng = np.random.default_rng(11)
    interaction = rng.normal(scale=0.5, size=(300, 600))
    components = refit_common_variance(interaction, splits=40, seed=5)
    assert components.common_raw == pytest.approx(0.0, abs=3.0 * components.common_se)
    assert components.fixed_games > 0.0


def _honest_inputs(
    n_boot: int, *, common_skill: float = 0.0, refit_noise: float = 0.0, seed: int = 20260818
):

    rng = np.random.default_rng(seed)
    n_games = 600
    actual = rng.integers(0, 2, size=n_games).astype(float)
    block_ids = np.repeat(np.arange(30), 20)
    baseline = np.clip(rng.normal(0.5, 0.12, size=n_games), 0.02, 0.98)
    candidate = np.clip(baseline + 0.03, 0.02, 0.98)
    draws = candidate[np.newaxis, :] + np.zeros((n_boot, 1))
    if common_skill:
        skill = rng.normal(scale=common_skill, size=(n_boot, 1))
        draws = draws + skill * (actual - 0.5)[np.newaxis, :]
    if refit_noise:
        draws = draws + rng.normal(scale=refit_noise, size=(n_boot, n_games))
    return actual, block_ids, baseline, candidate, np.clip(draws, 0.01, 0.99)


def test_refit_aware_interval_reduces_to_the_naive_one_without_refit_variance() -> None:
    actual, block_ids, baseline, candidate, _ = _honest_inputs(40)
    result = refit_aware_interval(
        actual,
        baseline[np.newaxis, :],
        np.tile(candidate, (40, 1)),
        block_ids,
        samples=4_000,
        seed=9,
    )
    assert result.decomposition.refit_sd == pytest.approx(0.0)
    assert result.decomposition.inflation_factor == pytest.approx(1.0)
    assert result.honest.lower == pytest.approx(result.naive.lower)
    assert result.honest.upper == pytest.approx(result.naive.upper)
    assert result.honest.probability_positive == pytest.approx(result.naive.probability_positive)


def test_refit_aware_interval_ignores_pure_interaction_noise() -> None:

    actual, block_ids, baseline, candidate, refits = _honest_inputs(200, refit_noise=0.12)
    result = refit_aware_interval(
        actual,
        baseline[np.newaxis, :],
        refits,
        block_ids,
        point_baseline_prob=baseline,
        point_candidate_prob=candidate,
        samples=8_000,
        seed=9,
    )
    assert result.decomposition.refit_fixed_games_sd > 0.0
    assert result.decomposition.interaction_double_counted_factor > 1.05
    assert result.decomposition.inflation_factor < 1.02


def test_refit_aware_interval_widens_by_the_derived_factor() -> None:
    actual, block_ids, baseline, candidate, refits = _honest_inputs(
        400, common_skill=0.30, refit_noise=0.05
    )
    result = refit_aware_interval(
        actual,
        baseline[np.newaxis, :],
        refits,
        block_ids,
        point_baseline_prob=baseline,
        point_candidate_prob=candidate,
        samples=8_000,
        seed=9,
    )
    decomposition = result.decomposition
    assert decomposition.refit_sd > 0.0
    assert decomposition.inflation_factor == pytest.approx(
        math.hypot(decomposition.conditional_sd, decomposition.refit_sd)
        / decomposition.conditional_sd
    )
    assert decomposition.inflation_factor_upper >= decomposition.inflation_factor
    naive_width = result.naive.upper - result.naive.lower
    honest_width = result.honest.upper - result.honest.lower
    assert honest_width > naive_width
    assert honest_width / naive_width == pytest.approx(decomposition.inflation_factor, rel=0.02)
    assert result.honest.estimate == pytest.approx(result.naive.estimate)


def test_refit_aware_interval_requires_at_least_two_refit_draws() -> None:
    actual, block_ids, baseline, candidate, _ = _honest_inputs(2)
    with pytest.raises(ValueError, match="At least 2 refit draws"):
        refit_aware_interval(
            actual, baseline[np.newaxis, :], candidate[np.newaxis, :], block_ids, samples=1_000
        )


def test_normal_helpers_round_trip() -> None:
    for probability in (0.001, 0.014, 0.11, 0.5, 0.758, 0.8735, 0.999):
        assert _normal_cdf(_normal_quantile(probability)) == pytest.approx(probability, abs=1e-9)
    with pytest.raises(ValueError, match="p must be strictly between 0 and 1"):
        _normal_quantile(0.0)


def test_inflate_recorded_interval_agrees_from_interval_or_from_probability() -> None:

    from_interval = inflate_recorded_interval(0.537, -0.419, 1.532, inflation_factor=1.326)
    implied = inflate_recorded_interval(
        0.537, None, None, inflation_factor=1.326, probability_positive=0.8735
    )
    assert from_interval.probability_positive == pytest.approx(
        implied.probability_positive, abs=0.02
    )
    assert from_interval.kind == "recorded_inflated"
    assert from_interval.estimate == pytest.approx(0.537)
    assert from_interval.probability_positive < 0.8735


def test_inflate_recorded_interval_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="inflation_factor must be at least 1"):
        inflate_recorded_interval(0.5, -0.4, 1.5, inflation_factor=0.9)
    with pytest.raises(ValueError, match="Cannot recover a conditional SD"):
        inflate_recorded_interval(0.5, None, None, inflation_factor=1.2)
