from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.variance_reduction import (
    build_cuped_covariates,
    covariate_adjusted_paired_comparisons,
    cuped_adjust,
    fast_block_bootstrap_means,
    paired_block_groups,
    plant_accuracy_effect,
    plant_null_candidate,
    required_sample_size,
    screening_ladder_decision,
)


def _synthetic_paired_predictions(
    n_games: int, n_weeks: int, seed: int, edge: float
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    actual = rng.integers(0, 2, size=n_games).astype(float)
    baseline_probability = np.clip(rng.normal(0.5, 0.12, size=n_games), 0.02, 0.98)
    direction = np.where(actual >= 0.5, 1.0, -1.0)
    candidate_probability = np.clip(baseline_probability + edge * direction, 0.02, 0.98)
    season = 2020 + (np.arange(n_games) % 2)
    week = 1 + (np.arange(n_games) % n_weeks)
    rows = []
    for index in range(n_games):
        rows.append(
            {
                "feature_set": "baseline",
                "game_id": f"g{index}",
                "season": int(season[index]),
                "week": int(week[index]),
                "home_cover": actual[index],
                "home_cover_probability": baseline_probability[index],
            }
        )
        rows.append(
            {
                "feature_set": "candidate",
                "game_id": f"g{index}",
                "season": int(season[index]),
                "week": int(week[index]),
                "home_cover": actual[index],
                "home_cover_probability": candidate_probability[index],
            }
        )
    return pd.DataFrame(rows)


def _synthetic_covariates(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    game_ids = predictions["game_id"].drop_duplicates().sort_values().reset_index(drop=True)
    n_games = len(game_ids)
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "abs_spread_line": rng.uniform(0.5, 14.0, size=n_games),
            "total_line": rng.uniform(35.0, 65.0, size=n_games),
            "on_key_number": rng.integers(0, 2, size=n_games).astype(float),
            "abs_rest_diff": rng.uniform(0.0, 7.0, size=n_games),
            "week_number": rng.integers(1, 16, size=n_games).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# cuped_adjust: the unbiasedness identity
# ---------------------------------------------------------------------------


def test_cuped_adjust_preserves_mean_for_arbitrary_theta() -> None:
    rng = np.random.default_rng(1)
    n = 500
    values = rng.normal(0.02, 1.0, size=n)
    covariates = rng.normal(size=(n, 4))
    covariates[:, 1] = rng.uniform(-10, 10, size=n)

    # The identity does not depend on theta being any particular value: any
    # theta leaves the sample mean unchanged, because the centered covariate
    # sums to exactly zero.
    for theta in (
        np.zeros(4),
        np.ones(4) * 1e6,
        rng.normal(size=4) * 1000,
    ):
        centered = covariates - covariates.mean(axis=0)
        adjusted = values - centered @ theta
        assert adjusted.mean() == pytest.approx(values.mean(), abs=1e-8)


def test_cuped_adjust_matches_manual_ols_and_preserves_mean() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.normal(size=n)
    noise = rng.normal(scale=0.5, size=n)
    values = 3.0 * x + noise  # strongly correlated with the covariate

    adjusted, theta, means = cuped_adjust(values, x[:, None])
    assert adjusted.mean() == pytest.approx(values.mean(), abs=1e-8)
    assert theta[0] == pytest.approx(3.0, rel=0.05)
    assert means[0] == pytest.approx(x.mean())
    # Variance should shrink a lot: only the noise term should remain.
    assert np.var(adjusted, ddof=1) < 0.1 * np.var(values, ddof=1)


def test_cuped_adjust_handles_2d_series_and_uncorrelated_covariate() -> None:
    rng = np.random.default_rng(3)
    n = 1000
    values = rng.normal(size=(n, 2))
    covariates = rng.normal(size=(n, 3))
    adjusted, theta, _ = cuped_adjust(values, covariates)
    assert adjusted.shape == values.shape
    assert theta.shape == (3, 2)
    np.testing.assert_allclose(adjusted.mean(axis=0), values.mean(axis=0), atol=1e-8)
    # Uncorrelated covariates should not meaningfully inflate variance.
    assert np.var(adjusted[:, 0], ddof=1) == pytest.approx(np.var(values[:, 0], ddof=1), rel=0.2)


def test_cuped_adjust_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="n_covariates"):
        cuped_adjust(np.zeros(10), np.zeros((9, 2)))


# ---------------------------------------------------------------------------
# build_cuped_covariates
# ---------------------------------------------------------------------------


def test_build_cuped_covariates_derives_expected_columns() -> None:
    features = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "spread_line": [-3.0, 7.0, 2.5, -6.5],
            "total_line": [45.0, np.nan, 51.0, 60.0],
            "rest_diff": [0.0, np.nan, -3.0, 4.0],
            "week": [1, 2, 3, 4],
        }
    )
    covariates = build_cuped_covariates(features)
    assert list(covariates["abs_spread_line"]) == [3.0, 7.0, 2.5, 6.5]
    assert list(covariates["on_key_number"]) == [1.0, 1.0, 0.0, 0.0]
    # Missing rest_diff is imputed to 0, not dropped.
    assert list(covariates["abs_rest_diff"]) == [0.0, 0.0, 3.0, 4.0]
    assert covariates["total_line"].isna().sum() == 1


def test_build_cuped_covariates_requires_core_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_cuped_covariates(pd.DataFrame({"game_id": [1]}))


# ---------------------------------------------------------------------------
# fast block bootstrap plumbing
# ---------------------------------------------------------------------------


def test_paired_block_groups_and_fast_bootstrap_recover_known_mean() -> None:
    rng = np.random.default_rng(4)
    n = 400
    paired = pd.DataFrame(
        {
            "season_baseline": rng.integers(2018, 2021, size=n),
            "week_baseline": rng.integers(1, 18, size=n),
        }
    )
    values = rng.normal(0.01, 1.0, size=(n, 1))
    group_of_row, n_groups = paired_block_groups(paired, "week")
    assert group_of_row.shape == (n,)
    assert n_groups == paired.groupby(["season_baseline", "week_baseline"]).ngroups

    draws = fast_block_bootstrap_means(values, group_of_row, n_groups, samples=3000, seed=99)
    assert draws.shape == (3000, 1)
    assert draws.mean() == pytest.approx(values.mean(), abs=0.05)


# ---------------------------------------------------------------------------
# covariate_adjusted_paired_comparisons
# ---------------------------------------------------------------------------


def test_covariate_adjusted_matches_raw_point_estimate_and_reference_bootstrap() -> None:
    predictions = _synthetic_paired_predictions(n_games=600, n_weeks=17, seed=10, edge=0.05)
    covariates = _synthetic_covariates(predictions, seed=11)

    reference = paired_feature_comparisons(
        predictions, baseline_feature_set="baseline", samples=2000, block="week", seed=20260818
    )
    result = covariate_adjusted_paired_comparisons(
        predictions,
        covariates,
        baseline_feature_set="baseline",
        samples=2000,
        block="week",
        seed=20260818,
    )
    comparisons = result.comparisons
    assert set(comparisons["metric"]) == set(reference["metric"])

    for metric in reference["metric"]:
        ref_row = reference.loc[reference["metric"].eq(metric)].iloc[0]
        adj_row = comparisons.loc[comparisons["metric"].eq(metric)].iloc[0]
        # The point estimate must be identical between the adjusted and raw
        # (production) estimator: this is the unbiasedness guarantee, not a
        # coincidence of the random covariates used here.
        assert adj_row["raw_estimate"] == pytest.approx(ref_row["estimate"], abs=1e-9)
        assert adj_row["estimate"] == pytest.approx(ref_row["estimate"], abs=1e-9)
        # Independent bootstrap draws of the same underlying data should
        # land in the same ballpark.
        assert adj_row["raw_lower"] == pytest.approx(ref_row["lower"], abs=0.03)
        assert adj_row["raw_upper"] == pytest.approx(ref_row["upper"], abs=0.03)


def test_covariate_adjusted_reports_variance_reduction_for_correlated_covariate() -> None:
    rng = np.random.default_rng(12)
    n_games = 1200
    actual = rng.integers(0, 2, size=n_games).astype(float)
    abs_spread = rng.uniform(0.5, 21.0, size=n_games)
    direction = np.where(actual >= 0.5, 1.0, -1.0)
    idiosyncratic_noise = rng.normal(scale=0.02, size=n_games)
    baseline_probability = np.clip(0.5 + idiosyncratic_noise, 0.05, 0.95)
    # The candidate's edge over baseline scales with |spread| -- e.g. a
    # feature that only helps in blowout-prone games. The resulting
    # brier_improvement is therefore strongly (though not perfectly) linear
    # in abs_spread, which is exactly the structure CUPED should remove.
    candidate_probability = np.clip(
        baseline_probability + 0.01 * abs_spread * direction, 0.02, 0.98
    )

    season = np.full(n_games, 2022)
    week = 1 + (np.arange(n_games) % 16)
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature_set": "baseline",
                    "game_id": [f"g{i}" for i in range(n_games)],
                    "season": season,
                    "week": week,
                    "home_cover": actual,
                    "home_cover_probability": baseline_probability,
                }
            ),
            pd.DataFrame(
                {
                    "feature_set": "candidate",
                    "game_id": [f"g{i}" for i in range(n_games)],
                    "season": season,
                    "week": week,
                    "home_cover": actual,
                    "home_cover_probability": candidate_probability,
                }
            ),
        ],
        ignore_index=True,
    )
    covariates = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n_games)],
            "abs_spread_line": abs_spread,
            "total_line": rng.uniform(35, 65, size=n_games),
            "on_key_number": rng.integers(0, 2, size=n_games).astype(float),
            "abs_rest_diff": rng.uniform(0, 7, size=n_games),
            "week_number": week.astype(float),
        }
    )
    result = covariate_adjusted_paired_comparisons(
        predictions,
        covariates,
        baseline_feature_set="baseline",
        samples=1500,
        block="week",
        seed=555,
    )
    brier_row = result.comparisons.loc[result.comparisons["metric"].eq("brier_improvement")].iloc[0]
    assert brier_row["variance_reduction_pct"] > 0.05
    assert brier_row["effective_sample_multiplier"] > 1.0

    spread_effect = result.covariate_effects.loc[
        (result.covariate_effects["metric"].eq("brier_improvement"))
        & (result.covariate_effects["covariate"].eq("abs_spread_line"))
    ].iloc[0]
    assert spread_effect["univariate_variance_reduction_pct"] > 0.05


def test_covariate_adjusted_validates_inputs() -> None:
    predictions = _synthetic_paired_predictions(n_games=20, n_weeks=5, seed=1, edge=0.0)
    covariates = _synthetic_covariates(predictions, seed=2)
    with pytest.raises(ValueError, match="Unknown paired baseline"):
        covariate_adjusted_paired_comparisons(
            predictions, covariates, baseline_feature_set="missing"
        )
    with pytest.raises(ValueError, match="Covariates are missing columns"):
        covariate_adjusted_paired_comparisons(
            predictions,
            covariates.drop(columns=["total_line"]),
            baseline_feature_set="baseline",
        )
    duplicated = pd.concat([covariates, covariates.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per game_id"):
        covariate_adjusted_paired_comparisons(
            predictions, duplicated, baseline_feature_set="baseline"
        )


# ---------------------------------------------------------------------------
# planted effects
# ---------------------------------------------------------------------------


def test_plant_accuracy_effect_hits_target_within_rounding() -> None:
    rng = np.random.default_rng(20)
    n = 9000
    baseline_probability = np.clip(rng.normal(0.5, 0.12, size=n), 0.02, 0.98)
    actual = rng.integers(0, 2, size=n).astype(float)
    for target in (0.0025, 0.005, 0.01, 0.02):
        candidate, achieved, delta = plant_accuracy_effect(
            baseline_probability, actual, target_accuracy_delta=target
        )
        assert achieved == pytest.approx(target, abs=1.5 / n)
        assert delta >= 0.0
        assert candidate.min() >= 1e-6
        assert candidate.max() <= 1.0 - 1e-6


def test_plant_accuracy_effect_never_decreases_accuracy() -> None:
    rng = np.random.default_rng(21)
    n = 500
    baseline_probability = np.clip(rng.normal(0.5, 0.15, size=n), 0.02, 0.98)
    actual = rng.integers(0, 2, size=n).astype(float)
    baseline_accuracy = ((baseline_probability >= 0.5) == actual).mean()
    for target in (0.0, 0.01, 0.05, 0.15):
        candidate, achieved, _ = plant_accuracy_effect(
            baseline_probability, actual, target_accuracy_delta=target
        )
        candidate_accuracy = ((candidate >= 0.5) == actual).mean()
        assert candidate_accuracy >= baseline_accuracy - 1e-9
        assert achieved >= -1e-9


def test_plant_accuracy_effect_rejects_negative_target() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        plant_accuracy_effect(np.array([0.5]), np.array([1.0]), target_accuracy_delta=-0.01)


def test_plant_accuracy_effect_with_noise_hits_target_and_is_two_sided() -> None:
    rng = np.random.default_rng(23)
    n = 9000
    baseline_probability = np.clip(rng.normal(0.5, 0.12, size=n), 0.02, 0.98)
    actual = rng.integers(0, 2, size=n).astype(float)
    noise = rng.normal(scale=0.01, size=n)

    candidate, achieved, delta = plant_accuracy_effect(
        baseline_probability,
        actual,
        target_accuracy_delta=0.01,
        probability_noise=noise,
    )
    assert achieved == pytest.approx(0.01, abs=5.0 / n)
    assert delta >= 0.0

    baseline_error = np.square(baseline_probability - actual)
    candidate_error = np.square(candidate - actual)
    brier_improvement = baseline_error - candidate_error
    # With real per-game noise, the candidate should be WORSE than baseline
    # on plenty of individual games even though better on average -- unlike
    # the noiseless mechanism, where every game's brier_improvement is
    # non-negative by construction.
    assert (brier_improvement < 0).mean() > 0.1
    assert (brier_improvement > 0).mean() > 0.1
    assert brier_improvement.mean() > 0.0


def test_plant_accuracy_effect_noise_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="probability_noise"):
        plant_accuracy_effect(
            np.array([0.5, 0.6]),
            np.array([1.0, 0.0]),
            target_accuracy_delta=0.1,
            probability_noise=np.array([0.01]),
        )


def test_plant_null_candidate_has_zero_expected_accuracy_effect() -> None:
    rng = np.random.default_rng(22)
    n = 800
    baseline_probability = np.clip(rng.normal(0.5, 0.12, size=n), 0.02, 0.98)
    actual = rng.integers(0, 2, size=n).astype(float)
    baseline_accuracy = ((baseline_probability >= 0.5) == actual).mean()

    achieved_deltas = []
    for seed in range(200):
        candidate = plant_null_candidate(baseline_probability, actual, magnitude=0.01, seed=seed)
        candidate_accuracy = ((candidate >= 0.5) == actual).mean()
        achieved_deltas.append(candidate_accuracy - baseline_accuracy)
    assert abs(np.mean(achieved_deltas)) < 0.01


# ---------------------------------------------------------------------------
# screening ladder decision rule
# ---------------------------------------------------------------------------


def test_screening_ladder_decision_flags_only_candidates_that_clear_threshold() -> None:
    comparisons = pd.DataFrame(
        [
            {
                "candidate_feature_set": "a",
                "metric": "brier_improvement",
                "probability_positive": 0.9,
                "estimate": 0.001,
            },
            {
                "candidate_feature_set": "a",
                "metric": "accuracy_improvement",
                "probability_positive": 0.55,
                "estimate": 0.002,
            },
            {
                "candidate_feature_set": "b",
                "metric": "brier_improvement",
                "probability_positive": 0.6,
                "estimate": 0.0002,
            },
            {
                "candidate_feature_set": "b",
                "metric": "accuracy_improvement",
                "probability_positive": 0.5,
                "estimate": 0.0,
            },
        ]
    )
    decision = screening_ladder_decision(comparisons, screen_probability_threshold=0.75)
    decision = decision.set_index("candidate_feature_set")
    assert decision.loc["a", "spend_confirmation_window"]
    assert not decision.loc["b", "spend_confirmation_window"]
    assert decision.loc["a", "confirm_probability_positive"] == pytest.approx(0.55)


def test_screening_ladder_decision_validates_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        screening_ladder_decision(pd.DataFrame({"metric": ["x"]}))


# ---------------------------------------------------------------------------
# required_sample_size
# ---------------------------------------------------------------------------


def test_required_sample_size_interpolates_between_grid_points() -> None:
    power_by_n = pd.DataFrame({"n_games": [100, 200, 400, 800], "power": [0.2, 0.5, 0.7, 0.95]})
    required = required_sample_size(power_by_n, target_power=0.80)
    assert required is not None
    assert 400 < required < 800


def test_required_sample_size_returns_none_when_unreached() -> None:
    power_by_n = pd.DataFrame({"n_games": [100, 200], "power": [0.1, 0.2]})
    assert required_sample_size(power_by_n, target_power=0.80) is None


def test_required_sample_size_returns_smallest_when_already_met() -> None:
    power_by_n = pd.DataFrame({"n_games": [100, 200], "power": [0.9, 0.95]})
    assert required_sample_size(power_by_n, target_power=0.80) == pytest.approx(100.0)
