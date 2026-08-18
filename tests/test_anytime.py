from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.anytime import (
    ANYTIME_METRICS,
    DEFAULT_ALPHA,
    PeekingTrialResult,
    anova_intraclass_correlation,
    anytime_summary,
    block_bootstrap_ci_fast,
    bootstrap_intraclass_correlation,
    confidence_sequence_from_block_stats,
    default_prior_variance,
    paired_anytime_comparisons,
    run_peeking_trial,
    simulate_block_sequence,
)
from nfl_ats.experiments import paired_feature_comparisons


def _synthetic_predictions(
    rng: np.random.Generator,
    *,
    seasons: tuple[int, ...] = (2020, 2021),
    weeks_per_season: int = 5,
    baseline_probability: float = 0.5,
    candidate_edge: float = 0.2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    game = 0
    for season in seasons:
        for week in range(1, weeks_per_season + 1):
            n_games = int(rng.integers(3, 12))
            for _ in range(n_games):
                actual = float(rng.integers(0, 2))
                candidate_probability = 0.5 + candidate_edge if actual else 0.5 - candidate_edge
                for feature_set, probability in (
                    ("baseline", baseline_probability),
                    ("candidate", candidate_probability),
                ):
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "game_id": f"g{game}",
                            "season": season,
                            "week": week,
                            "home_cover": actual,
                            "home_cover_probability": probability,
                        }
                    )
                game += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engine correctness
# ---------------------------------------------------------------------------


def test_confidence_sequence_matches_hand_computation_at_the_default_icc() -> None:
    """icc=0 (independence, the project's default): variance is k * s^2, not k^2 * s^2."""

    sizes = np.array([10.0])
    sums = np.array([3.0])
    trace = confidence_sequence_from_block_stats(sizes, sums, alpha=0.05, prior_variance=0.01)

    variance_process = 10.0  # k * s^2 * (1 + (k-1)*0) = 10 * 1.0 * 1 = 10
    denom = 1.0 + 0.01 * variance_process
    expected_log_e = -0.5 * np.log(denom) + 0.01 * 3.0**2 / (2.0 * denom)
    expected_radius = (1.0 / 10.0) * np.sqrt(
        (2.0 * denom / 0.01) * np.log((1.0 / 0.05) * np.sqrt(denom))
    )
    assert trace["log_e_value"].iloc[0] == pytest.approx(expected_log_e)
    assert trace["cumulative_mean"].iloc[0] == pytest.approx(0.3)
    assert trace["lower"].iloc[0] == pytest.approx(0.3 - expected_radius)
    assert trace["upper"].iloc[0] == pytest.approx(0.3 + expected_radius)


def test_confidence_sequence_matches_hand_computation_at_the_worst_case_icc() -> None:
    """icc=1 (explicit stress-test override): variance is the full k^2 * s^2."""

    sizes = np.array([10.0])
    sums = np.array([3.0])
    trace = confidence_sequence_from_block_stats(
        sizes, sums, alpha=0.05, prior_variance=0.01, intraclass_correlation=1.0
    )

    denom = 1.0 + 0.01 * 100.0
    expected_log_e = -0.5 * np.log(denom) + 0.01 * 3.0**2 / (2.0 * denom)
    expected_radius = (1.0 / 10.0) * np.sqrt(
        (2.0 * denom / 0.01) * np.log((1.0 / 0.05) * np.sqrt(denom))
    )
    assert trace["log_e_value"].iloc[0] == pytest.approx(expected_log_e)
    assert trace["lower"].iloc[0] == pytest.approx(0.3 - expected_radius)
    assert trace["upper"].iloc[0] == pytest.approx(0.3 + expected_radius)


def test_confidence_sequence_excludes_zero_iff_e_value_clears_threshold() -> None:
    """The e-value and confidence-sequence readings of the same martingale must agree.

    They are algebraically dual (see the module docstring); this pins that
    identity so any future refactor of one formula that breaks the other is
    caught immediately, on random inputs rather than a single example.
    """

    rng = np.random.default_rng(20260818)
    for _ in range(500):
        n_blocks = int(rng.integers(1, 12))
        sizes = rng.integers(1, 60, size=n_blocks).astype(float)
        sums = rng.uniform(-1.0, 1.0, size=n_blocks) * sizes
        alpha = float(rng.uniform(0.01, 0.2))
        prior_variance = float(rng.uniform(1e-6, 1.0))
        trace = confidence_sequence_from_block_stats(
            sizes, sums, alpha=alpha, prior_variance=prior_variance
        )
        threshold = np.log(1.0 / alpha)
        np.testing.assert_array_equal(
            trace["excludes_zero"].to_numpy(), trace["log_e_value"].to_numpy() >= threshold
        )


def test_confidence_sequence_rejects_bad_inputs() -> None:
    sizes = np.array([10.0])
    sums = np.array([3.0])
    with pytest.raises(ValueError, match="alpha must be"):
        confidence_sequence_from_block_stats(sizes, sums, alpha=0.0, prior_variance=0.1)
    with pytest.raises(ValueError, match="prior_variance must be"):
        confidence_sequence_from_block_stats(sizes, sums, alpha=0.05, prior_variance=0.0)
    with pytest.raises(ValueError, match="At least one block"):
        confidence_sequence_from_block_stats(
            np.array([]), np.array([]), alpha=0.05, prior_variance=0.1
        )
    with pytest.raises(ValueError, match="same length"):
        confidence_sequence_from_block_stats(
            np.array([1.0, 2.0]), np.array([1.0]), alpha=0.05, prior_variance=0.1
        )
    with pytest.raises(ValueError, match="at least one game"):
        confidence_sequence_from_block_stats(
            np.array([0.0]), np.array([0.0]), alpha=0.05, prior_variance=0.1
        )


def test_default_prior_variance_derivation_and_guards() -> None:
    # Default (icc=0, proxy=1): rho = 1 / (target_games * k), the independent case.
    assert default_prior_variance(16.0, target_games=800) == pytest.approx(1.0 / (800 * 16.0))
    # An explicit override changes the reference point to
    # k * proxy * (1 + (k - 1) * icc), never the formula's shape.
    overridden = default_prior_variance(
        16.0, target_games=800, per_game_variance_proxy=0.6, intraclass_correlation=0.1
    )
    expected_reference = 800 * (16.0 * 0.6 * (1.0 + 15.0 * 0.1))
    assert overridden == pytest.approx(1.0 / expected_reference)
    # The worst-case override reduces to the k^2 form.
    worst_case = default_prior_variance(16.0, target_games=800, intraclass_correlation=1.0)
    assert worst_case == pytest.approx(1.0 / (800 * 16.0**2))
    with pytest.raises(ValueError, match="average_block_size"):
        default_prior_variance(0.0)
    with pytest.raises(ValueError, match="target_games"):
        default_prior_variance(16.0, target_games=0)


# ---------------------------------------------------------------------------
# DataFrame surface: shares the paired_feature_comparisons contract.
# ---------------------------------------------------------------------------


def test_paired_anytime_comparisons_matches_fixed_sample_point_estimate() -> None:
    """The final look's cumulative mean must equal the plain per-game mean.

    Both methods estimate the same quantity: at the last look, the anytime
    engine's cumulative sum/cumulative games collapses to the ordinary
    per-game mean regardless of how uneven the weekly blocks are, which
    ``paired_feature_comparisons``'s ``estimate`` column already reports.
    Any drift between the two definitions of "improvement" would break this.
    """

    rng = np.random.default_rng(11)
    predictions = _synthetic_predictions(rng)
    fixed = paired_feature_comparisons(
        predictions, baseline_feature_set="baseline", samples=50, block="week", seed=1
    )
    trace = paired_anytime_comparisons(
        predictions, baseline_feature_set="baseline", metric="accuracy_improvement", block="week"
    )
    summary = anytime_summary(trace)
    fixed_estimate = fixed.loc[fixed["metric"].eq("accuracy_improvement"), "estimate"].iloc[0]
    assert summary["final_estimate"].iloc[0] == pytest.approx(fixed_estimate)
    assert summary["games"].iloc[0] == int(fixed["paired_games"].iloc[0])


def test_paired_anytime_comparisons_detects_a_dominant_candidate() -> None:
    """Power sanity check at the project's actual operating configuration.

    At the fully conservative worst case (every game in a block moving in
    lockstep, ``intraclass_correlation=1.0``, per-game variance at
    Hoeffding's worst case) this method needs on the order of a million
    games to resolve even a large effect -- documented and quantified in
    ``docs/anytime_valid.md``. This test uses the project's standing
    configuration instead: ``per_game_variance_proxy=0.55`` (measured on
    real CFB ``market`` vs ``market_residual`` predictions) and the DEFAULT
    ``intraclass_correlation=0.0`` (independence -- a modelling decision,
    not an estimate; see the module docstring), so a dominant candidate is
    detectable within a realistic number of games.
    """

    rng = np.random.default_rng(5)
    predictions = _synthetic_predictions(
        rng, seasons=tuple(range(2015, 2021)), weeks_per_season=17, candidate_edge=0.45
    )
    trace = paired_anytime_comparisons(
        predictions,
        baseline_feature_set="baseline",
        metric="accuracy_improvement",
        block="week",
        target_games=int(predictions["game_id"].nunique()),
        per_game_variance_proxy=0.55,
    )
    summary = anytime_summary(trace)
    assert bool(summary["final_excludes_zero"].iloc[0])
    assert summary["first_excluding_zero_look"].iloc[0] is not None
    # Once excluded, a near-certain win should stay excluded through the end.
    tail = trace.sort_values("look").tail(5)
    assert tail["excludes_zero"].all()


def test_paired_anytime_comparisons_rejects_log_loss_metric() -> None:
    rng = np.random.default_rng(1)
    predictions = _synthetic_predictions(rng)
    with pytest.raises(ValueError, match="log_loss_improvement is unbounded"):
        paired_anytime_comparisons(
            predictions, baseline_feature_set="baseline", metric="log_loss_improvement"
        )


def test_paired_anytime_comparisons_validates_columns_and_baseline() -> None:
    rng = np.random.default_rng(1)
    predictions = _synthetic_predictions(rng)
    with pytest.raises(ValueError, match="missing paired columns"):
        paired_anytime_comparisons(
            predictions.drop(columns=["week"]), baseline_feature_set="baseline"
        )
    with pytest.raises(ValueError, match="Unknown paired baseline"):
        paired_anytime_comparisons(predictions, baseline_feature_set="nonexistent")
    with pytest.raises(ValueError, match="block must be"):
        paired_anytime_comparisons(predictions, baseline_feature_set="baseline", block="day")  # type: ignore[arg-type]


def test_paired_anytime_comparisons_detects_mismatched_pairing() -> None:
    rng = np.random.default_rng(1)
    predictions = _synthetic_predictions(rng)
    corrupted = predictions.copy()
    candidate_mask = corrupted["feature_set"].eq("candidate")
    corrupted.loc[candidate_mask, "home_cover"] = 1.0 - corrupted.loc[candidate_mask, "home_cover"]
    with pytest.raises(ValueError, match="Paired home_cover values differ"):
        paired_anytime_comparisons(corrupted, baseline_feature_set="baseline")


# ---------------------------------------------------------------------------
# Simulation building blocks used at scale by scripts/anytime_validate.py.
# ---------------------------------------------------------------------------


def test_simulate_block_sequence_is_bounded_and_unbiased_in_expectation() -> None:
    rng = np.random.default_rng(42)
    block_sizes = [15] * 40
    means = []
    for _ in range(200):
        blocks = simulate_block_sequence(rng, block_sizes, true_mean=0.013)
        all_values = np.concatenate(blocks)
        assert np.all(all_values >= -1.0) and np.all(all_values <= 1.0)
        means.append(float(all_values.mean()))
    assert np.mean(means) == pytest.approx(0.013, abs=0.01)


def test_block_bootstrap_ci_fast_brackets_the_weighted_mean() -> None:
    rng = np.random.default_rng(7)
    sizes = np.array([10.0, 20.0, 5.0, 15.0])
    sums = np.array([1.0, -2.0, 0.5, 3.0])
    weighted_mean = sums.sum() / sizes.sum()
    lower, upper = block_bootstrap_ci_fast(sizes, sums, samples=5_000, alpha=0.05, rng=rng)
    assert lower <= weighted_mean <= upper


def test_run_peeking_trial_under_a_true_null_rarely_excludes_zero() -> None:
    """A single deterministic-seed sanity check, not the full calibration study.

    The full false-alarm-RATE study over many universes lives in
    ``scripts/anytime_validate.py`` (heavy, and belongs in the validation
    deliverable rather than the fast test suite). This just confirms the
    trial machinery runs and returns internally consistent results.
    """

    rng = np.random.default_rng(20260818)
    block_sizes = [16] * 18
    result = run_peeking_trial(
        rng,
        block_sizes,
        true_mean=0.0,
        alpha=DEFAULT_ALPHA,
        prior_variance=default_prior_variance(16.0, target_games=800),
        fixed_sample_bootstrap_samples=200,
    )
    assert isinstance(result, PeekingTrialResult)
    if result.cs_excluded:
        assert result.cs_first_look is not None
        assert 1 <= result.cs_first_look <= len(block_sizes)
    else:
        assert result.cs_first_look is None
    if result.fixed_sample_excluded:
        assert result.fixed_sample_first_look is not None


def test_anytime_metrics_are_exactly_the_bounded_ones() -> None:
    assert set(ANYTIME_METRICS) == {"accuracy_improvement", "brier_improvement"}


# ---------------------------------------------------------------------------
# Measuring the intraclass correlation instead of assuming it.
# ---------------------------------------------------------------------------


def test_anova_intraclass_correlation_recovers_the_perfectly_correlated_case() -> None:
    """Every game in a block shares its block's value exactly: ICC must be 1."""

    rng = np.random.default_rng(3)
    block_means = rng.normal(size=12)
    blocks = [np.full(rng.integers(3, 9), mean) for mean in block_means]
    assert anova_intraclass_correlation(blocks) == pytest.approx(1.0, abs=1e-9)


def test_anova_intraclass_correlation_is_near_zero_for_independent_data() -> None:
    """Pure iid noise, arbitrarily grouped into blocks: ICC should land near zero."""

    rng = np.random.default_rng(4)
    blocks = [rng.normal(size=int(rng.integers(10, 60))) for _ in range(150)]
    icc = anova_intraclass_correlation(blocks)
    assert abs(icc) < 0.05


def test_anova_intraclass_correlation_guards() -> None:
    with pytest.raises(ValueError, match="At least two blocks"):
        anova_intraclass_correlation([np.array([1.0, 2.0])])
    with pytest.raises(ValueError, match="at least one observation"):
        anova_intraclass_correlation([np.array([]), np.array([1.0])])
    with pytest.raises(ValueError, match="within-block degrees of freedom"):
        anova_intraclass_correlation([np.array([1.0]), np.array([2.0])])


def test_bootstrap_intraclass_correlation_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(5)
    blocks = [rng.normal(size=int(rng.integers(10, 40))) for _ in range(80)]
    result = bootstrap_intraclass_correlation(blocks, samples=500, seed=1)
    assert result["lower"] <= result["estimate"] <= result["upper"]
    assert result["n_blocks"] == 80
    assert result["confidence"] == pytest.approx(0.95)


def test_default_icc_zero_holds_calibration_even_stress_tested() -> None:
    """Regression pin for the standing project decision (2026-08-18,
    docs/anytime_valid.md): ``intraclass_correlation`` defaults to 0.0
    (independence, a modelling decision -- disjoint teams, no shared outcome
    mechanism -- not an estimate). This pins that the default keeps
    calibration valid even when the TRUE simulated correlation is the full
    worst case (1.0) while the default (0.0) is what the confidence sequence
    is told to assume -- the specific scenario a wrong independence decision
    would fail under. ``scripts/anytime_validate.py`` runs the full-scale
    version against real CFB data; this is the fast regression guard.
    """

    rng = np.random.default_rng(6)
    block_sizes = [16] * 18
    prior_variance = default_prior_variance(16.0, target_games=285, per_game_variance_proxy=0.55)
    false_alarms = 0
    trials = 400
    for _ in range(trials):
        result = run_peeking_trial(
            rng,
            block_sizes,
            true_mean=0.0,
            prior_variance=prior_variance,
            simulated_total_variance=0.55,
            simulated_intraclass_correlation=1.0,  # deliberately worse than assumed
            assumed_per_game_variance_proxy=0.55,
            # assumed_intraclass_correlation left at its default, 0.0.
            check_fixed_sample=False,
        )
        false_alarms += int(result.cs_excluded)
    assert false_alarms / trials <= 0.10  # nominal alpha is 0.05; generous margin for n=400
