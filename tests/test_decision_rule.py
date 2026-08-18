from __future__ import annotations

import math
import random

import pytest

from nfl_ats.decision_rule import (
    DecisionRuleError,
    EffectMeasurement,
    EmpiricalPrior,
    combine_standard_errors,
    evaluate_candidate,
    fit_empirical_prior,
    model_average,
    norm_cdf,
    norm_ppf,
    se_from_interval,
    se_from_probability_positive,
)


def test_norm_cdf_and_ppf_round_trip() -> None:
    for p in (0.01, 0.1, 0.5, 0.758, 0.899, 0.99):
        assert norm_ppf(p) == pytest.approx(norm_ppf(p))
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-6)
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)


def test_norm_ppf_rejects_out_of_range() -> None:
    with pytest.raises(DecisionRuleError):
        norm_ppf(0.0)
    with pytest.raises(DecisionRuleError):
        norm_ppf(1.0)


def test_se_from_interval_matches_known_95_percent_width() -> None:
    # A 95% CI of [-1.96, 1.96] around 0 is exactly +/- 1 standard error.
    se = se_from_interval(-1.959963984540054, 1.959963984540054)
    assert se == pytest.approx(1.0, abs=1e-9)


def test_se_from_interval_rejects_inverted_bounds() -> None:
    with pytest.raises(DecisionRuleError):
        se_from_interval(1.0, -1.0)


def test_se_from_probability_positive_matches_interval_derived_se() -> None:
    # A measurement at effect=1.0 with se=1.0 has probability_positive = Phi(1.0).
    effect, se = 1.0, 1.0
    pp = norm_cdf(effect / se)
    recovered = se_from_probability_positive(effect, pp)
    assert recovered == pytest.approx(se, rel=1e-6)


def test_se_from_probability_positive_rejects_half() -> None:
    with pytest.raises(DecisionRuleError):
        se_from_probability_positive(1.0, 0.5)


def test_combine_standard_errors_is_quadrature() -> None:
    assert combine_standard_errors(3.0, 4.0) == pytest.approx(5.0)
    assert combine_standard_errors(1.0) == pytest.approx(1.0)
    with pytest.raises(DecisionRuleError):
        combine_standard_errors(-1.0)
    with pytest.raises(DecisionRuleError):
        combine_standard_errors()


def test_effect_measurement_rejects_bad_inputs() -> None:
    with pytest.raises(DecisionRuleError):
        EffectMeasurement(label="x", estimate=1.0, standard_error=0.0)
    with pytest.raises(DecisionRuleError):
        EffectMeasurement(label="x", estimate=1.0, standard_error=-1.0)
    with pytest.raises(DecisionRuleError):
        EffectMeasurement(label="x", estimate=float("nan"), standard_error=1.0)


def test_fit_empirical_prior_needs_at_least_two_measurements() -> None:
    with pytest.raises(DecisionRuleError):
        fit_empirical_prior([EffectMeasurement(label="a", estimate=1.0, standard_error=1.0)])


def test_fit_empirical_prior_recovers_a_known_normal_prior() -> None:
    """Simulate measurements from a KNOWN N(mu, tau^2) prior and confirm the
    Paule-Mandel fit recovers mu and tau to within sampling noise on a large
    sample -- the core correctness check for the empirical Bayes machinery.
    """

    rng = random.Random(20260818)
    true_mu, true_tau = 0.4, 0.8
    measurements = []
    for i in range(400):
        se = rng.uniform(0.2, 1.5)
        theta_i = rng.gauss(true_mu, true_tau)
        y_i = rng.gauss(theta_i, se)
        measurements.append(EffectMeasurement(label=f"sim_{i}", estimate=y_i, standard_error=se))

    prior = fit_empirical_prior(measurements)
    assert prior.mean == pytest.approx(true_mu, abs=0.15)
    assert prior.sd == pytest.approx(true_tau, abs=0.15)
    assert prior.n_measurements == 400


def test_fit_empirical_prior_gives_zero_tau_when_homogeneous() -> None:
    """When every measurement estimates the exact same quantity (no real
    between-study heterogeneity), Q at tau^2=0 should not exceed K-1 by much
    and the fitted tau should land near zero, not be forced positive.
    """

    rng = random.Random(1)
    true_effect = 0.5
    measurements = [
        EffectMeasurement(label=f"m{i}", estimate=rng.gauss(true_effect, 1.0), standard_error=1.0)
        for i in range(300)
    ]
    prior = fit_empirical_prior(measurements)
    assert prior.sd < 0.3  # should be small; exact value depends on the draw
    assert prior.mean == pytest.approx(true_effect, abs=0.2)


def test_dersimonian_laird_and_paule_mandel_agree_when_var_is_near_zero() -> None:
    measurements = [
        EffectMeasurement(label="a", estimate=1.0, standard_error=0.5),
        EffectMeasurement(label="b", estimate=1.0, standard_error=0.5),
        EffectMeasurement(label="c", estimate=1.0, standard_error=0.5),
    ]
    pm = fit_empirical_prior(measurements, method="paule_mandel")
    dl = fit_empirical_prior(measurements, method="dersimonian_laird")
    assert pm.mean == pytest.approx(1.0, abs=1e-6)
    assert dl.mean == pytest.approx(1.0, abs=1e-6)
    assert pm.variance == pytest.approx(0.0, abs=1e-6)
    assert dl.variance == pytest.approx(0.0, abs=1e-6)


def test_shrinkage_factor_bounds_and_monotonicity() -> None:
    prior = EmpiricalPrior(mean=0.0, variance=1.0, n_measurements=10)
    assert prior.shrinkage_factor(0.0) == pytest.approx(1.0)
    tiny = prior.shrinkage_factor(0.01)
    huge = prior.shrinkage_factor(100.0)
    assert 0.0 < huge < tiny < 1.0


def test_evaluate_candidate_shrinks_toward_the_prior_mean() -> None:
    prior = EmpiricalPrior(mean=0.0, variance=0.25, n_measurements=100)
    noisy_measurement = EffectMeasurement(label="x", estimate=2.0, standard_error=2.0)
    result = evaluate_candidate(noisy_measurement, prior)
    # Posterior mean must lie strictly between the prior mean and the raw estimate.
    assert 0.0 < result.posterior_mean < 2.0
    # Posterior sd must be smaller than either input sd.
    assert result.posterior_sd < prior.sd
    assert result.posterior_sd < noisy_measurement.standard_error
    assert result.verdict == "use"


def test_evaluate_candidate_verdict_flips_exactly_at_zero() -> None:
    prior = EmpiricalPrior(mean=0.0, variance=1.0, n_measurements=50)
    positive = evaluate_candidate(
        EffectMeasurement(label="p", estimate=0.5, standard_error=1.0), prior
    )
    negative = evaluate_candidate(
        EffectMeasurement(label="n", estimate=-0.5, standard_error=1.0), prior
    )
    assert positive.verdict == "use"
    assert negative.verdict == "dont_use"
    assert positive.probability_positive > 0.5
    assert negative.probability_positive < 0.5


def test_evaluate_candidate_zero_variance_prior_ignores_the_data() -> None:
    prior = EmpiricalPrior(mean=0.3, variance=0.0, n_measurements=50)
    result = evaluate_candidate(
        EffectMeasurement(label="x", estimate=99.0, standard_error=0.01), prior
    )
    assert result.posterior_mean == pytest.approx(0.3)
    assert result.posterior_sd == pytest.approx(0.0)


def test_expected_costs_are_nonnegative_and_their_difference_is_the_posterior_mean() -> None:
    prior = EmpiricalPrior(mean=-0.1, variance=0.3, n_measurements=200)
    for estimate, se in [(1.5, 1.0), (-1.5, 1.0), (0.01, 0.5), (-0.01, 0.5)]:
        result = evaluate_candidate(
            EffectMeasurement(label="x", estimate=estimate, standard_error=se), prior
        )
        assert result.expected_cost_if_use_is_wrong >= 0.0
        assert result.expected_cost_if_skip_is_wrong >= 0.0
        diff = result.expected_cost_if_skip_is_wrong - result.expected_cost_if_use_is_wrong
        assert diff == pytest.approx(result.posterior_mean, abs=1e-9)


def test_mod07_regression_event_is_predicted_within_the_documented_range() -> None:
    """Pinned regression test using the real MOD-07 numbers from
    docs/mod07_stack.md and docs/pool_edge_plan.md: +1.97 pts on 456 games
    (week-blocked 95% CI [-1.10, 5.00]) delivered +0.33 pts on 1,537 games.
    A prior fit broadly in this project's empirically observed range
    (tau roughly 0.4-0.7 accuracy points, mean near zero) should shrink the
    raw estimate to somewhere between 0 and the raw value, and much closer to
    the delivered number than the raw number was.
    """

    prior = EmpiricalPrior(mean=-0.13, variance=0.55**2, n_measurements=210)
    raw = EffectMeasurement(
        label="mod07_raw", estimate=1.97, standard_error=se_from_interval(-1.10, 5.00)
    )
    result = evaluate_candidate(raw, prior)
    delivered = 0.33
    assert 0.0 < result.posterior_mean < 1.97
    assert abs(result.posterior_mean - delivered) < abs(raw.estimate - delivered)


def test_model_average_stack_sums_weighted_contributions() -> None:
    prior = EmpiricalPrior(mean=0.0, variance=0.5, n_measurements=100)
    a = evaluate_candidate(EffectMeasurement(label="a", estimate=1.0, standard_error=0.5), prior)
    b = evaluate_candidate(EffectMeasurement(label="b", estimate=-1.0, standard_error=0.5), prior)
    result = model_average([a, b], mode="stack", weight_by="probability_positive")
    expected = a.probability_positive * a.posterior_mean + b.probability_positive * b.posterior_mean
    assert result.combined_expected_gain == pytest.approx(expected)
    assert result.included == ("a",)
    assert set(result.weights) == {"a", "b"}


def test_model_average_blend_weights_sum_to_one() -> None:
    prior = EmpiricalPrior(mean=0.0, variance=0.5, n_measurements=100)
    candidates = [
        evaluate_candidate(EffectMeasurement(label=f"c{i}", estimate=v, standard_error=0.4), prior)
        for i, v in enumerate([0.5, -0.2, 0.1, -0.4])
    ]
    result = model_average(candidates, mode="blend", weight_by="probability_positive")
    assert sum(result.weights.values()) == pytest.approx(1.0)
    # Combined value must be a weighted average, hence bounded by the inputs.
    assert min(c.posterior_mean for c in candidates) <= result.combined_expected_gain
    assert result.combined_expected_gain <= max(c.posterior_mean for c in candidates)


def test_model_average_rejects_unknown_mode_or_weighting() -> None:
    prior = EmpiricalPrior(mean=0.0, variance=0.5, n_measurements=100)
    result = evaluate_candidate(
        EffectMeasurement(label="a", estimate=1.0, standard_error=0.5), prior
    )
    with pytest.raises(DecisionRuleError):
        model_average([result], mode="nonsense")
    with pytest.raises(DecisionRuleError):
        model_average([result], weight_by="nonsense")


def test_model_average_handles_empty_input() -> None:
    result = model_average([], mode="stack")
    assert result.combined_expected_gain == 0.0
    assert result.included == ()
    assert result.weights == {}


def test_evaluate_candidate_raises_on_invalid_prior_variance() -> None:
    with pytest.raises(DecisionRuleError):
        EmpiricalPrior(mean=0.0, variance=-1.0, n_measurements=10)


def test_fit_empirical_prior_rejects_unknown_method() -> None:
    measurements = [
        EffectMeasurement(label="a", estimate=1.0, standard_error=1.0),
        EffectMeasurement(label="b", estimate=1.0, standard_error=1.0),
    ]
    with pytest.raises(DecisionRuleError):
        fit_empirical_prior(measurements, method="not_a_method")


def test_norm_cdf_matches_math_erf_definition() -> None:
    for x in (-3.0, -1.0, 0.0, 1.0, 3.0):
        expected = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        assert norm_cdf(x) == pytest.approx(expected)
