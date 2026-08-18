"""Empirical Bayes decision rule for forced-pick candidates.

FLAWED -- DO NOT USE FOR DECISIONS. Marked by the project owner 2026-08-18;
the shrinkage is wrong and must be redone. See the banner at the top of
``docs/decision_rule.md`` for the three defects: the prior's reference class
counts correlated variants of one idea as independent draws, candidates are
treated as exchangeable when some carry independent evidence of being real,
and the resulting negative prior mean is then applied to everything. No
posterior produced here may be cited or acted on, and nothing here
re-classifies a registry entry. Not wired into any production path.

``AGENTS.md`` establishes ("A promotion bar is not a decision bar") that this
project's pool submits 285 forced picks a season with no abstain option.
Declining to use a candidate signal is not caution -- it is an active bet
that the signal is worth exactly zero, and that bet must clear the same
expected-value bar as any other. A p-value or a predeclared
``probability_positive`` threshold answers a *publication* question ("can we
claim this is real"), not the *decision* question this module answers ("does
using it beat not using it, in expectation").

The mechanism is standard empirical Bayes / random-effects shrinkage, reused
for a purpose ``weak_signals.pooled_effect`` does not cover. That function
pools several measurements of the SAME signal. This module instead treats
this project's entire history of measured effects -- 282 recorded rows
across 27 experiment artifacts, deduplicated to 210 independent point
estimates (see ``scripts/decision_load_measurements.py``) -- as a *sample
from the distribution of true effect sizes this domain produces*. Fitting a
prior to that sample, then shrinking any single new estimate toward it, is
exactly the empirical-Bayes / James-Stein construction: it borrows strength
from "how big do effects in this project usually turn out to be" to correct
for the fact that any one small-sample estimate is a noisy draw.

Two pieces of arithmetic, both pure functions with no side effects:

- ``fit_empirical_prior`` -- fit ``theta_i ~ N(mu, tau^2)`` by the
  Paule-Mandel method-of-moments iteration (a refinement of the
  DerSimonian-Laird estimator already used in ``weak_signals.pooled_effect``;
  see the module docstring there). Paule-Mandel iterates the moment equation
  to convergence rather than solving it in one step, which matters here
  because the 210 measurements have wildly heterogeneous standard errors (248
  to 11,780 paired games) -- DerSimonian-Laird is known to underestimate
  between-study variance in exactly that regime, and it is checked against
  Paule-Mandel in ``docs/decision_rule.md``.
- ``evaluate_candidate`` -- combine that prior with one new estimate via the
  normal-normal conjugate update, and report everything the decision needs:
  the posterior mean, ``probability_positive`` (never "contains zero" --
  binding per ``AGENTS.md``), and the expected cost of being wrong in each
  direction.

``model_average`` implements the model-averaging half of the task: weighting
every candidate by its posterior conviction instead of gating a subset in or
out by threshold. Nothing here spends a rotation window, scores a model, or
writes the registry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_Z95 = 1.959963984540054  # two-sided 95% normal multiplier, kept for callers
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


class DecisionRuleError(ValueError):
    """Raised for invalid inputs to the prior fit or the decision arithmetic."""


# ---------------------------------------------------------------------------
# Normal-distribution primitives (no scipy dependency; math.erf is stdlib)
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """Standard normal CDF, exact via the error function."""

    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    """Standard normal density."""

    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via bisection.

    Bisection rather than a rational approximation: this module runs on at
    most a few hundred values, accuracy matters more than speed, and it keeps
    the dependency footprint at "stdlib only."
    """

    if not 0.0 < p < 1.0:
        raise DecisionRuleError(f"norm_ppf requires 0 < p < 1, got {p!r}")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def se_from_interval(lower: float, upper: float, confidence: float = 0.95) -> float:
    """Recover a standard error from a symmetric normal confidence interval.

    Every artifact in this project reports intervals this way (bootstrap
    percentile intervals are close enough to symmetric at these sample sizes
    to invert safely -- the project's own ``resolved_standard_error`` in
    ``weak_signals.py`` makes the identical assumption for the 95% case).
    """

    if upper < lower:
        raise DecisionRuleError(f"Inverted interval: lower={lower} > upper={upper}")
    z = norm_ppf(0.5 + confidence / 2.0)
    return (upper - lower) / (2.0 * z)


def se_from_probability_positive(effect: float, probability_positive: float) -> float:
    """Recover a standard error from an effect and its probability_positive.

    Used when a registry entry carries ``probability_positive`` but no
    interval (``ecdf_smoothing_accuracy``, ``ridge_alpha_global``). Assumes
    the estimate is approximately normal around the effect, which is the same
    assumption the bootstrap procedures that produced these numbers already
    make.
    """

    if not 0.0 < probability_positive < 1.0:
        raise DecisionRuleError(
            f"probability_positive must be in (0, 1), got {probability_positive!r}"
        )
    z = norm_ppf(probability_positive)
    if abs(z) < 1e-9:
        raise DecisionRuleError("probability_positive == 0.5 implies zero effect; no SE to recover")
    return abs(effect / z)


def combine_standard_errors(*standard_errors: float) -> float:
    """Combine independent noise sources in quadrature: sqrt(sum of squares).

    Used to add an undisclosed noise floor (e.g. the calibration-step
    instability documented in ``docs/purged_cv.md``) on top of a
    measurement's own reported sampling SE, when that SE was derived from a
    procedure (block bootstrap) that does not capture the extra source.
    """

    if not standard_errors:
        raise DecisionRuleError("combine_standard_errors needs at least one input")
    if any(se < 0.0 for se in standard_errors):
        raise DecisionRuleError("standard errors must be >= 0")
    return math.sqrt(sum(se * se for se in standard_errors))


def _expected_positive_part(mean: float, sd: float) -> float:
    """E[max(0, X)] for X ~ N(mean, sd^2). Standard partial-moment formula."""

    if sd <= 0.0:
        return max(0.0, mean)
    z = mean / sd
    return sd * norm_pdf(z) + mean * norm_cdf(z)


# ---------------------------------------------------------------------------
# Measurements and the prior fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectMeasurement:
    """One historical or candidate effect measurement, accuracy-points scale.

    Positive always favours the candidate over its baseline (the same sign
    convention ``weak_signals.py`` uses). Units must be ``accuracy_points``
    (percentage points of forced-pick accuracy) before construction --
    convert upstream, the way ``fourth_down_aggressiveness``'s ``ats_points``
    figure is converted in ``scripts/decision_apply.py`` using the project's
    own documented exchange rate.
    """

    label: str
    estimate: float
    standard_error: float
    n_games: int | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.estimate):
            raise DecisionRuleError(f"{self.label!r}: non-finite estimate")
        if not (math.isfinite(self.standard_error) and self.standard_error > 0.0):
            raise DecisionRuleError(
                f"{self.label!r}: standard_error must be finite and positive, "
                f"got {self.standard_error!r}"
            )


@dataclass(frozen=True)
class EmpiricalPrior:
    """A fitted N(mean, variance) prior over true effect sizes in this domain."""

    mean: float
    variance: float
    n_measurements: int
    method: str = "paule_mandel"

    def __post_init__(self) -> None:
        if self.variance < 0.0:
            raise DecisionRuleError(f"prior variance must be >= 0, got {self.variance!r}")
        if self.n_measurements < 1:
            raise DecisionRuleError("prior must be fit from at least one measurement")

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance)

    def shrinkage_factor(self, standard_error: float) -> float:
        """Fraction of posterior weight the new DATA gets (0..1).

        ``1 - this`` is the weight left on the prior mean. A measurement with
        ``standard_error == 0`` would get weight 1 (no shrinkage); a
        measurement infinitely noisy gets weight 0 (posterior = prior).
        """

        if standard_error < 0.0:
            raise DecisionRuleError("standard_error must be >= 0")
        se2 = standard_error * standard_error
        if self.variance == 0.0 and se2 == 0.0:
            return 0.5
        return self.variance / (self.variance + se2)


def fit_empirical_prior(
    measurements: Sequence[EffectMeasurement],
    *,
    method: str = "paule_mandel",
    max_iter: int = 200,
    max_tau2: float = 100.0,
) -> EmpiricalPrior:
    """Fit ``theta_i ~ N(mu, tau^2)`` from a sample of independent measurements.

    ``paule_mandel`` (default) iterates the method-of-moments equation
    ``Q(tau^2) == K - 1`` to convergence by bisection, where
    ``Q(tau^2) = sum_i w_i(tau^2) * (y_i - mu(tau^2))^2`` and
    ``w_i(tau^2) = 1 / (se_i^2 + tau^2)``. ``dersimonian_laird`` solves the
    same equation in one Newton-style step from ``tau^2 = 0`` (the estimator
    ``weak_signals.pooled_effect`` uses for pooling a handful of same-signal
    looks). On this project's 210-measurement sample the two disagree
    materially -- DL gives ``tau ~= 0.20``, Paule-Mandel gives
    ``tau ~= 0.55`` -- and the standardized residuals under Paule-Mandel have
    variance 0.997 (well calibrated) against 1.59 under DL (badly
    under-dispersed, i.e. DL's tau is too small for measurements this
    heterogeneous in precision). See ``docs/decision_rule.md`` section 1 for
    the check. Paule-Mandel is therefore the default; ``dersimonian_laird``
    remains available for comparison.
    """

    if len(measurements) < 2:
        raise DecisionRuleError("Need at least two measurements to fit a prior")
    if method not in ("paule_mandel", "dersimonian_laird"):
        raise DecisionRuleError(f"Unknown method {method!r}")

    y = [m.estimate for m in measurements]
    v = [m.standard_error * m.standard_error for m in measurements]
    k = len(measurements)

    def weighted_mean_and_q(tau2: float) -> tuple[float, float]:
        weights = [1.0 / (vi + tau2) for vi in v]
        total_w = sum(weights)
        mu = sum(w * yi for w, yi in zip(weights, y, strict=True)) / total_w
        q = sum(w * (yi - mu) ** 2 for w, yi in zip(weights, y, strict=True))
        return mu, q

    target = float(k - 1)

    if method == "dersimonian_laird":
        weights0 = [1.0 / vi for vi in v]
        total_w0 = sum(weights0)
        mu_fixed = sum(w * yi for w, yi in zip(weights0, y, strict=True)) / total_w0
        q0 = sum(w * (yi - mu_fixed) ** 2 for w, yi in zip(weights0, y, strict=True))
        c = total_w0 - sum(w * w for w in weights0) / total_w0
        tau2 = max(0.0, (q0 - target) / c) if c > 0.0 else 0.0
        mu, _ = weighted_mean_and_q(tau2)
        return EmpiricalPrior(mean=mu, variance=tau2, n_measurements=k, method=method)

    # Paule-Mandel: Q(tau2) is monotone non-increasing in tau2, so bisect for
    # the root of Q(tau2) - target. At tau2 = 0, Q is the ordinary
    # fixed-effect heterogeneity statistic; it can lie below target - 1 (no
    # detectable heterogeneity), in which case the root is 0.
    lo, hi = 0.0, max_tau2
    _, q_lo = weighted_mean_and_q(lo)
    if q_lo <= target:
        tau2 = 0.0
    else:
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            _, q_mid = weighted_mean_and_q(mid)
            if q_mid > target:
                lo = mid
            else:
                hi = mid
        tau2 = (lo + hi) / 2.0
    mu, _ = weighted_mean_and_q(tau2)
    return EmpiricalPrior(mean=mu, variance=tau2, n_measurements=k, method=method)


def leave_one_out_log_likelihood(
    measurements: Sequence[EffectMeasurement],
    *,
    method: str = "paule_mandel",
) -> float:
    """Total leave-one-out predictive log-likelihood of a fitted prior.

    For each measurement, fits the prior on every OTHER measurement in the
    sequence, then scores the held-out point's marginal likelihood under
    ``y_i ~ N(mu, tau^2 + se_i^2)`` -- the standard held-out predictive check
    for whether treating a set of measurements as exchangeable (one shared
    prior) is supported by the data, as opposed to merely fitting it well
    in-sample.

    This is the tool for the question "does conditioning on a covariate
    (stratifying into two groups, each with its own prior) actually improve
    prediction, or does it just overfit the 210-measurement sample". Compare
    the SUM of this quantity computed separately within each stratum against
    this quantity computed once over the pooled set: a stratified model with
    genuinely different sub-populations will score higher out-of-sample; one
    that only differs by sampling noise will usually score lower, because
    each stratum's own prior is fit on fewer points and pays a variance cost
    for a distinction that is not real. See ``docs/decision_rule.md`` section
    1.5 for the applied check.
    """

    if len(measurements) < 3:
        raise DecisionRuleError("Need at least three measurements for leave-one-out scoring")

    total = 0.0
    for index in range(len(measurements)):
        held_out = measurements[index]
        rest = list(measurements[:index]) + list(measurements[index + 1 :])
        prior = fit_empirical_prior(rest, method=method)
        predictive_variance = prior.variance + held_out.standard_error * held_out.standard_error
        z = (held_out.estimate - prior.mean) / math.sqrt(predictive_variance)
        total += -0.5 * math.log(2.0 * math.pi * predictive_variance) - 0.5 * z * z
    return total


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorResult:
    """Everything the forced-pick decision needs for one candidate."""

    label: str
    observed_estimate: float
    observed_se: float
    prior_mean: float
    prior_sd: float
    posterior_mean: float
    posterior_sd: float
    probability_positive: float
    shrinkage_factor: float
    expected_cost_if_use_is_wrong: float
    expected_cost_if_skip_is_wrong: float
    verdict: str

    @property
    def expected_gain(self) -> float:
        """Alias for posterior_mean: the EV of using this candidate."""

        return self.posterior_mean


def evaluate_candidate(measurement: EffectMeasurement, prior: EmpiricalPrior) -> PosteriorResult:
    """Shrink one measurement toward the prior and render the forced-pick verdict.

    Under forced picks the rule is not a significance threshold: **use the
    candidate iff its posterior mean is positive.** ``probability_positive``
    and the two expected-cost figures are reported for transparency, not as
    gates -- a candidate at posterior mean +0.02 with probability_positive
    0.54 is still, in expectation, better than not using it. Only a posterior
    mean of exactly zero makes the two choices equivalent.

    The two expected-cost figures are the standard partial first moments of
    the posterior normal: ``expected_cost_if_use_is_wrong`` is
    ``E[max(0, -theta)]`` (what "use" costs you, weighted by how likely and
    how large a negative true effect is); ``expected_cost_if_skip_is_wrong``
    is ``E[max(0, theta)]`` (the symmetric quantity for "skip"). Their
    difference is exactly the posterior mean, which is the sanity check
    ``tests/test_decision_rule.py`` pins.
    """

    if prior.variance == 0.0:
        post_var = 0.0
        post_mean = prior.mean
    else:
        se2 = measurement.standard_error * measurement.standard_error
        precision = 1.0 / prior.variance + 1.0 / se2
        post_var = 1.0 / precision
        post_mean = post_var * (prior.mean / prior.variance + measurement.estimate / se2)
    post_sd = math.sqrt(post_var)

    probability_positive = 1.0 if post_mean > 0.0 else 0.0
    if post_sd > 0.0:
        probability_positive = norm_cdf(post_mean / post_sd)

    cost_use_wrong = _expected_positive_part(-post_mean, post_sd)
    cost_skip_wrong = _expected_positive_part(post_mean, post_sd)

    return PosteriorResult(
        label=measurement.label,
        observed_estimate=measurement.estimate,
        observed_se=measurement.standard_error,
        prior_mean=prior.mean,
        prior_sd=prior.sd,
        posterior_mean=post_mean,
        posterior_sd=post_sd,
        probability_positive=probability_positive,
        shrinkage_factor=prior.shrinkage_factor(measurement.standard_error),
        expected_cost_if_use_is_wrong=cost_use_wrong,
        expected_cost_if_skip_is_wrong=cost_skip_wrong,
        verdict="use" if post_mean > 0.0 else "dont_use",
    )


# ---------------------------------------------------------------------------
# Model averaging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelAverageResult:
    """A conviction-weighted blend across candidates, replacing a binary gate."""

    mode: str
    weight_by: str
    weights: dict[str, float]
    combined_expected_gain: float
    included: tuple[str, ...]
    note: str


def model_average(
    results: Sequence[PosteriorResult],
    *,
    mode: str = "stack",
    weight_by: str = "probability_positive",
) -> ModelAverageResult:
    """Weight every candidate by posterior conviction instead of gating on a threshold.

    Two modes, because this project's candidates come in both shapes:

    - ``mode="stack"``: candidates are independent, additive features (e.g.
      ``hc_year_one_fade`` and ``penalty_discipline`` can both be true at
      once). Each contributes ``weight_k * posterior_mean_k`` to the total,
      where ``weight_k`` is its own conviction (``probability_positive`` by
      default, not renormalized across candidates) -- a continuous relaxation
      of "include if it clears a bar," which lets a 55%-confident candidate
      contribute a little instead of nothing.
    - ``mode="blend"``: candidates are mutually exclusive alternatives (e.g.
      the ``residual_location_*`` family -- only one recency/shrink setting
      can run in production at a time). Weights are normalized to sum to 1 and
      ``combined_expected_gain`` is the expected value of running a
      probability-weighted lottery over the alternatives rather than picking
      a single winner by threshold.

    ``weight_by="probability_positive"`` is the default; ``"posterior_mean"``
    (clipped at zero) is available for a magnitude-weighted variant.
    """

    if mode not in ("stack", "blend"):
        raise DecisionRuleError(f"Unknown mode {mode!r}")
    if weight_by not in ("probability_positive", "posterior_mean"):
        raise DecisionRuleError(f"Unknown weight_by {weight_by!r}")
    if not results:
        return ModelAverageResult(
            mode=mode,
            weight_by=weight_by,
            weights={},
            combined_expected_gain=0.0,
            included=(),
            note="no candidates supplied",
        )

    if weight_by == "probability_positive":
        raw = {r.label: r.probability_positive for r in results}
    else:
        raw = {r.label: max(0.0, r.posterior_mean) for r in results}

    if mode == "blend":
        total = sum(raw.values())
        weights = (
            {label: v / total for label, v in raw.items()}
            if total > 0.0
            else {label: 1.0 / len(raw) for label in raw}
        )
        combined = sum(weights[r.label] * r.posterior_mean for r in results)
        note = (
            "Mutually-exclusive blend: weights sum to 1; combined_expected_gain is "
            "the EV of a probability-weighted lottery over the alternatives, not a sum."
        )
    else:
        weights = dict(raw)
        combined = sum(weights[r.label] * r.posterior_mean for r in results)
        note = (
            "Additive stack: weights are each candidate's own conviction, not "
            "renormalized. combined_expected_gain assumes independence across "
            "candidates -- check for shared source data before trusting it as a "
            "single number."
        )

    included = tuple(r.label for r in results if r.posterior_mean > 0.0)
    return ModelAverageResult(
        mode=mode,
        weight_by=weight_by,
        weights=weights,
        combined_expected_gain=combined,
        included=included,
        note=note,
    )
