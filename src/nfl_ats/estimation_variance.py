"""Refit-aware uncertainty: the training-resample source no interval has ever carried.

Research item: ``docs/estimation_variance.md``. Every interval this project has
reported (``experiments.paired_feature_comparisons``,
``outcomes.outcome_bootstrap_intervals``, ``cfb_benchmark.cfb_benchmark_uncertainty``)
block-bootstraps GAMES around ONE already-fitted model. That answers "did this
fitted model beat that fitted model on these games", conditional on whichever
training rows the walk-forward happened to see. It has never answered the
question a promotion decision actually needs: "would a model fit THIS WAY beat
one fit THAT WAY in general" -- which requires resampling the TRAINING rows and
refitting, not just resampling the evaluation games.

This module adds that missing source as an opt-in layer, composed on top of the
existing model/prediction machinery rather than inside it:

- ``refit_predicted_values`` / ``point_predicted_values`` -- refit
  ``margin.make_margin_estimator("ridge", ...)`` on bootstrap resamples of the
  training rows (or once, unresampled, for the value every production run
  reports today) and predict on a fixed test frame. Reuses the project's own
  ridge/imputer/scaler pipeline verbatim; nothing about the estimator changes.
- ``home_cover_probability_from_center`` -- reproduces
  ``margin._smoothed_probability`` vectorized across many games sharing one
  out-of-time residual sample, so refit centers can be turned into cover
  probabilities without touching ``margin.py``.
- ``naive_block_bootstrap_interval`` -- the CURRENTLY REPORTED style: resamples
  games, never refits. A drop-in numpy analogue of
  ``experiments.paired_feature_comparisons``'s accuracy-metric arm, generalized
  off the pandas feature-set pivot so it can score any two fixed probability
  arrays (real predictions or a synthetic validation harness).
- ``refit_aware_paired_interval`` -- the FIRST honest interval (2026-08-18,
  ``docs/estimation_variance.md`` secs 1-3): combines an independent
  training-refit draw with an independent game-block resample in each outer
  iteration. Kept unchanged so every number that document reports still
  reproduces, but superseded by ``refit_aware_interval`` below, which fixes its
  two declared weaknesses (quantiles read off only ``n_boot`` draws, and each
  draw re-centred on a bootstrap refit rather than the reported point fit).
- ``paired_refit_predicted_values`` / ``refit_common_variance`` /
  ``refit_variance_decomposition`` / ``refit_aware_interval`` -- the durable
  honest path (``docs/estimation_variance.md`` Part II). Both arms refit on the
  SAME resampled training rows and scored on the SAME games (pairing is
  structural, not a seed coincidence); the delta's variance split into a
  training effect, a game effect and their INTERACTION; and the interval
  widened by the DERIVED factor ``sqrt(1 + Var(a)/Var_conditional)``. The
  interaction is the part that matters: it belongs to the game bootstrap
  already, and adding it a second time is why both 2026-08-18 honest intervals
  OVER-cover and why the "17-58% too narrow" band was overstated. On real CFB
  the honest factor is 1.003 (95% upper bound 1.099), not 1.33.
  ``inflate_recorded_interval`` applies a widening to an already-recorded
  registry row without re-running anything.
- ``MIN_BLOCKS_FOR_INTERVAL`` / ``guard_block_count`` /
  ``distinct_block_resamples`` -- the D4 degeneracy guard. Measured, not
  asserted: percentile block-bootstrap coverage of a known truth is 0.00 at 1
  block, ~0.48 at 2, ~0.79 at 4 and only reaches 0.90 at 10 blocks, so the
  "~4-5 block" floor in ``docs/anytime_valid.md`` sec 6 is far too generous.
- ``bagged_values`` / ``shrink_predicted_margin`` -- two variance-REDUCTION
  levers on the fit itself: averaging over bootstrap refits (bagging) and
  uniformly shrinking the predicted center toward the market line, mirroring
  ``residual_location.py``'s shrinkage of the residual READER but applied to
  the prediction CENTRE instead.
- ``picks_differ_fraction`` / ``mde80`` -- the ``f`` lever:
  ``MDE80 = 280 * sqrt(f / n)`` where ``f`` is the fraction of games on which
  two arms make different picks. ``gate_by_disagreement`` tests whether a
  candidate can be made more provable by only trusting it where it disagrees
  with the baseline enough to matter.

Nothing here is wired into ``margin.py``, ``experiments.py``,
``cfb_benchmark.py``, or the active model. Every function is additive:
composed over the existing public estimator/dataclass interfaces so no
production pick can move by importing this module.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.margin import make_margin_estimator

#: MDE80 = coefficient * sqrt(f / n), points of forced-pick accuracy. Derived
#: in the evaluator-power audit this module acts on; declared once so every
#: caller reports the same constant instead of retyping it.
DEFAULT_MDE80_COEFFICIENT = 280.0

#: Below this many bootstrap blocks the percentile block-bootstrap interval is
#: not a 95% interval and must not be reported as one. MEASURED, not adopted
#: from prose: ``scripts/estvar_refit_intervals.py --study degeneracy`` sweeps
#: the block count against the project's OWN estimand (paired forced-pick
#: accuracy delta, per-game values in {-1, 0, +1}, zero on the games where both
#: arms pick the same side) at f = 0.10 / 0.20 / 0.55, 2,000 replicates each,
#: and takes the smallest k whose coverage of a known truth is not
#: statistically below 0.90 -- the point at which the miss rate stops being
#: MORE than double the nominal 5%. Measured mean coverage:
#:
#:     k       1     2     3     4     5     6     8    10    13    17
#:     cov  0.000 0.466 0.705 0.760 0.817 0.845 0.882 0.896 0.913 0.921
#:
#: k=8 is 4.6 standard errors below 0.90; k=10 is 1.0 below, i.e. the first
#: block count that is not demonstrably worse. ``docs/anytime_valid.md`` sec 6's
#: "~4-5 blocks" floor is far too generous: coverage at 4 blocks is 0.76.
MIN_BLOCKS_FOR_INTERVAL = 10

#: Above this the coverage curve has plateaued (0.944 at 50 blocks, 0.947 at
#: 199); between the two constants it is usable but measurably narrow (0.90 to
#: 0.94). Same measurement. Note the plateau is at ~0.945, NOT 0.95: even with
#: hundreds of blocks a percentile bootstrap of this skewed, zero-inflated
#: statistic runs about half a point anti-conservative. That is a third,
#: smaller source of narrowness on top of D2 and D4, and it does not go away
#: with more blocks.
RELIABLE_BLOCKS_FOR_INTERVAL = 50

OnDegenerate = Literal["raise", "warn", "ignore"]

FloatArray = npt.NDArray[np.float64]


class BootstrapDegeneracyError(ValueError):
    """Raised when a block bootstrap has too few blocks to support an interval."""


class BootstrapDegeneracyWarning(UserWarning):
    """Warned when a block bootstrap's interval is reported below the floor."""


# ---------------------------------------------------------------------------
# D4: the degeneracy guard
# ---------------------------------------------------------------------------


def distinct_block_resamples(block_count: int) -> int:
    """How many distinct resample COMPOSITIONS exist at ``block_count`` blocks.

    Resampling ``k`` blocks with replacement from ``k`` blocks produces a
    multiset, and the bootstrap statistic depends only on the multiset (how
    many times each block was drawn), not on the draw order. So the number of
    genuinely distinct achievable values is the number of multisets,
    ``C(2k-1, k-1)``: 1 at k=1, 3 at k=2, 10 at k=3, 35 at k=4.

    ``docs/anytime_valid.md`` sec 6 states this formula correctly but then
    quotes "27 at k=3, 256 at k=4", which are ``k**k`` -- the count of ORDERED
    draw tuples, which over-counts because most orderings collapse to the same
    statistic. Its own later "``C(4+4-1, 3) = 35``" is the right number.
    """

    if block_count < 1:
        raise ValueError("block_count must be at least 1")
    return math.comb(2 * block_count - 1, block_count - 1)


@dataclass(frozen=True)
class BlockCountVerdict:
    """Whether a block count can support a percentile interval at all."""

    block_count: int
    distinct_resamples: int
    #: True when the resampling distribution is too coarse for the reported
    #: nominal coverage to be even approximately right.
    degenerate: bool
    #: True when the interval is usable but measurably narrower than nominal.
    marginal: bool
    #: The single worst case: exactly one achievable resample, so the interval
    #: collapses to a point and excludes zero unless the estimate is exactly 0.
    collapses_to_point: bool
    message: str


def block_count_verdict(
    block_count: int,
    *,
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
    reliable_blocks: int = RELIABLE_BLOCKS_FOR_INTERVAL,
) -> BlockCountVerdict:
    """Classify a block count against the measured coverage floor."""

    if block_count < 1:
        raise ValueError("block_count must be at least 1")
    distinct = distinct_block_resamples(block_count)
    collapses = block_count == 1
    degenerate = block_count < min_blocks
    marginal = (not degenerate) and block_count < reliable_blocks
    if collapses:
        message = (
            f"{block_count} bootstrap block: exactly one resample is achievable, so the "
            "'interval' collapses to a point and excludes zero by construction. This is "
            "not an interval."
        )
    elif degenerate:
        message = (
            f"{block_count} bootstrap blocks (< {min_blocks}) admit only {distinct:,} distinct "
            "resamples; measured coverage of a known truth at this block count is well under "
            "the nominal 95% (0.47 at 2 blocks, 0.76 at 4, 0.88 at 8), and the interval comes "
            "back with EXACTLY ZERO width in 25% of 2-block draws, 8% at 3 and 2% at 4. Report "
            "the estimate and probability_positive, not this interval."
        )
    elif marginal:
        message = (
            f"{block_count} bootstrap blocks: usable, but measured coverage runs ~0.90-0.94 "
            f"against nominal 0.95 until ~{reliable_blocks} blocks. Treat the width as a lower "
            "bound."
        )
    else:
        message = (
            f"{block_count} bootstrap blocks: coverage is within Monte-Carlo error of nominal."
        )
    return BlockCountVerdict(
        block_count=block_count,
        distinct_resamples=distinct,
        degenerate=degenerate,
        marginal=marginal,
        collapses_to_point=collapses,
        message=message,
    )


def guard_block_count(
    block_count: int,
    *,
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
    on_degenerate: OnDegenerate = "raise",
    context: str = "block bootstrap",
) -> BlockCountVerdict:
    """Refuse -- loudly -- to hand back an interval too coarse to be one.

    ``on_degenerate='raise'`` (the default for every estimator in this module)
    raises ``BootstrapDegeneracyError``. ``'warn'`` emits a
    ``BootstrapDegeneracyWarning`` and returns the verdict so the caller can
    flag the row; it exists for the three production functions, which are wired
    to warn-and-flag rather than raise so that adding this guard cannot change
    a single number any existing call site already reports.
    """

    verdict = block_count_verdict(block_count, min_blocks=min_blocks)
    if verdict.degenerate:
        message = f"{context}: {verdict.message}"
        if on_degenerate == "raise":
            raise BootstrapDegeneracyError(message)
        if on_degenerate == "warn":
            warnings.warn(message, BootstrapDegeneracyWarning, stacklevel=3)
    return verdict


# ---------------------------------------------------------------------------
# Refitting the mean model under resampled training rows
# ---------------------------------------------------------------------------


def bootstrap_row_indices(n: int, *, n_boot: int, seed: int) -> npt.NDArray[np.int64]:
    """``n_boot`` independent with-replacement resamples of ``range(n)``."""

    if n <= 0:
        raise ValueError("n must be positive")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def _feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> FloatArray:
    """Select numeric features once, preserving pandas' column-major layout."""

    return np.asfortranarray(
        frame.loc[:, list(columns)].to_numpy(dtype=np.float64, na_value=np.nan)
    )


def point_predicted_values(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    ridge_alpha: float,
    random_state: int = 42,
) -> FloatArray:
    """Single fit on every training row -- the value production reports today."""

    columns = list(feature_columns)
    estimator = make_margin_estimator("ridge", random_state, ridge_alpha=ridge_alpha)
    target = pd.to_numeric(training[target_column], errors="raise").to_numpy(dtype=float)
    estimator.fit(training.loc[:, columns], target)
    return np.asarray(estimator.predict(test.loc[:, columns]), dtype=np.float64)


def refit_predicted_values(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    ridge_alpha: float,
    n_boot: int,
    seed: int,
    random_state: int = 42,
) -> FloatArray:
    """Refit on ``n_boot`` row-bootstrap resamples of ``training``; predict on ``test``.

    Returns shape ``(n_boot, len(test))``. Row-level resampling (not block
    resampling) matches the audit this module acts on -- it is the model's OWN
    estimation variance, not a resampling of games, so it operates entirely on
    the training side and never touches the fixed test frame's rows or order.
    """

    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    columns = list(feature_columns)
    x_train = training.loc[:, columns]
    y_train = pd.to_numeric(training[target_column], errors="raise").to_numpy(dtype=float)
    x_test = test.loc[:, columns]
    n = len(training)
    if n == 0:
        raise ValueError("training must contain at least one row")
    indices = bootstrap_row_indices(n, n_boot=n_boot, seed=seed)
    predictions = np.empty((n_boot, len(test)), dtype=np.float64)
    for draw in range(n_boot):
        rows = indices[draw]
        estimator = make_margin_estimator("ridge", random_state, ridge_alpha=ridge_alpha)
        estimator.fit(x_train.iloc[rows], y_train[rows])
        predictions[draw] = np.asarray(estimator.predict(x_test), dtype=np.float64)
    return predictions


def refit_pick_flip_rate(point_values: FloatArray, refit_values: FloatArray) -> float:
    """Fraction of (draw, game) cells whose sign disagrees with the point estimate.

    This is the model's own pick instability under resampled training data --
    the quantity the estimation-variance audit measured at 15-22% for CFB. Sign
    is the right comparator here (not a 0.5 probability threshold): it is scale
    -free across ``margin`` and ``market_residual`` targets alike.
    """

    point = np.asarray(point_values, dtype=np.float64)
    refit = np.asarray(refit_values, dtype=np.float64)
    if refit.shape[-1] != point.shape[-1]:
        raise ValueError("point_values and refit_values must score the same games")
    return float(np.mean(np.sign(refit) != np.sign(point)[np.newaxis, :]))


def refit_value_sd(refit_values: FloatArray) -> FloatArray:
    """Per-game standard deviation of the predicted center across refit draws."""

    return np.asarray(np.std(refit_values, axis=0, ddof=1), dtype=np.float64)


def bagged_values(refit_values: FloatArray) -> FloatArray:
    """The bagged prediction: the mean predicted center across refit draws."""

    return np.asarray(np.mean(refit_values, axis=0), dtype=np.float64)


# ---------------------------------------------------------------------------
# Turning a predicted centre into a cover probability (mirrors margin.py)
# ---------------------------------------------------------------------------


def home_cover_probability_from_center(
    predicted_margin: FloatArray,
    lines: FloatArray,
    residuals: FloatArray,
) -> FloatArray:
    """Vectorized ``margin._smoothed_probability`` across many games sharing one sample.

    Reproduces the exact Laplace/KT continuity correction
    ``(successes + 0.5) / (n + 1)`` production reads off the out-of-time
    residual ECDF, just batched over games instead of called once per game.
    Does not import or call ``margin._smoothed_probability`` (a private
    function); the two are pinned equal by
    ``tests/test_estimation_variance.py``.
    """

    centers = np.asarray(predicted_margin, dtype=np.float64)
    thresholds = np.asarray(lines, dtype=np.float64) - centers
    sample = np.asarray(residuals, dtype=np.float64)
    successes = np.sum(sample[np.newaxis, :] > thresholds[:, np.newaxis], axis=1).astype(np.float64)
    return (successes + 0.5) / (float(len(sample)) + 1.0)


def shrink_predicted_margin(
    spread: FloatArray, raw_residual_prediction: FloatArray, *, shrink_fraction: float
) -> FloatArray:
    """Shrink the predicted CENTRE toward the market line by a uniform fraction.

    ``predicted_margin = spread + shrink_fraction * raw_residual_prediction``.
    ``shrink_fraction=1`` reproduces the production centre exactly;
    ``shrink_fraction=0`` collapses the centre onto the market line (the
    ``market`` arm's centre). This shrinks the estimator's OWN prediction, the
    complement of ``residual_location.shrunk_survival``, which shrinks the
    residual sample's location instead -- ``docs/residual_location.md`` found
    that lever inert; this module asks whether the centre is.
    """

    if not 0.0 <= shrink_fraction <= 1.0:
        raise ValueError("shrink_fraction must be between 0 and 1")
    return np.asarray(spread, dtype=np.float64) + shrink_fraction * np.asarray(
        raw_residual_prediction, dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Paired intervals: naive (currently reported) vs. refit-aware (honest)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedInterval:
    estimate: float
    lower: float
    upper: float
    probability_positive: float
    samples: int
    #: 'naive' resamples games only (today's standard); 'refit_aware' also
    #: resamples training rows and refits, once per outer draw.
    kind: str
    #: How many bootstrap blocks the interval was built from. ``None`` when the
    #: constructor did not record it (kept for backward compatibility).
    block_count: int | None = None
    #: True when ``block_count`` is below the measured coverage floor, so the
    #: bounds are NOT a valid interval. Downstream reporting must not render a
    #: degenerate interval as a normal one.
    degenerate: bool = False


def _paired_accuracy_improvement(
    actual: FloatArray, baseline_prob: FloatArray, candidate_prob: FloatArray
) -> FloatArray:
    actual = np.asarray(actual, dtype=np.float64)
    return np.asarray(
        ((np.asarray(candidate_prob) >= 0.5) == actual).astype(np.float64)
        - ((np.asarray(baseline_prob) >= 0.5) == actual).astype(np.float64),
        dtype=np.float64,
    )


def _grouped_positions(block_ids: npt.NDArray[Any]) -> list[npt.NDArray[np.int64]]:
    _, block_index = np.unique(np.asarray(block_ids), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    return [np.flatnonzero(block_index == group) for group in range(int(block_index.max()) + 1)]


def naive_block_bootstrap_interval(
    actual: FloatArray,
    baseline_prob: FloatArray,
    candidate_prob: FloatArray,
    block_ids: npt.NDArray[Any],
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260812,
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
    on_degenerate: OnDegenerate = "warn",
) -> PairedInterval:
    """The style every recorded interval uses today: resample games, never refit.

    A numpy analogue of ``experiments.paired_feature_comparisons``'s accuracy
    metric, generalized off the pandas feature-set pivot so it can score any
    two FIXED probability arrays. Pinned equal to
    ``paired_feature_comparisons`` on identical inputs by
    ``tests/test_estimation_variance.py``.

    ``on_degenerate`` defaults to ``'warn'`` here ON PURPOSE: this function
    exists to REPRODUCE what the project reports today, so making it refuse
    would change the very baseline every comparison is measured against. It
    still warns below ``MIN_BLOCKS_FOR_INTERVAL`` and stamps ``degenerate=True``
    on the result so no caller can render it as a normal interval by accident.
    The honest estimator (``refit_aware_interval``) raises instead.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    improvements = _paired_accuracy_improvement(actual, baseline_prob, candidate_prob)
    grouped = _grouped_positions(block_ids)
    verdict = guard_block_count(
        len(grouped),
        min_blocks=min_blocks,
        on_degenerate=on_degenerate,
        context="naive_block_bootstrap_interval",
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        positions = np.concatenate([grouped[index] for index in selected])
        draws[sample_index] = float(np.mean(improvements[positions]))
    tail = (1.0 - confidence) / 2.0
    return PairedInterval(
        estimate=float(np.mean(improvements)),
        lower=float(np.quantile(draws, tail)),
        upper=float(np.quantile(draws, 1.0 - tail)),
        probability_positive=float(np.mean(draws > 0.0)),
        samples=samples,
        kind="naive",
        block_count=len(grouped),
        degenerate=verdict.degenerate,
    )


def refit_aware_paired_interval(
    actual: FloatArray,
    baseline_prob_refits: FloatArray,
    candidate_prob_refits: FloatArray,
    block_ids: npt.NDArray[Any],
    *,
    confidence: float = 0.95,
    seed: int = 20260812,
) -> PairedInterval:
    """The honest interval: an independent refit draw AND game-block resample per iteration.

    ``baseline_prob_refits``/``candidate_prob_refits`` are shape
    ``(n_boot, n_games)`` -- one row per independent training-resample-and
    -refit (see ``refit_predicted_values`` +
    ``home_cover_probability_from_center``). For outer draw ``b`` this pairs
    that refit's own probabilities with an INDEPENDENT block-bootstrap
    resample of the test games, so a single loop of ``n_boot`` iterations
    combines both variance sources (they arise from disjoint data -- training
    rows vs. test games -- so combining them additively in one draw is exact,
    not an approximation) without a nested double bootstrap. An arm with no
    training-refit variance of its own (e.g. the unconditional ``market``
    baseline, which fits no estimator) can be passed as a ``(1, n_games)``
    array; it is broadcast to every draw, correctly contributing zero refit
    variance from that side.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    baseline, candidate, n_boot = _broadcast_refits(baseline_prob_refits, candidate_prob_refits)
    if n_boot < 10:
        raise ValueError("At least 10 refit draws are required for an interval")

    grouped = _grouped_positions(block_ids)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for draw_index in range(n_boot):
        improvements = _paired_accuracy_improvement(
            actual, baseline[draw_index], candidate[draw_index]
        )
        selected = rng.integers(0, len(grouped), size=len(grouped))
        positions = np.concatenate([grouped[index] for index in selected])
        draws[draw_index] = float(np.mean(improvements[positions]))

    point_improvements = _paired_accuracy_improvement(
        actual, np.mean(baseline, axis=0), np.mean(candidate, axis=0)
    )
    tail = (1.0 - confidence) / 2.0
    return PairedInterval(
        estimate=float(np.mean(point_improvements)),
        lower=float(np.quantile(draws, tail)),
        upper=float(np.quantile(draws, 1.0 - tail)),
        probability_positive=float(np.mean(draws > 0.0)),
        samples=n_boot,
        kind="refit_aware",
    )


# ---------------------------------------------------------------------------
# D2: paired refits, the variance decomposition, and the honest interval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedRefits:
    """Two arms refit on the SAME resampled training rows, scored on the SAME games.

    ``baseline`` and ``candidate`` are ``(n_boot, n_test)`` predicted centres.
    Row ``b`` of both arrays came from ONE resample of the training rows
    (``row_indices[b]``), which is what makes the pair a paired comparison.
    """

    row_indices: npt.NDArray[np.int64]
    baseline: FloatArray
    candidate: FloatArray
    #: False only when built deliberately unpaired, which inflates the interval
    #: by adding independent noise to each arm. Kept so the error is testable.
    paired: bool = True


def paired_refit_predicted_values(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    baseline_feature_columns: Sequence[str],
    candidate_feature_columns: Sequence[str],
    target_column: str,
    ridge_alpha: float,
    n_boot: int,
    seed: int,
    baseline_ridge_alpha: float | None = None,
    candidate_ridge_alpha: float | None = None,
    random_state: int = 42,
    paired: bool = True,
) -> PairedRefits:
    """Refit BOTH arms on each bootstrap resample of the SAME training rows.

    **This pairing is the whole point.** Paired deltas -- "does arm B beat arm
    A on these same games" -- are this project's primary estimand, not levels.
    If each arm is refit on its OWN independent resample, the two arms' refit
    noise no longer cancels and the delta's variance becomes
    ``Var_A + Var_B`` instead of ``Var(A - B)``, which for two models sharing
    most of their features and all of their training rows is a large
    over-statement in the opposite direction to the D2 defect. Fitting both
    arms on one resample keeps the shared component shared, so only the part
    of the fit that actually differs between the arms contributes.

    ``refit_predicted_values`` called twice with the same ``seed`` and the same
    ``training`` frame happens to produce the same row indices, so the two
    existing scripts that do that are correct today -- but nothing in the API
    said so or enforced it, and a caller passing ``seed`` and ``seed + 1``
    would have silently got the unpaired, over-wide answer. This function makes
    the pairing structural.

    Arms that fit no estimator at all (the unconditional ``market`` baseline)
    do not belong here: pass their fixed probabilities as a ``(1, n_games)``
    array to the interval functions instead, which correctly contributes zero
    refit variance from that side.
    """

    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    n = len(training)
    if n == 0:
        raise ValueError("training must contain at least one row")
    baseline_alpha = ridge_alpha if baseline_ridge_alpha is None else baseline_ridge_alpha
    candidate_alpha = ridge_alpha if candidate_ridge_alpha is None else candidate_ridge_alpha
    target = pd.to_numeric(training[target_column], errors="raise").to_numpy(dtype=float)
    baseline_train = _feature_matrix(training, baseline_feature_columns)
    baseline_test = _feature_matrix(test, baseline_feature_columns)
    candidate_train = _feature_matrix(training, candidate_feature_columns)
    candidate_test = _feature_matrix(test, candidate_feature_columns)

    indices = bootstrap_row_indices(n, n_boot=n_boot, seed=seed)
    candidate_indices = (
        indices if paired else bootstrap_row_indices(n, n_boot=n_boot, seed=seed + 991)
    )

    baseline_out = np.empty((n_boot, len(test)), dtype=np.float64)
    candidate_out = np.empty((n_boot, len(test)), dtype=np.float64)
    for draw in range(n_boot):
        rows = indices[draw]
        baseline_estimator = make_margin_estimator(
            "ridge", random_state, ridge_alpha=baseline_alpha
        )
        baseline_estimator.fit(np.asfortranarray(baseline_train[rows]), target[rows])
        baseline_out[draw] = np.asarray(baseline_estimator.predict(baseline_test), dtype=np.float64)
        candidate_rows = candidate_indices[draw]
        candidate_estimator = make_margin_estimator(
            "ridge", random_state, ridge_alpha=candidate_alpha
        )
        candidate_estimator.fit(
            np.asfortranarray(candidate_train[candidate_rows]), target[candidate_rows]
        )
        candidate_out[draw] = np.asarray(
            candidate_estimator.predict(candidate_test), dtype=np.float64
        )
    return PairedRefits(
        row_indices=indices, baseline=baseline_out, candidate=candidate_out, paired=paired
    )


def _block_sums_and_counts(
    values: FloatArray, block_ids: npt.NDArray[Any]
) -> tuple[FloatArray, FloatArray]:
    _, block_index = np.unique(np.asarray(block_ids), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = int(block_index.max()) + 1
    sums = np.bincount(block_index, weights=values, minlength=block_count).astype(np.float64)
    counts = np.bincount(block_index, minlength=block_count).astype(np.float64)
    return sums, counts


def block_bootstrap_means(
    values: FloatArray,
    block_ids: npt.NDArray[Any],
    *,
    samples: int,
    seed: int,
) -> FloatArray:
    """Vectorized block-bootstrap draws of ``mean(values)``.

    Resampling ``k`` blocks with replacement and concatenating them, then
    taking the mean, is identical to weighting each block's SUM and COUNT by
    how many times it was drawn. Drawing the count vector from a
    ``Multinomial(k, 1/k)`` is exactly drawing ``k`` block indices uniformly
    with replacement, so this returns the same distribution as the explicit
    concatenate-and-average loop in ``naive_block_bootstrap_interval`` and
    ``experiments.paired_feature_comparisons`` -- at a fraction of the cost,
    which is what makes 20,000 samples affordable inside a refit study.
    Pinned distributionally against the loop in the tests.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    sums, counts = _block_sums_and_counts(np.asarray(values, dtype=np.float64), block_ids)
    block_count = len(sums)
    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    return np.asarray((drawn @ sums) / (drawn @ counts), dtype=np.float64)


@dataclass(frozen=True)
class RefitCommonVariance:
    """Training-side variance, split into the part the game bootstrap already carries and the part it does not."""  # noqa: E501

    #: ``Var(a)``: the honest additive term, clamped at zero.
    common: float
    #: ``Var(a)`` before clamping. Negative values are informative -- they mean
    #: the true common variance is at or below the estimator's resolution.
    common_raw: float
    #: Standard error of ``common_raw``. The covariance is estimated from
    #: ``n_boot`` draws, so it is intrinsically noisier than either variance it
    #: is built from; quoting the point estimate without this is how a number
    #: indistinguishable from zero gets reported as a correction.
    common_se: float
    #: ``Var(a) + Var(e)``: the raw spread of refit deltas on the fixed test
    #: games. What the 2026-08-18 estimators used as the additive term.
    fixed_games: float
    draws: int


def refit_common_variance(
    improvements: FloatArray, *, splits: int = 40, seed: int = 20260818
) -> RefitCommonVariance:
    """Separate the training-refit variance that is NOT already in the game bootstrap.

    Write a paired delta as ``Delta(T, G) = mu + a(T) + b(G) + e(T, G)``: a
    training effect, a game-sampling effect, and their INTERACTION. Then

    - the conditional block bootstrap (resample games, one fit) measures
      ``Var(b) + Var(e)``;
    - the spread of refit deltas on the FIXED test games measures
      ``Var(a) + Var(e)``;
    - the honest total is ``Var(a) + Var(b) + Var(e)``.

    So adding the two measured quantities **double-counts ``Var(e)``**, and for
    a forced-pick accuracy delta ``Var(e)`` is not a rounding term: it is the
    dominant part of the refit spread. A refit that flips the pick on game *i*
    moves the delta by ``+-1/n`` depending on whether that particular game
    covered -- pure interaction. Both honest intervals built on 2026-08-18
    (``refit_aware_paired_interval``, and this module's first
    ``refit_aware_interval``) add the two, and both consequently OVER-cover:
    measured 0.987-1.000 against a nominal 0.95 (``docs/estimation_variance.md``
    Part II sec 9c). That is conservative rather than wrong, but it is still a
    defect, and it is what inflated the published 17-58% band.

    ``Var(a)`` alone is recovered by splitting the games into two disjoint
    halves and taking the COVARIANCE of the two halves' refit-delta series
    across draws. ``a(T)`` is common to both halves, so it survives; the
    interaction averages of two disjoint game sets are uncorrelated, so
    ``Var(e)`` cancels. Splitting by GAME rather than by block is correct here:
    within-week game correlation is mandated to be exactly zero in this project
    (``AGENTS.md``), and the across-week component that a training perturbation
    shares IS ``a(T)`` by construction. Averaged over ``splits`` random halvings
    to damp the choice of split.

    The returned ``common`` is clamped at zero (it is a difference of estimates
    and can come out negative when the truth is near zero), and ``common_se``
    is reported alongside precisely because that clamping is not free: on the
    real CFB comparison the point estimate is an order of magnitude BELOW its
    own standard error, so the honest statement is an upper bound, not a
    correction.
    """

    matrix = np.atleast_2d(np.asarray(improvements, dtype=np.float64))
    n_boot, n_games = matrix.shape
    fixed_games_variance = float(np.var(np.mean(matrix, axis=1), ddof=1)) if n_boot > 1 else 0.0
    if n_boot < 3 or n_games < 4 or splits < 1:
        return RefitCommonVariance(0.0, 0.0, 0.0, fixed_games_variance, n_boot)
    rng = np.random.default_rng(seed)
    half = n_games // 2
    covariances = np.empty(splits, dtype=np.float64)
    half_variances = np.empty(splits, dtype=np.float64)
    for index in range(splits):
        order = rng.permutation(n_games)
        first = np.mean(matrix[:, order[:half]], axis=1)
        second = np.mean(matrix[:, order[half : 2 * half]], axis=1)
        moments = np.cov(first, second, ddof=1)
        covariances[index] = float(moments[0, 1])
        half_variances[index] = float(0.5 * (moments[0, 0] + moments[1, 1]))
    raw = float(np.mean(covariances))
    # Standard error of a covariance from n paired draws. The split-to-split
    # spread understates it badly (every split reuses the same draws), so it is
    # computed from the moments instead.
    variance_product = float(np.mean(half_variances)) ** 2
    standard_error = math.sqrt(max(0.0, variance_product + raw * raw) / max(1, n_boot - 1))
    return RefitCommonVariance(
        common=max(0.0, raw),
        common_raw=raw,
        common_se=standard_error,
        fixed_games=fixed_games_variance,
        draws=n_boot,
    )


@dataclass(frozen=True)
class VarianceDecomposition:
    """Where a paired delta's uncertainty comes from, and what it costs to ignore.

    ``inflation_factor`` is DERIVED, never typed in:
    ``sqrt(1 + (refit_sd / conditional_sd) ** 2)``, where ``refit_sd`` is the
    INTERACTION-FREE training component from ``refit_common_variance``. Using
    the raw fixed-games refit spread instead double-counts the training-by-game
    interaction and over-states the factor -- see that function's docstring, and
    ``refit_fixed_games_sd`` below for the size of the difference.
    """

    #: SD of the delta from resampling GAMES around one fit -- what is reported.
    conditional_sd: float
    #: The honest additive training term: ``sqrt(Var(a))``, the part of the
    #: refit spread the game bootstrap does NOT already carry.
    refit_sd: float
    #: Diagnostic: the raw SD of refit deltas on the fixed test games, i.e.
    #: ``sqrt(Var(a) + Var(e))``. This is what the 2026-08-18 estimators added,
    #: and the gap between it and ``refit_sd`` is the double-counted
    #: interaction. Never use it as the additive term.
    refit_fixed_games_sd: float
    #: sqrt(conditional_sd**2 + refit_sd**2).
    total_sd: float
    #: total_sd / conditional_sd. 1.0 means refitting adds nothing.
    inflation_factor: float
    #: One-sided 95% UPPER bound on ``inflation_factor``, from the standard
    #: error of the common-variance estimate. When the point estimate sits
    #: below its own standard error -- which is what happens on real CFB -- this
    #: bound is the honest number to quote, and the point estimate alone is not.
    inflation_factor_upper: float
    #: What the factor would have been under the double-counting decomposition,
    #: kept so any re-reading of the published 17-58% band is reproducible.
    interaction_double_counted_factor: float
    refit_draws: int
    block_count: int
    #: Fraction of (draw, game) cells where the two arms' refits pick opposite
    #: sides -- the ``f`` that drives how large the inflation is.
    paired_refit_flip_fraction: float
    paired: bool


def refit_variance_decomposition(
    actual: FloatArray,
    baseline_prob_refits: FloatArray,
    candidate_prob_refits: FloatArray,
    block_ids: npt.NDArray[Any],
    *,
    point_baseline_prob: FloatArray | None = None,
    point_candidate_prob: FloatArray | None = None,
    samples: int = 20_000,
    seed: int = 20260812,
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
    on_degenerate: OnDegenerate = "raise",
    paired: bool = True,
) -> VarianceDecomposition:
    """Split a paired delta's SD into game-sampling and training-refit parts.

    The refit component is measured on the FIXED test games (no block resample
    inside it), so it is a clean estimate of the training-side variance alone
    and does not double-count the game-sampling variance the conditional
    bootstrap already carries.
    """

    baseline, candidate, _ = _broadcast_refits(baseline_prob_refits, candidate_prob_refits)
    grouped_ids = np.asarray(block_ids)
    verdict = guard_block_count(
        len(np.unique(grouped_ids)),
        min_blocks=min_blocks,
        on_degenerate=on_degenerate,
        context="refit_variance_decomposition",
    )
    point_baseline = (
        np.mean(baseline, axis=0)
        if point_baseline_prob is None
        else np.asarray(point_baseline_prob, dtype=np.float64)
    )
    point_candidate = (
        np.mean(candidate, axis=0)
        if point_candidate_prob is None
        else np.asarray(point_candidate_prob, dtype=np.float64)
    )
    point_improvements = _paired_accuracy_improvement(actual, point_baseline, point_candidate)
    conditional_draws = block_bootstrap_means(
        point_improvements, grouped_ids, samples=samples, seed=seed
    )
    return _decompose(
        actual,
        baseline,
        candidate,
        conditional_draws=conditional_draws,
        block_count=verdict.block_count,
        paired=paired,
    )


def _decompose(
    actual: FloatArray,
    baseline: FloatArray,
    candidate: FloatArray,
    *,
    conditional_draws: FloatArray,
    block_count: int,
    paired: bool,
    splits: int = 40,
    seed: int = 20260818,
) -> VarianceDecomposition:
    """Combine an already-drawn conditional bootstrap with the refit spread."""

    n_boot = len(baseline)
    conditional_sd = float(np.std(conditional_draws, ddof=1))
    improvements = np.empty((n_boot, baseline.shape[1]), dtype=np.float64)
    flips = 0.0
    for draw in range(n_boot):
        improvements[draw] = _paired_accuracy_improvement(actual, baseline[draw], candidate[draw])
        flips += float(np.mean((candidate[draw] >= 0.5) != (baseline[draw] >= 0.5)))
    components = refit_common_variance(improvements, splits=splits, seed=seed)
    refit_sd = math.sqrt(components.common)
    total_sd = float(math.hypot(conditional_sd, refit_sd))
    double_counted_sd = float(math.hypot(conditional_sd, math.sqrt(components.fixed_games)))
    upper_common = max(0.0, components.common_raw + 1.6448536269514722 * components.common_se)
    upper_sd = float(math.hypot(conditional_sd, math.sqrt(upper_common)))
    return VarianceDecomposition(
        conditional_sd=conditional_sd,
        refit_sd=refit_sd,
        refit_fixed_games_sd=math.sqrt(components.fixed_games),
        total_sd=total_sd,
        inflation_factor=float(total_sd / conditional_sd) if conditional_sd > 0.0 else 1.0,
        inflation_factor_upper=(float(upper_sd / conditional_sd) if conditional_sd > 0.0 else 1.0),
        interaction_double_counted_factor=(
            float(double_counted_sd / conditional_sd) if conditional_sd > 0.0 else 1.0
        ),
        refit_draws=n_boot,
        block_count=block_count,
        paired_refit_flip_fraction=float(flips / n_boot),
        paired=paired,
    )


def _broadcast_refits(
    baseline_prob_refits: FloatArray, candidate_prob_refits: FloatArray
) -> tuple[FloatArray, FloatArray, int]:
    baseline = np.atleast_2d(np.asarray(baseline_prob_refits, dtype=np.float64))
    candidate = np.atleast_2d(np.asarray(candidate_prob_refits, dtype=np.float64))
    n_boot = max(len(baseline), len(candidate))
    if len(baseline) == 1 and n_boot > 1:
        baseline = np.repeat(baseline, n_boot, axis=0)
    if len(candidate) == 1 and n_boot > 1:
        candidate = np.repeat(candidate, n_boot, axis=0)
    if len(baseline) != n_boot or len(candidate) != n_boot:
        raise ValueError("baseline/candidate refit draw counts must match (or be broadcastable)")
    return baseline, candidate, n_boot


@dataclass(frozen=True)
class RefitAwareResult:
    """Both intervals plus the decomposition that explains the gap between them."""

    honest: PairedInterval
    naive: PairedInterval
    decomposition: VarianceDecomposition


def refit_aware_interval(
    actual: FloatArray,
    baseline_prob_refits: FloatArray,
    candidate_prob_refits: FloatArray,
    block_ids: npt.NDArray[Any],
    *,
    point_baseline_prob: FloatArray | None = None,
    point_candidate_prob: FloatArray | None = None,
    samples: int = 20_000,
    confidence: float = 0.95,
    seed: int = 20260812,
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
    on_degenerate: OnDegenerate = "raise",
    paired: bool = True,
) -> RefitAwareResult:
    """The honest interval: game-sampling variance PLUS training-refit variance.

    Construction, and why it is not the same as
    ``refit_aware_paired_interval`` (kept, unchanged, for reproducibility):

    1. The conditional block-bootstrap draws are taken around the POINT fit --
       the value production actually reports -- with the full ``samples``
       budget, so the interval's shape and quantile resolution are as fine as
       the naive interval's. ``refit_aware_paired_interval`` instead read its
       quantiles off only ``n_boot`` draws, which at the ``N_BOOT=120`` used in
       ``docs/estimation_variance.md`` sec 3 is a coarse 95% quantile (its own
       declared limitation 3), and it re-centred each draw on a bootstrap
       REFIT, which is systematically slightly worse than the full-data fit,
       so the interval inherited that bias as a shift.
    2. The refit draws supply ONE number: ``refit_sd``, the INTERACTION-FREE
       training component (``refit_common_variance``). Taking the raw spread of
       refit deltas on the fixed test games instead -- which is what BOTH
       2026-08-18 estimators did -- double-counts the training-by-game
       interaction that the conditional bootstrap already carries, and makes
       the interval over-cover (measured 0.987-1.000 against nominal 0.95).
    3. The conditional draws are then rescaled about THEIR OWN mean by
       ``inflation_factor = sqrt(1 + (refit_sd/conditional_sd)**2)``, and the
       bounds and ``probability_positive`` are read off the rescaled draws.
       Rescaling rather than re-centring keeps the bootstrap's own skew, and
       scaling about the draws' mean rather than the point estimate makes the
       honest interval an exact widening of the interval this project already
       reports: at ``inflation_factor == 1`` it returns the naive bounds to
       floating-point equality, so any difference between the two IS the refit
       variance and nothing else.

    ``probability_positive`` is the headline this returns, not "excludes zero":
    per ``AGENTS.md``, widening an interval is never grounds to close a signal.
    """

    baseline, candidate, n_boot = _broadcast_refits(baseline_prob_refits, candidate_prob_refits)
    if n_boot < 2:
        raise ValueError("At least 2 refit draws are required to estimate refit variance")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    grouped_ids = np.asarray(block_ids)
    verdict = guard_block_count(
        len(np.unique(grouped_ids)),
        min_blocks=min_blocks,
        on_degenerate=on_degenerate,
        context="refit_aware_interval",
    )
    point_baseline = (
        np.mean(baseline, axis=0)
        if point_baseline_prob is None
        else np.asarray(point_baseline_prob, dtype=np.float64)
    )
    point_candidate = (
        np.mean(candidate, axis=0)
        if point_candidate_prob is None
        else np.asarray(point_candidate_prob, dtype=np.float64)
    )
    point_improvements = _paired_accuracy_improvement(actual, point_baseline, point_candidate)
    estimate = float(np.mean(point_improvements))
    conditional_draws = block_bootstrap_means(
        point_improvements, grouped_ids, samples=samples, seed=seed
    )
    decomposition = _decompose(
        actual,
        baseline,
        candidate,
        conditional_draws=conditional_draws,
        block_count=verdict.block_count,
        paired=paired,
    )
    tail = (1.0 - confidence) / 2.0
    naive = PairedInterval(
        estimate=estimate,
        lower=float(np.quantile(conditional_draws, tail)),
        upper=float(np.quantile(conditional_draws, 1.0 - tail)),
        probability_positive=float(np.mean(conditional_draws > 0.0)),
        samples=samples,
        kind="naive",
        block_count=verdict.block_count,
        degenerate=verdict.degenerate,
    )
    center = float(np.mean(conditional_draws))
    scaled = center + decomposition.inflation_factor * (conditional_draws - center)
    honest = PairedInterval(
        estimate=estimate,
        lower=float(np.quantile(scaled, tail)),
        upper=float(np.quantile(scaled, 1.0 - tail)),
        probability_positive=float(np.mean(scaled > 0.0)),
        samples=samples,
        kind="refit_aware",
        block_count=verdict.block_count,
        degenerate=verdict.degenerate,
    )
    return RefitAwareResult(honest=honest, naive=naive, decomposition=decomposition)


def inflate_recorded_interval(
    estimate: float,
    lower: float | None,
    upper: float | None,
    *,
    inflation_factor: float,
    confidence: float = 0.95,
    probability_positive: float | None = None,
) -> PairedInterval:
    """Re-read an ALREADY RECORDED interval under the honest widening.

    The cheap path for the registry: an entry recorded ``estimate`` and either
    a conditional interval or a conditional ``probability_positive``. Both pin
    down the same conditional SD, so either can be widened without re-running
    anything. The implied SD is taken from the interval when present (it is the
    more direct record) and from ``probability_positive`` otherwise, via
    ``sd = estimate / Phi_inv(P+)``.

    Returns bounds and a ``probability_positive`` under the widened SD, using a
    normal reference distribution -- the recorded rows do not keep their
    bootstrap draws, so the bootstrap's own skew cannot be preserved here. The
    ``kind`` is stamped ``'recorded_inflated'`` so no caller mistakes this for
    a re-measurement.
    """

    if inflation_factor < 1.0:
        raise ValueError("inflation_factor must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    z = float(_normal_quantile(1.0 - (1.0 - confidence) / 2.0))
    conditional_sd: float | None = None
    if lower is not None and upper is not None and upper > lower:
        conditional_sd = (float(upper) - float(lower)) / (2.0 * z)
    elif probability_positive is not None and 0.0 < probability_positive < 1.0:
        quantile = _normal_quantile(float(probability_positive))
        if quantile != 0.0:
            conditional_sd = abs(float(estimate) / quantile)
    if conditional_sd is None or conditional_sd <= 0.0:
        raise ValueError(
            "Cannot recover a conditional SD: the entry records neither a usable interval "
            "nor a usable probability_positive"
        )
    honest_sd = conditional_sd * float(inflation_factor)
    return PairedInterval(
        estimate=float(estimate),
        lower=float(estimate) - z * honest_sd,
        upper=float(estimate) + z * honest_sd,
        probability_positive=float(_normal_cdf(float(estimate) / honest_sd)),
        samples=0,
        kind="recorded_inflated",
    )


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, refined).

    Accurate to better than 1e-9 after one Halley step, which is far finer than
    any registry entry's recorded precision. Avoids a scipy dependency in a
    module the whole project imports.
    """

    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    error = _normal_cdf(x) - p
    density = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if density > 0.0:
        u = error / density
        x = x - u / (1.0 + 0.5 * x * u)
    return x


# ---------------------------------------------------------------------------
# The f lever: MDE80 = 280 * sqrt(f / n), and gating a candidate's influence
# ---------------------------------------------------------------------------


def picks_differ_fraction(baseline_prob: FloatArray, candidate_prob: FloatArray) -> float:
    """``f``: the fraction of games where the two arms' forced picks differ."""

    baseline_pick = np.asarray(baseline_prob, dtype=np.float64) >= 0.5
    candidate_pick = np.asarray(candidate_prob, dtype=np.float64) >= 0.5
    return float(np.mean(baseline_pick != candidate_pick))


def mde80(f: float, n: int, *, coefficient: float = DEFAULT_MDE80_COEFFICIENT) -> float:
    """Minimum detectable effect (accuracy points) at 80% power: ``coefficient*sqrt(f/n)``."""

    if f < 0.0:
        raise ValueError("f must be non-negative")
    if n <= 0:
        raise ValueError("n must be positive")
    return coefficient * math.sqrt(f / float(n))


def gate_by_disagreement(
    baseline_prob: FloatArray, candidate_prob: FloatArray, *, threshold: float
) -> FloatArray:
    """Defer to the baseline wherever the candidate's opinion is too close to call.

    The surgical-candidate design: only let the candidate move a pick where it
    disagrees with the baseline by at least ``threshold`` in probability space;
    elsewhere its probability is replaced by the baseline's EXACTLY, so it
    contributes zero to ``f`` (and to every downstream metric) on that game.
    ``threshold=0`` recovers the candidate untouched; a large enough threshold
    recovers the baseline untouched (probabilities are bounded in ``[0, 1]``,
    so no candidate can disagree by more than 1).
    """

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    baseline = np.asarray(baseline_prob, dtype=np.float64)
    candidate = np.asarray(candidate_prob, dtype=np.float64)
    disagreement = np.abs(candidate - baseline)
    return np.asarray(np.where(disagreement >= threshold, candidate, baseline), dtype=np.float64)
