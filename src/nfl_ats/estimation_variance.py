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
- ``refit_aware_paired_interval`` -- the HONEST interval: combines an
  independent training-refit draw with an independent game-block resample in
  each outer iteration, so its spread reflects both variance sources at once
  without a nested double bootstrap.
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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.margin import make_margin_estimator

#: MDE80 = coefficient * sqrt(f / n), points of forced-pick accuracy. Derived
#: in the evaluator-power audit this module acts on; declared once so every
#: caller reports the same constant instead of retyping it.
DEFAULT_MDE80_COEFFICIENT = 280.0

FloatArray = npt.NDArray[np.float64]


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
) -> PairedInterval:
    """The style every recorded interval uses today: resample games, never refit.

    A numpy analogue of ``experiments.paired_feature_comparisons``'s accuracy
    metric, generalized off the pandas feature-set pivot so it can score any
    two FIXED probability arrays. Pinned equal to
    ``paired_feature_comparisons`` on identical inputs by
    ``tests/test_estimation_variance.py``.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    improvements = _paired_accuracy_improvement(actual, baseline_prob, candidate_prob)
    grouped = _grouped_positions(block_ids)
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
    baseline = np.atleast_2d(np.asarray(baseline_prob_refits, dtype=np.float64))
    candidate = np.atleast_2d(np.asarray(candidate_prob_refits, dtype=np.float64))
    n_boot = max(len(baseline), len(candidate))
    if len(baseline) == 1 and n_boot > 1:
        baseline = np.repeat(baseline, n_boot, axis=0)
    if len(candidate) == 1 and n_boot > 1:
        candidate = np.repeat(candidate, n_boot, axis=0)
    if len(baseline) != n_boot or len(candidate) != n_boot:
        raise ValueError("baseline/candidate refit draw counts must match (or be broadcastable)")
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
