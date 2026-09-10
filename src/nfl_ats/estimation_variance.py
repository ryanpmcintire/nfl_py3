from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.margin import make_margin_estimator

DEFAULT_MDE80_COEFFICIENT = 280.0

MIN_BLOCKS_FOR_INTERVAL = 10

RELIABLE_BLOCKS_FOR_INTERVAL = 50

OnDegenerate = Literal["raise", "warn", "ignore"]

FloatArray = npt.NDArray[np.float64]


class BootstrapDegeneracyError(ValueError):
    pass


class BootstrapDegeneracyWarning(UserWarning):
    pass


def distinct_block_resamples(block_count: int) -> int:

    if block_count < 1:
        raise ValueError("block_count must be at least 1")
    return math.comb(2 * block_count - 1, block_count - 1)


@dataclass(frozen=True)
class BlockCountVerdict:
    block_count: int
    distinct_resamples: int
    degenerate: bool
    marginal: bool
    collapses_to_point: bool
    message: str


def block_count_verdict(
    block_count: int,
    *,
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
    reliable_blocks: int = RELIABLE_BLOCKS_FOR_INTERVAL,
) -> BlockCountVerdict:

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

    verdict = block_count_verdict(block_count, min_blocks=min_blocks)
    if verdict.degenerate:
        message = f"{context}: {verdict.message}"
        if on_degenerate == "raise":
            raise BootstrapDegeneracyError(message)
        if on_degenerate == "warn":
            warnings.warn(message, BootstrapDegeneracyWarning, stacklevel=3)
    return verdict


def bootstrap_row_indices(n: int, *, n_boot: int, seed: int) -> npt.NDArray[np.int64]:

    if n <= 0:
        raise ValueError("n must be positive")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def _feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> FloatArray:

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

    point = np.asarray(point_values, dtype=np.float64)
    refit = np.asarray(refit_values, dtype=np.float64)
    if refit.shape[-1] != point.shape[-1]:
        raise ValueError("point_values and refit_values must score the same games")
    return float(np.mean(np.sign(refit) != np.sign(point)[np.newaxis, :]))


def refit_value_sd(refit_values: FloatArray) -> FloatArray:

    return np.asarray(np.std(refit_values, axis=0, ddof=1), dtype=np.float64)


def bagged_values(refit_values: FloatArray) -> FloatArray:

    return np.asarray(np.mean(refit_values, axis=0), dtype=np.float64)


def home_cover_probability_from_center(
    predicted_margin: FloatArray,
    lines: FloatArray,
    residuals: FloatArray,
) -> FloatArray:

    centers = np.asarray(predicted_margin, dtype=np.float64)
    thresholds = np.asarray(lines, dtype=np.float64) - centers
    sample = np.asarray(residuals, dtype=np.float64)
    successes = np.sum(sample[np.newaxis, :] > thresholds[:, np.newaxis], axis=1).astype(np.float64)
    return (successes + 0.5) / (float(len(sample)) + 1.0)


def shrink_predicted_margin(
    spread: FloatArray, raw_residual_prediction: FloatArray, *, shrink_fraction: float
) -> FloatArray:

    if not 0.0 <= shrink_fraction <= 1.0:
        raise ValueError("shrink_fraction must be between 0 and 1")
    return np.asarray(spread, dtype=np.float64) + shrink_fraction * np.asarray(
        raw_residual_prediction, dtype=np.float64
    )


@dataclass(frozen=True)
class PairedInterval:
    estimate: float
    lower: float
    upper: float
    probability_positive: float
    samples: int
    kind: str
    block_count: int | None = None
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
        probability_positive=float(probability_positive_from_draws(draws)),
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
        probability_positive=float(probability_positive_from_draws(draws)),
        samples=n_boot,
        kind="refit_aware",
    )


@dataclass(frozen=True)
class PairedRefits:
    row_indices: npt.NDArray[np.int64]
    baseline: FloatArray
    candidate: FloatArray
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

    if samples < 10:
        raise ValueError("samples must be at least 10")
    sums, counts = _block_sums_and_counts(np.asarray(values, dtype=np.float64), block_ids)
    block_count = len(sums)
    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    return np.asarray((drawn @ sums) / (drawn @ counts), dtype=np.float64)


@dataclass(frozen=True)
class RefitCommonVariance:
    common: float
    common_raw: float
    common_se: float
    fixed_games: float
    draws: int


def refit_common_variance(
    improvements: FloatArray, *, splits: int = 40, seed: int = 20260818
) -> RefitCommonVariance:

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
    conditional_sd: float
    refit_sd: float
    refit_fixed_games_sd: float
    total_sd: float
    inflation_factor: float
    inflation_factor_upper: float
    interaction_double_counted_factor: float
    refit_draws: int
    block_count: int
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
        probability_positive=float(probability_positive_from_draws(conditional_draws)),
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
        probability_positive=float(probability_positive_from_draws(scaled)),
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


def picks_differ_fraction(baseline_prob: FloatArray, candidate_prob: FloatArray) -> float:

    baseline_pick = np.asarray(baseline_prob, dtype=np.float64) >= 0.5
    candidate_pick = np.asarray(candidate_prob, dtype=np.float64) >= 0.5
    return float(np.mean(baseline_pick != candidate_pick))


def mde80(f: float, n: int, *, coefficient: float = DEFAULT_MDE80_COEFFICIENT) -> float:

    if f < 0.0:
        raise ValueError("f must be non-negative")
    if n <= 0:
        raise ValueError("n must be positive")
    return coefficient * math.sqrt(f / float(n))


def gate_by_disagreement(
    baseline_prob: FloatArray, candidate_prob: FloatArray, *, threshold: float
) -> FloatArray:

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    baseline = np.asarray(baseline_prob, dtype=np.float64)
    candidate = np.asarray(candidate_prob, dtype=np.float64)
    disagreement = np.abs(candidate - baseline)
    return np.asarray(np.where(disagreement >= threshold, candidate, baseline), dtype=np.float64)
