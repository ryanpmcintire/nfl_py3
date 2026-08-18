"""Variance-reduction tools layered on top of ``experiments.paired_feature_comparisons``.

The evaluator's binding constraint is statistical power: at this project's
sample sizes the paired bootstrap resolves roughly a 2-accuracy-point
difference, while real feature effects run 0.2-1.3 points, so most
comparisons come back unresolved and burn a scarce evaluation window without
answering anything. This module adds two independent levers for shrinking
the comparison's confidence interval -- neither one changes a pick, a
feature profile, a calibration method, or a model artifact; both only change
how precisely an already-computed pair of forced-pick probability streams is
compared.

Lever 1 -- covariate-adjusted (CUPED-style) comparison
    ``covariate_adjusted_paired_comparisons`` subtracts off the part of each
    game's paired improvement that a strictly pregame, arm-identical
    covariate (spread magnitude, total line, key-number status, rest
    differential, week number) already predicts, then bootstraps the
    residual exactly as ``paired_feature_comparisons`` bootstraps the raw
    improvement. ``cuped_adjust`` is the load-bearing primitive: the
    adjustment's point estimate equals the raw sample mean *for any theta*,
    proven algebraically (the centered covariate sums to zero, so the
    adjustment term sums to zero) and pinned by
    ``tests/test_variance_reduction.py::test_cuped_adjust_preserves_mean_for_arbitrary_theta``.
    Only the bootstrap's spread can change.

Lever 2 -- a screening ladder on continuous metrics
    ``screening_ladder_decision`` formalizes what this repo has already
    observed informally (MOD-16, the ridge-alpha sweep): Brier and log-loss
    intervals can exclude zero while forced-pick accuracy stays a coin flip,
    because a proper scoring rule uses each game's predicted probability
    magnitude, not just which side of 0.5 it landed on. The rule this module
    encodes is: screen candidates on a continuous metric's
    ``probability_positive`` at an affordable sample size, and only spend a
    scarce (NFL) confirmation window on forced-pick accuracy for candidates
    that clear the screen. A continuous-metric result never substitutes for
    the accuracy verdict -- the pool grades forced picks, not Brier score.

Both levers are validated empirically against planted effects of known size
in ``scripts/variance_planted_effects.py``, not by asymptotic theory alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.experiments import PairedBlock, _paired_row_improvements
from nfl_ats.key_numbers import DEFAULT_KEY_NUMBERS

IMPROVEMENT_METRICS: tuple[str, ...] = (
    "accuracy_improvement",
    "brier_improvement",
    "log_loss_improvement",
)

# Candidate pregame, arm-identical covariates for the CUPED adjustment. Every
# one is knowable before kickoff, and by the paired design each is literally
# the same value for the baseline and candidate arm on a given game -- so
# adjusting on it cannot bias which arm looks better, only how precisely the
# comparison resolves it. Ordering matches the task's own suggestion: spread
# magnitude, total line, key-number status, rest differential, week number.
DEFAULT_CUPED_COVARIATES: tuple[str, ...] = (
    "abs_spread_line",
    "total_line",
    "on_key_number",
    "abs_rest_diff",
    "week_number",
)


def build_cuped_covariates(
    features: pd.DataFrame, key_numbers: tuple[int, ...] = DEFAULT_KEY_NUMBERS
) -> pd.DataFrame:
    """Derive ``DEFAULT_CUPED_COVARIATES`` from a CFB/NFL game feature table.

    Every source column (``spread_line``, ``total_line``, ``rest_diff``,
    ``week``) is set before kickoff, so nothing here can leak the outcome.
    ``rest_diff`` is missing for a team's first game of a season (no prior
    game to measure rest from, ~6% of CFB rows); it is imputed to 0 (average
    rest) rather than dropping the game, matching this project's convention
    of not discarding rows over one missing contextual feature. A handful of
    ``total_line``/``*_team_games`` gaps (<0.2%) are left as NaN here and
    imputed downstream, at adjustment time, from the paired sample's own
    mean -- so the imputation value is always computed from data available
    at comparison time, never a global constant baked in ahead of time.
    """

    required = {"game_id", "spread_line", "week"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Covariate source is missing columns: {', '.join(missing)}")

    frame = features.loc[:, ["game_id"]].copy()
    abs_spread = pd.to_numeric(features["spread_line"], errors="coerce").abs()
    frame["abs_spread_line"] = abs_spread
    frame["total_line"] = (
        pd.to_numeric(features["total_line"], errors="coerce")
        if "total_line" in features.columns
        else np.nan
    )
    key_number_set = frozenset(float(value) for value in key_numbers)
    rounded = abs_spread.round()
    on_key = np.isclose(abs_spread, rounded) & rounded.isin(key_number_set)
    frame["on_key_number"] = on_key.astype(float)
    if "rest_diff" in features.columns:
        rest_diff = pd.to_numeric(features["rest_diff"], errors="coerce").abs()
    else:
        rest_diff = pd.Series(np.nan, index=features.index)
    frame["abs_rest_diff"] = rest_diff.fillna(0.0)
    frame["week_number"] = pd.to_numeric(features["week"], errors="coerce")
    return frame


def _impute_with_sample_mean(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    """Fill remaining NaNs in-place with each column's own sample mean.

    Returns the fraction of rows imputed per column, for reporting. Using
    the paired sample's own mean keeps the imputation leak-safe (it never
    reaches outside the comparison being run) and, since CUPED centers on
    the sample mean anyway, an imputed value contributes exactly zero to
    the adjustment term for that row.
    """

    coverage: dict[str, float] = {}
    for column in columns:
        values = frame[column]
        missing = values.isna()
        coverage[column] = float(missing.mean())
        if missing.any():
            frame.loc[missing, column] = float(values.mean())
    return coverage


def cuped_adjust(
    values: np.ndarray, covariates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract the part of ``values`` linearly predictable from ``covariates``.

    ``values`` may be 1-D ``(n_games,)`` or 2-D ``(n_games, n_series)`` (one
    OLS fit per series, sharing the same design matrix). Returns
    ``(adjusted, theta, covariate_means)`` with ``adjusted`` the same shape
    as ``values``.

    Unbiasedness: ``mean(adjusted, axis=0) == mean(values, axis=0)`` for ANY
    theta, not just the fitted one. ``covariates - covariate_means`` sums to
    exactly zero over the sample by construction, so
    ``theta @ (covariates - covariate_means)`` also sums to exactly zero no
    matter what theta is -- fitting theta on the same sample that gets
    adjusted cannot move the point estimate, it can only reduce (or, for a
    useless/overfit theta, leave unchanged) the per-game variance the
    bootstrap then resamples from.
    """

    values_array = np.asarray(values, dtype=float)
    covariates_array = np.asarray(covariates, dtype=float)
    if covariates_array.ndim != 2 or covariates_array.shape[0] != values_array.shape[0]:
        raise ValueError("covariates must be a (n_games, n_covariates) array")
    squeeze = values_array.ndim == 1
    target = values_array[:, None] if squeeze else values_array

    means = covariates_array.mean(axis=0)
    centered = covariates_array - means
    centered_target = target - target.mean(axis=0, keepdims=True)
    theta, *_ = np.linalg.lstsq(centered, centered_target, rcond=None)
    adjusted = target - centered @ theta
    if squeeze:
        return adjusted[:, 0], theta[:, 0], means
    return adjusted, theta, means


def paired_block_groups(paired: pd.DataFrame, block: PairedBlock) -> tuple[np.ndarray, int]:
    """Assign each paired row an integer block id (0-indexed, contiguous).

    Mirrors ``experiments.paired_feature_comparisons``'s own grouping
    exactly: ``week`` blocks on ``(season_baseline, week_baseline)``,
    ``season`` blocks on ``season_baseline`` alone.
    """

    group_columns = ["season_baseline", "week_baseline"] if block == "week" else ["season_baseline"]
    grouped_indices = list(paired.groupby(group_columns, sort=False).indices.values())
    group_of_row = np.empty(len(paired), dtype=np.int64)
    for group_id, positions in enumerate(grouped_indices):
        group_of_row[positions] = group_id
    return group_of_row, len(grouped_indices)


def fast_block_bootstrap_means(
    values: np.ndarray,
    group_of_row: np.ndarray,
    n_groups: int,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Block-bootstrap means of one or more per-game series at once.

    ``values`` is ``(n_games, n_series)``. Returns ``(samples, n_series)``.
    Reuses the sufficient-statistics trick already established in
    ``outcomes.outcome_bootstrap_intervals``: the mean of concatenated
    per-game values across resampled blocks equals (sum of resampled block
    sums) / (sum of resampled block sizes), so materializing resampled rows
    is never necessary -- every series shares one draw matrix, which also
    means raw and adjusted series here are compared under identical
    resampling noise (common random numbers), not independent bootstrap
    runs.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    n_series = values.shape[1]
    group_sum = np.zeros((n_groups, n_series), dtype=np.float64)
    np.add.at(group_sum, group_of_row, values)
    group_size = np.bincount(group_of_row, minlength=n_groups).astype(np.float64)

    generator = np.random.default_rng(seed)
    draws = generator.integers(0, n_groups, size=(samples, n_groups))
    sample_rows = np.repeat(np.arange(samples), n_groups)
    counts = np.zeros((samples, n_groups), dtype=np.float64)
    np.add.at(counts, (sample_rows, draws.ravel()), 1.0)

    numerator = counts @ group_sum
    denominator = counts @ group_size
    return numerator / denominator[:, None]


@dataclass(frozen=True)
class CupedComparisonResult:
    """Output of ``covariate_adjusted_paired_comparisons``."""

    comparisons: pd.DataFrame
    covariate_effects: pd.DataFrame


def _paired_frame(
    predictions: pd.DataFrame, baseline_feature_set: str, candidate_name: str
) -> pd.DataFrame:
    columns = ["game_id", "season", "week", "home_cover", "home_cover_probability"]
    baseline = predictions.loc[predictions["feature_set"].eq(baseline_feature_set), columns]
    candidate = predictions.loc[predictions["feature_set"].eq(candidate_name), columns]
    paired = baseline.merge(
        candidate,
        on="game_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    paired = paired.loc[
        paired["home_cover_baseline"].notna() & paired["home_cover_candidate"].notna()
    ].copy()
    if paired.empty:
        raise ValueError(f"No paired completed games for {candidate_name}")
    for column in ("season", "week", "home_cover"):
        if not paired[f"{column}_baseline"].equals(paired[f"{column}_candidate"]):
            raise ValueError(f"Paired {column} values differ for {candidate_name}")
    return paired


def covariate_adjusted_paired_comparisons(
    predictions: pd.DataFrame,
    covariates: pd.DataFrame,
    *,
    baseline_feature_set: str,
    covariate_columns: tuple[str, ...] = DEFAULT_CUPED_COVARIATES,
    samples: int = 2_000,
    confidence: float = 0.95,
    block: PairedBlock = "week",
    seed: int = 20260818,
) -> CupedComparisonResult:
    """CUPED-adjusted counterpart of ``experiments.paired_feature_comparisons``.

    Same required ``predictions`` schema (``feature_set``, ``game_id``,
    ``season``, ``week``, ``home_cover``, ``home_cover_probability``) and the
    same per-game improvement definitions (accuracy / Brier / log-loss). This
    version additionally requires ``covariates``: one row per ``game_id``
    with every column in ``covariate_columns`` (see
    ``build_cuped_covariates``). Returned ``comparisons`` carries the same
    columns as ``paired_feature_comparisons`` PLUS ``raw_estimate``,
    ``raw_lower``, ``raw_upper``, ``raw_probability_positive``,
    ``variance_reduction_pct``, and ``effective_sample_multiplier`` (how many
    times more data the unadjusted method would need to match this
    estimator's precision: ``raw_variance / adjusted_variance``).
    ``estimate`` is unchanged from the unadjusted mean by construction (see
    ``cuped_adjust``) -- only ``lower``/``upper``/``probability_positive``
    differ from the unadjusted comparison.

    ``covariate_effects`` reports, per (candidate, metric, covariate), the
    fitted joint-model theta AND the variance reduction each covariate would
    achieve alone (univariate CUPED) -- which covariates actually pay is an
    empirical question this answers directly, not something to assume from
    the covariate list.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    required_predictions = {
        "feature_set",
        "game_id",
        "season",
        "week",
        "home_cover",
        "home_cover_probability",
    }
    missing_predictions = sorted(required_predictions.difference(predictions.columns))
    if missing_predictions:
        raise ValueError(
            f"Predictions are missing paired columns: {', '.join(missing_predictions)}"
        )
    required_covariates = {"game_id", *covariate_columns}
    missing_covariates = sorted(required_covariates.difference(covariates.columns))
    if missing_covariates:
        raise ValueError(f"Covariates are missing columns: {', '.join(missing_covariates)}")
    if covariates["game_id"].duplicated().any():
        raise ValueError("Covariates must contain one row per game_id")

    feature_sets = set(predictions["feature_set"].astype(str))
    if baseline_feature_set not in feature_sets:
        raise ValueError(f"Unknown paired baseline feature set: {baseline_feature_set}")

    tail = (1.0 - confidence) / 2.0
    comparison_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for candidate_name in sorted(feature_sets.difference((baseline_feature_set,))):
        paired = _paired_frame(predictions, baseline_feature_set, candidate_name)
        paired = paired.merge(
            covariates.loc[:, ["game_id", *covariate_columns]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )
        coverage = _impute_with_sample_mean(paired, covariate_columns)

        raw = _paired_row_improvements(paired)
        raw_matrix = raw.to_numpy(dtype=float)
        covariate_matrix = paired.loc[:, list(covariate_columns)].to_numpy(dtype=float)
        adjusted_matrix, theta, covariate_means = cuped_adjust(raw_matrix, covariate_matrix)
        if not np.allclose(adjusted_matrix.mean(axis=0), raw_matrix.mean(axis=0), atol=1e-9):
            raise AssertionError("CUPED adjustment moved the point estimate; this is a bug")

        group_of_row, n_groups = paired_block_groups(paired, block)
        combined = np.concatenate([raw_matrix, adjusted_matrix], axis=1)
        draws = fast_block_bootstrap_means(
            combined, group_of_row, n_groups, samples=samples, seed=seed
        )
        n_metrics = len(IMPROVEMENT_METRICS)
        raw_draws = draws[:, :n_metrics]
        adjusted_draws = draws[:, n_metrics:]

        for metric_index, metric in enumerate(IMPROVEMENT_METRICS):
            raw_variance = float(np.var(raw_draws[:, metric_index], ddof=1))
            adjusted_variance = float(np.var(adjusted_draws[:, metric_index], ddof=1))
            variance_reduction = 1.0 - adjusted_variance / raw_variance if raw_variance > 0 else 0.0
            effective_multiplier = (
                raw_variance / adjusted_variance if adjusted_variance > 0 else float("inf")
            )
            comparison_rows.append(
                {
                    "baseline_feature_set": baseline_feature_set,
                    "candidate_feature_set": candidate_name,
                    "metric": metric,
                    "estimate": float(adjusted_matrix[:, metric_index].mean()),
                    "raw_estimate": float(raw_matrix[:, metric_index].mean()),
                    "lower": float(np.quantile(adjusted_draws[:, metric_index], tail)),
                    "upper": float(np.quantile(adjusted_draws[:, metric_index], 1.0 - tail)),
                    "probability_positive": float(np.mean(adjusted_draws[:, metric_index] > 0.0)),
                    "raw_lower": float(np.quantile(raw_draws[:, metric_index], tail)),
                    "raw_upper": float(np.quantile(raw_draws[:, metric_index], 1.0 - tail)),
                    "raw_probability_positive": float(np.mean(raw_draws[:, metric_index] > 0.0)),
                    "raw_variance": raw_variance,
                    "adjusted_variance": adjusted_variance,
                    "variance_reduction_pct": variance_reduction,
                    "effective_sample_multiplier": effective_multiplier,
                    "confidence": confidence,
                    "block": block,
                    "samples": samples,
                    "paired_games": len(paired),
                    "covariate_columns": ",".join(covariate_columns),
                }
            )
            for covariate_index, covariate_name in enumerate(covariate_columns):
                uni_adjusted, _, _ = cuped_adjust(
                    raw_matrix[:, metric_index], covariate_matrix[:, [covariate_index]]
                )
                uni_variance = float(np.var(uni_adjusted, ddof=1))
                raw_row_variance = float(np.var(raw_matrix[:, metric_index], ddof=1))
                effect_rows.append(
                    {
                        "candidate_feature_set": candidate_name,
                        "metric": metric,
                        "covariate": covariate_name,
                        "theta": float(theta[covariate_index, metric_index]),
                        "covariate_mean": float(covariate_means[covariate_index]),
                        "missing_fraction": coverage[covariate_name],
                        "univariate_variance_reduction_pct": (
                            1.0 - uni_variance / raw_row_variance if raw_row_variance > 0 else 0.0
                        ),
                    }
                )
    return CupedComparisonResult(
        comparisons=pd.DataFrame(comparison_rows), covariate_effects=pd.DataFrame(effect_rows)
    )


def plant_accuracy_effect(
    baseline_probability: np.ndarray,
    actual: np.ndarray,
    *,
    target_accuracy_delta: float,
    probability_noise: np.ndarray | None = None,
    probability_floor: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Build a candidate probability stream with an (approximately) planted accuracy edge.

    The candidate is ``baseline + probability_noise + delta * direction``,
    where ``direction`` always points toward the realized outcome (up if the
    home team covered, down if it did not) and ``probability_noise`` is
    caller-supplied, zero-mean, outcome-INdependent per-game noise (default
    all zeros). Adding ``delta * direction`` can only ever flip an incorrect
    forced pick to correct, never the reverse, for ANY fixed noise draw --
    so forced-pick accuracy is still a monotone, non-decreasing function of
    ``delta`` alone, and the ``delta`` that hits a target accuracy edge is
    still exactly solvable by sorting each game's remaining distance to its
    flip point (now measured from ``baseline + noise`` instead of from
    ``baseline``) and taking the k-th smallest.

    ``probability_noise`` matters for validating continuous metrics
    specifically: with no noise, EVERY game's probability moves toward the
    truth, so Brier/log-loss improvement is non-negative for every single
    game by construction -- a degenerate, always-detectable signal that
    would make a continuous-metric power curve meaningless. Real per-game
    noise (uncorrelated with the outcome) makes some individual games worse
    under the candidate even though the average is better, which is what
    any real model's edge actually looks like game to game.

    If ``probability_noise`` alone already over/undershoots
    ``target_accuracy_delta`` at ``delta=0``, the achieved delta will differ
    from the target by that amount (reported, not silently forced) -- this
    function only ever pushes with a non-negative ``delta``, it cannot undo
    noise that already moved accuracy past the target.

    This is a genuine, outcome-correlated synthetic effect -- a stand-in for
    "a model with real predictive skill of a known size" -- constructed for
    validating the evaluator's detection power, not a claim about any real
    feature. Returns ``(candidate_probability, achieved_accuracy_delta,
    delta)``.
    """

    if target_accuracy_delta < 0:
        raise ValueError("target_accuracy_delta must be non-negative; see plant_null_candidate")
    baseline_array = np.asarray(baseline_probability, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    n_games = len(baseline_array)
    if n_games == 0:
        raise ValueError("baseline_probability must be non-empty")
    noise_array = (
        np.zeros(n_games)
        if probability_noise is None
        else np.asarray(probability_noise, dtype=float)
    )
    if noise_array.shape != baseline_array.shape:
        raise ValueError("probability_noise must have the same shape as baseline_probability")

    baseline_pick = (baseline_array >= 0.5).astype(float)
    baseline_accuracy = float((baseline_pick == actual_array).mean())
    pre_shift = baseline_array + noise_array
    pre_pick = (pre_shift >= 0.5).astype(float)
    already_correct = pre_pick == actual_array
    direction = np.where(actual_array >= 0.5, 1.0, -1.0)
    distance_to_flip = np.where(
        already_correct,
        np.inf,
        np.where(direction > 0.0, 0.5 - pre_shift, pre_shift - 0.5),
    )
    accuracy_at_zero_delta = float((pre_pick == actual_array).mean())
    remaining_target = target_accuracy_delta - (accuracy_at_zero_delta - baseline_accuracy)
    max_flips = int((~already_correct).sum())
    target_flips = min(max_flips, max(0, round(remaining_target * n_games)))
    if target_flips == 0:
        delta = 0.0
    else:
        ordered = np.sort(distance_to_flip)
        delta = float(ordered[target_flips - 1]) + 1e-9

    candidate = np.clip(pre_shift + delta * direction, probability_floor, 1.0 - probability_floor)
    candidate_pick = (candidate >= 0.5).astype(float)
    achieved = float((candidate_pick == actual_array).mean() - baseline_accuracy)
    return candidate, achieved, delta


def plant_null_candidate(
    baseline_probability: np.ndarray,
    actual: np.ndarray,
    *,
    magnitude: float,
    seed: int,
    probability_noise: np.ndarray | None = None,
    probability_floor: float = 1e-6,
) -> np.ndarray:
    """Build a genuine-null candidate: same shift SIZE, an unrelated direction.

    The candidate is ``baseline + probability_noise + magnitude *
    direction``, where ``direction`` points toward a PERMUTED copy of
    ``actual`` instead of the real outcome -- same per-game shift structure
    as ``plant_accuracy_effect``, zero expected correlation with the true
    label. ``magnitude`` is applied directly (unconditionally, to every
    game) rather than solved for via ``plant_accuracy_effect``'s
    target-seeking search: that search can legitimately land on ``delta=0``
    when noise alone already lands close to the (tiny) null target relative
    to the shuffled labels, which would make the "null" arm collapse to
    ``clip(baseline + noise)`` -- identical across every seed sharing the
    same noise draw, and no longer a fresh independent null realization.
    Applying ``magnitude`` directly avoids that collapse, so different
    seeds always produce genuinely different null datasets.
    """

    baseline_array = np.asarray(baseline_probability, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    noise_array = (
        np.zeros_like(baseline_array)
        if probability_noise is None
        else np.asarray(probability_noise, dtype=float)
    )
    rng = np.random.default_rng(seed)
    shuffled_actual = rng.permutation(actual_array)
    direction = np.where(shuffled_actual >= 0.5, 1.0, -1.0)
    result: np.ndarray = np.clip(
        baseline_array + noise_array + magnitude * direction,
        probability_floor,
        1.0 - probability_floor,
    )
    return result


def screening_ladder_decision(
    comparisons: pd.DataFrame,
    *,
    screen_metric: str = "brier_improvement",
    screen_probability_threshold: float = 0.75,
    confirm_metric: str = "accuracy_improvement",
) -> pd.DataFrame:
    """Apply a predeclared screen-then-confirm rule to a paired-comparison table.

    ``comparisons`` is the output of ``experiments.paired_feature_comparisons``
    or ``covariate_adjusted_paired_comparisons().comparisons`` -- one row per
    (candidate, metric). For each candidate, ``spend_confirmation_window`` is
    True when its ``screen_metric`` row's ``probability_positive`` clears
    ``screen_probability_threshold``. This function only ever recommends
    WHERE to spend a scarce (e.g. NFL rotation-registry) confirmation window;
    it never substitutes the continuous-metric verdict for the accuracy
    verdict the pool actually grades -- ``confirm_metric``'s own numbers are
    passed through unchanged, decided by nothing here.
    """

    required = {"candidate_feature_set", "metric", "probability_positive"}
    missing = sorted(required.difference(comparisons.columns))
    if missing:
        raise ValueError(f"comparisons is missing columns: {', '.join(missing)}")

    screen = comparisons.loc[comparisons["metric"].eq(screen_metric)].set_index(
        "candidate_feature_set"
    )
    confirm = comparisons.loc[comparisons["metric"].eq(confirm_metric)].set_index(
        "candidate_feature_set"
    )
    rows: list[dict[str, Any]] = []
    for candidate_name in sorted(set(screen.index) & set(confirm.index)):
        screen_row = screen.loc[candidate_name]
        confirm_row = confirm.loc[candidate_name]
        passes = bool(screen_row["probability_positive"] >= screen_probability_threshold)
        rows.append(
            {
                "candidate_feature_set": candidate_name,
                "screen_metric": screen_metric,
                "screen_probability_positive": float(screen_row["probability_positive"]),
                "screen_threshold": screen_probability_threshold,
                "spend_confirmation_window": passes,
                "confirm_metric": confirm_metric,
                "confirm_estimate": float(confirm_row.get("estimate", np.nan)),
                "confirm_probability_positive": float(confirm_row["probability_positive"]),
            }
        )
    return pd.DataFrame(rows)


def required_sample_size(
    power_by_n: pd.DataFrame,
    *,
    n_column: str = "n_games",
    power_column: str = "power",
    target_power: float = 0.80,
) -> float | None:
    """Linearly interpolate the smallest ``n_column`` value reaching ``target_power``.

    ``power_by_n`` must be sorted (or is sorted here) by ``n_column``
    ascending. Returns ``None`` if even the largest available sample size
    falls short of ``target_power`` -- that is reported as a lower bound on
    the answer, never guessed by extrapolation.
    """

    ordered = power_by_n.sort_values(n_column).reset_index(drop=True)
    if ordered.empty:
        raise ValueError("power_by_n must have at least one row")
    powers = ordered[power_column].to_numpy(dtype=float)
    sizes = ordered[n_column].to_numpy(dtype=float)
    if powers[0] >= target_power:
        return float(sizes[0])
    for index in range(1, len(ordered)):
        if powers[index] >= target_power:
            low_power, high_power = powers[index - 1], powers[index]
            low_size, high_size = sizes[index - 1], sizes[index]
            if high_power == low_power:
                return float(high_size)
            fraction = (target_power - low_power) / (high_power - low_power)
            return float(low_size + fraction * (high_size - low_size))
    return None
