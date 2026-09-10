from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.experiments import PairedBlock

DEFAULT_ALPHA = 0.05

ANYTIME_METRICS: tuple[str, ...] = ("accuracy_improvement", "brier_improvement")

DEFAULT_TARGET_GAMES = 800

DEFAULT_INTRACLASS_CORRELATION = 0.0
WORST_CASE_INTRACLASS_CORRELATION = 1.0

_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"feature_set", "game_id", "season", "week", "home_cover", "home_cover_probability"}
)


def _validate_predictions_columns(predictions: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing paired columns: {', '.join(missing)}")


def _paired_rows(
    predictions: pd.DataFrame, baseline_feature_set: str, candidate_feature_set: str
) -> pd.DataFrame:
    columns = ["game_id", "season", "week", "home_cover", "home_cover_probability"]
    baseline = predictions.loc[predictions["feature_set"].eq(baseline_feature_set), columns]
    candidate = predictions.loc[predictions["feature_set"].eq(candidate_feature_set), columns]
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
        raise ValueError(f"No paired completed games for {candidate_feature_set}")
    for column in ("season", "week", "home_cover"):
        if not paired[f"{column}_baseline"].equals(paired[f"{column}_candidate"]):
            raise ValueError(f"Paired {column} values differ for {candidate_feature_set}")
    return paired


def _row_improvements(paired: pd.DataFrame) -> pd.DataFrame:
    actual = paired["home_cover_baseline"].to_numpy(dtype=float)
    baseline_p = paired["home_cover_probability_baseline"].to_numpy(dtype=float)
    candidate_p = paired["home_cover_probability_candidate"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "accuracy_improvement": ((candidate_p >= 0.5) == actual).astype(float)
            - ((baseline_p >= 0.5) == actual).astype(float),
            "brier_improvement": np.square(baseline_p - actual) - np.square(candidate_p - actual),
        },
        index=paired.index,
    )


def _ordered_blocks(
    paired: pd.DataFrame, values: pd.Series, block: PairedBlock
) -> tuple[pd.DataFrame, list[npt.NDArray[np.float64]]]:

    frame = paired[["season_baseline", "week_baseline"]].copy()
    frame.columns = pd.Index(["season", "week"])
    frame["value"] = values.to_numpy(dtype=float)
    group_columns = ["season", "week"] if block == "week" else ["season"]
    grouped = frame.groupby(group_columns, sort=True)
    identity = grouped.size().reset_index()[group_columns]
    block_arrays = [group["value"].to_numpy(dtype=np.float64) for _, group in grouped]
    return identity, block_arrays


def _per_block_variance(
    block_size: npt.NDArray[np.float64] | float,
    per_game_variance_proxy: float,
    intraclass_correlation: float,
) -> Any:

    return (
        block_size * per_game_variance_proxy * (1.0 + (block_size - 1.0) * intraclass_correlation)
    )


def default_prior_variance(
    average_block_size: float,
    *,
    target_games: int = DEFAULT_TARGET_GAMES,
    per_game_variance_proxy: float = 1.0,
    intraclass_correlation: float = 0.0,
) -> float:

    if average_block_size <= 0.0:
        raise ValueError("average_block_size must be positive")
    if target_games <= 0:
        raise ValueError("target_games must be positive")
    reference_variance_process = float(target_games) * float(
        _per_block_variance(average_block_size, per_game_variance_proxy, intraclass_correlation)
    )
    return 1.0 / reference_variance_process


def confidence_sequence_from_block_stats(
    block_sizes: npt.NDArray[np.float64],
    block_sums: npt.NDArray[np.float64],
    *,
    alpha: float = DEFAULT_ALPHA,
    prior_variance: float,
    per_game_variance_proxy: float = 1.0,
    intraclass_correlation: float = 0.0,
) -> pd.DataFrame:

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if prior_variance <= 0.0:
        raise ValueError("prior_variance must be positive")
    if len(block_sizes) == 0:
        raise ValueError("At least one block is required")
    if len(block_sizes) != len(block_sums):
        raise ValueError("block_sizes and block_sums must have the same length")
    if np.any(block_sizes <= 0):
        raise ValueError("Every block must contain at least one game")
    if per_game_variance_proxy <= 0.0:
        raise ValueError("per_game_variance_proxy must be positive")
    if not 0.0 <= intraclass_correlation <= 1.0:
        raise ValueError("intraclass_correlation must be between 0 and 1")

    cumulative_games = np.cumsum(block_sizes)
    cumulative_sum = np.cumsum(block_sums)
    cumulative_variance_process = np.cumsum(
        _per_block_variance(block_sizes, per_game_variance_proxy, intraclass_correlation)
    )

    denominator = 1.0 + prior_variance * cumulative_variance_process
    log_e_value = -0.5 * np.log(denominator) + (prior_variance * np.square(cumulative_sum)) / (
        2.0 * denominator
    )
    with np.errstate(over="ignore"):
        e_value = np.exp(np.clip(log_e_value, a_min=None, a_max=700.0))
    radius = (1.0 / cumulative_games) * np.sqrt(
        (2.0 * denominator / prior_variance) * np.log((1.0 / alpha) * np.sqrt(denominator))
    )
    mean = cumulative_sum / cumulative_games
    lower = mean - radius
    upper = mean + radius
    excludes_zero = (lower > 0.0) | (upper < 0.0)

    return pd.DataFrame(
        {
            "look": np.arange(1, len(block_sizes) + 1, dtype=int),
            "block_games": block_sizes.astype(int),
            "cumulative_games": cumulative_games.astype(int),
            "cumulative_variance_process": cumulative_variance_process,
            "cumulative_mean": mean,
            "log_e_value": log_e_value,
            "e_value": e_value,
            "lower": lower,
            "upper": upper,
            "excludes_zero": excludes_zero,
        }
    )


def paired_anytime_comparisons(
    predictions: pd.DataFrame,
    *,
    baseline_feature_set: str,
    metric: str = "accuracy_improvement",
    block: PairedBlock = "week",
    alpha: float = DEFAULT_ALPHA,
    prior_variance: float | None = None,
    target_games: int = DEFAULT_TARGET_GAMES,
    per_game_variance_proxy: float = 1.0,
    intraclass_correlation: float = 0.0,
) -> pd.DataFrame:

    if metric not in ANYTIME_METRICS:
        raise ValueError(
            f"Unsupported anytime metric {metric!r}; expected one of {ANYTIME_METRICS}. "
            "log_loss_improvement is unbounded and has no fixed sub-Gaussian proxy, so it "
            "cannot back this confidence sequence without a different (heavier-tailed) "
            "construction -- see the module docstring."
        )
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    _validate_predictions_columns(predictions)
    feature_sets = set(predictions["feature_set"].astype(str))
    if baseline_feature_set not in feature_sets:
        raise ValueError(f"Unknown paired baseline feature set: {baseline_feature_set}")

    traces: list[pd.DataFrame] = []
    for candidate_name in sorted(feature_sets.difference((baseline_feature_set,))):
        paired = _paired_rows(predictions, baseline_feature_set, candidate_name)
        values = _row_improvements(paired)[metric]
        identity, block_arrays = _ordered_blocks(paired, values, block)
        block_sizes = np.array([len(b) for b in block_arrays], dtype=np.float64)
        block_sums = np.array([float(b.sum()) for b in block_arrays], dtype=np.float64)
        average_block_size = float(np.mean(block_sizes))
        measured_icc_diagnostic = (
            anova_intraclass_correlation(block_arrays) if len(block_arrays) >= 2 else float("nan")
        )

        rho = (
            prior_variance
            if prior_variance is not None
            else default_prior_variance(
                average_block_size,
                target_games=target_games,
                per_game_variance_proxy=per_game_variance_proxy,
                intraclass_correlation=intraclass_correlation,
            )
        )
        trace = confidence_sequence_from_block_stats(
            block_sizes,
            block_sums,
            alpha=alpha,
            prior_variance=rho,
            per_game_variance_proxy=per_game_variance_proxy,
            intraclass_correlation=intraclass_correlation,
        )
        trace.insert(0, "candidate_feature_set", candidate_name)
        trace.insert(0, "baseline_feature_set", baseline_feature_set)
        trace["metric"] = metric
        trace["block"] = block
        trace["prior_variance"] = rho
        trace["per_game_variance_proxy"] = per_game_variance_proxy
        trace["intraclass_correlation"] = intraclass_correlation
        trace["measured_icc_diagnostic"] = measured_icc_diagnostic
        trace["season"] = identity["season"].to_numpy()
        trace["week"] = identity["week"].to_numpy() if block == "week" else None
        traces.append(trace)
    return pd.concat(traces, ignore_index=True)


def anytime_summary(trace: pd.DataFrame) -> pd.DataFrame:

    group_columns = ["baseline_feature_set", "candidate_feature_set", "metric", "block"]
    rows: list[dict[str, Any]] = []
    for keys, group in trace.groupby(group_columns, sort=True):
        ordered = group.sort_values("look")
        final = ordered.iloc[-1]
        hits = ordered.loc[ordered["excludes_zero"]]
        identity = dict(zip(group_columns, keys, strict=True))
        rows.append(
            {
                **identity,
                "alpha": DEFAULT_ALPHA,
                "prior_variance": float(final["prior_variance"]),
                "intraclass_correlation": float(final["intraclass_correlation"]),
                "measured_icc_diagnostic": float(final["measured_icc_diagnostic"]),
                "looks": int(final["look"]),
                "games": int(final["cumulative_games"]),
                "final_estimate": float(final["cumulative_mean"]),
                "final_lower": float(final["lower"]),
                "final_upper": float(final["upper"]),
                "final_e_value": float(final["e_value"]),
                "final_log_e_value": float(final["log_e_value"]),
                "final_excludes_zero": bool(final["excludes_zero"]),
                "first_excluding_zero_look": (
                    int(hits.iloc[0]["look"]) if not hits.empty else None
                ),
                "first_excluding_zero_games": (
                    int(hits.iloc[0]["cumulative_games"]) if not hits.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


def simulate_block_sequence(
    rng: np.random.Generator,
    block_sizes: Sequence[int],
    *,
    true_mean: float,
    total_variance: float = 0.545,
    intraclass_correlation: float = 0.10,
) -> list[npt.NDArray[np.float64]]:

    if total_variance <= 0.0:
        raise ValueError("total_variance must be positive")
    if not 0.0 <= intraclass_correlation <= 1.0:
        raise ValueError("intraclass_correlation must be between 0 and 1")
    shock_scale = float(np.sqrt(intraclass_correlation * total_variance))
    noise_scale = float(np.sqrt((1.0 - intraclass_correlation) * total_variance))
    blocks: list[npt.NDArray[np.float64]] = []
    for size in block_sizes:
        if size <= 0:
            raise ValueError("block sizes must be positive")
        shock = rng.normal(0.0, shock_scale) if shock_scale > 0.0 else 0.0
        noise = rng.normal(0.0, noise_scale, size=size) if noise_scale > 0.0 else np.zeros(size)
        blocks.append(np.clip(true_mean + shock + noise, -1.0, 1.0))
    return blocks


def block_bootstrap_ci_fast(
    block_sizes: npt.NDArray[np.float64],
    block_sums: npt.NDArray[np.float64],
    *,
    samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:

    if samples < 1:
        raise ValueError("samples must be at least 1")
    n_blocks = len(block_sizes)
    if n_blocks == 0:
        raise ValueError("At least one block is required")
    draws = rng.integers(0, n_blocks, size=(samples, n_blocks))
    pooled_sum = block_sums[draws].sum(axis=1)
    pooled_count = block_sizes[draws].sum(axis=1)
    means = pooled_sum / pooled_count
    tail = alpha / 2.0
    lower = float(np.quantile(means, tail))
    upper = float(np.quantile(means, 1.0 - tail))
    return lower, upper


@dataclass(frozen=True)
class PeekingTrialResult:
    cs_excluded: bool
    cs_first_look: int | None
    cs_first_games: int | None
    fixed_sample_excluded: bool
    fixed_sample_first_look: int | None
    fixed_sample_first_games: int | None


def anova_intraclass_correlation(
    block_values: Sequence[npt.NDArray[np.float64]],
) -> float:

    blocks = [np.asarray(b, dtype=np.float64) for b in block_values]
    if len(blocks) < 2:
        raise ValueError("At least two blocks are required to estimate ICC")
    sizes = np.array([len(b) for b in blocks], dtype=np.float64)
    if np.any(sizes < 1):
        raise ValueError("Every block must contain at least one observation")
    k = len(blocks)
    n_total = float(sizes.sum())
    grand_mean = float(np.concatenate(blocks).mean())
    block_means = np.array([float(b.mean()) for b in blocks], dtype=np.float64)

    ssb = float(np.sum(sizes * (block_means - grand_mean) ** 2))
    msb = ssb / (k - 1)
    ssw = float(sum(np.sum((b - m) ** 2) for b, m in zip(blocks, block_means, strict=True)))
    degrees_within = n_total - k
    if degrees_within <= 0:
        raise ValueError("Not enough within-block degrees of freedom to estimate ICC")
    msw = ssw / degrees_within

    n0 = (n_total - float(np.sum(sizes**2)) / n_total) / (k - 1)
    denominator = msb + (n0 - 1.0) * msw
    if denominator == 0.0:
        return 0.0
    return (msb - msw) / denominator


def bootstrap_intraclass_correlation(
    block_values: Sequence[npt.NDArray[np.float64]],
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260818,
) -> dict[str, float]:

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    point_estimate = anova_intraclass_correlation(block_values)
    blocks = list(block_values)
    n_blocks = len(blocks)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = generator.integers(0, n_blocks, size=n_blocks)
        draws[sample_index] = anova_intraclass_correlation([blocks[i] for i in selected])
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": point_estimate,
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
        "confidence": confidence,
        "samples": samples,
        "n_blocks": n_blocks,
    }
