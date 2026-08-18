"""Group-wise ridge penalty screen on the free CFB benchmark (12,206 games).

Rotation rule 8: the CFB benchmark is unreserved, so this costs no NFL
confirmation window. It is the same instrument that resolved MOD-06 and MOD-16.

The question. The active model puts one global ridge penalty (alpha 10) on every
feature block, which assumes every block has the same signal-to-noise. That is
false by construction: the market columns are a near-sufficient statistic for
team quality, while the team-state columns are noisy rolling estimates. This
screen asks whether reallocating penalty across blocks -- holding the AVERAGE
penalty fixed -- moves forced-pick accuracy.

The grid is predeclared in ``PENALTY_GRID`` below and is deliberately symmetric:
for every "market light / state heavy" arm there is an equal and opposite arm.
If the two directions help equally, the movement is noise, not signal.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/groupwise_ridge_screen.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_DISTRIBUTION_FRACTION,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_REGRESSOR,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
    cfb_walk_forward_benchmark,
)
from nfl_ats.cfb_features import (
    CFB_CONTEXT_FEATURES,
    CFB_EXPERIENCE_FEATURES,
    CFB_MARKET_FEATURES,
    CFB_MODEL_FEATURE_COLUMNS,
    CFB_STATE_METRICS,
)
from nfl_ats.margin import MarginModel, column_penalty_multipliers, make_margin_estimator

# ---------------------------------------------------------------------------
# Blocks and the predeclared grid
# ---------------------------------------------------------------------------

# The CFB feature contract's own declared blocks. The team-state columns split
# into offense and defense along CFB_STATE_METRICS, whose first four entries are
# offensive and last four defensive.
_OFFENSE_METRICS = CFB_STATE_METRICS[:4]
_DEFENSE_METRICS = CFB_STATE_METRICS[4:]


def cfb_feature_groups(columns: Sequence[str] = CFB_MODEL_FEATURE_COLUMNS) -> tuple[str, ...]:
    """Label each CFB feature column with its block."""

    lookup: dict[str, str] = {}
    for column in CFB_MARKET_FEATURES:
        lookup[column] = "market"
    for column in CFB_CONTEXT_FEATURES:
        lookup[column] = "context"
    for column in CFB_EXPERIENCE_FEATURES:
        lookup[column] = "experience"
    for metric in _OFFENSE_METRICS:
        for side in ("home", "away", "diff"):
            lookup[f"{side}_{metric}"] = "offense"
    for metric in _DEFENSE_METRICS:
        for side in ("home", "away", "diff"):
            lookup[f"{side}_{metric}"] = "defense"
    missing = sorted(set(columns).difference(lookup))
    if missing:
        raise ValueError(f"No CFB block covers: {', '.join(missing)}")
    return tuple(lookup[column] for column in columns)


# PREDECLARED, fixed before any result was read. Each entry is a per-block
# penalty multiplier; multipliers are renormalised to a count-weighted geometric
# mean of 1, so every arm carries the SAME average penalty as `uniform` and only
# the allocation across blocks differs. That keeps this axis orthogonal to the
# global-alpha axis MOD-06 already swept and closed.
PENALTY_GRID: dict[str, dict[str, float]] = {
    # Baseline: exactly the frozen benchmark.
    "uniform": {},
    # The lead's hypothesis: light on market, heavy on the noisy state blocks.
    "market_light_3": {"market": 1.0 / 3.0, "offense": 3.0, "defense": 3.0},
    "market_light_10": {"market": 0.1, "offense": 10.0, "defense": 10.0},
    # Equal and opposite falsification controls.
    "market_heavy_3": {"market": 3.0, "offense": 1.0 / 3.0, "defense": 1.0 / 3.0},
    "market_heavy_10": {"market": 10.0, "offense": 0.1, "defense": 0.1},
    # State moved against everything else, market held with context/experience.
    "state_heavy_3": {"offense": 3.0, "defense": 3.0},
    "state_light_3": {"offense": 1.0 / 3.0, "defense": 1.0 / 3.0},
}

BASELINE_ARM = "uniform"

# Global penalty levels the block grid is run at.
#
# 10.0 is the frozen benchmark. The other two are DERIVED, not chosen, from the
# measured eigenspectrum of the standardised design: ridge shrinks the principal
# direction with eigenvalue d by alpha/(d + alpha). At the final training cut
# (n = 11,738) only 41 of the 63 transformed directions are non-null -- the
# `diff = home - away` identity makes the design rank-deficient -- and their
# eigenvalues have median 6.02e3. At alpha = 10 the MEDIAN direction is
# therefore shrunk by 0.17%: the penalty is inert, so reallocating it across
# blocks is bounded to be inert too, whatever the ratio. alpha = 1e3 puts
# median shrinkage at 0.14 and alpha = 1e4 at 0.62, bracketing the range where
# a penalty has any leverage at all. Testing block ratios only at an inert
# global level would be a rigged screen.
GLOBAL_ALPHAS: tuple[float, ...] = (CFB_BENCHMARK_RIDGE_ALPHA, 1_000.0, 10_000.0)

BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260817
FEATURES_PATH = Path("data/processed/cfb_game_features.parquet")
OUTPUT_DIR = Path("artifacts/groupwise_ridge")

_PASSTHROUGH = (
    "game_id",
    "season",
    "week",
    "gameday",
    "spread_line",
    "result",
    "ats_margin",
    "home_cover",
)


def penalty_configuration(name: str) -> tuple[dict[str, float] | None, dict[str, float]]:
    """Return (per-column multipliers, normalised per-block multipliers)."""

    blocks = PENALTY_GRID[name]
    columns = list(CFB_MODEL_FEATURE_COLUMNS)
    groups = cfb_feature_groups(columns)
    per_column = column_penalty_multipliers(columns, groups, blocks, normalize=True)
    per_block: dict[str, float] = {}
    for column, group in zip(columns, groups, strict=True):
        per_block.setdefault(group, per_column[column])
    if name == BASELINE_ARM:
        # A uniform grid normalises to all-ones, which is bit-identical to the
        # frozen single-penalty path; pass None so that is literally true.
        return None, per_block
    return per_column, per_block


# ---------------------------------------------------------------------------
# Walk-forward (mirrors cfb_benchmark.cfb_walk_forward_benchmark exactly)
# ---------------------------------------------------------------------------


def _fit_residual_model(
    training: pd.DataFrame,
    column_penalties: Mapping[str, float] | None,
    ridge_alpha: float,
) -> MarginModel:
    frame = training.loc[pd.to_numeric(training["ats_margin"], errors="coerce").notna()].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    distribution_rows = int(len(frame) * CFB_BENCHMARK_DISTRIBUTION_FRACTION)
    split = len(frame) - distribution_rows
    columns = list(CFB_MODEL_FEATURE_COLUMNS)
    target = pd.to_numeric(frame["ats_margin"], errors="raise").to_numpy(dtype=float)

    temporary = make_margin_estimator(
        CFB_BENCHMARK_REGRESSOR,
        42,
        ridge_alpha=ridge_alpha,
        column_penalties=column_penalties,
    )
    temporary.fit(frame.iloc[:split].loc[:, columns], target[:split])
    prediction = np.asarray(temporary.predict(frame.iloc[split:].loc[:, columns]), dtype=float)
    residuals = np.asarray(target[split:] - prediction, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]

    estimator = make_margin_estimator(
        CFB_BENCHMARK_REGRESSOR,
        42,
        ridge_alpha=ridge_alpha,
        column_penalties=column_penalties,
    )
    estimator.fit(frame.loc[:, columns], target)
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name=CFB_BENCHMARK_REGRESSOR,
        ridge_alpha=ridge_alpha,
        target="market_residual",
        feature_columns=tuple(columns),
        training_rows=len(frame),
        distribution_rows=len(residuals),
        training_max_gameday=frame["gameday"].max().date().isoformat(),
        column_penalties=None if column_penalties is None else dict(column_penalties),
    )


def run_arm(completed: pd.DataFrame, name: str, ridge_alpha: float) -> pd.DataFrame:
    """Score every eligible CFB week for one penalty configuration."""

    column_penalties, _ = penalty_configuration(name)
    test = completed.loc[
        completed["season"].between(CFB_BENCHMARK_START_SEASON, CFB_BENCHMARK_END_SEASON)
    ]
    batches: list[pd.DataFrame] = []
    for _, weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < CFB_BENCHMARK_MIN_TRAIN_GAMES:
            continue
        model = _fit_residual_model(training, column_penalties, ridge_alpha)
        batch = weekly_games.loc[:, list(_PASSTHROUGH)].copy()
        forecasts = model.predict(weekly_games)
        for column in ("home_cover_probability", "predicted_market_residual", "predicted_margin"):
            batch[column] = forecasts[column]
        batch["arm"] = name
        batch["ridge_alpha"] = ridge_alpha
        batches.append(batch)
    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    return predictions.sort_values(["gameday", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics and paired blocked bootstrap
# ---------------------------------------------------------------------------


def _components(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row additive pieces of accuracy, Brier, and log loss."""

    evaluated = frame.loc[frame["home_cover"].notna()].copy()
    actual = evaluated["home_cover"].astype(int).to_numpy()
    probability = evaluated["home_cover_probability"].to_numpy(dtype=float)
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    evaluated["_correct"] = ((probability >= 0.5).astype(int) == actual).astype(float)
    evaluated["_brier"] = np.square(probability - actual)
    evaluated["_logloss"] = -(actual * np.log(clipped) + (1 - actual) * np.log(1.0 - clipped))
    evaluated["_n"] = 1.0
    return evaluated


def arm_summary(frame: pd.DataFrame, window: str) -> dict[str, Any]:
    subset = frame if window == "all" else frame.loc[frame["evaluation_window"].eq(window)]
    parts = _components(subset)
    return {
        "arm": str(frame["arm"].iloc[0]),
        "ridge_alpha": float(frame["ridge_alpha"].iloc[0]),
        "evaluation_window": window,
        "games": len(subset),
        "evaluated": len(parts),
        "accuracy": float(parts["_correct"].mean()),
        "brier": float(parts["_brier"].mean()),
        "log_loss": float(parts["_logloss"].mean()),
        "mean_abs_residual": float(subset["predicted_market_residual"].abs().mean()),
    }


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    window: str,
    block: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Blocked paired bootstrap of candidate-minus-baseline on the same games."""

    base = _components(
        baseline if window == "all" else baseline.loc[baseline["evaluation_window"].eq(window)]
    )
    cand = _components(
        candidate if window == "all" else candidate.loc[candidate["evaluation_window"].eq(window)]
    )
    merged = base.merge(cand, on="game_id", suffixes=("_b", "_c"), validate="one_to_one")
    if len(merged) != len(base) or len(merged) != len(cand):
        raise ValueError("Arms do not cover the same games")

    group_columns = ["season_b", "week_b"] if block == "week" else ["season_b"]
    indices = list(merged.groupby(group_columns, sort=False).indices.values())
    stats = {
        "accuracy": (merged["_correct_c"].to_numpy(), merged["_correct_b"].to_numpy()),
        "brier": (merged["_brier_b"].to_numpy(), merged["_brier_c"].to_numpy()),
        "log_loss": (merged["_logloss_b"].to_numpy(), merged["_logloss_c"].to_numpy()),
    }
    generator = np.random.default_rng(seed)
    draws = {metric: np.empty(samples, dtype=float) for metric in stats}
    for draw in range(samples):
        selected = generator.integers(0, len(indices), size=len(indices))
        positions = np.concatenate([indices[index] for index in selected])
        for metric, (better, worse) in stats.items():
            draws[metric][draw] = float(better[positions].mean() - worse[positions].mean())

    rows: list[dict[str, Any]] = []
    for metric, (better, worse) in stats.items():
        sample = draws[metric]
        rows.append(
            {
                "arm": str(candidate["arm"].iloc[0]),
                "ridge_alpha": float(candidate["ridge_alpha"].iloc[0]),
                "baseline_arm": str(baseline["arm"].iloc[0]),
                "baseline_ridge_alpha": float(baseline["ridge_alpha"].iloc[0]),
                "evaluation_window": window,
                "block": block,
                "metric": metric,
                # Oriented so positive always means the candidate is better.
                "improvement": float(better.mean() - worse.mean()),
                "lower": float(np.quantile(sample, 0.025)),
                "upper": float(np.quantile(sample, 0.975)),
                "probability_positive": float(np.mean(sample > 0.0)),
                "games": len(merged),
                "blocks": len(indices),
                "samples": int(samples),
            }
        )
    return rows


def pick_flips(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    """The diagnostic that separates this from MOD-06: do any picks move at all?"""

    merged = baseline.merge(candidate, on="game_id", suffixes=("_b", "_c"), validate="one_to_one")
    scored_b = merged["home_cover_probability_b"].to_numpy(dtype=float) >= 0.5
    scored_c = merged["home_cover_probability_c"].to_numpy(dtype=float) >= 0.5
    sign_b = np.sign(merged["predicted_market_residual_b"].to_numpy(dtype=float))
    sign_c = np.sign(merged["predicted_market_residual_c"].to_numpy(dtype=float))
    flipped = scored_b != scored_c
    evaluated = merged["home_cover_b"].notna().to_numpy()
    actual = merged["home_cover_b"].to_numpy()
    both = flipped & evaluated
    candidate_right = int(np.nansum((scored_c[both] == actual[both].astype(bool)).astype(int)))
    return {
        "arm": str(candidate["arm"].iloc[0]),
        "ridge_alpha": float(candidate["ridge_alpha"].iloc[0]),
        "baseline_arm": str(baseline["arm"].iloc[0]),
        "baseline_ridge_alpha": float(baseline["ridge_alpha"].iloc[0]),
        "games": len(merged),
        "scored_pick_flips": int(flipped.sum()),
        "scored_pick_flip_rate": float(flipped.mean()),
        "residual_sign_flips": int((sign_b != sign_c).sum()),
        "flipped_and_evaluated": int(both.sum()),
        "candidate_correct_on_flips": candidate_right,
        "baseline_correct_on_flips": int(both.sum()) - candidate_right,
        "spearman_residual_correlation": float(
            merged["predicted_market_residual_b"].corr(
                merged["predicted_market_residual_c"], method="spearman"
            )
        ),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Run only the baseline and the strongest arm at the frozen alpha.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    features = pd.read_parquet(FEATURES_PATH)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    completed = features.loc[
        pd.to_numeric(features["result"], errors="coerce").notna()
        & pd.to_numeric(features["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    if args.structural_only:
        arms = [BASELINE_ARM, "market_light_10"]
        alphas: tuple[float, ...] = (CFB_BENCHMARK_RIDGE_ALPHA,)
    else:
        arms = list(PENALTY_GRID)
        alphas = GLOBAL_ALPHAS

    predictions: dict[tuple[str, float], pd.DataFrame] = {}
    for alpha in alphas:
        for name in arms:
            print(f"[arm] alpha={alpha:g} {name} ...", flush=True)
            predictions[(name, alpha)] = run_arm(completed, name, alpha)

    # Correctness gate: the uniform arm at the frozen alpha must reproduce the
    # frozen benchmark exactly. Anything else means the opt-in path leaked.
    frozen = cfb_walk_forward_benchmark(features)
    frozen_residual = (
        frozen.predictions.loc[frozen.predictions["method"].eq("market_residual")]
        .sort_values("game_id")
        .reset_index(drop=True)
    )
    mine = (
        predictions[(BASELINE_ARM, CFB_BENCHMARK_RIDGE_ALPHA)]
        .sort_values("game_id")
        .reset_index(drop=True)
    )
    if not frozen_residual["game_id"].equals(mine["game_id"]):
        raise AssertionError("Uniform arm does not cover the frozen benchmark's games")
    delta = float(
        np.max(
            np.abs(
                frozen_residual["predicted_market_residual"].to_numpy(dtype=float)
                - mine["predicted_market_residual"].to_numpy(dtype=float)
            )
        )
    )
    print(f"[gate] uniform arm vs frozen benchmark, max |delta residual| = {delta:.3e}")
    if delta > 0.0:
        raise AssertionError("Uniform arm is not bit-identical to the frozen benchmark")

    summaries = [
        arm_summary(predictions[(name, alpha)], window)
        for alpha in alphas
        for name in arms
        for window in ("clean_core", "all")
    ]
    # Every comparison is against `uniform` AT THE SAME GLOBAL ALPHA, which is
    # what isolates the block allocation from the global penalty level.
    flips = [
        pick_flips(predictions[(BASELINE_ARM, alpha)], predictions[(name, alpha)])
        for alpha in alphas
        for name in arms
    ]
    # Plus one cross-alpha row per level, so the global-penalty axis (MOD-06's)
    # is visible beside the block axis rather than confounded with it.
    flips.extend(
        pick_flips(
            predictions[(BASELINE_ARM, CFB_BENCHMARK_RIDGE_ALPHA)],
            predictions[(BASELINE_ARM, alpha)],
        )
        for alpha in alphas
        if alpha != CFB_BENCHMARK_RIDGE_ALPHA
    )
    comparisons: list[dict[str, Any]] = []
    for alpha in alphas:
        for name in arms:
            if name == BASELINE_ARM:
                continue
            for window in ("clean_core", "all"):
                for block in ("week", "season"):
                    comparisons.extend(
                        paired_bootstrap(
                            predictions[(BASELINE_ARM, alpha)],
                            predictions[(name, alpha)],
                            window=window,
                            block=block,
                        )
                    )
    for alpha in alphas:
        if alpha == CFB_BENCHMARK_RIDGE_ALPHA:
            continue
        comparisons.extend(
            paired_bootstrap(
                predictions[(BASELINE_ARM, CFB_BENCHMARK_RIDGE_ALPHA)],
                predictions[(BASELINE_ARM, alpha)],
                window="clean_core",
                block="week",
            )
        )

    summary_frame = pd.DataFrame(summaries)
    flip_frame = pd.DataFrame(flips)
    comparison_frame = pd.DataFrame(comparisons)
    pd.set_option("display.width", 200)
    print("\n=== Arm summaries ===")
    print(summary_frame.to_string(index=False))
    print("\n=== Pick flips vs uniform ===")
    print(flip_frame.to_string(index=False))
    print("\n=== Paired blocked comparisons vs uniform ===")
    print(comparison_frame.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output_dir / "arm_summaries.csv", index=False)
    flip_frame.to_csv(args.output_dir / "pick_flips.csv", index=False)
    comparison_frame.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    (args.output_dir / "grid.json").write_text(
        json.dumps(
            {name: penalty_configuration(name)[1] for name in PENALTY_GRID},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
