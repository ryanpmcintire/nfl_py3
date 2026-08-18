"""Empirical validation: does variance reduction actually buy detection power?

This script does NOT touch the rotation registry, does NOT change any pick,
model, or feature profile, and does NOT write anywhere under ``registry/``
or ``artifacts/`` used by other work. It only reads the already-frozen CFB
market-residual benchmark (``cfb_benchmark.cfb_walk_forward_benchmark``,
identical recipe to the standing artifacts) and the cached CFB feature table
(``data/processed/cfb_game_features.parquet``), and writes its own results
under the given ``--output`` directory (defaults to a scratch path, never
``artifacts/``).

Method
------
1. Build the real CFB clean-core baseline arm (``market_residual``,
   2006-2025 walk-forward, frozen recipe) -- real games, real covariates,
   real market probabilities.
2. Plant a KNOWN forced-pick accuracy edge on top of that baseline
   (``variance_reduction.plant_accuracy_effect``): a candidate arm whose
   probability always moves toward the realized outcome by just enough to
   flip exactly the targeted number of games, for magnitudes 0.25/0.5/1.0/2.0
   accuracy points. This is a real, outcome-correlated synthetic effect (a
   stand-in for "a model with known, real predictive skill"), not a claim
   about any actual feature.
3. Also build several genuine-NULL candidate arms
   (``variance_reduction.plant_null_candidate``): the same shift mechanics,
   keyed to a permuted copy of the outcome, so the shift has zero expected
   correlation with the truth.
4. For a grid of evaluation-window sizes (number of week-blocks sampled
   without replacement from the real 2006-2025 walk), repeatedly measure
   whether each of four methods detects the planted effect (a week-blocked
   95% paired-bootstrap interval that excludes zero from below):
   - ``current``: forced-pick accuracy, unadjusted (today's
     ``paired_feature_comparisons``).
   - ``covariate_adjusted``: forced-pick accuracy, CUPED-adjusted
     (``variance_reduction.covariate_adjusted_paired_comparisons``).
   - ``continuous_brier`` / ``continuous_log_loss``: Brier / log-loss
     improvement, unadjusted -- the screening-ladder metric.
   - ``combined``: Brier improvement, CUPED-adjusted (both levers stacked).
5. Convert each method's power curve into a required-sample-size-for-80%-
   power number, report the planted-effect table, the false-positive check
   on the null arms, and a screen -> confirm concordance check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_walk_forward_benchmark,
)
from nfl_ats.experiments import _paired_row_improvements
from nfl_ats.variance_reduction import (
    DEFAULT_CUPED_COVARIATES,
    build_cuped_covariates,
    cuped_adjust,
    fast_block_bootstrap_means,
    plant_accuracy_effect,
    plant_null_candidate,
    required_sample_size,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data/processed/cfb_game_features.parquet"

POSITIVE_MAGNITUDES: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
MAGNITUDE_LABELS: dict[float, str] = {
    0.0025: "0.25pt",
    0.005: "0.50pt",
    0.01: "1.00pt",
    0.02: "2.00pt",
}
NULL_SEEDS: tuple[int, ...] = (901, 902, 903, 904, 905)
# Idiosyncratic per-game noise added to every planted arm's probability
# BEFORE the systematic accuracy-targeting shift. Without it, every game's
# candidate probability moves toward the truth, making brier/log-loss
# improvement non-negative for every single game by construction -- a
# degenerate signal any bootstrap detects at any sample size. Tuned (see
# scripts/variance_planted_effects.py history) so the resulting per-game
# Brier improvement is genuinely two-sided (30-45% of games worse under the
# candidate) while the population-average Brier/log-loss improvement stays
# positive, matching what a real, imperfect-but-net-positive model looks
# like game to game.
PROBABILITY_NOISE_SD = 0.01
NOISE_SEED = 20260818
BLOCK_GRID: tuple[int, ...] = (3, 6, 12, 25, 50, 100, 0)  # 0 means "all blocks"
BOOTSTRAP_SAMPLES = 800
REPLICATIONS_POSITIVE = 400
REPLICATIONS_NULL = 200
DETECTION_TAIL = 0.025  # one-sided lower bound of a two-sided 95% interval
SCREEN_PROBABILITY_THRESHOLD = 0.75
CONCORDANCE_TARGET_BLOCKS = 12  # ~ a realistic, affordable screening window


def _load_baseline_predictions(features_path: Path) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    result = cfb_walk_forward_benchmark(
        features,
        start_season=CFB_BENCHMARK_START_SEASON,
        end_season=CFB_BENCHMARK_END_SEASON,
        min_train_games=CFB_BENCHMARK_MIN_TRAIN_GAMES,
        ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
    )
    predictions = result.predictions
    clean = predictions.loc[
        predictions["evaluation_window"].eq("clean_core")
        & predictions["method"].eq("market_residual")
    ].copy()
    clean = clean.loc[clean["home_cover"].notna() & clean["home_cover_probability"].notna()].copy()
    return clean.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _block_index(paired: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
    grouped_indices = list(paired.groupby(["season", "week"], sort=True).indices.values())
    block_of_row = np.empty(len(paired), dtype=np.int64)
    for block_id, positions in enumerate(grouped_indices):
        block_of_row[positions] = block_id
    return grouped_indices, block_of_row


def _run_replication(
    raw_full: np.ndarray,
    covariate_full: np.ndarray,
    block_game_indices: list[np.ndarray],
    n_blocks_total: int,
    *,
    n_blocks_sample: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if n_blocks_sample <= 0 or n_blocks_sample >= n_blocks_total:
        chosen_blocks = list(range(n_blocks_total))
    else:
        chosen_blocks = list(rng.choice(n_blocks_total, size=n_blocks_sample, replace=False))

    chunks = [block_game_indices[block_id] for block_id in chosen_blocks]
    local_group_of_row = np.concatenate(
        [np.full(len(chunk), local_id, dtype=np.int64) for local_id, chunk in enumerate(chunks)]
    )
    subset_positions = np.concatenate(chunks)
    n_games = len(subset_positions)

    raw_subset = raw_full[subset_positions]
    covariate_subset = covariate_full[subset_positions]
    adjusted_subset, _, _ = cuped_adjust(raw_subset, covariate_subset)
    combined = np.concatenate([raw_subset, adjusted_subset], axis=1)

    draws = fast_block_bootstrap_means(
        combined,
        local_group_of_row,
        len(chunks),
        samples=BOOTSTRAP_SAMPLES,
        seed=seed,
    )
    lower = np.quantile(draws, DETECTION_TAIL, axis=0)
    positive_share = np.mean(draws > 0.0, axis=0)

    accuracy_raw, brier_raw, log_loss_raw = 0, 1, 2
    accuracy_adj, brier_adj = 3, 4
    return {
        "n_games": n_games,
        "current_detect": bool(lower[accuracy_raw] > 0.0),
        "covariate_adjusted_detect": bool(lower[accuracy_adj] > 0.0),
        "continuous_brier_detect": bool(lower[brier_raw] > 0.0),
        "continuous_brier_screen75": bool(
            positive_share[brier_raw] >= SCREEN_PROBABILITY_THRESHOLD
        ),
        "continuous_log_loss_detect": bool(lower[log_loss_raw] > 0.0),
        "combined_detect": bool(lower[brier_adj] > 0.0),
    }


def _power_curve_for_arm(
    raw_full: np.ndarray,
    covariate_full: np.ndarray,
    block_game_indices: list[np.ndarray],
    n_blocks_total: int,
    *,
    arm_name: str,
    seed_offset: int,
    replications: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_blocks_sample in BLOCK_GRID:
        outcomes: list[dict[str, Any]] = []
        for replication in range(replications):
            seed = seed_offset + n_blocks_sample * 1_000_003 + replication
            outcomes.append(
                _run_replication(
                    raw_full,
                    covariate_full,
                    block_game_indices,
                    n_blocks_total,
                    n_blocks_sample=n_blocks_sample,
                    seed=seed,
                )
            )
        frame = pd.DataFrame(outcomes)
        row: dict[str, Any] = {
            "arm": arm_name,
            "n_blocks": n_blocks_sample if n_blocks_sample else n_blocks_total,
            "mean_n_games": float(frame["n_games"].mean()),
            "replications": len(frame),
        }
        for column in (
            "current_detect",
            "covariate_adjusted_detect",
            "continuous_brier_detect",
            "continuous_brier_screen75",
            "continuous_log_loss_detect",
            "combined_detect",
        ):
            row[column.replace("_detect", "_power").replace("_screen75", "_screen75_rate")] = float(
                frame[column].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/"
            "56edf890-1650-456a-b560-8d8b00b374b6/scratchpad/variance"
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    baseline = _load_baseline_predictions(args.features)
    features = pd.read_parquet(args.features)
    covariates_by_game = build_cuped_covariates(features).set_index("game_id")
    covariates_by_game = covariates_by_game.reindex(baseline["game_id"])
    for column in DEFAULT_CUPED_COVARIATES:
        missing = covariates_by_game[column].isna()
        if missing.any():
            covariates_by_game.loc[missing, column] = covariates_by_game[column].mean()
    covariate_full = covariates_by_game.loc[:, list(DEFAULT_CUPED_COVARIATES)].to_numpy(dtype=float)

    actual = baseline["home_cover"].to_numpy(dtype=float)
    baseline_probability = baseline["home_cover_probability"].to_numpy(dtype=float)
    block_game_indices, _block_of_row = _block_index(baseline)
    n_blocks_total = len(block_game_indices)
    print(
        f"Loaded {len(baseline)} clean-core market_residual games across "
        f"{n_blocks_total} week blocks in {perf_counter() - started:.1f}s"
    )

    baseline_raw = _paired_row_improvements(
        pd.DataFrame(
            {
                "home_cover_baseline": actual,
                "home_cover_probability_baseline": baseline_probability,
                "home_cover_probability_candidate": baseline_probability,
            }
        )
    ).to_numpy(dtype=float)
    assert np.allclose(baseline_raw, 0.0)

    # --- Build planted arms -------------------------------------------------
    probability_noise = np.random.default_rng(NOISE_SEED).normal(
        scale=PROBABILITY_NOISE_SD, size=len(baseline)
    )
    arm_raw_matrices: dict[str, np.ndarray] = {}
    plant_report: list[dict[str, Any]] = []
    for magnitude in POSITIVE_MAGNITUDES:
        candidate, achieved, delta = plant_accuracy_effect(
            baseline_probability,
            actual,
            target_accuracy_delta=magnitude,
            probability_noise=probability_noise,
        )
        paired = pd.DataFrame(
            {
                "home_cover_baseline": actual,
                "home_cover_probability_baseline": baseline_probability,
                "home_cover_probability_candidate": candidate,
            }
        )
        raw = _paired_row_improvements(paired).to_numpy(dtype=float)
        arm_name = MAGNITUDE_LABELS[magnitude]
        arm_raw_matrices[arm_name] = raw
        plant_report.append(
            {
                "arm": arm_name,
                "target_accuracy_delta": magnitude,
                "achieved_accuracy_delta": achieved,
                "delta_probability_shift": delta,
                "brier_improvement_mean": float(raw[:, 1].mean()),
                "log_loss_improvement_mean": float(raw[:, 2].mean()),
            }
        )

    null_deltas = [
        plant_accuracy_effect(
            baseline_probability,
            actual,
            target_accuracy_delta=magnitude,
            probability_noise=probability_noise,
        )[2]
        for magnitude in POSITIVE_MAGNITUDES
    ]
    null_magnitude = float(np.mean(null_deltas))
    for seed in NULL_SEEDS:
        candidate = plant_null_candidate(
            baseline_probability,
            actual,
            magnitude=null_magnitude,
            seed=seed,
            probability_noise=probability_noise,
        )
        paired = pd.DataFrame(
            {
                "home_cover_baseline": actual,
                "home_cover_probability_baseline": baseline_probability,
                "home_cover_probability_candidate": candidate,
            }
        )
        raw = _paired_row_improvements(paired).to_numpy(dtype=float)
        arm_name = f"null_seed_{seed}"
        arm_raw_matrices[arm_name] = raw
        plant_report.append(
            {
                "arm": arm_name,
                "target_accuracy_delta": 0.0,
                "achieved_accuracy_delta": float(raw[:, 0].mean()),
                "delta_probability_shift": null_magnitude,
                "brier_improvement_mean": float(raw[:, 1].mean()),
                "log_loss_improvement_mean": float(raw[:, 2].mean()),
            }
        )
    pd.DataFrame(plant_report).to_csv(args.output / "planted_effects.csv", index=False)
    print(
        f"Planted {len(arm_raw_matrices)} arms ({len(POSITIVE_MAGNITUDES)} positive, "
        f"{len(NULL_SEEDS)} null); null shift magnitude = {null_magnitude:.4f}"
    )

    # --- Power curves ---------------------------------------------------------
    power_frames: list[pd.DataFrame] = []
    for seed_offset, (arm_name, raw_full) in enumerate(arm_raw_matrices.items()):
        replications = (
            REPLICATIONS_NULL if arm_name.startswith("null_seed") else REPLICATIONS_POSITIVE
        )
        curve = _power_curve_for_arm(
            raw_full,
            covariate_full,
            block_game_indices,
            n_blocks_total,
            arm_name=arm_name,
            seed_offset=seed_offset * 10_000_019,
            replications=replications,
        )
        power_frames.append(curve)
        print(
            f"  {arm_name}: n_blocks={list(curve['n_blocks'])}, "
            f"current_power={list(np.round(curve['current_power'], 3))}"
        )
    power_table = pd.concat(power_frames, ignore_index=True)
    power_table.to_csv(args.output / "power_curves.csv", index=False)

    # Pool the null arms into a single false-positive-rate curve.
    null_rows = power_table.loc[power_table["arm"].str.startswith("null_seed")]
    pooled_columns = {
        "mean_n_games": "mean_n_games",
        "current_power": "current_fpr",
        "covariate_adjusted_power": "covariate_adjusted_fpr",
        "continuous_brier_power": "continuous_brier_fpr",
        "continuous_brier_screen75_rate": "continuous_brier_screen75_fpr",
        "continuous_log_loss_power": "continuous_log_loss_fpr",
        "combined_power": "combined_fpr",
    }
    pooled_rows: list[dict[str, Any]] = []
    for n_blocks_value, group in null_rows.groupby("n_blocks", sort=True):
        weights = group["replications"].to_numpy(dtype=float)
        pooled_row: dict[str, Any] = {"n_blocks": int(n_blocks_value)}
        for source_column, target_column in pooled_columns.items():
            pooled_row[target_column] = float(
                np.average(group[source_column].to_numpy(dtype=float), weights=weights)
            )
        pooled_rows.append(pooled_row)
    pooled_null = pd.DataFrame(pooled_rows)
    pooled_null.to_csv(args.output / "false_positive_rates.csv", index=False)
    print("Pooled null false-positive rates (should be ~0.025 for *_detect, ~0.25 for screen75):")
    print(pooled_null.to_string(index=False))

    # --- Required sample size per magnitude x method ---------------------------
    method_columns = {
        "current": "current_power",
        "covariate_adjusted": "covariate_adjusted_power",
        "continuous_brier": "continuous_brier_power",
        "continuous_log_loss": "continuous_log_loss_power",
        "combined": "combined_power",
    }
    required_rows: list[dict[str, Any]] = []
    for magnitude in POSITIVE_MAGNITUDES:
        arm_name = MAGNITUDE_LABELS[magnitude]
        curve = power_table.loc[power_table["arm"].eq(arm_name)].sort_values("mean_n_games")
        row: dict[str, Any] = {"arm": arm_name, "target_accuracy_delta": magnitude}
        for method_name, column in method_columns.items():
            required = required_sample_size(
                curve.rename(columns={"mean_n_games": "n_games", column: "power"}),
                target_power=0.80,
            )
            row[f"required_n_games_{method_name}"] = required
        required_rows.append(row)
    required_table = pd.DataFrame(required_rows)
    current_n = required_table["required_n_games_current"]
    for method_name in method_columns:
        if method_name == "current":
            continue
        required_table[f"multiplier_vs_current_{method_name}"] = (
            current_n / required_table[f"required_n_games_{method_name}"]
        )
    required_table.to_csv(args.output / "required_sample_sizes.csv", index=False)
    print("\nRequired games for 80% power (None = grid exhausted, still <80% at full clean core):")
    print(required_table.to_string(index=False))

    # --- Screen -> confirm concordance at an affordable window ------------------
    concordance_rows: list[dict[str, Any]] = []
    full_power = power_table.loc[power_table["n_blocks"].eq(n_blocks_total)]
    screen_power = power_table.loc[power_table["n_blocks"].eq(CONCORDANCE_TARGET_BLOCKS)]
    for arm_name in arm_raw_matrices:
        full_row = full_power.loc[full_power["arm"].eq(arm_name)]
        screen_row = screen_power.loc[screen_power["arm"].eq(arm_name)]
        if full_row.empty or screen_row.empty:
            continue
        concordance_rows.append(
            {
                "arm": arm_name,
                "screen_n_blocks": CONCORDANCE_TARGET_BLOCKS,
                "screen_mean_n_games": float(screen_row["mean_n_games"].iloc[0]),
                "screen_brier_screen75_rate": float(
                    screen_row["continuous_brier_screen75_rate"].iloc[0]
                ),
                "screen_accuracy_power": float(screen_row["current_power"].iloc[0]),
                "full_n_games": float(full_row["mean_n_games"].iloc[0]),
                "full_accuracy_power": float(full_row["current_power"].iloc[0]),
            }
        )
    concordance_table = pd.DataFrame(concordance_rows)
    concordance_table.to_csv(args.output / "screen_confirm_concordance.csv", index=False)
    print("\nScreen (12 blocks) vs full-clean-core accuracy resolution, per arm:")
    print(concordance_table.to_string(index=False))

    metadata = {
        "n_games_clean_core": len(baseline),
        "n_blocks_total": n_blocks_total,
        "block_grid": list(BLOCK_GRID),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "replications_positive": REPLICATIONS_POSITIVE,
        "replications_null": REPLICATIONS_NULL,
        "detection_tail": DETECTION_TAIL,
        "screen_probability_threshold": SCREEN_PROBABILITY_THRESHOLD,
        "null_shift_magnitude": null_magnitude,
        "covariate_columns": list(DEFAULT_CUPED_COVARIATES),
        "total_seconds": perf_counter() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nDone in {metadata['total_seconds']:.1f}s. Output: {args.output}")


if __name__ == "__main__":
    main()
