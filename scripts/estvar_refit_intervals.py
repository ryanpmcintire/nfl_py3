"""Validate the refit-aware interval and derive the D4 block floor (estvar).

Research item: ``docs/estimation_variance.md``. Two defects of the measuring
instrument are properties of the interval machinery itself, so every result the
project produces inherits them until they are fixed:

- **D2** -- the block bootstrap resamples GAMES but never REFITS, so every
  reported interval is conditional on one fit and is measurably too narrow.
- **D4** -- the block bootstrap is degenerate at small block counts, and the
  "~4-5 block" floor recorded in ``docs/anytime_valid.md`` sec 6 was asserted
  from a combinatorial argument, never measured against coverage.

This script measures both, and derives the one constant the fix needs (the
variance-inflation factor) rather than typing it in. Five studies, selected
with ``--study``:

``degeneracy``
    Percentile block-bootstrap coverage of a KNOWN truth as a function of the
    block count ``k``, on the project's OWN estimand -- the paired forced-pick
    accuracy delta, whose per-game value is in ``{-1, 0, +1}`` and is zero on
    the ``1 - f`` games where both arms pick the same side. Swept at three
    values of ``f`` spanning the project's real range. Fixes
    ``MIN_BLOCKS_FOR_INTERVAL``.

``coverage``
    Planted-effect coverage with a known ``Delta_true``, comparing the naive
    (currently reported) interval, the 2026-08-18 ``refit_aware_paired_interval``
    and the new ``refit_aware_interval``. Run at two block counts: the one
    ``docs/estimation_variance.md`` sec 2 used (20 blocks, where D4 is still
    binding and confounds the D2 measurement) and a D4-free one (80 blocks),
    which is where the D2 fix can actually be judged.

``inflation``
    What the inflation factor depends on. Sweeps the training-set size and the
    evaluation-set size and reports ``f``, ``conditional_sd``, ``refit_sd`` and
    the derived factor, so the factor can be predicted for a configuration
    instead of being borrowed from another one.

``refits``
    How many refits the cheap path needs before the factor stops moving.

``cfb``
    One real comparison on CFB clean-core data (free and unlimited per rule 8
    of ``docs/rotation_registry.md``; this script only READS the registry):
    a thin market+context+experience feature set against the full
    ``CFB_MODEL_FEATURE_COLUMNS``, annual refit cadence, scored with the naive,
    the old honest and the new honest interval, plus paired-vs-unpaired refits.

Writes JSON to the scratchpad only. Never touches ``artifacts/``, the active
model, ``registry/*.json``, or any production pick.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.estimation_variance import (
    MIN_BLOCKS_FOR_INTERVAL,
    RELIABLE_BLOCKS_FOR_INTERVAL,
    distinct_block_resamples,
    naive_block_bootstrap_interval,
    paired_refit_predicted_values,
    picks_differ_fraction,
    point_predicted_values,
    refit_aware_interval,
    refit_aware_paired_interval,
    refit_variance_decomposition,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"
OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\d08576b1-0b16-4ebc-9ebc-f0bc62b943d3\scratchpad\estvar_refit"
)

RIDGE_ALPHA = 10.0
NOISE_SD = 13.0
BASELINE_COLUMNS = [f"b{index}" for index in range(4)]
EXTRA_COLUMNS = [f"e{index}" for index in range(3)]
ALL_COLUMNS = BASELINE_COLUMNS + EXTRA_COLUMNS
BASELINE_COEF = np.array([2.0, -1.5, 1.0, 0.8])


# ---------------------------------------------------------------------------
# Shared synthetic DGP (same shape as scripts/estvar_planted_effects.py, so the
# numbers here are directly comparable to docs/estimation_variance.md sec 2)
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def make_frame(n: int, *, seed: int, extra_coef: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_base = rng.normal(size=(n, len(BASELINE_COLUMNS)))
    x_extra = rng.normal(size=(n, len(EXTRA_COLUMNS)))
    mean = x_base @ BASELINE_COEF + x_extra @ extra_coef
    target = mean + rng.normal(scale=NOISE_SD, size=n)
    frame = pd.DataFrame(x_base, columns=BASELINE_COLUMNS)
    for index, column in enumerate(EXTRA_COLUMNS):
        frame[column] = x_extra[:, index]
    frame["target"] = target
    frame["actual"] = (target > 0.0).astype(float)
    return frame


def block_ids_for(n: int, *, block_size: int) -> np.ndarray:
    return (np.arange(n) // block_size).astype(np.int64)


def _accuracy_delta(actual: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.mean(((candidate >= 0.5) == actual).astype(float))
        - np.mean(((baseline >= 0.5) == actual).astype(float))
    )


def true_effect_mc(
    *, extra_coef: np.ndarray, n_train: int, replicates: int, seed: int, population: int = 20_000
) -> tuple[float, float]:
    """``Delta_true``: the expectation over fresh TRAINING draws and fresh games."""

    values = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        train = make_frame(n_train, seed=seed + 2 * draw, extra_coef=extra_coef)
        test = make_frame(population, seed=seed + 2 * draw + 1, extra_coef=extra_coef)
        baseline = point_predicted_values(
            train,
            test,
            feature_columns=BASELINE_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        candidate = point_predicted_values(
            train,
            test,
            feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
        )
        values[draw] = _accuracy_delta(
            test["actual"].to_numpy(), sigmoid(baseline), sigmoid(candidate)
        )
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(replicates))


# ---------------------------------------------------------------------------
# Study 1: the D4 block floor, measured on the real estimand
# ---------------------------------------------------------------------------


def degeneracy_study(*, replicates: int, samples: int, seed: int) -> dict[str, Any]:
    """Coverage of the percentile block bootstrap vs. the number of blocks.

    The per-game improvement is exactly the shape ``_paired_accuracy_improvement``
    produces: zero on the games where both arms pick the same side, and +1/-1 on
    the fraction ``f`` where they differ. That is a heavily zero-inflated, skewed
    variable, which is why the block count needed for honest coverage is much
    larger than a combinatorial count of distinct resamples suggests.
    """

    rows: list[dict[str, Any]] = []
    block_size = 14
    for f, p_win in ((0.10, 0.52), (0.20, 0.51), (0.55, 0.505)):
        truth = f * (2.0 * p_win - 1.0)
        for k in (1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 20, 25, 35, 50, 68, 100, 199):
            rng = np.random.default_rng(seed + 1_000 * k + int(f * 100))
            covers = 0
            widths = np.empty(replicates, dtype=np.float64)
            for rep in range(replicates):
                n = k * block_size
                differ = rng.random(n) < f
                win = rng.random(n) < p_win
                values = np.where(differ, np.where(win, 1.0, -1.0), 0.0)
                block_sum = values.reshape(k, block_size).sum(axis=1)
                counts = rng.multinomial(k, np.full(k, 1.0 / k), size=samples)
                draws = (counts @ block_sum) / float(n)
                lower = float(np.quantile(draws, 0.025))
                upper = float(np.quantile(draws, 0.975))
                covers += int(lower <= truth <= upper)
                widths[rep] = upper - lower
            rows.append(
                {
                    "f": f,
                    "blocks": k,
                    "distinct_resamples": float(distinct_block_resamples(k)),
                    "coverage": covers / replicates,
                    "mean_width_points": float(np.mean(widths)) * 100.0,
                    "zero_width_fraction": float(np.mean(widths <= 0.0)),
                }
            )
    return {
        "study": "degeneracy",
        "nominal_coverage": 0.95,
        "replicates": replicates,
        "bootstrap_samples": samples,
        "block_size_games": block_size,
        "rows": rows,
        "adopted_min_blocks": MIN_BLOCKS_FOR_INTERVAL,
        "adopted_reliable_blocks": RELIABLE_BLOCKS_FOR_INTERVAL,
    }


# ---------------------------------------------------------------------------
# Study 2: planted-effect coverage of naive vs. old-honest vs. new-honest
# ---------------------------------------------------------------------------


def coverage_cell(
    *,
    label: str,
    extra_coef: np.ndarray,
    n_train: int,
    n_test: int,
    block_size: int,
    replicates: int,
    n_boot: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    truth, truth_se = true_effect_mc(
        extra_coef=extra_coef, n_train=n_train, replicates=250, seed=seed
    )
    blocks = block_ids_for(n_test, block_size=block_size)
    block_count = len(np.unique(blocks))

    naive_hits = old_hits = new_hits = 0
    naive_widths: list[float] = []
    old_widths: list[float] = []
    new_widths: list[float] = []
    factors: list[float] = []
    fractions: list[float] = []
    naive_pp: list[float] = []
    new_pp: list[float] = []
    for rep in range(replicates):
        rep_seed = seed + 100_000 + rep
        train = make_frame(n_train, seed=rep_seed, extra_coef=extra_coef)
        test = make_frame(n_test, seed=rep_seed + 500_000, extra_coef=extra_coef)
        actual = test["actual"].to_numpy()

        baseline_point = sigmoid(
            point_predicted_values(
                train,
                test,
                feature_columns=BASELINE_COLUMNS,
                target_column="target",
                ridge_alpha=RIDGE_ALPHA,
            )
        )
        candidate_point = sigmoid(
            point_predicted_values(
                train,
                test,
                feature_columns=ALL_COLUMNS,
                target_column="target",
                ridge_alpha=RIDGE_ALPHA,
            )
        )
        refits = paired_refit_predicted_values(
            train,
            test,
            baseline_feature_columns=BASELINE_COLUMNS,
            candidate_feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=n_boot,
            seed=rep_seed + 1,
        )
        baseline_refits = sigmoid(refits.baseline)
        candidate_refits = sigmoid(refits.candidate)

        result = refit_aware_interval(
            actual,
            baseline_refits,
            candidate_refits,
            blocks,
            point_baseline_prob=baseline_point,
            point_candidate_prob=candidate_point,
            samples=samples,
            seed=rep_seed,
            on_degenerate="warn",
        )
        old = refit_aware_paired_interval(
            actual, baseline_refits, candidate_refits, blocks, seed=rep_seed
        )

        naive_hits += int(result.naive.lower <= truth <= result.naive.upper)
        old_hits += int(old.lower <= truth <= old.upper)
        new_hits += int(result.honest.lower <= truth <= result.honest.upper)
        naive_widths.append(result.naive.upper - result.naive.lower)
        old_widths.append(old.upper - old.lower)
        new_widths.append(result.honest.upper - result.honest.lower)
        factors.append(result.decomposition.inflation_factor)
        fractions.append(picks_differ_fraction(baseline_point, candidate_point))
        naive_pp.append(result.naive.probability_positive)
        new_pp.append(result.honest.probability_positive)

    return {
        "label": label,
        "delta_true_points": truth * 100.0,
        "delta_true_mc_se_points": truth_se * 100.0,
        "n_train": n_train,
        "n_test": n_test,
        "blocks": block_count,
        "replicates": replicates,
        "n_boot": n_boot,
        "bootstrap_samples": samples,
        "mean_f": float(np.mean(fractions)),
        "naive_coverage": naive_hits / replicates,
        "old_honest_coverage": old_hits / replicates,
        "new_honest_coverage": new_hits / replicates,
        "naive_width_points": float(np.mean(naive_widths)) * 100.0,
        "old_honest_width_points": float(np.mean(old_widths)) * 100.0,
        "new_honest_width_points": float(np.mean(new_widths)) * 100.0,
        "mean_inflation_factor": float(np.mean(factors)),
        "naive_mean_probability_positive": float(np.mean(naive_pp)),
        "new_honest_mean_probability_positive": float(np.mean(new_pp)),
    }


def coverage_study(*, replicates: int, n_boot: int, samples: int, seed: int) -> dict[str, Any]:
    null_coef = np.zeros(len(EXTRA_COLUMNS))
    real_coef = np.array([1.6, -1.2, 0.9])
    cells = []
    for dgp_label, coef in (("null", null_coef), ("real_effect", real_coef)):
        cells.append(
            coverage_cell(
                label=f"{dgp_label}/20 blocks (the doc's own configuration)",
                extra_coef=coef,
                n_train=1_200,
                n_test=400,
                block_size=20,
                replicates=replicates,
                n_boot=n_boot,
                samples=samples,
                seed=seed + (0 if dgp_label == "null" else 7),
            )
        )
        cells.append(
            coverage_cell(
                label=f"{dgp_label}/80 blocks (D4-free, so this isolates D2)",
                extra_coef=coef,
                n_train=1_200,
                n_test=1_120,
                block_size=14,
                replicates=replicates,
                n_boot=n_boot,
                samples=samples,
                seed=seed + (3 if dgp_label == "null" else 11),
            )
        )
    return {"study": "coverage", "nominal_coverage": 0.95, "cells": cells}


# ---------------------------------------------------------------------------
# Study 3: what drives the inflation factor
# ---------------------------------------------------------------------------


def inflation_study(*, n_boot: int, samples: int, seed: int, replicates: int) -> dict[str, Any]:
    real_coef = np.array([1.6, -1.2, 0.9])
    rows: list[dict[str, Any]] = []
    for n_train in (300, 600, 1_200, 2_400, 4_800):
        for n_test, block_size in ((420, 14), (1_120, 14), (4_200, 14)):
            factors: list[float] = []
            cond: list[float] = []
            refit: list[float] = []
            fractions: list[float] = []
            for rep in range(replicates):
                rep_seed = seed + 31 * rep + n_train + n_test
                train = make_frame(n_train, seed=rep_seed, extra_coef=real_coef)
                test = make_frame(n_test, seed=rep_seed + 900_000, extra_coef=real_coef)
                actual = test["actual"].to_numpy()
                blocks = block_ids_for(n_test, block_size=block_size)
                baseline_point = sigmoid(
                    point_predicted_values(
                        train,
                        test,
                        feature_columns=BASELINE_COLUMNS,
                        target_column="target",
                        ridge_alpha=RIDGE_ALPHA,
                    )
                )
                candidate_point = sigmoid(
                    point_predicted_values(
                        train,
                        test,
                        feature_columns=ALL_COLUMNS,
                        target_column="target",
                        ridge_alpha=RIDGE_ALPHA,
                    )
                )
                refits = paired_refit_predicted_values(
                    train,
                    test,
                    baseline_feature_columns=BASELINE_COLUMNS,
                    candidate_feature_columns=ALL_COLUMNS,
                    target_column="target",
                    ridge_alpha=RIDGE_ALPHA,
                    n_boot=n_boot,
                    seed=rep_seed + 1,
                )
                decomposition = refit_variance_decomposition(
                    actual,
                    sigmoid(refits.baseline),
                    sigmoid(refits.candidate),
                    blocks,
                    point_baseline_prob=baseline_point,
                    point_candidate_prob=candidate_point,
                    samples=samples,
                    seed=rep_seed,
                )
                factors.append(decomposition.inflation_factor)
                cond.append(decomposition.conditional_sd)
                refit.append(decomposition.refit_sd)
                fractions.append(picks_differ_fraction(baseline_point, candidate_point))
            rows.append(
                {
                    "n_train": n_train,
                    "n_test": n_test,
                    "blocks": n_test // block_size,
                    "f": float(np.mean(fractions)),
                    "conditional_sd_points": float(np.mean(cond)) * 100.0,
                    "refit_sd_points": float(np.mean(refit)) * 100.0,
                    "inflation_factor": float(np.mean(factors)),
                    "width_understatement_pct": (float(np.mean(factors)) - 1.0) * 100.0,
                }
            )
    return {"study": "inflation", "n_boot": n_boot, "replicates": replicates, "rows": rows}


# ---------------------------------------------------------------------------
# Study 4: how many refits the cheap path needs
# ---------------------------------------------------------------------------


def refits_study(*, samples: int, seed: int, max_refits: int, replicates: int) -> dict[str, Any]:
    real_coef = np.array([1.6, -1.2, 0.9])
    counts = [m for m in (5, 10, 20, 40, 80, 160, 320, 640) if m <= max_refits]
    per_count: dict[int, list[float]] = {m: [] for m in counts}
    for rep in range(replicates):
        rep_seed = seed + 137 * rep
        train = make_frame(1_200, seed=rep_seed, extra_coef=real_coef)
        test = make_frame(1_120, seed=rep_seed + 900_000, extra_coef=real_coef)
        actual = test["actual"].to_numpy()
        blocks = block_ids_for(1_120, block_size=14)
        baseline_point = sigmoid(
            point_predicted_values(
                train,
                test,
                feature_columns=BASELINE_COLUMNS,
                target_column="target",
                ridge_alpha=RIDGE_ALPHA,
            )
        )
        candidate_point = sigmoid(
            point_predicted_values(
                train,
                test,
                feature_columns=ALL_COLUMNS,
                target_column="target",
                ridge_alpha=RIDGE_ALPHA,
            )
        )
        refits = paired_refit_predicted_values(
            train,
            test,
            baseline_feature_columns=BASELINE_COLUMNS,
            candidate_feature_columns=ALL_COLUMNS,
            target_column="target",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=max(counts),
            seed=rep_seed + 1,
        )
        baseline_refits = sigmoid(refits.baseline)
        candidate_refits = sigmoid(refits.candidate)
        for m in counts:
            decomposition = refit_variance_decomposition(
                actual,
                baseline_refits[:m],
                candidate_refits[:m],
                blocks,
                point_baseline_prob=baseline_point,
                point_candidate_prob=candidate_point,
                samples=samples,
                seed=rep_seed,
            )
            per_count[m].append(decomposition.inflation_factor)
    reference = float(np.mean(per_count[max(counts)]))
    rows = [
        {
            "refits": m,
            "mean_inflation_factor": float(np.mean(per_count[m])),
            "sd_across_replicates": float(np.std(per_count[m], ddof=1)),
            "bias_vs_max_refits": float(np.mean(per_count[m])) - reference,
        }
        for m in counts
    ]
    return {
        "study": "refits",
        "replicates": replicates,
        "reference_refits": max(counts),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Study 5: one real CFB comparison
# ---------------------------------------------------------------------------


def cfb_study(
    *, features_path: Path, n_boot: int, samples: int, seed: int, min_train_games: int
) -> dict[str, Any]:
    from nfl_ats.cfb_benchmark import (
        CFB_BENCHMARK_RIDGE_ALPHA,
        CFB_CLEAN_CORE_SEASONS,
        fit_cfb_residual_model,
    )
    from nfl_ats.cfb_features import (
        CFB_CONTEXT_FEATURES,
        CFB_EXPERIENCE_FEATURES,
        CFB_MARKET_FEATURES,
        CFB_MODEL_FEATURE_COLUMNS,
    )
    from nfl_ats.estimation_variance import home_cover_probability_from_center

    thin_columns = tuple(CFB_MARKET_FEATURES + CFB_CONTEXT_FEATURES + CFB_EXPERIENCE_FEATURES)
    full_columns = tuple(CFB_MODEL_FEATURE_COLUMNS)

    frame = pd.read_parquet(features_path)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = (
        frame.loc[
            pd.to_numeric(frame["result"], errors="coerce").notna()
            & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
        ]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )
    seasons = tuple(sorted(CFB_CLEAN_CORE_SEASONS))

    actual_parts: list[np.ndarray] = []
    block_parts: list[np.ndarray] = []
    thin_point_parts: list[np.ndarray] = []
    full_point_parts: list[np.ndarray] = []
    thin_refit_parts: list[np.ndarray] = []
    full_refit_parts: list[np.ndarray] = []
    unpaired_full_parts: list[np.ndarray] = []

    def refit_probabilities(
        raw: np.ndarray, spread: np.ndarray, residuals: np.ndarray
    ) -> np.ndarray:
        """Loop over draws: one flattened call would need gigabytes of scratch."""

        out = np.empty_like(raw)
        for draw in range(raw.shape[0]):
            out[draw] = home_cover_probability_from_center(spread + raw[draw], spread, residuals)
        return out

    for season in seasons:
        test = completed.loc[completed["season"].eq(season)]
        if test.empty:
            continue
        cutoff = test["gameday"].min()
        train = completed.loc[completed["gameday"].lt(cutoff)]
        if len(train) < min_train_games:
            continue
        test = test.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        test = test.loc[pd.to_numeric(test["home_cover"], errors="coerce").notna()].reset_index(
            drop=True
        )
        spread = pd.to_numeric(test["spread_line"], errors="coerce").to_numpy(dtype=float)

        thin_model = fit_cfb_residual_model(
            train, ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA, feature_columns=thin_columns
        )
        full_model = fit_cfb_residual_model(
            train, ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA, feature_columns=full_columns
        )
        paired = paired_refit_predicted_values(
            train,
            test,
            baseline_feature_columns=thin_columns,
            candidate_feature_columns=full_columns,
            target_column="ats_margin",
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            n_boot=n_boot,
            seed=seed + season,
        )
        unpaired = paired_refit_predicted_values(
            train,
            test,
            baseline_feature_columns=thin_columns,
            candidate_feature_columns=full_columns,
            target_column="ats_margin",
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            n_boot=n_boot,
            seed=seed + season,
            paired=False,
        )

        actual_parts.append(pd.to_numeric(test["home_cover"], errors="raise").to_numpy(dtype=float))
        block_parts.append(season * 1000 + test["week"].to_numpy(dtype=np.int64))
        thin_point_parts.append(
            thin_model.predict(test)["home_cover_probability"].to_numpy(dtype=float)
        )
        full_point_parts.append(
            full_model.predict(test)["home_cover_probability"].to_numpy(dtype=float)
        )
        thin_refit_parts.append(refit_probabilities(paired.baseline, spread, thin_model.residuals))
        full_refit_parts.append(refit_probabilities(paired.candidate, spread, full_model.residuals))
        unpaired_full_parts.append(
            refit_probabilities(unpaired.candidate, spread, full_model.residuals)
        )
        print(f"  season {season}: train={len(train)} test={len(test)}", flush=True)

    actual = np.concatenate(actual_parts)
    blocks = np.concatenate(block_parts)
    thin_point = np.concatenate(thin_point_parts)
    full_point = np.concatenate(full_point_parts)
    thin_refits = np.hstack(thin_refit_parts)
    full_refits = np.hstack(full_refit_parts)
    unpaired_refits = np.hstack(unpaired_full_parts)

    naive = naive_block_bootstrap_interval(
        actual, thin_point, full_point, blocks, samples=samples, seed=seed
    )
    old = refit_aware_paired_interval(actual, thin_refits, full_refits, blocks, seed=seed)
    new = refit_aware_interval(
        actual,
        thin_refits,
        full_refits,
        blocks,
        point_baseline_prob=thin_point,
        point_candidate_prob=full_point,
        samples=samples,
        seed=seed,
    )
    unpaired = refit_aware_interval(
        actual,
        thin_refits,
        unpaired_refits,
        blocks,
        point_baseline_prob=thin_point,
        point_candidate_prob=full_point,
        samples=samples,
        seed=seed,
        paired=False,
    )

    def as_row(interval: Any, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "estimate_points": interval.estimate * 100.0,
            "lower_points": interval.lower * 100.0,
            "upper_points": interval.upper * 100.0,
            "width_points": (interval.upper - interval.lower) * 100.0,
            "probability_positive": interval.probability_positive,
            "blocks": interval.block_count,
        }

    return {
        "study": "cfb",
        "comparison": "thin (market+context+experience) vs full CFB_MODEL_FEATURE_COLUMNS",
        "games": len(actual),
        "blocks": len(np.unique(blocks)),
        "n_boot": n_boot,
        "f": picks_differ_fraction(thin_point, full_point),
        "intervals": [
            as_row(naive, "naive (reported style)"),
            as_row(old, "refit_aware_paired_interval (2026-08-18)"),
            as_row(new.honest, "refit_aware_interval (paired refits)"),
            as_row(unpaired.honest, "refit_aware_interval (UNPAIRED refits -- wrong)"),
        ],
        "decomposition": _decomposition_row(new.decomposition),
        "unpaired_decomposition": _decomposition_row(unpaired.decomposition),
    }


def _decomposition_row(decomposition: Any) -> dict[str, Any]:
    return {
        "conditional_sd_points": decomposition.conditional_sd * 100.0,
        "refit_common_sd_points": decomposition.refit_sd * 100.0,
        "refit_fixed_games_sd_points": decomposition.refit_fixed_games_sd * 100.0,
        "inflation_factor": decomposition.inflation_factor,
        "inflation_factor_upper95": decomposition.inflation_factor_upper,
        "interaction_double_counted_factor": decomposition.interaction_double_counted_factor,
        "refit_draws": decomposition.refit_draws,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        action="append",
        choices=["degeneracy", "coverage", "inflation", "refits", "cfb", "all"],
        help="Which study to run (repeatable). Default: all.",
    )
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--n-boot", type=int, default=60)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-refits", type=int, default=320)
    parser.add_argument("--cfb-features", type=Path, default=DEFAULT_CFB_FEATURES)
    parser.add_argument("--cfb-min-train-games", type=int, default=1_500)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    selected = set(args.study or ["all"])
    if "all" in selected:
        selected = {"degeneracy", "coverage", "inflation", "refits", "cfb"}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name in ("degeneracy", "coverage", "inflation", "refits", "cfb"):
        if name not in selected:
            continue
        started = time.time()
        if name == "degeneracy":
            payload = degeneracy_study(replicates=2_000, samples=4_000, seed=args.seed)
        elif name == "coverage":
            payload = coverage_study(
                replicates=args.replicates,
                n_boot=args.n_boot,
                samples=args.samples,
                seed=args.seed,
            )
        elif name == "inflation":
            payload = inflation_study(
                n_boot=args.n_boot, samples=args.samples, seed=args.seed, replicates=12
            )
        elif name == "refits":
            payload = refits_study(
                samples=args.samples,
                seed=args.seed,
                max_refits=args.max_refits,
                replicates=12,
            )
        else:
            payload = cfb_study(
                features_path=args.cfb_features,
                n_boot=args.n_boot,
                samples=args.samples,
                seed=args.seed,
                min_train_games=args.cfb_min_train_games,
            )
        payload["elapsed_seconds"] = round(time.time() - started, 1)
        results[name] = payload
        path = args.out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{name}] {payload['elapsed_seconds']}s -> {path}")
        print(json.dumps(payload, indent=2)[:6_000])

    (args.out_dir / "all.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
