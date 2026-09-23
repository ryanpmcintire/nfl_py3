from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability import (
    COUNTED_FLAG_COLUMNS,
    FLAG_SUM_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    signed_composition_flags,
)
from nfl_ats.pick_probability_fit import (
    FIT_RIDGE,
    SCHEDULE_COLUMNS,
    _arrest_incidents,
    _forecast_temperatures,
    _protection_back_side,
)
from nfl_ats.pick_probability_fit import _fit_logit as fit_logit
from nfl_ats.signal_atlas import _cell as signal_cell
from nfl_ats.snapshots import latest_snapshot, load_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
POPULATION_PATH = (
    ARTIFACTS_ROOT / "extended_fit_population" / "20260923T205910Z" / "population.parquet"
)
REDDIT_PARQUET = DATA_ROOT / "processed" / "game_features_weak_stack_reddit.parquet"
OUTPUT_ROOT = ARTIFACTS_ROOT / "extended_population_regrade"

BASE_FEATURES = ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)
EARLY_WINDOW_MAX_WEEK = 4
REDDIT_TERM = "reddit_home_comment_ratio_elevated"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
INTERVAL_LEVEL = 0.95
RELIABILITY_EDGES = [0.0, 0.5, 1.0]


def design_matrix(
    frame: pd.DataFrame, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    columns = [np.ones(len(frame))]
    for name in features:
        columns.append(((frame[name].astype(float) - means[name]) / stds[name]).to_numpy())
    return np.column_stack(columns)


def standardisers(
    train: pd.DataFrame, features: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    means = {name: float(train[name].mean()) for name in features}
    stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in features}
    return means, stds


def natural_coefficients(
    beta: np.ndarray, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    natural = {name: float(beta[index + 1] / stds[name]) for index, name in enumerate(features)}
    intercept = float(beta[0])
    for index, name in enumerate(features):
        intercept -= float(beta[index + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def predict(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    beta: np.ndarray,
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    z = np.clip(design_matrix(frame, features, means, stds) @ beta, -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def loso(
    population: pd.DataFrame, features: tuple[str, ...]
) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    seasons = sorted(int(value) for value in population["season"].unique())
    out = pd.Series(np.nan, index=population.index, dtype=float)
    folds: dict[str, dict[str, float]] = {}
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = standardisers(train, features)
        beta = fit_logit(
            design_matrix(train, features, means, stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        out.loc[test.index] = predict(test, features, beta, means, stds)
        folds[str(held)] = natural_coefficients(beta, features, means, stds)
    return out, folds


def in_sample_fit(
    population: pd.DataFrame, features: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, float]]:
    means, stds = standardisers(population, features)
    beta = fit_logit(
        design_matrix(population, features, means, stds),
        population["home_covered"].astype(float).to_numpy(),
        FIT_RIDGE,
    )
    return predict(population, features, beta, means, stds), natural_coefficients(
        beta, features, means, stds
    )


def accuracy(probability: pd.Series, target: pd.Series) -> float:
    picked_home = probability.ge(0.5)
    correct = picked_home.astype(float).eq(target.astype(float))
    return float(correct.mean())


def build_predictions(population: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    schedule_columns = [c for c in SCHEDULE_COLUMNS if c in schedules.columns and c != "week"]
    joined = population.merge(
        schedules[schedule_columns].drop_duplicates("game_id"),
        on=["game_id", "season"],
        how="left",
    )
    return joined


def flag_coverage(flags: pd.DataFrame, population: pd.DataFrame) -> dict[str, dict[str, int]]:
    merged = flags.merge(population[["game_id", "season"]], on="game_id", how="left")
    pre = merged.loc[merged["season"].lt(2020)]
    post = merged.loc[merged["season"].ge(2020)]
    coverage = {}
    for column in COUNTED_FLAG_COLUMNS:
        coverage[column] = {
            "nonzero_pre_2020": int(pre[column].astype(float).ne(0.0).sum()),
            "nonzero_post_2020": int(post[column].astype(float).ne(0.0).sum()),
        }
    return coverage


def add_reddit_column(population: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    reddit = pd.read_parquet(REDDIT_PARQUET, columns=["game_id", REDDIT_TERM])
    lookup = reddit.drop_duplicates("game_id").set_index("game_id")[REDDIT_TERM]
    enriched = population.copy()
    mapped = enriched["game_id"].astype(str).map(lookup)
    coverage = {
        "matched_games": int(mapped.notna().sum()),
        "unmatched_games": int(mapped.isna().sum()),
        "positive_flag_games_among_matched": int((mapped.fillna(0.0) > 0).sum()),
    }
    enriched[REDDIT_TERM] = mapped.fillna(0.0).astype(float)
    return enriched, coverage


def paired_cells(
    population: pd.DataFrame, base_oos: pd.Series, variant_oos: pd.Series
) -> dict[str, dict]:
    frame = population.copy()
    frame["rating_full"] = variant_oos
    frame["rating_reduced"] = base_oos
    logit = frame["model_logit"].astype(float).clip(-35.0, 35.0)
    frame["model_probability"] = 1.0 / (1.0 + np.exp(-logit))
    scored = frame.loc[frame["rating_full"].notna() & frame["rating_reduced"].notna()].copy()
    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": RELIABILITY_EDGES,
    }
    overall = signal_cell(scored, "rating", "overall", declaration)
    recent = scored.loc[scored["season"].ge(2020)]
    recent_cell = signal_cell(recent, "rating", "overall", declaration)
    return {"all_seasons_2011_2025": overall, "held_out_2020_2025_only": recent_cell}


def run_look(
    name: str,
    mechanism: str,
    population: pd.DataFrame,
    features: tuple[str, ...],
    base_oos: pd.Series,
    base_is_accuracy: float,
) -> dict:
    variant_oos, variant_folds = loso(population, features)
    variant_is, variant_is_coefficients = in_sample_fit(population, features)
    variant_is_accuracy = accuracy(
        pd.Series(variant_is, index=population.index), population["home_covered"]
    )
    scored_mask = variant_oos.notna() & base_oos.notna()
    variant_oos_accuracy = accuracy(
        variant_oos.loc[scored_mask], population.loc[scored_mask, "home_covered"]
    )
    base_oos_accuracy = accuracy(
        base_oos.loc[scored_mask], population.loc[scored_mask, "home_covered"]
    )
    cells = paired_cells(population, base_oos, variant_oos)
    return {
        "name": name,
        "mechanism": mechanism,
        "features": list(features),
        "coverage_games": int(scored_mask.sum()),
        "variant_in_sample_accuracy": variant_is_accuracy,
        "variant_oos_accuracy": variant_oos_accuracy,
        "variant_accuracy_gap": variant_is_accuracy - variant_oos_accuracy,
        "base_oos_accuracy_same_rows": base_oos_accuracy,
        "base_in_sample_accuracy": base_is_accuracy,
        "fold_coefficients": variant_folds,
        "in_sample_coefficients": variant_is_coefficients,
        "cells": cells,
    }


def main() -> None:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    population = pd.read_parquet(POPULATION_PATH)
    population = population.reset_index(drop=True)
    schedules, _team_stats = load_snapshot(latest_snapshot(DATA_ROOT / "raw"))

    predictions = build_predictions(population, schedules)
    incidents = _arrest_incidents(DATA_ROOT)
    forecasts = _forecast_temperatures(DATA_ROOT)
    protection = _protection_back_side(schedules, DATA_ROOT)
    flags = signed_composition_flags(
        predictions,
        schedules,
        incidents=incidents,
        forecasts_tuesday_noon=forecasts,
        protection_back_side=protection,
    )

    coverage = flag_coverage(flags, population)
    dropped_flags = [column for column, stats in coverage.items() if stats["nonzero_pre_2020"] == 0]
    kept_flags = [column for column in COUNTED_FLAG_COLUMNS if column not in dropped_flags]

    population = population.merge(
        flags[["game_id", *COUNTED_FLAG_COLUMNS]], on="game_id", how="left", validate="one_to_one"
    )
    for column in COUNTED_FLAG_COLUMNS:
        population[column] = population[column].fillna(0.0).astype(float)

    recomputed_sum = population[list(COUNTED_FLAG_COLUMNS)].sum(axis=1)
    sum_mismatch = int(recomputed_sum.round(6).ne(population[FLAG_SUM_COLUMN].round(6)).sum())

    population["gated_flag_sum"] = population[FLAG_SUM_COLUMN] - population["flag_protection"] * (
        population["week"].astype(int).gt(EARLY_WINDOW_MAX_WEEK).astype(float)
    )

    population, reddit_coverage = add_reddit_column(population)

    base_oos, base_folds = loso(population, BASE_FEATURES)
    base_is, base_is_coefficients = in_sample_fit(population, BASE_FEATURES)
    base_is_accuracy = accuracy(
        pd.Series(base_is, index=population.index), population["home_covered"]
    )
    base_oos_accuracy_overall = accuracy(
        base_oos.loc[base_oos.notna()], population.loc[base_oos.notna(), "home_covered"]
    )

    features_gated = ("model_logit", "gated_flag_sum", MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)
    features_reddit = (*BASE_FEATURES, REDDIT_TERM)
    features_separate = ("model_logit", *kept_flags, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)

    looks = [
        run_look(
            "week_gated_protection_flag_sum",
            "PBP08 protection-mismatch flag edge reads concentrated in weeks 1-4 "
            "(prior-season carryover still unpriced); gating its contribution to "
            "weeks<=4 re-tests that read at 2.5x the games.",
            population,
            features_gated,
            base_oos,
            base_is_accuracy,
        ),
        run_look(
            "reddit_home_comment_ratio_elevated",
            "Elevated home-side Reddit comment-volume ratio proxies public/local "
            "attention not fully priced into the market-move term; re-tests a "
            "2020-2025-only unresolved read at 2.5x the games.",
            population,
            features_reddit,
            base_oos,
            base_is_accuracy,
        ),
        run_look(
            "composition_flags_separate",
            "An equal-weight sum can hide member-specific sign/magnitude "
            "differences (as week_gated_protection_flag_sum tests for one "
            "member); entering the members jointly lets the fit choose each "
            "member's own weight.",
            population,
            features_separate,
            base_oos,
            base_is_accuracy,
        ),
    ]

    summary = {
        "schema_version": 1,
        "created_at_utc": now.isoformat(),
        "command": "python scripts/extended_population_regrade.py",
        "population_path": str(POPULATION_PATH.relative_to(REPO_ROOT)),
        "population_games": len(population),
        "population_seasons": sorted(int(s) for s in population["season"].unique()),
        "base_features": list(BASE_FEATURES),
        "base_in_sample_accuracy": base_is_accuracy,
        "base_oos_accuracy_overall": base_oos_accuracy_overall,
        "base_fold_coefficients": base_folds,
        "base_in_sample_coefficients": base_is_coefficients,
        "flag_pre_2020_coverage": coverage,
        "flags_dropped_no_pre_2020_coverage": dropped_flags,
        "flags_kept_for_separate_look": kept_flags,
        "composition_flag_sum_reproduction_mismatch_rows": sum_mismatch,
        "reddit_coverage": reddit_coverage,
        "predeclared_family": "extended_population_regrade_2011_2025",
        "look_count": len(looks),
        "looks": looks,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    population.to_parquet(output_dir / "population_used.parquet", index=False)
    print(json.dumps({"output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
