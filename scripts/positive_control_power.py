import argparse
import json
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import pooled_signal_second_fit as base_fit  # noqa: E402

from nfl_ats.pick_probability_fit import FIT_RIDGE, build_fit_population  # noqa: E402

BASE_TERMS = base_fit.BASE_FEATURES
DEFAULT_FIT_ITERATIONS = 25
DEFAULT_SIMS = 200
DEFAULT_DRAWS = 400
DEFAULT_GRID = (0.10, 0.25, 0.45, 0.70, 1.00, 1.40, 1.80)
TARGET_POWER = 0.80
PREVALENCE_CASES = (
    ("binary_p03", "binary", 0.03),
    ("binary_p10", "binary", 0.10),
    ("binary_p50", "binary", 0.50),
    ("continuous_std", "continuous", None),
)
EXTENDED_POPULATION_PATH = (
    REPO / "artifacts" / "extended_fit_population" / "20260923T205910Z" / "population.parquet"
)


def season_block_bootstrap(frame, value_col, draws, seed):
    rng = np.random.default_rng(seed)
    seasons = sorted(int(value) for value in frame["season"].unique())
    grouped = {
        season: frame.loc[frame["season"].eq(season), value_col].to_numpy() for season in seasons
    }
    point = float(frame[value_col].mean())
    n = len(seasons)
    if n < 2:
        return point, point, point, 1.0 if point > 0 else 0.0
    effects = np.empty(draws)
    for draw in range(draws):
        picked = rng.integers(0, n, size=n)
        pooled = np.concatenate([grouped[seasons[i]] for i in picked])
        effects[draw] = pooled.mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    probability_positive = float((effects > 0.0).mean())
    return point, float(low), float(high), probability_positive


def synth_term(rng, n, kind, prevalence):
    if kind == "binary":
        return rng.binomial(1, prevalence, size=n).astype(float)
    return rng.standard_normal(n)


def standardize(values):
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0)) or 1.0
    return (values - mean) / std, mean, std


def run_cell(
    population,
    base_probability_dgp,
    kind,
    prevalence,
    coef,
    sims,
    draws,
    fit_iterations,
    seed,
):
    detections = 0
    points = []
    epsilon = 1e-6
    base_logit = np.log(
        np.clip(base_probability_dgp, epsilon, 1.0 - epsilon)
        / (1.0 - np.clip(base_probability_dgp, epsilon, 1.0 - epsilon))
    )
    n = len(population)
    for sim in range(sims):
        rng = np.random.default_rng(seed + sim)
        term_raw = synth_term(rng, n, kind, prevalence)
        term_std, _, _ = standardize(term_raw)
        synth_logit = np.clip(base_logit + coef * term_std, -35.0, 35.0)
        synth_prob = 1.0 / (1.0 + np.exp(-synth_logit))
        synth_outcome = rng.binomial(1, synth_prob).astype(float)

        sim_pop = population[["season", "week", *BASE_TERMS]].copy()
        sim_pop["synthetic_term"] = term_raw
        sim_pop["home_covered"] = synth_outcome

        base_oos, _, _, _ = base_fit.loso_fit(sim_pop, list(BASE_TERMS), FIT_RIDGE, fit_iterations)
        new_oos, _, _, _ = base_fit.loso_fit(
            sim_pop, [*BASE_TERMS, "synthetic_term"], FIT_RIDGE, fit_iterations
        )
        mask = base_oos.notna() & new_oos.notna()
        scored = sim_pop.loc[mask].copy()
        scored["base_correct"] = (
            base_oos.loc[mask].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
        )
        scored["new_correct"] = (
            new_oos.loc[mask].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
        )
        scored["diff_vs_four_term"] = scored["new_correct"] - scored["base_correct"]

        point, low, _high, _pp = season_block_bootstrap(
            scored, "diff_vs_four_term", draws, seed + 900000 + sim
        )
        points.append(point * 100.0)
        if low > 0.0:
            detections += 1

    detection_rate = detections / sims
    return {
        "coefficient_logit_sd": coef,
        "sims": sims,
        "detection_rate": detection_rate,
        "mean_effect_accuracy_points": float(np.mean(points)),
        "median_effect_accuracy_points": float(np.median(points)),
    }


def mde_from_grid(grid_results, target_power):
    ordered = sorted(grid_results, key=lambda row: row["mean_effect_accuracy_points"])
    if not ordered:
        return None, "no grid points evaluated"
    if ordered[0]["detection_rate"] >= target_power:
        return ordered[0]["mean_effect_accuracy_points"], "below smallest tested effect"
    if ordered[-1]["detection_rate"] < target_power:
        return None, "not reached within tested coefficient range"
    for lower, upper in pairwise(ordered):
        if lower["detection_rate"] < target_power <= upper["detection_rate"]:
            span = upper["detection_rate"] - lower["detection_rate"]
            if span <= 0:
                return upper["mean_effect_accuracy_points"], "interpolated (flat span)"
            frac = (target_power - lower["detection_rate"]) / span
            mde = lower["mean_effect_accuracy_points"] + frac * (
                upper["mean_effect_accuracy_points"] - lower["mean_effect_accuracy_points"]
            )
            return float(mde), "linear interpolation between grid points"
    return None, "power was not monotone in the tested grid"


def population_seasons_and_games(population, label):
    return {
        "label": label,
        "games": len(population),
        "seasons": sorted(int(value) for value in population["season"].unique()),
        "season_blocks": int(population["season"].nunique()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--fit-iterations", type=int, default=DEFAULT_FIT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--grid", default=",".join(str(value) for value in DEFAULT_GRID)
    )
    args = parser.parse_args(argv)
    grid = tuple(float(value) for value in args.grid.split(","))

    started = time.time()
    served_population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    extended_population = pd.read_parquet(EXTENDED_POPULATION_PATH)

    populations = {
        "served_2020_2025": served_population,
        "extended_2011_2025": extended_population,
    }

    results = {
        "command": "python scripts/positive_control_power.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sims_per_cell": args.sims,
        "bootstrap_draws": args.draws,
        "fit_iterations": args.fit_iterations,
        "seed": args.seed,
        "coefficient_grid_logit_sd": list(grid),
        "target_power": TARGET_POWER,
        "base_terms": list(BASE_TERMS),
        "populations": {},
        "population_provenance_served": provenance,
        "extended_population_path": str(EXTENDED_POPULATION_PATH.relative_to(REPO)).replace(
            "\\", "/"
        ),
        "cells": {},
        "mde_table": {},
    }

    for pop_label, population in populations.items():
        results["populations"][pop_label] = population_seasons_and_games(population, pop_label)
        base_fit_outputs = base_fit.loso_fit(
            population, list(BASE_TERMS), FIT_RIDGE, args.fit_iterations
        )
        base_in_sample = base_fit_outputs[2]
        for prev_label, kind, prevalence in PREVALENCE_CASES:
            cell_key = f"{pop_label}__{prev_label}"
            grid_rows = []
            for coef in grid:
                row = run_cell(
                    population,
                    base_in_sample,
                    kind,
                    prevalence,
                    coef,
                    args.sims,
                    args.draws,
                    args.fit_iterations,
                    args.seed,
                )
                grid_rows.append(row)
                print(
                    json.dumps(
                        {
                            "cell": cell_key,
                            "coef": coef,
                            "detection_rate": row["detection_rate"],
                            "mean_effect_accuracy_points": row["mean_effect_accuracy_points"],
                            "elapsed_s": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
            mde, mde_note = mde_from_grid(grid_rows, TARGET_POWER)
            results["cells"][cell_key] = {
                "population": pop_label,
                "prevalence_case": prev_label,
                "term_kind": kind,
                "prevalence": prevalence,
                "grid": grid_rows,
                "minimum_detectable_effect_accuracy_points_at_80pct_power": mde,
                "mde_note": mde_note,
            }
            results["mde_table"][cell_key] = mde

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "positive_control_power" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results["artifact"] = str(out_path.relative_to(REPO)).replace("\\", "/")
    results["elapsed_s"] = round(time.time() - started, 1)
    summary = {
        "artifact": results["artifact"],
        "mde_table": results["mde_table"],
        "elapsed_s": results["elapsed_s"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
