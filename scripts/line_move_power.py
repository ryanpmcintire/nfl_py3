import argparse
import json
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import pooled_signal_second_fit as base_fit  # noqa: E402

from nfl_ats.pick_probability_fit import FIT_RIDGE, build_fit_population  # noqa: E402

BASE_FEATURES = ("model_logit", "composition_flag_sum")
DEFAULT_FIT_ITERATIONS = 20
DEFAULT_SIMS = 150
DEFAULT_DRAWS = 300
DEFAULT_GRID_POINTS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
TARGET_POWER = 0.80
PREVALENCE_CASES = (
    ("binary_p03", "binary", 0.03),
    ("binary_p10", "binary", 0.10),
    ("binary_p50", "binary", 0.50),
    ("continuous_std", "continuous", None),
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


def points_per_logit_sd(base_logit, open_move):
    logit_std, _, logit_sd = standardize(base_logit)
    move = np.asarray(open_move, dtype=float)
    move_centered = move - float(np.mean(move))
    denominator = float(np.sum(logit_std * logit_std))
    slope = float(np.sum(logit_std * move_centered) / denominator) if denominator else 0.0
    return slope, logit_sd


def accuracy_points_per_move_point(population, base_oos, open_move):
    mask = base_oos.notna()
    pick_home = base_oos.loc[mask].ge(0.5).to_numpy()
    home_covered = population.loc[mask, "home_covered"].astype(float).to_numpy()
    correct = (pick_home.astype(float) == home_covered).astype(float)
    move_signed = np.where(pick_home, 1.0, -1.0) * open_move[mask.to_numpy()]
    x = move_signed - float(np.mean(move_signed))
    y = correct - float(np.mean(correct))
    denominator = float(np.sum(x * x))
    slope_probability_per_point = float(np.sum(x * y) / denominator) if denominator else 0.0
    return slope_probability_per_point * 100.0, int(mask.sum())


def run_cell(
    population,
    base_logit,
    open_move,
    points_slope,
    kind,
    prevalence,
    coef_points,
    sims,
    draws,
    fit_iterations,
    seed,
):
    coef_logit = coef_points / points_slope if points_slope else 0.0
    detections = 0
    points = []
    n = len(population)
    for sim in range(sims):
        rng = np.random.default_rng(seed + sim)
        term_raw = synth_term(rng, n, kind, prevalence)
        term_std, _, _ = standardize(term_raw)
        synth_logit = np.clip(base_logit + coef_logit * term_std, -35.0, 35.0)
        synth_prob = 1.0 / (1.0 + np.exp(-synth_logit))
        synth_outcome = rng.binomial(1, synth_prob).astype(float)
        synthetic_move = open_move + coef_points * term_std

        sim_pop = population[["season", "week", *BASE_FEATURES]].copy()
        sim_pop["synthetic_term"] = term_raw
        sim_pop["home_covered"] = synth_outcome

        base_oos, _, _, _ = base_fit.loso_fit(
            sim_pop, list(BASE_FEATURES), FIT_RIDGE, fit_iterations
        )
        variant_oos, _, _, _ = base_fit.loso_fit(
            sim_pop, [*BASE_FEATURES, "synthetic_term"], FIT_RIDGE, fit_iterations
        )
        mask = base_oos.notna() & variant_oos.notna()
        scored = sim_pop.loc[mask].copy()
        scored["synthetic_move"] = synthetic_move[mask.to_numpy()]
        base_pick_home = base_oos.loc[mask].ge(0.5)
        variant_pick_home = variant_oos.loc[mask].ge(0.5)
        lm_base = np.where(base_pick_home, 1.0, -1.0) * scored["synthetic_move"]
        lm_variant = np.where(variant_pick_home, 1.0, -1.0) * scored["synthetic_move"]
        scored["diff_line_move"] = lm_variant - lm_base

        point, low, _high, _pp = season_block_bootstrap(
            scored, "diff_line_move", draws, seed + 900000 + sim
        )
        points.append(point)
        if low > 0.0:
            detections += 1

    detection_rate = detections / sims
    return {
        "coefficient_points": coef_points,
        "coefficient_logit_sd_equivalent": coef_logit,
        "sims": sims,
        "detection_rate": detection_rate,
        "mean_effect_line_move_points": float(np.mean(points)),
        "median_effect_line_move_points": float(np.median(points)),
    }


def mde_from_grid(grid_results, target_power):
    ordered = sorted(grid_results, key=lambda row: row["mean_effect_line_move_points"])
    if not ordered:
        return None, "no grid points evaluated"
    if ordered[0]["detection_rate"] >= target_power:
        return ordered[0]["mean_effect_line_move_points"], "below smallest tested effect"
    if ordered[-1]["detection_rate"] < target_power:
        return None, "not reached within tested coefficient range"
    for lower, upper in pairwise(ordered):
        if lower["detection_rate"] < target_power <= upper["detection_rate"]:
            span = upper["detection_rate"] - lower["detection_rate"]
            if span <= 0:
                return upper["mean_effect_line_move_points"], "interpolated (flat span)"
            frac = (target_power - lower["detection_rate"]) / span
            mde = lower["mean_effect_line_move_points"] + frac * (
                upper["mean_effect_line_move_points"] - lower["mean_effect_line_move_points"]
            )
            return float(mde), "linear interpolation between grid points"
    return None, "power was not monotone in the tested grid"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--fit-iterations", type=int, default=DEFAULT_FIT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--grid", default=",".join(str(value) for value in DEFAULT_GRID_POINTS)
    )
    args = parser.parse_args(argv)
    grid = tuple(float(value) for value in args.grid.split(","))

    started = time.time()
    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    population = population.dropna(subset=["open_move"]).reset_index(drop=True)

    base_fit_outputs = base_fit.loso_fit(
        population, list(BASE_FEATURES), FIT_RIDGE, args.fit_iterations
    )
    base_oos_real = base_fit_outputs[0]
    base_in_sample = base_fit_outputs[2]
    epsilon = 1e-6
    clipped = np.clip(base_in_sample, epsilon, 1.0 - epsilon)
    base_logit = np.log(clipped / (1.0 - clipped))
    open_move = population["open_move"].astype(float).to_numpy()

    points_slope, logit_sd = points_per_logit_sd(base_logit, open_move)
    accuracy_pts_per_move_pt, conversion_games = accuracy_points_per_move_point(
        population, base_oos_real, open_move
    )

    results = {
        "command": "python scripts/line_move_power.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "unit": "positive-control power unit 2: line-move yardstick MDE",
        "base_features": list(BASE_FEATURES),
        "population_label": "served_2020_2025",
        "games": len(population),
        "seasons": sorted(int(value) for value in population["season"].unique()),
        "season_blocks": int(population["season"].nunique()),
        "sims_per_cell": args.sims,
        "bootstrap_draws": args.draws,
        "fit_iterations": args.fit_iterations,
        "seed": args.seed,
        "coefficient_grid_points": list(grid),
        "target_power": TARGET_POWER,
        "population_provenance": provenance,
        "points_per_base_logit_sd_empirical": points_slope,
        "base_logit_sd_empirical": logit_sd,
        "injection_note": (
            "coefficient is declared in line-move points; the matching logit-SD shift "
            "used to make the synthetic term genuinely informative about the synthetic "
            "outcome is derived by dividing the points coefficient by the empirical "
            "points-per-logit-SD slope measured on the real population (points_per_"
            "base_logit_sd_empirical), so a term with a real relationship to the "
            "subsequent move of the declared point size also carries a matched, "
            "non-arbitrary relationship to the synthetic outcome."
        ),
        "accuracy_points_per_line_move_point_empirical": accuracy_pts_per_move_pt,
        "accuracy_conversion_games": conversion_games,
        "accuracy_conversion_note": (
            "OLS slope of pick-correct (0/1) on the base model's real signed line move "
            "toward its own real pick, times 100, using the real 2020-2025 population "
            "and its real outcomes (no synthetic injection); used only to convert each "
            "line-move MDE into an implied accuracy-point equivalent, not to detect "
            "anything."
        ),
        "cells": {},
        "mde_table_line_move_points": {},
        "mde_table_accuracy_equivalent": {},
    }

    for prev_label, kind, prevalence in PREVALENCE_CASES:
        grid_rows = []
        for coef_points in grid:
            row = run_cell(
                population,
                base_logit,
                open_move,
                points_slope,
                kind,
                prevalence,
                coef_points,
                args.sims,
                args.draws,
                args.fit_iterations,
                args.seed,
            )
            grid_rows.append(row)
            print(
                json.dumps(
                    {
                        "cell": prev_label,
                        "coef_points": coef_points,
                        "detection_rate": row["detection_rate"],
                        "mean_effect_line_move_points": row["mean_effect_line_move_points"],
                        "elapsed_s": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
        mde, mde_note = mde_from_grid(grid_rows, TARGET_POWER)
        accuracy_equivalent = mde * accuracy_pts_per_move_pt if mde is not None else None
        results["cells"][prev_label] = {
            "prevalence_case": prev_label,
            "term_kind": kind,
            "prevalence": prevalence,
            "grid": grid_rows,
            "minimum_detectable_effect_line_move_points_at_80pct_power": mde,
            "mde_note": mde_note,
            "accuracy_point_equivalent": accuracy_equivalent,
        }
        results["mde_table_line_move_points"][prev_label] = mde
        results["mde_table_accuracy_equivalent"][prev_label] = accuracy_equivalent

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "line_move_power" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results["artifact"] = str(out_path.relative_to(REPO)).replace("\\", "/")
    results["elapsed_s"] = round(time.time() - started, 1)
    summary = {
        "artifact": results["artifact"],
        "mde_table_line_move_points": results["mde_table_line_move_points"],
        "mde_table_accuracy_equivalent": results["mde_table_accuracy_equivalent"],
        "accuracy_points_per_line_move_point_empirical": accuracy_pts_per_move_pt,
        "elapsed_s": results["elapsed_s"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
