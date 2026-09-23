import argparse
import json
import sys
import time
from datetime import UTC, datetime
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import pooled_signal_second_fit as base_fit  # noqa: E402
from line_move_power import (  # noqa: E402
    PREVALENCE_CASES,
    TARGET_POWER,
    mde_from_grid,
    season_block_bootstrap,
    standardize,
    synth_term,
)

from nfl_ats.pick_probability_fit import FIT_RIDGE, build_fit_population  # noqa: E402

BASE_FEATURES = ("model_logit", "composition_flag_sum")
DEFAULT_FIT_ITERATIONS = 20
DEFAULT_SIMS = 150
DEFAULT_DRAWS = 300
DEFAULT_GRID_POINTS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35)


def run_cell(
    base_logit, open_move, seasons, kind, prevalence, coef_points, coef_logit, sims, draws, seed
):
    n = len(base_logit)
    base_pick_home = base_logit >= 0.0
    detections = 0
    points = []
    for sim in range(sims):
        rng = np.random.default_rng(seed + sim)
        term_raw = synth_term(rng, n, kind, prevalence)
        term_std, _, _ = standardize(term_raw)
        variant_logit = base_logit + coef_logit * term_std
        variant_pick_home = variant_logit >= 0.0
        synthetic_move = open_move + coef_points * term_std
        lm_base = np.where(base_pick_home, 1.0, -1.0) * synthetic_move
        lm_variant = np.where(variant_pick_home, 1.0, -1.0) * synthetic_move
        frame = pd.DataFrame({"season": seasons, "diff_line_move": lm_variant - lm_base})
        point, low, _high, _pp = season_block_bootstrap(
            frame, "diff_line_move", draws, seed + 900000 + sim
        )
        points.append(point)
        if low > 0.0:
            detections += 1
    detection_rate = detections / sims
    return {
        "coefficient_points": coef_points,
        "coefficient_logit_equivalent": coef_logit,
        "sims": sims,
        "detection_rate": detection_rate,
        "mean_effect_line_move_points": float(np.mean(points)),
        "median_effect_line_move_points": float(np.median(points)),
    }


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
    population = population.dropna(subset=["open_move", "margin_vs_open"]).reset_index(drop=True)

    margin_sd_points = float(population["margin_vs_open"].std(ddof=0))
    logistic_scale = margin_sd_points * sqrt(3.0) / pi

    base_oos, _, _, _ = base_fit.loso_fit(
        population, list(BASE_FEATURES), FIT_RIDGE, args.fit_iterations
    )
    mask = base_oos.notna()
    epsilon = 1e-6
    clipped = np.clip(base_oos.loc[mask].to_numpy(dtype=float), epsilon, 1.0 - epsilon)
    base_logit = np.log(clipped / (1.0 - clipped))
    open_move = population.loc[mask, "open_move"].astype(float).to_numpy()
    seasons = population.loc[mask, "season"].astype(int).to_numpy()

    results = {
        "command": "python scripts/line_move_power_direct.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "unit": "positive-control power unit 3: conversion-free direct line-move control",
        "base_features": list(BASE_FEATURES),
        "population_label": "served_2020_2025",
        "games": len(population),
        "games_scored": int(mask.sum()),
        "seasons": sorted(int(value) for value in population["season"].unique()),
        "season_blocks": int(population.loc[mask, "season"].nunique()),
        "sims_per_cell": args.sims,
        "bootstrap_draws": args.draws,
        "fit_iterations": args.fit_iterations,
        "seed": args.seed,
        "coefficient_grid_points": list(grid),
        "target_power": TARGET_POWER,
        "population_provenance": provenance,
        "margin_sd_points_empirical": margin_sd_points,
        "logistic_scale_points": logistic_scale,
        "design_note": (
            "Conversion-free control: the base model (model_logit, composition_flag_sum) is "
            "LOSO-refit exactly once against the REAL, unmodified 2020-2025 home_covered "
            "outcomes (no synthetic Bernoulli draw anywhere in this script). Its real "
            "out-of-fold logit is deterministic and reused across every simulation. Per "
            "simulation, a synthetic Tuesday-knowable term is drawn (binary at declared "
            "prevalence, or standard-normal continuous) and its only effect is a known, "
            "declared additive shift of coef_points line-move points added directly to the "
            "REAL close-minus-open move series (synthetic_move = real open_move + "
            "coef_points * standardized_term), together with a matched shift of the real "
            "base logit by coef_logit = coef_points / logistic_scale standardized-term units. "
            "logistic_scale converts points to logit units via the closed-form variance match "
            "between a logistic distribution and a Normal(0, margin_sd_points) using the "
            "margin_sd_points_empirical field, which is the plain standard deviation of the "
            "real 2020-2025 margin-vs-open-spread series (a descriptive statistic of one "
            "variable, not a regression slope fit between two variables as unit 2's "
            "points_per_base_logit_sd_empirical was) -- this is the sense in which the "
            "control is conversion-free relative to unit 2. The variant pick is a "
            "deterministic threshold (variant_logit >= 0), never a redrawn outcome, so no "
            "synthetic cover outcome is ever generated, and no accuracy-point conversion is "
            "computed or reported; the MDE is read directly in line-move points."
        ),
        "cells": {},
        "mde_table_line_move_points": {},
    }

    for prev_label, kind, prevalence in PREVALENCE_CASES:
        grid_rows = []
        for coef_points in grid:
            coef_logit = coef_points / logistic_scale if logistic_scale else 0.0
            row = run_cell(
                base_logit,
                open_move,
                seasons,
                kind,
                prevalence,
                coef_points,
                coef_logit,
                args.sims,
                args.draws,
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
        results["cells"][prev_label] = {
            "prevalence_case": prev_label,
            "term_kind": kind,
            "prevalence": prevalence,
            "grid": grid_rows,
            "minimum_detectable_effect_line_move_points_at_80pct_power": mde,
            "mde_note": mde_note,
        }
        results["mde_table_line_move_points"][prev_label] = mde

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "line_move_power_direct" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results["artifact"] = str(out_path.relative_to(REPO)).replace("\\", "/")
    results["elapsed_s"] = round(time.time() - started, 1)
    summary = {
        "artifact": results["artifact"],
        "mde_table_line_move_points": results["mde_table_line_move_points"],
        "margin_sd_points_empirical": margin_sd_points,
        "elapsed_s": results["elapsed_s"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
