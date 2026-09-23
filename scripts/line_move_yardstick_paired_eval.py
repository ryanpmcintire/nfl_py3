from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.pick_probability_fit import (  # noqa: E402
    FIT_FEATURES,
    FIT_RIDGE,
    _fit_logit,
    build_fit_population,
)

OPENER_ONLY_FEATURES = ("model_logit", "composition_flag_sum")
BOOTSTRAP_DRAWS = 2000


def _standardise(
    train: pd.DataFrame, features: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    means = {name: float(train[name].mean()) for name in features}
    stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in features}
    return means, stds


def _design(
    frame: pd.DataFrame, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    columns = [np.ones(len(frame))]
    for name in features:
        columns.append(((frame[name].astype(float) - means[name]) / stds[name]).to_numpy())
    return np.column_stack(columns)


def _predict(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    beta: np.ndarray,
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    z = np.clip(_design(frame, features, means, stds) @ beta, -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def loso_probabilities(population: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=population.index, dtype=float)
    for held in sorted(int(value) for value in population["season"].unique()):
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = _standardise(train, features)
        beta = _fit_logit(
            _design(train, features, means, stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        out.loc[test.index] = _predict(test, features, beta, means, stds)
    return out


def block_bootstrap(
    frame: pd.DataFrame, block_cols: list[str], value_col: str, draws: int, seed: int
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    blocks = frame.groupby(block_cols, sort=False).indices
    positions = list(blocks.values())
    n = len(positions)
    values = frame[value_col].to_numpy()
    point = float(values.mean())
    effects = np.empty(draws)
    for draw in range(draws):
        picked = rng.integers(0, n, size=n)
        idx = np.concatenate([positions[i] for i in picked])
        effects[draw] = values[idx].mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    probability_positive = float((effects > 0.0).mean())
    return point, float(low), float(high), probability_positive


def cell_stats(
    label: str, frame: pd.DataFrame, diff_col: str, seed: int, draws: int
) -> dict[str, object]:
    season_point, season_low, season_high, season_pp = block_bootstrap(
        frame, ["season"], diff_col, draws, seed
    )
    _week_point, week_low, week_high, week_pp = block_bootstrap(
        frame, ["season", "week"], diff_col, draws, seed + 1
    )
    decisive = frame.loc[frame[diff_col].ne(0.0)]
    per_season = (
        frame.groupby("season")[diff_col]
        .agg(["mean", "size"])
        .rename(columns={"mean": "mean_points", "size": "games"})
    )
    return {
        "label": label,
        "games": len(frame),
        "mean_points": season_point,
        "season_block_interval_low": season_low,
        "season_block_interval_high": season_high,
        "season_block_probability_positive": season_pp,
        "week_block_interval_low": week_low,
        "week_block_interval_high": week_high,
        "week_block_probability_positive": week_pp,
        "sample_blocks_season": int(frame["season"].nunique()),
        "decisive_games": len(decisive),
        "decisive_mean_points": float(decisive[diff_col].mean()) if len(decisive) else 0.0,
        "decisive_side_a_wins": int((decisive[diff_col] > 0.0).sum()),
        "decisive_side_b_wins": int((decisive[diff_col] < 0.0).sum()),
        "per_season": {
            str(season): {
                "mean_points": float(row["mean_points"]),
                "games": int(row["games"]),
            }
            for season, row in per_season.iterrows()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)

    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    population["served_probability"] = loso_probabilities(population, FIT_FEATURES)
    population["opener_only_probability"] = loso_probabilities(population, OPENER_ONLY_FEATURES)

    population["served_pick_home"] = population["served_probability"].ge(0.5)
    population["opener_only_pick_home"] = population["opener_only_probability"].ge(0.5)
    population["model_only_pick_home"] = population["model_pick_home"]
    population["always_fav_pick_home"] = population["tue_open_home_spread"].lt(0.0)
    pickem = population["tue_open_home_spread"].eq(0.0)

    for name in ("served", "opener_only", "model_only", "always_fav"):
        pick_col = f"{name}_pick_home"
        population[f"lm_{name}"] = (
            np.where(population[pick_col], 1.0, -1.0) * population["open_move"]
        )

    population["diff_served_vs_model_only"] = population["lm_served"] - population["lm_model_only"]
    population["diff_model_only_vs_always_fav"] = (
        population["lm_model_only"] - population["lm_always_fav"]
    )
    population["diff_served_vs_always_fav"] = population["lm_served"] - population["lm_always_fav"]
    population["diff_opener_only_vs_model_only"] = (
        population["lm_opener_only"] - population["lm_model_only"]
    )
    population["diff_opener_only_vs_always_fav"] = (
        population["lm_opener_only"] - population["lm_always_fav"]
    )

    non_pickem = population.loc[~pickem]

    cells = [
        cell_stats(
            "served_vs_model_only_CONTAMINATED",
            population,
            "diff_served_vs_model_only",
            args.seed,
            args.draws,
        ),
        cell_stats(
            "served_vs_always_favourite_CONTAMINATED",
            non_pickem,
            "diff_served_vs_always_fav",
            args.seed,
            args.draws,
        ),
        cell_stats(
            "model_only_vs_always_favourite",
            non_pickem,
            "diff_model_only_vs_always_fav",
            args.seed,
            args.draws,
        ),
        cell_stats(
            "opener_only_vs_model_only",
            population,
            "diff_opener_only_vs_model_only",
            args.seed,
            args.draws,
        ),
        cell_stats(
            "opener_only_vs_always_favourite",
            non_pickem,
            "diff_opener_only_vs_always_fav",
            args.seed,
            args.draws,
        ),
    ]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = REPO / "artifacts" / "line_move_yardstick" / stamp
    directory.mkdir(parents=True, exist_ok=True)
    results = {
        "command": "python scripts/line_move_yardstick_paired_eval.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "opener_evaluation_source": provenance["opener_evaluation"],
        "population_games": len(population),
        "pickem_games_excluded_for_favourite_cells": int(pickem.sum()),
        "provenance": provenance,
        "contamination_finding": (
            "The four-term served probability's FIT_FEATURES include "
            "market_move_toward_home / market_move_available, built from "
            "leader_median_through_sunday_prekick_v1 sharp-book movement measured "
            "AFTER the Tuesday opener and up through Sunday pre-kick. Any "
            "opener-to-close line_move_toward_pick comparison that includes the "
            "served four-term side is measuring the side partly against "
            "information the side's own market-move term already consumed. "
            "served_vs_model_only and served_vs_always_favourite are CONTAMINATED "
            "for this yardstick; opener_only_vs_model_only and "
            "opener_only_vs_always_favourite substitute a LOSO refit of the same "
            "logit using only (model_logit, composition_flag_sum), with the two "
            "market-move terms dropped, as the opener-only variant."
        ),
        "seed": args.seed,
        "bootstrap_draws": args.draws,
        "cells": cells,
    }
    (directory / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifact": str(directory.relative_to(REPO)), **results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
