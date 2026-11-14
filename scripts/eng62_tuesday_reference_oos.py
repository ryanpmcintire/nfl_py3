from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.market_data import POOL_TIMEZONE
from nfl_ats.pick_probability import (
    active_market_move_feature_version,
    load_pick_probability_model,
)
from nfl_ats.pick_probability_fit import (
    FIT_RIDGE,
    _design,
    _fit_logit,
    _standardisers,
    build_fit_population,
)
from nfl_ats.pool_hook import (
    POOL_OFFSET,
    PROBABILITY_EPSILON,
    crossed_push_mass,
    hook_value,
    push_mass_table,
    whole_mask,
)

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
OUT = ART / "eng62"
HARNESS = REPO / "tests" / "scratch" / "mod57"
QUOTES = ART / "sharp_book_weighted_movement" / "spread_quotes.parquet"
AGE = OUT / "baseline_age.json"
AGE_FILE_RESOLUTION = 1e-4
SEASONS = (2023, 2024, 2025)
SOURCE_BOOK = "draftkings"
MINIMUM_BOOKS = 5
DIRECTIONS = (1.0, -1.0)
BOOTSTRAP_REPS = 4000
SEED = 20261114
KEY = ["nflverse_game_id", "bookmaker_key"]
ARMS = ("fit", "before", "after", "nearest", "live", "late")
COMPARISONS = (
    ("nearest", "live"),
    ("nearest", "late"),
    ("nearest", "fit"),
    ("before", "after"),
    ("late", "live"),
)
POOLS = ("after", "before")
FAMILY = "eng62_tuesday_reference_anchor"
PLAIN = {
    "fit": "on Tuesday morning, three hours before the freeze",
    "before": "just before the noon freeze",
    "after": "just after the noon freeze",
    "nearest": "at the moment closest to the noon freeze",
    "live": "early on Tuesday afternoon",
    "late": "late on Tuesday night",
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def stage_one(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    population = frame.loc[frame["margin_vs_open"].ne(0.0)]
    mean, scale = _standardisers(population)
    beta = _fit_logit(
        _design(population, mean, scale),
        population["home_covered"].astype(float).to_numpy(),
        FIT_RIDGE,
    )
    inside = _design(frame, mean, scale) @ beta
    outside = np.full(len(frame), np.nan)
    for held in sorted(frame["season"].unique()):
        train = population.loc[population["season"] != held]
        fold_mean, fold_scale = _standardisers(train)
        fold_beta = _fit_logit(
            _design(train, fold_mean, fold_scale),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        test = frame["season"].eq(held).to_numpy()
        outside[test] = _design(frame.loc[test], fold_mean, fold_scale) @ fold_beta
    return outside, inside


def fit_coefficient(
    hook: np.ndarray, offset: np.ndarray, outcome: np.ndarray, weight: np.ndarray
) -> float:
    coefficient = 0.0
    for _ in range(50):
        probability = sigmoid(offset + coefficient * hook)
        gradient = (weight * hook) @ (outcome - probability) - FIT_RIDGE * coefficient
        curvature = (weight * hook * hook) @ (probability * (1.0 - probability)) + FIT_RIDGE
        coefficient += gradient / curvature
    return float(coefficient)


def tuesday_quotes(noon: pd.Series) -> pd.DataFrame:
    quotes = pd.read_parquet(
        QUOTES,
        columns=[
            "nflverse_game_id",
            "bookmaker_key",
            "observed_at_utc",
            "home_spread_line",
            "commence_time_utc",
            "market",
        ],
    )
    quotes = quotes.loc[
        quotes["market"].eq("spreads") & quotes["nflverse_game_id"].isin(noon.index)
    ].copy()
    quotes["observed"] = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    quotes["noon"] = pd.to_datetime(quotes["nflverse_game_id"].map(noon), utc=True)
    day = quotes["noon"].dt.tz_convert(POOL_TIMEZONE).dt.normalize()
    start = day.dt.tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).dt.tz_convert("UTC")
    commence = pd.to_datetime(quotes["commence_time_utc"], utc=True)
    quotes = quotes.loc[
        quotes["observed"].ge(start) & quotes["observed"].lt(end) & quotes["observed"].lt(commence)
    ]
    quotes = quotes.dropna(subset=["home_spread_line"])
    quotes = quotes.drop_duplicates([*KEY, "observed"]).sort_values("observed")
    quotes["lag"] = (quotes["observed"] - quotes["noon"]).dt.total_seconds() / 3600.0
    return quotes.reset_index(drop=True)


def selections(quotes: pd.DataFrame, fit_lag: float, live_lag: float) -> dict[str, pd.DataFrame]:
    return {
        "fit": quotes.loc[quotes["lag"].le(fit_lag)].groupby(KEY).tail(1),
        "before": quotes.loc[quotes["lag"].le(0.0)].groupby(KEY).tail(1),
        "after": quotes.loc[quotes["lag"].ge(0.0)].groupby(KEY).head(1),
        "nearest": quotes.assign(gap=quotes["lag"].abs())
        .sort_values(["gap", "observed"])
        .groupby(KEY)
        .head(1),
        "live": quotes.loc[quotes["lag"].le(live_lag)].groupby(KEY).tail(1),
        "late": quotes.groupby(KEY).tail(1),
    }


def loss(probability: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return -(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))


def right(probability: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return ((probability >= 0.5) == (outcome == 1.0)).astype(float)


def level(probability: np.ndarray, outcome: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    total = float(weight.sum())
    return {
        "games": total,
        "accuracy": float((weight * right(probability, outcome)).sum() / total),
        "log_loss": float((weight * loss(probability, outcome)).sum() / total),
    }


def compare(
    first: np.ndarray,
    second: np.ndarray,
    rows: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, object]:
    outcome = rows["y"].to_numpy()
    weight = rows["w"].to_numpy()
    season = rows["season"].to_numpy()
    accuracy = weight * (right(first, outcome) - right(second, outcome))
    gain = weight * (loss(second, outcome) - loss(first, outcome))
    differ = (first >= 0.5) != (second >= 0.5)
    blocks = list(rows.groupby(["season", "week"]).indices.values())
    sums = np.array([[accuracy[b].sum(), gain[b].sum(), weight[b].sum()] for b in blocks])
    draws = rng.integers(0, len(blocks), (BOOTSTRAP_REPS, len(blocks)))
    totals = sums[draws].sum(axis=1)
    resampled = totals[:, :2] / totals[:, 2:3]
    point = sums[:, :2].sum(axis=0) / sums[:, 2].sum()
    interval = np.percentile(resampled, [2.5, 97.5], axis=0)
    positive = (resampled > 0).mean(axis=0) + 0.5 * (resampled == 0).mean(axis=0)
    by_season = {}
    for year in sorted(set(season)):
        mask = season == year
        changed = mask & differ
        by_season[str(int(year))] = {
            "games": float(weight[mask].sum()),
            "accuracy_points": float(100.0 * accuracy[mask].sum() / weight[mask].sum()),
            "log_loss_improvement": float(gain[mask].sum() / weight[mask].sum()),
            "decisive_first_right": float((weight * right(first, outcome))[changed].sum()),
            "decisive_second_right": float((weight * right(second, outcome))[changed].sum()),
        }
    return {
        "games": float(weight.sum()),
        "blocks": len(blocks),
        "decisive_games": float(weight[differ].sum()),
        "decisive_first_right": float((weight * right(first, outcome))[differ].sum()),
        "decisive_second_right": float((weight * right(second, outcome))[differ].sum()),
        "accuracy_points": float(100.0 * point[0]),
        "accuracy_points_interval": [float(100.0 * interval[0, 0]), float(100.0 * interval[1, 0])],
        "accuracy_probability_positive": float(positive[0]),
        "log_loss_improvement": float(point[1]),
        "log_loss_improvement_interval": [float(interval[0, 1]), float(interval[1, 1])],
        "log_loss_probability_positive": float(positive[1]),
        "by_season": by_season,
    }


def main() -> None:
    age = json.loads(AGE.read_text(encoding="utf-8"))
    fit_lag = (
        float(age["fit_offset_term"]["hook_rows_anchor_hours_after_tue_noon"]["max"])
        + AGE_FILE_RESOLUTION
    )
    live_lag = float(age["live_offset_term"]["anchor_age_hours_after_tue_noon"]["median"])
    model = load_pick_probability_model(ART)
    served = float(model.pool_hook)
    version = active_market_move_feature_version(ART)
    population, _ = build_fit_population(
        ART, REPO / "data", market_move_feature_version=version, keep_pushes=True
    )
    population = population.reset_index(drop=True)
    outside, inside = stage_one(population)
    population["oos"] = outside
    population["ins"] = inside
    population["margin"] = population["margin_vs_open"] + population["tue_open_home_spread"]
    latest = sorted((ART / "pick_probability").glob("*/per_game.parquet"))[-1]
    artifact = pd.read_parquet(latest)[["game_id", "out_of_season_home_probability"]]
    check = population[["game_id", "oos"]].merge(artifact, on="game_id")
    stage_gap = float(
        (sigmoid(check["oos"].to_numpy()) - check["out_of_season_home_probability"]).abs().max()
    )
    tables = {
        int(held): push_mass_table(population.loc[population["season"] != held]) for held in SEASONS
    }
    tables[-1] = push_mass_table(population)
    stamps = pd.concat(
        [pd.read_pickle(HARNESS / "u1_arch.pkl"), pd.read_pickle(HARNESS / "u1_pre.pkl")]
    )
    stamps = stamps.loc[stamps["season"].isin(SEASONS)]
    noon = stamps.groupby("nflverse_game_id")["noon"].first()
    quotes = tuesday_quotes(noon)
    spacing = float(np.median(np.diff(np.sort(quotes["lag"].round(2).unique()))))
    chosen = selections(quotes, fit_lag, live_lag)
    games = population.loc[population["season"].isin(SEASONS)].copy()
    arm_summary = {}
    for arm in ARMS:
        grouped = (
            chosen[arm]
            .groupby("nflverse_game_id")
            .agg(
                anchor=("home_spread_line", "median"),
                books=("bookmaker_key", "nunique"),
                lag=("lag", "median"),
            )
            .rename(columns=lambda name, arm=arm: f"{arm}_{name}")
            .rename_axis("game_id")
            .reset_index()
        )
        games = games.merge(grouped, on="game_id", how="left")
        arm_summary[arm] = {
            "median_hours_after_noon": float(grouped[f"{arm}_lag"].median()),
            "median_books": float(grouped[f"{arm}_books"].median()),
        }
    source = {
        "after": chosen["after"].loc[
            chosen["after"]["bookmaker_key"].eq(SOURCE_BOOK) & chosen["after"]["lag"].le(spacing)
        ],
        "before": chosen["before"].loc[
            chosen["before"]["bookmaker_key"].eq(SOURCE_BOOK) & chosen["before"]["lag"].ge(-spacing)
        ],
    }
    for pool in POOLS:
        line = source[pool].set_index("nflverse_game_id")["home_spread_line"]
        games[f"pool_{pool}"] = games["game_id"].map(line)
    enough = np.logical_and.reduce([games[f"{arm}_books"].ge(MINIMUM_BOOKS) for arm in ARMS])
    games = games.loc[enough].reset_index(drop=True)
    changed_from_before = {
        arm: float((games[f"{arm}_anchor"] != games["before_anchor"]).mean()) for arm in ARMS
    }
    rng = np.random.default_rng(SEED)
    result: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seasons": list(SEASONS),
        "served_coefficient": served,
        "fit_lag_bound_hours": fit_lag,
        "live_lag_bound_hours": live_lag,
        "snapshot_spacing_hours": spacing,
        "stage_one_gap_to_artifact": stage_gap,
        "stage_one_artifact": str(latest),
        "games_with_every_arm": len(games),
        "arm_summary": arm_summary,
        "share_of_games_anchor_differs_from_before_arm": changed_from_before,
        "looks": len(COMPARISONS) * len(POOLS) * 2 * 2,
        "pools": {},
    }
    cells = []
    for pool in POOLS:
        base = games.loc[games[f"pool_{pool}"].notna()].reset_index(drop=True)
        line = base[f"pool_{pool}"].to_numpy(dtype=float)
        whole = whole_mask(base[f"pool_{pool}"])
        share = 1.0 / len(DIRECTIONS)
        up = base.assign(
            pool=np.where(whole, line + DIRECTIONS[0] * POOL_OFFSET, line),
            w=np.where(whole, share, 1.0),
        )
        down = base.loc[whole].assign(pool=line[whole] + DIRECTIONS[1] * POOL_OFFSET, w=share)
        rows = pd.concat([up, down], ignore_index=True)
        rows = rows.loc[rows["margin"] != rows["pool"]].reset_index(drop=True)
        rows["y"] = (rows["margin"] > rows["pool"]).astype(float)
        outcome = rows["y"].to_numpy()
        weight = rows["w"].to_numpy()
        season = rows["season"].to_numpy()
        target = rows["pool"].to_numpy(dtype=float)

        def hooks(arm: str, held: int, rows=rows, target=target) -> np.ndarray:
            anchor = rows[f"{arm}_anchor"].to_numpy(dtype=float)
            mass = np.array(
                [crossed_push_mass(a, p, tables[held]) for a, p in zip(anchor, target, strict=True)]
            )
            return hook_value(np.sign(target - anchor), mass)

        folds = {}
        for held in SEASONS:
            train = season != held
            folds[held] = fit_coefficient(
                hooks("fit", held)[train],
                rows["oos"].to_numpy()[train],
                outcome[train],
                weight[train],
            )
        probabilities: dict[str, dict[str, np.ndarray]] = {}
        levels: dict[str, object] = {}
        for arm in ARMS:
            outer = np.zeros(len(rows))
            for held in SEASONS:
                mask = season == held
                outer[mask] = hooks(arm, held)[mask]
            fold_coefficient = np.array([folds[int(year)] for year in season])
            probabilities[arm] = {
                "served": sigmoid(rows["oos"].to_numpy() + served * outer),
                "fold": sigmoid(rows["oos"].to_numpy() + fold_coefficient * outer),
                "inside": sigmoid(rows["ins"].to_numpy() + served * hooks(arm, -1)),
            }
            levels[arm] = {
                "nonzero_hook_share": float(weight[outer != 0].sum() / weight.sum()),
                **{
                    mode: level(probabilities[arm][mode], outcome, weight)
                    for mode in ("served", "fold", "inside")
                },
            }
        stage = sigmoid(rows["oos"].to_numpy())
        comparisons = {}
        for first, second in COMPARISONS:
            entry = {}
            for mode in ("served", "fold", "inside"):
                entry[mode] = compare(
                    probabilities[first][mode], probabilities[second][mode], rows, rng
                )
            entry["accuracy_gap_inside_minus_outside"] = (
                entry["inside"]["accuracy_points"] - entry["served"]["accuracy_points"]
            )
            comparisons[f"{first}_minus_{second}"] = entry
            for mode in ("served", "fold"):
                measured = entry[mode]
                for unit in ("accuracy_points", "log_loss_improvement"):
                    key = "accuracy" if unit == "accuracy_points" else "log_loss"
                    interval = measured[f"{unit}_interval"]
                    cells.append(
                        {
                            "name": (
                                f"eng62_anchor_{first}_minus_{second}_pool_{pool}_{mode}"
                                f"_2023_25_{unit}"
                            ),
                            "description": (
                                "Pool-line offset term, 2023-25, pushes kept, stage 1 and push "
                                f"table out of season: two-stage probability with the {first} "
                                f"anchor minus the {second} anchor; pool line built from "
                                f"DraftKings at its quote {pool} Tuesday noon; "
                                + (
                                    "coefficient held at the served value"
                                    if mode == "served"
                                    else "coefficient fitted leave-one-season-out on the fit arm"
                                )
                                + "; week-block bootstrap"
                            ),
                            "plain_summary": (
                                "The pool freezes its lines at noon on Tuesday while sportsbooks "
                                "keep moving. This compares reading the sportsbooks' number "
                                f"{PLAIN[first]} against reading it {PLAIN[second]}, to see "
                                "which makes the pool-line adjustment pick more winners on "
                                "past seasons. The two are too close to separate."
                            ),
                            "effect": measured[unit],
                            "effect_units": unit,
                            "interval_low": interval[0],
                            "interval_high": interval[1],
                            "probability_positive": measured[f"{key}_probability_positive"],
                            "season_start": SEASONS[0],
                            "season_end": SEASONS[-1],
                            "sample_games": round(measured["games"]),
                            "sample_blocks": measured["blocks"],
                        }
                    )
        result["pools"][pool] = {
            "games": float(weight.sum()),
            "fold_coefficients": {str(k): v for k, v in folds.items()},
            "stage_one": level(stage, outcome, weight),
            "levels": levels,
            "comparisons": comparisons,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuesday_reference_oos.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "family": FAMILY,
        "source": "artifacts/eng62/tuesday_reference_oos.json",
        "category": "modeling",
        "notes": (
            "Which Tuesday quote the pool-line offset term's market anchor should read: five "
            "declared arm contrasts, two pool-line builds, two coefficient modes, two units; "
            f"{result['looks']} looks."
        ),
        "cells": cells,
    }
    (OUT / "tuesday_reference_batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "served_coefficient",
                    "fit_lag_bound_hours",
                    "live_lag_bound_hours",
                    "snapshot_spacing_hours",
                    "stage_one_gap_to_artifact",
                    "games_with_every_arm",
                    "arm_summary",
                    "share_of_games_anchor_differs_from_before_arm",
                    "looks",
                )
            }
        )
    )
    for pool in POOLS:
        block = result["pools"][pool]
        print("POOL", pool, "games", block["games"], "folds", block["fold_coefficients"])
        print(" stage1", block["stage_one"])
        for arm in ARMS:
            item = block["levels"][arm]
            print(
                f" {arm:8s} hook share {item['nonzero_hook_share']:.3f}"
                f" served acc {item['served']['accuracy']:.4f} ll {item['served']['log_loss']:.5f}"
                f" fold acc {item['fold']['accuracy']:.4f} ll {item['fold']['log_loss']:.5f}"
                f" inside acc {item['inside']['accuracy']:.4f} ll {item['inside']['log_loss']:.5f}"
            )
        for name, entry in block["comparisons"].items():
            for mode in ("served", "fold", "inside"):
                m = entry[mode]
                seasons = " ".join(
                    f"{year}:{value['accuracy_points']:+.2f}/{value['log_loss_improvement']:+.5f}"
                    f"({value['decisive_first_right']:g}-{value['decisive_second_right']:g})"
                    for year, value in m["by_season"].items()
                )
                print(
                    f" {name:20s} {mode:7s} decisive {m['decisive_first_right']:g}-"
                    f"{m['decisive_second_right']:g} of {m['decisive_games']:g}"
                    f" acc {m['accuracy_points']:+.3f} [{m['accuracy_points_interval'][0]:+.3f},"
                    f"{m['accuracy_points_interval'][1]:+.3f}] P+"
                    f" {m['accuracy_probability_positive']:.3f}"
                    f" ll {m['log_loss_improvement']:+.5f}"
                    f" [{m['log_loss_improvement_interval'][0]:+.5f},"
                    f"{m['log_loss_improvement_interval'][1]:+.5f}] P+"
                    f" {m['log_loss_probability_positive']:.3f} | {seasons}"
                )


if __name__ == "__main__":
    main()
