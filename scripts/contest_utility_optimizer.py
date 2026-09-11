from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.pool import (
    Entry,
    FieldModel,
    PoolFormat,
    _field_scores,
    simulate_pool_finish,
)

OPENER_ARTIFACT = Path("artifacts/opener_evaluation/20260910T211255Z/per_game.parquet")
BEST_PICK_SUMMARY = Path("artifacts/best_pick_sunday_renomination/20260911T021349Z/summary.json")
SCHEDULE_ARTIFACT = Path("data/raw/20260908T162105Z/schedules.parquet")

REG_WEEK_GAMES = (16, 16, 16, 16, 14, 15, 15, 13, 14, 14, 15, 14, 16, 14, 16, 16, 16, 16)
PLAYOFF_GAMES = 13
SEASON_PICKS = sum(REG_WEEK_GAMES) + PLAYOFF_GAMES

FIELD_SIZES = (26, 101, 285, 1001)
HEADLINE_FIELD = 285
PUBLIC_LEANS = (0.50, 0.55, 0.65, 0.75, 0.85)
HEADLINE_LEAN = 0.65
PRIZE_FRACTIONS = (0.0, 0.10, 0.15, 0.25)
BEST_PICK_BONUSES = (1.0, 2.0)
ACCURACY_GRID = (0.500, 0.510, 0.520, 0.530, 0.540, 0.550, 0.560, 0.570)
FLIP_COUNTS = (0, 10, 25, 50)

STAR_RULES = (
    "max_probability",
    "big_spread",
    "max_variance",
    "contrarian",
    "contrarian_big",
    "arbitrary",
)

LEAD54_WIN_RATE_DELTA_PP = -0.137
LEAD54_WIN_RATE_LOW_PP = -0.935
LEAD54_WIN_RATE_HIGH_PP = 0.661

PROB_FLOOR = 0.30
PROB_CEILING = 0.70


def _prize_places(fraction: float, entries: int) -> int:
    if fraction <= 0.0:
        return 1
    return max(1, round(fraction * entries))


def load_population(column: str) -> pd.DataFrame:
    frame = pd.read_parquet(OPENER_ARTIFACT)
    frame = frame.dropna(subset=[column, "tue_open_home_spread"]).copy()
    frame["abs_spread"] = frame["tue_open_home_spread"].abs()
    pick_home = frame["pick_home_at_open_probability_rule"].astype(bool)
    if column == "correct_at_open":
        pick_home = frame["pick_home_at_open"].astype(bool)
    spread = frame["tue_open_home_spread"].to_numpy(dtype=float)
    favourite = np.where(
        spread < 0.0,
        pick_home.to_numpy(),
        np.where(spread > 0.0, ~pick_home.to_numpy(), np.nan),
    )
    frame["pick_favourite"] = favourite
    frame["correct"] = frame[column].astype(float)
    frame["week_key"] = frame["season"].astype(str) + "-" + frame["week"].astype(str)
    return frame[["week_key", "season", "week", "abs_spread", "pick_favourite", "correct"]]


def fit_logistic(
    spread: np.ndarray, correct: np.ndarray, iterations: int = 40
) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(spread), spread])
    beta = np.zeros(2, dtype=float)
    for _ in range(iterations):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(mu * (1.0 - mu), 1e-8, None)
        gradient = design.T @ (correct - mu)
        hessian = design.T @ (design * weight[:, None])
        try:
            step = np.linalg.solve(hessian + 1e-8 * np.eye(2), gradient)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def curve_probability(intercept: float, slope: float, spread: np.ndarray) -> np.ndarray:
    eta = intercept + slope * spread
    return np.clip(1.0 / (1.0 + np.exp(-eta)), PROB_FLOOR, PROB_CEILING)


def shift_to_mean(intercept: float, slope: float, spread: np.ndarray, target: float) -> float:
    low, high = -5.0, 5.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        value = float(np.mean(curve_probability(intercept + mid, slope, spread)))
        if value < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def week_blocked_resample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    keys = frame["week_key"].unique()
    positions = frame.groupby("week_key", observed=True).indices
    drawn = rng.choice(keys, size=len(keys), replace=True)
    taken = np.concatenate([positions[key] for key in drawn])
    return frame.take(taken).reset_index(drop=True)


def week_slices() -> list[slice]:
    counts = list(REG_WEEK_GAMES)
    counts[-1] = counts[-1] + PLAYOFF_GAMES
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(len(counts))]


def pool_format(bonus: float) -> PoolFormat:
    counts = list(REG_WEEK_GAMES)
    counts[-1] = counts[-1] + PLAYOFF_GAMES
    return PoolFormat(weekly_games=tuple(counts), best_pick_bonus=bonus)


def draw_season(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = rng.integers(0, len(frame), size=SEASON_PICKS)
    spread = frame["abs_spread"].to_numpy(dtype=float)[index]
    favourite = frame["pick_favourite"].to_numpy(dtype=float)[index]
    coin = rng.random(SEASON_PICKS) < 0.5
    favourite = np.where(np.isnan(favourite), coin.astype(float), favourite)
    eligible = np.zeros(SEASON_PICKS, dtype=bool)
    eligible[: sum(REG_WEEK_GAMES)] = True
    return spread, favourite.astype(bool), eligible


def choose_stars(
    rule: str,
    probability: np.ndarray,
    spread: np.ndarray,
    favourite: np.ndarray,
    eligible: np.ndarray,
    slices: list[slice],
    rng: np.random.Generator,
) -> np.ndarray:
    stars = []
    for slot in slices:
        offset = slot.start
        mask = eligible[slot]
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            stars.append(-1)
            continue
        sub_p = probability[slot][idx]
        sub_s = spread[slot][idx]
        sub_f = favourite[slot][idx]
        if rule == "max_probability":
            choice = idx[int(np.argmax(sub_p))]
        elif rule == "big_spread":
            choice = idx[int(np.argmax(sub_s))]
        elif rule == "max_variance":
            choice = idx[int(np.argmin(np.abs(sub_p - 0.5)))]
        elif rule == "contrarian":
            off = np.nonzero(~sub_f)[0]
            choice = (
                idx[int(off[np.argmax(sub_p[off])])] if off.size else idx[int(np.argmax(sub_p))]
            )
        elif rule == "contrarian_big":
            off = np.nonzero(~sub_f)[0]
            choice = (
                idx[int(off[np.argmax(sub_s[off])])] if off.size else idx[int(np.argmax(sub_s))]
            )
        elif rule == "arbitrary":
            choice = idx[int(rng.integers(0, idx.size))]
        else:
            raise ValueError(f"unknown star rule {rule!r}")
        stars.append(offset + int(choice))
    return np.asarray(stars, dtype=int)


def flip_sides(
    probability: np.ndarray, favourite: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return probability, favourite
    candidates = np.nonzero(favourite)[0]
    if candidates.size == 0:
        return probability, favourite
    order = candidates[np.argsort(np.abs(probability[candidates] - 0.5))]
    chosen = order[: min(count, order.size)]
    flipped = probability.copy()
    flipped[chosen] = 1.0 - flipped[chosen]
    side = favourite.copy()
    side[chosen] = ~side[chosen]
    return flipped, side


def build_entry_direct(probability: np.ndarray, favourite: np.ndarray, stars: np.ndarray) -> Entry:
    return Entry(
        cover_probability=np.asarray(probability, dtype=float),
        on_public_side=np.asarray(favourite, dtype=bool),
        best_pick_index=np.asarray(stars, dtype=int),
    )


def simulate_detailed(
    entry: Entry,
    field: FieldModel,
    fmt: PoolFormat,
    *,
    samples: int,
    seed: int,
    chunk: int = 2_000,
    prize_places: tuple[int, ...] = (1,),
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    outright = 0.0
    shared = 0.0
    tied = 0.0
    rank_sum = 0.0
    score_sum = 0.0
    score_square_sum = 0.0
    in_money = dict.fromkeys(prize_places, 0.0)
    boundary = dict.fromkeys(prize_places, 0.0)
    drawn = 0
    while drawn < samples:
        size = min(chunk, samples - drawn)
        covered = generator.random((size, fmt.games)) < entry.cover_probability
        ours = covered.sum(axis=1).astype(float)
        for index in entry.best_pick_index:
            if int(index) < 0:
                continue
            hit = covered[:, int(index)]
            ours += np.where(hit, fmt.best_pick_bonus, -fmt.best_pick_penalty)
        public_won = np.where(entry.on_public_side, covered, ~covered)
        rivals = _field_scores(public_won, fmt, field, generator)
        beaten = (rivals > ours[:, None]).sum(axis=1)
        level = (rivals == ours[:, None]).sum(axis=1)
        outright += float(np.count_nonzero(beaten + level == 0))
        tied += float(np.count_nonzero((beaten == 0) & (level > 0)))
        shared += float(np.where(beaten == 0, 1.0 / (1.0 + level), 0.0).sum())
        rank_sum += float((beaten + 1).sum())
        score_sum += float(ours.sum())
        score_square_sum += float((ours**2).sum())
        for places in prize_places:
            in_money[places] += float(np.count_nonzero(beaten < places))
            boundary[places] += float(
                np.count_nonzero((beaten < places) & (beaten + level >= places))
            )
        drawn += size
    mean_score = score_sum / samples
    variance = max(0.0, score_square_sum / samples - mean_score**2)
    result = {
        "probability_first": shared / samples,
        "probability_outright": outright / samples,
        "probability_tied_first": tied / samples,
        "expected_score": mean_score,
        "score_sd": math.sqrt(variance),
        "expected_rank": rank_sum / samples,
        "expected_percentile": (rank_sum / samples) / (field.entrants + 1.0),
    }
    for places in prize_places:
        result[f"probability_paid_{places}"] = in_money[places] / samples
        result[f"probability_boundary_tie_{places}"] = boundary[places] / samples
    return result


def tie_out_simulator(frame: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(20260911)
    intercept, slope = fit_logistic(
        frame["abs_spread"].to_numpy(dtype=float), frame["correct"].to_numpy(dtype=float)
    )
    spread, favourite, eligible = draw_season(frame, rng)
    probability = curve_probability(intercept, slope, spread)
    stars = choose_stars(
        "max_probability", probability, spread, favourite, eligible, week_slices(), rng
    )
    entry = build_entry_direct(probability, favourite, stars)
    fmt = pool_format(1.0)
    field = FieldModel(entrants=HEADLINE_FIELD - 1, public_lean=HEADLINE_LEAN)
    library = simulate_pool_finish(entry, field, fmt, samples=4_000, seed=4242, chunk=2_000)
    mine = simulate_detailed(entry, field, fmt, samples=4_000, seed=4242, prize_places=(1,))
    return {
        "library_probability_first": library["probability_first"],
        "script_probability_first": mine["probability_first"],
        "library_expected_rank": library["expected_rank"],
        "script_expected_rank": mine["expected_rank"],
        "library_score_sd": library["score_sd"],
        "script_score_sd": mine["score_sd"],
        "library_probability_tied_first": library["probability_tied_first"],
        "script_probability_boundary_tie_1": mine["probability_boundary_tie_1"],
        "identical": bool(
            abs(library["probability_first"] - mine["probability_first"]) < 1e-12
            and abs(library["expected_rank"] - mine["expected_rank"]) < 1e-12
            and abs(library["probability_tied_first"] - mine["probability_boundary_tie_1"]) < 1e-12
        ),
    }


def measured_inputs(frame: pd.DataFrame, served_frame: pd.DataFrame) -> dict[str, Any]:
    spread = frame["abs_spread"].to_numpy(dtype=float)
    correct = frame["correct"].to_numpy(dtype=float)
    intercept, slope = fit_logistic(spread, correct)
    edges = [-0.01, 2.5, 4.5, 7.0, 10.0, 60.0]
    labels = ["0-2.5", "3-4.5", "5-7", "7.5-10", "10.5+"]
    bucket = pd.cut(frame["abs_spread"], edges, labels=labels)
    table = frame.groupby(bucket, observed=True)["correct"].agg(["size", "mean"])
    best_pick = json.loads(BEST_PICK_SUMMARY.read_text(encoding="utf-8"))["hit_rates"]
    small = frame.loc[frame["abs_spread"] <= 4.5, "correct"]
    big = frame.loc[frame["abs_spread"] >= 7.5, "correct"]
    return {
        "games": len(frame),
        "weeks": int(frame["week_key"].nunique()),
        "accuracy_raw_pick": float(correct.mean()),
        "accuracy_served_rule": float(served_frame["correct"].mean()),
        "pick_favourite_share": float(np.nanmean(frame["pick_favourite"].to_numpy(dtype=float))),
        "logistic_intercept": intercept,
        "logistic_slope": slope,
        "bucket_table": {
            str(key): {"games": int(row["size"]), "accuracy": float(row["mean"])}
            for key, row in table.iterrows()
        },
        "curve_at_spread": {
            str(value): float(curve_probability(intercept, slope, np.array([value]))[0])
            for value in (1.0, 2.5, 3.0, 6.0, 9.0, 13.0)
        },
        "best_pick_hit_rates": {
            key: {"accuracy": value["accuracy"], "weeks": value["weeks_scored"]}
            for key, value in best_pick.items()
        },
        "small_spread_accuracy": float(small.mean()),
        "small_spread_games": int(small.size),
        "big_spread_accuracy": float(big.mean()),
        "big_spread_games": int(big.size),
    }


def cell_one(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, Any]:
    best_pick = json.loads(BEST_PICK_SUMMARY.read_text(encoding="utf-8"))["hit_rates"]
    p_ordinary = float(frame["correct"].mean())
    p_best = float(best_pick["s3_sunday_full"]["accuracy"])
    rows = []
    for bonus in BEST_PICK_BONUSES:
        weight = 1.0 + bonus
        ordinary_games = SEASON_PICKS - len(REG_WEEK_GAMES)
        var_ordinary = ordinary_games * p_ordinary * (1.0 - p_ordinary)
        var_best = len(REG_WEEK_GAMES) * weight**2 * p_best * (1.0 - p_best)
        var_star_only = len(REG_WEEK_GAMES) * (weight**2 - 1.0) * p_best * (1.0 - p_best)
        mean_ordinary = ordinary_games * p_ordinary
        mean_best = len(REG_WEEK_GAMES) * weight * p_best
        rows.append(
            {
                "best_pick_bonus": bonus,
                "ordinary_games": ordinary_games,
                "best_pick_games": len(REG_WEEK_GAMES),
                "p_ordinary": p_ordinary,
                "p_best_pick": p_best,
                "variance_ordinary": var_ordinary,
                "variance_best_pick": var_best,
                "variance_total": var_ordinary + var_best,
                "best_pick_variance_share": var_best / (var_ordinary + var_best),
                "variance_added_by_starring": var_star_only,
                "variance_added_share": var_star_only / (var_ordinary + var_best),
                "expected_score": mean_ordinary + mean_best,
                "best_pick_expected_share": mean_best / (mean_ordinary + mean_best),
                "score_sd": math.sqrt(var_ordinary + var_best),
            }
        )
    simulated = []
    base = frame
    for bonus in BEST_PICK_BONUSES:
        sds = []
        for replicate in range(replicates):
            local = np.random.default_rng(seed + replicate)
            boot = week_blocked_resample(base, local)
            intercept, slope = fit_logistic(
                boot["abs_spread"].to_numpy(dtype=float), boot["correct"].to_numpy(dtype=float)
            )
            spread, favourite, eligible = draw_season(boot, local)
            probability = curve_probability(intercept, slope, spread)
            stars = choose_stars(
                "max_probability", probability, spread, favourite, eligible, week_slices(), local
            )
            entry = build_entry_direct(probability, favourite, stars)
            fmt = pool_format(bonus)
            out = simulate_detailed(
                entry,
                FieldModel(entrants=HEADLINE_FIELD - 1, public_lean=HEADLINE_LEAN),
                fmt,
                samples=4_000,
                seed=90_000 + replicate,
                prize_places=(1,),
            )
            sds.append(out["score_sd"])
        no_star = []
        for replicate in range(replicates):
            local = np.random.default_rng(seed + replicate)
            boot = week_blocked_resample(base, local)
            intercept, slope = fit_logistic(
                boot["abs_spread"].to_numpy(dtype=float), boot["correct"].to_numpy(dtype=float)
            )
            spread, favourite, eligible = draw_season(boot, local)
            probability = curve_probability(intercept, slope, spread)
            stars = np.full(len(week_slices()), -1, dtype=int)
            entry = build_entry_direct(probability, favourite, stars)
            fmt = pool_format(bonus)
            out = simulate_detailed(
                entry,
                FieldModel(entrants=HEADLINE_FIELD - 1, public_lean=HEADLINE_LEAN),
                fmt,
                samples=4_000,
                seed=90_000 + replicate,
                prize_places=(1,),
            )
            no_star.append(out["score_sd"])
        starred = np.asarray(sds)
        bare = np.asarray(no_star)
        share = 1.0 - (bare**2) / (starred**2)
        simulated.append(
            {
                "best_pick_bonus": bonus,
                "score_sd_with_star": float(starred.mean()),
                "score_sd_without_star": float(bare.mean()),
                "variance_share_from_star": float(share.mean()),
                "variance_share_low": float(np.percentile(share, 2.5)),
                "variance_share_high": float(np.percentile(share, 97.5)),
            }
        )
    return {"exact": rows, "simulated": simulated}


def _replicate_inputs(
    base: pd.DataFrame, seed: int, replicate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    local = np.random.default_rng(seed + replicate)
    boot = week_blocked_resample(base, local)
    intercept, slope = fit_logistic(
        boot["abs_spread"].to_numpy(dtype=float), boot["correct"].to_numpy(dtype=float)
    )
    spread, favourite, eligible = draw_season(boot, local)
    probability = curve_probability(intercept, slope, spread)
    return spread, favourite, eligible, probability, intercept, slope


def _samples_for(entrants: int, budget: int) -> int:
    return int(max(2_000, min(budget, 1_600_000 // max(1, entrants))))


def cell_two(base: pd.DataFrame, replicates: int, seed: int, budget: int) -> dict[str, Any]:
    slices = week_slices()
    settings = []
    for lean in PUBLIC_LEANS:
        settings.append((HEADLINE_FIELD, lean))
    for size in FIELD_SIZES:
        if size != HEADLINE_FIELD:
            settings.append((size, HEADLINE_LEAN))
    results: dict[str, list[dict[str, float]]] = {}
    flips: dict[str, list[dict[str, float]]] = {}
    for entries, lean in settings:
        places = tuple(sorted({_prize_places(fraction, entries) for fraction in PRIZE_FRACTIONS}))
        entrants = entries - 1
        samples = _samples_for(entrants, budget)
        per_arm: dict[str, list[dict[str, float]]] = {rule: [] for rule in STAR_RULES}
        per_flip: dict[int, list[dict[str, float]]] = {count: [] for count in FLIP_COUNTS}
        for replicate in range(replicates):
            spread, favourite, eligible, probability, _, _ = _replicate_inputs(
                base, seed, replicate
            )
            star_rng = np.random.default_rng(700_000 + replicate)
            fmt = pool_format(1.0)
            field = FieldModel(entrants=entrants, public_lean=lean)
            sim_seed = 500_000 + replicate
            for rule in STAR_RULES:
                stars = choose_stars(
                    rule, probability, spread, favourite, eligible, slices, star_rng
                )
                entry = build_entry_direct(probability, favourite, stars)
                per_arm[rule].append(
                    simulate_detailed(
                        entry, field, fmt, samples=samples, seed=sim_seed, prize_places=places
                    )
                )
            for count in FLIP_COUNTS:
                if count == 0:
                    zero = dict(per_arm["max_probability"][-1])
                    zero["mean_probability"] = float(probability.mean())
                    per_flip[count].append(zero)
                    continue
                flipped_p, flipped_side = flip_sides(probability, favourite, count)
                stars = choose_stars(
                    "max_probability", flipped_p, spread, flipped_side, eligible, slices, star_rng
                )
                entry = build_entry_direct(flipped_p, flipped_side, stars)
                out = simulate_detailed(
                    entry, field, fmt, samples=samples, seed=sim_seed, prize_places=places
                )
                out["mean_probability"] = float(flipped_p.mean())
                per_flip[count].append(out)
        key = f"entries={entries},lean={lean}"
        results[key] = _summarise_arms(per_arm, "max_probability", places, entries, lean, samples)
        flips[key] = _summarise_flips(per_flip, places, entries, lean, samples)
    return {"stars": results, "side_flips": flips}


def _summarise_arms(
    per_arm: dict[str, list[dict[str, float]]],
    baseline: str,
    places: tuple[int, ...],
    entries: int,
    lean: float,
    samples: int,
) -> list[dict[str, float]]:
    rows = []
    base_rows = per_arm[baseline]
    for rule, runs in per_arm.items():
        row: dict[str, Any] = {
            "arm": rule,
            "entries": entries,
            "public_lean": lean,
            "samples": samples,
            "replicates": len(runs),
        }
        for metric in ("probability_first", "expected_score", "expected_percentile"):
            values = np.asarray([run[metric] for run in runs])
            row[metric] = float(values.mean())
        for place in places:
            paid = np.asarray([run[f"probability_paid_{place}"] for run in runs])
            row[f"probability_paid_{place}"] = float(paid.mean())
        if rule != baseline:
            for metric in ("probability_first", "expected_percentile"):
                delta = np.asarray(
                    [run[metric] - ref[metric] for run, ref in zip(runs, base_rows, strict=True)]
                )
                row[f"delta_{metric}"] = float(delta.mean())
                row[f"delta_{metric}_low"] = float(np.percentile(delta, 2.5))
                row[f"delta_{metric}_high"] = float(np.percentile(delta, 97.5))
                sign = delta if metric == "probability_first" else -delta
                row[f"delta_{metric}_probability_positive"] = float(
                    probability_positive_from_draws(sign)
                )
            for place in places:
                delta = np.asarray(
                    [
                        run[f"probability_paid_{place}"] - ref[f"probability_paid_{place}"]
                        for run, ref in zip(runs, base_rows, strict=True)
                    ]
                )
                row[f"delta_probability_paid_{place}"] = float(delta.mean())
                row[f"delta_probability_paid_{place}_low"] = float(np.percentile(delta, 2.5))
                row[f"delta_probability_paid_{place}_high"] = float(np.percentile(delta, 97.5))
                row[f"delta_probability_paid_{place}_probability_positive"] = float(
                    probability_positive_from_draws(delta)
                )
        rows.append(row)
    return rows


def _summarise_flips(
    per_flip: dict[int, list[dict[str, float]]],
    places: tuple[int, ...],
    entries: int,
    lean: float,
    samples: int,
) -> list[dict[str, float]]:
    rows = []
    base_rows = per_flip[0]
    for count, runs in per_flip.items():
        row: dict[str, Any] = {
            "flips": count,
            "entries": entries,
            "public_lean": lean,
            "samples": samples,
            "replicates": len(runs),
            "mean_probability": float(np.mean([run["mean_probability"] for run in runs])),
            "probability_first": float(np.mean([run["probability_first"] for run in runs])),
        }
        for place in places:
            row[f"probability_paid_{place}"] = float(
                np.mean([run[f"probability_paid_{place}"] for run in runs])
            )
        if count != 0:
            delta = np.asarray(
                [
                    run["probability_first"] - ref["probability_first"]
                    for run, ref in zip(runs, base_rows, strict=True)
                ]
            )
            row["delta_probability_first"] = float(delta.mean())
            row["delta_probability_first_low"] = float(np.percentile(delta, 2.5))
            row["delta_probability_first_high"] = float(np.percentile(delta, 97.5))
            row["delta_probability_first_probability_positive"] = float(
                probability_positive_from_draws(delta)
            )
            for place in places:
                paid = np.asarray(
                    [
                        run[f"probability_paid_{place}"] - ref[f"probability_paid_{place}"]
                        for run, ref in zip(runs, base_rows, strict=True)
                    ]
                )
                row[f"delta_probability_paid_{place}"] = float(paid.mean())
                row[f"delta_probability_paid_{place}_low"] = float(np.percentile(paid, 2.5))
                row[f"delta_probability_paid_{place}_high"] = float(np.percentile(paid, 97.5))
                row[f"delta_probability_paid_{place}_probability_positive"] = float(
                    probability_positive_from_draws(paid)
                )
        rows.append(row)
    return rows


def cell_three(base: pd.DataFrame, replicates: int, seed: int, budget: int) -> dict[str, Any]:
    slices = week_slices()
    rows = []
    for entries in FIELD_SIZES:
        places = tuple(sorted({_prize_places(fraction, entries) for fraction in PRIZE_FRACTIONS}))
        entrants = entries - 1
        samples = _samples_for(entrants, budget)
        runs = []
        for replicate in range(replicates):
            spread, favourite, eligible, probability, _, _ = _replicate_inputs(
                base, seed, replicate
            )
            star_rng = np.random.default_rng(700_000 + replicate)
            stars = choose_stars(
                "max_probability", probability, spread, favourite, eligible, slices, star_rng
            )
            entry = build_entry_direct(probability, favourite, stars)
            runs.append(
                simulate_detailed(
                    entry,
                    FieldModel(entrants=entrants, public_lean=HEADLINE_LEAN),
                    pool_format(1.0),
                    samples=samples,
                    seed=500_000 + replicate,
                    prize_places=places,
                )
            )
        for place in places:
            boundary = np.asarray([run[f"probability_boundary_tie_{place}"] for run in runs])
            rows.append(
                {
                    "entries": entries,
                    "prize_places": place,
                    "prize_fraction": place / entries,
                    "samples": samples,
                    "replicates": len(runs),
                    "probability_boundary_tie": float(boundary.mean()),
                    "probability_boundary_tie_low": float(np.percentile(boundary, 2.5)),
                    "probability_boundary_tie_high": float(np.percentile(boundary, 97.5)),
                }
            )
    return {"boundary_ties": rows}


def cell_four(base: pd.DataFrame, replicates: int, seed: int, budget: int) -> dict[str, Any]:
    slices = week_slices()
    rows = []
    curves: dict[str, list[dict[str, float]]] = {}
    for entries in FIELD_SIZES:
        places = tuple(sorted({_prize_places(fraction, entries) for fraction in PRIZE_FRACTIONS}))
        entrants = entries - 1
        samples = _samples_for(entrants, budget)
        per_level: dict[float, list[dict[str, float]]] = {level: [] for level in ACCURACY_GRID}
        for replicate in range(replicates):
            spread, favourite, eligible, _, intercept, slope = _replicate_inputs(
                base, seed, replicate
            )
            star_rng = np.random.default_rng(700_000 + replicate)
            for level in ACCURACY_GRID:
                offset = shift_to_mean(intercept, slope, spread, level)
                probability = curve_probability(intercept + offset, slope, spread)
                stars = choose_stars(
                    "max_probability", probability, spread, favourite, eligible, slices, star_rng
                )
                entry = build_entry_direct(probability, favourite, stars)
                per_level[level].append(
                    simulate_detailed(
                        entry,
                        FieldModel(entrants=entrants, public_lean=HEADLINE_LEAN),
                        pool_format(1.0),
                        samples=samples,
                        seed=500_000 + replicate,
                        prize_places=places,
                    )
                )
        level_rows = []
        for level, runs in per_level.items():
            row: dict[str, Any] = {
                "entries": entries,
                "accuracy": level,
                "samples": samples,
                "probability_first": float(np.mean([r["probability_first"] for r in runs])),
                "expected_percentile": float(np.mean([r["expected_percentile"] for r in runs])),
            }
            for place in places:
                row[f"probability_paid_{place}"] = float(
                    np.mean([r[f"probability_paid_{place}"] for r in runs])
                )
            level_rows.append(row)
        curves[str(entries)] = level_rows
        grid = list(ACCURACY_GRID)
        served_key = min(grid, key=lambda value: abs(value - 0.530))
        next_key = min(grid, key=lambda value: abs(value - 0.540))
        if served_key != next_key:
            base_row = next(r for r in level_rows if r["accuracy"] == served_key)
            next_row = next(r for r in level_rows if r["accuracy"] == next_key)
            deltas = []
            for replicate in range(replicates):
                low = per_level[served_key][replicate]
                high = per_level[next_key][replicate]
                deltas.append(high["probability_first"] - low["probability_first"])
            draws = np.asarray(deltas)
            rows.append(
                {
                    "entries": entries,
                    "from_accuracy": served_key,
                    "to_accuracy": next_key,
                    "probability_first_from": base_row["probability_first"],
                    "probability_first_to": next_row["probability_first"],
                    "delta_probability_first": float(draws.mean()),
                    "delta_probability_first_low": float(np.percentile(draws, 2.5)),
                    "delta_probability_first_high": float(np.percentile(draws, 97.5)),
                    "delta_probability_first_probability_positive": float(
                        probability_positive_from_draws(draws)
                    ),
                    "delta_expected_percentile": next_row["expected_percentile"]
                    - base_row["expected_percentile"],
                    "probability_first_per_accuracy_point": float(draws.mean()),
                }
            )
    return {"curves": curves, "one_point": rows}


def instrument_controls(base: pd.DataFrame, seed: int, budget: int) -> dict[str, Any]:
    slices = week_slices()
    entrants = HEADLINE_FIELD - 1
    samples = _samples_for(entrants, budget)
    rng = np.random.default_rng(seed)
    spread, favourite, eligible = draw_season(base, rng)
    flat = np.full(SEASON_PICKS, 0.5)
    matched = rng.random(SEASON_PICKS) < HEADLINE_LEAN
    stars = choose_stars("arbitrary", flat, spread, favourite, eligible, slices, rng)
    field = FieldModel(entrants=entrants, public_lean=HEADLINE_LEAN)
    fmt = pool_format(1.0)
    cases = {
        "flat_coin_flip_field_matched_sides": (flat, matched),
        "flat_coin_flip_our_measured_sides": (flat, favourite),
        "perfect_card": (np.full(SEASON_PICKS, 1.0), favourite),
    }
    rows = []
    for name, (probability, side) in cases.items():
        entry = build_entry_direct(probability, side, stars)
        out = simulate_detailed(
            entry, field, fmt, samples=samples * 4, seed=seed, prize_places=(1,)
        )
        rows.append(
            {
                "control": name,
                "favourite_share": float(np.mean(side)),
                "probability_first": out["probability_first"],
                "fair_share": 1.0 / HEADLINE_FIELD,
                "expected_score": out["expected_score"],
                "samples": samples * 4,
            }
        )
    return {"controls": rows}


def best_pick_value(base: pd.DataFrame, replicates: int, seed: int, budget: int) -> dict[str, Any]:
    slices = week_slices()
    rates = json.loads(BEST_PICK_SUMMARY.read_text(encoding="utf-8"))["hit_rates"]
    arms = {
        "card_average": float(base["correct"].mean()),
        "tuesday_nominator": float(rates["t0_tuesday"]["accuracy"]),
        "sunday_nominator": float(rates["s3_sunday_full"]["accuracy"]),
    }
    entrants = HEADLINE_FIELD - 1
    samples = _samples_for(entrants, budget)
    per_arm: dict[str, list[dict[str, float]]] = {name: [] for name in arms}
    for replicate in range(replicates):
        spread, favourite, eligible, probability, _, _ = _replicate_inputs(base, seed, replicate)
        star_rng = np.random.default_rng(700_000 + replicate)
        stars = choose_stars(
            "max_probability", probability, spread, favourite, eligible, slices, star_rng
        )
        for name, value in arms.items():
            adjusted = probability.copy()
            adjusted[stars[stars >= 0]] = value
            entry = build_entry_direct(adjusted, favourite, stars)
            per_arm[name].append(
                simulate_detailed(
                    entry,
                    FieldModel(entrants=entrants, public_lean=HEADLINE_LEAN),
                    pool_format(1.0),
                    samples=samples,
                    seed=500_000 + replicate,
                    prize_places=(1, _prize_places(0.15, HEADLINE_FIELD)),
                )
            )
    rows = []
    reference = per_arm["card_average"]
    for name, runs in per_arm.items():
        row: dict[str, Any] = {
            "arm": name,
            "star_hit_rate": arms[name],
            "entries": HEADLINE_FIELD,
            "public_lean": HEADLINE_LEAN,
            "samples": samples,
            "replicates": replicates,
            "probability_first": float(np.mean([r["probability_first"] for r in runs])),
            "expected_score": float(np.mean([r["expected_score"] for r in runs])),
        }
        if name != "card_average":
            delta = np.asarray(
                [
                    run["probability_first"] - ref["probability_first"]
                    for run, ref in zip(runs, reference, strict=True)
                ]
            )
            row["delta_probability_first"] = float(delta.mean())
            row["delta_probability_first_low"] = float(np.percentile(delta, 2.5))
            row["delta_probability_first_high"] = float(np.percentile(delta, 97.5))
            row["delta_probability_first_probability_positive"] = float(
                probability_positive_from_draws(delta)
            )
        rows.append(row)
    return {"arms": rows}


def spread_gradient(frame: pd.DataFrame, samples: int = 20_000) -> dict[str, Any]:
    from nfl_ats.clv import week_blocked_bootstrap

    working = frame.copy()
    working["small"] = (working["abs_spread"] <= 4.5).astype(float)
    working["big"] = (working["abs_spread"] >= 7.5).astype(float)

    def metric(part: pd.DataFrame) -> dict[str, float]:
        small = part.loc[part["small"] > 0.5, "correct"]
        big = part.loc[part["big"] > 0.5, "correct"]
        if small.empty or big.empty:
            return {"gradient_accuracy_points": float("nan")}
        return {"gradient_accuracy_points": 100.0 * (float(small.mean()) - float(big.mean()))}

    table = week_blocked_bootstrap(
        working,
        metric,
        block="week",
        samples=samples,
        seed=20260911,
        metric_columns=["season", "week", "small", "big", "correct"],
    )
    row = table.iloc[0]
    return {
        "effect_accuracy_points": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
        "small_games": int((working["small"] > 0.5).sum()),
        "big_games": int((working["big"] > 0.5).sum()),
        "blocks": int(working.groupby(["season", "week"]).ngroups),
    }


def _rate_table(payload: dict[str, Any]) -> dict[int, dict[str, float]]:
    rates: dict[int, dict[str, float]] = {}
    for key, rows in payload["cell_four_payout_units"]["curves"].items():
        entries = int(key)
        low = next(r for r in rows if abs(r["accuracy"] - 0.530) < 1e-9)
        high = next(r for r in rows if abs(r["accuracy"] - 0.540) < 1e-9)
        rate = {"probability_first": high["probability_first"] - low["probability_first"]}
        for name in low:
            if name.startswith("probability_paid_"):
                rate[name] = high[name] - low[name]
        rate["expected_percentile"] = high["expected_percentile"] - low["expected_percentile"]
        rates[entries] = rate
    return rates


def derive(artifact: Path) -> dict[str, Any]:
    payload = json.loads((artifact / "results.json").read_text(encoding="utf-8"))
    rates = _rate_table(payload)
    equivalents = []
    for key, rows in payload["cell_two_star_choice"]["stars"].items():
        for row in rows:
            if "delta_probability_first" not in row:
                continue
            entries = int(row["entries"])
            rate = rates[entries]["probability_first"]
            equivalents.append(
                {
                    "setting": key,
                    "arm": row["arm"],
                    "metric": "probability_first",
                    "delta": row["delta_probability_first"],
                    "delta_low": row["delta_probability_first_low"],
                    "delta_high": row["delta_probability_first_high"],
                    "probability_positive": row["delta_probability_first_probability_positive"],
                    "accuracy_point_equivalent": row["delta_probability_first"] / rate,
                    "accuracy_point_equivalent_low": row["delta_probability_first_low"] / rate,
                    "accuracy_point_equivalent_high": row["delta_probability_first_high"] / rate,
                }
            )
            for name in list(row):
                if not name.startswith("delta_probability_paid_") or not name.endswith("positive"):
                    continue
                stem = name[: -len("_probability_positive")]
                place = stem.split("_")[-1]
                rate_paid = rates[entries].get(f"probability_paid_{place}")
                if rate_paid is None or rate_paid == 0.0:
                    continue
                equivalents.append(
                    {
                        "setting": key,
                        "arm": row["arm"],
                        "metric": f"probability_paid_{place}",
                        "delta": row[stem],
                        "delta_low": row[f"{stem}_low"],
                        "delta_high": row[f"{stem}_high"],
                        "probability_positive": row[name],
                        "accuracy_point_equivalent": row[stem] / rate_paid,
                        "accuracy_point_equivalent_low": row[f"{stem}_low"] / rate_paid,
                        "accuracy_point_equivalent_high": row[f"{stem}_high"] / rate_paid,
                    }
                )
    flip_equivalents = []
    for key, rows in payload["cell_two_star_choice"]["side_flips"].items():
        for row in rows:
            if "delta_probability_first" not in row:
                continue
            entries = int(row["entries"])
            rate = rates[entries]["probability_first"]
            flip_equivalents.append(
                {
                    "setting": key,
                    "flips": row["flips"],
                    "mean_probability": row["mean_probability"],
                    "delta_probability_first": row["delta_probability_first"],
                    "delta_low": row["delta_probability_first_low"],
                    "delta_high": row["delta_probability_first_high"],
                    "probability_positive": row["delta_probability_first_probability_positive"],
                    "accuracy_point_equivalent": row["delta_probability_first"] / rate,
                }
            )
    generator = np.random.default_rng(20260911)
    spread_sd = (LEAD54_WIN_RATE_HIGH_PP - LEAD54_WIN_RATE_LOW_PP) / (2.0 * 1.959964)
    win_rate_draws = generator.normal(LEAD54_WIN_RATE_DELTA_PP, spread_sd, size=200_000) / 100.0
    tiebreaker = []
    for row in payload["cell_three_tiebreaker"]["boundary_ties"]:
        entries = int(row["entries"])
        place = int(row["prize_places"])
        rate_paid = rates[entries].get(f"probability_paid_{place}")
        tie_draws = generator.uniform(
            row["probability_boundary_tie_low"], row["probability_boundary_tie_high"], size=200_000
        )
        effect = tie_draws * win_rate_draws
        entry = {
            "entries": entries,
            "prize_places": place,
            "prize_fraction": row["prize_fraction"],
            "probability_boundary_tie": row["probability_boundary_tie"],
            "delta_probability_paid_season": float(effect.mean()),
            "delta_probability_paid_season_low": float(np.percentile(effect, 2.5)),
            "delta_probability_paid_season_high": float(np.percentile(effect, 97.5)),
            "probability_positive": float(probability_positive_from_draws(effect)),
        }
        if rate_paid:
            entry["accuracy_point_equivalent"] = entry["delta_probability_paid_season"] / rate_paid
            entry["accuracy_point_equivalent_low"] = (
                entry["delta_probability_paid_season_low"] / rate_paid
            )
            entry["accuracy_point_equivalent_high"] = (
                entry["delta_probability_paid_season_high"] / rate_paid
            )
        tiebreaker.append(entry)
    derived = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_artifact": str(artifact),
        "accuracy_point_rates": {str(k): v for k, v in rates.items()},
        "star_equivalents": equivalents,
        "flip_equivalents": flip_equivalents,
        "tiebreaker": tiebreaker,
        "lead54_win_rate_delta_pp": LEAD54_WIN_RATE_DELTA_PP,
        "lead54_win_rate_interval_pp": [LEAD54_WIN_RATE_LOW_PP, LEAD54_WIN_RATE_HIGH_PP],
    }
    (artifact / "derived.json").write_text(json.dumps(derived, indent=2), encoding="utf-8")
    pd.DataFrame(equivalents).to_csv(artifact / "star_equivalents.csv", index=False)
    pd.DataFrame(tiebreaker).to_csv(artifact / "tiebreaker_value.csv", index=False)
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "POL-05 contest utility optimizer: score the four predeclared cells of "
            "docs/contest_utility_optimizer.md against the pool simulator."
        )
    )
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--cell-one-replicates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--sample-budget", type=int, default=5_000)
    parser.add_argument(
        "--out-root", type=Path, default=Path("artifacts/contest_utility_optimizer")
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--derive-from", type=Path, default=None)
    parser.add_argument("--gradient-only", action="store_true")
    parser.add_argument("--best-pick-value-only", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()

    if args.controls_only:
        print(
            json.dumps(
                instrument_controls(
                    load_population("correct_at_open"), args.seed, args.sample_budget
                ),
                indent=2,
            )
        )
        return

    if args.best_pick_value_only:
        print(
            json.dumps(
                best_pick_value(
                    load_population("correct_at_open"),
                    args.replicates,
                    args.seed,
                    args.sample_budget,
                ),
                indent=2,
            )
        )
        return

    if args.gradient_only:
        print(json.dumps(spread_gradient(load_population("correct_at_open")), indent=2))
        print(
            json.dumps(
                spread_gradient(load_population("correct_at_open_probability_rule")), indent=2
            )
        )
        return

    if args.derive_from is not None:
        print(json.dumps(derive(args.derive_from), indent=2))
        return

    replicates = 4 if args.quick else args.replicates
    cell_one_replicates = 2 if args.quick else args.cell_one_replicates
    budget = 2_000 if args.quick else args.sample_budget

    base = load_population("correct_at_open")
    served = load_population("correct_at_open_probability_rule")

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "opener_artifact": str(OPENER_ARTIFACT),
        "best_pick_artifact": str(BEST_PICK_SUMMARY),
        "schedule_artifact": str(SCHEDULE_ARTIFACT),
        "season_picks": SEASON_PICKS,
        "regular_season_picks": sum(REG_WEEK_GAMES),
        "playoff_picks": PLAYOFF_GAMES,
        "best_pick_weeks": len(REG_WEEK_GAMES),
        "replicates": replicates,
        "sample_budget": budget,
        "seed": args.seed,
        "assumed_field_sizes": list(FIELD_SIZES),
        "assumed_public_leans": list(PUBLIC_LEANS),
        "assumed_prize_fractions": list(PRIZE_FRACTIONS),
        "assumed_best_pick_bonuses": list(BEST_PICK_BONUSES),
    }
    payload["tie_out"] = tie_out_simulator(base)
    payload["inputs"] = measured_inputs(base, served)
    payload["cell_one_best_pick_weighting"] = cell_one(base, cell_one_replicates, args.seed)
    payload["cell_two_star_choice"] = cell_two(base, replicates, args.seed, budget)
    payload["cell_three_tiebreaker"] = cell_three(base, replicates, args.seed, budget)
    payload["cell_four_payout_units"] = cell_four(base, replicates, args.seed, budget)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    star_rows = [row for rows in payload["cell_two_star_choice"]["stars"].values() for row in rows]
    pd.DataFrame(star_rows).to_csv(out_dir / "star_arms.csv", index=False)
    flip_rows = [
        row for rows in payload["cell_two_star_choice"]["side_flips"].values() for row in rows
    ]
    pd.DataFrame(flip_rows).to_csv(out_dir / "side_flips.csv", index=False)
    pd.DataFrame(payload["cell_three_tiebreaker"]["boundary_ties"]).to_csv(
        out_dir / "boundary_ties.csv", index=False
    )
    curve_rows = [
        row for rows in payload["cell_four_payout_units"]["curves"].values() for row in rows
    ]
    pd.DataFrame(curve_rows).to_csv(out_dir / "accuracy_curve.csv", index=False)

    print(json.dumps({"artifact": str(out_dir), "tie_out": payload["tie_out"]}, indent=2))
    print(json.dumps(payload["inputs"], indent=2))
    print(json.dumps(payload["cell_one_best_pick_weighting"], indent=2))
    print(json.dumps(payload["cell_three_tiebreaker"], indent=2))
    print(json.dumps(payload["cell_four_payout_units"]["one_point"], indent=2))
    headline = payload["cell_two_star_choice"]["stars"].get(
        f"entries={HEADLINE_FIELD},lean={HEADLINE_LEAN}", []
    )
    print(json.dumps(headline, indent=2))
    print(
        json.dumps(
            payload["cell_two_star_choice"]["side_flips"].get(
                f"entries={HEADLINE_FIELD},lean={HEADLINE_LEAN}", []
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
