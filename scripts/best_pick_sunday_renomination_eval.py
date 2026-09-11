from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.best_pick_nomination import (
    NOMINATION_RIDGE_ALPHA,
    dispersion_pool_from_frame,
    select_nominee,
)
from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.io import atomic_json, atomic_parquet, atomic_text, run_id
from nfl_ats.market_data import load_quote_history, spread_consensus, tuesday_opener_quotes
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock

OPENER_ARCHIVE = "artifacts/ridge_alpha_promotion/20260818T221459Z/opener_paired.parquet"
FEATURE_TABLE = "data/processed/game_features_weak_stack.parquet"
FEATURE_PROFILE = "weak_stack"
REGRESSOR = "ridge"
TIE_DECIMALS = 12

ARMS = (
    "t0_tuesday",
    "t0b_tuesday_frozen_side",
    "s1_sunday_literal",
    "s2_sunday_frozen_line",
    "s3_sunday_full",
    "pc_oracle",
)

PAIRED_CELLS = (
    ("t0b_tuesday_frozen_side", "t0_tuesday"),
    ("s1_sunday_literal", "t0_tuesday"),
    ("s2_sunday_frozen_line", "t0_tuesday"),
    ("s3_sunday_full", "t0_tuesday"),
    ("s2_sunday_frozen_line", "t0b_tuesday_frozen_side"),
    ("pc_oracle", "t0_tuesday"),
)


def smoothed_probability(samples: np.ndarray, threshold: float) -> float:
    successes = float(np.count_nonzero(samples > threshold))
    return (successes + 0.5) / (len(samples) + 1.0)


def select_with_tie_rule(frame: pd.DataFrame, stat_column: str) -> tuple[tuple[str, ...], int]:
    if frame.empty:
        raise ValueError("tie rule needs at least one candidate")
    table = frame.copy()
    stat = pd.to_numeric(table[stat_column], errors="coerce").round(TIE_DECIMALS)
    table = table.loc[stat.eq(stat.max())]
    if len(table) == 1:
        return (str(table.iloc[0]["game_id"]),), 0
    dispersion = pd.to_numeric(table["tie_dispersion"], errors="coerce").round(6)
    if dispersion.notna().any() and dispersion.nunique(dropna=True) > 1:
        table = table.loc[dispersion.eq(dispersion.min())]
        if len(table) == 1:
            return (str(table.iloc[0]["game_id"]),), 1
    magnitude = pd.to_numeric(table["tue_open_home_spread"], errors="coerce").abs().round(6)
    if magnitude.nunique(dropna=True) > 1:
        table = table.loc[magnitude.eq(magnitude.min())]
        if len(table) == 1:
            return (str(table.iloc[0]["game_id"]),), 2
    kickoff = pd.to_datetime(table["kickoff"], utc=True)
    if kickoff.nunique() > 1:
        table = table.loc[kickoff.eq(kickoff.min())]
        if len(table) == 1:
            return (str(table.iloc[0]["game_id"]),), 3
    return tuple(sorted(str(value) for value in table["game_id"])), 4


def score_selection(week_frame: pd.DataFrame, chosen: tuple[str, ...]) -> float:
    outcomes = week_frame.set_index("game_id").loc[list(chosen), "baseline_correct_open"]
    resolved = pd.to_numeric(outcomes, errors="coerce").dropna()
    return float(resolved.mean()) if len(resolved) else float("nan")


def sunday_market_read(
    quotes: pd.DataFrame, game_ids: set[str], instant: pd.Timestamp, sunday_date: Any
) -> pd.DataFrame:
    window = quotes.loc[
        quotes["nflverse_game_id"].isin(game_ids)
        & quotes["observed_at_utc"].le(instant)
        & quotes["et_date"].eq(sunday_date)
        & quotes["observed_at_utc"].lt(quotes["commence_time_utc"])
    ]
    if window.empty:
        return pd.DataFrame(columns=["game_id", "sunday_home_spread", "sunday_std", "sunday_books"])
    consensus = spread_consensus(window)
    return consensus.rename(
        columns={
            "nflverse_game_id": "game_id",
            "consensus_home_spread": "sunday_home_spread",
            "spread_std": "sunday_std",
            "bookmakers": "sunday_books",
        }
    )[["game_id", "sunday_home_spread", "sunday_std", "sunday_books"]]


def overridden(target: pd.DataFrame, lines: pd.Series) -> pd.DataFrame:
    frame = target.copy()
    frame["spread_line"] = pd.to_numeric(frame["game_id"].map(lines), errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(frame["spread_line"].to_numpy(dtype=float)).all():
        raise ValueError("line override left a game without a spread")
    return frame


def build_week(
    features: pd.DataFrame,
    archive_week: pd.DataFrame,
    tuesday_dispersion: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    season: int,
    week: int,
    instant_offset: pd.Timedelta,
) -> dict[str, Any] | None:
    game_ids = set(archive_week["game_id"].astype(str))
    target_all, models = fit_margin_models_for_week(
        features,
        season=season,
        week=week,
        regressor=REGRESSOR,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        feature_profile=FEATURE_PROFILE,
        ridge_alpha=NOMINATION_RIDGE_ALPHA,
        methods=("market_residual",),
    )
    model = models["market_residual"]
    target_all = target_all.copy()
    target_all["game_id"] = target_all["game_id"].astype(str)
    target = target_all.loc[target_all["game_id"].isin(game_ids)].reset_index(drop=True)
    if len(target) != len(game_ids):
        return None

    kickoffs = pd.to_datetime(target["kickoff"], utc=True)
    lock = sunday_pick_lock(kickoffs)
    instant = lock - instant_offset
    sunday_date = lock.tz_convert("America/New_York").date()

    archive = archive_week.copy()
    archive["game_id"] = archive["game_id"].astype(str)
    sunday = sunday_market_read(quotes, game_ids, instant, sunday_date)
    table = (
        target[["game_id", "kickoff"]]
        .merge(archive, on="game_id", how="inner", validate="one_to_one")
        .merge(tuesday_dispersion, on="game_id", how="left", validate="one_to_one")
        .merge(sunday, on="game_id", how="left", validate="one_to_one")
    )
    table["kickoff"] = pd.to_datetime(table["kickoff"], utc=True)
    table["deadline"] = table["kickoff"].map(lambda value: pick_deadline(value, lock))
    table["playable"] = table["deadline"].gt(instant)

    tue_lines = table.set_index("game_id")["tue_open_home_spread"]
    tue_frame = overridden(target, tue_lines)
    tue_forecast = model.predict(tue_frame)
    tue_probability = pd.Series(
        tue_forecast["home_cover_probability"].to_numpy(), index=tue_frame["game_id"].astype(str)
    )
    tue_margin = pd.Series(
        tue_forecast["predicted_margin"].to_numpy(), index=tue_frame["game_id"].astype(str)
    )

    playable_missing_sunday = int(table.loc[table["playable"], "sunday_home_spread"].isna().sum())
    if playable_missing_sunday:
        return {
            "season": season,
            "week": week,
            "skipped": "missing_sunday_line",
            "playable_missing_sunday": playable_missing_sunday,
        }

    sun_lines = table.set_index("game_id")["sunday_home_spread"].fillna(tue_lines)
    sun_frame = overridden(target, sun_lines)
    sun_forecast = model.predict(sun_frame)
    sun_probability = pd.Series(
        sun_forecast["home_cover_probability"].to_numpy(), index=sun_frame["game_id"].astype(str)
    )
    sun_margin = pd.Series(
        sun_forecast["predicted_margin"].to_numpy(), index=sun_frame["game_id"].astype(str)
    )

    residuals = np.asarray(model.residuals, dtype=float)
    table["p_tue_at_tue"] = table["game_id"].map(tue_probability)
    table["p_sun_at_sun"] = table["game_id"].map(sun_probability)
    table["predicted_margin_tue"] = table["game_id"].map(tue_margin)
    table["predicted_margin_sun"] = table["game_id"].map(sun_margin)
    table["p_home_tue_at_frozen"] = [
        smoothed_probability(centre + residuals, float(line))
        for centre, line in zip(
            table["predicted_margin_tue"], table["tue_open_home_spread"], strict=True
        )
    ]
    table["p_home_sun_at_frozen"] = [
        smoothed_probability(centre + residuals, float(line))
        for centre, line in zip(
            table["predicted_margin_sun"], table["tue_open_home_spread"], strict=True
        )
    ]
    pick_home = table["baseline_pick_home"].astype(bool)
    table["stat_t0"] = (table["p_tue_at_tue"] - 0.5).abs()
    table["stat_t0b"] = np.where(
        pick_home, table["p_home_tue_at_frozen"], 1.0 - table["p_home_tue_at_frozen"]
    )
    table["stat_s1"] = (table["p_sun_at_sun"] - 0.5).abs()
    table["stat_s2"] = np.where(
        pick_home, table["p_home_sun_at_frozen"], 1.0 - table["p_home_sun_at_frozen"]
    )
    table["stat_pc"] = pd.to_numeric(table["baseline_correct_open"], errors="coerce").fillna(-1.0)

    playable_rows = table.loc[table["playable"]]
    sunday_pool = dispersion_pool_from_frame(
        playable_rows[["game_id", "sunday_std"]].rename(columns={"sunday_std": "spread_std"})
    )
    table["sunday_pool_pass"] = (
        table["game_id"].map(sunday_pool.frame.set_index("game_id")["pool_pass"]).fillna(False)
    )
    tuesday_pool = dispersion_pool_from_frame(
        table[["game_id", "opener_std"]].rename(columns={"opener_std": "spread_std"})
    )
    table["tuesday_pool_pass"] = table["game_id"].map(
        tuesday_pool.frame.set_index("game_id")["pool_pass"]
    )

    served_pool = table.loc[table["tuesday_pool_pass"].astype(bool)].rename(
        columns={"stat_t0": "candidate_dist", "opener_std": "spread_std"}
    )
    served_nominee, served_tied, served_break = select_nominee(served_pool)

    table["tie_dispersion"] = table["opener_std"]
    tuesday_candidates = table.loc[table["tuesday_pool_pass"].astype(bool)]
    t0_choice, t0_level = select_with_tie_rule(tuesday_candidates, "stat_t0")
    t0b_choice, t0b_level = select_with_tie_rule(tuesday_candidates, "stat_t0b")

    t0_deadline = table.set_index("game_id").loc[list(t0_choice), "deadline"].min()
    frozen = bool(t0_deadline <= instant)

    playable_tuesday = tuesday_candidates.loc[tuesday_candidates["playable"]]
    sunday_table = table.copy()
    sunday_table["tie_dispersion"] = sunday_table["sunday_std"]
    playable_tuesday_sun = sunday_table.loc[
        sunday_table["tuesday_pool_pass"].astype(bool) & sunday_table["playable"]
    ]
    playable_sunday_pool = sunday_table.loc[
        sunday_table["sunday_pool_pass"].astype(bool) & sunday_table["playable"]
    ]

    if frozen or playable_tuesday_sun.empty:
        s1_choice, s1_level = t0_choice, t0_level
        s2_choice, s2_level = t0_choice, t0_level
        pc_choice, pc_level = t0_choice, t0_level
    else:
        s1_choice, s1_level = select_with_tie_rule(playable_tuesday_sun, "stat_s1")
        s2_choice, s2_level = select_with_tie_rule(playable_tuesday_sun, "stat_s2")
        pc_choice, pc_level = select_with_tie_rule(playable_tuesday, "stat_pc")
    if frozen or playable_sunday_pool.empty:
        s3_choice, s3_level = t0_choice, t0_level
    else:
        s3_choice, s3_level = select_with_tie_rule(playable_sunday_pool, "stat_s2")

    choices = {
        "t0_tuesday": (t0_choice, t0_level),
        "t0b_tuesday_frozen_side": (t0b_choice, t0b_level),
        "s1_sunday_literal": (s1_choice, s1_level),
        "s2_sunday_frozen_line": (s2_choice, s2_level),
        "s3_sunday_full": (s3_choice, s3_level),
        "pc_oracle": (pc_choice, pc_level),
    }

    row: dict[str, Any] = {
        "season": season,
        "week": week,
        "skipped": None,
        "n_games": len(table),
        "n_playable": int(table["playable"].sum()),
        "n_tuesday_pool": int(table["tuesday_pool_pass"].sum()),
        "n_sunday_pool": int(table["sunday_pool_pass"].sum()),
        "n_playable_tuesday_pool": len(playable_tuesday),
        "instant_utc": instant,
        "sunday_lock_utc": lock,
        "frozen_nominee": frozen,
        "tuesday_pool_fallback": tuesday_pool.fallback_reason,
        "sunday_pool_fallback": sunday_pool.fallback_reason,
        "median_tuesday_std": float(pd.to_numeric(table["opener_std"]).median()),
        "median_sunday_std": float(pd.to_numeric(table["sunday_std"]).median()),
        "median_sunday_books": float(pd.to_numeric(table["sunday_books"]).median()),
        "mean_abs_line_move": float(
            (table["sunday_home_spread"] - table["tue_open_home_spread"]).abs().mean()
        ),
        "margin_move_slope": float(
            (
                (table["predicted_margin_sun"] - table["predicted_margin_tue"])
                / (table["sunday_home_spread"] - table["tue_open_home_spread"])
            )
            .replace([np.inf, -np.inf], np.nan)
            .median()
        ),
        "max_abs_prob_change_own_line": float(
            (table["p_sun_at_sun"] - table["p_tue_at_tue"]).abs().max()
        ),
        "max_abs_prob_change_frozen_line": float(
            (table["p_home_sun_at_frozen"] - table["p_home_tue_at_frozen"]).abs().max()
        ),
        "served_nominee": served_nominee,
        "served_n_tied": served_tied,
        "served_tie_break": served_break,
        "archive_candidate_corr": float(
            pd.to_numeric(table["p_tue_at_tue"]).corr(pd.to_numeric(table["candidate_prob_open"]))
        ),
    }
    for arm, (chosen, level) in choices.items():
        row[f"{arm}_game_id"] = "|".join(chosen)
        row[f"{arm}_tie_level"] = level
        row[f"{arm}_n_tied"] = len(chosen)
        row[f"{arm}_correct"] = score_selection(table, chosen)
    return row


def paired_cell(weekly: pd.DataFrame, arm: str, base: str, *, samples: int, seed: int) -> dict:
    columns = [
        "season",
        "week",
        f"{arm}_correct",
        f"{base}_correct",
        f"{arm}_game_id",
        f"{base}_game_id",
    ]
    frame = weekly.loc[
        weekly[f"{arm}_correct"].notna() & weekly[f"{base}_correct"].notna(), columns
    ].copy()
    frame["delta"] = (frame[f"{arm}_correct"] - frame[f"{base}_correct"]) * 100.0
    frame["arm_points"] = frame[f"{arm}_correct"] * 100.0
    frame["base_points"] = frame[f"{base}_correct"] * 100.0

    def metric(sample: pd.DataFrame) -> dict[str, float]:
        return {"delta_accuracy_points": float(sample["delta"].mean())}

    boot = week_blocked_bootstrap(
        frame,
        metric,
        block="week",
        samples=samples,
        seed=seed,
        metric_columns=["season", "week", "delta"],
    )
    record = boot.iloc[0]
    differs = frame[f"{arm}_game_id"].ne(frame[f"{base}_game_id"])
    outcome_differs = frame["delta"].ne(0.0)
    return {
        "arm": arm,
        "base": base,
        "weeks_paired": len(frame),
        "arm_accuracy": float(frame[f"{arm}_correct"].mean()),
        "base_accuracy": float(frame[f"{base}_correct"].mean()),
        "effect_accuracy_points": float(record["estimate"]),
        "interval_low": float(record["lower"]),
        "interval_high": float(record["upper"]),
        "probability_positive": float(record["probability_positive"]),
        "weeks_nominee_differs": int(differs.sum()),
        "weeks_outcome_differs": int(outcome_differs.sum()),
        "weeks_arm_better": int((frame["delta"] > 0).sum()),
        "weeks_base_better": int((frame["delta"] < 0).sum()),
    }


def season_table(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, grp in weekly.groupby("season"):
        row: dict[str, Any] = {"season": int(season), "weeks": len(grp)}
        for arm in ARMS:
            resolved = grp[f"{arm}_correct"].dropna()
            row[f"{arm}_weeks"] = len(resolved)
            row[f"{arm}_accuracy"] = float(resolved.mean()) if len(resolved) else float("nan")
        both = grp.loc[
            grp["s2_sunday_frozen_line_correct"].notna() & grp["t0_tuesday_correct"].notna()
        ]
        row["s2_minus_t0_points"] = (
            float(
                (both["s2_sunday_frozen_line_correct"] - both["t0_tuesday_correct"]).mean() * 100.0
            )
            if len(both)
            else float("nan")
        )
        row["s2_nominee_differs"] = int(
            both["s2_sunday_frozen_line_game_id"].ne(both["t0_tuesday_game_id"]).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    archive = pd.read_parquet(repo / OPENER_ARCHIVE)
    archive["game_id"] = archive["game_id"].astype(str)
    features = pd.read_parquet(repo / FEATURE_TABLE)
    features["game_id"] = features["game_id"].astype(str)

    quotes = load_quote_history(repo / "data" / "market" / "raw")
    quotes = quotes.loc[quotes["sport_key"].astype(str).eq("americanfootball_nfl")].copy()
    quotes["observed_at_utc"] = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    quotes["commence_time_utc"] = pd.to_datetime(quotes["commence_time_utc"], utc=True)
    quotes["nflverse_game_id"] = quotes["nflverse_game_id"].astype(str)
    quotes["et_date"] = quotes["observed_at_utc"].dt.tz_convert("America/New_York").dt.date

    opener = tuesday_opener_quotes(quotes)
    tuesday_dispersion = (
        opener.rename(columns={"nflverse_game_id": "game_id"})[["game_id", "opener_std"]]
        .assign(game_id=lambda frame: frame["game_id"].astype(str))
        .drop_duplicates("game_id")
    )

    if args.seasons:
        wanted = {int(value) for value in args.seasons.split(",")}
        archive = archive.loc[archive["season"].isin(wanted)]

    offset = pd.Timedelta(minutes=args.instant_minutes_before_lock)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for (season, week), archive_week in archive.groupby(["season", "week"]):
        built = build_week(
            features,
            archive_week,
            tuesday_dispersion,
            quotes,
            season=int(season),
            week=int(week),
            instant_offset=offset,
        )
        if built is None:
            skipped.append({"season": int(season), "week": int(week), "reason": "feature_gap"})
            continue
        if built.get("skipped"):
            skipped.append(built)
            continue
        rows.append(built)
    weekly = pd.DataFrame(rows)

    cells = [
        paired_cell(weekly, arm, base, samples=args.samples, seed=args.seed)
        for arm, base in PAIRED_CELLS
    ]
    seasons = season_table(weekly)

    hit_rates = {}
    for arm in ARMS:
        resolved = weekly[f"{arm}_correct"].dropna()
        hit_rates[arm] = {
            "weeks_scored": len(resolved),
            "accuracy": float(resolved.mean()),
            "hits": float(resolved.sum()),
            "weeks_tied_at_level_5": int((weekly[f"{arm}_tie_level"] == 4).sum()),
            "mean_tied_games": float(weekly[f"{arm}_n_tied"].mean()),
        }

    served_matches = int(
        weekly.apply(
            lambda row: row["served_nominee"] in str(row["t0_tuesday_game_id"]).split("|"), axis=1
        ).sum()
    )
    summary: dict[str, Any] = {
        "family": "best_pick_renomination",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "instant_minutes_before_lock": args.instant_minutes_before_lock,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "population": {
            "weeks": len(weekly),
            "games": int(weekly["n_games"].sum()),
            "seasons": sorted(int(value) for value in weekly["season"].unique()),
            "weeks_skipped": skipped,
            "weeks_frozen_nominee": int(weekly["frozen_nominee"].sum()),
            "median_sunday_books": float(weekly["median_sunday_books"].median()),
            "mean_abs_line_move": float(weekly["mean_abs_line_move"].mean()),
            "margin_move_slope_median": float(weekly["margin_move_slope"].median()),
            "max_abs_prob_change_own_line": float(weekly["max_abs_prob_change_own_line"].max()),
            "max_abs_prob_change_frozen_line": float(
                weekly["max_abs_prob_change_frozen_line"].max()
            ),
            "archive_candidate_corr_median": float(weekly["archive_candidate_corr"].median()),
            "served_nominee_matches_tie_rule_weeks": served_matches,
            "served_tie_break_counts": weekly["served_tie_break"].value_counts().to_dict(),
        },
        "hit_rates": hit_rates,
        "cells": cells,
        "seasons": seasons.to_dict(orient="records"),
    }

    out = repo / "artifacts" / "best_pick_sunday_renomination" / run_id()
    out.mkdir(parents=True, exist_ok=True)
    atomic_parquet(weekly, out / "weekly.parquet")
    atomic_text(weekly.to_csv(index=False), out / "weekly.csv")
    atomic_text(seasons.to_csv(index=False), out / "seasons.csv")
    atomic_json(summary, out / "summary.json")
    summary["artifact"] = str(out.relative_to(repo))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-nominate the weekly Best Pick from Sunday-morning refreshed probabilities and "
            "score it against the frozen Tuesday nomination (LEAD-53)."
        )
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--instant-minutes-before-lock", type=int, default=210)
    parser.add_argument("--seasons", default="")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
