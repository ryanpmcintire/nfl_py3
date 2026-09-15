from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_ats.best_pick_nomination import SERVED_SPREAD_THRESHOLD, dispersion_pool_from_frame
from nfl_ats.clv import week_blocked_bootstrap

PAIRED = Path("artifacts/ridge_alpha_promotion/20260818T221459Z/opener_paired.parquet")
DISPERSION = Path("artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet")
SERVED = Path("artifacts/opener_evaluation/20260913T134234Z/per_game.parquet")
OUT_ROOT = Path("artifacts/best_pick_served_score_ranker")
SEED = 20260914
SAMPLES = 20_000
ARMS = (
    "a0_incumbent",
    "a1_served_score",
    "a2_served_score_no_dispersion",
    "control_foresight",
)


def build_frame() -> pd.DataFrame:
    paired = pd.read_parquet(PAIRED)[
        ["game_id", "season", "week", "tue_open_home_spread", "candidate_prob_open"]
    ].copy()
    paired["game_id"] = paired["game_id"].astype(str)
    dispersion = pd.read_parquet(DISPERSION)[["game_id", "spread_std"]].copy()
    dispersion["game_id"] = dispersion["game_id"].astype(str)
    served = pd.read_parquet(SERVED)[
        ["game_id", "home_cover_probability_at_open", "correct_at_open_probability_rule"]
    ].copy()
    served["game_id"] = served["game_id"].astype(str)
    frame = paired.merge(dispersion, on="game_id", how="left", validate="one_to_one").merge(
        served, on="game_id", how="left", validate="one_to_one"
    )
    frame["candidate_dist"] = (frame["candidate_prob_open"] - 0.5).abs()
    frame["served_dist"] = (frame["home_cover_probability_at_open"] - 0.5).abs()
    frame["small_spread"] = frame["tue_open_home_spread"].abs() < SERVED_SPREAD_THRESHOLD
    return frame


def week_pool(week_frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    pool = dispersion_pool_from_frame(week_frame[["game_id", "spread_std"]])
    table = week_frame.merge(
        pool.frame[["game_id", "pool_pass"]], on="game_id", how="inner", validate="one_to_one"
    )
    return table.loc[table["pool_pass"]].copy(), pool.fallback


def screen(pool: pd.DataFrame) -> pd.DataFrame:
    small = pool.loc[pool["small_spread"]]
    return small if not small.empty else pool


def nominate(pool: pd.DataFrame, score: str) -> str | None:
    usable = pool.dropna(subset=[score])
    if usable.empty:
        return None
    top = usable[score].max()
    tied = usable.loc[usable[score].eq(top)]
    ordered = tied.sort_values(
        ["spread_std", "game_id"], ascending=[True, True], na_position="last"
    )
    return str(ordered.iloc[0]["game_id"])


def foresight(pool: pd.DataFrame) -> str | None:
    usable = pool.dropna(subset=["correct_at_open_probability_rule"])
    if usable.empty:
        return None
    ordered = usable.sort_values(
        ["correct_at_open_probability_rule", "game_id"], ascending=[False, True]
    )
    return str(ordered.iloc[0]["game_id"])


def weekly_table(frame: pd.DataFrame) -> pd.DataFrame:
    correctness = frame.set_index("game_id")["correct_at_open_probability_rule"].to_dict()
    abs_spread = frame.set_index("game_id")["tue_open_home_spread"].abs().to_dict()
    rows: list[dict[str, object]] = []
    for (season, week), week_frame in frame.groupby(["season", "week"], sort=True):
        pool, fallback = week_pool(week_frame)
        if pool.empty:
            continue
        screened = screen(pool)
        row: dict[str, object] = {
            "season": int(season),
            "week": int(week),
            "n_games": len(week_frame),
            "n_pool": len(pool),
            "n_screened": len(screened),
            "pool_fallback": bool(fallback),
        }
        nominees = {
            "a0_incumbent": nominate(screened, "candidate_dist"),
            "a1_served_score": nominate(screened, "served_dist"),
            "a2_served_score_no_dispersion": nominate(screen(week_frame), "served_dist"),
            "control_foresight": foresight(screened),
        }
        for arm, game_id in nominees.items():
            row[f"{arm}_game_id"] = game_id
            row[f"{arm}_correct"] = correctness.get(game_id) if game_id else None
            row[f"{arm}_abs_spread"] = abs_spread.get(game_id) if game_id else None
        rows.append(row)
    return pd.DataFrame(rows)


def paired_cell(weekly: pd.DataFrame, arm: str) -> dict[str, float | int]:
    pair = weekly.dropna(subset=["a0_incumbent_correct", f"{arm}_correct"]).copy()
    pair["delta"] = pair[f"{arm}_correct"] - pair["a0_incumbent_correct"]

    def metric(sample: pd.DataFrame) -> dict[str, float]:
        return {"delta": float(sample["delta"].mean() * 100.0)}

    boot = week_blocked_bootstrap(
        pair, metric, block="week", samples=SAMPLES, seed=SEED, metric_columns=["delta"]
    )
    record = boot.loc[boot["metric"] == "delta"].iloc[0]
    return {
        "weeks_paired": len(pair),
        "nominee_differs": int((pair[f"{arm}_game_id"] != pair["a0_incumbent_game_id"]).sum()),
        "outcome_differs": int((pair["delta"] != 0).sum()),
        "delta_accuracy_points": float(record["estimate"]),
        "ci_low": float(record["lower"]),
        "ci_high": float(record["upper"]),
        "probability_positive": float(record["probability_positive"]),
    }


def run() -> dict[str, object]:
    frame = build_frame()
    weekly = weekly_table(frame)
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "weeks": len(weekly),
        "games": len(frame),
        "seed": SEED,
        "samples": SAMPLES,
        "arms": {},
        "cells": {},
    }
    for arm in ARMS:
        graded = weekly[f"{arm}_correct"].dropna()
        summary["arms"][arm] = {
            "weeks_scored": len(graded),
            "weeks_won": int(graded.sum()),
            "hit_rate": float(graded.mean()) if len(graded) else None,
        }
    for arm in ARMS[1:]:
        summary["cells"][f"{arm}_vs_a0"] = paired_cell(weekly, arm)

    graded = weekly.dropna(subset=["a0_incumbent_correct", "a1_served_score_correct"])
    differs = graded["a0_incumbent_game_id"] != graded["a1_served_score_game_id"]
    summary["coherence"] = {
        "weeks": len(graded),
        "star_is_not_card_top_pick": int(differs.sum()),
        "a0_hit_when_arms_agree": float(graded.loc[~differs, "a0_incumbent_correct"].mean()),
        "a0_hit_when_arms_differ": float(graded.loc[differs, "a0_incumbent_correct"].mean()),
        "a1_hit_when_arms_differ": float(graded.loc[differs, "a1_served_score_correct"].mean()),
    }
    summary["seasons"] = [
        {
            "season": int(season),
            "weeks": int(block["a0_incumbent_correct"].notna().sum()),
            "a0": float(block["a0_incumbent_correct"].dropna().mean()),
            "a1": float(block["a1_served_score_correct"].dropna().mean()),
        }
        for season, block in weekly.groupby("season", sort=True)
    ]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out / "weekly.csv", index=False)
    weekly.to_parquet(out / "weekly.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifact"] = str(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="score the card's served decision score as the Best Pick ranker "
        "against the incumbent alpha=2000 ranker on the frozen 2020-2025 opener archive"
    )
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
