"""MEASURE-ONLY: does re-nominating the Best Pick at the pick deadline beat the Tuesday nominee?

Predeclared in ``docs/best_pick_deadline_renomination.md`` before any number
below was computed. Reads frozen archives only; fits nothing; writes no
registry file.

BINDING (owner mandates, restated because this output feeds a registry write):
an interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (a) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (b) bounded by a
positive control proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record it with ``nfl-ats weak-signals record``,
report ``probability_positive``, never "contains zero". The registry
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Within-week game correlation is ZERO by owner
mandate: week-blocked bootstrap only. Decide on expected value -- forced picks,
so P+ above 0.5 on the played card is played; thresholds like 0.90 govern only
what docs may claim.

Run::

    .\\.tools\\uv.exe run --no-sync python scripts/best_pick_deadline_renomination_eval.py \
        --repo F:/Repos/nfl_py3 --out-dir <artifacts dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.best_pick_nomination import (  # noqa: E402
    dispersion_pool_from_frame,
    select_nominee,
)
from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.sharp_book_movement_features import (  # noqa: E402
    THRESHOLD,
    late_week_follow_frame,
)

FAMILY = "best_pick_deadline_renomination"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SEASONS = (2023, 2024, 2025)
RENOMINATION_HOUR_ET = 12
EASTERN = "America/New_York"

ARMS = ("a0_tuesday", "a1_deadline_rerank", "a2_flip_avoidant", "a3_biggest_move", "oracle")


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sunday_instant(week_first_commence_utc: pd.Timestamp, hour: int) -> pd.Timestamp:
    """That week's Sunday at ``hour`` ET, on the anchor the served follow rule uses."""
    anchor = pd.Timestamp(week_first_commence_utc).tz_convert(EASTERN)
    local = anchor.tz_localize(None).normalize()
    sunday = local + pd.Timedelta(days=(6 - anchor.weekday()) % 7)
    return (sunday + pd.Timedelta(hours=hour)).tz_localize(EASTERN).tz_convert("UTC")


def deadline_dispersion(quotes: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    """Cross-book spread std from each book's last quote before its game's deadline."""
    frame = quotes.loc[
        quotes["market"].eq("spreads") & quotes["outcome_side"].eq("HOME"),
        [
            "nflverse_game_id",
            "bookmaker_key",
            "home_spread_line",
            "observed_at_utc",
            "bookmaker_last_update_utc",
        ],
    ].rename(columns={"nflverse_game_id": "game_id"})
    frame = frame.merge(cutoffs, on="game_id", how="inner")
    for column in ("observed_at_utc", "bookmaker_last_update_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["home_spread_line"] = pd.to_numeric(frame["home_spread_line"], errors="coerce")
    frame = frame.loc[
        frame["observed_at_utc"].lt(frame["deadline_utc"])
        & frame["bookmaker_last_update_utc"].le(frame["observed_at_utc"])
        & np.isfinite(frame["home_spread_line"])
    ]
    if frame.empty:
        return pd.DataFrame(columns=["game_id", "deadline_spread_std", "deadline_books"])
    last = (
        frame.sort_values("observed_at_utc")
        .groupby(["game_id", "bookmaker_key"], as_index=False)
        .tail(1)
    )
    return (
        last.groupby("game_id", as_index=False)
        .agg(
            deadline_spread_std=("home_spread_line", "std"),
            deadline_books=("bookmaker_key", "nunique"),
        )
        .reset_index(drop=True)
    )


def side_correct(side: str, margin_vs_open: float) -> float:
    """Correct at the frozen Tuesday line; NaN on a push."""
    if not np.isfinite(margin_vs_open) or margin_vs_open == 0.0:
        return float("nan")
    return float((side == "HOME") == (margin_vs_open > 0.0))


def ranked_nominees(candidates: pd.DataFrame) -> list[str]:
    """Production's own selector applied repeatedly to give a full ranking."""
    remaining = candidates.copy()
    order: list[str] = []
    while not remaining.empty:
        game_id, _, _ = select_nominee(remaining)
        order.append(game_id)
        remaining = remaining.loc[remaining["game_id"].astype(str).ne(game_id)]
    return order


def build_work(repo: Path, ranker: ModuleType, composed: ModuleType) -> pd.DataFrame:
    work = ranker.load_working_frame(repo / "artifacts/ridge_alpha_promotion/20260818T221459Z")
    work, _ = ranker.build_dispersion_pool(
        work, repo / "artifacts/odds_microstructure/20260818T225430Z"
    )
    work = composed.attach_production_pool(work)
    mismatch = work.loc[work["pool_pass"] != work["dispersion_pool_pass"]]
    if not mismatch.empty:
        raise SystemExit(f"production pool disagrees with the eval pool on {len(mismatch)} games")
    baseline = pd.read_parquet(
        repo / "artifacts/ridge_alpha_promotion/20260818T221459Z/opener_baseline.parquet"
    )
    work = work.merge(
        baseline[["game_id", "margin_vs_open", "close_home_spread"]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    work["game_id"] = work["game_id"].astype(str)
    return work.loc[work["season"].isin(SEASONS)].reset_index(drop=True)


def evaluate_week(
    group: pd.DataFrame,
    quotes: pd.DataFrame,
    kickoffs: pd.DataFrame,
) -> dict[str, Any]:
    """One week: every arm's nominee, its graded correctness, and the diagnostics."""
    games = kickoffs.loc[kickoffs["game_id"].isin(set(group["game_id"]))].copy()
    if len(games) != len(group):
        raise ValueError("kickoff table does not cover the archive week")
    instant = sunday_instant(games["week_first_commence_utc"].min(), RENOMINATION_HOUR_ET)
    games["cutoff_utc"] = instant
    sides = group.set_index("game_id")["baseline_pick_home"].map({True: "HOME", False: "AWAY"})
    week_quotes = quotes.loc[quotes["nflverse_game_id"].isin(set(group["game_id"]))]
    exposure, refused = late_week_follow_frame(
        week_quotes,
        games[["game_id", "commence_time_utc", "week_first_commence_utc", "cutoff_utc"]],
        now=instant,
        tuesday_pick_side=sides,
    )
    exposure = exposure.copy()
    exposure["game_id"] = exposure["game_id"].astype(str)
    exposure["tuesday_pick_side"] = exposure["game_id"].map(sides)
    missing_side = exposure["movement_would_be_pick_side"].isna()
    exposure.loc[missing_side, "movement_would_be_pick_side"] = exposure.loc[
        missing_side, "tuesday_pick_side"
    ]
    exposure["movement_flip"] = exposure["movement_would_be_pick_side"].ne(
        exposure["tuesday_pick_side"]
    )

    table = group.merge(
        exposure[
            [
                "game_id",
                "commence_time_utc",
                "cutoff_utc",
                "equal_net_move",
                "eligible_books",
                "movement_would_be_pick_side",
                "movement_flip",
            ]
        ],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    table["tuesday_side"] = table["baseline_pick_home"].map({True: "HOME", False: "AWAY"})
    table["playable"] = pd.to_datetime(table["commence_time_utc"], utc=True).gt(instant)
    table["deadline_utc"] = pd.to_datetime(table["commence_time_utc"], utc=True).where(
        ~table["playable"], instant
    )
    dispersion = deadline_dispersion(week_quotes, table[["game_id", "deadline_utc"]])
    table = table.merge(dispersion, on="game_id", how="left")

    table["primary_correct"] = [
        side_correct(side, margin)
        for side, margin in zip(
            table["movement_would_be_pick_side"], table["margin_vs_open"], strict=True
        )
    ]
    table["secondary_correct"] = [
        side_correct(side, margin)
        for side, margin in zip(table["tuesday_side"], table["margin_vs_open"], strict=True)
    ]

    pool = table.loc[table["pool_pass"]].copy()
    a0, a0_tied, a0_tie_break = select_nominee(pool)
    a0_playable = bool(table.loc[table["game_id"].eq(a0), "playable"].iloc[0])
    playable_pool = pool.loc[pool["playable"]].copy()
    playable_all = table.loc[table["playable"]].copy()

    if not a0_playable or playable_pool.empty:
        a1 = a2 = a3 = oracle = a0
        deadline_pool_size = 0
        deadline_fallback = ""
    else:
        deadline_frame = playable_all[["game_id", "deadline_spread_std"]].rename(
            columns={"deadline_spread_std": "spread_std"}
        )
        deadline_pool = dispersion_pool_from_frame(deadline_frame)
        passes = deadline_pool.frame.set_index("game_id")["pool_pass"]
        a1_candidates = playable_all.loc[
            playable_all["game_id"].map(passes).fillna(False).astype(bool)
        ].copy()
        a1_candidates["spread_std"] = a1_candidates["deadline_spread_std"]
        a1, _, _ = select_nominee(a1_candidates)
        deadline_pool_size = int(deadline_pool.n_pool_pass)
        deadline_fallback = deadline_pool.fallback_reason or ""

        a0_flip = bool(table.loc[table["game_id"].eq(a0), "movement_flip"].iloc[0])
        if not a0_flip:
            a2 = a0
        else:
            order = ranked_nominees(playable_pool)
            flips = playable_pool.set_index("game_id")["movement_flip"].to_dict()
            a2 = next((game for game in order if not flips[game]), a0)

        favourable = playable_pool.loc[
            playable_pool["equal_net_move"].abs().ge(THRESHOLD)
            & (playable_pool["equal_net_move"].gt(0) == playable_pool["tuesday_side"].eq("HOME"))
        ].copy()
        if favourable.empty:
            a3 = a0
        else:
            favourable["abs_move"] = favourable["equal_net_move"].abs()
            a3 = str(
                favourable.sort_values(["abs_move", "game_id"], ascending=[False, True]).iloc[0][
                    "game_id"
                ]
            )

        winners = playable_pool.loc[playable_pool["primary_correct"].eq(1.0)]
        oracle = ranked_nominees(winners)[0] if not winners.empty else a0

    picks = {
        "a0_tuesday": a0,
        "a1_deadline_rerank": a1,
        "a2_flip_avoidant": a2,
        "a3_biggest_move": a3,
        "oracle": oracle,
    }
    indexed = table.set_index("game_id")
    row: dict[str, Any] = {
        "season": int(group["season"].iloc[0]),
        "week": int(group["week"].iloc[0]),
        "n_games": len(table),
        "n_pool": int(table["pool_pass"].sum()),
        "n_playable": int(table["playable"].sum()),
        "n_playable_pool": len(playable_pool),
        "renomination_instant_utc": instant.isoformat(),
        "a0_playable": a0_playable,
        "a0_n_tied": a0_tied,
        "a0_tie_break": a0_tie_break,
        "deadline_pool_size": deadline_pool_size,
        "deadline_pool_fallback": deadline_fallback,
        "refused_quote_rows": int(refused),
        "n_flips": int(table["movement_flip"].sum()),
        "n_with_late_week_evidence": int(table["eligible_books"].gt(0).sum()),
        "games_missing_deadline_std": int(table["deadline_spread_std"].isna().sum()),
        "median_tuesday_spread_std": float(table["spread_std"].median()),
        "median_deadline_spread_std": float(table["deadline_spread_std"].median()),
        "secondary_matches_archive": int(
            (
                table["secondary_correct"].fillna(-1) == table["baseline_correct_open"].fillna(-1)
            ).sum()
        ),
    }
    for arm, game_id in picks.items():
        row[f"{arm}_game_id"] = game_id
        row[f"{arm}_correct"] = float(indexed.loc[game_id, "primary_correct"])
        row[f"{arm}_correct_secondary"] = float(indexed.loc[game_id, "secondary_correct"])
    return row


def paired_cell(weekly: pd.DataFrame, arm: str, base: str, column: str) -> dict[str, Any]:
    """Week-blocked paired delta in accuracy points, arm minus base."""
    frame = weekly.dropna(subset=[f"{arm}_{column}", f"{base}_{column}"]).copy()
    frame["delta"] = (frame[f"{arm}_{column}"] - frame[f"{base}_{column}"]) * 100.0
    result = week_blocked_bootstrap(
        frame,
        lambda rows: {"mean": float(rows["delta"].mean())},
        block="week",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    ).iloc[0]
    differs = frame.loc[frame[f"{arm}_game_id"] != frame[f"{base}_game_id"]]
    return {
        "weeks_paired": len(frame),
        "arm_hits": int(frame[f"{arm}_{column}"].sum()),
        "base_hits": int(frame[f"{base}_{column}"].sum()),
        "arm_accuracy": float(frame[f"{arm}_{column}"].mean()),
        "base_accuracy": float(frame[f"{base}_{column}"].mean()),
        "effect_accuracy_points": float(result["estimate"]),
        "interval_low": float(result["lower"]),
        "interval_high": float(result["upper"]),
        "probability_positive": float(result["probability_positive"]),
        "weeks_nominee_differs": len(differs),
        "weeks_outcome_differs": int((differs["delta"] != 0).sum()),
    }


def run(repo: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    ranker = _load_script(REPO / "scripts/best_pick_opener_ranker_eval.py", "_bp_ranker")
    composed = _load_script(REPO / "scripts/best_pick_composed_rule_eval.py", "_bp_composed")
    work = build_work(repo, ranker, composed)
    cache = repo / "artifacts/experiments/sharp_book_movement"
    quotes = pd.read_parquet(cache / "quotes.parquet")
    kickoffs = pd.read_parquet(cache / "kickoff.parquet").rename(
        columns={"nflverse_game_id": "game_id"}
    )
    kickoffs["game_id"] = kickoffs["game_id"].astype(str)
    covered = set(kickoffs["game_id"])
    work = work.loc[work["game_id"].isin(covered)].reset_index(drop=True)

    rows = [
        evaluate_week(group, quotes, kickoffs)
        for _, group in work.groupby(["season", "week"], sort=True)
    ]
    weekly = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)

    cells: dict[str, Any] = {}
    for arm in ARMS:
        if arm == "a0_tuesday":
            continue
        cells[f"{arm}_vs_a0_primary"] = paired_cell(weekly, arm, "a0_tuesday", "correct")
        cells[f"{arm}_vs_a0_secondary"] = paired_cell(
            weekly, arm, "a0_tuesday", "correct_secondary"
        )

    hit_rates = {
        arm: {
            "weeks_scored": int(weekly[f"{arm}_correct"].notna().sum()),
            "hits": int(weekly[f"{arm}_correct"].sum()),
            "accuracy": float(weekly[f"{arm}_correct"].mean()),
            "weeks_scored_secondary": int(weekly[f"{arm}_correct_secondary"].notna().sum()),
            "hits_secondary": int(weekly[f"{arm}_correct_secondary"].sum()),
            "accuracy_secondary": float(weekly[f"{arm}_correct_secondary"].mean()),
        }
        for arm in ARMS
    }

    summary: dict[str, Any] = {
        "family": FAMILY,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "renomination_hour_et": RENOMINATION_HOUR_ET,
        "follow_threshold": THRESHOLD,
        "population": {
            "games": len(work),
            "weeks": len(weekly),
            "seasons": sorted(int(season) for season in work["season"].unique()),
            "pushes": int(work["baseline_correct_open"].isna().sum()),
            "weeks_a0_already_played_at_instant": int((~weekly["a0_playable"]).sum()),
            "weeks_with_any_flip": int(weekly["n_flips"].gt(0).sum()),
            "total_flips": int(weekly["n_flips"].sum()),
            "games_missing_deadline_std": int(weekly["games_missing_deadline_std"].sum()),
            "secondary_matches_archive_games": int(weekly["secondary_matches_archive"].sum()),
        },
        "hit_rates": hit_rates,
        "cells": cells,
        "registered_challenger_is_a_static_noop": {
            "weeks_a1b_differs_from_a0": 0,
            "note": (
                "best_pick_sunday_renomination reuses the frozen Tuesday pool and only "
                "refreshes probabilities; on a static feature table the refreshed "
                "probability equals the Tuesday probability, so its nominee equals A0 in "
                "every week by construction. Declared before the run, not a measured "
                "finding."
            ),
        },
        "binding_note": (
            "An interval containing zero is never a rejection; probability_positive is the "
            "continuous evidence. Closing grounds are limited to refuted_mechanism (resolved "
            "wrong sign or zero split-half reliability) and bounded_by_control."
        ),
    }
    return summary, weekly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summary, weekly = run(args.repo)
    print(json.dumps(summary, indent=2, default=str))
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.repo / "artifacts" / FAMILY / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    weekly.to_parquet(out_dir / "weekly.parquet")
    weekly.to_csv(out_dir / "weekly.csv", index=False)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
