"""MKT-15: leader-only / leader-weighted / leader-first vs the served equal-book follow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_REPO = Path("F:/Repos/nfl_py3")
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import pick_correct  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.io import atomic_csv, atomic_parquet, run_id  # noqa: E402
from nfl_ats.provenance import sha256_file, write_stamped_artifact  # noqa: E402
from nfl_ats.sharp_book_movement_features import (  # noqa: E402
    LEADER_BOOKS,
    LEADERSHIP_WEIGHTS,
    THRESHOLD,
    sharp_book_movement_features,
)

SOURCE = DATA_REPO / "artifacts/experiments/sharp_book_movement"
OUTPUT = DATA_REPO / "artifacts/sharp_weighted_follow"
SAMPLES = 20_000
SEED = 20260821
LEGACY_SEED = 2026090518
BASELINE_PICK = "pick_home_at_open_probability_rule"
BASELINE_CORRECT = "correct_at_open_probability_rule"
ARMS = ("s1_leader_only", "s2_leader_weighted", "s3_leader_first", "s4_served_equal")


def per_book_net_moves(quotes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per-game/per-book Wednesday-to-deadline net move, mirroring the served window logic."""
    result = games.copy()
    kickoff = pd.to_datetime(result.commence_time_utc, utc=True)
    anchor = pd.to_datetime(result.week_first_commence_utc, utc=True).dt.tz_convert(
        "America/New_York"
    )
    local = anchor.dt.tz_localize(None).dt.normalize()
    sunday = local + pd.to_timedelta((6 - anchor.dt.weekday) % 7, unit="D")
    deadline = (
        (sunday + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    cutoffs = [kickoff, deadline]
    if "cutoff_utc" in result:
        cutoffs.append(pd.to_datetime(result.cutoff_utc, utc=True))
    result["cutoff_utc"] = pd.concat(cutoffs, axis=1).min(axis=1)
    result["_monday"] = (
        (sunday - pd.Timedelta(days=6)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    result["_wednesday"] = (
        (sunday - pd.Timedelta(days=4)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    result["_sunday"] = sunday.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    needed = {
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "observed_at_utc",
        "bookmaker_last_update_utc",
    }
    q = quotes.loc[
        quotes.market.eq("spreads") & quotes.bookmaker_key.isin(LEADERSHIP_WEIGHTS),
        sorted(needed),
    ].rename(columns={"nflverse_game_id": "game_id"})
    q = q.merge(result[["game_id", "cutoff_utc", "_monday", "_wednesday", "_sunday"]], on="game_id")
    for column in ("observed_at_utc", "bookmaker_last_update_utc"):
        q[column] = pd.to_datetime(q[column], utc=True, errors="coerce")
    q["home_spread_line"] = pd.to_numeric(q.home_spread_line, errors="coerce")
    q = q.loc[
        q.observed_at_utc.lt(q.cutoff_utc)
        & q.observed_at_utc.ge(q._monday)
        & q.observed_at_utc.lt(q._sunday)
        & q.bookmaker_last_update_utc.le(q.observed_at_utc)
        & np.isfinite(q.home_spread_line)
    ].copy()
    keys = ["game_id", "bookmaker_key", "observed_at_utc"]
    q = q.sort_values(keys).drop_duplicates(keys)
    q["move"] = q.groupby(["game_id", "bookmaker_key"]).home_spread_line.diff()
    q = q.loc[q.observed_at_utc.ge(q._wednesday) & q.move.notna()].copy()
    return q.groupby(["game_id", "bookmaker_key"], as_index=False).move.sum()


def arm_aggregates(books: pd.DataFrame, game_ids: pd.Series) -> pd.DataFrame:
    """Equal, leader-median and 2:1 leader-weighted aggregates, one row per game."""
    frame = pd.DataFrame({"game_id": game_ids.to_numpy()})
    equal = books.groupby("game_id").move.agg(["mean", "size"])
    leaders = books.loc[books.bookmaker_key.isin(LEADER_BOOKS)]
    lead = leaders.groupby("game_id").move.agg(["median", "size"])
    weighted = books.assign(
        weight=np.where(books.bookmaker_key.isin(LEADER_BOOKS), 2.0, 1.0)
    ).assign(term=lambda d: d.move * d.weight)
    weighted_sum = weighted.groupby("game_id").agg(term=("term", "sum"), weight=("weight", "sum"))
    frame["equal_net"] = frame.game_id.map(equal["mean"])
    frame["equal_books"] = frame.game_id.map(equal["size"]).fillna(0).astype(int)
    frame["leader_median_net"] = frame.game_id.map(lead["median"])
    frame["leader_books"] = frame.game_id.map(lead["size"]).fillna(0).astype(int)
    frame["leader_weighted_net"] = frame.game_id.map(weighted_sum.term / weighted_sum.weight)
    return frame


def flag_reliability(frame: pd.DataFrame, flag: str) -> dict[str, Any]:
    """Odd/even-week split-half reliability of a team's exposure to the firing flag."""
    stacked = pd.concat(
        [
            frame[["season", "week", side, flag]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ]
    )
    stacked["parity"] = stacked.week % 2
    halves = stacked.groupby(["season", "team", "parity"])[flag].mean().unstack("parity").dropna()
    value = halves[0].corr(halves[1]) if len(halves) > 1 else float("nan")
    return {
        "team_seasons": len(halves),
        "correlation": float(value) if pd.notna(value) else None,
    }


def follow_side(net: pd.Series, baseline_home: pd.Series) -> pd.Series:
    """Follow a qualifying move; keep the Tuesday side otherwise."""
    fires = net.abs().ge(THRESHOLD).fillna(False)
    return baseline_home.astype(bool).mask(fires, net.gt(0))


def summarize(
    frame: pd.DataFrame, candidate: str, baseline: str, *, block: str, seed: int
) -> dict[str, Any]:
    """Paired block bootstrap on game-weighted accuracy points."""
    paired = frame.dropna(subset=[candidate, baseline]).copy()
    paired["delta"] = (paired[candidate] - paired[baseline]) * 100.0
    keys = ["season", "week"] if block == "week" else ["season"]
    blocks = paired.groupby(keys, sort=False).delta.agg(["sum", "count"])
    selected = np.random.default_rng(seed).integers(0, len(blocks), (SAMPLES, len(blocks)))
    draws = blocks["sum"].to_numpy()[selected].sum(axis=1) / blocks["count"].to_numpy()[
        selected
    ].sum(axis=1)
    return {
        "block": block,
        "seed": seed,
        "games": len(paired),
        "blocks": len(blocks),
        "candidate_accuracy": float(paired[candidate].mean()),
        "baseline_accuracy": float(paired[baseline].mean()),
        "effect": float(paired.delta.mean()),
        "interval_low": float(np.quantile(draws, 0.025)),
        "interval_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(probability_positive_from_draws(draws)),
        "season_effects": {
            str(s): float(v) for s, v in paired.groupby("season").delta.mean().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-game", type=Path, default=SOURCE / "20260905T205038Z/per_game.parquet"
    )
    parser.add_argument("--quotes", type=Path, default=SOURCE / "quotes.parquet")
    parser.add_argument("--kickoff", type=Path, default=SOURCE / "kickoff.parquet")
    args = parser.parse_args()

    output = OUTPUT / run_id()
    per_game = pd.read_parquet(args.per_game)
    quotes = pd.read_parquet(args.quotes)
    kickoff = pd.read_parquet(args.kickoff)
    teams = pd.read_parquet(
        SOURCE / "20260905T205038Z/per_game.parquet", columns=["game_id", "home_team", "away_team"]
    )
    games = kickoff.rename(columns={"nflverse_game_id": "game_id"})
    per_game = per_game.loc[per_game.game_id.isin(games.game_id)].reset_index(drop=True)
    if len(per_game) != len(games):
        raise ValueError("Baseline card does not cover the archive population one-to-one")

    exposure = sharp_book_movement_features(quotes, games.copy())
    books = per_book_net_moves(quotes, games.copy())
    aggregates = arm_aggregates(books, exposure.game_id)

    check = exposure[["game_id", "equal_net_move", "eligible_books"]].merge(
        aggregates[["game_id", "equal_net", "equal_books"]], on="game_id", validate="one_to_one"
    )
    parity = {
        "equal_net_max_abs_diff": float(
            (check.equal_net.fillna(0.0) - check.equal_net_move).abs().max()
        ),
        "eligible_books_mismatches": int(
            (check.equal_books != check.eligible_books.astype(int)).sum()
        ),
        "archive_equal_net_max_abs_diff": (
            float(
                (
                    exposure.set_index("game_id").equal_net_move
                    - per_game.set_index("game_id").equal_net_move
                )
                .abs()
                .max()
            )
            if "equal_net_move" in per_game.columns
            else None
        ),
    }
    if parity["equal_net_max_abs_diff"] > 1e-9 or parity["eligible_books_mismatches"]:
        raise ValueError(
            f"Per-book reconstruction does not reproduce the served aggregate: {parity}"
        )

    frame = per_game.merge(aggregates, on="game_id", validate="one_to_one")
    frame = frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    baseline_home = frame[BASELINE_PICK].astype(bool)

    frame["s4_served_equal_pick"] = follow_side(frame.equal_net, baseline_home)
    frame["s1_leader_only_pick"] = follow_side(frame.leader_median_net, baseline_home)
    frame["s2_leader_weighted_pick"] = follow_side(frame.leader_weighted_net, baseline_home)
    leader_fires = frame.leader_median_net.abs().ge(THRESHOLD).fillna(False)
    frame["s3_leader_first_pick"] = frame.s4_served_equal_pick.mask(
        leader_fires, frame.leader_median_net.gt(0)
    )
    reachable = frame.equal_books.gt(0)
    frame["pc_oracle_pick"] = baseline_home.mask(reachable, frame.margin_vs_open.gt(0))

    for arm in (*ARMS, "pc_oracle"):
        frame[f"{arm}_correct"] = pick_correct(frame[f"{arm}_pick"], frame.margin_vs_open)

    frame = frame.drop(columns=["home_team", "away_team"], errors="ignore").merge(
        teams, on="game_id", validate="one_to_one"
    )
    fire_flags = {
        "s4_served_equal": frame.equal_net.abs().ge(THRESHOLD).fillna(False),
        "s1_leader_only": frame.leader_median_net.abs().ge(THRESHOLD).fillna(False),
        "s2_leader_weighted": frame.leader_weighted_net.abs().ge(THRESHOLD).fillna(False),
        "s3_leader_first": leader_fires | frame.equal_net.abs().ge(THRESHOLD).fillna(False),
        "pc_oracle": reachable,
    }
    for arm, flag in fire_flags.items():
        frame[f"{arm}_fired"] = flag

    cells: dict[str, Any] = {}
    for arm in (*ARMS, "pc_oracle"):
        switches = int(frame[f"{arm}_pick"].ne(baseline_home).sum())
        differ = int(frame[f"{arm}_pick"].ne(frame.s4_served_equal_pick).sum())
        cell: dict[str, Any] = {
            "switches_vs_tuesday": switches,
            "sides_differing_from_s4": differ,
            "fires": int(fire_flags[arm].sum()),
            "fire_flag_reliability": flag_reliability(frame, f"{arm}_fired"),
        }
        for block in ("week", "season"):
            cell[block] = summarize(
                frame, f"{arm}_correct", BASELINE_CORRECT, block=block, seed=SEED
            )
        if arm != "s4_served_equal":
            cell["vs_s4"] = {
                block: summarize(
                    frame, f"{arm}_correct", "s4_served_equal_correct", block=block, seed=SEED
                )
                for block in ("week", "season")
            }
        cells[arm] = cell
    cells["s4_served_equal"]["legacy_replay"] = summarize(
        frame, "s4_served_equal_correct", BASELINE_CORRECT, block="week", seed=LEGACY_SEED
    )

    gate: dict[str, Any] = {}
    grid = np.round(np.arange(0.05, 6.01, 0.05), 2)
    targets = {
        "s1_gate_matched": ("leader_median_net", int(cells["s4_served_equal"]["fires"])),
        "s4_gate_matched": ("equal_net", int(cells["s1_leader_only"]["fires"])),
    }
    for name, (column, target) in targets.items():
        counts = np.array([int(frame[column].abs().ge(t).fillna(False).sum()) for t in grid])
        chosen = float(grid[int(np.argmin(np.abs(counts - target)))])
        fires = frame[column].abs().ge(chosen).fillna(False)
        frame[f"{name}_pick"] = baseline_home.mask(fires, frame[column].gt(0))
        frame[f"{name}_correct"] = pick_correct(frame[f"{name}_pick"], frame.margin_vs_open)
        peer = "s1_leader_only" if name == "s4_gate_matched" else "s4_served_equal"
        gate[name] = {
            "threshold": chosen,
            "target_fires": target,
            "fires": int(fires.sum()),
            "switches_vs_tuesday": int(frame[f"{name}_pick"].ne(baseline_home).sum()),
            "week": summarize(frame, f"{name}_correct", BASELINE_CORRECT, block="week", seed=SEED),
            "season": summarize(
                frame, f"{name}_correct", BASELINE_CORRECT, block="season", seed=SEED
            ),
            "vs_peer": {
                "peer": peer,
                "week": summarize(
                    frame, f"{name}_correct", f"{peer}_correct", block="week", seed=SEED
                ),
                "season": summarize(
                    frame, f"{name}_correct", f"{peer}_correct", block="season", seed=SEED
                ),
            },
        }

    atomic_parquet(frame, output / "per_game.parquet")
    atomic_csv(
        pd.json_normalize(
            [
                {
                    "arm": name,
                    "switches": cell["switches_vs_tuesday"],
                    "differ_from_s4": cell["sides_differing_from_s4"],
                    "fires": cell["fires"],
                    "effect": cell["week"]["effect"],
                    "week_low": cell["week"]["interval_low"],
                    "week_high": cell["week"]["interval_high"],
                    "week_pplus": cell["week"]["probability_positive"],
                    "season_low": cell["season"]["interval_low"],
                    "season_high": cell["season"]["interval_high"],
                    "season_pplus": cell["season"]["probability_positive"],
                }
                for name, cell in cells.items()
            ]
        ),
        output / "cells.csv",
    )
    metadata = {
        "cells": cells,
        "gate_matched_diagnostic": gate,
        "parity": parity,
        "population": len(frame),
        "pushes": int(frame[BASELINE_CORRECT].isna().sum()),
        "configuration": {
            "threshold": THRESHOLD,
            "leader_books": list(LEADER_BOOKS),
            "bootstrap_samples": SAMPLES,
            "seed": SEED,
            "legacy_seed": LEGACY_SEED,
            "baseline": BASELINE_CORRECT,
            "predeclaration_sha256": sha256_file(REPO / "docs/sharp_weighted_follow.md"),
            "script_sha256": sha256_file(Path(__file__)),
            "feature_module_sha256": sha256_file(
                REPO / "src/nfl_ats/sharp_book_movement_features.py"
            ),
            "quotes_cache_sha256": sha256_file(args.quotes),
            "kickoff_cache_sha256": sha256_file(args.kickoff),
            "per_game_sha256": sha256_file(args.per_game),
        },
    }
    write_stamped_artifact(json.loads(json.dumps(metadata, default=str)), output / "metadata.json")
    print(json.dumps(cells, indent=2), flush=True)
    print(json.dumps(parity, indent=2), flush=True)
    print(f"artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
