from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.evidence_conventions import binomial_two_sided_p, probability_positive_from_draws
from nfl_ats.four_overlay_composition import COMPOSITION_ORDER
from nfl_ats.overlay_composition import DEFAULT_FEATURES, DEFAULT_INCIDENTS, build_predictions_frame
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.unserved_tilt_marginals import build_served_card_flip_sets


def find_latest_per_game(repo_root: Path) -> Path:
    candidates = sorted((repo_root / "artifacts" / "opener_evaluation").glob("*/per_game.parquet"))
    if not candidates:
        raise ValueError("no opener_evaluation per_game.parquet found under artifacts")
    return candidates[-1]


def load_split_library(repo_root: Path) -> dict:
    path = repo_root / "registry" / "split_library.json"
    return json.loads(path.read_text(encoding="utf-8"))


def assign_cells(split: str, library: dict, frame: pd.DataFrame) -> pd.Series:
    if split not in library.get("splits", {}):
        raise ValueError("split " + str(split) + " is not declared in the split library")
    entry = library["splits"][split]
    cells = list(entry.get("cells", []))
    if split == "week_in_season":
        expected = ["weeks_1_4", "weeks_5_12", "weeks_13_18"]
        if cells != expected:
            raise ValueError(
                "week_in_season cells changed; expected " + str(expected) + " got " + str(cells)
            )
        weeks = pd.to_numeric(frame["week"], errors="coerce")
        labels = pd.Series("", index=frame.index, dtype=object)
        labels = labels.mask(weeks <= 4, "weeks_1_4")
        labels = labels.mask((weeks >= 5) & (weeks <= 12), "weeks_5_12")
        labels = labels.mask(weeks >= 13, "weeks_13_18")
        unassigned = labels.eq("")
        if bool(unassigned.any()):
            raise ValueError("week_in_season left weeks unassigned")
        return labels
    if split == "spread_band":
        expected = ["short", "long"]
        if cells != expected:
            raise ValueError(
                "spread_band cells changed; expected " + str(expected) + " got " + str(cells)
            )
        if "spread_line" not in frame.columns:
            raise ValueError("spread_band requires spread_line from the served predictions frame")
        spread = pd.to_numeric(frame["spread_line"], errors="coerce")
        absolute = spread.abs()
        labels = pd.Series("", index=frame.index, dtype=object)
        labels = labels.mask(absolute <= 7.0, "short")
        labels = labels.mask(absolute > 7.0, "long")
        unassigned = labels.eq("")
        if bool(unassigned.any()):
            raise ValueError("spread_band left spreads unassigned")
        return labels
    raise ValueError(
        "split "
        + str(split)
        + " is not implemented; only week_in_season and spread_band are implemented"
    )


def week_blocked_stats(delta: np.ndarray, frame: pd.DataFrame, samples: int, seed: int) -> dict:
    grouped = list(frame.groupby(["season", "week"], sort=False, dropna=False).indices.values())
    total_blocks = len(grouped)
    if len(delta) == 0 or total_blocks == 0:
        return {
            "estimate": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "ppos": float("nan"),
            "ppos_strict": float("nan"),
            "se": float("nan"),
            "blocks": int(total_blocks),
            "draws": np.empty(0, dtype=float),
        }
    sums = np.array([delta[list(idx)].sum() for idx in grouped], dtype=float)
    counts = np.array([len(idx) for idx in grouped], dtype=float)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        chosen = generator.integers(0, total_blocks, size=total_blocks)
        draws[i] = sums[chosen].sum() / counts[chosen].sum()
    estimate = float(np.mean(delta) * 100.0)
    lo = float(np.quantile(draws, 0.05) * 100.0)
    hi = float(np.quantile(draws, 0.95) * 100.0)
    ppos = float(probability_positive_from_draws(draws * 100.0))
    ppos_strict = float(np.mean(draws > 0.0))
    se = float(np.std(draws * 100.0, ddof=1))
    return {
        "estimate": estimate,
        "lo": lo,
        "hi": hi,
        "ppos": ppos,
        "ppos_strict": ppos_strict,
        "se": se,
        "blocks": int(total_blocks),
        "draws": draws,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Conditional signal atlas leave-one-out cell scorer"
    )
    parser.add_argument("--signal", type=str, required=True, help="served member id to score")
    parser.add_argument(
        "--split", type=str, required=True, help="split name from the split library"
    )
    parser.add_argument(
        "--season-start", type=int, default=2020, help="first season scored inclusive"
    )
    parser.add_argument("--season-end", type=int, default=2025, help="last season scored inclusive")
    parser.add_argument("--draws", type=int, default=2000, help="bootstrap draws per cell")
    parser.add_argument(
        "--seed", type=int, default=20260821, help="bootstrap seed reused in every cell"
    )
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    library = load_split_library(repo_root)
    library_version = library.get("version")
    if args.signal not in COMPOSITION_ORDER:
        raise ValueError("signal " + str(args.signal) + " is not one of the nine served members")
    per_game_path = find_latest_per_game(repo_root)
    metadata = json.loads(per_game_path.with_name("metadata.json").read_text(encoding="utf-8"))
    per_game = pd.read_parquet(per_game_path)
    per_game = per_game.loc[
        per_game["season"].between(args.season_start, args.season_end)
    ].reset_index(drop=True)
    snapshot = latest_snapshot(data_root / "raw")
    schedules, _team_stats = load_snapshot(snapshot)
    predictions = build_predictions_frame(per_game, schedules)
    members = build_served_card_flip_sets(
        predictions,
        schedules,
        per_game,
        data_root,
        repo_root,
        repo_root / DEFAULT_FEATURES,
        repo_root / DEFAULT_INCIDENTS,
    )
    ordered = [str(name) for name in COMPOSITION_ORDER]
    if [str(name) for name in members] != ordered:
        raise ValueError("served flip-set order does not match the nine-member order")
    signal_ids = {str(v) for v in members[args.signal]}
    other_union: set[str] = set()
    for name, ids in members.items():
        if str(name) != str(args.signal):
            other_union |= {str(v) for v in ids}
    served_union: set[str] = set()
    for ids in members.values():
        served_union |= {str(v) for v in ids}
    unique_ids = {v for v in signal_ids if v not in other_union}
    eval_frame = predictions[["game_id", "season", "week", "spread_line"]].merge(
        per_game[["game_id", "correct_at_open_probability_rule"]], on="game_id", how="left"
    )
    eval_frame["game_id"] = eval_frame["game_id"].astype(str)
    eval_frame["correct_raw"] = pd.to_numeric(
        eval_frame["correct_at_open_probability_rule"], errors="coerce"
    )
    game_ids = eval_frame["game_id"]
    served_mask = game_ids.isin(served_union)
    raw = eval_frame["correct_raw"]
    correct_served = raw.where(~served_mask, 1.0 - raw)
    no_signal_mask = served_mask & ~game_ids.isin(unique_ids)
    correct_no_signal = raw.where(~no_signal_mask, 1.0 - raw)
    valid = raw.notna().to_numpy()
    scored = eval_frame.loc[valid].reset_index(drop=True)
    delta_all = (correct_served - correct_no_signal).loc[valid].to_numpy(dtype=float)
    scored = scored.copy()
    scored["cell"] = assign_cells(args.split, library, scored)
    split_entry = library["splits"][args.split]
    cells_in_order = list(split_entry.get("cells", []))
    rows = []
    for cell in cells_in_order:
        mask = (scored["cell"].to_numpy() == cell).astype(bool)
        cell_delta = delta_all[mask]
        cell_frame = scored.loc[mask, ["season", "week"]].reset_index(drop=True)
        stats = week_blocked_stats(cell_delta, cell_frame, args.draws, args.seed)
        decisive = cell_delta[cell_delta != 0.0]
        wins = int(np.count_nonzero(decisive > 0.0))
        losses = int(np.count_nonzero(decisive < 0.0))
        decisive_n = int(decisive.size)
        exact_p = float(binomial_two_sided_p(wins, decisive_n))
        cell_game_ids = scored.loc[mask, "game_id"].astype(str)
        rows.append(
            {
                "cell": str(cell),
                "n": int(mask.sum()),
                "member_flips_scored": int(cell_game_ids.isin(signal_ids).sum()),
                "unique_flips_scored": int(cell_game_ids.isin(unique_ids).sum()),
                "decisive_wins": wins,
                "decisive_losses": losses,
                "decisive_n": decisive_n,
                "estimate_accuracy_points": float(stats["estimate"]),
                "lower_accuracy_points": float(stats["lo"]),
                "upper_accuracy_points": float(stats["hi"]),
                "probability_positive": float(stats["ppos"]),
                "probability_positive_strict": float(stats["ppos_strict"]),
                "standard_error_accuracy_points": float(stats["se"]),
                "blocks": int(stats["blocks"]),
                "exact_two_sided_p": exact_p,
            }
        )
    scored_ids = set(scored["game_id"].astype(str))
    payload = {
        "signal": str(args.signal),
        "split": str(args.split),
        "split_library_version": library_version,
        "split_library_cells": cells_in_order,
        "split_library_source_columns": split_entry.get("source_columns"),
        "split_library_leakage_note": split_entry.get("leakage_note"),
        "season_start": int(args.season_start),
        "season_end": int(args.season_end),
        "draws": int(args.draws),
        "seed": int(args.seed),
        "interval": "90pct week-blocked",
        "block": "week",
        "per_game_artifact": str(per_game_path.relative_to(repo_root)).replace("\\", "/"),
        "active_model_id": metadata.get("active_model_id"),
        "population": {
            "n_games": len(eval_frame),
            "n_scored": int(valid.sum()),
            "n_pushes": int((~valid).sum()),
            "seasons": sorted(int(s) for s in eval_frame["season"].dropna().unique()),
            "week_blocks_scored": int(scored[["season", "week"]].drop_duplicates().shape[0]),
        },
        "flip_counts": {
            "served_union_archive": len(served_union),
            "served_union_scored": int(game_ids[valid].isin(served_union).sum()),
            "signal_all_archive": len(signal_ids),
            "signal_all_scored": int(game_ids[valid].isin(signal_ids).sum()),
            "signal_unique_archive": len(unique_ids),
            "signal_unique_scored": int(game_ids[valid].isin(unique_ids).sum()),
            "signal_overlap_other8_archive": len(signal_ids & other_union),
            "scored_ids_not_in_archive": len(scored_ids - set(game_ids.astype(str))),
        },
        "members_in_order": ordered,
        "cells": rows,
        "fidelity": {
            "per_game_artifact": str(per_game_path.relative_to(repo_root)).replace("\\", "/"),
            "active_model_id": metadata.get("active_model_id"),
            "split_library_version": library_version,
            "member_order_matches_nine_member_order": True,
            "flip_set_builder": "build_served_card_flip_sets nine-member union",
            "leave_one_out": "served minus signal-unique flips, paired per game",
            "bootstrap": "week-blocked, same block count, tails 0.05 and 0.95",
            "draws_per_cell": int(args.draws),
            "seed_per_cell": int(args.seed),
            "grade": "opener",
        },
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
