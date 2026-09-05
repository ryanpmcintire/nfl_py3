"""Run the predeclared CX14 refresh screens; never fetch, publish, or train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.coach_fade_overlay import apply_coach_fade_overlay
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
from nfl_ats.injury_trajectory_features import (
    LEADS,
    build_flags,
    prepare_revisions,
    split_half_reliability,
)
from nfl_ats.io import atomic_parquet, run_id
from nfl_ats.overlay_composition import (
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    build_predictions_frame,
    load_inputs,
    reconstruct_arrest_flip_set,
)
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.spread_gap_zone_fade_overlay import apply_spread_gap_zone_fade_overlay

OUTPUT = Path("artifacts/experiments/injury_trajectory_leads")
RAW = Path("data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet")
CANONICAL = Path("data/players/raw/20260905T123614Z/injuries.parquet")
SEED = 20260905
SAMPLES = 20000


def summarize(frame: pd.DataFrame, lead: str) -> dict[str, Any]:
    delta = (frame[f"{lead}_correct"] - frame.production_correct) * 100
    blocks = pd.DataFrame({"season": frame.season, "week": frame.week, "delta": delta})
    groups = blocks.groupby(["season", "week"]).delta.agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    draws = np.empty(SAMPLES)
    for start in range(0, SAMPLES, 500):
        indices = rng.integers(0, len(groups), size=(min(500, SAMPLES - start), len(groups)))
        draws[start : start + len(indices)] = groups["sum"].to_numpy()[indices].sum(
            axis=1
        ) / groups["count"].to_numpy()[indices].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "games": len(frame),
        "blocks": len(groups),
        "flips": int(frame[f"{lead}_flip"].sum()),
        "production_accuracy": float(frame.production_correct.mean() * 100),
        "candidate_accuracy": float(frame[f"{lead}_correct"].mean() * 100),
        "effect": float(delta.mean()),
        "interval_low": float(low),
        "interval_high": float(high),
        "probability_positive": float((draws > 0).mean()),
        "probability_tie": float((draws == 0).mean()),
        "standard_error": float(draws.std(ddof=1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    raw = pd.read_parquet(RAW)
    canonical = pd.read_parquet(CANONICAL)
    keys = ["season", "game_type", "week", "team", "gsis_id", "date_modified"]
    basis = canonical[[*keys, "observed_at_basis"]].drop_duplicates()
    raw = raw.merge(basis, on=keys, how="left", validate="many_to_one")
    revisions, coverage = prepare_revisions(raw)
    # Source diagnosis precedes any arm computation, including when running normally.
    print(json.dumps({"coverage": coverage}, indent=2), flush=True)
    if args.coverage_only:
        return 0
    match = find_matching_opener_evaluation(Path("artifacts"))
    if match is None:
        raise ValueError("No opener evaluation matches the active model")
    per_game_path = match[1] / "per_game.parquet"
    per_game, schedules, _, snapshot, _ = load_inputs(per_game_path, Path("data"))
    predictions = build_predictions_frame(per_game, schedules)
    results = [
        apply_coach_fade_overlay(predictions, schedules),
        apply_division_revenge_tilt_overlay(predictions, schedules),
        apply_spread_gap_zone_fade_overlay(predictions),
    ]
    union = {flip.game_id for result in results for flip in result.flips}
    arrest_ids, _ = reconstruct_arrest_flip_set(per_game, DEFAULT_FEATURES, DEFAULT_INCIDENTS)
    union |= arrest_ids
    frame = per_game.merge(
        schedules[["game_id", "home_team", "away_team", "gameday", "gametime"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    local = pd.to_datetime(frame.gameday.astype(str).str[:10] + " " + frame.gametime.astype(str))
    frame["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    frame = frame.loc[
        frame.season.between(2022, 2025) & frame.correct_at_open_probability_rule.notna()
    ].copy()
    frame = frame.sort_values(["kickoff", "game_id"]).reset_index(drop=True)
    frame["production_pick_home"] = frame.pick_home_at_open_probability_rule.astype(
        bool
    ) ^ frame.game_id.isin(union)
    frame["production_correct"] = frame.production_pick_home.eq(frame.margin_vs_open.gt(0)).astype(
        float
    )
    flags = build_flags(frame, revisions)
    summaries: dict[str, Any] = {}
    for lead in LEADS:
        lookup = flags.set_index(["game_id", "team"])[lead]
        home = pd.Series(
            [bool(lookup.loc[(g.game_id, g.home_team)]) for g in frame.itertuples()],
            index=frame.index,
        )
        away = pd.Series(
            [bool(lookup.loc[(g.game_id, g.away_team)]) for g in frame.itertuples()],
            index=frame.index,
        )
        frame[f"{lead}_home_flag"] = home
        frame[f"{lead}_away_flag"] = away
        frame[f"{lead}_flip"] = home.ne(away) & frame.production_pick_home.eq(home)
        pick = frame.production_pick_home ^ frame[f"{lead}_flip"]
        frame[f"{lead}_pick_home"] = pick
        frame[f"{lead}_correct"] = pick.eq(frame.margin_vs_open.gt(0)).astype(float)
        summaries[lead] = {
            **summarize(frame, lead),
            "by_season": {
                str(season): summarize(group, lead) for season, group in frame.groupby("season")
            },
            "reliability": split_half_reliability(flags, lead),
            "covered_team_games": int(flags[f"{lead}_covered"].sum()),
            "flagged_team_games": int(flags[lead].sum()),
            "coverage_by_season": flags.groupby("season")[[f"{lead}_covered", lead]]
            .sum()
            .to_dict(),
            "classification": "unresolved_below_power",
            "positive_control": {
                "status": "unavailable",
                "reason": (
                    "No actual historical Sunday inactive list in the local capture source; "
                    "no zero-snap substitution."
                ),
            },
        }
    destination = OUTPUT / run_id()
    metadata = {
        "source_per_game": str(per_game_path),
        "source_per_game_sha256": sha256_file(per_game_path),
        "active_model_id": match[0]["active_model_id"],
        "schedule_snapshot": snapshot,
        "raw_injuries": str(RAW),
        "raw_sha256": sha256_file(RAW),
        "canonical_injuries": str(CANONICAL),
        "canonical_sha256": sha256_file(CANONICAL),
        "coverage": coverage,
        "results": summaries,
        "predeclaration": "docs/injury_trajectory_leads.md",
        "seed": SEED,
        "bootstrap_samples": SAMPLES,
        "provenance": artifact_provenance(
            {"command": "cx14-injury-trajectory-screen", "seed": SEED, "samples": SAMPLES},
            per_game_path,
        ),
    }
    # Keep the experiment helper's registry sidecar within this lane's allowed output tree.
    write_experiment_artifact(
        destination,
        "results.json",
        metadata,
        command="cx14-injury-trajectory-screen",
        metrics={lead: result["effect"] for lead, result in summaries.items()},
        registry_root=OUTPUT / "registry",
    )
    atomic_parquet(frame, destination / "paired_predictions.parquet")
    atomic_parquet(flags, destination / "team_flags.parquet")
    print(json.dumps({"output": str(destination), "results": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
