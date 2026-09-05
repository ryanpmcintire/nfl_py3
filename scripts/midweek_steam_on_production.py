"""CX16 LEAD-04 coverage and frozen refresh-overlay screen; local inputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import HISTORICAL_CAPTURE_KIND, load_snapshot_manifest_index, pick_correct
from nfl_ats.midweek_steam_features import attach_midweek_steam, spread_move_events
from nfl_ats.provenance import (
    configuration_hash,
    git_state,
    sha256_file,
    stamp_sidecar,
    write_experiment_artifact,
    write_stamped_artifact,
)
from nfl_ats.transaction_flag_features import default_schedule

ROOT = Path("artifacts/experiments/midweek_steam")
SEASONS = (2023, 2024, 2025)
SAMPLES = 20_000
SEED = 2026090516


def summarize(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    from nfl_ats.overlay_composition import blocked_bootstrap_matrix

    valid = frame.loc[frame[column].notna()].reset_index(drop=True)
    if valid.empty:
        return {"games": 0, "estimate": None, "probability_positive": None}
    stats = blocked_bootstrap_matrix(
        valid[[column]].to_numpy(dtype=float),
        valid[["season", "week"]],
        block="week",
        samples=SAMPLES,
        seed=SEED,
    )
    return {
        "games": len(valid),
        "blocks": int(stats["block_count"]),
        **{
            key: float(stats[key][0])
            for key in ("estimate", "lower", "upper", "standard_error", "probability_positive")
        },
    }


def score(output: Path) -> dict[str, Any]:
    from nfl_ats.coach_fade_overlay import apply_coach_fade_overlay
    from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
    from nfl_ats.four_overlay_composition import POLICY_FINGERPRINT, POLICY_ID
    from nfl_ats.overlay_composition import (
        DEFAULT_FEATURES,
        DEFAULT_INCIDENTS,
        build_predictions_frame,
        reconstruct_arrest_flip_set,
    )
    from nfl_ats.public_board import find_matching_opener_evaluation
    from nfl_ats.spread_gap_zone_fade_overlay import apply_spread_gap_zone_fade_overlay

    declaration = Path("docs/midweek_steam_lead.md")
    if not declaration.is_file():
        raise ValueError("Write the predeclaration before scoring")
    frozen_design = (
        declaration.read_text(encoding="utf-8").split("## Measured results", 1)[0].rstrip() + "\n"
    )
    match = find_matching_opener_evaluation(Path("artifacts"))
    if match is None:
        raise ValueError("No opener evaluation matches the active model")
    metadata, directory = match
    path = directory / "per_game.parquet"
    per_game = pd.read_parquet(path)
    per_game = per_game.loc[per_game["season"].isin(SEASONS)].copy()
    schedules = default_schedule()
    predictions = build_predictions_frame(per_game, schedules)
    overlays = [
        apply_coach_fade_overlay(predictions, schedules),
        apply_division_revenge_tilt_overlay(predictions, schedules),
        apply_spread_gap_zone_fade_overlay(predictions),
    ]
    arrest_ids, _ = reconstruct_arrest_flip_set(per_game, DEFAULT_FEATURES, DEFAULT_INCIDENTS)
    union = arrest_ids | {flip.game_id for result in overlays for flip in result.flips}
    games = pd.read_parquet(output / "games.parquet")
    events = pd.read_parquet(output / "events.parquet")
    # Recompute cutoff/flags from events so DST fixes cannot leave cached flags stale.
    games = attach_midweek_steam(games, events)
    exposure_ids = set(events.loc[events["before_cutoff"], "game_id"])
    games["any_steam_exposure"] = games["game_id"].isin(exposure_ids).astype(float)
    team_games = pd.concat([games.assign(team=games[side]) for side in ("home_team", "away_team")])
    team_games["parity"] = team_games["week"] % 2
    halves = (
        team_games.groupby(["season", "team", "parity"])["any_steam_exposure"]
        .mean()
        .unstack("parity")
        .dropna()
    )

    def reliability(frame: pd.DataFrame) -> float | None:
        if len(frame) < 2 or frame[0].nunique() < 2 or frame[1].nunique() < 2:
            return None
        return float(frame[0].corr(frame[1]))

    paired = (
        per_game.merge(
            games[["game_id", "kickoff", "decision_cutoff_utc", "midweek_steam_side"]],
            on="game_id",
            validate="one_to_one",
        )
        .sort_values(["season", "week", "kickoff", "game_id"])
        .reset_index(drop=True)
    )
    raw = paired["home_cover_probability_at_open"].ge(0.5)
    paired["production_pick_home"] = raw ^ paired["game_id"].isin(union)
    flag = paired["midweek_steam_side"].ne("")
    paired["candidate_pick_home"] = paired["production_pick_home"].where(
        ~flag, paired["midweek_steam_side"].eq("HOME")
    )
    margin = paired["margin_vs_open"]
    paired["production_correct"] = pick_correct(paired["production_pick_home"], margin).where(
        margin.notna()
    )
    paired["candidate_correct"] = pick_correct(paired["candidate_pick_home"], margin).where(
        margin.notna()
    )
    paired["delta_accuracy_points"] = 100 * (
        paired["candidate_correct"] - paired["production_correct"]
    )
    move = paired["close_home_spread"] - paired["tue_open_home_spread"]
    oracle = paired["production_pick_home"].where(move.eq(0) | move.isna(), move.gt(0))
    paired["oracle_pick_home"] = oracle
    paired["oracle_correct"] = pick_correct(oracle, margin).where(margin.notna())
    paired["oracle_delta_accuracy_points"] = 100 * (
        paired["oracle_correct"] - paired["production_correct"]
    )
    paired["steam_correct"] = pick_correct(paired["midweek_steam_side"].eq("HOME"), margin).where(
        flag & margin.notna()
    )
    paired["steam_vs_half_points"] = 100 * (paired["steam_correct"] - 0.5)
    paired["favorite_correct"] = pick_correct(paired["tue_open_home_spread"].ge(0), margin).where(
        margin.notna()
    )
    valid = paired["production_correct"].notna()
    paired["flipped"] = paired["production_pick_home"].ne(paired["candidate_pick_home"])
    summary = summarize(paired, "delta_accuracy_points")
    payload = {
        "active_model_id": metadata["active_model_id"],
        "feature_table_sha256": metadata["feature_table_sha256"],
        "policy_id": POLICY_ID,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "opener_artifact": str(path),
        "opener_sha256": sha256_file(path),
        "coverage": json.loads((output / "coverage.json").read_text()),
        "paired_games_including_pushes": len(paired),
        "unpaired_hourly_games": len(games) - len(paired),
        "push_or_unsettled_games": int((~valid).sum()),
        "flips_nonpush": int(paired.loc[valid, "flipped"].sum()),
        "production_correct": int(paired["production_correct"].sum()),
        "candidate_correct": int(paired["candidate_correct"].sum()),
        "production_accuracy": float(paired["production_correct"].mean()),
        "candidate_accuracy": float(paired["candidate_correct"].mean()),
        "raw_probability_accuracy": float(paired["correct_at_open_probability_rule"].mean()),
        "opener_favorite_accuracy": float(paired["favorite_correct"].mean()),
        "effect_accuracy_points": summary,
        "oracle_accuracy": float(paired["oracle_correct"].mean()),
        "oracle_effect_accuracy_points": summarize(paired, "oracle_delta_accuracy_points"),
        "steam_side_accuracy": float(paired["steam_correct"].mean()),
        "steam_side_vs_half_accuracy_points": summarize(paired, "steam_vs_half_points"),
        "split_half_reliability": reliability(halves),
        "team_seasons": len(halves),
        "reliability_by_season": {
            str(s): reliability(f) for s, f in halves.groupby(level="season")
        },
        "per_season": [
            {
                "season": int(s),
                "production_accuracy": float(f["production_correct"].mean()),
                "candidate_accuracy": float(f["candidate_correct"].mean()),
                "flips_nonpush": int(f.loc[f["production_correct"].notna(), "flipped"].sum()),
                **summarize(f, "delta_accuracy_points"),
            }
            for s, f in paired.groupby("season")
        ],
        "bootstrap_samples": SAMPLES,
        "seed": SEED,
        "provenance": {
            "configuration": {
                "predeclaration_sha256": hashlib.sha256(frozen_design.encode()).hexdigest(),
                "script_sha256": sha256_file(Path(__file__)),
                "feature_module_sha256": sha256_file(Path("src/nfl_ats/midweek_steam_features.py")),
                "events_sha256": sha256_file(output / "events.parquet"),
                "seed": SEED,
                "bootstrap_samples": SAMPLES,
            },
            "code": git_state(Path.cwd()),
        },
    }
    payload["provenance"]["configuration_sha256"] = configuration_hash(
        payload["provenance"]["configuration"]
    )
    for name, frame in (("paired", paired), ("reliability", halves.reset_index())):
        table = output / f"{name}.parquet"
        frame.to_parquet(table, index=False)
        stamp_sidecar(table)
    # Keep experiment provenance within this lane's writable artifact tree;
    # the shared weak-signal registry is written separately via its record CLI.
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="midweek-steam-on-production",
        metrics=summary,
        registry_root=output / "experiment_provenance",
    )
    return payload


def coverage(output: Path) -> dict[str, Any]:
    index = load_snapshot_manifest_index(Path("data/market/raw"))
    index = index.loc[
        index["capture_kind"].eq(HISTORICAL_CAPTURE_KIND)
        & index["decision_label"].eq("intraday_hourly")
        & index["season"].isin(SEASONS)
    ].copy()
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "observed_at_utc",
        "market",
        "home_spread_line",
        "bookmaker_last_update_utc",
        "commence_time_utc",
    ]
    frames = []
    missing = []
    for row in index.itertuples(index=False):
        path = Path(row.dir) / "quotes.parquet"
        if not path.exists():
            missing.append(str(path))
            continue
        q = pd.read_parquet(path, columns=columns, filters=[("market", "==", "spreads")])
        q = q.drop_duplicates(
            [
                "nflverse_game_id",
                "bookmaker_key",
                "observed_at_utc",
                "home_spread_line",
                "bookmaker_last_update_utc",
            ]
        )
        frames.append(q)
    quotes = pd.concat(frames, ignore_index=True)
    quotes["observed_at_utc"] = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    schedule = default_schedule()
    games = schedule.loc[
        schedule["season"].isin(SEASONS) & schedule["game_type"].eq("REG"),
        ["game_id", "season", "week", "home_team", "away_team"],
    ].copy()
    kickoff = quotes.groupby("nflverse_game_id")["commence_time_utc"].min().rename("kickoff")
    games = games.merge(kickoff, left_on="game_id", right_index=True, validate="one_to_one")
    quotes = quotes.loc[quotes["nflverse_game_id"].isin(games["game_id"])].copy()
    events = spread_move_events(quotes)
    games = attach_midweek_steam(games, events)
    events = events.merge(
        games[["game_id", "season", "decision_cutoff_utc", "midweek_start_utc"]], on="game_id"
    )
    outside_week_events = len(events.loc[events["observed_at_utc"].lt(events["midweek_start_utc"])])
    events = events.loc[events["observed_at_utc"].ge(events["midweek_start_utc"])].copy()
    events["before_cutoff"] = events["observed_at_utc"].lt(events["decision_cutoff_utc"])
    quotes["weekday"] = quotes["observed_at_utc"].dt.tz_convert("America/New_York").dt.weekday
    quotes["season"] = quotes["nflverse_game_id"].str[:4].astype(int)
    quotes = quotes.merge(
        games[["game_id", "midweek_start_utc"]], left_on="nflverse_game_id", right_on="game_id"
    )
    index["weekday"] = index["snapshot_timestamp_utc"].dt.tz_convert("America/New_York").dt.weekday
    summary = []
    for season in SEASONS:
        q = quotes.loc[quotes["season"].eq(season)]
        mid = q.loc[q["weekday"].between(2, 5) & q["observed_at_utc"].ge(q["midweek_start_utc"])]
        ev = events.loc[events["season"].eq(season)]
        summary.append(
            {
                "season": season,
                "games": int(q["nflverse_game_id"].nunique()),
                "quote_rows_deduplicated": len(q),
                "midweek_games": int(mid["nflverse_game_id"].nunique()),
                "midweek_game_snapshots": len(
                    mid[["nflverse_game_id", "observed_at_utc"]].drop_duplicates()
                ),
                "manifests_by_weekday": {
                    str(k): int(v)
                    for k, v in index.loc[index["season"].eq(season)]
                    .groupby("weekday")
                    .size()
                    .items()
                },
                "games_by_midweek_day": {
                    str(k): int(v)
                    for k, v in mid.groupby("weekday")["nflverse_game_id"].nunique().items()
                },
                "steam_events": len(ev),
                "steam_events_before_cutoff": int(ev["before_cutoff"].sum()),
                "steam_games_before_cutoff": int(ev.loc[ev["before_cutoff"], "game_id"].nunique()),
                "flagged_games": int(
                    games.loc[games["season"].eq(season), "midweek_steam_exposure"].sum()
                ),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("events", events), ("games", games)):
        path = output / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        stamp_sidecar(path)
    result = {
        "seasons": summary,
        "missing_quote_files": missing,
        "manifest_count": len(index),
        "outside_game_week_events": outside_week_events,
        "definition": "docs/midweek_steam_lead.md",
    }
    write_stamped_artifact(result, output / "coverage.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("coverage", "score"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            coverage(args.output) if args.mode == "coverage" else score(args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
