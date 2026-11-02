from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nfl_ats.cli_common import _injury_first_seen_index
from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.players import (
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    enrich_with_player_features,
    load_player_snapshot,
    load_player_value_snapshot,
)
from nfl_ats.players import player_snapshot_from_root
from nfl_ats.players import player_value_snapshot_from_root
from nfl_ats.quarterbacks import depth_snapshot_from_root, load_depth_snapshot

OUTPUT = Path("tests/scratch/codex/lead89_stage_timed")
CLOCKS = Path("tests/scratch/codex/lead89_unit2/input_clock_audit.parquet")
MANIFEST = Path("data/processed/game_features_weak_stack.manifest.json")
RATES = Path("data/processed/weak_stack_availability_rates.parquet")
ACTIVE = Path("data/processed/game_features_weak_stack.parquet")
BASE = Path("data/processed/game_features_pbp.parquet")
COLUMNS = (
    "diff_injury_offense_unavailability",
    "diff_injury_defense_unavailability",
    "diff_injury_special_teams_unavailability",
    "diff_injury_offensive_line_unavailability",
    "diff_injury_skill_unavailability",
    "diff_injury_front_unavailability",
    "diff_injury_secondary_unavailability",
    "diff_injury_skill_epa_value_lost",
    "diff_injury_defense_disruption_value_lost",
    "diff_qb_expected_epa_per_dropback",
    "diff_qb_start_probability",
)
SENTINEL = "__known_no_absence__"


def carry_rows(injuries, played, cutoff_by_team_week):
    keyed = {k: g for k, g in injuries.groupby(["season", "week", "team"], observed=True)}
    weeks = injuries.groupby(["season", "team"], observed=True).week.apply(lambda s: sorted(set(s)))
    out, counts = [], []
    for (season, week, team), cutoff in cutoff_by_team_week.items():
        current = keyed.get((season, week, team))
        if current is not None and current.effective_observed_at.le(cutoff).any():
            counts.append((season, week, team, "current_week_report"))
            continue
        earlier = [w for w in weeks.get((season, team), []) if w < week]
        if not earlier:
            counts.append((season, week, team, "nothing_known"))
            continue
        prior = keyed[(season, earlier[-1], team)]
        prior = prior.loc[prior.effective_observed_at.le(cutoff)]
        if prior.empty:
            counts.append((season, week, team, "nothing_known"))
            continue
        prior = prior.sort_values("effective_observed_at").drop_duplicates("gsis_id", keep="last")
        absent = ~prior.gsis_id.map(lambda g: (season, earlier[-1], team, g) in played)
        carried = prior.loc[absent].copy()
        if carried.empty:
            sentinel = prior.iloc[[0]].copy()
            sentinel["gsis_id"] = SENTINEL
            sentinel["report_status"] = None
            sentinel["practice_status"] = None
            carried = sentinel
            counts.append((season, week, team, "carried_known_no_absence"))
        else:
            counts.append((season, week, team, "carried_prior_week_absences"))
        carried["week"] = week
        out.append(carried)
    extra = pd.concat(out, ignore_index=True) if out else injuries.iloc[0:0]
    return extra, pd.DataFrame(counts, columns=["season", "week", "team", "case"])


def main() -> None:
    mode = sys.argv[1]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    features = pd.read_parquet(BASE)
    features["kickoff"] = pd.to_datetime(features.kickoff, utc=True)
    root = Path("data")
    player_snapshot = player_snapshot_from_root(
        root / "players" / "raw" / manifest["source_player_snapshot"]
    )
    pbp = load_pbp_snapshot(
        pbp_snapshot_from_root(root / "pbp" / "raw" / manifest["source_pbp_snapshot"])
    )
    values = load_player_value_snapshot(
        player_value_snapshot_from_root(
            root / "players" / "values" / "raw" / manifest["source_player_value_snapshot"]
        )
    )
    depth_snapshot = depth_snapshot_from_root(
        root / "quarterbacks" / "depth" / "raw" / manifest["source_depth_snapshot"]
    )
    depth = load_depth_snapshot(depth_snapshot)
    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    first_seen = _injury_first_seen_index(sorted(features.season.astype(int).unique().tolist()))
    canonical = canonicalize_injuries(injuries, timestamp_fallback="drop", first_seen=first_seen)
    rates = pd.read_parquet(RATES)
    games = features.copy()
    counts = None
    if mode != "control":
        clocks = pd.read_parquet(CLOCKS)
        clocks = clocks.loc[clocks.stage.eq(mode)].copy()
        clocks["decision_at"] = pd.to_datetime(clocks.decision_at, utc=True)
        cutoff_game = clocks.set_index("game_id").decision_at
        games["cutoff"] = games.game_id.map(cutoff_game)
        games["kickoff"] = games.cutoff.map(lambda v: v) + pd.Timedelta(hours=24)
        games["kickoff"] = games.kickoff.fillna(features.kickoff)
        team_week = {}
        for row in games.loc[games.cutoff.notna()].itertuples():
            for team in (row.home_team, row.away_team):
                team_week[(int(row.season), int(row.week), str(team))] = row.cutoff
        ids = attach_snap_player_ids(canonicalize_snaps(snaps), canonicalize_rosters(rosters))
        snap_cols = [c for c in ids.columns if c.endswith("_snaps")]
        ids = ids.loc[ids[snap_cols].fillna(0).sum(axis=1).gt(0) & ids.gsis_id.notna()]
        played = set(
            zip(ids.season.astype(int), ids.week.astype(int), ids.team.astype(str), ids.gsis_id.astype(str), strict=True)
        )
        extra, counts = carry_rows(canonical, played, team_week)
        canonical = pd.concat([canonical, extra], ignore_index=True)
        games = games.drop(columns="cutoff")
    with threadpool_limits(limits=1):
        enriched = enrich_with_player_features(
            games,
            canonical,
            rosters,
            snaps,
            pbp,
            player_stats=values,
            availability_rates=rates,
            depth_charts=depth,
            decision_hours_before_kickoff=24,
            injury_timestamp_fallback="drop",
            injury_first_seen=first_seen,
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = enriched[["game_id", *COLUMNS]]
    result.to_parquet(OUTPUT / f"stage_features_{mode}.parquet")
    summary = {"mode": mode, "rows": len(result)}
    if counts is not None:
        counts.to_parquet(OUTPUT / f"stage_cases_{mode}.parquet")
        summary["cases"] = counts.case.value_counts().to_dict()
    else:
        active = pd.read_parquet(ACTIVE).set_index("game_id")[list(COLUMNS)]
        merged = result.set_index("game_id").join(active, rsuffix="_active")
        summary["max_abs_diff_vs_active"] = {
            c: float(np.nanmax(np.abs(merged[c] - merged[c + "_active"]))) for c in COLUMNS
        }
    print(json.dumps(summary, default=str, indent=1))


if __name__ == "__main__":
    main()
