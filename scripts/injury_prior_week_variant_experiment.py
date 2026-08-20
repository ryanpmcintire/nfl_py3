"""Predeclared in ``docs/injury_prior_week_variant.md``: does a Tuesday-visible,
one-week-lagged injury signal (last week's final report / last week's actual
absences) carry playable ATS information, given that
``docs/injury_news_sourcing.md`` sec 5.1 measured the *current*-week
Saturday-cutoff ``injury_value_lost_narrowed`` edge (+1.316 pts, P+ 0.8875)
collapses to +0.000 pts (P+ 0.3965) under a true Tuesday-noon lock, because
99.57% of official injury-report rows are not yet filed by Tuesday noon?

Free re-read precedent, no new rotation window: rebuilds
``injury_value_lost_narrowed``'s D-A contrast (``docs/injury_value_lost.md``
sec 4) on the ALREADY-SPENT ``mod07_weak_signal_stack`` ``[2020, 2021]``
opener window, exactly as ``scripts/injury_tuesday_cutoff_experiment.py``
already does. Nothing here touches ``rotation_registry.json`` and
``[2022, 2023]`` is never referenced.

Two new arms, both replacing the production ``injuries`` table with a
synthetic table keyed at the CURRENT (season, week, team) but built only
from PRIOR-week information (see the predeclaration doc for the exact
construction rules and the exclusion/scope-limit language, quoted nowhere
here to avoid drift -- the doc is the source of truth):

* ``prior_week_report`` -- last week's Friday-final official designation,
  kept only for players who recorded zero snaps in that prior game.
* ``prior_week_absence`` -- every player on last week's active
  (ACT/INA) roster who recorded zero snaps in that prior game, severity
  fixed at 1.0 ("Out"), report-independent.

A fresh ``saturday`` arm (real, unmodified injuries,
``decision_hours_before_kickoff=24``) is also rebuilt for a reproduction
check against the recorded +1.316/P+0.8875, and to supply paired per-game
data for a channel-delta contrast against each new arm (same technique
``scripts/injury_tuesday_cutoff_experiment.py`` used for its Tuesday
arms).

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/injury_prior_week_variant_experiment.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import opener_pick_evaluation, week_blocked_bootstrap
from nfl_ats.modeling import regular_season_rows
from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.players import (
    _ACTIVE_ROSTER_STATUSES,
    INJURY_REQUIRED_COLUMNS,
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    enrich_with_player_features,
    load_player_snapshot,
    load_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.rotation import load_registry

REPO = Path(__file__).resolve().parents[1]

REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
MIN_TRAIN_GAMES = 500
SAMPLES = 20000
SEED = 20260819

# Exact snapshots scripts/injury_tuesday_cutoff_experiment.py used, matching
# the on-disk game_features_player_value.parquet manifest -- kept identical
# here so results are directly comparable to the recorded +1.316 (Saturday)
# and 0.000 (Tuesday-official) arms on the same 456 games.
PLAYER_SNAPSHOT_ID = "20260812T200527Z"
PBP_SNAPSHOT_ID = "20260812T142851Z"
PLAYER_VALUE_SNAPSHOT_ID = "20260813T121050Z"

RECORDED_SATURDAY_D_MINUS_A = {"delta_points": 1.316, "probability_positive": 0.8875}
RECORDED_TUESDAY_OFFICIAL_D_MINUS_A = {"delta_points": 0.000, "probability_positive": 0.3965}


def _config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


# ---------------------------------------------------------------------------
# Team schedule: each team's actual immediately-preceding game (bye-safe)
# ---------------------------------------------------------------------------


def team_schedule_prior_week(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with that team's own immediately
    preceding (season, prior_week, prior_game_id, prior_kickoff), derived from
    the actual sorted sequence of games the team played (not week-1
    arithmetic), so a bye week is skipped correctly. Week 1 of each season has
    no prior game -- ``prior_week`` is NaN, excluded by construction.
    """

    rows = []
    for side in ("home_team", "away_team"):
        rows.append(
            games[["season", "week", "game_id", "kickoff", side]].rename(columns={side: "team"})
        )
    long = pd.concat(rows, ignore_index=True).drop_duplicates(["season", "team", "week"])
    long = long.sort_values(["team", "season", "week"]).reset_index(drop=True)
    grouped = long.groupby(["team", "season"], sort=False)
    long["prior_week"] = grouped["week"].shift(1)
    long["prior_game_id"] = grouped["game_id"].shift(1)
    long["prior_kickoff"] = grouped["kickoff"].shift(1)
    return long


# ---------------------------------------------------------------------------
# Played / active-roster-absent tables, built exactly like production inputs
# ---------------------------------------------------------------------------


def build_played_and_active(
    injuries: pd.DataFrame, rosters: pd.DataFrame, snaps: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (injuries_c, played, active_roster) -- all canonicalized the
    same way ``enrich_with_player_features`` canonicalizes its own inputs, so
    the prior-week tables below are built from the identical data production
    would see.
    """

    injuries_c = canonicalize_injuries(injuries)
    rosters_c = canonicalize_rosters(rosters)
    snaps_c = canonicalize_snaps(snaps)
    snaps_linked = attach_snap_player_ids(snaps_c, rosters_c)
    total_snaps = (
        snaps_linked["offense_snaps"].fillna(0)
        + snaps_linked["defense_snaps"].fillna(0)
        + snaps_linked["st_snaps"].fillna(0)
    )
    played = (
        snaps_linked.loc[
            snaps_linked["gsis_id"].notna() & (total_snaps > 0),
            ["season", "week", "team", "gsis_id"],
        ]
        .assign(gsis_id=lambda f: f["gsis_id"].astype(str))
        .drop_duplicates()
        .reset_index(drop=True)
    )
    active_roster = (
        rosters_c.loc[
            rosters_c["status"].isin(_ACTIVE_ROSTER_STATUSES),
            ["season", "week", "team", "gsis_id", "position"],
        ]
        .drop_duplicates(["season", "week", "team", "gsis_id"])
        .reset_index(drop=True)
    )
    return injuries_c, played, active_roster


def build_prior_week_report(
    injuries_c: pd.DataFrame, played: pd.DataFrame, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Arm A: last week's Friday-final designation, kept only for players who
    recorded zero snaps in that prior game (the "not resolved by kickoff"
    filter predeclared in docs/injury_prior_week_variant.md sec 2)."""

    final_report = (
        injuries_c.sort_values("date_modified")
        .drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
        .reset_index(drop=True)
    )
    played_flagged = played.assign(_played=True)
    final_report_played = final_report.merge(
        played_flagged, on=["season", "week", "team", "gsis_id"], how="left"
    )
    resolved = int(final_report_played["_played"].notna().sum())
    unresolved = final_report_played.loc[final_report_played["_played"].isna()].drop(
        columns="_played"
    )

    prior = unresolved.rename(columns={"week": "prior_week"})
    mapped = schedule.dropna(subset=["prior_week"]).merge(
        prior, on=["season", "team", "prior_week"], how="inner"
    )
    synthetic = pd.DataFrame(
        {
            "season": mapped["season"].astype(int),
            "game_type": "REG",
            "team": mapped["team"],
            "week": mapped["week"].astype(int),
            "gsis_id": mapped["gsis_id"],
            "position": mapped["position"],
            "report_status": mapped["report_status"],
            "practice_status": mapped["practice_status"],
            "date_modified": mapped["date_modified"],
        }
    )
    stats = {
        "final_report_rows": len(final_report),
        "resolved_dropped_played_normally": resolved,
        "unresolved_carried_forward": len(unresolved),
        "synthetic_rows_after_schedule_mapping": len(synthetic),
    }
    return synthetic[list(INJURY_REQUIRED_COLUMNS)], stats


def build_prior_week_absence(
    played: pd.DataFrame, active_roster: pd.DataFrame, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Arm B: last week's active-roster (ACT/INA) players who recorded zero
    snaps in that prior game -- report-independent."""

    played_flagged = played.assign(_played=True)
    merged = active_roster.merge(
        played_flagged, on=["season", "week", "team", "gsis_id"], how="left"
    )
    absentees = merged.loc[merged["_played"].isna()].drop(columns="_played")

    prior = absentees.rename(columns={"week": "prior_week"})
    mapped = schedule.dropna(subset=["prior_week"]).merge(
        prior, on=["season", "team", "prior_week"], how="inner"
    )
    date_modified = mapped["prior_kickoff"] + pd.Timedelta(hours=1)
    synthetic = pd.DataFrame(
        {
            "season": mapped["season"].astype(int),
            "game_type": "REG",
            "team": mapped["team"],
            "week": mapped["week"].astype(int),
            "gsis_id": mapped["gsis_id"],
            "position": mapped["position"],
            "report_status": "Out",
            "practice_status": pd.NA,
            "date_modified": date_modified,
        }
    )
    stats = {
        "active_roster_rows": len(active_roster),
        "absentee_rows": len(absentees),
        "synthetic_rows_after_schedule_mapping": len(synthetic),
    }
    return synthetic[list(INJURY_REQUIRED_COLUMNS)], stats


# ---------------------------------------------------------------------------
# Arm construction / contrast (re-derived from scripts/injury_tuesday_cutoff_experiment.py)
# ---------------------------------------------------------------------------


def spent_window_split(
    features: pd.DataFrame, seasons: tuple[int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    span = list(range(int(seasons[0]), int(seasons[1]) + 1))
    missing = [s for s in span if s not in set(frame["season"].astype(int))]
    if missing:
        raise ValueError(f"Feature table is missing window seasons: {missing}")
    window = frame.loc[frame["season"].astype(int).isin(span)].copy()
    cutoff = window["gameday"].min()
    training = frame.loc[frame["gameday"].lt(cutoff) & frame["result"].notna()].copy()
    if training.empty:
        raise ValueError("No completed games before the window; cannot forward-chain")
    return training, window


def arm(
    features: pd.DataFrame,
    *,
    seasons: tuple[int, int],
    profile: str,
    market_root: Path,
    min_train_games: int,
) -> pd.DataFrame:
    training, window = spent_window_split(features, seasons)
    scoped = pd.concat([training, window], ignore_index=True)
    scored = opener_pick_evaluation(
        market_root,
        scoped,
        active_model_config=_config(profile),
        min_train_games=min_train_games,
    )
    span = sorted(window["season"].astype(int).unique())
    scored = scored.loc[scored["season"].isin(span)]
    return scored.loc[scored["correct_at_open"].notna()].copy()


def paired_frame(left_arm: pd.DataFrame, right_arm: pd.DataFrame) -> pd.DataFrame:
    left = left_arm[["game_id", "season", "week", "correct_at_open", "pick_home_at_open"]].rename(
        columns={"correct_at_open": "left_correct", "pick_home_at_open": "left_pick_home"}
    )
    right = right_arm[["game_id", "correct_at_open", "pick_home_at_open"]].rename(
        columns={"correct_at_open": "right_correct", "pick_home_at_open": "right_pick_home"}
    )
    paired = left.merge(right, on="game_id", how="inner")
    paired["left_correct"] = paired["left_correct"].astype(float)
    paired["right_correct"] = paired["right_correct"].astype(float)
    return paired.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _metric_fn(frame: pd.DataFrame) -> dict[str, float]:
    return {"right_minus_left": float(frame["right_correct"].mean() - frame["left_correct"].mean())}


def contrast(left_arm: pd.DataFrame, right_arm: pd.DataFrame, *, name: str) -> dict[str, Any]:
    paired = paired_frame(left_arm, right_arm)
    if paired.empty:
        raise ValueError(f"{name}: no game scored by both arms")
    bootstrap = week_blocked_bootstrap(paired, _metric_fn, block="week", samples=SAMPLES, seed=SEED)
    row = bootstrap.iloc[0]
    disagree = paired.loc[paired["left_pick_home"].ne(paired["right_pick_home"])]
    return {
        "contrast": name,
        "paired_games": len(paired),
        "weeks": int(paired.groupby(["season", "week"]).ngroups),
        "left_accuracy": float(paired["left_correct"].mean()),
        "right_accuracy": float(paired["right_correct"].mean()),
        "delta_points": float(row["estimate"]) * 100.0,
        "bootstrap_lower_points": float(row["lower"]) * 100.0,
        "bootstrap_upper_points": float(row["upper"]) * 100.0,
        "probability_positive": float(row["probability_positive"]),
        "picks_disagreeing": len(disagree),
    }


def channel_delta(
    arm_a_sat: pd.DataFrame,
    arm_d_sat: pd.DataFrame,
    arm_a_candidate: pd.DataFrame,
    arm_d_candidate: pd.DataFrame,
    *,
    name: str,
) -> dict[str, Any]:
    """Paired Saturday-D-minus-A vs candidate-D-minus-A: isolates whether the
    candidate channel differs from the whole Saturday channel, on the same
    games."""

    sat = paired_frame(arm_a_sat, arm_d_sat).rename(
        columns={"left_correct": "sat_a", "right_correct": "sat_d"}
    )[["game_id", "season", "week", "sat_a", "sat_d"]]
    cand = paired_frame(arm_a_candidate, arm_d_candidate).rename(
        columns={"left_correct": "cand_a", "right_correct": "cand_d"}
    )[["game_id", "cand_a", "cand_d"]]
    merged = sat.merge(cand, on="game_id", how="inner")
    merged["sat_diff"] = merged["sat_d"] - merged["sat_a"]
    merged["cand_diff"] = merged["cand_d"] - merged["cand_a"]

    def metric(frame: pd.DataFrame) -> dict[str, float]:
        return {"channel_delta": float(frame["sat_diff"].mean() - frame["cand_diff"].mean())}

    bootstrap = week_blocked_bootstrap(merged, metric, block="week", samples=SAMPLES, seed=SEED)
    row = bootstrap.iloc[0]
    return {
        "contrast": name,
        "paired_games": len(merged),
        "weeks": int(merged.groupby(["season", "week"]).ngroups),
        "saturday_D_minus_A_points": float(merged["sat_diff"].mean()) * 100.0,
        "candidate_D_minus_A_points": float(merged["cand_diff"].mean()) * 100.0,
        "channel_delta_points": float(row["estimate"]) * 100.0,
        "bootstrap_lower_points": float(row["lower"]) * 100.0,
        "bootstrap_upper_points": float(row["upper"]) * 100.0,
        "probability_positive": float(row["probability_positive"]),
    }


def build_enriched(
    *,
    games: pd.DataFrame,
    injuries: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame,
    pbp: pd.DataFrame,
    player_stats: pd.DataFrame,
    decision_hours_before_kickoff: float,
) -> pd.DataFrame:
    return enrich_with_player_features(
        games,
        injuries,
        rosters,
        snaps,
        pbp,
        player_stats,
        decision_hours_before_kickoff=decision_hours_before_kickoff,
        role_span=8,
        qb_span=12,
        qb_min_dropbacks=20,
        offseason_retention=0.75,
        value_span=16,
        value_prior_snaps=200.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="mod07_weak_signal_stack")
    parser.add_argument(
        "--features", type=Path, default=REPO / "data/processed/game_features_pbp.parquet"
    )
    parser.add_argument("--market-root", type=Path, default=REPO / "data/market/raw")
    parser.add_argument(
        "--out", type=Path, default=REPO / "artifacts/injury_prior_week_variant/result.json"
    )
    args = parser.parse_args()

    registry = load_registry(None)
    declared = registry.families[args.family]
    spent = [w for w in declared.windows if w.state == "spent"]
    if not spent:
        raise SystemExit(f"{args.family} has no spent window.")
    seasons = (int(spent[-1].seasons[0]), int(spent[-1].seasons[1]))

    games = pd.read_parquet(args.features)
    player_root = REPO / "data/players/raw"
    pbp_root = REPO / "data/pbp/raw"
    player_value_root = REPO / "data/players/values/raw"

    player_snapshot = player_snapshot_from_root(player_root / PLAYER_SNAPSHOT_ID)
    pbp_snapshot = pbp_snapshot_from_root(pbp_root / PBP_SNAPSHOT_ID)
    player_value_snapshot = player_value_snapshot_from_root(
        player_value_root / PLAYER_VALUE_SNAPSHOT_ID
    )

    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    pbp = load_pbp_snapshot(pbp_snapshot)
    player_stats = load_player_value_snapshot(player_value_snapshot)

    schedule = team_schedule_prior_week(games)
    week1_team_games = int(schedule["prior_week"].isna().sum())

    injuries_c, played, active_roster = build_played_and_active(injuries, rosters, snaps)
    prior_week_report_injuries, report_stats = build_prior_week_report(injuries_c, played, schedule)
    prior_week_absence_injuries, absence_stats = build_prior_week_absence(
        played, active_roster, schedule
    )

    coverage = {
        "team_games_total": len(schedule),
        "team_games_week1_no_prior": week1_team_games,
        "prior_week_report": report_stats,
        "prior_week_absence": absence_stats,
    }

    print("Building Saturday-cutoff arm (fresh rebuild, real injuries, decision_hours=24)...")
    enriched_saturday = build_enriched(
        games=games,
        injuries=injuries,
        rosters=rosters,
        snaps=snaps,
        pbp=pbp,
        player_stats=player_stats,
        decision_hours_before_kickoff=24,
    )
    print("Building prior_week_report arm (decision_hours=24, synthetic injuries)...")
    enriched_prior_report = build_enriched(
        games=games,
        injuries=prior_week_report_injuries,
        rosters=rosters,
        snaps=snaps,
        pbp=pbp,
        player_stats=player_stats,
        decision_hours_before_kickoff=24,
    )
    print("Building prior_week_absence arm (decision_hours=24, synthetic injuries)...")
    enriched_prior_absence = build_enriched(
        games=games,
        injuries=prior_week_absence_injuries,
        rosters=rosters,
        snaps=snaps,
        pbp=pbp,
        player_stats=player_stats,
        decision_hours_before_kickoff=24,
    )

    results: dict[str, Any] = {"window": list(seasons), "grade": "opener", "coverage": coverage}
    arm_pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for label, enriched in (
        ("saturday", enriched_saturday),
        ("prior_week_report", enriched_prior_report),
        ("prior_week_absence", enriched_prior_absence),
    ):
        arm_a = arm(
            enriched,
            seasons=seasons,
            profile="player",
            market_root=args.market_root,
            min_train_games=MIN_TRAIN_GAMES,
        )
        arm_d = arm(
            enriched,
            seasons=seasons,
            profile="player_value",
            market_root=args.market_root,
            min_train_games=MIN_TRAIN_GAMES,
        )
        arm_pairs[label] = (arm_a, arm_d)
        results[label] = {"contrast": contrast(arm_a, arm_d, name=f"D_minus_A_{label}")}

    results["saturday_reproduction_check"] = {
        "recorded_delta_points": RECORDED_SATURDAY_D_MINUS_A["delta_points"],
        "reproduced_delta_points": results["saturday"]["contrast"]["delta_points"],
        "recorded_probability_positive": RECORDED_SATURDAY_D_MINUS_A["probability_positive"],
        "reproduced_probability_positive": results["saturday"]["contrast"]["probability_positive"],
    }
    results["tuesday_official_reference"] = RECORDED_TUESDAY_OFFICIAL_D_MINUS_A

    for label in ("prior_week_report", "prior_week_absence"):
        sat_a, sat_d = arm_pairs["saturday"]
        cand_a, cand_d = arm_pairs[label]
        results[f"{label}_channel_delta_vs_saturday"] = channel_delta(
            sat_a, sat_d, cand_a, cand_d, name=f"saturday_minus_{label}_channel"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "injury-prior-week-variant-experiment",
        "family": args.family,
        "features": str(args.features),
        "market_root": str(args.market_root),
    }
    results["provenance"] = artifact_provenance(configuration, args.features, project_root=REPO)
    write_experiment_artifact(
        args.out.parent,
        args.out.name,
        results,
        command="injury-prior-week-variant-experiment",
        metrics=results,
        notes=(
            "Prior-week (Tuesday-visible) injury-report/absence variants vs. the "
            "Saturday-cutoff reproduction, on the already-spent mod07_weak_signal_stack "
            "opener window -- no new rotation window drawn."
        ),
    )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
