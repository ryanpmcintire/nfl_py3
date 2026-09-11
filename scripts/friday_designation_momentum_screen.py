from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.io import atomic_parquet, run_id
from nfl_ats.players import attach_snap_player_ids
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.transaction_flag_features import default_schedule
from nfl_ats.transaction_wire_features import canonical_team

INJURIES_PATH = Path("data/raw/nflverse_injuries/20260910T203021Z/injuries.parquet")
SNAP_COUNTS_PATH = Path("data/players/raw/20260817T184901Z/snap_counts.parquet")
WEEKLY_ROSTERS_PATH = Path("data/players/raw/20260817T184901Z/weekly_rosters.parquet")
OUTPUT = Path("artifacts/experiments/friday_designation_momentum")
SEED = 20260910
SAMPLES = 20000
STARTER_SNAP_SHARE_THRESHOLD = 0.5
TRAJECTORY_SEASON_START = 2010
TRAJECTORY_SEASON_END = 2024

PRACTICE_RANK = {
    "did not participate in practice": 0.0,
    "limited participation in practice": 1.0,
    "full participation in practice": 2.0,
    "out (definitely will not play)": 0.0,
}


def load_canonical_injuries() -> pd.DataFrame:
    frame = pd.read_parquet(INJURIES_PATH)
    frame = frame.loc[
        frame["game_type"].eq("REG")
        & frame["season"].between(TRAJECTORY_SEASON_START, TRAJECTORY_SEASON_END)
    ].copy()
    frame["team"] = frame["team"].astype(str).map(canonical_team)
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["date_modified"] = pd.to_datetime(frame["date_modified"], utc=True, errors="coerce")
    frame["practice_rank"] = (
        frame["practice_status"].astype("string").str.strip().str.lower().map(PRACTICE_RANK)
    )
    return frame


def classify_revision_timing(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    eastern = frame["date_modified"].dt.tz_convert("America/New_York")
    frame["revision_weekday"] = eastern.dt.weekday
    frame["revision_class"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    known = frame["date_modified"].notna() & frame["revision_weekday"].notna()
    frame.loc[known & frame["revision_weekday"].le(2), "revision_class"] = "static"
    frame.loc[known & frame["revision_weekday"].ge(3), "revision_class"] = "revised_late"
    return frame


def coverage_report(raw_all_seasons: pd.DataFrame, classified: pd.DataFrame) -> dict[str, Any]:
    by_season: dict[str, Any] = {}
    for season, group in raw_all_seasons.groupby("season"):
        classified_season = classified.loc[classified["season"].eq(season)]
        by_season[str(season)] = {
            "raw_player_week_rows": len(group),
            "real_date_modified_rows": int(group["date_modified"].notna().sum()),
            "resolved_practice_rank_rows": int(group["practice_rank"].notna().sum()),
            "resolved_revision_class_rows": int(classified_season["revision_class"].notna().sum()),
        }
    return {
        "definition": (
            "revision_class is 'static' when date_modified's Eastern weekday is "
            "Mon/Tue/Wed (no revision recorded after the initial filing) and "
            "'revised_late' when it is Thu/Fri/Sat/Sun (the record was touched "
            "again later in the week). This is NOT the predeclared Wed-vs-Fri "
            "per-player practice-level trajectory: the canonical source stores "
            "exactly one row per player-week (measured: 90887 of 90889 "
            "season/week/team/gsis_id groups in the full 2009-2026 archive have "
            "count 1), so no player ever has both an early-week and a Friday "
            "practice observation to sequence. Reported here as the closest "
            "available proxy, not the literal construct."
        ),
        "row_count_check": {
            "single_row_player_weeks": 90887,
            "multi_row_player_weeks": 2,
            "total_player_week_groups": 90889,
        },
        "by_season": by_season,
    }


def attach_participation(injuries: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    home = reg[["season", "week", "home_team", "game_id"]].rename(columns={"home_team": "team"})
    away = reg[["season", "week", "away_team", "game_id"]].rename(columns={"away_team": "team"})
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["team"] = team_games["team"].astype(str).map(canonical_team)
    merged = injuries.merge(
        team_games, on=["season", "week", "team"], how="left", validate="many_to_one"
    )

    snaps = pd.read_parquet(SNAP_COUNTS_PATH)
    rosters = pd.read_parquet(WEEKLY_ROSTERS_PATH)
    snaps = attach_snap_player_ids(snaps, rosters)
    snaps = snaps.loc[snaps["gsis_id"].notna()].copy()
    snaps["team"] = snaps["team"].astype(str).map(canonical_team)
    snaps["season"] = pd.to_numeric(snaps["season"], errors="raise").astype(int)
    snaps["week"] = pd.to_numeric(snaps["week"], errors="raise").astype(int)
    snaps["total_snaps"] = snaps[["offense_snaps", "defense_snaps", "st_snaps"]].sum(axis=1)
    snaps["snap_share"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps = snaps.sort_values(["gsis_id", "season", "week"])
    snaps["trailing_share"] = snaps.groupby("gsis_id")["snap_share"].shift(1)
    snaps = snaps.drop_duplicates(["game_id", "gsis_id"], keep="first")

    merged = merged.merge(
        snaps[["game_id", "gsis_id", "total_snaps", "trailing_share"]],
        on=["game_id", "gsis_id"],
        how="left",
        validate="many_to_one",
    )
    merged["played"] = merged["total_snaps"].fillna(0.0).gt(0.0)
    merged["inactive"] = ~merged["played"]
    merged["snap_row_matched"] = merged["total_snaps"].notna()
    return merged


def base_rate_table(scored: pd.DataFrame) -> dict[str, Any]:
    covered = scored.loc[scored["revision_class"].notna() & scored["practice_rank"].notna()].copy()
    covered["practice_level"] = covered["practice_rank"].map({0.0: "dnp", 1.0: "lp", 2.0: "fp"})
    by_class: dict[str, Any] = {}
    for cls, group in covered.groupby("revision_class", observed=True):
        by_class[str(cls)] = {
            "n": len(group),
            "inactive_rate": float(group["inactive"].mean()),
            "matched_to_a_snap_row": float(group["snap_row_matched"].mean()),
        }
    by_class_level: dict[str, Any] = {}
    for key, group in covered.groupby(["revision_class", "practice_level"], observed=True):
        by_class_level[f"{key[0]}__{key[1]}"] = {
            "n": len(group),
            "inactive_rate": float(group["inactive"].mean()),
        }
    return {
        "n_covered_player_weeks": len(covered),
        "by_revision_class": by_class,
        "by_revision_class_and_practice_level": by_class_level,
    }


def split_half_reliability(scored: pd.DataFrame) -> dict[str, Any]:
    covered = scored.loc[scored["revision_class"].notna()].copy()
    covered["half"] = covered["season"] % 2
    gap_rows = []
    for (team, season, half), group in covered.groupby(["team", "season", "half"], observed=True):
        rates = group.groupby("revision_class", observed=True)["inactive"].mean()
        gap_rows.append(
            {
                "team": team,
                "season": season,
                "half": half,
                "revised_late_rate": rates.get("revised_late", np.nan),
                "static_rate": rates.get("static", np.nan),
            }
        )
    gap_frame = pd.DataFrame(gap_rows)
    gap_frame["gap"] = gap_frame["revised_late_rate"] - gap_frame["static_rate"]

    def _team_half_reliability(column: str) -> dict[str, Any]:
        team_half = (
            gap_frame.dropna(subset=[column])
            .groupby(["team", "half"], observed=True)[column]
            .mean()
            .unstack("half")
        )
        if not {0, 1}.issubset(team_half.columns):
            return {"correlation": None, "n_teams": 0, "status": "missing_halves"}
        team_half = team_half.dropna()
        if len(team_half) < 2 or team_half[0].nunique() < 2 or team_half[1].nunique() < 2:
            return {"correlation": None, "n_teams": len(team_half), "status": "undefined"}
        return {
            "correlation": float(team_half[0].corr(team_half[1])),
            "n_teams": len(team_half),
            "status": "measured",
        }

    return {
        "unit": "team, odd-season mean vs even-season mean",
        "revised_late_absence_rate_reliability": _team_half_reliability("revised_late_rate"),
        "revised_late_minus_static_gap_reliability": _team_half_reliability("gap"),
    }


def walk_forward_severity(scored: pd.DataFrame) -> dict[int, dict[str, float]]:
    covered = scored.loc[scored["revision_class"].notna()].copy()
    overall_mean = float(covered["inactive"].mean())
    seasons = sorted(covered["season"].unique())
    severity: dict[int, dict[str, float]] = {}
    for season in seasons:
        prior = covered.loc[covered["season"].lt(season)]
        rates: dict[str, float] = {}
        for cls in ("static", "revised_late"):
            cls_prior = prior.loc[prior["revision_class"].eq(cls)]
            rates[cls] = (
                float(cls_prior["inactive"].mean()) if len(cls_prior) >= 30 else overall_mean
            )
        severity[int(season)] = rates
    return severity


def build_team_game_feature(scored: pd.DataFrame) -> pd.DataFrame:
    severity = walk_forward_severity(scored)
    eligible = scored.loc[
        scored["revision_class"].notna() & scored["trailing_share"].ge(STARTER_SNAP_SHARE_THRESHOLD)
    ].copy()
    eligible["severity"] = eligible.apply(
        lambda row: severity[int(row["season"])][str(row["revision_class"])], axis=1
    )
    eligible["contribution"] = eligible["trailing_share"] * eligible["severity"]
    eligible["revised_late_flag"] = eligible["revision_class"].eq("revised_late")

    grouped = eligible.groupby(["game_id", "team"], observed=True)
    team_value = grouped["contribution"].sum().rename("expected_absence_value")
    team_flag = grouped["revised_late_flag"].any().rename("flagged")
    team_starters = grouped.size().rename("n_starters_on_report")
    team_frame = pd.concat([team_value, team_flag, team_starters], axis=1).reset_index()
    return team_frame


def screen_against_production(
    per_game: pd.DataFrame, schedule: pd.DataFrame, team_frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reg = schedule.loc[schedule["game_type"].eq("REG"), ["game_id", "home_team", "away_team"]]
    frame = per_game.merge(reg, on="game_id", how="left", validate="one_to_one")
    frame = frame.loc[frame["correct_at_open_probability_rule"].notna()].copy()
    frame["home_team"] = frame["home_team"].astype(str).map(canonical_team)
    frame["away_team"] = frame["away_team"].astype(str).map(canonical_team)

    home_lookup = team_frame.set_index(["game_id", "team"])
    for side in ("home", "away"):
        team_col = f"{side}_team"
        for metric, default in (
            ("expected_absence_value", 0.0),
            ("flagged", False),
            ("n_starters_on_report", 0),
        ):
            values = []
            for game_id, team in zip(frame["game_id"], frame[team_col], strict=True):
                key = (game_id, team)
                if key in home_lookup.index:
                    values.append(home_lookup.loc[key, metric])
                else:
                    values.append(default)
            frame[f"{side}_{metric}"] = values

    frame["home_flagged"] = frame["home_flagged"].astype(bool)
    frame["away_flagged"] = frame["away_flagged"].astype(bool)
    frame["diff_expected_absence_value"] = (
        frame["home_expected_absence_value"] - frame["away_expected_absence_value"]
    )
    frame["production_pick_home"] = frame["pick_home_at_open_probability_rule"].astype(bool)
    asymmetric = frame["home_flagged"] != frame["away_flagged"]
    fade_home = frame["home_flagged"] & ~frame["away_flagged"]
    flip_condition = asymmetric & (
        (fade_home & frame["production_pick_home"]) | (~fade_home & ~frame["production_pick_home"])
    )
    frame["candidate_pick_home"] = np.where(
        flip_condition, ~frame["production_pick_home"], frame["production_pick_home"]
    )
    frame["flip"] = flip_condition
    frame["margin_home_positive"] = frame["margin_vs_open"].gt(0)
    frame["production_correct"] = (
        frame["production_pick_home"].eq(frame["margin_home_positive"]).astype(float)
    )
    frame["candidate_correct"] = (
        frame["candidate_pick_home"].eq(frame["margin_home_positive"]).astype(float)
    )
    coverage = {
        "n_games": len(frame),
        "n_games_home_flagged": int(frame["home_flagged"].sum()),
        "n_games_away_flagged": int(frame["away_flagged"].sum()),
        "n_games_asymmetric": int(asymmetric.sum()),
        "n_flips": int(frame["flip"].sum()),
    }
    return frame, coverage


def week_blocked_summary(frame: pd.DataFrame, seed: int, samples: int) -> dict[str, Any]:
    delta = (frame["candidate_correct"] - frame["production_correct"]) * 100.0
    blocks = pd.DataFrame({"season": frame["season"], "week": frame["week"], "delta": delta})
    groups = blocks.groupby(["season", "week"])["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    group_sums = groups["sum"].to_numpy()
    group_counts = groups["count"].to_numpy()
    n_groups = len(groups)
    for start in range(0, samples, 500):
        batch = min(500, samples - start)
        indices = rng.integers(0, n_groups, size=(batch, n_groups))
        draws[start : start + batch] = group_sums[indices].sum(axis=1) / group_counts[indices].sum(
            axis=1
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "n_games": len(frame),
        "n_week_blocks": int(n_groups),
        "production_accuracy": float(frame["production_correct"].mean() * 100.0),
        "candidate_accuracy": float(frame["candidate_correct"].mean() * 100.0),
        "effect_accuracy_points": float(delta.mean()),
        "interval_low": float(low),
        "interval_high": float(high),
        "probability_positive": float(probability_positive_from_draws(draws)),
        "probability_tie": float((draws == 0.0).mean()),
        "standard_error": float(draws.std(ddof=1)),
    }


def positive_control(frame: pd.DataFrame, seed: int, samples: int) -> dict[str, Any]:
    control = frame.copy()
    control["candidate_pick_home"] = control["margin_home_positive"]
    control["candidate_correct"] = 1.0
    return week_blocked_summary(control, seed, samples)


def main() -> int:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument(
        "--window-seasons",
        type=str,
        default=None,
        help="comma-separated season list restricting the production screen population",
    )
    args = parser.parse_args()

    raw = load_canonical_injuries()
    classified = classify_revision_timing(raw)
    coverage = coverage_report(raw, classified)
    print(json.dumps(coverage, indent=2), flush=True)
    if args.coverage_only:
        return 0

    schedule = default_schedule()
    scored = attach_participation(classified, schedule)
    base_rates = base_rate_table(scored)
    reliability = split_half_reliability(scored)
    team_frame = build_team_game_feature(scored)

    match = find_matching_opener_evaluation(Path("artifacts"))
    if match is None:
        raise ValueError("No opener evaluation matches the active model")
    active_manifest, evaluation_dir = match
    per_game = pd.read_parquet(evaluation_dir / "per_game.parquet")
    screened, screen_coverage = screen_against_production(per_game, schedule, team_frame)

    if args.window_seasons:
        seasons = {int(token) for token in args.window_seasons.split(",")}
        windowed = screened.loc[screened["season"].isin(seasons)].copy()
    else:
        windowed = screened

    full_population_summary = week_blocked_summary(screened, SEED, SAMPLES)
    window_summary = week_blocked_summary(windowed, SEED, SAMPLES)
    control_summary = positive_control(windowed, SEED, SAMPLES)
    by_season_summary = {
        str(season): week_blocked_summary(group, SEED, SAMPLES)
        for season, group in screened.groupby("season")
        if group["candidate_correct"].notna().sum() >= 10
    }

    results = {
        "coverage": coverage,
        "base_rates": base_rates,
        "reliability": reliability,
        "screen_coverage_full_population": screen_coverage,
        "full_population_seasons": sorted(int(s) for s in screened["season"].unique()),
        "full_population_summary": full_population_summary,
        "window_seasons": sorted(int(s) for s in windowed["season"].unique()),
        "window_summary": window_summary,
        "window_positive_control": control_summary,
        "by_season_summary": by_season_summary,
        "active_model_id": active_manifest.get("active_model_id"),
        "opener_evaluation_dir": str(evaluation_dir),
    }
    print(json.dumps(results, indent=2), flush=True)

    destination = OUTPUT / run_id()
    metadata = {
        "raw_injuries_path": str(INJURIES_PATH),
        "raw_injuries_sha256": sha256_file(INJURIES_PATH),
        "snap_counts_path": str(SNAP_COUNTS_PATH),
        "snap_counts_sha256": sha256_file(SNAP_COUNTS_PATH),
        "weekly_rosters_path": str(WEEKLY_ROSTERS_PATH),
        "weekly_rosters_sha256": sha256_file(WEEKLY_ROSTERS_PATH),
        "opener_evaluation_dir": str(evaluation_dir),
        "opener_evaluation_sha256": sha256_file(evaluation_dir / "per_game.parquet"),
        "active_model_id": active_manifest.get("active_model_id"),
        "results": results,
        "seed": SEED,
        "bootstrap_samples": SAMPLES,
        "predeclaration": "ROADMAP.md LEAD-08; docs/friday_designation_momentum.md",
        "provenance": artifact_provenance(
            {"command": "friday-designation-momentum-screen", "seed": SEED, "samples": SAMPLES},
            evaluation_dir / "per_game.parquet",
        ),
    }
    write_experiment_artifact(
        destination,
        "results.json",
        metadata,
        command="friday-designation-momentum-screen",
        metrics={"window_effect_accuracy_points": window_summary["effect_accuracy_points"]},
        registry_root=OUTPUT / "registry",
    )
    atomic_parquet(screened, destination / "paired_predictions.parquet")
    atomic_parquet(team_frame, destination / "team_game_feature.parquet")
    print(json.dumps({"output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
