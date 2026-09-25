from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from line_move_yardstick_paired_eval import cell_stats  # noqa: E402
from news_trigger_historical import classify, loso_logistic  # noqa: E402

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN  # noqa: E402
from nfl_ats.pick_probability_fit import FIT_FEATURES, build_fit_population  # noqa: E402
from nfl_ats.players import (  # noqa: E402
    attach_snap_player_ids,
    canonicalize_rosters,
    canonicalize_snaps,
    latest_player_snapshot,
    load_player_snapshot,
)
from nfl_ats.signal_atlas import _cell as signal_cell  # noqa: E402

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
PLAYER_RAW_ROOT = DATA_ROOT / "players" / "raw"
OUTPUT_ROOT = ARTIFACTS_ROOT / "news_trigger_player_value"

POINT_IN_TIME_COLUMN = "home_injury_observed_at"
FIRST_SEASON = 2020
LAST_SEASON = 2024
QUESTIONABLE_SEVERITY = 0.35
ROLE_SPAN = 8
ROLE_ALPHA = 2.0 / (ROLE_SPAN + 1.0)
FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF = -0.320
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260925
CANDIDATE_A_FEATURES = (*FIT_FEATURES, "news_trigger_value_shift")
CANDIDATE_B_FEATURES = (*FIT_FEATURES, "news_trigger_value_shift_v2")


def gate_visible(kickoff_et: pd.Series) -> pd.Series:
    weekday = kickoff_et.dt.day_name()
    hour = kickoff_et.dt.hour
    monday_block = weekday.eq("Monday")
    late_sunday_block = weekday.eq("Sunday") & hour.ge(16)
    return ~(monday_block | late_sunday_block)


def load_kickoff(data_root: Path) -> pd.DataFrame:
    return pd.read_parquet(
        data_root / "processed" / "game_features.parquet",
        columns=["game_id", "kickoff"],
    )


def load_injury_channel_totals(data_root: Path) -> pd.DataFrame:
    table = pd.read_parquet(
        data_root / "processed" / "game_features_player_value.parquet",
        columns=[
            "game_id",
            "home_injury_skill_epa_value_lost",
            "home_injury_defense_disruption_value_lost",
            "away_injury_skill_epa_value_lost",
            "away_injury_defense_disruption_value_lost",
            POINT_IN_TIME_COLUMN,
            "away_injury_observed_at",
        ],
    )
    table["point_in_time"] = (
        table[POINT_IN_TIME_COLUMN].notna() & table["away_injury_observed_at"].notna()
    )
    return table


def trailing_role_shares(data_root: Path) -> pd.DataFrame:
    snapshot = latest_player_snapshot(PLAYER_RAW_ROOT)
    _injuries, rosters, snaps = load_player_snapshot(snapshot)
    canonical_rosters = canonicalize_rosters(rosters)
    snaps_with_ids = attach_snap_player_ids(canonicalize_snaps(snaps), canonical_rosters)
    rows = snaps_with_ids.loc[snaps_with_ids["gsis_id"].notna()].copy()
    for column in ("offense_pct", "defense_pct"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    rows = rows.sort_values(["team", "gsis_id", "season", "week", "game_id"])
    grouped = rows.groupby(["team", "gsis_id"], sort=False)
    rows["trailing_offense_pct"] = grouped["offense_pct"].transform(
        lambda series: series.shift(1).ewm(alpha=ROLE_ALPHA, adjust=False).mean()
    )
    rows["trailing_defense_pct"] = grouped["defense_pct"].transform(
        lambda series: series.shift(1).ewm(alpha=ROLE_ALPHA, adjust=False).mean()
    )
    rows["trailing_offense_pct"] = rows["trailing_offense_pct"].fillna(0.0)
    rows["trailing_defense_pct"] = rows["trailing_defense_pct"].fillna(0.0)
    return rows[
        ["game_id", "team", "gsis_id", "trailing_offense_pct", "trailing_defense_pct"]
    ].drop_duplicates(["game_id", "team", "gsis_id"])


def team_resolution_table(data_root: Path) -> pd.DataFrame:
    outcomes = pd.read_parquet(
        data_root / "processed" / "injury_play_outcomes.parquet",
        columns=[
            "game_id",
            "team",
            "gsis_id",
            "report_category",
            "unavailable",
            "fixed_unavailability",
        ],
    )
    roles = trailing_role_shares(data_root)
    outcomes = outcomes.merge(roles, on=["game_id", "team", "gsis_id"], how="left")
    outcomes["trailing_offense_pct"] = outcomes["trailing_offense_pct"].fillna(0.0)
    outcomes["trailing_defense_pct"] = outcomes["trailing_defense_pct"].fillna(0.0)
    outcomes["skill_severity_share"] = (
        outcomes["fixed_unavailability"] * outcomes["trailing_offense_pct"]
    )
    outcomes["defense_severity_share"] = (
        outcomes["fixed_unavailability"] * outcomes["trailing_defense_pct"]
    )

    severity_sum = (
        outcomes.groupby(["game_id", "team"], observed=True)["fixed_unavailability"]
        .sum()
        .rename("severity_sum")
    )
    channel_denoms = outcomes.groupby(["game_id", "team"], observed=True)[
        ["skill_severity_share", "defense_severity_share"]
    ].sum()
    channel_denoms = channel_denoms.rename(
        columns={"skill_severity_share": "skill_denom", "defense_severity_share": "defense_denom"}
    )

    questionable = outcomes.loc[outcomes["report_category"] == "questionable"].copy()
    questionable["resolution_delta"] = questionable["unavailable"] - QUESTIONABLE_SEVERITY
    questionable["weighted_offense_resolution"] = (
        questionable["trailing_offense_pct"] * questionable["resolution_delta"]
    )
    questionable["weighted_defense_resolution"] = (
        questionable["trailing_defense_pct"] * questionable["resolution_delta"]
    )
    resolution_sum = questionable.groupby(["game_id", "team"], observed=True)[
        "resolution_delta"
    ].sum()
    resolution_sum = resolution_sum.rename("resolution_sum")
    questionable_count = questionable.groupby(["game_id", "team"], observed=True)[
        "resolution_delta"
    ].size()
    questionable_count = questionable_count.rename("questionable_count")
    weighted_sums = questionable.groupby(["game_id", "team"], observed=True)[
        ["weighted_offense_resolution", "weighted_defense_resolution"]
    ].sum()

    merged = pd.concat(
        [severity_sum, channel_denoms, resolution_sum, questionable_count, weighted_sums], axis=1
    ).reset_index()
    for column in (
        "severity_sum",
        "skill_denom",
        "defense_denom",
        "resolution_sum",
        "weighted_offense_resolution",
        "weighted_defense_resolution",
    ):
        merged[column] = merged[column].fillna(0.0)
    merged["questionable_count"] = merged["questionable_count"].fillna(0).astype(int)
    return merged


def build_population(artifacts_root: Path, data_root: Path) -> tuple[pd.DataFrame, dict]:
    population, provenance = build_fit_population(artifacts_root, data_root)
    population = population.merge(
        load_kickoff(data_root), on="game_id", how="left", validate="one_to_one"
    )
    population = population.merge(
        load_injury_channel_totals(data_root), on="game_id", how="left", validate="one_to_one"
    )

    resolutions = team_resolution_table(data_root)
    rename_map = {
        "severity_sum": "severity_sum",
        "skill_denom": "skill_denom",
        "defense_denom": "defense_denom",
        "resolution_sum": "resolution_sum",
        "questionable_count": "questionable_count",
        "weighted_offense_resolution": "weighted_offense_resolution",
        "weighted_defense_resolution": "weighted_defense_resolution",
    }
    home = resolutions.rename(
        columns={"team": "home_team", **{k: f"home_{v}" for k, v in rename_map.items()}}
    )
    away = resolutions.rename(
        columns={"team": "away_team", **{k: f"away_{v}" for k, v in rename_map.items()}}
    )
    population = population.merge(home, on=["game_id", "home_team"], how="left")
    population = population.merge(away, on=["game_id", "away_team"], how="left")
    for side in ("home", "away"):
        for column in rename_map.values():
            key = f"{side}_{column}"
            fill = 0 if column == "questionable_count" else 0.0
            population[key] = population[key].fillna(fill)

    home_v_severity_only = np.where(
        population["home_severity_sum"] > 0.0,
        (
            population["home_injury_skill_epa_value_lost"]
            + population["home_injury_defense_disruption_value_lost"]
        )
        / population["home_severity_sum"],
        0.0,
    )
    away_v_severity_only = np.where(
        population["away_severity_sum"] > 0.0,
        (
            population["away_injury_skill_epa_value_lost"]
            + population["away_injury_defense_disruption_value_lost"]
        )
        / population["away_severity_sum"],
        0.0,
    )
    shift_a_home = home_v_severity_only * population["home_resolution_sum"]
    shift_a_away = away_v_severity_only * population["away_resolution_sum"]

    home_v_role_skill = np.where(
        population["home_skill_denom"] > 0.0,
        population["home_injury_skill_epa_value_lost"] / population["home_skill_denom"],
        0.0,
    )
    home_v_role_defense = np.where(
        population["home_defense_denom"] > 0.0,
        population["home_injury_defense_disruption_value_lost"] / population["home_defense_denom"],
        0.0,
    )
    away_v_role_skill = np.where(
        population["away_skill_denom"] > 0.0,
        population["away_injury_skill_epa_value_lost"] / population["away_skill_denom"],
        0.0,
    )
    away_v_role_defense = np.where(
        population["away_defense_denom"] > 0.0,
        population["away_injury_defense_disruption_value_lost"] / population["away_defense_denom"],
        0.0,
    )
    shift_b_home = (
        home_v_role_skill * population["home_weighted_offense_resolution"]
        + home_v_role_defense * population["home_weighted_defense_resolution"]
    )
    shift_b_away = (
        away_v_role_skill * population["away_weighted_offense_resolution"]
        + away_v_role_defense * population["away_weighted_defense_resolution"]
    )

    kickoff_et = population["kickoff"].dt.tz_convert("America/New_York")
    population["visible_before_deadline"] = gate_visible(kickoff_et)
    population["news_trigger_value_shift"] = np.where(
        population["visible_before_deadline"], shift_a_home - shift_a_away, 0.0
    )
    population["news_trigger_value_shift_v2"] = np.where(
        population["visible_before_deadline"], shift_b_home - shift_b_away, 0.0
    )
    population["news_trigger_margin_points_v2"] = (
        FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF * population["news_trigger_value_shift_v2"]
    )
    population["questionable_players_pregame"] = (
        population["home_questionable_count"] + population["away_questionable_count"]
    )
    return population, provenance


def accuracy_cell(scoped: pd.DataFrame, baseline_oos: pd.Series, candidate_oos: pd.Series) -> dict:
    scored = scoped.copy()
    scored["rating_full"] = candidate_oos
    scored["rating_reduced"] = baseline_oos
    scored = scored.loc[scored["rating_full"].notna() & scored["rating_reduced"].notna()].copy()
    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": 0.95,
        "reliability_edges": [0.0, 0.5, 1.0],
    }
    return signal_cell(scored, "rating", "overall", declaration)


def line_move_cell(
    label: str, scoped: pd.DataFrame, baseline_oos: pd.Series, candidate_oos: pd.Series
) -> dict:
    valid = scoped[MOVE_AVAILABLE_COLUMN].eq(1.0) & baseline_oos.notna() & candidate_oos.notna()
    sub = scoped.loc[valid].copy()
    base_pick_home = baseline_oos.loc[valid].ge(0.5)
    candidate_pick_home = candidate_oos.loc[valid].ge(0.5)
    move = pd.to_numeric(sub[MOVE_COLUMN], errors="coerce").astype(float)
    lm_base = np.where(base_pick_home, 1.0, -1.0) * move
    lm_candidate = np.where(candidate_pick_home, 1.0, -1.0) * move
    sub["diff_line_move"] = lm_candidate - lm_base
    return cell_stats(label, sub, "diff_line_move", BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)


def main() -> None:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    population, provenance = build_population(ARTIFACTS_ROOT, DATA_ROOT)

    scoped = population.loc[
        population["season"].between(FIRST_SEASON, LAST_SEASON)
        & population["point_in_time"].fillna(False)
        & population["news_trigger_value_shift"].notna()
        & population["news_trigger_value_shift_v2"].notna()
    ].reset_index(drop=True)

    identification = {
        "scoped_games": len(scoped),
        "games_with_any_questionable_pregame": int(
            scoped["questionable_players_pregame"].gt(0).sum()
        ),
        "games_passing_visibility_gate": int(scoped["visible_before_deadline"].sum()),
        "games_with_nonzero_resolved_shift_v2": int(
            scoped["news_trigger_value_shift_v2"].ne(0.0).sum()
        ),
        "mean_abs_margin_points_v2_when_nonzero": float(
            scoped.loc[
                scoped["news_trigger_value_shift_v2"].ne(0.0), "news_trigger_margin_points_v2"
            ]
            .abs()
            .mean()
        )
        if scoped["news_trigger_value_shift_v2"].ne(0.0).any()
        else 0.0,
        "inactives_post_time_check": (
            "54/55 data/players/inactives/*/inactives.parquet snapshots empty; the one "
            "nonempty snapshot carries only captured_at_utc, not a per-game post time; "
            "visibility gate kept unchanged"
        ),
    }

    baseline_oos, baseline_folds = loso_logistic(scoped, FIT_FEATURES)
    candidate_a_oos, candidate_a_folds = loso_logistic(scoped, CANDIDATE_A_FEATURES)
    candidate_b_oos, candidate_b_folds = loso_logistic(scoped, CANDIDATE_B_FEATURES)

    accuracy_a = accuracy_cell(scoped, baseline_oos, candidate_a_oos)
    accuracy_b = accuracy_cell(scoped, baseline_oos, candidate_b_oos)
    line_move_a = line_move_cell("severity_only_line_move", scoped, baseline_oos, candidate_a_oos)
    line_move_b = line_move_cell(
        "per_player_role_share_line_move", scoped, baseline_oos, candidate_b_oos
    )

    summary = {
        "predeclaration": {
            "candidate_a": "news_trigger_value_shift (Unit 1 severity-only, recomputed here)",
            "candidate_b": "news_trigger_value_shift_v2 (per-player trailing role-share, "
            "channel-split skill/defense value-per-role-share, this unit)",
            "role_span_games": ROLE_SPAN,
            "visibility_gate": "not Monday and not (Sunday and kickoff_hour_et >= 16), unchanged",
            "seasons": [FIRST_SEASON, LAST_SEASON],
            "opener_evaluation": provenance["opener_evaluation"],
        },
        "identification": identification,
        "baseline_folds": baseline_folds,
        "candidate_a_folds": candidate_a_folds,
        "candidate_b_folds": candidate_b_folds,
        "candidate_a_accuracy_cell": accuracy_a,
        "candidate_a_classification": classify(accuracy_a["probability_positive"]),
        "candidate_b_accuracy_cell": accuracy_b,
        "candidate_b_classification": classify(accuracy_b["probability_positive"]),
        "candidate_a_line_move_cell": line_move_a,
        "candidate_a_line_move_classification": classify(
            line_move_a["season_block_probability_positive"]
        ),
        "candidate_b_line_move_cell": line_move_b,
        "candidate_b_line_move_classification": classify(
            line_move_b["season_block_probability_positive"]
        ),
    }

    scoped.to_parquet(output_dir / "scoped_population.parquet", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(identification, indent=2))
    print(
        "candidate_a accuracy_delta:",
        accuracy_a["accuracy_delta_points"],
        "probability_positive:",
        accuracy_a["probability_positive"],
        "decisive:",
        accuracy_a["decisive_games"],
        accuracy_a["full_decisive_wins"],
        accuracy_a["reduced_decisive_wins"],
    )
    print(
        "candidate_b accuracy_delta:",
        accuracy_b["accuracy_delta_points"],
        "probability_positive:",
        accuracy_b["probability_positive"],
        "decisive:",
        accuracy_b["decisive_games"],
        accuracy_b["full_decisive_wins"],
        accuracy_b["reduced_decisive_wins"],
    )
    print(
        "candidate_a line_move mean_points:",
        line_move_a["mean_points"],
        "probability_positive:",
        line_move_a["season_block_probability_positive"],
        "decisive:",
        line_move_a["decisive_games"],
        line_move_a["decisive_side_a_wins"],
        line_move_a["decisive_side_b_wins"],
    )
    print(
        "candidate_b line_move mean_points:",
        line_move_b["mean_points"],
        "probability_positive:",
        line_move_b["season_block_probability_positive"],
        "decisive:",
        line_move_b["decisive_games"],
        line_move_b["decisive_side_a_wins"],
        line_move_b["decisive_side_b_wins"],
    )
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
