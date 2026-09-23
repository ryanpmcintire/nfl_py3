from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_ats.availability import (
    availability_rate_lookup,
    build_season_lagged_availability_rates,
    fixed_unavailability,
)
from nfl_ats.availability import position_group as availability_position_group
from nfl_ats.availability import practice_category as availability_practice_category
from nfl_ats.availability import report_category as availability_report_category

REPO_ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = REPO_ROOT / "artifacts" / "margin_predictions" / "2026-week-03-20260923T161033Z"
INJURY_SNAPSHOT = (
    REPO_ROOT / "data" / "raw" / "nflverse_injuries" / "20260923T203033Z" / "injuries.parquet"
)
PLAYER_VALUE_TABLE = REPO_ROOT / "data" / "processed" / "game_features_player_value.parquet"
INJURY_PLAY_OUTCOMES_TABLE = REPO_ROOT / "data" / "processed" / "injury_play_outcomes.parquet"
TARGET_SEASON = 2026
SPARSE_CELL_FLOOR = 250
SIGNAL_REPORT_CATEGORIES = frozenset(("out", "doubtful", "questionable"))
SIGNAL_PRACTICE_CATEGORIES = frozenset(("dnp", "limited"))
MAX_ENUMERATED_PLAYERS_PER_SIDE = 12
FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF = -0.320
UNIT_COUPLING_MULTIPLIER = {
    "offensive_line": 1.192,
    "skill": 1.235,
    "front": 1.175,
    "secondary": 1.245,
}
_OFFENSIVE_LINE = frozenset(("C", "G", "OG", "OL", "OT", "T"))
_SKILL = frozenset(("FB", "HB", "QB", "RB", "TE", "WR"))
_FRONT = frozenset(("DE", "DL", "DT", "EDGE", "ILB", "LB", "NT", "OLB"))
_SECONDARY = frozenset(("CB", "DB", "FS", "S", "SAF", "SS"))
LINE_SWEEP_MIN_OFFSET = -4.0
LINE_SWEEP_MAX_OFFSET = 4.0


def position_group(position: str) -> str:
    normalized = str(position).strip().upper()
    if normalized in _OFFENSIVE_LINE:
        return "offensive_line"
    if normalized in _SKILL:
        return "skill"
    if normalized in _FRONT:
        return "front"
    if normalized in _SECONDARY:
        return "secondary"
    return "other"


def build_availability_lookup(
    target_season: int,
) -> tuple[dict[tuple[int, str, str, str], float], dict[tuple[int, str, str, str], int]]:
    outcomes = pd.read_parquet(INJURY_PLAY_OUTCOMES_TABLE)
    rates = build_season_lagged_availability_rates(outcomes, target_seasons=[target_season])
    lookup = availability_rate_lookup(rates)
    observations = {
        (
            int(row.target_season),
            str(row.report_category),
            str(row.practice_category),
            str(row.position_group),
        ): int(row.observations)
        for row in rates.itertuples(index=False)
    }
    return lookup, observations


def resolve_severity(
    lookup: dict[tuple[int, str, str, str], float],
    observations: dict[tuple[int, str, str, str], int],
    *,
    target_season: int,
    report_status: object,
    practice_status: object,
    position: object,
    sparse_cell_floor: int,
) -> tuple[float, str]:
    report = availability_report_category(report_status)
    practice = availability_practice_category(practice_status)
    group = availability_position_group(position)
    for key in (
        (target_season, report, practice, group),
        (target_season, report, practice, "__all__"),
    ):
        if key in lookup and observations.get(key, 0) >= sparse_cell_floor:
            return lookup[key], "season_lagged_rate"
    return fixed_unavailability(report_status, practice_status), "fixed_status_prior"


def borderline_players(
    injuries: pd.DataFrame,
    team: str,
    lookup: dict[tuple[int, str, str, str], float],
    observations: dict[tuple[int, str, str, str], int],
    target_season: int,
    sparse_cell_floor: int = SPARSE_CELL_FLOOR,
) -> list[dict]:
    rows = injuries.loc[injuries["team"].astype(str).eq(team)]
    players = []
    for _, row in rows.iterrows():
        report_cat = availability_report_category(row.get("report_status"))
        practice_cat = availability_practice_category(row["practice_status"])
        if (
            report_cat not in SIGNAL_REPORT_CATEGORIES
            and practice_cat not in SIGNAL_PRACTICE_CATEGORIES
        ):
            continue
        severity, source = resolve_severity(
            lookup,
            observations,
            target_season=target_season,
            report_status=row.get("report_status"),
            practice_status=row["practice_status"],
            position=row["position"],
            sparse_cell_floor=sparse_cell_floor,
        )
        if severity <= 0.0:
            continue
        players.append(
            {
                "gsis_id": str(row["gsis_id"]),
                "full_name": str(row["full_name"]),
                "position": str(row["position"]),
                "practice_status": str(row["practice_status"]),
                "unit": position_group(row["position"]),
                "sit_probability": severity,
                "severity_source": source,
            }
        )
    return players


def side_subsets(players: list[dict]) -> tuple[list[dict], bool]:
    if not players:
        return [{"sit_ids": frozenset(), "probability": 1.0, "sit_fraction": 0.0}], False
    ordered = sorted(players, key=lambda player: player["sit_probability"], reverse=True)
    truncated = len(ordered) > MAX_ENUMERATED_PLAYERS_PER_SIDE
    kept = ordered[:MAX_ENUMERATED_PLAYERS_PER_SIDE]
    remainder = ordered[MAX_ENUMERATED_PLAYERS_PER_SIDE:]
    expected_remainder_severity = sum(player["sit_probability"] for player in remainder)
    total_severity = sum(player["sit_probability"] for player in players)
    subsets = []
    for mask in itertools.product((False, True), repeat=len(kept)):
        sitting = [player for player, sit in zip(kept, mask, strict=True) if sit]
        independent_probability = 1.0
        for player, sit in zip(kept, mask, strict=True):
            independent_probability *= (
                player["sit_probability"] if sit else 1.0 - player["sit_probability"]
            )
        coupled_probability = independent_probability
        for left, right in itertools.combinations(sitting, 2):
            if left["unit"] == right["unit"] and left["unit"] in UNIT_COUPLING_MULTIPLIER:
                coupled_probability *= UNIT_COUPLING_MULTIPLIER[left["unit"]]
        sit_severity = sum(player["sit_probability"] for player in sitting)
        sit_severity += expected_remainder_severity
        sit_fraction = sit_severity / total_severity if total_severity > 0 else 0.0
        subsets.append(
            {
                "sit_ids": frozenset(player["gsis_id"] for player in sitting),
                "probability": coupled_probability,
                "sit_fraction": sit_fraction,
            }
        )
    normalizer = sum(subset["probability"] for subset in subsets)
    for subset in subsets:
        subset["probability"] = subset["probability"] / normalizer
    return subsets, truncated


def interpolate_cover_probability(sweep_game: pd.DataFrame, offset: float) -> tuple[float, bool]:
    clamped = min(max(offset, LINE_SWEEP_MIN_OFFSET), LINE_SWEEP_MAX_OFFSET)
    was_clamped = not (LINE_SWEEP_MIN_OFFSET <= offset <= LINE_SWEEP_MAX_OFFSET)
    ordered = sweep_game.sort_values("line_offset")
    offsets = ordered["line_offset"].to_numpy()
    probabilities = ordered["home_cover_probability"].to_numpy()
    interpolated = float(
        pd.Series(probabilities, index=offsets)
        .reindex(sorted({*offsets.tolist(), clamped}))
        .interpolate(method="index")
        .loc[clamped]
    )
    return interpolated, was_clamped


def main() -> None:
    recommendations = pd.read_csv(FORECAST_DIR / "recommendations.csv")
    sweep = pd.read_parquet(FORECAST_DIR / "line_sweep.parquet")
    sweep = sweep.loc[sweep["method"].eq("market_residual")].copy()
    injuries = pd.read_parquet(INJURY_SNAPSHOT)
    injuries = injuries.loc[injuries["season"].eq(2026) & injuries["week"].eq(3)].copy()
    lookup, observations = build_availability_lookup(TARGET_SEASON)

    player_value = pd.read_parquet(PLAYER_VALUE_TABLE)
    scoped_games = int(
        (
            player_value["home_injury_observed_at"].notna()
            & player_value["away_injury_observed_at"].notna()
        ).sum()
    )
    scoped_seasons = sorted(
        player_value.loc[
            player_value["home_injury_observed_at"].notna()
            & player_value["away_injury_observed_at"].notna(),
            "season",
        ]
        .unique()
        .tolist()
    )

    rows = []
    for _, game in recommendations.iterrows():
        game_id = str(game["game_id"])
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        spread_line = float(game["spread_line"])
        served_probability = float(game["home_cover_probability"])
        base_value_lost_diff = float(
            game["diff_injury_skill_epa_value_lost"]
            + game["diff_injury_defense_disruption_value_lost"]
        )
        base_home_value_lost = float(
            game["home_injury_skill_epa_value_lost"]
            + game["home_injury_defense_disruption_value_lost"]
        )
        base_away_value_lost = float(
            game["away_injury_skill_epa_value_lost"]
            + game["away_injury_defense_disruption_value_lost"]
        )

        home_players = borderline_players(
            injuries, home_team, lookup, observations, TARGET_SEASON
        )
        away_players = borderline_players(
            injuries, away_team, lookup, observations, TARGET_SEASON
        )
        home_subsets, home_truncated = side_subsets(home_players)
        away_subsets, away_truncated = side_subsets(away_players)
        side_truncated = home_truncated or away_truncated
        sweep_game = sweep.loc[sweep["game_id"].eq(game_id)]

        mixed_probability = 0.0
        clamped_any = False
        scenario_count = 0
        for home_subset, away_subset in itertools.product(home_subsets, away_subsets):
            scenario_probability = home_subset["probability"] * away_subset["probability"]
            scenario_home_value = base_home_value_lost * home_subset["sit_fraction"]
            scenario_away_value = base_away_value_lost * away_subset["sit_fraction"]
            scenario_value_lost_diff = scenario_home_value - scenario_away_value
            margin_shift = FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF * (
                scenario_value_lost_diff - base_value_lost_diff
            )
            if sweep_game.empty:
                scenario_cover_probability = served_probability
            else:
                scenario_cover_probability, was_clamped = interpolate_cover_probability(
                    sweep_game, -margin_shift
                )
                clamped_any = clamped_any or was_clamped
            mixed_probability += scenario_probability * scenario_cover_probability
            scenario_count += 1

        shift_points = (mixed_probability - served_probability) * 100.0
        served_side = "home" if served_probability >= 0.5 else "away"
        scenario_side = "home" if mixed_probability >= 0.5 else "away"
        rows.append(
            {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "spread_line": spread_line,
                "n_borderline_home": len(home_players),
                "n_borderline_away": len(away_players),
                "borderline_home_players": ", ".join(
                    f"{player['full_name']}({player['position']},{player['practice_status']})"
                    for player in home_players
                ),
                "borderline_away_players": ", ".join(
                    f"{player['full_name']}({player['position']},{player['practice_status']})"
                    for player in away_players
                ),
                "base_value_lost_diff": base_value_lost_diff,
                "scenario_count": scenario_count,
                "served_home_cover_probability": served_probability,
                "scenario_mixed_home_cover_probability": mixed_probability,
                "shift_points": shift_points,
                "served_pick_side": served_side,
                "scenario_pick_side": scenario_side,
                "side_differs": served_side != scenario_side,
                "line_offset_clamped": clamped_any,
                "side_truncated_at_cap": side_truncated,
            }
        )

    result = pd.DataFrame(rows).sort_values("game_id").reset_index(drop=True)
    all_borderline_players = [
        player
        for _, game in recommendations.iterrows()
        for player in borderline_players(
            injuries, str(game["home_team"]), lookup, observations, TARGET_SEASON
        )
        + borderline_players(
            injuries, str(game["away_team"]), lookup, observations, TARGET_SEASON
        )
    ]
    severity_source_counts = {
        source: sum(1 for player in all_borderline_players if player["severity_source"] == source)
        for source in ("season_lagged_rate", "fixed_status_prior")
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = REPO_ROOT / "artifacts" / "injury_scenario_producer" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "week3_scenario_comparison.csv", index=False)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_forecast": str(FORECAST_DIR),
        "source_injury_snapshot": str(INJURY_SNAPSHOT),
        "fitted_slope_points_per_unit_value_lost_diff": FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF,
        "fitted_slope_source": (
            "docs/lanes/injury-scenario-producer.md Unit 2 result, pooled block-bootstrap "
            "slope, unresolved_below_power on practical OOS improvement"
        ),
        "unit_coupling_multiplier": UNIT_COUPLING_MULTIPLIER,
        "coupling_source": (
            "docs/absence_pairwise_dependence.md section 8, "
            "observed/permuted-null excess ratios"
        ),
        "severity_source": "nfl_ats.availability.resolve_unavailability season-lagged rates "
        "fit on data/processed/injury_play_outcomes.parquet, sparse cells "
        f"(observations < {SPARSE_CELL_FLOOR}) fall back to fixed_unavailability",
        "severity_source_counts": severity_source_counts,
        "sparse_cell_floor_observations": SPARSE_CELL_FLOOR,
        "games": len(result),
        "games_with_any_borderline_player": int(
            ((result["n_borderline_home"] > 0) | (result["n_borderline_away"] > 0)).sum()
        ),
        "games_with_shift_over_1_point": int((result["shift_points"].abs() > 1.0).sum()),
        "games_with_side_difference": int(result["side_differs"].sum()),
        "games_with_side_truncated_at_cap": int(result["side_truncated_at_cap"].sum()),
        "max_enumerated_players_per_side": MAX_ENUMERATED_PLAYERS_PER_SIDE,
        "historical_games_with_pregame_attested_injury_inputs": scoped_games,
        "historical_games_seasons": scoped_seasons,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(result.to_string())


if __name__ == "__main__":
    main()
