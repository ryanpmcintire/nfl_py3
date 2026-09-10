from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.backup_qb_fade_overlay import apply_backup_qb_fade_overlay
from nfl_ats.bye_edge_fade_overlay import apply_bye_edge_fade_overlay
from nfl_ats.coach_fade_overlay import apply_coach_fade_overlay
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
from nfl_ats.forecast_cold_visitor_tilt_overlay import apply_forecast_cold_visitor_tilt_overlay
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (
    apply_precip_high_total_tilt_overlay,
)
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (
    apply_warm_team_cold_late_tilt_overlay,
)
from nfl_ats.four_overlay_composition import (
    BYE_EDGE_FADE,
    COACH_FADE,
    COMPOSITION_ORDER,
    DIVISION_REVENGE_TILT,
    FORECAST_COLD_VISITOR_TILT,
    INTERIM_HC_FIRST_GAME_TILT,
    PBP08_PROTECTION_MISMATCH_TILT,
    PLAYER_ARRESTS_BACK_SIDE_POLICY,
    POLICY_ID,
    PRECIP_HIGH_TOTAL_TILT,
    TANK_ZONE_FADE_TILT,
)
from nfl_ats.injury_value_tilt_overlay import apply_injury_value_tilt_overlay
from nfl_ats.interim_hc_first_game_tilt_overlay import apply_interim_hc_first_game_tilt_overlay
from nfl_ats.overlay_composition import (
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    blocked_bootstrap_matrix,
    build_delta_matrix,
    build_predictions_frame,
    load_inputs,
    reconstruct_arrest_flip_set,
)
from nfl_ats.pace_mismatch_dog_tilt_overlay import (
    apply_pace_mismatch_dog_tilt_overlay,
    pace_mismatch_flags_fail_open,
)
from nfl_ats.pbp import latest_pbp_snapshot as latest_pbp_snapshot_meta
from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp08_matchup_flags import build_flag_table
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    SCREEN_SEASON_START,
    apply_pbp08_protection_mismatch_tilt,
)
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    latest_pbp_snapshot as latest_pbp08_snapshot_dir,
)
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    latest_schedules as latest_pbp08_schedules,
)
from nfl_ats.provenance import sha256_file, write_stamped_artifact
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.special_teams_return_tilt_overlay import apply_special_teams_return_tilt_overlay
from nfl_ats.spread_gap_zone_fade_overlay import apply_spread_gap_zone_fade_overlay
from nfl_ats.surface_switch_tilt_overlay import apply_surface_switch_tilt_overlay
from nfl_ats.tank_zone_fade_tilt_overlay import apply_tank_zone_fade_tilt_overlay
from nfl_ats.third_down_reversion_fade_overlay import apply_third_down_reversion_fade_overlay
from nfl_ats.turnover_luck_rebound_tilt_overlay import apply_turnover_luck_rebound_tilt_overlay

DEFAULT_OUTPUT_ROOT = Path("artifacts/unserved_tilt_marginals")
DEFAULT_SAMPLES = 20_000
DEFAULT_SEED = 20260821
CONFIDENCE = 0.95

TUESDAY_NOON_ARCHIVE = Path("raw/forecast_archive/full_2020_2025/forecasts.parquet")
KICKOFF_NEAREST_ARCHIVE = Path("raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet")

SERVED_MEMBERS: tuple[str, ...] = (
    "coach_fade",
    "division_revenge_tilt",
    "player_arrests_back_side_policy",
)

SERVED_CARD_MEMBERS: tuple[str, ...] = COMPOSITION_ORDER

CARD_CHOICES: tuple[str, ...] = ("served", "three")

CANDIDATE_MEMBERS: tuple[str, ...] = (
    "tank_zone_fade_tilt_overlay",
    "forecast_cold_visitor_tilt_overlay",
    "forecast_weather_kn_warm_team_cold_late_tilt_overlay",
    "forecast_weather_kn_precip_high_total_tilt_overlay",
    "special_teams_return_tilt_overlay",
    "pace_mismatch_dog_tilt_overlay",
    "turnover_luck_rebound_tilt_overlay",
    "interim_hc_first_game_tilt_overlay",
    "bye_edge_fade_overlay",
    "pbp08_protection_mismatch_tilt_overlay",
    "third_down_reversion_fade_overlay",
    "injury_value_tilt_overlay",
    "surface_switch_tilt_overlay",
    "backup_qb_fade_overlay",
    "spread_gap_zone_fade_overlay",
)

ALREADY_ENUMERATED: frozenset[str] = frozenset(
    {
        "injury_value_tilt_overlay",
        "surface_switch_tilt_overlay",
        "backup_qb_fade_overlay",
        "spread_gap_zone_fade_overlay",
    }
)


def _flip_ids(result: Any) -> set[str]:
    return {str(flip.game_id) for flip in result.flips}


def _verify_complement(
    predictions: pd.DataFrame, name: str, result: Any, flip_ids: set[str]
) -> None:

    if not flip_ids:
        return
    baseline = predictions.set_index("game_id")["home_cover_probability"]
    overlaid = result.overlaid_predictions.set_index("game_id")["home_cover_probability"]
    ids = sorted(flip_ids)
    actual = overlaid.loc[ids].to_numpy(dtype=float)
    expected = 1.0 - baseline.loc[ids].to_numpy(dtype=float)
    if not np.allclose(actual, expected, atol=1e-9):
        raise AssertionError(f"{name} did not complement the baseline pick; OR union is undefined")


def build_pbp08_flag_table(data_root: Path) -> tuple[pd.DataFrame, str]:

    snapshot = latest_pbp08_snapshot_dir(data_root)
    schedules_path = latest_pbp08_schedules(data_root)
    if snapshot is None or schedules_path is None:
        raise FileNotFoundError(
            f"missing pbp snapshot ({snapshot}) or schedules ({schedules_path})"
        )
    schedule = pd.read_parquet(schedules_path)
    schedule = schedule.loc[schedule["game_type"].astype(str).eq("REG")]
    schedule = schedule.loc[schedule["season"].ge(SCREEN_SEASON_START)].copy()
    return build_flag_table(schedule, snapshot), snapshot.name


def build_candidate_flip_sets(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    player_features: pd.DataFrame,
    data_root: Path,
    repo_root: Path,
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:

    archive_ids = set(predictions["game_id"].astype(str))
    notes: dict[str, dict[str, Any]] = {}
    flips: dict[str, set[str]] = {}

    total_lines = schedules[["game_id", "total_line"]].drop_duplicates("game_id")
    with_total = predictions.merge(total_lines, on="game_id", how="left", validate="one_to_one")

    forecasts_tue = pd.read_parquet(data_root / TUESDAY_NOON_ARCHIVE)
    forecasts_kn = pd.read_parquet(data_root / KICKOFF_NEAREST_ARCHIVE)

    pace_flags = pace_mismatch_flags_fail_open(data_root)
    pbp08_flags, pbp08_snapshot = build_pbp08_flag_table(data_root)
    pbp08_flags = pbp08_flags.loc[pbp08_flags["game_id"].astype(str).isin(archive_ids)]

    pbp_snapshot = latest_pbp_snapshot_meta(data_root / "pbp" / "raw")
    pbp = load_pbp_snapshot(pbp_snapshot)

    def register(name: str, result: Any, source: dict[str, Any], frame: pd.DataFrame) -> None:
        ids = _flip_ids(result)
        _verify_complement(frame, name, result, ids)
        flips[name] = ids
        notes[name] = source

    register(
        "tank_zone_fade_tilt_overlay",
        apply_tank_zone_fade_tilt_overlay(predictions, schedules),
        {"inputs": ["schedules"], "registered_window": "REG weeks 14-18 only"},
        predictions,
    )
    register(
        "forecast_cold_visitor_tilt_overlay",
        apply_forecast_cold_visitor_tilt_overlay(predictions, schedules, forecasts_tue),
        {
            "inputs": ["schedules", "forecast_archive"],
            "forecast_archive": str(TUESDAY_NOON_ARCHIVE),
            "cutoff_mode": "tuesday_noon",
            "mos_model": "MEX",
            "archive_seasons": [2020, 2025],
        },
        predictions,
    )
    register(
        "forecast_weather_kn_warm_team_cold_late_tilt_overlay",
        apply_warm_team_cold_late_tilt_overlay(predictions, schedules, forecasts_kn),
        {
            "inputs": ["schedules", "forecast_archive"],
            "forecast_archive": str(KICKOFF_NEAREST_ARCHIVE),
            "cutoff_mode": "kickoff_nearest",
            "mos_model": "GFS",
            "archive_seasons": [2009, 2025],
        },
        predictions,
    )
    register(
        "forecast_weather_kn_precip_high_total_tilt_overlay",
        apply_precip_high_total_tilt_overlay(with_total, schedules, forecasts_kn),
        {
            "inputs": ["schedules", "forecast_archive", "schedule total_line"],
            "forecast_archive": str(KICKOFF_NEAREST_ARCHIVE),
            "cutoff_mode": "kickoff_nearest",
            "mos_model": "GFS",
            "total_line_caveat": (
                "total_line comes from the nflverse schedule snapshot (a settled line), not "
                "the point-in-time Tuesday total the live overlay reads off the card"
            ),
        },
        with_total,
    )
    register(
        "special_teams_return_tilt_overlay",
        apply_special_teams_return_tilt_overlay(predictions, schedules, data_root),
        {
            "inputs": ["schedules", "special_teams team_season snapshot"],
            "team_season_root": str(data_root / "raw" / "special_teams"),
        },
        predictions,
    )
    register(
        "pace_mismatch_dog_tilt_overlay",
        apply_pace_mismatch_dog_tilt_overlay(predictions, pace_flags),
        {
            "inputs": ["pace_mismatch_flags_fail_open(data_root)"],
            "flag_rows": len(pace_flags),
        },
        predictions,
    )
    register(
        "turnover_luck_rebound_tilt_overlay",
        apply_turnover_luck_rebound_tilt_overlay(predictions, schedules, pbp),
        {"inputs": ["schedules", "pbp"], "pbp_snapshot": pbp_snapshot.snapshot_id},
        predictions,
    )
    register(
        "interim_hc_first_game_tilt_overlay",
        apply_interim_hc_first_game_tilt_overlay(predictions, repo_root),
        {"inputs": ["repo_root interim-coach snapshot"]},
        predictions,
    )
    register(
        "bye_edge_fade_overlay",
        apply_bye_edge_fade_overlay(predictions, schedules),
        {"inputs": ["schedules"]},
        predictions,
    )
    register(
        "pbp08_protection_mismatch_tilt_overlay",
        apply_pbp08_protection_mismatch_tilt(predictions, pbp08_flags),
        {
            "inputs": ["pbp08 flag table"],
            "pbp_snapshot": pbp08_snapshot,
            "flag_build": (
                "build_flag_table over the full REG history from "
                f"{SCREEN_SEASON_START}, filtered to the archive games; the quartile "
                "thresholds expand over strictly-earlier week blocks, so one build matches "
                "the per-week flags_for_week_fail_open path"
            ),
        },
        predictions,
    )
    register(
        "third_down_reversion_fade_overlay",
        apply_third_down_reversion_fade_overlay(predictions, schedules, pbp),
        {"inputs": ["schedules", "pbp"], "pbp_snapshot": pbp_snapshot.snapshot_id},
        predictions,
    )
    register(
        "injury_value_tilt_overlay",
        apply_injury_value_tilt_overlay(predictions, player_features),
        {"inputs": ["game_features_player value-lost differentials"]},
        predictions,
    )
    register(
        "surface_switch_tilt_overlay",
        apply_surface_switch_tilt_overlay(predictions, schedules),
        {"inputs": ["schedules"]},
        predictions,
    )
    register(
        "backup_qb_fade_overlay",
        apply_backup_qb_fade_overlay(predictions, schedules),
        {"inputs": ["schedules"]},
        predictions,
    )
    register(
        "spread_gap_zone_fade_overlay",
        apply_spread_gap_zone_fade_overlay(predictions),
        {"inputs": ["prediction spread_line only"]},
        predictions,
    )
    return flips, notes


def build_served_flip_set(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    per_game: pd.DataFrame,
    features: Path,
    incidents: Path,
) -> tuple[set[str], dict[str, set[str]]]:

    coach = apply_coach_fade_overlay(predictions, schedules, enabled=True)
    division = apply_division_revenge_tilt_overlay(predictions, schedules, enabled=True)
    arrest_ids, _scored = reconstruct_arrest_flip_set(per_game, features, incidents)
    members = {
        "coach_fade": _flip_ids(coach),
        "division_revenge_tilt": _flip_ids(division),
        "player_arrests_back_side_policy": {str(game_id) for game_id in arrest_ids},
    }
    _verify_complement(predictions, "coach_fade", coach, members["coach_fade"])
    _verify_complement(
        predictions, "division_revenge_tilt", division, members["division_revenge_tilt"]
    )
    union: set[str] = set()
    for ids in members.values():
        union |= ids
    return union, members


def build_served_card_flip_sets(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    per_game: pd.DataFrame,
    data_root: Path,
    repo_root: Path,
    features: Path,
    incidents: Path,
) -> dict[str, set[str]]:

    archive_ids = set(predictions["game_id"].astype(str))
    total_lines = schedules[["game_id", "total_line"]].drop_duplicates("game_id")
    with_total = predictions.merge(total_lines, on="game_id", how="left", validate="one_to_one")
    forecasts_tue = pd.read_parquet(data_root / TUESDAY_NOON_ARCHIVE)
    forecasts_kn = pd.read_parquet(data_root / KICKOFF_NEAREST_ARCHIVE)
    pbp08_flags, _snapshot = build_pbp08_flag_table(data_root)
    pbp08_flags = pbp08_flags.loc[pbp08_flags["game_id"].astype(str).isin(archive_ids)]

    members: dict[str, set[str]] = {}

    def register(name: str, result: Any, frame: pd.DataFrame) -> None:
        ids = _flip_ids(result)
        _verify_complement(frame, name, result, ids)
        members[name] = ids

    register(
        COACH_FADE,
        apply_coach_fade_overlay(predictions, schedules, enabled=True),
        predictions,
    )
    register(
        DIVISION_REVENGE_TILT,
        apply_division_revenge_tilt_overlay(predictions, schedules, enabled=True),
        predictions,
    )
    arrest_ids, _scored = reconstruct_arrest_flip_set(per_game, features, incidents)
    members[PLAYER_ARRESTS_BACK_SIDE_POLICY] = {str(game_id) for game_id in arrest_ids}
    register(
        BYE_EDGE_FADE,
        apply_bye_edge_fade_overlay(predictions, schedules),
        predictions,
    )
    register(
        FORECAST_COLD_VISITOR_TILT,
        apply_forecast_cold_visitor_tilt_overlay(predictions, schedules, forecasts_tue),
        predictions,
    )
    register(
        PBP08_PROTECTION_MISMATCH_TILT,
        apply_pbp08_protection_mismatch_tilt(predictions, pbp08_flags),
        predictions,
    )
    register(
        INTERIM_HC_FIRST_GAME_TILT,
        apply_interim_hc_first_game_tilt_overlay(predictions, repo_root),
        predictions,
    )
    register(
        TANK_ZONE_FADE_TILT,
        apply_tank_zone_fade_tilt_overlay(predictions, schedules),
        predictions,
    )
    register(
        PRECIP_HIGH_TOTAL_TILT,
        apply_precip_high_total_tilt_overlay(with_total, schedules, forecasts_kn),
        with_total,
    )
    missing = [name for name in SERVED_CARD_MEMBERS if name not in members]
    if missing:
        raise ValueError(f"served policy {POLICY_ID} members not built: {', '.join(missing)}")
    return {name: members[name] for name in SERVED_CARD_MEMBERS}


def served_card_flip_set(
    per_game: pd.DataFrame,
    *,
    data_root: Path,
    repo_root: Path,
    features: Path = DEFAULT_FEATURES,
    incidents: Path = DEFAULT_INCIDENTS,
    schedules: pd.DataFrame | None = None,
    card: str = "served",
) -> tuple[set[str], dict[str, set[str]]]:

    if card not in CARD_CHOICES:
        raise ValueError(f"card must be one of {CARD_CHOICES}, got {card!r}")
    if schedules is None:
        snapshot = latest_snapshot(data_root / "raw")
        schedules, _team_stats = load_snapshot(snapshot)
    predictions = build_predictions_frame(per_game, schedules)
    if card == "three":
        return build_served_flip_set(predictions, schedules, per_game, features, incidents)
    members = build_served_card_flip_sets(
        predictions, schedules, per_game, data_root, repo_root, features, incidents
    )
    union: set[str] = set()
    for ids in members.values():
        union |= ids
    return union, members


def _season_rows(
    eval_frame: pd.DataFrame, valid_mask: np.ndarray, deltas: np.ndarray, column: int
) -> list[dict[str, Any]]:
    frame = eval_frame.loc[valid_mask, ["season"]].reset_index(drop=True).copy()
    frame["delta"] = deltas[:, column]
    rows: list[dict[str, Any]] = []
    for season, group in frame.groupby("season", sort=True):
        rows.append(
            {
                "season": int(str(season)),
                "games": len(group),
                "delta_accuracy_points": float(group["delta"].mean() * 100.0),
                "incremental_flips": int(np.count_nonzero(group["delta"].to_numpy() != 0.0)),
            }
        )
    return rows


def run_unserved_tilt_marginals(
    *,
    per_game_artifact: Path,
    data_root: Path,
    repo_root: Path,
    features: Path,
    incidents: Path = DEFAULT_INCIDENTS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    card: str = "served",
) -> dict[str, Any]:

    started = perf_counter()
    if card not in CARD_CHOICES:
        raise ValueError(f"card must be one of {CARD_CHOICES}, got {card!r}")
    per_game_artifact = per_game_artifact.resolve()
    metadata = json.loads(per_game_artifact.with_name("metadata.json").read_text(encoding="utf-8"))
    per_game, schedules, player_features, snapshot_name, player_feature_path = load_inputs(
        per_game_artifact, data_root
    )
    predictions = build_predictions_frame(per_game, schedules)
    if card == "three":
        served_ids, served_members = build_served_flip_set(
            predictions, schedules, per_game, features, incidents
        )
    else:
        served_members = build_served_card_flip_sets(
            predictions, schedules, per_game, data_root, repo_root, features, incidents
        )
        served_ids = set()
        for ids in served_members.values():
            served_ids |= ids
    candidate_flips, candidate_notes = build_candidate_flip_sets(
        predictions, schedules, player_features, data_root, repo_root
    )

    eval_frame = predictions[["game_id", "season", "week"]].merge(
        per_game[["game_id", "correct_at_open_probability_rule"]], on="game_id", how="left"
    )
    eval_frame["correct_raw"] = pd.to_numeric(
        eval_frame["correct_at_open_probability_rule"], errors="coerce"
    )
    game_ids = eval_frame["game_id"].astype(str)
    served_mask = game_ids.isin(served_ids)
    eval_frame["correct_served"] = eval_frame["correct_raw"].where(
        ~served_mask, 1.0 - eval_frame["correct_raw"]
    )

    valid_mask = eval_frame["correct_raw"].notna().to_numpy()
    valid_blocks = eval_frame.loc[valid_mask, ["season", "week"]].reset_index(drop=True)

    incremental = {name: ids - served_ids for name, ids in candidate_flips.items()}
    subsets: list[tuple[str, ...]] = [(name,) for name in CANDIDATE_MEMBERS]
    full_matrix = build_delta_matrix(
        eval_frame["correct_served"], game_ids, incremental, CANDIDATE_MEMBERS, subsets
    )
    deltas = full_matrix[valid_mask]
    week_stats = blocked_bootstrap_matrix(
        deltas, valid_blocks, block="week", samples=samples, seed=seed
    )
    season_stats = blocked_bootstrap_matrix(
        deltas, valid_blocks, block="season", samples=samples, seed=seed
    )

    raw_valid = eval_frame.loc[valid_mask, "correct_raw"].to_numpy(dtype=float)
    served_valid = eval_frame.loc[valid_mask, "correct_served"].to_numpy(dtype=float)
    raw_accuracy = float(raw_valid.mean())
    served_accuracy = float(served_valid.mean())
    archive_seasons = sorted(int(season) for season in eval_frame["season"].dropna().unique())

    member_records: list[dict[str, Any]] = []
    for column, name in enumerate(CANDIDATE_MEMBERS):
        ids = candidate_flips[name]
        new_ids = incremental[name]
        seasons_hit = sorted(
            {int(season) for season in eval_frame.loc[game_ids.isin(ids), "season"].dropna()}
        )
        member_records.append(
            {
                "member": name,
                "already_enumerated_by_overlay_composition": name in ALREADY_ENUMERATED,
                "inputs": candidate_notes[name],
                "flip_count_on_archive": len(ids),
                "flips_overlapping_served_card": len(ids & served_ids),
                "incremental_flip_count": len(new_ids),
                "incremental_flips_on_scored_games": int(
                    np.count_nonzero(deltas[:, column] != 0.0)
                ),
                "seasons_with_a_flip": seasons_hit,
                "no_op_seasons": [s for s in archive_seasons if s not in set(seasons_hit)],
                "served_card_accuracy": served_accuracy,
                "candidate_accuracy": float((served_valid + deltas[:, column]).mean()),
                "delta_estimate_accuracy_points": float(deltas[:, column].mean() * 100.0),
                "week_blocked": {
                    "estimate_accuracy_points": float(week_stats["estimate"][column] * 100.0),
                    "lower_accuracy_points": float(week_stats["lower"][column] * 100.0),
                    "upper_accuracy_points": float(week_stats["upper"][column] * 100.0),
                    "probability_positive": float(week_stats["probability_positive"][column]),
                    "standard_error_accuracy_points": float(
                        week_stats["standard_error"][column] * 100.0
                    ),
                    "block": "week",
                    "blocks": int(week_stats["block_count"]),
                    "bootstrap_samples": samples,
                    "confidence": CONFIDENCE,
                },
                "season_blocked": {
                    "estimate_accuracy_points": float(season_stats["estimate"][column] * 100.0),
                    "lower_accuracy_points": float(season_stats["lower"][column] * 100.0),
                    "upper_accuracy_points": float(season_stats["upper"][column] * 100.0),
                    "probability_positive": float(season_stats["probability_positive"][column]),
                    "standard_error_accuracy_points": float(
                        season_stats["standard_error"][column] * 100.0
                    ),
                    "block": "season",
                    "blocks": int(season_stats["block_count"]),
                    "bootstrap_samples": samples,
                    "confidence": CONFIDENCE,
                },
                "per_season": _season_rows(eval_frame, valid_mask, deltas, column),
            }
        )

    positive = tuple(
        str(record["member"])
        for record in member_records
        if record["week_blocked"]["probability_positive"] > 0.5
    )
    combo_names: list[str] = []
    combo_sets: dict[str, set[str]] = {}
    combo_members: dict[str, list[str]] = {}
    if positive:
        union: set[str] = set()
        for name in positive:
            union |= incremental[name]
        combo_sets["all_probability_positive_above_half"] = union
        combo_members["all_probability_positive_above_half"] = list(positive)
        combo_names.append("all_probability_positive_above_half")

    chosen: list[str] = []
    remaining = list(CANDIDATE_MEMBERS)
    greedy_steps: list[dict[str, Any]] = []
    while remaining:
        best_name = ""
        best_delta = -np.inf
        best_column = np.zeros(len(served_valid), dtype=float)
        for name in remaining:
            trial_ids: set[str] = set()
            for picked in [*chosen, name]:
                trial_ids |= incremental[picked]
            flipped = game_ids.isin(trial_ids).to_numpy()[valid_mask]
            candidate = np.where(flipped, 1.0 - served_valid, served_valid)
            delta = float((candidate - served_valid).mean())
            if delta > best_delta:
                best_delta = delta
                best_name = name
                best_column = candidate - served_valid
        chosen.append(best_name)
        remaining.remove(best_name)
        greedy_steps.append(
            {
                "step": len(greedy_steps) + 1,
                "added": best_name,
                "members_so_far": list(chosen),
                "delta_vs_served_accuracy_points": float(best_column.mean() * 100.0),
                "cumulative_accuracy": float((served_valid + best_column).mean()),
            }
        )
        key = f"greedy_step_{len(greedy_steps):02d}"
        step_ids: set[str] = set()
        for picked in chosen:
            step_ids |= incremental[picked]
        combo_sets[key] = step_ids
        combo_members[key] = list(chosen)
        combo_names.append(key)

    combo_subsets: list[tuple[str, ...]] = [(name,) for name in combo_names]
    combo_matrix = build_delta_matrix(
        eval_frame["correct_served"], game_ids, combo_sets, tuple(combo_names), combo_subsets
    )
    combo_deltas = combo_matrix[valid_mask]
    combo_week = blocked_bootstrap_matrix(
        combo_deltas, valid_blocks, block="week", samples=samples, seed=seed
    )
    combo_season = blocked_bootstrap_matrix(
        combo_deltas, valid_blocks, block="season", samples=samples, seed=seed
    )
    combo_records: list[dict[str, Any]] = []
    for column, name in enumerate(combo_names):
        combo_records.append(
            {
                "label": name,
                "members": combo_members[name],
                "incremental_flip_count": len(combo_sets[name]),
                "incremental_flips_on_scored_games": int(
                    np.count_nonzero(combo_deltas[:, column] != 0.0)
                ),
                "served_card_accuracy": served_accuracy,
                "candidate_accuracy": float((served_valid + combo_deltas[:, column]).mean()),
                "delta_estimate_accuracy_points": float(combo_deltas[:, column].mean() * 100.0),
                "week_blocked": {
                    "estimate_accuracy_points": float(combo_week["estimate"][column] * 100.0),
                    "lower_accuracy_points": float(combo_week["lower"][column] * 100.0),
                    "upper_accuracy_points": float(combo_week["upper"][column] * 100.0),
                    "probability_positive": float(combo_week["probability_positive"][column]),
                    "standard_error_accuracy_points": float(
                        combo_week["standard_error"][column] * 100.0
                    ),
                },
                "season_blocked": {
                    "estimate_accuracy_points": float(combo_season["estimate"][column] * 100.0),
                    "lower_accuracy_points": float(combo_season["lower"][column] * 100.0),
                    "upper_accuracy_points": float(combo_season["upper"][column] * 100.0),
                    "probability_positive": float(combo_season["probability_positive"][column]),
                    "standard_error_accuracy_points": float(
                        combo_season["standard_error"][column] * 100.0
                    ),
                },
            }
        )

    payload: dict[str, Any] = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Every registered tilt overlay that nfl_ats.overlay_composition never enumerated, "
            "scored as a marginal ON TOP OF the three-member played card at the opener grade."
        ),
        "predeclaration_note": (
            "Not a predeclared, window-spending measurement. It re-uses the same frozen "
            "2020-2025 opener archive the composition study scored, so every row is an "
            "unresolved_below_power read on already-looked-at seasons. Per AGENTS.md an "
            "interval containing zero is never grounds to decline a member; the decision bar "
            "is expected value, i.e. probability_positive above 0.5."
        ),
        "attribution_note": (
            "The greedy forward ordering and the all-positive union are POST-HOC ATTRIBUTION "
            "on the same games the members were scored on. They propose an ordering; they "
            "confirm nothing."
        ),
        "combination_rule": (
            "Joint OR against the raw model card, complemented exactly once, identical to "
            "nfl_ats.four_overlay_composition. A member's marginal is (played union OR that "
            "member) minus the played union, so a flip the card already makes contributes "
            "nothing."
        ),
        "served_policy": {
            "card": card,
            "policy_id": POLICY_ID if card == "served" else "three_member_union",
            "members": list(SERVED_CARD_MEMBERS if card == "served" else SERVED_MEMBERS),
            "member_flip_counts": {name: len(ids) for name, ids in served_members.items()},
            "union_flip_count": len(served_ids),
        },
        "source_artifact": str(per_game_artifact),
        "source_artifact_sha256": sha256_file(per_game_artifact),
        "active_model_id": metadata.get("active_model_id"),
        "active_model_config": metadata.get("active_model_config"),
        "probability_method": metadata.get("probability_method"),
        "feature_table_sha256": metadata.get("feature_table_sha256"),
        "schedule_snapshot": snapshot_name,
        "player_feature_table": str(player_feature_path),
        "incidents_table": str(incidents),
        "grading_rule": (
            "production probability rule (home_cover_probability >= 0.5) at the "
            "Tuesday-opener decision line"
        ),
        "seasons": [archive_seasons[0], archive_seasons[-1]],
        "n_games": len(eval_frame),
        "n_pushes": int((~valid_mask).sum()),
        "n_scored_games": int(valid_mask.sum()),
        "raw_model_accuracy": raw_accuracy,
        "served_card_accuracy": served_accuracy,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "confidence": CONFIDENCE,
        "week_block_count": int(valid_blocks.drop_duplicates().shape[0]),
        "season_block_count": int(eval_frame["season"].nunique()),
        "members": member_records,
        "combinations": combo_records,
        "members_probability_positive_above_half": list(positive),
        "greedy_forward_selection": greedy_steps,
    }
    payload["timing"] = {"total_seconds": perf_counter() - started}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    stamped = write_stamped_artifact(payload, output_dir / "result.json")
    table = pd.DataFrame(
        [
            {
                "member": record["member"],
                "flips": record["flip_count_on_archive"],
                "flips_overlapping_served": record["flips_overlapping_served_card"],
                "incremental_flips": record["incremental_flip_count"],
                "served_accuracy": record["served_card_accuracy"],
                "candidate_accuracy": record["candidate_accuracy"],
                "delta_accuracy_points": record["delta_estimate_accuracy_points"],
                "week_lower": record["week_blocked"]["lower_accuracy_points"],
                "week_upper": record["week_blocked"]["upper_accuracy_points"],
                "week_probability_positive": record["week_blocked"]["probability_positive"],
                "season_lower": record["season_blocked"]["lower_accuracy_points"],
                "season_upper": record["season_blocked"]["upper_accuracy_points"],
                "season_probability_positive": record["season_blocked"]["probability_positive"],
            }
            for record in member_records
        ]
    ).sort_values("delta_accuracy_points", ascending=False)
    table.to_csv(output_dir / "marginals.csv", index=False)
    return {**stamped, "artifact_directory": str(output_dir)}


__all__ = [
    "CANDIDATE_MEMBERS",
    "CARD_CHOICES",
    "SERVED_CARD_MEMBERS",
    "SERVED_MEMBERS",
    "build_candidate_flip_sets",
    "build_pbp08_flag_table",
    "build_served_card_flip_sets",
    "build_served_flip_set",
    "run_unserved_tilt_marginals",
    "served_card_flip_set",
]
