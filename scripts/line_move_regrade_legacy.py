from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from line_move_yardstick_paired_eval import cell_stats  # noqa: E402
from roof_state_screen import build_prediction_table as roof_state_prediction_table  # noqa: E402

from nfl_ats.bye_edge_fade_overlay import bye_edge_flag_by_game  # noqa: E402
from nfl_ats.coach_fade_overlay import OVERLAY_WEEK_MAX, year_one_by_game  # noqa: E402
from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.division_revenge_tilt_overlay import division_revenge_side_by_game  # noqa: E402
from nfl_ats.forecast_cold_visitor_tilt_overlay import (  # noqa: E402
    forecast_cold_visitor_flag_by_game,
)
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (  # noqa: E402
    precip_high_total_flag_by_game,
)
from nfl_ats.interim_hc_first_game_tilt_overlay import (  # noqa: E402
    interim_first_game_flag_by_game_fail_open,
)
from nfl_ats.interim_playcaller_first_game_back_overlay import (  # noqa: E402
    games_after_playcaller_change_flag_by_game,
    load_counted_playcaller_change_events,
)
from nfl_ats.pick_probability_fit import build_fit_population  # noqa: E402
from nfl_ats.post_bye_new_playcaller_back_overlay import (  # noqa: E402
    load_coordinator_history,
    post_bye_new_oc_flag_by_game,
)
from nfl_ats.rain_on_grass_dog_challenger import rain_on_grass_flag_by_game  # noqa: E402
from nfl_ats.rookie_prior_surplus_tilt_overlay import rookie_prior_surplus_flags  # noqa: E402
from nfl_ats.schedule_flag_features import (  # noqa: E402
    ATS_STREAK_REGRESS_COLUMN,
    DIVISION_DOG_COLUMN,
    DOME_SHOOTOUT_COLUMN,
    HOME_THURSDAY_COLUMN,
    WEEK1_DOG_COLUMN,
    default_opener_lines,
    default_schedule,
    derive_ats_streak_regress_features,
    derive_division_dog_features,
    derive_dome_shootout_favorite_features,
    derive_home_thursday_features,
    derive_week1_dog_features,
)
from nfl_ats.special_teams_return_tilt_overlay import (  # noqa: E402
    special_teams_return_flag_by_game_fail_open,
)
from nfl_ats.tank_zone_fade_tilt_overlay import tank_zone_flag_by_game  # noqa: E402
from nfl_ats.transaction_flag_features import (  # noqa: E402
    DEADLINE_INTEGRATION_DRAG_COLUMN,
    SUSPENSION_RETURN_RUST_COLUMN,
    attach_deadline_integration_drag_features,
    attach_suspension_return_rust_features,
)

BASE_FEATURES = ("model_logit", "composition_flag_sum")
LOW_TOTAL_MAX = 42.0
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
FORECAST_ARCHIVE = (
    REPO / "data" / "raw" / "forecast_archive" / "pool_decision_2009_2025" / "forecasts.parquet"
)

SELECTION_RULE = (
    "Predeclared before fitting (2026-09-23): among registry families in category "
    "schedule/environment/offfield/health with classification unresolved_below_power and "
    "effect_units accuracy_points, rank distinct families by |effect|/standard_error "
    "(standard_error from the field, else (interval_high-interval_low)/(2*1.96)); take each "
    "family's own highest-ratio entry; exclude CFB-only/benchmark-transfer families (out of "
    "NFL 2020-2025 scope), positive-control and refuted_mechanism/closed entries, families "
    "already graded on line-move today (players_on_field_rating_v1, opener_error_transfer_v1/2/3, "
    "pbp08_protection_mismatch_gated_flag_sum_fit, reddit/pooled_signal_sixth_fit), referee/crew "
    "families (this repo's own crew_tilt_stacked_on_production entry documents the underlying "
    "flag as a 'late-week officiating-crew tilt', not Tuesday-knowable), and health-category "
    "families whose construction depends on in-week injury/practice reports (Friday designations, "
    "Wednesday DNP, weekly Out reports) rather than information knowable by Tuesday noon. Of the "
    "families with a confirmed standalone rebuildable NFL feature builder already in src/nfl_ats "
    "(no CFB pipeline, no in-week injury report, no referee-crew timing), take the top 8 by ratio."
)

RATIO_TABLE = (
    ("deadline_integration_drag_on_production", 2.218, "offfield"),
    ("tank_zone_fade_tilt__spread_band_short_2020_2025", 2.098, "onfield/standings"),
    ("bye_edge_fade__spread_band_long_2020_2025", 1.940, "onfield/rest"),
    ("roof_state_predicted_open_fade_on_production", 1.572, "environment/weather-forecast"),
    (
        "precip_high_total_tilt__week_in_season_weeks_1_4_2020_2025",
        1.481,
        "onfield/weather-forecast",
    ),
    ("week1_dog_on_production", 1.473, "schedule"),
    ("interim_hc_first_game_tilt__week_in_season_weeks_5_12_2020_2025", 1.449, "onfield/coach"),
    ("division_dog_on_production", 1.338, "schedule"),
)

SELECTION_RULE_BATCH2 = (
    "Batch 2 (2026-09-23, same predeclared rule as batch 1): continue down the SAME "
    "|effect|/standard_error ranking over classification=unresolved_below_power, "
    "effect_units=accuracy_points, category in schedule/environment/health/offfield/onfield, "
    "one entry per family, starting immediately after the 8 batch-1 families. Same exclusions: "
    "CFB-only/benchmark-transfer, referee/crew families, health families depending on in-week "
    "injury/practice-report data, composite/pooled-atlas entries (not one rebuildable column), "
    "ablations of already-served composition members, fitted team-rating pipelines (e.g. "
    "apm_unit_feature's play-by-play ridge fit, same class of exclusion as graph_ratings_v2), "
    "families already graded on line movement (batch 1's 8 plus the done lane's 5), and any "
    "family with no confirmed standalone rebuildable NFL feature builder in src/nfl_ats. "
    "player_arrests_back_side_policy (ratio 2.163/1.566) skipped: confirmed a live served "
    "composition member via its own overlay_leave_one_out_2026_08_26 LOO-ablation entry, not a "
    "legacy accuracy-only family. fluview_* families skipped again: Tuesday-safety of the CDC "
    "surveillance release timing still not confirmed. Of the remainder, take the next 8 by ratio."
)

RATIO_TABLE_BATCH2 = (
    ("interim_playcaller_first_game_back_on_production", 0.979, "offfield/coach"),
    ("xlg06_rookie_prior_surplus_tilt_on_production", 0.955, "onfield/roster"),
    ("forecast_cold_visitor_tilt_on_production", 0.847, "environment/weather-forecast"),
    ("ats_streak_regress_on_production", 0.835, "schedule"),
    ("post_bye_new_playcaller_back_on_production", 0.835, "offfield/coach"),
    ("home_thursday_on_production", 0.719, "schedule"),
    ("low_total_div_home_dog_on_production", 0.712, "schedule"),
    ("division_revenge_tilt_on_production", 0.573, "onfield"),
)

SELECTION_RULE_BATCH3 = (
    "Batch 3 (2026-09-23, session 4): re-derived the ranking with the SAME predeclared rule text "
    "as batch 2 (classification=unresolved_below_power, effect_units=accuracy_points, category in "
    "schedule/environment/health/offfield/onfield, one entry per family by max ratio), but this "
    "time filled standard_error with the documented fallback (interval_high-interval_low)/(2*1.96) "
    "when the stored field is null, which changes the ranked list to 388 families instead of the "
    "narrower list batch 1/2 used (division_revenge_tilt, batch 2's own pick, is rank 253 at ratio "
    "0.573 in this fuller list). Walking down from rank 253 with the SAME exclusion classes as "
    "batch 1/2 (referee/crew families; health as a block; composite/pooled-atlas clusters, not one "
    "rebuildable column; fitted team-rating/team-style pipelines; era-scope mismatches with near-"
    "zero overlap on the ~2020-2025 population) only 4 candidates had a confirmed live standalone "
    "Tuesday-safe builder still present in src/nfl_ats within this session's budget: a large share "
    "of the remaining high-ratio legacy families have no matching module because their source "
    "files were deleted in the repository-cut commit b7ed31d. Two higher-ratio families with "
    "confirmed live builders were deliberately NOT included and are flagged to the orchestrator "
    "instead of graded: special_teams_return_top_quartile (ratio 1.694, onfield) and "
    "hc_year_one_fade (ratio 1.495, offfield, same module as the already-served coach_fade "
    "backup) both rank above every "
    "family batch 1/2 actually selected and neither was picked by those batches for a reason not "
    "re-derivable this session; grading them now without resolving why they were skipped would "
    "risk silently overriding an earlier exclusion. This batch keeps coach_fade at its existing "
    "batch-fixed identity/rank (~0.52) rather than swapping in hc_year_one_fade. Only 4 of the "
    "nominal 8-per-batch cadence graded; ranks below coach_fade (<0.513) are unexamined."
)

RATIO_TABLE_BATCH3 = (
    ("suspension_return_rust_on_production", 0.572, "offfield"),
    ("rain_on_grass_dog_on_production", 0.567, "environment"),
    ("dome_shootout_favorite_on_production", 0.525, "schedule"),
    ("coach_fade_on_production", 0.520, "onfield"),
)


def add_tank_zone_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = tank_zone_flag_by_game(schedule)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "tank_zone_home", "tank_zone_away"]]
    out = population.merge(flags, on="game_id", how="left")
    out["tank_zone_home"] = out["tank_zone_home"].fillna(False)
    out["tank_zone_away"] = out["tank_zone_away"].fillna(False)
    out["tank_zone_term"] = np.where(
        out["tank_zone_home"], 1.0, np.where(out["tank_zone_away"], -1.0, 0.0)
    )
    return out


def add_bye_edge_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = bye_edge_flag_by_game(schedule)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "home_off_bye", "away_off_bye"]]
    out = population.merge(flags, on="game_id", how="left")
    out["home_off_bye"] = out["home_off_bye"].fillna(False)
    out["away_off_bye"] = out["away_off_bye"].fillna(False)
    out["bye_edge_term"] = np.where(
        out["home_off_bye"], 1.0, np.where(out["away_off_bye"], -1.0, 0.0)
    )
    return out


def add_roof_state_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    table = roof_state_prediction_table()[["game_id", "predicted_open"]].drop_duplicates(
        subset="game_id"
    )
    table["game_id"] = table["game_id"].astype(str)
    out = population.merge(table, on="game_id", how="left")
    out["predicted_open"] = out["predicted_open"].fillna(False)
    out["roof_state_term"] = out["predicted_open"].astype(float)
    return out


def add_precip_high_total_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    forecasts = pd.read_parquet(FORECAST_ARCHIVE)
    total_lines = schedule[["game_id", "total_line"]].copy()
    total_lines["game_id"] = total_lines["game_id"].astype(str)
    flags = precip_high_total_flag_by_game(schedule, forecasts, total_lines)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "precip_high_total_flag"]]
    out = population.merge(flags, on="game_id", how="left")
    out["precip_high_total_flag"] = out["precip_high_total_flag"].fillna(False)
    out["precip_high_total_term"] = out["precip_high_total_flag"].astype(float)
    return out


def add_week1_dog_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    opener_lines = population[["game_id", "tue_open_home_spread"]].copy()
    derived = derive_week1_dog_features(schedule, opener_lines)
    derived = derived.drop_duplicates(subset="game_id")
    out = population.merge(derived, on="game_id", how="left")
    out[WEEK1_DOG_COLUMN] = out[WEEK1_DOG_COLUMN].fillna(0.0)
    return out


def add_division_dog_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    opener_lines = population[["game_id", "tue_open_home_spread"]].copy()
    derived = derive_division_dog_features(schedule, opener_lines)
    derived = derived.drop_duplicates(subset="game_id")
    out = population.merge(derived, on="game_id", how="left")
    out[DIVISION_DOG_COLUMN] = out[DIVISION_DOG_COLUMN].fillna(0.0)
    return out


def add_interim_hc_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = interim_first_game_flag_by_game_fail_open(REPO)
    sched = schedule[["game_id", "home_team", "away_team"]].copy()
    sched["game_id"] = sched["game_id"].astype(str)
    if flags.empty:
        sched["interim_hc_term"] = 0.0
    else:
        flags = flags.copy()
        flags["game_id"] = flags["game_id"].astype(str)
        home_hits = set(zip(flags["game_id"], flags["team"], strict=False))
        sched["home_first_game"] = [
            (gid, team) in home_hits
            for gid, team in zip(sched["game_id"], sched["home_team"], strict=False)
        ]
        sched["away_first_game"] = [
            (gid, team) in home_hits
            for gid, team in zip(sched["game_id"], sched["away_team"], strict=False)
        ]
        sched["interim_hc_term"] = np.where(
            sched["home_first_game"], 1.0, np.where(sched["away_first_game"], -1.0, 0.0)
        )
    out = population.merge(sched[["game_id", "interim_hc_term"]], on="game_id", how="left")
    out["interim_hc_term"] = out["interim_hc_term"].fillna(0.0)
    return out


def add_deadline_drag_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    features = population[["game_id"]].copy()
    features["game_id"] = features["game_id"].astype(str)
    try:
        attached = attach_deadline_integration_drag_features(features, schedule=schedule)
    except DataContractError:
        attached = features.copy()
        attached[DEADLINE_INTEGRATION_DRAG_COLUMN] = 0.0
    attached = attached.drop_duplicates(subset="game_id")
    out = population.merge(attached, on="game_id", how="left")
    out[DEADLINE_INTEGRATION_DRAG_COLUMN] = out[DEADLINE_INTEGRATION_DRAG_COLUMN].fillna(0.0)
    return out


def add_playcaller_change_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    events = load_counted_playcaller_change_events(REPO / "data")
    flags = games_after_playcaller_change_flag_by_game(schedule, events)
    flags = flags.drop_duplicates(subset="game_id")
    out = population.merge(flags, on="game_id", how="left")
    out["home_flagged"] = out["home_flagged"].fillna(False)
    out["away_flagged"] = out["away_flagged"].fillna(False)
    out["playcaller_change_term"] = np.where(
        out["home_flagged"], 1.0, np.where(out["away_flagged"], -1.0, 0.0)
    )
    return out


def add_rookie_priors_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    games = schedule[["game_id", "season", "week", "home_team", "away_team"]].copy()
    games["game_id"] = games["game_id"].astype(str)
    flags = rookie_prior_surplus_flags(REPO, games)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "flag"]]
    out = population.merge(flags, on="game_id", how="left")
    out["flag"] = out["flag"].fillna(0.0)
    out["rookie_priors_term"] = out["flag"].astype(float)
    return out


def add_forecast_cold_visitor_term(
    population: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    forecasts = pd.read_parquet(FORECAST_ARCHIVE)
    flags = forecast_cold_visitor_flag_by_game(schedule, forecasts)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "forecast_cold_visitor_flag"]]
    out = population.merge(flags, on="game_id", how="left")
    out["forecast_cold_visitor_flag"] = out["forecast_cold_visitor_flag"].fillna(False)
    out["forecast_cold_visitor_term"] = out["forecast_cold_visitor_flag"].astype(float)
    return out


def add_ats_streak_regress_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    derived = derive_ats_streak_regress_features(schedule)
    derived = derived.drop_duplicates(subset="game_id")
    out = population.merge(derived, on="game_id", how="left")
    out[ATS_STREAK_REGRESS_COLUMN] = out[ATS_STREAK_REGRESS_COLUMN].fillna(0.0)
    return out


def add_post_bye_new_oc_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    coordinator_history = load_coordinator_history(REPO / "data")
    flags = post_bye_new_oc_flag_by_game(schedule, coordinator_history)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "home_flagged", "away_flagged"]]
    out = population.merge(flags, on="game_id", how="left")
    out["home_flagged"] = out["home_flagged"].fillna(False)
    out["away_flagged"] = out["away_flagged"].fillna(False)
    out["post_bye_new_oc_term"] = np.where(
        out["home_flagged"], 1.0, np.where(out["away_flagged"], -1.0, 0.0)
    )
    return out


def add_home_thursday_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    derived = derive_home_thursday_features(schedule)
    derived = derived.drop_duplicates(subset="game_id")
    out = population.merge(derived, on="game_id", how="left")
    out[HOME_THURSDAY_COLUMN] = out[HOME_THURSDAY_COLUMN].fillna(0.0)
    return out


def add_low_total_div_home_dog_term(
    population: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    cols = ["game_id", "div_game", "total_line", "spread_line", "game_type"]
    sched = schedule[cols].copy()
    sched["game_id"] = sched["game_id"].astype(str)
    div_game = pd.to_numeric(sched["div_game"], errors="coerce").eq(1.0)
    total_line = pd.to_numeric(sched["total_line"], errors="coerce")
    spread_line = pd.to_numeric(sched["spread_line"], errors="coerce")
    low_total = total_line.notna() & total_line.le(LOW_TOTAL_MAX)
    home_dog = spread_line.notna() & spread_line.lt(0.0)
    reg = sched["game_type"].astype(str).eq("REG")
    eligible = div_game & low_total & home_dog & reg
    sched["low_total_div_home_dog_term"] = np.where(eligible, -1.0, 0.0)
    out = population.merge(
        sched[["game_id", "low_total_div_home_dog_term"]], on="game_id", how="left"
    )
    out["low_total_div_home_dog_term"] = out["low_total_div_home_dog_term"].fillna(0.0)
    return out


def add_division_revenge_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = division_revenge_side_by_game(schedule)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "revenge_home", "revenge_away"]]
    out = population.merge(flags, on="game_id", how="left")
    out["revenge_home"] = out["revenge_home"].fillna(False)
    out["revenge_away"] = out["revenge_away"].fillna(False)
    out["division_revenge_term"] = np.where(
        out["revenge_home"], 1.0, np.where(out["revenge_away"], -1.0, 0.0)
    )
    return out


def add_suspension_return_rust_term(
    population: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    features = population[["game_id"]].copy()
    features["game_id"] = features["game_id"].astype(str)
    try:
        attached = attach_suspension_return_rust_features(features, schedule=schedule)
    except DataContractError:
        attached = features.copy()
        attached[SUSPENSION_RETURN_RUST_COLUMN] = 0.0
    attached = attached.drop_duplicates(subset="game_id")
    out = population.merge(attached, on="game_id", how="left")
    out[SUSPENSION_RETURN_RUST_COLUMN] = out[SUSPENSION_RETURN_RUST_COLUMN].fillna(0.0)
    return out


def add_rain_on_grass_dog_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    forecasts = pd.read_parquet(FORECAST_ARCHIVE)
    flags = rain_on_grass_flag_by_game(schedule, forecasts)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "rain_on_grass_flag"]]
    cols = ["game_id", "spread_line", "game_type"]
    sched = schedule[cols].copy()
    sched["game_id"] = sched["game_id"].astype(str)
    sched = sched.merge(flags, on="game_id", how="left")
    sched["rain_on_grass_flag"] = sched["rain_on_grass_flag"].fillna(False)
    spread_line = pd.to_numeric(sched["spread_line"], errors="coerce")
    reg = sched["game_type"].astype(str).eq("REG")
    home_dog = sched["rain_on_grass_flag"] & spread_line.notna() & spread_line.lt(0.0) & reg
    away_dog = sched["rain_on_grass_flag"] & spread_line.notna() & spread_line.gt(0.0) & reg
    sched["rain_on_grass_dog_term"] = np.where(home_dog, 1.0, np.where(away_dog, -1.0, 0.0))
    out = population.merge(sched[["game_id", "rain_on_grass_dog_term"]], on="game_id", how="left")
    out["rain_on_grass_dog_term"] = out["rain_on_grass_dog_term"].fillna(0.0)
    return out


def add_dome_shootout_favorite_term(
    population: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    opener_lines = default_opener_lines(schedule)
    derived = derive_dome_shootout_favorite_features(schedule, opener_lines)
    derived = derived.drop_duplicates(subset="game_id")[["game_id", DOME_SHOOTOUT_COLUMN]]
    out = population.merge(derived, on="game_id", how="left")
    out[DOME_SHOOTOUT_COLUMN] = out[DOME_SHOOTOUT_COLUMN].fillna(0.0)
    return out


def add_coach_fade_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = year_one_by_game(schedule)
    flags = flags.drop_duplicates(subset="game_id")[
        ["game_id", "year_one_home", "year_one_away"]
    ]
    out = population.merge(flags, on="game_id", how="left")
    out["year_one_home"] = out["year_one_home"].fillna(False)
    out["year_one_away"] = out["year_one_away"].fillna(False)
    out["coach_fade_term"] = np.where(
        out["year_one_home"], 1.0, np.where(out["year_one_away"], -1.0, 0.0)
    )
    return out


def add_special_teams_return_term(
    population: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    flags = special_teams_return_flag_by_game_fail_open(REPO / "data", schedule)
    flags = flags.drop_duplicates(subset="game_id")[
        ["game_id", "home_return_top_quartile", "away_return_top_quartile"]
    ]
    flags["game_id"] = flags["game_id"].astype(str)
    out = population.merge(flags, on="game_id", how="left")
    out["home_return_top_quartile"] = out["home_return_top_quartile"].fillna(False)
    out["away_return_top_quartile"] = out["away_return_top_quartile"].fillna(False)
    both_flagged = out["home_return_top_quartile"] & out["away_return_top_quartile"]
    out["special_teams_return_term"] = np.where(
        out["home_return_top_quartile"] & ~both_flagged,
        1.0,
        np.where(out["away_return_top_quartile"] & ~both_flagged, -1.0, 0.0),
    )
    return out


def add_hc_year_one_fade_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = year_one_by_game(schedule)
    flags = flags.drop_duplicates(subset="game_id")[
        ["game_id", "year_one_home", "year_one_away"]
    ]
    out = population.merge(flags, on="game_id", how="left")
    out["year_one_home"] = out["year_one_home"].fillna(False)
    out["year_one_away"] = out["year_one_away"].fillna(False)
    week_eligible = pd.to_numeric(out["week"], errors="coerce").le(OVERLAY_WEEK_MAX)
    out["hc_year_one_fade_term"] = np.where(
        out["year_one_home"] & week_eligible,
        1.0,
        np.where(out["year_one_away"] & week_eligible, -1.0, 0.0),
    )
    return out


def loso(frame: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    frame = frame.reset_index(drop=True)
    x_cols = list(features)
    oos = np.full(len(frame), np.nan)
    folds: dict[str, list[float]] = {}
    for season in sorted(frame["season"].unique()):
        train = frame.loc[frame["season"] != season]
        test_idx = frame.index[frame["season"] == season]
        means = {name: float(train[name].mean()) for name in x_cols}
        stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in x_cols}
        design_train = np.column_stack(
            [np.ones(len(train))]
            + [((train[name] - means[name]) / stds[name]).to_numpy() for name in x_cols]
        )
        y_train = train["home_covered"].astype(float).to_numpy()
        beta = np.zeros(design_train.shape[1])
        ridge = 1e-3
        for _ in range(50):
            z = np.clip(design_train @ beta, -35.0, 35.0)
            p = 1.0 / (1.0 + np.exp(-z))
            w = np.clip(p * (1.0 - p), 1e-6, None)
            gradient = design_train.T @ (y_train - p) - ridge * beta
            hessian = (design_train.T * w) @ design_train + ridge * np.eye(design_train.shape[1])
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            beta = beta + step
        test = frame.loc[test_idx]
        design_test = np.column_stack(
            [np.ones(len(test))]
            + [((test[name] - means[name]) / stds[name]).to_numpy() for name in x_cols]
        )
        z_test = np.clip(design_test @ beta, -35.0, 35.0)
        oos[test_idx] = 1.0 / (1.0 + np.exp(-z_test))
        folds[str(int(season))] = beta.tolist()
    return oos, folds


TERM_DECLARATIONS = (
    {
        "label": "tank_zone_fade_tilt",
        "term_columns": ("tank_zone_term",),
        "builder": add_tank_zone_term,
    },
    {"label": "bye_edge_fade", "term_columns": ("bye_edge_term",), "builder": add_bye_edge_term},
    {
        "label": "roof_state_predicted_open",
        "term_columns": ("roof_state_term",),
        "builder": add_roof_state_term,
    },
    {
        "label": "precip_high_total_tilt",
        "term_columns": ("precip_high_total_term",),
        "builder": add_precip_high_total_term,
    },
    {"label": "week1_dog", "term_columns": (WEEK1_DOG_COLUMN,), "builder": add_week1_dog_term},
    {
        "label": "interim_hc_first_game_tilt",
        "term_columns": ("interim_hc_term",),
        "builder": add_interim_hc_term,
    },
    {
        "label": "division_dog",
        "term_columns": (DIVISION_DOG_COLUMN,),
        "builder": add_division_dog_term,
    },
    {
        "label": "deadline_integration_drag",
        "term_columns": (DEADLINE_INTEGRATION_DRAG_COLUMN,),
        "builder": add_deadline_drag_term,
    },
)

TERM_DECLARATIONS_BATCH2 = (
    {
        "label": "playcaller_change",
        "term_columns": ("playcaller_change_term",),
        "builder": add_playcaller_change_term,
    },
    {
        "label": "rookie_priors",
        "term_columns": ("rookie_priors_term",),
        "builder": add_rookie_priors_term,
    },
    {
        "label": "forecast_cold_visitor_tilt",
        "term_columns": ("forecast_cold_visitor_term",),
        "builder": add_forecast_cold_visitor_term,
    },
    {
        "label": "ats_streak_regress",
        "term_columns": (ATS_STREAK_REGRESS_COLUMN,),
        "builder": add_ats_streak_regress_term,
    },
    {
        "label": "post_bye_new_oc",
        "term_columns": ("post_bye_new_oc_term",),
        "builder": add_post_bye_new_oc_term,
    },
    {
        "label": "home_thursday",
        "term_columns": (HOME_THURSDAY_COLUMN,),
        "builder": add_home_thursday_term,
    },
    {
        "label": "low_total_div_home_dog",
        "term_columns": ("low_total_div_home_dog_term",),
        "builder": add_low_total_div_home_dog_term,
    },
    {
        "label": "division_revenge_tilt",
        "term_columns": ("division_revenge_term",),
        "builder": add_division_revenge_term,
    },
)

TERM_DECLARATIONS_BATCH3 = (
    {
        "label": "suspension_return_rust_on_production",
        "term_columns": (SUSPENSION_RETURN_RUST_COLUMN,),
        "builder": add_suspension_return_rust_term,
    },
    {
        "label": "rain_on_grass_dog_on_production",
        "term_columns": ("rain_on_grass_dog_term",),
        "builder": add_rain_on_grass_dog_term,
    },
    {
        "label": "dome_shootout_favorite_on_production",
        "term_columns": (DOME_SHOOTOUT_COLUMN,),
        "builder": add_dome_shootout_favorite_term,
    },
    {
        "label": "coach_fade_on_production",
        "term_columns": ("coach_fade_term",),
        "builder": add_coach_fade_term,
    },
)

SELECTION_RULE_BATCH4 = (
    "Batch 4 (2026-09-23, session 6): the 2 families batch 3 confirmed had a live builder but "
    "deliberately left ungraded, flagged as an open ranking discrepancy -- "
    "special_teams_return_top_quartile (ratio 1.694, onfield) and hc_year_one_fade (ratio 1.495, "
    "offfield). special_teams_return_top_quartile builds from "
    "special_teams_return_tilt_overlay.special_teams_return_flag_by_game_fail_open, local data "
    "confirmed present. hc_year_one_fade shares its builder (coach_fade_overlay.year_one_by_game) "
    "with batch 3's coach_fade_on_production, but the registered hc_year_one_fade family is "
    "restricted to weeks 1-8 (docs/hc_year_one_fade.md) while batch 3's term applied the flag to "
    "all weeks with no cutoff -- batch 3 therefore graded an unrestricted variant, not the "
    "registered family; this batch applies the same week<=8 cutoff apply_coach_fade_overlay uses "
    "(OVERLAY_WEEK_MAX), so the two terms are not duplicates despite the shared flag builder."
)

RATIO_TABLE_BATCH4 = (
    ("special_teams_return_top_quartile_on_production", 1.694, "onfield"),
    ("hc_year_one_fade_on_production", 1.495, "offfield"),
)

TERM_DECLARATIONS_BATCH4 = (
    {
        "label": "special_teams_return_top_quartile_on_production",
        "term_columns": ("special_teams_return_term",),
        "builder": add_special_teams_return_term,
    },
    {
        "label": "hc_year_one_fade_on_production",
        "term_columns": ("hc_year_one_fade_term",),
        "builder": add_hc_year_one_fade_term,
    },
)


def variant_report(
    declaration: dict, population: pd.DataFrame, schedule: pd.DataFrame, seed: int, draws: int
) -> dict[str, object]:
    enriched = declaration["builder"](population, schedule)
    term_columns = list(declaration["term_columns"])
    subset = enriched.dropna(subset=[*term_columns, "open_move"]).reset_index(drop=True)

    features_variant = (*BASE_FEATURES, *term_columns)
    base_oos, base_folds = loso(subset, BASE_FEATURES)
    variant_oos, variant_folds = loso(subset, features_variant)

    frame = subset.copy()
    frame["base_oos_p"] = base_oos
    frame["variant_oos_p"] = variant_oos
    frame = frame.loc[frame["base_oos_p"].notna() & frame["variant_oos_p"].notna()].copy()

    frame["base_pick_home"] = frame["base_oos_p"].ge(0.5)
    frame["variant_pick_home"] = frame["variant_oos_p"].ge(0.5)
    frame["lm_base"] = np.where(frame["base_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["lm_variant"] = np.where(frame["variant_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["diff_line_move"] = frame["lm_variant"] - frame["lm_base"]

    frame["base_correct"] = (
        frame["base_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["variant_correct"] = (
        frame["variant_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["diff_accuracy"] = frame["variant_correct"] - frame["base_correct"]

    line_move_cell = cell_stats(
        f"{declaration['label']}_line_move_toward_pick", frame, "diff_line_move", seed, draws
    )
    accuracy_cell = cell_stats(
        f"{declaration['label']}_accuracy_companion", frame, "diff_accuracy", seed + 500, draws
    )
    positive_rate = (
        float(frame[term_columns[0]].astype(bool).mean()) if len(term_columns) == 1 else None
    )
    return {
        "label": declaration["label"],
        "term_columns": term_columns,
        "base_features": list(BASE_FEATURES),
        "variant_features": list(features_variant),
        "coverage_games_before_pairing": len(subset),
        "paired_games": len(frame),
        "seasons": sorted(int(v) for v in frame["season"].unique()),
        "term_nonzero_rate": positive_rate,
        "base_fold_coefficients": base_folds,
        "variant_fold_coefficients": variant_folds,
        "line_move_toward_pick_cell": line_move_cell,
        "accuracy_companion_cell": accuracy_cell,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    if args.batch == 4:
        term_declarations = TERM_DECLARATIONS_BATCH4
        selection_rule = SELECTION_RULE_BATCH4
        ratio_table = RATIO_TABLE_BATCH4
    elif args.batch == 3:
        term_declarations = TERM_DECLARATIONS_BATCH3
        selection_rule = SELECTION_RULE_BATCH3
        ratio_table = RATIO_TABLE_BATCH3
    elif args.batch == 2:
        term_declarations = TERM_DECLARATIONS_BATCH2
        selection_rule = SELECTION_RULE_BATCH2
        ratio_table = RATIO_TABLE_BATCH2
    else:
        term_declarations = TERM_DECLARATIONS
        selection_rule = SELECTION_RULE
        ratio_table = RATIO_TABLE

    if args.only:
        term_declarations = tuple(
            d for d in term_declarations if d["label"] == args.only
        )

    now = datetime.now(UTC)
    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    population = population.dropna(subset=["open_move"]).reset_index(drop=True)
    population["game_id"] = population["game_id"].astype(str)
    schedule = default_schedule()
    schedule["game_id"] = schedule["game_id"].astype(str)

    reports = []
    for declaration in term_declarations:
        try:
            reports.append(
                variant_report(declaration, population, schedule, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
            )
        except Exception as exc:
            reports.append({"label": declaration["label"], "error": f"{type(exc).__name__}: {exc}"})

    results = {
        "command": (
            f"python scripts/line_move_regrade_legacy.py --batch {args.batch}"
            + (f" --only {args.only}" if args.only else "")
        ),
        "created_at_utc": now.isoformat(),
        "unit": "Tuesday-terms line-move regrade, legacy registry families",
        "batch": args.batch,
        "selection_rule": selection_rule,
        "predeclared_ratio_table": [
            {"registry_name": name, "prior_abs_effect_over_se": ratio, "category": cat}
            for name, ratio, cat in ratio_table
        ],
        "base_population_games": len(population),
        "seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "opener_evaluation_provenance": provenance,
        "terms": reports,
    }

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "line_move_regrade_legacy" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        __import__("json").dumps(results, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "artifact": str(out_dir.relative_to(REPO)).replace("\\", "/"),
        "base_population_games": len(population),
        "terms": [
            {
                "label": report.get("label"),
                "error": report.get("error"),
                "paired_games": report.get("paired_games"),
                "line_move_mean_points": (report.get("line_move_toward_pick_cell") or {}).get(
                    "mean_points"
                ),
                "line_move_season_block_interval": [
                    (report.get("line_move_toward_pick_cell") or {}).get(
                        "season_block_interval_low"
                    ),
                    (report.get("line_move_toward_pick_cell") or {}).get(
                        "season_block_interval_high"
                    ),
                ],
                "line_move_season_block_probability_positive": (
                    report.get("line_move_toward_pick_cell") or {}
                ).get("season_block_probability_positive"),
                "accuracy_mean_points": (report.get("accuracy_companion_cell") or {}).get(
                    "mean_points"
                ),
                "accuracy_season_block_probability_positive": (
                    report.get("accuracy_companion_cell") or {}
                ).get("season_block_probability_positive"),
            }
            for report in reports
        ],
    }
    print(__import__("json").dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
