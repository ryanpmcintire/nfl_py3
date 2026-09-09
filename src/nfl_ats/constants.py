"""Shared schema, feature, and walk-forward definitions."""

from __future__ import annotations

MIN_FITTABLE_TRAIN_GAMES = 50

DEFAULT_MIN_TRAIN_GAMES = 500

DEFAULT_MIN_CALIBRATION_GAMES = 200

DEFAULT_OFFSEASON_RETENTION = 0.67

EARLY_SEASON_GAME_COUNT = 256

SCHEDULE_REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "result",
    "spread_line",
)

TEAM_ABBREVIATION_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "OAK": "LV",
    "SD": "LAC",
    "SL": "LA",
    "STL": "LA",
}

TEAM_STATS_REQUIRED_COLUMNS = (
    "game_id",
    "team",
    "attempts",
    "carries",
    "passing_epa",
    "rushing_epa",
)

STATE_METRICS = (
    "off_epa_per_play",
    "off_pass_epa_per_play",
    "off_rush_epa_per_play",
    "off_cpoe",
    "off_yards_per_play",
    "off_turnover_rate",
    "off_sack_rate",
    "point_diff",
    "ats_residual",
    "def_epa_per_play",
    "def_pass_epa_per_play",
    "def_rush_epa_per_play",
    "def_yards_per_play",
    "def_takeaway_rate",
    "def_sack_rate",
)

PBP_STATE_METRICS = (
    "pbp_off_epa_per_play",
    "pbp_off_early_down_epa",
    "pbp_off_success_rate",
    "pbp_off_explosive_rate",
    "pbp_off_pass_rate",
    "pbp_off_proe",
    "pbp_pressure_allowed_rate",
    "pbp_sack_allowed_rate",
    "pbp_start_yardline_100",
    "pbp_drives",
    "pbp_def_epa_per_play",
    "pbp_def_early_down_epa",
    "pbp_def_success_rate_allowed",
    "pbp_def_explosive_rate_allowed",
    "pbp_pressure_rate",
    "pbp_sack_rate",
)

PBP_OPPONENT_ADJUSTMENT_METRICS = (
    ("pbp_off_epa_per_play", "pbp_matchup_epa_per_play"),
    ("pbp_off_early_down_epa", "pbp_matchup_early_down_epa"),
    ("pbp_off_success_rate", "pbp_matchup_success_rate"),
    ("pbp_off_explosive_rate", "pbp_matchup_explosive_rate"),
    ("pbp_pressure_allowed_rate", "pbp_matchup_pressure_allowed_rate"),
    ("pbp_sack_allowed_rate", "pbp_matchup_sack_allowed_rate"),
)

DRIVE_STATE_METRICS = (
    "drive_points_per_drive",
    "drive_yards_per_drive",
    "drive_plays_per_drive",
    "drive_seconds_per_drive",
    "drive_scoring_rate",
    "drive_turnover_rate",
    "drive_points_per_drive_allowed",
    "drive_yards_per_drive_allowed",
    "drive_plays_per_drive_allowed",
    "drive_seconds_per_drive_allowed",
    "drive_scoring_rate_allowed",
    "drive_takeaway_rate",
)

PBP_ENRICHMENT_STATE_METRICS = PBP_STATE_METRICS + DRIVE_STATE_METRICS

QB_STATE_METRICS = (
    "qb_epa_per_dropback",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_interception_rate",
    "qb_explosive_pass_rate",
)

QB_DEPTH_STATE_METRICS = (
    "depth_qb_start_probability",
    "depth_qb_expected_epa_per_dropback",
    "depth_qb_expected_cpoe",
    "depth_qb_starter_epa_per_dropback",
    "depth_qb_starter_cpoe",
    "depth_qb_starter_experience_log",
    "depth_qb_backup_epa_per_dropback",
    "depth_qb_backup_cpoe",
    "depth_qb_backup_experience_log",
    "depth_qb_backup_adjustment_epa_per_dropback",
    "depth_qb_backup_adjustment_cpoe",
)

PLAYER_QB_STATE_METRICS = (
    "qb_expected_epa_per_dropback",
    "qb_starter_epa_per_dropback",
    "qb_starter_cpoe",
    "qb_start_probability",
    "qb_starter_experience_log",
)
PLAYER_INJURY_STATE_METRICS = (
    "injury_offense_unavailability",
    "injury_defense_unavailability",
    "injury_special_teams_unavailability",
    "injury_offensive_line_unavailability",
    "injury_skill_unavailability",
    "injury_front_unavailability",
    "injury_secondary_unavailability",
)
PLAYER_CONTINUITY_STATE_METRICS = (
    "offense_lineup_continuity",
    "offensive_line_continuity",
    "skill_lineup_continuity",
    "defense_lineup_continuity",
    "front_lineup_continuity",
    "secondary_lineup_continuity",
    "special_teams_lineup_continuity",
    "active_roster_continuity",
    "active_roster_mean_experience",
)
ROSTER_RETURNING_SNAP_STATE_METRICS = (
    "returning_offense_snap_share",
    "returning_defense_snap_share",
    "returning_special_teams_snap_share",
)
SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS = (
    "diff_defense_lineup_continuity",
    "diff_front_lineup_continuity",
    "diff_offense_lineup_continuity",
    "diff_offensive_line_continuity",
    "diff_secondary_lineup_continuity",
    "diff_skill_lineup_continuity",
    "diff_special_teams_lineup_continuity",
)
PLAYER_VALUE_STATE_METRICS = (
    "injury_skill_epa_value_lost",
    "injury_defense_disruption_value_lost",
)
PLAYER_PARTICIPATION_STATE_METRICS = (
    "injury_offense_participation_value_lost",
    "injury_defense_participation_value_lost",
)
PLAYER_STATE_METRICS = (
    PLAYER_QB_STATE_METRICS + PLAYER_INJURY_STATE_METRICS + PLAYER_CONTINUITY_STATE_METRICS
)
PLAYER_ALL_STATE_METRICS = (
    PLAYER_STATE_METRICS + PLAYER_VALUE_STATE_METRICS + ROSTER_RETURNING_SNAP_STATE_METRICS
)

GRAPH_FEATURE_COLUMNS = (
    "home_graph_pagerank",
    "away_graph_pagerank",
    "graph_pagerank_diff",
    "home_graph_offense",
    "away_graph_offense",
    "home_graph_defense",
    "away_graph_defense",
    "graph_matchup_diff",
    "home_schedule_rating",
    "away_schedule_rating",
    "schedule_rating_diff",
    "schedule_predicted_margin",
)

BIAS_METRICS = (
    "bias_playoff_holdover",
    "bias_prior_week_ats",
    "bias_week2_anchor",
)

BIAS_FEATURE_COLUMNS = tuple(
    column
    for metric in BIAS_METRICS
    for column in (f"{metric}_home", f"{metric}_away", f"{metric}_diff")
)

SURFACE_SWITCH_FEATURE_COLUMNS = ("surface_switch_flag",)

TRAVEL_GEOMETRY_FEATURE_COLUMNS = (
    "travel_home_distance_mi",
    "travel_away_distance_mi",
    "travel_home_tz_change_hours",
    "travel_away_tz_change_hours",
    "travel_home_body_clock_direction",
    "travel_away_body_clock_direction",
    "travel_international_game",
    "travel_neutral_site",
    "travel_home_prior_game_distance_mi",
    "travel_away_prior_game_distance_mi",
)

REST_CONTEXT_FEATURE_COLUMNS = (
    "rest_home_days",
    "rest_away_days",
    "rest_days_diff",
    "rest_home_off_bye",
    "rest_away_off_bye",
    "rest_home_short_week",
    "rest_away_short_week",
    "rest_home_mini_bye",
    "rest_away_mini_bye",
    "rest_away_consecutive_road_games",
)

FORECAST_WEATHER_FEATURE_COLUMNS = (
    "forecast_temp_f",
    "forecast_wind_mph",
    "forecast_precip_prob_pct",
    "forecast_is_outdoors",
    "forecast_temp_f_outdoor",
    "forecast_wind_mph_outdoor",
)

OBSERVED_WEATHER_FEATURE_COLUMNS = (
    "observed_temp_f",
    "observed_wind_mph",
    "observed_is_outdoors",
    "observed_temp_f_outdoor",
    "observed_wind_mph_outdoor",
)

GRAPH_TEAM_STAT_OFF_SACK_RATE_FEATURE_COLUMNS = ("graph_v2_team_stat_off_sack_rate_katz_diff",)

GRAPH_TEAM_STAT_DEF_YARDS_PER_PLAY_FEATURE_COLUMNS = (
    "graph_v2_team_stat_def_yards_per_play_katz_diff",
)

GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS = (
    "graph_v2_team_stat_off_rush_epa_per_play_katz_diff",
)

FLUVIEW_HOME_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS = ("fluview_home_market_elevated",)
FLUVIEW_AWAY_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS = ("fluview_away_market_elevated",)


ILLNESS_AWAY_ACTIVE_GE1_ON_PRODUCTION_FEATURE_COLUMNS = ("illness_away_active_ge1",)
ILLNESS_HOME_GE2_ON_PRODUCTION_FEATURE_COLUMNS = ("illness_home_ge2",)

REDDIT_HOME_RATIO_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS = ("reddit_home_comment_ratio_elevated",)
REDDIT_AWAY_SPIKE_ON_PRODUCTION_FEATURE_COLUMNS = ("reddit_away_spike_value",)

TEAM_STYLE_PACE_MISMATCH_ON_PRODUCTION_FEATURE_COLUMNS = ("team_style_pace_mismatch_flag",)

REDZONE_THIRD_DOWN_OVER_FADE_ON_PRODUCTION_FEATURE_COLUMNS = ("redzone_third_down_over_fade_diff",)

POST_OT_FATIGUE_ON_PRODUCTION_FEATURE_COLUMNS = ("post_ot_fatigue_flag",)
MNF_ROAD_SHORT_WEEK_ON_PRODUCTION_FEATURE_COLUMNS = ("mnf_road_short_week_flag",)
HOME_THURSDAY_ON_PRODUCTION_FEATURE_COLUMNS = ("home_thursday_flag",)

NEW_STADIUM_HOME_ON_PRODUCTION_FEATURE_COLUMNS = ("new_stadium_home_flag",)
DOME_SHOOTOUT_FAVORITE_ON_PRODUCTION_FEATURE_COLUMNS = ("dome_shootout_favorite_flag",)
LOW_TOTAL_DIV_HOME_DOG_ON_PRODUCTION_FEATURE_COLUMNS = ("low_total_div_home_dog_flag",)
SEPT_HEAT_HOME_ON_PRODUCTION_FEATURE_COLUMNS = ("sept_heat_home_flag",)

GAP_V3_BIAS_METRICS = (
    "gap_division_revenge",
    "gap_sandwich_spot",
    "gap_post_blowout_win_letdown",
    "gap_post_blowout_loss_bounce",
)
GAP_V3_BIAS_FEATURE_COLUMNS = tuple(
    column
    for metric in GAP_V3_BIAS_METRICS
    for column in (f"{metric}_home", f"{metric}_away", f"{metric}_diff")
)
GAP_V3_PENALTY_FEATURE_COLUMNS = ("diff_penalty_rate_prior",)
GAP_V3_TRAVEL_FEATURE_COLUMNS = (
    "gap_thursday_pure_flag",
    "gap_return_trip_hangover_flag",
)

PLAYER_VALUE_JS_PRIOR_STATE_METRICS = (
    "injury_skill_epa_value_lost_js_prior",
    "injury_defense_disruption_value_lost_js_prior",
)

SCHEDULE_FEATURES = (
    "spread_line",
    "total_line",
    "rest_diff",
    "neutral_site",
    "div_game",
    "temp",
    "wind",
    "week_sin",
    "week_cos",
    "elo_diff",
    "elo_home_win_prob",
    "home_team_games",
    "away_team_games",
)


def _team_state_features(metrics: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        column
        for metric in metrics
        for column in (f"home_{metric}", f"away_{metric}", f"diff_{metric}")
    )


def _difference_features(metrics: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"diff_{metric}" for metric in metrics)


PBP_OPPONENT_ADJUSTED_FEATURE_COLUMNS = _team_state_features(
    tuple(derived for _, derived in PBP_OPPONENT_ADJUSTMENT_METRICS)
)


FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "market": ("spread_line", "total_line"),
    "context": (
        "rest_diff",
        "neutral_site",
        "div_game",
        "temp",
        "wind",
        "week_sin",
        "week_cos",
    ),
    "elo": ("elo_diff", "elo_home_win_prob"),
    "experience": ("home_team_games", "away_team_games"),
    "offense": _team_state_features(STATE_METRICS[:7]),
    "results": _team_state_features(STATE_METRICS[7:9]),
    "defense": _team_state_features(STATE_METRICS[9:]),
    "pbp": _team_state_features(PBP_STATE_METRICS),
    "pbp_opponent_adjusted": PBP_OPPONENT_ADJUSTED_FEATURE_COLUMNS,
    "drive": _team_state_features(DRIVE_STATE_METRICS),
    "quarterback": _team_state_features(QB_STATE_METRICS),
    "quarterback_depth": _difference_features(QB_DEPTH_STATE_METRICS),
    "player_qb": _difference_features(PLAYER_QB_STATE_METRICS),
    "player_injuries": _difference_features(PLAYER_INJURY_STATE_METRICS),
    "player_continuity": _difference_features(PLAYER_CONTINUITY_STATE_METRICS),
    "roster_returning_snaps": _difference_features(ROSTER_RETURNING_SNAP_STATE_METRICS),
    "player_values": _difference_features(PLAYER_VALUE_STATE_METRICS),
    "player_participation_values": _difference_features(PLAYER_PARTICIPATION_STATE_METRICS),
    "graph": GRAPH_FEATURE_COLUMNS[:8],
    "schedule_rating": GRAPH_FEATURE_COLUMNS[8:],
    "bias": BIAS_FEATURE_COLUMNS,
    "surface_switch": SURFACE_SWITCH_FEATURE_COLUMNS,
    "travel_geometry": TRAVEL_GEOMETRY_FEATURE_COLUMNS,
    "rest_context": REST_CONTEXT_FEATURE_COLUMNS,
    "player_values_js_prior": _difference_features(PLAYER_VALUE_JS_PRIOR_STATE_METRICS),
    "gap_v3_bias": GAP_V3_BIAS_FEATURE_COLUMNS,
    "gap_v3_penalty": GAP_V3_PENALTY_FEATURE_COLUMNS,
    "gap_v3_travel": GAP_V3_TRAVEL_FEATURE_COLUMNS,
    "forecast_weather": FORECAST_WEATHER_FEATURE_COLUMNS,
    "observed_weather": OBSERVED_WEATHER_FEATURE_COLUMNS,
    "graph_team_stat_off_sack_rate": GRAPH_TEAM_STAT_OFF_SACK_RATE_FEATURE_COLUMNS,
    "graph_team_stat_def_yards_per_play": GRAPH_TEAM_STAT_DEF_YARDS_PER_PLAY_FEATURE_COLUMNS,
    "graph_team_stat_off_rush_epa_per_play": GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS,
    "fluview_home_elevated_on_production": FLUVIEW_HOME_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS,
    "fluview_away_elevated_on_production": FLUVIEW_AWAY_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS,
    "illness_away_active_ge1_on_production": (
        ILLNESS_AWAY_ACTIVE_GE1_ON_PRODUCTION_FEATURE_COLUMNS
    ),
    "illness_home_ge2_on_production": ILLNESS_HOME_GE2_ON_PRODUCTION_FEATURE_COLUMNS,
    "reddit_home_ratio_elevated_on_production": (
        REDDIT_HOME_RATIO_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS
    ),
    "reddit_away_spike_on_production": REDDIT_AWAY_SPIKE_ON_PRODUCTION_FEATURE_COLUMNS,
    "team_style_pace_mismatch_on_production": (
        TEAM_STYLE_PACE_MISMATCH_ON_PRODUCTION_FEATURE_COLUMNS
    ),
    "redzone_third_down_over_fade_on_production": (
        REDZONE_THIRD_DOWN_OVER_FADE_ON_PRODUCTION_FEATURE_COLUMNS
    ),
    "post_ot_fatigue_on_production": POST_OT_FATIGUE_ON_PRODUCTION_FEATURE_COLUMNS,
    "mnf_road_short_week_on_production": MNF_ROAD_SHORT_WEEK_ON_PRODUCTION_FEATURE_COLUMNS,
    "home_thursday_on_production": HOME_THURSDAY_ON_PRODUCTION_FEATURE_COLUMNS,
}

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "market": FEATURE_FAMILIES["market"],
    "market_context": FEATURE_FAMILIES["market"] + FEATURE_FAMILIES["context"],
    "market_elo": FEATURE_FAMILIES["market"] + FEATURE_FAMILIES["elo"],
    "football": (
        FEATURE_FAMILIES["context"]
        + FEATURE_FAMILIES["elo"]
        + FEATURE_FAMILIES["experience"]
        + FEATURE_FAMILIES["offense"]
        + FEATURE_FAMILIES["results"]
        + FEATURE_FAMILIES["defense"]
    ),
    "full_without_ats": (
        FEATURE_FAMILIES["market"]
        + FEATURE_FAMILIES["context"]
        + FEATURE_FAMILIES["elo"]
        + FEATURE_FAMILIES["experience"]
        + FEATURE_FAMILIES["offense"]
        + tuple(column for column in FEATURE_FAMILIES["results"] if "ats_residual" not in column)
        + FEATURE_FAMILIES["defense"]
    ),
}


def model_feature_columns() -> list[str]:
    """Return the explicit model allowlist.

    Labels, scores, final margins, identifiers, and kickoff timestamps are
    intentionally absent. Adding a model input requires changing this function
    and the accompanying leakage tests.
    """

    columns = list(SCHEDULE_FEATURES)
    for metric in STATE_METRICS:
        columns.extend((f"home_{metric}", f"away_{metric}", f"diff_{metric}"))
    return columns


MODEL_FEATURE_COLUMNS = tuple(model_feature_columns())
FEATURE_SETS["full"] = MODEL_FEATURE_COLUMNS
FEATURE_SETS["graph"] = FEATURE_FAMILIES["graph"]
FEATURE_SETS["schedule_rating"] = FEATURE_FAMILIES["schedule_rating"]
FEATURE_SETS["market_graph"] = FEATURE_SETS["market_context"] + FEATURE_FAMILIES["graph"]
FEATURE_SETS["market_schedule"] = (
    FEATURE_SETS["market_context"] + FEATURE_FAMILIES["schedule_rating"]
)
FEATURE_SETS["market_graph_schedule"] = (
    FEATURE_SETS["market_context"] + FEATURE_FAMILIES["graph"] + FEATURE_FAMILIES["schedule_rating"]
)
FEATURE_SETS["football_graph"] = FEATURE_SETS["football"] + FEATURE_FAMILIES["graph"]
FEATURE_SETS["football_schedule"] = FEATURE_SETS["football"] + FEATURE_FAMILIES["schedule_rating"]
FEATURE_SETS["football_graph_schedule"] = (
    FEATURE_SETS["football"] + FEATURE_FAMILIES["graph"] + FEATURE_FAMILIES["schedule_rating"]
)
FEATURE_SETS["full_graph"] = FEATURE_SETS["full"] + FEATURE_FAMILIES["graph"]
FEATURE_SETS["full_schedule"] = FEATURE_SETS["full"] + FEATURE_FAMILIES["schedule_rating"]
FEATURE_SETS["full_graph_schedule"] = (
    FEATURE_SETS["full"] + FEATURE_FAMILIES["graph"] + FEATURE_FAMILIES["schedule_rating"]
)
FEATURE_SETS["football_pbp"] = FEATURE_SETS["football"] + FEATURE_FAMILIES["pbp"]
FEATURE_SETS["full_pbp"] = FEATURE_SETS["full"] + FEATURE_FAMILIES["pbp"]
FEATURE_SETS["football_pbp_adjusted"] = (
    FEATURE_SETS["football_pbp"] + FEATURE_FAMILIES["pbp_opponent_adjusted"]
)
FEATURE_SETS["full_pbp_adjusted"] = (
    FEATURE_SETS["full_pbp"] + FEATURE_FAMILIES["pbp_opponent_adjusted"]
)
FEATURE_SETS["football_drive"] = FEATURE_SETS["football_pbp"] + FEATURE_FAMILIES["drive"]
FEATURE_SETS["full_drive"] = FEATURE_SETS["full_pbp"] + FEATURE_FAMILIES["drive"]
FEATURE_SETS["market_player_qb"] = FEATURE_SETS["market_context"] + FEATURE_FAMILIES["player_qb"]
FEATURE_SETS["market_player_injuries"] = (
    FEATURE_SETS["market_context"] + FEATURE_FAMILIES["player_injuries"]
)
FEATURE_SETS["market_player_continuity"] = (
    FEATURE_SETS["market_context"] + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["market_player_qb_injuries"] = (
    FEATURE_SETS["market_context"]
    + FEATURE_FAMILIES["player_qb"]
    + FEATURE_FAMILIES["player_injuries"]
)
FEATURE_SETS["market_player_qb_continuity"] = (
    FEATURE_SETS["market_context"]
    + FEATURE_FAMILIES["player_qb"]
    + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["market_player_injuries_continuity"] = (
    FEATURE_SETS["market_context"]
    + FEATURE_FAMILIES["player_injuries"]
    + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["market_player"] = (
    FEATURE_SETS["market_context"]
    + FEATURE_FAMILIES["player_qb"]
    + FEATURE_FAMILIES["player_injuries"]
    + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["football_player_qb"] = FEATURE_SETS["football"] + FEATURE_FAMILIES["player_qb"]
FEATURE_SETS["full_player_qb"] = FEATURE_SETS["full"] + FEATURE_FAMILIES["player_qb"]
FEATURE_SETS["football_player_injuries"] = (
    FEATURE_SETS["football"] + FEATURE_FAMILIES["player_injuries"]
)
FEATURE_SETS["full_player_injuries"] = FEATURE_SETS["full"] + FEATURE_FAMILIES["player_injuries"]
FEATURE_SETS["football_player_continuity"] = (
    FEATURE_SETS["football"] + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["full_player_continuity"] = (
    FEATURE_SETS["full"] + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["football_player_qb_injuries"] = (
    FEATURE_SETS["football"] + FEATURE_FAMILIES["player_qb"] + FEATURE_FAMILIES["player_injuries"]
)
FEATURE_SETS["full_player_qb_injuries"] = (
    FEATURE_SETS["full"] + FEATURE_FAMILIES["player_qb"] + FEATURE_FAMILIES["player_injuries"]
)
FEATURE_SETS["football_player_qb_continuity"] = (
    FEATURE_SETS["football"] + FEATURE_FAMILIES["player_qb"] + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["full_player_qb_continuity"] = (
    FEATURE_SETS["full"] + FEATURE_FAMILIES["player_qb"] + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["football_player_injuries_continuity"] = (
    FEATURE_SETS["football"]
    + FEATURE_FAMILIES["player_injuries"]
    + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["full_player_injuries_continuity"] = (
    FEATURE_SETS["full"]
    + FEATURE_FAMILIES["player_injuries"]
    + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["football_player"] = (
    FEATURE_SETS["football"]
    + FEATURE_FAMILIES["player_qb"]
    + FEATURE_FAMILIES["player_injuries"]
    + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["full_player"] = (
    FEATURE_SETS["full"]
    + FEATURE_FAMILIES["player_qb"]
    + FEATURE_FAMILIES["player_injuries"]
    + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["football_player_injury_value"] = (
    FEATURE_SETS["football"]
    + FEATURE_FAMILIES["player_injuries"]
    + FEATURE_FAMILIES["player_values"]
)
FEATURE_SETS["full_player_injury_value"] = (
    FEATURE_SETS["full"] + FEATURE_FAMILIES["player_injuries"] + FEATURE_FAMILIES["player_values"]
)
FEATURE_SETS["football_player_value"] = (
    FEATURE_SETS["football_player"] + FEATURE_FAMILIES["player_values"]
)
FEATURE_SETS["full_player_value"] = FEATURE_SETS["full_player"] + FEATURE_FAMILIES["player_values"]
FEATURE_SETS["football_player_participation"] = (
    FEATURE_SETS["football_player_value"] + FEATURE_FAMILIES["player_participation_values"]
)
FEATURE_SETS["full_player_participation"] = (
    FEATURE_SETS["full_player_value"] + FEATURE_FAMILIES["player_participation_values"]
)
FEATURE_SETS["football_weak_stack"] = (
    FEATURE_SETS["football_player_value"] + FEATURE_FAMILIES["bias"]
)
FEATURE_SETS["full_weak_stack"] = FEATURE_SETS["full_player_value"] + FEATURE_FAMILIES["bias"]
ROSTER_CONTINUITY_DATA_AVAILABLE = "roster_continuity_data_available"
FEATURE_FAMILIES["roster_continuity_source_availability"] = (ROSTER_CONTINUITY_DATA_AVAILABLE,)
FEATURE_SETS["football_weak_stack_source_availability"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["roster_continuity_source_availability"]
)
FEATURE_SETS["full_weak_stack_source_availability"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["roster_continuity_source_availability"]
)
FEATURE_SETS["football_weak_stack_surface"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["surface_switch"]
)
FEATURE_SETS["full_weak_stack_surface"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["surface_switch"]
)
FEATURE_SETS["football_weak_stack_js_prior"] = (
    tuple(
        column
        for column in FEATURE_SETS["football_weak_stack"]
        if column not in FEATURE_FAMILIES["player_values"]
    )
    + FEATURE_FAMILIES["player_values_js_prior"]
)
FEATURE_SETS["full_weak_stack_js_prior"] = (
    tuple(
        column
        for column in FEATURE_SETS["full_weak_stack"]
        if column not in FEATURE_FAMILIES["player_values"]
    )
    + FEATURE_FAMILIES["player_values_js_prior"]
)
FEATURE_SETS["football_weak_stack_v3"] = (
    FEATURE_SETS["football_weak_stack_surface"]
    + FEATURE_FAMILIES["gap_v3_bias"]
    + FEATURE_FAMILIES["gap_v3_penalty"]
    + FEATURE_FAMILIES["gap_v3_travel"]
)
FEATURE_SETS["full_weak_stack_v3"] = (
    FEATURE_SETS["full_weak_stack_surface"]
    + FEATURE_FAMILIES["gap_v3_bias"]
    + FEATURE_FAMILIES["gap_v3_penalty"]
    + FEATURE_FAMILIES["gap_v3_travel"]
)
FEATURE_SETS["football_weak_stack_v4"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["forecast_weather"]
)
FEATURE_SETS["full_weak_stack_v4"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["forecast_weather"]
)
FEATURE_SETS["football_weak_stack_oracle_weather"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["observed_weather"]
)
FEATURE_SETS["full_weak_stack_oracle_weather"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["observed_weather"]
)
FEATURE_SETS["football_weak_stack_graph_sack"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_off_sack_rate"]
)
FEATURE_SETS["full_weak_stack_graph_sack"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_off_sack_rate"]
)

FEATURE_SETS["football_weak_stack_graph_def_ypp"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_def_yards_per_play"]
)
FEATURE_SETS["full_weak_stack_graph_def_ypp"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_def_yards_per_play"]
)

FEATURE_SETS["football_weak_stack_graph_off_rush_epa"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_off_rush_epa_per_play"]
)
FEATURE_SETS["full_weak_stack_graph_off_rush_epa"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_off_rush_epa_per_play"]
)

FEATURE_SETS["football_weak_stack_fluview_home"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["fluview_home_elevated_on_production"]
)
FEATURE_SETS["full_weak_stack_fluview_home"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["fluview_home_elevated_on_production"]
)
FEATURE_SETS["football_weak_stack_fluview_away"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["fluview_away_elevated_on_production"]
)
FEATURE_SETS["full_weak_stack_fluview_away"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["fluview_away_elevated_on_production"]
)

FEATURE_SETS["football_weak_stack_illness_away"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["illness_away_active_ge1_on_production"]
)
FEATURE_SETS["full_weak_stack_illness_away"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["illness_away_active_ge1_on_production"]
)
FEATURE_SETS["football_weak_stack_illness_home"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["illness_home_ge2_on_production"]
)
FEATURE_SETS["full_weak_stack_illness_home"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["illness_home_ge2_on_production"]
)
FEATURE_SETS["football_weak_stack_reddit_ratio_home"] = (
    FEATURE_SETS["football_weak_stack"]
    + FEATURE_FAMILIES["reddit_home_ratio_elevated_on_production"]
)
FEATURE_SETS["full_weak_stack_reddit_ratio_home"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["reddit_home_ratio_elevated_on_production"]
)
FEATURE_SETS["football_weak_stack_reddit_spike_away"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["reddit_away_spike_on_production"]
)
FEATURE_SETS["full_weak_stack_reddit_spike_away"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["reddit_away_spike_on_production"]
)
FEATURE_SETS["football_weak_stack_team_style_pace"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["team_style_pace_mismatch_on_production"]
)
FEATURE_SETS["full_weak_stack_team_style_pace"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["team_style_pace_mismatch_on_production"]
)
FEATURE_SETS["football_weak_stack_redzone_third_down"] = (
    FEATURE_SETS["football_weak_stack"]
    + FEATURE_FAMILIES["redzone_third_down_over_fade_on_production"]
)
FEATURE_SETS["full_weak_stack_redzone_third_down"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["redzone_third_down_over_fade_on_production"]
)

FEATURE_SETS["football_weak_stack_post_ot"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["post_ot_fatigue_on_production"]
)
FEATURE_SETS["full_weak_stack_post_ot"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["post_ot_fatigue_on_production"]
)
FEATURE_SETS["football_weak_stack_mnf_road"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["mnf_road_short_week_on_production"]
)
FEATURE_SETS["full_weak_stack_mnf_road"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["mnf_road_short_week_on_production"]
)
FEATURE_SETS["football_weak_stack_home_thursday"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["home_thursday_on_production"]
)
FEATURE_SETS["full_weak_stack_home_thursday"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["home_thursday_on_production"]
)

PER13_DURABILITY_SUFFIX = "_durability"
PER13_DURABILITY_SWAPPED_BASE_COLUMNS = (
    FEATURE_FAMILIES["player_injuries"] + FEATURE_FAMILIES["player_values"]
)
PER13_DURABILITY_INJURY_FEATURE_COLUMNS = tuple(
    f"{column}{PER13_DURABILITY_SUFFIX}" for column in FEATURE_FAMILIES["player_injuries"]
)
PER13_DURABILITY_VALUE_FEATURE_COLUMNS = tuple(
    f"{column}{PER13_DURABILITY_SUFFIX}" for column in FEATURE_FAMILIES["player_values"]
)
PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS = (
    PER13_DURABILITY_INJURY_FEATURE_COLUMNS + PER13_DURABILITY_VALUE_FEATURE_COLUMNS
)
FEATURE_FAMILIES["player_injuries_durability"] = PER13_DURABILITY_INJURY_FEATURE_COLUMNS
FEATURE_FAMILIES["player_values_durability"] = PER13_DURABILITY_VALUE_FEATURE_COLUMNS
FEATURE_SETS["football_weak_stack_durability"] = (
    tuple(
        column
        for column in FEATURE_SETS["football_weak_stack"]
        if column not in PER13_DURABILITY_SWAPPED_BASE_COLUMNS
    )
    + PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_SETS["full_weak_stack_durability"] = (
    tuple(
        column
        for column in FEATURE_SETS["full_weak_stack"]
        if column not in PER13_DURABILITY_SWAPPED_BASE_COLUMNS
    )
    + PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS
)

IDENTIFIER_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "home_team",
)

OUTCOME_COLUMNS = (
    "result",
    "ats_margin",
    "home_cover",
    "away_score",
    "home_score",
)


OPENER_SOFTNESS_FADE_ON_PRODUCTION_FEATURE_COLUMNS = ("opener_softness_fade_signal",)

ML_SPREAD_DIVERGENCE_ON_PRODUCTION_FEATURE_COLUMNS = ("ml_spread_divergence_signal",)

FEATURE_FAMILIES["opener_softness_fade_on_production"] = (
    OPENER_SOFTNESS_FADE_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["ml_spread_divergence_on_production"] = (
    ML_SPREAD_DIVERGENCE_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_opener_softness"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["opener_softness_fade_on_production"]
)
FEATURE_SETS["full_weak_stack_opener_softness"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["opener_softness_fade_on_production"]
)
FEATURE_SETS["football_weak_stack_ml_divergence"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["ml_spread_divergence_on_production"]
)
FEATURE_SETS["full_weak_stack_ml_divergence"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["ml_spread_divergence_on_production"]
)


FEATURE_FAMILIES["new_stadium_home_on_production"] = NEW_STADIUM_HOME_ON_PRODUCTION_FEATURE_COLUMNS
FEATURE_FAMILIES["dome_shootout_favorite_on_production"] = (
    DOME_SHOOTOUT_FAVORITE_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["low_total_div_home_dog_on_production"] = (
    LOW_TOTAL_DIV_HOME_DOG_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["sept_heat_home_on_production"] = SEPT_HEAT_HOME_ON_PRODUCTION_FEATURE_COLUMNS

FEATURE_SETS["football_weak_stack_new_stadium"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["new_stadium_home_on_production"]
)
FEATURE_SETS["full_weak_stack_new_stadium"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["new_stadium_home_on_production"]
)
FEATURE_SETS["football_weak_stack_dome_shootout"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["dome_shootout_favorite_on_production"]
)
FEATURE_SETS["full_weak_stack_dome_shootout"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["dome_shootout_favorite_on_production"]
)
FEATURE_SETS["football_weak_stack_low_total_div_dog"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["low_total_div_home_dog_on_production"]
)
FEATURE_SETS["full_weak_stack_low_total_div_dog"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["low_total_div_home_dog_on_production"]
)
FEATURE_SETS["football_weak_stack_sept_heat"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["sept_heat_home_on_production"]
)
FEATURE_SETS["full_weak_stack_sept_heat"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["sept_heat_home_on_production"]
)


ROAD_FAV_BIG_FADE_ON_PRODUCTION_FEATURE_COLUMNS = ("road_fav_big_fade_flag",)
DIVISION_DOG_ON_PRODUCTION_FEATURE_COLUMNS = ("division_dog_flag",)
WEEK1_DOG_ON_PRODUCTION_FEATURE_COLUMNS = ("week1_dog_flag",)
ATS_STREAK_REGRESS_ON_PRODUCTION_FEATURE_COLUMNS = ("ats_streak_regress_flag",)

FEATURE_FAMILIES["road_fav_big_fade_on_production"] = (
    ROAD_FAV_BIG_FADE_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["division_dog_on_production"] = DIVISION_DOG_ON_PRODUCTION_FEATURE_COLUMNS
FEATURE_FAMILIES["week1_dog_on_production"] = WEEK1_DOG_ON_PRODUCTION_FEATURE_COLUMNS
FEATURE_FAMILIES["ats_streak_regress_on_production"] = (
    ATS_STREAK_REGRESS_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_road_fav_fade"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["road_fav_big_fade_on_production"]
)
FEATURE_SETS["full_weak_stack_road_fav_fade"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["road_fav_big_fade_on_production"]
)
FEATURE_SETS["football_weak_stack_division_dog"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["division_dog_on_production"]
)
FEATURE_SETS["full_weak_stack_division_dog"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["division_dog_on_production"]
)
FEATURE_SETS["football_weak_stack_week1_dog"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["week1_dog_on_production"]
)
FEATURE_SETS["full_weak_stack_week1_dog"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["week1_dog_on_production"]
)
FEATURE_SETS["football_weak_stack_ats_streak_regress"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["ats_streak_regress_on_production"]
)
FEATURE_SETS["full_weak_stack_ats_streak_regress"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["ats_streak_regress_on_production"]
)


OPENING_DRIVE_EPA_ON_PRODUCTION_FEATURE_COLUMNS = ("opening_drive_epa",)
Q3_POINT_DIFF_ON_PRODUCTION_FEATURE_COLUMNS = ("q3_point_diff",)
FOURTH_DOWN_INTERACTION_ON_PRODUCTION_FEATURE_COLUMNS = ("fourth_down_interaction",)

FEATURE_FAMILIES["opening_drive_script_on_production"] = (
    OPENING_DRIVE_EPA_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["q3_adjustment_on_production"] = Q3_POINT_DIFF_ON_PRODUCTION_FEATURE_COLUMNS
FEATURE_FAMILIES["fourth_down_aggression_interaction_on_production"] = (
    FOURTH_DOWN_INTERACTION_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_opening_drive_epa"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["opening_drive_script_on_production"]
)
FEATURE_SETS["full_weak_stack_opening_drive_epa"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["opening_drive_script_on_production"]
)
FEATURE_SETS["football_weak_stack_q3_point_diff"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["q3_adjustment_on_production"]
)
FEATURE_SETS["full_weak_stack_q3_point_diff"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["q3_adjustment_on_production"]
)
FEATURE_SETS["football_weak_stack_fourth_down_interaction"] = (
    FEATURE_SETS["football_weak_stack"]
    + FEATURE_FAMILIES["fourth_down_aggression_interaction_on_production"]
)
FEATURE_SETS["full_weak_stack_fourth_down_interaction"] = (
    FEATURE_SETS["full_weak_stack"]
    + FEATURE_FAMILIES["fourth_down_aggression_interaction_on_production"]
)


ROOKIE_QB_DEBUT_FADE_ON_PRODUCTION_FEATURE_COLUMNS = ("rookie_qb_debut_fade_flag",)
QB_REVENGE_ON_PRODUCTION_FEATURE_COLUMNS = ("qb_revenge_flag",)

FEATURE_FAMILIES["rookie_qb_debut_fade_on_production"] = (
    ROOKIE_QB_DEBUT_FADE_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["qb_revenge_on_production"] = QB_REVENGE_ON_PRODUCTION_FEATURE_COLUMNS

FEATURE_SETS["football_weak_stack_rookie_qb_debut_fade"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["rookie_qb_debut_fade_on_production"]
)
FEATURE_SETS["full_weak_stack_rookie_qb_debut_fade"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["rookie_qb_debut_fade_on_production"]
)
FEATURE_SETS["football_weak_stack_qb_revenge"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["qb_revenge_on_production"]
)
FEATURE_SETS["full_weak_stack_qb_revenge"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["qb_revenge_on_production"]
)


HOLDOUT_SLOW_START_ON_PRODUCTION_FEATURE_COLUMNS = ("holdout_slow_start_flag",)
DEADLINE_INTEGRATION_DRAG_ON_PRODUCTION_FEATURE_COLUMNS = ("deadline_integration_drag_flag",)
SUSPENSION_RETURN_RUST_ON_PRODUCTION_FEATURE_COLUMNS = ("suspension_return_rust_flag",)

FEATURE_FAMILIES["holdout_slow_start_on_production"] = (
    HOLDOUT_SLOW_START_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["deadline_integration_drag_on_production"] = (
    DEADLINE_INTEGRATION_DRAG_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["suspension_return_rust_on_production"] = (
    SUSPENSION_RETURN_RUST_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_holdout_slow_start"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["holdout_slow_start_on_production"]
)
FEATURE_SETS["full_weak_stack_holdout_slow_start"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["holdout_slow_start_on_production"]
)
FEATURE_SETS["football_weak_stack_deadline_drag"] = (
    FEATURE_SETS["football_weak_stack"]
    + FEATURE_FAMILIES["deadline_integration_drag_on_production"]
)
FEATURE_SETS["full_weak_stack_deadline_drag"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["deadline_integration_drag_on_production"]
)
FEATURE_SETS["football_weak_stack_suspension_rust"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["suspension_return_rust_on_production"]
)
FEATURE_SETS["full_weak_stack_suspension_rust"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["suspension_return_rust_on_production"]
)


CREW_SECOND_MEETING_FAVORITE_ON_PRODUCTION_FEATURE_COLUMNS = ("crew_second_meeting_favorite_flag",)
ROOKIE_CREW_UNDERDOG_ON_PRODUCTION_FEATURE_COLUMNS = ("rookie_crew_underdog_flag",)

FEATURE_FAMILIES["crew_second_meeting_favorite_on_production"] = (
    CREW_SECOND_MEETING_FAVORITE_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["rookie_crew_underdog_on_production"] = (
    ROOKIE_CREW_UNDERDOG_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_crew_second_meeting_favorite"] = (
    FEATURE_SETS["football_weak_stack"]
    + FEATURE_FAMILIES["crew_second_meeting_favorite_on_production"]
)
FEATURE_SETS["full_weak_stack_crew_second_meeting_favorite"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["crew_second_meeting_favorite_on_production"]
)
FEATURE_SETS["football_weak_stack_rookie_crew_underdog"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["rookie_crew_underdog_on_production"]
)
FEATURE_SETS["full_weak_stack_rookie_crew_underdog"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["rookie_crew_underdog_on_production"]
)


OPEN_CORNER_WIND_DOG_ON_PRODUCTION_FEATURE_COLUMNS = ("open_corner_wind_dog_flag",)
RAIN_ON_GRASS_DOG_ON_PRODUCTION_FEATURE_COLUMNS = ("rain_on_grass_dog_flag",)

FEATURE_FAMILIES["open_corner_wind_dog_on_production"] = (
    OPEN_CORNER_WIND_DOG_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["rain_on_grass_dog_on_production"] = (
    RAIN_ON_GRASS_DOG_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_open_corner_wind_dog"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["open_corner_wind_dog_on_production"]
)
FEATURE_SETS["full_weak_stack_open_corner_wind_dog"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["open_corner_wind_dog_on_production"]
)
FEATURE_SETS["football_weak_stack_rain_on_grass_dog"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["rain_on_grass_dog_on_production"]
)
FEATURE_SETS["full_weak_stack_rain_on_grass_dog"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["rain_on_grass_dog_on_production"]
)


FEATURE_SETS["football_weak_stack_qb_revenge_deadline_drag"] = (
    FEATURE_SETS["football_weak_stack"]
    + FEATURE_FAMILIES["qb_revenge_on_production"]
    + FEATURE_FAMILIES["deadline_integration_drag_on_production"]
)
FEATURE_SETS["full_weak_stack_qb_revenge_deadline_drag"] = (
    FEATURE_SETS["full_weak_stack"]
    + FEATURE_FAMILIES["qb_revenge_on_production"]
    + FEATURE_FAMILIES["deadline_integration_drag_on_production"]
)


IR_RETURN_REINFORCEMENT_ON_PRODUCTION_FEATURE_COLUMNS = ("ir_return_reinforcement_flag",)
SPECIALIST_ABSENCE_FADE_ON_PRODUCTION_FEATURE_COLUMNS = ("specialist_absence_fade_flag",)

FEATURE_FAMILIES["ir_return_bump_on_production"] = (
    IR_RETURN_REINFORCEMENT_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["specialist_absence_fade_on_production"] = (
    SPECIALIST_ABSENCE_FADE_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_ir_return_reinforcement"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["ir_return_bump_on_production"]
)
FEATURE_SETS["full_weak_stack_ir_return_reinforcement"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["ir_return_bump_on_production"]
)
FEATURE_SETS["football_weak_stack_specialist_absence_fade"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["specialist_absence_fade_on_production"]
)
FEATURE_SETS["full_weak_stack_specialist_absence_fade"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["specialist_absence_fade_on_production"]
)


ROOKIE_WALL_DEPENDENCE_ON_PRODUCTION_FEATURE_COLUMNS = ("rookie_wall_dependence_fade_flag",)
KICKER_CHANGE_UNDERDOG_ON_PRODUCTION_FEATURE_COLUMNS = ("kicker_change_underdog_flag",)

FEATURE_FAMILIES["rookie_wall_dependence_on_production"] = (
    ROOKIE_WALL_DEPENDENCE_ON_PRODUCTION_FEATURE_COLUMNS
)
FEATURE_FAMILIES["kicker_change_underdog_on_production"] = (
    KICKER_CHANGE_UNDERDOG_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_rookie_wall_dependence"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["rookie_wall_dependence_on_production"]
)
FEATURE_SETS["full_weak_stack_rookie_wall_dependence"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["rookie_wall_dependence_on_production"]
)
FEATURE_SETS["football_weak_stack_kicker_change_underdog"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["kicker_change_underdog_on_production"]
)
FEATURE_SETS["full_weak_stack_kicker_change_underdog"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["kicker_change_underdog_on_production"]
)


BACKUP_TENURE_GAP_ON_PRODUCTION_FEATURE_COLUMNS = ("backup_tenure_gap_flag",)

FEATURE_FAMILIES["backup_tenure_gap_on_production"] = (
    BACKUP_TENURE_GAP_ON_PRODUCTION_FEATURE_COLUMNS
)

FEATURE_SETS["football_weak_stack_backup_tenure_gap"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["backup_tenure_gap_on_production"]
)
FEATURE_SETS["full_weak_stack_backup_tenure_gap"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["backup_tenure_gap_on_production"]
)

FEATURE_FAMILIES["fluview_away_ili_asof_on_production"] = ("fluview_away_market_ili_asof",)
FEATURE_FAMILIES["apm_unit_on_production"] = (
    "home_apm_off_rating",
    "home_apm_def_rating",
    "away_apm_off_rating",
    "away_apm_def_rating",
    "apm_off_rating_diff",
    "apm_def_rating_diff",
)

from nfl_ats.spread_regime import SPREAD_REGIME_COLUMNS  # noqa: E402

FEATURE_FAMILIES["spread_regime"] = SPREAD_REGIME_COLUMNS
for _regime_prefix in ("football", "full"):
    FEATURE_SETS[f"{_regime_prefix}_weak_stack_spread_regime"] = (
        *FEATURE_SETS[f"{_regime_prefix}_weak_stack"],
        *SPREAD_REGIME_COLUMNS,
    )

from nfl_ats.home_dog_location import (  # noqa: E402
    HOME_DOG_HINGE_COLUMNS,
    HOME_DOG_POINTS_COLUMNS,
)

FEATURE_FAMILIES["home_dog_location_points"] = HOME_DOG_POINTS_COLUMNS
FEATURE_FAMILIES["home_dog_location_hinge"] = HOME_DOG_HINGE_COLUMNS[1:]
for _home_dog_prefix in ("football", "full"):
    FEATURE_SETS[f"{_home_dog_prefix}_weak_stack_home_dog_points"] = (
        *FEATURE_SETS[f"{_home_dog_prefix}_weak_stack"],
        *HOME_DOG_POINTS_COLUMNS,
    )
    FEATURE_SETS[f"{_home_dog_prefix}_weak_stack_home_dog_hinge_7"] = (
        *FEATURE_SETS[f"{_home_dog_prefix}_weak_stack"],
        *HOME_DOG_HINGE_COLUMNS,
    )

from nfl_ats.home_side_location import HOME_SIDE_HINGE_COLUMNS  # noqa: E402

FEATURE_FAMILIES["home_side_location_hinge"] = HOME_SIDE_HINGE_COLUMNS
for _home_side_prefix in ("football", "full"):
    FEATURE_SETS[f"{_home_side_prefix}_weak_stack_home_side_hinge_7"] = (
        *FEATURE_SETS[f"{_home_side_prefix}_weak_stack"],
        *HOME_SIDE_HINGE_COLUMNS,
    )


FEATURE_FAMILIES["per07_coord_change_on_production"] = (
    "coord_new_oc_diff",
    "coord_new_dc_diff",
    "coord_new_hc_diff",
)
for _coord_prefix in ("football", "full"):
    FEATURE_SETS[f"{_coord_prefix}_weak_stack_coord_change"] = (
        *FEATURE_SETS[f"{_coord_prefix}_weak_stack"],
        *FEATURE_FAMILIES["per07_coord_change_on_production"],
    )
