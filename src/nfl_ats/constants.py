"""Shared schema and feature definitions."""

from __future__ import annotations

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

# nflverse schedules retain some historical abbreviations while its current
# team-stat feeds use stable franchise IDs. Feature state must use one identity
# across both sources and across relocations.
TEAM_ABBREVIATION_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
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

# Each value is measured from one team's perspective in a completed game. The
# feature builder turns it into an exponentially weighted, pregame team state.
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

# Play-by-play states live in the optional enriched feature table. Keeping the
# names explicit makes changes to the play filter or aggregation contract
# reviewable instead of silently changing a model's inputs.
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

# Each source metric is modeled as an offense effect plus an opposing-defense
# effect at a weekly cutoff. The derived name describes the expected result of
# the actual matchup, rather than either team's unadjusted rolling average.
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

# These conservative player features are derived only from earlier-game snaps,
# earlier-week roster rows, and injury revisions visible at the declared
# decision timestamp. They are kept in small research families so QB,
# availability, and continuity signal can be admitted or rejected separately.
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
PLAYER_STATE_METRICS = (
    PLAYER_QB_STATE_METRICS
    + PLAYER_INJURY_STATE_METRICS
    + PLAYER_CONTINUITY_STATE_METRICS
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
    "player_qb": _difference_features(PLAYER_QB_STATE_METRICS),
    "player_injuries": _difference_features(PLAYER_INJURY_STATE_METRICS),
    "player_continuity": _difference_features(PLAYER_CONTINUITY_STATE_METRICS),
    "graph": GRAPH_FEATURE_COLUMNS[:8],
    "schedule_rating": GRAPH_FEATURE_COLUMNS[8:],
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
FEATURE_SETS["market_player"] = (
    FEATURE_SETS["market_context"]
    + FEATURE_FAMILIES["player_qb"]
    + FEATURE_FAMILIES["player_injuries"]
    + FEATURE_FAMILIES["player_continuity"]
)
FEATURE_SETS["football_player_qb"] = FEATURE_SETS["football"] + FEATURE_FAMILIES["player_qb"]
FEATURE_SETS["full_player_qb"] = FEATURE_SETS["full"] + FEATURE_FAMILIES["player_qb"]
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
