"""Shared schema, feature, and walk-forward definitions."""

from __future__ import annotations

# Smallest training set `margin.fit_margin_model` can actually fit. This is
# DERIVED, and exactly, from that function's own preconditions rather than
# chosen: it requires len(training) >= 50, int(0.20 * n) >= min_distribution_rows
# (10), and n - int(0.20 * n) >= 40. All three are first satisfied together at
# n = 50 and the middle one fails at 49 (int(9.8) = 9), so 50 is the true
# feasibility boundary. Below it the estimator raises; at or above it, it fits.
#
# This is a FEASIBILITY floor and answers only "can a model be produced at all".
# It deliberately says nothing about whether the fit is any good.
MIN_FITTABLE_TRAIN_GAMES = 50

# Default completed games required in front of a target week before a
# walk-forward model is fitted, used as a reporting/quality guard.
#
# STILL UNDERIVED at 500, and measurement says no derivable value exists,
# because the quantity it was meant to protect has no threshold. Measured two
# ways on 2026-08-17:
#
#   * NFL, holding warm test rows fixed (2012-2025, 3,573 games) and truncating
#     training to the most recent N: forced-pick ATS accuracy is FLAT from
#     N=50 to the full ~2,600 (.509 / .499 / .508), and every paired delta
#     straddles zero. Brier and residual MAE degrade smoothly with no cliff at
#     500 or anywhere else, and the Brier half is fully repaired by the
#     already-derived 200-row calibration floor (raw .313 -> calibrated .2504
#     at N=50).
#   * CFB, 12,500 games: the segment this floor REFUSES (train rows < 500)
#     scores 0.4906, week-blocked [0.4313, 0.5511], probability_positive 0.376
#     on 12 independent blocks -- unresolved and leaning mildly negative, not
#     demonstrably bad.
#
# The real failure mode below ~500 is OVER-CONFIDENCE rather than error: mean
# |predicted residual| is 8.84 at N=50 against 1.68 at full training. That is a
# regularization problem, not a sample-size threshold, and a cliff is the wrong
# instrument for it.
#
# What changed on 2026-08-17 is that this number no longer gates an
# irreversible decision. The rotation registry's warm-up floor -- which
# permanently determines which seasons any future family may draw -- now
# derives from MIN_FITTABLE_TRAIN_GAMES above, because rule 9 asks whether a
# window's first week can be SCORED, which is a feasibility question. 500
# survives here only as a default for reporting runs, where being conservative
# costs nothing and binds nothing (every NFL evaluation window sits over 1,000
# games clear of it).
#
# Frozen, predeclared runs pin their own FROZEN_* copies (see `experiments.py`)
# so that changing this value cannot retroactively alter a recorded artifact.
# Never point a frozen run at this constant.
DEFAULT_MIN_TRAIN_GAMES = 500

# Minimum prior out-of-sample prediction rows before the cover-probability
# stream is calibrated rather than passed through raw.
#
# DERIVED, unlike the training floor above. Measured on the real 2009-2025
# walk-forward stream by opening the gate and bucketing calibrated-vs-raw Brier
# by the history each week's calibrator actually had: 100-199 rows makes Brier
# worse (0.206 -> 0.284, on 16 games), while 200-399 already improves it
# (0.269 -> 0.250, on 204 games). 200 is the smallest demonstrated-safe value.
# See `calibration.calibrate_cover_prediction_stream` for the full note.
DEFAULT_MIN_CALIBRATION_GAMES = 200

# Fraction of a team's end-of-season state carried into the next season, after
# regressing toward the league mean:
#   current = league_mean + retention ** gap * (current - league_mean)
#
# MEASURED 2026-08-17 AND WRONG AT 0.67 -- roughly twice what the data
# supports. Three independent routes agree, none of them overlapping 0.67:
#
#   * Fitting the retention slope per metric and horizon across 486
#     season-to-season transitions: all 24 metric x horizon cells have a 95%
#     upper bound below 0.67. At the first-4-games horizon the constant
#     actually governs, the median fitted value is 0.337 (range 0.195-0.382).
#   * The shipped feature table's own behaviour: the slope of
#     `result ~ diff_point_diff` is 0.333 in week 1 against 0.588 in weeks
#     9-18, a ratio of 0.566, implying 0.67 x 0.566 = 0.379.
#   * Plain regression of next-season on prior-season point differential, both
#     centred within season: 0.400, season-blocked 95% [0.347, 0.460] over the
#     full next season and 0.475 [0.391, 0.573] over its first four games.
#
# So the honest value is around 0.35-0.45, and at 0.67 the carried state is
# inflated enough that carrying NOTHING forward beats it on full-season point
# differential (RMSE 6.16 at retention 0 vs 6.38 at 0.67, and 5.75 at ~0.30).
#
# NOT YET CHANGED, deliberately: this constant shapes the feature table, so
# moving it moves every prediction and is a scored change needing a screen on
# CFB first (free under rotation rule 8) rather than a quiet edit. It is named
# here, rather than repeated as a literal in five modules, so that the fix is
# one edit and the defect is visible while it waits. A single global value is
# itself an undefended assumption -- the fitted retention ranges from 0.195
# (`off_turnover_rate`) to 0.382 (`off_sack_rate`) -- so per-metric values
# should be considered at the same time.
#
# A shared ridge coefficient cannot absorb this. With EWM span 8 the offseason
# initial condition still carries weight 0.78/0.60/0.47/0.37 after 1-4 games,
# so it is load-bearing for roughly weeks 1-6 -- about a third of the slate --
# while the coefficient is fit mostly on late-season rows. The early-season
# state feature is over-weighted by about 1.8x and no single coefficient fixes
# a week-varying scale error.
DEFAULT_OFFSEASON_RETENTION = 0.67

# Regular-season games in a pre-2021 NFL season (16 games x 32 teams / 2). The
# schedule expanded to 272 in 2021, so this is the conservative figure for
# counting how many prior seasons a warm-up requirement consumes at the start
# of the feature table, which is the only place the question arises.
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
# The participation family is opt-in so rebuilding the established v2 player
# table without a participation snapshot preserves its exact feature contract.
PLAYER_ALL_STATE_METRICS = PLAYER_STATE_METRICS + PLAYER_VALUE_STATE_METRICS

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

# Peer-reviewed opener-bias signals (MOD-07). They are computed from the
# schedules frame alone and ride along in the canonical table, but they stay
# out of MODEL_FEATURE_COLUMNS on purpose: the frozen feature sets must keep
# their exact contract, so only an explicitly opted-in profile may read them.
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

# Surface-switch tilt candidate (docs/surface_switch_feature_arm.md), the
# feature_arm sibling of the already-live surface_switch_tilt_overlay pick-
# level challenger. Computed from schedules alone (this season's full REG
# modal home surface, a structural stadium fact fixed before Week 1 -- never
# an outcome column) and ridden along in the canonical table on the same
# BIAS_METRICS precedent: it stays out of MODEL_FEATURE_COLUMNS, so only an
# explicitly opted-in profile (weak_stack_surface) may read it.
SURFACE_SWITCH_FEATURE_COLUMNS = ("surface_switch_flag",)

# weak_stack_v4 candidate profile (docs/weak_stack_v4.md): the six continuous
# forecast-weather columns joined by game_id from the completed
# kickoff-nearest archive. weak_stack_v3 already tested fifteen hand-coded
# situational FLAGS and was refused at the opener on EV, and the registered
# forecast-weather signals are that same cell shape (the strongest fires on
# 1.51% of the slate), so this family hands ridge the RAW variables instead.
# Same BIAS_METRICS/SURFACE_SWITCH_FEATURE_COLUMNS precedent: these stay out
# of MODEL_FEATURE_COLUMNS, so only the explicitly opted-in weak_stack_v4
# profile reads them. Built in nfl_ats.forecast_weather_features.
FORECAST_WEATHER_FEATURE_COLUMNS = (
    "forecast_temp_f",
    "forecast_wind_mph",
    "forecast_precip_prob_pct",
    "forecast_is_outdoors",
    "forecast_temp_f_outdoor",
    "forecast_wind_mph_outdoor",
)

# POSITIVE CONTROL ONLY: the weather that ACTUALLY happened, which is not
# knowable before kickoff. Never a production feature -- only the opted-in
# weak_stack_oracle_weather profile reads it, and that profile exists solely
# to bound what perfect weather knowledge could ever be worth.
OBSERVED_WEATHER_FEATURE_COLUMNS = (
    "observed_temp_f",
    "observed_wind_mph",
    "observed_is_outdoors",
    "observed_temp_f_outdoor",
    "observed_wind_mph_outdoor",
)

# weak_stack_graph_sack candidate profile (docs/graph_team_stat_on_production.md):
# the ONE graph-propagated `team_stat` column that led the 38-family screen
# (docs/graph_ratings_v2_screen.md section 8, off_sack_rate, +2.949 accuracy
# points against a BARE market baseline), tested here stacked on PRODUCTION
# weak_stack instead -- the project's own "composition is not the signal"
# lesson is that a component positive alone can go negative once stacked on
# what is actually played. Computed by `add_graph_ratings_v2_features` at the
# structural configuration frozen in docs/graph_ratings_v2_screen.md section 5
# (inherited, not refit, here) and additively joined by game_id in
# nfl_ats.graph_team_stat_production_feature. Same BIAS_METRICS/
# SURFACE_SWITCH_FEATURE_COLUMNS precedent: stays out of MODEL_FEATURE_COLUMNS,
# so only the explicitly opted-in weak_stack_graph_sack profile reads it.
GRAPH_TEAM_STAT_OFF_SACK_RATE_FEATURE_COLUMNS = ("graph_v2_team_stat_off_sack_rate_katz_diff",)

# weak_stack_graph_def_ypp candidate profile (docs/graph_team_stat_def_ypp_on_production.md):
# the ONE graph-propagated `team_stat` column that LED the 38-family screen by
# the CONSERVATIVE, null-adjusted reference (docs/graph_ratings_v2_screen.md
# section 8, def_yards_per_play, 95.5th percentile of its own permutation null,
# +2.145 accuracy points against a null centred at +0.279 -- the least
# artifact-contaminated of the three cells the doc names), tested here stacked
# on PRODUCTION weak_stack instead of a bare baseline -- same "composition is
# not the signal" reasoning as weak_stack_graph_sack above. Computed by
# `add_graph_ratings_v2_features` at the SAME structural configuration frozen
# in docs/graph_ratings_v2_screen.md section 5 (inherited, not refit, here) and
# additively joined by game_id in nfl_ats.graph_team_stat_def_ypp_production_feature.
# Same BIAS_METRICS/SURFACE_SWITCH_FEATURE_COLUMNS precedent: stays out of
# MODEL_FEATURE_COLUMNS, so only the explicitly opted-in
# weak_stack_graph_def_ypp profile reads it.
GRAPH_TEAM_STAT_DEF_YARDS_PER_PLAY_FEATURE_COLUMNS = (
    "graph_v2_team_stat_def_yards_per_play_katz_diff",
)

# weak_stack_fluview_home / weak_stack_fluview_away candidate profiles
# (docs/fluview_on_production.md): the two FluView home-market illness
# indicators that led docs/fluview_battery.md's five-cell screen against a
# BARE market baseline (fluview_away_market_elevated +0.368 accuracy points
# P+ 0.883, fluview_home_market_elevated +0.309 P+ 0.818 -- both week-blocked,
# recorded 2026-08-20), tested here stacked on PRODUCTION weak_stack instead
# -- the project's own "composition is not the signal" lesson is that a
# component positive alone can go negative once stacked on what is actually
# played. Computed at the frozen as-of/threshold construction
# (nfl_ats.fluview_production_feature, reusing scripts/fluview_battery_screen.py's
# checkpoint/merge_asof mechanism unchanged) and additively joined by game_id.
# Same BIAS_METRICS/SURFACE_SWITCH_FEATURE_COLUMNS precedent: these stay out
# of MODEL_FEATURE_COLUMNS, so only the explicitly opted-in
# weak_stack_fluview_home/_away profiles read them. Each profile carries
# EXACTLY ONE of the two columns -- the same "one new column" shape as
# weak_stack_graph_sack above -- even though both live in the same widened
# parquet table (game_features_weak_stack_fluview.parquet).
FLUVIEW_HOME_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS = ("fluview_home_market_elevated",)
FLUVIEW_AWAY_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS = ("fluview_away_market_elevated",)

# weak_stack_v3 candidate profile (docs/weak_stack_v3.md): every NFL registry
# signal with probability_positive >= 0.60 in accuracy_points units, not
# already inside FEATURE_SETS["football_weak_stack"], that is buildable this
# session from data already local to the repo. Same BIAS_METRICS/
# SURFACE_SWITCH_FEATURE_COLUMNS precedent: these stay out of
# MODEL_FEATURE_COLUMNS, so only the explicitly opted-in weak_stack_v3
# profile reads them. Three sub-families, computed in
# nfl_ats.weak_stack_v3_features from the newest schedules snapshot (and,
# for penalty rate, the newest PBP snapshot) alone -- never from
# result/spread_line -- ported from already-reviewed constructs rather than
# re-derived: division_revenge/sandwich_spot mirror
# nfl_ats.experiment_runner.FLAG_BUILDERS' division_revenge_game/
# sandwich_spot; post_blowout_win_letdown/loss_bounce mirror
# scripts/nfl_bias_battery_screen.py's identically-named hypotheses;
# diff_penalty_rate_prior mirrors scripts/weak_stack_v2_eval.py's already-
# verified reconstruction of the registered `penalty_discipline` signal;
# the two travel/rest flags mirror scripts/nfl_travel_rest_battery_screen.py
# cells 4 and 8 (registry/stadium_coordinates.json).
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
# diff-only, matching diff_penalty_rate_prior's own home/away-optional
# precedent in scripts/weak_stack_v2_eval.py (the registry's
# `weak_stack_v2_penalty_only`/`penalty_discipline` entries score this exact
# column name).
GAP_V3_PENALTY_FEATURE_COLUMNS = ("diff_penalty_rate_prior",)
# Both game-level, not home/away-split: thursday_pure is a plain calendar
# fact about the game itself, and return_trip_hangover is inherently a
# home-side-only construct (the home team's own preceding road trip), same
# shape precedent as SURFACE_SWITCH_FEATURE_COLUMNS' single unsigned column.
GAP_V3_TRAVEL_FEATURE_COLUMNS = (
    "gap_thursday_pure_flag",
    "gap_return_trip_hangover_flag",
)

# MOD-06's one live arm (docs/mod06_position_prior_shrinkage.md):
# players.py's PLAYER_VALUE_STATE_METRICS shrink a thin player's per-snap
# value rate toward ZERO via career/(career+value_prior_snaps). The candidate
# is the SAME two metrics, computed identically except the shrinkage target
# is a point-in-time-safe, data-derived position/channel prior instead of
# zero (``players.py::enrich_with_player_features(value_shrinkage_target=
# "position_prior")``, an opt-in parameter -- the default "zero" path is
# bit-identical to today's production feature). Distinct column names so a
# feature_arm baseline/candidate pair can be fit from the SAME features file
# (the runner's own constraint) without one arm overwriting the other's
# values under an identical name.
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
    "player_qb": _difference_features(PLAYER_QB_STATE_METRICS),
    "player_injuries": _difference_features(PLAYER_INJURY_STATE_METRICS),
    "player_continuity": _difference_features(PLAYER_CONTINUITY_STATE_METRICS),
    "player_values": _difference_features(PLAYER_VALUE_STATE_METRICS),
    "player_participation_values": _difference_features(PLAYER_PARTICIPATION_STATE_METRICS),
    "graph": GRAPH_FEATURE_COLUMNS[:8],
    "schedule_rating": GRAPH_FEATURE_COLUMNS[8:],
    "bias": BIAS_FEATURE_COLUMNS,
    "surface_switch": SURFACE_SWITCH_FEATURE_COLUMNS,
    "player_values_js_prior": _difference_features(PLAYER_VALUE_JS_PRIOR_STATE_METRICS),
    "gap_v3_bias": GAP_V3_BIAS_FEATURE_COLUMNS,
    "gap_v3_penalty": GAP_V3_PENALTY_FEATURE_COLUMNS,
    "gap_v3_travel": GAP_V3_TRAVEL_FEATURE_COLUMNS,
    "forecast_weather": FORECAST_WEATHER_FEATURE_COLUMNS,
    "observed_weather": OBSERVED_WEATHER_FEATURE_COLUMNS,
    "graph_team_stat_off_sack_rate": GRAPH_TEAM_STAT_OFF_SACK_RATE_FEATURE_COLUMNS,
    "graph_team_stat_def_yards_per_play": GRAPH_TEAM_STAT_DEF_YARDS_PER_PLAY_FEATURE_COLUMNS,
    "fluview_home_elevated_on_production": FLUVIEW_HOME_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS,
    "fluview_away_elevated_on_production": FLUVIEW_AWAY_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS,
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
# MOD-07 weak-signal stack (SPEC-4): the surviving weak signals -- the player
# value composite, plus the documented early-season opener biases -- in one set.
# The injury columns of the candidate table carry LEARNED availability semantics
# by construction (it is built through build-learned-availability-features), so
# no separate availability family appears here; the two tables must never be
# mixed in one run. Bias columns stay out of every frozen set: only these two
# entries admit them.
FEATURE_SETS["football_weak_stack"] = (
    FEATURE_SETS["football_player_value"] + FEATURE_FAMILIES["bias"]
)
FEATURE_SETS["full_weak_stack"] = FEATURE_SETS["full_player_value"] + FEATURE_FAMILIES["bias"]
# MOD-08 candidate profile (docs/surface_switch_feature_arm.md): weak_stack
# plus the surface-switch tilt feature, wiring the project's strongest
# prospective lead in as a training-time FEATURE rather than a post-prediction
# pick-level overlay. Declared for a feature_arm experiment comparing it
# against weak_stack itself; never mixed with any other profile.
FEATURE_SETS["football_weak_stack_surface"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["surface_switch"]
)
FEATURE_SETS["full_weak_stack_surface"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["surface_switch"]
)
# MOD-06 candidate profile (docs/mod06_position_prior_shrinkage.md): weak_stack
# with the player_values family REPLACED by its js_prior counterpart (same two
# underlying metrics, shrunk toward a data-derived position/channel prior
# instead of zero) -- everything else (bias, injuries, continuity, QB) held
# identical, isolating exactly the shrinkage-target variable under test.
# Declared for a feature_arm experiment comparing it against weak_stack
# itself; never mixed with any other profile.
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
# weak_stack_v3 candidate profile (docs/weak_stack_v3.md): weak_stack_surface
# (weak_stack + surface_switch_flag, itself a registry gap candidate --
# surface_switch_feature_arm, P+ 0.6181 -- reused rather than re-added) plus
# the three new gap sub-families above. Declared for a feature_arm-style
# opener-graded comparison against the active weak_stack profile; never
# mixed with any other profile, and never referenced by the active model.
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
# weak_stack_v4 candidate profile (docs/weak_stack_v4.md): PRODUCTION
# weak_stack plus the forecast-weather family. Deliberately built on
# weak_stack and NOT on weak_stack_v3 or weak_stack_surface -- the question is
# whether forecast weather adds to PRODUCTION, and stacking it onto a profile
# already refused at the opener would confound the answer. Declared for an
# opener-graded comparison against the active weak_stack profile; never mixed
# with any other profile, and never referenced by the active model.
FEATURE_SETS["football_weak_stack_v4"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["forecast_weather"]
)
FEATURE_SETS["full_weak_stack_v4"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["forecast_weather"]
)
# POSITIVE CONTROL ONLY (docs/weak_stack_v4.md, "wind oracle"): production
# weak_stack plus the weather that ACTUALLY happened. Deliberately leaky and
# NEVER promotable -- it exists to bound the whole weather channel. If a model
# handed perfect weather cannot beat the baseline, no improvement in
# forecasting can, which is the admissible `bounded_by_control` ground; if it
# can, the oracle-minus-forecast gap is exactly the headroom a better wind
# source could buy.
FEATURE_SETS["football_weak_stack_oracle_weather"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["observed_weather"]
)
FEATURE_SETS["full_weak_stack_oracle_weather"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["observed_weather"]
)
# weak_stack_graph_sack candidate profile (docs/graph_team_stat_on_production.md):
# PRODUCTION weak_stack plus the one graph_team_stat_off_sack_rate column.
# Built on PRODUCTION weak_stack directly, never on weak_stack_v3/_surface/_v4
# -- same reasoning as weak_stack_v4 above, stated there and restated here:
# the question is whether the graph feature adds to what is actually played,
# and stacking it onto an undecided/refused profile would confound the
# answer. Declared for a close-graded rotation-window comparison against the
# active weak_stack profile; never mixed with any other candidate profile,
# and never referenced by the active model.
FEATURE_SETS["football_weak_stack_graph_sack"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_off_sack_rate"]
)
FEATURE_SETS["full_weak_stack_graph_sack"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_off_sack_rate"]
)

# weak_stack_graph_def_ypp candidate profile
# (docs/graph_team_stat_def_ypp_on_production.md): PRODUCTION weak_stack plus
# the one graph_team_stat_def_yards_per_play column. Built on PRODUCTION
# weak_stack directly, never on weak_stack_v3/_surface/_v4/_graph_sack -- same
# reasoning as weak_stack_graph_sack above, stated there and restated here:
# the question is whether the graph feature adds to what is actually played,
# and stacking it onto an undecided/refused profile would confound the
# answer. Declared for a close-graded rotation-window comparison against the
# active weak_stack profile; never mixed with any other candidate profile,
# and never referenced by the active model.
FEATURE_SETS["football_weak_stack_graph_def_ypp"] = (
    FEATURE_SETS["football_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_def_yards_per_play"]
)
FEATURE_SETS["full_weak_stack_graph_def_ypp"] = (
    FEATURE_SETS["full_weak_stack"] + FEATURE_FAMILIES["graph_team_stat_def_yards_per_play"]
)

# weak_stack_fluview_home / weak_stack_fluview_away candidate profiles
# (docs/fluview_on_production.md): PRODUCTION weak_stack plus exactly one of
# the two FluView elevated-illness columns. Built on PRODUCTION weak_stack
# directly, never on weak_stack_v3/_surface/_v4/_graph_sack -- same reasoning
# as weak_stack_graph_sack above, stated there and restated here: the
# question is whether the FluView feature adds to what is actually played,
# and stacking it onto an undecided/refused profile would confound the
# answer. Declared for a close-graded rotation-window comparison against the
# active weak_stack profile; never mixed with any other candidate profile,
# and never referenced by the active model.
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
