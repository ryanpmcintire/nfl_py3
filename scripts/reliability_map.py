"""Split-half reliability map across the project's whole continuous
team-week feature surface (owner-directed cross-cutting audit, 2026-08-26).

**Why**: the registry (``registry/weak_signals.json``) holds 99 NFL
accuracy-point leads at ``probability_positive >= 0.80``, but only 17 carry a
``reliability`` number. Reliability is the ONE measurement that can
legitimately close a line of work (see binding taxonomy below), and it is
the strongest available predictor of whether an effect survives out of
sample. Rather than reconstruct 82 individual signals, this script measures
reliability on the underlying CONTINUOUS TEAM-WEEK FEATURES that feed them,
reused directly by many signals across the registry.

**Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)**: an interval
or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment -- at this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two closing grounds: (1)
refuted mechanism -- RESOLVED wrong sign (whole interval on the wrong side
of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record with ``probability_positive``, never
"contains zero". If a record command errors, the verdict is wrong, not the
validator. This script does not close anything -- it only measures a
feature-level reliability map. A feature reliability point estimate at or
below zero is a *candidate* for the reliability closing ground on signals
built from it, but only when the bootstrap interval is actually resolved
below/at zero, never merely because it contains zero.

**Method**: reshapes ``data/processed/game_features_weak_stack_v4.parquet``
(4,902 rows x 281 cols) from game-level ``home_<x>``/``away_<x>`` (and
``<x>_home``/``<x>_away``) pairs into team-week long form -- each REG-season
game contributes two rows, one per side -- then runs
``nfl_ats.cfb_qb_dependence.split_half_reliability`` (imported, not
reimplemented; the same function backing the FluView/team-style/PBP-08
reliability precedents) on every numeric team-attributable metric discovered
that way. ``data/processed/game_features_weak_stack_v3.parquet`` (290 cols)
supplies four ``gap_*`` families v4 dropped (division revenge, sandwich
spot, post-blowout win/loss), joined onto v4 by ``game_id``.
``data/processed/game_features.parquet`` (98 cols) was checked and confirmed
to add nothing beyond v3/v4 (measured this session, see docstring of
``discover_family_pairs`` -- its column set is a strict subset of v3's).

Columns are partitioned into exactly five buckets so every one of the 281
(+4 gap) columns is accounted for, never silently dropped:
1. identifiers/metadata (game_id, season, week, ...)
2. outcome/label columns (result, home_cover, home_score, away_score, ...)
3. team-attributable numeric pairs -- SWEPT (this script's subject)
4. excluded pairs -- matched the home_/away_ pattern but are non-numeric
   (team, qb_id, observed_at) or explicitly market/outcome, not a team trait
   (spread_odds, score)
5. game-level/venue-level columns with no per-team split at all (spread_line,
   total_line, weather at the venue, calendar encoding, matchup diffs, ...)
   -- AGENTS.md's own example bucket for "handled separately, not applicable".

Writes ``artifacts/reliability_map/<UTC timestamp>/results.json`` +
``results.csv`` and cross-references the registry's 99 NFL accuracy_points
leads at ``probability_positive >= 0.80`` against the families this map
covers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

RELIABILITY_SEED = 20260826
N_BOOT = 4000

V4_PATH = REPO / "data" / "processed" / "game_features_weak_stack_v4.parquet"
V3_PATH = REPO / "data" / "processed" / "game_features_weak_stack_v3.parquet"

# v3-only families v4 dropped that still fit the "<x>_home"/"<x>_away" pair
# pattern (task step 2/3) -- pulled onto the v4 frame via a game_id merge so
# the rest of the pipeline treats them identically to a native v4 column.
V3_ONLY_PAIR_BASES = (
    "gap_division_revenge",
    "gap_sandwich_spot",
    "gap_post_blowout_win_letdown",
    "gap_post_blowout_loss_bounce",
)

IDENTIFIER_COLUMNS = frozenset(
    {
        "game_id",
        "season",
        "week",
        "gameday",
        "away_team",
        "home_team",
        "weekday",
        "gametime",
        "kickoff",
        "game_type",
        "location",
        "pbp_feature_version",
        "player_feature_version",
        "home_projected_qb_id",
        "away_projected_qb_id",
        "home_injury_observed_at",
        "away_injury_observed_at",
    }
)

# Realized game outcomes -- not pregame predictive features. Reported
# separately so nobody reads an outcome's own split-half reliability as a
# feature's.
OUTCOME_COLUMNS = frozenset({"result", "ats_margin", "home_cover", "away_score", "home_score"})

# home_/away_-prefixed pairs that match the auto-discovery pattern but are a
# market artifact of the BET, not a trait of a TEAM (juice on either side of
# the spread). Explicitly excluded rather than swept.
MARKET_PAIR_EXCLUSIONS = frozenset({"home_spread_odds", "away_spread_odds"})


def discover_family_pairs(
    columns: list[str], dtypes: dict[str, Any]
) -> tuple[dict[str, tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Auto-detect ``home_<x>``/``away_<x>`` and ``<x>_home``/``<x>_away`` pairs.

    Returns ``(numeric_families, excluded)`` where ``numeric_families`` maps
    metric name -> ``(home_col, away_col, pattern)`` for pairs that are both
    numeric and not in the outcome/market exclusion lists, and ``excluded``
    lists every home_/away_-shaped column with a reason it did not make the
    cut (non-numeric dtype, no counterpart, or explicit market/outcome
    exclusion) -- nothing is silently dropped.
    """

    colset = set(columns)
    numeric_families: dict[str, tuple[str, str, str]] = {}
    excluded: list[tuple[str, str, str]] = []
    claimed: set[str] = set()

    def is_numeric(col: str) -> bool:
        return pd.api.types.is_numeric_dtype(dtypes[col])

    for col in columns:
        if col.startswith("diff_") or col.endswith("_diff"):
            continue
        if col in IDENTIFIER_COLUMNS or col in OUTCOME_COLUMNS:
            continue
        if col.startswith("home_"):
            base = col[len("home_") :]
            away_col = f"away_{base}"
        elif col.endswith("_home"):
            base = col[: -len("_home")]
            away_col = f"{base}_away"
        else:
            continue
        if away_col not in colset:
            continue
        claimed.add(col)
        claimed.add(away_col)
        if col in MARKET_PAIR_EXCLUSIONS or away_col in MARKET_PAIR_EXCLUSIONS:
            excluded.append((base, col, "market artifact of the bet, not a team trait"))
            continue
        if not (is_numeric(col) and is_numeric(away_col)):
            excluded.append((base, col, f"non-numeric dtype ({dtypes[col]})"))
            continue
        if base not in numeric_families:
            pattern = "prefix" if col.startswith("home_") else "suffix"
            numeric_families[base] = (col, away_col, pattern)

    # Anything shaped like a home/away column that never got claimed above
    # (e.g. home_cover has no away_cover counterpart at all).
    for col in columns:
        if col in claimed or col in IDENTIFIER_COLUMNS or col in OUTCOME_COLUMNS:
            continue
        if col.startswith("diff_") or col.endswith("_diff"):
            continue
        if col.startswith("home_") or col.startswith("away_") or col.endswith(("_home", "_away")):
            excluded.append((col, col, "no home/away counterpart column"))

    return numeric_families, excluded


def load_feature_table() -> pd.DataFrame:
    v4 = pd.read_parquet(V4_PATH)
    v3 = pd.read_parquet(V3_PATH)
    v3_pair_cols = ["game_id"] + [
        f"{base}_{side}" for base in V3_ONLY_PAIR_BASES for side in ("home", "away")
    ]
    missing = [c for c in v3_pair_cols if c not in v3.columns]
    if missing:
        raise SystemExit(f"expected v3-only columns missing: {missing}")
    merged = v4.merge(v3[v3_pair_cols], on="game_id", how="left", validate="one_to_one")
    return merged


def build_long_frame(
    features: pd.DataFrame, families: dict[str, tuple[str, str, str]]
) -> pd.DataFrame:
    """One row per (REG-season game, side): ``team_id``, ``season``, ``week``,
    plus every discovered metric column, home/away values folded into one
    column per metric. Restricted to ``game_type == "REG"`` -- POST weeks
    give a team at most 1-4 games a season, almost always below the >=2
    observations-per-half floor ``split_half_reliability`` already enforces,
    so including them would only add rows that get dropped downstream while
    diluting week-number semantics (POST week numbers are not the same
    clock as REG week numbers).
    """

    reg = features.loc[features["game_type"] == "REG"].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)

    pieces = []
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        piece = reg[["game_id", "season", "week", team_col]].rename(columns={team_col: "team_id"})
        for metric, (home_col, away_col, _pattern) in families.items():
            source_col = home_col if side == "home" else away_col
            piece[metric] = pd.to_numeric(reg[source_col], errors="coerce")
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def is_constant(series: pd.Series) -> bool:
    values = series.dropna()
    return len(values) == 0 or bool(values.nunique(dropna=True) <= 1)


def run_sweep(
    long: pd.DataFrame, families: dict[str, tuple[str, str, str]], *, seed: int, n_boot: int
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    for metric in sorted(families):
        if is_constant(long[metric]):
            skipped.append(metric)
            continue
        result = split_half_reliability(long, metric, seed=seed, n_boot=n_boot)
        home_col, away_col, pattern = families[metric]
        result["home_column"] = home_col
        result["away_column"] = away_col
        result["pattern"] = pattern
        results.append(result)
    return results, skipped


# ---------------------------------------------------------------------------
# Cross-reference: the registry's 99 NFL accuracy_points leads at P+ >= 0.80
# against the families this map covers.
#
# This mapping is INFERRED -- built by reading each lead's ``description``
# field in registry/weak_signals.json against the family names this script
# discovers, not by opening every signal's source module. A handful (marked
# ``read_from_description``) name their source column explicitly in that
# description (e.g. "game_features_pbp.parquet's home_pbp_off_pass_rate");
# the rest are this session's best-effort keyword match and should be read
# as a starting point for a signal-by-signal audit, not a verified fact.
# ---------------------------------------------------------------------------

# lead name -> (matched family keys or [], status, note)
# status one of:
#   "covered_new"                  -- no prior registry reliability; this map
#                                      now supplies one for a directly-matching
#                                      family.
#   "corroborates_existing"        -- registry already has a reliability for
#                                      this lead; this map supplies an
#                                      independent, closely-related family
#                                      number alongside it.
#   "not_covered_external_source"  -- a real team/market construct, but its
#                                      data source is not one of these three
#                                      parquet tables at all (a genuine
#                                      coverage gap, not a by-design exclusion).
#   "not_applicable"                -- game/venue-level (AGENTS.md's own
#                                      example bucket) or a pure model/
#                                      composition construct, not a feature.
LEAD_CROSSREF: dict[str, tuple[list[str], str, str]] = {
    "bias_battery_division_revenge_game": (
        ["gap_division_revenge"],
        "covered_new",
        "Same construct as gap_division_revenge_home/away (v3).",
    ),
    "bias_battery_division_revenge_game_opener": (
        ["gap_division_revenge"],
        "covered_new",
        "Opener-grade re-screen of the same underlying construct.",
    ),
    "divisional_rematch_revenge_home_loser": (
        ["gap_division_revenge"],
        "covered_new",
        "Revenge-game construct, same family.",
    ),
    "overlay_subset_production_plus_division_revenge": (
        ["gap_division_revenge"],
        "covered_new",
        "One ingredient (division_revenge_tilt_overlay) of an OR-composition; "
        "the composition itself is not a single feature.",
    ),
    "injury_value_lost_prior_week_absence_saturday_channel": (
        ["injury_skill_epa_value_lost", "injury_defense_disruption_value_lost"],
        "covered_new",
        "Channel-delta built on the injury value-lost construct; the delta "
        "itself is not a single column, but its underlying family is swept.",
    ),
    "injury_value_lost_prior_week_report_saturday_channel": (
        ["injury_skill_epa_value_lost", "injury_defense_disruption_value_lost"],
        "covered_new",
        "Same underlying family as above.",
    ),
    "injury_value_lost_tuesday_saturday_channel_official_only": (
        ["injury_skill_epa_value_lost", "injury_defense_disruption_value_lost"],
        "covered_new",
        "Same underlying family as above.",
    ),
    "injury_value_lost_tuesday_saturday_channel_pft_augmented": (
        ["injury_skill_epa_value_lost", "injury_defense_disruption_value_lost"],
        "covered_new",
        "Same underlying family as above.",
    ),
    "player_family_base_vs_continuity": (
        [
            "offense_lineup_continuity",
            "offensive_line_continuity",
            "skill_lineup_continuity",
            "defense_lineup_continuity",
            "front_lineup_continuity",
            "secondary_lineup_continuity",
            "special_teams_lineup_continuity",
            "active_roster_continuity",
            "active_roster_mean_experience",
        ],
        "covered_new",
        "Lineup-continuity feature BLOCK vs base; every member column is swept.",
    ),
    "player_family_base_vs_injuries_continuity": (
        [
            "injury_offense_unavailability",
            "injury_defense_unavailability",
            "injury_special_teams_unavailability",
            "injury_offensive_line_unavailability",
            "injury_skill_unavailability",
            "injury_front_unavailability",
            "injury_secondary_unavailability",
            "offense_lineup_continuity",
            "defense_lineup_continuity",
        ],
        "covered_new",
        "Injuries+continuity BLOCK; member columns swept.",
    ),
    "player_family_base_vs_qb_continuity": (
        [
            "qb_expected_epa_per_dropback",
            "qb_starter_epa_per_dropback",
            "qb_starter_cpoe",
            "qb_start_probability",
            "qb_starter_experience_log",
        ],
        "covered_new",
        "QB+continuity BLOCK; member columns swept.",
    ),
    "player_family_base_vs_value": (
        ["injury_skill_epa_value_lost", "injury_defense_disruption_value_lost"],
        "covered_new",
        "Injury value-lost BLOCK; member columns swept.",
    ),
    # -- Already has a registry reliability; this map corroborates with an
    # independent, closely related feature-level number.
    "pbp08_pass_mismatch": (
        ["pbp_off_pass_rate", "pbp_off_pass_epa_per_play"],
        "corroborates_existing",
        "Description names 'prior-4-game pass-OVE offense'; closest table "
        "columns are pbp_off_pass_rate / off_pass_epa_per_play (not an exact "
        "match -- the lead's own EWM window differs from this map's raw "
        "column).",
    ),
    "pbp08_protection_mismatch": (
        ["pbp_pressure_allowed_rate", "pbp_pressure_rate"],
        "corroborates_existing",
        "Description names 'pressure-rate-allowed offense' / "
        "'pressure-generating defense', matching pbp_pressure_allowed_rate / "
        "pbp_pressure_rate directly (read_from_description).",
    ),
    "penalty_crew_holding_tilt_run_heavy": (
        ["pbp_off_pass_rate"],
        "corroborates_existing",
        "Description explicitly names \"game_features_pbp.parquet's "
        'home_pbp_off_pass_rate" (read_from_description).',
    ),
    "qb_age_second_year_jump": (
        ["qb_starter_experience_log"],
        "corroborates_existing",
        "Closest continuous proxy for QB experience/tenure in this table; "
        "not an exact match for the 'year-2 starter' flag itself.",
    ),
    "close_game_luck_turnover_under_rebound": (
        ["off_turnover_rate", "def_takeaway_rate"],
        "corroborates_existing",
        "Description names 'centered turnover differential'; closest table "
        "columns are off_turnover_rate / def_takeaway_rate.",
    ),
    "injury_value_lost_gradient": (
        ["injury_skill_epa_value_lost", "injury_defense_disruption_value_lost"],
        "corroborates_existing",
        "Same value-lost construct this map's family sweeps directly.",
    ),
    "injury_value_lost_narrowed": (
        ["injury_skill_epa_value_lost", "injury_defense_disruption_value_lost"],
        "corroborates_existing",
        "Same value-lost construct this map's family sweeps directly.",
    ),
}

# Leads whose description ties them to a real team/market construct that is
# simply not present in these three parquet tables at all -- a genuine
# coverage gap, not a by-design exclusion. Grouped by data source for the
# report rather than repeated per lead.
NOT_COVERED_EXTERNAL_SOURCE = {
    "best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered": (
        "cross-book opener spread dispersion (market microstructure)"
    ),
    "bye_overval_fade_full_slate_post2011": "bye-week scheduling (no bye column in this table)",
    "redteam_bye_overval_fade_full_slate_withinseason_bye_map": (
        "bye-week scheduling (no bye column in this table)"
    ),
    "environmental_battery_aqi_high_outdoor": "home-county AQI (external air-quality feed)",
    "era_trend_hc_year_one_fade": "head-coach tenure (not in this table)",
    "hc_year_one_fade": "head-coach tenure (not in this table)",
    "interim_hc_first_game": "interim head-coach status (not in this table)",
    "interim_hc_home_within_interim": "interim head-coach status (not in this table)",
    "ffc_adp_cellA_highadp_underdog_back_ppr_w14": "fantasy ADP roster data (external)",
    "ffc_adp_cellC_highadp_underdog_back_ppr_w12": "fantasy ADP roster data (external)",
    "motivation_ladder_tank_zone_wk14_18": "league standings/draft position (not in this table)",
    "movement_attribution_pop_threshold_attributed_any": (
        "line-movement attribution (market microstructure)"
    ),
    "movement_attribution_pop_threshold_injury": (
        "line-movement attribution (market microstructure)"
    ),
    "movement_attribution_pop_unfiltered_attributed_any": (
        "line-movement attribution (market microstructure)"
    ),
    "movement_attribution_pop_unfiltered_injury": (
        "line-movement attribution (market microstructure)"
    ),
    "movement_rule_composed_chain": "observed line movement (market microstructure)",
    "nflcom_refresh_out2_starters_on_chain": (
        "NFL.com injury report (distinct external source from this table's injury_* family)"
    ),
    "nflcom_refresh_out2_starters_on_chain_gate_admitted": (
        "NFL.com injury report (distinct external source)"
    ),
    "observed_movement_oracle_full_slate": "observed line movement (market microstructure)",
    "observed_movement_oracle_sunday_am_realism": (
        "observed line movement (market microstructure)"
    ),
    "observed_movement_threshold_0_5": "observed line movement (market microstructure)",
    "observed_movement_threshold_1_0": "observed line movement (market microstructure)",
    "observed_movement_threshold_1_0_sunday_am_realism": (
        "observed line movement (market microstructure)"
    ),
    "odds_microstructure_H3_3_0a_full_week_oracle_2020_2025_sanity_check": (
        "line movement oracle (market microstructure)"
    ),
    "odds_microstructure_H3_3_0b_full_week_oracle_2023_2025_baseline": (
        "line movement oracle (market microstructure)"
    ),
    "odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025": (
        "line movement oracle (market microstructure)"
    ),
    "opener_error_mining_favorite_side_road_favorite": (
        "production-rule slice by market spread side"
    ),
    "opener_error_mining_movement_agreement_agrees_corrected": (
        "production-rule slice by observed line movement"
    ),
    "opener_error_mining_movement_agreement_disagrees": (
        "production-rule slice by observed line movement"
    ),
    "opener_error_mining_movement_agreement_disagrees_overlay_paired_delta": (
        "production-rule slice by observed line movement"
    ),
    "opener_error_mining_movement_agreement_disagrees_overlay_paired_delta_move_ge_1_0": (
        "production-rule slice by observed line movement"
    ),
    "opener_error_mining_rest_diff_even": (
        "rest_diff is diff-only in this table (no home/away pair to split)"
    ),
    "opener_error_mining_slate_primetime": "kickoff slot (calendar, not a team trait)",
    "opener_error_mining_spread_magnitude_0_2p5": "market spread magnitude",
    "opener_error_mining_spread_magnitude_3_6p5": "market spread magnitude",
    "opener_error_mining_total_bucket_below_42": "market total_line bucket",
    "overlay_subset_production_chain_coach_arrest": (
        "coach fade + player arrests (external roster/news data)"
    ),
    "overlay_subset_production_plus_spread_gap_zone": "market spread-gap construct",
    "pick_conditioned_spread_gap_zone_pre2018": "market spread-gap construct",
    "player_arrests_recent_14d_back_side_policy_opener": "USA Today arrests feed (external)",
    "pt_post_mnf_sunday_era_2009_2017": (
        "prior Thu/Mon-night schedule pattern (not in this table; rest_diff has no home/away pair)"
    ),
    "roof_battery_home_cover_open_vs_closed": "venue roof status (not in this table)",
    "sagarin_battery_top_decile_close": (
        "Sagarin rating divergence (distinct external rating from this table's schedule_rating "
        "family)"
    ),
    "sbr_opener_era_2020_2021": "SBR-derived opener line substitution (market data source)",
    "sbr_opener_pooled_2011_2021": "SBR-derived opener line substitution (market data source)",
    "surface_familiarity_r1_turf_venue_visitor_split": (
        "venue surface x visitor modal surface (venue-level, no team-attributable pair)"
    ),
    "surface_familiarity_r3_era_2018_2025": (
        "venue surface x visitor modal surface (venue-level, no team-attributable pair)"
    ),
    # Added to the registry concurrently by another session mid-audit (this
    # script observed candidates go 99 -> 100 between runs); triaged the same
    # way as the rest, not part of the original 99 this task was scoped to.
    "dst_transition_eastbound_interaction": (
        "timezone travel distance x DST transition window (not in this table)"
    ),
}

NOT_APPLICABLE = {
    "forecast_weather_kn_precip_high_total_full": "venue/game-level weather forecast",
    "forecast_weather_kn_precip_high_total_pre2020": "venue/game-level weather forecast",
    "forecast_weather_kn_temp_gap_cold_visitor_full": "venue/game-level weather forecast",
    "forecast_weather_kn_temp_gap_cold_visitor_pre2020": "venue/game-level weather forecast",
    "forecast_weather_kn_warm_team_cold_late_full": "venue/game-level weather forecast",
    "forecast_weather_kn_warm_team_cold_late_pre2020": "venue/game-level weather forecast",
    "forecast_weather_temp_gap_cold_visitor": "venue/game-level weather forecast",
    "forecast_weather_warm_team_cold_late": "venue/game-level weather forecast",
    "weather_battery_dome_team_outdoors_cold": "venue/game-level actual weather",
    "weather_battery_extreme_cold": "venue/game-level actual weather",
    "weather_battery_surface_switch_grass_to_turf": "venue-level surface switch flag",
    "weather_battery_thursday_outdoor_cold": "venue/game-level actual weather",
    "weather_battery_warm_team_cold_late": "venue/game-level actual weather",
    "weather_followup_surface_switch_x_outdoor_cold": "venue-level surface + weather",
    "weather_followup_temp_gap_cold_visitor": "venue/game-level actual weather",
    "weather_followup_wind_gap_visitor": "venue/game-level actual weather",
    "wxtot_precip60_top_total": "venue-level weather forecast x market total_line",
    "wxtot_wind15_bottom_total": "venue-level weather forecast x market total_line",
    "era_trend_production_model_opener_proxy_edge": "model-level accuracy trend, not a feature",
    "era_weighting_nfl_half_life_16": "sample-weighting scheme, not a feature",
    "era_weighting_nfl_half_life_8": "sample-weighting scheme, not a feature",
    "mod08_smooth_cdf_mapping": "probability-calibration construct, not a feature",
    "overlay_subset_holdout_2020_2022_reverse": "composition/holdout design, not a single feature",
}

# The 10 already-reliable leads this map does NOT independently touch
# (their data source lives outside these three parquet tables).
ALREADY_RELIABLE_NOT_CROSSREFERENCED = frozenset(
    {
        "attention_battery_both_cold",
        "fluview_away_market_elevated",
        "fluview_home_market_elevated",
        "penalty_crew_high_flag_heavy_underdog_opener",
        "redzone_reversion_c3_third_down_over_fade",
        "redzone_reversion_c5_hot_offense_vs_stingy_defense",
        "special_teams_return_top_quartile",
        "team_style_short_game_identity",
        "vi_disp_homecover_top_vs_bottom_tercile",
        "vi_dispersion_bottom_tercile_underdog",
    }
)


def cross_reference(reliability_by_metric: dict[str, dict[str, Any]]) -> dict[str, Any]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    candidates = {
        name: signal
        for name, signal in registry.signals.items()
        if signal.league == "nfl"
        and signal.effect_units == "accuracy_points"
        and (signal.probability_positive or 0.0) >= 0.80
    }

    rows: list[dict[str, Any]] = []
    for name, signal in sorted(candidates.items()):
        if name in LEAD_CROSSREF:
            families, status, note = LEAD_CROSSREF[name]
            matched = {
                f: reliability_by_metric[f]["spearman_brown_full_length_reliability"]
                for f in families
                if f in reliability_by_metric
            }
            rows.append(
                {
                    "lead": name,
                    "registry_probability_positive": signal.probability_positive,
                    "registry_reliability": signal.reliability,
                    "status": status,
                    "matched_families": matched,
                    "note": note,
                }
            )
        elif name in NOT_COVERED_EXTERNAL_SOURCE:
            rows.append(
                {
                    "lead": name,
                    "registry_probability_positive": signal.probability_positive,
                    "registry_reliability": signal.reliability,
                    "status": "not_covered_external_source",
                    "matched_families": {},
                    "note": NOT_COVERED_EXTERNAL_SOURCE[name],
                }
            )
        elif name in NOT_APPLICABLE:
            rows.append(
                {
                    "lead": name,
                    "registry_probability_positive": signal.probability_positive,
                    "registry_reliability": signal.reliability,
                    "status": "not_applicable",
                    "matched_families": {},
                    "note": NOT_APPLICABLE[name],
                }
            )
        elif name in ALREADY_RELIABLE_NOT_CROSSREFERENCED:
            rows.append(
                {
                    "lead": name,
                    "registry_probability_positive": signal.probability_positive,
                    "registry_reliability": signal.reliability,
                    "status": "already_reliable_not_crossreferenced",
                    "matched_families": {},
                    "note": "Already has its own registry reliability; data source lives outside "
                    "game_features_weak_stack_v3/v4/game_features.parquet.",
                }
            )
        else:
            rows.append(
                {
                    "lead": name,
                    "registry_probability_positive": signal.probability_positive,
                    "registry_reliability": signal.reliability,
                    "status": "unclassified",
                    "matched_families": {},
                    "note": (
                        "Not yet triaged in this script's LEAD_CROSSREF/NOT_COVERED/"
                        "NOT_APPLICABLE tables -- gap in the mapping, not a measurement."
                    ),
                }
            )

    n_covered_new = sum(1 for r in rows if r["status"] == "covered_new")
    n_corroborates = sum(1 for r in rows if r["status"] == "corroborates_existing")
    n_not_covered = sum(1 for r in rows if r["status"] == "not_covered_external_source")
    n_not_applicable = sum(1 for r in rows if r["status"] == "not_applicable")
    n_already_not_crossref = sum(
        1 for r in rows if r["status"] == "already_reliable_not_crossreferenced"
    )
    n_unclassified = sum(1 for r in rows if r["status"] == "unclassified")

    flagged_zero_or_below = [
        {
            "metric": m,
            "spearman_brown_full_length_reliability": d["spearman_brown_full_length_reliability"],
            "pearson_r_ci95": d["pearson_r_ci95"],
            "probability_positive": d["probability_positive"],
        }
        for m, d in reliability_by_metric.items()
        if d["spearman_brown_full_length_reliability"] is not None
        and not np.isnan(d["spearman_brown_full_length_reliability"])
        and d["spearman_brown_full_length_reliability"] <= 0.0
    ]

    return {
        "n_candidates": len(candidates),
        "n_covered_new": n_covered_new,
        "n_corroborates_existing": n_corroborates,
        "n_not_covered_external_source": n_not_covered,
        "n_not_applicable": n_not_applicable,
        "n_already_reliable_not_crossreferenced": n_already_not_crossref,
        "n_unclassified": n_unclassified,
        "rows": rows,
        "feature_families_at_or_below_zero_reliability": flagged_zero_or_below,
        "note": (
            "LEAD_CROSSREF/NOT_COVERED_EXTERNAL_SOURCE/NOT_APPLICABLE are this session's "
            "INFERRED reading of each lead's description field against the discovered "
            "family names, not a verified per-signal source-code audit (label per "
            "AGENTS.md 'label how you know it'). A feature family's reliability at or "
            "below zero here is NOT itself a closing ground for the leads it touches -- "
            "only a resolved-below-zero bootstrap interval on THAT SPECIFIC lead's own "
            "reliability measurement is admissible, per the taxonomy in this file's "
            "module docstring."
        ),
    }


def main() -> None:
    started = time.time()
    print(f"=== loading {V4_PATH.name} + v3-only gap_* columns ===")
    features = load_feature_table()
    print(f"merged feature table: {features.shape}")

    dtypes = {c: features[c].dtype for c in features.columns}
    families, excluded_pairs = discover_family_pairs(list(features.columns), dtypes)
    print(f"discovered {len(families)} team-attributable numeric families")
    print(f"excluded {len(excluded_pairs)} home/away-shaped columns (non-numeric/market/no pair):")
    for base, col, reason in excluded_pairs:
        print(f"  {base} ({col}): {reason}")

    claimed_cols = {c for pair in families.values() for c in pair[:2]}
    excluded_cols = {col for _base, col, _reason in excluded_pairs}
    accounted = (
        IDENTIFIER_COLUMNS
        | OUTCOME_COLUMNS
        | claimed_cols
        | excluded_cols
        | {c for c in features.columns if c.startswith("diff_") or c.endswith("_diff")}
    )
    game_level_singletons = sorted(set(features.columns) - accounted)
    print(
        f"\n{len(game_level_singletons)} game-level/venue-level singleton columns "
        "(not team-attributable):"
    )
    for col in game_level_singletons:
        print(f"  {col}")

    long = build_long_frame(features, families)
    n_teams = long["team_id"].nunique()
    print(f"\nteam-week long frame: {long.shape} ({n_teams} distinct team_id values)")

    results, skipped = run_sweep(long, families, seed=RELIABILITY_SEED, n_boot=N_BOOT)
    print(f"\nswept {len(results)} metrics, skipped {len(skipped)} constant/all-missing metrics:")
    for metric in skipped:
        print(f"  {metric}")

    ranked = sorted(
        results,
        key=lambda r: (
            r["spearman_brown_full_length_reliability"]
            if not np.isnan(r["spearman_brown_full_length_reliability"])
            else -999.0
        ),
        reverse=True,
    )
    print("\n=== ranked by Spearman-Brown full-length reliability ===")
    header = (
        f"{'metric':<44} {'SB reliability':>15} {'pearson r':>10} {'95% CI':>20} "
        f"{'P+':>7} {'n_ts':>6}"
    )
    print(header)
    for row in ranked:
        ci = row["pearson_r_ci95"]
        ci_str = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if not np.isnan(ci[0]) else "n/a"
        sb = row["spearman_brown_full_length_reliability"]
        sb_str = f"{sb:+.4f}" if not np.isnan(sb) else "n/a"
        pr = row["pearson_r"]
        pr_str = f"{pr:+.4f}" if not np.isnan(pr) else "n/a"
        pp = row["probability_positive"]
        pp_str = f"{pp:.3f}" if not np.isnan(pp) else "n/a"
        print(
            f"{row['metric']:<44} {sb_str:>15} {pr_str:>10} {ci_str:>20} {pp_str:>7} "
            f"{row['n_team_seasons']:>6}"
        )

    reliability_by_metric = {r["metric"]: r for r in results}
    print("\n=== cross-referencing registry NFL accuracy_points leads (P+ >= 0.80) ===")
    crossref = cross_reference(reliability_by_metric)
    print(
        f"candidates={crossref['n_candidates']} covered_new={crossref['n_covered_new']} "
        f"corroborates_existing={crossref['n_corroborates_existing']} "
        f"not_covered_external_source={crossref['n_not_covered_external_source']} "
        f"not_applicable={crossref['n_not_applicable']} "
        f"already_reliable_not_crossreferenced="
        f"{crossref['n_already_reliable_not_crossreferenced']} "
        f"unclassified={crossref['n_unclassified']}"
    )
    zero_or_below = crossref["feature_families_at_or_below_zero_reliability"]
    print(f"\n{len(zero_or_below)} swept families at or below zero Spearman-Brown reliability:")
    for entry in zero_or_below:
        print(
            f"  {entry['metric']}: {entry['spearman_brown_full_length_reliability']:+.4f} "
            f"CI={entry['pearson_r_ci95']} P+={entry['probability_positive']:.3f}"
        )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_map" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(
        [
            {
                "metric": r["metric"],
                "home_column": r["home_column"],
                "away_column": r["away_column"],
                "pattern": r["pattern"],
                "n_team_seasons": r["n_team_seasons"],
                "pearson_r": r["pearson_r"],
                "pearson_r_ci95_low": r["pearson_r_ci95"][0],
                "pearson_r_ci95_high": r["pearson_r_ci95"][1],
                "spearman_rho": r["spearman_rho"],
                "spearman_brown_full_length_reliability": r[
                    "spearman_brown_full_length_reliability"
                ],
                "probability_positive": r["probability_positive"],
            }
            for r in ranked
        ]
    )
    csv_path = output_dir / "results.csv"
    results_df.to_csv(csv_path, index=False)

    configuration = {
        "command": "reliability-map",
        "v4_path": str(V4_PATH),
        "v3_path": str(V3_PATH),
        "reliability_seed": RELIABILITY_SEED,
        "n_boot": N_BOOT,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "reliability_seed": RELIABILITY_SEED,
        "n_boot": N_BOOT,
        "n_families_discovered": len(families),
        "n_families_swept": len(results),
        "n_families_skipped_constant": len(skipped),
        "skipped_constant_families": skipped,
        "excluded_home_away_pairs": [
            {"base": base, "column": col, "reason": reason} for base, col, reason in excluded_pairs
        ],
        "game_level_singleton_columns": game_level_singletons,
        "families": {
            base: {"home_column": pair[0], "away_column": pair[1], "pattern": pair[2]}
            for base, pair in families.items()
        },
        "results": results,
        "ranked_metrics": [r["metric"] for r in ranked],
        "registry_crossreference": crossref,
        "csv_path": str(csv_path),
        "provenance": artifact_provenance(configuration, V4_PATH, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-map",
        metrics={
            "n_families_swept": len(results),
            "n_families_skipped_constant": len(skipped),
            "n_covered_new": crossref["n_covered_new"],
            "n_corroborates_existing": crossref["n_corroborates_existing"],
            "n_at_or_below_zero_reliability": len(zero_or_below),
        },
        notes=(
            "Measure-only reliability map across the game_features_weak_stack team-week "
            "feature surface; every family swept regardless of sign or interval shape, "
            "per AGENTS.md's binding closing-grounds taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
