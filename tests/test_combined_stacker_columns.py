"""Leakage regression tests for scripts/combined_stacker_look.py's four columns.

docs/combined_stacker_predeclaration.md section 2 requires every new candidate
column to ship with a test proving it is computable from information available
strictly before each game's prediction timestamp, and section 7 voids the look
if any of these fail.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "combined_stacker_look.py"

_spec = importlib.util.spec_from_file_location("combined_stacker_look", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
csl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csl)


def _base_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    defaults = {
        "game_type": "REG",
        "result": 24.0,
        "temp": 55.0,
        "week": 10,
        "spread_line": 3.0,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    return frame


def _pv_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame


def test_injury_tertile_boundary_uses_strictly_prior_completed_games_only() -> None:
    base = _base_frame(
        [
            {
                "game_id": "early",
                "gameday": pd.Timestamp("2022-09-08"),
                "home_team": "GB",
                "away_team": "MIA",
            },
            {
                "game_id": "late",
                "gameday": pd.Timestamp("2022-12-01"),
                "home_team": "MIA",
                "away_team": "GB",
            },
        ]
    )
    pv = _pv_frame(
        [
            {
                "game_id": "early",
                "home_injury_skill_epa_value_lost": 0.1,
                "home_injury_defense_disruption_value_lost": 0.0,
                "away_injury_skill_epa_value_lost": 9.9,
                "away_injury_defense_disruption_value_lost": 0.0,
            },
            {
                "game_id": "late",
                "home_injury_skill_epa_value_lost": 9.9,
                "home_injury_defense_disruption_value_lost": 9.9,
                "away_injury_skill_epa_value_lost": 0.1,
                "away_injury_defense_disruption_value_lost": 0.0,
            },
        ]
    )
    out = csl.attach_injury_tercile_columns(base, pv)
    assert out.loc[out["game_id"].eq("early"), "ivl_home_top_tertile"].iat[0] == 0.0
    assert out.loc[out["game_id"].eq("early"), "ivl_away_top_tertile"].iat[0] == 0.0
    assert out.loc[out["game_id"].eq("late"), "ivl_home_top_tertile"].iat[0] == 1.0


def test_injury_missing_value_fails_closed_to_not_flagged() -> None:
    base = _base_frame(
        [
            {
                "game_id": "a",
                "gameday": pd.Timestamp("2023-10-01"),
                "home_team": "GB",
                "away_team": "MIA",
            }
        ]
    )
    pv = _pv_frame(
        [
            {
                "game_id": "other",
                "home_injury_skill_epa_value_lost": None,
                "home_injury_defense_disruption_value_lost": None,
                "away_injury_skill_epa_value_lost": None,
                "away_injury_defense_disruption_value_lost": None,
            }
        ]
    )
    out = csl.attach_injury_tercile_columns(base, pv)
    assert out["ivl_home_top_tertile"].iat[0] == 0.0
    assert out["ivl_away_top_tertile"].iat[0] == 0.0


def test_climate_temp_excludes_same_and_later_season_games() -> None:
    base = _base_frame(
        [
            {
                "game_id": "h1",
                "gameday": pd.Timestamp("2022-09-11"),
                "home_team": "GB",
                "away_team": "MIA",
                "temp": 70.0,
            },
            {
                "game_id": "v",
                "gameday": pd.Timestamp("2022-10-02"),
                "home_team": "KC",
                "away_team": "GB",
            },
            {
                "game_id": "h2",
                "gameday": pd.Timestamp("2022-11-06"),
                "home_team": "GB",
                "away_team": "MIA",
                "temp": 30.0,
            },
        ]
    )
    schedules = pd.DataFrame(
        [
            {"game_id": "h1", "roof": "outdoors"},
            {"game_id": "v", "roof": "outdoors"},
            {"game_id": "h2", "roof": "outdoors"},
        ]
    )
    forecasts = pd.DataFrame(
        [
            {"game_id": "v", "forecast_temp_f": 40.0},
        ]
    )
    out = csl.attach_weather_columns(base, forecasts, schedules)
    visitor_row = out.loc[out["game_id"].eq("v")].iloc[0]
    assert visitor_row["csl_climate_temp"] == 70.0
    assert visitor_row["kn_temp_gap_cold_visitor_pre2020"] == 1.0


def test_temp_gap_requires_outdoor_and_threshold() -> None:
    base = _base_frame(
        [
            {
                "game_id": "dome",
                "gameday": pd.Timestamp("2022-11-06"),
                "home_team": "DET",
                "away_team": "MIA",
            },
            {
                "game_id": "warm_gap",
                "gameday": pd.Timestamp("2022-11-06"),
                "home_team": "GB",
                "away_team": "MIA",
            },
        ]
    )
    schedules = pd.DataFrame(
        [
            {"game_id": "dome", "roof": "dome"},
            {"game_id": "warm_gap", "roof": "outdoors"},
        ]
    )
    forecasts = pd.DataFrame(
        [
            {"game_id": "dome", "forecast_temp_f": 20.0},
            {"game_id": "warm_gap", "forecast_temp_f": 80.0},
        ]
    )
    prior_home = _base_frame(
        [
            {
                "game_id": "p",
                "gameday": pd.Timestamp("2022-09-11"),
                "home_team": "MIA",
                "away_team": "GB",
                "temp": 85.0,
            }
        ]
    )
    combined = pd.concat([prior_home, base], ignore_index=True)
    schedules_all = pd.concat(
        [schedules, pd.DataFrame([{"game_id": "p", "roof": "outdoors"}])], ignore_index=True
    )
    out = csl.attach_weather_columns(combined, forecasts, schedules_all)
    assert out.loc[out["game_id"].eq("dome"), "kn_temp_gap_cold_visitor_pre2020"].iat[0] == 0.0
    assert out.loc[out["game_id"].eq("warm_gap"), "kn_temp_gap_cold_visitor_pre2020"].iat[0] == 0.0


def test_warm_team_cold_late_gates() -> None:
    base = _base_frame(
        [
            {
                "game_id": "cold_late",
                "gameday": pd.Timestamp("2022-12-11"),
                "week": 14,
                "home_team": "GB",
                "away_team": "MIA",
            },
            {
                "game_id": "cold_early",
                "gameday": pd.Timestamp("2022-10-09"),
                "week": 5,
                "home_team": "GB",
                "away_team": "MIA",
            },
            {
                "game_id": "cold_late_cold_team",
                "gameday": pd.Timestamp("2022-12-12"),
                "week": 14,
                "home_team": "KC",
                "away_team": "CHI",
            },
        ]
    )
    schedules = pd.DataFrame(
        [{"game_id": game_id, "roof": "outdoors"} for game_id in base["game_id"]]
    )
    forecasts = pd.DataFrame(
        [{"game_id": game_id, "forecast_temp_f": 25.0} for game_id in base["game_id"]]
    )
    out = csl.attach_weather_columns(base, forecasts, schedules)
    flags = out.set_index("game_id")["warm_team_cold_late_pre2020"]
    assert flags["cold_late"] == 1.0
    assert flags["cold_early"] == 0.0
    assert flags["cold_late_cold_team"] == 0.0


def test_spread_gap_zone_bounds_are_frozen() -> None:
    base = _base_frame(
        [
            {"game_id": "at7", "spread_line": -7.0},
            {"game_id": "in8", "spread_line": -8.0},
            {"game_id": "at10", "spread_line": 10.0},
            {"game_id": "beyond", "spread_line": 10.5},
            {"game_id": "missing", "spread_line": None},
        ]
    )
    out = csl.attach_spread_gap_zone(base)
    flags = out.set_index("game_id")["spread_gap_zone"]
    assert flags["at7"] == 0.0
    assert flags["in8"] == 1.0
    assert flags["at10"] == 1.0
    assert flags["beyond"] == 0.0
    assert flags["missing"] == 0.0


def test_candidate_profile_registers_ninety_four_columns() -> None:
    from nfl_ats import margin as margin_module
    from nfl_ats.constants import FEATURE_SETS

    feature_sets_snapshot = dict(FEATURE_SETS)
    profiles_snapshot = margin_module.MARGIN_FEATURE_PROFILES
    profile_sets_snapshot = dict(margin_module._MARGIN_PROFILE_FEATURE_SETS)
    try:
        csl.register_candidate_profile()
        columns = margin_module.margin_feature_columns("market_residual", csl.CANDIDATE_PROFILE)
        baseline_columns = margin_module.margin_feature_columns("market_residual", "weak_stack")
        assert len(columns) == len(baseline_columns) + len(csl.CANDIDATE_COLUMNS)
        assert tuple(columns[: len(baseline_columns)]) == tuple(baseline_columns)
        assert set(columns[len(baseline_columns) :]) == set(csl.CANDIDATE_COLUMNS)
    finally:
        FEATURE_SETS.clear()
        FEATURE_SETS.update(feature_sets_snapshot)
        margin_module.MARGIN_FEATURE_PROFILES = profiles_snapshot
        margin_module._MARGIN_PROFILE_FEATURE_SETS.clear()
        margin_module._MARGIN_PROFILE_FEATURE_SETS.update(profile_sets_snapshot)
