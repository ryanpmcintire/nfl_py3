"""Deterministic, pregame schedule context for NFL games.

ENV-04's reusable family is derived only from the ordered schedule within a
team-season.  Results, lines, observed conditions, and future rows are never
read.  Season openers keep rest-dependent values missing instead of silently
pretending that an unknown turnaround is a normal week.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.constants import REST_CONTEXT_FEATURE_COLUMNS, TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError, require_columns

OFF_BYE_MIN_DAYS = 13
SHORT_WEEK_MAX_DAYS = 5
MINI_BYE_MIN_DAYS = 9
MINI_BYE_MAX_DAYS = 11

REQUIRED_SCHEDULE_COLUMNS = (
    "game_id",
    "season",
    "gameday",
    "home_team",
    "away_team",
    "location",
)


def _canonical_team(raw: object) -> str:
    team = str(raw).strip()
    return TEAM_ABBREVIATION_ALIASES.get(team, team)


def _validate_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    require_columns(schedules, REQUIRED_SCHEDULE_COLUMNS, "rest context schedules")
    frame = schedules.loc[:, list(REQUIRED_SCHEDULE_COLUMNS)].copy()
    if frame["game_id"].isna().any() or frame["game_id"].astype(str).str.strip().eq("").any():
        raise DataContractError("rest context schedules contains a blank game_id")
    frame["game_id"] = frame["game_id"].astype(str)
    duplicates = frame.loc[frame["game_id"].duplicated(), "game_id"].unique()
    if len(duplicates):
        raise DataContractError(
            f"rest context schedules contains duplicate game_id values: {duplicates[:5]}"
        )

    seasons = pd.to_numeric(frame["season"], errors="coerce")
    if seasons.isna().any() or (~np.isfinite(seasons)).any() or (seasons % 1.0 != 0.0).any():
        raise DataContractError("rest context schedules contains an invalid season")
    frame["season"] = seasons.astype(int)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce", format="mixed")
    if frame["gameday"].isna().any():
        raise DataContractError("rest context schedules contains an invalid gameday")

    for side in ("home", "away"):
        column = f"{side}_team"
        if frame[column].isna().any():
            raise DataContractError(f"rest context schedules contains a null {column}")
        frame[column] = frame[column].map(_canonical_team)
        if frame[column].eq("").any():
            raise DataContractError(f"rest context schedules contains a blank {column}")
    if frame["home_team"].eq(frame["away_team"]).any():
        raise DataContractError("rest context schedules contains identical home and away teams")

    locations = frame["location"].astype("string").str.strip().str.lower()
    if locations.isna().any() or not locations.isin({"home", "neutral"}).all():
        invalid = sorted(
            frame.loc[~locations.isin({"home", "neutral"}), "location"].astype(str).unique()
        )
        raise DataContractError(
            f"rest context schedules contains invalid location values: {invalid[:5]}"
        )
    frame["location_norm"] = locations

    appearances = pd.concat(
        [
            frame[["season", "gameday", "home_team"]].rename(columns={"home_team": "team"}),
            frame[["season", "gameday", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    if appearances.duplicated(["season", "gameday", "team"]).any():
        raise DataContractError(
            "rest context schedules contains multiple games for one team on one gameday"
        )
    return frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _flag(value: float, predicate: bool) -> float:
    return float(predicate) if math.isfinite(value) else math.nan


def build_rest_context_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic ENV-04 feature row per scheduled game.

    ``rest_away_consecutive_road_games`` counts the current true road game, so
    values 1, 2, and 3 mean the first, second, and third straight road game.
    A neutral-site or true-home appearance breaks the streak.  ``mini_bye`` is
    a 9--11 day turnaround, separated from both ordinary rest and the >=13-day
    off-bye contract frozen in ``docs/travel_rest_battery.md``.
    """

    frame = _validate_schedules(schedules)
    previous_date: dict[tuple[str, int], pd.Timestamp] = {}
    road_streak: dict[tuple[str, int], int] = {}
    rows: list[dict[str, Any]] = []

    for game in frame.itertuples(index=False):
        season = int(cast("Any", game.season))
        gameday = pd.Timestamp(cast("Any", game.gameday))
        location = str(game.location_norm)
        rests: dict[str, float] = {}
        for side in ("home", "away"):
            team = str(getattr(game, f"{side}_team"))
            prior = previous_date.get((team, season))
            rests[side] = float((gameday - prior).days) if prior is not None else math.nan

        away_key = (str(game.away_team), season)
        away_is_true_road = location == "home"
        away_road_games = road_streak.get(away_key, 0) + 1 if away_is_true_road else 0
        home_rest = rests["home"]
        away_rest = rests["away"]
        rows.append(
            {
                "game_id": str(game.game_id),
                "rest_home_days": home_rest,
                "rest_away_days": away_rest,
                "rest_days_diff": (
                    home_rest - away_rest
                    if math.isfinite(home_rest) and math.isfinite(away_rest)
                    else math.nan
                ),
                "rest_home_off_bye": _flag(home_rest, home_rest >= OFF_BYE_MIN_DAYS),
                "rest_away_off_bye": _flag(away_rest, away_rest >= OFF_BYE_MIN_DAYS),
                "rest_home_short_week": _flag(home_rest, home_rest <= SHORT_WEEK_MAX_DAYS),
                "rest_away_short_week": _flag(away_rest, away_rest <= SHORT_WEEK_MAX_DAYS),
                "rest_home_mini_bye": _flag(
                    home_rest, MINI_BYE_MIN_DAYS <= home_rest <= MINI_BYE_MAX_DAYS
                ),
                "rest_away_mini_bye": _flag(
                    away_rest, MINI_BYE_MIN_DAYS <= away_rest <= MINI_BYE_MAX_DAYS
                ),
                "rest_away_consecutive_road_games": float(away_road_games),
            }
        )

        for side in ("home", "away"):
            team = str(getattr(game, f"{side}_team"))
            key = (team, season)
            previous_date[key] = gameday
            is_true_road = side == "away" and away_is_true_road
            road_streak[key] = road_streak.get(key, 0) + 1 if is_true_road else 0

    result = pd.DataFrame(rows, columns=["game_id", *REST_CONTEXT_FEATURE_COLUMNS])
    for column in REST_CONTEXT_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    result.attrs["rest_context_provenance"] = {
        "schedule_columns_read": list(REQUIRED_SCHEDULE_COLUMNS),
        "outcome_columns_read": [],
        "rest_rule": "calendar-day gap from prior same-season appearance",
        "road_rule": "true road only; neutral and home appearances break the streak",
    }
    return result


def add_rest_context_features(games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach ENV-04 columns to unique caller-selected games in caller order."""

    require_columns(games, ("game_id",), "rest context games")
    if games["game_id"].isna().any() or games["game_id"].astype(str).duplicated().any():
        raise DataContractError("rest context games requires unique non-null game_id values")
    features = build_rest_context_features(schedules)
    selected = games.copy()
    selected["game_id"] = selected["game_id"].astype(str)
    merged = selected.merge(features, on="game_id", how="left", sort=False, validate="one_to_one")
    merged.attrs["rest_context_provenance"] = features.attrs["rest_context_provenance"]
    return merged


__all__ = [
    "MINI_BYE_MAX_DAYS",
    "MINI_BYE_MIN_DAYS",
    "OFF_BYE_MIN_DAYS",
    "REQUIRED_SCHEDULE_COLUMNS",
    "SHORT_WEEK_MAX_DAYS",
    "add_rest_context_features",
    "build_rest_context_features",
]
