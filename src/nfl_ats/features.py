"""Leak-safe, one-row-per-game feature engineering."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    GRAPH_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    STATE_METRICS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError, validate_schedules, validate_team_stats
from nfl_ats.graph_ratings import GraphRatingConfig, add_schedule_strength_features


def _numeric(frame: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _sum_available(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    result = frame[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    return pd.Series(result, index=frame.index, dtype="float64")


def add_ats_outcomes(schedules: pd.DataFrame) -> pd.DataFrame:
    """Add a single, documented ATS target using nflverse sign conventions."""

    result = schedules.copy()
    result["result"] = _numeric(result, "result")
    result["spread_line"] = _numeric(result, "spread_line")
    result["ats_margin"] = result["result"] - result["spread_line"]
    result["home_cover"] = np.select(
        [result["ats_margin"] > 0, result["ats_margin"] < 0],
        [1.0, 0.0],
        default=np.nan,
    )
    return result


def _regular_season_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    validate_schedules(schedules)
    regular = schedules.loc[schedules["game_type"].eq("REG")].copy()
    regular["gameday"] = pd.to_datetime(regular["gameday"], errors="raise")
    regular["season"] = pd.to_numeric(regular["season"], errors="raise").astype(int)
    regular["week"] = pd.to_numeric(regular["week"], errors="raise").astype(int)
    for column in ("home_team", "away_team"):
        regular[column] = regular[column].replace(TEAM_ABBREVIATION_ALIASES)
    return regular.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _kickoff_utc(games: pd.DataFrame) -> pd.Series:
    """Combine nflverse game date and Eastern kickoff time into UTC."""

    if "gametime" not in games:
        return pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns, UTC]")
    date_text = pd.to_datetime(games["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = games["gametime"].astype("string")
    local = pd.to_datetime(date_text + " " + time_text, errors="coerce")
    return local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def build_team_game_metrics(
    schedules: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Create offense and opponent-derived defense metrics for completed games."""

    validate_team_stats(team_stats)
    stats = team_stats.copy()
    stats["team"] = stats["team"].replace(TEAM_ABBREVIATION_ALIASES)
    if "season_type" in stats:
        stats = stats.loc[stats["season_type"].eq("REG")].copy()

    games = _regular_season_schedules(schedules)
    schedule_columns = [
        "game_id",
        "season",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "result",
        "spread_line",
    ]
    for column in schedule_columns:
        if column not in games:
            games[column] = np.nan
    if "season" in stats:
        stats = stats.drop(columns="season")
    stats = stats.merge(games[schedule_columns], on="game_id", how="inner", validate="many_to_one")

    team_rows_per_game = stats.groupby("game_id")["team"].nunique()
    malformed = team_rows_per_game.loc[team_rows_per_game != 2]
    if not malformed.empty:
        examples = ", ".join(malformed.index.astype(str).tolist()[:5])
        raise DataContractError(f"Expected two team-stat rows per game; malformed: {examples}")

    attempts = _numeric(stats, "attempts", 0.0)
    carries = _numeric(stats, "carries", 0.0)
    sacks = _numeric(stats, "sacks_suffered", 0.0).fillna(0.0)
    pass_plays = (attempts + sacks).replace(0.0, np.nan)
    rush_plays = carries.replace(0.0, np.nan)
    total_plays = (pass_plays.fillna(0.0) + rush_plays.fillna(0.0)).replace(0.0, np.nan)

    passing_epa = _numeric(stats, "passing_epa")
    rushing_epa = _numeric(stats, "rushing_epa")
    passing_yards = _numeric(stats, "passing_yards")
    rushing_yards = _numeric(stats, "rushing_yards")
    turnovers = _sum_available(
        stats,
        (
            "passing_interceptions",
            "sack_fumbles_lost",
            "rushing_fumbles_lost",
            "receiving_fumbles_lost",
        ),
    )

    metrics = stats[["game_id", "season", "gameday", "team", "home_team", "away_team"]].copy()
    metrics["off_epa_per_play"] = (passing_epa.fillna(0.0) + rushing_epa.fillna(0.0)) / total_plays
    metrics["off_pass_epa_per_play"] = passing_epa / pass_plays
    metrics["off_rush_epa_per_play"] = rushing_epa / rush_plays
    metrics["off_cpoe"] = _numeric(stats, "passing_cpoe")
    metrics["off_yards_per_play"] = (
        passing_yards.fillna(0.0) + rushing_yards.fillna(0.0)
    ) / total_plays
    metrics["off_turnover_rate"] = turnovers / total_plays
    metrics["off_sack_rate"] = sacks / pass_plays

    is_home = stats["team"].eq(stats["home_team"])
    home_score = _numeric(stats, "home_score")
    away_score = _numeric(stats, "away_score")
    metrics["point_diff"] = np.where(is_home, home_score - away_score, away_score - home_score)
    team_spread = np.where(is_home, _numeric(stats, "spread_line"), -_numeric(stats, "spread_line"))
    metrics["ats_residual"] = metrics["point_diff"] - team_spread

    offense_columns = [
        "off_epa_per_play",
        "off_pass_epa_per_play",
        "off_rush_epa_per_play",
        "off_yards_per_play",
        "off_turnover_rate",
        "off_sack_rate",
    ]
    opponent = metrics[["game_id", "team", *offense_columns]].rename(
        columns={
            "team": "opponent_team",
            "off_epa_per_play": "def_epa_per_play",
            "off_pass_epa_per_play": "def_pass_epa_per_play",
            "off_rush_epa_per_play": "def_rush_epa_per_play",
            "off_yards_per_play": "def_yards_per_play",
            "off_turnover_rate": "def_takeaway_rate",
            "off_sack_rate": "def_sack_rate",
        }
    )
    paired = metrics.merge(opponent, on="game_id", how="inner", validate="many_to_many")
    paired = paired.loc[paired["team"].ne(paired["opponent_team"])].copy()
    if len(paired) != len(metrics):
        raise DataContractError("Unable to pair each team-stat row with exactly one opponent")

    return paired[["game_id", "season", "gameday", "team", *STATE_METRICS]].sort_values(
        ["gameday", "game_id", "team"]
    )


def build_team_states(
    team_games: pd.DataFrame,
    span: int = 8,
    min_periods: int = 3,
    offseason_retention: float = 0.67,
) -> pd.DataFrame:
    """Calculate state after each completed game.

    These rows deliberately include the game on the row. `attach_team_states`
    uses a strict earlier-than lookup, making that state available only to the
    team's next game.
    """

    if span < 2:
        raise ValueError("span must be at least 2")
    if min_periods < 1:
        raise ValueError("min_periods must be positive")
    if not 0.0 <= offseason_retention <= 1.0:
        raise ValueError("offseason_retention must be between 0 and 1")

    states = team_games.copy().sort_values(["team", "gameday", "game_id"])
    states["season"] = pd.to_numeric(states["season"], errors="raise").astype(int)
    alpha = 2.0 / (span + 1.0)
    for metric in STATE_METRICS:
        states[metric] = pd.to_numeric(states[metric], errors="coerce")
        league_means = states.groupby("season", sort=False)[metric].mean().to_dict()
        states[f"league_mean_{metric}"] = states["season"].map(league_means)
        output = pd.Series(np.nan, index=states.index, dtype="float64")
        for _, group in states.groupby("team", sort=False):
            current = math.nan
            observations = 0
            previous_season: int | None = None
            for index, row in group.iterrows():
                season = int(row["season"])
                if (
                    previous_season is not None
                    and season != previous_season
                    and math.isfinite(current)
                ):
                    league_mean = float(league_means.get(previous_season, 0.0))
                    gap = max(1, season - previous_season)
                    retention = offseason_retention**gap
                    current = league_mean + retention * (current - league_mean)
                value = float(row[metric])
                if math.isfinite(value):
                    current = (
                        value
                        if not math.isfinite(current)
                        else alpha * value + (1.0 - alpha) * current
                    )
                    observations += 1
                if observations >= min_periods:
                    output.at[index] = current
                previous_season = season
        states[f"state_{metric}"] = output
    states["team_games"] = states.groupby(["team", "season"], sort=False).cumcount() + 1
    return states[
        [
            "game_id",
            "season",
            "gameday",
            "team",
            "team_games",
            *[f"state_{m}" for m in STATE_METRICS],
            *[f"league_mean_{m}" for m in STATE_METRICS],
        ]
    ]


def attach_team_states(
    games: pd.DataFrame,
    states: pd.DataFrame,
    offseason_retention: float = 0.67,
) -> pd.DataFrame:
    """Attach the most recent state strictly before each game's date."""

    result = games.copy()
    state_columns = [f"state_{metric}" for metric in STATE_METRICS]
    groups: dict[str, pd.DataFrame] = {
        str(team): group.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        for team, group in states.groupby("team", sort=False)
    }

    for side in ("home", "away"):
        for metric in STATE_METRICS:
            result[f"{side}_{metric}"] = np.nan
        result[f"{side}_team_games"] = np.nan

        for index, game in result.iterrows():
            team = str(game[f"{side}_team"])
            history = groups.get(team)
            if history is None or history.empty:
                continue
            dates = history["gameday"].to_numpy(dtype="datetime64[ns]")
            game_date = np.datetime64(pd.Timestamp(game["gameday"]), "ns")
            position = int(np.searchsorted(dates, game_date, side="left")) - 1
            if position < 0:
                continue
            state = history.iloc[position]
            game_season = int(game["season"])
            state_season = int(state["season"])
            season_gap = game_season - state_season
            result.at[index, f"{side}_team_games"] = state["team_games"] if season_gap == 0 else 0
            for metric, state_column in zip(STATE_METRICS, state_columns, strict=True):
                value = state[state_column]
                if season_gap > 0 and pd.notna(value):
                    league_mean = state[f"league_mean_{metric}"]
                    retention = offseason_retention ** max(1, season_gap)
                    value = league_mean + retention * (value - league_mean)
                result.at[index, f"{side}_{metric}"] = value

    for metric in STATE_METRICS:
        result[f"diff_{metric}"] = result[f"home_{metric}"] - result[f"away_{metric}"]
    return result


def add_elo_features(
    games: pd.DataFrame,
    k_factor: float = 20.0,
    home_field_elo: float = 55.0,
    offseason_retention: float = 0.67,
) -> pd.DataFrame:
    """Add pregame Elo ratings using only previously completed games."""

    result = games.copy().sort_values(["gameday", "game_id"]).reset_index(drop=True)
    result["home_elo"] = np.nan
    result["away_elo"] = np.nan
    result["elo_diff"] = np.nan
    result["elo_home_win_prob"] = np.nan
    ratings: dict[str, float] = {}
    previous_season: int | None = None

    for index, game in result.iterrows():
        season = int(game["season"])
        if previous_season is not None and season != previous_season:
            ratings = {
                team: 1500.0 + (rating - 1500.0) * offseason_retention
                for team, rating in ratings.items()
            }
        previous_season = season

        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_rating = ratings.setdefault(home_team, 1500.0)
        away_rating = ratings.setdefault(away_team, 1500.0)
        neutral = str(game.get("location", "Home")).lower() == "neutral"
        home_advantage = 0.0 if neutral else home_field_elo
        difference = home_rating + home_advantage - away_rating
        expected_home = 1.0 / (1.0 + 10.0 ** (-difference / 400.0))

        result.at[index, "home_elo"] = home_rating
        result.at[index, "away_elo"] = away_rating
        result.at[index, "elo_diff"] = difference
        result.at[index, "elo_home_win_prob"] = expected_home

        game_result = pd.to_numeric(pd.Series([game.get("result")]), errors="coerce").iloc[0]
        if pd.isna(game_result):
            continue
        observed_home = 1.0 if game_result > 0 else 0.0 if game_result < 0 else 0.5
        adjustment = k_factor * (observed_home - expected_home)
        ratings[home_team] = home_rating + adjustment
        ratings[away_team] = away_rating - adjustment

    return result


def build_game_features(
    schedules: pd.DataFrame,
    team_stats: pd.DataFrame,
    span: int = 8,
    min_periods: int = 3,
    offseason_retention: float = 0.67,
    graph_half_life_weeks: float = 8.0,
    graph_ridge_alpha: float = 8.0,
    graph_min_games: int = 16,
) -> pd.DataFrame:
    """Build the canonical model table with one row per regular-season game."""

    games = add_ats_outcomes(_regular_season_schedules(schedules))
    games["kickoff"] = _kickoff_utc(games)
    games = add_elo_features(games)
    games = add_schedule_strength_features(
        games,
        GraphRatingConfig(
            half_life_weeks=graph_half_life_weeks,
            offseason_retention=offseason_retention,
            ridge_alpha=graph_ridge_alpha,
            min_games=graph_min_games,
        ),
    )
    team_games = build_team_game_metrics(games, team_stats)
    states = build_team_states(
        team_games,
        span=span,
        min_periods=min_periods,
        offseason_retention=offseason_retention,
    )
    games = attach_team_states(games, states, offseason_retention=offseason_retention)

    for column in (
        "total_line",
        "home_rest",
        "away_rest",
        "div_game",
        "temp",
        "wind",
        "home_spread_odds",
        "away_spread_odds",
        "home_score",
        "away_score",
    ):
        if column not in games:
            games[column] = np.nan
        games[column] = _numeric(games, column)

    games["rest_diff"] = games["home_rest"] - games["away_rest"]
    location = (
        games["location"]
        if "location" in games
        else pd.Series("Home", index=games.index, dtype="string")
    )
    games["neutral_site"] = location.astype(str).str.lower().eq("neutral").astype(int)
    games["week_sin"] = np.sin(2.0 * np.pi * games["week"] / 18.0)
    games["week_cos"] = np.cos(2.0 * np.pi * games["week"] / 18.0)

    for column in MODEL_FEATURE_COLUMNS:
        if column not in games:
            games[column] = np.nan
        games[column] = pd.to_numeric(games[column], errors="coerce")

    games = games.replace([np.inf, -np.inf], np.nan)
    if "location" not in games:
        games["location"] = "Home"
    for column in ("weekday", "gametime"):
        if column not in games:
            games[column] = pd.NA
    passthrough = [
        *IDENTIFIER_COLUMNS,
        "weekday",
        "gametime",
        "kickoff",
        "game_type",
        "location",
        "home_spread_odds",
        "away_spread_odds",
        *OUTCOME_COLUMNS,
    ]
    ordered = list(dict.fromkeys([*passthrough, *MODEL_FEATURE_COLUMNS, *GRAPH_FEATURE_COLUMNS]))
    return games[ordered].sort_values(["gameday", "game_id"]).reset_index(drop=True)
