"""Leak-safe, regularized opponent adjustment for team-game PBP metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nfl_ats.constants import (
    PBP_OPPONENT_ADJUSTMENT_METRICS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError, require_columns


def _canonical_team(value: object) -> str:
    team = str(value)
    return TEAM_ABBREVIATION_ALIASES.get(team, team)


def _fit_metric(
    history: pd.DataFrame,
    *,
    metric: str,
    teams: tuple[str, ...],
    cutoff: pd.Timestamp,
    half_life_weeks: float,
    ridge_alpha: float,
    min_team_games: int,
) -> tuple[float, dict[str, float], dict[str, float]] | None:
    usable = history.loc[history[metric].notna()].copy()
    if (
        len(usable) < min_team_games
        or usable["team"].nunique() < 2
        or usable["opponent"].nunique() < 2
    ):
        return None

    positions = {team: index for index, team in enumerate(teams)}
    design = np.zeros((len(usable), len(teams) * 2), dtype=float)
    for row_index, (team, opponent) in enumerate(
        zip(usable["team"], usable["opponent"], strict=True)
    ):
        design[row_index, positions[str(team)]] = 1.0
        design[row_index, len(teams) + positions[str(opponent)]] = 1.0

    age_weeks = (cutoff - usable["gameday"]).dt.total_seconds().to_numpy(dtype=float) / (
        7.0 * 24.0 * 60.0 * 60.0
    )
    weights = np.power(0.5, np.maximum(age_weeks, 0.0) / half_life_weeks)
    estimator = Ridge(alpha=ridge_alpha, fit_intercept=True)
    estimator.fit(
        design,
        pd.to_numeric(usable[metric], errors="raise").to_numpy(dtype=float),
        sample_weight=weights,
    )
    coefficients = np.asarray(estimator.coef_, dtype=float)
    offense = {team: float(coefficients[index]) for team, index in positions.items()}
    defense = {team: float(coefficients[len(teams) + index]) for team, index in positions.items()}
    return float(estimator.intercept_), offense, defense


def add_opponent_adjusted_pbp_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    half_life_weeks: float = 16.0,
    ridge_alpha: float = 10.0,
    min_team_games: int = 64,
) -> pd.DataFrame:
    """Add weekly, point-in-time matchup expectations from prior PBP games.

    For each efficiency metric, a weighted ridge model decomposes an observed
    team-game result into an offensive-team effect and an opposing-defense
    effect. Every game in an NFL week is scored before that week's observations
    are eligible for fitting.
    """

    if not math.isfinite(half_life_weeks) or half_life_weeks <= 0:
        raise ValueError("half_life_weeks must be positive")
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive")
    if min_team_games < 2:
        raise ValueError("min_team_games must be at least 2")

    require_columns(
        games,
        ("game_id", "season", "week", "gameday", "home_team", "away_team"),
        "game features",
    )
    require_columns(
        team_games,
        (
            "game_id",
            "season",
            "week",
            "team",
            "opponent",
            *(source for source, _ in PBP_OPPONENT_ADJUSTMENT_METRICS),
        ),
        "PBP team games",
    )
    if team_games.duplicated(["game_id", "team"]).any():
        raise DataContractError("PBP team games contain duplicate game/team rows")

    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    if result[["home_team", "away_team"]].isna().any(axis=None):
        raise DataContractError("game features contain a null home or away team")
    canonical_home = result["home_team"].map(_canonical_team)
    canonical_away = result["away_team"].map(_canonical_team)

    history = team_games.copy()
    history["team"] = history["team"].map(_canonical_team)
    history["opponent"] = history["opponent"].map(_canonical_team)
    if history["team"].eq(history["opponent"]).any():
        raise DataContractError("PBP team games contain a team paired with itself")
    game_dates = result[["game_id", "gameday"]].drop_duplicates("game_id")
    history = history.merge(game_dates, on="game_id", how="left", validate="many_to_one")
    if history["gameday"].isna().any():
        raise DataContractError("PBP contains games absent from the canonical schedule table")
    history["gameday"] = pd.to_datetime(history["gameday"], errors="raise")

    teams = tuple(
        sorted(
            set(history["team"])
            | set(history["opponent"])
            | set(canonical_home)
            | set(canonical_away)
        )
    )
    for _, derived in PBP_OPPONENT_ADJUSTMENT_METRICS:
        for side in ("home", "away"):
            result[f"{side}_{derived}"] = np.nan
        result[f"diff_{derived}"] = np.nan

    week_order = (
        result[["season", "week", "gameday"]]
        .groupby(["season", "week"], sort=True, as_index=False)
        .agg(cutoff=("gameday", "min"))
        .sort_values("cutoff")
    )
    for week in week_order.itertuples(index=False):
        season = int(str(week.season))
        week_number = int(str(week.week))
        cutoff = pd.Timestamp(str(week.cutoff))
        prior_week = (history["season"].lt(season)) | (
            history["season"].eq(season) & history["week"].lt(week_number)
        )
        eligible = history.loc[prior_week & history["gameday"].lt(cutoff)]
        target_indexes = result.index[result["season"].eq(season) & result["week"].eq(week_number)]
        for source, derived in PBP_OPPONENT_ADJUSTMENT_METRICS:
            fitted = _fit_metric(
                eligible,
                metric=source,
                teams=teams,
                cutoff=cutoff,
                half_life_weeks=half_life_weeks,
                ridge_alpha=ridge_alpha,
                min_team_games=min_team_games,
            )
            if fitted is None:
                continue
            intercept, offense, defense = fitted
            for index in target_indexes:
                home = str(canonical_home.at[index])
                away = str(canonical_away.at[index])
                home_expectation = intercept + offense.get(home, 0.0) + defense.get(away, 0.0)
                away_expectation = intercept + offense.get(away, 0.0) + defense.get(home, 0.0)
                result.at[index, f"home_{derived}"] = home_expectation
                result.at[index, f"away_{derived}"] = away_expectation
                result.at[index, f"diff_{derived}"] = home_expectation - away_expectation

    return result
