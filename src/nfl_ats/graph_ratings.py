"""Leak-safe temporal graph and regularized schedule-strength ratings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from nfl_ats.constants import GRAPH_FEATURE_COLUMNS

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GraphRatingConfig:
    """Fixed research configuration for schedule-strength features."""

    half_life_weeks: float = 8.0
    offseason_retention: float = 0.67
    damping: float = 0.85
    prior_weight: float = 1.0
    ridge_alpha: float = 8.0
    min_games: int = 16
    offseason_age_weeks: int = 8

    def validate(self) -> None:
        if self.half_life_weeks <= 0:
            raise ValueError("half_life_weeks must be positive")
        if not 0.0 <= self.offseason_retention <= 1.0:
            raise ValueError("offseason_retention must be between 0 and 1")
        if not 0.0 < self.damping < 1.0:
            raise ValueError("damping must be between 0 and 1")
        if self.prior_weight <= 0:
            raise ValueError("prior_weight must be positive")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.min_games < 1:
            raise ValueError("min_games must be positive")
        if self.offseason_age_weeks < 0:
            raise ValueError("offseason_age_weeks cannot be negative")


def _standardize(values: FloatArray) -> FloatArray:
    centered = values - float(values.mean())
    scale = float(values.std())
    if not math.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values)
    return centered / scale


def _prior_matrix(size: int, weight: float) -> FloatArray:
    prior = np.full((size, size), weight, dtype=np.float64)
    np.fill_diagonal(prior, 0.0)
    return prior


def _expand_matrix(
    matrix: FloatArray,
    existing_teams: list[str],
    desired_teams: list[str],
) -> FloatArray:
    if existing_teams == desired_teams:
        return matrix
    expanded = np.zeros((len(desired_teams), len(desired_teams)), dtype=np.float64)
    desired_index = {team: index for index, team in enumerate(desired_teams)}
    for old_row, row_team in enumerate(existing_teams):
        new_row = desired_index[row_team]
        for old_column, column_team in enumerate(existing_teams):
            expanded[new_row, desired_index[column_team]] = matrix[old_row, old_column]
    return expanded


def _page_rank(
    adjacency: FloatArray,
    *,
    damping: float,
    tolerance: float = 1e-12,
    iterations: int = 500,
) -> FloatArray:
    size = adjacency.shape[0]
    if size == 0:
        return np.empty(0, dtype=np.float64)
    row_sums = adjacency.sum(axis=1)
    transition: FloatArray = np.divide(
        adjacency,
        row_sums[:, None],
        out=np.full_like(adjacency, 1.0 / size),
        where=row_sums[:, None] > 0,
    )
    scores = np.full(size, 1.0 / size, dtype=np.float64)
    teleport = (1.0 - damping) / size
    for _ in range(iterations):
        updated: FloatArray = np.asarray(
            teleport + damping * (transition.T @ scores), dtype=np.float64
        )
        if float(np.abs(updated - scores).sum()) <= tolerance:
            return updated / float(updated.sum())
        scores = updated
    return scores / float(scores.sum())


def _hits(adjacency: FloatArray, *, iterations: int = 500) -> tuple[FloatArray, FloatArray]:
    """Return hub and authority vectors from a nonnegative weighted graph."""

    size = adjacency.shape[0]
    if size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    authority = np.full(size, 1.0 / math.sqrt(size), dtype=np.float64)
    hub = authority.copy()
    for _ in range(iterations):
        new_hub = adjacency @ authority
        hub_norm = float(np.linalg.norm(new_hub))
        if hub_norm > 0:
            new_hub /= hub_norm
        new_authority = adjacency.T @ new_hub
        authority_norm = float(np.linalg.norm(new_authority))
        if authority_norm > 0:
            new_authority /= authority_norm
        change = float(np.abs(new_hub - hub).sum() + np.abs(new_authority - authority).sum())
        hub, authority = new_hub, new_authority
        if change <= 1e-12:
            break
    return hub, authority


def _ridge_ratings(
    history: list[tuple[str, str, float, int, int]],
    teams: list[str],
    *,
    current_step: int,
    current_season: int,
    config: GraphRatingConfig,
) -> tuple[dict[str, float], float]:
    if len(history) < config.min_games:
        return dict.fromkeys(teams, math.nan), math.nan
    team_index = {team: index for index, team in enumerate(teams)}
    design = np.zeros((len(history), len(teams)), dtype=np.float64)
    targets = np.empty(len(history), dtype=np.float64)
    weights = np.empty(len(history), dtype=np.float64)
    for row, (home, away, margin, step, season) in enumerate(history):
        design[row, team_index[home]] = 1.0
        design[row, team_index[away]] = -1.0
        targets[row] = margin
        age = current_step - step
        age += config.offseason_age_weeks * max(0, current_season - season)
        weights[row] = 0.5 ** (age / config.half_life_weeks)
    estimator = Ridge(alpha=config.ridge_alpha, fit_intercept=True)
    estimator.fit(design, targets, sample_weight=weights)
    ratings = {team: float(estimator.coef_[team_index[team]]) for team in teams}
    return ratings, float(estimator.intercept_)


def _empty_feature_frame(games: pd.DataFrame) -> pd.DataFrame:
    output = games.copy()
    for column in GRAPH_FEATURE_COLUMNS:
        output[column] = np.nan
    return output


def add_schedule_strength_features(
    games: pd.DataFrame,
    config: GraphRatingConfig | None = None,
) -> pd.DataFrame:
    """Add pregame PageRank, HITS, and ridge schedule ratings.

    Ratings for every game in an NFL week are computed before any result from
    that week is incorporated. Graph edges decay weekly, and prior seasons are
    additionally shrunk toward a symmetric graph. The ridge comparator uses
    the same historical games and an equivalent time-decay policy.
    """

    settings = config or GraphRatingConfig()
    settings.validate()
    required = {"season", "week", "gameday", "game_id", "home_team", "away_team", "result"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise ValueError(f"Graph ratings require columns: {', '.join(missing)}")

    result = _empty_feature_frame(games)
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    result = result.sort_values(["gameday", "game_id"]).copy()
    graph_teams: list[str] = []
    performance = np.zeros((0, 0), dtype=np.float64)
    scoring = np.zeros((0, 0), dtype=np.float64)
    history: list[tuple[str, str, float, int, int]] = []
    weekly_decay = 0.5 ** (1.0 / settings.half_life_weeks)
    previous_season: int | None = None
    step = 0

    for (season_value, _), weekly_games in result.groupby(["season", "week"], sort=True):
        season = int(str(season_value))
        if previous_season is not None and season != previous_season:
            gap = max(1, season - previous_season)
            retention = settings.offseason_retention**gap
            performance *= retention
            scoring *= retention
        previous_season = season
        performance *= weekly_decay
        scoring *= weekly_decay

        current_teams = set(weekly_games["home_team"].astype(str)) | set(
            weekly_games["away_team"].astype(str)
        )
        desired_teams = sorted(set(graph_teams) | current_teams)
        performance = _expand_matrix(performance, graph_teams, desired_teams)
        scoring = _expand_matrix(scoring, graph_teams, desired_teams)
        graph_teams = desired_teams
        team_index = {team: index for index, team in enumerate(graph_teams)}

        if len(history) >= settings.min_games:
            prior = _prior_matrix(len(graph_teams), settings.prior_weight)
            page_rank = _standardize(_page_rank(performance + prior, damping=settings.damping))
            defensive_vulnerability, offensive_strength = _hits(scoring + prior)
            offense = _standardize(offensive_strength)
            defense = -_standardize(defensive_vulnerability)
            ridge, home_field = _ridge_ratings(
                history,
                graph_teams,
                current_step=step,
                current_season=season,
                config=settings,
            )

            for index, game in weekly_games.iterrows():
                home = str(game["home_team"])
                away = str(game["away_team"])
                home_index = team_index[home]
                away_index = team_index[away]
                home_offense = float(offense[home_index])
                away_offense = float(offense[away_index])
                home_defense = float(defense[home_index])
                away_defense = float(defense[away_index])
                result.at[index, "home_graph_pagerank"] = page_rank[home_index]
                result.at[index, "away_graph_pagerank"] = page_rank[away_index]
                result.at[index, "graph_pagerank_diff"] = (
                    page_rank[home_index] - page_rank[away_index]
                )
                result.at[index, "home_graph_offense"] = home_offense
                result.at[index, "away_graph_offense"] = away_offense
                result.at[index, "home_graph_defense"] = home_defense
                result.at[index, "away_graph_defense"] = away_defense
                result.at[index, "graph_matchup_diff"] = (home_offense - away_defense) - (
                    away_offense - home_defense
                )
                result.at[index, "home_schedule_rating"] = ridge[home]
                result.at[index, "away_schedule_rating"] = ridge[away]
                result.at[index, "schedule_rating_diff"] = ridge[home] - ridge[away]
                result.at[index, "schedule_predicted_margin"] = (
                    home_field + ridge[home] - ridge[away]
                )

        for _, game in weekly_games.iterrows():
            margin = pd.to_numeric(pd.Series([game["result"]]), errors="coerce").iloc[0]
            if pd.isna(margin):
                continue
            home = str(game["home_team"])
            away = str(game["away_team"])
            home_index = team_index[home]
            away_index = team_index[away]
            numeric_margin = float(margin)
            if numeric_margin > 0:
                performance[away_index, home_index] += 1.0 + min(numeric_margin, 28.0) / 14.0
            elif numeric_margin < 0:
                performance[home_index, away_index] += 1.0 + min(-numeric_margin, 28.0) / 14.0
            else:
                performance[home_index, away_index] += 0.5
                performance[away_index, home_index] += 0.5

            home_score = pd.to_numeric(pd.Series([game.get("home_score")]), errors="coerce").iloc[0]
            away_score = pd.to_numeric(pd.Series([game.get("away_score")]), errors="coerce").iloc[0]
            if pd.notna(home_score) and pd.notna(away_score):
                # Edge direction follows the old project: defense -> offense.
                scoring[away_index, home_index] += max(0.0, float(home_score))
                scoring[home_index, away_index] += max(0.0, float(away_score))
            history.append((home, away, numeric_margin, step, season))
        step += 1

    for column in GRAPH_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values(["gameday", "game_id"]).reset_index(drop=True)
