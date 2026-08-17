"""Leak-safe, regularized opponent adjustment for team-game PBP metrics.

The module is two layers. The lower one — :class:`OpponentEffects`,
:func:`opponent_adjustment_weeks`, :func:`eligible_opponent_history`, and
:func:`fit_opponent_effects` — is the league-agnostic estimator and its
point-in-time contract: one weighted ridge decomposition of an observed
team-game metric into an offense effect and an opposing-defense effect, fit
only from strictly earlier weeks and strictly earlier game dates. The upper
one is the NFL feature builder that turns those effects into matchup
expectations.

Other leagues (see ``nfl_ats.cfb_opponent_adjustment``) reuse the lower layer
rather than re-implementing it, so there is exactly one estimator and one
leakage contract to audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nfl_ats.constants import (
    PBP_OPPONENT_ADJUSTMENT_METRICS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError, require_columns

# Every opponent-adjustment history table needs these columns, whatever the
# league: who played whom, when, and in which ordered week.
OPPONENT_HISTORY_COLUMNS: tuple[str, ...] = ("team", "opponent", "season", "week", "gameday")


@dataclass(frozen=True)
class OpponentEffects:
    """One week's decomposition of a metric into offense and defense effects.

    ``offense[t] + defense[o] + intercept`` is the expected value of the
    metric when team ``t`` faces opponent ``o``. Teams absent from the fit's
    eligible history keep a 0.0 effect, i.e. they are treated as league
    average until they have been observed.
    """

    intercept: float
    offense: dict[str, float]
    defense: dict[str, float]

    def expectation(self, team: str, opponent: str) -> float:
        """The matchup expectation for ``team`` facing ``opponent``."""

        return self.intercept + self.offense.get(team, 0.0) + self.defense.get(opponent, 0.0)

    def offense_rating(self, team: str) -> float:
        """``team``'s opponent-adjusted metric against an average defense."""

        return self.intercept + self.offense.get(team, 0.0)

    def defense_rating(self, team: str) -> float:
        """``team``'s opponent-adjusted metric allowed to an average offense."""

        return self.intercept + self.defense.get(team, 0.0)


def _canonical_team(value: object) -> str:
    team = str(value)
    return TEAM_ABBREVIATION_ALIASES.get(team, team)


def opponent_adjustment_weeks(games: pd.DataFrame) -> pd.DataFrame:
    """Every (season, week) with its earliest kickoff, in chronological order.

    The earliest kickoff of a week is that week's cutoff: no game played in
    the week — not even one played days before the rest of it — is eligible
    for the week's own fit.
    """

    require_columns(games, ("season", "week", "gameday"), "opponent adjustment games")
    return (
        games[["season", "week", "gameday"]]
        .groupby(["season", "week"], sort=True, as_index=False)
        .agg(cutoff=("gameday", "min"))
        .sort_values("cutoff")
    )


def eligible_opponent_history(
    history: pd.DataFrame, *, season: int, week: int, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """History a (season, week) fit may see: earlier week AND earlier date.

    Both conditions are required, so a mislabeled week cannot smuggle a
    same-day or later game into an earlier week's fit, and a game labeled to
    an earlier week but played after the cutoff is excluded as well.
    """

    prior_week = (history["season"].lt(season)) | (
        history["season"].eq(season) & history["week"].lt(week)
    )
    return history.loc[prior_week & history["gameday"].lt(cutoff)]


def fit_opponent_effects(
    history: pd.DataFrame,
    *,
    metric: str,
    teams: tuple[str, ...],
    cutoff: pd.Timestamp,
    half_life_weeks: float,
    ridge_alpha: float,
    min_team_games: int,
    include_opponent: bool = True,
) -> OpponentEffects | None:
    """Decompose one metric into offense and opposing-defense effects.

    ``history`` must already be restricted to rows the cutoff allows (see
    :func:`eligible_opponent_history`). Observations are weighted by an
    exponential decay in weeks before ``cutoff``. Returns ``None`` when the
    eligible history is too thin to fit, so callers leave the week unscored
    instead of inventing a value.

    ``include_opponent=False`` drops the opposing-defense block, leaving a
    time-decayed, ridge-shrunk team mean and an all-zero defense map. That is
    the control for "what does the *opponent* block buy?": everything else —
    decay, penalty, warm-up, cutoff — is held identical.
    """

    usable = history.loc[history[metric].notna()].copy()
    if (
        len(usable) < min_team_games
        or usable["team"].nunique() < 2
        or usable["opponent"].nunique() < 2
    ):
        return None

    universe = pd.Index(teams)
    team_positions = universe.get_indexer(pd.Index(usable["team"].astype(str)))
    opponent_positions = universe.get_indexer(pd.Index(usable["opponent"].astype(str)))
    if (team_positions < 0).any() or (opponent_positions < 0).any():
        raise DataContractError("Opponent adjustment history names a team outside the declared set")
    blocks = 2 if include_opponent else 1
    design = np.zeros((len(usable), len(teams) * blocks), dtype=float)
    rows = np.arange(len(usable))
    design[rows, team_positions] = 1.0
    if include_opponent:
        design[rows, len(teams) + opponent_positions] = 1.0

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
    positions = {team: index for index, team in enumerate(teams)}
    offense = {team: float(coefficients[index]) for team, index in positions.items()}
    defense = {
        team: float(coefficients[len(teams) + index]) if include_opponent else 0.0
        for team, index in positions.items()
    }
    return OpponentEffects(intercept=float(estimator.intercept_), offense=offense, defense=defense)


def validate_opponent_adjustment_parameters(
    *, half_life_weeks: float, ridge_alpha: float, min_team_games: int
) -> None:
    """Reject adjustment parameters that cannot produce a meaningful fit."""

    if not math.isfinite(half_life_weeks) or half_life_weeks <= 0:
        raise ValueError("half_life_weeks must be positive")
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive")
    if min_team_games < 2:
        raise ValueError("min_team_games must be at least 2")


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

    validate_opponent_adjustment_parameters(
        half_life_weeks=half_life_weeks,
        ridge_alpha=ridge_alpha,
        min_team_games=min_team_games,
    )

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

    for week in opponent_adjustment_weeks(result).itertuples(index=False):
        season = int(str(week.season))
        week_number = int(str(week.week))
        cutoff = pd.Timestamp(str(week.cutoff))
        eligible = eligible_opponent_history(
            history, season=season, week=week_number, cutoff=cutoff
        )
        target_indexes = result.index[result["season"].eq(season) & result["week"].eq(week_number)]
        for source, derived in PBP_OPPONENT_ADJUSTMENT_METRICS:
            effects = fit_opponent_effects(
                eligible,
                metric=source,
                teams=teams,
                cutoff=cutoff,
                half_life_weeks=half_life_weeks,
                ridge_alpha=ridge_alpha,
                min_team_games=min_team_games,
            )
            if effects is None:
                continue
            for index in target_indexes:
                home = str(canonical_home.at[index])
                away = str(canonical_away.at[index])
                home_expectation = effects.expectation(home, away)
                away_expectation = effects.expectation(away, home)
                result.at[index, f"home_{derived}"] = home_expectation
                result.at[index, f"away_{derived}"] = away_expectation
                result.at[index, f"diff_{derived}"] = home_expectation - away_expectation

    return result
