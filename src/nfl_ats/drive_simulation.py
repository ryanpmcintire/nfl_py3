"""Leak-safe empirical simulation at the offensive-drive level."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Final, SupportsIndex, cast

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.pbp import analysis_plays, build_drive_table

_GAME_SECONDS: Final = 3_600
_LATE_GAME_SECONDS: Final = 900
_OBSERVATION_COLUMNS: Final = (
    "posteam",
    "defteam",
    "game_state",
    "start_yardline_100",
    "drive_seconds",
    "outcome",
    "points",
)


@dataclass(frozen=True)
class DriveSimulatorModel:
    """Pregame drive observations frozen at an explicit training cutoff."""

    observations: pd.DataFrame
    training_max_gameday: str
    training_games: int
    training_drives: int


@dataclass(frozen=True)
class DriveSimulationResult:
    """Simulation-level scores and the possession trace that produced them."""

    games: pd.DataFrame
    drives: pd.DataFrame
    samples: int
    seed: int


def _as_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = index(cast(SupportsIndex, value))
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _game_state(seconds_remaining: int | float, score_differential: int | float) -> str:
    if seconds_remaining > _LATE_GAME_SECONDS or score_differential == 0:
        return "neutral"
    return "late_leading" if score_differential > 0 else "late_trailing"


def _outcome(row: pd.Series) -> str:
    points = int(row["points"])
    result = str(row["result"]).lower()
    if points >= 6 or "touchdown" in result:
        return "touchdown"
    if points == 3 or ("field goal" in result and "miss" not in result):
        return "field_goal"
    if bool(row["drive_turnover"]):
        return "turnover"
    if "punt" in result:
        return "punt"
    if "field goal" in result:
        return "missed_field_goal"
    return "scoreless"


def _validate_schedule(games: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "gameday", "home_team", "away_team"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise DataContractError(
            f"Drive simulator schedule is missing columns: {', '.join(missing)}"
        )
    if games.empty:
        raise ValueError("Drive simulator requires at least one scheduled game")
    if games["game_id"].isna().any() or games["game_id"].duplicated().any():
        raise DataContractError("Drive simulator schedule requires unique, non-null game_id values")
    if games[["home_team", "away_team"]].isna().any(axis=None):
        raise DataContractError("Drive simulator schedule requires both teams")
    if games["home_team"].astype(str).eq(games["away_team"].astype(str)).any():
        raise DataContractError("Drive simulator schedule cannot match a team against itself")
    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="coerce")
    if result["gameday"].isna().any():
        raise DataContractError("Drive simulator schedule requires valid gamedays")
    return result


def fit_drive_simulator(
    pbp: pd.DataFrame,
    games: pd.DataFrame,
    *,
    training_max_gameday: str | pd.Timestamp,
) -> DriveSimulatorModel:
    """Fit empirical drive pools using only games at or before ``training_max_gameday``.

    The schedule supplies the point-in-time boundary. Play-by-play belonging to
    later games may be present in ``pbp`` but is excluded before any drive pool
    is constructed, making the cutoff invariant directly regression-testable.
    """

    schedule = _validate_schedule(games)
    cutoff = pd.to_datetime(training_max_gameday, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("training_max_gameday must be a valid date")
    cutoff = pd.Timestamp(cutoff).normalize()
    training_schedule = schedule.loc[schedule["gameday"].dt.normalize().le(cutoff)].copy()
    if training_schedule.empty:
        raise DataContractError("Drive simulator cutoff leaves no training games")

    training_ids = set(training_schedule["game_id"].astype(str))
    pbp_game_ids = pbp["game_id"].astype(str) if "game_id" in pbp else pd.Series(dtype="string")
    training_pbp = pbp.loc[pbp_game_ids.isin(training_ids)].copy()
    drives = build_drive_table(training_pbp)
    if drives.empty:
        raise DataContractError("Drive simulator found no eligible training drives")

    plays = analysis_plays(training_pbp)
    drive_keys = ["game_id", "season", "week", "posteam", "defteam", "fixed_drive"]
    drive_states = (
        plays.sort_values(["game_id", "fixed_drive", "play_id"])
        .groupby(drive_keys, sort=False, dropna=False)
        .agg(score_differential=("score_differential", "first"))
        .reset_index()
    )
    drives = drives.merge(
        drive_states,
        on=drive_keys,
        how="left",
        validate="one_to_one",
    )
    for column in (
        "start_yardline_100",
        "drive_seconds",
        "drive_points",
        "start_game_seconds",
        "score_differential",
    ):
        drives[column] = pd.to_numeric(drives[column], errors="coerce")
    drives = drives.loc[
        drives["posteam"].notna()
        & drives["defteam"].notna()
        & drives["start_yardline_100"].between(1.0, 99.0, inclusive="both")
        & drives["drive_seconds"].notna()
        & drives["drive_points"].notna()
        & drives["start_game_seconds"].notna()
    ].copy()
    if drives.empty:
        raise DataContractError("Drive simulator found no complete training drives")

    drives["points"] = np.rint(drives["drive_points"].clip(lower=0, upper=8)).astype(int)
    drives["game_state"] = [
        _game_state(seconds, differential)
        for seconds, differential in zip(
            drives["start_game_seconds"], drives["score_differential"].fillna(0), strict=True
        )
    ]
    drives["outcome"] = drives.apply(_outcome, axis="columns")
    observations = drives.loc[:, list(_OBSERVATION_COLUMNS)].reset_index(drop=True)
    observations["posteam"] = observations["posteam"].astype(str)
    observations["defteam"] = observations["defteam"].astype(str)
    observations["drive_seconds"] = observations["drive_seconds"].clip(lower=1, upper=900)

    observed_training_games = int(drives["game_id"].nunique())
    return DriveSimulatorModel(
        observations=observations,
        training_max_gameday=cutoff.date().isoformat(),
        training_games=observed_training_games,
        training_drives=len(observations),
    )


def _candidate_pool(
    observations: pd.DataFrame,
    *,
    offense: str,
    defense: str,
    game_state: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, str]:
    state_rows = observations.loc[observations["game_state"].eq(game_state)]
    offense_rows = state_rows.loc[state_rows["posteam"].eq(offense)]
    defense_rows = state_rows.loc[state_rows["defteam"].eq(defense)]
    if not offense_rows.empty and not defense_rows.empty:
        if bool(rng.integers(0, 2)):
            return offense_rows, "offense_state"
        return defense_rows, "defense_state"
    if not offense_rows.empty:
        return offense_rows, "offense_state"
    if not defense_rows.empty:
        return defense_rows, "defense_state"
    if not state_rows.empty:
        return state_rows, "league_state"

    neutral = observations.loc[observations["game_state"].eq("neutral")]
    offense_rows = neutral.loc[neutral["posteam"].eq(offense)]
    defense_rows = neutral.loc[neutral["defteam"].eq(defense)]
    if not offense_rows.empty and not defense_rows.empty:
        if bool(rng.integers(0, 2)):
            return offense_rows, "offense_neutral"
        return defense_rows, "defense_neutral"
    if not offense_rows.empty:
        return offense_rows, "offense_neutral"
    if not defense_rows.empty:
        return defense_rows, "defense_neutral"
    return observations, "league_all"


def simulate_drive_distribution(
    model: DriveSimulatorModel,
    future_games: pd.DataFrame,
    *,
    samples: int = 1_000,
    seed: int = 42,
    max_drives_per_game: int = 64,
) -> DriveSimulationResult:
    """Simulate regulation games as alternating empirical possessions.

    One historical drive row supplies the starting field position, outcome,
    points, and pace together, retaining their observed dependence. In the
    final 15 minutes, a possession uses separate leading or trailing pools.
    Team offense and opponent defense pools are sampled symmetrically when
    both exist; named fallback levels are retained in the audit trace.
    """

    sample_count = _as_positive_integer(samples, "samples")
    max_drives = _as_positive_integer(max_drives_per_game, "max_drives_per_game")
    games = _validate_schedule(future_games)
    cutoff = pd.Timestamp(model.training_max_gameday)
    if games["gameday"].dt.normalize().le(cutoff.normalize()).any():
        raise DataContractError(
            "Drive simulation is leak-safe only when every target gameday is strictly "
            "after the model training cutoff"
        )
    observations = model.observations
    missing = sorted(set(_OBSERVATION_COLUMNS).difference(observations.columns))
    if missing or observations.empty:
        detail = f": {', '.join(missing)}" if missing else ""
        raise DataContractError(f"Drive simulator model has no usable observations{detail}")

    rng = np.random.default_rng(seed)
    game_rows: list[dict[str, object]] = []
    drive_rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        game_id = str(game.game_id)
        home_team = str(game.home_team)
        away_team = str(game.away_team)
        for simulation_id in range(sample_count):
            scores = {home_team: 0, away_team: 0}
            possessions = {home_team: 0, away_team: 0}
            offense = home_team if bool(rng.integers(0, 2)) else away_team
            first_possession = offense
            seconds_remaining = _GAME_SECONDS
            drive_number = 0
            while seconds_remaining > 0 and drive_number < max_drives:
                defense = away_team if offense == home_team else home_team
                differential = scores[offense] - scores[defense]
                state = _game_state(seconds_remaining, differential)
                pool, profile_source = _candidate_pool(
                    observations,
                    offense=offense,
                    defense=defense,
                    game_state=state,
                    rng=rng,
                )
                sampled = pool.iloc[int(rng.integers(0, len(pool)))]
                sampled_duration = max(1, round(float(sampled["drive_seconds"])))
                duration = min(seconds_remaining, sampled_duration)
                points = int(sampled["points"])
                before = seconds_remaining
                scores[offense] += points
                possessions[offense] += 1
                seconds_remaining -= duration
                drive_rows.append(
                    {
                        "game_id": game_id,
                        "simulation_id": simulation_id,
                        "drive_number": drive_number,
                        "offense": offense,
                        "defense": defense,
                        "game_state": state,
                        "profile_source": profile_source,
                        "offense_score_before": scores[offense] - points,
                        "defense_score_before": scores[defense],
                        "game_seconds_before": before,
                        "start_yardline_100": float(sampled["start_yardline_100"]),
                        "outcome": str(sampled["outcome"]),
                        "points": points,
                        "sampled_drive_seconds": sampled_duration,
                        "drive_seconds": duration,
                        "game_seconds_after": seconds_remaining,
                    }
                )
                drive_number += 1
                offense = defense
            game_rows.append(
                {
                    "game_id": game_id,
                    "gameday": pd.Timestamp(str(game.gameday)),
                    "simulation_id": simulation_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "first_possession": first_possession,
                    "home_score": scores[home_team],
                    "away_score": scores[away_team],
                    "home_margin": scores[home_team] - scores[away_team],
                    "home_possessions": possessions[home_team],
                    "away_possessions": possessions[away_team],
                    "total_possessions": drive_number,
                    "clock_expired": seconds_remaining == 0,
                    "training_max_gameday": model.training_max_gameday,
                    "seed": seed,
                }
            )
    return DriveSimulationResult(
        games=pd.DataFrame(game_rows),
        drives=pd.DataFrame(drive_rows),
        samples=sample_count,
        seed=seed,
    )
