from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.pbp import analysis_plays, load_pbp_snapshot, snapshot_from_root
from nfl_ats.pbp08_matchup_flags import MIN_WINDOW_OBS, WINDOW_GAMES
from nfl_ats.published_picks import frozen_picks, game_deadlines

SCHEDULE_SEASON_START = 2020


def _active_forecast(artifacts_root: Path) -> tuple[pd.DataFrame, int, int]:
    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise DataContractError("No synchronized active ATS model is available")
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise DataContractError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    recommendations_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not recommendations_path.is_file():
        raise DataContractError("Linked weekly forecast is missing metadata or recommendations")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(recommendations_path)
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "kickoff",
        "home_team",
        "away_team",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(
            "Active forecast is missing pressure coverage columns: " + ", ".join(missing)
        )
    return predictions, int(metadata["season"]), int(metadata["week"])


def _unfrozen_targets(
    artifacts_root: Path,
    predictions: pd.DataFrame,
    *,
    season: int,
    week: int,
    now: datetime,
) -> pd.DataFrame:
    deadlines = game_deadlines(predictions)
    game_ids = predictions["game_id"].astype(str)
    missing_deadlines = sorted(set(game_ids).difference(deadlines))
    if missing_deadlines:
        raise DataContractError(
            "Pressure coverage cannot determine pool deadlines for active forecast games: "
            + ", ".join(missing_deadlines)
        )
    locked = frozen_picks(
        artifacts_root,
        now=now,
        season=season,
        week=week,
    )
    instant = pd.Timestamp(now.astimezone(UTC))
    protected = {
        game_id for game_id in game_ids if game_id in locked and deadlines[game_id] <= instant
    }
    return predictions.loc[~game_ids.isin(protected)].copy()


def _latest_schedules(data_root: Path, target_season: int) -> pd.DataFrame:
    candidates = sorted(data_root.glob("raw/*/schedules.parquet"), reverse=True)
    if not candidates:
        raise DataContractError("No schedules snapshot is available for pressure coverage")
    schedule = pd.read_parquet(candidates[0])
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "game_type",
        "home_team",
        "away_team",
    }
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(
            "Schedules are missing pressure coverage columns: " + ", ".join(missing)
        )
    schedule = schedule.loc[schedule["game_type"].astype(str).eq("REG")].copy()
    seasons = pd.to_numeric(schedule["season"], errors="coerce")
    schedule = schedule.loc[seasons.between(SCHEDULE_SEASON_START, target_season)].copy()
    schedule["season"] = pd.to_numeric(schedule["season"], errors="raise").astype(int)
    schedule["week"] = pd.to_numeric(schedule["week"], errors="raise").astype(int)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    if schedule["gameday"].isna().any():
        raise DataContractError("Schedules contain games without a usable gameday")
    for column in ("home_team", "away_team"):
        schedule[column] = schedule[column].astype(str).replace(TEAM_ABBREVIATION_ALIASES)
    return schedule


def _required_team_games(schedule: pd.DataFrame, targets: pd.DataFrame) -> list[dict[str, Any]]:
    schedule_ids = set(schedule["game_id"].astype(str))
    target_ids = set(targets["game_id"].astype(str))
    missing_targets = sorted(target_ids.difference(schedule_ids))
    if missing_targets:
        raise DataContractError(
            "Active forecast games are missing from the current schedules snapshot: "
            + ", ".join(missing_targets)
        )
    sides = []
    for column in ("home_team", "away_team"):
        sides.append(
            schedule[["game_id", "season", "week", "gameday", column]]
            .rename(columns={column: "team"})
            .copy()
        )
    team_games = pd.concat(sides, ignore_index=True)
    team_games["game_id"] = team_games["game_id"].astype(str)
    team_games = team_games.sort_values(["team", "gameday", "game_id"])
    required: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        target_season = int(str(target.season))
        target_week = int(str(target.week))
        target_day = pd.to_datetime(str(target.gameday), errors="coerce")
        if pd.isna(target_day):
            raise DataContractError(f"Active forecast game {target.game_id} has no usable gameday")
        for raw_team in (target.home_team, target.away_team):
            team = str(TEAM_ABBREVIATION_ALIASES.get(str(raw_team), str(raw_team)))
            prior = team_games.loc[
                team_games["team"].eq(team)
                & (
                    team_games["season"].lt(target_season)
                    | (team_games["season"].eq(target_season) & team_games["week"].lt(target_week))
                )
                & team_games["gameday"].lt(target_day)
            ].tail(WINDOW_GAMES)
            if len(prior) < MIN_WINDOW_OBS:
                continue
            for game in prior.itertuples(index=False):
                required.append(
                    {
                        "target_game_id": str(target.game_id),
                        "team": team,
                        "game_id": str(game.game_id),
                    }
                )
    return required


def _pressure_usable_by_team(data_root: Path) -> tuple[Path, set[tuple[str, str]]]:
    raw_root = data_root / "pbp" / "raw"
    if not raw_root.is_dir():
        raise DataContractError(f"No play-by-play snapshot directory is available at {raw_root}")
    directories = sorted((path for path in raw_root.iterdir() if path.is_dir()), reverse=True)
    if not directories:
        raise DataContractError(f"No play-by-play snapshot is available in {raw_root}")
    directory = directories[0]
    try:
        snapshot = snapshot_from_root(directory)
        plays = analysis_plays(load_pbp_snapshot(snapshot))
    except (FileNotFoundError, KeyError, OSError, ValueError, DataContractError) as error:
        raise DataContractError(
            f"Latest play-by-play input {directory} is incomplete or unusable ({error}); "
            "complete or remove that input and refresh play-by-play before publishing"
        ) from error
    plays = plays.loc[plays["competitive_play"]].copy()
    for column in ("qb_dropback", "sack", "qb_hit"):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    plays["qb_dropback_present"] = plays["qb_dropback"].notna()
    qb_complete: list[set[tuple[str, str]]] = []
    for team_column in ("posteam", "defteam"):
        plays[team_column] = plays[team_column].replace(TEAM_ABBREVIATION_ALIASES)
        grouped = plays.groupby(["game_id", team_column], sort=False)["qb_dropback_present"].agg(
            ["size", "all"]
        )
        complete: set[tuple[str, str]] = set()
        for key, row in grouped.iterrows():
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and int(row["size"]) > 0
                and bool(row["all"])
            ):
                complete.add((str(key[0]), str(key[1])))
        qb_complete.append(complete)
    dropbacks = plays.loc[plays["qb_dropback"].eq(1)].copy()
    dropbacks["pressure_fields_present"] = dropbacks[["sack", "qb_hit"]].notna().all(axis=1)
    pressure_complete: list[set[tuple[str, str]]] = []
    for team_column in ("posteam", "defteam"):
        grouped = dropbacks.groupby(["game_id", team_column], sort=False)[
            "pressure_fields_present"
        ].agg(["size", "all"])
        complete = set()
        for key, row in grouped.iterrows():
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and int(row["size"]) > 0
                and bool(row["all"])
            ):
                complete.add((str(key[0]), str(key[1])))
        pressure_complete.append(complete)
    usable = qb_complete[0] & qb_complete[1] & pressure_complete[0] & pressure_complete[1]
    return directory, usable


def require_pressure_coverage_for_unlocked_card(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> None:
    instant = now or datetime.now(UTC)
    predictions, season, week = _active_forecast(artifacts_root)
    targets = _unfrozen_targets(
        artifacts_root,
        predictions,
        season=season,
        week=week,
        now=instant,
    )
    if targets.empty:
        return
    schedule = _latest_schedules(data_root, int(pd.to_numeric(targets["season"]).max()))
    required = _required_team_games(schedule, targets)
    if not required:
        return
    snapshot_path, usable = _pressure_usable_by_team(data_root)
    missing = [row for row in required if (row["game_id"], row["team"]) not in usable]
    if not missing:
        return
    details = ", ".join(
        f"{row['game_id']}:{row['team']} (for {row['target_game_id']})" for row in missing[:12]
    )
    suffix = f"; plus {len(missing) - 12} more" if len(missing) > 12 else ""
    raise DataContractError(
        "Pressure coverage is incomplete for unlocked served probabilities. "
        f"Latest play-by-play snapshot {snapshot_path.name} lacks usable competitive-dropback "
        f"sack/qb_hit input for rolling-window team-games: {details}{suffix}. "
        "Refresh play-by-play through the completed prior games before publishing; null pressure "
        "fields are missing input, while recorded zero values are valid."
    )
