"""nflverse acquisition and input schema validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.constants import SCHEDULE_REQUIRED_COLUMNS, TEAM_STATS_REQUIRED_COLUMNS
from nfl_ats.snapshots import Snapshot, write_snapshot


class DataContractError(ValueError):
    """Raised when upstream data no longer satisfies the expected contract."""


def require_columns(frame: pd.DataFrame, required: Iterable[str], dataset: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise DataContractError(f"{dataset} is missing required columns: {', '.join(missing)}")


def validate_schedules(frame: pd.DataFrame) -> None:
    require_columns(frame, SCHEDULE_REQUIRED_COLUMNS, "schedules")
    if frame["game_id"].isna().any():
        raise DataContractError("schedules contains null game_id values")
    duplicates = frame.loc[frame["game_id"].duplicated(), "game_id"].unique()
    if len(duplicates):
        raise DataContractError(f"schedules contains duplicate game_id values: {duplicates[:5]}")
    same_team = frame["home_team"].eq(frame["away_team"])
    if same_team.any():
        raise DataContractError("schedules contains a game with identical home and away teams")


def validate_team_stats(frame: pd.DataFrame) -> None:
    require_columns(frame, TEAM_STATS_REQUIRED_COLUMNS, "team_stats")
    key_columns = ["game_id", "team"]
    if frame[key_columns].isna().any(axis=None):
        raise DataContractError("team_stats contains null game_id or team values")
    duplicates = frame.duplicated(key_columns)
    if duplicates.any():
        raise DataContractError("team_stats contains duplicate game_id/team rows")


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        converted = frame.to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted
    raise TypeError(f"Unsupported dataframe type: {type(frame)!r}")


def fetch_nflverse(
    seasons: list[int],
    raw_root: Path,
    team_stat_seasons: list[int] | None = None,
) -> Snapshot:
    """Download schedules and weekly team stats into an immutable snapshot."""

    if not seasons:
        raise ValueError("At least one season is required")
    if seasons != sorted(set(seasons)):
        raise ValueError("Seasons must be unique and sorted")
    resolved_team_stat_seasons = seasons if team_stat_seasons is None else team_stat_seasons
    if not resolved_team_stat_seasons:
        raise ValueError("At least one team-stat season is required")
    if resolved_team_stat_seasons != sorted(set(resolved_team_stat_seasons)):
        raise ValueError("Team-stat seasons must be unique and sorted")
    if not set(resolved_team_stat_seasons).issubset(seasons):
        raise ValueError("Team-stat seasons must be included in schedule seasons")

    import nflreadpy as nfl

    schedules = _to_pandas(nfl.load_schedules(seasons=seasons))
    team_stats = _to_pandas(
        nfl.load_team_stats(seasons=resolved_team_stat_seasons, summary_level="week")
    )
    validate_schedules(schedules)
    validate_team_stats(team_stats)
    return write_snapshot(
        schedules,
        team_stats,
        seasons,
        raw_root,
        team_stat_seasons=resolved_team_stat_seasons,
    )


def check_nflverse_contract(schedule_season: int, stats_season: int) -> dict[str, Any]:
    """Fetch small current source slices and fail loudly on schema drift."""

    import nflreadpy as nfl

    schedules = _to_pandas(nfl.load_schedules(seasons=[schedule_season]))
    team_stats = _to_pandas(nfl.load_team_stats(seasons=[stats_season], summary_level="week"))
    validate_schedules(schedules)
    validate_team_stats(team_stats)
    if schedules.empty:
        raise DataContractError(f"schedules returned no rows for {schedule_season}")
    if team_stats.empty:
        raise DataContractError(f"team_stats returned no rows for {stats_season}")
    return {
        "checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "schedule_season": schedule_season,
        "schedule_rows": len(schedules),
        "stats_season": stats_season,
        "team_stat_rows": len(team_stats),
        "required_schedule_columns": list(SCHEDULE_REQUIRED_COLUMNS),
        "required_team_stat_columns": list(TEAM_STATS_REQUIRED_COLUMNS),
    }
