from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.travel_geometry import haversine_mi, load_stadium_coordinate_registry

DAYS_PER_WEEK = 7
PRIOR_GAME_DAY_NAMES = {"Thursday", "Monday"}


def _previous_game(games: pd.DataFrame, team: str, season: int, week: int) -> pd.Series | None:
    played = games[
        (games["season"] == season)
        & (games["week"] < week)
        & ((games["home_team"] == team) | (games["away_team"] == team))
    ]
    return played.sort_values("week").iloc[-1] if len(played) else None


def _team_note(games: pd.DataFrame, team: str, season: int, week: int, rest: object) -> str:
    previous = _previous_game(games, team, season, week)
    notes: list[str] = []
    if previous is not None and int(previous["week"]) < week - 1:
        notes.append("off a bye")
    elif previous is not None and str(previous["weekday"]) in PRIOR_GAME_DAY_NAMES:
        notes.append(f"off a {previous['weekday']} game")
    if pd.notna(rest) and int(rest) < DAYS_PER_WEEK:
        notes.append("short week")
    days = f"{team} {int(rest)} days" if pd.notna(rest) else f"{team} rest unknown"
    return days + (f" ({', '.join(notes)})" if notes else "")


def _home_stadium(games: pd.DataFrame, team: str, season: int, week: int) -> str | None:
    home = games[
        (games["home_team"] == team) & (games["season"] == season) & (games["week"] < week)
    ].sort_values("week")
    if not len(home):
        home = games[(games["home_team"] == team) & (games["season"] == season)].sort_values("week")
    return str(home.iloc[-1]["stadium"]) if len(home) else None


def build_schedule_facts(
    data_root: Path, *, season: int, week: int, game_ids: set[str]
) -> dict[str, str]:
    schedules, _ = load_snapshot(latest_snapshot(Path(data_root) / "raw"))
    venues = load_stadium_coordinate_registry().venues
    out: dict[str, str] = {}
    for row in schedules[
        (schedules["season"] == season) & (schedules["week"] == week)
    ].itertuples():
        if row.game_id not in game_ids:
            continue
        away_note = _team_note(schedules, str(row.away_team), season, week, row.away_rest)
        home_note = _team_note(schedules, str(row.home_team), season, week, row.home_rest)
        text = f"Rest: {away_note}, {home_note}."
        origin_name = _home_stadium(schedules, str(row.away_team), season, week)
        origin = venues.get(origin_name) if origin_name else None
        destination = venues.get(str(row.stadium))
        if origin is not None and destination is not None:
            miles = round(haversine_mi(origin, destination), -1)
            text += f" {row.away_team} travels about {int(miles):,} miles."
        out[row.game_id] = text
    return out
