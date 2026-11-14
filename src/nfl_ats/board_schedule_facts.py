from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.travel_geometry import (
    StadiumLocation,
    _offset_hours,
    haversine_mi,
    load_stadium_coordinate_registry,
)

DAYS_PER_WEEK = 7
PRIOR_GAME_DAY_NAMES = {"Thursday", "Monday"}
NEUTRAL_LOCATION = "Neutral"
GAMES_IN_A_ROW_MINIMUM = 2
ORDINAL_WORDS = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}


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
    true_home = games[(games["home_team"] == team) & (games["location"] != NEUTRAL_LOCATION)]
    home = true_home[(true_home["season"] == season) & (true_home["week"] < week)].sort_values(
        "week"
    )
    if not len(home):
        home = true_home[true_home["season"] == season].sort_values("week")
    return str(home.iloc[-1]["stadium"]) if len(home) else None


def _away_streak(games: pd.DataFrame, team: str, season: int, week: int) -> int:
    through_this_week = games[
        (games["season"] == season)
        & (games["week"] <= week)
        & ((games["home_team"] == team) | (games["away_team"] == team))
    ].sort_values("week", ascending=False)
    streak = 0
    for game in through_this_week.itertuples():
        if str(game.away_team) != team and str(game.location) != NEUTRAL_LOCATION:
            break
        streak += 1
    return streak


def _streak_note(games: pd.DataFrame, team: str, season: int, week: int) -> str:
    streak = _away_streak(games, team, season, week)
    if streak < GAMES_IN_A_ROW_MINIMUM:
        return ""
    ordinal = ORDINAL_WORDS.get(streak, f"{streak}th")
    return f"{team} is away from home for the {ordinal} game in a row."


def _clock_note(origin: StadiumLocation, destination: StadiumLocation, gameday: object) -> str:
    day = pd.Timestamp(str(gameday))
    shift = _offset_hours(destination, day) - _offset_hours(origin, day)
    if not shift:
        return ""
    hours = abs(shift)
    unit = "hour" if hours == 1 else "hours"
    return f"{hours:g} {unit} {'ahead of' if shift > 0 else 'behind'} home"


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
        destination = venues.get(str(row.stadium))
        neutral = str(row.location) == NEUTRAL_LOCATION
        travellers = (row.away_team, row.home_team) if neutral else (row.away_team,)
        legs: list[tuple[str, str, str]] = []
        for team in travellers:
            origin_name = _home_stadium(schedules, str(team), season, week)
            origin = venues.get(origin_name) if origin_name else None
            if origin is not None and destination is not None:
                miles = f"about {int(round(haversine_mi(origin, destination), -1)):,} miles"
                legs.append((str(team), miles, _clock_note(origin, destination, row.gameday)))
        if neutral and legs:
            sites = ", ".join(
                f"{team} {miles}" + (f" ({clock})" if clock else "") for team, miles, clock in legs
            )
            text += f" Played at {row.stadium}, a neutral site: {sites}."
        elif legs:
            team, miles, clock = legs[0]
            text += f" {team} travels {miles}" + (f" to a time zone {clock}" if clock else "") + "."
        streaks = [
            note
            for team in (row.away_team, row.home_team)
            if (note := _streak_note(schedules, str(team), season, week))
        ]
        if streaks:
            text += " " + " ".join(streaks)
        out[row.game_id] = text
    return out
