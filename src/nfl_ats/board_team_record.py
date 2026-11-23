from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from nfl_ats.board_content import _grade_decisions, _record_text

RELOCATED_TEAM_CODES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
DECISION_COLUMNS = {"season", "week", "pick_side", "home_team", "away_team", "game_id"}


def _team(code: object) -> str:
    text = str(code)
    return RELOCATED_TEAM_CODES.get(text, text)


def _season_records(
    decisions: pd.DataFrame, outcomes: pd.DataFrame, *, season: int, week: int
) -> dict[str, str]:
    if decisions.empty or not DECISION_COLUMNS.issubset(decisions.columns):
        return {}
    seasons = pd.to_numeric(decisions["season"], errors="coerce")
    weeks = pd.to_numeric(decisions["week"], errors="coerce")
    prior = decisions.loc[seasons.eq(season) & weeks.lt(week)]
    if prior.empty:
        return {}
    picked = (
        prior["home_team"]
        .where(prior["pick_side"].astype(str).eq("HOME"), prior["away_team"])
        .map(_team)
    )
    out: dict[str, str] = {}
    for team, rows in prior.groupby(picked):
        wins, losses, pushes, _pending = _grade_decisions(rows, outcomes)
        if wins + losses + pushes:
            out[str(team)] = f"{_record_text(wins, losses, pushes)} this season"
    return out


def build_team_records(
    artifacts_root: Path,
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    season: int,
    week: int,
    picks: Mapping[str, str],
) -> dict[str, str]:
    this_season = _season_records(decisions, outcomes, season=season, week=week)
    return {
        game_id: this_season[_team(team)]
        for game_id, team in picks.items()
        if _team(team) in this_season
    }
