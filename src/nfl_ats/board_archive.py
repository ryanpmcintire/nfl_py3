from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from nfl_ats.board_content import AttributionPanel, BoardContent, GameDive, GameRow
from nfl_ats.board_site_content import HistoryPickRow, SiteContent
from nfl_ats.tiebreaker import newest_schedules_path


def _game(row: HistoryPickRow, gameday: date) -> GameRow:
    result = (
        "push"
        if row.status.lower() == "push"
        else "win"
        if row.correct is True
        else "loss"
        if row.correct is False
        else None
    )
    score = None
    if row.score_text:
        away_score, home_score = row.score_text.split(" at ")
        score = f"{row.away_team} {away_score} at {row.home_team} {home_score}"
    return GameRow(
        game_id=row.game_id,
        gameday=gameday,
        weekday_name=gameday.strftime("%A"),
        home=row.home_team,
        away=row.away_team,
        market_spread=(
            row.decision_home_spread if row.decision_home_spread is not None else float("nan")
        ),
        pick_team=row.pick_team,
        pick_probability=row.confidence if row.confidence is not None else float("nan"),
        confidence_word="",
        is_best=row.best_pick,
        is_flipped=False,
        final=score is not None,
        cover_result=result,
        final_score_text=score,
    )


def _dive(game: GameRow) -> GameDive:
    return GameDive(
        game_id=game.game_id,
        matchup_label=f"{game.away} at {game.home}",
        pick_team=game.pick_team,
        pick_spread_text=game.pick_spread_text,
        home=game.home,
        kickoff_group_label=game.kickoff_group_label,
        probability_text=game.probability_text,
        is_best=game.is_best,
        attribution=AttributionPanel(
            available=False,
            unavailable_note="The game breakdown was not saved with this pick.",
        ),
        cover_curve=(),
        cover_curve_offset_zero_note=None,
        adjuster=None,
    )


def build_archived_boards(
    content: SiteContent, data_root: Path, current: tuple[int, int]
) -> tuple[BoardContent, ...]:
    grouped: dict[tuple[int, int], list[HistoryPickRow]] = {}
    for row in content.history.picks:
        if row.season is not None and row.week is not None and (row.season, row.week) < current:
            grouped.setdefault((row.season, row.week), []).append(row)
    if not grouped:
        return ()
    schedule = pd.read_parquet(newest_schedules_path(data_root), columns=["game_id", "gameday"])
    dates = {
        str(game_id): pd.Timestamp(gameday).date()
        for game_id, gameday in zip(schedule["game_id"], schedule["gameday"], strict=True)
    }
    boards = []
    for (season, week), rows in sorted(grouped.items(), reverse=True):
        games = tuple(
            sorted(
                (_game(row, dates[row.game_id]) for row in rows),
                key=lambda game: (game.gameday, game.game_id),
            )
        )
        boards.append(
            replace(
                content.board,
                season=season,
                week=week,
                week_label=f"Week {week}",
                games=games,
                dives=tuple(_dive(game) for game in games),
                best_pick_game_id=next((game.game_id for game in games if game.is_best), None),
                best_pick_note="",
                best_pick_ranking=(),
                best_pick_gap_points=None,
                flip_count=0,
                strong_count=0,
                calibrated_probability=False,
            )
        )
    return tuple(boards)
