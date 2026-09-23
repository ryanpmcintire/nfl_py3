from __future__ import annotations

import html
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.board_site_content import HistoryPickRow, SiteContent
from nfl_ats.market_data_halves import current_week_kickoff_window
from nfl_ats.splash_lines import load_splash_capture
from nfl_ats.tiebreaker import newest_schedules_path


def _week_key(season: int, week: int) -> str:
    return f"{season}-{week}"


def _current_week(data_root: Path, generated_at: datetime) -> tuple[int, int]:
    schedule = pd.read_parquet(
        newest_schedules_path(data_root),
        columns=["season", "week", "gameday", "gametime"],
    )
    local_kickoffs = pd.to_datetime(
        schedule["gameday"].astype("string") + " " + schedule["gametime"].astype("string"),
        errors="coerce",
    )
    kickoffs = local_kickoffs.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    start, end = current_week_kickoff_window(generated_at)
    selected = schedule.loc[(kickoffs >= start) & (kickoffs < end), ["season", "week"]]
    records = cast(list[dict[str, Any]], selected.to_dict("records"))
    weeks = {
        (int(row["season"]), int(row["week"]))
        for row in records
        if pd.notna(row["season"]) and pd.notna(row["week"])
    }
    if len(weeks) != 1:
        raise ValueError("current NFL week is not unique in the schedule")
    return next(iter(weeks))


def _game_data(row: HistoryPickRow) -> dict[str, Any]:
    if row.status.lower() == "push":
        result = "Push"
    elif row.correct is True:
        result = "Won"
    elif row.correct is False:
        result = "Lost"
    else:
        result = "Pending"
    return {
        "awayTeam": row.away_team,
        "homeTeam": row.home_team,
        "pickTeam": row.pick_team,
        "pickLine": row.pick_line_text,
        "confidence": f"{row.confidence:.1%}" if row.confidence is not None else "—",
        "status": result,
        "correct": row.correct,
        "score": row.score_text,
        "bestPick": row.best_pick,
    }


def _archive_weeks(
    rows: tuple[HistoryPickRow, ...], current: tuple[int, int]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[HistoryPickRow]] = {}
    for row in rows:
        if row.season is None or row.week is None:
            continue
        key = (row.season, row.week)
        if key >= current:
            continue
        grouped.setdefault(key, []).append(row)
    weeks: list[dict[str, Any]] = []
    for season, week in sorted(grouped, reverse=True):
        games = [_game_data(row) for row in grouped[(season, week)]]
        weeks.append(
            {
                "season": season,
                "week": week,
                "key": _week_key(season, week),
                "label": f"{season} · Week {week}",
                "games": games,
            }
        )
    return weeks


def _readiness(
    current: tuple[int, int],
    published: tuple[int, int] | None,
    current_lines_available: bool | None,
) -> dict[str, str]:
    if published == current:
        return {"status": "ready", "message": "This week's published card is ready."}
    if current_lines_available is False:
        return {
            "status": "missing-pool-lines",
            "message": (
                f"Week {current[1]} picks are waiting for the pool's lines. "
                "Published cards remain available in the selector."
            ),
        }
    return {
        "status": "unavailable",
        "message": "This week's card is unavailable. Please try again after the next update.",
    }


def _archive_markup(weeks: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for week in weeks:
        rows: list[str] = []
        for index, game in enumerate(week["games"]):
            rows.append(
                f'<tr class="game week-archive-row{" is-best" if game["bestPick"] else ""}" '
                f'data-archive-index="{index}" tabindex="0"'
                f' aria-label="Inspect {html.escape(game["awayTeam"])} '
                f'at {html.escape(game["homeTeam"])}">'
                '<td class="kickoff" data-label="Kickoff">&mdash;</td>'
                '<td class="matchup" data-label="Matchup">'
                '<button type="button" class="week-game-link">'
                f"{html.escape(game['awayTeam'])} at <b>{html.escape(game['homeTeam'])}</b>"
                "</button></td>"
                '<td class="pick" data-label="Pick">'
                f"{html.escape(game['pickTeam'])} {html.escape(game['pickLine'])}</td>"
                '<td class="market-now" data-label="Books now">&mdash;</td>'
                f'<td class="prob" data-label="Cover chance">{html.escape(game["confidence"])}</td>'
                '<td class="flipline" data-label="Flips at">&mdash;</td>'
                '<td class="conf" data-label="Confidence">&mdash;</td></tr>'
            )
        sections.append(
            f'<div class="week-grid week-saved-card" data-week-panel="{week["key"]}">'
            '<section class="board-col"><div class="section-head">'
            f"<h2>Week {week['week']} / The complete card</h2>"
            f'<span class="sub">{len(week["games"])} games · select a game to inspect</span></div>'
            '<p class="week-saved-note">The picks and pool lines published for this week.</p>'
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Kickoff</th><th>Matchup</th><th>Pick</th><th>Books now</th>"
            "<th>Cover chance</th><th>Flips at</th><th>Confidence</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
            '<section class="inspector-col"><div class="section-head">'
            "<h2>Game inspector</h2>"
            '<span class="sub">The published pick and final result</span></div>'
            '<div class="week-archive-detail" aria-live="polite"></div></section></div>'
        )
    return f'<template id="week-archive">{"".join(sections)}</template>'


def enhance(
    document: str,
    content: SiteContent,
    *,
    data_root: Path | None = None,
    generated_at: datetime | None = None,
) -> str:
    now = (generated_at or datetime.now(UTC)).astimezone(UTC)
    resolved_data_root = data_root or Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))
    try:
        current = _current_week(resolved_data_root, now)
    except (OSError, ValueError, KeyError):
        module_path = Path(__file__).parent
        css = (module_path / "board_week_navigation.css").read_text(encoding="utf-8")
        browser = (
            '<section id="week-browser" aria-labelledby="week-browser-title">'
            '<h2 id="week-browser-title">Current week unavailable</h2>'
            '<p id="week-readiness" data-readiness="unavailable">'
            "The current NFL week could not be matched to the schedule. "
            "Please try again after the schedule refresh.</p></section>"
        )
        main_marker = '<main id="main-content"'
        main_start = document.find(main_marker)
        opening_end = document.find(">", main_start)
        if main_start < 0 or opening_end < 0:
            raise ValueError("board page is missing main-content") from None
        document = (
            document[:opening_end]
            + ' data-current-card-ready="false"'
            + document[opening_end : opening_end + 1]
            + browser
            + document[opening_end + 1 :]
        )
        document = document.replace("<body", '<body data-week-live-visible="true"', 1)
        return document.replace("</head>", f"<style>{css}</style></head>", 1)
    published = None
    if content.board.season is not None and content.board.week is not None and content.board.games:
        published = (content.board.season, content.board.week)
    weeks = _archive_weeks(content.history.picks, current)
    try:
        current_lines_available: bool | None = (
            load_splash_capture(resolved_data_root, *current) is not None
        )
    except (OSError, ValueError, KeyError):
        current_lines_available = None
    readiness = _readiness(current, published, current_lines_available)
    payload = {
        "current": {
            "season": current[0],
            "week": current[1],
            "key": _week_key(*current),
        },
        "published": (
            {
                "season": published[0],
                "week": published[1],
                "key": _week_key(*published),
            }
            if published is not None
            else None
        ),
        "readiness": readiness,
        "weeks": weeks,
    }
    options = [
        f'<option value="{_week_key(*current)}">'
        f"{current[0]} · Week {current[1]} — Current week</option>"
    ]
    options.extend(
        f'<option value="{week["key"]}">{week["label"]} — Published picks</option>'
        for week in weeks
    )
    readiness_hidden = " hidden" if readiness["status"] == "ready" else ""
    browser = (
        '<section id="week-browser" aria-labelledby="week-browser-label">'
        '<label id="week-browser-label" for="week-select">Choose a week</label>'
        f'<select id="week-select">{"".join(options)}</select>'
        '<button id="week-current" type="button" disabled>Current week</button>'
        f'<p id="week-readiness" data-readiness="{readiness["status"]}"{readiness_hidden}>'
        f"{html.escape(readiness['message'])}</p>"
        '<p id="week-selection-status" aria-live="polite">'
        f"{current[0]} · Week {current[1]} · Current week</p></section>"
    )
    archive = _archive_markup(weeks)
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    data_node = f'<script id="week-navigation-data" type="application/json">{data_json}</script>'
    module_path = Path(__file__).parent
    css = (module_path / "board_week_navigation.css").read_text(encoding="utf-8")
    javascript = (module_path / "board_week_navigation.js").read_text(encoding="utf-8")
    main_marker = '<main id="main-content"'
    main_start = document.find(main_marker)
    if main_start < 0:
        raise ValueError("board page is missing main-content")
    opening_end = document.find(">", main_start)
    if opening_end < 0:
        raise ValueError("board page has an incomplete main-content element")
    ready = published == current
    document = (
        document[:opening_end]
        + f' data-current-card-ready="{str(ready).lower()}"'
        + document[opening_end : opening_end + 1]
        + browser
        + document[opening_end + 1 :]
    )
    main_end = document.rfind("</main>")
    if main_end < 0:
        raise ValueError("board page is missing the main-content close tag")
    document = document[:main_end] + archive + data_node + document[main_end:]
    document = document.replace("<body", f'<body data-week-live-visible="{str(ready).lower()}"', 1)
    document = document.replace("</head>", f"<style>{css}</style></head>", 1)
    return document.replace("</body>", f"<script>{javascript}</script></body>", 1)


__all__ = ["enhance"]
