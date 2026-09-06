"""Interactive dashboard presentation over the normal, guarded site content.

Assets are packaged with the renderer and inlined so every generated page is
self-contained. No saved preview, alternate forecast, or browser storage is used.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nfl_ats.board_content import BoardContent, GameRow

_ROOT = Path(__file__).parent
_STYLE = (_ROOT / "board_interactive.css").read_text(encoding="utf-8")
_SCRIPTS = "\n".join(
    (_ROOT / f"board_interactive_{name}.js").read_text(encoding="utf-8")
    for name in ("layout", "motion", "experience")
)


def _lineup_date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    local = parsed.astimezone(ZoneInfo("America/New_York"))
    return f"{local:%A, %B} {local.day}, {local.year} at {local:%I:%M %p %Z}"


def _final_scores(game: GameRow) -> dict[str, int] | None:
    if not game.final or not game.final_score_text:
        return None
    match = re.fullmatch(
        rf"{re.escape(game.away)} (\d+) at {re.escape(game.home)} (\d+)",
        game.final_score_text,
    )
    return {"away": int(match[1]), "home": int(match[2])} if match else None


def card_payload(board: BoardContent) -> dict[str, object]:
    """Use exactly the picks, explanations and lineups rendered on this card."""
    dives = {dive.game_id: dive for dive in board.dives}
    games = []
    for game in board.games:
        dive = dives.get(game.game_id)
        lineups = (
            {
                lineup.team: _lineup_date(lineup.as_of)
                for lineup in (dive.away_lineup, dive.home_lineup)
                if lineup is not None
            }
            if dive
            else {}
        )
        games.append(
            {
                "id": game.game_id,
                "away": game.away,
                "home": game.home,
                "pick": game.pick_team,
                "spread": -game.market_spread
                if game.pick_team == game.home
                else game.market_spread,
                "score": game.probability_text,
                "kickoff": f"{game.weekday_name}, {game.gameday:%B} {game.gameday.day}",
                "explanation": game.explanation_text,
                "adjusted": game.is_flipped,
                "final": _final_scores(game),
                "lineups": lineups,
            }
        )
    return {
        "season": board.season,
        "week": board.week,
        "weekLabel": board.week_label,
        "games": games,
    }


def enhance(document: str, *, page: str, board: BoardContent) -> str:
    """Apply the approved interactive design to a freshly rendered site page."""
    key = "week" if page == "index.html" else Path(page).stem
    payload = json.dumps(card_payload(board), ensure_ascii=True, allow_nan=False).replace(
        "<", "\\u003c"
    )
    document = document.replace("</head>", f"<style>{_STYLE}</style>\n</head>", 1)
    document = document.replace("<body>", f'<body data-interactive-page="{key}">', 1)
    return document.replace(
        "</body>",
        f'<script id="interactive-card" type="application/json">{payload}</script>\n'
        "<script>window.BALL_CARD = JSON.parse("
        'document.getElementById("interactive-card").textContent);'
        f"\n{_SCRIPTS}\n</script>\n</body>",
        1,
    )
