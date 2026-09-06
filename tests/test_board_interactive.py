"""The interactive site must follow the published card across refreshes."""

from __future__ import annotations

import json
import re
from dataclasses import replace

from _board_content_fixtures import build_fixture_content

from nfl_ats import board_interactive, board_terminal
from nfl_ats.lineup_view import TeamLineup


def test_interactive_payload_tracks_new_week_and_changed_pick() -> None:
    board = build_fixture_content()
    game = replace(
        board.games[0],
        game_id="2027_08_NE_SEA",
        pick_team="NE",
        market_spread=6.5,
        pick_probability=0.612,
        explanation_text="Updated explanation </script><script>bad()</script>",
    )
    board = replace(board, season=2027, week=8, week_label="Week 8", games=(game,))
    html = board_interactive.enhance(board_terminal.render(board), page="index.html", board=board)
    match = re.search(r'<script id="interactive-card" type="application/json">(.*?)</script>', html)
    assert match is not None
    payload = json.loads(match[1])
    assert payload["season"] == 2027
    assert payload["weekLabel"] == "Week 8"
    assert payload["games"][0]["pick"] == "NE"
    assert payload["games"][0]["spread"] == 6.5
    assert payload["games"][0]["score"] == "61.2%"
    assert payload["games"][0]["explanation"] == game.explanation_text
    assert "<script>bad()" not in match[1]


def test_receipts_require_confirmed_final_and_preserve_home_spread_sign() -> None:
    board = build_fixture_content()
    game = replace(board.games[0], pick_team="SEA", market_spread=3.5)
    pending = replace(game, final=False, final_score_text="NE 21 at SEA 24")
    final = replace(game, final=True, final_score_text="NE 21 at SEA 24")
    malformed = replace(final, final_score_text="score unavailable")
    payload = board_interactive.card_payload(replace(board, games=(pending, final, malformed)))
    games = payload["games"]
    assert isinstance(games, list)
    assert games[0]["final"] is None
    assert games[1]["final"] == {"away": 21, "home": 24}
    assert games[1]["spread"] == -3.5
    assert games[2]["final"] is None


def test_lineup_date_comes_from_same_rendered_lineup() -> None:
    board = build_fixture_content()
    lineup = TeamLineup("NE", (), "2027-10-20T15:38:00Z", None, "unavailable")
    dive = replace(board.dives[0], away_lineup=lineup)
    payload = board_interactive.card_payload(replace(board, dives=(dive,)))
    games = payload["games"]
    assert isinstance(games, list)
    assert games[0]["lineups"]["NE"] == "Wednesday, October 20, 2027 at 11:38 AM EDT"
