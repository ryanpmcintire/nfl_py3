"""The board's "Flips at" column (owner request, 2026-09-01): the first
half-point line at which the PLAYED pick would switch to the other team.

Semantics under test (see ``board_content._flip_line``): the played pick is
the raw model plus the four-member policy re-evaluated at each hypothetical
line -- ``played(L) = raw(L)`` complemented once if any member fires at L.
The spread-gap zone member fires purely on ``|L|`` in [7.5, 10] (for every
game), so a zone-flipped pick reverts on a half-point move out of the zone
and an unflipped pick near the zone flips by entering it (owner catch #1,
2026-09-01 -- the first draft dashed these out). The scan is bounded to the
±4 span the on-page chart and slider explore, and a pick nothing switches
in that span reports "held" -- a bounded claim on purpose (owner catch #2,
same day: an unbounded scan produced "at any line", asserting the pick at
absurd hypothetical spreads where the spread-blind fix-up rules have no
evidence).

Cell format (owner feedback, same day, replacing a first draft that printed
the flipped-to team's handicap in the opposite orientation from the Pick
column): the CURRENT pick's own handicap at the flip line, then the team it
switches to -- ``NYJ +2.5 → TEN``; the held state reads ``IND within ±4``.
"""

from __future__ import annotations

import math

import pandas as pd
from _board_content_fixtures import build_fixture_content

from nfl_ats import board_terminal
from nfl_ats.board_content import GameRow, _flip_line
from nfl_ats.spread_explorer import SpreadExplorerGameParams


def _params(
    card_line: float, center: float, game_id: str = "2026_01_NYJ_TEN"
) -> dict[str, SpreadExplorerGameParams]:
    return {
        game_id: SpreadExplorerGameParams(
            game_id=game_id,
            home_team="TEN",
            away_team="NYJ",
            center=center,
            residual_mean=0.0,
            residual_std=6.0,
            card_line=card_line,
            card_home_cover_probability=1.0
            - 0.5 * (1.0 + math.erf((card_line - center) / (6.0 * math.sqrt(2.0)))),
        )
    }


def test_widget_path_finds_the_owner_example_crossing() -> None:
    # Centre 2.7 with the card at 3.0: home cover probability is below 0.5
    # at the quoted line (pick NYJ), and crosses above it at 2.5 -- the real
    # Week 1 NYJ @ TEN shape the owner read off the adjuster.
    line, held = _flip_line("2026_01_NYJ_TEN", "TEN", "NYJ", (), pd.DataFrame(), _params(3.0, 2.7))
    assert (line, held) == (2.5, False)


def test_widget_path_flips_a_home_pick_upward() -> None:
    line, held = _flip_line("2026_01_NYJ_TEN", "TEN", "TEN", (), pd.DataFrame(), _params(3.0, 5.2))
    assert (line, held) == (5.5, False)


def test_unflipped_pick_near_the_zone_flips_by_entering_it() -> None:
    # DET -7 shape: home pick, raw model likes home well past 7.5 (centre
    # 8.5), but at 7.5 the spread-gap zone fires and fades the pick -- the
    # flip is the ZONE EDGE, not the distant raw crossing.
    line, held = _flip_line("2026_01_NYJ_TEN", "TEN", "TEN", (), pd.DataFrame(), _params(7.0, 8.5))
    assert (line, held) == (7.5, False)


def test_zone_flipped_pick_reverts_on_a_half_point_move_out_of_the_zone() -> None:
    # CLE @ JAX shape: card at 7.5 (inside the zone), zone member fired, so
    # the played pick is the raw complement (away). One half-point down and
    # the zone stops firing -- the pick reverts to the raw side at 7.0.
    line, held = _flip_line(
        "2026_01_NYJ_TEN",
        "TEN",
        "NYJ",
        ("spread_gap_zone_fade",),
        pd.DataFrame(),
        _params(7.5, 8.5),
    )
    assert (line, held) == (7.0, False)


def test_coach_fade_game_flips_only_where_the_zone_overrides() -> None:
    # Coach-fade shape at card -3.5 (zone edge -7.5 is inside the ±4 span):
    # the member fired against raw NYJ, playing TEN. The raw crossing at -5
    # changes nothing -- the member stops firing and the played side is
    # STILL TEN -- but at -7.5 the raw side is TEN, the zone complements
    # it, and the played pick becomes NYJ.
    line, held = _flip_line(
        "2026_01_NYJ_TEN",
        "TEN",
        "TEN",
        ("coach_fade",),
        pd.DataFrame(),
        _params(-3.5, -5.0),
    )
    assert (line, held) == (-7.5, False)


def test_coach_fade_game_with_no_switch_in_the_span_is_held() -> None:
    # Centre -15: inside the entire explored span the raw side stays the
    # faded side, the member keeps firing, and nothing switches the played
    # pick. The honest answer is "held within the span" -- NEVER a claim
    # about lines beyond it.
    line, held = _flip_line(
        "2026_01_NYJ_TEN",
        "TEN",
        "TEN",
        ("coach_fade",),
        pd.DataFrame(),
        _params(-3.5, -15.0),
    )
    assert (line, held) == (None, True)


def test_unflipped_pick_with_a_distant_crossing_is_held_not_extrapolated() -> None:
    # Raw crossing at 9.0 sits outside the card's ±4 span (card 3.0), and
    # no zone line falls inside it either: the bounded scan reports held
    # rather than quoting a number the on-page explorer cannot even show.
    line, held = _flip_line("2026_01_NYJ_TEN", "TEN", "TEN", (), pd.DataFrame(), _params(3.0, 9.0))
    assert (line, held) == (None, True)


def test_sweep_fallback_reads_the_nearest_crossing_row() -> None:
    sweep = pd.DataFrame(
        {
            "game_id": ["g"] * 5,
            "line_offset": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "alternative_line": [2.0, 2.5, 3.0, 3.5, 4.0],
            "home_cover_probability": [0.53, 0.51, 0.47, 0.44, 0.41],
        }
    )
    line, held = _flip_line("g", "TEN", "NYJ", (), sweep, {})
    assert (line, held) == (2.5, False)


def test_no_source_means_no_flip_line_and_no_held_claim() -> None:
    assert _flip_line("g", "TEN", "NYJ", (), pd.DataFrame(), {}) == (None, False)


def test_flip_line_text_names_the_pick_then_the_switch() -> None:
    base = build_fixture_content().games[0]
    nyj_ten = GameRow(
        game_id="2026_01_NYJ_TEN",
        gameday=base.gameday,
        weekday_name="Sunday",
        home="TEN",
        away="NYJ",
        market_spread=3.0,
        pick_team="NYJ",
        pick_probability=0.505,
        confidence_word="slight",
        is_best=False,
        is_flipped=False,
        flip_line=2.5,
    )
    assert nyj_ten.flip_line_text == "NYJ +2.5 → TEN"
    # A home pick states its own (laying) handicap at the flip line, then
    # the away team it would switch to.
    away_flip = GameRow(
        game_id="2026_01_ARI_LAC",
        gameday=base.gameday,
        weekday_name="Sunday",
        home="LAC",
        away="ARI",
        market_spread=10.5,
        pick_team="LAC",
        pick_probability=0.55,
        confidence_word="lean",
        is_best=False,
        is_flipped=False,
        flip_line=13.0,
    )
    assert away_flip.flip_line_text == "LAC -13 → ARI"
    held = GameRow(
        game_id="2026_01_BAL_IND",
        gameday=base.gameday,
        weekday_name="Sunday",
        home="IND",
        away="BAL",
        market_spread=-3.5,
        pick_team="IND",
        pick_probability=0.516,
        confidence_word="slight",
        is_best=False,
        is_flipped=True,
        flip_line=None,
        flip_held=True,
    )
    assert held.flip_line_text == "IND within ±4"
    assert (
        GameRow(
            game_id="2026_01_GB_MIN",
            gameday=base.gameday,
            weekday_name="Sunday",
            home="MIN",
            away="GB",
            market_spread=1.5,
            pick_team="MIN",
            pick_probability=0.55,
            confidence_word="lean",
            is_best=False,
            is_flipped=False,
            flip_line=None,
        ).flip_line_text
        == ""
    )


def test_board_renders_the_flips_at_column() -> None:
    content = build_fixture_content()
    html = board_terminal.render(content)
    assert "Flips&nbsp;at" in html
    assert 'data-label="Flips at">NYJ +2.5 → TEN</td>' in html
    # The held coach-fade fixture row states the bounded claim, with the
    # explanatory title on that state only -- and never the unbounded one.
    assert "IND within ±4" in html
    assert "at any line" not in html
    assert "changes this pick" in html
    # Six columns: the day-group separator spans all of them.
    assert 'colspan="6"' in html
    assert 'colspan="5"' not in html
