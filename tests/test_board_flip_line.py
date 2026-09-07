"""The board's "Flips at" column (owner request, 2026-09-01): the first
half-point line at which the PLAYED pick would switch to the other team.

Semantics under test: the three-member played policy uses the model crossing
and fired pick-conditioned rules. The retired spread-gap thresholds never
change the played side. The scan stays within the displayed four-point span.

Cell format (owner feedback, same day, replacing a first draft that printed
the flipped-to team's handicap in the opposite orientation from the Pick
column): the CURRENT pick's own handicap at the flip line, then the team it
switches to -- ``NYJ +2.5 → TEN``; the held state reads ``IND holds from +7.5 to -0.5``.
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
    line, held, reason = _flip_line(
        "2026_01_NYJ_TEN", "TEN", "NYJ", 3.0, (), pd.DataFrame(), _params(3.0, 2.7)
    )
    assert (line, held, reason) == (2.5, False, "model")


def test_widget_path_flips_a_home_pick_upward() -> None:
    line, held, reason = _flip_line(
        "2026_01_NYJ_TEN", "TEN", "TEN", 3.0, (), pd.DataFrame(), _params(3.0, 5.2)
    )
    assert (line, held, reason) == (5.5, False, "model")


def test_pick_near_retired_zone_waits_for_model_crossing() -> None:
    # Passing the retired boundary leaves the model side intact.
    line, held, reason = _flip_line(
        "2026_01_NYJ_TEN", "TEN", "TEN", 7.0, (), pd.DataFrame(), _params(7.0, 8.5)
    )
    assert (line, held, reason) == (8.5, False, "model")


def test_retired_zone_member_cannot_toggle_the_pick() -> None:
    args = ("2026_01_NYJ_TEN", "TEN", "TEN", 7.5)
    expected = _flip_line(*args, (), pd.DataFrame(), _params(7.5, 8.5))
    actual = _flip_line(*args, ("spread_gap_zone_fade",), pd.DataFrame(), _params(7.5, 8.5))
    assert actual == expected == (8.5, False, "model")


def test_coach_fade_game_never_re_fires_the_zone_four_points_away() -> None:
    # The owner-catch-#3 case, exactly: a 3.5-point game whose raw crossing
    # sits inside the span. The coach fade keeps the pick through the raw
    # crossing (the member just stops firing, same side), and the zone edge
    # at -7.5 is FOUR points from the real line -- out of the rule's
    # evidence, frozen off -- so no "give it more points and lose it" cell.
    line, held, reason = _flip_line(
        "2026_01_NYJ_TEN",
        "TEN",
        "TEN",
        -3.5,
        ("coach_fade",),
        pd.DataFrame(),
        _params(-3.5, -5.0),
    )
    assert (line, held, reason) == (None, True, None)


def test_coach_fade_game_with_no_crossing_in_the_span_is_held() -> None:
    line, held, reason = _flip_line(
        "2026_01_NYJ_TEN",
        "TEN",
        "TEN",
        -3.5,
        ("coach_fade",),
        pd.DataFrame(),
        _params(-3.5, -15.0),
    )
    assert (line, held, reason) == (None, True, None)


def test_unflipped_pick_with_a_distant_crossing_is_held_not_extrapolated() -> None:
    # Raw crossing at 9.0 sits outside the card's ±4 span (card 3.0), and
    # the zone edges are 4+ points away (frozen off): the bounded scan
    # reports held rather than quoting a number the on-page explorer cannot
    # even show.
    line, held, reason = _flip_line(
        "2026_01_NYJ_TEN", "TEN", "TEN", 3.0, (), pd.DataFrame(), _params(3.0, 9.0)
    )
    assert (line, held, reason) == (None, True, None)


def test_sweep_fallback_reads_the_nearest_crossing_row() -> None:
    sweep = pd.DataFrame(
        {
            "game_id": ["g"] * 5,
            "line_offset": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "alternative_line": [2.0, 2.5, 3.0, 3.5, 4.0],
            "home_cover_probability": [0.53, 0.51, 0.47, 0.44, 0.41],
        }
    )
    line, held, reason = _flip_line("g", "TEN", "NYJ", 3.0, (), sweep, {})
    assert (line, held, reason) == (2.5, False, "model")


def test_no_source_means_no_flip_line_and_no_held_claim() -> None:
    assert _flip_line("g", "TEN", "NYJ", 3.0, (), pd.DataFrame(), {}) == (None, False, None)


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
    # BAL -3.5 at IND, pick IND +3.5, scanned four points either way: the two
    # ends of that span in IND's orientation, most points first.
    assert held.flip_line_text == "IND holds from +7.5 to -0.5"
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
    assert "IND holds from +7.5 to -0.5" in html
    # A rule-driven switch names the rule in the cell (owner question, 2026-09-07).
    assert "ARI +10 → LAC (spread-gap rule)" in html
    assert "not the model changing its mind" in html
    assert "coach fade rule backs IND whichever side the model leans" in html
    assert "at any line" not in html
    assert "changes this pick" in html
    # Six columns: the day-group separator spans all of them.
    assert 'colspan="6"' in html
    assert 'colspan="5"' not in html
