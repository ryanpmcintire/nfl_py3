"""The board's "Flips at" column (owner request, 2026-09-01): the first
half-point line at which the displayed pick would switch to the other team.

Source-order contract under test (see ``board_content._flip_line``):

1. a policy-flipped game has NO flip line -- the overlay pins the side by
   member rule, so no spread move changes it;
2. the guarded Gaussian read (the spread adjuster's own math) is preferred,
   so the column can never disagree with the slider on the same page;
3. real ``line_sweep`` rows are the fallback when the adjuster is degraded.

The rendered-cell tests use the shared hand-built fixture, whose
``2026_01_NYJ_TEN`` row mirrors the real Week 1 card's example: pick NYJ at
TEN -3, flips to TEN at -2.5.
"""

from __future__ import annotations

import math

import pandas as pd
from _board_content_fixtures import build_fixture_content

from nfl_ats import board_terminal
from nfl_ats.board_content import GameRow, _flip_line
from nfl_ats.spread_explorer import SpreadExplorerGameParams


def _params(card_line: float, center: float) -> SpreadExplorerGameParams:
    return SpreadExplorerGameParams(
        game_id="2026_01_NYJ_TEN",
        home_team="TEN",
        away_team="NYJ",
        center=center,
        residual_mean=0.0,
        residual_std=6.0,
        card_line=card_line,
        card_home_cover_probability=1.0
        - 0.5 * (1.0 + math.erf((card_line - center) / (6.0 * math.sqrt(2.0)))),
    )


def test_widget_path_finds_the_owner_example_crossing() -> None:
    # Centre 2.7 with the card at 3.0: home cover probability is below 0.5
    # at the quoted line (pick NYJ), and crosses above it at 2.5 -- the real
    # Week 1 NYJ @ TEN shape the owner read off the adjuster.
    line = _flip_line(
        "2026_01_NYJ_TEN",
        "TEN",
        "NYJ",
        False,
        pd.DataFrame(),
        {"2026_01_NYJ_TEN": _params(card_line=3.0, center=2.7)},
    )
    assert line == 2.5


def test_widget_path_flips_a_home_pick_upward() -> None:
    # Centre 5.2 with the card at 3.0: the home side covers at the quoted
    # line, and keeps covering until the line passes the centre -- the first
    # flipped grid line is 5.5.
    line = _flip_line(
        "2026_01_NYJ_TEN",
        "TEN",
        "TEN",
        False,
        pd.DataFrame(),
        {"2026_01_NYJ_TEN": _params(card_line=3.0, center=5.2)},
    )
    assert line == 5.5


def test_policy_flipped_game_has_no_flip_line_even_with_params() -> None:
    line = _flip_line(
        "2026_01_NYJ_TEN",
        "TEN",
        "TEN",
        True,
        pd.DataFrame(),
        {"2026_01_NYJ_TEN": _params(card_line=3.0, center=2.7)},
    )
    assert line is None


def test_sweep_fallback_reads_the_nearest_crossing_row() -> None:
    sweep = pd.DataFrame(
        {
            "game_id": ["g"] * 5,
            "line_offset": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "alternative_line": [2.0, 2.5, 3.0, 3.5, 4.0],
            "home_cover_probability": [0.53, 0.51, 0.47, 0.44, 0.41],
        }
    )
    line = _flip_line("g", "TEN", "NYJ", False, sweep, {})
    assert line == 2.5


def test_no_source_means_no_flip_line() -> None:
    assert _flip_line("g", "TEN", "NYJ", False, pd.DataFrame(), {}) is None


def test_flip_line_text_names_the_flipped_to_team_with_its_own_handicap() -> None:
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
    assert nyj_ten.flip_line_text == "TEN -2.5"
    # An away-team flip states the away side's own (positive) handicap.
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
    assert away_flip.flip_line_text == "ARI +13"
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
    assert 'data-label="Flips at">TEN -2.5</td>' in html
    # The policy-flipped fixture row renders the pinned em-dash, with the
    # explanatory title only on that pinned state.
    assert "title='Policy overlay pins this pick'" in html
    # Six columns now: the day-group separator spans all of them.
    assert 'colspan="6"' in html
    assert 'colspan="5"' not in html
