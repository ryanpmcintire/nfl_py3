"""Tests for :mod:`nfl_ats.board_content`'s cover-curve / spread-explorer
fallback (2026-08-31 full-site conversion, item 4: "keep the spread
explorer's exact published-card math and guard").

``_build_cover_curve`` prefers REAL swept ``line_sweep`` rows wherever they
exist; these tests exercise the Gaussian closed-form fallback path used when
they do not, and the build-time guard that must fire before that fallback is
ever trusted.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nfl_ats import board_content
from nfl_ats.board_content import GameRow
from nfl_ats.data import DataContractError
from nfl_ats.public_board import assert_spread_explorer_matches_card
from nfl_ats.spread_explorer import (
    SpreadExplorerGameParams,
    widget_home_cover_probability,
)


def _game(pick_team: str, home: str, away: str) -> GameRow:
    return GameRow(
        game_id="2026_01_TEST",
        gameday=__import__("datetime").date(2026, 9, 10),
        weekday_name="Thursday",
        home=home,
        away=away,
        market_spread=3.0,
        pick_team=pick_team,
        pick_probability=0.55,
        confidence_word="lean",
        is_best=True,
        is_flipped=False,
    )


def _params(game_id: str = "2026_01_TEST") -> SpreadExplorerGameParams:
    return SpreadExplorerGameParams(
        game_id=game_id,
        home_team="SEA",
        away_team="NE",
        center=1.2,
        residual_mean=0.4,
        residual_std=6.5,
        card_line=3.0,
        card_home_cover_probability=widget_home_cover_probability(3.0, 1.2, 0.4, 6.5),
    )


def test_cover_curve_empty_when_no_game() -> None:
    assert board_content._build_cover_curve(pd.DataFrame(), None) == ()


def test_cover_curve_prefers_real_sweep_over_gaussian_fallback() -> None:
    game = _game("SEA", home="SEA", away="NE")
    sweep = pd.DataFrame(
        {
            "game_id": ["2026_01_TEST", "2026_01_TEST"],
            "line_offset": [-1.0, 0.0],
            "home_cover_probability": [0.4, 0.5],
        }
    )
    params = {game.game_id: _params()}
    curve = board_content._build_cover_curve(sweep, game, params)
    # Real sweep has only 2 rows -- the fallback grid has many more -- so a
    # curve this short proves the real rows won, not the Gaussian fallback.
    assert len(curve) == 2
    assert curve[-1].offset == 0.0
    assert curve[-1].probability == pytest.approx(0.5)


def test_cover_curve_falls_back_to_gaussian_when_sweep_is_empty() -> None:
    game = _game("SEA", home="SEA", away="NE")
    params = {game.game_id: _params()}
    curve = board_content._build_cover_curve(pd.DataFrame(), game, params)
    assert len(curve) > 2
    zero_point = next(point for point in curve if point.offset == 0.0)
    # Pick is the home team, so the curve's own home-oriented published
    # probability should reproduce card_home_cover_probability at offset 0.
    assert zero_point.probability == pytest.approx(params[game.game_id].card_home_cover_probability)


def test_cover_curve_gaussian_fallback_orients_to_away_pick() -> None:
    game = _game("NE", home="SEA", away="NE")  # pick is the AWAY team
    params = {game.game_id: _params()}
    curve = board_content._build_cover_curve(pd.DataFrame(), game, params)
    zero_point = next(point for point in curve if point.offset == 0.0)
    home_probability = params[game.game_id].card_home_cover_probability
    assert zero_point.probability == pytest.approx(1.0 - home_probability)


def test_cover_curve_empty_when_no_sweep_and_no_gaussian_params() -> None:
    game = _game("SEA", home="SEA", away="NE")
    assert board_content._build_cover_curve(pd.DataFrame(), game, None) == ()
    assert board_content._build_cover_curve(pd.DataFrame(), game, {}) == ()


def test_load_spread_explorer_params_skips_non_gaussian_methods() -> None:
    metadata = {"probability_method": "ecdf"}
    result = board_content._load_spread_explorer_params(
        metadata, pd.DataFrame({"game_id": ["x"]}), board_content._default_data_root()
    )
    assert result == {}


def test_load_spread_explorer_params_skips_empty_predictions() -> None:
    metadata = {"probability_method": "gaussian"}
    result = board_content._load_spread_explorer_params(
        metadata, pd.DataFrame(), board_content._default_data_root()
    )
    assert result == {}


def test_assert_spread_explorer_matches_card_guard_fires_on_mismatch() -> None:
    """The REQUIRED build-time guard (preserved verbatim from
    ``public_board._assert_spread_explorer_matches_card`` via the public
    wrapper) must raise when the widget formula disagrees with the
    published card -- proving board_content.py cannot silently trust a
    Gaussian read that would show a different number than the one already
    on the page."""

    params = {"2026_01_TEST": _params()}
    predictions = pd.DataFrame(
        {"game_id": ["2026_01_TEST"], "home_cover_probability": [0.999]}
    )  # deliberately wrong vs. the widget's own computed probability
    with pytest.raises(DataContractError):
        assert_spread_explorer_matches_card(params, predictions)


def test_assert_spread_explorer_matches_card_guard_passes_on_match() -> None:
    game_params = _params()
    params = {game_params.game_id: game_params}
    predictions = pd.DataFrame(
        {
            "game_id": [game_params.game_id],
            "home_cover_probability": [game_params.card_home_cover_probability],
        }
    )
    assert_spread_explorer_matches_card(params, predictions)  # must not raise


def test_guard_fires_for_any_game_in_a_multi_game_week() -> None:
    """2026-08-31 owner redirect: the line-offset adjuster now covers EVERY
    game's deep dive, not only the Best Pick's -- so the guard must catch a
    mismatch on ANY game in the week, not just the first/only one a
    single-game test would exercise."""

    good_game_params = _params("2026_01_GOOD")
    bad_game_params = SpreadExplorerGameParams(
        game_id="2026_01_BAD",
        home_team="KC",
        away_team="DEN",
        center=0.5,
        residual_mean=0.1,
        residual_std=7.0,
        card_line=-3.0,
        card_home_cover_probability=widget_home_cover_probability(-3.0, 0.5, 0.1, 7.0),
    )
    params = {good_game_params.game_id: good_game_params, bad_game_params.game_id: bad_game_params}
    predictions = pd.DataFrame(
        {
            "game_id": [good_game_params.game_id, bad_game_params.game_id],
            "home_cover_probability": [
                good_game_params.card_home_cover_probability,
                0.999,  # deliberately wrong for the SECOND game only
            ],
        }
    )
    with pytest.raises(DataContractError):
        assert_spread_explorer_matches_card(params, predictions)


def test_cover_curve_fallback_offsets_match_sweep_half_width_and_step() -> None:
    """Regression guard: the fallback grid must span the SAME domain as a
    real sweep (``SWEEP_HALF_WIDTH``), never wider or coarser -- otherwise a
    chart built from the fallback would look different from one built from
    real rows for no real reason."""

    offsets = board_content._COVER_CURVE_FALLBACK_OFFSETS
    assert math.isclose(min(offsets), -board_content.SWEEP_HALF_WIDTH)
    assert math.isclose(max(offsets), board_content.SWEEP_HALF_WIDTH)
    assert offsets == tuple(sorted(offsets))
