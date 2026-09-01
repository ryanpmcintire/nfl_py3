"""Tests for the owner-approved improvement batch's new pure functions in
:mod:`nfl_ats.board_content`:

* item 1 -- flip member labels and the raw-vs-played flip note
  (:func:`_flip_member_labels`, :func:`_flip_note`);
* item 3 -- the paired prospective scoreboard
  (:func:`_build_prospective_scoreboard`, :func:`_grade_decisions`);
* item 4 -- in-season finals and the running record strip
  (:func:`_game_final_state`, :func:`_build_season_record`).

These are pure functions over hand-built ``pandas.DataFrame``s (the same
schemas ``nfl_ats.clv.load_paper_decisions`` /
``nfl_ats.prospective_scoring.load_challenger_decisions`` /
``data/processed/game_features.parquet`` carry), so no real artifact tree or
parquet fixture is needed -- exactly the discipline
``tests/test_board_content.py`` already uses for the cover-curve fallback.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from nfl_ats import board_content
from nfl_ats.board_content import GameRow
from nfl_ats.four_overlay_composition import (
    INCUMBENT_CHALLENGER_ID,
    POLICY_ID,
)

# ---------------------------------------------------------------------------
# item 1 -- flip member labels and the raw-vs-played flip note
# ---------------------------------------------------------------------------


def test_flip_member_labels_empty_when_view_is_none() -> None:
    assert board_content._flip_member_labels(None, "2026_01_BAL_IND") == ()


def test_flip_member_labels_from_production_overlay_provenance() -> None:
    game_provenance = SimpleNamespace(game_id="2026_01_BAL_IND", member_ids=("coach_fade",))
    other_game = SimpleNamespace(game_id="2026_01_CLE_JAX", member_ids=("spread_gap_zone_fade",))
    view = SimpleNamespace(production_overlay=SimpleNamespace(games=(game_provenance, other_game)))
    assert board_content._flip_member_labels(view, "2026_01_BAL_IND") == ("coach fade",)
    assert board_content._flip_member_labels(view, "2026_01_CLE_JAX") == ("spread-gap zone",)
    assert board_content._flip_member_labels(view, "2026_01_UNFLIPPED") == ()


def test_flip_member_labels_reports_every_member_on_an_overlap() -> None:
    """The joint-OR policy can flip one game via more than one member -- the
    label tuple must name every member that fired, not just one."""

    game_provenance = SimpleNamespace(
        game_id="2026_01_BAL_IND", member_ids=("coach_fade", "spread_gap_zone_fade")
    )
    view = SimpleNamespace(production_overlay=SimpleNamespace(games=(game_provenance,)))
    labels = board_content._flip_member_labels(view, "2026_01_BAL_IND")
    assert labels == ("coach fade", "spread-gap zone")


def test_flip_member_labels_legacy_fallback_with_no_production_overlay() -> None:
    coach_flip = SimpleNamespace(game_id="2026_01_BAL_IND")
    arrest_flip = SimpleNamespace(game_id="2026_01_CLE_JAX")
    view = SimpleNamespace(
        production_overlay=None,
        overlay=SimpleNamespace(flips=(coach_flip,)),
        arrest_overlay=SimpleNamespace(enabled=True, flips=(arrest_flip,)),
    )
    assert board_content._flip_member_labels(view, "2026_01_BAL_IND") == ("coach fade",)
    assert board_content._flip_member_labels(view, "2026_01_CLE_JAX") == ("player arrests",)
    assert board_content._flip_member_labels(view, "2026_01_UNFLIPPED") == ()


def test_flip_member_labels_legacy_fallback_ignores_disabled_arrest_overlay() -> None:
    arrest_flip = SimpleNamespace(game_id="2026_01_CLE_JAX")
    view = SimpleNamespace(
        production_overlay=None,
        overlay=SimpleNamespace(flips=()),
        arrest_overlay=SimpleNamespace(enabled=False, flips=(arrest_flip,)),
    )
    assert board_content._flip_member_labels(view, "2026_01_CLE_JAX") == ()


def _dive_game(**overrides: object) -> GameRow:
    return _game_row(**overrides)


def test_flip_note_is_none_when_not_flipped() -> None:
    game = _dive_game(flip_member_labels=())
    assert board_content._flip_note(game, raw_home_cover_probability=0.4) is None


def test_flip_note_is_none_when_raw_probability_unavailable() -> None:
    game = _dive_game(pick_team="IND", home="IND", away="BAL", flip_member_labels=("coach fade",))
    assert board_content._flip_note(game, raw_home_cover_probability=None) is None


def test_flip_note_names_raw_side_vs_played_side() -> None:
    # Raw model favored the home team (BAL), the played card flips to the
    # away team (IND) via coach fade.
    game = _dive_game(home="BAL", away="IND", pick_team="IND", flip_member_labels=("coach fade",))
    note = board_content._flip_note(game, raw_home_cover_probability=0.6)
    assert note is not None
    assert "BAL" in note
    assert "IND" in note
    assert "coach fade" in note


def test_flip_note_is_none_when_raw_side_equals_played_side() -> None:
    """A flip can toggle the probability without toggling the final team
    (e.g. a near-50% game) -- nothing to say in that case."""

    game = _dive_game(home="BAL", away="IND", pick_team="BAL", flip_member_labels=("coach fade",))
    assert board_content._flip_note(game, raw_home_cover_probability=0.6) is None


# ---------------------------------------------------------------------------
# item 4 -- in-season finals
# ---------------------------------------------------------------------------


def test_game_final_state_is_upcoming_when_result_is_missing() -> None:
    final, cover, score_text = board_content._game_final_state(
        home="SEA",
        away="NE",
        pick_team="SEA",
        market_spread=-3.0,
        result=None,
        home_score=None,
        away_score=None,
    )
    assert final is False
    assert cover is None
    assert score_text is None


def test_game_final_state_win_when_pick_covers() -> None:
    # market_spread is home-oriented (nflverse convention, ``public_board
    # .spread_words``): +3.0 means SEA (home) is favored by 3 ("SEA -3.0").
    # SEA wins 24-13 (home margin +11), well past the +3.0 line -> covers.
    final, cover, score_text = board_content._game_final_state(
        home="SEA",
        away="NE",
        pick_team="SEA",
        market_spread=3.0,
        result=11.0,
        home_score=24.0,
        away_score=13.0,
    )
    assert final is True
    assert cover == "win"
    assert score_text == "NE 13 at SEA 24"


def test_game_final_state_loss_when_pick_does_not_cover() -> None:
    # SEA favored by 3 (market_spread=+3.0); SEA wins by only 1 (home margin
    # +1), short of the +3.0 line, so the SEA pick does NOT cover.
    final, cover, _ = board_content._game_final_state(
        home="SEA",
        away="NE",
        pick_team="SEA",
        market_spread=3.0,
        result=1.0,
        home_score=21.0,
        away_score=20.0,
    )
    assert final is True
    assert cover == "loss"


def test_game_final_state_loss_for_away_pick_when_home_covers() -> None:
    # Pick is the AWAY team; home comfortably covers its own +3.0 favorite
    # line -> the away pick loses.
    final, cover, _ = board_content._game_final_state(
        home="SEA",
        away="NE",
        pick_team="NE",
        market_spread=3.0,
        result=11.0,
        home_score=24.0,
        away_score=13.0,
    )
    assert final is True
    assert cover == "loss"


def test_game_final_state_push_on_exact_margin() -> None:
    # Home wins by exactly the home-favorite line (+3.0) -> a push.
    final, cover, _ = board_content._game_final_state(
        home="SEA",
        away="NE",
        pick_team="SEA",
        market_spread=3.0,
        result=3.0,
        home_score=23.0,
        away_score=20.0,
    )
    assert final is True
    assert cover == "push"


def test_game_row_cover_result_label_and_flip_pill_text() -> None:
    win_game = _game_row(cover_result="win")
    loss_game = _game_row(cover_result="loss")
    push_game = _game_row(cover_result="push")
    assert win_game.cover_result_label == "Covered"
    assert loss_game.cover_result_label == "No cover"
    assert push_game.cover_result_label == "Push"

    flipped = _game_row(flip_member_labels=("coach fade",))
    assert flipped.flip_pill_text == "⇄ coach fade"
    overlap = _game_row(flip_member_labels=("coach fade", "spread-gap zone"))
    assert overlap.flip_pill_text == "⇄ coach fade + spread-gap zone"


def _game_row(**overrides: object) -> GameRow:
    defaults: dict[str, object] = {
        "game_id": "2026_01_TEST",
        "gameday": date(2026, 9, 10),
        "weekday_name": "Thursday",
        "home": "SEA",
        "away": "NE",
        "market_spread": -3.0,
        "pick_team": "SEA",
        "pick_probability": 0.55,
        "confidence_word": "lean",
        "is_best": False,
        "is_flipped": False,
    }
    defaults.update(overrides)
    return GameRow(**defaults)  # type: ignore[arg-type]


def _outcomes(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["game_id", "result", "home_score", "away_score"])


def _decisions(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_grade_decisions_counts_win_loss_push_and_pending() -> None:
    decisions = _decisions(
        [
            {"game_id": "G1", "pick_side": "HOME", "decision_home_spread": 3.0},
            {"game_id": "G2", "pick_side": "AWAY", "decision_home_spread": 3.0},
            {"game_id": "G3", "pick_side": "HOME", "decision_home_spread": 3.0},
            {"game_id": "G4", "pick_side": "HOME", "decision_home_spread": 3.0},
        ]
    )
    outcomes = _outcomes(
        [
            ("G1", 11.0, 24.0, 13.0),  # HOME pick, margin +8 -> win
            ("G2", 11.0, 24.0, 13.0),  # AWAY pick, home covers -> loss
            ("G3", 3.0, 23.0, 20.0),  # push
            # G4 has no outcome row -> pending
        ]
    )
    wins, losses, pushes, pending = board_content._grade_decisions(decisions, outcomes)
    assert (wins, losses, pushes, pending) == (1, 1, 1, 1)


def test_grade_decisions_empty_inputs_are_zero() -> None:
    empty = pd.DataFrame(columns=["game_id", "pick_side", "decision_home_spread"])
    outcomes = _outcomes([])
    assert board_content._grade_decisions(empty, outcomes) == (0, 0, 0, 0)


def test_record_text_omits_pushes_when_zero() -> None:
    assert board_content._record_text(3, 1, 0) == "3-1"
    assert board_content._record_text(3, 1, 2) == "3-1-2"


# ---------------------------------------------------------------------------
# item 3 -- prospective scoreboard
# ---------------------------------------------------------------------------


def test_prospective_scoreboard_dormant_when_both_ledgers_empty() -> None:
    empty = pd.DataFrame()
    scoreboard = board_content._build_prospective_scoreboard(empty, empty, empty)
    assert scoreboard.dormant is True
    assert scoreboard.detail_text is None
    assert "Week 1" in scoreboard.headline_text


def test_prospective_scoreboard_reports_paired_record_once_settled() -> None:
    paper = _decisions(
        [
            {
                "game_id": "2026_01_A",
                "decision_policy_id": POLICY_ID,
                "pick_side": "HOME",
                "decision_home_spread": 3.0,
            },
            {
                "game_id": "2026_01_B",
                "decision_policy_id": POLICY_ID,
                "pick_side": "AWAY",
                "decision_home_spread": 3.0,
            },
            {
                # a game recorded but not yet kicked off
                "game_id": "2026_01_C",
                "decision_policy_id": POLICY_ID,
                "pick_side": "HOME",
                "decision_home_spread": 3.0,
            },
        ]
    )
    challenger = _decisions(
        [
            {
                "game_id": "2026_01_A",
                "challenger_id": INCUMBENT_CHALLENGER_ID,
                "pick_side": "AWAY",
                "decision_home_spread": 3.0,
            },
            {
                "game_id": "2026_01_B",
                "challenger_id": INCUMBENT_CHALLENGER_ID,
                "pick_side": "AWAY",
                "decision_home_spread": 3.0,
            },
        ]
    )
    outcomes = _outcomes(
        [
            ("2026_01_A", 11.0, 24.0, 13.0),  # played HOME wins; incumbent AWAY loses
            ("2026_01_B", 11.0, 24.0, 13.0),  # played AWAY loses; incumbent AWAY loses too
        ]
    )
    scoreboard = board_content._build_prospective_scoreboard(paper, challenger, outcomes)
    assert scoreboard.dormant is False
    assert "played policy 1-1" in scoreboard.headline_text
    assert "prior chain 0-2" in scoreboard.headline_text
    assert scoreboard.detail_text is not None
    assert "1 recorded game" in scoreboard.detail_text


def test_prospective_scoreboard_ignores_rows_from_a_different_policy_or_challenger() -> None:
    """Only ``POLICY_ID``/``INCUMBENT_CHALLENGER_ID`` rows count -- a stray
    row from a different decision policy or challenger must never leak into
    the paired record."""

    paper = _decisions(
        [
            {
                "game_id": "2026_01_A",
                "decision_policy_id": "some_other_policy",
                "pick_side": "HOME",
                "decision_home_spread": 3.0,
            }
        ]
    )
    challenger = _decisions(
        [
            {
                "game_id": "2026_01_A",
                "challenger_id": "some_other_challenger",
                "pick_side": "AWAY",
                "decision_home_spread": 3.0,
            }
        ]
    )
    scoreboard = board_content._build_prospective_scoreboard(paper, challenger, _outcomes([]))
    assert scoreboard.dormant is True


# ---------------------------------------------------------------------------
# item 4 -- the hero's running record strip
# ---------------------------------------------------------------------------


def test_season_record_is_none_when_ledger_is_empty() -> None:
    empty = pd.DataFrame(columns=["season", "week", "pick_side", "decision_home_spread"])
    record = board_content._build_season_record(empty, _outcomes([]), season=2026, week=1)
    assert record is None


def test_season_record_is_none_when_nothing_this_season_is_graded() -> None:
    decisions = _decisions(
        [
            {
                "game_id": "2026_01_A",
                "season": 2026,
                "week": 1,
                "pick_side": "HOME",
                "decision_home_spread": 3.0,
            }
        ]
    )
    record = board_content._build_season_record(decisions, _outcomes([]), season=2026, week=1)
    assert record is None  # nothing settled yet -- stay dormant, not "0-0"


def test_season_record_reports_week_season_and_best_pick_splits() -> None:
    decisions = _decisions(
        [
            {
                "game_id": "2026_01_A",
                "season": 2026,
                "week": 1,
                "pick_side": "HOME",
                "decision_home_spread": 3.0,
                "is_best_pick": True,
            },
            {
                "game_id": "2026_01_B",
                "season": 2026,
                "week": 1,
                "pick_side": "AWAY",
                "decision_home_spread": 3.0,
                "is_best_pick": False,
            },
            {
                # a prior week this season, already graded
                "game_id": "2025_18_C",
                "season": 2026,
                "week": 18,
                "pick_side": "HOME",
                "decision_home_spread": 3.0,
                "is_best_pick": False,
            },
        ]
    )
    outcomes = _outcomes(
        [
            ("2026_01_A", 11.0, 24.0, 13.0),  # HOME pick wins -> Best Pick win
            ("2026_01_B", 11.0, 24.0, 13.0),  # AWAY pick loses
            ("2025_18_C", 11.0, 24.0, 13.0),  # HOME pick wins
        ]
    )
    record = board_content._build_season_record(decisions, outcomes, season=2026, week=1)
    assert record is not None
    assert "1-1" in record.week_record_text  # this week: A win, B loss
    assert "2-1" in record.season_record_text  # season to date: A, C win, B loss
    assert record.best_pick_record_text == "Best Pick: 1-0"


def test_season_record_best_pick_is_none_when_no_best_pick_settled() -> None:
    decisions = _decisions(
        [
            {
                "game_id": "2026_01_A",
                "season": 2026,
                "week": 1,
                "pick_side": "HOME",
                "decision_home_spread": 3.0,
                "is_best_pick": False,
            }
        ]
    )
    outcomes = _outcomes([("2026_01_A", 11.0, 24.0, 13.0)])
    record = board_content._build_season_record(decisions, outcomes, season=2026, week=1)
    assert record is not None
    assert record.best_pick_record_text is None
