from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.pool import (
    Entry,
    FieldModel,
    PoolFormat,
    build_ats_pool_card,
    build_entry,
    build_straight_up_pool_card,
    deviate,
    head_to_head_win_probability,
    pool_card_markdown,
    simulate_pool_finish,
    straight_up_pool_markdown,
    strategy_comparison,
)


def test_pool_card_forces_and_ranks_every_game() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "gameday": pd.to_datetime(["2022-09-11"] * 3),
            "away_team": ["A", "C", "E"],
            "home_team": ["B", "D", "F"],
            "spread_line": [3.0, -2.5, 1.0],
            "home_cover_probability": [0.60, 0.45, 0.51],
        }
    )
    card = build_ats_pool_card(predictions)
    assert card["pool_pick"].tolist() == ["B", "C", "F"]
    assert card["pick_line"].tolist() == [-3.0, -2.5, -1.0]
    assert card["confidence_rank"].tolist() == [1, 2, 3]
    assert "ATS pool card: 2022 week 1" in pool_card_markdown(card, 2022, 1)


def test_pool_card_requires_contract() -> None:
    with pytest.raises(ValueError, match="missing pool columns"):
        build_ats_pool_card(pd.DataFrame({"game_id": ["g1"]}))


def test_straight_up_card_selects_named_method_and_ranks() -> None:
    rows = []
    for method, probabilities in (
        ("market", [0.55, 0.40]),
        ("market_residual", [0.70, 0.48]),
    ):
        for index, probability in enumerate(probabilities, start=1):
            rows.append(
                {
                    "game_id": f"g{index}",
                    "gameday": "2022-09-11",
                    "away_team": f"A{index}",
                    "home_team": f"H{index}",
                    "method": method,
                    "home_win_probability": probability,
                    "market_spread": 3.0,
                    "fair_spread": 4.0,
                    "predicted_market_residual": 1.0,
                }
            )
    card = build_straight_up_pool_card(pd.DataFrame(rows))
    assert card["pool_pick"].tolist() == ["H1", "A2"]
    assert card["confidence_rank"].tolist() == [1, 2]
    assert "Straight-up pool card: 2022 week 1" in straight_up_pool_markdown(card, 2022, 1)


# ---------------------------------------------------------------------------
# POL-05 contest simulator
#
# Every test below is a case whose answer is known before the simulator runs.
# A Monte Carlo that has never been pinned to a closed form is a random number
# generator with a plot attached.
# ---------------------------------------------------------------------------

SEASON = PoolFormat(weekly_games=(16,) * 17 + (14,), best_pick_bonus=1.0)


def test_format_arithmetic_matches_the_pool() -> None:
    assert SEASON.games == 286
    assert SEASON.weeks == 18
    assert [slot.start for slot in SEASON.week_slices()][:3] == [0, 16, 32]
    assert SEASON.week_slices()[-1] == slice(272, 286)


def test_head_to_head_closed_form_on_hand_computable_cases() -> None:
    # One disagreement: we win it exactly as often as we are right.
    assert head_to_head_win_probability(1, 0.6) == pytest.approx(0.6)
    # Two disagreements at a coin flip: we only lead by sweeping both.
    assert head_to_head_win_probability(2, 0.5) == pytest.approx(0.25)
    # Three at a coin flip: no ties are possible, so it is symmetric.
    assert head_to_head_win_probability(3, 0.5) == pytest.approx(0.5)
    # Agreeing on everything can never produce a lead.
    assert head_to_head_win_probability(0, 0.9) == 0.0


def test_a_symmetric_pool_is_a_lottery() -> None:
    """No edge, no correlation, no bonus: everyone's chance is 1/(N+1)."""

    fmt = PoolFormat(weekly_games=(16,) * 4, best_pick_bonus=0.0)
    entry = build_entry(fmt, cover_probability=0.5, public_agreement=0.5, seed=11)
    result = simulate_pool_finish(
        entry, FieldModel(entrants=9, public_lean=0.5), fmt, samples=40_000, seed=7
    )
    assert result["probability_first"] == pytest.approx(0.1, abs=0.01)


def test_a_field_that_copies_our_card_can_only_tie() -> None:
    """public_lean 1.0 with our side public everywhere: identical entries."""

    fmt = PoolFormat(weekly_games=(16,) * 4, best_pick_bonus=0.0)
    entry = build_entry(fmt, cover_probability=0.53, public_agreement=np.ones(64), seed=3)
    result = simulate_pool_finish(
        entry, FieldModel(entrants=4, public_lean=1.0), fmt, samples=4_000, seed=5
    )
    assert result["probability_outright"] == 0.0
    assert result["probability_tied_first"] == 1.0
    assert result["probability_first"] == pytest.approx(0.2)


def test_a_perfect_card_always_finishes_first() -> None:
    fmt = PoolFormat(weekly_games=(16,) * 4, best_pick_bonus=0.0)
    entry = build_entry(fmt, cover_probability=1.0, public_agreement=0.5, seed=3)
    result = simulate_pool_finish(
        entry, FieldModel(entrants=50, public_lean=0.6), fmt, samples=2_000, seed=5
    )
    assert result["probability_outright"] == pytest.approx(1.0)


def test_simulator_reproduces_the_head_to_head_closed_form() -> None:
    """One deterministic opponent: the margin is decided only by disagreements."""

    fmt = PoolFormat(weekly_games=(10,) * 3, best_pick_bonus=0.0)
    public = np.array([True] * 21 + [False] * 9)  # we differ on nine games
    entry = build_entry(fmt, cover_probability=0.55, public_agreement=public)
    result = simulate_pool_finish(
        entry, FieldModel(entrants=1, public_lean=1.0), fmt, samples=60_000, seed=17
    )
    expected = head_to_head_win_probability(9, 0.55)
    assert result["probability_outright"] == pytest.approx(expected, abs=0.01)


def test_expected_score_matches_the_stated_probabilities() -> None:
    fmt = PoolFormat(weekly_games=(16,) * 4, best_pick_bonus=2.0)
    entry = build_entry(fmt, cover_probability=0.525, public_agreement=0.5, seed=3)
    result = simulate_pool_finish(entry, FieldModel(entrants=5), fmt, samples=20_000, seed=9)
    # 64 forced picks at 0.525 plus a 2-point bonus on four nominated games.
    assert result["expected_score"] == pytest.approx(64 * 0.525 + 4 * 2.0 * 0.525, abs=0.15)


def test_more_entrants_can_only_make_first_place_rarer() -> None:
    fmt = PoolFormat(weekly_games=(16,) * 4, best_pick_bonus=0.0)
    entry = build_entry(fmt, cover_probability=0.55, public_agreement=0.5, seed=3)
    chances = [
        simulate_pool_finish(entry, FieldModel(entrants=size), fmt, samples=6_000, seed=13)[
            "probability_first"
        ]
        for size in (5, 25, 100)
    ]
    assert chances[0] > chances[1] > chances[2]


def test_deviating_flips_both_the_probability_and_the_public_flag() -> None:
    fmt = PoolFormat(weekly_games=(4,), best_pick_bonus=0.0)
    entry = build_entry(fmt, cover_probability=0.56, public_agreement=np.ones(4))
    flipped = deviate(entry, np.array([0, 2]))
    assert flipped.cover_probability.tolist() == pytest.approx([0.44, 0.56, 0.44, 0.56])
    assert flipped.on_public_side.tolist() == [False, True, False, True]
    assert entry.cover_probability.tolist() == [0.56] * 4  # original untouched


def test_strategy_comparison_ranks_by_probability_first() -> None:
    fmt = PoolFormat(weekly_games=(16,) * 4, best_pick_bonus=0.0)
    weak = build_entry(fmt, cover_probability=0.50, public_agreement=0.5, seed=3)
    strong = build_entry(fmt, cover_probability=0.58, public_agreement=0.5, seed=3)
    frame = strategy_comparison(
        {"weak": weak, "strong": strong}, FieldModel(entrants=20), fmt, samples=4_000
    )
    assert frame["strategy"].tolist() == ["strong", "weak"]
    assert frame["probability_first"].is_monotonic_decreasing


def test_simulator_rejects_a_card_that_does_not_fit_the_format() -> None:
    fmt = PoolFormat(weekly_games=(16,) * 4)
    entry = build_entry(
        PoolFormat(weekly_games=(16,) * 3), cover_probability=0.5, public_agreement=0.5
    )
    with pytest.raises(ValueError, match="does not match"):
        simulate_pool_finish(entry, FieldModel(entrants=3), fmt, samples=200)


def test_input_contracts() -> None:
    with pytest.raises(ValueError, match="weekly_games"):
        PoolFormat(weekly_games=())
    with pytest.raises(ValueError, match="entrants"):
        FieldModel(entrants=0)
    with pytest.raises(ValueError, match="public_lean"):
        FieldModel(entrants=5, public_lean=1.5)
    with pytest.raises(ValueError, match="cover_probability"):
        Entry(np.array([1.5]), np.array([True]), np.array([0]))
