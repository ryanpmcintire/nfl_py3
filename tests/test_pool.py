from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.pool import (
    build_ats_pool_card,
    build_straight_up_pool_card,
    pool_card_markdown,
    straight_up_pool_markdown,
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
