from __future__ import annotations

import math

import pytest

from nfl_ats.odds import (
    choose_bet,
    implied_probability,
    market_hold,
    no_vig_probabilities,
    profit_per_unit,
    settle_bet,
)


def test_american_odds_math() -> None:
    assert implied_probability(-110) == pytest.approx(110 / 210)
    assert implied_probability(150) == pytest.approx(0.4)
    assert implied_probability(None) == pytest.approx(110 / 210)
    assert profit_per_unit(-110) == pytest.approx(100 / 110)
    assert profit_per_unit(150) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="zero"):
        implied_probability(0)
    home, away = no_vig_probabilities(-110, -110)
    assert home == pytest.approx(0.5)
    assert away == pytest.approx(0.5)
    assert market_hold(-110, -110) == pytest.approx(2 * 110 / 210 - 1)


def test_vig_aware_bet_selection_and_settlement() -> None:
    home = choose_bet(0.58, -110, -105, min_edge=0.02)
    assert home.side == "HOME"
    assert home.edge == pytest.approx(0.58 - 110 / 210)
    assert settle_bet(home.side, 1, home.odds) == pytest.approx(100 / 110)
    assert settle_bet(home.side, 0, home.odds) == -1.0

    away = choose_bet(0.40, -110, 120, min_edge=0.02)
    assert away.side == "AWAY"
    assert settle_bet(away.side, 0, away.odds) == pytest.approx(1.2)

    passed = choose_bet(0.50, min_edge=0.02)
    assert passed.side == "PASS"
    assert math.isnan(passed.odds)
    assert settle_bet("PASS", 1, None) == 0.0


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_invalid_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        choose_bet(probability)
