from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.portfolio import (
    kelly_fraction,
    simulate_bankroll_paths,
    simulate_paper_bankroll,
)


def test_kelly_fraction_for_american_odds() -> None:
    assert kelly_fraction(0.55, -110) == pytest.approx(0.055)
    assert kelly_fraction(0.50, -110) == 0.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        kelly_fraction(1.1, -110)


def test_weekly_paper_sizing_caps_and_settles() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2022, 2022, 2022, 2022],
            "week": [1, 1, 1, 2],
            "gameday": pd.to_datetime(["2022-09-11", "2022-09-11", "2022-09-11", "2022-09-18"]),
            "bet_side": ["HOME", "AWAY", "PASS", "HOME"],
            "bet_odds": [-110.0, -110.0, float("nan"), -110.0],
            "home_cover_probability": [0.65, 0.35, 0.50, 0.60],
            "home_cover": [1.0, 1.0, 0.0, 0.0],
            "ats_margin": [3.0, 2.0, -1.0, -2.0],
        }
    )
    result = simulate_paper_bankroll(
        predictions,
        initial_bankroll=100.0,
        kelly_multiplier=1.0,
        max_bet_fraction=0.10,
        max_week_fraction=0.10,
    )
    week_one = result.ledger.loc[result.ledger["week"].eq(1)]
    assert week_one["stake_fraction"].sum() == pytest.approx(0.10)
    assert result.metrics["resolved_bets"] == 3
    assert result.metrics["wins"] == 1
    assert result.metrics["losses"] == 2
    assert result.metrics["final_bankroll"] < 100.0
    assert result.metrics["max_drawdown"] < 0.0


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("initial_bankroll", 0.0, "positive"),
        ("kelly_multiplier", 1.1, "between"),
        ("max_bet_fraction", 0.0, r"in \(0, 1\]"),
        ("max_week_fraction", 0.0, r"in \(0, 1\]"),
    ],
)
def test_portfolio_configuration_guards(keyword: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_paper_bankroll(pd.DataFrame(), **{keyword: value})


def test_portfolio_requires_prediction_contract() -> None:
    with pytest.raises(ValueError, match="missing portfolio columns"):
        simulate_paper_bankroll(pd.DataFrame({"season": [2022]}))


def test_probability_haircut_reduces_stakes() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2022],
            "week": [1],
            "gameday": pd.to_datetime(["2022-09-11"]),
            "bet_side": ["HOME"],
            "bet_odds": [-110.0],
            "home_cover_probability": [0.58],
            "home_cover": [1.0],
            "ats_margin": [3.0],
        }
    )
    raw = simulate_paper_bankroll(predictions, kelly_multiplier=1.0, max_bet_fraction=0.20)
    conservative = simulate_paper_bankroll(
        predictions,
        kelly_multiplier=1.0,
        max_bet_fraction=0.20,
        probability_haircut=0.02,
    )
    assert conservative.ledger.loc[0, "stake"] < raw.ledger.loc[0, "stake"]
    assert conservative.ledger.loc[0, "bet_probability"] == pytest.approx(0.56)


def test_bankroll_monte_carlo_is_deterministic_and_bounded() -> None:
    predictions = pd.DataFrame(
        {
            "season": [2022, 2022],
            "week": [1, 2],
            "bet_side": ["HOME", "AWAY"],
            "bet_odds": [-110.0, -110.0],
            "home_cover_probability": [0.60, 0.40],
        }
    )
    first = simulate_bankroll_paths(predictions, paths=200, seed=7)
    second = simulate_bankroll_paths(predictions, paths=200, seed=7)
    assert first.metrics == second.metrics
    assert first.paths.equals(second.paths)
    assert 0.0 <= first.metrics["probability_of_loss"] <= 1.0
    assert first.metrics["terminal_bankroll_p05"] <= first.metrics["terminal_bankroll_p95"]
