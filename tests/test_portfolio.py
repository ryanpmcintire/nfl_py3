from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.portfolio import (
    kelly_fraction,
    simulate_bankroll_paths,
    simulate_paper_bankroll,
    size_correlated_paper_portfolio,
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


def _correlated_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["spread-a", "spread-b", "pass"],
            "home_team": ["BUF", "BUF", "DET"],
            "away_team": ["MIA", "MIA", "CHI"],
            "bet_side": ["HOME", "HOME", "PASS"],
            "bet_odds": [-110.0, -110.0, float("nan")],
            "home_cover_probability": [0.57, 0.57, 0.50],
        }
    )


def test_correlated_paper_sizing_is_deterministic_and_reduces_duplicate_team_risk() -> None:
    candidates = _correlated_candidates()
    independent = size_correlated_paper_portfolio(
        candidates,
        team_factor_strength=0.0,
        max_bet_fraction=0.20,
        max_total_fraction=0.50,
    )
    correlated = size_correlated_paper_portfolio(
        candidates,
        team_factor_strength=1.0,
        max_bet_fraction=0.20,
        max_total_fraction=0.50,
    )
    repeated = size_correlated_paper_portfolio(
        candidates,
        team_factor_strength=1.0,
        max_bet_fraction=0.20,
        max_total_fraction=0.50,
    )

    assert correlated.allocations.equals(repeated.allocations)
    assert correlated.covariance.equals(repeated.covariance)
    assert correlated.metrics == repeated.metrics
    assert correlated.metrics["paper_only"] is True
    assert correlated.metrics["total_stake_fraction"] < independent.metrics["total_stake_fraction"]
    assert correlated.covariance.loc["spread-a", "spread-b"] > 0.0
    assert correlated.allocations.loc[2, "stake_fraction"] == 0.0


def test_optional_factor_exposure_changes_covariance_and_obeys_absolute_limit() -> None:
    candidates = _correlated_candidates().iloc[:2].copy()
    candidates.loc[1, ["home_team", "away_team"]] = ["KC", "LV"]
    exposures = pd.DataFrame(
        {"weather:windy": [1.0, 1.0], "market:prime_time": [0.5, -0.5]},
        index=["spread-a", "spread-b"],
    )
    result = size_correlated_paper_portfolio(
        candidates,
        factor_exposures=exposures,
        factor_strengths={"weather:windy": 0.5, "market:prime_time": 0.2},
        factor_limits={"weather:windy": 0.025},
        team_factor_strength=0.0,
        max_bet_fraction=0.20,
        max_total_fraction=0.50,
    )

    allocations = result.allocations.set_index("game_id")["stake_fraction"]
    weather_exposure = float(allocations @ exposures["weather:windy"])
    assert result.covariance.loc["spread-a", "spread-b"] > 0.0
    assert weather_exposure <= 0.025 + 1e-9
    assert result.metrics["covariance_source"] == "factor_scenario"


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (
            pd.DataFrame([[1.0]], index=["spread-a"], columns=["spread-a"]),
            "labels must exactly match",
        ),
        (
            pd.DataFrame(
                [[1.0, 0.2], [0.1, 1.0]],
                index=["spread-a", "spread-b"],
                columns=["spread-a", "spread-b"],
            ),
            "symmetric",
        ),
        (
            pd.DataFrame(
                [[1.0, 2.0], [2.0, 1.0]],
                index=["spread-a", "spread-b"],
                columns=["spread-a", "spread-b"],
            ),
            "positive semidefinite",
        ),
    ],
)
def test_explicit_covariance_fails_closed(covariance: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        size_correlated_paper_portfolio(
            _correlated_candidates().iloc[:2],
            covariance=covariance,
        )


def test_explicit_labeled_covariance_is_reordered_safely() -> None:
    labels = ["spread-b", "spread-a"]
    covariance = pd.DataFrame(
        [[0.9, 0.3], [0.3, 0.8]],
        index=labels,
        columns=labels,
    )
    result = size_correlated_paper_portfolio(
        _correlated_candidates().iloc[:2],
        covariance=covariance,
        max_bet_fraction=0.20,
        max_total_fraction=0.50,
    )

    assert list(result.covariance.index) == ["spread-a", "spread-b"]
    assert result.covariance.loc["spread-a", "spread-a"] == pytest.approx(0.8)
    assert result.metrics["covariance_source"] == "explicit"


def test_optional_factor_contract_refuses_ambiguous_or_missing_inputs() -> None:
    candidates = _correlated_candidates().iloc[:2]
    unlabeled = pd.DataFrame({"wind": [1.0, 1.0]}, index=["spread-a", "spread-b"])
    with pytest.raises(ValueError, match="must start with"):
        size_correlated_paper_portfolio(candidates, factor_exposures=unlabeled)

    weather = pd.DataFrame({"weather:wind": [1.0, 1.0]}, index=["spread-a", "spread-b"])
    with pytest.raises(ValueError, match="name each optional factor exactly"):
        size_correlated_paper_portfolio(candidates, factor_exposures=weather)
    with pytest.raises(ValueError, match="unknown factor"):
        size_correlated_paper_portfolio(
            candidates,
            factor_limits={"weather:not_supplied": 0.0},
            kelly_multiplier=0.0,
        )


def test_correlated_sizing_enforces_total_and_per_candidate_caps() -> None:
    result = size_correlated_paper_portfolio(
        _correlated_candidates().iloc[:2],
        team_factor_strength=0.0,
        kelly_multiplier=1.0,
        max_bet_fraction=0.02,
        max_total_fraction=0.03,
    )
    fractions = result.allocations["stake_fraction"]
    assert fractions.max() <= 0.02 + 1e-9
    assert fractions.sum() <= 0.03 + 1e-9
