from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.portfolio import (
    simulate_bankroll_paths,
    simulate_paper_bankroll,
    size_correlated_paper_portfolio,
)
from nfl_ats.probability_uncertainty import conservative_probability_audit


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["home", "away", "pass"],
            "season": [2026, 2026, 2026],
            "week": [1, 1, 1],
            "gameday": pd.to_datetime(["2026-09-13"] * 3),
            "home_team": ["BUF", "KC", "DET"],
            "away_team": ["MIA", "LV", "CHI"],
            "bet_side": ["HOME", "AWAY", "PASS"],
            "bet_odds": [-110.0, -110.0, float("nan")],
            "home_cover_probability": [0.60, 0.42, 0.50],
            "home_cover": [1.0, 0.0, 1.0],
            "ats_margin": [3.0, -2.0, 1.0],
        }
    )


def _uncertainty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bet_side": ["AWAY", "HOME"],
            "probability_lower_bound": [float("nan"), 0.54],
            "posterior_sd": [0.02, float("nan")],
        },
        index=pd.Index(["away", "home"], name="game_id"),
    )


def test_uncertainty_contract_supports_mixed_lower_bound_and_posterior_rows() -> None:
    audit = conservative_probability_audit(
        _candidates(), probability_uncertainty=_uncertainty(), posterior_z=2.0
    )

    assert audit.loc[0, "raw_bet_probability"] == pytest.approx(0.60)
    assert audit.loc[0, "conservative_bet_probability"] == pytest.approx(0.54)
    assert audit.loc[0, "probability_uncertainty_method"] == "supplied_lower_bound"
    assert audit.loc[1, "raw_bet_probability"] == pytest.approx(0.58)
    assert audit.loc[1, "conservative_bet_probability"] == pytest.approx(0.54)
    assert audit.loc[1, "probability_uncertainty_method"] == "posterior_sd"
    assert audit.loc[1, "posterior_z"] == pytest.approx(2.0)
    assert audit.loc[2, "probability_uncertainty_method"] == "PASS"


def test_paper_bankroll_uses_candidate_specific_probabilities_without_changing_sides() -> None:
    candidates = _candidates()
    point = simulate_paper_bankroll(candidates, kelly_multiplier=1.0, max_bet_fraction=0.5)
    conservative = simulate_paper_bankroll(
        candidates,
        kelly_multiplier=1.0,
        max_bet_fraction=0.5,
        probability_uncertainty=_uncertainty(),
        posterior_z=2.0,
    )

    sized = conservative.ledger.set_index("game_id")
    point_sized = point.ledger.set_index("game_id")
    assert sized.loc[candidates["game_id"], "bet_side"].tolist() == candidates["bet_side"].tolist()
    assert sized.loc["home", "bet_probability"] == pytest.approx(0.54)
    assert sized.loc["away", "bet_probability"] == pytest.approx(0.54)
    assert sized.loc["home", "stake"] < point_sized.loc["home", "stake"]
    assert sized.loc["away", "stake"] < point_sized.loc["away", "stake"]
    assert conservative.metrics["probability_uncertainty_methods"] == [
        "posterior_sd",
        "supplied_lower_bound",
    ]


def test_uncertainty_that_removes_edge_sizes_zero_but_does_not_change_pick() -> None:
    uncertainty = _uncertainty()
    uncertainty.loc["home", "probability_lower_bound"] = 0.48

    result = simulate_paper_bankroll(
        _candidates(),
        kelly_multiplier=1.0,
        max_bet_fraction=0.5,
        probability_uncertainty=uncertainty,
    )

    home = result.ledger.set_index("game_id").loc["home"]
    assert home["bet_side"] == "HOME"
    assert home["bet_probability"] == pytest.approx(0.5)
    assert home["stake"] == 0.0


def test_correlated_sizing_uses_same_auditable_conservative_probabilities() -> None:
    result = size_correlated_paper_portfolio(
        _candidates(),
        probability_uncertainty=_uncertainty(),
        posterior_z=2.0,
        team_factor_strength=0.0,
        kelly_multiplier=1.0,
        max_bet_fraction=0.5,
        max_total_fraction=1.0,
    )

    assert result.allocations.loc[0, "conservative_bet_probability"] == pytest.approx(0.54)
    assert result.allocations.loc[1, "conservative_bet_probability"] == pytest.approx(0.54)
    assert result.allocations.loc[0, "probability_uncertainty_method"] == "supplied_lower_bound"
    assert result.metrics["probability_uncertainty_methods"] == [
        "posterior_sd",
        "supplied_lower_bound",
    ]


def test_bankroll_paths_use_uncertainty_and_remain_deterministic() -> None:
    first = simulate_bankroll_paths(
        _candidates(),
        paths=200,
        seed=17,
        probability_uncertainty=_uncertainty(),
        posterior_z=2.0,
    )
    second = simulate_bankroll_paths(
        _candidates(),
        paths=200,
        seed=17,
        probability_uncertainty=_uncertainty(),
        posterior_z=2.0,
    )

    assert first.paths.equals(second.paths)
    assert first.metrics == second.metrics
    assert first.metrics["probability_uncertainty_methods"] == [
        "posterior_sd",
        "supplied_lower_bound",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(index="away"), "exactly match"),
        (lambda frame: frame.assign(bet_side=["HOME", "HOME"]), "bet_side must match"),
        (
            lambda frame: frame.assign(probability_lower_bound=[0.7, 0.54]),
            "exactly one",
        ),
        (
            lambda frame: frame.assign(probability_lower_bound=[float("nan"), 0.7]),
            "cannot exceed",
        ),
        (lambda frame: frame.assign(posterior_sd=[-0.1, float("nan")]), "non-negative"),
    ],
)
def test_uncertainty_contract_fails_closed(mutate: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        conservative_probability_audit(
            _candidates(),
            probability_uncertainty=mutate(_uncertainty()),  # type: ignore[operator]
        )


def test_scalar_haircut_and_candidate_uncertainty_cannot_be_ambiguously_stacked() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        simulate_paper_bankroll(
            _candidates(),
            probability_haircut=0.01,
            probability_uncertainty=_uncertainty(),
        )


@pytest.mark.parametrize("posterior_z", [0.0, float("nan"), float("inf")])
def test_posterior_multiplier_must_be_positive_and_finite(posterior_z: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        conservative_probability_audit(_candidates(), posterior_z=posterior_z)
