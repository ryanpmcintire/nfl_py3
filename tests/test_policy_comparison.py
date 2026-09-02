from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.policy_comparison import POLICY_ORDER, compare_paper_policies


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d", "e", "f"],
            "season": [2025] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "gameday": [
                "2025-09-07",
                "2025-09-07",
                "2025-09-14",
                "2025-09-14",
                "2025-09-21",
                "2025-09-21",
            ],
            "home_team": ["BUF", "BUF", "KC", "DET", "SEA", "GB"],
            "away_team": ["MIA", "MIA", "LV", "CHI", "ARI", "MIN"],
            "bet_side": ["HOME", "HOME", "AWAY", "PASS", "HOME", "AWAY"],
            "bet_odds": [-110.0, -110.0, -105.0, np.nan, -110.0, -110.0],
            "home_cover_probability": [0.57, 0.57, 0.42, 0.50, 0.56, 0.40],
        }
    )


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d", "e", "f"],
            "home_cover": [1.0, 1.0, 1.0, 0.0, np.nan, np.nan],
            "ats_margin": [3.0, 2.0, 1.0, -4.0, np.nan, 0.0],
        }
    )


def test_policy_comparison_preserves_every_prediction_for_every_policy() -> None:
    result = compare_paper_policies(_decisions(), _outcomes())

    assert list(result.metrics["policy"]) == list(POLICY_ORDER)
    assert len(result.ledger) == len(_decisions()) * len(POLICY_ORDER)
    assert result.ledger.groupby("policy")["game_id"].nunique().eq(6).all()
    assert result.ledger.groupby("game_id")["policy"].nunique().eq(4).all()
    assert result.metrics["paper_only"].eq(True).all()
    assert result.configuration["paper_only"] is True
    assert "winner" not in result.configuration


def test_policy_comparison_is_deterministic() -> None:
    first = compare_paper_policies(_decisions(), _outcomes())
    second = compare_paper_policies(_decisions(), _outcomes())

    assert first.ledger.equals(second.ledger)
    assert first.metrics.equals(second.metrics)
    assert first.configuration == second.configuration


def test_flat_units_are_constant_and_weekly_stakes_are_simultaneous() -> None:
    result = compare_paper_policies(
        _decisions(),
        _outcomes(),
        initial_bankroll=100.0,
        flat_unit_fraction=0.01,
    )
    flat = result.ledger.loc[result.ledger["policy"].eq("flat_unit")]
    played = flat.loc[flat["bet_side"].ne("PASS")]

    assert played["stake"].to_numpy(dtype=float) == pytest.approx(np.ones(len(played)))
    week_one = flat.loc[flat["week"].eq(1)]
    assert week_one["bankroll_before_week"].nunique() == 1
    assert week_one["bankroll_after_week"].nunique() == 1


def test_policies_apply_distinct_sizing_rules_and_common_caps() -> None:
    result = compare_paper_policies(
        _decisions(),
        _outcomes(),
        max_bet_fraction=0.20,
        max_week_fraction=0.50,
        team_factor_strength=1.0,
    )
    ledger = result.ledger
    week_one = ledger.loc[ledger["week"].eq(1)]
    quarter = week_one.loc[week_one["policy"].eq("quarter_kelly"), "stake_fraction"].sum()
    constrained = week_one.loc[
        week_one["policy"].eq("risk_constrained_kelly"), "stake_fraction"
    ].sum()
    confidence = ledger.loc[
        ledger["policy"].eq("confidence_tier") & ledger["game_id"].eq("c"),
        "stake_fraction",
    ].item()

    assert constrained < quarter
    assert confidence == pytest.approx(0.02)
    weekly_exposure = ledger.groupby(["policy", "season", "week"])["stake_fraction"].sum()
    assert weekly_exposure.le(0.50 + 1e-9).all()
    assert ledger["stake_fraction"].le(0.20 + 1e-9).all()


def test_pushes_and_unresolved_rows_are_preserved_without_fabricated_profit() -> None:
    result = compare_paper_policies(_decisions(), _outcomes())
    active = result.ledger.loc[result.ledger["policy"].eq("flat_unit")].set_index("game_id")

    assert active.loc["e", "paper_result"] == "UNRESOLVED"
    assert np.isnan(active.loc["e", "profit"])
    assert active.loc["f", "paper_result"] == "PUSH"
    assert active.loc["f", "profit"] == 0.0
    assert result.metrics["pushes"].eq(1).all()
    assert result.metrics["resolved_bets"].eq(4).all()
    assert result.metrics["max_drawdown"].le(0.0).all()


def test_outcomes_cannot_change_precomputed_policy_stakes() -> None:
    alternate = _outcomes().copy()
    alternate.loc[alternate["game_id"].eq("e"), ["ats_margin", "home_cover"]] = [3.0, 1.0]
    alternate.loc[alternate["game_id"].eq("f"), ["ats_margin", "home_cover"]] = [-2.0, 0.0]
    original = compare_paper_policies(_decisions(), _outcomes()).ledger
    changed = compare_paper_policies(_decisions(), alternate).ledger

    identity = ["policy", "game_id"]
    original_stakes = original.set_index(identity)["stake"].sort_index()
    changed_stakes = changed.set_index(identity)["stake"].sort_index()
    assert original_stakes.equals(changed_stakes)


def test_optional_factor_limit_flows_to_risk_constrained_policy() -> None:
    decisions = _decisions()
    active_ids = decisions.loc[decisions["bet_side"].ne("PASS"), "game_id"]
    exposures = pd.DataFrame(
        {"weather:wind": 1.0},
        index=active_ids,
    )
    result = compare_paper_policies(
        decisions,
        _outcomes(),
        max_bet_fraction=0.20,
        max_week_fraction=0.50,
        factor_exposures=exposures,
        factor_strengths={"weather:wind": 0.2},
        factor_limits={"weather:wind": 0.015},
    )
    risk = result.ledger.loc[result.ledger["policy"].eq("risk_constrained_kelly")]
    exposure = risk.groupby(["season", "week"])["stake_fraction"].sum()
    assert exposure.le(0.015 + 1e-9).all()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda outcomes: outcomes.assign(
                home_cover=outcomes["home_cover"].where(outcomes["game_id"].ne("a"), 0.0)
            ),
            "must agree",
        ),
        (
            lambda outcomes: outcomes.loc[outcomes["game_id"].ne("a")],
            "exactly match",
        ),
    ],
)
def test_outcome_contract_fails_closed(mutator: object, message: str) -> None:
    changed = mutator(_outcomes())  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        compare_paper_policies(_decisions(), changed)


def test_optional_prediction_timestamp_guard_prevents_post_kickoff_replay() -> None:
    decisions = _decisions().assign(
        kickoff="2025-09-07T17:00:00Z",
        prediction_timestamp="2025-09-07T16:00:00Z",
    )
    decisions.loc[1:, "kickoff"] = [
        "2025-09-07T20:00:00Z",
        "2025-09-14T17:00:00Z",
        "2025-09-14T20:00:00Z",
        "2025-09-21T17:00:00Z",
        "2025-09-21T20:00:00Z",
    ]
    decisions.loc[2:, "prediction_timestamp"] = [
        "2025-09-14T16:00:00Z",
        "2025-09-14T16:00:00Z",
        "2025-09-21T16:00:00Z",
        "2025-09-21T16:00:00Z",
    ]
    decisions.loc[0, "prediction_timestamp"] = decisions.loc[0, "kickoff"]

    with pytest.raises(ValueError, match="strictly before kickoff"):
        compare_paper_policies(decisions, _outcomes())

    with pytest.raises(ValueError, match="must be supplied together"):
        compare_paper_policies(_decisions().assign(prediction_timestamp="2025-09-01"), _outcomes())


def test_explicit_covariance_requires_every_active_week() -> None:
    with pytest.raises(ValueError, match="every active season/week exactly"):
        compare_paper_policies(
            _decisions(),
            _outcomes(),
            covariance_by_week={(2025, 1): pd.DataFrame()},
        )


def test_configuration_guards_are_explicit() -> None:
    with pytest.raises(ValueError, match="strictly descending"):
        compare_paper_policies(
            _decisions(),
            _outcomes(),
            confidence_tiers=((0.55, 0.01), (0.58, 0.02)),
        )
