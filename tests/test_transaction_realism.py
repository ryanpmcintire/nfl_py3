from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pandas as pd
import pytest

from nfl_ats.transaction_realism import (
    ExecutionPolicy,
    execute_paper_orders,
    settle_paper_executions,
)


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["filled", "limited", "stale", "unavailable", "closed", "pass"],
            "bet_side": ["HOME", "AWAY", "HOME", "HOME", "AWAY", "PASS"],
            "requested_stake": [10.0, 10.0, 10.0, 10.0, 10.0, 0.0],
            "requested_spread_line": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "requested_odds": [-110.0] * 6,
            "offered_spread_line": [3.5, 3.5, 3.0, 3.0, 3.0, 3.0],
            "offered_odds": [-105.0, -120.0, -110.0, -110.0, -110.0, -110.0],
            "quote_timestamp": [
                "2026-09-13T16:55:00Z",
                "2026-09-13T16:55:00Z",
                "2026-09-13T16:30:00Z",
                "2026-09-13T16:55:00Z",
                "2026-09-13T16:55:00Z",
                "2026-09-13T16:55:00Z",
            ],
            "kickoff": [
                "2026-09-13T17:05:00Z",
                "2026-09-13T17:05:00Z",
                "2026-09-13T17:05:00Z",
                "2026-09-13T17:05:00Z",
                "2026-09-13T16:59:00Z",
                "2026-09-13T17:05:00Z",
            ],
            "book_available": [True, True, True, False, True, True],
            "max_stake": [float("nan"), 4.0, 10.0, 10.0, 10.0, 10.0],
        }
    )


def test_execution_applies_limits_staleness_availability_and_kickoff() -> None:
    executions = execute_paper_orders(
        _orders(),
        execution_time="2026-09-13T17:00:00Z",
        policy=ExecutionPolicy(max_quote_age=timedelta(minutes=15)),
    ).set_index("game_id")

    assert executions.loc["filled", "execution_status"] == "FILLED"
    assert bool(executions.loc["filled", "limit_known"]) is False
    assert executions.loc["filled", "line_value_change"] == pytest.approx(-0.5)
    assert executions.loc["filled", "price_value_change"] > 0.0
    assert executions.loc["limited", "filled_stake"] == pytest.approx(4.0)
    assert executions.loc["limited", "unfilled_stake"] == pytest.approx(6.0)
    assert bool(executions.loc["limited", "limit_applied"]) is True
    assert executions.loc["limited", "line_value_change"] == pytest.approx(0.5)
    assert executions.loc["limited", "break_even_probability_change"] > 0.0
    assert executions.loc["stale", "rejection_reason"] == "quote_stale"
    assert executions.loc["unavailable", "rejection_reason"] == "book_unavailable"
    assert executions.loc["closed", "rejection_reason"] == "kickoff_closed"
    assert executions.loc["pass", "execution_status"] == "PASS"


def test_execution_rejects_future_or_incomplete_quotes() -> None:
    orders = _orders().iloc[:2].copy()
    orders.loc[0, "quote_timestamp"] = "2026-09-13T17:01:00Z"
    orders.loc[1, "offered_odds"] = float("nan")

    executions = execute_paper_orders(orders, execution_time="2026-09-13T17:00:00Z").set_index(
        "game_id"
    )
    assert executions.loc["filled", "rejection_reason"] == "quote_after_execution"
    assert executions.loc["limited", "rejection_reason"] == "quote_missing"


def test_settlement_uses_executed_line_and_preserves_pushes() -> None:
    orders = _orders().iloc[:2].copy()
    executions = execute_paper_orders(orders, execution_time="2026-09-13T17:00:00Z")
    results = pd.DataFrame({"game_id": ["filled", "limited"], "home_margin": [3.5, 7.0]})

    settled = settle_paper_executions(executions, results).set_index("game_id")
    assert settled.loc["filled", "settlement"] == "PUSH"
    assert settled.loc["filled", "profit"] == 0.0
    assert settled.loc["limited", "settlement"] == "LOSS"
    assert settled.loc["limited", "profit"] == pytest.approx(-4.0)


def test_unsettled_and_rejected_orders_never_invent_results() -> None:
    orders = _orders().iloc[[0, 3]].copy()
    executions = execute_paper_orders(orders, execution_time="2026-09-13T17:00:00Z")
    results = pd.DataFrame({"game_id": ["filled"], "home_margin": [float("nan")]})

    settled = settle_paper_executions(executions, results).set_index("game_id")
    assert settled.loc["filled", "settlement"] == "UNSETTLED"
    assert pd.isna(settled.loc["filled", "profit"])
    assert settled.loc["unavailable", "settlement"] == "NOT_FILLED"
    assert settled.loc["unavailable", "profit"] == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda frame: frame.drop(columns="kickoff"),
        lambda frame: frame.assign(game_id="duplicate"),
        lambda frame: frame.assign(bet_side="MAYBE"),
        lambda frame: frame.assign(requested_stake=-1.0),
        lambda frame: frame.assign(max_stake=-1.0),
    ],
)
def test_execution_contract_fails_closed(
    mutation: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    frame = mutation(_orders())
    with pytest.raises(ValueError):
        execute_paper_orders(frame, execution_time="2026-09-13T17:00:00Z")


def test_policy_rejects_negative_quote_age_limit() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ExecutionPolicy(max_quote_age=timedelta(seconds=-1))
