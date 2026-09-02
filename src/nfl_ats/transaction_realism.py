"""Paper-only execution and settlement with observable transaction constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import cast

import numpy as np
import pandas as pd

from nfl_ats.odds import implied_probability, profit_per_unit


@dataclass(frozen=True)
class ExecutionPolicy:
    """Rules governing whether a quoted paper order can be filled."""

    max_quote_age: timedelta = timedelta(minutes=15)

    def __post_init__(self) -> None:
        if self.max_quote_age < timedelta(0):
            raise ValueError("max_quote_age cannot be negative")


_ORDER_COLUMNS = {
    "game_id",
    "bet_side",
    "requested_stake",
    "requested_spread_line",
    "requested_odds",
    "offered_spread_line",
    "offered_odds",
    "quote_timestamp",
    "kickoff",
    "book_available",
    "max_stake",
}


def _as_utc(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(str(value), errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    return pd.Timestamp(timestamp)


def _finite_number(value: object) -> float | None:
    try:
        number = float(cast(str | float | int, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def execute_paper_orders(
    orders: pd.DataFrame,
    *,
    execution_time: object,
    policy: ExecutionPolicy | None = None,
) -> pd.DataFrame:
    """Apply availability, quote-age, kickoff, limit, line and price realism.

    The function records a fill or a named rejection for every requested paper
    order. It never contacts a book and cannot place a wager. Unknown limits
    are represented by ``NaN`` and do not invent a cap; ``limit_known`` keeps
    that uncertainty visible in the returned audit rows.
    """

    missing = sorted(_ORDER_COLUMNS.difference(orders.columns))
    if missing:
        raise ValueError(f"Paper orders are missing columns: {', '.join(missing)}")
    if orders["game_id"].isna().any() or orders["game_id"].duplicated().any():
        raise ValueError("Paper orders require unique, non-null game_id values")

    now = _as_utc(execution_time, name="execution_time")
    resolved_policy = policy or ExecutionPolicy()
    rows: list[dict[str, object]] = []
    for _, order in orders.iterrows():
        side = str(order["bet_side"]).upper()
        if side not in {"HOME", "AWAY", "PASS"}:
            raise ValueError(f"invalid bet_side for {order['game_id']}: {side}")
        requested_stake = _finite_number(order["requested_stake"])
        if requested_stake is None or requested_stake < 0.0:
            raise ValueError("requested_stake must be finite and non-negative")

        kickoff = _as_utc(order["kickoff"], name="kickoff")
        quote = pd.to_datetime(order["quote_timestamp"], errors="coerce", utc=True)
        requested_line = _finite_number(order["requested_spread_line"])
        requested_odds = _finite_number(order["requested_odds"])
        offered_line = _finite_number(order["offered_spread_line"])
        offered_odds = _finite_number(order["offered_odds"])
        max_stake = _finite_number(order["max_stake"])
        if max_stake is not None and max_stake < 0.0:
            raise ValueError("max_stake must be non-negative when known")

        reason: str | None = None
        if side == "PASS" or requested_stake == 0.0:
            reason = "pass"
        elif not bool(order["book_available"]):
            reason = "book_unavailable"
        elif pd.isna(quote) or offered_line is None or offered_odds is None:
            reason = "quote_missing"
        elif offered_odds == 0.0:
            reason = "invalid_price"
        elif pd.Timestamp(quote) > now:
            reason = "quote_after_execution"
        elif now >= kickoff:
            reason = "kickoff_closed"
        elif now - pd.Timestamp(quote) > resolved_policy.max_quote_age:
            reason = "quote_stale"

        filled_stake = 0.0 if reason is not None else requested_stake
        limit_applied = False
        if reason is None and max_stake is not None and filled_stake > max_stake:
            filled_stake = max_stake
            limit_applied = True

        line_value_change = np.nan
        price_value_change = np.nan
        break_even_change = np.nan
        if reason is None and requested_line is not None and requested_odds not in (None, 0.0):
            assert offered_line is not None
            assert offered_odds is not None
            canonical_move = offered_line - requested_line
            line_value_change = canonical_move if side == "AWAY" else -canonical_move
            price_value_change = profit_per_unit(offered_odds) - profit_per_unit(requested_odds)
            break_even_change = implied_probability(offered_odds) - implied_probability(
                requested_odds
            )

        rows.append(
            {
                **{str(key): value for key, value in order.items()},
                "execution_time": now,
                "quote_age_seconds": (
                    (now - pd.Timestamp(quote)).total_seconds() if not pd.isna(quote) else np.nan
                ),
                "execution_status": "REJECTED"
                if reason not in (None, "pass")
                else ("PASS" if reason == "pass" else "FILLED"),
                "rejection_reason": reason if reason not in (None, "pass") else None,
                "filled_stake": filled_stake,
                "unfilled_stake": requested_stake - filled_stake,
                "limit_known": max_stake is not None,
                "limit_applied": limit_applied,
                "executed_spread_line": offered_line if reason is None else np.nan,
                "executed_odds": offered_odds if reason is None else np.nan,
                "line_value_change": line_value_change,
                "price_value_change": price_value_change,
                "break_even_probability_change": break_even_change,
            }
        )
    return pd.DataFrame(rows)


def settle_paper_executions(executions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Settle filled ATS paper orders at their actually executed line and price."""

    required_execution = {
        "game_id",
        "bet_side",
        "execution_status",
        "filled_stake",
        "executed_spread_line",
        "executed_odds",
    }
    missing = sorted(required_execution.difference(executions.columns))
    if missing:
        raise ValueError(f"Executions are missing settlement columns: {', '.join(missing)}")
    if not {"game_id", "home_margin"}.issubset(results.columns):
        raise ValueError("Results require game_id and home_margin")
    if results["game_id"].isna().any() or results["game_id"].duplicated().any():
        raise ValueError("Results require unique, non-null game_id values")

    settled = executions.merge(
        results.loc[:, ["game_id", "home_margin"]], on="game_id", how="left", validate="one_to_one"
    )
    outcomes: list[str] = []
    profits: list[float] = []
    for _, row in settled.iterrows():
        if row["execution_status"] != "FILLED":
            outcomes.append("NOT_FILLED")
            profits.append(0.0)
            continue
        home_margin = _finite_number(row["home_margin"])
        if home_margin is None:
            outcomes.append("UNSETTLED")
            profits.append(np.nan)
            continue
        ats_margin = home_margin - float(row["executed_spread_line"])
        if math.isclose(ats_margin, 0.0, abs_tol=1e-9):
            outcomes.append("PUSH")
            profits.append(0.0)
            continue
        home_cover = ats_margin > 0.0
        won = home_cover if row["bet_side"] == "HOME" else not home_cover
        stake = float(row["filled_stake"])
        outcomes.append("WIN" if won else "LOSS")
        profits.append(stake * profit_per_unit(row["executed_odds"]) if won else -stake)

    settled["settlement"] = outcomes
    settled["profit"] = profits
    return settled


__all__ = ["ExecutionPolicy", "execute_paper_orders", "settle_paper_executions"]
