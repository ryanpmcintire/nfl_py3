"""Odds conversion, wager selection, and settlement helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_AMERICAN_ODDS = -110.0


def _resolved_odds(american_odds: float | int | None) -> float:
    if american_odds is None:
        return DEFAULT_AMERICAN_ODDS
    odds = float(american_odds)
    return odds if math.isfinite(odds) else DEFAULT_AMERICAN_ODDS


def implied_probability(american_odds: float | int | None) -> float:
    """Return the break-even probability for American odds."""

    odds = _resolved_odds(american_odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def profit_per_unit(american_odds: float | int | None) -> float:
    """Return net profit on one unit risked for a winning wager."""

    odds = _resolved_odds(american_odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def no_vig_probabilities(
    home_odds: float | int | None,
    away_odds: float | int | None,
) -> tuple[float, float]:
    """Normalize two implied probabilities to remove the bookmaker margin."""

    home_raw = implied_probability(home_odds)
    away_raw = implied_probability(away_odds)
    total = home_raw + away_raw
    return home_raw / total, away_raw / total


def market_hold(home_odds: float | int | None, away_odds: float | int | None) -> float:
    """Return the two-way implied probability above 100%."""

    return implied_probability(home_odds) + implied_probability(away_odds) - 1.0


@dataclass(frozen=True)
class BetDecision:
    """A vig-aware wager recommendation for one game."""

    side: str
    edge: float
    odds: float
    break_even_probability: float


def choose_bet(
    home_cover_probability: float,
    home_odds: float | int | None = None,
    away_odds: float | int | None = None,
    min_edge: float = 0.02,
) -> BetDecision:
    """Choose HOME, AWAY, or PASS based on probability above break-even."""

    if not 0.0 <= home_cover_probability <= 1.0:
        raise ValueError("home_cover_probability must be between 0 and 1")
    if min_edge < 0:
        raise ValueError("min_edge cannot be negative")

    resolved_home_odds = _resolved_odds(home_odds)
    resolved_away_odds = _resolved_odds(away_odds)
    home_break_even = implied_probability(resolved_home_odds)
    away_break_even = implied_probability(resolved_away_odds)
    home_edge = home_cover_probability - home_break_even
    away_edge = (1.0 - home_cover_probability) - away_break_even

    if max(home_edge, away_edge) < min_edge:
        return BetDecision("PASS", max(home_edge, away_edge), math.nan, math.nan)
    if home_edge >= away_edge:
        return BetDecision("HOME", home_edge, resolved_home_odds, home_break_even)
    return BetDecision("AWAY", away_edge, resolved_away_odds, away_break_even)


def settle_bet(side: str, home_cover: float | int, odds: float | int | None) -> float:
    """Settle a one-unit wager; pushes should be excluded before calling."""

    if side == "PASS":
        return 0.0
    home_won = int(home_cover) == 1
    bet_won = home_won if side == "HOME" else not home_won
    return profit_per_unit(odds) if bet_won else -1.0
