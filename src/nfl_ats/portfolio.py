"""Paper-only bankroll sizing and settlement analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.odds import profit_per_unit, settle_bet


@dataclass(frozen=True)
class PortfolioResult:
    ledger: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(frozen=True)
class BankrollSimulation:
    paths: pd.DataFrame
    metrics: dict[str, Any]


def kelly_fraction(win_probability: float, american_odds: float | int | None) -> float:
    """Return the full-Kelly bankroll fraction for a two-outcome wager."""

    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be between 0 and 1")
    net_profit = profit_per_unit(american_odds)
    fraction = (net_profit * win_probability - (1.0 - win_probability)) / net_profit
    return max(0.0, float(fraction))


def _drawdown(bankroll: pd.Series) -> pd.Series:
    peaks = bankroll.cummax()
    return bankroll / peaks - 1.0


def simulate_paper_bankroll(
    predictions: pd.DataFrame,
    initial_bankroll: float = 100.0,
    kelly_multiplier: float = 0.25,
    max_bet_fraction: float = 0.02,
    max_week_fraction: float = 0.10,
    probability_haircut: float = 0.0,
) -> PortfolioResult:
    """Size simultaneous weekly paper bets from the same starting bankroll.

    Desired fractional-Kelly stakes are capped per game, then scaled pro rata
    if aggregate risk exceeds the weekly cap. No result from one game changes a
    different stake in the same NFL week.
    """

    if initial_bankroll <= 0:
        raise ValueError("initial_bankroll must be positive")
    if not 0.0 <= kelly_multiplier <= 1.0:
        raise ValueError("kelly_multiplier must be between 0 and 1")
    if not 0.0 < max_bet_fraction <= 1.0:
        raise ValueError("max_bet_fraction must be in (0, 1]")
    if not 0.0 < max_week_fraction <= 1.0:
        raise ValueError("max_week_fraction must be in (0, 1]")
    if not 0.0 <= probability_haircut < 0.5:
        raise ValueError("probability_haircut must be in [0, 0.5)")
    required = {
        "season",
        "week",
        "gameday",
        "bet_side",
        "bet_odds",
        "home_cover_probability",
        "home_cover",
        "ats_margin",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing portfolio columns: {', '.join(missing)}")

    ledger = predictions.copy().sort_values(["gameday", "game_id"]).reset_index(drop=True)
    for column in (
        "bet_probability",
        "raw_bet_probability",
        "full_kelly_fraction",
        "stake_fraction",
        "stake",
        "profit",
        "bankroll_before_week",
        "bankroll_after_week",
    ):
        ledger[column] = 0.0

    bankroll = float(initial_bankroll)
    weekly_bankroll = [bankroll]
    resolved_bets = 0
    wins = 0
    losses = 0
    pushes = 0
    for _, indices in ledger.groupby(["season", "week"], sort=True).groups.items():
        week_indices = list(indices)
        desired: dict[int, float] = {}
        for index in week_indices:
            row = ledger.loc[index]
            side = str(row["bet_side"])
            if side == "PASS":
                continue
            home_probability = float(row["home_cover_probability"])
            probability = home_probability if side == "HOME" else 1.0 - home_probability
            conservative_probability = max(0.5, probability - probability_haircut)
            full_fraction = kelly_fraction(conservative_probability, float(row["bet_odds"]))
            desired_fraction = min(max_bet_fraction, kelly_multiplier * full_fraction)
            ledger.at[index, "raw_bet_probability"] = probability
            ledger.at[index, "bet_probability"] = conservative_probability
            ledger.at[index, "full_kelly_fraction"] = full_fraction
            desired[index] = desired_fraction

        total_desired = sum(desired.values())
        scale = min(1.0, max_week_fraction / total_desired) if total_desired else 1.0
        weekly_profit = 0.0
        for index in week_indices:
            ledger.at[index, "bankroll_before_week"] = bankroll
            fraction = desired.get(index, 0.0) * scale
            stake = bankroll * fraction
            ledger.at[index, "stake_fraction"] = fraction
            ledger.at[index, "stake"] = stake
            if stake == 0.0:
                continue

            row = ledger.loc[index]
            ats_margin = float(row["ats_margin"])
            if not math.isfinite(ats_margin):
                ledger.at[index, "profit"] = np.nan
                continue
            if ats_margin == 0.0:
                profit = 0.0
                pushes += 1
            else:
                profit = stake * settle_bet(
                    str(row["bet_side"]), float(row["home_cover"]), float(row["bet_odds"])
                )
                wins += int(profit > 0)
                losses += int(profit < 0)
            resolved_bets += 1
            ledger.at[index, "profit"] = profit
            weekly_profit += profit

        bankroll += weekly_profit
        ledger.loc[week_indices, "bankroll_after_week"] = bankroll
        weekly_bankroll.append(bankroll)

    bankroll_path = pd.Series(weekly_bankroll, dtype="float64")
    max_drawdown = float(_drawdown(bankroll_path).min())
    total_staked = float(ledger["stake"].sum())
    net_profit = bankroll - initial_bankroll
    metrics = {
        "initial_bankroll": float(initial_bankroll),
        "final_bankroll": bankroll,
        "return": net_profit / initial_bankroll,
        "net_profit": net_profit,
        "max_drawdown": max_drawdown,
        "total_staked": total_staked,
        "turnover": total_staked / initial_bankroll,
        "resolved_bets": resolved_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "kelly_multiplier": kelly_multiplier,
        "max_bet_fraction": max_bet_fraction,
        "max_week_fraction": max_week_fraction,
        "probability_haircut": probability_haircut,
    }
    return PortfolioResult(ledger=ledger, metrics=metrics)


def simulate_bankroll_paths(
    predictions: pd.DataFrame,
    *,
    paths: int = 5_000,
    seed: int = 20260812,
    initial_bankroll: float = 100.0,
    kelly_multiplier: float = 0.25,
    max_bet_fraction: float = 0.02,
    max_week_fraction: float = 0.10,
    probability_haircut: float = 0.0,
    ruin_fraction: float = 0.50,
) -> BankrollSimulation:
    """Simulate bankroll paths conditional on the model's stated probabilities.

    This quantifies the risk implied by the probabilities; it does not validate
    that those probabilities are correct.
    """

    if paths < 100:
        raise ValueError("paths must be at least 100")
    if initial_bankroll <= 0:
        raise ValueError("initial_bankroll must be positive")
    if not 0.0 <= kelly_multiplier <= 1.0:
        raise ValueError("kelly_multiplier must be between 0 and 1")
    if not 0.0 < max_bet_fraction <= 1.0:
        raise ValueError("max_bet_fraction must be in (0, 1]")
    if not 0.0 < max_week_fraction <= 1.0:
        raise ValueError("max_week_fraction must be in (0, 1]")
    if not 0.0 <= probability_haircut < 0.5:
        raise ValueError("probability_haircut must be in [0, 0.5)")
    if not 0.0 < ruin_fraction < 1.0:
        raise ValueError("ruin_fraction must be between 0 and 1")
    required = {
        "season",
        "week",
        "bet_side",
        "bet_odds",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing simulation columns: {', '.join(missing)}")

    generator = np.random.default_rng(seed)
    bankroll = np.full(paths, initial_bankroll, dtype=float)
    peak = bankroll.copy()
    maximum_drawdown = np.zeros(paths, dtype=float)
    weekly_columns: dict[str, np.ndarray] = {"start": bankroll.copy()}
    weeks_simulated = 0
    bets_simulated = 0
    for (season, week), group in predictions.groupby(["season", "week"], sort=True):
        bets = group.loc[group["bet_side"].ne("PASS")].copy()
        if bets.empty:
            continue
        probabilities: list[float] = []
        returns: list[float] = []
        fractions: list[float] = []
        for _, row in bets.iterrows():
            home_probability = float(row["home_cover_probability"])
            side_probability = (
                home_probability if row["bet_side"] == "HOME" else 1.0 - home_probability
            )
            conservative = max(0.5, side_probability - probability_haircut)
            fraction = min(
                max_bet_fraction,
                kelly_multiplier * kelly_fraction(conservative, float(row["bet_odds"])),
            )
            probabilities.append(conservative)
            returns.append(profit_per_unit(float(row["bet_odds"])))
            fractions.append(fraction)
        total = sum(fractions)
        scale = min(1.0, max_week_fraction / total) if total else 1.0
        stake_fractions = np.asarray(fractions, dtype=float) * scale
        win_probability = np.asarray(probabilities, dtype=float)
        win_return = np.asarray(returns, dtype=float)
        wins = generator.random((paths, len(bets))) < win_probability
        payoffs = np.where(wins, win_return, -1.0)
        bankroll *= 1.0 + payoffs @ stake_fractions
        peak = np.maximum(peak, bankroll)
        maximum_drawdown = np.minimum(maximum_drawdown, bankroll / peak - 1.0)
        weekly_columns[f"{season}-W{week}"] = bankroll.copy()
        weeks_simulated += 1
        bets_simulated += len(bets)

    path_frame = pd.DataFrame(weekly_columns)
    terminal_return = bankroll / initial_bankroll - 1.0
    metrics = {
        "paths": paths,
        "seed": seed,
        "weeks_simulated": weeks_simulated,
        "bets_simulated": bets_simulated,
        "initial_bankroll": initial_bankroll,
        "terminal_bankroll_mean": float(bankroll.mean()),
        "terminal_bankroll_median": float(np.median(bankroll)),
        "terminal_bankroll_p05": float(np.quantile(bankroll, 0.05)),
        "terminal_bankroll_p95": float(np.quantile(bankroll, 0.95)),
        "probability_of_loss": float(np.mean(terminal_return < 0.0)),
        "probability_of_ruin": float(np.mean(bankroll <= initial_bankroll * ruin_fraction)),
        "median_max_drawdown": float(np.median(maximum_drawdown)),
        "p05_max_drawdown": float(np.quantile(maximum_drawdown, 0.05)),
        "kelly_multiplier": kelly_multiplier,
        "max_bet_fraction": max_bet_fraction,
        "max_week_fraction": max_week_fraction,
        "probability_haircut": probability_haircut,
        "conditional_on_model_probabilities": True,
    }
    return BankrollSimulation(paths=path_frame, metrics=metrics)
