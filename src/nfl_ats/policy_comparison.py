"""Deterministic, paper-only comparison of bankroll sizing policies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.odds import settle_bet
from nfl_ats.portfolio import kelly_fraction, size_correlated_paper_portfolio

POLICY_ORDER = (
    "flat_unit",
    "confidence_tier",
    "quarter_kelly",
    "risk_constrained_kelly",
)
DEFAULT_CONFIDENCE_TIERS = ((0.58, 0.02), (0.55, 0.01), (0.50, 0.005))


@dataclass(frozen=True)
class PolicyComparison:
    """Prediction-level policy ledger and one comparable metrics row per policy."""

    ledger: pd.DataFrame
    metrics: pd.DataFrame
    configuration: dict[str, Any]


def _validate_configuration(
    *,
    initial_bankroll: float,
    flat_unit_fraction: float,
    confidence_tiers: Sequence[tuple[float, float]],
    max_bet_fraction: float,
    max_week_fraction: float,
    probability_haircut: float,
) -> tuple[tuple[float, float], ...]:
    if not math.isfinite(initial_bankroll) or initial_bankroll <= 0.0:
        raise ValueError("initial_bankroll must be finite and positive")
    if not math.isfinite(flat_unit_fraction) or not 0.0 < flat_unit_fraction <= 1.0:
        raise ValueError("flat_unit_fraction must be in (0, 1]")
    if not math.isfinite(max_bet_fraction) or not 0.0 < max_bet_fraction <= 1.0:
        raise ValueError("max_bet_fraction must be in (0, 1]")
    if not math.isfinite(max_week_fraction) or not 0.0 < max_week_fraction <= 1.0:
        raise ValueError("max_week_fraction must be in (0, 1]")
    if not math.isfinite(probability_haircut) or not 0.0 <= probability_haircut < 0.5:
        raise ValueError("probability_haircut must be in [0, 0.5)")
    tiers = tuple((float(threshold), float(fraction)) for threshold, fraction in confidence_tiers)
    if not tiers:
        raise ValueError("confidence_tiers cannot be empty")
    thresholds = [threshold for threshold, _ in tiers]
    if any(not math.isfinite(value) or not 0.5 <= value < 1.0 for value in thresholds):
        raise ValueError("confidence tier thresholds must be finite and in [0.5, 1)")
    if any(left <= right for left, right in pairwise(thresholds)):
        raise ValueError("confidence tier thresholds must be strictly descending")
    fractions = [fraction for _, fraction in tiers]
    if any(not math.isfinite(value) or not 0.0 <= value <= max_bet_fraction for value in fractions):
        raise ValueError("confidence tier fractions must be in [0, max_bet_fraction]")
    if any(left < right for left, right in pairwise(fractions)):
        raise ValueError("confidence tier fractions must not increase as confidence falls")
    return tiers


def _prepare_replay(decisions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    decision_columns = {
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "bet_side",
        "bet_odds",
        "home_cover_probability",
    }
    outcome_columns = {"game_id", "home_cover", "ats_margin"}
    missing_decisions = sorted(decision_columns.difference(decisions.columns))
    missing_outcomes = sorted(outcome_columns.difference(outcomes.columns))
    if missing_decisions:
        raise ValueError(f"Decisions are missing policy columns: {', '.join(missing_decisions)}")
    if missing_outcomes:
        raise ValueError(f"Outcomes are missing policy columns: {', '.join(missing_outcomes)}")
    if (
        decisions["game_id"].isna().any()
        or decisions["game_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("decision game_id values must be non-empty")
    if decisions["game_id"].astype(str).duplicated().any():
        raise ValueError("decision game_id values must be unique")
    if outcomes["game_id"].astype(str).duplicated().any():
        raise ValueError("outcome game_id values must be unique")
    decision_ids = set(decisions["game_id"].astype(str))
    outcome_ids = set(outcomes["game_id"].astype(str))
    if decision_ids != outcome_ids:
        raise ValueError("outcome game_id values must exactly match decisions")

    prepared = decisions.copy()
    prepared["game_id"] = prepared["game_id"].astype(str)
    prepared["gameday"] = pd.to_datetime(prepared["gameday"], errors="raise", utc=True)
    if prepared["gameday"].isna().any():
        raise ValueError("gameday must be present for every decision")
    for column in ("season", "week"):
        values = pd.to_numeric(prepared[column], errors="raise")
        if (
            values.isna().any()
            or not np.isfinite(values.to_numpy(dtype=float)).all()
            or not values.mod(1.0).eq(0.0).all()
        ):
            raise ValueError(f"{column} must contain finite integers")
        prepared[column] = values.astype(int)
    for column in ("home_team", "away_team"):
        if prepared[column].isna().any() or prepared[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} values must be non-empty")
    if prepared["home_team"].astype(str).eq(prepared["away_team"].astype(str)).any():
        raise ValueError("home_team and away_team must differ")
    sides = set(prepared["bet_side"].astype(str))
    if not sides.issubset({"HOME", "AWAY", "PASS"}):
        raise ValueError("bet_side must be HOME, AWAY, or PASS")
    active = prepared["bet_side"].ne("PASS")
    probabilities = pd.to_numeric(prepared.loc[active, "home_cover_probability"], errors="raise")
    if probabilities.isna().any() or not probabilities.between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("active home_cover_probability values must be strictly between 0 and 1")
    odds = pd.to_numeric(prepared.loc[active, "bet_odds"], errors="raise")
    if odds.isna().any() or not np.isfinite(odds.to_numpy(dtype=float)).all() or odds.eq(0.0).any():
        raise ValueError("active bet_odds must be finite, non-zero American odds")
    timestamp_columns = {"prediction_timestamp", "kickoff"}.intersection(prepared.columns)
    if len(timestamp_columns) == 1:
        raise ValueError("prediction_timestamp and kickoff must be supplied together")
    if len(timestamp_columns) == 2:
        predicted_at = pd.to_datetime(prepared["prediction_timestamp"], errors="raise", utc=True)
        kickoff = pd.to_datetime(prepared["kickoff"], errors="raise", utc=True)
        if predicted_at.isna().any() or kickoff.isna().any() or predicted_at.ge(kickoff).any():
            raise ValueError("every prediction_timestamp must be strictly before kickoff")

    outcome_frame = outcomes[["game_id", "home_cover", "ats_margin"]].copy()
    outcome_frame["game_id"] = outcome_frame["game_id"].astype(str)
    replay = prepared.merge(outcome_frame, on="game_id", how="left", validate="one_to_one")
    margin = pd.to_numeric(replay["ats_margin"], errors="raise")
    home_cover = pd.to_numeric(replay["home_cover"], errors="raise")
    if not home_cover.dropna().isin([0.0, 1.0]).all():
        raise ValueError("resolved home_cover values must be 0 or 1")
    resolved = margin.notna()
    if not np.isfinite(margin.loc[resolved].to_numpy(dtype=float)).all():
        raise ValueError("resolved ats_margin values must be finite")
    if home_cover.loc[margin.gt(0.0)].ne(1.0).any() or home_cover.loc[margin.lt(0.0)].ne(0.0).any():
        raise ValueError("home_cover must agree with the sign of non-push ats_margin")
    if home_cover.loc[margin.ne(0.0) & resolved].isna().any():
        raise ValueError("non-push outcomes require home_cover")
    replay["home_cover"] = home_cover
    replay["ats_margin"] = margin
    replay["prediction_row"] = np.arange(len(replay), dtype=int)
    return replay.sort_values(["gameday", "game_id"], kind="stable").reset_index(drop=True)


def _side_probabilities(
    frame: pd.DataFrame, probability_haircut: float
) -> tuple[np.ndarray, np.ndarray]:
    home_probability = pd.to_numeric(frame["home_cover_probability"], errors="coerce").to_numpy(
        dtype=float
    )
    raw = np.where(frame["bet_side"].eq("HOME"), home_probability, 1.0 - home_probability)
    conservative = np.clip(raw - probability_haircut, 0.0, 1.0)
    return raw, conservative


def _confidence_fractions(
    conservative_probability: np.ndarray,
    tiers: tuple[tuple[float, float], ...],
) -> np.ndarray:
    fractions = np.zeros(len(conservative_probability), dtype=float)
    for position, probability in enumerate(conservative_probability):
        fractions[position] = next(
            (fraction for threshold, fraction in tiers if probability >= threshold), 0.0
        )
    return fractions


def _scale_fractions(fractions: np.ndarray, max_week_fraction: float) -> np.ndarray:
    total = float(fractions.sum())
    if total <= max_week_fraction or total == 0.0:
        return fractions
    return fractions * (max_week_fraction / total)


def _policy_fractions(
    policy: str,
    week: pd.DataFrame,
    *,
    bankroll: float,
    initial_bankroll: float,
    flat_unit_fraction: float,
    confidence_tiers: tuple[tuple[float, float], ...],
    max_bet_fraction: float,
    max_week_fraction: float,
    probability_haircut: float,
    team_factor_strength: float,
    factor_exposures: pd.DataFrame | None,
    factor_strengths: Mapping[str, float] | None,
    factor_limits: Mapping[str, float] | None,
    covariance: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_probability, conservative_probability = _side_probabilities(week, probability_haircut)
    active = week["bet_side"].ne("PASS").to_numpy()
    fractions = np.zeros(len(week), dtype=float)
    if policy == "flat_unit":
        flat_fraction = min(max_bet_fraction, initial_bankroll * flat_unit_fraction / bankroll)
        fractions[active] = flat_fraction
        fractions = _scale_fractions(fractions, max_week_fraction)
    elif policy == "confidence_tier":
        fractions[active] = np.minimum(
            max_bet_fraction,
            _confidence_fractions(conservative_probability[active], confidence_tiers),
        )
        fractions = _scale_fractions(fractions, max_week_fraction)
    elif policy == "quarter_kelly":
        active_rows = week.loc[active]
        odds = pd.to_numeric(active_rows["bet_odds"], errors="raise").to_numpy(dtype=float)
        fractions[active] = np.minimum(
            max_bet_fraction,
            np.asarray(
                [
                    0.25 * kelly_fraction(float(probability), float(price))
                    for probability, price in zip(
                        conservative_probability[active], odds, strict=True
                    )
                ]
            ),
        )
        fractions = _scale_fractions(fractions, max_week_fraction)
    elif policy == "risk_constrained_kelly":
        active_ids = week.loc[active, "game_id"].astype(str).tolist()
        weekly_exposures = (
            factor_exposures.loc[active_ids] if factor_exposures is not None else None
        )
        sizing = size_correlated_paper_portfolio(
            week,
            covariance=covariance,
            factor_exposures=weekly_exposures,
            factor_strengths=None if covariance is not None else factor_strengths,
            factor_limits=factor_limits,
            team_factor_strength=team_factor_strength,
            kelly_multiplier=0.25,
            max_bet_fraction=max_bet_fraction,
            max_total_fraction=max_week_fraction,
            probability_haircut=probability_haircut,
        )
        fractions = sizing.allocations["stake_fraction"].to_numpy(dtype=float)
    else:  # pragma: no cover - internal invariant
        raise AssertionError(f"Unknown policy: {policy}")
    return fractions, raw_probability, conservative_probability


def _replay_policy(
    policy: str,
    replay: pd.DataFrame,
    *,
    initial_bankroll: float,
    flat_unit_fraction: float,
    confidence_tiers: tuple[tuple[float, float], ...],
    max_bet_fraction: float,
    max_week_fraction: float,
    probability_haircut: float,
    team_factor_strength: float,
    factor_exposures: pd.DataFrame | None,
    factor_strengths: Mapping[str, float] | None,
    factor_limits: Mapping[str, float] | None,
    covariance_by_week: Mapping[tuple[int, int], pd.DataFrame] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bankroll = float(initial_bankroll)
    bankroll_path = [bankroll]
    ledger_parts: list[pd.DataFrame] = []
    wins = losses = pushes = resolved_bets = 0
    for (_, _), week in replay.groupby(["season", "week"], sort=False):
        week = week.copy().reset_index(drop=True)
        season = int(week["season"].iloc[0])
        week_number = int(week["week"].iloc[0])
        covariance = covariance_by_week.get((season, week_number)) if covariance_by_week else None
        fractions, raw_probability, conservative_probability = _policy_fractions(
            policy,
            week,
            bankroll=bankroll,
            initial_bankroll=initial_bankroll,
            flat_unit_fraction=flat_unit_fraction,
            confidence_tiers=confidence_tiers,
            max_bet_fraction=max_bet_fraction,
            max_week_fraction=max_week_fraction,
            probability_haircut=probability_haircut,
            team_factor_strength=team_factor_strength,
            factor_exposures=factor_exposures,
            factor_strengths=factor_strengths,
            factor_limits=factor_limits,
            covariance=covariance,
        )
        stakes = bankroll * fractions
        profits = np.zeros(len(week), dtype=float)
        results: list[str] = []
        sides = week["bet_side"].astype(str).to_numpy()
        ats_margins = week["ats_margin"].to_numpy(dtype=float)
        home_covers = week["home_cover"].to_numpy(dtype=float)
        bet_odds = pd.to_numeric(week["bet_odds"], errors="coerce").to_numpy(dtype=float)
        for position, side in enumerate(sides):
            if side == "PASS" or stakes[position] == 0.0:
                results.append("PASS" if side == "PASS" else "NO_STAKE")
                continue
            if not math.isfinite(ats_margins[position]):
                profits[position] = np.nan
                results.append("UNRESOLVED")
                continue
            resolved_bets += 1
            if ats_margins[position] == 0.0:
                pushes += 1
                results.append("PUSH")
                continue
            profit = stakes[position] * settle_bet(
                str(side), home_covers[position], bet_odds[position]
            )
            profits[position] = profit
            if profit > 0.0:
                wins += 1
                results.append("WIN")
            else:
                losses += 1
                results.append("LOSS")
        before_week = bankroll
        bankroll += float(np.nansum(profits))
        bankroll_path.append(bankroll)
        week["policy"] = policy
        week["side_probability"] = raw_probability
        week["sizing_probability"] = conservative_probability
        week["stake_fraction"] = fractions
        week["stake"] = stakes
        week["profit"] = profits
        week["paper_result"] = results
        week["bankroll_before_week"] = before_week
        week["bankroll_after_week"] = bankroll
        ledger_parts.append(week)

    ledger = pd.concat(ledger_parts, ignore_index=True) if ledger_parts else replay.copy()
    path = pd.Series(bankroll_path, dtype=float)
    drawdown = path / path.cummax() - 1.0
    total_staked = float(ledger["stake"].sum()) if "stake" in ledger else 0.0
    metrics = {
        "policy": policy,
        "paper_only": True,
        "initial_bankroll": initial_bankroll,
        "final_bankroll": bankroll,
        "return": bankroll / initial_bankroll - 1.0,
        "net_profit": bankroll - initial_bankroll,
        "peak_bankroll": float(path.max()),
        "trough_bankroll": float(path.min()),
        "max_drawdown": float(drawdown.min()),
        "total_staked": total_staked,
        "turnover": total_staked / initial_bankroll,
        "resolved_bets": resolved_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "weeks": max(0, len(bankroll_path) - 1),
    }
    return ledger, metrics


def compare_paper_policies(
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    initial_bankroll: float = 100.0,
    flat_unit_fraction: float = 0.01,
    confidence_tiers: Sequence[tuple[float, float]] = DEFAULT_CONFIDENCE_TIERS,
    max_bet_fraction: float = 0.02,
    max_week_fraction: float = 0.10,
    probability_haircut: float = 0.0,
    team_factor_strength: float = 0.10,
    factor_exposures: pd.DataFrame | None = None,
    factor_strengths: Mapping[str, float] | None = None,
    factor_limits: Mapping[str, float] | None = None,
    covariance_by_week: Mapping[tuple[int, int], pd.DataFrame] | None = None,
) -> PolicyComparison:
    """Replay four sizing policies on identical chronological paper decisions.

    The function performs no model selection and does not declare a winner. It
    preserves one output row per input prediction per policy, including PASS and
    unresolved rows, and only settles a week after all simultaneous stakes have
    been fixed from that week's starting hypothetical bankroll.
    """

    tiers = _validate_configuration(
        initial_bankroll=initial_bankroll,
        flat_unit_fraction=flat_unit_fraction,
        confidence_tiers=confidence_tiers,
        max_bet_fraction=max_bet_fraction,
        max_week_fraction=max_week_fraction,
        probability_haircut=probability_haircut,
    )
    replay = _prepare_replay(decisions, outcomes)
    active_ids = replay.loc[replay["bet_side"].ne("PASS"), "game_id"].astype(str).tolist()
    prepared_exposures: pd.DataFrame | None = None
    if factor_exposures is not None:
        prepared_exposures = factor_exposures.copy()
        prepared_exposures.index = prepared_exposures.index.astype(str)
        if prepared_exposures.index.has_duplicates or set(prepared_exposures.index) != set(
            active_ids
        ):
            raise ValueError(
                "factor exposure index must exactly match active decision game_id values"
            )
        prepared_exposures = prepared_exposures.loc[active_ids]

    active_weeks = {
        (int(str(season)), int(str(week)))
        for (season, week), group in replay.groupby(["season", "week"], sort=False)
        if group["bet_side"].ne("PASS").any()
    }
    if covariance_by_week is not None and set(covariance_by_week) != active_weeks:
        raise ValueError("covariance_by_week must name every active season/week exactly")
    if not math.isfinite(team_factor_strength) or team_factor_strength < 0.0:
        raise ValueError("team_factor_strength must be finite and non-negative")

    ledgers: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        policy_ledger, policy_metrics = _replay_policy(
            policy,
            replay,
            initial_bankroll=initial_bankroll,
            flat_unit_fraction=flat_unit_fraction,
            confidence_tiers=tiers,
            max_bet_fraction=max_bet_fraction,
            max_week_fraction=max_week_fraction,
            probability_haircut=probability_haircut,
            team_factor_strength=team_factor_strength,
            factor_exposures=prepared_exposures,
            factor_strengths=factor_strengths,
            factor_limits=factor_limits,
            covariance_by_week=covariance_by_week,
        )
        ledgers.append(policy_ledger)
        metrics.append(policy_metrics)
    ledger = pd.concat(ledgers, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    configuration = {
        "paper_only": True,
        "policies": list(POLICY_ORDER),
        "initial_bankroll": initial_bankroll,
        "flat_unit_fraction": flat_unit_fraction,
        "confidence_tiers": [list(tier) for tier in tiers],
        "max_bet_fraction": max_bet_fraction,
        "max_week_fraction": max_week_fraction,
        "probability_haircut": probability_haircut,
        "team_factor_strength": team_factor_strength,
        "factor_limits": dict(factor_limits or {}),
        "covariance_source": "explicit_by_week"
        if covariance_by_week is not None
        else "factor_scenario",
    }
    return PolicyComparison(ledger=ledger, metrics=metric_frame, configuration=configuration)
