"""Paper-only bankroll sizing and settlement analytics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.odds import profit_per_unit, settle_bet
from nfl_ats.probability_uncertainty import (
    AUDIT_COLUMNS,
    conservative_probability_audit,
)


@dataclass(frozen=True)
class PortfolioResult:
    ledger: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(frozen=True)
class BankrollSimulation:
    paths: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(frozen=True)
class CorrelatedPortfolioSizing:
    """A deterministic, paper-only allocation and its risk inputs."""

    allocations: pd.DataFrame
    covariance: pd.DataFrame
    factor_exposures: pd.DataFrame
    metrics: dict[str, Any]


def kelly_fraction(win_probability: float, american_odds: float | int | None) -> float:
    """Return the full-Kelly bankroll fraction for a two-outcome wager."""

    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be between 0 and 1")
    net_profit = profit_per_unit(american_odds)
    fraction = (net_profit * win_probability - (1.0 - win_probability)) / net_profit
    return max(0.0, float(fraction))


def _validate_labeled_covariance(
    covariance: pd.DataFrame,
    candidate_ids: list[str],
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Validate and order a covariance matrix without positional guesswork."""

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a labeled pandas DataFrame")
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise ValueError("covariance row and column labels must be unique")
    row_labels = [str(value) for value in covariance.index]
    column_labels = [str(value) for value in covariance.columns]
    expected = set(candidate_ids)
    if set(row_labels) != expected or set(column_labels) != expected:
        raise ValueError("covariance labels must exactly match active candidate game_id values")
    labeled = covariance.copy()
    labeled.index = row_labels
    labeled.columns = column_labels
    matrix = labeled.loc[candidate_ids, candidate_ids].to_numpy(dtype=float)
    if matrix.shape != (len(candidate_ids), len(candidate_ids)):
        raise ValueError("covariance must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        raise ValueError("covariance must be symmetric")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        raise ValueError("covariance diagonal must be strictly positive")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    if float(eigenvalues.min(initial=0.0)) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    return matrix


def _team_factor_exposures(active: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(set(active["home_team"].astype(str)) | set(active["away_team"].astype(str)))
    exposures = pd.DataFrame(
        0.0,
        index=active["game_id"].astype(str),
        columns=[f"team:{team}" for team in teams],
    )
    for row in active.itertuples(index=False):
        candidate_id = str(row.game_id)
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        if row.bet_side == "HOME":
            exposures.at[candidate_id, f"team:{home_team}"] = 1.0
            exposures.at[candidate_id, f"team:{away_team}"] = -1.0
        else:
            exposures.at[candidate_id, f"team:{home_team}"] = -1.0
            exposures.at[candidate_id, f"team:{away_team}"] = 1.0
    return exposures


def _validate_optional_exposures(
    factor_exposures: pd.DataFrame | None,
    candidate_ids: list[str],
) -> pd.DataFrame:
    if factor_exposures is None:
        return pd.DataFrame(index=candidate_ids, dtype=float)
    if not isinstance(factor_exposures, pd.DataFrame):
        raise TypeError("factor_exposures must be a labeled pandas DataFrame")
    if factor_exposures.index.has_duplicates or factor_exposures.columns.has_duplicates:
        raise ValueError("factor exposure row and column labels must be unique")
    labels = [str(value) for value in factor_exposures.index]
    if set(labels) != set(candidate_ids):
        raise ValueError("factor exposure index must exactly match active candidate game_id values")
    result = factor_exposures.copy()
    result.index = labels
    result.columns = [str(value) for value in result.columns]
    if any(column.startswith("team:") for column in result.columns):
        raise ValueError("optional factor names cannot use the reserved 'team:' prefix")
    allowed_prefixes = ("total:", "weather:", "market:")
    invalid = [column for column in result.columns if not column.startswith(allowed_prefixes)]
    if invalid:
        raise ValueError(
            "optional factor names must start with total:, weather:, or market:: "
            + ", ".join(invalid)
        )
    result = result.loc[candidate_ids].astype(float)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("factor exposures must contain only finite values")
    return result


def _factor_covariance(
    payoff_variance: np.ndarray,
    exposures: pd.DataFrame,
    strengths: Mapping[str, float],
) -> np.ndarray:
    columns = list(exposures.columns)
    if set(strengths) != set(columns):
        missing = sorted(set(columns).difference(strengths))
        extra = sorted(set(strengths).difference(columns))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unknown " + ", ".join(extra))
        raise ValueError(
            "factor_strengths must name every factor exactly (" + "; ".join(detail) + ")"
        )
    weights = np.asarray([float(strengths[column]) for column in columns], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("factor strengths must be finite and non-negative")
    loadings = exposures.to_numpy(dtype=float)
    raw_correlation = np.eye(len(exposures), dtype=float)
    if columns:
        raw_correlation += (loadings * weights) @ loadings.T
    normalizer = np.sqrt(np.diag(raw_correlation))
    correlation = raw_correlation / np.outer(normalizer, normalizer)
    standard_deviation = np.sqrt(payoff_variance)
    return cast(np.ndarray, correlation * np.outer(standard_deviation, standard_deviation))


def _project_feasible(
    point: np.ndarray,
    upper_bounds: np.ndarray,
    halfspaces: list[tuple[np.ndarray, float]],
    *,
    tolerance: float,
    max_iterations: int = 5_000,
) -> np.ndarray:
    """Project onto box/halfspace intersections with deterministic Dykstra iterations."""

    projections = 1 + len(halfspaces)
    corrections = [np.zeros_like(point) for _ in range(projections)]
    current = point.copy()
    for _ in range(max_iterations):
        before = current.copy()
        shifted = current + corrections[0]
        projected = np.clip(shifted, 0.0, upper_bounds)
        corrections[0] = shifted - projected
        current = projected
        for position, (normal, limit) in enumerate(halfspaces, start=1):
            shifted = current + corrections[position]
            excess = float(normal @ shifted - limit)
            denominator = float(normal @ normal)
            projected = (
                shifted - (excess / denominator) * normal
                if excess > 0.0 and denominator > 0.0
                else shifted
            )
            corrections[position] = shifted - projected
            current = projected
        if float(np.max(np.abs(current - before), initial=0.0)) <= tolerance:
            return cast(np.ndarray, current)
    raise RuntimeError("portfolio constraint projection did not converge")


def _quadratic_kelly_allocation(
    expected_return: np.ndarray,
    covariance: np.ndarray,
    upper_bounds: np.ndarray,
    exposures: pd.DataFrame,
    factor_limits: Mapping[str, float],
    *,
    max_total_fraction: float,
    kelly_multiplier: float,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    halfspaces: list[tuple[np.ndarray, float]] = [
        (np.ones_like(expected_return), max_total_fraction)
    ]
    for factor, raw_limit in factor_limits.items():
        if factor not in exposures.columns:
            raise ValueError(f"factor limit names unknown factor: {factor}")
        limit = float(raw_limit)
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError("factor exposure limits must be finite and non-negative")
        normal = exposures[factor].to_numpy(dtype=float)
        if np.any(normal):
            halfspaces.extend(((normal, limit), (-normal, limit)))
    if kelly_multiplier == 0.0 or not np.any(upper_bounds > 0.0):
        return np.zeros_like(expected_return), 0

    risk_aversion = 1.0 / kelly_multiplier
    hessian = risk_aversion * covariance
    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    if not math.isfinite(lipschitz) or lipschitz <= 0.0:
        raise ValueError("covariance must imply positive portfolio variance")
    step = 1.0 / lipschitz
    allocation = np.zeros_like(expected_return)
    for iteration in range(1, 20_001):
        gradient = hessian @ allocation - expected_return
        candidate = _project_feasible(
            allocation - step * gradient,
            upper_bounds,
            halfspaces,
            tolerance=tolerance / 10.0,
        )
        if float(np.max(np.abs(candidate - allocation), initial=0.0)) <= tolerance:
            return candidate, iteration
        allocation = candidate
    raise RuntimeError("correlated portfolio optimizer did not converge")


def size_correlated_paper_portfolio(
    candidates: pd.DataFrame,
    *,
    covariance: pd.DataFrame | None = None,
    factor_exposures: pd.DataFrame | None = None,
    factor_strengths: Mapping[str, float] | None = None,
    factor_limits: Mapping[str, float] | None = None,
    team_factor_strength: float = 0.10,
    kelly_multiplier: float = 0.25,
    max_bet_fraction: float = 0.02,
    max_total_fraction: float = 0.10,
    probability_haircut: float = 0.0,
    probability_uncertainty: pd.DataFrame | None = None,
    posterior_z: float = 1.645,
    tolerance: float = 1e-10,
) -> CorrelatedPortfolioSizing:
    """Allocate one simultaneous slate with a correlated fractional-Kelly approximation.

    This function only returns paper fractions: it performs no settlement, I/O,
    or external action. Shared-team exposures are derived from HOME/AWAY sides.
    Optional ``total:``, ``weather:``, and ``market:`` exposure columns let a
    caller make other common risk channels explicit. The quadratic objective is
    deterministic and subject to per-candidate, aggregate, and optional absolute
    factor-exposure limits.
    """

    if not 0.0 <= kelly_multiplier <= 1.0:
        raise ValueError("kelly_multiplier must be between 0 and 1")
    if not 0.0 < max_bet_fraction <= 1.0:
        raise ValueError("max_bet_fraction must be in (0, 1]")
    if not 0.0 < max_total_fraction <= 1.0:
        raise ValueError("max_total_fraction must be in (0, 1]")
    if not 0.0 <= probability_haircut < 0.5:
        raise ValueError("probability_haircut must be in [0, 0.5)")
    if not math.isfinite(team_factor_strength) or team_factor_strength < 0.0:
        raise ValueError("team_factor_strength must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    required = {
        "game_id",
        "bet_side",
        "bet_odds",
        "home_cover_probability",
        "home_team",
        "away_team",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(
            f"Candidates are missing correlated portfolio columns: {', '.join(missing)}"
        )
    if candidates["game_id"].astype(str).duplicated().any():
        raise ValueError("candidate game_id values must be unique")
    if (
        candidates["game_id"].isna().any()
        or candidates["game_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("candidate game_id values must be non-empty")
    sides = set(candidates["bet_side"].astype(str))
    invalid_sides = sorted(sides.difference({"HOME", "AWAY", "PASS"}))
    if invalid_sides:
        raise ValueError("bet_side must be HOME, AWAY, or PASS")
    for column in ("home_team", "away_team"):
        if (
            candidates[column].isna().any()
            or candidates[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{column} values must be non-empty")
    if candidates["home_team"].astype(str).eq(candidates["away_team"].astype(str)).any():
        raise ValueError("home_team and away_team must differ")

    allocations = candidates.copy().reset_index(drop=True)
    probability_audit = conservative_probability_audit(
        allocations,
        probability_haircut=probability_haircut,
        probability_uncertainty=probability_uncertainty,
        posterior_z=posterior_z,
    )
    for column in AUDIT_COLUMNS:
        allocations[column] = probability_audit[column]
    allocations["candidate_expected_return"] = 0.0
    allocations["independent_kelly_fraction"] = 0.0
    allocations["stake_fraction"] = 0.0
    active = allocations.loc[allocations["bet_side"].ne("PASS")].copy()
    candidate_ids = active["game_id"].astype(str).tolist()
    if not candidate_ids:
        empty = pd.DataFrame(index=pd.Index([], name="game_id"), dtype=float)
        return CorrelatedPortfolioSizing(
            allocations=allocations,
            covariance=empty,
            factor_exposures=empty,
            metrics={
                "paper_only": True,
                "method": "quadratic_fractional_kelly",
                "active_candidates": 0,
                "total_stake_fraction": 0.0,
                "expected_profit_fraction": 0.0,
                "portfolio_variance": 0.0,
                "probability_uncertainty_methods": [],
                "posterior_z": posterior_z,
                "iterations": 0,
            },
        )

    raw_home_probability = pd.to_numeric(active["home_cover_probability"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(raw_home_probability).all() or np.any(
        (raw_home_probability <= 0.0) | (raw_home_probability >= 1.0)
    ):
        raise ValueError("home_cover_probability must be finite and strictly between 0 and 1")
    probability = active["conservative_bet_probability"].to_numpy(dtype=float)
    odds = pd.to_numeric(active["bet_odds"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(odds).all() or np.any(odds == 0.0):
        raise ValueError("bet_odds must be finite, non-zero American odds")
    win_return = np.asarray([profit_per_unit(value) for value in odds], dtype=float)
    expected_return = probability * win_return - (1.0 - probability)
    payoff_variance = (
        probability * (win_return - expected_return) ** 2
        + (1.0 - probability) * (-1.0 - expected_return) ** 2
    )
    independent_kelly = np.asarray(
        [kelly_fraction(float(p), float(price)) for p, price in zip(probability, odds, strict=True)]
    )
    upper_bounds = np.minimum(max_bet_fraction, kelly_multiplier * independent_kelly)

    team_exposures = _team_factor_exposures(active)
    optional_exposures = _validate_optional_exposures(factor_exposures, candidate_ids)
    exposures = pd.concat([team_exposures, optional_exposures], axis="columns")
    if covariance is None:
        optional_strengths = dict(factor_strengths or {})
        expected_optional = set(optional_exposures.columns)
        if set(optional_strengths) != expected_optional:
            missing_strengths = sorted(expected_optional.difference(optional_strengths))
            extra_strengths = sorted(set(optional_strengths).difference(expected_optional))
            detail = []
            if missing_strengths:
                detail.append("missing " + ", ".join(missing_strengths))
            if extra_strengths:
                detail.append("unknown " + ", ".join(extra_strengths))
            raise ValueError(
                "factor_strengths must name each optional factor exactly ("
                + "; ".join(detail)
                + ")"
            )
        strengths = dict.fromkeys(team_exposures.columns, team_factor_strength)
        strengths.update(optional_strengths)
        covariance_matrix = _factor_covariance(payoff_variance, exposures, strengths)
    else:
        if factor_strengths:
            raise ValueError("factor_strengths cannot be combined with an explicit covariance")
        covariance_matrix = _validate_labeled_covariance(covariance, candidate_ids)

    limits = dict(factor_limits or {})
    fractions, iterations = _quadratic_kelly_allocation(
        expected_return,
        covariance_matrix,
        upper_bounds,
        exposures,
        limits,
        max_total_fraction=max_total_fraction,
        kelly_multiplier=kelly_multiplier,
        tolerance=tolerance,
    )
    active_positions = active.index.to_numpy()
    allocations.loc[active_positions, "candidate_expected_return"] = expected_return
    allocations.loc[active_positions, "independent_kelly_fraction"] = independent_kelly
    allocations.loc[active_positions, "stake_fraction"] = fractions
    covariance_frame = pd.DataFrame(
        covariance_matrix,
        index=pd.Index(candidate_ids, name="game_id"),
        columns=candidate_ids,
    )
    portfolio_variance = float(fractions @ covariance_matrix @ fractions)
    realized_factor_exposure = {
        column: float(exposures[column].to_numpy(dtype=float) @ fractions)
        for column in exposures.columns
        if np.any(exposures[column].to_numpy(dtype=float))
    }
    if float(fractions.sum()) > max_total_fraction + tolerance * 10.0:
        raise RuntimeError("correlated portfolio optimizer violated the aggregate exposure cap")
    for factor, limit in limits.items():
        if abs(realized_factor_exposure.get(factor, 0.0)) > float(limit) + tolerance * 10.0:
            raise RuntimeError(f"correlated portfolio optimizer violated factor cap: {factor}")
    metrics = {
        "paper_only": True,
        "method": "quadratic_fractional_kelly",
        "active_candidates": len(candidate_ids),
        "total_stake_fraction": float(fractions.sum()),
        "expected_profit_fraction": float(expected_return @ fractions),
        "portfolio_variance": portfolio_variance,
        "portfolio_standard_deviation": math.sqrt(max(0.0, portfolio_variance)),
        "quadratic_objective": float(
            expected_return @ fractions - 0.5 * portfolio_variance / kelly_multiplier
            if kelly_multiplier > 0.0
            else 0.0
        ),
        "kelly_multiplier": kelly_multiplier,
        "max_bet_fraction": max_bet_fraction,
        "max_total_fraction": max_total_fraction,
        "probability_haircut": probability_haircut,
        "probability_uncertainty_methods": sorted(
            active["probability_uncertainty_method"].astype(str).unique().tolist()
        ),
        "posterior_z": posterior_z,
        "team_factor_strength": team_factor_strength,
        "factor_limits": limits,
        "realized_factor_exposure": realized_factor_exposure,
        "covariance_source": "explicit" if covariance is not None else "factor_scenario",
        "iterations": iterations,
    }
    return CorrelatedPortfolioSizing(
        allocations=allocations,
        covariance=covariance_frame,
        factor_exposures=exposures,
        metrics=metrics,
    )


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
    probability_uncertainty: pd.DataFrame | None = None,
    posterior_z: float = 1.645,
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
    probability_audit = conservative_probability_audit(
        ledger,
        probability_haircut=probability_haircut,
        probability_uncertainty=probability_uncertainty,
        posterior_z=posterior_z,
    )
    for column in AUDIT_COLUMNS:
        ledger[column] = probability_audit[column]
    ledger["bet_probability"] = ledger["conservative_bet_probability"]
    for column in (
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
            conservative_probability = float(
                cast(Any, ledger.at[index, "conservative_bet_probability"])
            )
            full_fraction = kelly_fraction(conservative_probability, float(row["bet_odds"]))
            desired_fraction = min(max_bet_fraction, kelly_multiplier * full_fraction)
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
        "probability_uncertainty_methods": sorted(
            ledger.loc[ledger["bet_side"].ne("PASS"), "probability_uncertainty_method"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "posterior_z": posterior_z,
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
    probability_uncertainty: pd.DataFrame | None = None,
    posterior_z: float = 1.645,
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

    working = predictions.reset_index(drop=True)
    probability_audit = conservative_probability_audit(
        working,
        probability_haircut=probability_haircut,
        probability_uncertainty=probability_uncertainty,
        posterior_z=posterior_z,
    )
    generator = np.random.default_rng(seed)
    bankroll = np.full(paths, initial_bankroll, dtype=float)
    peak = bankroll.copy()
    maximum_drawdown = np.zeros(paths, dtype=float)
    weekly_columns: dict[str, np.ndarray] = {"start": bankroll.copy()}
    weeks_simulated = 0
    bets_simulated = 0
    for (season, week), group in working.groupby(["season", "week"], sort=True):
        bets = group.loc[group["bet_side"].ne("PASS")].copy()
        if bets.empty:
            continue
        probabilities: list[float] = []
        returns: list[float] = []
        fractions: list[float] = []
        for index, row in bets.iterrows():
            conservative = float(
                cast(Any, probability_audit.at[index, "conservative_bet_probability"])
            )
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
        "probability_uncertainty_methods": sorted(
            probability_audit.loc[working["bet_side"].ne("PASS"), "probability_uncertainty_method"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "posterior_z": posterior_z,
        "conditional_on_model_probabilities": True,
    }
    return BankrollSimulation(paths=path_frame, metrics=metrics)
