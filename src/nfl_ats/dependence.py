"""Diagnostics for serial dependence in out-of-time forecast errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.data import require_columns


@dataclass(frozen=True)
class DependenceAudit:
    summary: dict[str, Any]
    team_summary: pd.DataFrame
    team_residuals: pd.DataFrame


def team_residual_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the model's cover-probability error from each team's perspective."""

    required = (
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_cover",
        "home_cover_probability",
    )
    require_columns(predictions, required, "prediction dependence input")
    games = predictions.loc[
        predictions["home_cover"].notna() & predictions["home_cover_probability"].notna()
    ].copy()
    if games.empty:
        raise ValueError("No evaluated predictions are available for a dependence audit")
    games["gameday"] = pd.to_datetime(games["gameday"], errors="raise")
    actual = pd.to_numeric(games["home_cover"], errors="raise")
    probability = pd.to_numeric(games["home_cover_probability"], errors="raise")
    home_error = actual - probability
    shared = ["game_id", "season", "week", "gameday"]
    home = games[shared].copy()
    home["team"] = games["home_team"].astype("string")
    home["side"] = "HOME"
    home["probability_error"] = home_error
    away = games[shared].copy()
    away["team"] = games["away_team"].astype("string")
    away["side"] = "AWAY"
    away["probability_error"] = -home_error
    panel = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "gameday", "game_id"], ignore_index=True
    )
    panel["previous_error"] = panel.groupby("team", sort=False)["probability_error"].shift()
    panel["previous_gameday"] = panel.groupby("team", sort=False)["gameday"].shift()
    panel["days_since_previous"] = (panel["gameday"] - panel["previous_gameday"]).dt.days
    return panel


def _correlation(frame: pd.DataFrame) -> float:
    pairs = frame.loc[frame["previous_error"].notna()]
    if len(pairs) < 3:
        return float("nan")
    current = pairs["probability_error"].to_numpy(dtype=float)
    previous = pairs["previous_error"].to_numpy(dtype=float)
    if np.std(current) == 0.0 or np.std(previous) == 0.0:
        return float("nan")
    return float(np.corrcoef(current, previous)[0, 1])


def _team_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team, group in panel.groupby("team", sort=True):
        pairs = group.loc[group["previous_error"].notna()]
        rows.append(
            {
                "team": str(team),
                "games": len(group),
                "lag_pairs": len(pairs),
                "mean_probability_error": float(group["probability_error"].mean()),
                "lag1_error_correlation": _correlation(group),
            }
        )
    return pd.DataFrame(rows)


def _permuted_correlation(predictions: pd.DataFrame, generator: np.random.Generator) -> float:
    shuffled = predictions.copy()
    residual = pd.to_numeric(shuffled["home_cover"], errors="raise") - pd.to_numeric(
        shuffled["home_cover_probability"], errors="raise"
    )
    shuffled_residual = residual.copy()
    for indices in shuffled.groupby("season", sort=False).indices.values():
        positions = np.asarray(indices, dtype=int)
        shuffled_residual.iloc[positions] = generator.permutation(
            residual.iloc[positions].to_numpy(dtype=float)
        )
    shuffled["home_cover"] = (
        pd.to_numeric(shuffled["home_cover_probability"], errors="raise") + shuffled_residual
    )
    return _correlation(team_residual_panel(shuffled))


def prediction_dependence_audit(
    predictions: pd.DataFrame,
    *,
    permutations: int = 1_000,
    confidence: float = 0.95,
    seed: int = 20260812,
) -> DependenceAudit:
    """Test lag-one team error correlation against season-preserving shuffles."""

    if permutations < 100:
        raise ValueError("permutations must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    required = ("season", "home_cover", "home_cover_probability")
    require_columns(predictions, required, "prediction dependence input")
    evaluated = predictions.loc[
        predictions["home_cover"].notna() & predictions["home_cover_probability"].notna()
    ].reset_index(drop=True)
    panel = team_residual_panel(evaluated)
    team_summary = _team_summary(panel)
    observed = _correlation(panel)
    generator = np.random.default_rng(seed)
    null = np.asarray(
        [_permuted_correlation(evaluated, generator) for _ in range(permutations)],
        dtype=float,
    )
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or not len(null):
        p_value = float("nan")
        lower = float("nan")
        upper = float("nan")
    else:
        p_value = float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))
        tail = (1.0 - confidence) / 2.0
        lower, upper = (float(value) for value in np.quantile(null, [tail, 1.0 - tail]))
    finite_team = team_summary["lag1_error_correlation"].dropna()
    summary = {
        "games": len(evaluated),
        "team_game_rows": len(panel),
        "teams": int(panel["team"].nunique()),
        "team_lag_pairs": int(panel["previous_error"].notna().sum()),
        "pooled_team_lag1_error_correlation": observed,
        "median_team_lag1_error_correlation": (
            float(finite_team.median()) if not finite_team.empty else None
        ),
        "positive_team_correlations": int(finite_team.gt(0.0).sum()),
        "negative_team_correlations": int(finite_team.lt(0.0).sum()),
        "season_shuffle_null_lower": lower,
        "season_shuffle_null_upper": upper,
        "season_shuffle_two_sided_p_value": p_value,
        "confidence": confidence,
        "permutations": len(null),
        "seed": seed,
        "interpretation": (
            "Dependence affects uncertainty estimates, not the validity of chronological "
            "validation and test partitions."
        ),
    }
    return DependenceAudit(summary=summary, team_summary=team_summary, team_residuals=panel)
