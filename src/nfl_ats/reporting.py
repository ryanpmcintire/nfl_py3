"""Read-only artifact discovery and dashboard-ready summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from nfl_ats.estimation_variance import MIN_BLOCKS_FOR_INTERVAL, OnDegenerate, guard_block_count
from nfl_ats.odds import settle_bet

BootstrapBlock = Literal["week", "season"]


def _expected_calibration_error(
    actual: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probability, boundaries[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += selected.mean() * abs(actual[selected].mean() - probability[selected].mean())
    return float(error)


def _realized_profit(row: pd.Series) -> float:
    if row["bet_side"] == "PASS":
        return 0.0
    if pd.isna(row["home_cover"]):
        return 0.0
    return settle_bet(str(row["bet_side"]), float(row["home_cover"]), float(row["bet_odds"]))


def _evaluation_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    evaluated = predictions.loc[predictions["home_cover"].notna()].copy()
    if evaluated.empty:
        raise ValueError("Predictions contain no completed, non-push games")
    actual = evaluated["home_cover"].to_numpy(dtype=int)
    probability = evaluated["home_cover_probability"].to_numpy(dtype=float)
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    classification = probability >= 0.5
    metrics = {
        "accuracy": float(np.mean(classification == actual)),
        "brier_score": float(np.mean(np.square(probability - actual))),
        "log_loss": float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))),
        "expected_calibration_error": _expected_calibration_error(actual, probability),
    }
    if {"bet_side", "bet_odds"}.issubset(predictions.columns):
        wagered = predictions.loc[predictions["bet_side"].ne("PASS")].copy()
        if wagered.empty:
            metrics["roi"] = 0.0
        else:
            metrics["roi"] = float(wagered.apply(_realized_profit, axis=1).mean())
    return metrics


def artifact_directories(root: Path, required_file: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and (path / required_file).is_file()),
        reverse=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    evaluated = predictions.loc[predictions["home_cover"].notna()].copy()
    if evaluated.empty:
        return pd.DataFrame(
            columns=["bin", "games", "mean_probability", "observed_cover_rate", "gap"]
        )
    probability = evaluated["home_cover_probability"].astype(float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    evaluated["bin"] = pd.cut(
        probability,
        boundaries.tolist(),
        include_lowest=True,
        duplicates="drop",
    )
    table = (
        evaluated.groupby("bin", observed=True)
        .agg(
            games=("home_cover", "size"),
            mean_probability=("home_cover_probability", "mean"),
            observed_cover_rate=("home_cover", "mean"),
        )
        .reset_index()
    )
    table["bin"] = table["bin"].astype(str)
    table["gap"] = table["observed_cover_rate"] - table["mean_probability"]
    return table


def season_scorecard(predictions: pd.DataFrame) -> pd.DataFrame:
    evaluated = predictions.loc[predictions["home_cover"].notna()].copy()
    if evaluated.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "games",
                "accuracy",
                "brier_score",
                "log_loss",
                "expected_calibration_error",
            ]
        )
    evaluated["correct"] = (
        (evaluated["home_cover_probability"].ge(0.5)).astype(int)
        == evaluated["home_cover"].astype(int)
    ).astype(int)
    evaluated["brier"] = (evaluated["home_cover_probability"] - evaluated["home_cover"]) ** 2
    probability = evaluated["home_cover_probability"].clip(1e-15, 1.0 - 1e-15)
    evaluated["log_loss_row"] = -(
        evaluated["home_cover"] * np.log(probability)
        + (1.0 - evaluated["home_cover"]) * np.log(1.0 - probability)
    )
    scorecard = (
        evaluated.groupby("season", as_index=False)
        .agg(
            games=("game_id", "size"),
            accuracy=("correct", "mean"),
            brier_score=("brier", "mean"),
            log_loss=("log_loss_row", "mean"),
        )
        .sort_values("season")
    )
    calibration = {
        season: _expected_calibration_error(
            group["home_cover"].to_numpy(dtype=int),
            group["home_cover_probability"].to_numpy(dtype=float),
        )
        for season, group in evaluated.groupby("season", sort=False)
    }
    scorecard["expected_calibration_error"] = scorecard["season"].map(calibration)
    if {"bet_side", "bet_odds"}.issubset(predictions.columns):
        wagered = predictions.loc[predictions["bet_side"].ne("PASS")].copy()
        wagered["profit_units"] = wagered.apply(_realized_profit, axis=1)
        wagered["win"] = wagered["profit_units"].gt(0).astype(int)
        wagered["loss"] = wagered["profit_units"].lt(0).astype(int)
        wagered["bet_push"] = wagered["profit_units"].eq(0).astype(int)
        betting = wagered.groupby("season").agg(
            bets=("game_id", "size"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            bet_pushes=("bet_push", "sum"),
            net_profit_units=("profit_units", "sum"),
        )
        for column in ("bets", "wins", "losses", "bet_pushes", "net_profit_units"):
            scorecard[column] = scorecard["season"].map(betting[column]).fillna(0)
        scorecard[["bets", "wins", "losses", "bet_pushes"]] = scorecard[
            ["bets", "wins", "losses", "bet_pushes"]
        ].astype(int)
        scorecard["bet_coverage"] = scorecard["bets"] / scorecard["games"]
        scorecard["roi"] = np.where(
            scorecard["bets"].gt(0), scorecard["net_profit_units"] / scorecard["bets"], 0.0
        )
    return scorecard


def block_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    # See paired_feature_comparisons: 2,000 leaves ~6-7% of the reported SE as
    # the bootstrap's own noise, which is enough to flip a threshold-adjacent
    # verdict between seeds. 20,000 is ~5x quieter and effectively free.
    samples: int = 20_000,
    confidence: float = 0.95,
    block: BootstrapBlock = "week",
    seed: int = 20260812,
    # D4 guard. Default 'warn' + a flagged output column, never 'raise':
    # refusing would change what existing call sites return, and the point is
    # that the flag TRAVELS with the number into the CSV a registry entry cites.
    # A caller that is about to record a verdict should pass 'raise'.
    on_degenerate: OnDegenerate = "warn",
    min_blocks: int = MIN_BLOCKS_FOR_INTERVAL,
) -> pd.DataFrame:
    """Estimate metric uncertainty by resampling whole NFL weeks or seasons.

    Games within a block stay together, preserving much more of the schedule
    dependence than an ordinary row bootstrap. The interval is descriptive of
    this historical sample; it is not a guarantee about a future season.

    Every row carries ``blocks`` and ``degenerate_blocks``. Below the measured
    floor (``estimation_variance.MIN_BLOCKS_FOR_INTERVAL``) the percentile
    bootstrap's coverage is nowhere near nominal, so ``lower``/``upper`` on a
    flagged row are not a 95% interval and must not be read as one; report
    ``estimate`` instead. See ``experiments.paired_feature_comparisons`` for
    the same guard on paired deltas.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    required = {"season", "home_cover", "home_cover_probability"}
    if block == "week":
        required.add("week")
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing bootstrap columns: {', '.join(missing)}")
    if predictions.loc[predictions["home_cover"].notna()].empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "estimate",
                "lower",
                "upper",
                "confidence",
                "block",
                "samples",
                "blocks",
                "degenerate_blocks",
            ]
        )

    group_columns = ["season", "week"] if block == "week" else ["season"]
    grouped_indices = list(
        predictions.groupby(group_columns, sort=False, dropna=False).indices.values()
    )
    if not grouped_indices:
        raise ValueError("Predictions contain no bootstrap blocks")
    block_verdict = guard_block_count(
        len(grouped_indices),
        min_blocks=min_blocks,
        on_degenerate=on_degenerate,
        context=f"block_bootstrap_intervals(block={block})",
    )

    estimate = _evaluation_metrics(predictions)
    metric_names = list(estimate)
    draws = np.empty((samples, len(metric_names)), dtype=float)
    generator = np.random.default_rng(seed)
    for sample_index in range(samples):
        selected = generator.integers(0, len(grouped_indices), size=len(grouped_indices))
        positions = np.concatenate([grouped_indices[index] for index in selected])
        sampled_metrics = _evaluation_metrics(predictions.iloc[positions])
        draws[sample_index] = [sampled_metrics[name] for name in metric_names]

    tail = (1.0 - confidence) / 2.0
    lower = np.quantile(draws, tail, axis=0)
    upper = np.quantile(draws, 1.0 - tail, axis=0)
    return pd.DataFrame(
        {
            "metric": metric_names,
            "estimate": [estimate[name] for name in metric_names],
            "lower": lower,
            "upper": upper,
            "confidence": confidence,
            "block": block,
            "samples": samples,
            "blocks": block_verdict.block_count,
            # True => lower/upper are NOT a valid interval at this block
            # count. See the docstring; do not render as one.
            "degenerate_blocks": block_verdict.degenerate,
        }
    )


def bankroll_curve(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=["season_week", "bankroll", "drawdown"])
    curve = (
        ledger.sort_values(["gameday", "game_id"])
        .groupby(["season", "week"], as_index=False)
        .agg(gameday=("gameday", "min"), bankroll=("bankroll_after_week", "last"))
    )
    initial = float(ledger["bankroll_before_week"].iloc[0])
    first_date = pd.to_datetime(curve["gameday"]).min() - pd.Timedelta(days=1)
    curve = pd.concat(
        [
            pd.DataFrame(
                {"season": [np.nan], "week": [0], "gameday": [first_date], "bankroll": [initial]}
            ),
            curve,
        ],
        ignore_index=True,
    )
    curve["drawdown"] = curve["bankroll"] / curve["bankroll"].cummax() - 1.0
    curve["season_week"] = curve.apply(
        lambda row: (
            "start" if pd.isna(row["season"]) else f"{int(row['season'])}-W{int(row['week']):02d}"
        ),
        axis=1,
    )
    return curve


def feature_coverage(features: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        missing = int(features[column].isna().sum())
        rows.append(
            {
                "feature": column,
                "available": len(features) - missing,
                "missing": missing,
                "coverage": 1.0 - missing / len(features) if len(features) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["coverage", "feature"], ignore_index=True)
