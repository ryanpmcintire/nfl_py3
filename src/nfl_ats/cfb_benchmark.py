"""Chronological CFB-only market-residual ATS benchmark (XLG-03).

One frozen configuration, evaluated once, mirroring the active NFL
market-residual model: a Ridge regression (alpha 10, no calibration) on the
declared CFB feature contract predicts the home ATS margin
(``result - spread_line``), with cover probabilities read from an out-of-time
empirical residual distribution exactly as in ``margin.fit_margin_model``.
Training is strictly earlier than each scored week, with the NFL minimum
training floor of 500 games taken verbatim.

The benchmark is CFB-only: no NFL rows are read and no NFL outcomes are
touched. Its purpose is the instrument - sample, controls, and detection
power for future CFB->NFL transfer claims - not an edge claim. The headline
window is the clean core (2012-2019 and 2021-2025); 2006-2011 is reported as
a thin-line-regime sensitivity split only, and 2020 as the sparse-provider
regime split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.data import DataContractError
from nfl_ats.margin import MarginModel, fit_market_baseline, make_margin_estimator
from nfl_ats.outcomes import outcome_bootstrap_intervals, summarize_outcome_method

# Frozen benchmark configuration; fixed upfront to mirror the active NFL
# market-residual configuration, never tuned on the evaluation window.
CFB_BENCHMARK_TARGET = "market_residual"
CFB_BENCHMARK_REGRESSOR = "ridge"
CFB_BENCHMARK_RIDGE_ALPHA = 10.0
CFB_BENCHMARK_CALIBRATION = "none"
CFB_BENCHMARK_MIN_TRAIN_GAMES = 500
CFB_BENCHMARK_DISTRIBUTION_FRACTION = 0.20
CFB_BENCHMARK_START_SEASON = 2006
CFB_BENCHMARK_END_SEASON = 2025
CFB_BENCHMARK_BOOTSTRAP_SAMPLES = 1_000
CFB_BENCHMARK_BOOTSTRAP_SEED = 20260812
CFB_BENCHMARK_METHODS = ("market", "market_residual")

CFB_CLEAN_CORE_SEASONS = tuple(range(2012, 2020)) + tuple(range(2021, 2026))
CFB_THIN_REGIME_SEASONS = tuple(range(2006, 2012))

_PREDICTION_PASSTHROUGH = (
    "game_id",
    "season",
    "week",
    "gameday",
    "source_regime",
    "spread_line",
    "spread_book_count",
    "spread_dispersion",
    "home_spread_odds",
    "away_spread_odds",
    "result",
    "ats_margin",
    "home_cover",
)


@dataclass(frozen=True)
class CfbBenchmarkResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    season_summary: pd.DataFrame


def cfb_evaluation_window(season: int) -> str:
    """Name the reporting split a season belongs to."""

    if season in CFB_THIN_REGIME_SEASONS:
        return "thin_2006_2011"
    if season == 2020:
        return "regime_2020"
    if season in CFB_CLEAN_CORE_SEASONS:
        return "clean_core"
    raise ValueError(f"Season {season} is outside the CFB benchmark window")


def fit_cfb_residual_model(
    training: pd.DataFrame,
    *,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    distribution_fraction: float = CFB_BENCHMARK_DISTRIBUTION_FRACTION,
    min_distribution_rows: int = 10,
    random_state: int = 42,
    feature_columns: tuple[str, ...] = CFB_MODEL_FEATURE_COLUMNS,
) -> MarginModel:
    """Fit the CFB market-residual Ridge, mirroring ``fit_margin_model``.

    The recipe is identical to the NFL implementation: sort chronologically,
    hold out the trailing ``distribution_fraction`` as an out-of-time residual
    distribution, then refit on all rows. Only the feature contract differs
    (the declared CFB columns instead of a registered NFL feature set).
    ``feature_columns`` defaults to the frozen XLG-03 contract; a declared
    candidate family (e.g. the XLG-04 role-continuity columns) may extend it
    without touching the frozen benchmark path.
    """

    required = {"game_id", "gameday", "ats_margin", *feature_columns}
    missing = sorted(required.difference(training.columns))
    if missing:
        raise DataContractError(f"CFB margin training is missing columns: {', '.join(missing)}")
    if not 0.10 <= distribution_fraction < 0.5:
        raise ValueError("distribution_fraction must be in [0.10, 0.5)")

    frame = training.loc[pd.to_numeric(training["ats_margin"], errors="coerce").notna()].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError("At least 50 completed games are required for a CFB margin model")
    distribution_rows = int(len(frame) * distribution_fraction)
    if distribution_rows < min_distribution_rows or len(frame) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time residual distribution")

    split = len(frame) - distribution_rows
    target = pd.to_numeric(frame["ats_margin"], errors="raise").to_numpy(dtype=float)
    temporary = make_margin_estimator(
        CFB_BENCHMARK_REGRESSOR, random_state, ridge_alpha=ridge_alpha
    )
    temporary.fit(frame.iloc[:split].loc[:, list(feature_columns)], target[:split])
    calibration_prediction = np.asarray(
        temporary.predict(frame.iloc[split:].loc[:, list(feature_columns)]), dtype=float
    )
    residuals = np.asarray(target[split:] - calibration_prediction, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < min_distribution_rows:
        raise ValueError("Out-of-time residual distribution has too few finite values")

    estimator = make_margin_estimator(
        CFB_BENCHMARK_REGRESSOR, random_state, ridge_alpha=ridge_alpha
    )
    estimator.fit(frame.loc[:, list(feature_columns)], target)
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name=CFB_BENCHMARK_REGRESSOR,
        ridge_alpha=ridge_alpha,
        target="market_residual",
        feature_columns=feature_columns,
        training_rows=len(frame),
        distribution_rows=len(residuals),
        training_max_gameday=frame["gameday"].max().date().isoformat(),
    )


def _score_week(weekly_games: pd.DataFrame, models: dict[str, MarginModel]) -> list[pd.DataFrame]:
    batches: list[pd.DataFrame] = []
    for method, model in models.items():
        batch = weekly_games.loc[:, list(_PREDICTION_PASSTHROUGH)].copy()
        forecasts = model.predict(weekly_games)
        for column in forecasts:
            batch[column] = forecasts[column]
        batch["method"] = method
        batch["model_name"] = model.model_name
        batch["train_rows"] = model.training_rows
        batch["distribution_rows"] = model.distribution_rows
        batch["train_max_gameday"] = model.training_max_gameday
        # The benchmark is a forced-pick instrument, not a wagering policy.
        batch["bet_side"] = "PASS"
        batch["bet_odds"] = np.nan
        batches.append(batch)
    return batches


def cfb_walk_forward_benchmark(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
) -> CfbBenchmarkResult:
    """Score every eligible CFB week with strictly earlier training."""

    if end_season < start_season:
        raise ValueError("end_season cannot be earlier than start_season")
    required = {*_PREDICTION_PASSTHROUGH, *CFB_MODEL_FEATURE_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"CFB features are missing columns: {', '.join(missing)}")

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {start_season} to {end_season}")

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        models = {
            "market": fit_market_baseline(training),
            "market_residual": fit_cfb_residual_model(training, ridge_alpha=ridge_alpha),
        }
        batches.extend(_score_week(weekly_games, models))
    if not batches:
        raise ValueError("No CFB week had enough prior training games")

    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)
    summary, season_summary = summarize_cfb_benchmark(predictions)
    return CfbBenchmarkResult(
        predictions=predictions, summary=summary, season_summary=season_summary
    )


def summarize_cfb_benchmark(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summaries per evaluation window/method and per season/method."""

    summary_rows: list[dict[str, Any]] = []
    windows = ["clean_core", "thin_2006_2011", "regime_2020", "all"]
    for window in windows:
        subset = (
            predictions
            if window == "all"
            else predictions.loc[predictions["evaluation_window"].eq(window)]
        )
        if subset.empty:
            continue
        for _, group in subset.groupby("method", sort=True):
            summary_rows.append({"evaluation_window": window, **summarize_outcome_method(group)})
    season_rows: list[dict[str, Any]] = []
    for (_, season), group in predictions.groupby(["method", "season"], sort=True):
        season_rows.append(
            {
                "season": int(str(season)),
                "evaluation_window": cfb_evaluation_window(int(str(season))),
                **summarize_outcome_method(group),
            }
        )
    summary = pd.DataFrame(summary_rows)
    season_summary = pd.DataFrame(season_rows).sort_values(["method", "season"], ignore_index=True)
    return summary, season_summary


def cfb_benchmark_uncertainty(
    predictions: pd.DataFrame,
    *,
    samples: int = CFB_BENCHMARK_BOOTSTRAP_SAMPLES,
    seed: int = CFB_BENCHMARK_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Week- and season-blocked intervals on the clean-core window.

    Reuses the NFL outcome bootstrap: within-block schedule dependence is
    preserved and every method's delta against the market baseline carries its
    own blocked interval.
    """

    clean = predictions.loc[predictions["evaluation_window"].eq("clean_core")].copy()
    if clean.empty:
        raise ValueError("No clean-core predictions available for uncertainty estimation")
    frames = [
        outcome_bootstrap_intervals(clean, samples=samples, block="week", seed=seed).assign(
            evaluation_window="clean_core"
        ),
        outcome_bootstrap_intervals(clean, samples=samples, block="season", seed=seed).assign(
            evaluation_window="clean_core"
        ),
    ]
    return pd.concat(frames, ignore_index=True)
