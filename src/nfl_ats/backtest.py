"""Expanding-window backtests and honest betting metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from nfl_ats.modeling import (
    CoverModel,
    fit_cover_model,
    regular_season_rows,
    validate_model_frame,
)
from nfl_ats.odds import choose_bet, market_hold, no_vig_probabilities, settle_bet
from nfl_ats.prediction_safety import validate_prediction_card


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _prediction_rows(
    games: pd.DataFrame,
    model: CoverModel,
    min_edge: float,
) -> pd.DataFrame:
    scored = games.copy()
    scored["home_cover_probability"] = model.predict_home_cover(scored)
    decisions = [
        choose_bet(
            probability,
            _optional_number(row.get("home_spread_odds")),
            _optional_number(row.get("away_spread_odds")),
            min_edge=min_edge,
        )
        for probability, (_, row) in zip(
            scored["home_cover_probability"], scored.iterrows(), strict=True
        )
    ]
    scored["pick"] = np.where(scored["home_cover_probability"] >= 0.5, "HOME", "AWAY")
    scored["bet_side"] = [decision.side for decision in decisions]
    scored["edge"] = [decision.edge for decision in decisions]
    scored["bet_odds"] = [decision.odds for decision in decisions]
    scored["break_even_probability"] = [decision.break_even_probability for decision in decisions]
    market_probabilities = [
        no_vig_probabilities(
            _optional_number(row.get("home_spread_odds")),
            _optional_number(row.get("away_spread_odds")),
        )
        for _, row in scored.iterrows()
    ]
    scored["market_home_no_vig_probability"] = [pair[0] for pair in market_probabilities]
    scored["market_hold"] = [
        market_hold(
            _optional_number(row.get("home_spread_odds")),
            _optional_number(row.get("away_spread_odds")),
        )
        for _, row in scored.iterrows()
    ]
    scored["train_rows"] = model.training_rows
    scored["train_max_gameday"] = model.training_max_gameday
    validate_prediction_card(
        scored,
        min_edge=min_edge,
        feature_columns=model.feature_columns,
    )
    return scored


def score_week(
    features: pd.DataFrame,
    season: int,
    week: int,
    model_name: str = "logistic",
    min_edge: float = 0.02,
    min_train_games: int = 500,
    feature_set: str = "full",
) -> tuple[pd.DataFrame, CoverModel]:
    validate_model_frame(features)
    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)].copy()
    if target.empty:
        raise ValueError(f"No games found for {season} week {week}")
    missing_lines = target.loc[target["spread_line"].isna(), "game_id"].astype(str).tolist()
    if missing_lines:
        examples = ", ".join(missing_lines[:5])
        raise ValueError(
            f"Cannot score an ATS card with missing spread lines; refresh market data: {examples}"
        )
    cutoff = target["gameday"].min()
    training = frame.loc[frame["gameday"].lt(cutoff) & frame["home_cover"].notna()].copy()
    if len(training) < min_train_games:
        raise ValueError(
            f"Only {len(training)} eligible games precede {season} week {week}; "
            f"need {min_train_games}"
        )
    model = fit_cover_model(training, model_name=model_name, feature_set=feature_set)
    return _prediction_rows(target, model, min_edge), model


def _calibration_error(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probability, boundaries[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += selected.mean() * abs(actual[selected].mean() - probability[selected].mean())
    return float(error)


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    evaluated = predictions.loc[predictions["home_cover"].notna()].copy()
    if evaluated.empty:
        raise ValueError("Backtest produced no non-push predictions")
    actual = evaluated["home_cover"].astype(int).to_numpy()
    probability = evaluated["home_cover_probability"].to_numpy(dtype=float)
    classification = (probability >= 0.5).astype(int)

    wagered = predictions.loc[predictions["bet_side"].ne("PASS")].copy()
    wagered["profit_units"] = wagered.apply(
        lambda row: (
            0.0
            if pd.isna(row["home_cover"])
            else settle_bet(row["bet_side"], row["home_cover"], row["bet_odds"])
        ),
        axis=1,
    )
    wins = int((wagered["profit_units"] > 0).sum())
    losses = int((wagered["profit_units"] < 0).sum())
    pushes = int((wagered["profit_units"] == 0).sum())
    net_profit = float(wagered["profit_units"].sum())
    ats_margin = pd.to_numeric(predictions["ats_margin"], errors="coerce").dropna()

    return {
        "games_scored": len(predictions),
        "games_evaluated": len(evaluated),
        "pushes": int(predictions["home_cover"].isna().sum()),
        "accuracy": float(accuracy_score(actual, classification)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "expected_calibration_error": _calibration_error(actual, probability),
        "market_margin_mae": float(ats_margin.abs().mean()),
        "market_margin_rmse": float(np.sqrt(np.square(ats_margin).mean())),
        "bets": len(wagered),
        "bet_coverage": float(len(wagered) / len(predictions)),
        "wins": wins,
        "losses": losses,
        "bet_pushes": pushes,
        "net_profit_units": net_profit,
        "roi": float(net_profit / len(wagered)) if len(wagered) else 0.0,
    }


def walk_forward_backtest(
    features: pd.DataFrame,
    start_season: int,
    end_season: int | None = None,
    model_name: str = "logistic",
    min_edge: float = 0.02,
    min_train_games: int = 500,
    feature_set: str = "full",
) -> BacktestResult:
    """Retrain before each week using only games from earlier dates.

    ``start_season`` and ``end_season`` bound only the scored rows. Training
    always includes every eligible game strictly before that week's cutoff,
    which mirrors a model that is refreshed as new results become available.
    """

    validate_model_frame(features)
    if end_season is not None and end_season < start_season:
        raise ValueError("end_season cannot be earlier than start_season")
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["ats_margin"].notna()].copy()
    test_mask = completed["season"].ge(start_season)
    if end_season is not None:
        test_mask &= completed["season"].le(end_season)
    test = completed.loc[test_mask]
    if test.empty:
        end_label = f" through {end_season}" if end_season is not None else ""
        raise ValueError(f"No completed games found from season {start_season}{end_label}")

    batches: list[pd.DataFrame] = []
    for (season, week), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff) & completed["home_cover"].notna()]
        if len(training) < min_train_games:
            continue
        model = fit_cover_model(training, model_name=model_name, feature_set=feature_set)
        batch = _prediction_rows(weekly_games, model, min_edge)
        batch["test_season"] = int(str(season))
        batch["test_week"] = int(str(week))
        batches.append(batch)

    if not batches:
        raise ValueError("No backtest window had enough prior training games")
    predictions = pd.concat(batches, ignore_index=True).sort_values(["gameday", "game_id"])
    metrics = summarize_predictions(predictions)
    metrics.update(
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "start_season": start_season,
            "end_season": end_season,
            "min_edge": min_edge,
            "min_train_games": min_train_games,
            "first_test_gameday": predictions["gameday"].min().date().isoformat(),
            "last_test_gameday": predictions["gameday"].max().date().isoformat(),
        }
    )
    return BacktestResult(predictions.reset_index(drop=True), metrics)
