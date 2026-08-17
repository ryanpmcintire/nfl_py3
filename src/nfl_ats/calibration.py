"""Leak-safe calibration of chronological out-of-sample ATS probabilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from nfl_ats.odds import choose_bet

CoverCalibrationMethod = Literal["none", "platt", "isotonic", "beta"]
COVER_CALIBRATION_METHODS: tuple[CoverCalibrationMethod, ...] = (
    "none",
    "platt",
    "isotonic",
    "beta",
)
_PROBABILITY_EPSILON = 1e-6


def normalize_cover_calibration_method(method: str) -> CoverCalibrationMethod:
    """Return a known calibration method or fail before model fitting."""

    if method not in COVER_CALIBRATION_METHODS:
        choices = ", ".join(COVER_CALIBRATION_METHODS)
        raise ValueError(f"Unknown cover calibration method {method!r}; choose one of {choices}")
    return method


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.clip(probabilities.astype(float), _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON),
        dtype=float,
    )


def _logit_features(probabilities: np.ndarray) -> np.ndarray:
    clipped = _clip(probabilities)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _beta_features(probabilities: np.ndarray) -> np.ndarray:
    clipped = _clip(probabilities)
    return np.column_stack((np.log(clipped), -np.log1p(-clipped)))


def _calibrated_probabilities(
    method: CoverCalibrationMethod,
    training_probability: np.ndarray,
    training_outcome: np.ndarray,
    target_probability: np.ndarray,
) -> np.ndarray:
    if method == "none":
        return target_probability.astype(float)
    if len(np.unique(training_outcome)) != 2:
        raise ValueError("Calibration history must contain both ATS outcome classes")
    if method == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(training_probability, training_outcome)
        calibrated = np.asarray(calibrator.predict(target_probability), dtype=float)
    else:
        training_features = (
            _logit_features(training_probability)
            if method == "platt"
            else _beta_features(training_probability)
        )
        target_features = (
            _logit_features(target_probability)
            if method == "platt"
            else _beta_features(target_probability)
        )
        calibrator = LogisticRegression(C=1_000_000.0, max_iter=1_000, solver="lbfgs")
        calibrator.fit(training_features, training_outcome)
        calibrated = np.asarray(calibrator.predict_proba(target_features)[:, 1], dtype=float)
    return _clip(calibrated)


def _replace_decisions(frame: pd.DataFrame, *, min_edge: float) -> pd.DataFrame:
    result = frame.copy()
    decisions = [
        choose_bet(
            float(row["home_cover_probability"]),
            row.get("home_spread_odds"),
            row.get("away_spread_odds"),
            min_edge=min_edge,
        )
        for _, row in result.iterrows()
    ]
    result["bet_side"] = [decision.side for decision in decisions]
    result["edge"] = [decision.edge for decision in decisions]
    result["bet_odds"] = [decision.odds for decision in decisions]
    result["break_even_probability"] = [decision.break_even_probability for decision in decisions]
    return result


def calibrate_cover_prediction_stream(
    predictions: pd.DataFrame,
    *,
    method: str,
    evaluation_start_season: int,
    # DERIVED, not inherited (2026-08-17). The previous default of 400 was an
    # undocumented constant that nobody had ever tested; it demanded 200
    # observations per parameter for a two-parameter Platt sigmoid and, via the
    # rotation registry's warm-up rule, permanently shrank the confirmation-window
    # pool. Measured on the real 2009-2025 walk-forward stream by bucketing
    # calibrated-vs-raw Brier by the history each week's calibrator actually had:
    # 100-199 rows makes Brier WORSE (0.206 -> 0.284, though only 16 games);
    # 200-399 rows already IMPROVES it (0.269 -> 0.250 on 204 games), as does
    # every larger bucket with diminishing returns. So a floor is real but 400 is
    # twice what the evidence supports; 200 is the smallest demonstrated-safe
    # value. Raising it again needs evidence, not caution.
    min_calibration_games: int = 200,
    min_edge: float = 0.02,
) -> pd.DataFrame:
    """Calibrate weekly predictions using only earlier out-of-sample predictions.

    The input must itself be a chronological walk-forward prediction stream.
    Every target week is calibrated from completed prediction rows strictly
    before the first kickoff in that week. No in-sample training predictions
    or same-week outcomes enter the calibrator.
    """

    calibration_method = normalize_cover_calibration_method(method)
    if min_calibration_games < 2:
        raise ValueError("min_calibration_games must be at least 2")
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "method",
        "home_cover",
        "home_cover_probability",
        "home_spread_odds",
        "away_spread_odds",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Calibration predictions are missing columns: {', '.join(missing)}")
    if predictions.empty:
        raise ValueError("Calibration predictions cannot be empty")
    if not predictions["method"].astype(str).eq("market_residual").all():
        raise ValueError("Cover calibration currently supports only market_residual predictions")
    if predictions["game_id"].duplicated().any():
        raise ValueError("Calibration requires exactly one raw prediction per game")

    frame = predictions.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame["home_cover_probability"] = pd.to_numeric(
        frame["home_cover_probability"], errors="coerce"
    )
    raw_probability = frame["home_cover_probability"].to_numpy(dtype=float)
    if (
        not np.isfinite(raw_probability).all()
        or not np.logical_and(raw_probability >= 0.0, raw_probability <= 1.0).all()
    ):
        raise ValueError("Raw home-cover probabilities must be finite and between zero and one")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    evaluation = frame.loc[frame["season"].ge(evaluation_start_season)]
    if evaluation.empty:
        raise ValueError(f"No calibration evaluation rows from season {evaluation_start_season}")

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_rows in evaluation.groupby(["season", "week"], sort=True):
        cutoff = weekly_rows["gameday"].min()
        history = frame.loc[
            frame["gameday"].lt(cutoff)
            & frame["home_cover"].notna()
            & frame["home_cover_probability"].notna()
        ]
        batch = weekly_rows.copy()
        batch["raw_home_cover_probability"] = batch["home_cover_probability"].astype(float)
        batch["calibration_method"] = calibration_method
        if calibration_method == "none":
            batch["calibration_rows"] = 0
            batch["calibration_max_gameday"] = pd.NaT
        else:
            if len(history) < min_calibration_games:
                season = int(str(batch["season"].iloc[0]))
                week = int(str(batch["week"].iloc[0]))
                raise ValueError(
                    f"Only {len(history)} prior prediction rows calibrate {season} week {week}; "
                    f"need {min_calibration_games}"
                )
            training_probability = history["home_cover_probability"].to_numpy(dtype=float)
            training_outcome = history["home_cover"].to_numpy(dtype=int)
            target_probability = batch["home_cover_probability"].to_numpy(dtype=float)
            batch["home_cover_probability"] = _calibrated_probabilities(
                calibration_method,
                training_probability,
                training_outcome,
                target_probability,
            )
            batch["calibration_rows"] = len(history)
            batch["calibration_max_gameday"] = history["gameday"].max()
        batches.append(_replace_decisions(batch, min_edge=min_edge))

    return pd.concat(batches, ignore_index=True).sort_values(
        ["gameday", "game_id"], ignore_index=True
    )
