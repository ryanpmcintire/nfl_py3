"""Leak-safe calibration of chronological out-of-sample ATS probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from nfl_ats.constants import DEFAULT_MIN_CALIBRATION_GAMES
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
    min_calibration_games: int = DEFAULT_MIN_CALIBRATION_GAMES,
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


# ---------------------------------------------------------------------------
# Residual-distribution smoothing (research item: docs/ecdf_smoothing.md)
# ---------------------------------------------------------------------------
#
# ``margin.MarginModel`` builds its predictive distribution for every game by
# adding a fixed out-of-time residual SAMPLE (typically a few hundred draws --
# see ``fit_margin_model``'s 20% chronological holdout) to that game's
# predicted centre, then reads cover/win/loss probabilities off the resulting
# discretized empirical CDF (``margin._smoothed_probability``: a Laplace/KT
# continuity-corrected count, not a fitted density). That is an ECDF, with
# whatever sampling noise a few-hundred-draw ECDF carries.
#
# Everything below is an OPT-IN alternative reader of the SAME residual draws
# -- it never touches ``margin.py`` and is never called by the production
# prediction path unless a caller explicitly builds a ``ResidualSmoother``.
# The frozen active model is therefore bit-identical whether or not this
# module is imported; ``tests/test_calibration_ecdf_smoothing.py`` pins that.
#
# IMPORTANT: replacing the ECDF with a smoothed density is NOT the "rescale
# the point prediction" operation MOD-06 closed (docs/pool_edge_plan.md: a
# positive scalar rescaling can never flip ``sign(predicted residual)``).
# The pool's actual forced pick is ``home_cover_probability >= 0.5``
# (`nfl_ats.pool.build_ats_pool_card`), which is the EMPIRICAL MEDIAN of
# ``center + residuals`` compared against the line, not the point residual
# compared against zero. Because the raw held-out residual sample has a
# nonzero, noisy mean/median (it corrects for the temporary fit model's own
# out-of-time bias), those two decision rules already disagree for a real,
# measurable share of games under the CURRENT unsmoothed model. Smoothing
# changes the estimated location/shape of the distribution -- not its scale
# -- so it can move which side of 0.5 a game near that boundary falls on.
# That is a real, distinct lever from rescaling, which is exactly why it
# needs its own predeclared confirmation window rather than shipping on a
# measurement.

ResidualSmoothingMethod = Literal["ecdf", "gaussian", "gaussian_kde", "skew_normal"]
RESIDUAL_SMOOTHING_METHODS: tuple[ResidualSmoothingMethod, ...] = (
    "ecdf",
    "gaussian",
    "gaussian_kde",
    "skew_normal",
)
_SURVIVAL_EPSILON = 1e-9


def normalize_residual_smoothing_method(method: str) -> ResidualSmoothingMethod:
    """Return a known residual-smoothing method or fail before any fit."""

    if method not in RESIDUAL_SMOOTHING_METHODS:
        choices = ", ".join(RESIDUAL_SMOOTHING_METHODS)
        raise ValueError(f"Unknown residual smoothing method {method!r}; choose one of {choices}")
    return method


@dataclass(frozen=True)
class ResidualSmoother:
    """A fitted reader of one out-of-time residual sample.

    ``method="ecdf"`` is the CONTROL arm: it reproduces
    ``margin._smoothed_probability``'s continuity-corrected empirical CDF
    from the same draws, to floating-point precision (pinned by a test), so
    every comparison in this module is "smoothed vs the production math",
    never "smoothed vs some other reimplementation of the production math".
    The other three methods fit a continuous density to the same draws
    instead of resampling them directly.
    """

    method: ResidualSmoothingMethod
    n: int
    residuals: npt.NDArray[np.float64]
    mean: float
    std: float
    kde: Any | None
    skew_params: tuple[float, float, float] | None

    def survival(self, thresholds: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
        """P(residual > threshold), vectorized over one or many thresholds."""

        values = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
        if self.method == "ecdf":
            counts = np.array(
                [np.count_nonzero(self.residuals > threshold) for threshold in values],
                dtype=np.float64,
            )
            result = (counts + 0.5) / (self.n + 1.0)
        elif self.method == "gaussian":
            result = stats.norm.sf(values, loc=self.mean, scale=self.std)
        elif self.method == "skew_normal":
            if self.skew_params is None:  # pragma: no cover - fit_residual_smoother guarantees
                raise RuntimeError("skew_normal smoother is missing fitted parameters")
            shape, loc, scale = self.skew_params
            result = stats.skewnorm.sf(values, shape, loc=loc, scale=scale)
        else:
            if self.kde is None:  # pragma: no cover - fit_residual_smoother guarantees
                raise RuntimeError("gaussian_kde smoother is missing a fitted kde")
            result = np.array(
                [self.kde.integrate_box_1d(threshold, np.inf) for threshold in values],
                dtype=np.float64,
            )
        return np.clip(result, _SURVIVAL_EPSILON, 1.0 - _SURVIVAL_EPSILON)


def fit_residual_smoother(
    residuals: npt.NDArray[np.float64], method: str = "ecdf"
) -> ResidualSmoother:
    """Fit one out-of-time residual sample under the requested method.

    ``residuals`` is exactly the array ``MarginModel.residuals`` holds for one
    fitted margin model (one week, in a walk-forward); every game scored by
    that model shares this one fitted smoother.
    """

    normalized = normalize_residual_smoothing_method(method)
    values = np.asarray(residuals, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        raise ValueError("At least 10 residual draws are required to fit a smoother")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Residual sample has degenerate (non-positive) spread")
    kde = None
    skew_params: tuple[float, float, float] | None = None
    if normalized == "gaussian_kde":
        kde = stats.gaussian_kde(values)
    elif normalized == "skew_normal":
        shape, loc, scale = stats.skewnorm.fit(values)
        skew_params = (float(shape), float(loc), float(scale))
    return ResidualSmoother(
        method=normalized,
        n=len(values),
        residuals=values,
        mean=mean,
        std=std,
        kde=kde,
        skew_params=skew_params,
    )


def smoothed_home_cover_probability(
    residuals: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64] | pd.Series,
    lines: npt.NDArray[np.float64] | pd.Series,
    *,
    method: str = "ecdf",
) -> npt.NDArray[np.float64]:
    """Home-cover probability under an opt-in (possibly smoothed) residual model.

    Mirrors ``MarginModel.predict``'s ``home_cover_probability`` for the
    ``margin``/``market_residual`` targets: ``centers`` is each game's
    predicted margin, ``lines`` its quoted spread, and ``residuals`` the
    SAME out-of-time draws the frozen model would add to that centre.
    ``method="ecdf"`` reproduces the production probability; any other
    method re-estimates the residual distribution's shape/location from the
    same draws instead of resampling them, which can move which side of 0.5
    a game near the decision boundary falls on (see the module docstring
    above for why that is a distinct lever from rescaling).
    """

    smoother = fit_residual_smoother(residuals, method=method)
    thresholds = np.asarray(lines, dtype=np.float64) - np.asarray(centers, dtype=np.float64)
    return smoother.survival(thresholds)
