"""Shared arm grid and weighted-ridge fit helper for MOD-14 (era weighting).

Predeclaration: ``docs/era_weighting_screen.md``. This module is imported by
both ``scripts/era_weighting_cfb_screen.py`` and
``scripts/era_weighting_nfl_screen.py`` so the two leagues run the exact
same predeclared grid and weighting arithmetic. Nothing here edits
``src/nfl_ats`` -- ``make_margin_estimator``/``MarginModel`` are imported and
reused verbatim; the only new logic is the ``sample_weight`` hook the
production fitters (``fit_margin_model``, ``fit_cfb_residual_model``) do not
expose, plus the season-decay/rolling-window arithmetic itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.margin import MarginModel, make_margin_estimator

ArmKind = Literal["uniform", "half_life", "rolling"]


@dataclass(frozen=True)
class WeightingArm:
    name: str
    kind: ArmKind
    # Half-life in seasons (half_life arms) or window length in seasons
    # (rolling arms); unused (None) for the uniform baseline.
    parameter: float | None = None


# Predeclared grid (docs/era_weighting_screen.md Section 2): one baseline
# plus six candidates. half_life=infinity and rolling=all both collapse to
# `baseline` mathematically, so they are represented once, not run twice.
ERA_WEIGHTING_ARMS: tuple[WeightingArm, ...] = (
    WeightingArm("baseline", "uniform", None),
    WeightingArm("half_life_2", "half_life", 2.0),
    WeightingArm("half_life_4", "half_life", 4.0),
    WeightingArm("half_life_8", "half_life", 8.0),
    WeightingArm("half_life_16", "half_life", 16.0),
    WeightingArm("rolling_6", "rolling", 6.0),
    WeightingArm("rolling_10", "rolling", 10.0),
)

CANDIDATE_ARM_NAMES: tuple[str, ...] = tuple(
    arm.name for arm in ERA_WEIGHTING_ARMS if arm.kind != "uniform"
)
BASELINE_ARM_NAME = "baseline"


def half_life_weights(
    seasons: npt.NDArray[np.float64], predict_season: int, half_life: float
) -> npt.NDArray[np.float64]:
    """Season-granularity exponential decay: weight 1.0 for the predicted season.

    ``elapsed`` is clamped at zero rather than allowed to go negative -- a
    training row can share the predicted season (earlier weeks of the same
    season) but never postdate it, since training is already restricted to
    strictly-earlier gamedays upstream of this function.
    """

    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    elapsed = np.clip(predict_season - seasons.astype(np.float64), 0.0, None)
    return np.power(0.5, elapsed / half_life)


def rolling_window_floor_season(predict_season: int, window_seasons: float) -> int:
    """First season kept by a rolling window (inclusive of the predicted season)."""

    return int(predict_season - int(window_seasons) + 1)


def fit_weighted_ridge_margin(
    sorted_frame: pd.DataFrame,
    *,
    target: npt.NDArray[np.float64],
    feature_columns: tuple[str, ...],
    weights: npt.NDArray[np.float64],
    ridge_alpha: float = 10.0,
    distribution_fraction: float = 0.20,
    min_distribution_rows: int = 10,
    min_rows: int = 50,
    random_state: int = 42,
    model_name: str = "ridge",
) -> MarginModel:
    """Generic weighted mirror of ``fit_cfb_residual_model``/``fit_margin_model``.

    ``sorted_frame`` must already be chronologically sorted and filtered to
    completed, target-notna rows by the caller (CFB and NFL differ in that
    upstream filtering -- ``regular_season_rows`` for NFL, none for CFB --
    so this function stays league-agnostic and takes the finished frame).
    ``target``/``weights`` are aligned 1:1 with ``sorted_frame``'s row order.

    The weight vector is routed to the Ridge step only
    (``regressor__sample_weight``); the imputer/scaler steps of
    ``make_margin_estimator``'s pipeline are fit unweighted, exactly as the
    frozen production pipeline already does for every existing (unweighted)
    arm.
    """

    if len(sorted_frame) < min_rows:
        raise ValueError(f"At least {min_rows} completed games are required to fit")
    if len(sorted_frame) != len(target) or len(sorted_frame) != len(weights):
        raise ValueError("sorted_frame, target, and weights must be the same length")
    if not 0.10 <= distribution_fraction < 0.5:
        raise ValueError("distribution_fraction must be in [0.10, 0.5)")

    distribution_rows = int(len(sorted_frame) * distribution_fraction)
    if distribution_rows < min_distribution_rows or len(sorted_frame) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time residual distribution")
    split = len(sorted_frame) - distribution_rows

    columns = list(feature_columns)
    temporary = make_margin_estimator(model_name, random_state, ridge_alpha=ridge_alpha)
    temporary.fit(
        sorted_frame.iloc[:split].loc[:, columns],
        target[:split],
        regressor__sample_weight=weights[:split],
    )
    calibration_prediction = np.asarray(
        temporary.predict(sorted_frame.iloc[split:].loc[:, columns]), dtype=float
    )
    residuals = np.asarray(target[split:] - calibration_prediction, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < min_distribution_rows:
        raise ValueError("Out-of-time residual distribution has too few finite values")

    estimator = make_margin_estimator(model_name, random_state, ridge_alpha=ridge_alpha)
    estimator.fit(
        sorted_frame.loc[:, columns],
        target,
        regressor__sample_weight=weights,
    )
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name=model_name,
        ridge_alpha=ridge_alpha if model_name == "ridge" else None,
        target="market_residual",
        feature_columns=feature_columns,
        training_rows=len(sorted_frame),
        distribution_rows=len(residuals),
        training_max_gameday=pd.to_datetime(sorted_frame["gameday"]).max().date().isoformat(),
    )
