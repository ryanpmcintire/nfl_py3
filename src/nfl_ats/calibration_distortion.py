from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.data import require_columns
from nfl_ats.purged_cv import synthetic_signal_beta

__all__ = [
    "additive_plant_gamma",
    "full_pipeline_home_pick",
    "implied_pick_threshold",
    "orthogonal_carrier",
    "plant_additive_effect",
    "sign_only_home_pick",
    "standardized_carrier",
]


def implied_pick_threshold(residuals: npt.ArrayLike) -> float:

    values = np.sort(np.asarray(residuals, dtype=np.float64))
    n = int(values.size)
    if n == 0:
        raise ValueError("implied_pick_threshold requires at least one residual")
    return float(-values[n - math.ceil(n / 2)])


def full_pipeline_home_pick(
    residuals: npt.ArrayLike, predicted_market_residual: npt.ArrayLike
) -> npt.NDArray[np.bool_]:

    sorted_residuals = np.sort(np.asarray(residuals, dtype=np.float64))
    n = int(sorted_residuals.size)
    if n == 0:
        raise ValueError("full_pipeline_home_pick requires at least one residual")
    thresholds = -np.asarray(predicted_market_residual, dtype=np.float64)
    count_above = n - np.searchsorted(sorted_residuals, thresholds, side="right")
    return np.asarray(count_above >= n / 2.0, dtype=bool)


def sign_only_home_pick(predicted_market_residual: npt.ArrayLike) -> npt.NDArray[np.bool_]:

    return np.asarray(np.asarray(predicted_market_residual, dtype=np.float64) > 0.0, dtype=bool)


def orthogonal_carrier(n: int, *, seed: int) -> npt.NDArray[np.float64]:

    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(np.array([-1.0, 1.0]), size=n), dtype=np.float64)


def standardized_carrier(frame: pd.DataFrame, column: str) -> npt.NDArray[np.float64]:

    if column not in frame.columns:
        raise KeyError(f"Carrier column {column!r} is not in the frame")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Carrier column {column!r} has no finite values")
    values = np.where(np.isfinite(values), values, float(np.median(finite)))
    scale = float(np.std(values))
    if scale <= 0.0:
        raise ValueError(f"Carrier column {column!r} is constant")
    return np.asarray((values - float(np.mean(values))) / scale, dtype=np.float64)


def additive_plant_gamma(target_accuracy: float, noise_std: float) -> float:

    return synthetic_signal_beta(target_accuracy, noise_std)


def plant_additive_effect(
    frame: pd.DataFrame,
    *,
    carrier: npt.ArrayLike,
    gamma: float,
    column: str | None = None,
) -> pd.DataFrame:

    require_columns(frame, ("spread_line", "ats_margin"), "additive plant frame")
    values = np.asarray(carrier, dtype=np.float64)
    if values.size != len(frame):
        raise ValueError("carrier length must match the frame")
    if not np.isfinite(values).all():
        raise ValueError("carrier must be finite")
    if not math.isfinite(gamma):
        raise ValueError("gamma must be finite")

    working = frame.copy()
    base = pd.to_numeric(working["ats_margin"], errors="raise").to_numpy(dtype=float)
    planted = base + gamma * values
    spread = pd.to_numeric(working["spread_line"], errors="raise").to_numpy(dtype=float)
    working["ats_margin"] = planted
    working["result"] = spread + planted
    working["home_cover"] = np.select([planted > 0, planted < 0], [1.0, 0.0], default=np.nan)
    if column is not None:
        working[column] = values
    return working
