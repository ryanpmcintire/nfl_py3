from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    MIN_BAND_GAMES,
    MassPreservingRead,
    band_read,
    residual_location,
    walk_forward_reads,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

ACCURACY_POINT_CEILING = 100.0
_ROBUST_SIGMAS = 3.0
_MAD_TO_SIGMA = 1.4826
_MEAN_ABS_DEVIATION_TO_SIGMA = 1.2533

KEY_NUMBERS: tuple[float, ...] = (3.0, 7.0, 10.0, 14.0)
SERVED_ATOMS: tuple[float, ...] = (3.0, 7.0)
NEIGHBOURHOOD_HALF_WIDTH = 0.5
_TOLERANCE = 1e-9


def distance_to_nearest_atom(
    lines: Sequence[float] | FloatArray | pd.Series, atoms: Iterable[float] = SERVED_ATOMS
) -> FloatArray:

    values = np.abs(np.asarray(pd.to_numeric(pd.Series(lines), errors="raise"), dtype=float))
    keys = np.asarray(tuple(atoms), dtype=float)
    if keys.size == 0:
        raise ValueError("At least one key-number atom is required")
    return np.asarray(np.min(np.abs(values[:, None] - keys[None, :]), axis=1), dtype=np.float64)


def key_neighbourhood_mask(
    lines: Sequence[float] | FloatArray | pd.Series,
    atoms: Iterable[float] = SERVED_ATOMS,
    half_width: float = NEIGHBOURHOOD_HALF_WIDTH,
    *,
    half_point_only: bool = False,
) -> BoolArray:

    distance = distance_to_nearest_atom(lines, atoms)
    if half_point_only:
        return np.asarray(np.abs(distance - float(half_width)) < _TOLERANCE, dtype=bool)
    return np.asarray(distance <= float(half_width) + _TOLERANCE, dtype=bool)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    atoms: tuple[float, ...] | None
    half_width: float = NEIGHBOURHOOD_HALF_WIDTH
    half_point_only: bool = False
    description: str = ""

    def mask(self, lines: Sequence[float] | FloatArray | pd.Series) -> BoolArray:
        size = len(pd.Series(lines))
        if self.atoms is None:
            return np.ones(size, dtype=bool)
        return key_neighbourhood_mask(
            lines, self.atoms, self.half_width, half_point_only=self.half_point_only
        )


ARMS: dict[str, ArmSpec] = {
    "G1": ArmSpec(
        "G1",
        None,
        description="the discrete read decides the side on every game",
    ),
    "G2": ArmSpec(
        "G2",
        SERVED_ATOMS,
        description="on 3 and 7 and the half points either side of them",
    ),
    "G3": ArmSpec(
        "G3",
        KEY_NUMBERS,
        description="on 3, 7, 10 and 14 and the half points either side of them",
    ),
    "G4": ArmSpec(
        "G4",
        SERVED_ATOMS,
        half_point_only=True,
        description="only on the half points either side of 3 and 7",
    ),
}


def apply_arm(
    lines: Sequence[float] | FloatArray | pd.Series,
    baseline_probability: FloatArray,
    baseline_push: FloatArray,
    discrete_probability: FloatArray,
    discrete_push: FloatArray,
    arm: ArmSpec | str,
) -> dict[str, np.ndarray]:

    spec = ARMS[arm] if isinstance(arm, str) else arm
    touched = spec.mask(lines)
    base_p = np.asarray(baseline_probability, dtype=float)
    base_push = np.asarray(baseline_push, dtype=float)
    candidate_p = np.asarray(discrete_probability, dtype=float)
    candidate_push = np.asarray(discrete_push, dtype=float)
    shapes = {base_p.shape, base_push.shape, candidate_p.shape, candidate_push.shape}
    if len(shapes) != 1 or base_p.shape != touched.shape:
        raise ValueError("Every arm input needs one value per quoted line")
    return {
        "touched": touched,
        "probability": np.where(touched, candidate_p, base_p),
        "push": np.where(touched, candidate_push, base_push),
    }


def discrete_side_read(
    pool_line: FloatArray,
    pool_margin: FloatArray,
    line: float,
    point: float,
    *,
    half_width: float = BAND_HALF_WIDTH,
    min_band_games: int = MIN_BAND_GAMES,
) -> MassPreservingRead:

    return band_read(
        np.asarray(pool_line, dtype=float),
        np.asarray(pool_margin, dtype=float),
        float(line),
        float(point),
        float(half_width),
        int(min_band_games),
    )


def walk_forward_side_reads(
    pool: pd.DataFrame, targets: pd.DataFrame, half_width: float = BAND_HALF_WIDTH
) -> pd.DataFrame:

    return walk_forward_reads(pool, targets, float(half_width))


def plausible_standard_error_floor(
    reference: Iterable[tuple[float, int]], sample_games: int
) -> float | None:

    positive = [
        (float(error), float(games)) for error, games in reference if error > 0.0 and games > 0
    ]
    if len(positive) < 3 or sample_games <= 0:
        return None
    scale = statistics.median([error**2 * games for error, games in positive])
    if scale <= 0.0:
        return None
    log_ratios = [math.log(error / math.sqrt(scale / games)) for error, games in positive]
    centre = statistics.median(log_ratios)
    deviations = [abs(ratio - centre) for ratio in log_ratios]
    robust_sigma = _MAD_TO_SIGMA * statistics.median(deviations)
    if robust_sigma <= 0.0:
        robust_sigma = _MEAN_ABS_DEVIATION_TO_SIGMA * statistics.fmean(deviations)
    if robust_sigma <= 0.0:
        return None
    cutoff = centre - _ROBUST_SIGMAS * robust_sigma
    return math.exp(cutoff) * math.sqrt(scale / float(sample_games))


def floor_degenerate_cell(
    metrics: dict[str, float],
    reference: Iterable[tuple[float, int]],
    *,
    ceiling: float | None = ACCURACY_POINT_CEILING,
) -> tuple[dict[str, float], str | None]:

    error = float(metrics.get("standard_error", 0.0))
    lower, upper = float(metrics["lower"]), float(metrics["upper"])
    if error > 0.0 and upper > lower:
        return dict(metrics), None
    games = int(metrics.get("n", 0))
    floor = plausible_standard_error_floor(reference, games)
    if floor is None or floor <= 0.0:
        return dict(metrics), None
    effect = float(metrics["delta"])
    half = 1.959963984540054 * floor
    low, high = effect - half, effect + half
    if ceiling is not None:
        low, high = max(low, -ceiling), min(high, ceiling)
    floored = dict(metrics)
    floored.update(standard_error=floor, lower=low, upper=high)
    return floored, (
        f"Degenerate cell: the paired difference is identical on all {games} games, so the "
        f"block bootstrap returns a zero-width band that no re-measurement can widen. Per "
        f"docs/weak_signal_pooling.md (defect 4, 2026-09-08) the band is FLOORED to what this "
        f"sample size supports -- standard error {floor:.4f}, from this family's own "
        f"median SE^2*n curve at three robust sigmas -- and the row is kept, not dropped. "
        f"The point estimate is the measurement and is unchanged."
    )


DISCRETE_MARGIN_METHODS = (
    "discrete_conditional_lattice",
    "discrete_conditional_key_neighbourhood",
)


def discrete_conditional_cover_probability(
    residuals: FloatArray,
    centers: FloatArray | pd.Series,
    lines: FloatArray | pd.Series,
    history: pd.DataFrame,
    *,
    method: str = "discrete_conditional_lattice",
    atoms: Iterable[float] = SERVED_ATOMS,
    half_width: float = NEIGHBOURHOOD_HALF_WIDTH,
    band_half_width: float = BAND_HALF_WIDTH,
    min_band_games: int = MIN_BAND_GAMES,
    smooth: FloatArray | None = None,
) -> FloatArray:

    if method not in DISCRETE_MARGIN_METHODS:
        raise ValueError(f"Unknown discrete margin method: {method}")
    required = {"spread_line", "result"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"Discrete margin history is missing columns: {', '.join(missing)}")
    prior = history[["spread_line", "result"]].to_numpy(dtype=float)
    prior = prior[np.isfinite(prior).all(axis=1)]
    if len(prior) == 0:
        raise ValueError("The discrete conditional read requires prior completed games")
    if not np.equal(prior[:, 1], np.round(prior[:, 1])).all():
        raise ValueError("Actual margins must be integers")
    location = residual_location(np.asarray(residuals, dtype=float), "gaussian_median")
    center_values = np.asarray(centers, dtype=float)
    line_values = np.asarray(lines, dtype=float)
    probabilities = np.array(
        [
            discrete_side_read(
                prior[:, 0],
                prior[:, 1],
                float(line),
                float(center) + location,
                half_width=band_half_width,
                min_band_games=min_band_games,
            ).home_cover_probability
            for center, line in zip(center_values, line_values, strict=True)
        ],
        dtype=np.float64,
    )
    if method == "discrete_conditional_lattice":
        return probabilities
    if smooth is None:
        raise ValueError(
            "discrete_conditional_key_neighbourhood needs the incumbent smooth probabilities "
            "for the games it does not touch"
        )
    touched = key_neighbourhood_mask(line_values, atoms, half_width)
    return np.asarray(
        np.where(touched, probabilities, np.asarray(smooth, dtype=float)), dtype=np.float64
    )
