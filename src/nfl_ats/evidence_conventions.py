from __future__ import annotations

import math
from typing import Any, overload

import numpy as np
import numpy.typing as npt

__all__ = [
    "ZERO_ATOM_CREDIT",
    "binomial_two_sided_p",
    "probability_positive_from_draws",
]

ZERO_ATOM_CREDIT = 0.5


@overload
def probability_positive_from_draws(
    draws: Any, *, axis: None = ..., ignore_nan: bool = ...
) -> float: ...


@overload
def probability_positive_from_draws(
    draws: Any, *, axis: int, ignore_nan: bool = ...
) -> npt.NDArray[np.float64]: ...


def probability_positive_from_draws(
    draws: Any, *, axis: int | None = None, ignore_nan: bool = False
) -> float | npt.NDArray[np.float64]:

    values = np.asarray(draws, dtype=float)
    credit = (values > 0.0).astype(float) + ZERO_ATOM_CREDIT * (values == 0.0).astype(float)
    if ignore_nan:
        credit = np.where(np.isfinite(values), credit, np.nan)
        reducer = np.nanmean
    else:
        reducer = np.mean
    if axis is None:
        if credit.size == 0:
            return float("nan")
        return float(reducer(credit))
    return np.asarray(reducer(credit, axis=axis), dtype=np.float64)


def binomial_two_sided_p(favourable: int, total: int) -> float:

    if total <= 0:
        return 1.0
    if favourable < 0 or favourable > total:
        raise ValueError(f"favourable={favourable} is outside 0..{total}")

    observed_distance = abs(2 * favourable - total)
    log_half = total * math.log(0.5)
    log_total_factorial = math.lgamma(total + 1)

    def log_pmf(k: int) -> float:
        return log_total_factorial - math.lgamma(k + 1) - math.lgamma(total - k + 1) + log_half

    return min(
        1.0,
        math.fsum(
            math.exp(log_pmf(k))
            for k in range(total + 1)
            if abs(2 * k - total) >= observed_distance
        ),
    )
