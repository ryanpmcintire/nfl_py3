"""Shared conventions for turning bootstrap draws into reported evidence.

One module, one place to change, because the two conventions below were
previously copy-pasted across ~100 call sites in ``src/`` and ``scripts/`` and
drifted into a defect that only ever pointed one way: it turned measurements
that said NOTHING into resolved-looking negatives.

Two things live here.

``probability_positive_from_draws`` -- the zero-atom convention
---------------------------------------------------------------
Every screen in this repository reports ``probability_positive``: the share of
block-bootstrap resamples in which the candidate beat the baseline. It was
computed as ``np.mean(draws > 0.0)``, a strict inequality, and that is wrong
for the quantity actually being measured.

A paired accuracy delta is ``(candidate wins - baseline wins) / games`` over
the handful of games where the two arms DISAGREE. That statistic has a large
atom of probability sitting at exactly zero -- and in the limiting case, a
candidate that makes IDENTICAL picks on every game puts ALL of its mass there.
Under the strict ``>`` such a candidate records ``probability_positive = 0.0``:
the strongest negative the scale can express, awarded to a no-op. That is not
a measurement, it is an arithmetic accident, and it is exactly the "an
interval crossing zero is not grounds for rejection" failure the project's
binding invariant exists to prevent, arriving through the back door.

The convention here splits the zero atom evenly, because "no difference at
all" is evidence for neither arm:

    P+ = P(draws > 0) + ZERO_ATOM_CREDIT * P(draws == 0)

so a pure no-op scores 0.5 -- "this told us nothing" -- and a candidate is only
pushed below 0.5 by resamples in which it actually LOST. NaN draws are counted
in the denominator and credited nothing, matching the previous behaviour
exactly (``nan > 0`` and ``nan == 0`` are both False).

``binomial_two_sided_p`` -- the sign-test p-value, in log space
---------------------------------------------------------------
The exact two-sided binomial p-value against a fair coin, summing every
outcome no more likely than the observed one. The previous implementation
evaluated ``math.comb(total, k) * 0.5**total`` directly: ``math.comb`` is an
exact int, ``0.5**total`` underflows to 0.0, and the int-to-float conversion
overflows above ~1.8e308. Measured 2026-09-08: fine at n=1020, ``OverflowError``
at n>=1030. The eligible NFL pool is n=1489, so ``nfl-ats weak-signals pool``
-- the command AGENTS.md tells every session to re-run -- was unexecutable.

The construction is unchanged; only the arithmetic moved. For p = 0.5 the
pmf is symmetric about n/2, so "no more likely than observed" is the exact
integer predicate ``|2k - n| >= |2*observed - n|`` -- no floating-point
tolerance is needed at all -- and the surviving terms are summed from
``math.lgamma`` in log space, where each term is a probability in [0, 1] and
cannot overflow.
"""

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

#: How much credit a resample of exactly zero gives the candidate. Half,
#: because an exact tie is evidence for neither arm. See the module docstring.
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
    """Share of resamples favouring the candidate, splitting the zero atom.

    ``P(draws > 0) + 0.5 * P(draws == 0)``. Use this everywhere instead of
    ``np.mean(draws > 0.0)``; see the module docstring for why the strict
    inequality is a defect rather than a stylistic choice.

    Args:
        draws: Resampled values of a candidate-minus-baseline quantity, under
            this repository's universal sign convention that positive favours
            the candidate.
        axis: Reduce along this axis (typically ``0``, one column per metric)
            and return an array. ``None`` reduces everything to one float.
        ignore_nan: Drop non-finite draws from the denominator instead of
            counting them against the candidate. Set it wherever the previous
            code used ``np.nanmean``, so a resample that failed to produce a
            number stays a failed resample rather than becoming a loss.

    Returns:
        A float in [0, 1] when ``axis`` is ``None``, otherwise an array of
        them. An empty ``draws`` with ``axis=None`` returns NaN rather than
        inventing a probability.
    """

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
    """Exact two-sided binomial p-value against a fair coin, computed in log space.

    Sums every outcome no more likely than the observed one. Safe at any
    ``total``; the direct-pmf version this replaces raised ``OverflowError``
    above n ~= 1030. Reproduces the value AGENTS.md quotes for the NFL pool
    (327 of 628 -> 0.31846963) exactly.
    """

    if total <= 0:
        return 1.0
    if favourable < 0 or favourable > total:
        raise ValueError(f"favourable={favourable} is outside 0..{total}")

    # For p = 0.5 the pmf is symmetric about total/2, so "no more likely than
    # the observed outcome" is the exact integer predicate below -- no
    # floating-point tolerance, and therefore no tie-inclusion ambiguity.
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
