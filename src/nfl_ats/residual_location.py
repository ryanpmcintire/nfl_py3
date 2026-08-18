"""Recency weighting and location shrinkage for the out-of-time residual ECDF.

Research item: ``docs/residual_location.md``. ``margin.MarginModel`` builds its
predictive distribution for every game by adding a fixed out-of-time residual
SAMPLE (the trailing 20% chronological holdout computed in
``fit_margin_model``/``fit_cfb_residual_model``) to that game's predicted
centre, then reads ``home_cover_probability`` off the discretized empirical
CDF (``margin._smoothed_probability``, a Laplace/Krichevsky-Trofimov
continuity-corrected count). The pool's forced pick is
``home_cover_probability >= 0.5`` (``nfl_ats.pool.build_ats_pool_card``,
``nfl_ats.backtest.py``), which is the empirical MEDIAN of
``center + residuals`` compared against the line, not
``sign(predicted_market_residual)``. Because the trailing residual sample has
a nonzero, sampling-noisy location, those two rules disagree on a measurable
share of games (11.8% on the 2018-2025 backtest), and the production rule
wins by +2.12 accuracy points -- entirely because of that location. Nobody
has ever modelled the location: it is currently an unweighted mean/median of
whatever the trailing-20% split happens to contain.

This module adds two OPT-IN alternative readers of the SAME residual draws:

- ``recency_weighted_survival`` / ``recency_weighted_home_cover_probability``
  -- an exponentially recency-weighted ECDF. Games within the residual sample
  are weighted by ``0.5 ** (games_ago / half_life_games)``, so the most
  recent draw gets weight 1 and weight halves every ``half_life_games``
  games back. ``half_life_games -> inf`` recovers the current unweighted
  ECDF exactly.
- ``shrunk_survival`` / ``shrunk_home_cover_probability`` -- shrinks the
  residual sample's location toward zero by a fraction before reading the
  ECDF. ``shrink_fraction=0`` recovers the current production probability
  exactly; ``shrink_fraction=1`` fully removes the location correction
  (residuals become the sample re-centered at zero), which is the shrinkage
  screen's connection to the already-measured cost of dropping the location
  correction entirely (production vs. ``sign(predicted_market_residual)``,
  +2.12 accuracy points, ``docs/pool_edge_plan.md``).

Neither function touches ``margin.py`` or ``calibration.py``; both take the
same ``residuals``/``centers``/``lines`` inputs ``calibration.
smoothed_home_cover_probability`` does, so the CFB screen in
``scripts/residual_location_screen.py`` can compare all three families
(ecdf control, density smoothing, and this module's recency/shrinkage
candidates) on identical weeks and mean models. The frozen active model is
unchanged whether or not this module is imported -- it is never called from
the production prediction path.

Residual ordering contract: ``MarginModel.residuals`` (and
``fit_cfb_residual_model``'s output) is built from a frame sorted ascending
by ``(gameday, game_id)`` before slicing the trailing holdout, so index 0 is
the OLDEST draw in the holdout and index ``n - 1`` is the MOST RECENT. Every
function below assumes that ordering; passing a shuffled array silently
produces a meaningless recency weighting (shrinkage is order-independent).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_CONTINUITY_EPSILON = 0.5


def _continuity_corrected(
    success_weight: npt.NDArray[np.float64] | float, total_weight: float
) -> npt.NDArray[np.float64]:
    """Laplace/KT continuity correction, generalized to weighted counts.

    Reproduces ``margin._smoothed_probability``'s
    ``(successes + 0.5) / (n + 1)`` exactly when every weight is 1. Always
    returns an array so every caller's vectorized-over-thresholds contract is
    uniform, even for a single scalar threshold.
    """

    successes = np.asarray(success_weight, dtype=np.float64)
    return (successes + _CONTINUITY_EPSILON) / (total_weight + 1.0)


def recency_weights(n: int, *, half_life_games: float) -> npt.NDArray[np.float64]:
    """Exponential recency weights for a length-``n`` chronological sample.

    Index 0 (oldest draw) gets the smallest weight; index ``n - 1`` (most
    recent draw) gets weight exactly 1. ``half_life_games`` must be finite
    and positive; use plain ``_smoothed_probability``/the ``ecdf`` control
    for "no weighting" rather than an infinite half-life.
    """

    if n <= 0:
        raise ValueError("recency_weights requires a positive sample size")
    if not np.isfinite(half_life_games) or half_life_games <= 0.0:
        raise ValueError("half_life_games must be finite and positive")
    games_ago = np.arange(n - 1, -1, -1, dtype=np.float64)
    return np.asarray(np.exp(-np.log(2.0) * games_ago / half_life_games), dtype=np.float64)


def recency_weighted_survival(
    residuals: npt.NDArray[np.float64],
    threshold: npt.NDArray[np.float64] | float,
    *,
    half_life_games: float,
) -> npt.NDArray[np.float64]:
    """P(residual > threshold) under exponential recency weighting.

    Vectorized over one or many thresholds, mirroring
    ``calibration.ResidualSmoother.survival``.
    """

    values = np.asarray(residuals, dtype=np.float64)
    weights = recency_weights(len(values), half_life_games=half_life_games)
    total_weight = float(weights.sum())
    thresholds = np.atleast_1d(np.asarray(threshold, dtype=np.float64))
    success_weight = np.array(
        [float(weights[values > t].sum()) for t in thresholds], dtype=np.float64
    )
    return _continuity_corrected(success_weight, total_weight)


def recency_weighted_home_cover_probability(
    residuals: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    lines: npt.NDArray[np.float64],
    *,
    half_life_games: float,
) -> npt.NDArray[np.float64]:
    """Home-cover probability under a recency-weighted residual ECDF.

    Mirrors ``calibration.smoothed_home_cover_probability``: ``centers`` is
    each game's predicted margin, ``lines`` its quoted spread, and
    ``residuals`` the same out-of-time draws the frozen model would add to
    that centre (in their original chronological order).
    """

    thresholds = np.asarray(lines, dtype=np.float64) - np.asarray(centers, dtype=np.float64)
    return recency_weighted_survival(residuals, thresholds, half_life_games=half_life_games)


def location_offset(residuals: npt.NDArray[np.float64], *, statistic: str = "mean") -> float:
    """The residual sample's location, as either its mean or its median."""

    values = np.asarray(residuals, dtype=np.float64)
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown location statistic {statistic!r}; choose 'mean' or 'median'")


def shrunk_survival(
    residuals: npt.NDArray[np.float64],
    threshold: npt.NDArray[np.float64] | float,
    *,
    shrink_fraction: float,
    statistic: str = "mean",
) -> npt.NDArray[np.float64]:
    """P(residual > threshold) after shrinking the sample's location toward zero.

    ``shrink_fraction=0`` reproduces ``margin._smoothed_probability`` exactly
    (the current production reading); ``shrink_fraction=1`` removes the
    location correction entirely by re-centering the sample at zero before
    reading the ECDF.
    """

    if not 0.0 <= shrink_fraction <= 1.0:
        raise ValueError("shrink_fraction must be between 0 and 1")
    values = np.asarray(residuals, dtype=np.float64)
    location = location_offset(values, statistic=statistic)
    shifted = values - shrink_fraction * location
    thresholds = np.atleast_1d(np.asarray(threshold, dtype=np.float64))
    successes = np.array(
        [float(np.count_nonzero(shifted > t)) for t in thresholds], dtype=np.float64
    )
    return _continuity_corrected(successes, float(len(shifted)))


def shrunk_home_cover_probability(
    residuals: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    lines: npt.NDArray[np.float64],
    *,
    shrink_fraction: float,
    statistic: str = "mean",
) -> npt.NDArray[np.float64]:
    """Home-cover probability after shrinking the residual location toward zero.

    Mirrors ``calibration.smoothed_home_cover_probability``, see
    ``shrunk_survival`` for the shrinkage contract.
    """

    thresholds = np.asarray(lines, dtype=np.float64) - np.asarray(centers, dtype=np.float64)
    return shrunk_survival(
        residuals, thresholds, shrink_fraction=shrink_fraction, statistic=statistic
    )
