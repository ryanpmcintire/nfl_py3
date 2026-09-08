"""Home-side-conditioned probability shapes around the incumbent point (MOD-18 lane T).

Both mappings leave the served point prediction alone and condition only the
cover probability on which side of the line the home team is: a home
favourite (positive home spread), a home underdog (negative) or a pick'em.
Predeclared in ``docs/home_side_mapping.md``; the shrinkage weight and the
support floor below are declared regularisation, not fitted parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from nfl_ats.conditional_margin import (
    BANDWIDTH,
    ConditionalMarginLattice,
    fit_conditional_margin,
)
from nfl_ats.spread_regime import spread_bucket

HOME_SIDE_MAPPING_METHODS = ("smooth_home_side_shift", "lattice_home_side")
SHIFT_PRIOR_GAMES = 100
LATTICE_SIDE_SUPPORT = 200
SUPPORT_HALF_WIDTH = 2 * BANDWIDTH
HOME_SIDES = ("home_favourite", "home_underdog", "pickem")


def home_side(lines: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """Positive home spread means the home team is favoured (nflverse sign)."""
    values = np.atleast_1d(np.asarray(lines, dtype=float))
    return np.asarray(
        np.where(values > 0, HOME_SIDES[0], np.where(values < 0, HOME_SIDES[1], HOME_SIDES[2])),
        dtype=str,
    )


def shift_cell(lines: pd.Series) -> pd.Series:
    """Home side crossed with the lane-J spread band, e.g. ``home_underdog/7.5-10``."""
    numeric = pd.to_numeric(lines, errors="raise")
    sides = pd.Series(home_side(numeric.to_numpy()), index=numeric.index)
    return sides + "/" + spread_bucket(numeric).astype(str)


@dataclass(frozen=True)
class HomeSideShift:
    """Shrunken mean residual per home-side/spread-band cell from prior games only."""

    shifts: dict[str, float]
    counts: dict[str, int]
    history_rows: int

    def shift(self, lines: npt.ArrayLike) -> npt.NDArray[np.float64]:
        cells = shift_cell(pd.Series(np.atleast_1d(np.asarray(lines, dtype=float))))
        return np.asarray([self.shifts.get(cell, 0.0) for cell in cells], dtype=np.float64)

    def rows(self, lines: npt.ArrayLike) -> npt.NDArray[np.int64]:
        cells = shift_cell(pd.Series(np.atleast_1d(np.asarray(lines, dtype=float))))
        return np.asarray([self.counts.get(cell, 0) for cell in cells], dtype=np.int64)


def fit_home_side_shift(history: pd.DataFrame) -> HomeSideShift:
    """``predicted_margin`` must be the incumbent mapping centre of an OOS forecast."""
    values = history[["spread_line", "result", "predicted_margin"]].apply(
        pd.to_numeric, errors="raise"
    )
    valid = values.loc[np.isfinite(values).all(axis=1)]
    residual = valid.result - valid.predicted_margin
    grouped = residual.groupby(shift_cell(valid.spread_line)).agg(["sum", "count"])
    shifts = {
        str(cell): float(row["sum"] / (row["count"] + SHIFT_PRIOR_GAMES))
        for cell, row in grouped.iterrows()
    }
    counts = {str(cell): int(row["count"]) for cell, row in grouped.iterrows()}
    return HomeSideShift(shifts, counts, len(valid))


def shifted_probability(
    smooth_probability: npt.ArrayLike, shift: npt.ArrayLike, scale: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Re-read a Gaussian cover probability after moving its centre by ``shift``.

    Identical to evaluating the Gaussian at ``centre + shift`` with the same
    ``scale``; expressed on the probability so an archived read can be shifted
    without reconstructing the residual sample.
    """
    p = np.clip(np.asarray(smooth_probability, dtype=float), 1e-12, 1 - 1e-12)
    z = stats.norm.ppf(p) + np.asarray(shift, dtype=float) / np.asarray(scale, dtype=float)
    return np.asarray(stats.norm.cdf(z), dtype=np.float64)


@dataclass(frozen=True)
class HomeSideLattice:
    """K1 lattice fitted on same-side prior games, or on all of them as fallback."""

    lattice: ConditionalMarginLattice
    used_side: bool
    side_support: int

    def probabilities(self, line: float) -> tuple[float, float, float]:
        return self.lattice.probabilities(line)

    def decision_probability(self, line: float) -> float:
        return self.lattice.decision_probability(line)


def fit_lattice_home_side(history: pd.DataFrame, center: float, line: float) -> HomeSideLattice:
    """Fit an already cutoff-filtered OOS history; no outcomes from target rows."""
    if not np.isfinite([center, line]).all():
        raise ValueError("Center and line must be finite")
    sides = home_side(history.spread_line.to_numpy(dtype=float))
    same = history.loc[sides == home_side(np.array([line]))[0]]
    predicted = pd.to_numeric(same.predicted_margin, errors="raise")
    support = int(predicted.sub(center).abs().le(SUPPORT_HALF_WIDTH).sum())
    used_side = support >= LATTICE_SIDE_SUPPORT
    lattice = fit_conditional_margin(same if used_side else history, center, line)
    return HomeSideLattice(lattice, used_side, support)


def _eligible(prior: pd.DataFrame, batch: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return prior.loc[
        (prior.gameday + pd.Timedelta(days=1)).lt(batch.gameday.min())
        & prior.season.ge(season - 4)
        & ~((prior.season == season) & (prior.week == week))
        & prior.result.notna()
        & ~prior.game_id.isin(batch.game_id)
    ]


def predict_home_side_mapping(
    history: pd.DataFrame, targets: pd.DataFrame, *, method: str = "smooth_home_side_shift"
) -> pd.DataFrame:
    """Fit a whole week before its first game, with a one-day completion allowance.

    ``history`` rows carry ``game_id, season, week, gameday, spread_line, result,
    predicted_margin`` (the mapping centre of an out-of-time forecast).
    ``smooth_home_side_shift`` targets also need ``p_smooth`` (the incumbent
    read) and ``smooth_scale`` (its Gaussian scale); ``lattice_home_side``
    targets need ``predicted_margin``.
    """
    if method not in HOME_SIDE_MAPPING_METHODS:
        raise ValueError(f"Unknown home-side mapping method: {method}")
    prior = history.copy()
    prior["gameday"] = pd.to_datetime(prior.gameday)
    if prior.game_id.duplicated().any():
        raise ValueError("History requires unique game ids")
    result = targets.copy()
    result["gameday"] = pd.to_datetime(result.gameday)
    outputs = []
    for (season, week), batch in result.groupby(["season", "week"], sort=True):
        eligible = _eligible(prior, batch, int(str(season)), int(str(week)))
        if method == "smooth_home_side_shift":
            fitted = fit_home_side_shift(eligible)
            shift = fitted.shift(batch.spread_line.to_numpy(dtype=float))
            probability = shifted_probability(batch.p_smooth, shift, batch.smooth_scale)
            for index, s, n, p in zip(
                batch.index,
                shift,
                fitted.rows(batch.spread_line.to_numpy(dtype=float)),
                probability,
                strict=True,
            ):
                outputs.append(
                    {
                        "index": index,
                        "home_cover_probability": float(p),
                        "home_side_shift": float(s),
                        "home_side_shift_rows": int(n),
                        "home_side_history_rows": fitted.history_rows,
                        "home_side_max_gameday": eligible.gameday.max(),
                    }
                )
            continue
        for index, row in batch.iterrows():
            fitted_lattice = fit_lattice_home_side(
                eligible, float(row.predicted_margin), float(row.spread_line)
            )
            cover, push, loss = fitted_lattice.probabilities(float(row.spread_line))
            outputs.append(
                {
                    "index": index,
                    "home_cover_probability": cover + 0.5 * push,
                    "home_cover_probability_excluding_push": cover,
                    "push_probability": push,
                    "home_loss_probability": loss,
                    "lattice_side_used": fitted_lattice.used_side,
                    "lattice_side_support": fitted_lattice.side_support,
                    "lattice_effective_rows": fitted_lattice.lattice.effective_rows,
                    "home_side_history_rows": len(eligible),
                    "home_side_max_gameday": eligible.gameday.max(),
                }
            )
    mapped = pd.DataFrame(outputs).set_index("index")
    for column in mapped:
        result[column] = mapped[column].reindex(result.index)
    return result
