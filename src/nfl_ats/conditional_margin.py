"""Prior-only conditional empirical integer margins (MOD-18 lane K)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

CONDITIONAL_MARGIN_METHODS = (
    "conditional_margin_lattice",
    "conditional_margin_lattice_keyshift",
    "conditional_margin_lattice_keyside",
)
BANDWIDTH = 2.5


def key_side(lines: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Nearest absolute key and under/on/over; ties choose the smaller key."""
    keys = np.array([3, 7, 10, 14])
    values = np.abs(lines)
    nearest = np.argmin(np.abs(values[:, None] - keys), axis=1)
    return np.asarray(3 * nearest + np.sign(values - keys[nearest]) + 1, dtype=np.int64)


@dataclass(frozen=True)
class ConditionalMarginLattice:
    """Weighted atoms on integer margins, never a pooled residual cloud."""

    margins: npt.NDArray[np.float64]
    mass: npt.NDArray[np.float64]
    effective_rows: float
    used_keyside: bool = False
    shift: int = 0

    def probabilities(self, line: float) -> tuple[float, float, float]:
        """Strict home cover, push and strict away cover, summing to one."""
        return (
            float(self.mass[self.margins > line].sum()),
            float(self.mass[self.margins == line].sum()),
            float(self.mass[self.margins < line].sum()),
        )

    def decision_probability(self, line: float) -> float:
        cover, push, _ = self.probabilities(line)
        return cover + 0.5 * push


def fit_conditional_margin(
    history: pd.DataFrame,
    center: float,
    line: float,
    *,
    method: str = "conditional_margin_lattice",
) -> ConditionalMarginLattice:
    """Fit an already cutoff-filtered OOS history; no outcomes from target rows."""
    from nfl_ats.tiebreaker import effective_sample_size

    if method not in CONDITIONAL_MARGIN_METHODS:
        raise ValueError(f"Unknown conditional margin method: {method}")
    if not np.isfinite([center, line]).all():
        raise ValueError("Center and line must be finite")
    values = history[["predicted_margin", "result", "spread_line"]].to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0:
        raise ValueError("Conditional margin mapping requires prior completed predictions")
    if not np.equal(values[:, 1], np.round(values[:, 1])).all():
        raise ValueError("Actual margins must be integers")
    distances = ((values[:, 0] - center) / BANDWIDTH) ** 2
    weights = np.exp(-0.5 * (distances - distances.min()))
    used_keyside = False
    if method == "conditional_margin_lattice_keyside":
        selected = key_side(values[:, 2]) == key_side(np.array([line]))[0]
        cell_weights = weights * selected
        if selected.sum() >= 50 and effective_sample_size(cell_weights) >= 30:
            weights = cell_weights
            used_keyside = True
    margins, inverse = np.unique(values[:, 1], return_inverse=True)
    mass = np.bincount(inverse, weights=weights).astype(float)
    mass /= mass.sum()
    shift = 0
    if method == "conditional_margin_lattice_keyshift":
        median = margins[np.searchsorted(np.cumsum(mass), 0.5)]
        shift = int(np.round(center - median))
        margins = margins + shift
    return ConditionalMarginLattice(
        margins, mass, effective_sample_size(weights), used_keyside, shift
    )


def predict_conditional_margin(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    method: str = "conditional_margin_lattice",
) -> pd.DataFrame:
    """Filter to completed games strictly before each whole prediction week.

    Inputs are OOS point predictions; a conservative one-day completion allowance
    excludes same-day games when the source provides dates without end times.
    """
    prior = history.copy()
    prior["gameday"] = pd.to_datetime(prior.gameday)
    if prior.game_id.duplicated().any():
        raise ValueError("History requires unique game ids")
    result = targets.copy()
    result["gameday"] = pd.to_datetime(result.gameday)
    outputs = []
    for (season, week), batch in result.groupby(["season", "week"], sort=True):
        cutoff = batch.gameday.min()
        eligible = prior.loc[
            (prior.gameday + pd.Timedelta(days=1)).lt(cutoff)
            & prior.season.ge(int(str(season)) - 4)
            & ~((prior.season == season) & (prior.week == week))
            & prior.result.notna()
            & ~prior.game_id.isin(batch.game_id)
        ]
        for index, row in batch.iterrows():
            lattice = fit_conditional_margin(
                eligible, float(row.predicted_margin), float(row.spread_line), method=method
            )
            cover, push, loss = lattice.probabilities(float(row.spread_line))
            outputs.append(
                {
                    "index": index,
                    "home_cover_probability": cover + 0.5 * push,
                    "home_cover_probability_excluding_push": cover,
                    "push_probability": push,
                    "home_loss_probability": loss,
                    "conditional_effective_rows": lattice.effective_rows,
                    "conditional_keyside_used": lattice.used_keyside,
                    "conditional_shift": lattice.shift,
                    "conditional_history_rows": len(eligible),
                    "conditional_max_gameday": eligible.gameday.max(),
                }
            )
    mapped = pd.DataFrame(outputs).set_index("index")
    for column in mapped:
        result[column] = mapped[column].reindex(result.index)
    return result
