from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

HYBRID_MARGIN_METHODS = ("hybrid_key_distance", "hybrid_key_distance_by_size")
GRID = tuple((w, t) for w in (0.0, 0.25, 0.5, 0.75, 1.0) for t in (0.5, 1.0, 2.0, 4.0))


def key_distance(lines: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.atleast_1d(np.asarray(lines, dtype=float))
    return np.min(np.abs(np.abs(values)[:, None] - [3, 7, 10, 14]), axis=1)


def size_cell(lines: npt.ArrayLike) -> npt.NDArray[np.int64]:
    return np.asarray(np.digitize(np.abs(lines), [3, 7], right=True), dtype=np.int64)


@dataclass(frozen=True)
class HybridWeights:
    global_pair: tuple[float, float]
    size_pairs: tuple[tuple[float, float], ...] = ()
    history_rows: int = 0

    def weights(self, lines: npt.ArrayLike) -> npt.NDArray[np.float64]:
        distances = key_distance(lines)
        pairs = np.array(self.size_pairs or (self.global_pair,) * 3)[size_cell(lines)]
        return np.asarray(pairs[:, 0] * np.exp(-distances / pairs[:, 1]), dtype=np.float64)


def fit_hybrid_weights(
    history: pd.DataFrame, *, method: str = "hybrid_key_distance"
) -> HybridWeights:
    if method not in HYBRID_MARGIN_METHODS:
        raise ValueError(f"Unknown hybrid margin method: {method}")
    valid = history.dropna(subset=["result", "spread_line", "p_smooth", "p_lattice"])
    valid = valid.loc[valid.result.ne(valid.spread_line)]
    if not np.isfinite(valid[["result", "spread_line", "p_smooth", "p_lattice"]]).all().all():
        raise ValueError("Hybrid history must be finite")
    if not valid[["p_smooth", "p_lattice"]].apply(lambda x: x.between(0, 1)).all().all():
        raise ValueError("Hybrid history probabilities must be in [0, 1]")

    def best(rows: pd.DataFrame) -> tuple[float, float]:
        if len(rows) < 200:
            return GRID[0]
        pairs = np.array(GRID)
        weights = pairs[:, 0] * np.exp(-key_distance(rows.spread_line)[:, None] / pairs[:, 1])
        smooth = rows.p_smooth.to_numpy()[:, None]
        lattice = rows.p_lattice.to_numpy()[:, None]
        truth = rows.result.gt(rows.spread_line).to_numpy()[:, None]
        loss = np.mean((smooth + weights * (lattice - smooth) - truth) ** 2, axis=0)
        return GRID[int(np.argmin(loss))]

    global_pair = best(valid)
    cells = [valid.loc[size_cell(valid.spread_line) == i] for i in range(3)]
    supported = method.endswith("by_size") and all(
        len(cell) >= 200 and cell.season.nunique() >= 3 for cell in cells
    )
    return HybridWeights(
        global_pair, tuple(best(c) for c in cells) if supported else (), len(valid)
    )


def predict_hybrid_margin(
    history: pd.DataFrame, targets: pd.DataFrame, *, method: str = "hybrid_key_distance"
) -> pd.DataFrame:
    prior = history.copy()
    prior["gameday"] = pd.to_datetime(prior.gameday)
    if prior.game_id.duplicated().any():
        raise ValueError("History requires unique game ids")
    result = targets.copy()
    result["gameday"] = pd.to_datetime(result.gameday)
    for (season, week), batch in result.groupby(["season", "week"], sort=True):
        eligible = prior.loc[
            (prior.gameday + pd.Timedelta(days=1)).lt(batch.gameday.min())
            & prior.season.ge(int(str(season)) - 4)
            & ~((prior.season == season) & (prior.week == week))
            & ~prior.game_id.isin(batch.game_id)
        ]
        fitted = fit_hybrid_weights(eligible, method=method)
        weights = fitted.weights(batch.spread_line)
        result.loc[batch.index, "hybrid_weight"] = weights
        result.loc[batch.index, "home_cover_probability"] = (
            weights * batch.p_lattice + (1 - weights) * batch.p_smooth
        )
        result.loc[batch.index, "hybrid_history_rows"] = fitted.history_rows
        result.loc[batch.index, "hybrid_size_supported"] = bool(fitted.size_pairs)
        pairs = np.array(fitted.size_pairs or (fitted.global_pair,) * 3)[
            size_cell(batch.spread_line)
        ]
        result.loc[batch.index, "hybrid_w0"] = pairs[:, 0]
        result.loc[batch.index, "hybrid_tau"] = pairs[:, 1]
    return result
