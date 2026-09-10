from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.tiebreaker import (
    _neighborhood,
    market_implied_scores,
    weighted_median,
)

PSEUDO_OBSERVATIONS = 1.0


def feasible_team_scores(finals: pd.DataFrame) -> npt.NDArray[np.int64]:

    values = np.concatenate(
        [
            finals["home_score"].to_numpy(dtype=float),
            finals["away_score"].to_numpy(dtype=float),
        ]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no completed finals to enumerate a feasible score set from")
    return np.unique(np.rint(values).astype(np.int64))


@dataclass(frozen=True)
class ScoreLattice:
    scores: npt.NDArray[np.int64]
    probabilities: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    weight_total: float
    effective_size: float
    label: str

    @property
    def support_size(self) -> int:

        return int(self.scores.size) ** 2

    def _index(self, score: int) -> int | None:
        position = int(np.searchsorted(self.scores, score))
        if position >= self.scores.size or int(self.scores[position]) != int(score):
            return None
        return position

    def probability(self, home_score: int, away_score: int) -> float:

        home_index = self._index(home_score)
        away_index = self._index(away_score)
        if home_index is None or away_index is None:
            return 0.0
        return float(self.probabilities[home_index, away_index])

    def smoothed_probability(self, home_score: int, away_score: int) -> float:

        denominator = self.weight_total + PSEUDO_OBSERVATIONS
        floor = PSEUDO_OBSERVATIONS / float(self.support_size)
        home_index = self._index(home_score)
        away_index = self._index(away_score)
        observed = (
            0.0
            if home_index is None or away_index is None
            else float(self.weights[home_index, away_index])
        )
        return (observed + floor) / denominator

    def top_scores(self, count: int = 3) -> tuple[tuple[int, int, float], ...]:

        if count <= 0:
            raise ValueError("count must be positive")
        flat = self.probabilities.ravel()
        alive = np.flatnonzero(flat > 0.0)
        home_index, away_index = np.divmod(alive, self.scores.size)
        order = sorted(
            range(alive.size),
            key=lambda k: (
                -float(flat[alive[k]]),
                int(self.scores[home_index[k]]),
                int(self.scores[away_index[k]]),
            ),
        )
        return tuple(
            (
                int(self.scores[home_index[k]]),
                int(self.scores[away_index[k]]),
                float(flat[alive[k]]),
            )
            for k in order[:count]
        )

    def _total_grid(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        totals = self.scores[:, None] + self.scores[None, :]
        flat_totals = totals.ravel()
        flat_probabilities = self.probabilities.ravel()
        unique = np.unique(flat_totals)
        mass = np.zeros(unique.size, dtype=float)
        np.add.at(mass, np.searchsorted(unique, flat_totals), flat_probabilities)
        return unique.astype(np.int64), mass

    def total_distribution(self) -> pd.Series:

        totals, mass = self._total_grid()
        return pd.Series(mass, index=pd.Index(totals, name="total"), name="probability")

    def modal_total(self) -> int:

        totals, mass = self._total_grid()
        return int(totals[int(np.argmax(mass))])

    def median_total(self) -> float:

        totals, mass = self._total_grid()
        return weighted_median(totals.astype(float), mass)

    def _margin_grid(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        margins = self.scores[:, None] - self.scores[None, :]
        flat_margins = margins.ravel()
        flat_probabilities = self.probabilities.ravel()
        unique = np.unique(flat_margins)
        mass = np.zeros(unique.size, dtype=float)
        np.add.at(mass, np.searchsorted(unique, flat_margins), flat_probabilities)
        return unique.astype(np.int64), mass

    def margin_distribution(self) -> pd.Series:

        margins, mass = self._margin_grid()
        return pd.Series(mass, index=pd.Index(margins, name="home_margin"), name="probability")

    def margin_probability(self, margin: float) -> float:

        if not float(margin).is_integer():
            return 0.0
        margins, mass = self._margin_grid()
        position = int(np.searchsorted(margins, int(margin)))
        if position >= margins.size or int(margins[position]) != int(margin):
            return 0.0
        return float(mass[position])

    def push_probability(self, spread_line: float) -> float:

        return self.margin_probability(spread_line)

    def condition_on_total(self, total: int) -> ScoreLattice:

        totals = self.scores[:, None] + self.scores[None, :]
        selected = totals == int(total)
        if not selected.any():
            return self
        weights = np.where(selected, self.weights, 0.0)
        if weights.sum() <= 0.0:
            weights = selected.astype(float)
        return ScoreLattice(
            scores=self.scores,
            probabilities=weights / weights.sum(),
            weights=weights,
            weight_total=float(weights.sum()),
            effective_size=self.effective_size,
            label=f"{self.label} | conditioned on total {int(total)}",
        )


def _lattice_weights(
    home_points: npt.NDArray[np.float64],
    away_points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    scores: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:

    size = int(scores.size)
    grid = np.zeros(size * size, dtype=float)
    if home_points.size == 0:
        return grid.reshape(size, size)
    lookup = np.full(int(scores.max()) + 2, -1, dtype=np.int64)
    lookup[scores] = np.arange(size, dtype=np.int64)

    home_floor = np.floor(home_points)
    away_floor = np.floor(away_points)
    home_fraction = home_points - home_floor
    away_fraction = away_points - away_floor
    for home_step, home_share in ((0, 1.0 - home_fraction), (1, home_fraction)):
        home_cell = (home_floor + home_step).astype(np.int64)
        for away_step, away_share in ((0, 1.0 - away_fraction), (1, away_fraction)):
            away_cell = (away_floor + away_step).astype(np.int64)
            share = weights * home_share * away_share
            usable = (
                (share > 0.0)
                & (home_cell >= 0)
                & (home_cell < lookup.size)
                & (away_cell >= 0)
                & (away_cell < lookup.size)
            )
            if not usable.any():
                continue
            home_index = lookup[home_cell[usable]]
            away_index = lookup[away_cell[usable]]
            feasible = (home_index >= 0) & (away_index >= 0)
            if not feasible.any():
                continue
            flat = home_index[feasible] * size + away_index[feasible]
            grid += np.bincount(flat, weights=share[usable][feasible], minlength=size * size)
    return grid.reshape(size, size)


def build_lattice(
    neighborhood: pd.DataFrame,
    weights: npt.NDArray[np.float64],
    guess_margin: float,
    guess_total_line: float,
    scores: npt.NDArray[np.int64],
    *,
    recentre: bool = True,
    effective_size: float = float("nan"),
    label: str = "",
) -> ScoreLattice:

    home_actual = neighborhood["home_score"].to_numpy(dtype=float)
    away_actual = neighborhood["away_score"].to_numpy(dtype=float)
    if recentre:
        guess_home, guess_away = market_implied_scores(guess_margin, guess_total_line)
        row_margin = neighborhood["spread_line"].to_numpy(dtype=float)
        row_total = neighborhood["total_line"].to_numpy(dtype=float)
        implied_home = (row_total + row_margin) / 2.0
        implied_away = (row_total - row_margin) / 2.0
        home_points = guess_home + (home_actual - implied_home)
        away_points = guess_away + (away_actual - implied_away)
    else:
        home_points, away_points = home_actual, away_actual
    grid = _lattice_weights(home_points, away_points, weights, scores)
    total = float(grid.sum())
    if total <= 0.0:
        raise ValueError("lattice has no mass on the feasible support")
    return ScoreLattice(
        scores=scores,
        probabilities=grid / total,
        weights=grid,
        weight_total=total,
        effective_size=effective_size,
        label=label,
    )


def score_lattice(
    finals: pd.DataFrame,
    guess_margin: float,
    guess_total_line: float,
    *,
    scores: npt.NDArray[np.int64] | None = None,
    recentre: bool = True,
) -> ScoreLattice:

    neighborhood = _neighborhood(finals, guess_margin, guess_total_line)
    support = feasible_team_scores(finals) if scores is None else scores
    return build_lattice(
        neighborhood.frame,
        neighborhood.weights,
        guess_margin,
        guess_total_line,
        support,
        recentre=recentre,
        effective_size=neighborhood.effective_size,
        label=neighborhood.label,
    )


_TOTAL_PROXIMITY_TOLERANCES: tuple[float, ...] = (1.0, 2.0)

_NEAR_TIE_DISTANCE: float = 0.5

_MAX_CENTRE_DISTANCE: float = 3.0


def pick_consistent_top_score(
    lattice: ScoreLattice,
    *,
    pick_side: str,
    spread_line: float,
    served_total: float,
    centre_margin: float,
    total_tolerances: tuple[float, ...] = _TOTAL_PROXIMITY_TOLERANCES,
    near_tie_distance: float = _NEAR_TIE_DISTANCE,
    max_centre_distance: float = _MAX_CENTRE_DISTANCE,
) -> tuple[int, int, float, float] | None:

    if pick_side not in ("HOME", "AWAY"):
        raise ValueError(f"pick_side must be 'HOME' or 'AWAY', got {pick_side!r}")
    if not total_tolerances:
        raise ValueError("total_tolerances must be non-empty")
    home_grid, away_grid = np.meshgrid(lattice.scores, lattice.scores, indexing="ij")
    margin_grid = home_grid - away_grid
    total_grid = home_grid + away_grid
    side_admissible = (
        margin_grid > spread_line if pick_side == "HOME" else margin_grid < spread_line
    )
    distance = np.sqrt((margin_grid - centre_margin) ** 2 + (total_grid - served_total) ** 2)
    for tolerance in total_tolerances:
        admissible = side_admissible & (np.abs(total_grid - served_total) <= tolerance)
        if not np.any(admissible):
            continue
        masked_distance = np.where(admissible, distance, np.inf)
        min_distance = float(np.min(masked_distance))
        near_tie = admissible & (masked_distance <= min_distance + near_tie_distance)
        candidates = np.argwhere(near_tie)
        if len(candidates) > 1:
            order = sorted(
                range(len(candidates)),
                key=lambda k: (
                    -float(lattice.probabilities[candidates[k][0], candidates[k][1]]),
                    float(masked_distance[candidates[k][0], candidates[k][1]]),
                    int(lattice.scores[candidates[k][0]]),
                    int(lattice.scores[candidates[k][1]]),
                ),
            )
            candidates = candidates[[order[0]]]
        home_index, away_index = candidates[0]
        chosen_margin = float(lattice.scores[home_index]) - float(lattice.scores[away_index])
        chosen_total = float(lattice.scores[home_index]) + float(lattice.scores[away_index])
        if (
            abs(chosen_margin - centre_margin) > max_centre_distance
            or abs(chosen_total - served_total) > max_centre_distance
        ):
            return None
        return (
            int(lattice.scores[home_index]),
            int(lattice.scores[away_index]),
            float(lattice.probabilities[home_index, away_index]),
            float(tolerance),
        )
    return None


def pick_cover_probability(lattice: ScoreLattice, *, pick_side: str, spread_line: float) -> float:

    if pick_side not in ("HOME", "AWAY"):
        raise ValueError(f"pick_side must be 'HOME' or 'AWAY', got {pick_side!r}")
    home_grid, away_grid = np.meshgrid(lattice.scores, lattice.scores, indexing="ij")
    margin_grid = home_grid - away_grid
    admissible = margin_grid > spread_line if pick_side == "HOME" else margin_grid < spread_line
    return float(lattice.probabilities[admissible].sum())


def mode_list_probability(
    counts: dict[tuple[int, int], float],
    scores: npt.NDArray[np.int64],
    home_score: int,
    away_score: int,
) -> float:

    support_size = int(scores.size) ** 2
    denominator = sum(counts.values()) + PSEUDO_OBSERVATIONS
    floor = PSEUDO_OBSERVATIONS / float(support_size)
    return (counts.get((int(home_score), int(away_score)), 0.0) + floor) / denominator


def ranked_modes(
    counts: dict[tuple[int, int], float], count: int = 3
) -> tuple[tuple[int, int, float], ...]:

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple((home, away, weight) for (home, away), weight in ranked[:count])
