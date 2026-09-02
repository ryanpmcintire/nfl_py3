"""Correct-score lattice: a joint distribution over exact NFL final scores.

ROADMAP row MOD-05 measured the key-number lattice and closed with an
instruction rather than a dead end -- "build it for push probability,
alternative-line/half-point questions and correct-score products, **not** for
ATS accuracy". This module is the correct-score-products half of that, aimed
at the one place in the repository that publishes an exact final score:
:mod:`nfl_ats.tiebreaker`'s exact-score mode list.

What it is
----------
The shipped exact-score answer is a *mode list*: count the kernel-weighted
neighborhood's actual finals and print the three heaviest. That is not a
distribution -- it has no mass anywhere it has not literally seen a game, so
it cannot answer "how likely is a push at this line" or "what is the chance
of any final at all".

The lattice keeps the same neighborhood -- the same triangular kernel, the
same ``(1.0, 1.5)`` base bandwidths, the same continuous widening schedule and
the same Kish effective-sample-size floor of 150, all imported from
:mod:`nfl_ats.tiebreaker` rather than reimplemented -- and turns it into a
probability over the integer score lattice:

1. Each neighborhood game contributes its **market residual pair**
   ``(actual_margin - spread_line, actual_total - total_line)``, recentred on
   the target game's guess. In score space, which is where the lattice lives,
   that is ``home' = guess_home + (actual_home - implied_home)`` and likewise
   for the away score -- the same linear bijection
   ``(margin, total) <-> (home, away)``, nothing added or dropped.
2. Each recentred point is spread onto the integer lattice by a product
   triangular kernel of bandwidth exactly one score point per coordinate.
   **That bandwidth is derived, not chosen** (AGENTS.md: underived constants
   are defects): a triangular kernel whose half-width equals the lattice
   spacing is the unique mass-preserving linear interpolation of a real point
   onto that lattice, because ``sum over integers n of max(0, 1 - |n - x|)``
   is exactly 1 for every real ``x``. Each game's kernel weight lands on the
   at-most-four surrounding integer score pairs and none of it is invented.
3. The support is the **feasible score set enumerated from the data itself**
   -- every team score that has occurred in the training finals, crossed with
   itself. NFL scoring makes 1 and 4 unreachable in practice and the finals
   say so on their own; no hand list appears anywhere in this module.
4. Mass that lands outside the support (a negative score, a score above
   anything ever seen) is dropped and the grid renormalised, so the result is
   a proper distribution over feasible finals.

Products
--------
:meth:`ScoreLattice.top_scores` (the correct-score product),
:meth:`ScoreLattice.push_probability` (identically zero at a half-point line,
by construction), :meth:`ScoreLattice.margin_probability` for
alternative-line questions, and :meth:`ScoreLattice.modal_total` /
:meth:`ScoreLattice.median_total` for the closest-total metric. The median is
the |error|-minimising total and the mode is not; both are exposed so a
caller cannot accidentally quote the flattering one.

Declared limitation
-------------------
MOD-05's key-number spikes live in the **raw** margin, and recentring by the
market residual smears them by however much each neighborhood game's line
differs from the target's. :func:`build_lattice` therefore also supports
``recentre=False`` (smoothing without recentring), which is what isolates the
two effects. See ``docs/score_lattice.md`` for the predeclaration and the
measured comparison.
"""

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

#: Total pseudo-observation mass spread uniformly over the feasible support
#: when a lattice is turned into a log-loss-scoreable distribution. One
#: pseudo-count is the minimum smoothing that makes ``-log P(realised)``
#: finite, expressed in the same units as the kernel weights it is added to;
#: it is not a tuned parameter and the same value is used for every arm of
#: every comparison so the scores stay paired.
PSEUDO_OBSERVATIONS = 1.0


def feasible_team_scores(finals: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Every team score that has actually occurred in ``finals``, sorted.

    The feasible score set is read off the data and never written down by
    hand: 1 and 4 are unreachable under NFL scoring and simply do not appear,
    and a walk-forward caller passing a training slice gets the set that was
    knowable at that point in time rather than today's.
    """

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
    """A normalised joint distribution over feasible exact finals.

    ``scores`` is the sorted feasible team-score set ``V``; ``probabilities``
    is the ``len(V) x len(V)`` grid over ``V x V`` with the home score on
    axis 0, summing to 1. ``weights`` is the same grid before normalisation
    (kernel weight, not a count) and ``weight_total`` is its sum -- both are
    kept because the log-loss smoothing in :func:`smoothed_probability` is
    defined on the un-normalised grid.
    """

    scores: npt.NDArray[np.int64]
    probabilities: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    weight_total: float
    effective_size: float
    label: str

    @property
    def support_size(self) -> int:
        """``|S|`` -- the number of feasible ``(home, away)`` cells."""

        return int(self.scores.size) ** 2

    def _index(self, score: int) -> int | None:
        position = int(np.searchsorted(self.scores, score))
        if position >= self.scores.size or int(self.scores[position]) != int(score):
            return None
        return position

    def probability(self, home_score: int, away_score: int) -> float:
        """``P(home = h, away = a)``; 0 outside the feasible support."""

        home_index = self._index(home_score)
        away_index = self._index(away_score)
        if home_index is None or away_index is None:
            return 0.0
        return float(self.probabilities[home_index, away_index])

    def smoothed_probability(self, home_score: int, away_score: int) -> float:
        """``P`` with one pseudo-observation spread uniformly over ``S``.

        ``(W + PSEUDO_OBSERVATIONS / |S|) / (W.sum() + PSEUDO_OBSERVATIONS)``,
        which sums to exactly 1 over ``S`` and is finite everywhere, so
        ``-log`` of it is a usable score. A final outside the support gets the
        same floor an in-support zero-weight cell gets.
        """

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
        """The ``count`` most probable exact finals, ties broken by score.

        The tie rule matches :func:`nfl_ats.tiebreaker.build_report`'s mode
        list exactly (``sorted`` on ``(-probability, home, away)``), so the
        two arms are ordered by the same rule and a comparison between them
        never turns on iteration order.
        """

        if count <= 0:
            raise ValueError("count must be positive")
        flat = self.probabilities.ravel()
        alive = np.flatnonzero(flat > 0.0)
        # Rank by descending probability first, then by (home, away) ascending.
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
        """``P(home + away = T)`` for every feasible total ``T``."""

        totals, mass = self._total_grid()
        return pd.Series(mass, index=pd.Index(totals, name="total"), name="probability")

    def modal_total(self) -> int:
        """The single most probable total; ties go to the smaller total."""

        totals, mass = self._total_grid()
        return int(totals[int(np.argmax(mass))])

    def median_total(self) -> float:
        """The distribution's median total -- the |error|-minimising answer.

        Delegates to :func:`nfl_ats.tiebreaker.weighted_median` so the
        lattice's closest-total answer and the shipped one are produced by the
        same median, including its exact-half-point averaging rule.
        """

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
        """``P(home - away = m)`` for every feasible home margin ``m``."""

        margins, mass = self._margin_grid()
        return pd.Series(mass, index=pd.Index(margins, name="home_margin"), name="probability")

    def margin_probability(self, margin: float) -> float:
        """``P(home margin == margin)``; 0 for a non-integer margin."""

        if not float(margin).is_integer():
            return 0.0
        margins, mass = self._margin_grid()
        position = int(np.searchsorted(margins, int(margin)))
        if position >= margins.size or int(margins[position]) != int(margin):
            return 0.0
        return float(mass[position])

    def push_probability(self, spread_line: float) -> float:
        """``P(push)`` at a home spread quoted positive-home-favored.

        A push is ``home margin == spread_line``, so a half-point line returns
        exactly 0 -- the answer the lattice exists to be able to give.
        """

        return self.margin_probability(spread_line)

    def condition_on_total(self, total: int) -> ScoreLattice:
        """The lattice restricted to finals with ``home + away == total``.

        Used as the walk-forward positive control: conditioning on the
        realised total is a deliberate peek at the outcome, and an instrument
        that cannot detect the hit-rate jump it produces cannot bound anything.
        When the lattice has no mass on that total, feasible cells summing to
        it are given equal mass (still a peek, and it keeps the control from
        silently degenerating); when no feasible cell sums to it, the lattice
        is returned unchanged.
        """

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
    """Mass-preserving triangular interpolation onto the ``scores x scores`` grid.

    Bandwidth is one score point per coordinate, which is the lattice spacing,
    so ``max(0, 1 - |n - x|)`` summed over the integers ``n`` is exactly 1 for
    any real ``x`` and each point's weight is split across the four surrounding
    integer cells without being created or destroyed. Mass falling on an
    infeasible score (or off the board entirely) is dropped here and the caller
    renormalises -- that, not the kernel, is what makes 1 and 4 impossible.
    """

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
    """Turn a kernel-weighted neighborhood into a normalised score lattice.

    ``recentre=True`` (the predeclared primary arm) shifts every neighborhood
    final by the target's market position minus its own -- i.e. it uses the
    ``(margin, total)`` residual from the market rather than the raw final.
    ``recentre=False`` is the declared secondary arm: identical smoothing,
    no recentring, which is what preserves the raw key-number lattice
    MOD-05 measured.
    """

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
    """The lattice for one game, neighborhood and all.

    ``finals`` must already be restricted to whatever history the caller is
    allowed to see -- this function does no chronological filtering of its
    own, and a walk-forward caller is responsible for passing a strict prefix.
    The neighborhood comes from :func:`nfl_ats.tiebreaker._neighborhood`, so
    the lattice and the shipped mode list are always looking at exactly the
    same weighted games.
    """

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


def mode_list_probability(
    counts: dict[tuple[int, int], float],
    scores: npt.NDArray[np.int64],
    home_score: int,
    away_score: int,
) -> float:
    """The mode list's smoothed probability for one exact final.

    The shipped comparator is a weighted *count* table, not a distribution.
    This applies the same one-pseudo-observation smoothing
    :meth:`ScoreLattice.smoothed_probability` applies, over the same feasible
    support, so the two arms are scored by an identical formula and the
    log-loss comparison stays paired.
    """

    support_size = int(scores.size) ** 2
    denominator = sum(counts.values()) + PSEUDO_OBSERVATIONS
    floor = PSEUDO_OBSERVATIONS / float(support_size)
    return (counts.get((int(home_score), int(away_score)), 0.0) + floor) / denominator


def ranked_modes(
    counts: dict[tuple[int, int], float], count: int = 3
) -> tuple[tuple[int, int, float], ...]:
    """The comparator's top finals, ordered by the shipped tie rule.

    Identical to the ``ranked``/``common`` step of
    :func:`nfl_ats.tiebreaker.build_report`; kept here so the evaluation can
    call it on an arbitrary neighborhood without rebuilding a whole report.
    """

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple((home, away, weight) for (home, away), weight in ranked[:count])
