"""MOD-18 C2: the discrete conditional side read, at every line and near the key numbers.

Predeclaration: ``docs/mod18_discrete_margin_mapping.md``.

The construction is NOT new here. Lane K built it
(``nfl_ats.mass_preserving_lattice.band_read``: the empirical integer-margin
atoms of prior games quoted near the line, kept at their ABSOLUTE positions,
with the model's point entering only as an exponential tilt) and lane S
productionised it as the served push and alternative-line source. What is new
is WHERE the resulting two-way number is allowed to decide the SIDE:

* lane K served it on every game and lost the played card (-0.399 accuracy
  points, ``probability_positive`` 0.311);
* lane T served it only where ``abs(line)`` is exactly 3 or 7 and won it
  (+0.200, 0.7426) -- that arm is what production plays today;
* nothing had measured the middle, and the middle is the only part of the
  axis the owner's pool quotes. Every Splash Sports line is a half point, so
  lane T's exact-match test can never fire on a served card.

The mechanism, named, and measured before this module existed
(``docs/key_line_pick_read.md``): 14.58% of finals land exactly on ``abs(3)``.
On a whole-number line that block is the push; on a half-point line it falls
WHOLLY on one side of the number. Crossing the 3 atom from 2.5 to 3.5 the
empirical home-cover rate drops 7.81 points where the smooth read says 2.72,
and the two reads err in opposite directions on the two sides of the atom. So
the key-number mass matters MORE at the pool's lines, not less, and the
neighbourhood predicate below is a statement about that mass -- never a spread
value chosen because accuracy dipped there (AGENTS.md, "No unexplained
threshold flips on the played card").

Nothing in this module is served. It is the research arm plus two additive
``probability_method`` values; the served card changes only if the
coordinator promotes an arm.
"""

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
    """Distance from ``abs(line)`` to the nearest declared atom, per row."""

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
    """True where the quoted line sits ON, or one half point either side of,
    a declared key number.

    ``half_point_only`` keeps ONLY the adjacent half points and drops the
    atom itself, which makes the mask disjoint from lane T's served
    exact-match read -- the arm that measures the new territory alone.
    """

    distance = distance_to_nearest_atom(lines, atoms)
    if half_point_only:
        return np.asarray(np.abs(distance - float(half_width)) < _TOLERANCE, dtype=bool)
    return np.asarray(distance <= float(half_width) + _TOLERANCE, dtype=bool)


@dataclass(frozen=True)
class ArmSpec:
    """One frozen arm: which games the discrete read is allowed to decide."""

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
    """The whole arm: the discrete read where the arm applies, the baseline
    everywhere else, bit for bit.

    Returns ``touched`` / ``probability`` / ``push``. An untouched game's
    probability is the baseline object itself, never a recomputation of it.
    """

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
    """One game's discrete read, straight through lane K's core.

    Kept as a named entry point so a reader of this module can see that the
    distribution is the served one: ``band_read`` is the same function the
    card's push probability and the line sweep already go through, so an arm
    promoted from here can never disagree with the served push on the lattice.
    """

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
    """Lane K's walk-forward mapping for every target row, imported not copied.

    ``pool`` is :func:`nfl_ats.mass_preserving_lattice.prior_pool`'s frame;
    ``targets`` carries ``game_id``, ``season``, ``week``, ``gameday``,
    ``line`` and ``point``. The window is applied inside, per (season, week),
    so a caller cannot hand this a leaky frame.
    """

    return walk_forward_reads(pool, targets, float(half_width))


def plausible_standard_error_floor(
    reference: Iterable[tuple[float, int]], sample_games: int
) -> float | None:
    """The narrowest standard error a cell of this size can honestly claim.

    This is `docs/weak_signal_pooling.md`'s defect-4 remedy, applied at
    RECORD time instead of at pool time, and it is deliberately the same
    arithmetic as ``nfl_ats.weak_signals._plausibility_curve``: an honest
    estimator's standard error scales as ``sigma / sqrt(n)``, so ``SE^2 * n``
    is roughly constant across a commensurable family. The scale is the
    MEDIAN of ``SE^2 * n`` so a handful of degenerate bands cannot set it,
    and the cutoff is the log-ratios' median minus three MAD-based standard
    deviations -- derived from the family, never picked in advance.

    ``reference`` is the family's own (standard error, sample games) pairs;
    a lane's cells share units, archive, bootstrap and seed, so they are the
    most commensurable pool available. Returns ``None`` when the family is
    too thin to fit a curve, in which case the caller must say so rather
    than invent a band.

    Why this and not a re-measurement: a block bootstrap cannot rescue a cell
    whose paired difference is identical on every game -- every resample
    returns the same number, so the zero width is structural, not sampling
    noise. Flooring keeps the cell in the record at the weakest precision its
    sample supports, which is what the rule asks for. Nothing is dropped and
    nothing is widened to taste.
    """

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
    """Return ``metrics`` with a floored band when its own band has no width.

    A cell is degenerate when its bootstrap standard error is zero or its
    interval has zero width -- which happens when the paired difference is
    identical on every game in the slice, so the registry's
    ``standard_error > 0`` contract refuses it. The point estimate,
    ``probability_positive``, sample size and block count are NEVER touched:
    only the band is widened, to ``+/- 1.96`` floored standard errors around
    the measured effect, clipped at ``ceiling`` where the metric has one.

    The second element names what happened, for the row's notes; it is
    ``None`` when the cell was already admissible.
    """

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
    """``home_cover_probability`` under the discrete conditional distribution.

    ``history`` is the caller's already cutoff-filtered prior stream -- the
    same contract ``nfl_ats.conditional_margin.fit_conditional_margin``
    states, and the same responsibility: at THIS entry point leak-safety
    belongs to the caller, and the walk-forward window helper above is what
    a caller should use to build it. Only ``spread_line`` and integer
    ``result`` are read from it; the atoms are those games' realised margins.

    The point the atoms are tilted to is ``center`` plus the served
    ``gaussian_median`` residual location, which is exactly the served point
    the card's push read already tilts to.

    ``discrete_conditional_key_neighbourhood`` restricts the discrete read to
    lines on, or one half point either side of, ``atoms`` and needs
    ``smooth`` -- the incumbent probability for the untouched games.
    """

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
