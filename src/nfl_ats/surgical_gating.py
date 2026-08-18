"""Mechanism-based gating for the injury value-lost candidate.

Research item: ``docs/surgical_injury.md``. The candidate is the
``injury_value_lost_narrowed`` construct already isolated in
``docs/injury_value_lost.md`` section 4 (+1.316 accuracy points, `probability_positive`
0.8875, week-blocked 95% [-0.460, +3.247] on the 456-game [2020, 2021] opener
window; split-half reliability 0.87-0.93; survives a market-move control and
a drop-QB stress test). ``docs/estimation_variance.md`` section 5 establishes
that gating a candidate's influence to games where it matters can turn an
`f`-diluted comparison into a resolving one -- there, by gating on how much
the fitted probabilities *disagreed* (``estimation_variance.gate_by_disagreement``).
This module gates on a different, and for this candidate more principled,
axis: the injury CONSTRUCT's own magnitude, not the model's fitted opinion.
Injuries are sparse and lumpy -- most games carry no meaningful value lost,
and in those games the feature can only contribute noise to a fitted
coefficient while still perturbing the pick. The gate below defers to a
baseline pick on exactly those games.

Nothing here is wired into ``margin.py``, ``experiments.py``, any feature
profile, or the active model. Every function is additive and reads only
pregame-available quantities (the two ``diff_injury_*_value_lost`` columns
and a pair of already-made picks) -- never a result, a correctness flag, or
anything settled after kickoff.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

#: The two columns docs/injury_value_lost.md section 4's cleaner isolation (arm D)
#: adds on top of the frozen ``player`` baseline. Both are pregame-available:
#: built from severity x role-share x value-rate, all measurable before kickoff
#: (players.py's ``_injury_value_features``).
VALUE_LOST_DIFF_COLUMNS = (
    "diff_injury_skill_epa_value_lost",
    "diff_injury_defense_disruption_value_lost",
)

#: Frozen threshold, derived once by
#: ``scripts/surgical_value_lost_distribution.py`` on 2026-08-18 from the full
#: leak-safe history (4,431 completed REG games, 2009-2025,
#: ``data/processed/game_features_player_value.parquet``, via
#: ``modeling.regular_season_rows`` + ``result.notna()`` -- the exact filter
#: docs/injury_value_lost.md section 3.1 already used for this table's
#: split-half reliability). It is the CONDITIONAL MEDIAN of
#: ``raw_value_magnitude`` (below): the median taken over games where the
#: magnitude is strictly nonzero. The median is the one fixed quantile with
#: no further researcher degree of freedom (no "why 75 and not 70" to
#: answer); conditioning on nonzero is what keeps that property honest when
#: the point mass at exactly zero is large (29.86% of NFL games here; 55.7%
#: for this recipe's CFB analog, where the UNCONDITIONAL median is exactly
#: zero and therefore unusable as a threshold at all -- see
#: ``scripts/surgical_cfb_recipe_validation.py``). The conditional median is
#: the one definition of "the typical positive reading" that is well-defined
#: in both leagues without adapting the rule after seeing which league it is
#: applied to.
#:
#: NOT derived by sweeping candidate thresholds against any accuracy outcome.
#: Pinned exactly in tests/test_surgical_gating.py so it can never silently
#: drift toward a re-tuned value.
VALUE_LOST_MAGNITUDE_THRESHOLD = 2.247849687590416


def derive_conditional_median_threshold(magnitude: pd.Series | FloatArray) -> float:
    """The general derivation rule: median of a magnitude series among its nonzero values.

    Shared by ``scripts/surgical_value_lost_distribution.py`` (NFL) and
    ``scripts/surgical_cfb_recipe_validation.py`` (the CFB recipe check) so
    the exact same rule, not merely the same idea, produces both leagues'
    thresholds. The plain (unconditional) population median is not used
    because it collides with the point mass at exactly zero whenever that
    mass exceeds half the population -- true for the CFB analog (55.7% zero)
    though not for the NFL construct itself (29.86% zero) -- and a rule that
    silently changed definition between the two would no longer be one
    general recipe.
    """

    values = pd.Series(np.asarray(magnitude, dtype=np.float64))
    nonzero = values.loc[values > 0.0]
    if nonzero.empty:
        raise ValueError("magnitude has no nonzero values; cannot derive a conditional median")
    return float(nonzero.median())


def raw_value_magnitude(features: pd.DataFrame) -> pd.Series:
    """Per-game |diff skill-EPA value lost| + |diff defense-disruption value lost|.

    Reads the two columns directly from a feature table built with the
    ``player_value`` profile's fixed-severity table (or any table carrying
    ``VALUE_LOST_DIFF_COLUMNS``) -- the same quantities the ridge model
    itself consumes, already imputed to a finite value for every row. No
    manual home/away reconstruction and no window-specific standardization,
    so the number and its units are identical in every season, which is what
    makes the frozen threshold above portable to a future confirmation
    window without recomputing a scale.
    """

    total = pd.Series(0.0, index=features.index, dtype=float)
    for column in VALUE_LOST_DIFF_COLUMNS:
        total = total + pd.to_numeric(features[column], errors="raise").abs()
    return total


def gate_by_value_lost_magnitude(
    baseline_pick: FloatArray,
    candidate_pick: FloatArray,
    magnitude: FloatArray,
    *,
    threshold: float = VALUE_LOST_MAGNITUDE_THRESHOLD,
) -> FloatArray:
    """Defer to the baseline wherever the injury construct itself is small.

    Unlike ``estimation_variance.gate_by_disagreement`` (which gates on how
    far the fitted probabilities disagree), this gates on the CONSTRUCT's own
    magnitude -- a game with ``magnitude < threshold`` is deferred to the
    baseline regardless of what the candidate model happened to predict
    there, because the mechanism argues the candidate has nothing to read in
    that game. ``baseline_pick``/``candidate_pick`` may be probabilities in
    ``[0, 1]`` or already-forced 0/1 (or boolean) picks -- the function is
    agnostic; it only ever selects one array's value or the other's,
    elementwise, never averages or transforms them.
    """

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    baseline = np.asarray(baseline_pick, dtype=np.float64)
    candidate = np.asarray(candidate_pick, dtype=np.float64)
    mag = np.asarray(magnitude, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.shape != mag.shape:
        raise ValueError("baseline_pick, candidate_pick, and magnitude must share a shape")
    return np.asarray(np.where(mag >= threshold, candidate, baseline), dtype=np.float64)


def gate_active_fraction(
    magnitude: FloatArray, *, threshold: float = VALUE_LOST_MAGNITUDE_THRESHOLD
) -> float:
    """Fraction of games where the gate lets the candidate's pick through at all.

    An upper bound on the gated ``f``: the candidate can only move a pick on
    games where it is active, so gated ``f`` <= this fraction, whatever the
    underlying disagreement rate turns out to be.
    """

    mag = np.asarray(magnitude, dtype=np.float64)
    return float(np.mean(mag >= threshold))


def surgical_predeclaration_summary(
    diff_columns: Sequence[str] = VALUE_LOST_DIFF_COLUMNS,
) -> dict[str, object]:
    """The frozen recipe as a plain dict, for a future session to execute verbatim.

    Not itself a computation over any window -- a structured restatement of
    the predeclaration in ``docs/surgical_injury.md`` so a future caller can
    import the exact column names and threshold instead of retyping them.
    """

    return {
        "family": "injury_value_lost_narrowed_surgical",
        "inherits": ["injury_value_lost_narrowed", "mod07_weak_signal_stack"],
        "diff_columns": list(diff_columns),
        "gate": "gate_by_value_lost_magnitude",
        "threshold": VALUE_LOST_MAGNITUDE_THRESHOLD,
        "threshold_derivation": "population median of raw_value_magnitude, full 2009-2025 history",
    }
