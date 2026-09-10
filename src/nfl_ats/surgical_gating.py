from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

VALUE_LOST_DIFF_COLUMNS = (
    "diff_injury_skill_epa_value_lost",
    "diff_injury_defense_disruption_value_lost",
)

VALUE_LOST_MAGNITUDE_THRESHOLD = 2.247849687590416


def derive_conditional_median_threshold(magnitude: pd.Series | FloatArray) -> float:

    values = pd.Series(np.asarray(magnitude, dtype=np.float64))
    nonzero = values.loc[values > 0.0]
    if nonzero.empty:
        raise ValueError("magnitude has no nonzero values; cannot derive a conditional median")
    return float(nonzero.median())


def raw_value_magnitude(features: pd.DataFrame) -> pd.Series:

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

    mag = np.asarray(magnitude, dtype=np.float64)
    return float(np.mean(mag >= threshold))


def surgical_predeclaration_summary(
    diff_columns: Sequence[str] = VALUE_LOST_DIFF_COLUMNS,
) -> dict[str, object]:

    return {
        "family": "injury_value_lost_narrowed_surgical",
        "inherits": ["injury_value_lost_narrowed", "mod07_weak_signal_stack"],
        "diff_columns": list(diff_columns),
        "gate": "gate_by_value_lost_magnitude",
        "threshold": VALUE_LOST_MAGNITUDE_THRESHOLD,
        "threshold_derivation": "population median of raw_value_magnitude, full 2009-2025 history",
    }
