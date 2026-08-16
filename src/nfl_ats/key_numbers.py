"""Key-number calibration: a validation report, not a promotion claim.

NFL final margins cluster on a handful of "key numbers" (3 and 7 above all,
from a made field goal or touchdown-plus-extra-point deciding the game).
This module checks whether ``MarginModel``'s empirical predictive
distribution implies the right amount of probability mass on those numbers,
and whether its cover probabilities are well calibrated across different
line regions. Every number here comes from either an already-stored
walk-forward artifact or a fresh leak-safe walk-forward refit -- nothing is
scored with information from the future -- and none of it feeds back into
model selection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

DEFAULT_KEY_NUMBERS: tuple[int, ...] = (1, 2, 3, 4, 6, 7, 10, 14)

_LINE_BUCKET_ORDER: tuple[str, ...] = (
    "under_3",
    "three",
    "three_five_to_six_five",
    "seven",
    "over_seven",
)
LINE_BUCKET_LABELS: dict[str, str] = {
    "under_3": "|line| < 3",
    "three": "|line| = 3",
    "three_five_to_six_five": "3.5 <= |line| <= 6.5",
    "seven": "|line| = 7",
    "over_seven": "|line| > 7",
}


def implied_key_number_mass(
    distribution: npt.NDArray[np.float64],
    key_numbers: Sequence[int] = DEFAULT_KEY_NUMBERS,
) -> pd.DataFrame:
    """Per-game implied probability mass on each |final margin| key number.

    ``distribution`` has shape ``(n_games, n_samples)`` -- see
    ``MarginModel.distribution``. Samples are rounded to the nearest integer
    before comparison, since a real final score margin is always an integer;
    the continuous empirical bootstrap only approximates that support.
    Returns one row per game and one ``key_number_{k}`` column per key
    number, holding the fraction of that game's samples with
    ``|round(sample)| == k`` (a home win or a home loss by exactly k).
    """

    if not key_numbers:
        raise ValueError("At least one key number is required")
    if distribution.ndim != 2:
        raise ValueError("distribution must be a 2-D (games, samples) array")
    rounded = np.round(distribution)
    data = {
        f"key_number_{k}": (np.isclose(rounded, k) | np.isclose(rounded, -k)).mean(axis=1)
        for k in key_numbers
    }
    return pd.DataFrame(data)


def realized_key_number_frequency(
    results: pd.Series, key_numbers: Sequence[int] = DEFAULT_KEY_NUMBERS
) -> pd.Series:
    """Realized frequency of ``|final margin| == k`` for each key number."""

    if not key_numbers:
        raise ValueError("At least one key number is required")
    magnitude = pd.to_numeric(results, errors="coerce").abs().dropna()
    if magnitude.empty:
        raise ValueError("No completed games with a finite result")
    return pd.Series(
        {k: float(np.isclose(magnitude, k).mean()) for k in key_numbers},
        name="realized_frequency",
    )


def summarize_key_number_calibration(
    key_number_mass: pd.DataFrame,
    key_numbers: Sequence[int] = DEFAULT_KEY_NUMBERS,
) -> pd.DataFrame:
    """One row per (method, key number): implied mass vs realized frequency.

    ``key_number_mass`` must have one row per (method, game) with a
    ``method`` column, a ``result`` column (the completed game's realized
    margin), and one ``key_number_{k}`` column per key number -- the shape
    produced by walking ``implied_key_number_mass`` forward across weeks.
    """

    required = {"method", "result", *(f"key_number_{k}" for k in key_numbers)}
    missing = sorted(required.difference(key_number_mass.columns))
    if missing:
        raise ValueError(f"Key-number mass table is missing columns: {', '.join(missing)}")
    rows: list[dict[str, object]] = []
    for method, group in key_number_mass.groupby("method", sort=True):
        realized = realized_key_number_frequency(group["result"], key_numbers)
        for k in key_numbers:
            implied = float(group[f"key_number_{k}"].mean())
            rows.append(
                {
                    "method": method,
                    "key_number": k,
                    "implied_mass": implied,
                    "realized_frequency": float(realized[k]),
                    "gap": implied - float(realized[k]),
                    "games": len(group),
                }
            )
    return pd.DataFrame(rows)


def line_bucket(spread_line: pd.Series) -> pd.Series:
    """Classify each quoted line's magnitude into a key-number line region.

    Buckets: ``|line| < 3``, exactly ``3``, ``3.5`` to ``6.5``, exactly
    ``7``, and ``|line| > 7``. NFL spreads are quoted in half-point
    increments, so these five buckets partition the space without gaps.
    """

    magnitude = pd.to_numeric(spread_line, errors="coerce").abs().to_numpy(dtype=float)
    is_three = np.isclose(magnitude, 3.0, atol=1e-9)
    is_seven = np.isclose(magnitude, 7.0, atol=1e-9)
    conditions = [
        magnitude < 3.0,
        is_three,
        (magnitude > 3.0) & (magnitude < 7.0) & ~is_three & ~is_seven,
        is_seven,
        magnitude > 7.0,
    ]
    bucket = np.select(conditions, _LINE_BUCKET_ORDER, default="unclassified")
    return pd.Series(bucket, index=spread_line.index, name="line_bucket")


def cover_reliability_by_line_bucket(
    predictions: pd.DataFrame,
    *,
    probability_column: str = "home_cover_probability",
) -> pd.DataFrame:
    """Reliability of cover probabilities bucketed by line region.

    Compares the mean predicted probability against the realized cover rate
    (``home_cover``, which is already null for pushes) within each line
    bucket. ``predictions`` is expected to be a walk-forward outcome card
    (or a subset restricted to one method); pass a subset filtered to the
    method of interest for a single-method reliability table.
    """

    required = {"spread_line", probability_column, "home_cover"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Reliability check is missing columns: {', '.join(missing)}")
    frame = predictions.copy()
    frame["_line_bucket"] = line_bucket(frame["spread_line"])
    frame["_probability"] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["_actual"] = pd.to_numeric(frame["home_cover"], errors="coerce")
    frame = frame.dropna(subset=["_probability", "_actual"])
    if frame.empty:
        raise ValueError("No non-push rows with a finite predicted probability")

    rows: list[dict[str, object]] = []
    for bucket, group in frame.groupby("_line_bucket", sort=False):
        mean_predicted = float(group["_probability"].mean())
        realized = float(group["_actual"].mean())
        rows.append(
            {
                "line_bucket": bucket,
                "line_bucket_label": LINE_BUCKET_LABELS.get(str(bucket), str(bucket)),
                "games": len(group),
                "mean_predicted_probability": mean_predicted,
                "realized_cover_rate": realized,
                "calibration_gap": mean_predicted - realized,
            }
        )
    order = {label: index for index, label in enumerate(_LINE_BUCKET_ORDER)}
    table = pd.DataFrame(rows)
    table["_order"] = table["line_bucket"].map(order).fillna(len(order))
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
