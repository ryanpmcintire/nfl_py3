from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PICK_SIDE_FLOOR = 0.5

DISPLAYED_PICK_PROBABILITY_COLUMN = "displayed_pick_probability"
DISPLAYED_STRENGTH_WORD_COLUMN = "displayed_strength_word"

STRENGTH_WORDS = ("slight", "lean", "strong")
STRENGTH_ROUNDING_PLACES = 3


@dataclass(frozen=True)
class StrengthBands:
    lean_min: float
    strong_min: float

    def word(self, probability: float) -> str:
        shown = round(float(probability), STRENGTH_ROUNDING_PLACES)
        if shown >= self.strong_min:
            return STRENGTH_WORDS[2]
        if shown >= self.lean_min:
            return STRENGTH_WORDS[1]
        return STRENGTH_WORDS[0]

    def words(self, probability: pd.Series) -> pd.Series:
        value = pd.to_numeric(probability, errors="coerce").round(STRENGTH_ROUNDING_PLACES)
        return pd.Series(
            np.select(
                [value.ge(self.strong_min), value.ge(self.lean_min)],
                [STRENGTH_WORDS[2], STRENGTH_WORDS[1]],
                default=STRENGTH_WORDS[0],
            ),
            index=probability.index,
            dtype=object,
        ).where(value.notna())

    def to_dict(self) -> dict[str, float]:
        return {"lean_min": self.lean_min, "strong_min": self.strong_min}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> StrengthBands | None:
        if not isinstance(payload, Mapping):
            return None
        try:
            lean = float(payload["lean_min"])
            strong = float(payload["strong_min"])
        except (KeyError, TypeError, ValueError):
            return None
        return cls(lean_min=lean, strong_min=strong) if lean <= strong else None


def attach_displayed_confidence(
    predictions: pd.DataFrame, bands: StrengthBands | None
) -> pd.DataFrame:

    frame = predictions.copy()
    if "home_cover_probability" not in frame:
        return frame
    home_probability = pd.to_numeric(frame["home_cover_probability"], errors="coerce")
    pick_home = home_probability.ge(0.5)
    stated = home_probability.where(pick_home, 1.0 - home_probability)
    frame[DISPLAYED_PICK_PROBABILITY_COLUMN] = stated
    if bands is not None:
        frame[DISPLAYED_STRENGTH_WORD_COLUMN] = bands.words(
            frame[DISPLAYED_PICK_PROBABILITY_COLUMN]
        )
    return frame


def displayed_pick_probability(row: Mapping[str, Any] | pd.Series) -> float | None:

    value: Any = row.get(DISPLAYED_PICK_PROBABILITY_COLUMN)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(number) else number


def displayed_strength_word(row: Mapping[str, Any] | pd.Series) -> str | None:

    value: Any = row.get(DISPLAYED_STRENGTH_WORD_COLUMN)
    word = str(value).strip().lower() if isinstance(value, str) else ""
    return word if word in STRENGTH_WORDS else None


__all__ = [
    "DISPLAYED_PICK_PROBABILITY_COLUMN",
    "DISPLAYED_STRENGTH_WORD_COLUMN",
    "PICK_SIDE_FLOOR",
    "STRENGTH_ROUNDING_PLACES",
    "STRENGTH_WORDS",
    "StrengthBands",
    "attach_displayed_confidence",
    "displayed_pick_probability",
    "displayed_strength_word",
]
