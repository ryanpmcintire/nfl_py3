"""Replay the correction recorded by a card, without changing historical fits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from nfl_ats.calibration import ResidualSmoothingMethod
from nfl_ats.home_side_location import (
    center_offset_for_frame,
    center_offsets_from_metadata,
    served_center_offsets,
)
from nfl_ats.margin import MarginModel


@dataclass(frozen=True)
class CardRefit:
    center_offsets: Mapping[str, float] | None
    probability_method: ResidualSmoothingMethod
    warnings: tuple[str, ...] = ()

    def predict(self, model: MarginModel, frame: pd.DataFrame) -> pd.DataFrame:
        if self.center_offsets is None:
            return model.predict(frame, probability_method=self.probability_method)
        return model.predict(
            frame,
            probability_method=self.probability_method,
            center_offset=center_offset_for_frame(frame, self.center_offsets),
        )


def load_card_refit(
    metadata: Mapping[str, object],
    card: pd.DataFrame,
    forecast_dir: Path,
    *,
    historical_method: ResidualSmoothingMethod = "gaussian",
) -> CardRefit:
    """Prefer the card's metadata; fall back to its sidecar, never refit offsets.

    Old cards retain the caller's original mapping and uncorrected center.
    Warnings are returned for inclusion in the recorder's result.
    """

    offsets = center_offsets_from_metadata(metadata, card)
    if offsets is None:
        offsets = served_center_offsets(forecast_dir)
    if offsets is None:
        return CardRefit(
            None,
            historical_method,
            ("No served home-side offset recorded; retaining historical refit behavior.",),
        )
    return CardRefit(
        offsets,
        cast(ResidualSmoothingMethod, metadata.get("probability_method", historical_method)),
    )
