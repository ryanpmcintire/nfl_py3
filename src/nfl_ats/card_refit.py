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
from nfl_ats.key_line_pick_read import apply_pick_overrides, load_pick_overrides
from nfl_ats.margin import MarginModel


@dataclass(frozen=True)
class CardRefit:
    center_offsets: Mapping[str, float] | None
    probability_method: ResidualSmoothingMethod
    warnings: tuple[str, ...] = ()
    pick_overrides: Mapping[str, float] | None = None

    def predict(
        self, model: MarginModel, frame: pd.DataFrame, *, replay_served_pick: bool = True
    ) -> pd.DataFrame:

        if self.center_offsets is None:
            result = model.predict(frame, probability_method=self.probability_method)
        else:
            result = model.predict(
                frame,
                probability_method=self.probability_method,
                center_offset=center_offset_for_frame(frame, self.center_offsets),
            )
        if replay_served_pick and self.pick_overrides:
            result = result.copy()
            result["home_cover_probability"] = apply_pick_overrides(
                result["home_cover_probability"], frame["game_id"], self.pick_overrides
            )
        return result


def load_card_refit(
    metadata: Mapping[str, object],
    card: pd.DataFrame,
    forecast_dir: Path,
    *,
    historical_method: ResidualSmoothingMethod = "gaussian",
) -> CardRefit:

    pick_overrides = load_pick_overrides(metadata, forecast_dir)
    offsets = center_offsets_from_metadata(metadata, card)
    if offsets is None:
        offsets = served_center_offsets(forecast_dir)
    if offsets is None:
        return CardRefit(
            None,
            historical_method,
            ("No served home-side offset recorded; retaining historical refit behavior.",),
            pick_overrides,
        )
    return CardRefit(
        offsets,
        cast(ResidualSmoothingMethod, metadata.get("probability_method", historical_method)),
        (),
        pick_overrides,
    )
