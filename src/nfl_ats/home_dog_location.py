from __future__ import annotations

import pandas as pd

HOME_DOG_POINTS_COLUMNS = ("home_dog_points",)
HOME_DOG_HINGE_COLUMNS = ("home_dog_points", "home_dog_hinge_7")
HINGE_POINTS = 7.0


def attach_home_dog_location(frame: pd.DataFrame) -> pd.DataFrame:

    result = frame.copy()
    line = pd.to_numeric(result["spread_line"], errors="raise")
    dog_points = (-line).clip(lower=0.0)
    result["home_dog_points"] = dog_points.where(line.notna())
    result["home_dog_hinge_7"] = (dog_points - HINGE_POINTS).clip(lower=0.0).where(line.notna())
    return result
