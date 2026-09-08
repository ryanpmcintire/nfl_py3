"""MOD-18 lane Q: row-local home-underdog location features (docs/home_dog_location.md).

Lane L's Diagnosis D located a point-forecast error in the games where the
home team is a big underdog: at 10.5+ the actual home margin beats the
model's point forecast by about five points when the home side is the dog,
and by nothing when the home side is the favourite. These two columns let
the ridge learn that correction from prior games instead of bolting a flip
on at a threshold. Both are functions of the row's own available line only.
"""

from __future__ import annotations

import pandas as pd

#: Q1 -- points by which the home team is the underdog at the line the row is
#: scored at; zero when the home team is favoured or the game is a pick'em.
HOME_DOG_POINTS_COLUMNS = ("home_dog_points",)
#: Q2 -- Q1 plus the same quantity above seven points, so the fitted
#: correction may bend where lane L located the error.
HOME_DOG_HINGE_COLUMNS = ("home_dog_points", "home_dog_hinge_7")
HINGE_POINTS = 7.0


def attach_home_dog_location(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the home-underdog columns from ``spread_line`` (positive = home favoured).

    Uses only this row's line: no other game, no outcome, no timestamp. A row
    with no line gets missing values so the ridge imputer treats it the way
    it treats every other missing feature.
    """

    result = frame.copy()
    line = pd.to_numeric(result["spread_line"], errors="raise")
    dog_points = (-line).clip(lower=0.0)
    result["home_dog_points"] = dog_points.where(line.notna())
    result["home_dog_hinge_7"] = (dog_points - HINGE_POINTS).clip(lower=0.0).where(line.notna())
    return result
