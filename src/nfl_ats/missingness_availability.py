"""Explicit, source-level availability features for missingness experiments.

This module deliberately reads only the presence of the source columns.  It
does not read an outcome, a line, or any row outside the game being enriched.
"""

from __future__ import annotations

import pandas as pd

from nfl_ats.constants import SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS

ROSTER_CONTINUITY_DATA_AVAILABLE = "roster_continuity_data_available"


def add_roster_continuity_availability(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with one shared source-availability flag.

    The seven lineup-continuity columns have one measured source-era
    transition.  A row is available only when every member is available;
    partial rows (the 2013 transition regime) remain explicitly unavailable
    rather than being mistaken for a complete source regime.
    """

    missing = sorted(set(SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            "Roster-continuity availability needs source columns: " + ", ".join(missing)
        )
    result = frame.copy()
    result[ROSTER_CONTINUITY_DATA_AVAILABLE] = (
        result.loc[:, list(SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS)].notna().all(axis=1).astype(float)
    )
    return result
