"""Exact previous-season unit priors, with explicit pregame availability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError

UNIT_PRIOR_COLUMNS = ("unit_prior_off_ol_diff", "unit_prior_off_skill_diff")
UNITS = ("OFF_OL", "OFF_SKILL")
RATING_COLUMNS = ("season", "team", "unit", "rating", "finalized_at")


def attach_unit_prior_features(games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Join S-1 only; unavailable histories stay NaN and row order is preserved.

    Ratings are final team-unit estimates, not pooled player correlations.
    ``prediction_timestamp`` and ``finalized_at`` must be explicit UTC-aware
    timestamps (UTC is assumed for timezone-naive inputs). A strict timestamp
    comparison protects early-season decisions from unfinished source fits.
    """
    for name, frame, required in (
        ("games", games, ("season", "home_team", "away_team", "prediction_timestamp")),
        ("ratings", ratings, RATING_COLUMNS),
    ):
        missing = set(required).difference(frame.columns)
        if missing:
            raise DataContractError(f"{name} missing columns: {sorted(missing)}")
        if frame[list(required)].isna().any().any():
            raise DataContractError(f"{name} has null required values")
        seasons = pd.to_numeric(frame["season"], errors="coerce")
        if seasons.isna().any() or not np.isfinite(seasons).all() or seasons.mod(1).ne(0).any():
            raise DataContractError(f"{name} requires integer seasons")
    source = ratings.copy()
    source["season"] = pd.to_numeric(source["season"]).astype(int)
    if source.duplicated(["season", "team", "unit"]).any():
        raise DataContractError("duplicate season/team/unit rating")
    source["rating"] = pd.to_numeric(source["rating"], errors="coerce")
    if not np.isfinite(source["rating"]).all():
        raise DataContractError("ratings must be finite")
    source["finalized_at"] = pd.to_datetime(source["finalized_at"], utc=True, errors="coerce")
    decisions = pd.to_datetime(games["prediction_timestamp"], utc=True, errors="coerce")
    if source["finalized_at"].isna().any() or decisions.isna().any():
        raise DataContractError("invalid availability timestamp")
    output = games.copy()
    keys = pd.DataFrame({"season": pd.to_numeric(games["season"]).to_numpy() - 1})
    for unit, column in zip(UNITS, UNIT_PRIOR_COLUMNS, strict=True):
        values = {}
        for side in ("home", "away"):
            keys["team"] = games[f"{side}_team"].to_numpy()
            joined = keys.merge(
                source.loc[source["unit"].eq(unit)],
                on=["season", "team"],
                how="left",
                sort=False,
                validate="many_to_one",
            )
            available = joined["finalized_at"].lt(decisions.reset_index(drop=True))
            values[side] = joined["rating"].where(available).to_numpy(dtype=float)
        output[column] = values["home"] - values["away"]
    return output
