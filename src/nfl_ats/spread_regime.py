"""Frozen MOD-18 row-local spread features and chronological cell calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPREAD_REGIME_COLUMNS = (
    "regime_fav_3p5_6p5",
    "regime_fav_7p5_10",
    "regime_fav_10p5_plus",
    "regime_signed_absolute",
    "regime_distance_3",
    "regime_distance_7",
    "regime_distance_10",
    "regime_distance_14",
)
BUCKETS = ("0-3", "3.5-6.5", "7", "7.5-10", "10.5+")


def spread_bucket(spread: pd.Series) -> pd.Series:
    """Disjoint half-point line buckets; 7.5 belongs to the fade zone."""
    size = pd.to_numeric(spread, errors="raise").abs()
    return pd.Series(
        np.select(
            [size.le(3), size.le(6.5), size.lt(7.5), size.le(10)],
            BUCKETS[:4],
            default=BUCKETS[4],
        ),
        index=spread.index,
    ).where(size.notna())


def attach_spread_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """Use only this row's available line, never other games or outcomes."""
    result = frame.copy()
    line = pd.to_numeric(result["spread_line"], errors="raise")
    size, sign = line.abs(), np.sign(line)
    for column, mask in zip(
        SPREAD_REGIME_COLUMNS[:3],
        (size.between(3.5, 6.5), size.between(7.5, 10), size.ge(10.5)),
        strict=True,
    ):
        result[column] = (sign * mask.astype(float)).where(line.notna())
    result["regime_signed_absolute"] = sign * size
    for key in (3, 7, 10, 14):
        result[f"regime_distance_{key}"] = sign * (size - key).abs()
    return result


def calibrate_spread_stream(predictions: pd.DataFrame) -> pd.DataFrame:
    """Shrink prior cell correctness toward stated p with 20 fixed pseudo-games.

    Input probabilities must be out-of-sample. A game's result becomes usable
    the next day; no result from the target season-week enters its calibration.
    The returned home probability may cross 0.5; input rows are never mutated.
    """
    frame = predictions.copy()
    if frame.game_id.duplicated().any():
        raise ValueError("One prediction per game is required")
    frame["gameday"] = pd.to_datetime(frame.gameday, errors="raise")
    p = pd.to_numeric(frame.home_cover_probability, errors="raise")
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("Probabilities must be finite in [0, 1]")
    pick = p.ge(0.5)
    stated = p.where(pick, 1 - p)
    line = frame.spread_line
    side: pd.Series[str] = pd.Series(
        np.where(pick.eq(line.gt(0)), "favourite", "underdog"), index=frame.index, dtype=str
    )
    side = side.where(line.ne(0), "pickem")
    band = pd.cut(stated, [0.499999, 0.55, 0.60, 1.000001], right=False).astype(str)
    cell = spread_bucket(line).astype(str) + "/" + side + "/" + band
    correct = pick.eq(frame.home_cover.eq(1)).astype(float).where(frame.home_cover.notna())
    frame["raw_home_cover_probability"] = p
    frame["calibration_rows"] = 0
    frame["calibration_max_gameday"] = pd.NaT
    for (season, week), batch in frame.groupby(["season", "week"], sort=True):
        cutoff = batch.gameday.min()
        prior = (
            (frame.gameday + pd.Timedelta(days=1)).le(cutoff)
            & ~(frame.season.eq(int(str(season))) & frame.week.eq(int(str(week))))
            & correct.notna()
        )
        for index in batch.index:
            history = prior & cell.eq(cell.loc[index])
            n = int(history.sum())
            calibrated = (float(correct.loc[history].sum()) + 20 * stated.loc[index]) / (n + 20)
            frame.loc[index, "home_cover_probability"] = (
                calibrated if pick.loc[index] else 1 - calibrated
            )
            frame.loc[index, "calibration_rows"] = n
            frame.loc[index, "calibration_max_gameday"] = frame.loc[history, "gameday"].max()
    return frame
