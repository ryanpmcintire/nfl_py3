from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.forecast_cold_visitor_tilt_overlay import tuesday_noon_et_cutoff_utc
from nfl_ats.market_data import POOL_TIMEZONE, load_quote_history
from nfl_ats.splash_lines import load_splash_capture

HOOK_COLUMN = "pool_hook"
HOOK_DIRECTION_COLUMN = "pool_hook_direction"
HOOK_PUSH_MASS_COLUMN = "pool_hook_push_mass"
HOOK_PUSH_WINDOW = 1.0
POOL_OFFSET = 0.5
PROBABILITY_EPSILON = 1e-6


def _logit(probability: np.ndarray | float) -> np.ndarray | float:
    return np.log(probability / (1.0 - probability))


def is_whole(line: float) -> bool:
    return bool(line % 1 == 0)


def whole_mask(line: pd.Series) -> np.ndarray:
    return np.asarray((line % 1) == 0, dtype=bool)


def push_mass_at(
    margin: pd.Series, line: pd.Series, anchor: float, window: float = HOOK_PUSH_WINDOW
) -> float:
    near = (line - anchor).abs() <= window
    if not bool(near.any()):
        return float("nan")
    return float((margin[near] == anchor).mean())


def hook_value(direction: np.ndarray, push_mass: np.ndarray) -> np.ndarray:
    sign = np.sign(direction)
    value = _logit(
        np.clip(0.5 * (1.0 - sign * push_mass), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    )
    return np.where((sign != 0) & np.isfinite(push_mass), value, 0.0)


def push_mass_table(frame: pd.DataFrame) -> dict[float, float]:
    margin = frame["margin_vs_open"] + frame["tue_open_home_spread"]
    line = frame["tue_open_home_spread"]
    table: dict[float, float] = {}
    for anchor in range(int(np.floor(line.min())), int(np.ceil(line.max())) + 1):
        value = push_mass_at(margin, line, float(anchor))
        if np.isfinite(value):
            table[float(anchor)] = value
    return table


def hook_rows(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.reset_index(drop=True)
    line = work["tue_open_home_spread"].astype(float)
    margin = work["margin_vs_open"].astype(float) + line
    whole = np.flatnonzero(whole_mask(line))
    prior = np.full(len(work), np.nan)
    cache: dict[tuple[int, float], float] = {}
    for index in whole:
        key = (int(work["season"].iloc[index]), float(line.iloc[index]))
        if key not in cache:
            earlier = work["season"].lt(key[0]).to_numpy()
            cache[key] = push_mass_at(margin[earlier], line[earlier], key[1])
        prior[index] = cache[key]
    pieces = []
    for direction in (1.0, -1.0):
        piece = work.iloc[whole][["game_id", "season"]].copy()
        piece["row"] = whole
        piece["pool_line"] = line.iloc[whole].to_numpy() + direction * POOL_OFFSET
        piece[HOOK_COLUMN] = hook_value(np.full(len(whole), direction), prior[whole])
        piece["home_covered"] = (margin.iloc[whole].to_numpy() > piece["pool_line"]).astype(float)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def tuesday_market_lines(
    data_root: Path, season: int, week: int, game_ids: list[str]
) -> dict[str, float]:
    capture = load_splash_capture(data_root, season, week)
    if capture is None:
        return {}
    opened = tuesday_noon_et_cutoff_utc(pd.Timestamp(capture.captured_at_et).tz_convert("UTC"))
    quotes = load_quote_history(data_root / "market" / "raw", since=opened)
    if quotes.empty:
        return {}
    quotes = quotes.loc[quotes["market"].eq("spreads") & quotes["nflverse_game_id"].isin(game_ids)]
    observed = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    commence = pd.to_datetime(quotes["commence_time_utc"], utc=True)
    quotes = quotes.loc[observed.ge(opened) & observed.lt(commence)]
    if quotes.empty:
        return {}
    observed = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    snapshots = (
        quotes.groupby(["nflverse_game_id", "observed_at_utc"])["home_spread_line"]
        .median()
        .reset_index()
        .sort_values("observed_at_utc")
    )
    first = snapshots.groupby("nflverse_game_id").first()["home_spread_line"]
    lines = {str(game): float(value) for game, value in first.items()}
    wednesday = (opened.tz_convert(POOL_TIMEZONE).normalize() + pd.Timedelta(days=1)).tz_convert(
        "UTC"
    )
    tuesday = quotes.loc[observed.lt(wednesday)].sort_values("observed_at_utc")
    if not tuesday.empty:
        latest = tuesday.groupby(["nflverse_game_id", "bookmaker_key"])["home_spread_line"].last()
        settled = latest.groupby(level="nflverse_game_id").median()
        lines.update({str(game): float(value) for game, value in settled.items()})
    return lines


def crossed_push_mass(anchor: float, pool: float, push_table: Mapping[float, float]) -> float:
    if not np.isfinite(anchor) or not np.isfinite(pool) or anchor == pool:
        return 0.0
    above = pool > anchor
    low, high = (anchor, pool) if above else (pool, anchor)
    total = 0.0
    for number in range(int(np.floor(low)), int(np.ceil(high)) + 1):
        if (above and number >= pool) or (not above and number <= pool):
            continue
        distance = (number - anchor) if above else (anchor - number)
        weight = float(np.clip(1.0 + distance / POOL_OFFSET, 0.0, 2.0))
        total += weight * float(push_table.get(float(number), 0.0))
    return float(min(total, 1.0 - PROBABILITY_EPSILON))


def served_hook_frame(
    predictions: pd.DataFrame, data_root: Path | None, push_table: Mapping[float, float]
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "game_id": predictions["game_id"].astype(str),
            HOOK_DIRECTION_COLUMN: 0.0,
            HOOK_PUSH_MASS_COLUMN: 0.0,
            HOOK_COLUMN: 0.0,
        }
    ).reset_index(drop=True)
    if (
        data_root is None
        or not push_table
        or not {"season", "week", "spread_line"} <= set(predictions.columns)
    ):
        return result
    frame = predictions.reset_index(drop=True)
    for (season, week), group in frame.groupby(["season", "week"]):
        market = tuesday_market_lines(
            data_root, int(str(season)), int(str(week)), group["game_id"].astype(str).tolist()
        )
        for index in group.index:
            game = str(frame.loc[index, "game_id"])
            anchor = market.get(game)
            pool = float(frame.loc[index, "spread_line"])
            if anchor is None:
                continue
            mass = crossed_push_mass(float(anchor), pool, push_table)
            if mass <= 0.0:
                continue
            result.loc[index, HOOK_DIRECTION_COLUMN] = float(np.sign(pool - anchor))
            result.loc[index, HOOK_PUSH_MASS_COLUMN] = mass
    result[HOOK_COLUMN] = hook_value(
        result[HOOK_DIRECTION_COLUMN].to_numpy(dtype=float),
        result[HOOK_PUSH_MASS_COLUMN].to_numpy(dtype=float),
    )
    return result
