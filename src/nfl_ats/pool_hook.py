from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.forecast_cold_visitor_tilt_overlay import tuesday_noon_et_cutoff_utc
from nfl_ats.market_data import load_quote_history
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


def stack_pool_lines(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.reset_index(drop=True).copy()
    line = work["tue_open_home_spread"].astype(float)
    margin = work["margin_vs_open"].astype(float) + line
    whole = whole_mask(line)
    prior = np.full(len(work), np.nan)
    cache: dict[tuple[int, float], float] = {}
    for index in np.flatnonzero(whole):
        key = (int(work["season"].iloc[index]), float(line.iloc[index]))
        if key not in cache:
            earlier = work["season"].lt(key[0]).to_numpy()
            cache[key] = push_mass_at(margin[earlier], line[earlier], key[1])
        prior[index] = cache[key]
    up = work.copy()
    up["pool_line"] = np.where(whole, line + POOL_OFFSET, line)
    up[HOOK_COLUMN] = hook_value(np.where(whole, 1.0, 0.0), prior)
    down = work.loc[whole].copy()
    down["pool_line"] = line[whole] - POOL_OFFSET
    down[HOOK_COLUMN] = hook_value(np.full(len(down), -1.0), prior[whole])
    stacked = pd.concat([up, down], ignore_index=True)
    stacked_margin = stacked["margin_vs_open"].astype(float) + stacked[
        "tue_open_home_spread"
    ].astype(float)
    stacked["home_covered"] = stacked_margin.gt(stacked["pool_line"]).astype(float)
    return stacked


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
    snapshots = (
        quotes.groupby(["nflverse_game_id", "observed_at_utc"])["home_spread_line"]
        .median()
        .reset_index()
        .sort_values("observed_at_utc")
    )
    first = snapshots.groupby("nflverse_game_id").first()["home_spread_line"]
    return {str(game): float(value) for game, value in first.items()}


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
            if anchor is None or not is_whole(anchor):
                continue
            offset = pool - anchor
            mass = push_table.get(float(anchor))
            if mass is None or not np.isclose(abs(offset), POOL_OFFSET):
                continue
            result.loc[index, HOOK_DIRECTION_COLUMN] = float(np.sign(offset))
            result.loc[index, HOOK_PUSH_MASS_COLUMN] = float(mass)
    result[HOOK_COLUMN] = hook_value(
        result[HOOK_DIRECTION_COLUMN].to_numpy(dtype=float),
        result[HOOK_PUSH_MASS_COLUMN].to_numpy(dtype=float),
    )
    return result
