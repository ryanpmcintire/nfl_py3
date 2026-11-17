from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.forecast_weather_features import load_forecast_archive
from nfl_ats.nfl_week import pool_decision_cutoff

FORECAST_ARCHIVE = (
    Path("raw") / "forecast_archive" / "pool_decision_2009_2025_aware" / "forecasts.parquet"
)
BULLETIN_ROOT = Path("raw") / "forecast_archive" / "mos_bulletins"
STATION_MAP = Path("registry") / "reference" / "stadium_station_map.csv"
BULLETIN_MODELS = ("GFS", "MEX")
OPEN_AIR_ROOFS = ("outdoors", "open")
KNOTS_TO_MPH = 1.15078
GFS_MOS_PROJECTION_STEP = pd.Timedelta(hours=3)

Bulletins = dict[tuple[str, str], list[tuple[pd.Timestamp, pd.DataFrame]]]


def _bulletins(root: Path, model: str, station: str, cache: Bulletins) -> list[Any]:
    key = (model, station)
    if key not in cache:
        loaded = []
        for path in sorted((root / model / station).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = pd.DataFrame(payload.get("data", []))
            if rows.empty:
                continue
            loaded.append((pd.Timestamp(payload["cached_at_utc"]), rows))
        cache[key] = loaded
    return cache[key]


def _nearest_bulletin(
    root: Path,
    station: str,
    kickoff: pd.Timestamp,
    cutoff: pd.Timestamp,
    cache: Bulletins,
) -> tuple[float, float] | None:
    best = None
    for model in BULLETIN_MODELS:
        for cached_at, rows in _bulletins(root, model, station, cache):
            if cached_at > cutoff:
                continue
            forecast_time = pd.to_datetime(rows["ftime"], utc=True)
            run_time = pd.to_datetime(rows["runtime"], utc=True)
            gap = (forecast_time - kickoff).abs()
            order = np.lexsort((-run_time.astype("int64").to_numpy(), gap.to_numpy()))
            i = int(order[0])
            rank = (gap.iloc[i], -run_time.iloc[i].value)
            if gap.iloc[i] <= GFS_MOS_PROJECTION_STEP and (best is None or rank < best[0]):
                best = (
                    rank,
                    float(rows["tmp"].iloc[i]),
                    float(rows["wsp"].iloc[i]) * KNOTS_TO_MPH,
                )
    return None if best is None else (best[1], best[2])


def attach_pregame_forecast_weather(
    games: pd.DataFrame,
    schedules: pd.DataFrame,
    data_root: Path,
    station_map: Path = STATION_MAP,
) -> tuple[pd.DataFrame, dict[str, int]]:
    archive = load_forecast_archive(data_root / FORECAST_ARCHIVE)
    archive = archive.set_index("game_id")
    open_air = archive["roof"].astype(str).str.lower().isin(OPEN_AIR_ROOFS)
    out = games.copy()
    ids = out["game_id"].astype(str)
    archived = ids.isin(set(archive.index))
    out["temp"] = ids.map(archive["forecast_temp_f"].where(open_air)).astype(float)
    out["wind"] = ids.map(archive["forecast_wind_mph"].where(open_air)).astype(float)
    venue = schedules[["game_id", "stadium", "roof"]].assign(
        game_id=schedules["game_id"].astype(str)
    )
    stations = pd.read_csv(station_map)[["stadium", "icao_station"]].drop_duplicates("stadium")
    venue = venue.drop_duplicates("game_id").merge(stations, on="stadium", how="left")
    venue = venue.set_index("game_id")
    cache: Bulletins = {}
    live_open_air = 0
    live_matched = 0
    for index in out.index[~archived.to_numpy()]:
        game_id = ids.at[index]
        if game_id not in venue.index:
            continue
        roof = str(venue.at[game_id, "roof"]).lower()
        station = venue.at[game_id, "icao_station"]
        kickoff = pd.Timestamp(out.at[index, "kickoff"])
        if roof not in OPEN_AIR_ROOFS or not isinstance(station, str) or pd.isna(kickoff):
            continue
        live_open_air += 1
        kickoff = kickoff.tz_convert("UTC")
        cutoff = pd.Timestamp(pool_decision_cutoff(kickoff.to_pydatetime()))
        found = _nearest_bulletin(data_root / BULLETIN_ROOT, station, kickoff, cutoff, cache)
        if found is not None:
            live_matched += 1
            out.at[index, "temp"], out.at[index, "wind"] = found
    coverage = {
        "archived_rows": int(archived.sum()),
        "archived_with_forecast": int((archived & out["temp"].notna()).sum()),
        "unarchived_rows": int((~archived).sum()),
        "unarchived_open_air": live_open_air,
        "unarchived_with_forecast": live_matched,
    }
    return out, coverage
