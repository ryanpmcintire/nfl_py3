from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ats.snapshots import latest_snapshot, load_snapshot

EASTERN = ZoneInfo("America/New_York")
KNOTS_TO_MPH = 1.15078
INDOOR_ROOFS = {"dome", "closed"}
FORECAST_STEP_HOURS = 12
STATION_MAP = Path(__file__).resolve().parents[2] / "registry" / "reference" / "stadium_station_map.csv"


def _kickoff_utc(gameday: object, gametime: object) -> pd.Timestamp | None:
    try:
        local = pd.Timestamp(f"{pd.Timestamp(gameday).date()} {gametime}").tz_localize(EASTERN)
    except Exception:
        return None
    return local.tz_convert("UTC")


def _bulletin(root: Path, station: str, now: pd.Timestamp, kickoff: pd.Timestamp):
    best = None
    for path in sorted((root / station).glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            cached = pd.Timestamp(payload["cached_at_utc"])
        except Exception:
            continue
        if cached <= now and cached < kickoff:
            best = payload
    return best


def _nearest(rows: list[dict], kickoff: pd.Timestamp, key: str):
    window = timedelta(hours=FORECAST_STEP_HOURS / 2)
    scored = []
    for row in rows:
        if row.get(key) is None:
            continue
        valid = pd.Timestamp(str(row["ftime"]).replace(" ", "T") + "+00:00")
        gap = abs(valid - kickoff)
        if gap <= window:
            scored.append((gap, row))
    return min(scored, key=lambda item: item[0])[1] if scored else None


def _chip(payload: dict, kickoff: pd.Timestamp) -> str:
    rows = payload.get("data") or []
    temp_row = _nearest(rows, kickoff, "tmp")
    if temp_row is None:
        return ""
    wind_row = _nearest(rows, kickoff, "wsp")
    rain_row = _nearest(rows, kickoff, "p12") or _nearest(rows, kickoff, "p06")
    parts = [f"{int(round(float(temp_row['tmp'])))}°F"]
    if wind_row is not None:
        parts.append(f"wind {int(round(float(wind_row['wsp']) * KNOTS_TO_MPH))} mph")
    if rain_row is not None:
        rain = rain_row["p12"] if rain_row.get("p12") is not None else rain_row["p06"]
        parts.append(f"{int(round(float(rain)))}% chance of rain")
    cached = pd.Timestamp(payload["cached_at_utc"]).tz_convert(EASTERN)
    return f"Kickoff forecast as of {cached.strftime('%A')}: " + ", ".join(parts)


def build_forecast_chips(
    data_root: Path, *, season: int, week: int, game_ids: set[str], now: datetime
) -> dict[str, str]:
    raw = Path(data_root) / "raw"
    schedules, _ = load_snapshot(latest_snapshot(raw))
    week_rows = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    stations = pd.read_csv(STATION_MAP)
    stations = stations[stations["mappable"].astype(str).str.lower() == "true"]
    by_team = {str(r.teams): str(r.icao_station) for r in stations.itertuples()}
    root = raw / "forecast_archive" / "mos_bulletins" / "MEX"
    stamp = pd.Timestamp(now)
    out: dict[str, str] = {}
    for row in week_rows.itertuples():
        if row.game_id not in game_ids:
            continue
        roof = str(row.roof) if pd.notna(row.roof) else ""
        if roof in INDOOR_ROOFS:
            out[row.game_id] = "Indoors"
            continue
        if roof != "outdoors":
            continue
        station = by_team.get(str(row.home_team))
        kickoff = _kickoff_utc(row.gameday, row.gametime)
        if not station or kickoff is None:
            continue
        payload = _bulletin(root, station, stamp, kickoff)
        chip = _chip(payload, kickoff) if payload else ""
        if chip:
            out[row.game_id] = chip
    return out
