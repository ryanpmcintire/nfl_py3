"""Build the stadium -> county FIPS reference table for environmental-exposure joins.

One-time (re-runnable, idempotent) build feeding both the air-quality (EPA
AQS, county-keyed) and drought (US Drought Monitor, county-FIPS-keyed)
ingestion scripts. Reuses two reference tables already in the repo rather
than re-deriving stadium locations:

  - registry/stadium_coordinates.json: stadium display-name (verbatim from
    nflverse schedules' `stadium` column) -> {lat, lon, tz, city}. Built
    2026-08-19 for the travel/rest battery; covers every `stadium` string
    seen in REG games 2009-2025 plus a few 2026 international venues.
  - registry/reference/stadium_station_map.csv: same stadium keys -> a
    `mappable` boolean (true = domestic US venue with weather-station
    coverage, false = international/no-coverage). Built for the GFS-MOS
    forecast-archive work; reused here purely for its domestic/international
    split, not its ICAO station mapping.

For each `mappable == true` (domestic) stadium, this script calls the FCC's
public Census Block Conversions API (no auth, no key -- **measured** 2026-08-20:
`curl -s "https://geo.fcc.gov/api/census/block/find?latitude=44.5013&longitude=-88.0622&format=json"`
returned HTTP 200 with `{"County":{"FIPS":"55009","name":"Brown County"}, ...}`
for Lambeau Field's coordinates) with each stadium's (lat, lon) and records
the returned county FIPS + county/state name. International rows are kept in
the output with `county_fips` empty and `in_scope=False` -- explicit
out-of-scope rows, not silent omissions, per the task's requirement to flag
international games rather than drop them.

Dome / retractable-roof flag is read directly from the already-ingested
nflverse `schedules.parquet`'s own `roof` column (values: outdoors / dome /
closed / open) rather than re-derived, since that column is **measured**
(this script) to be fully populated 2009-2025 (only the not-yet-played tail
of 2026 is null) -- contradicting docs/data_source_scout_v4.md's caution that
pre-2020 completeness of nflverse's `roof` field was unconfirmed. Each
stadium's roof value(s) across its games are recorded verbatim (usually a
single constant value; retractable-roof venues legitimately show more than
one).

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/build_stadium_county_fips.py

Writes registry/reference/stadium_county_fips.csv (static reference table,
tracked in git like its two sibling reference tables -- NOT under
data/raw/, since this is a derived lookup table, not a raw external
snapshot).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
COORDS_PATH = REPO / "registry/stadium_coordinates.json"
STATION_MAP_PATH = REPO / "registry/reference/stadium_station_map.csv"
OUT_PATH = REPO / "registry/reference/stadium_county_fips.csv"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"

FCC_URL = "https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&format=json"


def _latest_schedules() -> Path:
    candidates = sorted(
        p for p in (REPO / "data/raw").glob("*/schedules.parquet") if p.parent.name[0].isdigit()
    )
    if not candidates:
        raise FileNotFoundError("No data/raw/<snapshot>/schedules.parquet found.")
    return candidates[-1]


def fcc_county_fips(lat: float, lon: float, *, retries: int = 3) -> dict:
    url = FCC_URL.format(lat=lat, lon=lon)
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read())
            county = payload.get("County", {})
            state = payload.get("State", {})
            return {
                "county_fips": county.get("FIPS", ""),
                "county_name": county.get("name", ""),
                "state_code": state.get("code", ""),
                "state_fips": state.get("FIPS", ""),
                "fcc_status": payload.get("status", ""),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FCC lookup failed for ({lat}, {lon}): {last_error}")


def main() -> None:
    coords = json.loads(COORDS_PATH.read_text(encoding="utf-8"))
    coords = {k: v for k, v in coords.items() if k != "_README"}

    station_map = pd.read_csv(STATION_MAP_PATH)
    mappable = dict(zip(station_map["stadium"], station_map["mappable"], strict=True))
    teams_by_stadium = dict(zip(station_map["stadium"], station_map["teams"], strict=True))

    schedules_path = _latest_schedules()
    sched = pd.read_parquet(schedules_path, columns=["stadium", "home_team", "season", "roof"])
    roof_by_stadium = sched.groupby("stadium")["roof"].apply(
        lambda s: sorted(set(s.dropna().tolist()))
    )
    seasons_by_stadium = sched.groupby("stadium")["season"].agg(["min", "max"])
    home_team_by_stadium = sched.groupby("stadium")["home_team"].apply(
        lambda s: sorted(set(s.tolist()))
    )

    all_stadiums = sorted(set(coords) | set(mappable))
    rows = []
    fips_cache: dict[tuple[float, float], dict] = {}
    for stadium in all_stadiums:
        info = coords.get(stadium)
        is_mappable = bool(mappable.get(stadium, False))
        row = {
            "stadium": stadium,
            "teams": teams_by_stadium.get(stadium, ";".join(home_team_by_stadium.get(stadium, []))),
            "lat": info["lat"] if info else None,
            "lon": info["lon"] if info else None,
            "city": info["city"] if info else None,
            "tz": info["tz"] if info else None,
            "in_scope": is_mappable,
            "roof_values_seen": ";".join(roof_by_stadium.get(stadium, [])),
            "season_min": (
                int(seasons_by_stadium.loc[stadium, "min"])
                if stadium in seasons_by_stadium.index
                else None
            ),
            "season_max": (
                int(seasons_by_stadium.loc[stadium, "max"])
                if stadium in seasons_by_stadium.index
                else None
            ),
        }
        if is_mappable and info is not None:
            key = (info["lat"], info["lon"])
            if key not in fips_cache:
                print(f"FCC lookup: {stadium} ({info['lat']}, {info['lon']})")
                fips_cache[key] = fcc_county_fips(info["lat"], info["lon"])
            row.update(fips_cache[key])
        else:
            row.update(
                {
                    "county_fips": "",
                    "county_name": "",
                    "state_code": "",
                    "state_fips": "",
                    "fcc_status": "out_of_scope_international"
                    if not is_mappable
                    else "missing_coordinates",
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("stadium").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(
        f"\nWrote {OUT_PATH} ({len(out)} stadium rows, "
        f"{int(out['in_scope'].sum())} in-scope domestic, "
        f"{int((~out['in_scope']).sum())} out-of-scope international/unmapped)"
    )
    print(f"Distinct county FIPS: {out.loc[out['in_scope'], 'county_fips'].nunique()}")


if __name__ == "__main__":
    main()
