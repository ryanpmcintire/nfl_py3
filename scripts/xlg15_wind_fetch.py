import argparse
import gzip
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "xlg15"
SCHED = ROOT / "data" / "cfb" / "schedules" / "raw" / "20260816T162105Z"
SEASONS = [*range(2014, 2020), *range(2023, 2026)]
STATIONS_PER_VENUE = 2
WINDOW_HOURS = 4
KMH_PER_MPH = 1.609344
EARTH_KM = 6371.0
WORKERS = 6
HOURLY_COLS = [
    "date",
    "hour",
    "temp",
    "dwpt",
    "rhum",
    "prcp",
    "snow",
    "wdir",
    "wspd",
    "wpgt",
    "pres",
    "tsun",
    "coco",
]


def cfbd_key():
    k = os.environ.get("CFBD_API_KEY")
    if k:
        return k
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
        return winreg.QueryValueEx(h, "CFBD_API_KEY")[0]


def get(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "research-client", **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def venues():
    p = OUT / "cfbd_venues.json"
    if not p.exists():
        raw = get(
            "https://api.collegefootballdata.com/venues",
            {"Authorization": f"Bearer {cfbd_key()}", "accept": "application/json"},
        )
        p.write_bytes(raw)
    rows = []
    for v in json.loads(p.read_text()):
        loc = v.get("location") or {}
        lat = loc.get("y", v.get("latitude"))
        lon = loc.get("x", v.get("longitude"))
        rows.append(
            {
                "venue_id": v["id"],
                "lat": lat,
                "lon": lon,
                "dome": bool(v.get("dome")),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["lat", "lon"])


def games(seasons):
    parts = []
    for s in seasons:
        d = pd.read_parquet(SCHED / f"season={s}" / "schedules.parquet")
        parts.append(d[d.season_type.eq("regular") & d.completed.fillna(False)])
    g = pd.concat(parts, ignore_index=True)
    g["kick"] = pd.to_datetime(g.start_date, utc=True)
    return g[["game_id", "season", "week", "venue_id", "kick", "neutral_site"]]


def hav(lat1, lon1, lat2, lon2):
    a = np.radians
    dl = a(lat2 - lat1)
    dn = a(lon2 - lon1)
    h = np.sin(dl / 2) ** 2 + np.cos(a(lat1)) * np.cos(a(lat2)) * np.sin(dn / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(h))


def stations():
    p = OUT / "meteostat_stations_lite.json.gz"
    if not p.exists():
        p.write_bytes(get("https://bulk.meteostat.net/v2/stations/lite.json.gz"))
    rows = []
    for s in json.loads(gzip.decompress(p.read_bytes())):
        inv = (s.get("inventory") or {}).get("hourly") or {}
        loc = s.get("location") or {}
        if inv.get("start") and inv.get("end") and loc.get("latitude") is not None:
            rows.append(
                {
                    "station": s["id"],
                    "slat": loc["latitude"],
                    "slon": loc["longitude"],
                    "start": pd.Timestamp(inv["start"], tz="UTC"),
                    "end": pd.Timestamp(inv["end"], tz="UTC"),
                }
            )
    return pd.DataFrame(rows)


def hourly(station, seasons, cache):
    p = OUT / cache / f"{station}.parquet"
    if p.exists():
        return station, pd.read_parquet(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = gzip.decompress(get(f"https://bulk.meteostat.net/v2/hourly/{station}.csv.gz"))
    except Exception:
        return station, None
    d = pd.read_csv(io.BytesIO(raw), header=None, names=HOURLY_COLS, usecols=[0, 1, 8])
    d["t"] = pd.to_datetime(d.date, utc=True) + pd.to_timedelta(d.hour, unit="h")
    d = d[d.t.dt.year.isin(seasons)][["t", "wspd"]].dropna()
    d.to_parquet(p)
    return station, d


def main():
    ap = argparse.ArgumentParser(
        description="Realized kickoff-window station wind for CFB games (Meteostat)."
    )
    ap.add_argument("--seasons", type=int, nargs="+", default=SEASONS, help="CFB seasons to build")
    ap.add_argument(
        "--out", default="cfb_game_wind.parquet", help="output parquet name under artifacts/xlg15"
    )
    ap.add_argument(
        "--cache",
        default="meteostat_hourly",
        help="hourly station cache directory under artifacts/xlg15",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v = venues()
    g = games(args.seasons).merge(v, on="venue_id", how="left")
    print("games", len(g), "with venue coords", int(g.lat.notna().sum()), flush=True)
    g = g.dropna(subset=["lat"])
    st = stations()
    span = g.groupby("venue_id").agg(
        lat=("lat", "first"), lon=("lon", "first"), lo=("kick", "min"), hi=("kick", "max")
    )
    pick = []
    for vid, r in span.iterrows():
        c = st[(st.start <= r.lo) & (st.end >= r.hi)].copy()
        c["dist_km"] = hav(r.lat, r.lon, c.slat.values, c.slon.values)
        c = c.nsmallest(STATIONS_PER_VENUE, "dist_km")
        for rank, (_, s) in enumerate(c.iterrows()):
            pick.append({"venue_id": vid, "station": s.station, "dist_km": s.dist_km, "rank": rank})
    pick = pd.DataFrame(pick)
    need = sorted(pick.station.unique())
    print("venues", len(span), "stations", len(need), flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        data = dict(ex.map(partial(hourly, seasons=args.seasons, cache=args.cache), need))
    print("downloaded", sum(d is not None for d in data.values()), flush=True)
    out = []
    for _, r in g.iterrows():
        k0 = r.kick.floor("h")
        win = pd.date_range(k0, periods=WINDOW_HOURS, freq="h")
        best = None
        for _, s in pick[pick.venue_id == r.venue_id].sort_values("rank").iterrows():
            d = data.get(s.station)
            if d is None:
                continue
            w = d[d.t.isin(win)].wspd
            if len(w):
                best = (s.station, s.dist_km, float(w.mean()) / KMH_PER_MPH, len(w))
                break
        if best:
            out.append(
                {
                    "game_id": r.game_id,
                    "season": r.season,
                    "venue_id": r.venue_id,
                    "dome": r.dome,
                    "station": best[0],
                    "dist_km": best[1],
                    "wind_mph": best[2],
                    "n_obs": best[3],
                }
            )
    o = pd.DataFrame(out)
    o.to_parquet(OUT / args.out)
    print(
        "wind rows",
        len(o),
        "outdoor",
        int((~o.dome).sum()),
        "dist_km median",
        round(float(o.dist_km.median()), 1),
        "p90",
        round(float(o.dist_km.quantile(0.9)), 1),
        "wind median",
        round(float(o.wind_mph.median()), 2),
        flush=True,
    )


if __name__ == "__main__":
    main()
