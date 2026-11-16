from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_candidate as v4c  # noqa: E402
from mod27_fenced_repairs_replay import paired, record  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v4_weather_openmeteo"
OM = ROOT / "data/external/open_meteo/kickoff_forecast_weather.parquet"
LIVE = ROOT / "data/external/open_meteo/live"
COORDS = ROOT / "registry/stadium_coordinates.json"
OM_SEASONS = range(2021, 2026)
PREV = {"temp": "prev_temperature_2m_previous_day1", "wind": "prev_wind_speed_10m_previous_day1"}
HIST = {"temp": "hist_temperature_2m", "wind": "hist_wind_speed_10m"}
MOS = {"temp": "forecast_temp_f", "wind": "forecast_wind_mph"}
UA = "Mozilla/5.0 (compatible; research-script/1.0)"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
LIVE_HOURLY = "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability"
NOT_OUTDOOR = ("dome", "closed")


def archive():
    fc = pd.read_parquet(v4c.ARCHIVE)
    fc = fc.loc[fc["fetch_status"].eq("ok")].drop_duplicates("game_id").set_index("game_id")
    fc["open_air"] = fc["roof"].astype(str).str.lower().isin(v4c.OPEN_AIR)
    return fc


def openmeteo():
    om = pd.read_parquet(OM)
    om["game_id"] = om["game_id"].astype(str)
    return om.drop_duplicates("game_id").set_index("game_id")


def stats(a, b):
    m = a.notna() & b.notna()
    d = (a[m] - b[m]).astype(float)
    return {
        "n": int(m.sum()),
        "pearson": float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else None,
        "mean_diff_openmeteo_minus_mos": float(d.mean()) if m.any() else None,
        "sd_diff": float(d.std()) if m.any() else None,
        "mean_abs_diff": float(d.abs().mean()) if m.any() else None,
        "mean_mos": float(b[m].mean()) if m.any() else None,
        "mean_openmeteo": float(a[m].mean()) if m.any() else None,
    }


def run_agree(_args):
    fc = archive()
    om = openmeteo()
    keep = fc["open_air"] & fc["season"].isin(OM_SEASONS)
    j = fc.loc[keep].join(om, how="left", rsuffix="_om")
    out = {"open_air_games_2021_2025": int(keep.sum()), "units": "MOS tmp F, wsp knots x 1.15078 mph, 10 m sustained; Open-Meteo temperature_2m F, wind_speed_10m mph, best_match"}
    for lead, cols in (("previous_day1_24h", PREV), ("historical_forecast_0_6h", HIST)):
        out[lead] = {}
        for var, c in cols.items():
            out[lead][var] = {"all": stats(j[c], j[MOS[var]])}
            for s in OM_SEASONS:
                out[lead][var][str(s)] = stats(j.loc[j["season"].eq(s), c], j.loc[j["season"].eq(s), MOS[var]])
    gap = (j["forecast_valid_utc"] - j["kickoff_utc"]).abs().dt.total_seconds() / 3600
    out["mos_valid_minus_kickoff_abs_hours"] = {"median": float(gap.median()), "p90": float(gap.quantile(0.9))}
    (OUT / "agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1)[:5000])


def om_replace(table, fc, om):
    out = table.copy()
    ids = out["game_id"]
    open_air = fc["open_air"].reindex(ids).fillna(False).to_numpy(bool)
    season_ok = out["season"].isin(OM_SEASONS).to_numpy(bool)
    counts = {}
    for var, c in PREV.items():
        v = om[c].reindex(ids).to_numpy(float)
        m = open_air & season_ok & ~np.isnan(v)
        out.loc[m, var] = v[m]
        counts[var] = {
            str(s): int((m & out["season"].eq(s).to_numpy()).sum()) for s in OM_SEASONS
        }
        counts[var]["open_air_2021_2025"] = int((open_air & season_ok).sum())
    return out, counts


def run_grade(_args):
    v4c.OUT = OUT
    v4c.CACHED = True
    for name in ("S", "W", "QW"):
        src = ROOT / "artifacts/v4_candidate" / f"opener_{name}.parquet"
        if not (OUT / src.name).exists():
            shutil.copy2(src, OUT / src.name)
    fc = archive()
    om = openmeteo()
    table = pd.read_parquet(v4c.TABLE)
    table["game_id"] = table["game_id"].astype(str)
    q_table, _ = v4c.qb_repaired(table)
    w_table, _, coverage = v4c.forecast_weather(table)
    qw_table, _, _ = v4c.forecast_weather(q_table)
    ow_table, counts = om_replace(w_table, fc, om)
    oqw_table, _ = om_replace(qw_table, fc, om)
    tables = {
        "S": (table, v4c.FULL),
        "W": (w_table, v4c.FULL),
        "QW": (qw_table, v4c.FULL),
        "OW": (ow_table, v4c.FULL),
        "OQW": (oqw_table, v4c.FULL),
    }
    pop = pd.read_parquet(v4c.PICK / "per_game.parquet")
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    graded = v4c.grade(tables, pop)
    ids = pop["game_id"].to_numpy()
    y = pop["home_covered"].to_numpy(float)
    probs = {}
    for name in ("W", "QW", "OW", "OQW"):
        mp = pd.read_parquet(OUT / f"opener_{name}.parquet")["home_cover_probability_at_open"]
        probs[name] = v4c.stage_two(pop, mp.reindex(ids).to_numpy(float))[0]
    frame = pop.assign(week_block=pop["season"] * 100 + pop["week"])
    for arm, base in (("OW", "W"), ("OQW", "QW")):
        a, b = probs[arm], probs[base]
        changed = (a >= 0.5) != (b >= 0.5)
        graded[arm][f"vs_{base}"] = {
            "season_block": paired(frame, b, a, "season"),
            "week_block": paired(frame, b, a, "week_block"),
            "changed_picks": int(changed.sum()),
            "changed_record_arm": record(a[changed], y[changed]) if changed.any() else "0-0",
            "changed_record_base": record(b[changed], y[changed]) if changed.any() else "0-0",
            "changed_games": pop.loc[changed, "game_id"].tolist(),
        }
    graded["openmeteo_replacements"] = counts
    graded["mos_coverage"] = coverage
    (OUT / "results.json").write_text(json.dumps(graded, indent=2, default=str), encoding="utf-8")
    brief = {
        k: {kk: graded[k].get(kk) for kk in ("record", "log_loss", "vs_S", "vs_W", "vs_QW")}
        for k in ("W", "QW", "OW", "OQW")
    }
    brief["identity"] = graded["identity"]
    brief["openmeteo_replacements"] = counts
    print(json.dumps(brief, default=str)[:8000])


def run_live(args):
    LIVE.mkdir(parents=True, exist_ok=True)
    sched = pd.read_parquet(sorted(glob.glob(str(ROOT / "data/raw/2026*/schedules.parquet")))[-1])
    g = sched[(sched["season"] == v4c.LIVE_SEASON) & (sched["week"] == args.week)].copy()
    g["game_id"] = g["game_id"].astype(str)
    g["kick"], _ = v4c.kickoff_and_cutoff(g)
    coords = json.loads(COORDS.read_text(encoding="utf-8"))
    fetched = datetime.now(UTC)
    stamp = fetched.strftime("%Y%m%dT%H%M%SZ")
    rows = []
    raw = {}
    for _, r in g.iterrows():
        roof = str(r["roof"]).lower() if pd.notna(r["roof"]) else "unknown"
        row = {"game_id": r["game_id"], "stadium": r["stadium"], "roof": roof, "kickoff_utc": r["kick"].isoformat()}
        if roof in NOT_OUTDOOR:
            row["status"] = "indoor_skipped"
            rows.append(row)
            continue
        c = coords.get(r["stadium"])
        if c is None:
            row["status"] = "no_coordinates"
            rows.append(row)
            continue
        day = r["kick"].date().isoformat()
        q = urllib.parse.urlencode(
            {
                "latitude": c["lat"],
                "longitude": c["lon"],
                "hourly": LIVE_HOURLY,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "GMT",
                "start_date": day,
                "end_date": day,
            }
        )
        req = urllib.request.Request(f"{FORECAST_API}?{q}", headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        raw[r["game_id"]] = data
        h = pd.DataFrame(data.get("hourly", {}))
        h["time"] = pd.to_datetime(h["time"], utc=True)
        hit = h[h["time"].eq(r["kick"].floor("h"))]
        if hit.empty:
            row["status"] = "kick_hour_missing"
        else:
            x = hit.iloc[0]
            row.update(
                status="ok",
                temp_f=float(x["temperature_2m"]),
                wind_mph=float(x["wind_speed_10m"]),
                gust_mph=float(x["wind_gusts_10m"]),
                precip_prob_pct=float(x["precipitation_probability"]) if pd.notna(x["precipitation_probability"]) else None,
                lead_hours=round((r["kick"] - fetched).total_seconds() / 3600, 1),
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["fetched_utc"] = fetched.isoformat(timespec="seconds")
    df["source"] = "open-meteo forecast api best_match"
    df.to_csv(LIVE / f"week{args.week:02d}_{stamp}.csv", index=False)
    (LIVE / f"week{args.week:02d}_{stamp}_raw.json").write_text(
        json.dumps({"fetched_utc": fetched.isoformat(), "responses": raw}), encoding="utf-8"
    )
    print(df.to_string())


def main():
    parser = argparse.ArgumentParser(description="Open-Meteo weather for the v4 weather arm.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("agree").set_defaults(fn=run_agree)
    sub.add_parser("grade").set_defaults(fn=run_grade)
    live = sub.add_parser("live")
    live.add_argument("--week", type=int, required=True)
    live.set_defaults(fn=run_live)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
