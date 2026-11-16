from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ats.nfl_week import pool_decision_cutoff  # noqa: E402

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "forecast_archive"
    / "pool_decision_2009_2025_aware"
    / "forecasts.parquet"
)
LIVE_BULLETINS = ROOT / "data" / "raw" / "forecast_archive" / "mos_bulletins"
STATION_MAP = ROOT / "registry" / "reference" / "stadium_station_map.csv"
OUT = ROOT / "artifacts" / "weather_match_rule"
CACHE = OUT / "bulletins"
URL = "https://mesonet.agron.iastate.edu/api/1/mos.json?station={station}&model={model}&runtime={runtime}"
OPEN_AIR = ("outdoors", "open")
KNOTS_TO_MPH = 1.15078
EASTERN = ZoneInfo("America/New_York")
CURRENT_WINDOW_H = 3.0


def open_air_archive() -> pd.DataFrame:
    a = pd.read_parquet(ARCHIVE)
    a = a[(a["fetch_status"] == "ok") & a["roof"].astype(str).str.lower().isin(OPEN_AIR)].copy()
    a = a.dropna(subset=["actual_temp_f", "actual_wind_mph"])
    a["tuesday_runtime"] = a["kickoff_utc"].map(tuesday_runtime)
    return a


def tuesday_runtime(kickoff: pd.Timestamp) -> pd.Timestamp:
    et = kickoff.tz_convert(EASTERN)
    tuesday = (et - timedelta(days=(et.weekday() - 1) % 7)).normalize()
    return tuesday.tz_convert("UTC").floor("12h")


def cache_path(model: str, station: str, runtime: pd.Timestamp) -> Path:
    return CACHE / model / station / f"{runtime:%Y%m%dT%H}.json"


def fetch_one(job: tuple[str, str, pd.Timestamp]) -> str:
    model, station, runtime = job
    path = cache_path(model, station, runtime)
    if path.exists():
        return "cached"
    url = URL.format(station=station, model=model, runtime=f"{runtime:%Y-%m-%dT%H:00Z}")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"url": url, "data": payload.get("data", [])}), encoding="utf-8"
            )
            return "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"url": url, "data": [], "status": 404}), encoding="utf-8"
                )
                return "missing"
            last = repr(exc)
            time.sleep(2**attempt)
        except Exception as exc:
            last = repr(exc)
            time.sleep(2**attempt)
    return f"error {last}"


def jobs_for(a: pd.DataFrame) -> list[tuple[str, str, pd.Timestamp]]:
    jobs = set()
    for row in a.itertuples():
        jobs.add(("GFS", row.icao_station, row.issuance_runtime_utc))
        jobs.add(("MEX", row.icao_station, row.issuance_runtime_utc))
        jobs.add(("MEX", row.icao_station, row.tuesday_runtime))
    return sorted(jobs, key=lambda j: (j[2], j[0], j[1]))


def cmd_fetch(args: argparse.Namespace) -> int:
    a = open_air_archive()
    jobs = jobs_for(a)
    print(f"games {len(a)} requests {len(jobs)}", flush=True)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for i, status in enumerate(pool.map(fetch_one, jobs)):
            key = status.split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "error":
                print(jobs[i], status, flush=True)
            if i % 500 == 0:
                print(i, counts, flush=True)
    print("done", counts, flush=True)
    return 0


def projections(model: str, station: str, runtime: pd.Timestamp) -> pd.DataFrame:
    path = cache_path(model, station, runtime)
    if not path.exists():
        return pd.DataFrame()
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["data"])
    if rows.empty:
        return rows
    rows = rows.dropna(subset=["tmp", "wsp"])
    return pd.DataFrame(
        {
            "ftime": pd.to_datetime(rows["ftime"], utc=True),
            "tmp": rows["tmp"].astype(float),
            "wind": rows["wsp"].astype(float) * KNOTS_TO_MPH,
        }
    ).sort_values("ftime")


def long_table(a: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for row in a.itertuples():
        for model, runtime, arm in (
            ("GFS", row.issuance_runtime_utc, "deadline"),
            ("MEX", row.issuance_runtime_utc, "deadline"),
            ("MEX", row.tuesday_runtime, "tuesday"),
        ):
            p = projections(model, row.icao_station, runtime)
            if p.empty:
                continue
            p = p.assign(
                game_id=row.game_id,
                season=row.season,
                station=row.icao_station,
                model=model,
                arm=arm,
                lead_h=(row.kickoff_utc - runtime).total_seconds() / 3600,
                offset_h=(p["ftime"] - row.kickoff_utc).dt.total_seconds() / 3600,
                actual_temp=row.actual_temp_f,
                actual_wind=row.actual_wind_mph,
            )
            parts.append(p)
    t = pd.concat(parts, ignore_index=True)
    t["gap_h"] = t["offset_h"].abs()
    t["temp_err"] = (t["tmp"] - t["actual_temp"]).abs()
    t["wind_err"] = (t["wind"] - t["actual_wind"]).abs()
    return t


def climatology(a: pd.DataFrame) -> pd.DataFrame:
    a = a.assign(month=a["kickoff_utc"].dt.tz_convert(EASTERN).dt.month)
    out = []
    for season in sorted(a["season"].unique()):
        train = a[a["season"] != season]
        test = a[a["season"] == season]
        tm = train.groupby(["icao_station", "month"])["actual_temp_f"].mean()
        tm_station = train.groupby("icao_station")["actual_temp_f"].mean()
        wm = train.groupby("icao_station")["actual_wind_mph"].mean()
        keys = list(zip(test["icao_station"], test["month"], strict=True))
        temp = pd.Series([tm.get(k, np.nan) for k in keys], index=test.index)
        temp = temp.fillna(test["icao_station"].map(tm_station))
        wind = test["icao_station"].map(wm)
        out.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"],
                    "clim_temp_err": (temp - test["actual_temp_f"]).abs(),
                    "clim_wind_err": (wind - test["actual_wind_mph"]).abs(),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def per_bulletin(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (game, model, arm), g in t.groupby(["game_id", "model", "arm"], sort=False):
        g = g.sort_values("offset_h")
        near = g.loc[g["gap_h"].idxmin()]
        before = g[g["offset_h"] <= 0]
        after = g[g["offset_h"] >= 0]
        rec = {
            "game_id": game,
            "season": near["season"],
            "model": model,
            "arm": arm,
            "lead_h": near["lead_h"],
            "near_gap_h": near["gap_h"],
            "near_temp": near["tmp"],
            "near_wind": near["wind"],
            "actual_temp": near["actual_temp"],
            "actual_wind": near["actual_wind"],
            "bracket_h": np.nan,
            "interp_temp": np.nan,
            "interp_wind": np.nan,
        }
        if not before.empty and not after.empty:
            b = before.iloc[-1]
            f = after.iloc[0]
            width = f["offset_h"] - b["offset_h"]
            w = 0.0 if width == 0 else -b["offset_h"] / width
            rec["bracket_h"] = width
            rec["interp_temp"] = b["tmp"] + w * (f["tmp"] - b["tmp"])
            rec["interp_wind"] = b["wind"] + w * (f["wind"] - b["wind"])
        rows.append(rec)
    p = pd.DataFrame(rows)
    for v in ("temp", "wind"):
        p[f"near_{v}_err"] = (p[f"near_{v}"] - p[f"actual_{v}"]).abs()
        p[f"interp_{v}_err"] = (p[f"interp_{v}"] - p[f"actual_{v}"]).abs()
    return p


def boot_diff(p: pd.DataFrame, a_col: str, b_col: str, seed: int) -> dict[str, float]:
    d = (p[a_col] - p[b_col]).to_numpy()
    seasons = p["season"].to_numpy()
    uniq = np.unique(seasons)
    groups = {s: d[seasons == s] for s in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        draws.append(np.concatenate([groups[s] for s in pick]).mean())
    draws = np.array(draws)
    return {
        "mean": float(d.mean()),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
        "probability_negative": float((draws < 0).mean()),
        "n": len(d),
    }


def lead_band(lead_h: pd.Series) -> pd.Series:
    return pd.cut(lead_h, [0, 24, 72, 120, 240], labels=["<=1d", "1-3d", "3-5d", "5-10d"])


def cmd_analyze(args: argparse.Namespace) -> int:
    a = open_air_archive()
    t = long_table(a)
    clim = climatology(a).set_index("game_id")
    report: dict[str, object] = {"games": len(a), "projection_rows": len(t)}
    t = t[t["gap_h"] <= 24].copy()
    t["gap_bin"] = np.floor(t["gap_h"]).astype(int)
    t["lead_band"] = lead_band(t["lead_h"]).astype(str)
    clim_in = clim.reindex(t["game_id"].unique())
    report["climatology_mae"] = {
        "temp": float(clim_in["clim_temp_err"].mean()),
        "wind": float(clim_in["clim_wind_err"].mean()),
    }
    curve = (
        t.groupby(["model", "lead_band", "gap_bin"], observed=True)
        .agg(n=("temp_err", "size"), temp_mae=("temp_err", "mean"), wind_mae=("wind_err", "mean"))
        .reset_index()
    )
    t2 = t.join(clim, on="game_id")
    curve_rel = (
        t2.groupby(["model", "lead_band", "gap_bin"], observed=True)
        .agg(temp_clim=("clim_temp_err", "mean"), wind_clim=("clim_wind_err", "mean"))
        .reset_index()
    )
    curve = curve.merge(curve_rel, on=["model", "lead_band", "gap_bin"])
    curve.to_csv(OUT / "gap_curve.csv", index=False)
    cuts = {}
    for (model, band), g in curve.groupby(["model", "lead_band"]):
        g = g.sort_values("gap_bin")
        entry = {}
        for v in ("temp", "wind"):
            worse = g[g[f"{v}_mae"] >= g[f"{v}_clim"]]
            entry[f"{v}_first_gap_bin_not_beating_climatology"] = (
                int(worse["gap_bin"].iloc[0]) if not worse.empty else None
            )
        entry["max_gap_bin_seen"] = int(g["gap_bin"].max()) if not g.empty else None
        cuts[f"{model}|{band}"] = entry
    report["climatology_crossing"] = cuts
    p = per_bulletin(t)
    p = p.join(clim, on="game_id")
    p["lead_band"] = lead_band(p["lead_h"]).astype(str)
    p.to_parquet(OUT / "per_bulletin.parquet", index=False)
    both = p.dropna(subset=["interp_temp"])
    comp = {}
    for (model, arm), g in both.groupby(["model", "arm"]):
        comp[f"{model}|{arm}"] = {
            "bracket_h_median": float(g["bracket_h"].median()),
            "near_temp_mae": float(g["near_temp_err"].mean()),
            "interp_temp_mae": float(g["interp_temp_err"].mean()),
            "near_wind_mae": float(g["near_wind_err"].mean()),
            "interp_wind_mae": float(g["interp_wind_err"].mean()),
            "clim_temp_mae": float(g["clim_temp_err"].mean()),
            "clim_wind_mae": float(g["clim_wind_err"].mean()),
            "interp_minus_near_temp": boot_diff(g, "interp_temp_err", "near_temp_err", 1),
            "interp_minus_near_wind": boot_diff(g, "interp_wind_err", "near_wind_err", 2),
        }
    report["interp_vs_nearest"] = comp
    near_curve = (
        p.assign(near_bin=np.floor(p["near_gap_h"]).astype(int))
        .groupby(["model", "arm", "near_bin"])
        .agg(
            n=("near_temp_err", "size"),
            near_temp=("near_temp_err", "mean"),
            interp_temp=("interp_temp_err", "mean"),
            near_wind=("near_wind_err", "mean"),
            interp_wind=("interp_wind_err", "mean"),
            clim_temp=("clim_temp_err", "mean"),
            clim_wind=("clim_wind_err", "mean"),
        )
        .reset_index()
    )
    near_curve.to_csv(OUT / "nearest_gap_curve.csv", index=False)
    ag = a.assign(
        archive_gap_h=(a["forecast_valid_utc"] - a["kickoff_utc"]).abs().dt.total_seconds() / 3600
    )
    report["archive_gap_h_counts"] = {
        "le_3": int((ag["archive_gap_h"] <= CURRENT_WINDOW_H).sum()),
        "gt_3": int((ag["archive_gap_h"] > CURRENT_WINDOW_H).sum()),
        "max": float(ag["archive_gap_h"].max()),
        "quantiles": {str(q): float(ag["archive_gap_h"].quantile(q)) for q in (0.5, 0.9, 0.99)},
    }
    gfs = p[(p["model"] == "GFS") & (p["arm"] == "deadline")].set_index("game_id")
    arch = ag.set_index("game_id")
    shared = gfs.index.intersection(arch.index)
    report["archive_vs_refetched_gfs_temp_equal"] = int(
        (gfs.loc[shared, "near_temp"] == arch.loc[shared, "forecast_temp_f"]).sum()
    )
    report["archive_rows_with_bracket"] = int(gfs.loc[shared, "interp_temp"].notna().sum())
    report["archive_rows_interp_changes_temp_or_wind"] = int(
        (
            (gfs.loc[shared, "interp_temp"].round(6) != gfs.loc[shared, "near_temp"].round(6))
            | (gfs.loc[shared, "interp_wind"].round(6) != gfs.loc[shared, "near_wind"].round(6))
        )
        .loc[gfs.loc[shared, "interp_temp"].notna()]
        .sum()
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(curve.to_string())
    print(near_curve.to_string())
    return 0


def live_candidates(schedules: Path) -> pd.DataFrame:
    s = pd.read_parquet(schedules)
    s = s[(s["season"] == 2026) & s["roof"].astype(str).str.lower().isin(OPEN_AIR)].copy()
    st = pd.read_csv(STATION_MAP)[["stadium", "icao_station"]].drop_duplicates("stadium")
    s = s.merge(st, on="stadium", how="left").dropna(subset=["icao_station", "gametime"])
    naive = pd.to_datetime(s["gameday"].astype(str) + " " + s["gametime"].astype(str))
    s["kickoff_utc"] = naive.dt.tz_localize(EASTERN).dt.tz_convert("UTC")
    return s


def live_bulletins(model: str, station: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    out = []
    for path in sorted((LIVE_BULLETINS / model / station).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = pd.DataFrame(payload.get("data", []))
        if rows.empty:
            continue
        rows = rows.dropna(subset=["tmp", "wsp"])
        frame = pd.DataFrame(
            {
                "ftime": pd.to_datetime(rows["ftime"], utc=True),
                "runtime": pd.to_datetime(rows["runtime"], utc=True),
                "tmp": rows["tmp"].astype(float),
                "wind": rows["wsp"].astype(float) * KNOTS_TO_MPH,
            }
        )
        out.append((pd.Timestamp(payload["cached_at_utc"]), frame))
    return out


def cmd_live(args: argparse.Namespace) -> int:
    cut = args.cut_h
    s = live_candidates(Path(args.schedules))
    rows = []
    for g in s.itertuples():
        cutoff = pd.Timestamp(pool_decision_cutoff(g.kickoff_utc.to_pydatetime()))
        rec = {"game_id": g.game_id, "week": g.week, "obs_temp": g.temp, "obs_wind": g.wind}
        for model in ("GFS", "MEX"):
            best = None
            for cached_at, frame in live_bulletins(model, g.icao_station):
                if cached_at > cutoff:
                    continue
                gap = (frame["ftime"] - g.kickoff_utc).abs() / pd.Timedelta(hours=1)
                i = int(
                    np.lexsort((-frame["runtime"].astype("int64").to_numpy(), gap.to_numpy()))[0]
                )
                key = (gap.iloc[i], -frame["runtime"].iloc[i].value)
                if best is None or key < best[0]:
                    best = (key, frame, i)
            if best is None:
                continue
            (gap_h, _), frame, i = best
            rec[f"{model}_gap_h"] = gap_h
            rec[f"{model}_near_temp"] = frame["tmp"].iloc[i]
            rec[f"{model}_near_wind"] = frame["wind"].iloc[i]
            run = frame[frame["runtime"] == frame["runtime"].iloc[i]]
            off = (run["ftime"] - g.kickoff_utc) / pd.Timedelta(hours=1)
            before, after = run[off <= 0], run[off >= 0]
            if not before.empty and not after.empty:
                b, f = before.iloc[-1], after.iloc[0]
                width = (f["ftime"] - b["ftime"]) / pd.Timedelta(hours=1)
                w = (
                    0.0
                    if width == 0
                    else ((g.kickoff_utc - b["ftime"]) / pd.Timedelta(hours=1)) / width
                )
                rec[f"{model}_interp_temp"] = b["tmp"] + w * (f["tmp"] - b["tmp"])
                rec[f"{model}_interp_wind"] = b["wind"] + w * (f["wind"] - b["wind"])
        rows.append(rec)
    out = pd.DataFrame(rows)
    for col in ("GFS_gap_h", "MEX_gap_h", "GFS_interp_temp", "GFS_interp_wind"):
        if col not in out:
            out[col] = np.nan
    gfs_first = out["GFS_gap_h"].fillna(np.inf) <= out["MEX_gap_h"].fillna(np.inf)
    old_model = np.where(gfs_first, "GFS", "MEX")
    old_gap = out[["GFS_gap_h", "MEX_gap_h"]].min(axis=1)
    old_ok = old_gap <= CURRENT_WINDOW_H
    out["old_temp"] = np.where(
        old_ok, np.where(gfs_first, out["GFS_near_temp"], out["MEX_near_temp"]), np.nan
    )
    out["old_wind"] = np.where(
        old_ok, np.where(gfs_first, out["GFS_near_wind"], out["MEX_near_wind"]), np.nan
    )
    gfs_ok = out["GFS_gap_h"] < cut
    mex_ok = out["MEX_gap_h"] < cut
    gfs_temp = out["GFS_interp_temp"].fillna(out["GFS_near_temp"])
    gfs_wind = out["GFS_interp_wind"].fillna(out["GFS_near_wind"])
    out["new_temp"] = np.where(gfs_ok, gfs_temp, np.where(mex_ok, out["MEX_near_temp"], np.nan))
    out["new_wind"] = np.where(gfs_ok, gfs_wind, np.nan)
    out["old_model"] = np.where(old_ok, old_model, "")
    out.to_csv(OUT / "live_2026.csv", index=False)
    summary = {"open_air_2026_rows": len(out)}
    for v in ("temp", "wind"):
        o, n = out[f"old_{v}"], out[f"new_{v}"]
        summary[v] = {
            "old_covered": int(o.notna().sum()),
            "new_covered": int(n.notna().sum()),
            "gained": int((o.isna() & n.notna()).sum()),
            "lost": int((o.notna() & n.isna()).sum()),
            "value_changed": int((o.notna() & n.notna() & ((o - n).abs() > 1e-9)).sum()),
        }
        played = out[f"obs_{v}"].notna()
        both = played & o.notna() & n.notna()
        summary[v]["played_both_n"] = int(both.sum())
        summary[v]["played_both_old_mae"] = float((o - out[f"obs_{v}"]).abs()[both].mean())
        summary[v]["played_both_new_mae"] = float((n - out[f"obs_{v}"]).abs()[both].mean())
    summary["old_wind_from_mex"] = int((old_ok & (old_model == "MEX")).sum())
    a = open_air_archive()
    gap = (a["forecast_valid_utc"] - a["kickoff_utc"]).abs() / pd.Timedelta(hours=1)
    summary["training"] = {
        "rows": len(a),
        "within_3h": int((gap <= CURRENT_WINDOW_H).sum()),
        "within_cut": int((gap < cut).sum()),
        "in_3h_to_cut": int(((gap > CURRENT_WINDOW_H) & (gap < cut)).sum()),
        "at_or_beyond_cut": int((gap >= cut).sum()),
    }
    (OUT / "live_2026.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure MOS forecast error against projection gap and lead."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="Fetch GFS and MEX bulletins for archived open-air games.")
    f.add_argument("--workers", type=int, default=4)
    sub.add_parser("analyze", help="Build gap and lead error curves.")
    lv = sub.add_parser(
        "live", help="Count 2026 rows matched under the 3 h rule and a derived rule."
    )
    lv.add_argument("--schedules", required=True)
    lv.add_argument(
        "--cut-h",
        type=float,
        required=True,
        help="Projection gap in hours at which the analyze curve stops beating climatology.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"fetch": cmd_fetch, "analyze": cmd_analyze, "live": cmd_live}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
