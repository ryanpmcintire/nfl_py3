import glob
import json
import sys
from pathlib import Path

import mkt21_line_path as M
import mkt23_unit2_term as U
import mkt26_move_dispersion as K
import mkt26_unit2_dispersion as V
import numpy as np
import pandas as pd
import sky04_lead_lag as L
from scipy.optimize import minimize_scalar

from nfl_ats import mass_preserving_lattice as P

OUT = K.OUT
FAMILY = K.FAMILY
ROOT = L.ROOT
OPENER_EVAL = ROOT / "artifacts/opener_evaluation/20260924T005838Z/per_game.parquet"
SERIES_2026 = OUT / "unit3_book_series_2026.parquet"
RAW = ROOT / "data/market/raw"
EXCLUDED_SUFFIXES = ("-reparse", "-ncaaf", "-futures")
SEASON_LIVE = 2026
WEEKS_LIVE = tuple(range(1, 11))
HIST = K.SEASONS
LOOKS = 2
POWER_Z = 2.8
PLAIN = (
    "Whether reading the margin distribution from past games whose spread moved the same way "
    "(moved or unmoved) improves the cover probability re-served at the deadline line, checked "
    "on the live 2026 season at the pool line."
)


def frames():
    gf = pd.read_parquet(
        ROOT / "data/processed/game_features.parquet",
        columns=["game_id", "season", "week", "gameday", "game_type", "result", "spread_line", "kickoff"],
    ).drop_duplicates("game_id")
    gf["game_id"] = gf.game_id.astype(str)
    gf["gameday"] = pd.to_datetime(gf.gameday)
    oe = pd.read_parquet(OPENER_EVAL)
    oe["game_id"] = oe.game_id.astype(str)
    return gf, oe


def move_classes(oe):
    f, _, _ = V.sources()
    cls = f.set_index("game_id").moved
    late = oe.set_index("game_id")
    late = ((late.close_home_spread - late.tue_open_home_spread).abs() > 0).astype(float)
    late = late[late.index.str[:4].astype(int) > V.SBR_LAST]
    cls = cls[~cls.index.isin(late.index)]
    return pd.concat([cls, late])


def week_arrays(pool, season, week, cutoff, classes=("all", "mv", "un")):
    el = P.prior_pool_for_week(pool, season=int(season), week=int(week), cutoff=pd.Timestamp(cutoff))
    parts = {"all": el, "mv": el[el.moved == 1], "un": el[el.moved == 0]}
    return {k: (parts[k].line.to_numpy(float), parts[k].result.to_numpy(float)) for k in classes}


def read(arr, anchor, point, line):
    v, m = V.lattice(*arr, anchor, point)
    return V.cond_cover(v, m, line)


def fit_location(arr, rows):
    a = rows.tue_open_home_spread.to_numpy(float)
    b = a + rows.residual_at_open_served.to_numpy(float)
    t = rows.home_cover_probability_at_open.to_numpy(float)
    width = float(rows.residual_at_open_served.std(ddof=0))

    def obj(loc):
        return sum((read(arr, x, y + loc, x) - z) ** 2 for x, y, z in zip(a, b, t, strict=True))

    r = minimize_scalar(obj, bracket=(-width, width))
    return float(r.x)


def historical(gf, oe, cutoff, pool):
    h = oe[oe.season.isin(HIST)].reset_index(drop=True)
    locs, fid = {}, []
    for (s, w), grp in h.groupby(["season", "week"], sort=True):
        arr = week_arrays(pool, s, w, cutoff[(s, w)], ("all",))["all"]
        loc = fit_location(arr, grp)
        locs[(int(s), int(w))] = loc
        for r in grp.itertuples():
            fid.append(
                {
                    "game_id": r.game_id,
                    "open_fit": read(arr, r.tue_open_home_spread, r.tue_open_home_spread + r.residual_at_open_served + loc, r.tue_open_home_spread) - r.home_cover_probability_at_open,
                    "close_check": read(arr, r.close_home_spread, r.close_home_spread + r.residual_at_close_served + loc, r.close_home_spread) - r.home_cover_probability_at_close,
                    "close_no_location": read(arr, r.close_home_spread, r.close_home_spread + r.residual_at_close_served, r.close_home_spread) - r.home_cover_probability_at_close,
                }
            )
    fid = pd.DataFrame(fid)
    summary = {
        k: {"n": len(fid), "mean_abs_diff": float(fid[k].abs().mean()), "max_abs_diff": float(fid[k].abs().max())}
        for k in ("open_fit", "close_check", "close_no_location")
    }
    lv = np.array(list(locs.values()))
    summary["location_by_week"] = {"weeks": len(lv), "mean": float(lv.mean()), "min": float(lv.min()), "max": float(lv.max())}
    pd.DataFrame([{"season": s, "week": w, "residual_location": v} for (s, w), v in locs.items()]).to_parquet(OUT / "unit3_residual_location_2023_25.parquet")
    return locs, summary


def l2_hist(oe, locs, cutoff, pool, rng):
    d = K.deadline_frame(keep_pushes=True)
    d = d.dropna(subset=["tue_open_home_spread", "residual_at_close_served", "hs1"]).reset_index(drop=True)
    d["game_id"] = d.game_id.astype(str)
    d["loc"] = [locs.get((int(s), int(w)), np.nan) for s, w in zip(d.season, d.week, strict=True)]
    d = d.dropna(subset=["loc"]).reset_index(drop=True)
    d["point_dl"] = d.hs1 + d.residual_at_close_served + d["loc"]
    d["cls"] = (d.abs_move > 0).astype(float)
    rows = []
    for (s, w), grp in d.groupby(["season", "week"], sort=True):
        arr = week_arrays(pool, s, w, cutoff[(s, w)])
        for r in grp.itertuples():
            rows.append({"game_id": r.game_id, **{f"p_{k}": read(arr[k], r.hs1, r.point_dl, r.tue_open_home_spread) for k in arr}})
    d = d.merge(pd.DataFrame(rows), on="game_id")
    q = d[d.result != d.tue_open_home_spread].reset_index(drop=True)
    y = (q.result > q.tue_open_home_spread).to_numpy(float)
    pv, pu = q.p_mv.to_numpy(), q.p_un.to_numpy()
    res = V.compare(y, q.p_all.to_numpy(), np.where(q.cls == 1, pv, pu), q.season.values, rng, alt=(pv, pu, q.cls.to_numpy()))
    res["share_moved"] = float(q.cls.mean())
    return res


def series_2026():
    if SERIES_2026.exists():
        return pd.read_parquet(SERIES_2026)
    cols = ["observed_at_utc", "nflverse_game_id", "bookmaker_key", "market", "home_spread_line"]
    rows, used, skipped = [], 0, 0
    for d in sorted(glob.glob(str(RAW / f"{SEASON_LIVE}*"))):
        name = Path(d).name
        if name.endswith(EXCLUDED_SUFFIXES):
            skipped += 1
            continue
        f = Path(d) / "quotes.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f, columns=cols)
        q = q[(q.market == "spreads") & q.nflverse_game_id.astype(str).str.startswith(f"{SEASON_LIVE}_")]
        if len(q):
            used += 1
            rows.append(q.groupby(["nflverse_game_id", "bookmaker_key", "observed_at_utc"], as_index=False).home_spread_line.median())
    s = pd.concat(rows, ignore_index=True)
    s["observed_at_utc"] = pd.to_datetime(s.observed_at_utc, utc=True)
    s = s.drop_duplicates(["nflverse_game_id", "bookmaker_key", "observed_at_utc"]).sort_values("observed_at_utc")
    s.attrs = {}
    s.to_parquet(SERIES_2026)
    (OUT / "unit3_series_2026_sources.json").write_text(json.dumps({"folders_used": used, "folders_excluded_by_suffix": skipped, "excluded_suffixes": EXCLUDED_SUFFIXES}))
    return s


def forecasts_2026():
    out = []
    for d in sorted(glob.glob(str(ROOT / f"artifacts/margin_predictions/{SEASON_LIVE}-week-*"))):
        f = Path(d) / "discrete_push_read.json"
        if not f.exists():
            continue
        name = Path(d).name
        created = pd.to_datetime(name.split("-")[3], format="%Y%m%dT%H%M%SZ", errors="coerce")
        if pd.isna(created):
            continue
        created = created.tz_localize("UTC")
        j = json.loads(f.read_text())
        if not j.get("served") or not isinstance(j.get("games"), list):
            continue
        for g in j["games"]:
            out.append({"game_id": str(g["game_id"]), "created": created, "point": float(g["point"]), "fc_line": float(g["spread_line"]), "artifact": name})
    return pd.DataFrame(out)


def pool_lines_2026():
    out = {}
    for f in sorted(glob.glob(str(ROOT / f"data/splash/{SEASON_LIVE}_week*.json"))):
        for g in json.loads(Path(f).read_text()).get("games", []):
            out[str(g["game_id"])] = float(g["home_spread"])
    return pd.Series(out, name="pool_line")


def nearest_at(q, t, t_end, value):
    tt, te = pd.Timestamp(t, unit="s", tz="UTC"), pd.Timestamp(t_end, unit="s", tz="UTC")
    qq = q[q.observed_at_utc <= te]
    qq = qq.assign(gap=(qq.observed_at_utc - tt).abs()).sort_values("gap").groupby("bookmaker_key").first()
    return qq[value], float(qq.gap.median() / pd.Timedelta(hours=1)) if len(qq) else np.nan


def live(gf, oe, cutoff, pool, slope, rng):
    s = series_2026()
    g = gf[(gf.season == SEASON_LIVE) & (gf.game_type == "REG") & gf.week.isin(WEEKS_LIVE) & gf.result.notna()].copy()
    rows = []
    for r in g.itertuples():
        q = s[s.nflverse_game_id == r.game_id]
        if q.empty:
            continue
        t0, t1 = L.decision_ts(r.gameday), M.deadline_ts(r.gameday, r.kickoff)
        a, gap = nearest_at(q, t0, t1, "home_spread_line")
        b = K.last_at(q, t1, "home_spread_line")
        if a.empty or b.empty:
            continue
        rows.append({"game_id": r.game_id, "hs0": float(a.median()), "hs0_gap_hours": gap, "hs1": float(b.median()), "t1": pd.Timestamp(t1, unit="s", tz="UTC"), "books": len(b)})
    x = g.merge(pd.DataFrame(rows), on="game_id")
    fc = forecasts_2026()
    fc = fc.merge(x[["game_id", "t1"]], on="game_id")
    fc = fc[fc.created < fc.t1].sort_values("created").groupby("game_id").last().reset_index()
    x = x.merge(fc[["game_id", "point", "fc_line", "artifact"]], on="game_id")
    x = x.merge(pool_lines_2026(), left_on="game_id", right_index=True)
    x["result"] = np.rint(x.result.astype(float))
    x["abs_move"] = (x.hs1 - x.hs0).abs().round(4)
    x["cls"] = (x.abs_move > 0).astype(float)
    x["point_dl"] = x.point + (1 + slope) * (x.hs1 - x.fc_line)
    pool = pool.copy()
    lm = x.set_index("game_id").cls
    pool["moved"] = pool.moved.where(~pool.game_id.isin(lm.index), pool.game_id.map(lm))
    out = []
    for (s_, w), grp in x.groupby(["season", "week"], sort=True):
        arr = week_arrays(pool, s_, w, cutoff[(s_, w)])
        for r in grp.itertuples():
            out.append({"game_id": r.game_id, **{f"p_{k}": read(arr[k], r.hs1, r.point_dl, r.pool_line) for k in arr}})
    x = x.merge(pd.DataFrame(out), on="game_id")
    x.drop(columns=["t1"]).to_parquet(OUT / "unit3_live_rows_2026.parquet")
    push = x.result == x.pool_line
    q = x[~push].reset_index(drop=True)
    y = (q.result > q.pool_line).to_numpy(float)
    pv, pu = q.p_mv.to_numpy(), q.p_un.to_numpy()
    res = V.compare(y, q.p_all.to_numpy(), np.where(q.cls == 1, pv, pu), q.week.values, rng, alt=(pv, pu, q.cls.to_numpy()))
    res.update(
        {
            "games_with_result": int(len(g)),
            "games_read": int(len(x)),
            "pushes_at_pool_line": int(push.sum()),
            "whole_number_pool_lines": int((x.pool_line % 1 == 0).sum()),
            "share_moved": float(q.cls.mean()),
            "mean_abs_move": float(x.abs_move.mean()),
            "mean_abs_deadline_minus_pool_line": float((x.hs1 - x.pool_line).abs().mean()),
            "mean_books_at_deadline": float(x.books.mean()),
            "tuesday_quote_gap_hours_median_max": [float(x.hs0_gap_hours.median()), float(x.hs0_gap_hours.max())],
            "corr_tuesday_capture_vs_pool_line": float(np.corrcoef(x.hs0, x.pool_line)[0, 1]),
            "mean_abs_tuesday_capture_minus_pool_line": float((x.hs0 - x.pool_line).abs().mean()),
            "detectable_log_loss_gain_80pct": float(POWER_Z * res["standard_error"]),
            "base_record_at_pool_line": f"{int(((q.p_all > 0.5) == (y == 1)).sum())}-{int(((q.p_all > 0.5) != (y == 1)).sum())}",
            "arm_record_at_pool_line": f"{int(((np.where(q.cls == 1, pv, pu) > 0.5) == (y == 1)).sum())}-{int(((np.where(q.cls == 1, pv, pu) > 0.5) != (y == 1)).sum())}",
        }
    )
    return res


def run():
    rng = np.random.default_rng(L.SEED)
    gf, oe = frames()
    cutoff = gf.groupby(["season", "week"]).gameday.min()
    pool = P.prior_pool(gf, oe.set_index("game_id").tue_open_home_spread)
    pool["moved"] = pool.game_id.map(move_classes(oe))
    h = oe[oe.season.isin(HIST)]
    slope = float(np.polyfit(h.close_home_spread - h.tue_open_home_spread, h.residual_at_close_served - h.residual_at_open_served, 1)[0])
    locs, fid = historical(gf, oe, cutoff, pool)
    res = {"family": FAMILY, "looks": LOOKS, "fidelity": fid, "residual_slope_on_line_2023_25": slope}
    res["L2_hist_located"] = l2_hist(oe, locs, cutoff, pool, rng)
    res["L2_hist_located"]["detectable_log_loss_gain_80pct"] = float(POWER_Z * res["L2_hist_located"]["standard_error"])
    res["L2_live_2026_pool_line"] = live(gf, oe, cutoff, pool, slope, rng)
    (OUT / "unit3_result.json").write_text(json.dumps(res, indent=1, default=float))
    ev = f"{LOOKS} looks family {FAMILY}; no positive control"
    batches = {
        "unit3_record_batch_2026.json": (SEASON_LIVE, SEASON_LIVE, res["L2_live_2026_pool_line"]["games_read"], [
            V.cell("mkt26u3_L2_2026_reserve_pool_line_graded", res["L2_live_2026_pool_line"], "log_loss_improvement", f"2026 Weeks 1-10 live deadline captures (reparse folders excluded), re-served lattice at deadline line, same move-class pool vs full pool, graded at the pool line; {ev}"),
        ]),
        "unit3_record_batch_hist.json": (min(HIST), max(HIST), res["L2_hist_located"]["n"], [
            V.cell("mkt26u3_L2_2023_25_reserve_located_opener_graded", res["L2_hist_located"], "log_loss_improvement", f"unit 2 L2 rerun with per-week residual location reconstructed from stored opener reads and the 1537-game opener pool; {ev}"),
        ]),
    }
    for fname, (s0, s1, n, cells) in batches.items():
        batch = {
            "classification": "unresolved_below_power",
            "league": "nfl",
            "season_start": s0,
            "season_end": s1,
            "family": FAMILY,
            "category": "market",
            "source": "artifacts/mkt26/unit3_result.json",
            "sample_games": n,
            "plain_summary": PLAIN,
            "description": f"Unit 3: re-served deadline lattice, move-class pool; {LOOKS} looks",
            "cells": cells,
        }
        (OUT / fname).write_text(json.dumps(batch, indent=1))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(run())
