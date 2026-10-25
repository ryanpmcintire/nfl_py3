import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

PBP_DIR = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
OUT = REPO / "artifacts" / "sim10"
SEASONS = tuple(range(2009, 2018))
COLS = [
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "fixed_drive",
    "down",
    "play_type",
    "yardline_100",
    "game_seconds_remaining",
    "qtr",
    "posteam_score",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "epa",
    "yards_gained",
    "sack",
    "interception",
    "fumble_lost",
]
DIMS = ["yl", "sd", "lt", "too", "tod", "x", "season", "week"]
LAST_REG_WEEK = 17
HALF_SECS = {1: 1800.0, 2: 1800.0}
SUDDEN_DEATH = {2009, 2010, 2011}


def load_pbp():
    fr = []
    for s in SEASONS:
        f = pd.read_parquet(PBP_DIR / f"season={s}" / "plays.parquet", columns=COLS)
        fr.append(f[f["season_type"] == "REG"])
    p = (
        pd.concat(fr, ignore_index=True)
        .sort_values(["game_id", "play_id"], kind="stable")
        .reset_index(drop=True)
    )
    g = p["game_id"]
    p["gsr"] = p["game_seconds_remaining"].groupby(g).ffill().groupby(g).bfill().fillna(0.0)
    ok = (
        p["posteam"].notna() & p["posteam_score"].notna() & p["score_differential"].notna()
    ).to_numpy()
    ph = (p["posteam"] == p["home_team"]).to_numpy()
    ps = p["posteam_score"].to_numpy(float)
    sdf = p["score_differential"].to_numpy(float)
    for col, val in (("hs", np.where(ph, ps, ps - sdf)), ("as", np.where(ph, ps - sdf, ps))):
        p[col] = pd.Series(np.where(ok, val, np.nan)).groupby(g).ffill().fillna(0.0).to_numpy()
    p["half"] = np.where(p["qtr"] <= 2, 1, np.where(p["qtr"] <= 4, 2, 3))
    return p


def real_games():
    import mod25_generator as gen

    return gen.real_games(SEASONS)


def team_ratings(p):
    r = p[p["play_type"].isin(["run", "pass"]) & p["posteam"].notna()].copy()
    r["epa"] = r["epa"].fillna(0.0)
    off = (
        r.groupby(["game_id", "season", "posteam"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "off"})
    )
    de = (
        r.groupby(["game_id", "season", "defteam"], as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team", "epa": "dfn"})
    )
    t = off.merge(de, on=["game_id", "season", "team"], how="outer")
    for c in ("off", "dfn"):
        grp = t.groupby(["season", "team"])[c]
        s, n = grp.transform("sum"), grp.transform("count")
        t[f"{c}_logo"] = (s - t[c].fillna(0.0)) / (n - t[c].notna().astype(float))
        t[f"{c}_c"] = t.groupby("season")[c].transform("mean")
    return t


def build(args):
    t0 = time.time()
    p = load_pbp()
    games = real_games()
    games = games[games["game_id"].isin(p["game_id"].unique())].reset_index(drop=True)
    fin = games.set_index("game_id")[["home_score", "away_score"]]
    p = p[p["game_id"].isin(fin.index)].reset_index(drop=True)
    n = len(p)
    gid = p["game_id"].to_numpy()
    same_next = np.r_[gid[1:] == gid[:-1], False]
    hs, as_ = p["hs"].to_numpy(float), p["as"].to_numpy(float)
    fh = fin["home_score"].reindex(gid).to_numpy(float)
    fa = fin["away_score"].reindex(gid).to_numpy(float)
    hs_post = np.where(same_next, np.r_[hs[1:], 0.0], fh)
    as_post = np.where(same_next, np.r_[as_[1:], 0.0], fa)
    gsr = p["gsr"].to_numpy(float)
    t_ev = np.where(same_next, np.r_[gsr[1:], 0.0], gsr)
    dh, da = hs_post - hs, as_post - as_
    ev_rows = np.flatnonzero((dh != 0) | (da != 0))
    half = p["half"].to_numpy()
    print(
        "ot gsr range",
        float(p.loc[p["half"] == 3, "gsr"].min()),
        float(p.loc[p["half"] == 3, "gsr"].max()),
        flush=True,
    )

    tr = team_ratings(p)
    key = tr.set_index(["game_id", "team"])
    snap = p["down"].notna() & p["posteam"].notna() & p["fixed_drive"].notna()
    first = p[snap].groupby(["game_id", "fixed_drive"], sort=False).head(1).index.to_numpy()
    starts = np.sort(first)
    sg, sh = gid[starts], half[starts]
    nxt_same = np.r_[(sg[1:] == sg[:-1]) & (sh[1:] == sh[:-1]), False]
    hk = pd.Series(np.arange(n)).groupby([gid, half]).transform("max").to_numpy() + 1
    end = np.where(nxt_same, np.r_[starts[1:], 0], hk[starts])
    nstart = np.where(nxt_same, np.r_[starts[1:], 0], -1)

    pos = p["posteam"].to_numpy(object)
    dft = p["defteam"].to_numpy(object)
    home = p["home_team"].to_numpy(object)
    pos_home = pos == home
    yl = p["yardline_100"].to_numpy(float)
    sd = p["score_differential"].to_numpy(float)
    too = p["posteam_timeouts_remaining"].fillna(3).to_numpy(float)
    tod = p["defteam_timeouts_remaining"].fillna(3).to_numpy(float)
    season = p["season"].to_numpy(int)
    wk = p["week"].to_numpy(float)
    ot_total = {}
    for g_, d in p[p["half"] == 3].groupby("game_id"):
        ot_total[g_] = 600.0 if int(d["season"].iloc[0]) >= 2017 else 900.0

    def secs_left(r):
        h = half[r]
        if h == 1:
            return gsr[r] - 1800.0
        return gsr[r]

    ns = len(starts)
    S = {
        k: np.zeros(ns)
        for k in (
            "yl",
            "sd",
            "secs",
            "too",
            "tod",
            "x",
            "season",
            "half",
            "home",
            "el",
            "he",
            "nsame",
            "nyl",
            "uo",
            "ud",
            "week",
        )
    }
    ev_ptr = np.zeros(ns + 1, dtype=np.int64)
    ev_list = []
    pl_ptr = np.zeros(ns + 1, dtype=np.int64)
    pl_list = []
    rp = p["play_type"].isin(["run", "pass"]).to_numpy()
    x_off = key["off_logo"] - key["off_c"]
    x_def = key["dfn_logo"] - key["dfn_c"]
    xo = x_off.reindex(pd.MultiIndex.from_arrays([gid[starts], pos[starts]])).to_numpy(float)
    xd = x_def.reindex(pd.MultiIndex.from_arrays([gid[starts], dft[starts]])).to_numpy(float)
    xs = np.nan_to_num(xo, nan=0.0) + np.nan_to_num(xd, nan=0.0)
    for k in range(ns):
        r, e = starts[k], end[k]
        S["yl"][k] = yl[r]
        S["sd"][k] = sd[r]
        sl = secs_left(r)
        S["secs"][k] = sl
        S["too"][k] = too[r]
        S["tod"][k] = tod[r]
        S["x"][k] = xs[k]
        S["season"][k] = season[r]
        S["week"][k] = wk[r]
        S["half"][k] = half[r]
        S["home"][k] = pos_home[r]
        a, b = np.searchsorted(ev_rows, r), np.searchsorted(ev_rows, e)
        for i in ev_rows[a:b]:
            po = dh[i] if pos_home[r] else da[i]
            pd_ = da[i] if pos_home[r] else dh[i]
            ev_list.append((min(gsr[r] - t_ev[i], sl), po, pd_))
        ev_ptr[k + 1] = len(ev_list)
        rows = np.arange(r, e)
        rows = rows[rp[r:e] & (pos[r:e] == pos[r])]
        for i in rows:
            pl_list.append((i, min(gsr[r] - gsr[i], sl)))
        pl_ptr[k + 1] = len(pl_list)
        m = nstart[k]
        if m >= 0:
            S["el"][k] = gsr[r] - gsr[m]
            S["he"][k] = 0
            same = pos[m] == pos[r]
            S["nsame"][k] = same
            S["nyl"][k] = yl[m]
            S["uo"][k] = max(0.0, too[r] - (too[m] if same else tod[m]))
            S["ud"][k] = max(0.0, tod[r] - (tod[m] if same else too[m]))
        else:
            S["el"][k] = sl
            S["he"][k] = 1
    seg = pd.DataFrame(S)
    seg["game_id"] = gid[starts]
    seg["row"] = starts
    ev = np.array(ev_list, dtype=np.float64).reshape(-1, 3)
    kk = np.repeat(np.arange(ns), np.diff(ev_ptr))
    spo = np.bincount(kk, weights=ev[:, 1], minlength=ns)
    spd = np.bincount(kk, weights=ev[:, 2], minlength=ns)
    last = np.zeros(ns)
    np.maximum.at(last, kk, ev[:, 0])
    scored = (spo != 0) | (spd != 0)
    post = (seg["he"] == 0) & scored & (seg["half"] < 3)
    med = seg[post].groupby("season")["nyl"].median()
    otw = (seg["half"] == 3) & (seg["he"] == 1) & scored
    seg.loc[otw, "he"] = 0
    seg.loc[otw, "el"] = last[otw.to_numpy()]
    seg.loc[otw, "nsame"] = 0
    seg.loc[otw, "nyl"] = seg.loc[otw, "season"].map(med).to_numpy()
    fin_tied = (fin["home_score"] == fin["away_score"]).reindex(seg["game_id"]).to_numpy()
    otr = ((seg["half"] == 3) & (seg["he"] == 1) & ~scored).to_numpy() & ~fin_tied
    snaps = snap.to_numpy()
    for k in np.flatnonzero(otr):
        r = starts[k]
        rows = np.arange(r, end[k])
        ls = rows[snaps[r : end[k]]][-1]
        seg.loc[k, ["he", "el", "nsame", "nyl"]] = [0, gsr[r] - gsr[ls], 0, 100.0 - yl[ls]]
    pl = np.array(pl_list, dtype=np.float64).reshape(-1, 2)

    op = []
    for (g_, h), d in p.groupby(["game_id", "half"], sort=False):
        fs = d.index[snap.loc[d.index].to_numpy()]
        if len(fs) == 0:
            continue
        f0, r = d.index[0], fs[0]
        tot = HALF_SECS.get(h, ot_total.get(g_, 900.0))
        sl = secs_left(r)
        a, b = np.searchsorted(ev_rows, f0), np.searchsorted(ev_rows, r)
        ph_ = sum(dh[i] for i in ev_rows[a:b])
        pa_ = sum(da[i] for i in ev_rows[a:b])
        op.append((g_, h, int(season[r]), int(pos_home[r]), yl[r], sl, tot - sl, ph_, pa_))
    ops = pd.DataFrame(
        op, columns=["game_id", "half", "season", "recv_home", "yl", "secs", "el", "ph", "pa"]
    )
    q1 = ops[ops["half"] == 1].set_index("game_id")["recv_home"]
    ops["recv_q1_home"] = ops["game_id"].map(q1)

    rows = []
    for g_, d in p.groupby("game_id", sort=False):
        q = d["qtr"].to_numpy()
        m = d["hs"].to_numpy() - d["as"].to_numpy()
        rec = {"game_id": g_}
        for qq, nm in ((2, "m1"), (3, "m2"), (4, "m3"), (5, "mreg")):
            w = np.flatnonzero(q >= qq)
            rec[nm] = float(m[w[0]]) if len(w) else np.nan
        rows.append(rec)
    qm = pd.DataFrame(rows)
    rg = games.merge(qm, on="game_id", how="left")
    rg["went_ot"] = rg["game_id"].isin(ot_total.keys())
    rg.loc[~rg["went_ot"], "mreg"] = np.nan

    tot_ev = (
        pd.DataFrame({"g": gid[ev_rows], "h": dh[ev_rows], "a": da[ev_rows]})
        .groupby("g")[["h", "a"]]
        .sum()
    )
    chk = fin.join(tot_ev, how="left").fillna(0.0)
    bad = int(((chk["home_score"] != chk["h"]) | (chk["away_score"] != chk["a"])).sum())
    seg_pts = pd.DataFrame(
        {
            "g": seg["game_id"].repeat(np.diff(ev_ptr)).to_numpy(),
            "po": ev[:, 1],
            "pd": ev[:, 2],
            "home": seg["home"].repeat(np.diff(ev_ptr)).to_numpy(),
        }
    )
    seg_pts["h"] = np.where(seg_pts["home"] == 1, seg_pts["po"], seg_pts["pd"])
    seg_pts["a"] = np.where(seg_pts["home"] == 1, seg_pts["pd"], seg_pts["po"])
    sp = seg_pts.groupby("g")[["h", "a"]].sum()
    oph = ops.groupby("game_id")[["ph", "pa"]].sum()
    allp = sp.add(oph.rename(columns={"ph": "h", "pa": "a"}), fill_value=0.0)
    chk2 = fin.join(allp, how="left").fillna(0.0)
    badg = chk2.index[(chk2["home_score"] != chk2["h"]) | (chk2["away_score"] != chk2["a"])]
    bad2 = len(badg)
    seg["bad"] = seg["game_id"].isin(badg).astype(np.int8)
    ops["bad"] = ops["game_id"].isin(badg).astype(np.int8)

    tg = tr.dropna(subset=["off", "off_logo", "dfn_logo"]).copy()
    opp = tr.set_index(["game_id", "team"])
    gg = p[["game_id", "home_team", "away_team"]].drop_duplicates("game_id").set_index("game_id")
    tg["opp"] = np.where(
        gg["home_team"].reindex(tg["game_id"]).to_numpy() == tg["team"].to_numpy(),
        gg["away_team"].reindex(tg["game_id"]).to_numpy(),
        gg["home_team"].reindex(tg["game_id"]).to_numpy(),
    )
    od = opp["dfn_logo"] - opp["dfn_c"]
    tg["xd"] = od.reindex(pd.MultiIndex.from_arrays([tg["game_id"], tg["opp"]])).to_numpy()
    tg = tg.dropna(subset=["xd"])
    xx = (tg["off_logo"] - tg["off_c"]) + tg["xd"]
    yy = tg["off"] - tg["off_c"]
    b = float(np.cov(yy, xx)[0, 1] / xx.var(ddof=1))

    OUT.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(OUT / "segments.parquet")
    ops.to_parquet(OUT / "openings.parquet")
    rg.to_parquet(OUT / "real_games.parquet")
    keep = np.unique(pl[:, 0].astype(np.int64)) if len(pl) else np.zeros(0, dtype=np.int64)
    pp = p.loc[
        keep, ["play_type", "epa", "yards_gained", "sack", "interception", "fumble_lost"]
    ].copy()
    pp["row"] = keep
    pp.to_parquet(OUT / "plays.parquet")
    np.savez(OUT / "csr.npz", ev_ptr=ev_ptr, ev=ev, pl_ptr=pl_ptr, pl=pl)
    meta = {
        "eiv_b": b,
        "n_seg": int(ns),
        "n_games": len(fin),
        "recon_bad_events": bad,
        "recon_bad_segments": bad2,
        "x_sd": float(seg["x"].std()),
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    print(
        "segments by half",
        seg["half"].value_counts().to_dict(),
        "half-end share",
        float(seg["he"].mean()),
        flush=True,
    )
    print(
        "mean elapsed",
        float(seg.loc[seg["he"] == 0, "el"].mean()),
        "segs/game",
        float(ns / len(fin)),
        "seconds",
        round(time.time() - t0, 1),
        flush=True,
    )


FIT_DIMS = [0, 1, 2, 3, 4, 6, 7]


def segment_points(seg):
    c = np.load(OUT / "csr.npz")
    ptr, ev = c["ev_ptr"], c["ev"]
    k = np.repeat(np.arange(len(seg)), np.diff(ptr))
    po = np.bincount(k, weights=ev[:, 1], minlength=len(seg))
    pd_ = np.bincount(k, weights=ev[:, 2], minlength=len(seg))
    return po, pd_


def outcome_classes(seg):
    po, pd_ = segment_points(seg)
    cp = np.select([pd_ > 0, po == 0, po == 3, po == 6, po == 7, po == 8], [5, 0, 1, 2, 3, 4], 6)
    nyl = seg["nyl"].to_numpy()
    open_ = (cp == 0) & (seg["he"].to_numpy() == 0)
    t1, t2 = np.quantile(nyl[open_], [1 / 3, 2 / 3])
    cp = np.where(open_, 7 + (nyl > t1) + (nyl > t2), cp)
    el = seg["el"].to_numpy()
    qs = np.quantile(el, np.linspace(0.1, 0.9, 9))
    cls = cp * 10 + np.searchsorted(qs, el)
    return np.unique(cls, return_inverse=True)[1]


L0 = math.log1p(3600.0)


def game_lt(seg):
    secs = seg["secs"].clip(lower=0.0).to_numpy(float)
    return np.log1p(np.where(seg["half"].to_numpy() == 1, secs + 1800.0, secs))


def loso_ll(X, cls, half, home, season, qidx, h, alpha, g=0.0, lg=None, he=None):
    K = int(cls.max()) + 1
    gc = np.bincount(cls, minlength=K) + 0.5
    tot = 0.0
    n = 0
    for hf in (1, 2):
        for hm in (0, 1):
            st = (half == hf) & (home == hm)
            for s in SEASONS:
                qi = qidx[st[qidx] & (season[qidx] == s)]
                if len(qi) == 0:
                    continue
                pool = np.flatnonzero(st & (season != s))
                oh = np.zeros((len(pool), K), dtype=np.float64)
                oh[np.arange(len(pool)), cls[pool]] = 1.0
                pri = (oh.sum(axis=0) + gc / gc.sum() * 0.5 * K) / (len(pool) + 0.5 * K)
                z = np.zeros((len(qi), len(pool)))
                for d, hd in zip(FIT_DIMS, h, strict=True):
                    if np.isfinite(hd):
                        hq = hd * np.exp(g * (lg[qi] - L0))[:, None] if d == 1 else hd
                        z += ((X[qi, d][:, None] - X[pool, d][None, :]) / hq) ** 2
                W = np.exp(-0.5 * z)
                W *= ~((he[pool][None, :] == 1) & (X[pool, 2][None, :] < X[qi, 2][:, None]))
                cnt = W @ oh
                p = (cnt[np.arange(len(qi)), cls[qi]] + alpha * pri[cls[qi]]) / (
                    W.sum(axis=1) + alpha
                )
                tot += float(np.log(p).sum())
                n += len(qi)
    return tot / n


def cmd_fit(a):
    t0 = time.time()
    seg = pd.read_parquet(OUT / "segments.parquet")
    cls = outcome_classes(seg)
    keep = np.flatnonzero((seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3))
    seg = seg.iloc[keep].reset_index(drop=True)
    cls = cls[keep]
    X = design(seg)
    half, home, season = seg["half"].to_numpy(), seg["home"].to_numpy(), seg["season"].to_numpy()
    he = seg["he"].to_numpy()
    rng = np.random.default_rng(a.seed)
    qidx = np.sort(rng.choice(len(seg), size=min(a.queries, len(seg)), replace=False))
    lg = game_lt(seg)
    h = list(scott_bandwidths(seg)[FIT_DIMS])
    alpha = 1.0
    g = 0.0
    best = loso_ll(X, cls, half, home, season, qidx, h, alpha, g, lg, he)
    base = loso_ll(X, cls, half, home, season, qidx, [np.inf] * len(h), alpha, g, lg, he)
    print(f"start {best:.5f} marginal {base:.5f} h {np.round(h, 3).tolist()}", flush=True)
    grids = [[0.25, 0.5, 0.71, 1.41, 2.0, 4.0], [0.71, 0.84, 1.19, 1.41]]
    for gm in grids:
        improved = True
        rounds = 0
        while improved and rounds < 4:
            improved = False
            rounds += 1
            for i in range(len(h) + 2):
                if i < len(h):
                    cands = [h[i] * m for m in gm]
                elif i == len(h):
                    cands = [alpha * m for m in gm]
                else:
                    cands = [g + (m - 1.0) for m in gm]
                for v in cands:
                    hh, al, gg = list(h), alpha, g
                    if i < len(h):
                        hh[i] = v
                    elif i == len(h):
                        al = v
                    else:
                        gg = v
                    ll = loso_ll(X, cls, half, home, season, qidx, hh, al, gg, lg, he)
                    if ll > best + 1e-6:
                        best, h, alpha, g, improved = ll, hh, al, gg, True
                print(
                    f"  dim {i} best {best:.5f} h {np.round(h, 3).tolist()} "
                    f"alpha {alpha:.3f} g {g:.3f} {time.time() - t0:.0f}s",
                    flush=True,
                )
    seg_ot = pd.read_parquet(OUT / "segments.parquet")
    n_reg = len(seg) / 4.0
    n_ot = float(((seg_ot["half"] == 3) & (seg_ot["bad"] == 0)).sum()) / 2.0
    ratio = (n_ot / n_reg) ** (-1.0 / (len(FIT_DIMS) + 4))
    full = np.full(len(DIMS), np.inf)
    full[FIT_DIMS] = h
    full_ot = full * ratio
    res = {
        "h": [float(v) for v in full],
        "h_ot": [float(v) for v in full_ot],
        "alpha": alpha,
        "g_sd": g,
        "loso_ll": best,
        "marginal_ll": base,
        "queries": len(qidx),
        "dims": DIMS,
        "x": "exponential tilt to the EIV target, no kernel",
        "ot_scale": ratio,
    }
    (OUT / "bandwidths.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


TWO_MIN = 120.0
Q4_SECS = 900.0


def phase_of_state(half, secs):
    if half == 3:
        return 6
    base = 0 if half == 1 else 3
    return base + (0 if secs > Q4_SECS else (1 if secs > TWO_MIN else 2))


def regime(half, secs):
    if half == 1:
        return 0 if secs > TWO_MIN else 1
    if secs > Q4_SECS:
        return 2
    return 3 if secs > TWO_MIN else 4


def cmd_fitreg(a):
    t0 = time.time()
    base = json.loads((OUT / "bandwidths.json").read_text())
    seg = pd.read_parquet(OUT / "segments.parquet")
    cls = outcome_classes(seg)
    keep = np.flatnonzero((seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3))
    seg = seg.iloc[keep].reset_index(drop=True)
    cls = cls[keep]
    X = design(seg)
    half, home, season = seg["half"].to_numpy(), seg["home"].to_numpy(), seg["season"].to_numpy()
    he = seg["he"].to_numpy()
    lg = game_lt(seg)
    reg = np.array([regime(h_, s_) for h_, s_ in zip(half, seg["secs"].to_numpy(), strict=True)])
    rng = np.random.default_rng(a.seed)
    out = {}
    for r in range(5):
        cand = np.flatnonzero(reg == r)
        qidx = np.sort(rng.choice(cand, size=min(a.queries, len(cand)), replace=False))
        h = [base["h"][d] for d in FIT_DIMS]
        alpha = base["alpha"]

        def ll(hh, al, qi=qidx):
            return loso_ll(X, cls, half, home, season, qi, hh, al, 0.0, lg, he)

        best = ll(h, alpha)
        marg = ll([np.inf] * len(h), alpha)
        for gm in ([0.25, 0.5, 0.71, 1.41, 2.0, 4.0], [0.71, 0.84, 1.19, 1.41]):
            improved, rounds = True, 0
            while improved and rounds < 4:
                improved, rounds = False, rounds + 1
                for i in range(len(h) + 1):
                    for m in gm:
                        hh, al = list(h), alpha
                        if i < len(h):
                            hh[i] = h[i] * m
                        else:
                            al = alpha * m
                        v = ll(hh, al)
                        if v > best + 1e-6:
                            best, h, alpha, improved = v, hh, al, True
        full = np.full(len(DIMS), np.inf)
        full[FIT_DIMS] = h
        out[str(r)] = {
            "h": [float(v) for v in full],
            "alpha": alpha,
            "loso_ll": best,
            "marginal_ll": marg,
            "queries": len(qidx),
        }
        print(
            f"regime {r} n {len(cand)} ll {best:.5f} marginal {marg:.5f} "
            f"h {np.round(h, 3).tolist()} alpha {alpha:.3f} {time.time() - t0:.0f}s",
            flush=True,
        )
    base["h_reg"] = out
    (OUT / "bandwidths_reg.json").write_text(json.dumps(base, indent=1))


def cmd_fittilt(a):
    t0 = time.time()
    base = json.loads(Path(a.base).read_text())
    seg = with_mu(pd.read_parquet(OUT / "segments.parquet"))
    cls = outcome_classes(seg)
    K = int(cls.max()) + 1
    gfreq = (np.bincount(cls, minlength=K) + 0.5) / (len(cls) + 0.5 * K)
    oof = pd.read_parquet(OUT / "oof_targets.parquet")
    T = outcome_frame(seg)[OUT_TARGETS].to_numpy(np.float64)
    X = design(seg)
    half, home, season = seg["half"].to_numpy(), seg["home"].to_numpy(), seg["season"].to_numpy()
    he = seg["he"].to_numpy() == 1
    ok = (seg["bad"].to_numpy() == 0) & (half < 3)
    secs = seg["secs"].to_numpy()
    reg = np.array([regime(h_, s_) if h_ < 3 else -1 for h_, s_ in zip(half, secs, strict=True)])
    rng = np.random.default_rng(a.seed)
    fin_dims = [d for d in range(len(DIMS)) if d != 5]
    scott_w = scott_bandwidths(seg[ok])
    for r in [int(v) for v in a.regimes.split(",")]:
        cand = np.flatnonzero(ok & (reg == r))
        qs = np.sort(rng.choice(cand, size=min(a.queries, len(cand)), replace=False))
        hr = base["h_reg"][str(r)]
        h = pad_h(hr["h"])
        if not np.isfinite(h[7]):
            h[7] = scott_w[7]
        alpha = float(hr["alpha"])
        pools = {}
        for i in qs:
            key = (half[i], home[i], season[i])
            if key not in pools:
                sel = ok & (half == half[i]) & (home == home[i]) & (season != season[i])
                pools[key] = np.flatnonzero(sel)

        def ll(hh, al, qs=qs, pools=pools):
            tot = 0.0
            for i in qs:
                pool = pools[(half[i], home[i], season[i])]
                z = np.zeros(len(pool))
                for d in fin_dims:
                    if np.isfinite(hh[d]):
                        z += ((X[pool, d] - X[i, d]) / hh[d]) ** 2
                z[he[pool] & (X[pool, 2] < X[i, 2])] = np.inf
                w = np.exp(-0.5 * z)
                if w.max() <= 0:
                    tot += math.log(gfreq[cls[i]])
                    continue
                nz = np.flatnonzero(w > 1e-12 * w.max())
                tgt = np.r_[oof.loc[i, OUT_TARGETS].to_numpy(np.float64), X[i, 5]]
                wt = tilt_weights(w[nz], np.c_[T[pool[nz]], X[pool[nz], 5]], tgt)
                wt = wt / wt.sum() * w[nz].sum()
                pc = np.bincount(cls[pool], minlength=K)
                pri = (pc + gfreq * 0.5 * K) / (len(pool) + 0.5 * K)
                num = wt[cls[pool[nz]] == cls[i]].sum() + al * pri[cls[i]]
                tot += math.log(num / (wt.sum() + al))
            return tot / len(qs)

        best = ll(h, alpha)
        start = best
        for gm in ([0.5, 1.41, 2.0, 4.0], [0.71, 1.41]):
            improved, rounds = True, 0
            while improved and rounds < 3:
                improved, rounds = False, rounds + 1
                for d in [*fin_dims, -1]:
                    for m in gm:
                        hh, al = h.copy(), alpha
                        if d >= 0:
                            hh[d] = h[d] * m
                        else:
                            al = alpha * m
                        v = ll(hh, al)
                        if v > best + 1e-6:
                            best, h, alpha, improved = v, hh, al, True
                print(
                    f"  regime {r} round {rounds} ll {best:.5f} h {np.round(h, 3).tolist()} "
                    f"alpha {alpha:.3f} {time.time() - t0:.0f}s",
                    flush=True,
                )
        base["h_reg"][str(r)] = {
            "h": [float(v) for v in h],
            "alpha": alpha,
            "loso_ll_tilt": best,
            "loso_ll_tilt_start": start,
            "queries": len(qs),
        }
        print(f"regime {r} tilt-aware ll {start:.5f} -> {best:.5f}", flush=True)
    Path(a.out).write_text(json.dumps(base, indent=1))


def scott_bandwidths(seg):
    X = design(seg)
    n = len(seg) / 4.0
    return X.std(axis=0) * n ** (-1.0 / (X.shape[1] + 4))


def post_bin(half, secs):
    if half == 3:
        return 6
    if half == 1:
        return 0 if secs > Q4_SECS else (1 if secs > TWO_MIN else 2)
    return 3 if secs > Q4_SECS else (4 if secs > TWO_MIN else 5)


def posterior_table(seg, b):
    ok = seg["bad"].to_numpy() == 0
    bins = np.array(
        [
            post_bin(h_, s_)
            for h_, s_ in zip(seg["half"].to_numpy(), seg["secs"].to_numpy(), strict=True)
        ]
    )
    tab = {}
    for k in range(7):
        m = ok & (bins == k)
        x = seg["x"].to_numpy(float)[m]
        sdv = seg["sd"].to_numpy(float)[m]
        hm = seg["home"].to_numpy()[m]
        means = [float(sdv[hm == v].mean()) for v in (0, 1)]
        sdr = sdv - np.where(hm == 1, means[1], means[0])
        V = np.cov(np.c_[x, sdr].T)
        r = np.array([b * V[0, 0], V[0, 1]])
        cx, cs = np.linalg.solve(V, r)
        tab[k] = (float(cx), float(cs), means[0], means[1])
    return tab, bins


def epa_scale(seg):
    c = np.load(OUT / "csr.npz")
    pl, ptr = c["pl"], c["pl_ptr"]
    P = pd.read_parquet(OUT / "plays.parquet").set_index("row")
    ep = P["epa"].fillna(0.0).reindex(pl[:, 0].astype(np.int64)).to_numpy()
    k = np.repeat(np.arange(len(seg)), np.diff(ptr))
    es = np.bincount(k, weights=ep, minlength=len(seg))
    n = np.diff(ptr).astype(float)
    m = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3) & (n > 0)
    x, y, w = seg["mu"].to_numpy()[m], es[m] / n[m], n[m]
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def with_mu(seg):
    if "mu" in seg:
        return seg
    b = json.loads((OUT / "build_meta.json").read_text())["eiv_b"]
    if os.environ.get("SIM10_MU") == "pregame":
        return seg.assign(mu=b * seg["x"].to_numpy(float))
    tab, bins = posterior_table(seg, b)
    cx = np.array([tab[k][0] for k in bins])
    cs = np.array([tab[k][1] for k in bins])
    mh = np.array(
        [
            tab[k][3] if hm == 1 else tab[k][2]
            for k, hm in zip(bins, seg["home"].to_numpy(), strict=True)
        ]
    )
    return seg.assign(mu=cx * seg["x"].to_numpy(float) + cs * (seg["sd"].to_numpy(float) - mh))


def quarter_of(half, secs):
    return np.where(np.asarray(secs) > Q4_SECS, 1, 2) * (np.asarray(half) < 3)


def pad_h(h):
    h = np.asarray(h, dtype=np.float64)
    return np.r_[h, np.full(len(DIMS) - len(h), np.inf)] if len(h) < len(DIMS) else h


def design(seg):
    sv = seg["mu"] if "mu" in seg else seg["x"]
    return np.c_[
        seg["yl"],
        seg["sd"],
        np.log1p(seg["secs"].clip(lower=0.0)),
        seg["too"],
        seg["tod"],
        sv,
        seg["season"],
        seg["week"],
    ].astype(np.float64)


def tilt_weights(w, t, target):
    lo, hi = t.min(axis=0), t.max(axis=0)
    keep = hi > lo
    if not keep.any():
        return w
    t, lo, hi = t[:, keep], lo[keep], hi[keep]
    span = hi - lo
    target = np.clip(np.asarray(target, dtype=np.float64)[keep], lo + 1e-6 * span, hi - 1e-6 * span)
    sc = t.std(axis=0)
    tc = (t - target) / sc
    lw = np.log(w)
    beta = np.zeros(tc.shape[1])

    def dual(bb):
        e = lw + tc @ bb
        mx = e.max()
        return mx + math.log(np.exp(e - mx).sum()), e

    f0, e = dual(beta)
    for _ in range(100):
        u = np.exp(e - e.max())
        u /= u.sum()
        m = u @ tc
        if np.abs(m).max() < 1e-6:
            break
        C = (tc * u[:, None]).T @ tc - np.outer(m, m)
        step = np.linalg.solve(C + 1e-9 * np.eye(len(m)), -m)
        a = 1.0
        found = False
        while a > 1e-6:
            f1, e1 = dual(beta + a * step)
            if f1 <= f0 + 1e-4 * a * float(m @ step):
                found = True
                break
            a *= 0.5
        if not found:
            break
        beta = beta + a * step
        gain = f0 - f1
        f0, e = f1, e1
        if gain < 1e-12:
            break
    return np.exp(e - e.max())


def predict_outcome(t, mc, f):
    m, cal = mc
    if cal is None:
        return float(m.predict(f)[0]) if t == "el" else float(m.predict_proba(f)[0, 1])
    w = np.r_[np.ravel(cal.coef_)]
    c0 = float(np.ravel(np.atleast_1d(cal.intercept_))[0])
    if t == "el":
        z = cal_x(m.predict(f), f)[0]
        return c0 + float(w @ z)
    p = float(np.clip(m.predict_proba(f)[0, 1], 1e-6, 1 - 1e-6))
    z = cal_x(np.array([math.log(p / (1 - p))]), f)[0]
    return 1.0 / (1.0 + math.exp(-(c0 + float(w @ z))))


def strength_adjust(vals, gam, z):
    vals = np.asarray(vals, dtype=np.float64)
    out = vals.copy()
    for k, t in enumerate(OUT_TARGETS):
        if t == "el":
            out[..., k] = vals[..., k] + gam[..., k] * z
        else:
            p = np.clip(vals[..., k], 1e-6, 1 - 1e-6)
            out[..., k] = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + gam[..., k] * z)))
    return out


class Engine:
    def __init__(
        self,
        h,
        h_ot=None,
        g_sd=0.0,
        h_reg=None,
        tilt=(5,),
        models=False,
        pat=False,
        mu_free=(),
        dr=False,
        qstrata=False,
        eoh="outcome",
        strength=None,
        dr_oof=False,
        dr_mu=True,
        mu_label=False,
    ):
        seg = with_mu(pd.read_parquet(OUT / "segments.parquet"))
        self.dr_mu = bool(dr_mu)
        self.ptab = None
        if mu_label:
            eb = json.loads((OUT / "build_meta.json").read_text())["eiv_b"]
            self.ptab = posterior_table(seg, eb)[0]
        self.sgam = None
        self.trace = None
        if strength is not None:
            self.sgam = np.array([strength["terms"][t]["gamma"] for t in OUT_TARGETS]).T
            self.smbar = {
                int(k): np.asarray(v, dtype=np.float64) for k, v in strength["mbar"].items()
            }
        ops = pd.read_parquet(OUT / "openings.parquet")
        meta = json.loads((OUT / "build_meta.json").read_text())
        c = np.load(OUT / "csr.npz")
        self.b = meta["eiv_b"]
        self.s = epa_scale(seg)
        self.g = float(g_sd)
        self.tilt = tuple(tilt)
        self.mu_free = set(mu_free)
        self.qstrata = bool(qstrata)
        self.eoh = eoh
        self.score_k = [k for k, t in enumerate(OUT_TARGETS) if t != "el"]
        self.dr = bool(dr)
        self.pat = None
        if pat:
            import pickle

            with open(OUT / "pat_model.pkl", "rb") as fh:
                pm = pickle.load(fh)
            self.pat = (pm["model"], pm["rates"])
        self.models = None
        if models:
            import pickle

            with open(OUT / "outcome_models.pkl", "rb") as fh:
                mm = pickle.load(fh)
            self.models = {hm: [(t, mm["models"][(t, hm)]) for t in OUT_TARGETS] for hm in (0, 1)}
            od = outcome_frame(seg)
            self.out_arr = od[OUT_TARGETS].to_numpy(np.float64)
            self.m_arr = np.zeros_like(self.out_arr)
            fx = od[OUT_FEATS].to_numpy(np.float64)
            for hm in (0, 1):
                sel = od["home"].to_numpy() == hm
                for k, (t, mc) in enumerate(self.models[hm]):
                    m = mc[0]
                    if t == "el":
                        self.m_arr[sel, k] = m.predict(fx[sel])
                    else:
                        self.m_arr[sel, k] = m.predict_proba(fx[sel])[:, 1]
            if dr_oof:
                oo = pd.read_parquet(OUT / "oof_targets.parquet")
                ri = oo.index.to_numpy()
                self.m_arr[ri] = oo[OUT_TARGETS].to_numpy(np.float64)
            if self.sgam is not None:
                hv, sv = seg["half"].to_numpy(), seg["secs"].to_numpy(float)
                ph = np.array([phase_of_state(a, b) for a, b in zip(hv, sv, strict=True)])
                for hm in (0, 1):
                    sel = np.flatnonzero((od["home"].to_numpy() == hm) & (hv < 3))
                    z = seg["mu"].to_numpy(float)[sel] - self.smbar[hm][ph[sel]]
                    self.m_arr[sel] = strength_adjust(self.m_arr[sel], self.sgam[ph[sel]], z)
        self.h_reg = None if not h_reg else [pad_h(h_reg[str(r)]["h"]) for r in range(5)]
        self.h = pad_h(h)
        self.h_ot = pad_h(h if h_ot is None else h_ot)
        X = design(seg)
        ok = seg["bad"].to_numpy() == 0
        self.strata = {}
        qseg = quarter_of(seg["half"].to_numpy(), seg["secs"].to_numpy())
        for hf in (1, 2, 3):
            for hm in (0, 1):
                for qq in (0, 1, 2) if self.qstrata else (0,):
                    sel = ok & (seg["half"].to_numpy() == hf) & (seg["home"].to_numpy() == hm)
                    if self.qstrata and hf < 3:
                        sel = sel & (qseg == qq)
                    idx = np.flatnonzero(sel)
                    self.strata[(hf, hm, qq)] = (
                        idx,
                        np.ascontiguousarray(X[idx]),
                        seg["he"].to_numpy()[idx] == 1,
                    )
        for k in ("el", "he", "nsame", "nyl", "uo", "ud", "yl"):
            setattr(self, k, seg[k].to_numpy(np.float64))
        self.ev_ptr, self.ev = c["ev_ptr"], c["ev"]
        self.pl_ptr, self.pl = c["pl_ptr"], c["pl"]
        ops = ops[ops["bad"] == 0]
        ops = ops.assign(same_recv=(ops["recv_home"] == ops["recv_q1_home"]).astype(np.int8))
        ops["pr"] = np.where(ops["recv_home"] == 1, ops["ph"], ops["pa"])
        ops["pk"] = np.where(ops["recv_home"] == 1, ops["pa"], ops["ph"])
        self.ops = {
            h: ops[ops["half"] == h][
                ["recv_home", "same_recv", "yl", "el", "pr", "pk", "season"]
            ].to_numpy(np.float64)
            for h in (1, 2, 3)
        }

    def draw(self, hf, hm, q, rng):
        qq = quarter_of(np.array([hf]), np.array([math.expm1(q[2])]))[0] if self.qstrata else 0
        if hf == 3:
            qq = 0
        idx, X, hem = self.strata[(hf, hm, qq)]
        h = self.h_ot if hf == 3 else self.h
        if self.h_reg is not None and hf < 3:
            h = self.h_reg[regime(hf, math.expm1(q[2]))]
        fin = np.isfinite(h)
        if self.g != 0.0 and (self.h_reg is None or hf == 3):
            secs = math.expm1(q[2]) + (1800.0 if hf == 1 else 0.0)
            h = h.copy()
            h[1] = h[1] * math.exp(self.g * (math.log1p(secs) - L0))
        z = (((X[:, fin] - q[fin]) / h[fin]) ** 2).sum(axis=1)
        z[hem & (X[:, 2] < q[2])] = np.inf
        w = np.exp(-0.5 * (z - z.min()))
        nz = np.flatnonzero(w > 1e-12)
        w = w[nz]
        if self.models is not None and hf < 3:
            secs = math.expm1(q[2])
            gs = secs + 1800.0 if hf == 1 else secs
            f = np.array(
                [
                    [
                        q[0],
                        q[1],
                        secs,
                        hf,
                        q[3],
                        q[4],
                        q[5],
                        q[6],
                        gs,
                        q[1] * (GAME_SECS - gs) / GAME_SECS,
                    ]
                ]
            )
            tg = np.asarray([predict_outcome(t, m, f) for t, m in self.models[hm]])
            if self.sgam is not None:
                p_ = phase_of_state(hf, secs)
                tg = strength_adjust(tg, self.sgam[p_], q[5] - self.smbar[hm][p_])
            tg0 = tg
            mt = q[5:6]
            if self.ptab is not None:
                cx_, cs_, m0_, m1_ = self.ptab[post_bin(hf, secs)]
                mt = np.array([cx_ * q[5] + cs_ * (q[1] - (m1_ if hm == 1 else m0_))])
            if self.dr:
                w1 = tilt_weights(w, X[nz, 5:6], mt) if self.dr_mu else w
                u = w1 / w1.sum()
                adj = u @ self.out_arr[idx[nz]] - u @ self.m_arr[idx[nz]]
                tg = tg + adj * DR_MASK
            rg_ = regime(hf, secs)
            if rg_ in self.mu_free and self.eoh == "mu":
                w = tilt_weights(w, X[nz, 5:6], mt)
            elif rg_ in self.mu_free and self.eoh == "muscore":
                sk = self.score_k
                w = tilt_weights(
                    w, np.c_[self.out_arr[idx[nz]][:, sk], X[nz, 5]], np.r_[tg[sk], mt]
                )
            elif rg_ in self.mu_free:
                w = tilt_weights(w, self.out_arr[idx[nz]], np.asarray(tg))
            else:
                w = tilt_weights(w, np.c_[self.out_arr[idx[nz]], X[nz, 5]], np.r_[tg, mt])
            if self.trace is not None:
                u = w / w.sum()
                self.trace.append(
                    np.r_[tg0, tg, u @ self.out_arr[idx[nz]], u @ X[nz, 5], 1.0 / (u @ u)]
                )
        else:
            tdims = [d for d in self.tilt if d == 5 or fin[d]] if hf < 3 else []
            if tdims:
                w = tilt_weights(w, X[nz][:, tdims], q[tdims])
        c = np.cumsum(w)
        k = min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), len(nz) - 1)
        return idx[nz[k]]

    def pat_points(self, sdv, gsecs, season, rng):
        m, r = self.pat
        p2 = float(
            m.predict_proba(pat_x(np.array([sdv]), np.array([gsecs]), np.array([season])))[0, 1]
        )
        if rng.random() < p2:
            return 2.0 if rng.random() < r["two"] else 0.0
        xp = r["xp_post"] if season >= XP_RULE_SEASON else r["xp_pre"]
        return 1.0 if rng.random() < xp else 0.0

    def opening(self, hf, season, rng):
        o = self.ops[hf]
        cand = o[o[:, 6] == season] if hf < 3 else o
        if len(cand) == 0:
            cand = o
        return cand[rng.integers(len(cand))]

    def run_half(
        self, rng, hf, pos, yl, secs, sc, base, xq, season, evs, pls, ot_rule=None, week=9.0
    ):
        tos = [3.0, 3.0] if hf < 3 else [2.0, 2.0]
        n_poss = 0
        guard = 0
        while secs > 0 and guard < 80:
            guard += 1
            other = 1 - pos
            q = np.array(
                [
                    yl,
                    sc[pos] - sc[other],
                    math.log1p(secs),
                    tos[pos],
                    tos[other],
                    xq[pos],
                    season,
                    week,
                ]
            )
            j = self.draw(hf, 1 if pos == 0 else 0, q, rng)
            trunc = self.he[j] == 1 or self.el[j] >= secs
            now = base + secs
            scored = td = dsc = False
            skip = None
            for k in range(self.ev_ptr[j], self.ev_ptr[j + 1]):
                off_, po, pd_ = self.ev[k]
                if trunc and off_ > secs:
                    continue
                if skip is not None:
                    mine, theirs = (po, pd_) if skip == pos else (pd_, po)
                    skip = None
                    if mine in (1.0, 2.0) and theirs == 0:
                        continue
                sc[pos] += po
                sc[other] += pd_
                evs.append((now - off_, po if pos == 0 else pd_, pd_ if pos == 0 else po, hf))
                scored = True
                td = td or po >= 6
                dsc = dsc or pd_ > 0
                if self.pat is not None and hf < 3 and (po == 6) != (pd_ == 6):
                    scorer = pos if po == 6 else other
                    skip = scorer
                    pts = self.pat_points(sc[scorer] - sc[1 - scorer], now - off_, season, rng)
                    if pts:
                        sc[scorer] += pts
                        evs.append(
                            (
                                now - off_,
                                pts if scorer == 0 else 0.0,
                                0.0 if scorer == 0 else pts,
                                hf,
                            )
                        )
            a, b = self.pl_ptr[j], self.pl_ptr[j + 1]
            if b > a:
                r = self.pl[a:b]
                if trunc:
                    r = r[r[:, 1] <= secs]
                pls.append(np.c_[r[:, 0], np.full(len(r), pos)])
            if trunc:
                return
            secs -= self.el[j]
            tos[pos] = max(0.0, tos[pos] - self.uo[j])
            tos[other] = max(0.0, tos[other] - self.ud[j])
            n_poss += 1
            if hf == 3:
                if ot_rule == "sd" and scored:
                    return
                if ot_rule == "mod" and (td or dsc or (n_poss >= 2 and sc[0] != sc[1])):
                    return
            d = yl - self.yl[j]
            ny = self.nyl[j]
            if not scored:
                ny = ny + d if self.nsame[j] == 1 else ny - d
            yl = min(99.0, max(1.0, ny))
            if self.nsame[j] != 1:
                pos = other

    def game(self, rng, lat, season, week=9.0):
        week = float(min(week, LAST_REG_WEEK))
        sc = [0.0, 0.0]
        evs, pls = [], []
        xq = [(lat[0][0] + lat[1][1]) / self.s, (lat[1][0] + lat[0][1]) / self.s]
        o = self.opening(1, season, rng)
        recv1 = 0 if rng.random() < 0.5 else 1
        sc[recv1] += o[4]
        sc[1 - recv1] += o[5]
        if o[4] or o[5]:
            evs.append(
                (3600.0 - o[3], o[4] if recv1 == 0 else o[5], o[5] if recv1 == 0 else o[4], 1)
            )
        self.run_half(
            rng, 1, recv1, o[2], 1800.0 - o[3], sc, 1800.0, xq, season, evs, pls, week=week
        )
        o = self.opening(2, season, rng)
        recv2 = recv1 if o[1] == 1 else 1 - recv1
        sc[recv2] += o[4]
        sc[1 - recv2] += o[5]
        if o[4] or o[5]:
            evs.append(
                (1800.0 - o[3], o[4] if recv2 == 0 else o[5], o[5] if recv2 == 0 else o[4], 2)
            )
        self.run_half(rng, 2, recv2, o[2], 1800.0 - o[3], sc, 0.0, xq, season, evs, pls, week=week)
        reg = sc[0] - sc[1]
        went_ot = sc[0] == sc[1]
        if went_ot:
            tot = 600.0 if season >= 2017 else 900.0
            rule = "sd" if season in SUDDEN_DEATH else "mod"
            o = self.opening(3, season, rng)
            recv = 0 if rng.random() < 0.5 else 1
            sc[recv] += o[4]
            sc[1 - recv] += o[5]
            if not (o[4] or o[5]):
                self.run_half(
                    rng,
                    3,
                    recv,
                    o[2],
                    tot - o[3],
                    sc,
                    0.0,
                    xq,
                    season,
                    evs,
                    pls,
                    ot_rule=rule,
                    week=week,
                )
        e = np.array(evs, dtype=np.float64).reshape(-1, 4)
        reg_e = e[e[:, 3] < 3]
        dm = reg_e[:, 1] - reg_e[:, 2]
        ms = [float(dm[reg_e[:, 0] >= t].sum()) for t in (2700.0, 1800.0, 900.0)]
        pl = np.concatenate(pls) if pls else np.zeros((0, 2))
        return sc[0], sc[1], ms, reg, went_ot, pl


_W = {}


def _init(
    h,
    ho,
    g,
    hr=None,
    tilt=(5,),
    models=False,
    pat=False,
    mu_free=(),
    dr=False,
    qstrata=False,
    eoh="outcome",
    strength=None,
    dr_oof=False,
    dr_mu=True,
    mu_label=False,
):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("QBC", "1")
    _W["eng"] = Engine(
        h,
        ho,
        g,
        hr,
        tilt,
        models,
        pat,
        mu_free,
        dr,
        qstrata,
        eoh,
        strength,
        dr_oof,
        dr_mu,
        mu_label,
    )


def sim_world(task):
    import mod25_generator as gen

    w, wseed, n_seasons = task
    eng = _W["eng"]
    fit = load_latent_fit()
    setting = dict(gen.SETTING_DEFAULTS)
    grows, prow, ptm = [], [], []
    rng = np.random.default_rng(wseed)
    lat = gen.gen_world_latents(rng, n_seasons, fit, setting)
    gseeds = wseed.spawn(n_seasons)
    for sidx, (weekly, qb_out) in enumerate(lat):
        sched = gen.make_schedule(rng)
        ratings = gen.season_ratings(weekly, qb_out, fit, setting, 0.0, 0.0)
        season = SEASONS[(w * n_seasons + sidx) % len(SEASONS)]
        grng = np.random.default_rng(gseeds[sidx])
        sid = w * 1000 + sidx + 1
        for gi, (week, h, a) in enumerate(sched):
            rh, ra = ratings[week][h], ratings[week][a]
            hs, as_, ms, reg, ot, pl = eng.game(grng, (rh, ra), season, week)
            gidn = f"w{w:04d}s{sidx:02d}g{gi:03d}"
            dl = (rh[0] + ra[1]) - (ra[0] + rh[1])
            grows.append(
                (
                    gidn,
                    sid,
                    week,
                    f"W{w}T{h}",
                    f"W{w}T{a}",
                    hs,
                    as_,
                    ms[0],
                    ms[1],
                    ms[2],
                    reg if ot else np.nan,
                    ot,
                    season,
                    dl,
                )
            )
            if len(pl):
                prow.append(pl[:, 0].astype(np.int64))
                ptm.append((gidn, f"W{w}T{h}", f"W{w}T{a}", pl[:, 1].astype(np.int8)))
    return w, grows, prow, ptm


def sim_worlds(params, n_worlds, n_seasons, seed, workers=1):
    import multiprocessing as mp

    os.environ.setdefault("QBC", "1")
    ws = np.random.SeedSequence(seed).spawn(n_worlds)
    tasks = [(w, ws[w], n_seasons) for w in range(n_worlds)]
    if workers <= 1:
        _init(*params)
        out = [sim_world(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=params) as pool:
            out = pool.map(sim_world, tasks, chunksize=1)
    out.sort(key=lambda r: r[0])
    grows, prow, ptm = [], [], []
    for _, g_, p_, t_ in out:
        grows += g_
        prow += p_
        ptm += t_
    G = pd.DataFrame(
        grows,
        columns=[
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "m1",
            "m2",
            "m3",
            "mreg",
            "went_ot",
            "real_season",
            "lat_d",
        ],
    )
    G["gameday"] = pd.Timestamp("2000-09-05") + pd.to_timedelta(
        (G["season"] % 1000 - 1) * 365 + (G["week"] - 1) * 7, unit="D"
    )
    rows = np.concatenate(prow)
    P = pd.read_parquet(OUT / "plays.parquet").set_index("row").reindex(rows).reset_index(drop=True)
    n = np.array([len(x[3]) for x in ptm])
    P["game_id"] = np.repeat([x[0] for x in ptm], n)
    side = np.concatenate([x[3] for x in ptm])
    P["posteam"] = np.where(
        side == 0, np.repeat([x[1] for x in ptm], n), np.repeat([x[2] for x in ptm], n)
    )
    return G, P


def gate_stats(G, P, keep):
    import mod25_generator as gen
    import mod25d_variance as dv

    X = G[G["season"].isin(keep)]
    out = margin_stats(X)
    re_ = {}
    for sid, d in X.groupby("season"):
        meta = pd.DataFrame(
            {
                "season": d["season"].to_numpy(),
                "home_team": d["home_team"].to_numpy(),
                "away_team": d["away_team"].to_numpy(),
            }
        )
        r = dv.random_effects(
            meta, pd.Series((d["home_score"] - d["away_score"]).to_numpy(float)), iters=400
        )
        re_[sid] = (r["strength_diff_var"], r["noise_var"])
    out["strength_re_var"] = float(np.mean([v[0] for v in re_.values()]))
    out["noise_re_var"] = float(np.mean([v[1] for v in re_.values()]))
    ts = gen.team_stats_from_plays(P)
    gm = G[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ]
    met, _, d = gen.all_metrics(gm, ts, list(keep))
    out["r2_w10_18"] = met["r2_w10_18"]
    out["r2_w5_9"] = met["r2_w5_9"]
    out["epa_autocorr_lag1"] = met["epa_autocorr_lag1"]
    return out, (X, re_, d)


def boot(parts, nb, seed):
    X, re_, d = parts
    rng = np.random.default_rng(seed)
    seasons = sorted(re_)
    gs = {s: X[X["season"] == s] for s in seasons}
    ds = {s: d[(d["season"] == s) & (d["week"] >= 10)] for s in seasons}
    res = {}
    for _ in range(nb):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        st = margin_stats(pd.concat([gs[s] for s in pick]))
        st["strength_re_var"] = float(np.mean([re_[s][0] for s in pick]))
        st["noise_re_var"] = float(np.mean([re_[s][1] for s in pick]))
        b = pd.concat([ds[s] for s in pick])
        st["r2_w10_18"] = float(np.corrcoef(b["x"], b["y"])[0, 1] ** 2)
        for k, v in st.items():
            res.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in res.items()}


def compare(sim, simb, real, realb):
    lines = []
    for k in simb:
        dd = simb[k] - realb[k]
        lo, hi = np.percentile(dd, [5, 95])
        rlo, rhi = np.percentile(realb[k], [5, 95])
        lines.append(
            f"{k:18s} sim {sim[k]:9.4f}  real {real[k]:9.4f} [{rlo:9.4f},{rhi:9.4f}]  "
            f"delta {sim[k] - real[k]:+9.4f} [{lo:+9.4f},{hi:+9.4f}] "
            f"P+ {float((dd > 0).mean()):.3f}"
        )
    return lines


def margin_stats(X):
    m = (X["home_score"] - X["away_score"]).to_numpy(float)
    mr = np.where(X["went_ot"].to_numpy(bool), X["mreg"].to_numpy(float), m)
    prev = np.zeros(len(X))
    incs = []
    for q in ("m1", "m2", "m3"):
        d = X[q].to_numpy(float)
        incs.append(d - prev)
        prev = d
    incs.append(mr - prev)
    out = {
        "margin_sd": float(m.std(ddof=1)),
        "xq_cov_sum": float(mr.var(ddof=1) - sum(i.var(ddof=1) for i in incs)),
        "q4_slope": float(np.cov(incs[3], prev)[0, 1] / prev.var(ddof=1)),
        "mass_3": float(np.mean(np.abs(m) == 3)),
        "mass_7": float(np.mean(np.abs(m) == 7)),
        "mass_10": float(np.mean(np.abs(m) == 10)),
        "mass_14": float(np.mean(np.abs(m) == 14)),
        "pts_game": float((X["home_score"] + X["away_score"]).mean()),
        "home_edge": float(m.mean()),
        "ot_share": float(X["went_ot"].mean()),
        "tie_share": float(np.mean(m == 0)),
    }
    return out


def observable_slope(G, P):
    p = P[P["play_type"].isin(["run", "pass"])].copy()
    p["epa"] = p["epa"].fillna(0.0)
    gm = G.set_index("game_id")
    p["season"] = gm["season"].reindex(p["game_id"]).to_numpy()
    home = gm["home_team"].reindex(p["game_id"]).to_numpy()
    away = gm["away_team"].reindex(p["game_id"]).to_numpy()
    p["defteam"] = np.where(p["posteam"].to_numpy() == home, away, home)
    off = (
        p.groupby(["game_id", "season", "posteam"])["epa"]
        .mean()
        .rename("off")
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    de = (
        p.groupby(["game_id", "season", "defteam"])["epa"]
        .mean()
        .rename("dfn")
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    t = off.merge(de, on=["game_id", "season", "team"])
    for c in ("off", "dfn"):
        grp = t.groupby(["season", "team"])[c]
        t[c + "_l"] = (grp.transform("sum") - t[c]) / (grp.transform("count") - 1) - t.groupby(
            "season"
        )[c].transform("mean")
    k = t.set_index(["game_id", "team"])

    def side(col, tc):
        return k[col].reindex(pd.MultiIndex.from_arrays([G["game_id"], G[tc]])).to_numpy()

    D = (
        side("off_l", "home_team")
        + side("dfn_l", "away_team")
        - side("off_l", "away_team")
        - side("dfn_l", "home_team")
    )
    m = (G["home_score"] - G["away_score"]).to_numpy(float)
    ok = np.isfinite(D)
    return float(np.polyfit(D[ok], m[ok], 1)[0]), float(np.var(D[ok]))


def latent_logo_stats(fit, n_worlds=200, n_seasons=6, seed=5):
    import mod25_generator as gen

    os.environ.setdefault("QBC", "1")
    setting = dict(gen.SETTING_DEFAULTS)
    ws = np.random.SeedSequence(seed).spawn(n_worlds)
    out = []
    for w in range(n_worlds):
        rng = np.random.default_rng(ws[w])
        for weekly, qb_out in gen.gen_world_latents(rng, n_seasons, fit, setting):
            rt = gen.season_ratings(weekly, qb_out, fit, setting, 0.0, 0.0)
            A = np.array([[rt[wk + 1][t] for t in range(gen.N_TEAMS)] for wk in range(18)])
            A = A - A.mean(axis=1, keepdims=True)
            lo = (A.sum(axis=0, keepdims=True) - A) / (A.shape[0] - 1)
            cx = 0.5 * (np.mean(A[:, :, 0] * lo[:, :, 1]) + np.mean(A[:, :, 1] * lo[:, :, 0]))
            out.append((np.mean(A[:, :, 0] * lo[:, :, 0]), np.mean(A[:, :, 1] * lo[:, :, 1]), cx))
    return np.mean(np.array(out), axis=0)


def cmd_latfit(a):
    import mod25_generator as gen

    plays = gen.load_real_plays(SEASONS)
    tg = gen.team_game_epa(plays)
    res = {"off": gen.fit_ar1(tg, "off"), "def": gen.fit_ar1(tg, "dfn")}
    for k, c in (("off", "off"), ("def", "dfn")):
        res[k]["phi_6lag_log"] = res[k]["phi"]
        dd = tg.copy()
        dd["dev"] = dd[c] - dd.groupby("season")[c].transform("mean")
        ks, ac, nw = [], [], []
        for lag in range(1, LAST_REG_WEEK - 1):
            lagged = dd.groupby(["posteam", "season"])["dev"].shift(lag)
            okl = lagged.notna()
            ks.append(lag)
            ac.append(float(np.mean(dd.loc[okl, "dev"].to_numpy() * lagged[okl].to_numpy())))
            nw.append(float(okl.sum()))
        ks, ac, nw = np.array(ks), np.array(ac), np.array(nw)
        grid = np.linspace(0.95, 1.0, 501)
        sse = []
        for ph in grid:
            basis = ph**ks
            v = float(np.sum(nw * basis * ac) / np.sum(nw * basis * basis))
            sse.append(float(np.sum(nw * (ac - v * basis) ** 2)))
        res[k]["phi"] = float(grid[int(np.argmin(sse))])
        res[k]["autocov_all"] = ac.tolist()
    sm = tg.copy()
    for c in ("off", "dfn"):
        sm[c] = sm[c] - sm.groupby("season")[c].transform("mean")
    sm = sm.groupby(["posteam", "season"])[["off", "dfn"]].mean()
    res["season_mean_corr_off_def_raw"] = float(sm["off"].corr(sm["dfn"]))
    res["season_mean_sd"] = {"off": float(sm["off"].std()), "def": float(sm["dfn"].std())}
    d = tg.copy()
    for c in ("off", "dfn"):
        d[c + "_d"] = d[c] - d.groupby("season")[c].transform("mean")
        g_ = d.groupby(["season", "posteam"])[c + "_d"]
        d[c + "_l"] = (g_.transform("sum") - d[c + "_d"]) / (g_.transform("count") - 1)
    n_g = float(d.groupby(["season", "posteam"]).size().mean())
    lags = np.arange(1, round(n_g))
    wts = 2.0 * (n_g - lags)
    direct = {}
    for k, c in (("off", "off"), ("def", "dfn")):
        cov = float(np.cov(d[c + "_d"], d[c + "_l"])[0, 1])
        phi = res[k]["phi"]
        avg = float(np.sum(wts * phi**lags) / np.sum(wts))
        direct[k] = cov
        res[k]["logo_cov"] = cov
        res[k]["var_mu_ar"] = res[k]["var_mu"]
        res[k]["var_mu"] = cov / avg
    c_od = 0.5 * (
        float(np.cov(d["off_d"], d["dfn_l"])[0, 1]) + float(np.cov(d["dfn_d"], d["off_l"])[0, 1])
    )
    res["logo_cov_off_def"] = c_od
    res["season_mean_corr_off_def"] = c_od / math.sqrt(direct["off"] * direct["def"])
    res["off"].update(gen.fit_offseason(tg, "off", res["off"]["phi"], res["off"]["var_mu"]))
    res["def"].update(gen.fit_offseason(tg, "dfn", res["def"]["phi"], res["def"]["var_mu"]))
    res["qb"] = gen.fit_qb(plays, tg)
    target = np.array([res["off"]["logo_cov"], res["def"]["logo_cov"], res["logo_cov_off_def"]])
    for _ in range(6):
        got = latent_logo_stats(res)
        res["off"]["var_mu"] *= target[0] / got[0]
        res["def"]["var_mu"] *= target[1] / got[1]
        res["season_mean_corr_off_def"] *= target[2] / got[2]
    res["latent_logo_check"] = {
        "target": target.tolist(),
        "sim": latent_logo_stats(res, seed=6).tolist(),
    }
    res["seasons"] = list(SEASONS)
    (OUT / "latent_fit.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                k: (
                    v
                    if not isinstance(v, dict)
                    else {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
                )
                for k, v in res.items()
            },
            indent=1,
        )
    )


def load_latent_fit():
    p = (
        Path(os.environ["SIM10_LATENT"])
        if os.environ.get("SIM10_LATENT")
        else OUT / "latent_fit.json"
    )
    if p.exists():
        return json.loads(p.read_text())
    import mod25_generator as gen

    return gen.load_fit()


OUT_FEATS = ["yl", "sd", "secs", "half", "too", "tod", "mu", "season", "gsecs", "sdlate"]
GAME_SECS = 3600.0
MAX_POSS_PTS = 8.0
CAL_FEATS = ["yl"]
CAL_IDX = [OUT_FEATS.index(c) for c in CAL_FEATS]


N_PHASES = 6


def phase_of(F):
    gs = F[:, OUT_FEATS.index("gsecs")]
    q = np.minimum(3, ((GAME_SECS - gs) // Q4_SECS).astype(int))
    half_end = np.where(gs > GAME_SECS / 2, GAME_SECS / 2, 0.0)
    late = (gs - half_end) <= TWO_MIN
    q2 = np.where(late, 2, 1)
    q4 = np.where(late, 5, 4)
    return np.where(q == 1, q2, np.where(q == 3, q4, np.where(q == 0, 0, 3)))


def cal_x(base, F):
    oh = np.eye(N_PHASES)[phase_of(F)]
    return np.c_[np.asarray(base, dtype=np.float64)[:, None] * oh, oh[:, 1:], cal_design(F)]


def cal_design(F):
    sdv = F[:, OUT_FEATS.index("sd")]
    muv = F[:, OUT_FEATS.index("mu")]
    ph = phase_of(F)
    lead1 = (sdv >= 1) & (sdv <= MAX_POSS_PTS)
    trail1 = (sdv <= -1) & (sdv >= -MAX_POSS_PTS)
    cols = []
    for k in range(N_PHASES):
        qk = ph == k
        cols += [sdv * qk, lead1 * qk, trail1 * qk, (sdv == 0) * qk, muv * qk]
    return np.c_[F[:, CAL_IDX], *cols]


OUT_TARGETS = ["el", "td", "fg", "ds"]
DR_MASK = np.array([0.0, 1.0, 1.0, 1.0])


def add_game_feats(d):
    gs = np.where(
        d["half"].to_numpy() == 1, d["secs"].to_numpy(float) + 1800.0, d["secs"].to_numpy(float)
    )
    return d.assign(gsecs=gs, sdlate=d["sd"].to_numpy(float) * (GAME_SECS - gs) / GAME_SECS)


def outcome_frame(seg):
    seg = add_game_feats(with_mu(seg))
    po, pd_ = segment_points(seg)
    d = seg[[*OUT_FEATS, "home"]].copy()
    d["el"] = seg["el"].to_numpy(float)
    d["td"] = (po >= 6).astype(float)
    d["fg"] = (po == 3).astype(float)
    d["ds"] = (pd_ > 0).astype(float)
    return d


HGB_PARAMS = {}
MONO_SIGN = {"td": 1, "ds": -1}


def make_model(t, params=None):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    kw = {"early_stopping": True, "random_state": 0, "max_iter": 1000}
    kw.update(HGB_PARAMS if params is None else params)
    if t in MONO_SIGN:
        cst = [0] * len(OUT_FEATS)
        cst[OUT_FEATS.index("mu")] = MONO_SIGN[t]
        kw["monotonic_cst"] = cst
    if t == "el":
        return HistGradientBoostingRegressor(**kw)
    return HistGradientBoostingClassifier(**kw)


def cmd_hgbsel(a):
    seg = pd.read_parquet(OUT / "segments.parquet")
    d = outcome_frame(seg)
    d = d[(seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)].reset_index(drop=True)
    grid = [
        {"learning_rate": lr, "max_leaf_nodes": ml, "l2_regularization": l2}
        for lr in (0.05, 0.1)
        for ml in (15, 31, 63)
        for l2 in (0.0, 1.0)
    ]
    res = []
    for p in grid:
        tot = 0.0
        for hm in (0, 1):
            dh = d[d["home"].to_numpy() == hm].reset_index(drop=True)
            X = dh[OUT_FEATS].to_numpy(float)
            y = dh["td"].to_numpy(int)
            for s_ in SEASONS:
                te = dh["season"].to_numpy() == s_
                m = make_model("td", p).fit(X[~te], y[~te])
                pr = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
                tot += float(-np.sum(y[te] * np.log(pr) + (1 - y[te]) * np.log(1 - pr)))
        res.append((tot / len(d), p))
        print(f"{tot / len(d):.5f} {p}", flush=True)
    best = min(res, key=lambda r: r[0])
    (OUT / "hgb_params.json").write_text(
        json.dumps(
            {"best": best[1], "loso_logloss_td": best[0], "grid": [(r[0], r[1]) for r in res]},
            indent=1,
        )
    )
    print("best", best, flush=True)


def load_hgb_params():
    p = OUT / "hgb_params.json"
    if p.exists():
        HGB_PARAMS.update(json.loads(p.read_text())["best"])


def cmd_outfit(a):
    import pickle

    load_hgb_params()
    t0 = time.time()
    seg = pd.read_parquet(OUT / "segments.parquet")
    d = outcome_frame(seg)
    d = d[(seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)]
    d = d.assign(sidx=d.index).reset_index(drop=True)
    rep_ = {}
    models = {}
    oof = {}
    for hm in (0, 1):
        dh = d[d["home"].to_numpy() == hm].reset_index(drop=True)
        X = dh[OUT_FEATS].to_numpy(float)
        for t in OUT_TARGETS:
            y = dh[t].to_numpy(float)
            pred = np.zeros(len(dh))
            for s_ in SEASONS:
                te = dh["season"].to_numpy() == s_
                m = make_model(t).fit(X[~te], y[~te] if t == "el" else y[~te].astype(int))
                pred[te] = m.predict(X[te]) if t == "el" else m.predict_proba(X[te])[:, 1]
            if t == "el":
                rep_[f"{t}_{hm}"] = {
                    "loso_mse": float(np.mean((y - pred) ** 2)),
                    "marginal_mse": float(np.mean((y - y.mean()) ** 2)),
                }
            else:
                p = np.clip(pred, 1e-6, 1 - 1e-6)
                q = y.mean()
                rep_[f"{t}_{hm}"] = {
                    "loso_logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                    "marginal_logloss": float(-(q * math.log(q) + (1 - q) * math.log(1 - q))),
                }
            if t == "el":
                from sklearn.linear_model import LinearRegression

                CX = cal_x(pred, X)
                cal = LinearRegression().fit(CX, y)
                cp = np.zeros(len(y))
                for s_ in SEASONS:
                    te = dh["season"].to_numpy() == s_
                    cp[te] = LinearRegression().fit(CX[~te], y[~te]).predict(CX[te])
                rep_[f"{t}_{hm}"]["cal_loso_mse"] = float(np.mean((y - cp) ** 2))
                oof.setdefault(t, []).append(pd.Series(cp, index=dh["sidx"].to_numpy()))
            else:
                from sklearn.linear_model import LogisticRegression

                lz = np.log(p / (1 - p))
                CX = cal_x(lz, X)
                yi = y.astype(int)
                cal = LogisticRegression(penalty=None, max_iter=5000).fit(CX, yi)
                cp = np.zeros(len(y))
                for s_ in SEASONS:
                    te = dh["season"].to_numpy() == s_
                    lr = LogisticRegression(penalty=None, max_iter=5000).fit(CX[~te], yi[~te])
                    cp[te] = lr.predict_proba(CX[te])[:, 1]
                cp = np.clip(cp, 1e-6, 1 - 1e-6)
                rep_[f"{t}_{hm}"]["cal_loso_logloss"] = float(
                    -np.mean(y * np.log(cp) + (1 - y) * np.log(1 - cp))
                )
                oof.setdefault(t, []).append(pd.Series(cp, index=dh["sidx"].to_numpy()))
            rep_[f"{t}_{hm}"]["cal_coef"] = [float(v) for v in np.ravel(cal.coef_)]
            print(t, hm, rep_[f"{t}_{hm}"], f"{time.time() - t0:.0f}s", flush=True)
            if t == "el":
                raw_loss = rep_[f"{t}_{hm}"]["loso_mse"]
                cal_loss = rep_[f"{t}_{hm}"]["cal_loso_mse"]
            else:
                raw_loss = rep_[f"{t}_{hm}"]["loso_logloss"]
                cal_loss = rep_[f"{t}_{hm}"]["cal_loso_logloss"]
            use_cal = cal_loss < raw_loss
            rep_[f"{t}_{hm}"]["stacking_used"] = bool(use_cal)
            if not use_cal:
                oof[t][-1] = pd.Series(pred, index=dh["sidx"].to_numpy())
            fitted = make_model(t).fit(X, y if t == "el" else y.astype(int))
            models[(t, hm)] = (fitted, cal if use_cal else None)
    with open(OUT / "outcome_models.pkl", "wb") as fh:
        pickle.dump({"models": models, "feats": OUT_FEATS, "loso": rep_}, fh)
    (OUT / "outcome_models.json").write_text(json.dumps(rep_, indent=1))
    pd.DataFrame({t: pd.concat(v) for t, v in oof.items()}).sort_index().to_parquet(
        OUT / "oof_targets.parquet"
    )


XP_RULE_SEASON = 2015
PAT_SD_CLIP = 24


def pat_frame():
    p = load_pbp()
    g = p["game_id"].to_numpy()
    same = np.r_[g[1:] == g[:-1], False]
    hs, as_ = p["hs"].to_numpy(), p["as"].to_numpy()
    dh = np.where(same, np.r_[hs[1:], 0] - hs, 0)
    da = np.where(same, np.r_[as_[1:], 0] - as_, 0)
    td = np.flatnonzero(((dh == 6) | (da == 6)) & same)
    td = td[td + 2 < len(p)]
    nx = td + 1
    pt = p["play_type"].to_numpy()
    dn = p["down"].to_numpy().astype(float)
    kick = pt[nx] == "extra_point"
    two = np.isin(pt[nx], ["pass", "run"]) & np.isnan(dn[nx])
    hsc = dh[td] == 6
    d = pd.DataFrame(
        {
            "sd": np.where(hsc, hs[nx] - as_[nx], as_[nx] - hs[nx]),
            "gsecs": np.where(
                p["qtr"].to_numpy()[nx] <= 2, p["gsr"].to_numpy()[nx], p["gsr"].to_numpy()[nx]
            ),
            "qtr": p["qtr"].to_numpy()[nx],
            "season": p["season"].to_numpy()[nx],
            "kick": kick,
            "two": two,
            "made": np.where(hsc, dh[nx], da[nx]) > 0,
        }
    )
    return d[(d["kick"] | d["two"]) & (d["qtr"] <= 4)].reset_index(drop=True)


def pat_x(sdv, gsecs, season):
    return np.c_[
        np.clip(sdv, -PAT_SD_CLIP, PAT_SD_CLIP) + PAT_SD_CLIP,
        gsecs,
        (np.asarray(season) >= XP_RULE_SEASON).astype(float),
    ]


def cmd_patfit(a):
    import pickle

    from sklearn.ensemble import HistGradientBoostingClassifier

    d = pat_frame()
    X = pat_x(d["sd"].to_numpy(float), d["gsecs"].to_numpy(float), d["season"].to_numpy())
    y = d["two"].to_numpy(int)

    def mk():
        return HistGradientBoostingClassifier(
            categorical_features=[0], early_stopping=True, random_state=0
        )

    pred = np.zeros(len(d))
    for s_ in SEASONS:
        te = d["season"].to_numpy() == s_
        pred[te] = mk().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    p_ = np.clip(pred, 1e-6, 1 - 1e-6)
    q = y.mean()
    rep_ = {
        "loso_logloss": float(-np.mean(y * np.log(p_) + (1 - y) * np.log(1 - p_))),
        "marginal_logloss": float(-(q * math.log(q) + (1 - q) * math.log(1 - q))),
        "n": len(d),
    }
    era = d["season"].to_numpy() >= XP_RULE_SEASON
    rates = {
        "xp_pre": float(d.loc[d["kick"] & ~era, "made"].mean()),
        "xp_post": float(d.loc[d["kick"] & era, "made"].mean()),
        "two": float(d.loc[d["two"], "made"].mean()),
    }
    with open(OUT / "pat_model.pkl", "wb") as fh:
        pickle.dump({"model": mk().fit(X, y), "rates": rates, "loso": rep_}, fh)
    print(json.dumps({"loso": rep_, "rates": rates}), flush=True)


def cmd_real(a):
    import pickle

    import mod25_generator as gen

    rg = pd.read_parquet(OUT / "real_games.parquet")
    P = gen.load_real_plays(SEASONS)
    rg["gameday"] = pd.to_datetime(rg["gameday"])
    st, parts = gate_stats(rg, P, list(SEASONS))
    (OUT / "real_gates.json").write_text(json.dumps(st, indent=1))
    with open(OUT / "real_boot.pkl", "wb") as f:
        pickle.dump(boot(parts, 1000, 7), f)
    print(json.dumps(st, indent=1))


def load_h(path):
    if path:
        d = json.loads(Path(path).read_text())
        return (
            np.array(d["h"]),
            np.array(d.get("h_ot", d["h"])),
            float(d.get("g_sd", 0.0)),
            d.get("h_reg"),
        )
    seg = pd.read_parquet(OUT / "segments.parquet")
    h = scott_bandwidths(seg[seg["half"] < 3])
    ho = scott_bandwidths(seg[seg["half"] == 3])
    return h, ho, 0.0, None


def cmd_sim(a):
    t0 = time.time()
    h, ho, g, hr = load_h(a.h)
    tilt = tuple(int(v) for v in a.tilt.split(",")) if a.tilt else ()
    mf = tuple(int(v) for v in a.mufree.split(",")) if a.mufree else ()
    st = json.loads(Path(a.strength).read_text()) if a.strength else None
    G, P = sim_worlds(
        (
            h,
            ho,
            g,
            hr,
            tilt,
            bool(a.models),
            bool(a.pat),
            mf,
            bool(a.dr),
            bool(a.qstrata),
            a.eoh,
            st,
            bool(a.droof),
            bool(a.drmu),
            bool(a.mulabel),
        ),
        a.worlds,
        a.seasons,
        a.seed,
        a.workers,
    )
    keep = sorted(G.loc[G["season"] % 1000 > a.burn, "season"].unique())
    import pickle

    st, parts = gate_stats(G, P, keep)
    K = G[G["season"].isin(keep)]
    kk = float(np.polyfit(K["lat_d"], K["home_score"] - K["away_score"], 1)[0])
    vs = float(K.groupby("season")["lat_d"].var().mean())
    print(
        f"diag: margin slope on latent diff {kk:.2f}, within-season var of latent diff {vs:.5f}, "
        f"implied {kk * kk * vs:.1f}",
        flush=True,
    )
    osl, ovar = observable_slope(K, P[P["game_id"].isin(K["game_id"])])
    print(
        f"diag: margin slope on LOGO season EPA diff {osl:.2f} (real 36.25), "
        f"var of that diff {ovar:.5f}",
        flush=True,
    )
    real = json.loads((OUT / "real_gates.json").read_text())
    with open(OUT / "real_boot.pkl", "rb") as f:
        realb = pickle.load(f)
    simb = boot(parts, 1000, a.seed)
    od = OUT / f"{a.label}_s{a.seed}"
    od.mkdir(parents=True, exist_ok=True)
    G.to_parquet(od / "games.parquet")
    P[["game_id", "posteam", "play_type", "epa"]].to_parquet(od / "plays.parquet")
    lines = compare(st, simb, real, realb)
    (od / "gates.json").write_text(
        json.dumps(
            {
                "sim": st,
                "real": real,
                "h": list(map(float, h)),
                "h_ot": list(map(float, ho)),
                "seconds": time.time() - t0,
                "table": lines,
            },
            indent=1,
        )
    )
    for ln in lines:
        print(ln, flush=True)
    print("games", len(G), "seconds", round(time.time() - t0, 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("real")
    sub.add_parser("latfit")
    sub.add_parser("outfit")
    sub.add_parser("hgbsel")
    sub.add_parser("patfit")
    ft = sub.add_parser("fittilt")
    ft.add_argument("--regimes", default="1,4")
    ft.add_argument("--queries", type=int, default=800)
    ft.add_argument("--seed", type=int, default=9)
    ft.add_argument("--base", default=str(OUT / "bandwidths_reg.json"))
    ft.add_argument("--out", default=str(OUT / "bandwidths_tilt.json"))
    fr = sub.add_parser("fitreg")
    fr.add_argument("--queries", type=int, default=3000)
    fr.add_argument("--seed", type=int, default=5)
    f = sub.add_parser("fit")
    f.add_argument("--queries", type=int, default=8000)
    f.add_argument("--seed", type=int, default=3)
    s = sub.add_parser("sim")
    s.add_argument("--worlds", type=int, default=8)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--burn", type=int, default=2)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--tilt", default="5")
    s.add_argument("--models", type=int, default=1)
    s.add_argument("--pat", type=int, default=1)
    s.add_argument("--mufree", default="1,4")
    s.add_argument("--dr", type=int, default=1)
    s.add_argument("--qstrata", type=int, default=1)
    s.add_argument("--eoh", choices=("outcome", "mu", "muscore"), default="outcome")
    s.add_argument("--strength", default=str(REPO / "artifacts" / "sim11" / "strength_terms.json"))
    s.add_argument("--droof", type=int, default=1)
    s.add_argument("--drmu", type=int, default=0)
    s.add_argument("--mulabel", type=int, default=1)
    s.add_argument("--label", default="sim10")
    s.add_argument("--h", default=str(OUT / "bandwidths_week.json"))
    a = ap.parse_args()
    {
        "build": build,
        "patfit": cmd_patfit,
        "fittilt": cmd_fittilt,
        "hgbsel": cmd_hgbsel,
        "outfit": cmd_outfit,
        "latfit": cmd_latfit,
        "real": cmd_real,
        "sim": cmd_sim,
        "fit": cmd_fit,
        "fitreg": cmd_fitreg,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
