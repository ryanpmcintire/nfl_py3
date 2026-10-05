import os
import sys
from pathlib import Path

os.environ.setdefault("A2_LABEL", "crHpqokgndecsmfwtjo2as2")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_adj2 as a2  # noqa: E402
import mod25e_e91 as e91  # noqa: E402
import mod25e_late as late  # noqa: E402
import sim09_u4g as u4g  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / "e93"
NB = 300
rng = np.random.default_rng(93)
CNT = {"cand": 0, "pool_ok": 0, "match": 0, "applied": 0, "punt": 0, "to": 0, "fgmiss": 0}
POOL = {}
TBMODE = os.environ.get("E93_TB", "cond")
FGMODE = os.environ.get("E93_FG", "rule")


def load_pool():
    p = pd.read_parquet(u4g.OUT / "pool.parquet", columns=["gid", "yl", "flip", "code", "down"])
    p["nyl"] = p.groupby("gid", sort=False).yl.shift(-1)
    POOL["yl"] = p.yl.to_numpy(float)
    POOL["nyl"] = p.nyl.to_numpy(float)
    POOL["flip"] = p.flip.to_numpy(bool)
    POOL["code"] = p.code.to_numpy(int)
    POOL["down"] = p.down.to_numpy(float)
    POOL["year"] = p.gid.str[:4].astype(int).to_numpy()
    rel = POOL["nyl"] - (100.0 - POOL["yl"])
    rel = POOL["nyl"] - (100.0 - POOL["yl"])
    m = np.zeros(len(p), bool)
    return p, rel, m


def measure_fgmiss():
    rows = []
    for sd in a2.SEEDS:
        sdir = a2.ART / f"e5_{a2.LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            d = pd.read_parquet(f, columns=["code", "po", "pdf", "flip", "qtr", "idx"])
            d = d[(d.code == 3) & (d.flip == 1) & (d.po == 0) & (d.pdf == 0) & (d.qtr <= 4)]
            rows.append(d.idx.to_numpy().astype(int))
    ix = np.unique(np.concatenate(rows))
    rel = POOL["nyl"][ix] - (100.0 - POOL["yl"][ix])
    yl = POOL["yl"][ix]
    far = np.isfinite(rel) & (yl >= 30.0)
    near = np.isfinite(rel) & (yl <= 10.0)
    POOL["kick_back"] = float(-np.median(rel[far]))
    POOL["tb"] = float(np.median(POOL["nyl"][ix][near]))
    return ix, rel, yl


def replay(d):
    d = d.sort_values("g", kind="stable").reset_index(drop=True)
    ix = d.idx.to_numpy().astype(int)
    g = d.g.to_numpy()
    yl0 = d.yl.to_numpy(float)
    yl = yl0.copy()
    code = d.code.to_numpy(int)
    cand = (d.flip == 1).to_numpy() & (d.po == 0).to_numpy() & (d.pdf == 0).to_numpy() & (d.qtr <= 4).to_numpy() & np.r_[g[1:] == g[:-1], False] & np.isin(code, (0, 1, 2, 3))
    ci = np.flatnonzero(cand)
    nxt = np.minimum(ci + 1, len(d) - 1)
    pi = ix[ci]
    ok = (POOL["flip"][pi]) & (POOL["code"][pi] == code[ci]) & (POOL["down"][pi] == d.down.to_numpy(float)[ci])
    match = ok
    CNT["cand"] += len(ci)
    CNT["pool_ok"] += int(ok.sum())
    CNT["match"] += int((ok & (np.abs(yl0[nxt] - POOL["nyl"][pi]) < 1e-6)).sum())
    ci, nxt, pi = ci[match], nxt[match], pi[match]
    ys = yl0[ci]
    cd = code[ci]
    tb = POOL["tb"]
    base_start = yl0[nxt]
    s = base_start + (POOL["yl"][pi] - ys)
    s = np.where(s >= 100.0, tb, s)
    row_tb = (cd == 2) & (POOL["nyl"][pi] == tb)
    s = np.where(row_tb & ((ys <= POOL["yl"][pi]) | (TBMODE == "keep")), base_start, s)
    fgm = cd == 3
    if FGMODE == "rule":
        s = np.where(fgm, 100.0 - np.maximum(ys + POOL["kick_back"], 100.0 - tb), s)
    else:
        fg_row_tb = fgm & (POOL["nyl"][pi] == tb)
        s = np.where(fgm, np.minimum(base_start + (POOL["yl"][pi] - ys), tb), s)
        s = np.where(fg_row_tb & ((ys <= POOL["yl"][pi]) | (TBMODE == "keep")), tb, s)
    s = np.clip(s, 1.0, 99.0)
    yl[nxt] = s
    CNT["applied"] += len(ci)
    CNT["punt"] += int((cd == 2).sum())
    CNT["to"] += int(np.isin(cd, (0, 1)).sum())
    CNT["fgmiss"] += int(fgm.sum())
    d["yl"] = yl
    return d


def sim_replayed(sd):
    orig = pd.read_parquet

    def patched(path, *a, **k):
        if Path(str(path)).name.startswith("play_"):
            cols = k.get("columns")
            d = replay(orig(path))
            return d[cols] if cols else d
        return orig(path, *a, **k)

    pd.read_parquet = patched
    try:
        return a2.sim_drives(sd)
    finally:
        pd.read_parquet = orig


def boots(S):
    gk = pd.Index(sorted(S.gk.unique()))
    gi = gk.get_indexer(S.gk)
    W = np.vstack([np.ones((1, len(gk))), rng.multinomial(len(gk), np.ones(len(gk)) / len(gk), size=NB).astype(float)])
    return gi, W, len(gk)


def meanboot(v, gi, W):
    G = W.shape[1]
    return (W @ np.bincount(gi, weights=v, minlength=G)) / (W @ np.bincount(gi, minlength=G).astype(float))


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    p, rel, m = load_pool()
    fx, frel, fyl = measure_fgmiss()
    yrs = POOL["year"][fx]
    say(f"fg mode {FGMODE} touchback-row mode {TBMODE}")
    say(f"E93 relative start after flips; label {a2.LABEL} seeds {a2.SEEDS}; no sim run; boots {NB}")
    say(f"pool measured on sim-drawn missed-FG rows: spot offset behind LOS (median of -rel, yl>=30) {POOL['kick_back']:.2f} yd; touchback spot (median start on flip FG rows yl<=10) {POOL['tb']:.1f}")
    for a, b in ((2009, 2013), (2014, 2018), (2019, 2025)):
        mm = np.isfinite(frel) & (fyl >= 30.0) & (yrs >= a) & (yrs <= b)
        if mm.any():
            say(f"  era {a}-{b}: missed-FG offset median {-np.median(frel[mm]):.2f} n {int(mm.sum())}")
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = a2.decorate(ns, real, None, fpmap, None)
    base = pd.concat([a2.sim_drives(sd) for sd in a2.SEEDS], ignore_index=True)
    rep = pd.concat([sim_replayed(sd) for sd in a2.SEEDS], ignore_index=True)
    say(f"replay coverage: flip no-score plays {CNT['cand']}, pool-consistent {CNT['pool_ok']}, of which sim start equals drawn-row start (rest carry STF z) {CNT['match']}, applied {CNT['applied']} (punt {CNT['punt']}, turnover {CNT['to']}, missed FG {CNT['fgmiss']})")
    Db = a2.decorate(ns, base, None, fpmap, None)
    Dp = a2.decorate(ns, rep, None, fpmap, None)
    looks = 0
    res = {}
    for nm, D in (("real", Dr), ("base", Db), ("rep", Dp)):
        S = e91.centre(e91.prep(D), ["s_raw", "e_raw"])
        gi, W, G = boots(S)
        res[nm] = {}
        for gname in ("punt", "to", "fgmiss"):
            mk = (S.ptype == gname).to_numpy()
            g = gi[mk]
            e, s = S.e_raw_c.to_numpy()[mk], S.s_raw_c.to_numpy()[mk]
            res[nm][gname] = {
                "bse": e91.boot_slope(e, s, g, W),
                "lvl": meanboot(S.s_raw.to_numpy()[mk], g, W),
                "emean": meanboot(S.e_raw.to_numpy()[mk], g, W),
                "tb": meanboot((np.abs(S.s_raw.to_numpy()[mk] - POOL["tb"]) < 1e-6).astype(float), g, W),
                "n": int(mk.sum()),
            }
        H = a2.drive_x(D)
        hgi, hW, _ = boots(H)
        res[nm]["x"] = e91.boot_slope(H.xo.to_numpy(), H.ylc.to_numpy(), hgi, hW)
    for gname in ("punt", "to", "fgmiss"):
        say(f"group {gname} (n real {res['real'][gname]['n']} base {res['base'][gname]['n']} rep {res['rep'][gname]['n']})")
        for k, lab, d in (("bse", "slope of start on end spot", 4), ("lvl", "mean start yardline_100", 2), ("tb", "touchback share", 3), ("emean", "mean end spot e", 2)):
            r, b, q = (res[x][gname][k] for x in ("real", "base", "rep"))
            looks += 1
            say(f"  {lab:28s} real {e91.fmt(r, d)} | base {e91.fmt(b, d)} | replay {e91.fmt(q, d)} | replay-real {e91.fmt(q - r, d)} P(replay>real) {np.mean((q - r)[1:] > 0):.2f}")
    r, b, q = res["real"]["x"], res["base"]["x"], res["rep"]["x"]
    looks += 1
    say(f"E88 x slope (B H2 start on A H1 residual): real {e91.fmt(r)} base {e91.fmt(b)} replay {e91.fmt(q)}")
    say(f"  base-real {e91.fmt(b - r)} P(base>real) {np.mean((b - r)[1:] > 0):.2f}; replay-real {e91.fmt(q - r)} P(replay>real) {np.mean((q - r)[1:] > 0):.2f}; replay-base {e91.fmt(q - b)} P(replay<base) {np.mean((q - b)[1:] < 0):.2f}")
    say(f"looks {looks}")
    (OUTD / f"e93_{FGMODE}_{TBMODE}.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
