import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.argv = sys.argv[:1] + [str(NB)]
import mod25e_catchup as cu  # noqa: E402
import mod25e_late as late  # noqa: E402
import mod25e_srcconc as sc  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / os.environ.get("SELSTATE_OUT", "selstate")
OUT = []
rng = np.random.default_rng(61)
EDGES = np.array([-8.5, -0.5, 0.5, 8.5])
BN = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
CH = 20


def say(s=""):
    print(s)
    OUT.append(s)


def iv(d, pt):
    return f"{pt:+.4f} [{np.percentile(d, 5):+.4f},{np.percentile(d, 95):+.4f}] pp {float((d > 0).mean()):.2f}"


def stats_by_group(gid, q, b, x, y, ng):
    key = (gid * 4 + (q - 1)) * 5 + b
    k = 20 * ng
    n = np.bincount(key, minlength=k).astype(float).reshape(ng, 4, 5)
    return n, [np.bincount(key, weights=w, minlength=k).reshape(ng, 4, 5) for w in (x, y, x * x, x * y)]


def slope_boot(n, S, nb, ngr):
    Sx, Sy, Sxx, Sxy = S
    res = {"pool": [], "q": [[] for _ in range(4)], "cell": [[[] for _ in range(5)] for _ in range(4)]}
    ws = [np.ones(ngr)] + [np.bincount(rng.integers(0, ngr, ngr), minlength=ngr).astype(float) for _ in range(nb)]
    pts = None
    for i, w in enumerate(ws):
        N, X, Y, XX, XY = (np.tensordot(w, a, 1) for a in (n, Sx, Sy, Sxx, Sxy))
        Nq, Xq, Yq, XXq, XYq = (a.sum(1) for a in (N, X, Y, XX, XY))
        cv = XYq - Xq * Yq / Nq
        vr = XXq - Xq * Xq / Nq
        pool = cv.sum() / vr.sum()
        qs = cv / vr
        cell = Y / np.maximum(N, 1) - (Yq / Nq)[:, None]
        if i == 0:
            pts = (pool, qs, cell)
        else:
            res["pool"].append(pool)
            for q in range(4):
                res["q"][q].append(qs[q])
                for bb in range(5):
                    res["cell"][q][bb].append(cell[q, bb])
    return pts, res


def fmt_slopes(tag, pts, res):
    pool, qs, cell = pts
    say(f"  {tag}: pooled within-quarter slope per 7 pts of score diff {iv(np.array(res['pool']), pool)}")
    say("    by quarter per 7 pts: " + "; ".join(f"Q{q + 1} {iv(np.array(res['q'][q]), qs[q])}" for q in range(4)))
    for q in range(4):
        say(f"    Q{q + 1} band mean minus quarter mean: " + "; ".join(f"{BN[b]} {iv(np.array(res['cell'][q][b]), cell[q, b])}" for b in range(5)))


def load_sim(rv, uv):
    rows = []
    for sd in cu.SEEDS:
        sdir = cu.ART / f"e5_{cu.LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < late.BURN:
                continue
            d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "sd", "idx"])
            d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
            g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
            new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
            d["d"] = np.cumsum(new)
            ok = d.idx.notna().to_numpy()
            ix = np.where(ok, d.idx.fillna(0).to_numpy(), 0).astype(int)
            d["v"] = np.where(ok, rv[ix], np.nan)
            d["u"] = np.where(ok, uv[ix], np.nan)
            d["gk"] = f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str)
            rows.append(d[["gk", "d", "qtr", "sd", "v", "u"]])
    return pd.concat(rows, ignore_index=True)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    tables, trans, pbp = sc.engine()
    a = tables["arrays"]
    gcode, ots, dts, rv, gs, _ = sc.source_tables(tables, trans, pbp)
    home = a["is_home_off"].astype(bool)
    okp = np.isin(a["play_type_code"], (0, 1))
    uv = np.nan_to_num(gs.set_index(["g", "h"]).r.reindex(pd.MultiIndex.from_arrays([gcode, ~home])).to_numpy(), nan=0.0)
    PA = load_sim(rv, uv)
    P = PA[PA.v.notna()].reset_index(drop=True)
    gcodes, gnames = pd.factorize(P.gk)
    ngs = len(gnames)
    say("## S1. selection of source game-day residual by score state; v = source play's game-side offence EPA/play (leave-play-out, leave-game-out, opponent-adjusted), u = same-game opposing offence residual (defence side: positive = weak defence); sd = offence score diff at the draw; game bootstrap " + str(NB))
    say(f"  sim plays {len(P)}, sim team-games {ngs}; v mean {P.v.mean():+.4f} sd {P.v.std():.4f}; u mean {P.u.mean():+.4f} sd {P.u.std():.4f}")
    b = np.digitize(P.sd.to_numpy(), EDGES)
    q = P.qtr.to_numpy().astype(int)
    x = P.sd.to_numpy() / 7.0
    for nm in ("v", "u"):
        n, S = stats_by_group(gcodes, q, b, x, P[nm].to_numpy(), ngs)
        pts, res = slope_boot(n, S, NB, ngs)
        fmt_slopes(f"SIM {nm}", pts, res)
        if nm == "v":
            sim_pool = pts[0]
    qr = trans[["game_id", "play_id"]].merge(pbp[["game_id", "play_id", "qtr"]].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")["qtr"].to_numpy()
    sc_raw = trans["sc_raw"].to_numpy(dtype=float)
    m = okp & np.isfinite(qr) & (qr <= 4) & np.isfinite(sc_raw)
    gm = pd.factorize(gcode[m])
    rb = np.digitize(sc_raw[m], EDGES)
    for nm, arr in (("v", rv), ("u", uv)):
        n, S = stats_by_group(gm[0], qr[m].astype(int), rb, sc_raw[m] / 7.0, arr[m], len(gm[1]))
        pts, res = slope_boot(n, S, NB, len(gm[1]))
        fmt_slopes(f"REAL pool rows {nm}", pts, res)
        if nm == "v":
            real_pool = pts[0]
    say(f"  sim/real pooled v slope ratio {sim_pool / real_pool:.2f}")

    ns = late.load_xq()
    _, simd = cu.drives_all(ns)
    Ds, Gs = cu.prep(ns, simd)
    dm = PA.groupby(["gk", "d"], sort=False).agg(v=("v", "mean"), u=("u", "mean")).reset_index()
    say("## S2. persistence generated by the selection slope (catchup definitions: E = sum over 6 quarter pairs of 2cov(I_q, I_r) minus state-response part)")
    Dj = Ds[["gk", "sd0", "qi", "bi", "a"]].copy()
    Dj["k"] = Dj.groupby("gk").cumcount()
    dm["k"] = dm.groupby("gk").cumcount()
    say(f"  catchup sim drives {len(Ds)}; play-file drives {len(dm)}")
    J = Dj.merge(dm[["gk", "k", "v", "u"]], on=["gk", "k"], how="left")
    say(f"  drive match coverage {float(J.v.notna().mean()):.3f}")
    J["v"] = J.v.fillna(0.0)
    J["u"] = J.u.fillna(0.0)
    cell = J.qi.to_numpy() * 5 + J.bi.to_numpy()
    sdv = J.sd0.to_numpy() / 7.0
    dx = sdv - pd.Series(sdv).groupby(cell).transform("mean").to_numpy()
    comps, gam = [], {}
    for nm in ("v", "u"):
        y = J[nm].to_numpy()
        dy = y - pd.Series(y).groupby(cell).transform("mean").to_numpy()
        gam[nm] = float((dx * dy).sum() / (dx * dx).sum())
        comps.append(gam[nm] * dx)
    Xr = np.c_[comps[0], comps[1]]
    aw = J.a.to_numpy() - pd.Series(J.a.to_numpy()).groupby(cell).transform("mean").to_numpy()
    beta, *_ = np.linalg.lstsq(Xr, aw, rcond=None)
    say(f"  within-cell (quarter x band) slope per 7 pts on drive-start diff: gamma_v {gam['v']:+.4f}, gamma_u {gam['u']:+.4f}; pass-through of drive points on those fitted components beta_v {beta[0]:+.3f}, beta_u {beta[1]:+.3f}")
    Ds2 = Ds.copy()
    Ds2["a"] = Ds.a.to_numpy() - Xr @ beta

    def esum(D, W):
        I, IBS, IS, abar = cu.cross_terms(D, Gs, W)
        out = np.zeros(len(W))
        for i in range(len(W)):
            Wi = W[i:i + 1]
            for qq, rr in cu.PAIRS:
                out[i] += 2 * cu.wcov(Wi, I[None, :, qq], I[None, :, rr])[0] - 2 * cu.wcov(Wi, I[None, :, qq], IS[i:i + 1, :, rr])[0]
        return out

    W0 = np.ones((1, Gs))
    e0, e1 = esum(Ds, W0)[0], esum(Ds2, W0)[0]
    say(f"  E sum as simulated {e0:+.2f}; selection component removed {e1:+.2f}; generated {e0 - e1:+.2f}; share of E58 +14.8 {(e0 - e1) / 14.8:+.3f}, of E59 unexplained +8.7 {(e0 - e1) / 8.7:+.3f}")
    gen = []
    for _ in range(max(NB // CH, 1)):
        W = rng.multinomial(Gs, np.ones(Gs) / Gs, size=CH).astype(float)
        gen.extend(list(esum(Ds, W) - esum(Ds2, W)))
    gen = np.array(gen)
    say(f"  generated E bootstrap ({len(gen)} draws) {iv(gen, e0 - e1)}; share of 8.7 {iv(gen / 8.7, (e0 - e1) / 8.7)}")
    say("  beta in-sample, inferred: sim pass-through of source residual to drive points within cell, attributing all within-cell covariance with the fitted component to selection")
    say("looks: S1 2 outcomes x (1 pooled + 4 quarters + 20 cells) x sim and real = 100 interval prints; S2 gamma/beta point fits + 2 intervals; bootstrap " + str(NB))
    (OUTD / "selstate.txt").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()
