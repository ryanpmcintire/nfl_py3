import os
import sys
from pathlib import Path

os.environ.setdefault("A2_LABEL", "crHpqokgndecsmfwtjo2a")
os.environ.setdefault("A2_SEEDS", "11,12,13")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_late as late  # noqa: E402

ART = REPO / "artifacts" / "mod25e3"
LABEL = os.environ["A2_LABEL"]
SEEDS = tuple(int(x) for x in os.environ["A2_SEEDS"].split(","))
OUTD = ART / "adj2"
NB = 300
OUT = []
LOOKS = [0]
TYPES = ["punt", "to", "fgmiss", "fg", "td", "other", "defscore"]
rng = np.random.default_rng(88)
FINE = [-np.inf] + [float(x) + 0.5 for x in range(-21, 21, 3)] + [np.inf]


def say(s=""):
    print(s)
    OUT.append(s)


def sim_drives(sd):
    sdir = ART / f"e5_{LABEL}_s{sd}"
    sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
    sg["w"] = sg.game_id.str[1:5].astype(int)
    sg["s"] = sg.game_id.str[6:8].astype(int)
    sg["g"] = sg.game_id.str[-3:].astype(int)
    out = []
    for f in sorted(sdir.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < late.BURN:
            continue
        tm = sg[(sg.w == w) & (sg.s == s)].set_index("g")
        d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "yl", "gsr", "sd", "po", "pdf", "code", "down", "flip"])
        d = d[d.qtr <= 4].sort_values("g", kind="stable").reset_index(drop=True)
        g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
        sc = ((d.po > 0) | (d.pdf > 0)).to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | sc[:-1]]
        d["d"] = np.cumsum(new)
        d["pp"] = d.po - d.pdf
        gi = d.g.astype(int)
        ht = tm.home_team.reindex(gi).to_numpy()
        at = tm.away_team.reindex(gi).to_numpy()
        d["off"] = np.where(d.offhome == 1, ht, at)
        d["dfn"] = np.where(d.offhome == 1, at, ht)
        lv = d.code.isin([0, 1, 2, 3])
        gp = d.groupby("d", sort=True)
        gl = d[lv].groupby("d", sort=True)
        key = f"{sd}_{w}_{s}"
        D = pd.DataFrame({"gk": key + "_" + gp.g.first().astype(int).astype(str), "season": 0, "off": gp.off.first(), "dfn": gp.dfn.first(),
                          "s": np.where(gp.offhome.first() == 1, 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yl.first(), "gsr0": gp.gsr.first(), "n": gp.size(),
                          "sd0": gp.sd.first(), "p": gp.pp.sum()})
        lastc = gl.code.last().reindex(D.index)
        D["lastpt"] = np.select([lastc == 2, lastc == 3], ["punt", "field_goal"], "run")
        D["dlast"] = gl.down.last().reindex(D.index)
        D["ylast"] = gl.yl.last().reindex(D.index)
        flipl = gp.flip.last()
        D["tolast"] = ((flipl == 1) & (lastc.isin([0, 1]))).astype(int)
        D["tm"] = key + "_" + D.off.astype(str)
        D["td"] = key + "_" + D.dfn.astype(str)
        out.append(D)
    return late.finish(pd.concat(out, ignore_index=True))


def decorate(ns, D, edges, fpmap, ref):
    D, _ = ns["components"](D)
    q0 = np.clip(((3600 - D.gsr0.to_numpy()) // 900).astype(int), 0, 3)
    D["q0"] = q0
    D["h"] = (D.gsr0.to_numpy() <= 1800).astype(int)
    D["bi"] = pd.cut(D.sd0, FINE if os.environ.get("A2_FINE") == "1" else ns["LEAD_EDGES"], labels=False).astype(int)
    r1 = D.p.to_numpy() - D.p.mean() - D.c_strength.to_numpy()
    D["r1"] = r1
    cell = pd.Series(r1).groupby([q0, D.bi.to_numpy()]).transform("mean").to_numpy()
    D["e"] = r1 - cell
    fb = pd.cut(D.yl0, ns["YL_BINS"], right=False)
    D["fpv"] = fb.map(fpmap).astype(float).to_numpy()
    D["fpc"] = D.fpv - pd.Series(D.fpv.to_numpy()).groupby([q0, D.bi.to_numpy()]).transform("mean").to_numpy()
    D["rest"] = D.e - D.fpc
    D["ptype"] = pd.Series(np.r_[["end"], D.end.to_numpy()[:-1]]).where(D.gk.to_numpy() == np.r_[[""], D.gk.to_numpy()[:-1]], "end").to_numpy()
    D["ptype"] = D.ptype.replace({"downs": "to"}).where(D.ptype.replace({"downs": "to"}).isin(TYPES), "other")
    D["ylc"] = D.yl0 - D.groupby([q0, D.bi.to_numpy()]).yl0.transform("mean").to_numpy()
    tm = D.groupby("ptype").fpv.mean()
    D["fpmix"] = D.ptype.map(tm).astype(float)
    D["fpwin"] = D.fpv - D.fpmix
    return D


def side_table(D, own=False):
    D = D.assign(S=(D.s < 0).astype(int))
    cols = ["e", "fpc", "rest", "fpmix", "fpwin", "fpv"]
    A = D[D.h == 0].groupby(["gk", "S"]).agg(x=("e", "sum"), n0=("e", "size")).reset_index()
    parts = {"y_e": "e", "y_fpc": "fpc", "y_rest": "rest", "y_fpmix": "fpmix", "y_fpwin": "fpwin", "y_yl": "ylc"}
    H = D[D.h == 1]
    B = H.groupby(["gk", "S"]).agg(**{k: (v, "sum") for k, v in parts.items()}, y_n=("e", "size")).reset_index()
    for t in TYPES:
        Ht = H[H.ptype == t]
        Bt = Ht.groupby(["gk", "S"]).agg(**{f"fp_{t}": ("fpc", "sum"), f"n_{t}": ("fpc", "size"), f"yl_{t}": ("ylc", "sum")}).reset_index()
        B = B.merge(Bt, on=["gk", "S"], how="left")
    B = B.fillna(0.0)
    if not own:
        B["S"] = 1 - B.S
    T = A.merge(B, on=["gk", "S"], how="inner")
    return T


def slopes(T, ycols, W, gk):
    gi = gk.get_indexer(T.gk)
    G = len(gk)
    x = T.x.to_numpy()
    one = np.bincount(gi, minlength=G).astype(float)
    Sx = np.bincount(gi, weights=x, minlength=G)
    Sxx = np.bincount(gi, weights=x * x, minlength=G)
    N = W @ one
    mx = (W @ Sx) / N
    vx = (W @ Sxx) / N - mx**2
    res = {}
    for c in ycols:
        y = T[c].to_numpy()
        Sy = np.bincount(gi, weights=y, minlength=G)
        Sxy = np.bincount(gi, weights=x * y, minlength=G)
        res[c] = ((W @ Sxy) / N - mx * (W @ Sy) / N) / vx
    return res


def fmt(a, d=4):
    return f"{a[0]:+.{d}f} [{np.quantile(a[1:], .025):+.{d}f},{np.quantile(a[1:], .975):+.{d}f}]"


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    sim = pd.concat([sim_drives(sd) for sd in SEEDS], ignore_index=True)
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = decorate(ns, real, None, fpmap, None)
    Ds = decorate(ns, sim, None, fpmap, None)
    say(f"E88 channel decomposition of the opponent's H2 drives on side X's H1 residual; label {LABEL} seeds {SEEDS}; no sim run; real games {Dr.gk.nunique()} sim games {Ds.gk.nunique()}; boots {NB}")
    say("fp map = real mean (drive net points minus strength) by start-yardline bin; e = drive net points - strength - quarter x lead-band mean; x = X H1 sum e; y = other side H2 sums")
    out = {}
    for nm, D in (("real", Dr), ("sim", Ds)):
        T = side_table(D)
        gk = pd.Index(sorted(T.gk.unique()))
        G = len(gk)
        W = np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])
        ycols = ["y_e", "y_fpc", "y_rest", "y_fpmix", "y_fpwin", "y_n", "y_yl"] + [f"{p}_{t}" for t in TYPES for p in ("fp", "n", "yl")]
        out[nm] = (slopes(T, ycols, W, gk), len(T), T.x.std())
        say(f"{nm}: side-games {len(T)} sd(x) {T.x.std():.2f}; drives/B-H2 {T.y_n.mean():.3f}")
    R, S = out["real"][0], out["sim"][0]
    say("slope of Y on x (per point of X early residual); real | sim | sim-real (pp = P(sim>real))")
    for c in R:
        if c.startswith(("yl_", "fpmix", "fpwin")):
            continue
        d = S[c] - R[c]
        LOOKS[0] += 1
        say(f"  {c:12s} real {fmt(R[c])} | sim {fmt(S[c])} | s-r {fmt(d)} pp {np.mean(d[1:] > 0):.2f}")
    say("own side later (same side X H2 drives on X H1 residual)")
    for nm, D in (("real", Dr), ("sim", Ds)):
        T = side_table(D, True)
        gk = pd.Index(sorted(T.gk.unique()))
        G = len(gk)
        W = np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])
        r = slopes(T, ["y_e", "y_fpc", "y_rest", "y_n"], W, gk)
        say(f"  {nm}: " + " ".join(f"{c} {fmt(r[c])}" for c in r))
    say(f"looks {LOOKS[0]}")
    (OUTD / ("an1fine.txt" if os.environ.get("A2_FINE") == "1" else "an1.txt")).write_text("\n".join(OUT))


def xtable(D):
    D = D.assign(S=(D.s < 0).astype(int))
    return D[D.h == 0].groupby(["gk", "S"]).e.sum().rename("x").reset_index()


def real_fourth(ns, seasons):
    out = []
    for s in seasons:
        p = pd.read_parquet(late.PBP / f"season={s}" / "plays.parquet", columns=["game_id", "season_type", "posteam", "home_team", "qtr", "yardline_100", "play_type", "score_differential", "down", "ydstogo", "qb_kneel", "qb_spike"])
        p = p[(p.season_type == "REG") & (p.down == 4) & p.score_differential.notna() & (p.qtr >= 3) & (p.qtr <= 4) & p.play_type.isin(["run", "pass", "punt", "field_goal"]) & (p.qb_kneel != 1) & (p.qb_spike != 1)]
        out.append(pd.DataFrame({"gk": str(s) + "_" + p.game_id, "S": np.where(p.posteam == p.home_team, 0, 1), "qtr": p.qtr, "yl": p.yardline_100, "dist": p.ydstogo, "sd": p.score_differential,
                                 "a": np.select([p.play_type == "punt", p.play_type == "field_goal"], [2, 3], 1)}))
    return pd.concat(out, ignore_index=True)


def sim_fourth():
    out = []
    for sd in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < late.BURN:
                continue
            d = pd.read_parquet(f, columns=["g", "qtr", "down", "dist", "yl", "sd", "code", "offhome"])
            d = d[(d.qtr >= 3) & (d.qtr <= 4) & (d.down == 4) & d.code.isin([0, 1, 2, 3])]
            out.append(pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str), "S": np.where(d.offhome == 1, 0, 1), "qtr": d.qtr, "yl": d.yl, "dist": d.dist, "sd": d.sd,
                                     "a": np.select([d.code == 2, d.code == 3], [2, 3], 1)}))
    return pd.concat(out, ignore_index=True)


def fourth_fe(ns, F, T0, nm):
    F = F.merge(T0, on=["gk", "S"], how="inner")
    cell = (pd.cut(F.yl, ns["YL_BINS"], right=False).astype(str) + "|" + pd.cut(F.dist, [0, 1, 2, 4, 7, 10, 100]).astype(str) + "|" + pd.cut(F.sd, ns["LEAD_EDGES"], labels=False).astype(str) + "|" + F.qtr.astype(str)).to_numpy()
    for k, nmk in ((1, "go"), (2, "punt"), (3, "fg")):
        F["y" + nmk] = (F.a == k).astype(float)
    cols = ["ygo", "ypunt", "yfg"]
    for c in cols + ["x"]:
        F[c] = F[c] - F.groupby(cell)[c].transform("mean")
    gk = pd.Index(sorted(F.gk.unique()))
    G = len(gk)
    W = np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])
    r = slopes(F, cols, W, gk)
    say(f"  {nm}: n {len(F)} base go {np.mean(F.a == 1):.3f} punt {np.mean(F.a == 2):.3f} fg {np.mean(F.a == 3):.3f}")
    return r


def a4():
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    sim = pd.concat([sim_drives(sd) for sd in SEEDS], ignore_index=True)
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = decorate(ns, real, None, fpmap, None)
    Ds = decorate(ns, sim, None, fpmap, None)
    say(f"E88 A-side later 4th-down choice (Q3-Q4) vs A's own H1 residual x, cell FE yl bin x dist bin x lead band x quarter; label {LABEL}; boots {NB}")
    R = fourth_fe(ns, real_fourth(ns, ns["POOL"]), xtable(Dr), "real")
    S = fourth_fe(ns, sim_fourth(), xtable(Ds), "sim")
    for c in R:
        d = S[c] - R[c]
        LOOKS[0] += 1
        say(f"  {c:6s} real {fmt(R[c])} | sim {fmt(S[c])} | s-r {fmt(d)} pp {np.mean(d[1:] > 0):.2f}")
    say(f"looks {LOOKS[0]}")
    (OUTD / "a4.txt").write_text(chr(10).join(OUT))


def wx():
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = decorate(ns, real, None, fpmap, None)
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features.parquet", columns=["game_id", "season", "wind", "temp"])
    gf = gf[gf.season.between(2009, 2017)]
    gf["gk"] = gf.season.astype(int).astype(str) + "_" + gf.game_id
    g = gf.drop_duplicates("gk").set_index("gk")
    g["roof"] = np.where(g.wind.isna() & g.temp.isna(), "dome", "open")
    say(f"E88 wind/roof split of the real opponent and own later start-position slope; boots {NB}")
    say("roof counts " + str(g.roof.value_counts().to_dict()) + f"; wind mean {g.wind.mean():.1f} sd {g.wind.std():.1f}; wind missing {g.wind.isna().mean():.3f}")
    for own in (False, True):
        T = side_table(Dr, own)
        T["roof"] = g.roof.reindex(T.gk).to_numpy()
        T["wind"] = g.wind.reindex(T.gk).to_numpy()
        grp = {"dome": T.roof.isin(["dome", "closed"]), "open": ~T.roof.isin(["dome", "closed"]), "open wind>=10": ~T.roof.isin(["dome", "closed"]) & (T.wind >= 10), "open wind<10": ~T.roof.isin(["dome", "closed"]) & (T.wind < 10)}
        for nm, m in grp.items():
            U = T[m]
            gk = pd.Index(sorted(U.gk.unique()))
            G = len(gk)
            W = np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])
            r = slopes(U, ["y_e", "y_fpc", "y_rest"], W, gk)
            LOOKS[0] += 3
            say(f"  {'own' if own else 'opp'} {nm:14s} games {G:5d} " + " ".join(f"{c} {fmt(r[c])}" for c in r))
    say(f"looks {LOOKS[0]}")
    (OUTD / "wx.txt").write_text(chr(10).join(OUT))


def drive_x(D):
    X = xtable(D)
    H = D[D.h == 1].assign(S=lambda d: (d.s < 0).astype(int))
    H = H.assign(Sopp=1 - H.S)
    H = H.merge(X.rename(columns={"S": "Sopp", "x": "xo"}), on=["gk", "Sopp"], how="inner")
    return H


def lsq(x, y):
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm * xm).sum())


def refp(D, fpmap, ns, shift):
    D = D.copy()
    D["yl0"] = np.clip(D.yl0 + shift, 1, 99)
    fb = pd.cut(D.yl0, ns["YL_BINS"], right=False)
    D["fpv"] = fb.map(fpmap).astype(float).to_numpy()
    cl = [D.q0.to_numpy(), D.bi.to_numpy()]
    D["fpc"] = D.fpv - D.groupby(cl).fpv.transform("mean")
    D["ylc"] = D.yl0 - D.groupby(cl).yl0.transform("mean")
    D["rest"] = D.e - D.fpc
    return D


def fit():
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    sim = pd.concat([sim_drives(sd) for sd in SEEDS], ignore_index=True)
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = decorate(ns, real, None, fpmap, None)
    Ds = decorate(ns, sim, None, fpmap, None)
    Hr, Hs = drive_x(Dr), drive_x(Ds)
    say(f"E88 start-yardline response of side B H2 drives to A H1 residual x (state-adjusted ylc; per drive); LOSO by season; label {LABEL}")
    br = lsq(Hr.xo.to_numpy(), Hr.ylc.to_numpy())
    bs = lsq(Hs.xo.to_numpy(), Hs.ylc.to_numpy())
    folds = []
    sse0 = sse1 = 0.0
    for s in sorted(Hr.season.unique()):
        tr, te = Hr[Hr.season != s], Hr[Hr.season == s]
        b = lsq(tr.xo.to_numpy(), tr.ylc.to_numpy())
        folds.append(b)
        mx = tr.xo.mean()
        my = tr.ylc.mean()
        sse0 += float(((te.ylc - my) ** 2).sum())
        sse1 += float(((te.ylc - my - b * (te.xo - mx)) ** 2).sum())
    n = len(Hr)
    say(f"  real b {br:+.4f} yd/pt (fold sd {np.std(folds):.4f}, folds {len(folds)}); held-out SSE gain {100 * (1 - sse1 / sse0):.3f}% of {n} drives, held-out loglik gain {0.5 * n * np.log(sse0 / sse1):+.1f}")
    ws = []
    for sd in SEEDS:
        m = Hs.gk.str.startswith(f"{sd}_")
        ws.append(lsq(Hs[m].xo.to_numpy(), Hs[m].ylc.to_numpy()))
    say(f"  sim b {bs:+.4f} yd/pt (seed sd {np.std(ws):.4f}); delta real-sim {br - bs:+.4f} (fold sd {np.std(folds):.4f}); 1 sd of x = {Hr.xo.std():.2f} pts -> {(br - bs) * Hr.xo.std():+.2f} yd")
    delta = br - bs
    shift = np.where(Ds.h.to_numpy() == 1, 0.0, 0.0)
    xo = Ds[["gk", "s", "h"]].assign(S=(Ds.s < 0).astype(int))
    X = xtable(Ds)
    xo = xo.assign(Sopp=1 - xo.S).merge(X.rename(columns={"S": "Sopp", "x": "xo"}), on=["gk", "Sopp"], how="left")
    shift = np.where(Ds.h.to_numpy() == 1, delta * xo.xo.fillna(0.0).to_numpy(), 0.0)
    Dsa = refp(Ds, fpmap, ns, shift)
    say("replay (offline, sim drives with start yl shifted by delta * x on H2; real slope is the target, never fitted)")
    out = {}
    for nm, D in (("real", Dr), ("sim", Ds), ("sim+shift", Dsa)):
        T = side_table(D)
        gk = pd.Index(sorted(T.gk.unique()))
        G = len(gk)
        W = np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])
        out[nm] = slopes(T, ["y_e", "y_fpc", "y_yl", "y_n"], W, gk)
        say(f"  {nm:10s} " + " ".join(f"{c} {fmt(out[nm][c])}" for c in out[nm]))
    say(f"looks {LOOKS[0] + 1}")
    (OUTD / "fit.txt").write_text(chr(10).join(OUT))


if __name__ == "__main__":
    {"a4": a4, "wx": wx, "fit": fit}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
