import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_adj2 as a2  # noqa: E402
import mod25e_late as late  # noqa: E402
import mod25e_srcconc as sc  # noqa: E402

OUTD = a2.ART / "stform"
NB = 300
OUT = []
LOOKS = [0]
rng = np.random.default_rng(89)
fmt = a2.fmt


def say(s=""):
    print(s)
    OUT.append(s)


def real_epa(seasons):
    cols = ["game_id", "play_id", "season_type", "posteam", "defteam", "home_team", "qtr", "yardline_100", "play_type", "score_differential", "qb_kneel", "qb_spike", "epa"]
    out = []
    for s in seasons:
        p = pd.read_parquet(late.PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna()]
        p = p[p.qtr <= 4].sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        ko = (p.play_type == "kickoff").to_numpy()
        p = p.assign(kocum=np.cumsum(ko))
        p = p[~ko & p.play_type.notna()].copy()
        g, pos, kc, q = p.game_id.to_numpy(), p.posteam.to_numpy(), p.kocum.to_numpy(), p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        p["d"] = np.cumsum(new)
        kn = (p.qb_kneel == 1) | (p.qb_spike == 1)
        rp = p.play_type.isin(["run", "pass"]) & ~kn
        gp = p.groupby("d", sort=True)
        es = p.epa.where(rp, 0.0).fillna(0.0).groupby(p.d).sum()
        out.append(pd.DataFrame({"epa_sum": es.reindex(gp.size().index).to_numpy(), "yl_chk": gp.yardline_100.first().to_numpy()}))
    return pd.concat(out, ignore_index=True)


def sim_extra(sd, epa, ngm):
    sdir = a2.ART / f"e5_{a2.LABEL}_s{sd}"
    sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id"])
    out = []
    for f in sorted(sdir.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < late.BURN:
            continue
        d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "po", "pdf", "code", "idx"])
        d = d[d.qtr <= 4].sort_values("g", kind="stable").reset_index(drop=True)
        g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
        scr = ((d.po > 0) | (d.pdf > 0)).to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | scr[:-1]]
        d["d"] = np.cumsum(new)
        ix = d.idx.fillna(-1).to_numpy().astype(int)
        d["e"] = np.where((ix >= 0) & d.code.isin([0, 1]).to_numpy(), epa[np.clip(ix, 0, ngm - 1)], 0.0)
        d["ix"] = ix
        lv = d.code.isin([0, 1, 2, 3])
        gp = d.groupby("d", sort=True)
        last = d[lv].groupby("d", sort=True).ix.last().reindex(gp.size().index)
        out.append(pd.DataFrame({"epa_sum": gp.e.sum().to_numpy(), "lastidx": last.fillna(-1).to_numpy().astype(int)}))
    return pd.concat(out, ignore_index=True)


def sim_with_extra(epa, ngm):
    parts = []
    for sd in a2.SEEDS:
        D = a2.sim_drives(sd)
        X = sim_extra(sd, epa, ngm)
        assert len(D) == len(X), (len(D), len(X))
        parts.append((D, X))
    return parts


def decorate2(ns, D, fpmap, epa_sum, pidx=None):
    D = D.copy()
    D["epa_sum"] = epa_sum
    if pidx is not None:
        D["pidx"] = pidx
    D = a2.decorate(ns, D, None, fpmap, None)
    cl = [D.q0.to_numpy(), D.bi.to_numpy()]
    D["ee"] = D.epa_sum - D.groupby(cl).epa_sum.transform("mean")
    D["S"] = (D.s < 0).astype(int)
    D["ylw"] = D.ylc - D.groupby("ptype").ylc.transform("mean")
    return D


def side_games(D):
    H1 = D[D.h == 0].groupby(["gk", "S"]).agg(xe=("e", "sum"), xfp=("fpc", "sum"), xr=("rest", "sum"), xp=("ee", "sum"), sw1=("ylw", "mean"), n1=("e", "size")).reset_index()
    H2 = D[D.h == 1].groupby(["gk", "S"]).agg(y=("ylc", "sum"), ym=("ylc", "mean"), sw2=("ylw", "mean"), n2=("e", "size")).reset_index()
    return H1, H2


def pair_table(D):
    H1, H2 = side_games(D)
    A = H1.rename(columns={"sw1": "sA"})
    B2 = H2.copy()
    B2["S"] = 1 - B2.S
    B1 = H1[["gk", "S", "sw1"]].copy()
    B1["S"] = 1 - B1.S
    B1 = B1.rename(columns={"sw1": "sB"})
    T = A.merge(B2, on=["gk", "S"], how="inner").merge(B1, on=["gk", "S"], how="inner")
    T["x"] = T.xe
    T["fold"] = np.where(T.gk.str.match(r"^\d{4}_"), T.gk.str[:4], T.gk.str.rsplit("_", n=1).str[0])
    return T.dropna(subset=["sA", "sB", "sw2"]).reset_index(drop=True)


def boot_w(gk):
    G = len(gk)
    return np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])


def wls(T, ycol, xcols, W, gk, wcol=None):
    gi = gk.get_indexer(T.gk)
    X = np.c_[np.ones(len(T)), T[list(xcols)].to_numpy(float)]
    y = T[ycol].to_numpy(float)
    base = np.ones(len(T)) if wcol is None else T[wcol].to_numpy(float)
    res = np.zeros((len(W), len(xcols)))
    for b in range(len(W)):
        w = W[b][gi] * base
        Xw = X * w[:, None]
        beta = np.linalg.solve(X.T @ Xw, Xw.T @ y)
        res[b] = beta[1:]
    return res


def cmp(name, R, S, k=0):
    r, s = R[:, k], S[:, k]
    d = s - r
    LOOKS[0] += 1
    say(f"  {name:34s} real {fmt(r)} | sim {fmt(s)} | s-r {fmt(d)} pp {np.mean(d[1:] > 0):.2f}")


def wcov(T, a, b, W, gk):
    gi = gk.get_indexer(T.gk)
    x, y = T[a].to_numpy(float), T[b].to_numpy(float)
    out = np.zeros(len(W))
    for i in range(len(W)):
        w = W[i][gi]
        sw = w.sum()
        out[i] = (w * x * y).sum() / sw - (w * x).sum() / sw * (w * y).sum() / sw
    return out


def parity_table(D):
    D = D.copy()
    D["k"] = D.groupby(["gk", "S"]).cumcount() % 2
    P = D.groupby(["gk", "S", "k"]).ylw.mean().unstack()
    P.columns = ["mo", "me"]
    return P.dropna().reset_index()


def loso(T, xsets, wcol="n2"):
    res = {}
    n = len(T)
    for nm, cols in xsets.items():
        sse = 0.0
        for f in sorted(T.fold.unique()):
            tr, te = T[T.fold != f], T[T.fold == f]
            Xtr = np.c_[np.ones(len(tr)), tr[cols].to_numpy(float)] if cols else np.ones((len(tr), 1))
            Xte = np.c_[np.ones(len(te)), te[cols].to_numpy(float)] if cols else np.ones((len(te), 1))
            w = tr[wcol].to_numpy(float)
            beta = np.linalg.solve(Xtr.T @ (Xtr * w[:, None]), (Xtr * w[:, None]).T @ tr.ym.to_numpy(float))
            sse += float((te[wcol].to_numpy(float) * (te.ym.to_numpy(float) - Xte @ beta) ** 2).sum())
        res[nm] = sse
    s0 = res["const"]
    return {k: (100 * (1 - v / s0), 0.5 * n * np.log(s0 / v)) for k, v in res.items()}


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    rx = real_epa(ns["POOL"])
    assert len(rx) == len(real) and np.allclose(rx.yl_chk.to_numpy(), real.yl0.to_numpy(), equal_nan=True), "real epa alignment"
    say(f"E89 special-teams game form and EPA-only residual; label {a2.LABEL} seeds {a2.SEEDS}; no sim run; boots {NB}")
    say(f"S0 real drive alignment: {len(real)} drives, start yardline matches replica on all")
    tables, trans, pbp = sc.engine()
    right = pbp[["game_id", "play_id", "epa"]].drop_duplicates(["game_id", "play_id"])
    epa = trans[["game_id", "play_id"]].merge(right, on=["game_id", "play_id"], how="left")["epa"].fillna(0.0).to_numpy(dtype=np.float64)
    gid_pool = trans["game_id"].to_numpy()
    ngm = len(trans)
    parts = sim_with_extra(epa, ngm)
    sim = pd.concat([p[0] for p in parts], ignore_index=True)
    X = pd.concat([p[1] for p in parts], ignore_index=True)
    pidx = np.r_[-1, X.lastidx.to_numpy()[:-1]]
    pidx = np.where(np.r_[False, sim.gk.to_numpy()[1:] == sim.gk.to_numpy()[:-1]], pidx, -1)
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = decorate2(ns, real, fpmap, rx.epa_sum.to_numpy())
    Ds = decorate2(ns, sim, fpmap, X.epa_sum.to_numpy(), pidx)
    say(f"S0 sim drives {len(Ds)}; real games {Dr.gk.nunique()} sim games {Ds.gk.nunique()}; pool rows {ngm}; real EPA sum/drive mean {Dr.epa_sum.mean():+.4f} sim {Ds.epa_sum.mean():+.4f}; sd {Dr.epa_sum.std():.3f} | {Ds.epa_sum.std():.3f}")
    Tr, Ts = pair_table(Dr), pair_table(Ds)
    gr, gs = pd.Index(sorted(Tr.gk.unique())), pd.Index(sorted(Ts.gk.unique()))
    Wr, Ws = boot_w(gr), boot_w(gs)
    say(f"pair rows (A H1 -> B H2) real {len(Tr)} sim {len(Ts)}")

    say("S1 (c) which x explains the B H2 start slope: y = sum of B H2 state-adjusted start yardlines (yd) on A H1 x; per unit x")
    for xn, lab in (("xe", "x = drive net points resid"), ("xp", "x = run/pass EPA-sum resid"), ("xfp", "x = start-position part of points resid"), ("xr", "x = rest of points resid")):
        cmp(lab, wls(Tr, "y", [xn], Wr, gr), wls(Ts, "y", [xn], Ws, gs))
    say("  joint (xfp, xr) on y")
    R, S = wls(Tr, "y", ["xfp", "xr"], Wr, gr), wls(Ts, "y", ["xfp", "xr"], Ws, gs)
    cmp("  coef xfp", R, S, 0)
    cmp("  coef xr", R, S, 1)
    rr, ss = Tr[["xe", "xp", "xfp", "xr"]].corr().loc["xe"], Ts[["xe", "xp", "xfp", "xr"]].corr().loc["xe"]
    say(f"  corr(xe, .) real xp {rr.xp:.3f} xfp {rr.xfp:.3f} xr {rr.xr:.3f} | sim xp {ss.xp:.3f} xfp {ss.xfp:.3f} xr {ss.xr:.3f}")

    say("S2 (b) special-teams game form from the same game's other drive starts (within-type start residual, yd)")
    say("  sA = A H1 mean start resid (A returns / B coverage), sB = B H1 mean start resid; slope of y on x after adding them")
    R, S = wls(Tr, "y", ["xe", "sA", "sB"], Wr, gr), wls(Ts, "y", ["xe", "sA", "sB"], Ws, gs)
    cmp("coef xe given sA sB", R, S, 0)
    cmp("coef sA (per yd, sum y)", R, S, 1)
    cmp("coef sB (per yd, sum y)", R, S, 2)
    R1, S1 = wls(Tr, "y", ["xe"], Wr, gr), wls(Ts, "y", ["xe"], Ws, gs)
    say(f"  xe slope absorbed by sA sB: real {100 * (1 - R[0, 0] / R1[0, 0]):.1f}% sim {100 * (1 - S[0, 0] / S1[0, 0]):.1f}%")
    say("  game-level start-resid latent per side: cov(odd drives, even drives) of side-game mean resid, yd^2 (true variance); and H1-H2")
    for lab, TT in (("real", Dr), ("sim", Ds)):
        P = parity_table(TT)
        g = pd.Index(sorted(P.gk.unique()))
        W = boot_w(g)
        c = wcov(P, "mo", "me", W, g)
        LOOKS[0] += 1
        say(f"  {lab} odd-even cov {fmt(c, 3)} n {len(P)}; sd of game latent {np.sqrt(max(c[0], 0)):.3f} yd; probability_positive {np.mean(c[1:] > 0):.2f}")
    for lab, TT, W, g in (("real", Tr, Wr, gr), ("sim", Ts, Ws, gs)):
        c1 = wcov(TT, "sB", "sw2", W, g)
        c2 = wcov(TT, "sA", "sw2", W, g)
        LOOKS[0] += 2
        say(f"  {lab} cov(B H1 mean, B H2 mean) own-side {fmt(c1, 3)} pp {np.mean(c1[1:] > 0):.2f}; cov(A H1 mean, B H2 mean) cross {fmt(c2, 3)} pp {np.mean(c2[1:] > 0):.2f}")
    say("  LOSO (by season / sim world) held-out SSE gain % and loglik gain on mean B H2 start resid, weights n2")
    sets = {"const": [], "xe": ["xe"], "sA,sB": ["sA", "sB"], "xe,sA,sB": ["xe", "sA", "sB"]}
    for lab, TT in (("real", Tr), ("sim", Ts)):
        r = loso(TT, sets)
        LOOKS[0] += len(sets) - 1
        say(f"  {lab} " + "; ".join(f"{k} {v[0]:+.3f}% ll {v[1]:+.1f}" for k, v in r.items() if k != "const"))

    say("S3 (a) source-game environment: sim B H2 start after punt/turnover draws pool row idx; env = pool game's mean H2 start resid (real: own game, leave-drive-out)")
    H2r = Dr[Dr.h == 1]
    gsum = H2r.groupby(Dr.gk.str.split("_", n=1).str[1]).ylc.agg(["sum", "count"])
    sel_r = (Dr.h == 1) & Dr.ptype.isin(["punt", "to"])
    Dr = Dr.assign(gid=Dr.gk.str.split("_", n=1).str[1])
    gs_sum = Dr[Dr.h == 1].groupby("gk").ylc.agg(["sum", "count"]).reindex(Dr.gk).reset_index(drop=True)
    Dr["env"] = np.where(sel_r & (gs_sum["count"] > 1), (gs_sum["sum"] - Dr.ylc) / (gs_sum["count"] - 1), np.nan)
    sel_s = (Ds.h == 1) & Ds.ptype.isin(["punt", "to"]) & (Ds.pidx >= 0)
    gp = gid_pool[np.clip(Ds.pidx.to_numpy(), 0, ngm - 1)]
    gm = (gsum["sum"] / gsum["count"]).reindex(gp).to_numpy()
    Ds["env"] = np.where(sel_s, gm, np.nan)
    say(f"  sim punt/turnover H2 starts {int(sel_s.sum())} with pool game env found {int(np.isfinite(Ds.env[sel_s]).sum())}; real {int(sel_r.sum())} with env {int(np.isfinite(Dr.env[sel_r]).sum())}")
    res = {}
    for lab, D, Wc, g in (("real", Dr, Wr, gr), ("sim", Ds, Ws, gs)):
        Dd = D[np.isfinite(D.env)].copy()
        Dd["S"] = 1 - Dd.S
        Bh = Dd.groupby(["gk", "S"]).agg(ye=("env", "sum"), yl=("ylc", "sum"), nn=("env", "size")).reset_index()
        H1, _ = side_games(D)
        T = H1.merge(Bh, on=["gk", "S"], how="inner")
        T["x"] = T.xe
        res[lab] = (T, Wc, g)
    for yc, lab in (("ye", "env of drawn/own game"), ("yl", "start resid of same drives")):
        R = wls(res["real"][0], yc, ["x"], res["real"][1], res["real"][2])
        S = wls(res["sim"][0], yc, ["x"], res["sim"][1], res["sim"][2])
        cmp(lab, R, S)
    say(f"looks {LOOKS[0]}")
    (OUTD / "stform.txt").write_text(chr(10).join(OUT))


if __name__ == "__main__":
    main()
