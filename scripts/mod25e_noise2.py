import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ART = REPO / "artifacts" / "mod25e3"
LABEL = "crHpqokg"
SEEDS = (11, 12, 13)
OUTD = ART / "noise2"
POOL = tuple(range(2009, 2018))
BURN = 2
NB = 300
SEEDOFF = 1_000_000
LEAD = [-np.inf, -8.5, -0.5, 0.5, 8.5, np.inf]
LEADN = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
ENDS = ["td", "fg", "fgmiss", "punt", "lost", "defsc", "other"]


def sim_dir(sd):
    return ART / f"e5_{LABEL}_s{sd}"


def fmt(pt, bt):
    lo, hi = np.quantile(bt, [0.025, 0.975])
    return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}] pp {float((bt > 0).mean()):.2f}"


def sim_budget_inputs():
    import mod25e_budget as bud

    Ts, Gs = [], []
    for sd in SEEDS:
        bud.OUT = sim_dir(sd)
        T, G = bud.build_sim(BURN)
        T["sid"] = T["sid"] + sd * SEEDOFF
        G = G.copy()
        G["sid"] = G["sid"] + sd * SEEDOFF
        Ts.append(T)
        Gs.append(G)
    return pd.concat(Ts, ignore_index=True), pd.concat(Gs, ignore_index=True)


def cmd_budget(a):
    import mod25e_budget as bud
    from mod25e_analysis import per_game, season_re, stats

    OUTD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    sets = {}
    T, G = bud.build_real(POOL)
    D, X = per_game(T, G)
    sets["real_pool"] = (D, X, season_re(G))
    T, G = sim_budget_inputs()
    chk = T.groupby(["sid", "g"]).margin.first().rename("pm").reset_index().merge(G.assign(m=G.home_score - G.away_score)[["sid", "g", "m"]], on=["sid", "g"])
    print("sim games", len(chk), "play margin equals games margin", float((chk.pm == chk.m).mean()), flush=True)
    D, X = per_game(T, G)
    sets["sim"] = (D, X, season_re(G))
    point = {nm: stats(D, X, list(r.values())) for nm, (D, X, r) in sets.items()}
    sidx = {nm: ({k: v for k, v in D.groupby("sid")}, {k: v for k, v in X.groupby("sid")}, r) for nm, (D, X, r) in sets.items()}
    boots = {nm: [] for nm in sets}
    for b in range(a.boot):
        for nm, (Dg, Xg, r) in sidx.items():
            ks = list(Dg.keys())
            sel = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
            Db = pd.concat([Dg[k] for k in sel], ignore_index=True)
            Xb = pd.concat([Xg[k] for k in sel], ignore_index=True)
            boots[nm].append(stats(Db, Xb, [r[k] for k in sel if k in r]))
    L = []
    for k in point["sim"]:
        br = np.array([x[k] for x in boots["real_pool"]])
        bs = np.array([x[k] for x in boots["sim"]])
        d = bs - br
        d = d[np.isfinite(d)]
        L.append(f"{k:28s} real {point['real_pool'][k]:10.3f} sim {point['sim'][k]:10.3f} diff {fmt(point['sim'][k] - point['real_pool'][k], d)}")
    (OUTD / "budget.txt").write_text("\n".join(L))
    print("\n".join(L))


def sim_drive_frame():
    out = []
    for sd in SEEDS:
        gm = pd.read_parquet(sim_dir(sd) / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
        gm["w"] = gm.game_id.str[1:5].astype(int)
        gm["s"] = gm.game_id.str[6:8].astype(int)
        gm["g"] = gm.game_id.str[-3:].astype(int)
        gm = gm.set_index(["w", "s", "g"])
        for f in sorted(glob.glob(str(sim_dir(sd) / "play_*_*.parquet"))):
            w, s = (int(x) for x in Path(f).stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f)
            d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
            g = d.g.to_numpy()
            oh = d.offhome.to_numpy()
            q = d.qtr.to_numpy()
            new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
            d["d"] = np.cumsum(new)
            d["pp"] = d.po - d.pdf
            gp = d.groupby("d", sort=True)
            key = sd * 100000 + w * 1000 + s
            g0 = gp.g.first().astype(int)
            tm = gm.reindex(pd.MultiIndex.from_arrays([np.full(len(g0), w), np.full(len(g0), s), g0.to_numpy()]))
            offh = gp.offhome.first().to_numpy() == 1
            off = np.where(offh, tm.home_team.to_numpy(), tm.away_team.to_numpy())
            dfn = np.where(offh, tm.away_team.to_numpy(), tm.home_team.to_numpy())
            D = pd.DataFrame({"gk": f"{key}_" + g0.astype(str), "season": key, "off": off, "dfn": dfn, "s": np.where(offh, 1.0, -1.0),
                              "ql": gp.qtr.last(), "yl0": gp.yl.first(), "secs": gp.el.sum(), "n": gp.size(), "sd0": gp.sd.first(), "p": gp.pp.sum(),
                              "pt": np.where(gp.code.last() == 2, "punt", "x")})
            D["tm"] = D.season.astype(str) + "_" + D.off
            D["td"] = D.season.astype(str) + "_" + D.dfn
            out.append(D)
    return pd.concat(out, ignore_index=True)


def cmd_xq(a):
    import mod25e_late as late
    import mod25e_xq2 as x2

    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    rp = ns["real_drives"](ns["POOL"])[0]
    sm = sim_drive_frame()
    res = {"real_pool": ns["run"]("real_pool", rp), "sim": ns["run"]("sim", sm)}
    L = []
    for k, r in res.items():
        L.append(f"{k} games {r['games']} drives/g {r['drives_g']:.2f} mu {r['mu']:.3f} xq(sum comp) {r['total']:.2f} true_xq {r['true_xq']:.2f}")
        L.append("  comp " + " ".join(f"{c}={v:+.2f}" for c, v in zip(ns["COMP"], r["R"])))
        L.append("  count-diff var by q " + " ".join(f"{v:.2f}" for v in r["cnt_var_by_q"]))
    B = {k: np.array(r["boot"]) for k, r in res.items()}
    n = min(len(B["sim"]), len(B["real_pool"]))
    L.append("sim minus real_pool (2x cov components; 95% boot; probability_positive)")
    for i, c in enumerate(ns["COMP"] + ["total"]):
        sv, rv = B["sim"][:n], B["real_pool"][:n]
        sd = sv.sum(1) if c == "total" else sv[:, i]
        rd = rv.sum(1) if c == "total" else rv[:, i]
        L.append(f"  {c:10s} {fmt(float(sd.mean() - rd.mean()), sd - rd)}")
    srcs = {"real_pool": rp, "sim": sm}
    mats = {}
    rng = np.random.default_rng(7)
    for nm, D in srcs.items():
        D, I, gi, qi = x2.adj_incr(ns, D)
        C = np.cov(I, rowvar=False)
        G = len(I)
        Bt = []
        for _ in range(NB):
            Cb = np.cov(I[rng.integers(0, G, G)], rowvar=False)
            Bt.append([Cb[q, r] for q in range(4) for r in range(q + 1, 4)])
        mats[nm] = (C, np.array(Bt))
        L.append(f"{nm} games {G} var by q " + " ".join(f"{C[q, q]:.1f}" for q in range(4)))
    names = [f"{q + 1}{r + 1}" for q in range(4) for r in range(q + 1, 4)]
    d = 2 * (mats["sim"][1] - mats["real_pool"][1])
    pt = 2 * np.array([mats["sim"][0][q, r] - mats["real_pool"][0][q, r] for q in range(4) for r in range(q + 1, 4)])
    L.append("quarter-pair matrix sim minus real (strength-adjusted, 2x cov)")
    for i, nme in enumerate(names):
        L.append(f"  {nme} {fmt(pt[i], d[:, i])}")
    L.append(f"  total {fmt(pt.sum(), d.sum(1))}")
    (OUTD / "xq.txt").write_text("\n".join(L))
    print("\n".join(L))


def load_sim_plays():
    fr = []
    for sd in SEEDS:
        for f in sorted(glob.glob(str(sim_dir(sd) / "play_*_*.parquet"))):
            w, s = (int(x) for x in Path(f).stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f)
            d = d[d.qtr <= 4].copy()
            d["unit"] = f"{sd}_{w}_{s}"
            d["gid"] = d.unit + "_" + d.g.astype(int).astype(str)
            fr.append(d)
    return pd.concat(fr, ignore_index=True)


def drive_table(D):
    import mod25e_pts as P

    D = P.drives(D)
    X = P.drive_frame(D)
    gp = D.groupby("d", sort=True)
    X["net"] = X.pts - X.pdf
    X["q0"] = gp.qtr.first()
    X["sd0"] = gp.sd.first()
    X["lead"] = pd.cut(X.sd0, LEAD, labels=LEADN).astype(str)
    X["gid"] = gp.gid.first()
    X["span"] = X.yl0 - X.ylmin
    return X


def cells(X, edges):
    return {"fp": np.digitize(X.yl0.to_numpy(), edges), "qtr": X.q0.astype(int).to_numpy() - 1,
            "lead": pd.Categorical(X.lead, categories=LEADN).codes, "end": pd.Categorical(X.end, categories=ENDS).codes}


def unit_tensor(X, idx, K, units):
    ui = pd.Series(np.arange(len(units)), index=units).reindex(X.unit).to_numpy()
    N = np.zeros((len(units), K))
    S1 = np.zeros_like(N)
    S2 = np.zeros_like(N)
    np.add.at(N, (ui, idx), 1)
    np.add.at(S1, (ui, idx), X.net.to_numpy())
    np.add.at(S2, (ui, idx), X.net.to_numpy() ** 2)
    return N, S1, S2


def cmd_drv(a):
    import mod25e_pts as P

    OUTD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    R = drive_table(P.load_real())
    S = drive_table(load_sim_plays())
    ur = np.array(sorted(R.unit.unique()))
    us = np.array(sorted(S.unit.unique()))
    gr = R.groupby("unit").gid.nunique().reindex(ur).to_numpy().astype(float)
    gs = S.groupby("unit").gid.nunique().reindex(us).to_numpy().astype(float)
    edges = np.quantile(R.yl0, np.linspace(0, 1, 7)[1:-1])
    cr, cs = cells(R, edges), cells(S, edges)
    L = [f"games real {gr.sum():.0f} sim {gs.sum():.0f} units real {len(ur)} sim {len(us)} drives/g {len(R) / gr.sum():.3f} {len(S) / gs.sum():.3f}", f"fp edges {np.round(edges, 1).tolist()}"]
    dims = {"fp": len(edges) + 1, "qtr": 4, "lead": 5, "end": 7}
    labs = {"fp": [f"fp{i}" for i in range(len(edges) + 1)], "qtr": [f"Q{i + 1}" for i in range(4)], "lead": LEADN, "end": ENDS}
    nlook = 0
    for dim, K in dims.items():
        Nr, A1r, A2r = unit_tensor(R, cr[dim], K, ur)
        Ns, A1s, A2s = unit_tensor(S, cs[dim], K, us)

        def stat(ir, is_):
            g_r, g_s = gr[ir].sum(), gs[is_].sum()
            cnr, cns = Nr[ir].sum(0), Ns[is_].sum(0)
            nr, ns_ = cnr / g_r, cns / g_s
            m2r = A2r[ir].sum(0) / np.maximum(cnr, 1)
            m2s = A2s[is_].sum(0) / np.maximum(cns, 1)
            m1r = A1r[ir].sum(0) / np.maximum(cnr, 1)
            m1s = A1s[is_].sum(0) / np.maximum(cns, 1)
            ss = ns_ * m2s - nr * m2r
            vol = (ns_ - nr) * (m2s + m2r) / 2
            per = (ns_ + nr) / 2 * (m2s - m2r)
            return ss, vol, per, (m2s - m1s**2) - (m2r - m1r**2), m1s - m1r, ns_, nr

        base = stat(np.arange(len(ur)), np.arange(len(us)))
        bt = [stat(rng.integers(0, len(ur), len(ur)), rng.integers(0, len(us), len(us))) for _ in range(NB)]
        L.append(f"[{dim}] sum of x^2 per game (x = drive net points), sim minus real: total {fmt(base[0].sum(), np.array([t[0].sum() for t in bt]))}")
        for k in range(K):
            nlook += 1
            L.append(f"  {labs[dim][k]:9s} n/g r {base[6][k]:.3f} s {base[5][k]:.3f} ss {fmt(base[0][k], np.array([t[0][k] for t in bt]))} vol {base[1][k]:+.3f} per {fmt(base[2][k], np.array([t[2][k] for t in bt]))} var(x) diff {fmt(base[3][k], np.array([t[3][k] for t in bt]))} mean diff {base[4][k]:+.3f}")
    L.append(f"looks {nlook} (4 families: fp 6, qtr 4, lead 5, end 7)")
    K = 6 * 7
    Nr, A1r, A2r = unit_tensor(R, cr["fp"] * 7 + cr["end"], K, ur)
    Ns, A1s, A2s = unit_tensor(S, cs["fp"] * 7 + cs["end"], K, us)

    def sscell(ir, is_):
        return A2s[is_].sum(0) / gs[is_].sum() - A2r[ir].sum(0) / gr[ir].sum()

    base = sscell(np.arange(len(ur)), np.arange(len(us)))
    bt = np.array([sscell(rng.integers(0, len(ur), len(ur)), rng.integers(0, len(us), len(us))) for _ in range(NB)])
    L.append(f"[fp x end, 42 cells] sum x^2 per game sim minus real (total {base.sum():+.3f}); top 12 by abs diff")
    for k in np.argsort(-np.abs(base))[:12]:
        L.append(f"  fp{k // 7} {ENDS[k % 7]:7s} n/g r {Nr[:, k].sum() / gr.sum():.3f} s {Ns[:, k].sum() / gs.sum():.3f} {fmt(base[k], bt[:, k])}")
    for nm, X in (("real", R), ("sim", S)):
        tdx = X[X.end == "td"]
        L.append(f"{nm} TD drive span q25/50/75/90 {np.round(np.quantile(tdx.span, [.25, .5, .75, .9]), 1).tolist()} share span>=60 {float((tdx.span >= 60).mean()):.3f}; TD share {float((X.end == 'td').mean()):.4f}")
    (OUTD / "drv.txt").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--boot", type=int, default=NB)
    sub.add_parser("xq")
    sub.add_parser("drv")
    a = ap.parse_args()
    {"budget": cmd_budget, "xq": cmd_xq, "drv": cmd_drv}[a.cmd](a)


if __name__ == "__main__":
    main()
