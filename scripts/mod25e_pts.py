import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "pts"
SEEDS = (11, 12, 13)
BURN = 2
NB = 400
OUTC = ["td", "fg", "fgmiss", "punt", "lost", "defsc", "other"]
ZN = ["own", "mid", "rz"]


def load_real():
    import mod25e_f2pr as m

    R = pd.read_parquet(m.OUT / "real_fit.parquet")
    pool = pd.read_parquet(m.f2.u4g.OUT / "pool.parquet", columns=["down", "yl"])
    assert (pool.down.to_numpy() == R.down.to_numpy()).all() and (pool.yl.to_numpy() == R.yl.to_numpy()).all()
    R["ridx"] = np.arange(len(R))
    R = R[(R.qtr <= 4) & (R.season >= 2009) & (R.season <= 2017)].copy()
    R["offhome"] = (R.posteam == R.home_team).astype(float)
    R = R.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
    R["gid"] = R.game_id.astype(str)
    R["unit"] = R.gid
    return R


def load_sim():
    fr = []
    for sd in SEEDS:
        for f in sorted(glob.glob(str(ART / f"e5_crHp_s{sd}" / "play_*_*.parquet"))):
            w, s = (int(x) for x in Path(f).stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f)
            d = d[d.qtr <= 4].copy()
            d["unit"] = f"{sd}_{w}_{s}"
            d["gid"] = d.unit + "_" + d.g.astype(int).astype(str)
            fr.append(d)
    S = pd.concat(fr, ignore_index=True)
    S["ridx"] = S.idx.astype(int)
    return S


def drives(D):
    g = D.gid.to_numpy()
    oh = D.offhome.to_numpy()
    q = D.qtr.to_numpy()
    sc = ((D.po > 0) | (D.pdf > 0)).to_numpy()
    new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | sc[:-1]]
    D["d"] = np.cumsum(new)
    D["pts"] = D.po + D.pdf
    return D


def zone_of(yl):
    return np.where(yl <= 20, 2, np.where(yl <= 50, 1, 0))


def cell(D, edges):
    de, ye = edges
    di = np.digitize(D.dist.to_numpy(), de)
    yi = np.digitize(D.yl.to_numpy(), ye)
    return D.down.clip(1, 4).astype(int).to_numpy() * 10000 + di * 100 + yi


def build_v(R, edges):
    R = R.copy()
    R["cell"] = cell(R, edges)
    R["rem"] = R.groupby("d").pts.transform(lambda x: x[::-1].cumsum()[::-1])
    return R.groupby("cell").rem.mean()


def adj_table(D, v, edges):
    D = D.copy()
    D["cell"] = cell(D, edges)
    D["v"] = D.cell.map(v).fillna(float(v.mean())).to_numpy()
    dd = D.d.to_numpy()
    last = np.r_[dd[1:] != dd[:-1], True]
    nv = np.where(last, 0.0, np.r_[D.v.to_numpy()[1:], 0.0])
    D["adj"] = D.pts + nv - D.v
    first = np.r_[True, dd[1:] != dd[:-1]]
    D["v0"] = np.where(first, D.v, 0.0)
    D["ty"] = np.select([D.code == 0, D.code == 1, D.code == 2, D.code == 3], ["run", "pass", "punt", "fg"], "oth")
    D["zn"] = np.array(ZN)[zone_of(D.yl.to_numpy())]
    return D


def drive_frame(D):
    gp = D.groupby("d", sort=True)
    X = pd.DataFrame({"unit": gp.unit.first(), "yl0": gp.yl.first(), "pts": gp.pts.sum(), "pdf": gp.pdf.sum(), "ylmin": gp.yl.min()})
    lv = D[D.code.isin([0, 1, 2, 3])].groupby("d", sort=True)
    lc = lv.code.last().reindex(X.index)
    lfl = lv.flip.last().reindex(X.index)
    own = X.pts - X.pdf
    X["end"] = np.select([X.pdf > 0, own >= 6, (lc == 3) & (own == 3), lc == 3, lc == 2, (lfl == 1) & lc.isin([0, 1])], ["defsc", "td", "fg", "fgmiss", "punt", "lost"], "other")
    return X


def tensor(X, edges, units):
    b = np.digitize(X.yl0.to_numpy(), edges)
    o = pd.Categorical(X.end, categories=OUTC).codes
    N = np.zeros((len(units), len(edges) + 1, len(OUTC)))
    P = np.zeros_like(N)
    ui = pd.Series(np.arange(len(units)), index=units).reindex(X.unit).to_numpy()
    np.add.at(N, (ui, b, o), 1)
    np.add.at(P, (ui, b, o), X.pts.to_numpy())
    return N, P


def decomp(Ns, Ps, Gs, Nr, Pr, Gr):
    ns, nr = Ns / Gs, Nr / Gr
    ps, pr = Ps / Gs, Pr / Gr
    nbs, nbr = ns.sum(1), nr.sum(1)
    pis = ns / np.maximum(nbs[:, None], 1e-12)
    pir = nr / np.maximum(nbr[:, None], 1e-12)
    mus = np.where(ns > 0, ps / np.maximum(ns, 1e-12), 0.0)
    mur = np.where(nr > 0, pr / np.maximum(nr, 1e-12), 0.0)
    Ms = (pis * mus).sum(1)
    Mr = (pir * mur).sum(1)
    vol = (nbs - nbr) * (Ms + Mr) / 2
    nbar = ((nbs + nbr) / 2)[:, None]
    probe = nbar * (pis - pir) * (mus + mur) / 2
    ptsr = nbar * (pis + pir) / 2 * (mus - mur)
    return vol, probe, ptsr


def fmt(pt, bt):
    lo, hi = np.quantile(bt, [0.025, 0.975])
    return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}] pp {float((bt > 0).mean()):.2f}"


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    R = drives(load_real())
    S = drives(load_sim())
    Gr, Gs = R.gid.nunique(), S.gid.nunique()
    print("games real", Gr, "sim", Gs, "pts/g real", round(R.pts.sum() / Gr, 3), "sim", round(S.pts.sum() / Gs, 3), "drives/g", round(R.d.nunique() / Gr, 3), round(S.d.nunique() / Gs, 3), flush=True)
    Xr, Xs = drive_frame(R), drive_frame(S)
    ur = np.array(sorted(Xr.unit.unique()))
    us = np.array(sorted(Xs.unit.unique()))
    gur = R.groupby("unit").gid.nunique().reindex(ur).to_numpy()
    gus = S.groupby("unit").gid.nunique().reindex(us).to_numpy()
    edges = np.quantile(Xr.yl0, np.linspace(0, 1, 7)[1:-1])
    Nr, Pr = tensor(Xr, edges, ur)
    Ns, Ps = tensor(Xs, edges, us)

    def dec(ir, is_):
        return decomp(Ns[is_].sum(0), Ps[is_].sum(0), gus[is_].sum(), Nr[ir].sum(0), Pr[ir].sum(0), gur[ir].sum())

    base = dec(np.arange(len(ur)), np.arange(len(us)))
    bt = [dec(rng.integers(0, len(ur), len(ur)), rng.integers(0, len(us), len(us))) for _ in range(NB)]
    nbn = len(edges) + 1
    lab = [f"yl0<={edges[0]:.0f}"] + [f"yl0({edges[i-1]:.0f},{edges[i]:.0f}]" for i in range(1, len(edges))] + [f"yl0>{edges[-1]:.0f}"]
    rows = []
    for b in range(nbn):
        rows.append(("vol", b, None, base[0][b], np.array([t[0][b] for t in bt])))
        for k, o in enumerate(OUTC):
            rows.append(("prob", b, o, base[1][b, k], np.array([t[1][b, k] for t in bt])))
            rows.append(("pts", b, o, base[2][b, k], np.array([t[2][b, k] for t in bt])))
    print("drive-level exact total", round(sum(r[3] for r in rows), 4), "direct", round(S.pts.sum() / Gs - R.pts.sum() / Gr, 4))

    def agg(sel, name):
        rs = [r for r in rows if sel(r)]
        print(" ", name, fmt(sum(r[3] for r in rs), sum(r[4] for r in rs)))

    agg(lambda r: r[0] == "vol", "start volume (bin drives/g x bin value)")
    agg(lambda r: r[0] == "prob", "outcome probabilities within start bin")
    agg(lambda r: r[0] == "pts", "points per outcome class (PAT/2pt/def)")
    for o in OUTC:
        agg(lambda r, o=o: r[0] == "prob" and r[2] == o, f"prob {o}")
        agg(lambda r, o=o: r[0] == "pts" and r[2] == o, f"pts-per {o}")
    for b in range(nbn):
        agg(lambda r, b=b: r[1] == b, f"start bin {lab[b]} (all)")
    for b in range(nbn):
        agg(lambda r, b=b: r[1] == b and r[0] == "prob", f"start bin {lab[b]} prob only")
    print("  top 10 cells:")
    for r in sorted(rows, key=lambda r: -abs(r[3]))[:10]:
        print("   ", r[0], lab[r[1]], r[2], fmt(r[3], r[4]))

    de = np.unique(np.quantile(R[R.code.isin([0, 1])].dist, [0.2, 0.4, 0.6, 0.8]))
    ye = np.unique(np.quantile(R.yl, np.linspace(0, 1, 11)[1:-1]))
    ed = (de, ye)
    v = build_v(R, ed)
    Ra, Sa = adj_table(R, v, ed), adj_table(S, v, ed)
    print("ladder: real adj/g", round(Ra.adj.sum() / Gr, 4), "sim adj/g", round(Sa.adj.sum() / Gs, 4), "V0 real", round(Ra.v0.sum() / Gr, 3), "sim", round(Sa.v0.sum() / Gs, 3), flush=True)

    def catsum(A, units):
        A = A.copy()
        A["k"] = A.down.clip(1, 4).astype(int).astype(str) + " " + A.ty + " " + A.zn
        gp = A.groupby(["unit", "k"]).adj.sum().unstack("k").reindex(units).fillna(0.0)
        v0 = A.groupby("unit").v0.sum().reindex(units).fillna(0.0)
        return gp, v0

    gr_, v0r = catsum(Ra, ur)
    gs_, v0s = catsum(Sa, us)
    allk = sorted(set(gr_.columns) | set(gs_.columns))
    Mr = np.column_stack([v0r.to_numpy(), gr_.reindex(columns=allk, fill_value=0.0).to_numpy()])
    Ms = np.column_stack([v0s.to_numpy(), gs_.reindex(columns=allk, fill_value=0.0).to_numpy()])
    names = ["start V0"] + ["d" + k for k in allk]

    def lad(ir, is_):
        return Ms[is_].sum(0) / gus[is_].sum() - Mr[ir].sum(0) / gur[ir].sum()

    pt = lad(np.arange(len(ur)), np.arange(len(us)))
    bt = np.array([lad(rng.integers(0, len(ur), len(ur)), rng.integers(0, len(us), len(us))) for _ in range(NB)])
    print("ladder sum", round(pt.sum(), 4), fmt(pt.sum(), bt.sum(1)))
    order = np.argsort(pt)
    print("ladder cells, most negative:")
    for i in order[:12]:
        print("  ", names[i], fmt(pt[i], bt[:, i]))
    print("most positive:")
    for i in order[::-1][:4]:
        print("  ", names[i], fmt(pt[i], bt[:, i]))
    for d in (1, 2, 3, 4):
        ix = [i for i, k in enumerate(names) if k.startswith(f"d{d} ")]
        print("  down", d, fmt(pt[ix].sum(), bt[:, ix].sum(1)))
    for z in ZN:
        ix = [i for i, k in enumerate(names) if k.endswith(" " + z)]
        print("  zone", z, fmt(pt[ix].sum(), bt[:, ix].sum(1)))
    for t in ("run", "pass", "punt", "fg", "oth"):
        ix = [i for i, k in enumerate(names) if f" {t} " in k]
        print("  type", t, fmt(pt[ix].sum(), bt[:, ix].sum(1)))
    for d in (1, 2, 3, 4):
        for t in ("run", "pass"):
            ix = [i for i, k in enumerate(names) if k.startswith(f"d{d} {t} ")]
            print("  down", d, t, fmt(pt[ix].sum(), bt[:, ix].sum(1)))
    pd.DataFrame({"name": names, "pts_g": pt, "lo": np.quantile(bt, 0.025, axis=0), "hi": np.quantile(bt, 0.975, axis=0), "pp": (bt > 0).mean(0)}).to_csv(OUTD / "ladder.csv", index=False)

    ex = np.quantile(R[R.code.isin([0, 1])].yards, 0.94)
    print("explosive cut (real q.94 yards)", ex)

    def convtab(D):
        L = D[D.code.isin([0, 1])].copy()
        L["conv"] = ((L.yards >= L.dist) | (L.po > 0)).astype(int)
        L["db"] = np.digitize(L.dist, de)
        L["expl"] = (L.yards >= ex).astype(int)
        return L

    Lr, Ls = convtab(R), convtab(S)
    print("conversion by down x dist bin")
    for d in (1, 2, 3, 4):
        for b in range(len(de) + 1):
            a, c = Lr[(Lr.down == d) & (Lr.db == b)], Ls[(Ls.down == d) & (Ls.db == b)]
            if len(a) < 200 or len(c) < 200:
                continue
            print(f"  d{d} dist {a.dist.min():.0f}-{a.dist.max():.0f} real {a.conv.mean():.3f} n/g {len(a)/Gr:.2f} | sim {c.conv.mean():.3f} n/g {len(c)/Gs:.2f}")
    print("ypp, explosive by down x zone")
    for d in (1, 2, 3):
        for z in (0, 1, 2):
            a = Lr[(Lr.down == d) & (zone_of(Lr.yl.to_numpy()) == z)]
            c = Ls[(Ls.down == d) & (zone_of(Ls.yl.to_numpy()) == z)]
            print(f"  d{d} {ZN[z]} ypp {a.yards.mean():.2f} {c.yards.mean():.2f} | expl {a.expl.mean():.4f} {c.expl.mean():.4f} | n/g {len(a)/Gr:.2f} {len(c)/Gs:.2f}")
    ra, sa = Xr[Xr.ylmin <= 20], Xs[Xs.ylmin <= 20]
    print("rz arrival", round(len(ra) / len(Xr), 4), round(len(sa) / len(Xs), 4), "td|arr", round((ra.end == "td").mean(), 4), round((sa.end == "td").mean(), 4), "fg|arr", round((ra.end == "fg").mean(), 4), round((sa.end == "fg").mean(), 4))
    Fr, Fs = R[R.code == 3], S[S.code == 3]
    for lo_, hi_ in ((0, 29), (30, 39), (40, 49), (50, 99)):
        a = Fr[(Fr.yl + 17 >= lo_) & (Fr.yl + 17 <= hi_)]
        c = Fs[(Fs.yl + 17 >= lo_) & (Fs.yl + 17 <= hi_)]
        print(f"  FG dist {lo_}-{hi_} make {(a.po>0).mean():.3f} {(c.po>0).mean():.3f} n/g {len(a)/Gr:.2f} {len(c)/Gs:.2f}")
    for nm, D, Gn in (("real", R, Gr), ("sim", S, Gs)):
        t = D[D.po > 0]
        print(f"  {nm} PAT/2pt pts/g {((t.po - 6).clip(lower=0)).sum()/Gn:.3f} def pts/g {D.pdf.sum()/Gn:.3f} FG pts/g {D[D.code==3].po.sum()/Gn:.3f} TD6/g {((D.po >= 6) & (D.code != 3)).sum()/Gn:.3f}")

    import mod25e_f2pr as m

    RF = pd.read_parquet(m.OUT / "real_fit.parquet", columns=["down", "dist", "yl", "yards"])
    for d in (3, 2, 1):
        L3 = Ls[Ls.down == d].copy()
        ix = L3.ridx.to_numpy()
        L3["sdist"], L3["syl"], L3["sy"], L3["sdown"] = RF.dist.to_numpy()[ix], RF.yl.to_numpy()[ix], RF.yards.to_numpy()[ix], RF.down.to_numpy()[ix]
        print(f"source rows for sim down {d} run/pass n {len(L3)} source down share same {(L3.sdown == d).mean():.3f}; sim minus source yards {(L3.yards - L3.sy).mean():.3f}")
        for b in range(len(de) + 1):
            a = L3[L3.db == b]
            rr = Lr[(Lr.down == d) & (Lr.db == b)]
            if len(a) < 200:
                continue
            print(f"  dist {rr.dist.min():.0f}-{rr.dist.max():.0f}: sim dist {a.dist.mean():.2f} src dist {a.sdist.mean():.2f} | yl {a.yl.mean():.1f} src {a.syl.mean():.1f} | src yards>=sim dist {(a.sy >= a.dist).mean():.3f} src>=own dist {(a.sy >= a.sdist).mean():.3f} sim conv {a.conv.mean():.3f} real {rr.conv.mean():.3f} | ypp src {a.sy.mean():.2f} sim {a.yards.mean():.2f} real {rr.yards.mean():.2f}")


if __name__ == "__main__":
    main()


def second():
    import mod25e_f2pr as m

    rng = np.random.default_rng(11)
    R, S = drives(load_real()), drives(load_sim())
    RF = pd.read_parquet(m.OUT / "real_fit.parquet", columns=["yards", "yl", "dist", "down", "code"])
    for D in (R, S):
        D["tdp"] = (D.po >= 6) & D.code.isin([0, 1])
    Lr = R[R.code.isin([0, 1]) & (R.down == 1) & (R.dist == 10) & (R.yl > 20)].copy()
    Ls = S[S.code.isin([0, 1]) & (S.down == 1) & (S.dist == 10) & (S.yl > 20)].copy()
    ix = Ls.ridx.to_numpy()
    Ls["sy"], Ls["syl"], Ls["sdist"] = RF.yards.to_numpy()[ix], RF.yl.to_numpy()[ix], RF.dist.to_numpy()[ix]
    print("1st&10 snaps real", len(Lr), "sim", len(Ls), "pass share", Lr.code.mean().round(4), Ls.code.mean().round(4))
    for nm, L in (("real", Lr), ("sim", Ls)):
        y = L.yards
        print(f"  {nm} mean {y.mean():.3f} P<=0 {(y<=0).mean():.3f} 1-4 {((y>=1)&(y<=4)).mean():.3f} 5-9 {((y>=5)&(y<=9)).mean():.3f} 10+ {(y>=10).mean():.3f} 20+ {(y>=20).mean():.4f} sd {y.std():.2f}")
    ur = Lr.groupby("unit").yards.agg(["sum", "count"])
    us = Ls.groupby("unit").yards.agg(["sum", "count"])
    bt = []
    for _ in range(NB):
        a = ur.iloc[rng.integers(0, len(ur), len(ur))]
        b = us.iloc[rng.integers(0, len(us), len(us))]
        bt.append(b["sum"].sum() / b["count"].sum() - a["sum"].sum() / a["count"].sum())
    pt = Ls.yards.mean() - Lr.yards.mean()
    print("  1st&10 mean yards sim minus real", fmt(pt, np.array(bt)))
    print("  source yards mean", Ls.sy.mean().round(3), "source dist mean", Ls.sdist.mean().round(3), "source yl minus sim yl", (Ls.syl - Ls.yl).mean().round(3), "sim minus source yards", (Ls.yards - Ls.sy).mean().round(3))
    qe = np.quantile(Lr.yl, [0.25, 0.5, 0.75])
    for b in range(4):
        a = Lr[np.digitize(Lr.yl, qe) == b]
        c = Ls[np.digitize(Ls.yl, qe) == b]
        print(f"  yl band {b} ({a.yl.min():.0f}-{a.yl.max():.0f}): ypp real {a.yards.mean():.3f} sim {c.yards.mean():.3f} src {c.sy.mean():.3f} | P<=0 real {(a.yards<=0).mean():.3f} sim {(c.yards<=0).mean():.3f} src {(c.sy<=0).mean():.3f} | 20+ real {(a.yards>=20).mean():.4f} sim {(c.yards>=20).mean():.4f} src {(c.sy>=20).mean():.4f} | pass real {a.code.mean():.3f} sim {c.code.mean():.3f}")
    pool = RF[(RF.down == 1) & (RF.dist == 10) & (RF.yl > 20) & RF.code.isin([0, 1])]
    print("  pool-wide 1st&10 source rows n", len(pool), "mean yards", pool.yards.mean().round(3), "share of source rows drawn by sim is mean src", Ls.sy.mean().round(3))
    srcr = RF.iloc[Ls.ridx.to_numpy()]
    print("  drawn code mix pass", srcr.code.mean().round(4), "vs real pass", Lr.code.mean().round(4))
    Sg = S[S.code.isin([0, 1])]
    Rg = R[R.code.isin([0, 1])]
    print("yards by code overall ypp real", Rg.groupby("code").yards.mean().round(3).to_dict(), "sim", Sg.groupby("code").yards.mean().round(3).to_dict())
    print("sack-like negative share all downs real", (Rg.yards < 0).mean().round(4), "sim", (Sg.yards < 0).mean().round(4))


if __name__ == "__main__" and len(sys.argv) > 1:
    second()
