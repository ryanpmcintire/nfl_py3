import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_draw as dr  # noqa: E402
import mod25e_pts as P  # noqa: E402
import sim04_engine as sim  # noqa: E402

OUT = P.ART / "mix.log"
NB = P.NB
NSNAP = 20000
FEATS = ["sd", "gsr", "yl", "oto", "dto"]


def pp(v):
    return float((np.asarray(v) > 0).mean())


def fmt(pt, bt):
    lo, hi = np.quantile(bt, [0.025, 0.975])
    return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}] pp {pp(bt):.2f}"


def boot(parts, fn, rng):
    pt = fn(*[p.sum() for p in parts])
    bt = []
    for _ in range(NB):
        sums = []
        for p in parts:
            ix = rng.integers(0, len(p), len(p))
            sums.append(p.iloc[ix].sum())
        bt.append(fn(*sums))
    return pt, np.array(bt)


def crossfit_weights(Lr, Ls, feats):
    ur = sorted(Lr.unit.unique())
    us = sorted(Ls.unit.unique())
    fr = Lr.unit.map({u: i % 2 for i, u in enumerate(ur)}).to_numpy()
    fs = Ls.unit.map({u: i % 2 for i, u in enumerate(us)}).to_numpy()
    w = np.zeros(len(Lr))
    Xr = Lr[feats].to_numpy()
    Xs = Ls[feats].to_numpy()
    for f in (0, 1):
        tr_r, tr_s = Xr[fr != f], Xs[fs != f]
        X = np.vstack([tr_r, tr_s])
        y = np.r_[np.zeros(len(tr_r)), np.ones(len(tr_s))]
        clf = HistGradientBoostingClassifier(random_state=7).fit(X, y)
        p = clf.predict_proba(Xr[fr == f])[:, 1]
        w[fr == f] = p / (1 - p) * len(tr_r) / len(tr_s)
    return w / w.mean()


def cells(L):
    return (np.sign(L.sd).astype(int).astype(str) + "_" + L.qtr.astype(int).astype(str)).to_numpy()


def cell_weights(Lr, Ls):
    cr = pd.Series(cells(Lr)).value_counts(normalize=True)
    cs = pd.Series(cells(Ls)).value_counts(normalize=True)
    ratio = (cs / cr).fillna(0.0)
    return pd.Series(cells(Lr)).map(ratio).to_numpy()


def neighbors_state(G, down, phase, dist, fp, sc, tm, oto, dto, k):
    t = G["tables"]
    entry = t["nn_trees_cond"].get((down, phase)) or t["nn_trees_cond"][(down, 0)]
    tree, sub = entry
    feat = sim.feature_matrix(np.array([dist]), np.array([fp]), np.array([sc]), np.array([tm]), np.array([oto]), np.array([dto]), np.array([phase]))[0]
    _, ind = tree.query(feat, k=min(k, len(sub)))
    nb = sub[np.atleast_1d(ind)]
    key = G["ns"]["round_state_key"](down, phase, dist, fp, sc, tm, oto, dto)
    return nb, key


def chain_expect(G, L, a, yds, rng):
    ns = G["ns"]
    ix = rng.choice(len(L), size=min(NSNAP, len(L)), replace=False)
    ph = sim.vectorized_phase(L.qtr.to_numpy(), L.gsr.to_numpy())
    tm = sim.vectorized_time_raw(L.qtr.to_numpy(), L.gsr.to_numpy())
    code = a["play_type_code"]
    rows = []
    for i in ix:
        r = L.iloc[i]
        nb, key = neighbors_state(G, int(r.down), int(ph[i]), r.dist, r.yl, r.sd, tm[i], r.oto, r.dto, sim.K_STATE)
        w = ns["IPW"][nb] * np.exp(-0.5 * ((ns["DISTR"][nb] - key[2] * ns["DKR"]) / (ns["DKH"] * max(1.0, key[2] * ns["DKR"]) ** 0.5)) ** 2)
        rp = np.isin(code[nb], (0, 1))
        wr = w * rp
        nc, _ = neighbors_state(G, int(r.down), int(ph[i]), key[2] * sim.ROUND_DIST, key[3] * sim.ROUND_FP, key[4] * sim.ROUND_SCORE, key[5] * sim.ROUND_TIME, r.oto, r.dto, sim.K_STATE)
        wc = np.isin(code[nc], (0, 1))
        rows.append((L.index[i], yds[nb].mean(), (w * yds[nb]).sum() / w.sum(), (wr * yds[nb]).sum() / wr.sum(), rp.mean(), (wr * a["fp_raw"][nb]).sum() / wr.sum() - r.yl, yds[nc][wc].mean()))
    return pd.DataFrame(rows, columns=["ix", "eu", "ec", "ecr", "rpshare", "efp", "ecell"]).set_index("ix")


def slice_report(name, Lr, Ls, G, a, yds, RF, rng, lines):
    Ls = Ls.copy()
    ix = Ls.ridx.to_numpy()
    Ls["sy"] = RF.yards.to_numpy()[ix]
    Ls["scode"] = RF.code.to_numpy()[ix]
    s_row = a["off_row"] + a["def_row"]
    Ls["ss"] = s_row[ix]
    Ls["syl"] = RF.yl.to_numpy()[ix]
    Lr = Lr.copy()
    Lr["ss"] = s_row[Lr.ridx.to_numpy()]
    lines.append(f"== {name}: real n {len(Lr)} sim n {len(Ls)}; row alignment yards equal {(yds[ix] == Ls.sy.to_numpy()).mean():.4f}; sim code equals source code {(Ls.code.to_numpy() == Ls.scode.to_numpy()).mean():.4f}")
    feats = FEATS + (["dist"] if name != "1st&10" else [])
    Lr["wc"] = crossfit_weights(Lr, Ls, feats)
    Lr["wk"] = cell_weights(Lr, Ls)
    Lr["wk"] = Lr.wk / Lr.wk.mean()
    for nm, w in (("clf", Lr.wc), ("sign x qtr cells", Lr.wk)):
        lines.append(f"  weights {nm}: ess share {(w.sum() ** 2 / (w ** 2).sum()) / len(w):.3f}")
    for c in ("sd", "gsr", "yl", "oto", "dto"):
        lines.append(f"  mean {c}: real {Lr[c].mean():.3f} real-reweighted(clf) {np.average(Lr[c], weights=Lr.wc):.3f} sim {Ls[c].mean():.3f}")
    lines.append(f"  share trailing real {(Lr.sd < 0).mean():.3f} sim {(Ls.sd < 0).mean():.3f}; leading real {(Lr.sd > 0).mean():.3f} sim {(Ls.sd > 0).mean():.3f}")
    Lr["one"] = 1.0
    Ls["one"] = 1.0
    for w_ in ("wc", "wk"):
        Lr["wy_" + w_] = Lr[w_] * Lr.yards
    pr = Lr.groupby("unit")[["yards", "one", "wy_wc", "wc", "wy_wk", "wk"]].sum()
    ps = Ls.groupby("unit")[["yards", "one", "sy", "ss"]].sum()
    lines.append("  (a) state mix, mean yards per play:")
    for tag, fn, parts in (
        ("sim minus real", lambda r, s: s.yards / s.one - r.yards / r.one, (pr, ps)),
        ("mix effect real_rw(clf) minus real", lambda r: r.wy_wc / r.wc - r.yards / r.one, (pr,)),
        ("residual sim minus real_rw(clf)", lambda r, s: s.yards / s.one - r.wy_wc / r.wc, (pr, ps)),
        ("mix effect real_rw(cells) minus real", lambda r: r.wy_wk / r.wk - r.yards / r.one, (pr,)),
        ("residual sim minus real_rw(cells)", lambda r, s: s.yards / s.one - r.wy_wk / r.wk, (pr, ps)),
    ):
        pt, bt = boot(parts, fn, rng)
        lines.append(f"    {tag}: {fmt(pt, bt)}")
    slope = np.polyfit(Lr.ss, Lr.yards, 1)[0]
    pt, bt = boot((pr.assign(ss=Lr.groupby("unit").ss.sum()), ps), lambda r, s: s.ss / s.one - r.ss / r.one, rng)
    lines.append(f"  strength s=off+def epa/play of source rows: sim minus real mean {fmt(pt, bt)}; yards slope on s in real {slope:.2f} -> {slope * pt:+.3f} yd/play")
    lines.append("  (c) post-draw: logged minus source-row yards:")
    pt, bt = boot((ps,), lambda s: (s.yards - s.sy) / s.one, rng)
    lines.append(f"    all snaps {fmt(pt, bt)}; changed share {(Ls.yards != Ls.sy).mean():.4f}; pstop share {Ls.pstop.mean():.4f}")
    for v in (0, 1):
        q = Ls[Ls.pstop == v]
        lines.append(f"    pstop {v} n {len(q)} logged-source {(q.yards - q.sy).mean():+.4f} changed {(q.yards != q.sy).mean():.4f}")
    lines.append(f"    source-row yards mean {Ls.sy.mean():.3f}; real mean {Lr.yards.mean():.3f}; source minus real {Ls.sy.mean() - Lr.yards.mean():+.3f}")
    ce = chain_expect(G, Ls, a, yds, rng)
    Ls2 = Ls.loc[ce.index].join(ce)
    Ls2["one"] = 1.0
    for c in ("eu", "ec", "ecr"):
        Ls2["d_" + c] = Ls2[c] - Ls2.sy
    p2 = Ls2.groupby("unit")[["one", "sy", "yards", "eu", "ec", "ecr", "d_eu", "d_ec", "d_ecr"]].sum()
    lines.append(f"  (b) draw faithfulness at sim states (n {len(Ls2)}), expected minus realized source yards; nonrunpass neighbour share {1 - Ls2.rpshare.mean():.4f}:")
    for c, lab in (("d_eu", "uniform kNN K200"), ("d_ec", "kNN x IPW x dist kernel (all codes)"), ("d_ecr", "same, run/pass neighbours only")):
        pt, bt = boot((p2,), lambda s, c=c: s[c] / s.one, rng)
        lines.append(f"    {lab}: {fmt(pt, bt)}")
    lines.append(f"    source yl minus sim yl: realized {(Ls2.syl - Ls2.yl).mean():+.3f} chain expected {Ls2.efp.mean():+.3f}; cell-centre run/pass uniform expected yards {Ls2.ecell.mean():.3f} vs exact-state uniform run/pass chain {Ls2.ecr.mean():.3f}")
    qe = np.quantile(Lr.yl, [0.25, 0.5, 0.75])
    for b in range(4):
        q = Ls2[np.digitize(Ls2.yl, qe) == b]
        lines.append(f"    yl band {b}: chain run/pass {q.ecr.mean():.3f} cell-centre {q.ecell.mean():.3f} realized source {q.sy.mean():.3f} n {len(q)}")
    lines.append(f"    chain expected {Ls2.ec.mean():.3f} (all codes) {Ls2.ecr.mean():.3f} (run/pass) vs sim realized source {Ls2.sy.mean():.3f}, real plain {Lr.yards.mean():.3f}, real_rw {np.average(Lr.yards, weights=Lr.wc):.3f}")


def td_bins(R, S, rng, lines):
    for D in (R, S):
        D["tdp"] = ((D.po >= 6) & (D.code != 3)).astype(float)

    def dframe(D):
        g = D.groupby("d")
        f = g.first()
        f["td"] = g.tdp.max()
        f["one"] = 1.0
        return f

    fr, fs = dframe(R), dframe(S)
    qe = np.quantile(fr.yl, [0.25, 0.5, 0.75])
    lines.append(f"== TD per drive by start yl quartile (edges {qe}); real drives {len(fr)} sim {len(fs)}")
    for b in range(4):
        a = fr[np.digitize(fr.yl, qe) == b]
        c = fs[np.digitize(fs.yl, qe) == b]
        pa = a.groupby("unit")[["td", "one"]].sum()
        pc = c.groupby("unit")[["td", "one"]].sum()
        pt, bt = boot((pa, pc), lambda r, s: s.td / s.one - r.td / r.one, rng)
        lines.append(f"  bin {b}: real {a.td.mean():.4f} sim {c.td.mean():.4f} diff {fmt(pt, bt)} share real {len(a) / len(fr):.3f} sim {len(c) / len(fs):.3f}")
    pa = fr.groupby("unit")[["td", "one"]].sum()
    pc = fs.groupby("unit")[["td", "one"]].sum()
    pt, bt = boot((pa, pc), lambda r, s: s.td / s.one - r.td / r.one, rng)
    lines.append(f"  all: real {fr.td.mean():.4f} sim {fs.td.mean():.4f} diff {fmt(pt, bt)}")
    rate = fr.groupby(np.digitize(fr.yl, qe)).td.mean()
    mixs = np.bincount(np.digitize(fs.yl, qe), minlength=4) / len(fs)
    lines.append(f"  start-mix effect on TD rate (real rates x sim start shares minus real) {(rate.to_numpy() * mixs).sum() - fr.td.mean():+.4f}")


def main():
    rng = np.random.default_rng(11)
    G, _ = dr.build()
    a = G["tables"]["arrays"]
    yds = a["yards_gained"].astype(float)
    import mod25e_f2pr as m

    RF = pd.read_parquet(m.OUT / "real_fit.parquet", columns=["yards", "yl", "dist", "down", "code"])
    lines = [f"fp_raw equals real_fit yl {(a['fp_raw'] == RF.yl.to_numpy()).mean():.4f}; yards_gained equals real_fit yards {(yds == RF.yards.to_numpy()).mean():.4f}"]
    R, S = P.drives(P.load_real()), P.drives(P.load_sim())
    for name, f in (
        ("1st&10", lambda D: D.code.isin([0, 1]) & (D.down == 1) & (D.dist == 10) & (D.yl > 20)),
        ("2nd", lambda D: D.code.isin([0, 1]) & (D.down == 2) & (D.yl > 20)),
    ):
        slice_report(name, R[f(R)], S[f(S)], G, a, yds, RF, rng, lines)
    td_bins(R, S, rng, lines)
    txt = "\n".join(lines)
    OUT.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
