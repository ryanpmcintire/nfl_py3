import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CLK_LABEL", "crHpqokgndecsm")
os.environ.setdefault("CLK_SEEDS", "11")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.argv = [sys.argv[0], "100"]
h2 = types.ModuleType("h2lib")
h2.__file__ = str(REPO / "scripts" / "mod25e_half2.py")
exec((REPO / "scripts" / "mod25e_half2.py").read_text().rsplit("\nmain()", 1)[0], h2.__dict__)
c2 = h2.c2
NB = 100


def sim_full():
    pool = pd.read_parquet(c2.ART.parent / "sim09" / "u4g" / "pool.parquet", columns=["yl"])
    pyl = pool.yl.to_numpy()
    use = ["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "offhome", "yards", "idx"]
    out, off = [], 0
    for sd in c2.SEEDS:
        for f in sorted((c2.ART / f"e5_{c2.LABEL}_s{sd}").glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < c2.BURN:
                continue
            d = pd.read_parquet(f, columns=use)
            d = d[d.qtr <= 4].sort_values("g", kind="stable").reset_index(drop=True)
            g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
            sc = ((d.po > 0) | (d.pdf > 0)).to_numpy()
            new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | sc[:-1]]
            d["dd"] = np.cumsum(new) + off
            off += int(new.sum()) + 1
            d = d[d.code.isin([0, 1, 2, 3, 4, 5])].copy()
            d["gk"] = f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str)
            d["pyl"] = pyl[d.idx.astype(int).to_numpy()]
            out.append(d)
    d = pd.concat(out, ignore_index=True)
    c = d.code.astype(int).to_numpy()
    rp = c <= 1
    dd = d.dd.to_numpy()
    same = np.r_[dd[1:] == dd[:-1], False]
    nyl = np.r_[d.yl.to_numpy()[1:], np.nan]
    cont = same & (d.po.to_numpy() == 0) & (d.pdf.to_numpy() == 0) & (d.flip.to_numpy() == 0)
    return pd.DataFrame({"gk": d.gk.to_numpy(), "d": dd, "down": d.down.to_numpy(), "yl": d.yl.to_numpy(), "kind": c, "td": (rp & (d.po.to_numpy() >= 6)).astype(int),
                         "fgm": ((c == 3) & (d.po.to_numpy() == 3)).astype(int), "yd": d.yards.to_numpy(), "nyl": np.where(cont, nyl, np.nan), "pyl": d.pyl.to_numpy()})


def real_full(R):
    dd, gk = R.d.to_numpy(), R.gk.to_numpy()
    same = np.r_[(dd[1:] == dd[:-1]) & (gk[1:] == gk[:-1]), False]
    nyl = np.r_[R.yl.to_numpy()[1:], np.nan]
    cont = same & (R.td.to_numpy() == 0) & (R.to.to_numpy() == 0) & (R.fgm.to_numpy() == 0) & (R.kind.to_numpy() <= 1)
    out = R.copy()
    out["nyl"] = np.where(cont, nyl, np.nan)
    return out


def gsums(D, games, cols):
    k = pd.Series(np.arange(len(games)), index=games).reindex(D.gk).to_numpy()
    return np.column_stack([np.bincount(k, weights=np.asarray(v, dtype=float), minlength=len(games)) for v in cols])


def boot(Dr, Ds, fn, rng):
    rg, sg = np.sort(Dr.gk.unique()), np.sort(Ds.gk.unique())
    Mr, Ms = gsums(Dr, rg, fn(Dr)), gsums(Ds, sg, fn(Ds))

    def est(a, b):
        r, s = a.sum(0), b.sum(0)
        return r[0] / r[1], s[0] / s[1]

    r0, s0 = est(Mr, Ms)
    bs = np.array([np.subtract(*est(Mr[rng.integers(0, len(rg), len(rg))], Ms[rng.integers(0, len(sg), len(sg))])[::-1]) for _ in range(NB)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return r0, s0, s0 - r0, lo, hi


def tripframe(D):
    rz = D[(D.yl <= 20) & (D.kind <= 3)]
    g = rz.groupby("d").agg(gk=("gk", "first"), td=("td", "max"), fg=("fgm", "max")).reset_index(drop=True)
    g["none"] = ((g.td == 0) & (g.fg == 0)).astype(int)
    g["n"] = 1
    return g


def main():
    rng = np.random.default_rng(0)
    R, S0 = h2.real_plays(), h2.sim_plays()
    cuts = np.quantile(R[R.down == 1].yl, [0.25, 0.5, 0.75])
    LR, QR = h2.prep(R, cuts)
    LS, QS = h2.prep(S0, cuts)
    S = sim_full()
    R = real_full(R)
    print(f"label {c2.LABEL} seeds {c2.SEEDS} real plays {len(R)} sim plays {len(S)}")
    st = S[(S.td == 1) & (S.yl <= 10)]
    rt = R[(R.td == 1) & (R.yl <= 10) & (R.kind <= 1)]
    print(f"yards column on TD rows yl<=10: sim mean {st.yd.mean():.2f} share<0 {(st.yd < 0).mean():.3f} n {len(st)}; real mean {rt.yd.mean():.2f} share<0 {(rt.yd < 0).mean():.3f} n {len(rt)}")
    print(f"sim TD rows yl<=10: source-row yl mean {st.pyl.mean():.1f} vs state yl {st.yl.mean():.1f}; source yl>10 share {(st.pyl > 10).mean():.3f}; source yl>state+2 share {(st.pyl > st.yl + 2).mean():.3f}")
    sa = S[(S.kind <= 1) & (S.yl <= 10)]
    print(f"sim all run/pass yl<=10: source yl mean {sa.pyl.mean():.2f} vs state {sa.yl.mean():.2f}")
    ql_r, ql_s = set(QR.index), set(QS.index)
    subs = {"all plays": (R, S), "Q2 last drive": (R[R.d.isin(ql_r)], S[S.d.isin(ql_s)])}
    for nm, (Dr, Ds) in subs.items():
        print(f"== {nm}: real plays {len(Dr)} sim plays {len(Ds)}")
        for zl, zh in ((1, 10), (11, 20), (21, 40)):
            parts = []
            for lab, sel in (("rp", lambda D: D.kind <= 1), ("pass", lambda D: D.kind == 1), ("run", lambda D: D.kind == 0)):
                def fn(D, sel=sel):
                    m = ((D.yl >= zl) & (D.yl <= zh) & sel(D)).to_numpy().astype(float)
                    return [m * D.td.to_numpy(), m]
                r = boot(Dr, Ds, fn, rng)
                parts.append(f"{lab} {r[0]:.3f}/{r[1]:.3f} {r[2]:+.3f} [{r[3]:+.3f},{r[4]:+.3f}]")
            print(f"TD/play real/sim yl {zl}-{zh}: " + "; ".join(parts))
        for lab, kk in (("pass", 1), ("run", 0)):
            def fraw(D):
                m = ((D.yl <= 10) & (D.kind == kk)).to_numpy().astype(float)
                return [m * D.yd.to_numpy(), m]

            def fcont(D):
                m = ((D.yl <= 10) & (D.kind == kk) & D.nyl.notna()).to_numpy().astype(float)
                return [m * np.nan_to_num(D.yl.to_numpy() - D.nyl.to_numpy()), m]

            def ftd(D):
                td = (D.td == 1).to_numpy()
                m = ((D.yl <= 10) & (D.kind == kk) & (D.nyl.notna().to_numpy() | td)).to_numpy().astype(float)
                return [m * np.where(td, D.yl.to_numpy(), np.nan_to_num(D.yl.to_numpy() - D.nyl.to_numpy())), m]
            for tag, fn in (("raw column", fraw), ("continuing rows next-yl diff", fcont), ("continuing + TD at goal line", ftd)):
                r = boot(Dr, Ds, fn, rng)
                print(f"yd/{lab} yl<=10 {tag}: real {r[0]:+.2f} sim {r[1]:+.2f} diff {r[2]:+.2f} [{r[3]:+.2f},{r[4]:+.2f}]")
        rg, sg = np.sort(Dr.gk.unique()), np.sort(Ds.gk.unique())
        tr, ts = tripframe(Dr), tripframe(Ds)
        Mr, Ms = gsums(tr, rg, [tr.td, tr.fg, tr.none, tr.n]), gsums(ts, sg, [ts.td, ts.fg, ts.none, ts.n])

        def sh(M):
            t = M.sum(0)
            return t[:3] / t[3]
        e0 = sh(Ms) - sh(Mr)
        bs = np.array([sh(Ms[rng.integers(0, len(sg), len(sg))]) - sh(Mr[rng.integers(0, len(rg), len(rg))]) for _ in range(NB)])
        lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
        print(f"red-zone trips (drives with run/pass/kick at yl<=20) real {len(tr)} sim {len(ts)}")
        for j, o in enumerate(("TD", "FG", "none")):
            print(f"  {o}: real {sh(Mr)[j]:.3f} sim {sh(Ms)[j]:.3f} diff {e0[j]:+.3f} [{lo[j]:+.3f},{hi[j]:+.3f}]")


main()
