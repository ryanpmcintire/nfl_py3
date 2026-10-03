import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod25e_late as late  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "artifacts" / "mod25e3" / "crH"
SIMD = BASE / "play6"
OUT = BASE / "xq2"
NB = 300


def lib():
    ns = late.load_xq()
    ns["SIM"] = SIMD
    ns["OUT"] = OUT
    return ns


def cmd_decomp(a):
    OUT.mkdir(parents=True, exist_ok=True)
    ns = lib()
    sys.argv = ["x", "run"]
    ns["main"]()
    ns["diag"]()
    ns["common"]()


def cmd_late(a):
    late.SIMD = SIMD
    late.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    late.cmd_an(a)
    late.cmd_fourth(a)


def cmd_late2(a):
    late.SIMD = SIMD
    late.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    late.cmd_state(a)
    late.cmd_clock(a)
    late.cmd_fgdown(a)


def adj_incr(ns, D):
    D, _ = ns["components"](D)
    D = D.assign(a=D.p - D.c_strength)
    gk = pd.Index(sorted(D.gk.unique()))
    gi = gk.get_indexer(D.gk)
    qi = np.clip(D.ql.astype(int).to_numpy(), 1, 4) - 1
    I = np.zeros((len(gk), 4))
    np.add.at(I, (gi, qi), D.s.to_numpy() * D.a.to_numpy())
    return D, I, gi, qi


def pairs(I):
    C = np.cov(I, rowvar=False)
    return {(q, r): C[q, r] for q in range(4) for r in range(q + 1, 4)}


def cmd_matrix(a):
    OUT.mkdir(parents=True, exist_ok=True)
    ns = lib()
    srcs = {"real_pool": ns["real_drives"](ns["POOL"])[0], "real_late": ns["real_drives"](ns["LATE"])[0], "crH": ns["sim_drives"]()}
    rng = np.random.default_rng(7)
    res = {}
    L = []
    for nm, D in srcs.items():
        D, I, gi, qi = adj_incr(ns, D)
        srcs[nm] = D
        C = np.cov(I, rowvar=False)
        G = len(I)
        B = []
        for _ in range(NB):
            Cb = np.cov(I[rng.integers(0, G, G)], rowvar=False)
            B.append([Cb[q, r] for q in range(4) for r in range(q + 1, 4)])
        res[nm] = (C, np.array(B))
        L.append(f"{nm} games {G} var by q " + " ".join(f"{C[q, q]:.1f}" for q in range(4)) + " sum offdiag x2 " + f"{2 * sum(C[q, r] for q in range(4) for r in range(q + 1, 4)):+.2f}")
        L.append("  pair cov " + " ".join(f"{q + 1}{r + 1}:{C[q, r]:+.2f}" for q in range(4) for r in range(q + 1, 4)))
    names = [f"{q + 1}{r + 1}" for q in range(4) for r in range(q + 1, 4)]
    for ref in ("real_pool", "real_late"):
        L.append(f"crH minus {ref} (2x cov pairs; [5,95] boot; probability_positive)")
        d = 2 * (res["crH"][1] - res[ref][1])
        pt = 2 * np.array([res["crH"][0][q, r] - res[ref][0][q, r] for q in range(4) for r in range(q + 1, 4)])
        for i, n in enumerate(names):
            L.append(f"  {n} {pt[i]:+.2f} [{np.percentile(d[:, i], 5):+.2f},{np.percentile(d[:, i], 95):+.2f}] pp {(d[:, i] > 0).mean():.2f}")
        L.append(f"  total {pt.sum():+.2f} [{np.percentile(d.sum(1), 5):+.2f},{np.percentile(d.sum(1), 95):+.2f}] pp {(d.sum(1) > 0).mean():.2f}")
    (OUT / "matrix.txt").write_text("\n".join(L))
    print("\n".join(L))


def cell_rows(D, I, gi, qi, q, r):
    G = len(I)
    home = I[:, q]
    x = D.s.to_numpy() * home[gi]
    x = x - 0.0
    m = qi == r
    d = D[m].copy()
    d["x"] = x[m]
    d["ab"] = d.a - d.a.mean()
    med = d.x.abs().median()
    d["cell"] = np.select([d.x > med, d.x > 0, d.x > -med], ["gained big", "gained small", "lost small"], "lost big")
    d["out"] = np.select([d.p >= 6, d.p == 3, d.p < 0, d.pt == "punt"], ["td", "fg", "defscore", "punt"], "other")
    rows = []
    for cn in ("gained big", "gained small", "lost small", "lost big"):
        c = d[d.cell == cn]
        rows.append(f"    {cn:12s} drives/g {len(c) / G:.3f} mean adj pts {c.a.mean():+.3f} contrib {float((c.x * c.ab).sum() / G):+.2f} sd0 {c.sd0.mean():+.2f} yl0 {c.yl0.mean():.1f} secs {c.secs.mean():.0f} plays {c.n.mean():.2f} td {(c.out == 'td').mean():.3f} fg {(c.out == 'fg').mean():.3f} defsc {(c.out == 'defscore').mean():.3f} punt {(c.out == 'punt').mean():.3f} other {(c.out == 'other').mean():.3f}")
    rows.append(f"    total contrib {float((d.x * d.ab).sum() / G):+.2f} (|x| median {med:.2f})")
    return rows


def cmd_anatomy(a):
    OUT.mkdir(parents=True, exist_ok=True)
    ns = lib()
    q, r = a.q - 1, a.r - 1
    srcs = {"real_pool": ns["real_drives"](ns["POOL"])[0], "real_late": ns["real_drives"](ns["LATE"])[0], "crH": ns["sim_drives"]()}
    L = [f"anatomy of Q{q + 1} increment (offence perspective x) vs offence drives ending in Q{r + 1}"]
    for nm, D in srcs.items():
        D, I, gi, qi = adj_incr(ns, D)
        L.append(f"  {nm}")
        L.extend(cell_rows(D, I, gi, qi, q, r))
        Dr = D[qi == r]
        lead = np.select([Dr.sd0 > 0, Dr.sd0 < 0], ["lead", "trail"], "tied")
        for st in ("lead", "tied", "trail"):
            c = Dr[lead == st]
            L.append(f"    state {st:5s} drives/g {len(c) / len(I):.3f} mean adj {c.a.mean():+.3f} yl0 {c.yl0.mean():.1f} secs {c.secs.mean():.0f}")
    (OUT / f"anatomy_{a.q}{a.r}.txt").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("decomp", "late", "late2", "matrix"):
        sp.add_parser(n)
    s = sp.add_parser("anatomy")
    s.add_argument("--q", type=int, required=True)
    s.add_argument("--r", type=int, required=True)
    a = ap.parse_args()
    {"decomp": cmd_decomp, "late": cmd_late, "late2": cmd_late2, "matrix": cmd_matrix, "anatomy": cmd_anatomy}[a.cmd](a)


if __name__ == "__main__":
    main()
