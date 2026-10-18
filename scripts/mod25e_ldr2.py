import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_ldr as ld

OUTD = REPO / "artifacts" / "mod25e3" / "ldr2"
FIT = OUTD / "fit.json"
DOWN_SCHEMES = {"none": [], "d12_34": [2.5], "d1_2_34": [1.5, 2.5], "d12_3_4": [2.5, 3.5]}
LAMBDAS = (1.0, 5.0, 25.0)
BASELINE = ("seven", "none", 1.0)


def enabled():
    return os.environ.get("LDR2") == "1"


def cell_id(lead, gsr, down, edges, dedges):
    base = ld.cell_id(lead, gsr, edges)
    nd = len(dedges) + 1
    return base * nd + np.digitize(down, dedges)


def ncell(edges, dedges):
    return 4 * (len(edges) + 1) * (len(dedges) + 1)


def fit():
    F = ld.Frame()
    combos = [(n, dn, lam) for n in ld.TIME_SCHEMES for dn in DOWN_SCHEMES for lam in LAMBDAS]
    folds = {}
    ids = lambda ix, n, dn: cell_id(F.sd[ix], F.gsr[ix], F.down[ix], ld.TIME_SCHEMES[n], DOWN_SCHEMES[dn])
    for s in ld.POOL:
        rest = [u for u in ld.POOL if u != s]
        inner = {}
        for n, dn, lam in combos:
            nc = ncell(ld.TIME_SCHEMES[n], DOWN_SCHEMES[dn])
            ll = 0.0
            m = 0
            for t in rest:
                r_ix, r_pm = ld.stack(F, [u for u in rest if u != t], lambda u: {s, t, u})
                te_ix, te_pm = F.pmf_for(t, {s, t})
                r = ld.ratios(ids(r_ix, n, dn), F.cls[r_ix], r_pm, nc, lam)
                a, _ = ld.score(r, ids(te_ix, n, dn), F.cls[te_ix], te_pm)
                ll += a
                m += len(te_ix)
            inner[(n, dn, lam)] = ll / m
        tr_ix, tr_pm = ld.stack(F, rest, lambda u: {s, u})
        te_ix, te_pm = F.pmf_for(s, {s})
        res = {}
        for key in (max(inner, key=inner.get), BASELINE):
            n, dn, lam = key
            nc = ncell(ld.TIME_SCHEMES[n], DOWN_SCHEMES[dn])
            r = ld.ratios(ids(tr_ix, n, dn), F.cls[tr_ix], tr_pm, nc, lam)
            a_out, b_out = ld.score(r, ids(te_ix, n, dn), F.cls[te_ix], te_pm)
            a_in, _ = ld.score(r, ids(tr_ix, n, dn), F.cls[tr_ix], tr_pm)
            res[key] = (a_out, b_out, a_in)
        ch = max(inner, key=inner.get)
        a_out, b_out, a_in = res[ch]
        ab_out = res[BASELINE][0]
        l13 = (F.sd[te_ix] >= 1) & (F.sd[te_ix] <= 3) & (F.gsr[te_ix] <= 120)
        n, dn, lam = ch
        nc = ncell(ld.TIME_SCHEMES[n], DOWN_SCHEMES[dn])
        r = ld.ratios(ids(tr_ix, n, dn), F.cls[tr_ix], tr_pm, nc, lam)
        adj = te_pm * r[ids(te_ix, n, dn)]
        adj = adj / adj.sum(axis=1, keepdims=True)
        rb = ld.ratios(ids(tr_ix, *BASELINE[:2]), F.cls[tr_ix], tr_pm, ncell(ld.TIME_SCHEMES[BASELINE[0]], DOWN_SCHEMES[BASELINE[1]]), BASELINE[2])
        adjb = te_pm * rb[ids(te_ix, *BASELINE[:2])]
        adjb = adjb / adjb.sum(axis=1, keepdims=True)
        i = np.arange(len(te_ix))
        c = F.cls[te_ix]
        pas = np.isin(np.arange(ld.NCLS), [2, 3])
        folds[s] = dict(choice=f"{n}|{dn}|{lam:g}", n_te=int(len(te_ix)), out_adj=a_out / len(te_ix), out_base=ab_out / len(te_ix), out_nn=b_out / len(te_ix), in_adj=a_in / len(tr_ix), n13=int(l13.sum()), ll13_adj=float(np.log(np.maximum(adj[i, c], 1e-9))[l13].sum()), ll13_base=float(np.log(np.maximum(adjb[i, c], 1e-9))[l13].sum()), pass13_adj=float(adj[:, 2:4][l13].sum() / max(l13.sum(), 1)), pass13_base=float(adjb[:, 2:4][l13].sum() / max(l13.sum(), 1)), pass13_nn=float(te_pm[:, 2:4][l13].sum() / max(l13.sum(), 1)), pass13_real=float(np.isin(c[l13], [2, 3]).mean()) if l13.any() else None)
        print(s, folds[s], flush=True)
    n_tot = sum(f["n_te"] for f in folds.values())
    d_base = [f["out_adj"] - f["out_base"] for f in folds.values()]
    n13 = sum(f["n13"] for f in folds.values())
    pooled = dict(out_adj=sum(f["out_adj"] * f["n_te"] for f in folds.values()) / n_tot, out_base=sum(f["out_base"] * f["n_te"] for f in folds.values()) / n_tot, ll13_delta_per_query=sum(f["ll13_adj"] - f["ll13_base"] for f in folds.values()) / max(n13, 1), n13=n13)
    choices = [f["choice"] for f in folds.values()]
    modal = max(set(choices), key=lambda c: (choices.count(c), c))
    n, dn, lam = modal.split("|")
    lam = float(lam)
    tr_ix, tr_pm = ld.stack(F, ld.POOL, lambda u: {u})
    nc = ncell(ld.TIME_SCHEMES[n], DOWN_SCHEMES[dn])
    r = ld.ratios(ids(tr_ix, n, dn), F.cls[tr_ix], tr_pm, nc, lam)
    OUTD.mkdir(parents=True, exist_ok=True)
    spec = dict(scheme=n, down_scheme=dn, edges=ld.TIME_SCHEMES[n], down_edges=DOWN_SCHEMES[dn], lam=lam, ratio=r.tolist(), ncls=ld.NCLS, folds=folds, pooled=pooled, fold_deltas_vs_ldr=d_base, looks=len(combos) * len(ld.POOL))
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    print("modal", modal, "pooled", pooled, "deltas", [round(d, 4) for d in d_base])


def install_ldr2():
    import mod25d_variance as dv

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    ns = dv._G["ns"]
    a = dv._G["tables"]["arrays"]
    r = np.asarray(spec["ratio"])
    rmax = r.max(axis=1)
    edges = spec["edges"]
    dedges = spec["down_edges"]
    rowcls = ld.classes(a["play_type_code"], a["yards_gained"], a["dist_raw"], a["auto_first"])
    base = ns["pick_index_nn_conditioned"]
    tries = int(np.ceil(np.log(1e-3) / np.log(0.5)))

    def pick(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        j = base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        if phase != 3 or score < 1:
            return j
        c = int(cell_id(np.array([score]), np.array([time_raw]), np.array([down]), edges, dedges)[0])
        for _ in range(tries):
            if rng.random() * rmax[c] <= r[c, rowcls[j]]:
                return j
            j = base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        return j

    ns["pick_index_nn_conditioned"] = pick


if __name__ == "__main__":
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    fit()
