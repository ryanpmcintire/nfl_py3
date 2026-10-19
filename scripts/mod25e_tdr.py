import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from mod25e_ldr import NCLS, POOL, classes  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / "tdr"
FIT = OUTD / "fit.json"
PHASES = (0, 1, 2)
DEPTH = {"d2": [8.5], "d3": [8.5, 16.5]}
PERIOD = ("p0", "p1", "p2")
DOWN = ("n0", "n1")
LAMBDAS = (0.0, 1.0, 5.0, 25.0)
COUNT = {"calls": 0, "tries": 0, "forced": 0}


def enabled():
    return os.environ.get("TDR") == "1"


def cell_id(score, phase, time_raw, down, scheme):
    dname, pname, nname = scheme
    deficit = -np.asarray(score, float)
    de = DEPTH[dname]
    db = np.digitize(deficit, de)
    nd = len(de) + 1
    phase = np.asarray(phase)
    time_raw = np.asarray(time_raw, float)
    if pname == "p0":
        pb, npb = np.zeros(len(deficit), int), 1
    elif pname == "p1":
        pb, npb = (phase == 2).astype(int), 2
    else:
        pb, npb = np.where(phase == 2, 2, (time_raw <= 900.0).astype(int)), 3
    down = np.asarray(down)
    if nname == "n0":
        nb, nnb = np.zeros(len(deficit), int), 1
    else:
        nb, nnb = np.where(down <= 2, 0, np.where(down == 3, 1, 2)), 3
    return (db * npb + pb) * nnb + nb, nd * npb * nnb


class Frame:
    def __init__(self):
        import sim04_engine as se

        self.se = se
        pbp = se.load_reg_seasons(POOL)
        self.ratings = se.load_team_ratings()
        t = se.build_transition_frame(pbp, team_ratings=self.ratings).reset_index(drop=True)
        t = t[t["phase"].isin(PHASES)].reset_index(drop=True)
        self.t = t
        self.season = t["game_id"].str.slice(0, 4).astype(int).to_numpy()
        self.down = t["down_i"].to_numpy().astype(int)
        self.phase = t["phase"].to_numpy().astype(int)
        self.cls = classes(t["play_type_code"].to_numpy(), t["yards_gained"].to_numpy(), t["dist_raw"].to_numpy(), t["auto_first"].to_numpy())
        self.feats = se.feature_matrix(t["dist_raw"].to_numpy(), t["fp_raw"].to_numpy(), t["sc_raw"].to_numpy(), t["time_raw"].to_numpy(), t["off_to_raw"].to_numpy(), t["def_to_raw"].to_numpy(), t["phase"].to_numpy())
        self.off = t["off_row"].to_numpy()
        self.dfn = t["def_row"].to_numpy()
        self.home = t["is_home_off"].to_numpy()
        self.sd = t["sc_raw"].to_numpy(float)
        self.hs = t["time_raw"].to_numpy(float)
        self.game = t["game_id"].to_numpy()
        self.trail = self.sd <= -1
        gid_season = self.ratings["game_id"].str.slice(0, 4).astype(int).to_numpy()
        self.gid_season2 = np.concatenate([gid_season, gid_season])
        self.off_spread = np.concatenate([self.ratings["home_off_epa_per_play"].to_numpy(), self.ratings["away_off_epa_per_play"].to_numpy()])
        self.cache = {}

    def kernel_h(self, excl):
        keep = ~np.isin(self.gid_season2, list(excl))
        return self.se.TEAM_KERNEL_H_SCALE * float(np.nanstd(self.off_spread[keep]))

    def pmf_for(self, u, excl):
        key = (u, frozenset(excl))
        if key in self.cache:
            return self.cache[key]
        se = self.se
        h = self.kernel_h(excl)
        qmask = (self.season == u) & self.trail
        out = np.full((len(self.sd), NCLS), np.nan)
        for dn in (1, 2, 3, 4):
            for ph in PHASES:
                tr = np.flatnonzero(~np.isin(self.season, list(excl)) & (self.down == dn) & (self.phase == ph))
                te = np.flatnonzero(qmask & (self.down == dn) & (self.phase == ph))
                if len(te) == 0 or len(tr) == 0:
                    continue
                k = min(se.K_STATE, len(tr))
                _, ix = cKDTree(self.feats[tr]).query(self.feats[te], k=k)
                nb = tr[ix.reshape(len(te), k)]
                d2 = (self.off[nb] - self.off[te][:, None]) ** 2 + (self.dfn[nb] - self.dfn[te][:, None]) ** 2
                w = np.exp(-d2 / (2.0 * h * h))
                w = w * np.where(self.home[nb] == self.home[te][:, None], se.TEAM_KERNEL_LAMBDA, 1.0)
                bad = ~np.isfinite(w.sum(axis=1)) | (w.sum(axis=1) <= 0)
                w[bad] = 1.0
                w = w / w.sum(axis=1, keepdims=True)
                cl = self.cls[nb]
                for c in range(NCLS):
                    out[te, c] = (w * (cl == c)).sum(axis=1)
        idx = np.flatnonzero(qmask)
        res = (idx, out[idx])
        self.cache[key] = res
        return res

    def cells(self, ix, scheme):
        return cell_id(self.sd[ix], self.phase[ix], self.hs[ix], self.down[ix], scheme)


def ratios(cell, cls, pmf, ncell, lam):
    real = np.bincount(cell * NCLS + cls, minlength=ncell * NCLS).reshape(ncell, NCLS).astype(float)
    nn = np.stack([np.bincount(cell, weights=pmf[:, c], minlength=ncell) for c in range(NCLS)], axis=1)
    return (real + lam) / (nn + lam + 1e-12)


def adjust(r, cell, pmf):
    adj = pmf * r[cell]
    return adj / adj.sum(axis=1, keepdims=True)


def ll_of(adj, cls):
    return float(np.log(np.maximum(adj[np.arange(len(cls)), cls], 1e-9)).sum())


def fit():
    F = Frame()
    schemes = list(itertools.product(DEPTH, PERIOD, DOWN))
    combos = [(sc, lam) for sc in schemes for lam in LAMBDAS]
    folds = {}
    keep = {}
    stab = {}
    for s in POOL:
        rest = [u for u in POOL if u != s]
        pm = {u: F.pmf_for(u, {s, u}) for u in rest}
        inner = {}
        for sc, lam in combos:
            ll = 0.0
            n = 0
            cells = {u: F.cells(pm[u][0], sc) for u in rest}
            nc = cells[rest[0]][1]
            for t in rest:
                tr = [u for u in rest if u != t]
                c = np.concatenate([cells[u][0] for u in tr])
                cl = np.concatenate([F.cls[pm[u][0]] for u in tr])
                p = np.vstack([pm[u][1] for u in tr])
                r = ratios(c, cl, p, nc, lam)
                ll += ll_of(adjust(r, cells[t][0], pm[t][1]), F.cls[pm[t][0]])
                n += len(cells[t][0])
            inner[(sc, lam)] = ll / n
        sc, lam = max(inner, key=inner.get)
        c0 = [F.cells(pm[u][0], sc) for u in rest]
        nc = c0[0][1]
        c = np.concatenate([x[0] for x in c0])
        cl = np.concatenate([F.cls[pm[u][0]] for u in rest])
        p = np.vstack([pm[u][1] for u in rest])
        r = ratios(c, cl, p, nc, lam)
        a_in = ll_of(adjust(r, c, p), cl) / len(cl)
        b_in = ll_of(p, cl) / len(cl)
        te_ix, te_pm = F.pmf_for(s, {s})
        te_c = F.cells(te_ix, sc)[0]
        te_cls = F.cls[te_ix]
        te_adj = adjust(r, te_c, te_pm)
        a_out = ll_of(te_adj, te_cls) / len(te_ix)
        b_out = ll_of(te_pm, te_cls) / len(te_ix)
        keep[str(s)] = dict(ix=te_ix, pmf=te_pm, adj=te_adj, cls=te_cls, cell=te_c)
        stab[s] = ((np.bincount(cl, minlength=NCLS) + lam) / (p.sum(axis=0) + lam)).tolist()
        folds[s] = dict(choice="|".join(sc) + f"|{lam:g}", n_te=int(len(te_ix)), out_adj=a_out, out_nn=b_out, in_adj=a_in, in_nn=b_in)
        print(s, folds[s], flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez(OUTD / "heldout.npz", **{f"{k}_{n}": v for k, d in keep.items() for n, v in d.items()})
    n_tot = sum(f["n_te"] for f in folds.values())
    pooled = {k: sum(f[k] * f["n_te"] for f in folds.values()) / n_tot for k in ("out_adj", "out_nn")}
    pooled["in_adj"] = float(np.mean([f["in_adj"] for f in folds.values()]))
    pooled["in_nn"] = float(np.mean([f["in_nn"] for f in folds.values()]))
    deltas = [f["out_adj"] - f["out_nn"] for f in folds.values()]
    choices = [f["choice"] for f in folds.values()]
    modal = max(set(choices), key=lambda x: (choices.count(x), x))
    parts = modal.split("|")
    sc, lam = tuple(parts[:3]), float(parts[3])
    allp = {u: F.pmf_for(u, {u}) for u in POOL}
    cc = [F.cells(allp[u][0], sc) for u in POOL]
    nc = cc[0][1]
    r = ratios(np.concatenate([x[0] for x in cc]), np.concatenate([F.cls[allp[u][0]] for u in POOL]), np.vstack([allp[u][1] for u in POOL]), nc, lam)
    arr = np.array(list(stab.values()))
    spec = dict(scheme=list(sc), lam=lam, depth_edges=DEPTH[sc[0]], ratio=r.tolist(), ncls=NCLS, folds=folds, pooled=pooled, fold_deltas=deltas, class_ratio_mean=arr.mean(axis=0).tolist(), class_ratio_min=arr.min(axis=0).tolist(), class_ratio_max=arr.max(axis=0).tolist(), looks=len(combos) * len(POOL))
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    print("modal", modal, "pooled", pooled, "deltas", [round(d, 4) for d in deltas])
    return spec


def install_tdr():
    import mod25d_variance as dv

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    ns = dv._G["ns"]
    a = dv._G["tables"]["arrays"]
    r = np.asarray(spec["ratio"])
    rmax = r.max(axis=1)
    sc = tuple(spec["scheme"])
    rowcls = classes(a["play_type_code"], a["yards_gained"], a["dist_raw"], a["auto_first"])
    base = ns["pick_index_nn_conditioned"]
    tries = int(np.ceil(np.log(1e-3) / np.log(0.5)))

    def pick(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        fr = sys._getframe(1)
        while fr is not None and "qtr" not in fr.f_locals:
            fr = fr.f_back
        qtr = fr.f_locals["qtr"] if fr is not None else None  # noqa: F841
        j = base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        if phase not in PHASES or score > -1:
            return j
        c = int(cell_id(np.array([score]), np.array([phase]), np.array([time_raw]), np.array([down]), sc)[0][0])
        COUNT["calls"] += 1
        for _ in range(tries):
            COUNT["tries"] += 1
            if rng.random() * rmax[c] <= r[c, rowcls[j]]:
                return j
            j = base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        COUNT["forced"] += 1
        return j

    ns["pick_index_nn_conditioned"] = pick


if __name__ == "__main__":
    fit()
