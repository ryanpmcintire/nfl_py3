import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUTD = REPO / "artifacts" / "mod25e3" / "ldr"
FIT = OUTD / "fit.json"
LEAD_EDGES = [0.5, 3.5, 8.5, 16.5]
TIME_SCHEMES = {
    "two": [120.0],
    "four": [60.0, 120.0, 180.0],
    "seven": [30.0, 60.0, 90.0, 120.0, 180.0, 240.0],
}
LAMBDAS = (0.0, 1.0, 5.0, 25.0)
NCLS = 9
POOL = tuple(range(2009, 2018))
COUNT = {"calls": 0, "tries": 0, "forced": 0}


def enabled():
    return os.environ.get("LDR") == "1"


def classes(code, yards, dist, auto_first):
    code = np.asarray(code).astype(int)
    fd = ((np.asarray(yards) >= np.asarray(dist)) | np.asarray(auto_first).astype(bool)).astype(int)
    return np.where(code <= 1, code * 2 + fd, np.where(code <= 5, code + 2, 8))


def cell_id(lead, gsr, edges):
    lb = np.digitize(lead, LEAD_EDGES) - 1
    tb = np.digitize(gsr, edges)
    return lb * (len(edges) + 1) + tb


class Frame:
    def __init__(self):
        import sim04_engine as se

        self.se = se
        pbp = se.load_reg_seasons(POOL)
        self.ratings = se.load_team_ratings()
        t = se.build_transition_frame(pbp, team_ratings=self.ratings).reset_index(drop=True)
        t = t[t["phase"] == 3].reset_index(drop=True)
        self.t = t
        self.season = t["game_id"].str.slice(0, 4).astype(int).to_numpy()
        self.down = t["down_i"].to_numpy().astype(int)
        self.cls = classes(t["play_type_code"].to_numpy(), t["yards_gained"].to_numpy(), t["dist_raw"].to_numpy(), t["auto_first"].to_numpy())
        self.feats = se.feature_matrix(t["dist_raw"].to_numpy(), t["fp_raw"].to_numpy(), t["sc_raw"].to_numpy(), t["time_raw"].to_numpy(), t["off_to_raw"].to_numpy(), t["def_to_raw"].to_numpy(), t["phase"].to_numpy())
        self.off = t["off_row"].to_numpy()
        self.dfn = t["def_row"].to_numpy()
        self.home = t["is_home_off"].to_numpy()
        self.sd = t["sc_raw"].to_numpy(float)
        self.gsr = t["time_raw"].to_numpy(float)
        self.lead = self.sd >= 1
        gid_season = self.ratings["game_id"].str.slice(0, 4).astype(int).to_numpy()
        self.gid_season2 = np.concatenate([gid_season, gid_season])
        self.off_spread = np.concatenate([self.ratings["home_off_epa_per_play"].to_numpy(), self.ratings["away_off_epa_per_play"].to_numpy()])
        self.cache = {}

    def kernel_h(self, excl):
        keep = ~np.isin(self.gid_season2, list(excl))
        return self.se.TEAM_KERNEL_H_SCALE * float(np.nanstd(self.off_spread[keep]))

    def pmf_for(self, u, excl, weighted=True):
        key = (u, frozenset(excl), weighted)
        if key in self.cache:
            return self.cache[key]
        se = self.se
        h = self.kernel_h(excl)
        qmask = (self.season == u) & self.lead
        out = np.full((len(self.sd), NCLS), np.nan)
        for dn in (1, 2, 3, 4):
            tr = np.flatnonzero(~np.isin(self.season, list(excl)) & (self.down == dn))
            te = np.flatnonzero(qmask & (self.down == dn))
            if len(te) == 0 or len(tr) == 0:
                continue
            k = min(se.K_STATE, len(tr))
            _, ix = cKDTree(self.feats[tr]).query(self.feats[te], k=k)
            nb = tr[ix.reshape(len(te), k)]
            if weighted:
                d2 = (self.off[nb] - self.off[te][:, None]) ** 2 + (self.dfn[nb] - self.dfn[te][:, None]) ** 2
                w = np.exp(-d2 / (2.0 * h * h))
                w = w * np.where(self.home[nb] == self.home[te][:, None], se.TEAM_KERNEL_LAMBDA, 1.0)
                bad = ~np.isfinite(w.sum(axis=1)) | (w.sum(axis=1) <= 0)
                w[bad] = 1.0
            else:
                w = np.ones(nb.shape)
            w = w / w.sum(axis=1, keepdims=True)
            cl = self.cls[nb]
            for c in range(NCLS):
                out[te, c] = (w * (cl == c)).sum(axis=1)
        idx = np.flatnonzero(qmask)
        res = (idx, out[idx])
        self.cache[key] = res
        return res


def ratios(cell, cls, pmf, ncell, lam):
    real = np.zeros((ncell, NCLS))
    nn = np.zeros((ncell, NCLS))
    np.add.at(real, (cell, cls), 1.0)
    np.add.at(nn, cell, pmf)
    return (real + lam) / (nn + lam + 1e-12)


def score(r, cell, cls, pmf):
    adj = pmf * r[cell]
    adj = adj / adj.sum(axis=1, keepdims=True)
    i = np.arange(len(cls))
    return float(np.log(np.maximum(adj[i, cls], 1e-9)).sum()), float(np.log(np.maximum(pmf[i, cls], 1e-9)).sum())


def stack(F, seasons, excl_fn):
    ix, pm = [], []
    for u in seasons:
        i, p = F.pmf_for(u, excl_fn(u))
        ix.append(i)
        pm.append(p)
    return np.concatenate(ix), np.vstack(pm)


def fit():
    F = Frame()
    folds = {}
    combos = [(n, lam) for n in TIME_SCHEMES for lam in LAMBDAS]
    stab = {}
    for s in POOL:
        rest = [u for u in POOL if u != s]
        inner = {}
        for name, lam in combos:
            edges = TIME_SCHEMES[name]
            nc = 4 * (len(edges) + 1)
            ll = 0.0
            n = 0
            for t in rest:
                r_ix, r_pm = stack(F, [u for u in rest if u != t], lambda u: {s, t, u})
                te_ix, te_pm = F.pmf_for(t, {s, t})
                r = ratios(cell_id(F.sd[r_ix], F.gsr[r_ix], edges), F.cls[r_ix], r_pm, nc, lam)
                a, _ = score(r, cell_id(F.sd[te_ix], F.gsr[te_ix], edges), F.cls[te_ix], te_pm)
                ll += a
                n += len(te_ix)
            inner[(name, lam)] = ll / n
        name, lam = max(inner, key=inner.get)
        edges = TIME_SCHEMES[name]
        nc = 4 * (len(edges) + 1)
        tr_ix, tr_pm = stack(F, rest, lambda u: {s, u})
        r = ratios(cell_id(F.sd[tr_ix], F.gsr[tr_ix], edges), F.cls[tr_ix], tr_pm, nc, lam)
        a_in, b_in = score(r, cell_id(F.sd[tr_ix], F.gsr[tr_ix], edges), F.cls[tr_ix], tr_pm)
        te_ix, te_pm = F.pmf_for(s, {s})
        a_out, b_out = score(r, cell_id(F.sd[te_ix], F.gsr[te_ix], edges), F.cls[te_ix], te_pm)
        te_pm_u = F.pmf_for(s, {s}, weighted=False)[1]
        _, b_unw = score(r, cell_id(F.sd[te_ix], F.gsr[te_ix], edges), F.cls[te_ix], te_pm_u)
        stab[s] = ((np.bincount(F.cls[tr_ix], minlength=NCLS) + lam) / (tr_pm.sum(axis=0) + lam)).tolist()
        folds[s] = dict(choice=f"{name}|{lam:g}", n_te=int(len(te_ix)), out_adj=a_out / len(te_ix), out_nn=b_out / len(te_ix), out_nn_unweighted=b_unw / len(te_ix), in_adj=a_in / len(tr_ix), in_nn=b_in / len(tr_ix))
        print(s, folds[s], flush=True)
    n_tot = sum(f["n_te"] for f in folds.values())
    pooled = {k: sum(f[k] * f["n_te"] for f in folds.values()) / n_tot for k in ("out_adj", "out_nn", "out_nn_unweighted")}
    pooled["in_adj"] = float(np.mean([f["in_adj"] for f in folds.values()]))
    pooled["in_nn"] = float(np.mean([f["in_nn"] for f in folds.values()]))
    deltas = [f["out_adj"] - f["out_nn"] for f in folds.values()]
    choices = [f["choice"] for f in folds.values()]
    modal = max(set(choices), key=lambda c: (choices.count(c), c))
    name, lam = modal.split("|")
    lam = float(lam)
    edges = TIME_SCHEMES[name]
    nc = 4 * (len(edges) + 1)
    tr_ix, tr_pm = stack(F, POOL, lambda u: {u})
    r = ratios(cell_id(F.sd[tr_ix], F.gsr[tr_ix], edges), F.cls[tr_ix], tr_pm, nc, lam)
    arr = np.array(list(stab.values()))
    OUTD.mkdir(parents=True, exist_ok=True)
    spec = dict(scheme=name, edges=edges, lam=lam, lead_edges=LEAD_EDGES, ratio=r.tolist(), ncls=NCLS, folds=folds, pooled=pooled, fold_deltas=deltas, class_ratio_mean=arr.mean(axis=0).tolist(), class_ratio_min=arr.min(axis=0).tolist(), class_ratio_max=arr.max(axis=0).tolist(), n=int(len(tr_ix)), looks=len(combos) * len(POOL))
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    print("modal", modal, "pooled", pooled, "deltas", [round(d, 4) for d in deltas])
    print("class ratio mean", np.round(arr.mean(axis=0), 3), "min", np.round(arr.min(axis=0), 3), "max", np.round(arr.max(axis=0), 3))
    return spec


def install_ldr():
    import mod25d_variance as dv

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    ns = dv._G["ns"]
    a = dv._G["tables"]["arrays"]
    r = np.asarray(spec["ratio"])
    rmax = r.max(axis=1)
    edges = spec["edges"]
    rowcls = classes(a["play_type_code"], a["yards_gained"], a["dist_raw"], a["auto_first"])
    base = ns["pick_index_nn_conditioned"]
    tries = int(np.ceil(np.log(1e-3) / np.log(0.5)))

    def pick(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        j = base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        if phase != 3 or score < 1:
            return j
        c = int(cell_id(np.array([score]), np.array([time_raw]), edges)[0])
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
