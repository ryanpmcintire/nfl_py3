import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "dkern" / "fit.json"
GRID = (0.15, 0.3, 0.6, 1.2, 2.4)
FOLDS = 9
DOWNS = (2, 3, 4)


def enabled():
    return os.environ.get("DKF") == "1"


def fit():
    import mod25e_draw as dr

    G, _ = dr.build()
    a = G["tables"]["arrays"]
    term = a["possession_flip"] | (a["points_off"] > 0) | (a["points_def"] > 0)
    down = a["down_i"]
    code = a["play_type_code"]
    keep = np.isin(code, [0, 1]) & np.isin(down, DOWNS)
    dist = np.clip(np.round(a["dist_raw"]).astype(int), 1, 30)
    conv = ((~term & (a["next_down"] == 1)) | (a["points_off"] >= 6)).astype(float)
    grid = np.arange(1, 31)
    tot = np.zeros(len(GRID))
    cnt = 0
    per = {}
    for dn in DOWNS:
        m = keep & (down == dn)
        dd, cv = dist[m], conv[m]
        fold = np.minimum((np.arange(len(dd)) * FOLDS) // len(dd), FOLDS - 1)
        ll = np.zeros((FOLDS, len(GRID)))
        for f in range(FOLDS):
            tr = fold != f
            n = np.bincount(dd[tr], minlength=31)[1:].astype(float)
            k = np.bincount(dd[tr], weights=cv[tr], minlength=31)[1:]
            base = cv[tr].mean()
            te = ~tr
            for ci, c in enumerate(GRID):
                h = c * np.sqrt(np.maximum(1, grid))
                W = np.exp(-0.5 * ((grid[None, :] - grid[:, None]) / h[:, None]) ** 2)
                p = np.clip((W @ k + base) / (W @ n + 1.0), 1e-6, 1 - 1e-6)[dd[te] - 1]
                ll[f, ci] = -np.mean(cv[te] * np.log(p) + (1 - cv[te]) * np.log(1 - p))
        per[dn] = ll.mean(axis=0).tolist()
        tot += ll.mean(axis=0) * len(dd)
        cnt += len(dd)
    tot /= cnt
    best = float(GRID[int(tot.argmin())])
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps({"c": best, "grid": list(GRID), "pooled_ll": tot.tolist(), "per_down_ll": per, "folds": FOLDS}), encoding="utf-8")
    print("best c", best, "pooled", tot.tolist(), "per", per)


class ExactTree:
    def __init__(self, tree, sub_idx, dist_raw, k, scale):
        from scipy.spatial import KDTree

        self.tree = tree
        self.data = tree.data
        self.scale = scale
        self.groups = {}
        dd = np.round(dist_raw[sub_idx]).astype(int)
        for d in np.unique(dd):
            pos = np.flatnonzero(dd == d)
            if len(pos) >= k:
                self.groups[int(d)] = (pos, KDTree(self.data[pos]))

    def query(self, feat, k=1):
        g = self.groups.get(int(round(float(np.asarray(feat).ravel()[0]) * self.scale)))
        if g is None or len(g[0]) < k:
            return self.tree.query(feat, k=k)
        pos, tr = g
        dist, ind = tr.query(feat, k=k)
        return dist, pos[ind]


def install_dk():
    import mod25d_variance as dv

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    ns = dv._G["ns"]
    t = dv._G["tables"]
    assert "DKH" in ns
    ns["DKH"] = float(spec["c"])
    k = int(dv.sim.K_STATE)
    dist_raw = t["arrays"]["dist_raw"]
    for key, (tree, sub) in list(t["nn_trees_cond"].items()):
        t["nn_trees_cond"][key] = (ExactTree(tree, sub, dist_raw, k, dv.sim.SCALE_YDSTOGO), sub)
    t["nn_cache_cond"].clear()
    t["nn_weight_cache_cond"].clear()


if __name__ == "__main__":
    fit()
