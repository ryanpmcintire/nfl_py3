import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "dkern2" / "fit.json"
VAL = REPO / "artifacts" / "mod25e3" / "dkern2" / "validate.txt"
DKFIT = REPO / "artifacts" / "mod25e3" / "dkern" / "fit.json"
GRID = (5, 10, 20, 40, 80, 160, 200, 320, 640, 1280, 2560)
FOLDS = 9
DOWNS = (2, 3, 4)
GO = (0, 1)
OLD_MIN = 200


def enabled():
    return os.environ.get("DK2") == "1"


def credit_table(dd, yards, term, td):
    H = np.zeros((30, 100))
    y = np.clip(yards, 0, 99).astype(int)
    ok = ~term
    np.add.at(H, (dd[ok] - 1, y[ok]), 1.0)
    tail = np.cumsum(H[:, ::-1], axis=1)[:, ::-1]
    tdc = np.bincount(dd[td] - 1, minlength=30).astype(float)
    C = tdc[:, None] + tail[:, 1:31]
    n = np.bincount(dd - 1, minlength=30).astype(float)
    return C, n


def window_p(C, n, base, m):
    D = len(n)
    cn = np.r_[0.0, np.cumsum(n)]
    cc = np.vstack([np.zeros((1, D)), np.cumsum(C, axis=0)])
    p = np.zeros(D)
    for i in range(D):
        w = 0
        while True:
            lo, hi = max(i - w, 0), min(i + w + 1, D)
            if cn[hi] - cn[lo] >= m or (lo == 0 and hi == D):
                break
            w += 1
        p[i] = (cc[hi, i] - cc[lo, i] + base) / (cn[hi] - cn[lo] + 1.0)
    return np.clip(p, 1e-6, 1 - 1e-6)


def old_p(C, n, base):
    idx = np.arange(len(n))
    pooled = (C.sum(axis=0) + base) / (n.sum() + 1.0)
    own = (C[idx, idx] + base) / (n + 1.0)
    return np.clip(np.where(n >= OLD_MIN, own, pooled), 1e-6, 1 - 1e-6)


BINS = ((1, 1), (2, 3), (4, 6), (7, 9), (10, 30))


def nll(p, d, cv):
    q = p[d - 1]
    loss = -(cv * np.log(q) + (1 - cv) * np.log(1 - q))
    return float(np.mean([loss[(d >= lo) & (d <= hi)].mean() for lo, hi in BINS if ((d >= lo) & (d <= hi)).any()]))


def fit():
    import mod25e_draw as dr

    G, _ = dr.build()
    a = G["tables"]["arrays"]
    term = a["possession_flip"] | (a["points_off"] > 0) | (a["points_def"] > 0)
    down = a["down_i"]
    code = a["play_type_code"]
    keep = np.isin(code, GO) & np.isin(down, DOWNS)
    dist = np.clip(np.round(a["dist_raw"]).astype(int), 1, 30)
    conv = ((~term & (a["next_down"] == 1)) | (a["points_off"] >= 6)).astype(float)
    yds_all = np.round(a["yards_gained"]).astype(int)
    td_all = a["points_off"] >= 6
    kernel_c = json.loads(DKFIT.read_text(encoding="utf-8"))["c"]
    grid = np.arange(1, 31)
    out = {"grid": list(GRID), "folds": FOLDS, "kernel_c": kernel_c, "per_down": {}}
    for dn in DOWNS:
        m = keep & (down == dn)
        dd, cv = dist[m], conv[m]
        yy, tt, dt = yds_all[m], term[m], td_all[m]
        fold = np.minimum((np.arange(len(dd)) * FOLDS) // len(dd), FOLDS - 1)
        ll = np.zeros((FOLDS, len(GRID)))
        ll_old = np.zeros(FOLDS)
        ll_ker = np.zeros(FOLDS)
        for f in range(FOLDS):
            tr, te = fold != f, fold == f
            n = np.bincount(dd[tr], minlength=31)[1:].astype(float)
            kk = np.bincount(dd[tr], weights=cv[tr], minlength=31)[1:]
            base = cv[tr].mean()
            C, nn_ = credit_table(dd[tr], yy[tr], tt[tr], dt[tr])
            for gi, g in enumerate(GRID):
                ll[f, gi] = nll(window_p(C, nn_, base, g), dd[te], cv[te])
            ll_old[f] = nll(old_p(C, nn_, base), dd[te], cv[te])
            h = kernel_c * np.sqrt(np.maximum(1, grid))
            W = np.exp(-0.5 * ((grid[None, :] - grid[:, None]) / h[:, None]) ** 2)
            ll_ker[f] = nll(np.clip((W @ kk + base) / (W @ n + 1.0), 1e-6, 1 - 1e-6), dd[te], cv[te])
        mean = ll.mean(axis=0)
        ki = GRID.index(OLD_MIN) if OLD_MIN in GRID else int(np.abs(np.array(GRID) - OLD_MIN).argmin())
        bi = int(mean.argmin())
        if bi != ki and not (ll[:, bi] < ll[:, ki]).all():
            bi = ki
        nested = np.zeros(FOLDS)
        for f in range(FOLDS):
            others = np.delete(np.arange(FOLDS), f)
            nested[f] = ll[f, int(ll[others].mean(axis=0).argmin())]
        out["per_down"][str(dn)] = {
            "rows": int(len(dd)),
            "ll_by_min_group": mean.tolist(),
            "best_min_group": int(GRID[bi]),
            "argmin_ll_min_group": int(GRID[int(mean.argmin())]),
            "ll_new": float(mean[bi]),
            "ll_dkf1_exact200_fallback_pooled": float(ll_old.mean()),
            "ll_dkf1_kernel": float(ll_ker.mean()),
            "fold_wins_vs_dkf1": int((ll[:, bi] < ll_old).sum()),
            "fold_wins_vs_kernel": int((ll[:, bi] < ll_ker).sum()),
            "nested_ll": float(nested.mean()),
            "nested_fold_wins_vs_dkf1": int((nested < ll_old).sum()),
        }
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(out, indent=1))


class NearTree:
    def __init__(self, tree, sub_idx, dist_raw, code, m, k, scale, mixed):
        from scipy.spatial import KDTree

        self.KD = KDTree
        self.tree = tree
        self.data = np.asarray(tree.data)
        self.m = int(m)
        self.k = int(k)
        self.scale = scale
        self.dd = np.round(dist_raw[sub_idx]).astype(int)
        self.go = np.isin(code[sub_idx], GO)
        self.mixed = bool(mixed)
        self.cache = {}

    def pool(self, d0, which, need=None):
        need = self.m if need is None else need
        key = (d0, which, need)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        base = self.go if which == "go" else np.ones(len(self.dd), dtype=bool)
        ddv = np.clip(self.dd, 0, None)
        hist = np.bincount(ddv[base], minlength=max(int(ddv.max()), d0) + 2)
        cum = np.r_[0, np.cumsum(hist)]
        top = len(hist)
        w = 0
        while True:
            lo, hi = max(d0 - w, 0), min(d0 + w + 1, top)
            if cum[hi] - cum[lo] >= need or (lo == 0 and hi == top):
                break
            w += 1
        pos = np.flatnonzero(base & (ddv >= d0 - w) & (ddv <= d0 + w))
        if len(pos) == 0:
            pos = np.flatnonzero(base)
        out = (pos, self.KD(self.data[pos]) if which == "all" else None)
        self.cache[key] = out
        return out

    def query(self, feat, k=1):
        arr = np.asarray(feat, dtype=np.float64)
        f = arr.reshape(-1)
        d0 = int(round(float(f[0]) * self.scale))
        pos, tr = self.pool(d0, "all")
        kk = min(k, len(pos))
        dist, ind = tr.query(f, k=kk)
        got = pos[np.atleast_1d(ind)]
        if self.mixed:
            isgo = self.go[got]
            ng = int(isgo.sum())
            if ng:
                gpos, _ = self.pool(d0, "go", min(self.m, ng))
                gk = min(ng, len(gpos))
                gd = ((self.data[gpos] - f[None, :]) ** 2).sum(axis=1)
                gi = np.argpartition(gd, gk - 1)[:gk] if gk < len(gd) else np.arange(len(gd))
                got = np.concatenate([got[~isgo], gpos[gi]])
        dist = np.zeros(len(got))
        if arr.ndim == 2:
            return dist[None, :], got[None, :]
        return dist, got


def specs():
    return json.loads(FIT.read_text(encoding="utf-8"))


def min_group(spec, dn, k):
    if dn in DOWNS:
        return int(spec["per_down"][str(dn)]["best_min_group"])
    return k


def wrap_trees(t, spec, k, scale):
    arr = t["arrays"]
    out = {}
    for key, (tree, sub) in t["nn_trees_cond"].items():
        dn = key[0]
        out[key] = (NearTree(tree, sub, arr["dist_raw"], arr["play_type_code"], min_group(spec, dn, k), k, scale, dn == 4), sub)
    return out


def wrap_fourth(t, spec, k, scale):
    arr = t["arrays"]
    out = {}
    for key, (tree, sub) in t["nn_trees_4th"].items():
        if key[1] == 0:
            out[key] = (NearTree(tree, sub, arr["dist_raw"], arr["play_type_code"], min_group(spec, 4, k), k, scale, False), sub)
        else:
            out[key] = (tree, sub)
    return out

REDRAW = {
    ("mod25_mechanisms", "make_decide.<locals>.cpick"): ("trees", "cache"),
    ("mod25c_noise", "make_c_decide.<locals>.pick"): ("trees", "cache"),
    ("mod25c_noise", "make_c_decide.<locals>.decide"): ("otp", "otcache"),
    ("sim09_f2", "install_f2.<locals>.pick"): (None, "cache"),
    ("sim09_f2", "install_f2.<locals>.tree"): ("trees", None),
}


class NoCache(dict):
    def get(self, key, default=None):
        return default

    def __setitem__(self, key, value):
        return None


class NearDict(dict):
    def __init__(self, items, make):
        super().__init__()
        self.make = make
        for key, val in items:
            self[key] = val

    def __setitem__(self, key, val):
        super().__setitem__(key, self.make(key, val))


def redraw_maker(t, spec, k, scale):
    arr = t["arrays"]

    def make(key, val):
        if not (isinstance(val, tuple) and len(val) == 2 and hasattr(val[0], "data") and not isinstance(val[0], NearTree)):
            return val
        tree, sub = val
        return (NearTree(tree, sub, arr["dist_raw"], arr["play_type_code"], min_group(spec, key[0], k), k, scale, False), sub)

    return make


def wrap_redraw(root, t, spec, k, scale):
    make = redraw_maker(t, spec, k, scale)
    seen = set()
    stack = [root]
    hits = 0
    while stack:
        f = stack.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        names = REDRAW.get((f.__module__, f.__qualname__))
        for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ()):
            v = cell.cell_contents
            if callable(v) and getattr(v, "__closure__", None):
                stack.append(v)
            if names and isinstance(v, dict) and not isinstance(v, (NoCache, NearDict)):
                if name == names[0]:
                    cell.cell_contents = NearDict(list(v.items()), make)
                    hits += 1
                elif name == names[1]:
                    cell.cell_contents = NoCache()
                    hits += 1
    return hits


def install_dk2():
    import mod25d_variance as dv

    spec = specs()
    ns = dv._G["ns"]
    t = dv._G["tables"]
    assert "DKH" in ns
    ns["DKH"] = float(spec["kernel_c"])
    k = int(dv.sim.K_STATE)
    scale = dv.sim.SCALE_YDSTOGO
    t["nn_trees_cond"] = wrap_trees(t, spec, k, scale)
    t["nn_cache_cond"].clear()
    t["nn_weight_cache_cond"].clear()
    if t.get("nn_trees_4th") is not None:
        t["nn_trees_4th"] = wrap_fourth(t, spec, k, scale)
        t["nn_cache_4th"].clear()
    return wrap_redraw(ns["DECIDE"], t, spec, k, scale)


def validate(per=3000):
    import mod25e_draw as dr
    import mod25e_dkern as dk
    import mod25d_variance as dv
    import sim04_engine as sim

    G, _ = dr.build()
    t = G["tables"]
    a = t["arrays"]
    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = sim.build_transition_frame(pbp)
    ph, sc, tm = trans["phase"].to_numpy(), trans["sc_raw"].to_numpy(), trans["time_raw"].to_numpy()
    oto, dto = trans["off_to_raw"].to_numpy(), trans["def_to_raw"].to_numpy()
    spec = specs()
    c = float(spec["kernel_c"])
    k = int(sim.K_STATE)
    scale = sim.SCALE_YDSTOGO
    orig = t["nn_trees_cond"]
    old = {key: (dk.ExactTree(tree, sub, a["dist_raw"], k, scale), sub) for key, (tree, sub) in orig.items()}
    new = wrap_trees(t, spec, k, scale)
    term = a["possession_flip"] | (a["points_off"] > 0) | (a["points_def"] > 0)
    code = a["play_type_code"]
    draw = np.round(a["dist_raw"]).astype(int)
    yds = a["yards_gained"].astype(float)
    td = a["points_off"] >= 6
    own = (~term & (a["next_down"] == 1)) | td
    bins = [(1, 1), (2, 3), (4, 6), (7, 9), (10, 99)]
    rng = np.random.default_rng(3)
    lines = []

    def say(s):
        print(s)
        lines.append(s)

    say("down bin | n | real conv | P(sd==d) unweighted old/new | implied conv unweighted old/new | P(sd==d) dk-weighted old/new | implied dk-weighted old/new | go run share old/new/real")
    for dn in DOWNS:
        rows = np.flatnonzero((a["down_i"] == dn) & np.isin(code, GO))
        rows = rng.choice(rows, size=min(per, len(rows)), replace=False)
        res = {"old": [], "new": []}
        for r in rows:
            d = int(draw[r])
            feat = sim.feature_matrix(np.array([a["dist_raw"][r]]), np.array([a["fp_raw"][r]]), np.array([sc[r]]), np.array([tm[r]]), np.array([oto[r]]), np.array([dto[r]]), np.array([ph[r]]))[0]
            for name, trees in (("old", old), ("new", new)):
                tree, sub = trees.get((dn, int(ph[r]))) or trees[(dn, 0)]
                _, ind = tree.query(feat, k=min(k, len(sub)))
                nb = sub[np.atleast_1d(ind)]
                nb = nb[np.isin(code[nb], GO)]
                if len(nb) == 0:
                    continue
                w = np.exp(-0.5 * ((a["dist_raw"][nb] - d) / (c * max(1.0, d) ** 0.5)) ** 2)
                w = w / w.sum()
                cred = ((yds[nb] >= d) & ~term[nb]) | td[nb]
                res[name].append((d, float((w * (draw[nb] == d)).sum()), float((w * cred).sum()), float((w * (code[nb] == 0)).sum()), float((draw[nb] == d).mean()), float(cred.mean())))
        for lo, hi in bins:
            m = (a["down_i"] == dn) & np.isin(code, GO) & (draw >= lo) & (draw <= hi)
            if m.sum() == 0:
                continue
            real = own[m].mean()
            rrun = (code[m] == 0).mean()
            cells = {}
            for name in ("old", "new"):
                z = np.array([x for x in res[name] if lo <= x[0] <= hi])
                cells[name] = z.mean(axis=0)[1:] if len(z) else np.full(5, np.nan)
            n = sum(1 for x in res["new"] if lo <= x[0] <= hi)
            say(f"{dn} {lo}-{hi} | {n} | {real:.3f} | {cells['old'][3]:.3f}/{cells['new'][3]:.3f} | {cells['old'][4]:.3f}/{cells['new'][4]:.3f} | {cells['old'][0]:.3f}/{cells['new'][0]:.3f} | {cells['old'][1]:.3f}/{cells['new'][1]:.3f} | {cells['old'][2]:.3f}/{cells['new'][2]:.3f}/{rrun:.3f}")
    VAL.parent.mkdir(parents=True, exist_ok=True)
    VAL.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
    else:
        fit()
