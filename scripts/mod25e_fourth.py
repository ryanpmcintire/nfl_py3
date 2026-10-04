import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "artifacts" / "mod25e3" / "fd4"
TRAIN = tuple(range(2009, 2018))
SALT = 32452843
COLS = ("yl", "dist", "hs", "sd", "qtr", "oto", "dto", "era")
CLASSES = ("go", "fg", "punt")
NBOOT = 1000
KNN_PRIOR = 0.5


def enabled():
    return os.environ.get("FD4") == "1"


def load():
    import mod25_mechanisms as m25
    import pandas as pd
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(TRAIN)
    tr = sim.build_transition_frame(pbp)
    m = (tr["down_i"] == 4) & tr["play_type_code"].isin((0, 1, 2, 3)) & (tr["qtr_actual"] <= 4)
    d = tr.loc[m].reset_index(drop=True)
    code = d["play_type_code"].to_numpy()
    out = pd.DataFrame(
        {
            "yl": d["fp_raw"].to_numpy(float),
            "dist": d["dist_raw"].to_numpy(float),
            "hs": d["time_raw"].to_numpy(float),
            "sd": d["sc_raw"].to_numpy(float),
            "qtr": d["qtr_actual"].to_numpy(float),
            "oto": d["off_to_raw"].to_numpy(float),
            "dto": d["def_to_raw"].to_numpy(float),
            "era": d["game_id"].str[:4].astype(int).to_numpy(float),
            "gsr": d["gsr_actual"].to_numpy(float),
            "phase": d["phase"].to_numpy(int),
            "game": d["game_id"].to_numpy(),
            "y": m25.cls4_of(code).astype(int),
        }
    )
    return out


class Design:
    def __init__(self, full, nk, inter, era):
        self.full = full
        self.nk = nk
        self.inter = inter
        self.era = era
        self.vars = ("yl", "ld", "hs", "sd") if full else ("yl", "ld")

    def raw(self, X):
        import sim04_engine as sim

        return {
            "yl": X[:, 0],
            "ld": np.log(np.maximum(X[:, 1], 1.0)),
            "hs": np.log1p(np.maximum(X[:, 2], 0.0)),
            "sd": np.clip(X[:, 3], -sim.SCORE_CLIP, sim.SCORE_CLIP),
        }

    def fit(self, X):
        from sklearn.preprocessing import SplineTransformer

        r = self.raw(X)
        self.sp = {
            k: SplineTransformer(n_knots=self.nk, degree=3, knots="uniform", extrapolation="constant").fit(r[k][:, None])
            for k in self.vars
        }
        lin = X[:, [5, 6, 7]]
        self.mu = lin.mean(axis=0)
        self.sg = np.where(lin.std(axis=0) > 0, lin.std(axis=0), 1.0)
        return self

    def transform(self, X):
        r = self.raw(X)
        B = {k: self.sp[k].transform(r[k][:, None]) for k in self.vars}
        cols = [B[k] for k in self.vars]
        if self.full:
            q = np.column_stack([(X[:, 4] == v).astype(float) for v in (1, 2, 3, 4)])
            lin = (X[:, [5, 6, 7]] - self.mu) / self.sg
            cols += [q, lin[:, :2]]
            if self.era:
                cols.append(lin[:, 2:3])
        elif self.era:
            cols.append(((X[:, [7]] - self.mu[2]) / self.sg[2]))
        if self.inter:
            pairs = [("yl", "ld")] + ([("hs", "sd")] if self.full else [])
            for a, b in pairs:
                cols.append(np.einsum("ni,nj->nij", B[a], B[b]).reshape(len(X), -1))
            if self.full:
                cols.append(np.einsum("ni,nj->nij", q, B["hs"]).reshape(len(X), -1))
        return np.hstack(cols)


class LRModel:
    def __init__(self, full, nk, inter, C, era=False):
        self.args = (full, nk, inter, C, era)

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        full, nk, inter, C, era = self.args
        self.des = Design(full, nk, inter, era).fit(X)
        self.m = LogisticRegression(C=C, max_iter=500).fit(self.des.transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict_proba(self.des.transform(X))


class HGBModel:
    def __init__(self, depth, leaf, era=False, lr=0.05, it=150):
        self.args = (depth, leaf, era, lr, it)

    def cols(self):
        return [0, 1, 2, 3, 4, 5, 6] + ([7] if self.args[2] else [])

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier

        depth, leaf, _, lr, it = self.args
        self.m = HistGradientBoostingClassifier(
            max_depth=depth, min_samples_leaf=leaf, learning_rate=lr, max_iter=it, l2_regularization=1.0, random_state=1
        ).fit(X[:, self.cols()], y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X[:, self.cols()])


class PriorModel:
    def fit(self, X, y):
        c = np.bincount(y, minlength=3).astype(float)
        self.p = c / c.sum()
        return self

    def predict(self, X):
        return np.tile(self.p, (len(X), 1))


def build(spec):
    k = spec["kind"]
    if k == "prior":
        return PriorModel()
    if k == "lr":
        return LRModel(spec["full"], spec["nk"], spec["inter"], spec["C"], spec.get("era", False))
    return HGBModel(spec["depth"], spec["leaf"], spec.get("era", False), spec.get("lr", 0.05), spec.get("iters", 150))


def snap(ax, v):
    return ax[np.abs(ax[None, :] - v[:, None]).argmin(axis=1)]


def proxy_pol4(d, tr, te, snapped):
    import mod25_mechanisms as m25
    from sklearn.ensemble import HistGradientBoostingClassifier

    def feats(ix, sn):
        sd = np.clip(d["sd"].to_numpy()[ix], -24, 24)
        gsr = d["gsr"].to_numpy()[ix]
        yl = d["yl"].to_numpy()[ix]
        dist = d["dist"].to_numpy()[ix]
        otf = (d["oto"].to_numpy()[ix] > 0).astype(float)
        dtf = (d["dto"].to_numpy()[ix] > 0).astype(float)
        ot = np.zeros(len(ix))
        if sn:
            sd, gsr, yl, dist = snap(m25.SD_AX, sd), snap(m25.T4_AX, gsr), snap(m25.Y4_AX, yl), snap(m25.D4_AX, dist)
        return np.column_stack([sd, gsr, ot, yl, dist, otf, dtf])

    c4 = HistGradientBoostingClassifier(max_depth=5, max_iter=200, learning_rate=0.08, random_state=1).fit(
        feats(tr, False), d["y"].to_numpy()[tr]
    )
    return c4.predict_proba(feats(te, snapped))


def proxy_knn(d, tr, te):
    import sim04_engine as sim
    from scipy.spatial import cKDTree

    def fm(ix):
        return sim.feature_matrix(
            d["dist"].to_numpy()[ix],
            d["yl"].to_numpy()[ix],
            d["sd"].to_numpy()[ix],
            d["hs"].to_numpy()[ix],
            d["oto"].to_numpy()[ix],
            d["dto"].to_numpy()[ix],
            d["phase"].to_numpy()[ix],
        )

    Ftr, Fte = fm(tr), fm(te)
    ph_tr, ph_te = d["phase"].to_numpy()[tr], d["phase"].to_numpy()[te]
    ytr = d["y"].to_numpy()[tr]
    P = np.zeros((len(te), 3))
    for p in np.unique(ph_te):
        sub = np.flatnonzero(sim.phase_pool_mask(ph_tr, int(p)))
        k = min(sim.K_STATE, len(sub))
        _, ind = cKDTree(Ftr[sub]).query(Fte[ph_te == p], k=k)
        ind = np.atleast_2d(ind.reshape(-1, k))
        cnt = np.stack([(ytr[sub][ind] == c).sum(axis=1) for c in range(3)], axis=1).astype(float)
        P[ph_te == p] = (cnt + KNN_PRIOR) / (k + 3 * KNN_PRIOR)
    return P


def oof(d, name, fn, folds, seasons):
    f = OUT / "oof" / f"{name}.npy"
    if f.exists():
        return np.load(f)
    P = np.zeros((len(d), 3))
    for s in seasons:
        te = np.flatnonzero(seasons_arr(d) == s)
        tr = np.flatnonzero(seasons_arr(d) != s)
        P[te] = fn(tr, te)
    np.save(f, P)
    return P


def seasons_arr(d):
    return d["era"].to_numpy(int)


def sub_fit(d, spec):
    X = d[list(COLS)].to_numpy(float)
    y = d["y"].to_numpy()

    def fn(tr, te):
        return build(spec).fit(X[tr], y[tr]).predict(X[te])

    return fn


def ll_vec(P, y):
    return np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))


def fold_ll(P, y, s):
    return np.array([ll_vec(P[s == v], y[s == v]).mean() for v in np.unique(s)])


def boot_pp(diff, game, rng):
    g = np.unique(game, return_inverse=True)[1]
    ng = g.max() + 1
    sd = np.bincount(g, weights=diff, minlength=ng)
    n = np.bincount(g, minlength=ng).astype(float)
    ix = rng.integers(0, ng, size=(NBOOT, ng))
    est = sd[ix].sum(axis=1) / n[ix].sum(axis=1)
    return float(diff.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), float((est > 0).mean())


def regions(d):
    hs, q, sd = d["hs"].to_numpy(), d["qtr"].to_numpy(), d["sd"].to_numpy()
    yl = d["yl"].to_numpy()
    hend = ((q == 2) | (q == 4)) & (hs <= 120)
    q4t = (q == 4) & (hs > 120) & (sd < 0)
    return {
        "half-end hs<=120": hend,
        "Q4 trailing hs>120": q4t,
        "normal": ~hend & ~q4t,
        "half-end yl<=40": hend & (yl <= 40),
        "Q2 hs<=10 yl41-60": (q == 2) & (hs <= 10) & (yl >= 41) & (yl <= 60),
    }


def cmd_fit(a):
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oof").mkdir(exist_ok=True)
    t0 = time.time()
    d = load()
    X = d[list(COLS)].to_numpy(float)
    y = d["y"].to_numpy()
    s = seasons_arr(d)
    seas = sorted(set(s.tolist()))
    np.savez(OUT / "train.npz", X=X, y=y)
    lines = []

    def say(t):
        print(t, flush=True)
        lines.append(t)

    say(f"4th downs 2009-17 n {len(d)}; go/fg/punt share {np.bincount(y) / len(y)}; seasons {seas}")
    R = {}
    R["pol4_snap"] = oof(d, "pol4_snap", lambda tr, te: proxy_pol4(d, tr, te, True), None, seas)
    R["pol4_raw"] = oof(d, "pol4_raw", lambda tr, te: proxy_pol4(d, tr, te, False), None, seas)
    R["knn"] = oof(d, "knn", lambda tr, te: proxy_knn(d, tr, te), None, seas)
    R["prior"] = oof(d, "prior", sub_fit(d, {"kind": "prior"}), None, seas)
    say(f"proxies done {time.time() - t0:.0f}s")
    specs = {}
    for nk in (4, 6):
        for C in (0.03, 0.3, 3.0):
            specs[f"S1_nk{nk}_C{C}"] = {"kind": "lr", "full": False, "nk": nk, "inter": True, "C": C}
    for nk in (4, 6):
        for inter in (False, True):
            for C in (0.03, 0.3, 3.0):
                specs[f"LR_nk{nk}_i{int(inter)}_C{C}"] = {"kind": "lr", "full": True, "nk": nk, "inter": inter, "C": C}
    for depth in (2, 3, 4):
        for leaf in (50, 150, 450):
            specs[f"HGB_d{depth}_l{leaf}"] = {"kind": "hgb", "depth": depth, "leaf": leaf}
    for depth, leaf in ((4, 20), (5, 20), (5, 50), (6, 20), (6, 50)):
        specs[f"HGB_d{depth}_l{leaf}"] = {"kind": "hgb", "depth": depth, "leaf": leaf}
    for depth, leaf in ((5, 20), (6, 20)):
        specs[f"HGB_d{depth}_l{leaf}_f"] = {"kind": "hgb", "depth": depth, "leaf": leaf, "lr": 0.08, "iters": 200}
    for nm, sp in specs.items():
        R[nm] = oof(d, nm, sub_fit(d, sp), None, seas)
    say(f"grid done {time.time() - t0:.0f}s")
    score = {nm: float(ll_vec(P, y).mean()) for nm, P in R.items()}
    best = {g: min((n for n in specs if n.startswith(g)), key=lambda n: -score[n]) for g in ("S1", "LR", "HGB")}
    for g in ("LR", "HGB"):
        sp = dict(specs[best[g]], era=True)
        nm = best[g] + "_era"
        specs[nm] = sp
        R[nm] = oof(d, nm, sub_fit(d, sp), None, seas)
        score[nm] = float(ll_vec(R[nm], y).mean())
    final = max((n for n in specs if n.startswith(("LR", "HGB")) and not n.endswith("_era")), key=lambda n: score[n])
    FLe = fold_ll(R[final + "_era"], y, s) if final + "_era" in R else None
    refs = ["pol4_snap", "pol4_raw", "knn", "prior", best["S1"]]
    FL = {nm: fold_ll(P, y, s) for nm, P in R.items()}
    say(f"looks: {len(specs) + 4} fits per fold x {len(seas)} folds, 3 region sets, 3 bootstraps")
    say("name | LOSO mean LL | 9-fold wins vs pol4_snap / knn / prior / S1best")
    rows = sorted(R, key=lambda n: -score[n])
    for nm in rows:
        w = [int((FL[nm] > FL[r]).sum()) for r in ("pol4_snap", "knn", "prior", best["S1"])]
        say(f"{nm:22s} {score[nm]:.4f}  {w[0]}/{w[1]}/{w[2]}/{w[3]}")
    if FLe is not None:
        say(f"era variant of selected: LL {score[final + '_era']:.4f} vs {score[final]:.4f}, fold wins {int((FLe > fold_ll(R[final], y, s)).sum())}/9; era excluded from deployment unless 9/9")
    say(f"selected {final} {specs[final]}; grid spread best {max(score[n] for n in specs):.4f} median {float(np.median([score[n] for n in specs])):.4f}")
    ins = build(specs[final]).fit(X, y).predict(X)
    say(f"in-sample LL {ll_vec(ins, y).mean():.4f} vs LOSO {score[final]:.4f} gap {ll_vec(ins, y).mean() - score[final]:.4f}")
    rng = np.random.default_rng(1)
    game = d["game"].to_numpy()
    reg = regions(d)
    say("game-bootstrap per-play LL difference of selected minus reference (mean, 95% interval, probability_positive)")
    for r in ("pol4_snap", "knn", best["S1"]):
        diff = ll_vec(R[final], y) - ll_vec(R[r], y)
        say(f"all vs {r}: {boot_pp(diff, game, rng)}")
        m = reg["half-end hs<=120"]
        say(f"half-end vs {r}: {boot_pp(diff[m], game[m], rng)}")
    say("calibration by region: n | share go/fg/punt observed | mean predicted selected | pol4_snap | knn | LL selected / pol4_snap / knn / prior")
    for rn, m in reg.items():
        obs = np.bincount(y[m], minlength=3) / m.sum()
        f = lambda P: np.round(P[m].mean(axis=0), 3)
        lls = [round(float(ll_vec(R[k][m], y[m]).mean()), 4) for k in (final, "pol4_snap", "knn", "prior")]
        say(f"{rn:20s} n {int(m.sum()):5d} obs {np.round(obs, 3)} sel {f(R[final])} pol4 {f(R['pol4_snap'])} knn {f(R['knn'])} LL {lls}")
    say("reliability pooled (selected, class go/fg/punt, 10 equal-count bins): mean predicted -> observed")
    for c in range(3):
        p = R[final][:, c]
        order = np.argsort(p)
        parts = np.array_split(order, 10)
        say(CLASSES[c] + " " + " ".join(f"{p[i].mean():.3f}>{(y[i] == c).mean():.3f}" for i in parts))
    spec_out = {"selected": final, "spec": specs[final], "loso_ll": score[final], "pol4_snap_ll": score["pol4_snap"], "knn_ll": score["knn"], "era_at_sim": int(max(seas))}
    (OUT / "fit.json").write_text(json.dumps(spec_out, indent=1))
    (OUT / "fd4.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame({"name": rows, "ll": [score[n] for n in rows]}).to_csv(OUT / "grid.csv", index=False)


def _find(root, qualname):
    seen, stack, hits = set(), [root], []
    while stack:
        f = stack.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        if f.__qualname__ == qualname:
            hits.append(f)
        for cell in f.__closure__ or ():
            v = cell.cell_contents
            if callable(v) and getattr(v, "__closure__", None):
                stack.append(v)
    return hits


def install_fd4():
    import mod25d_variance as dv

    z = np.load(OUT / "train.npz")
    meta = json.loads((OUT / "fit.json").read_text())
    spec = meta["spec"]
    model = build(spec).fit(z["X"], z["y"])
    era = float(meta["era_at_sim"])
    ns = dv._G["ns"]
    seed = int(dv._G["cfg"].get("seed", 3))
    code = dv._G["tables"]["arrays"]["play_type_code"]
    root = ns["DECIDE"]
    holders = _find(root, "make_c_decide.<locals>.decide")
    if holders:
        h = holders[0]
        cell = h.__closure__[h.__code__.co_freevars.index("base_decide")]
        orig = cell.cell_contents
    else:
        cell = None
        orig = root
    bases = _find(orig, "make_decide.<locals>.decide")
    assert bases, "no make_decide.decide in the DECIDE chain"
    b = bases[0]
    cpick = b.__closure__[b.__code__.co_freevars.index("cpick")].cell_contents
    st = {"k": None, "rng": None, "cache": {}}

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        idx = orig(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
        c0 = int(code[idx])
        if down != 4 or in_ot or c0 == 6:
            return idx
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, seed)
        key = (float(yardline), float(distance), float(time_feat), float(score_diff), float(qtr), float(off_to), float(def_to))
        p = st["cache"].get(key)
        if p is None:
            p = model.predict(np.array([[*key, era]]))[0]
            st["cache"][key] = p
        ci = min(int(np.searchsorted(np.cumsum(p), st["rng"].random() * p.sum())), 2)
        cur = 0 if c0 in (0, 1) else (1 if c0 == 3 else (2 if c0 == 2 else -1))
        if ci == cur:
            return idx
        j = cpick(rng, 4, phase, ci, distance, yardline, score_diff, time_feat, off_to, def_to)
        return idx if j is None else j

    if cell is not None:
        cell.cell_contents = decide
    else:
        ns["DECIDE"] = decide


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    a = ap.parse_args()
    {"fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
