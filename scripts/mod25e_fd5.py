import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "artifacts" / "mod25e3" / "fd5"
SALT = 32452851
LATE_QTR = 4
LATE_SECONDS = 300.0
SCORE_EDGES = [-8.5, -3.5, -0.5, 0.5, 3.5, 8.5]
BASE = dict(max_depth=8, min_samples_leaf=20, learning_rate=0.05, max_iter=150)
LATE = dict(max_depth=4, min_samples_leaf=20, learning_rate=0.05, max_iter=100)


def enabled():
    return os.environ.get("FD5") == "1"


def features(X):
    sd = X[:, 3]
    hs = X[:, 2]
    f = np.column_stack(
        [
            np.digitize(sd, SCORE_EDGES),
            (sd < 0) & (sd >= -3),
            (sd < 0) & (sd >= -8),
            sd == 0,
            (sd > 0) & (sd <= 3),
            np.log1p(hs),
        ]
    ).astype(float)
    return np.hstack([X[:, :7], f])


def is_late(X):
    return (X[:, 4] == LATE_QTR) & (X[:, 2] <= LATE_SECONDS)


def make(kw):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(l2_regularization=1.0, random_state=1, **kw)


class Choice:
    def fit(self, X, y):
        F = features(X)
        late = is_late(X)
        self.base = make(BASE).fit(F, y)
        self.late = make(LATE).fit(F[late], y[late])
        return self

    def predict(self, X):
        F = features(X)
        P = self.base.predict_proba(F)
        late = is_late(X)
        if late.any():
            P[late] = self.late.predict_proba(F[late])
        return P


def cmd_fit(a):
    import mod25e_fourth as f4

    OUT.mkdir(parents=True, exist_ok=True)
    d = f4.load()
    X = d[list(f4.COLS)].to_numpy(float)
    y = d["y"].to_numpy()
    s = f4.seasons_arr(d)
    P = np.zeros((len(y), 3))
    for v in sorted(set(s.tolist())):
        te = s == v
        P[te] = Choice().fit(X[~te], y[~te]).predict(X[te])
    base = np.load(f4.OUT / "oof" / "HGB_d6_l50.npy")
    ll = lambda Q: f4.ll_vec(Q, y)
    fold = np.array([ll(P)[s == v].mean() - ll(base)[s == v].mean() for v in sorted(set(s.tolist()))])
    print("LOSO LL", ll(P).mean(), "FD4 LOSO LL", ll(base).mean(), "fold deltas", fold.round(4), "wins", int((fold > 0).sum()))
    m = Choice().fit(X, y)
    with open(OUT / "model.pkl", "wb") as fh:
        pickle.dump(m, fh)
    (OUT / "fit.json").write_text(json.dumps({"loso_ll": float(ll(P).mean()), "fd4_loso_ll": float(ll(base).mean()), "era_at_sim": 2017}, indent=1))


def install_fd5():
    import mod25d_variance as dv
    import mod25e_fourth as f4

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "Choice":
                return Choice
            return super().find_class(module, name)

    with open(OUT / "model.pkl", "rb") as fh:
        model = _Unpickler(fh).load()
    era = float(json.loads((OUT / "fit.json").read_text())["era_at_sim"])
    ns = dv._G["ns"]
    seed = int(dv._G["cfg"].get("seed", 3))
    code = dv._G["tables"]["arrays"]["play_type_code"]
    root = ns["DECIDE"]
    holders = f4._find(root, "make_c_decide.<locals>.decide")
    if holders:
        h = holders[0]
        cell = h.__closure__[h.__code__.co_freevars.index("base_decide")]
        orig = cell.cell_contents
    else:
        cell = None
        orig = root
    bases = f4._find(orig, "make_decide.<locals>.decide")
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
