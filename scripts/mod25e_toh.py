import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUTD = REPO / "artifacts" / "mod25e3" / "toh"
FIT = OUTD / "fit.json"
POOL = tuple(range(2009, 2018))
FEATS = ("lh", "sc", "tied", "otr", "q4", "run", "dto", "le30", "le60", "le120")
COUNT = {"calls": 0, "eligible": 0, "called": 0, "no_donor": 0}
_STATE = {"k": None, "rng": None}


def enabled():
    return os.environ.get("TOH") == "1"


def season_key():
    return os.environ.get("TOHS") or "all"


def design(qtr, gsr, score_diff, off_to, def_to, code):
    import sim04_engine as se

    half = np.where(np.asarray(qtr) == 2, np.asarray(gsr, float) - 1800.0, np.asarray(gsr, float))
    return np.column_stack([
        np.log1p(np.clip(half, 0.0, None)),
        se.scaled_score_diff(np.asarray(score_diff, float)),
        (np.asarray(score_diff) == 0).astype(float),
        np.asarray(off_to, float),
        (np.asarray(qtr) == 4).astype(float),
        (np.asarray(code) == 0).astype(float),
        np.asarray(def_to, float),
        (half <= 30.0).astype(float),
        (half <= 60.0).astype(float),
        (half <= 120.0).astype(float),
    ])


def late_mask(phase, sc, off_to, time_raw):
    return np.isin(phase, (1, 3)) & (np.asarray(sc) <= 0) & (np.asarray(off_to) >= 1) & (np.asarray(time_raw) <= 300.0)


def frame():
    import sim04_engine as se

    pbp = se.load_reg_seasons(POOL)
    tr = se.build_transition_frame(pbp).reset_index(drop=True)
    cols = [c for c in ("game_id", "play_id", "complete_pass", "season") if c in pbp.columns]
    tr = tr.merge(pbp[cols].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
    code = tr["play_type_code"].to_numpy()
    cp = tr["complete_pass"].fillna(0).to_numpy() == 1
    cls = np.where(code == 0, 1, np.where((code == 1) & cp, 2, 0))
    scored = ((tr["points_off"] > 0) | (tr["points_def"] > 0)).to_numpy()
    inb = (cls > 0) & ~scored & ~tr["possession_flip"].to_numpy().astype(bool)
    sel = inb & late_mask(tr["phase"].to_numpy(), tr["sc_raw"].to_numpy(), tr["off_to_raw"].to_numpy(), tr["time_raw"].to_numpy())
    return tr, code, inb, sel


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e6, max_iter=500).fit(X, y)
    return {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}


def predict(spec, X):
    z = X @ np.asarray(spec["coef"]) + spec["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def fit():
    tr, code, inb, sel = frame()
    season = tr["season"].to_numpy()
    X = design(tr["qtr_actual"].to_numpy(), tr["gsr_actual"].to_numpy(), tr["sc_raw"].to_numpy(), tr["off_to_raw"].to_numpy(), tr["def_to_raw"].to_numpy(), code)
    y = (tr["off_to_used"].to_numpy() > 0).astype(int)
    out = {"features": list(FEATS), "n": int(sel.sum()), "rate": float(y[sel].mean()), "fits": {}}
    for s in POOL:
        m = sel & (season != s)
        out["fits"][str(s)] = fit_lr(X[m], y[m])
    out["fits"]["all"] = fit_lr(X[sel], y[sel])
    OUTD.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


class Donors:
    def __init__(self, arrays, stop, mask_extra=None):
        import sim04_engine as se
        from scipy.spatial import cKDTree

        code = np.asarray(arrays["play_type_code"])
        ok = late_mask(arrays["phase"], arrays["sc_raw"], arrays["off_to_raw"], arrays["time_raw"]) & np.isin(code, (0, 1)) & (np.asarray(stop) == 0)
        if mask_extra is not None:
            ok &= mask_extra
        feat = se.feature_matrix(arrays["dist_raw"], arrays["fp_raw"], arrays["sc_raw"], arrays["time_raw"], arrays["off_to_raw"], arrays["def_to_raw"], arrays["phase"])
        y = np.asarray(arrays["off_to_used"]) > 0
        ph_a = np.asarray(arrays["phase"])
        self.trees = {}
        for ph in (1, 3):
            for c in (0, 1):
                for t in (0, 1):
                    ix = np.flatnonzero(ok & (ph_a == ph) & (code == c) & (y == bool(t)))
                    if len(ix):
                        self.trees[(ph, c, t)] = (cKDTree(feat[ix]), ix)

    def draw(self, ph, c, t, f, rng, k):
        tt = self.trees.get((ph, c, t))
        if tt is None:
            return None
        tree, ix = tt
        _, nb = tree.query(f, k=min(k, len(ix)))
        nb = np.atleast_1d(nb)
        return int(ix[nb[int(rng.integers(len(nb)))]])


def install_toh():
    import mod25d_variance as dv
    import sim04_engine as se

    if getattr(dv._G["pol"], "_toh", False):
        return
    spec = json.loads(FIT.read_text(encoding="utf-8"))["fits"][season_key()]
    a = dv._G["tables"]["arrays"]
    stop = np.asarray(dv._G["f2h"]["stop"])
    don = Donors(a, stop)
    base = dv._G["pol"]
    seed = int(dv._G["cfg"].get("seed", 3))

    def locals_of():
        fr = sys._getframe(2)
        while fr is not None and not ("offense" in fr.f_locals and "off_sim" in fr.f_locals):
            fr = fr.f_back
        return fr.f_locals

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        COUNT["calls"] += 1
        i = drawn.get("idx")
        code = int(drawn["play_type_code"])
        if i is None or code not in (0, 1) or drawn["flip"] or drawn["points_off"] > 0 or drawn["points_def"] > 0 or stop[int(i)] != 0:
            return drawn
        ph = se.compute_phase(qtr, clock_val)
        tf = se.continuous_time_feature(qtr, clock_val)
        L = locals_of()
        off_to = L["home_to"] if L["offense"] == "home" else L["away_to"]
        def_to = L["away_to"] if L["offense"] == "home" else L["home_to"]
        if not late_mask(np.array([ph]), np.array([score_diff]), np.array([off_to]), np.array([tf]))[0]:
            return drawn
        COUNT["eligible"] += 1
        if _STATE["k"] != dv._G.get("task_key"):
            _STATE["k"] = dv._G.get("task_key")
            _STATE["rng"] = dv.task_rng("toh", seed)
        r = _STATE["rng"]
        p = float(predict(spec, design([qtr], [clock_val], [score_diff], [off_to], [def_to], [code]))[0])
        t = int(r.random() < p)
        f = se.feature_matrix(np.array([distance]), np.array([float(yardline)]), np.array([score_diff]), np.array([tf]), np.array([off_to]), np.array([def_to]), np.array([ph]))[0]
        k = don.draw(ph, code, t, f, r, se.K_STATE)
        if k is None:
            COUNT["no_donor"] += 1
            return drawn
        COUNT["called"] += t
        new = dict(drawn)
        new["clock_elapsed"] = a["clock_elapsed"][k]
        new["off_to_used"] = a["off_to_used"][k]
        new["def_to_used"] = a["def_to_used"][k]
        return new

    pol._toh = True
    dv._G["pol"] = pol


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def replay():
    import pandas as pd
    import sim04_engine as se

    spec = json.loads(FIT.read_text(encoding="utf-8"))["fits"]
    tr, code, inb, sel = frame()
    season = tr["season"].to_numpy()
    arrays = {
        "play_type_code": code, "phase": tr["phase"].to_numpy(), "sc_raw": tr["sc_raw"].to_numpy(), "off_to_raw": tr["off_to_raw"].to_numpy(),
        "def_to_raw": tr["def_to_raw"].to_numpy(), "time_raw": tr["time_raw"].to_numpy(), "dist_raw": tr["dist_raw"].to_numpy(), "fp_raw": tr["fp_raw"].to_numpy(),
        "off_to_used": tr["off_to_used"].to_numpy(),
    }
    stop = (~inb).astype(int)
    X = design(tr["qtr_actual"].to_numpy(), tr["gsr_actual"].to_numpy(), tr["sc_raw"].to_numpy(), tr["off_to_raw"].to_numpy(), tr["def_to_raw"].to_numpy(), code)
    feat = se.feature_matrix(arrays["dist_raw"], arrays["fp_raw"], arrays["sc_raw"], arrays["time_raw"], arrays["off_to_raw"], arrays["def_to_raw"], arrays["phase"])
    rng = np.random.default_rng(0)
    clock = tr["clock_elapsed"].to_numpy()
    yreal = (tr["off_to_used"].to_numpy() > 0).astype(int)
    rows = []
    for s in POOL:
        d = Donors(arrays, stop, season != s)
        q = np.flatnonzero(sel & (season == s))
        p = predict(spec[str(s)], X[q])
        for n, i in enumerate(q):
            t = int(rng.random() < p[n])
            k = d.draw(int(arrays["phase"][i]), int(code[i]), t, feat[i], rng, se.K_STATE)
            rows.append((s, i, p[n], t if k is None else int(arrays["off_to_used"][k] > 0), np.nan if k is None else clock[k]))
    r = pd.DataFrame(rows, columns=["season", "i", "p", "y_new", "clock_new"])
    r["y_real"] = yreal[r.i]
    r["clock_real"] = clock[r.i]
    half = np.where(tr["qtr_actual"].to_numpy() == 2, tr["gsr_actual"].to_numpy() - 1800, tr["gsr_actual"].to_numpy())
    r["hs"] = pd.cut(half[r.i], [0, 30, 60, 120, 300])
    print("overall real %.4f toh %.4f mean_p %.4f n %d no_donor %d" % (r.y_real.mean(), r.y_new.mean(), r.p.mean(), len(r), int(r.clock_new.isna().sum())))
    print("hs n real [lo,hi] toh mean_p clock_real(y=1) clock_new(y=1) clock_real(y=0) clock_new(y=0)")
    for h, g in r.groupby("hs", observed=True):
        lo, hi = wilson(g.y_real.sum(), len(g))
        print(h, len(g), "%.3f [%.3f,%.3f] %.3f %.3f | %.1f %.1f | %.1f %.1f" % (g.y_real.mean(), lo, hi, g.y_new.mean(), g.p.mean(), g.clock_real[g.y_real == 1].mean(), g.clock_new[g.y_new == 1].mean(), g.clock_real[g.y_real == 0].mean(), g.clock_new[g.y_new == 0].mean()))
    pc = np.clip(r.p.to_numpy(), 1e-4, 1 - 1e-4)
    print("LOSO ll %.4f" % float(np.mean(r.y_real * np.log(pc) + (1 - r.y_real) * np.log(1 - pc))))
    print("per-season real/toh:", {int(s): (round(g.y_real.mean(), 3), round(g.y_new.mean(), 3)) for s, g in r.groupby("season")})
    print("in-bounds rows outside late scope (rule never fires):", int((~sel & inb).sum()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        replay()
    else:
        fit()
