import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "artifacts" / "mod25e3" / "pto"
FIT = OUT / "fit.json"
TIME_EDGES = (60.0, 120.0, 240.0)
DIST_SHORT = 3.0
DIST_LONG = 10.0
FP_EDGES = (10.0, 20.0, 50.0)
SCORE_EDGE = 8.0
PASS_CODE = 1
DOWNS = (1, 2, 3)
STATE = {"fit": None, "cls": None, "count": np.zeros(2), "fallback": 0}
NAMES_FULL = None


def enabled():
    on = os.environ.get("PTO") == "1"
    if on and os.environ.get("CLSDRAW") == "1":
        raise RuntimeError("PTO and CLSDRAW both set; the pass-turnover draw would be applied twice")
    return on


def turnover_mask(a):
    return (a["possession_flip"].astype(bool) & (a["points_off"] < 6)) | (a["points_def"] >= 6)


def score_bands(sc):
    sc = np.asarray(sc, dtype=float)
    return {
        "tr1_8": (sc < 0) & (sc >= -SCORE_EDGE),
        "tr9p": sc < -SCORE_EDGE,
        "ld1_8": (sc > 0) & (sc <= SCORE_EDGE),
        "ld9p": sc > SCORE_EDGE,
    }


def time_bins(t):
    t = np.asarray(t, dtype=float)
    return {
        "t60": t <= TIME_EDGES[0],
        "t120": (t > TIME_EDGES[0]) & (t <= TIME_EDGES[1]),
        "t240": (t > TIME_EDGES[1]) & (t <= TIME_EDGES[2]),
    }


def features(down, dist, fp, score, qtr, tsec, o_rt, d_rt, clock=True):
    down = np.atleast_1d(down)
    dist = np.atleast_1d(dist).astype(float)
    fp = np.atleast_1d(fp).astype(float)
    cols, names = [], []

    def add(nm, v):
        names.append(nm)
        cols.append(np.asarray(v, dtype=float))

    add("d2", down == 2)
    add("d3", down == 3)
    add("dist_le3", dist <= DIST_SHORT)
    add("dist_ge10", dist >= DIST_LONG)
    add("fp_le10", fp <= FP_EDGES[0])
    add("fp_11_20", (fp > FP_EDGES[0]) & (fp <= FP_EDGES[1]))
    add("fp_gt50", fp > FP_EDGES[2])
    sb = score_bands(score)
    for k, v in sb.items():
        add(k, v)
    if clock:
        tb = time_bins(tsec)
        h2 = np.atleast_1d(qtr) >= 3
        for tk, tv in tb.items():
            add(tk, tv)
            for sk, sv in sb.items():
                add(tk + "_" + sk, tv & sv)
            add(tk + "_h2", tv & h2)
    add("off", o_rt)
    add("def", d_rt)
    return np.column_stack(cols), names


def prob(fit, X):
    Z = (X - np.asarray(fit["mu"])) / np.asarray(fit["sd"])
    z = Z @ np.asarray(fit["coef"]) + fit["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def load_fit():
    if STATE["fit"] is None:
        STATE["fit"] = json.loads(FIT.read_text(encoding="utf-8"))
    return STATE["fit"]


def install_pto():
    import mod25e_geo as ge
    import sim04_engine as sim

    if os.environ.get("CLSDRAW") == "1":
        raise RuntimeError("PTO and CLSDRAW both set; the pass-turnover draw would be applied twice")
    if getattr(ge.Geo.fix, "_pto", False):
        return
    if getattr(ge.Geo.fix, "_clsdraw", False):
        raise RuntimeError("CLSDRAW fix already installed; PTO would stack on it")
    fit = load_fit()
    orig = ge.Geo.fix

    def rating_frame():
        fr = sys._getframe(2)
        while fr is not None and not ("offense" in fr.f_locals and "off_sim" in fr.f_locals):
            fr = fr.f_back
        return fr.f_locals

    def play_frame():
        fr = sys._getframe(2)
        while fr is not None and not ("clock_val" in fr.f_locals and "score_diff" in fr.f_locals and "qtr" in fr.f_locals):
            fr = fr.f_back
        return fr.f_locals

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        if not force and not self.violates(drawn, i, q):
            return None
        code = int(drawn["play_type_code"])
        pf = play_frame()
        qtr = int(pf["qtr"])
        if code != PASS_CODE or down not in DOWNS or qtr >= 5:
            return orig(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)
        P = self.pools.get((down, code))
        if P is None:
            return None
        pos = self.candidates(P, q, net_sim, float(f3q[0]))
        if len(pos) == 0:
            return None
        if STATE["cls"] is None:
            STATE["cls"] = turnover_mask(self.a)
        to = STATE["cls"]
        k = min(self.k, len(pos))
        dd = ((P["f3"][pos] - f3q[None, :]) ** 2).sum(axis=1)
        near = pos[np.argpartition(dd, k - 1)[:k]] if k < len(pos) else pos
        L = rating_frame()
        tsec = sim.continuous_time_feature(qtr, float(pf["clock_val"]))
        X, _ = features(down, float(pf["distance"]), q, float(pf["score_diff"]), qtr, tsec, L["off_sim"], L["def_sim"])
        p = float(prob(fit, X)[0])
        tn = to[P["ids"][near]]
        tw = to[P["ids"][pos]]
        want = bool(rng.random() < p)
        if (tn == want).any():
            sel = near[tn == want]
        elif (tw == want).any():
            sel = pos[tw == want]
        else:
            STATE["fallback"] += 1
            return orig(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)
        STATE["count"][int(want)] += 1
        ids = P["ids"][sel]
        w = weight_fn(ids)
        cum = np.cumsum(w)
        if not np.isfinite(cum[-1]) or cum[-1] <= 0.0:
            jp = int(rng.integers(len(ids)))
        else:
            jp = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), len(ids) - 1)
        j = int(ids[jp])
        jpos = int(sel[jp])
        shift = self.bias + self.gain * (net_sim - float(self.a["off_row"][j] + self.sign * self.a["def_row"][j]))
        new = dict(drawn)
        for key in (k_ for k_ in ge.REPICK_KEYS if k_ not in ge.ENGINE_KEEP):
            new[key] = self.a[key][j]
        new["flip"] = bool(self.a["possession_flip"][j])
        new["auto_first"] = bool(self.a["auto_first"][j])
        new["repeat_down"] = bool(self.a["repeat_down"][j])
        new["yards_gained"] = ge.sim_dither(self.a["yards_gained"][j], shift, rng) if os.environ.get("YDITH") == "1" else self.a["yards_gained"][j] + shift
        new["idx"] = j
        if os.environ.get("FLIPA") == "1" and new["flip"]:
            new["next_yardline"] = float(min(max(float(self.a["next_yardline"][j]) + (float(P["fp"][jpos]) - float(q)) + (float(new["yards_gained"]) - float(self.a["yards_gained"][j])), 1.0), 99.0))
        if os.environ.get("GEOX") in ("1", "2") and not P["td"][jpos] and not P["free"][jpos] and float(self.a["yards_gained"][j]) + shift >= q:
            tdp = pos[P["td"][pos] & (P["fp"][pos] >= q)]
            if len(tdp) == 0:
                tdp = np.flatnonzero(P["td"] & (P["fp"] >= q))
            if len(tdp):
                t = int(P["ids"][tdp[int(rng.integers(len(tdp)))]])
                for key in (k_ for k_ in ge.REPICK_KEYS if k_ not in ge.ENGINE_KEEP):
                    new[key] = self.a[key][t]
                new["flip"] = bool(self.a["possession_flip"][t])
                new["auto_first"] = bool(self.a["auto_first"][t])
                new["repeat_down"] = bool(self.a["repeat_down"][t])
                new["yards_gained"] = float(q)
        return new, j, shift

    fix._pto = True
    ge.Geo.fix = fix
