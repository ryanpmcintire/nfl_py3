import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "cdr" / "fit.json"
SALT = 4311
TAIL = 60
NBIN = TAIL + 2
HORIZON = (60.0, 120.0)
EDGES = {"none": [], "w": [30.0], "ww": [20.0, 45.0], "w4": [15.0, 30.0, 60.0], "w5": [10.0, 20.0, 30.0, 45.0, 60.0]}
SCOPE = ("q4_trail", "both_trail", "q4_all")
SMOOTH = (0.5, 2.0, 8.0, 32.0, 128.0)


def enabled():
    return os.environ.get("CDR") == "1"


def rows():
    import mod25e_clk as ck

    R = ck.real_rows()
    R = R[R.code.isin([0, 1]) & (R.oto > 0)].copy()
    R["rem"] = (R.hs - R.el).clip(lower=0)
    return R.reset_index(drop=True)


def pick(R, scope, hz):
    m = (R.hs <= hz) & (R.otu > 0)
    if scope == "q4_trail":
        m &= (R.qtr == 4) & (R.sd <= 0)
    elif scope == "both_trail":
        m &= R.sd <= 0
    else:
        m &= R.qtr == 4
    return m.to_numpy()


def hb(hs, edges):
    return np.searchsorted(np.asarray(edges, float), np.asarray(hs, float), side="left") if len(edges) else np.zeros(len(hs), int)


def ebin(x):
    return np.minimum(np.round(np.asarray(x, float)).astype(int), TAIL + 1).clip(0)


def trunc_ll(cnt_row, b, hs_row, a):
    lim = np.minimum(np.round(hs_row).astype(int), TAIL + 1)
    cs = np.cumsum(cnt_row + a / NBIN)
    return float(np.log((cnt_row[b] + a / NBIN) / cs[lim]))


def fold(R, mk, edges, a, mode):
    hsv = R.hs.to_numpy()[mk]
    h = hb(hsv, edges)
    tgt = ebin(R.rem.to_numpy()[mk] if mode == "rem" else R.el.to_numpy()[mk])
    ss = R.season.to_numpy()[mk]
    tot, n = 0.0, 0
    for s in np.unique(ss):
        te, tr = ss == s, ss != s
        cnt = np.zeros((int(h.max()) + 1, NBIN))
        np.add.at(cnt, (h[tr], tgt[tr]), 1.0)
        for i in np.flatnonzero(te):
            tot += trunc_ll(cnt[h[i]], tgt[i], hsv[i], a)
            n += 1
    return tot / max(n, 1), n


def fit():
    R = rows()
    out = {"results": [], "looks": 0}
    for scope, hz in itertools.product(SCOPE, HORIZON):
        mk = pick(R, scope, hz)
        if mk.sum() < 40:
            continue
        for en, a in itertools.product(EDGES, SMOOTH):
            llr, n = fold(R, mk, EDGES[en], a, "rem")
            lle, _ = fold(R, mk, EDGES[en], a, "el")
            out["results"].append({"scope": scope, "hz": hz, "edges": en, "a": a, "n": int(n), "ll_rem": llr, "ll_el": lle})
            out["looks"] += 2
    best = max(out["results"], key=lambda r: r["ll_rem"] - r["ll_el"])
    out["best"] = best
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    top = sorted(out["results"], key=lambda r: -(r["ll_rem"] - r["ll_el"]))[:8]
    for r in top:
        print(r["scope"], r["hz"], r["edges"], r["a"], r["n"], "rem", round(r["ll_rem"], 4), "el", round(r["ll_el"], 4), "gain", round(r["ll_rem"] - r["ll_el"], 4))
    print("looks", out["looks"], "best", best)


def install_cdr():
    import mod25d_variance as dv
    import mod25e_endgame as eg

    b = json.loads(FIT.read_text(encoding="utf-8"))["best"]
    edges = EDGES[b["edges"]]
    R = rows()
    mk = pick(R, b["scope"], b["hz"])
    h = hb(R.hs.to_numpy()[mk], edges)
    rem = R.rem.to_numpy()[mk]
    pool = {int(v): rem[h == v] for v in np.unique(h)}
    allp = rem
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if code not in (0, 1) or float(drawn["off_to_used"]) <= 0 or hs <= 0 or hs > b["hz"] or qtr not in ((4,) if b["scope"] == "q4_trail" or b["scope"] == "q4_all" else (2, 4)):
            return drawn
        if b["scope"] != "q4_all" and score_diff > 0:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        src = pool.get(int(hb([hs], edges)[0]), allp)
        ok = src[src <= hs]
        if not len(ok):
            return drawn
        r = float(ok[int(st["rng"].random() * len(ok))])
        new = float(hs - r)
        old = float(drawn["clock_elapsed"])
        drawn = dict(drawn)
        drawn["clock_elapsed"] = new
        lg = dv._G.get("log")
        if lg:
            tp = list(lg[-1])
            tp[10] = new
            lg[-1] = tuple(tp)
        es = dv._G.get("egst")
        if es is not None and es["ps"][0] is not None and abs(es["ps"][0] - (clock_val - old)) < 1e-6:
            stn = float(eg.stop_after(code, float(drawn["yards_gained"]), bool(drawn["flip"]), (drawn["points_off"] + drawn["points_def"]) > 0, float(drawn["off_to_used"]), float(drawn["def_to_used"]), new))
            es["ps"] = (clock_val - new, stn)
        return drawn

    dv._G["pol"] = pol
