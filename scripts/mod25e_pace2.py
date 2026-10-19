import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "pace2" / "fit.json"
CDW_FIT = REPO / "artifacts" / "mod25e3" / "cdw" / "fit.json"
CLQ_FIT = REPO / "artifacts" / "mod25e3" / "clq" / "fit.json"
CLK_FIT = REPO / "artifacts" / "mod25e3" / "clk" / "fit.json"
SALT = 4351
Q3_OFFSET = 900.0
Q3_START = 1800.0
Q4_HI = 900.0
Q4_LO = 420.0
FEATS = ("none", "down", "dist", "downdist", "band7", "downband")
GROUPS = (("q3", 0), ("q3", 1), ("q3", 2), ("q3", 3), ("q4m", 0), ("q4m", 1))
MIN_WINS = 5
NB = 62
SMOOTH = (2.0, 8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0)
NK = 4 * 7


def enabled():
    return os.environ.get("PACE2") == "1"


def band7(sd):
    sd = np.asarray(sd, float)
    return np.select([sd <= -17, sd <= -9, sd < 0, sd == 0, sd < 9, sd < 17], [0, 1, 2, 3, 4, 5], 6)


def dband(dist, cuts):
    return np.searchsorted(np.asarray(cuts, float), np.asarray(dist, float), side="left").clip(0, 2)


def feat_code(feat, down, dist, sd, cuts):
    d = np.clip(np.asarray(down, int), 1, 4) - 1
    s = dband(dist, cuts)
    b = band7(sd)
    if feat == "none":
        return np.zeros(len(d), int)
    if feat == "down":
        return d
    if feat == "dist":
        return s
    if feat == "downdist":
        return d * 3 + s
    if feat == "band7":
        return b
    return d * 7 + b


def domain_rows(dom):
    import pandas as pd

    import mod25e_clk as ck

    R = pd.read_parquet(ck.REAL)
    R = R[R.code.isin([0, 1]) & (R.penalty != 1)].copy()
    if dom == "q3":
        R = R[R.qtr == 3].copy()
        R["hs"] = R.gsr.astype(float) - Q3_OFFSET
        R = R[(R.hs > 0) & (R.gsr < Q3_START)]
        R["oT"], R["dT"] = R.otu, R.dtu
    else:
        R = R[R.qtr == 4].copy()
        R["hs"] = R.gsr.astype(float)
        R = R[(R.hs > Q4_LO) & (R.hs <= Q4_HI)]
        R["oT"], R["dT"] = R.oto, R.dto
    R = R[(R.el >= 0) & (R.el <= R.hs)].copy()
    R["el"] = R.el.astype(float)
    R["cls"] = ck.klass(R.code, R.yards, R.flip, R.po, R.pdf)
    return R[R.cls >= 0].reset_index(drop=True)


def parent_spec(dom, k):
    import mod25e_clk as ck

    nm = ck.CLASSES[k]
    path = CLQ_FIT if dom == "q3" else CLK_FIT
    b = json.loads(path.read_text(encoding="utf-8"))["classes"][nm]["best"]
    return {"edges": b["edges"], "sign": b["sign"], "to": b["to"], "a": float(b["a"])}


def fold_cuts(CS, seasons, excl):
    keep = ~np.isin(CS[1], [seasons[i] for i in excl])
    dd = CS[0][keep]
    return [float(np.quantile(dd, 1 / 3)), float(np.quantile(dd, 2 / 3))]


def cc_for(S, excl):
    key = tuple(sorted(excl))
    if key in S["cache"]:
        return S["cache"][key]
    cuts = fold_cuts(S["CS"], S["seasons"], key)
    Q = S["Q"]
    CC = {}
    for f in FEATS[1:]:
        fc = feat_code(f, Q.down.to_numpy(), Q.dist.to_numpy(), Q.sd.to_numpy(), cuts)
        cid = S["pi"] * NK + fc
        A = np.zeros((S["ns"], len(S["ucp"]) * NK, NB))
        np.add.at(A, (S["si"], cid, S["b"]), 1.0)
        CC[f] = (cid, A)
    if len(S["cache"]) > 4:
        S["cache"].clear()
    S["cache"][key] = CC
    return CC


def prep(dom, k, R, CS):
    import mod25e_clk as ck

    Q = R[R.cls == k].reset_index(drop=True)
    sp = parent_spec(dom, k)
    ed = ck.EDGES[sp["edges"]]
    cp = ck.cell(Q.hs.to_numpy(), Q.sd.to_numpy(), Q.oT.to_numpy(), Q.dT.to_numpy(), ed, sp["sign"], sp["to"])
    hb = ck.hbin(Q.hs.to_numpy(), ed)
    ucp, pi = np.unique(cp, return_inverse=True)
    uh, hi = np.unique(hb, return_inverse=True)
    gof = np.zeros(len(ucp), int)
    gof[pi] = hi
    b = ck.ebin(Q.el.to_numpy())
    seasons = np.unique(Q.season.to_numpy())
    si = np.searchsorted(seasons, Q.season.to_numpy())
    ns = len(seasons)
    CP = np.zeros((ns, len(ucp), NB))
    np.add.at(CP, (si, pi, b), 1.0)
    CH = np.zeros((ns, len(uh), NB))
    np.add.at(CH, (si, hi, b), 1.0)
    return {"Q": Q, "sp": sp, "pi": pi, "hi": hi, "gof": gof, "b": b, "si": si, "ns": ns, "seasons": seasons, "CP": CP, "CH": CH, "cache": {}, "CS": CS, "ucp": ucp}


def probs(S, excl, feat, a):
    keep = np.ones(S["ns"], bool)
    keep[list(excl)] = False
    ap = S["sp"]["a"]
    CH = S["CH"][keep].sum(0)
    ph = (CH + 1.0) / (CH.sum(1, keepdims=True) + NB)
    CP = S["CP"][keep].sum(0)
    npar = CP.sum(1, keepdims=True)
    Pp = (CP + ap * ph[S["gof"]]) / (npar + ap)
    if feat == "none":
        return Pp, Pp
    cid, A = cc_for(S, excl)[feat]
    AC = A[keep].sum(0)
    n = AC.sum(1, keepdims=True)
    par = np.repeat(Pp, NK, axis=0)
    Pc = (AC + a * par) / (n + a)
    return Pp, Pc


def score(S, te, Pp, Pc, feat, excl=()):
    b = S["b"][te]
    if feat == "none":
        P = Pp[S["pi"][te]]
    else:
        P = Pc[cc_for(S, excl)[feat][0][te]]
    return np.log(P[np.arange(len(b)), b]), P


def fit_group(dom, k, R, CS):
    S = prep(dom, k, R, CS)
    ns = S["ns"]
    cfgs = [("none", 0.0)] + [(f, a) for f in FEATS[1:] for a in SMOOTH]
    n = len(S["b"])
    sc = np.zeros(len(cfgs))
    inner = np.zeros((ns, ns, len(cfgs)))
    for s in range(ns):
        te = S["si"] == s
        for ci, (f, a) in enumerate(cfgs):
            Pp, Pc = probs(S, (s,), f, a)
            sc[ci] += score(S, te, Pp, Pc, f, (s,))[0].sum()
        for e in range(ns):
            if e == s:
                continue
            te = S["si"] == e
            for ci, (f, a) in enumerate(cfgs):
                Pp, Pc = probs(S, (s, e), f, a)
                inner[s, e, ci] = score(S, te, Pp, Pc, f, (s, e))[0].sum()
    d_all = np.zeros(n)
    ll_cur = np.zeros(n)
    mae_c = np.zeros(n)
    mae_n = np.zeros(n)
    ebin = np.arange(NB, dtype=float)
    y = np.minimum(S["Q"].el.to_numpy(), NB - 1)
    chosen = {}
    for s in range(ns):
        ci = int(np.argmax(inner[s].sum(0)))
        f, a = cfgs[ci]
        chosen[int(S["seasons"][s])] = [f, float(a)]
        te = np.where(S["si"] == s)[0]
        Pp, Pc = probs(S, (s,), "none", 0.0)
        l0, P0 = score(S, te, Pp, Pc, "none")
        Pp, Pc = probs(S, (s,), f, a)
        l1, P1 = score(S, te, Pp, Pc, f, (s,))
        d_all[te] = l1 - l0
        ll_cur[te] = l0
        mae_c[te] = np.abs(P0 @ ebin - y[te])
        mae_n[te] = np.abs(P1 @ ebin - y[te])
    g = S["Q"].game_id.to_numpy()
    ug, inv = np.unique(g, return_inverse=True)
    gs = np.bincount(inv, weights=d_all)
    ms = np.bincount(inv, weights=mae_c - mae_n)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(gs), (2000, len(gs)))
    bs = gs[idx].sum(1) / n
    bm = ms[idx].sum(1) / n
    fold = [float(d_all[S["si"] == s].mean()) for s in range(ns)]
    wins = int(sum(1 for x in fold if x > 0))
    f, a = cfgs[int(np.argmax(sc))]
    Pp, Pc = probs(S, (), f, a)
    lall, _ = score(S, np.arange(n), Pp, Pc, f, ())
    l0all, _ = score(S, np.arange(n), Pp, Pp, "none")
    bands = {}
    bd = band7(S["Q"].sd.to_numpy())
    for j in range(7):
        m = bd == j
        if m.sum():
            bands[str(j)] = [int(m.sum()), float(d_all[m].mean()), float((mae_c - mae_n)[m].mean())]
    return {
        "dom": dom,
        "k": k,
        "n": int(n),
        "parent": S["sp"],
        "per_fold_choice": chosen,
        "nested_gain_ll": float(d_all.mean()),
        "boot90_ll": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))],
        "probability_positive_ll": float((bs > 0).mean()),
        "mae_gain": float((mae_c - mae_n).mean()),
        "boot90_mae": [float(np.quantile(bm, 0.05)), float(np.quantile(bm, 0.95))],
        "probability_positive_mae": float((bm > 0).mean()),
        "fold_gain": fold,
        "wins": wins,
        "insample_gain_ll": float((lall - l0all).mean()),
        "full_cfg": [f, float(a)],
        "ll_cur": float(ll_cur.mean()),
        "by_band7": bands,
        "apply": bool(f != "none" and d_all.sum() > 0 and wins >= MIN_WINS),
        "looks": len(cfgs),
    }


def fit():
    out = {"groups": {}, "q3_offset": Q3_OFFSET, "q4_lo": Q4_LO, "q4_hi": Q4_HI}
    allR = {d: domain_rows(d) for d in ("q3", "q4m")}
    CS = (allR["q3"].dist.to_numpy(), allR["q3"].season.to_numpy())
    cuts = [float(np.quantile(CS[0], 1 / 3)), float(np.quantile(CS[0], 2 / 3))]
    out["dist_cuts"] = cuts
    out["dist_cuts_provenance"] = "served: tercile cuts over all seasons of q3 rows; held-out scoring used cuts recomputed from training seasons inside each fold"
    look = 0
    keys = ("n", "nested_gain_ll", "boot90_ll", "probability_positive_ll", "mae_gain", "boot90_mae", "probability_positive_mae", "wins", "insample_gain_ll", "full_cfg", "apply")
    for dom, k in GROUPS:
        r = fit_group(dom, k, allR[dom], CS)
        look += r["looks"]
        out["groups"][f"{dom}_{k}"] = r
        print(dom, k, json.dumps({x: r[x] for x in keys}), flush=True)
    out["looks"] = look
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")


def install_pace2():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    fit_ = json.loads(FIT.read_text(encoding="utf-8"))
    cuts = fit_["dist_cuts"]
    cdw_hz = 0.0
    if os.environ.get("CDW") in ("1", "2"):
        cdw_hz = float(json.loads(CDW_FIT.read_text(encoding="utf-8"))["chosen"]["spec"]["hz"])
    q4lo = max(fit_["q4_lo"], cdw_hz)
    plan = {}
    for cf in fit_["groups"].values():
        if not cf["apply"]:
            continue
        dom, k = cf["dom"], cf["k"]
        R = domain_rows(dom)
        Q = R[R.cls == k]
        sp = cf["parent"]
        ed = ck.EDGES[sp["edges"]]
        cp = ck.cell(Q.hs.to_numpy(), Q.sd.to_numpy(), Q.oT.to_numpy(), Q.dT.to_numpy(), ed, sp["sign"], sp["to"])
        hb = ck.hbin(Q.hs.to_numpy(), ed)
        feat, a = cf["full_cfg"]
        fc = feat_code(feat, Q.down.to_numpy(), Q.dist.to_numpy(), Q.sd.to_numpy(), cuts)
        el = Q.el.to_numpy(float)
        cid = cp * NK + fc
        ch = {int(v): el[cid == v] for v in np.unique(cid)}
        pa = {int(v): el[cp == v] for v in np.unique(cp)}
        mg = {int(v): el[hb == v] for v in np.unique(hb)}
        plan[(dom, k)] = {"feat": feat, "a": float(a), "ap": float(sp["a"]), "sp": sp, "ed": ed, "ch": ch, "pa": pa, "mg": mg}
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        while "offense" not in fr.f_locals:
            fr = fr.f_back
        off_to, def_to = fr.f_locals["off_to"], fr.f_locals["def_to"]
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        if qtr not in (3, 4) or code not in (0, 1):
            return drawn
        if qtr == 3:
            if clock_val >= Q3_START:
                return drawn
            hs = clock_val - Q3_OFFSET
            if hs <= 0:
                return drawn
            dom = "q3"
            oT, dT = float(drawn["off_to_used"]), float(drawn["def_to_used"])
        else:
            hs = clock_val
            if hs <= q4lo or hs > Q4_HI:
                return drawn
            dom = "q4m"
            oT, dT = float(off_to), float(def_to)
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        pl = plan.get((dom, k))
        if pl is None:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        rng = st["rng"]
        sp = pl["sp"]
        cv = int(ck.cell([hs], [score_diff], [oT], [dT], pl["ed"], sp["sign"], sp["to"])[0])
        hv = int(ck.hbin([hs], pl["ed"])[0])
        fcv = int(feat_code(pl["feat"], [down], [distance], [score_diff], cuts)[0])
        u1, u2, v = rng.random(), rng.random(), rng.random()
        c_ = pl["ch"].get(cv * NK + fcv)
        p_ = pl["pa"].get(cv)
        if c_ is not None and u1 < len(c_) / (len(c_) + pl["a"]):
            src = c_
        elif p_ is not None and u2 < len(p_) / (len(p_) + pl["ap"]):
            src = p_
        else:
            src = pl["mg"][hv]
        new = float(min(src[int(v * len(src))], hs))
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
