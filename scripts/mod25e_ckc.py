import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "ckc" / "fit.json"
SALT = 4312
NB = 61
HZ = 45.0
KLASSES = (0, 1, 2, 3, 5)


def enabled():
    return os.environ.get("CKC") == "1"


def rows():
    import mod25e_clk as ck

    R = ck.real_rows()
    R = R[R.cls.isin(KLASSES) & (R.hs <= HZ)].copy()
    R["cens"] = R.el >= R.hs
    return R.reset_index(drop=True)


def obs(R):
    e = np.minimum(np.round(R.el.to_numpy(float)).astype(int), NB)
    cb = np.minimum(np.ceil(R.hs.to_numpy(float)).astype(int), NB)
    ev = np.where(R.cens.to_numpy() | (e >= NB), -1, e)
    last = np.where(ev >= 0, ev, cb - 1)
    return ev, cb, last


def km_pmf(ci, ev, last, nc):
    D = np.zeros((nc, NB))
    H = np.zeros((nc, NB))
    m = ev >= 0
    np.add.at(D, (ci[m], ev[m]), 1.0)
    np.add.at(H, (ci, last), 1.0)
    N = np.cumsum(H[:, ::-1], axis=1)[:, ::-1]
    hz = np.where(N > 0, D / np.maximum(N, 1.0), 0.0)
    surv = np.cumprod(1 - hz, axis=1)
    prev = np.concatenate([np.ones((nc, 1)), surv[:, :-1]], axis=1)
    return np.concatenate([prev * hz, surv[:, -1:]], axis=1)


def smooth_pm(cn, pc, ph, a):
    return (cn[:, None] * pc + a * ph) / (cn[:, None] + a)


def fold_rows(sub, spec, a, mode):
    import mod25e_clk as ck

    edges = ck.EDGES[spec["edges"]]
    c = ck.cell(sub.hs.to_numpy(), sub.sd.to_numpy(), sub.oto.to_numpy(), sub.dto.to_numpy(), edges, spec["sign"], spec["to"])
    hb = ck.hbin(sub.hs.to_numpy(), edges)
    ev, cb, last = obs(sub)
    ss = sub.season.to_numpy()
    uc, ci = np.unique(c, return_inverse=True)
    nh = len(edges) + 1
    hbc = np.zeros(len(uc), int)
    hbc[ci] = hb
    out = np.zeros(len(sub))
    for s in np.unique(ss):
        te, tr = ss == s, ss != s
        cn = np.bincount(ci[tr], minlength=len(uc)).astype(float)
        hn = np.bincount(hb[tr], minlength=nh).astype(float)
        if mode == "km":
            pc = km_pmf(ci[tr], ev[tr], last[tr], len(uc))
            ph = km_pmf(hb[tr], ev[tr], last[tr], nh)
            ph = (hn[:, None] * ph + 1.0 / (NB + 1)) / (hn[:, None] + 1.0)
        else:
            elb = np.where(ev >= 0, ev, np.minimum(cb, NB))
            cnt = np.zeros((len(uc), NB + 1))
            np.add.at(cnt, (ci[tr], elb[tr]), 1.0)
            mh = np.zeros((nh, NB + 1))
            np.add.at(mh, (hb[tr], elb[tr]), 1.0)
            pc = cnt / np.maximum(cn[:, None], 1.0)
            ph = (mh + 1.0) / (mh.sum(axis=1, keepdims=True) + NB + 1)
        pm = smooth_pm(cn, pc, ph[hbc], a)
        tail = np.cumsum(pm[:, ::-1], axis=1)[:, ::-1]
        out[te] = np.where(ev[te] >= 0, np.log(pm[ci[te], np.maximum(ev[te], 0)]), np.log(tail[ci[te], cb[te]]))
    return out, ss


def fit():
    import mod25e_clk as ck

    R = rows()
    spec = json.loads(ck.FIT.read_text(encoding="utf-8"))["classes"]
    out = {"classes": {}, "looks": 0}
    for k in KLASSES:
        nm = ck.CLASSES[k]
        sub = R[R.cls == k].reset_index(drop=True)
        b = spec[nm]["best"]
        res = []
        for a in ck.SMOOTH:
            lk, ss = fold_rows(sub, b, a, "km")
            lp, _ = fold_rows(sub, b, a, "pool")
            sg = [float((lk[ss == s] - lp[ss == s]).sum()) for s in np.unique(ss)]
            res.append({"a": a, "ll_km": float(lk.mean()), "ll_pool": float(lp.mean()), "d": lk - lp, "seasons_pos": int(sum(x > 0 for x in sg)), "seasons": len(sg)})
        out["looks"] += 2 * len(res)
        bk = max(res, key=lambda r: r["ll_km"])
        bp = max(res, key=lambda r: r["ll_pool"])
        se = float(np.std(bk["d"], ddof=1) / np.sqrt(len(bk["d"])))
        gain = bk["ll_km"] - bp["ll_pool"]
        out["classes"][nm] = {"k": k, "n": int(len(sub)), "cens": float(sub.cens.mean()), "edges": b["edges"], "sign": b["sign"], "to": b["to"], "a": bk["a"], "a_pool": bp["a"], "ll_km": bk["ll_km"], "ll_pool": bp["ll_pool"], "gain": gain, "seasons_pos": bk["seasons_pos"], "seasons": bk["seasons"], "apply": bool(gain > 0)}
        print(nm, len(sub), "cens", round(float(sub.cens.mean()), 4), "km", round(bk["ll_km"], 4), "a", bk["a"], "pool", round(bp["ll_pool"], 4), "a", bp["a"], "gain", round(gain, 4), "se_same_a", round(se, 4), "seasons+", bk["seasons_pos"], "/", bk["seasons"], flush=True)
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"])


def tables():
    import mod25e_clk as ck

    spec = json.loads(FIT.read_text(encoding="utf-8"))["classes"]
    R = rows()
    T = {}
    for nm, s in spec.items():
        if not s["apply"]:
            continue
        k = s["k"]
        sub = R[R.cls == k].reset_index(drop=True)
        edges = ck.EDGES[s["edges"]]
        c = ck.cell(sub.hs.to_numpy(), sub.sd.to_numpy(), sub.oto.to_numpy(), sub.dto.to_numpy(), edges, s["sign"], s["to"])
        hb = ck.hbin(sub.hs.to_numpy(), edges)
        ev, cb, last = obs(sub)
        uc, ci = np.unique(c, return_inverse=True)
        nh = len(edges) + 1
        cn = np.bincount(ci, minlength=len(uc)).astype(float)
        hn = np.bincount(hb, minlength=nh).astype(float)
        pc = km_pmf(ci, ev, last, len(uc))
        ph = km_pmf(hb, ev, last, nh)
        ph = (hn[:, None] * ph + 1.0 / (NB + 1)) / (hn[:, None] + 1.0)
        hbc = np.zeros(len(uc), int)
        hbc[ci] = hb
        pm = smooth_pm(cn, pc, ph[hbc], s["a"])
        T[k] = {"edges": edges, "sign": s["sign"], "to": s["to"], "cdf": {int(cv): cdf_of(pm[i]) for i, cv in enumerate(uc)}, "hcdf": {h: cdf_of(ph[h]) for h in range(nh)}}
    return T


def cdf_of(pmf):
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    return cdf


def install_ckc():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg
    import sim09_u4g as u4

    T = tables()
    probs = u4.load_probs()
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        offense, off_to, def_to = fr.f_locals["offense"], fr.f_locals["off_to"], fr.f_locals["def_to"]  # noqa: F841
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        if qtr not in (2, 4) or code not in (0, 1, 5):
            return drawn
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if hs <= 0 or hs > HZ:
            return drawn
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        t = T.get(k)
        if t is None:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        rng = st["rng"]
        u = rng.random()
        cv = int(ck.cell([hs], [score_diff], [float(off_to)], [float(def_to)], t["edges"], t["sign"], t["to"])[0])
        cdf = t["cdf"].get(cv)
        if cdf is None:
            cdf = t["hcdf"][int(ck.hbin([hs], t["edges"])[0])]
        b = int(min(np.searchsorted(cdf, u, side="left"), NB))
        new = float(min(b, hs))
        ends = bool(drawn["flip"]) or (drawn["points_off"] + drawn["points_def"]) > 0
        w = u4.WARN_AT.get(qtr)
        if w is not None and clock_val > w and clock_val - new < w and rng.random() < probs[u4.cell_key(qtr, code, ends)]:
            new = float(clock_val - w)
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
