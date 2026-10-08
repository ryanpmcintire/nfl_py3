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


def hazards(c, hb, ev, last, nc, nh, a):
    D = np.zeros((nc, NB))
    H = np.zeros((nc, NB))
    m = ev >= 0
    np.add.at(D, (c[m], ev[m]), 1.0)
    np.add.at(H, (c, last), 1.0)
    N = np.cumsum(H[:, ::-1], axis=1)[:, ::-1]
    Dh = np.zeros((nh, NB))
    Nh = np.zeros((nh, NB))
    Dh_c = np.zeros((nh, NB))
    np.add.at(Dh_c, (hb[m], ev[m]), 1.0)
    Hh = np.zeros((nh, NB))
    np.add.at(Hh, (hb, last), 1.0)
    Nh = np.cumsum(Hh[:, ::-1], axis=1)[:, ::-1]
    Dh = Dh_c
    hg = np.clip(Dh.sum(axis=0) / np.maximum(Nh.sum(axis=0), 1.0), 1e-6, 1 - 1e-6)
    hh = (Dh + hg) / (Nh + 1.0)
    return D, N, hh, a


def cell_hz(D, N, hh_rows, a):
    return np.clip((D + a * hh_rows) / (N + a), 1e-6, 1 - 1e-6)


def ll_cens(hz, ev, cb):
    cum = np.concatenate([np.zeros((hz.shape[0], 1)), np.cumsum(np.log1p(-hz), axis=1)], axis=1)
    return cum, np.maximum(ev, 0)


def fold_ll(sub, spec, a, mode):
    import mod25e_clk as ck

    edges = ck.EDGES[spec["edges"]]
    c = ck.cell(sub.hs.to_numpy(), sub.sd.to_numpy(), sub.oto.to_numpy(), sub.dto.to_numpy(), edges, spec["sign"], spec["to"])
    hb = ck.hbin(sub.hs.to_numpy(), edges)
    ev, cb, last = obs(sub)
    ss = sub.season.to_numpy()
    uc, ci = np.unique(c, return_inverse=True)
    nh = len(edges) + 1
    tot, n = 0.0, 0
    for s in np.unique(ss):
        te, tr = ss == s, ss != s
        if mode == "haz":
            D, N, hh, _ = hazards(ci[tr], hb[tr], ev[tr], last[tr], len(uc), nh, a)
            hh_rows = np.zeros((len(uc), NB))
            hbc = np.zeros(len(uc), int)
            hbc[ci] = hb
            hh_rows = hh[hbc]
            hz = cell_hz(D, N, hh_rows, a)
            cum = np.concatenate([np.zeros((hz.shape[0], 1)), np.cumsum(np.log1p(-hz), axis=1)], axis=1)
            e = np.maximum(ev[te], 0)
            ll = np.where(ev[te] >= 0, cum[ci[te], e] + np.log(hz[ci[te], e]), cum[ci[te], cb[te]])
        else:
            elb = np.where(ev >= 0, ev, np.minimum(cb, NB))
            cnt = np.zeros((len(uc), NB + 1))
            np.add.at(cnt, (ci[tr], elb[tr]), 1.0)
            mh = np.zeros((nh, NB + 1))
            np.add.at(mh, (hb[tr], elb[tr]), 1.0)
            margh = (mh + 1.0) / (mh.sum(axis=1, keepdims=True) + NB + 1)
            hbc = np.zeros(len(uc), int)
            hbc[ci] = hb
            pm = (cnt + a * margh[hbc]) / (cnt.sum(axis=1, keepdims=True) + a)
            tail = np.cumsum(pm[:, ::-1], axis=1)[:, ::-1]
            ll = np.where(ev[te] >= 0, np.log(pm[ci[te], np.maximum(ev[te], 0)]), np.log(tail[ci[te], cb[te]]))
        tot += float(ll.sum())
        n += int(te.sum())
    return tot / n, n


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
            res.append({"a": a, "ll_haz": fold_ll(sub, b, a, "haz")[0], "ll_pool": fold_ll(sub, b, a, "pool")[0]})
        out["looks"] += 2 * len(res)
        bh = max(res, key=lambda r: r["ll_haz"])
        bp = max(res, key=lambda r: r["ll_pool"])
        out["classes"][nm] = {"k": k, "n": int(len(sub)), "cens": float(sub.cens.mean()), "edges": b["edges"], "sign": b["sign"], "to": b["to"], "a": bh["a"], "ll_haz": bh["ll_haz"], "ll_pool": bp["ll_pool"], "gain": bh["ll_haz"] - bp["ll_pool"]}
        print(nm, len(sub), "cens", round(float(sub.cens.mean()), 4), "haz", round(bh["ll_haz"], 4), "a", bh["a"], "pool", round(bp["ll_pool"], 4), "a", bp["a"], "gain", round(bh["ll_haz"] - bp["ll_pool"], 4), flush=True)
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"])


def tables():
    import mod25e_clk as ck

    spec = json.loads(FIT.read_text(encoding="utf-8"))["classes"]
    R = rows()
    T = {}
    for nm, s in spec.items():
        k = s["k"]
        sub = R[R.cls == k].reset_index(drop=True)
        edges = ck.EDGES[s["edges"]]
        c = ck.cell(sub.hs.to_numpy(), sub.sd.to_numpy(), sub.oto.to_numpy(), sub.dto.to_numpy(), edges, s["sign"], s["to"])
        hb = ck.hbin(sub.hs.to_numpy(), edges)
        ev, cb, last = obs(sub)
        uc, ci = np.unique(c, return_inverse=True)
        D, N, hh, _ = hazards(ci, hb, ev, last, len(uc), len(edges) + 1, s["a"])
        hbc = np.zeros(len(uc), int)
        hbc[ci] = hb
        hz = cell_hz(D, N, hh[hbc], s["a"])
        hz_h = np.clip(hh, 1e-6, 1 - 1e-6)
        cdfs = {}
        for i, cv in enumerate(uc):
            cdfs[int(cv)] = cdf_of(hz[i])
        T[k] = {"edges": edges, "sign": s["sign"], "to": s["to"], "cdf": cdfs, "hcdf": {h: cdf_of(hz_h[h]) for h in range(len(edges) + 1)}}
    return T


def cdf_of(hz):
    surv = np.concatenate([[1.0], np.cumprod(1 - hz)])
    pmf = np.concatenate([surv[:-1] * hz, surv[-1:]])
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
