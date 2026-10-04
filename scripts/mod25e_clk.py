import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "clk" / "fit.json"
REAL = REPO / "artifacts" / "sim09" / "f2" / "real_fit.parquet"
SALT_CK = 4301
TAIL = 60
NBIN = TAIL + 2
EDGES = {
    "none": [],
    "w": [120.0],
    "ww": [120.0, 240.0],
    "w4": [60.0, 120.0, 240.0, 480.0],
    "w9": [30.0, 60.0, 120.0, 180.0, 240.0, 360.0, 480.0, 720.0, 1080.0],
    "w12": [15.0, 30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 420.0, 600.0, 900.0, 1200.0],
    "w16": [10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0, 420.0, 600.0, 900.0, 1200.0, 1500.0],
    "w36": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 135.0, 150.0, 165.0, 180.0, 210.0, 240.0, 300.0, 420.0, 600.0, 900.0, 1200.0, 1500.0, 1700.0],
    "w24": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 180.0, 210.0, 240.0, 300.0, 420.0, 600.0, 900.0, 1200.0, 1500.0],
}
SIGNS = ("none", "sign", "band")
TOS = ("none", "any", "both")
SMOOTH = (0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0, 32768.0)
CLASSES = ("inc", "term", "run", "pass", "kneel", "spike")


def enabled():
    return os.environ.get("CLK") == "1"


def klass(code, yards, flip, po, pdf):
    code = np.asarray(code)
    ends = np.asarray(flip).astype(bool) | ((np.asarray(po) + np.asarray(pdf)) > 0)
    rp = (code == 0) | (code == 1)
    inc = (code == 1) & (np.asarray(yards) == 0) & ~ends
    return np.select([code == 5, code == 4, inc, rp & ends, code == 0, code == 1], [5, 4, 0, 1, 2, 3], -1)


def hbin(hs, edges):
    hs = np.asarray(hs, float)
    return np.searchsorted(np.asarray(edges, float), hs, side="left") if len(edges) else np.zeros(len(hs), int)


def cell(hs, sd, oto, dto, edges, sign, to):
    hs, sd = np.asarray(hs, float), np.asarray(sd, float)
    h = hbin(hs, edges)
    if sign == "sign":
        s = (np.sign(sd) + 1).astype(int)
    elif sign == "band":
        s = np.select([sd <= -9, sd < 0, sd == 0, sd < 9], [0, 1, 2, 3], 4)
    else:
        s = np.zeros(len(sd), int)
    oa, da = np.asarray(oto) > 0, np.asarray(dto) > 0
    t = (oa | da).astype(int) if to == "any" else (oa.astype(int) * 2 + da.astype(int) if to == "both" else np.zeros(len(sd), int))
    return (h * 8 + s) * 8 + t


def real_rows():
    import pandas as pd

    R = pd.read_parquet(REAL)
    R = R[R.qtr.isin([2, 4]) & R.code.isin([0, 1, 4, 5]) & (R.penalty != 1)].copy()
    R["hs"] = np.where(R.qtr == 2, R.gsr - 1800.0, R.gsr.astype(float))
    R = R[R.hs > 0]
    R["el"] = np.minimum(R.el.astype(float), R.hs)
    R["cls"] = klass(R.code, R.yards, R.flip, R.po, R.pdf)
    return R[R.cls >= 0].reset_index(drop=True)


def ebin(el):
    return np.minimum(np.round(np.asarray(el, float)).astype(int), TAIL + 1).clip(0)


def fold_ll(R, season, edges, sign, to, a, qs=0):
    tot, n = 0.0, 0
    b = ebin(R.el.to_numpy())
    c = cell(R.hs.to_numpy(), R.sd.to_numpy(), R.oto.to_numpy(), R.dto.to_numpy(), edges, sign, to)
    cls = R.cls.to_numpy()
    hb = hbin(R.hs.to_numpy(), edges)
    if qs:
        q4 = (R.qtr.to_numpy() == 4).astype(int)
        c = c * 2 + q4
        hb = hb * 2 + q4
    for s in np.unique(season):
        te = season == s
        tr = ~te
        for k in np.unique(cls):
            mk = cls == k
            if not (mk & te).any():
                continue
            mh = np.zeros((int(hb.max()) + 1, NBIN))
            np.add.at(mh, (hb[mk & tr], b[mk & tr]), 1.0)
            margh = (mh + 1.0) / (mh.sum(axis=1, keepdims=True) + NBIN)
            ct, bt = c[mk & tr], b[mk & tr]
            uc, inv = np.unique(ct, return_inverse=True)
            cnt = np.zeros((len(uc), NBIN))
            np.add.at(cnt, (inv, bt), 1.0)
            nn = cnt.sum(axis=1)
            idx = np.minimum(np.searchsorted(uc, c[mk & te]), len(uc) - 1)
            hit = uc[idx] == c[mk & te]
            be = b[mk & te]
            cc = np.where(hit, cnt[idx, be], 0.0)
            nh = np.where(hit, nn[idx], 0.0)
            p = (cc + a * margh[hb[mk & te], be]) / (nh + a)
            tot += float(np.log(p).sum())
            n += len(be)
    return tot / n, n


def fit():
    R = real_rows()
    out = {"classes": {}, "grid": {"edges": list(EDGES), "signs": list(SIGNS), "tos": list(TOS), "smooth": list(SMOOTH)}, "folds": int(R.season.nunique()), "n_rows": int(len(R))}
    looks = 0
    for k, nm in enumerate(CLASSES):
        sub = R[R.cls == k].reset_index(drop=True)
        ss = sub.season.to_numpy()
        base = fold_ll(sub, ss, EDGES["none"], "none", "none", SMOOTH[0])[0]
        res = []
        for en, sg, to, a in itertools.product(EDGES, SIGNS, TOS, SMOOTH):
            res.append({"edges": en, "sign": sg, "to": to, "a": a, "ll": fold_ll(sub, ss, EDGES[en], sg, to, a)[0]})
        looks += len(res)
        res.sort(key=lambda r: -r["ll"])
        out["classes"][nm] = {"n": int(len(sub)), "baseline_ll": base, "best": res[0], "top3": res[:3]}
        print(nm, len(sub), "base", round(base, 4), "best", res[0])
    out["looks"] = looks
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", looks)


FIT2 = REPO / "artifacts" / "mod25e3" / "clk" / "fit2.json"


def enabled2():
    return os.environ.get("CLK2") == "1"


def fit2():
    R = real_rows()
    spec = json.loads(FIT.read_text(encoding="utf-8"))["classes"]
    out = {"classes": {}, "looks": 0}
    for k, nm in enumerate(CLASSES):
        sub = R[R.cls == k].reset_index(drop=True)
        ss = sub.season.to_numpy()
        b = spec[nm]["best"]
        res = []
        for qs in (0, 1):
            for a in SMOOTH:
                res.append({"qs": qs, "a": a, "ll": fold_ll(sub, ss, EDGES[b["edges"]], b["sign"], b["to"], a, qs)[0]})
        out["looks"] += len(res)
        res.sort(key=lambda r: -r["ll"])
        pooled = max((r for r in res if r["qs"] == 0), key=lambda r: r["ll"])
        out["classes"][nm] = {"edges": b["edges"], "sign": b["sign"], "to": b["to"], "best": res[0], "pooled_best": pooled}
        print(nm, len(sub), "pooled", round(pooled["ll"], 4), "best", res[0], flush=True)
    FIT2.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"])


def install_ck():
    import mod25d_variance as dv
    import mod25e_endgame as eg
    import sim09_u4g as u4

    spec = json.loads(FIT.read_text(encoding="utf-8"))["classes"]
    sch = {k: (EDGES[spec[nm]["best"]["edges"]], spec[nm]["best"]["sign"], spec[nm]["best"]["to"], float(spec[nm]["best"]["a"])) for k, nm in enumerate(CLASSES)}
    qs = {k: 0 for k in range(len(CLASSES))}
    if enabled2():
        f2 = json.loads(FIT2.read_text(encoding="utf-8"))["classes"]
        for k, nm in enumerate(CLASSES):
            qs[k] = int(f2[nm]["best"]["qs"])
            sch[k] = sch[k][:3] + (float(f2[nm]["best"]["a"]),)
    R = real_rows()
    q4all = (R.qtr.to_numpy() == 4).astype(int)
    el = R.el.to_numpy(float)
    kk = R.cls.to_numpy()
    table, marg, margh = {}, {}, {}
    for k in range(len(CLASSES)):
        mk = kk == k
        marg[k] = el[mk]
        hk = hbin(R.hs.to_numpy()[mk], sch[k][0]) * 2 ** qs[k] + (q4all[mk] if qs[k] else 0)
        for hv in np.unique(hk):
            margh[(k, int(hv))] = el[mk][hk == hv]
        c = cell(R.hs.to_numpy()[mk], R.sd.to_numpy()[mk], R.oto.to_numpy()[mk], R.dto.to_numpy()[mk], *sch[k][:3])
        if qs[k]:
            c = c * 2 + q4all[mk]
        for cv in np.unique(c):
            table[(k, int(cv))] = el[mk][c == cv]
    probs = u4.load_probs()
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        offense, off_to, def_to = fr.f_locals["offense"], fr.f_locals["off_to"], fr.f_locals["def_to"]  # noqa: F841
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        if qtr not in (2, 4) or code not in (0, 1, 4, 5):
            return drawn
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if hs <= 0:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT_CK, cseed)
        rng = st["rng"]
        oto, dto = float(off_to), float(def_to)
        k = int(klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        cv = int(cell([hs], [score_diff], [oto], [dto], *sch[k][:3])[0])
        q4 = int(qtr == 4) if qs[k] else 0
        if qs[k]:
            cv = cv * 2 + q4
        a = sch[k][3]
        pool = table.get((k, cv))
        n = 0 if pool is None else len(pool)
        u, v = rng.random(), rng.random()
        hv = int(hbin([hs], sch[k][0])[0]) * 2 ** qs[k] + q4
        src = pool if (n and u < n / (n + a)) else margh.get((k, hv), marg[k])
        new = float(min(src[int(v * len(src))], hs))
        t = u4.WARN_AT.get(qtr)
        ends = bool(drawn["flip"]) or (drawn["points_off"] + drawn["points_def"]) > 0
        if t is not None and clock_val > t and clock_val - new < t and rng.random() < probs[u4.cell_key(qtr, code, ends)]:
            new = float(clock_val - t)
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


KN_FIT = REPO / "artifacts" / "mod25e3" / "clk" / "kneel_kn.json"
SALT_KN = 4302
NB_KN = 61
KN_SMOOTH = (0.5, 2.0, 8.0, 32.0, 128.0, 512.0)
KN_TO = ("none", "any", "count")


def kn_enabled():
    return os.environ.get("KN") == "1"


def kn_rows():
    import pandas as pd

    R = pd.read_parquet(REAL).sort_values(["g", "play_id"]).reset_index(drop=True)
    R["half"] = np.where(R.qtr >= 3, 2, 1)
    R["pstop"] = R.groupby(["g", "half"]).stop.shift(1).fillna(True).astype(float)
    K = R[(R.code == 4) & R.qtr.isin([2, 4]) & (R.penalty != 1)].copy()
    K["hs"] = np.where(K.qtr == 2, K.gsr - 1800.0, K.gsr.astype(float))
    K = K[K.hs > 0].reset_index(drop=True)
    K["cens"] = K.el >= K.hs
    return K


def kn_obs(K):
    e = np.minimum(np.round(K.el.to_numpy(float)).astype(int), NB_KN)
    cb = np.minimum(np.ceil(K.hs.to_numpy(float)).astype(int), NB_KN)
    ev = np.where(K.cens.to_numpy() | (e >= NB_KN), -1, e)
    cens_bin = np.where(ev >= 0, 0, cb)
    last = np.where(ev >= 0, ev, cb - 1)
    return ev, cens_bin, last


def kn_hazards(hb, ps, ev, last, nhb, pst, a):
    nc = nhb * 2
    cell = hb * 2 + ps if pst else hb * 2
    D = np.zeros((nc, NB_KN))
    H = np.zeros((nc, NB_KN))
    m = ev >= 0
    np.add.at(D, (cell[m], ev[m]), 1.0)
    np.add.at(H, (cell, last), 1.0)
    N = np.cumsum(H[:, ::-1], axis=1)[:, ::-1]
    Dh = D.reshape(nhb, 2, NB_KN).sum(axis=1)
    Nh = N.reshape(nhb, 2, NB_KN).sum(axis=1)
    hg = np.clip(Dh.sum(axis=0) / np.maximum(Nh.sum(axis=0), 1.0), 1e-6, 1 - 1e-6)
    hh = (Dh + a * hg) / (Nh + a)
    hz = (D + a * np.repeat(hh, 2, axis=0)) / (N + a)
    return np.clip(hz, 1e-6, 1 - 1e-6)


def kn_ll(hz, cell, ev, cbin):
    cum = np.concatenate([np.zeros((hz.shape[0], 1)), np.cumsum(np.log1p(-hz), axis=1)], axis=1)
    e = np.maximum(ev, 0)
    return np.where(ev >= 0, cum[cell, e] + np.log(hz[cell, e]), cum[cell, cbin])


def kn_fold(K, ev, cbin, last, edges, pst, a):
    hb = hbin(K.hs.to_numpy(), edges)
    ps = K.pstop.to_numpy().astype(int)
    nhb = len(edges) + 1
    cell = hb * 2 + ps if pst else hb * 2
    se = K.season.to_numpy()
    tot, n = 0.0, 0
    for s in np.unique(se):
        te, tr = se == s, se != s
        hz = kn_hazards(hb[tr], ps[tr], ev[tr], last[tr], nhb, pst, a)
        tot += float(kn_ll(hz, cell[te], ev[te], cbin[te]).sum())
        n += int(te.sum())
    return tot / n


def kn_to_fold(K, edges, scheme, a):
    hb = hbin(K.hs.to_numpy(), edges)
    dto = K.dto.to_numpy().astype(int)
    y = (K.dtu.to_numpy() > 0).astype(float)
    ng = 3 if scheme == "count" else 1
    g = np.clip(dto, 1, 3) - 1 if ng == 3 else np.zeros(len(K), int)
    se = K.season.to_numpy()
    nh = len(edges) + 1
    tot, n = 0.0, 0
    for s in np.unique(se):
        te, tr = se == s, se != s
        pg = y[tr].mean()
        k_h = np.bincount(hb[tr], weights=y[tr], minlength=nh)
        n_h = np.bincount(hb[tr], minlength=nh).astype(float)
        ph = (k_h + a * pg) / (n_h + a)
        c = hb * ng + g
        k_c = np.bincount(c[tr], weights=y[tr], minlength=nh * ng)
        n_c = np.bincount(c[tr], minlength=nh * ng).astype(float)
        pc = np.clip((k_c + a * np.repeat(ph, ng)) / (n_c + a), 1e-6, 1 - 1e-6)
        p = pc[c[te]]
        tot += float((y[te] * np.log(p) + (1 - y[te]) * np.log1p(-p)).sum())
        n += int(te.sum())
    return tot / n


def kn_fit():
    K = kn_rows()
    T = K[(K.dtu == 0) & (K.otu == 0)].reset_index(drop=True)
    ev, cbin, last = kn_obs(T)
    out = {"n_T": int(len(T)), "censored": float(T.cens.mean()), "T": [], "TO": []}
    out["T_base"] = kn_fold(T, ev, cbin, last, EDGES["none"], 0, 2048.0)
    for en, pst, a in itertools.product(EDGES, (0, 1), KN_SMOOTH):
        out["T"].append({"edges": en, "pstop": pst, "a": a, "ll": kn_fold(T, ev, cbin, last, EDGES[en], pst, a)})
    out["T"].sort(key=lambda r: -r["ll"])
    D = K[K.dto > 0].reset_index(drop=True)
    for en, sc, a in itertools.product(("none", "w", "ww", "w4", "w9"), KN_TO, KN_SMOOTH):
        out["TO"].append({"edges": en, "scheme": sc, "a": a, "ll": kn_to_fold(D, EDGES[en], sc, a)})
    out["TO"].sort(key=lambda r: -r["ll"])
    out["TO_base"] = kn_to_fold(D, [], "none", 1e9)
    out["n_TO"] = int(len(D))
    out["looks"] = len(out["T"]) + len(out["TO"])
    KN_FIT.write_text(json.dumps(out), encoding="utf-8")
    print("T base", out["T_base"], "best", out["T"][:3])
    print("TO base", out["TO_base"], "best", out["TO"][:3], "n", len(D), "looks", out["looks"])


def kn_tables():
    spec = json.loads(KN_FIT.read_text(encoding="utf-8"))
    bt, bo = spec["T"][0], spec["TO"][0]
    K = kn_rows()
    T = K[(K.dtu == 0) & (K.otu == 0)].reset_index(drop=True)
    ev, cbin, last = kn_obs(T)
    et = EDGES[bt["edges"]]
    hz = kn_hazards(hbin(T.hs.to_numpy(), et), T.pstop.to_numpy().astype(int), ev, last, len(et) + 1, bt["pstop"], float(bt["a"]))
    surv = np.concatenate([np.ones((hz.shape[0], 1)), np.cumprod(1 - hz, axis=1)], axis=1)
    pmf = np.concatenate([surv[:, :-1] * hz, surv[:, -1:]], axis=1)
    cdf = np.cumsum(pmf, axis=1)
    cdf[:, -1] = 1.0
    D = K[K.dto > 0].reset_index(drop=True)
    eo = EDGES[bo["edges"]]
    ng = 3 if bo["scheme"] == "count" else 1
    hb = hbin(D.hs.to_numpy(), eo)
    g = np.clip(D.dto.to_numpy().astype(int), 1, 3) - 1 if ng == 3 else np.zeros(len(D), int)
    y = (D.dtu.to_numpy() > 0).astype(float)
    a = float(bo["a"])
    nh = len(eo) + 1
    ph = (np.bincount(hb, weights=y, minlength=nh) + a * y.mean()) / (np.bincount(hb, minlength=nh) + a)
    c = hb * ng + g
    pc = (np.bincount(c, weights=y, minlength=nh * ng) + a * np.repeat(ph, ng)) / (np.bincount(c, minlength=nh * ng) + a)
    U = D[D.dtu > 0]
    pool = np.minimum(U.el.to_numpy(float), U.hs.to_numpy(float))
    return {"bt": bt, "bo": bo, "cdf": cdf, "pc": pc, "pool": pool, "ng": ng, "et": et, "eo": eo}


def kn_draw(tb, u, hs, ps, dto, ut):
    hb = int(hbin([hs], tb["eo"])[0])
    g = min(max(int(dto), 1), 3) - 1 if tb["ng"] == 3 else 0
    if dto > 0 and ut < tb["pc"][hb * tb["ng"] + g]:
        return True, float(min(tb["pool"][int(u * len(tb["pool"]))], hs))
    cell = int(hbin([hs], tb["et"])[0]) * 2 + (int(ps) if tb["bt"]["pstop"] else 0)
    b = int(min(np.searchsorted(tb["cdf"][cell], u, side="left"), NB_KN))
    return False, float(min(b, hs))


def install_kn():
    import mod25d_variance as dv
    import mod25e_endgame as eg
    import sim09_u4g as u4

    tb = kn_tables()
    probs = u4.load_probs()
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None, "prev": (None, 1.0)}
    wide = os.environ.get("KNW") == "1"
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        offense, off_to, def_to = fr.f_locals["offense"], fr.f_locals["off_to"], fr.f_locals["def_to"]  # noqa: F841
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT_KN, cseed)
            st["prev"] = (None, 1.0)
        pv = st["prev"]
        ps = pv[1] if (pv[0] is not None and abs(pv[0] - clock_val) < 1e-6) else 1.0
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        oto, dto = float(off_to), float(def_to)
        if qtr in (2, 4) and code == 4 and hs > 0 and not (oto > 0 and float(drawn["off_to_used"]) > 0):
            rng = st["rng"]
            u, ut, uw = rng.random(), rng.random(), rng.random()
            used, new = kn_draw(tb, u, hs, ps, dto, ut)
            t = u4.WARN_AT.get(qtr)
            if (not used) and t is not None and clock_val > t and clock_val - new < t and uw < probs[u4.cell_key(qtr, 4, False)]:
                new = float(clock_val - t)
            old = float(drawn["clock_elapsed"])
            drawn = dict(drawn)
            drawn["clock_elapsed"] = new
            drawn["def_to_used"] = 1.0 if used else 0.0
            drawn["off_to_used"] = 0.0
            lg = dv._G.get("log")
            if lg:
                tp = list(lg[-1])
                tp[10] = new
                lg[-1] = tuple(tp)
            es = dv._G.get("egst")
            if es is not None and es["ps"][0] is not None and abs(es["ps"][0] - (clock_val - old)) < 1e-6:
                es["ps"] = (clock_val - new, 1.0 if used else 0.0)
        stn = float(eg.stop_after(code, float(drawn["yards_gained"]), bool(drawn["flip"]), (drawn["points_off"] + drawn["points_def"]) > 0, float(drawn["off_to_used"]), float(drawn["def_to_used"]), float(drawn["clock_elapsed"])))
        st["prev"] = (clock_val - float(drawn["clock_elapsed"]), stn)
        es = dv._G.get("egst")
        if wide and es is not None and not eg.dr.in_window(qtr, clock_val):
            es["ps"] = st["prev"]
        return drawn

    dv._G["pol"] = pol


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "kn":
        kn_fit()
    elif len(sys.argv) > 1 and sys.argv[1] == "fit2":
        fit2()
    else:
        fit()
