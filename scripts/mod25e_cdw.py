import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "cdw" / "fit.json"
SALT = 4317
HZ_LO = 45.0
HZS = (300.0, 420.0, 600.0)
WIDTHS = (10.0, 20.0)
USEDS = ("none", "any", "split")
SIGNS = ("none", "sign", "band")
AS = (8.0, 32.0, 128.0, 512.0, 2048.0)
QSPLIT = (0, 1)
NB = 62
FIT2 = REPO / "artifacts" / "mod25e3" / "cdw" / "fit2.json"
SALT2 = 4351
EPS2 = (0.01, 0.1, 1.0, 4.0, 16.0, 64.0)
AS2 = (8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0, 32768.0)
USED2 = ("none", "split")
SIGN2 = ("none", "sign")
CLS2 = (0, 1, 2, 3)


def enabled():
    return os.environ.get("CDW") in ("1", "2")


def rows():
    import mod25e_clk as ck

    R = ck.real_rows()
    R = R[R.code.isin([0, 1]) & R.qtr.isin([2, 4]) & (R.hs > HZ_LO)].copy()
    return R.reset_index(drop=True)


def keys(hs, sd, qtr, otu, dtu, cls, spec):
    hs, sd = np.asarray(hs, float), np.asarray(sd, float)
    nh = int(np.ceil((spec["hz"] - HZ_LO) / spec["w"]))
    h = np.clip(np.floor((hs - HZ_LO - 1e-9) / spec["w"]).astype(int), 0, nh - 1)
    oa, da = np.asarray(otu) > 0, np.asarray(dtu) > 0
    if spec["used"] == "any":
        u = (oa | da).astype(int)
    elif spec["used"] == "split":
        u = np.where(oa, 1, np.where(da, 2, 0))
    else:
        u = np.zeros(len(hs), int)
    if spec["sign"] == "sign":
        s = (np.sign(sd) + 1).astype(int)
    elif spec["sign"] == "band":
        s = np.select([sd <= -9, sd < 0, sd == 0, sd < 9], [0, 1, 2, 3], 4)
    else:
        s = np.zeros(len(hs), int)
    q = (np.asarray(qtr) == 4).astype(int) if spec["qs"] else np.zeros(len(hs), int)
    c = np.asarray(cls).astype(int)
    m = (u * 4 + c) * nh + h
    f = (m * 5 + s) * 2 + q
    return f, m


def counts(f, eb, nk):
    C = np.zeros((nk, NB))
    np.add.at(C, (f, eb), 1.0)
    return C


def predict(R_tr, R_te, spec, a):
    ftr, mtr = keys(R_tr.hs, R_tr.sd, R_tr.qtr, R_tr.otu, R_tr.dtu, R_tr.cls, spec)
    fte, mte = keys(R_te.hs, R_te.sd, R_te.qtr, R_te.otu, R_te.dtu, R_te.cls, spec)
    ebt = np.minimum(np.round(R_tr.el.to_numpy()).astype(int), NB - 1)
    nk = int(max(ftr.max(), fte.max())) + 1
    nm = int(max(mtr.max(), mte.max())) + 1
    CF, CM = counts(ftr, ebt, nk), counts(mtr, ebt, nm)
    MP = (CM + 1.0) / (CM.sum(1, keepdims=True) + NB)
    cf = CF[fte]
    return (cf + a * MP[mte]) / (cf.sum(1, keepdims=True) + a)


def fold(R, spec, a):
    out = {}
    eb = np.minimum(np.round(R.el.to_numpy()).astype(int), NB - 1)
    hsr = R.hs.to_numpy()
    for s in sorted(R.season.unique()):
        te = (R.season == s).to_numpy() & (hsr <= spec["hz"])
        tr = (R.season != s).to_numpy() & (hsr <= spec["hz"])
        P = predict(R[tr], R[te], spec, a)
        out[int(s)] = (P, te, eb[te])
    return out


def stats_row(P, hs):
    b = np.arange(NB)[None, :]
    mean = (P * np.minimum(b, hs[:, None])).sum(1)
    p12 = (P * (b <= 12)).sum(1)
    p38 = (P * (b >= 38)).sum(1)
    return mean, p12, p38


def evaluate(R, spec, a, with_bias=False):
    f = fold(R, spec, a)
    ll, n, per, bias = 0.0, 0, {}, {}
    for s, (P, te, eb) in f.items():
        lls = float(np.log(P[np.arange(len(eb)), eb]).sum())
        ll += lls
        n += len(eb)
        per[s] = (lls, len(eb))
        if with_bias:
            Rt = R[te]
            m, p12, p38 = stats_row(P, Rt.hs.to_numpy())
            el = Rt.el.to_numpy()
            sub = ((Rt.hs > 120) & (Rt.hs <= 300) & (Rt.qtr == 4) & (Rt.sd >= 9) & ((Rt.otu + Rt.dtu) == 0) & (Rt.code == 0)).to_numpy()
            if sub.sum() >= 5:
                bias[s] = {"n": int(sub.sum()), "mean_el": (float(m[sub].mean()), float(el[sub].mean())), "p12": (float(p12[sub].mean()), float((el[sub] <= 12).mean())), "p38": (float(p38[sub].mean()), float((el[sub] >= 38).mean()))}
    return ll / n, n, per, bias


def fit():
    R = rows()
    res = []
    for hz, w, used, sign, a, qs in itertools.product(HZS, WIDTHS, USEDS, SIGNS, AS, QSPLIT):
        spec = {"hz": hz, "w": w, "used": used, "sign": sign, "qs": qs}
        ll, n, per, _ = evaluate(R, spec, a)
        res.append({"spec": spec, "a": a, "ll": ll, "n": n, "per": {str(k): v for k, v in per.items()}})
    out = {"looks": len(res), "results": res}
    best = {}
    for hz in HZS:
        sub = [r for r in res if r["spec"]["hz"] == hz]
        base = max([r for r in sub if r["spec"]["used"] == "none" and r["spec"]["sign"] == "none"], key=lambda r: r["ll"])
        clk = max([r for r in sub if r["spec"]["used"] == "none"], key=lambda r: r["ll"])
        cand = max([r for r in sub if r["spec"]["used"] != "none"], key=lambda r: r["ll"])
        sb = {k: v[0] / v[1] for k, v in base["per"].items()}
        sk = {k: v[0] / v[1] for k, v in clk["per"].items()}
        sc = {k: v[0] / v[1] for k, v in cand["per"].items()}
        best[str(int(hz))] = {"base": {"spec": base["spec"], "a": base["a"], "ll": base["ll"]}, "clk_like": {"spec": clk["spec"], "a": clk["a"], "ll": clk["ll"]}, "cand": {"spec": cand["spec"], "a": cand["a"], "ll": cand["ll"]},
                              "gain_vs_base": cand["ll"] - base["ll"], "gain_vs_clk": cand["ll"] - clk["ll"], "seasons_pos_base": sum(1 for k in sc if sc[k] > sb[k]), "seasons_pos_clk": sum(1 for k in sc if sc[k] > sk[k]), "seasons": len(sc), "n": cand["n"]}
    out["best"] = best
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", len(res))
    for hz, b in best.items():
        print(hz, json.dumps(b))
    for r in sorted(res, key=lambda r: -r["ll"])[:6]:
        print(r["spec"], r["a"], round(r["ll"], 4))


def choose(hz):
    d = json.loads(FIT.read_text(encoding="utf-8"))
    c = d["best"][str(int(hz))]["cand"]
    d["chosen"] = c
    FIT.write_text(json.dumps(d), encoding="utf-8")
    R = rows()
    for tag in ("cand", "clk_like"):
        b = d["best"][str(int(hz))][tag]
        ll, n, per, bias = evaluate(R, b["spec"], b["a"], True)
        print(tag, b["spec"], b["a"], round(ll, 4), n)
        for s, v in bias.items():
            print(" ", s, v["n"], "mean_el pred/act %.1f/%.1f" % v["mean_el"], "p12 %.3f/%.3f" % v["p12"], "p38 %.3f/%.3f" % v["p38"])


def install_cdw():
    if os.environ.get("CDW") == "2":
        install_cdw1()
        return install_cdw2()
    return install_cdw1()


def install_cdw1():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    b = json.loads(FIT.read_text(encoding="utf-8"))["chosen"]
    spec, a = b["spec"], float(b["a"])
    R = rows()
    R = R[R.hs <= spec["hz"]]
    f, m = keys(R.hs, R.sd, R.qtr, R.otu, R.dtu, R.cls, spec)
    eb = np.minimum(np.round(R.el.to_numpy()).astype(int), NB - 1)
    pool, marg = {}, {}
    for key in np.unique(f):
        pool[int(key)] = eb[f == key]
    for key in np.unique(m):
        marg[int(key)] = eb[m == key]
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if code not in (0, 1) or qtr not in (2, 4) or hs <= HZ_LO or hs > spec["hz"]:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        rng = st["rng"]
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        if k not in (0, 1, 2, 3):
            return drawn
        fk, mk = keys([hs], [score_diff], [qtr], [float(drawn["off_to_used"])], [float(drawn["def_to_used"])], [k], spec)
        src = pool.get(int(fk[0]))
        n = 0 if src is None else len(src)
        r1, r2 = rng.random(), rng.random()
        if n and r1 < n / (n + a):
            bb = int(src[int(r2 * n)])
        else:
            ms = marg.get(int(mk[0]))
            if ms is None:
                return drawn
            cnt = np.bincount(ms, minlength=NB).astype(float) + 1.0
            bb = int(np.searchsorted(np.cumsum(cnt) / cnt.sum(), r2))
        new = float(min(bb, hs))
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


def geom():
    sp = json.loads(FIT.read_text(encoding="utf-8"))["chosen"]["spec"]
    hz, w = float(sp["hz"]), float(sp["w"])
    return hz, w, int(np.ceil((hz - HZ_LO) / w))


def idx2(hs, sd, qtr, otu, dtu, sp, nh, w):
    hs, sd = np.asarray(hs, float), np.asarray(sd, float)
    h = np.clip(np.floor((hs - HZ_LO - 1e-9) / w).astype(int), 0, nh - 1)
    oa, da = np.asarray(otu) > 0, np.asarray(dtu) > 0
    u = np.where(oa, 1, np.where(da, 2, 0)) if sp[0] == "split" else np.zeros(len(hs), int)
    s = (np.sign(sd) + 1).astype(int) if sp[1] == "sign" else np.zeros(len(hs), int)
    q = (np.asarray(qtr) == 4).astype(int) if sp[2] else np.zeros(len(hs), int)
    return ((u * nh + h) * 3 + s) * 2 + q, u * nh + h, h


def tiers(R, sp, nh, w, ns):
    k1, k2, k3 = idx2(R.hs, R.sd, R.qtr, R.otu, R.dtu, sp, nh, w)
    b = np.minimum(np.round(R.el.to_numpy()).astype(int), NB - 1)
    si = R.sidx.to_numpy()
    T = []
    for k, K in ((k1, 3 * nh * 6), (k2, 3 * nh), (k3, nh)):
        C = np.zeros((ns, K, NB))
        np.add.at(C, (si, k, b), 1.0)
        T.append(C)
    C4 = np.zeros((ns, NB))
    np.add.at(C4, (si, b), 1.0)
    return T[0], T[1], T[2], C4, k1, k2, k3, b


def ll_at(T1, T2, T3, T4, k1, k2, k3, b, a1, a2, eps):
    p = (T4[b] + eps) / (T4.sum() + NB * eps)
    p = (T3[k3, b] + a2 * p) / (T3.sum(1)[k3] + a2)
    p = (T2[k2, b] + a2 * p) / (T2.sum(1)[k2] + a2)
    p = (T1[k1, b] + a1 * p) / (T1.sum(1)[k1] + a1)
    return float(np.log(p).sum())


def pmf2(T1, T2, T3, T4, a1, a2, eps):
    p4 = (T4 + eps) / (T4.sum() + NB * eps)
    p3 = (T3 + a2 * p4[None, :]) / (T3.sum(1, keepdims=True) + a2)
    nh = T3.shape[0]
    p2 = (T2 + a2 * p3[np.arange(T2.shape[0]) % nh]) / (T2.sum(1, keepdims=True) + a2)
    return (T1 + a1 * p2[np.arange(T1.shape[0]) // 6]) / (T1.sum(1, keepdims=True) + a1)


def fit2():
    hz, w, nh = geom()
    R = rows()
    R = R[R.hs <= hz].reset_index(drop=True)
    seasons = sorted(R.season.unique())
    ns = len(seasons)
    R["sidx"] = R.season.map({s: i for i, s in enumerate(seasons)})
    grid = [(a1, a2, e) for a1 in AS2 for a2 in AS2 for e in EPS2]
    specs = [(u, g, q) for u in USED2 for g in SIGN2 for q in QSPLIT]
    out = {"hz": hz, "w": w, "nh": nh, "seasons": [int(s) for s in seasons], "grid": grid, "specs": specs, "classes": {}}
    looks = 0
    for c in CLS2:
        Rc = R[R.cls == c].reset_index(drop=True)
        si = Rc.sidx.to_numpy()
        rows_s = [np.where(si == j)[0] for j in range(ns)]
        loso = np.zeros((len(specs), len(grid), ns))
        inner = np.zeros((len(specs), len(grid), ns, ns))
        for i, sp in enumerate(specs):
            T1, T2, T3, T4, k1, k2, k3, b = tiers(Rc, sp, nh, w, ns)
            t1, t2, t3, t4 = T1.sum(0), T2.sum(0), T3.sum(0), T4.sum(0)
            for s in range(ns):
                tr = (t1 - T1[s], t2 - T2[s], t3 - T3[s], t4 - T4[s])
                r = rows_s[s]
                for j, (a1, a2, e) in enumerate(grid):
                    loso[i, j, s] = ll_at(*tr, k1[r], k2[r], k3[r], b[r], a1, a2, e)
                for t in range(ns):
                    if t == s:
                        continue
                    tr2 = (tr[0] - T1[t], tr[1] - T2[t], tr[2] - T3[t], tr[3] - T4[t])
                    r2 = rows_s[t]
                    for j, (a1, a2, e) in enumerate(grid):
                        inner[i, j, s, t] = ll_at(*tr2, k1[r2], k2[r2], k3[r2], b[r2], a1, a2, e)
        looks += len(specs) * len(grid)
        flat = loso.reshape(-1, ns)
        best = int(np.argmax(flat.sum(1)))
        innerf = inner.reshape(-1, ns, ns).sum(2)
        pick = [int(np.argmax(innerf[:, s])) for s in range(ns)]
        nested = np.array([flat[pick[s], s] for s in range(ns)])
        bi, bj = divmod(best, len(grid))
        out["classes"][str(c)] = {"n": int(len(Rc)), "best_spec": list(specs[bi]), "best_grid": list(grid[bj]), "best_loso": flat[best].tolist(), "nested": nested.tolist(), "nested_pick": [[list(specs[p // len(grid)]), list(grid[p % len(grid)])] for p in pick]}
        print(c, len(Rc), specs[bi], grid[bj], "loso %.4f nested %.4f" % (flat[best].sum() / len(Rc), nested.sum() / len(Rc)), flush=True)
    out["looks"] = looks
    d = json.loads(FIT.read_text(encoding="utf-8"))["chosen"]
    base = {c: np.zeros(ns) for c in CLS2}
    for s, (P, te, eb) in fold(R, d["spec"], float(d["a"])).items():
        j = seasons.index(s)
        lls = np.log(P[np.arange(len(eb)), eb])
        cl = R[te].cls.to_numpy()
        for c in CLS2:
            base[c][j] = float(lls[cl == c].sum())
    for c in CLS2:
        o = out["classes"][str(c)]
        o["cdw1"] = base[c].tolist()
        o["nested_gain"] = (np.array(o["nested"]) - base[c]).tolist()
        o["loso_gain"] = (np.array(o["best_loso"]) - base[c]).tolist()
        print(c, "cdw1 %.4f" % (base[c].sum() / o["n"]), "nested gain/play %.4f wins %d/%d" % (sum(o["nested_gain"]) / o["n"], sum(1 for g in o["nested_gain"] if g > 0), ns), "loso gain/play %.4f wins %d/%d" % (sum(o["loso_gain"]) / o["n"], sum(1 for g in o["loso_gain"] if g > 0), ns))
    FIT2.parent.mkdir(parents=True, exist_ok=True)
    FIT2.write_text(json.dumps(out), encoding="utf-8")
    print("looks", looks)


def install_cdw2():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    d = json.loads(FIT2.read_text(encoding="utf-8"))
    hz, w, nh = float(d["hz"]), float(d["w"]), int(d["nh"])
    R = rows()
    R = R[R.hs <= hz].reset_index(drop=True)
    R["sidx"] = 0
    cdf, spc, pend, tail = {}, {}, {}, {}
    for c in CLS2:
        o = d["classes"][str(c)]
        if not all(g > 0 for g in o["nested_gain"]):
            continue
        tl = R[(R.cls == c) & (R.el >= NB - 1)]
        ends = (tl.el >= tl.hs - 1e-9).to_numpy()
        pend[c] = float(ends.mean()) if len(tl) else 1.0
        tail[c] = np.sort(tl.el.to_numpy()[~ends])
        sp = tuple(o["best_spec"])
        a1, a2, e = o["best_grid"]
        T1, T2, T3, T4, _, _, _, _ = tiers(R[R.cls == c].reset_index(drop=True), sp, nh, w, 1)
        cum = np.cumsum(pmf2(T1[0], T2[0], T3[0], T4[0], a1, a2, e), axis=1)
        cum[:, -1] = 1.0
        cdf[c], spc[c] = cum, sp
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if code not in (0, 1) or qtr not in (2, 4) or hs <= HZ_LO or hs > hz:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT2, cseed)
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        if k not in cdf:
            return drawn
        f = int(idx2([hs], [score_diff], [qtr], [float(drawn["off_to_used"])], [float(drawn["def_to_used"])], spc[k], nh, w)[0][0])
        bb = min(int(np.searchsorted(cdf[k][f], st["rng"].random(), side="right")), NB - 1)
        if bb == NB - 1:
            r2 = st["rng"].random()
            if r2 < pend[k] or not len(tail[k]):
                bb = hs
            else:
                bb = tail[k][min(int((r2 - pend[k]) / (1.0 - pend[k]) * len(tail[k])), len(tail[k]) - 1)]
        new = float(min(bb, hs))
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
    if len(sys.argv) > 1 and sys.argv[1] == "fit2":
        fit2()
    if len(sys.argv) > 2 and sys.argv[1] == "choose":
        choose(float(sys.argv[2]))
