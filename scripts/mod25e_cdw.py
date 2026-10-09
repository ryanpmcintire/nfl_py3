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


def enabled():
    return os.environ.get("CDW") == "1"


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
    if len(sys.argv) > 2 and sys.argv[1] == "choose":
        choose(float(sys.argv[2]))
