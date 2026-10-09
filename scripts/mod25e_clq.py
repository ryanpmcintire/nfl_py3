import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "clq" / "fit.json"
SALT = 4343
OFFSET = {1: 2700.0, 3: 900.0}
CODES = (0, 1, 4, 5)
EDGE_SETS = ("none", "w", "ww", "w4", "w9", "w12", "w16")
SIGNS = ("none", "sign", "band")
TOS = ("none", "any", "both")
QS = (0, 1)


def enabled():
    return os.environ.get("CLQ") == "1"


def rows():
    import pandas as pd

    import mod25e_clk as ck

    R = pd.read_parquet(ck.REAL)
    R = R[R.qtr.isin([1, 3]) & R.code.isin(CODES) & (R.penalty != 1)].copy()
    R["hs"] = R.gsr.astype(float) - R.qtr.map(OFFSET).astype(float)
    n0 = len(R)
    R = R[R.hs > 0]
    nh = len(R)
    bad = (R.el > R.hs) | (R.el < 0)
    drop = {"rows": int(n0), "hs_nonpositive": int(n0 - nh), "glitch": int(bad.sum())}
    R = R[~bad].copy()
    R["el"] = R.el.astype(float)
    R["cls"] = ck.klass(R.code, R.yards, R.flip, R.po, R.pdf)
    R = R[R.cls >= 0].reset_index(drop=True)
    R.attrs["drop"] = drop
    return R


def cells(R, edges, sign, to, qs):
    import mod25e_clk as ck

    c = ck.cell(R.hs.to_numpy(), R.sd.to_numpy(), R.otu.to_numpy(), R.dtu.to_numpy(), edges, sign, to)
    hb = ck.hbin(R.hs.to_numpy(), edges)
    if qs:
        q3 = (R.qtr.to_numpy() == 3).astype(int)
        c = c * 2 + q3
        hb = hb * 2 + q3
    return c, hb


def collapse(P, hs):
    nb = P.shape[1]
    h = np.minimum(hs.astype(int), nb - 1)
    ar = np.arange(nb)[None, :]
    S = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]
    out = np.where(ar < h[:, None], P, 0.0)
    out[np.arange(len(h)), h] = S[np.arange(len(h)), h]
    return out


def season_blocks(sub, edges, sign, to, qs):
    import mod25e_clk as ck

    b = ck.ebin(sub.el.to_numpy())
    hs = sub.hs.to_numpy()
    ss = sub.season.to_numpy()
    c, hb = cells(sub, edges, sign, to, qs)
    nbh = int(hb.max()) + 1
    out = {}
    for s in np.unique(ss):
        te, tr = ss == s, ss != s
        mh = np.zeros((nbh, ck.NBIN))
        np.add.at(mh, (hb[tr], b[tr]), 1.0)
        margh = (mh + 1.0) / (mh.sum(axis=1, keepdims=True) + ck.NBIN)
        uc, inv = np.unique(c[tr], return_inverse=True)
        cnt = np.zeros((len(uc), ck.NBIN))
        np.add.at(cnt, (inv, b[tr]), 1.0)
        nn = cnt.sum(axis=1)
        idx = np.minimum(np.searchsorted(uc, c[te]), len(uc) - 1)
        hit = uc[idx] == c[te]
        out[int(s)] = (np.where(hit[:, None], cnt[idx], 0.0), np.where(hit, nn[idx], 0.0), margh[hb[te]], b[te], hs[te])
    return out


def ll_of(P, bb):
    return float(np.log(P[np.arange(len(bb)), bb]).sum())


def baseline(sub, trunc):
    import mod25e_clk as ck

    el = sub.el.to_numpy()
    b_raw = ck.ebin(el)
    hs = sub.hs.to_numpy()
    ss = sub.season.to_numpy()
    out = {}
    for s in np.unique(ss):
        te, tr = ss == s, ss != s
        ok = tr & (el <= ck.TAIL)
        mean_ok = float(el[ok].mean())
        cl = np.where(el[tr] > ck.TAIL, mean_ok, el[tr])
        bc = ck.ebin(cl)
        cnt = np.bincount(bc, minlength=ck.NBIN).astype(float)
        p = (cnt + 1.0) / (cnt.sum() + ck.NBIN)
        P = np.tile(p, (int(te.sum()), 1))
        if trunc:
            P = collapse(P, hs[te])
        out[int(s)] = (ll_of(P, b_raw[te]), int(te.sum()), float((P * np.arange(ck.NBIN)[None, :]).sum(1).mean()))
    return out


def eval_spec(blocks, a, trunc=True):
    res = {}
    for s, (ct, nt, mg, bt, hst) in blocks.items():
        P = (ct + a * mg) / (nt + a)[:, None]
        if trunc:
            P = collapse(P, hst)
        res[s] = (ll_of(P, bt), len(bt), float((P * np.arange(P.shape[1])[None, :]).sum(1).mean()))
    return res


def fit():
    import mod25e_clk as ck

    R = rows()
    out = {"drop": R.attrs["drop"], "classes": {}, "smooth": list(ck.SMOOTH), "looks": 0, "n_rows": int(len(R)), "folds": int(R.season.nunique())}
    for k, nm in enumerate(ck.CLASSES):
        sub = R[R.cls == k].reset_index(drop=True)
        if len(sub) < 200:
            continue
        b0 = baseline(sub, False)
        b0t = baseline(sub, True)
        res = []
        for en, sg, to, qs in itertools.product(EDGE_SETS, SIGNS, TOS, QS):
            bl = season_blocks(sub, ck.EDGES[en], sg, to, qs)
            for a in ck.SMOOTH:
                ev = eval_spec(bl, a)
                tot = sum(v[0] for v in ev.values())
                res.append({"edges": en, "sign": sg, "to": to, "qs": qs, "a": a, "ll": tot / len(sub), "per": {str(s): v[0] / v[1] for s, v in ev.items()}, "ns": {str(s): v[1] for s, v in ev.items()}, "mean_el": {str(s): v[2] for s, v in ev.items()}})
        out["looks"] += len(res)
        res.sort(key=lambda r: -r["ll"])
        r0 = max((r for r in res if r["edges"] == "none" and r["sign"] == "none" and r["to"] == "none" and r["qs"] == 0), key=lambda r: r["ll"])
        best = res[0]
        pb0 = {str(s): v[0] / v[1] for s, v in b0.items()}
        pb0t = {str(s): v[0] / v[1] for s, v in b0t.items()}
        out["classes"][nm] = {
            "n": int(len(sub)),
            "base_untrunc_ll": sum(v[0] for v in b0.values()) / len(sub),
            "base_trunc_ll": sum(v[0] for v in b0t.values()) / len(sub),
            "base_per": pb0,
            "base_trunc_per": pb0t,
            "base_mean_el": float(np.mean([v[2] for v in b0.values()])),
            "class_only": {key: r0[key] for key in ("a", "ll", "per")},
            "best": {key: best[key] for key in ("edges", "sign", "to", "qs", "a", "ll", "per", "ns", "mean_el")},
            "top3": [{key: r[key] for key in ("edges", "sign", "to", "qs", "a", "ll")} for r in res[:3]],
            "wins_vs_base": int(sum(1 for s in pb0 if best["per"][s] > pb0[s])),
            "wins_vs_base_trunc": int(sum(1 for s in pb0t if best["per"][s] > pb0t[s])),
            "seasons": len(pb0),
        }
        c = out["classes"][nm]
        print(nm, len(sub), "base %.4f base_t %.4f class_only %.4f best %.4f" % (c["base_untrunc_ll"], c["base_trunc_ll"], c["class_only"]["ll"], best["ll"]), best["edges"], best["sign"], best["to"], best["qs"], best["a"], "wins", c["wins_vs_base"], c["wins_vs_base_trunc"], "of", c["seasons"], flush=True)
    tot = sum(v["n"] for v in out["classes"].values())
    for key, lab in (("base_untrunc_ll", "gain_vs_base"), ("base_trunc_ll", "gain_vs_base_trunc")):
        out[lab] = sum(v["n"] * (v["best"]["ll"] - v[key]) for v in out["classes"].values()) / tot
    out["gain_class_only_vs_base"] = sum(v["n"] * (v["class_only"]["ll"] - v["base_untrunc_ll"]) for v in out["classes"].values()) / tot
    out["gain_class_only_vs_base_trunc"] = sum(v["n"] * (v["class_only"]["ll"] - v["base_trunc_ll"]) for v in out["classes"].values()) / tot
    out["gain_base_trunc_vs_base"] = sum(v["n"] * (v["base_trunc_ll"] - v["base_untrunc_ll"]) for v in out["classes"].values()) / tot
    seasons = sorted({int(s) for v in out["classes"].values() for s in v["best"]["per"]})
    pf, nf = {}, {}
    for s in seasons:
        pf[str(s)] = sum((v["best"]["per"][str(s)] - v["base_per"][str(s)]) * v["best"]["ns"][str(s)] for v in out["classes"].values())
        nf[str(s)] = sum(v["best"]["ns"][str(s)] for v in out["classes"].values())
    out["fold_gain_per_play"] = {s: pf[s] / nf[s] for s in pf}
    out["fold_wins"] = int(sum(1 for s in pf if pf[s] > 0))
    out["n_total"] = tot
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"], "n", tot, "fold wins", out["fold_wins"], "of", len(seasons), {k: round(v, 4) for k, v in out["fold_gain_per_play"].items()})
    print("gain per play vs untruncated cleaned-pool", round(out["gain_vs_base"], 4), "vs truncated cleaned-pool", round(out["gain_vs_base_trunc"], 4), "class-only real vs base", round(out["gain_class_only_vs_base"], 4), "trunc alone", round(out["gain_base_trunc_vs_base"], 4))
    return out


def install_clq():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    d = json.loads(FIT.read_text(encoding="utf-8"))["classes"]
    R = rows()
    el = R.el.to_numpy(float)
    kk = R.cls.to_numpy()
    q3all = (R.qtr.to_numpy() == 3).astype(int)
    sch, table, marg, margh = {}, {}, {}, {}
    for k, nm in enumerate(ck.CLASSES):
        if nm not in d:
            continue
        b = d[nm]["best"]
        sch[k] = (ck.EDGES[b["edges"]], b["sign"], b["to"], int(b["qs"]), float(b["a"]))
        mk = kk == k
        marg[k] = el[mk]
        sub = R[mk]
        c, hb = cells(sub, sch[k][0], sch[k][1], sch[k][2], sch[k][3])
        e = el[mk]
        for hv in np.unique(hb):
            margh[(k, int(hv))] = e[hb == hv]
        for cv in np.unique(c):
            table[(k, int(cv))] = e[c == cv]
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        if qtr not in (1, 3) or code not in CODES:
            return drawn
        hs = clock_val - OFFSET[qtr]
        if hs <= 0:
            return drawn
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        if k not in sch:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        rng = st["rng"]
        edges, sign, to, qs, a = sch[k]
        cv = int(ck.cell([hs], [score_diff], [float(drawn["off_to_used"])], [float(drawn["def_to_used"])], edges, sign, to)[0])
        hv = int(ck.hbin([hs], edges)[0])
        if qs:
            q3 = int(qtr == 3)
            cv, hv = cv * 2 + q3, hv * 2 + q3
        pool = table.get((k, cv))
        n = 0 if pool is None else len(pool)
        u, v = rng.random(), rng.random()
        src = pool if (n and u < n / (n + a)) else margh.get((k, hv), marg[k])
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
