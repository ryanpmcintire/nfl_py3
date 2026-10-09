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
    return os.environ.get("CLQ") in ("1", "2")


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
    if os.environ.get("CLQ") == "2":
        return install_clq2()
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


FIT2 = REPO / "artifacts" / "mod25e3" / "clq" / "fit2.json"
SALT2 = 4344
START = {1: 3600.0, 3: 1800.0}
CODES2 = (0, 1, 4, 5, 6)
CLASSES2 = ("inc", "term", "run", "pass", "kneel", "spike", "noplay")
NH = 61
NPB = 62
OFFB = 71


def rows2():
    import pandas as pd

    import mod25e_clk as ck

    R = pd.read_parquet(ck.REAL)
    R = R[R.qtr.isin([1, 3]) & ((R.code.isin(CODES) & (R.penalty != 1)) | (R.code == 6))].copy()
    R["hs"] = R.gsr.astype(float) - R.qtr.map(OFFSET).astype(float)
    R["last"] = (R.next_qtr != R.qtr).to_numpy()
    n0 = len(R)
    R = R[R.hs > 0]
    nh = len(R)
    bad = (R.el > R.hs) | (R.el < 0)
    drop = {"rows": int(n0), "hs_nonpositive": int(n0 - nh), "glitch": int(bad.sum())}
    R = R[~bad].copy()
    R["el"] = R.el.astype(float)
    k = ck.klass(R.code, R.yards, R.flip, R.po, R.pdf)
    R["cls"] = np.where(R.code == 6, 6, k)
    R = R[R.cls >= 0].reset_index(drop=True)
    drop["censored_last"] = int(R["last"].sum())
    R.attrs["drop"] = drop
    return R


def obs(sub):
    import mod25e_clk as ck

    el = sub.el.to_numpy(float)
    hs = sub.hs.to_numpy(float)
    cens = sub["last"].to_numpy(bool)
    b = ck.ebin(el)
    h = np.minimum(np.round(hs).astype(int), NPB - 1)
    L = np.clip(np.where(cens, h - 1, np.minimum(b, NH - 1)), 0, NH - 1)
    ev = (~cens) & (b < NH)
    return b, L, ev, h


def counts(sub, edges, sign, to, qs, seasons):
    c, hb = cells(sub, edges, sign, to, qs)
    uc, ci = np.unique(c, return_inverse=True)
    uh, hi = np.unique(hb, return_inverse=True)
    b, L, ev, h = obs(sub)
    si = np.searchsorted(seasons, sub.season.to_numpy())
    ns = len(seasons)
    HL = np.zeros((ns, len(uc), NH))
    HD = np.zeros((ns, len(uc), NH))
    GL = np.zeros((ns, len(uh), NH))
    GD = np.zeros((ns, len(uh), NH))
    np.add.at(HL, (si, ci, L), 1.0)
    np.add.at(HD, (si[ev], ci[ev], b[ev]), 1.0)
    np.add.at(GL, (si, hi, L), 1.0)
    np.add.at(GD, (si[ev], hi[ev], b[ev]), 1.0)
    gof = np.zeros(len(uc), int)
    gof[ci] = hi
    return {"uc": uc, "uh": uh, "ci": ci, "hi": hi, "b": b, "h": h, "si": si, "HL": HL, "HD": HD, "GL": GL, "GD": GD, "gof": gof}


def revcum(x):
    return np.cumsum(x[..., ::-1], axis=-1)[..., ::-1]


def pmf_of(hz):
    cs = np.cumsum(np.log1p(-hz), axis=-1)
    S = np.concatenate([np.zeros(hz.shape[:-1] + (1,)), cs], axis=-1)
    return np.concatenate([np.exp(S[..., :-1]) * hz, np.exp(S[..., -1:])], axis=-1)


def margin_hazard(GL, GD, excl):
    keep = np.ones(GL.shape[0], bool)
    keep[list(excl)] = False
    R = revcum(GL[keep].sum(0))
    D = GD[keep].sum(0)
    return (D + 0.5) / (R + 1.0)


def ll_fold(cnt, t, excl, A):
    keep = np.ones(cnt["HL"].shape[0], bool)
    keep[list(excl)] = False
    te = cnt["si"] == t
    ci, b, h = cnt["ci"][te], cnt["b"][te], cnt["h"][te]
    sel, pos = np.unique(ci, return_inverse=True)
    R = revcum(cnt["HL"][keep][:, sel].sum(0))
    D = cnt["HD"][keep][:, sel].sum(0)
    hm = margin_hazard(cnt["GL"], cnt["GD"], excl)[cnt["gof"][sel]]
    out = np.zeros(len(A))
    for j, a in enumerate(A):
        P = pmf_of((D + a * hm) / (R + a))
        S = revcum(P)
        pr = np.where(b < h, P[pos, b], S[pos, h])
        out[j] = float(np.log(np.maximum(pr, 1e-300)).sum())
    return out, int(te.sum())


def baseline_np(sub, All):
    import mod25e_clk as ck

    el = sub.el.to_numpy(float)
    hs = sub.hs.to_numpy()
    ss = sub.season.to_numpy()
    b_raw = ck.ebin(el)
    ea = All.el.to_numpy(float)
    sa = All.season.to_numpy()
    out = {}
    for s in np.unique(ss):
        te, tr = ss == s, sa != s
        e = ea[tr]
        cl = np.where(e > ck.TAIL, float(e[e <= ck.TAIL].mean()), e)
        cnt = np.bincount(ck.ebin(cl), minlength=ck.NBIN).astype(float)
        p = (cnt + 1.0) / (cnt.sum() + ck.NBIN)
        P = collapse(np.tile(p, (int(te.sum()), 1)), hs[te])
        out[int(s)] = (ll_of(P, b_raw[te]), int(te.sum()))
    return out


def offset_rows():
    import pandas as pd

    import mod25e_clk as ck

    R = pd.read_parquet(ck.REAL, columns=["game_id", "play_id", "qtr", "gsr", "season"])
    R = R[R.qtr.isin([1, 3])].sort_values(["game_id", "play_id"], kind="stable").groupby(["game_id", "qtr"]).head(1)
    R["off"] = np.clip((R.qtr.map(START) - R.gsr.astype(float)).round().astype(int), 0, OFFB - 1)
    return R.reset_index(drop=True)


def offset_pmf(O, qs, a, excl=None):
    ok = np.ones(len(O), bool) if excl is None else (O.season.to_numpy() != excl)
    out = {}
    for q in ((1, 3) if qs else (0,)):
        m = ok & ((O.qtr.to_numpy() == q) if qs else True)
        cnt = np.bincount(O.off.to_numpy()[m], minlength=OFFB).astype(float)
        out[q] = (cnt + a / OFFB) / (cnt.sum() + a)
    return out


def fit_offset():
    import mod25e_clk as ck

    O = offset_rows()
    seasons = np.unique(O.season.to_numpy())
    res = []
    for qs in (0, 1):
        for a in ck.SMOOTH:
            tot = 0.0
            for s in seasons:
                P = offset_pmf(O, qs, a, s)
                te = O[O.season == s]
                for q, g in te.groupby("qtr"):
                    tot += float(np.log(P[int(q) if qs else 0][g.off.to_numpy()]).sum())
            res.append({"qs": qs, "a": a, "ll": tot / len(O)})
    res.sort(key=lambda r: -r["ll"])
    base = max((r for r in res if r["qs"] == 0), key=lambda r: r["ll"])
    return {"n": int(len(O)), "mean_off": float(O.off.mean()), "mean_off_q": {str(q): float(g.off.mean()) for q, g in O.groupby("qtr")}, "best": res[0], "pooled_best": base, "uniform_ll": float(-np.log(OFFB)), "looks": len(res), "top3": res[:3]}


def fit2():
    import itertools

    import pandas as pd

    import mod25e_clk as ck

    R = rows2()
    seasons = np.unique(R.season.to_numpy())
    ns = len(seasons)
    A = list(ck.SMOOTH)
    f1 = json.loads(FIT.read_text(encoding="utf-8"))["classes"]
    AllNP = pd.read_parquet(ck.REAL, columns=["code", "el", "season"])
    AllNP = AllNP[AllNP.code == 6]
    out = {"drop": R.attrs["drop"], "classes": {}, "smooth": A, "looks": 0, "folds": int(ns)}
    for k, nm in enumerate(CLASSES2):
        sub = R[R.cls == k].reset_index(drop=True)
        if len(sub) < 200:
            continue
        if k == 6:
            bt = baseline_np(sub, AllNP)
            base_per = {str(s): v[0] / v[1] for s, v in bt.items()}
            clq1_per = None
        else:
            base_per = f1[nm]["base_trunc_per"]
            clq1_per = f1[nm]["best"]["per"]
        specs, L1, L2, NS = [], [], [], None
        for en, sg, to, qs in itertools.product(EDGE_SETS, SIGNS, TOS, QS):
            cnt = counts(sub, ck.EDGES[en], sg, to, qs, seasons)
            l1 = np.zeros((len(A), ns))
            l2 = np.zeros((len(A), ns, ns))
            nt = np.zeros(ns)
            for t in range(ns):
                v, n = ll_fold(cnt, t, (t,), A)
                l1[:, t] = v
                nt[t] = n
                for s in range(ns):
                    if s != t:
                        l2[:, s, t] = ll_fold(cnt, t, (s, t), A)[0]
            specs.append((en, sg, to, qs))
            L1.append(l1)
            L2.append(l2)
            NS = nt
        L1 = np.array(L1)
        L2 = np.array(L2)
        out["looks"] += L1.shape[0] * L1.shape[1]
        tot = L1.sum(2)
        i, j = np.unravel_index(int(np.argmax(tot)), tot.shape)
        n_all = float(NS.sum())
        best = {"edges": specs[i][0], "sign": specs[i][1], "to": specs[i][2], "qs": specs[i][3], "a": A[j], "ll": float(tot[i, j] / n_all), "per": {str(int(s)): float(L1[i, j, t] / NS[t]) for t, s in enumerate(seasons)}, "ns": {str(int(s)): int(NS[t]) for t, s in enumerate(seasons)}}
        c0 = [x for x, sp in enumerate(specs) if sp == ("none", "none", "none", 0)][0]
        j0 = int(np.argmax(tot[c0]))
        class_only = {"a": A[j0], "ll": float(tot[c0, j0] / n_all), "per": {str(int(s)): float(L1[c0, j0, t] / NS[t]) for t, s in enumerate(seasons)}}
        nested, picks = 0.0, []
        for s in range(ns):
            sc = L2[:, :, s, :].sum(2)
            x, y = np.unravel_index(int(np.argmax(sc)), sc.shape)
            nested += float(L1[x, y, s])
            picks.append([specs[x][0], specs[x][1], specs[x][2], specs[x][3], A[y]])
        order = np.argsort(-tot.ravel())[:3]
        top3 = [{"edges": specs[o // len(A)][0], "sign": specs[o // len(A)][1], "to": specs[o // len(A)][2], "qs": specs[o // len(A)][3], "a": A[o % len(A)], "ll": float(tot.ravel()[o] / n_all)} for o in order]
        bp = sum(base_per[str(int(s))] * NS[t] for t, s in enumerate(seasons)) / n_all
        c1 = None if clq1_per is None else sum(clq1_per[str(int(s))] * NS[t] for t, s in enumerate(seasons)) / n_all
        out["classes"][nm] = {"n": int(len(sub)), "n_censored": int(sub["last"].sum()), "base_trunc_ll": bp, "base_trunc_per": base_per, "clq1_ll": c1, "clq1_per": clq1_per, "class_only": class_only, "best": best, "nested_ll": nested / n_all, "nested_picks": picks, "top3": top3, "wins_vs_base_trunc": int(sum(1 for s in base_per if best["per"][s] > base_per[s])), "wins_vs_clq1": None if clq1_per is None else int(sum(1 for s in clq1_per if best["per"][s] > clq1_per[s]))}
        c = out["classes"][nm]
        print(nm, len(sub), "cens", c["n_censored"], "base_t %.4f clq1 %s class_only %.4f best %.4f nested %.4f" % (bp, "%.4f" % c1 if c1 is not None else "-", class_only["ll"], best["ll"], c["nested_ll"]), best["edges"], best["sign"], best["to"], best["qs"], best["a"], "wins", c["wins_vs_base_trunc"], c["wins_vs_clq1"], flush=True)
    tot_n = sum(v["n"] for v in out["classes"].values())
    out["n_total"] = tot_n
    out["gain_best_vs_base_trunc"] = sum(v["n"] * (v["best"]["ll"] - v["base_trunc_ll"]) for v in out["classes"].values()) / tot_n
    out["gain_nested_vs_base_trunc"] = sum(v["n"] * (v["nested_ll"] - v["base_trunc_ll"]) for v in out["classes"].values()) / tot_n
    out["gain_class_only_vs_base_trunc"] = sum(v["n"] * (v["class_only"]["ll"] - v["base_trunc_ll"]) for v in out["classes"].values()) / tot_n
    c4 = [v for kk, v in out["classes"].items() if v["clq1_ll"] is not None]
    n4 = sum(v["n"] for v in c4)
    out["gain_best_vs_clq1_4cls"] = sum(v["n"] * (v["best"]["ll"] - v["clq1_ll"]) for v in c4) / n4
    out["gain_nested_vs_clq1_4cls"] = sum(v["n"] * (v["nested_ll"] - v["clq1_ll"]) for v in c4) / n4
    fb, f1g = {}, {}
    for s in seasons:
        ss = str(int(s))
        fb[ss] = sum((v["best"]["per"][ss] - v["base_trunc_per"][ss]) * v["best"]["ns"][ss] for v in out["classes"].values()) / sum(v["best"]["ns"][ss] for v in out["classes"].values())
        f1g[ss] = sum((v["best"]["per"][ss] - v["clq1_per"][ss]) * v["best"]["ns"][ss] for v in c4) / sum(v["best"]["ns"][ss] for v in c4)
    out["fold_gain_vs_base_trunc"] = fb
    out["fold_gain_vs_clq1"] = f1g
    out["fold_wins_vs_base_trunc"] = int(sum(1 for v in fb.values() if v > 0))
    out["fold_wins_vs_clq1"] = int(sum(1 for v in f1g.values() if v > 0))
    out["offset"] = fit_offset()
    out["looks"] += out["offset"]["looks"]
    FIT2.parent.mkdir(parents=True, exist_ok=True)
    FIT2.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"], "n", tot_n)
    print("gain per play best v cleaned-pool", round(out["gain_best_vs_base_trunc"], 4), "nested", round(out["gain_nested_vs_base_trunc"], 4), "class-only", round(out["gain_class_only_vs_base_trunc"], 4))
    print("gain v CLQ=1 (4 classes) best", round(out["gain_best_vs_clq1_4cls"], 4), "nested", round(out["gain_nested_vs_clq1_4cls"], 4), "fold wins v clq1", out["fold_wins_vs_clq1"], "v base", out["fold_wins_vs_base_trunc"], "of", ns)
    print("fold gain v base", {k: round(v, 4) for k, v in fb.items()})
    print("fold gain v clq1", {k: round(v, 4) for k, v in f1g.items()})
    print("offset", out["offset"])
    return out


def tables2(sub, spec, seasons):
    import mod25e_clk as ck

    edges, sign, to, qs, a = ck.EDGES[spec["edges"]], spec["sign"], spec["to"], int(spec["qs"]), float(spec["a"])
    cnt = counts(sub, edges, sign, to, qs, seasons)
    hmg = margin_hazard(cnt["GL"], cnt["GD"], ())
    R = revcum(cnt["HL"].sum(0))
    D = cnt["HD"].sum(0)
    P = pmf_of((D + a * hmg[cnt["gof"]]) / (R + a))
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0
    gc = np.cumsum(pmf_of(hmg), axis=1)
    gc[:, -1] = 1.0
    Rt = revcum(cnt["GL"].sum((0, 1)))
    mc = np.cumsum(pmf_of((cnt["GD"].sum((0, 1)) + 0.5) / (Rt + 1.0)))
    mc[-1] = 1.0
    tail = sub.el.to_numpy(float)[(~sub["last"].to_numpy(bool)) & (cnt["b"] >= NH)]
    return {"edges": edges, "sign": sign, "to": to, "qs": qs, "cell": {int(u): cdf[i] for i, u in enumerate(cnt["uc"])}, "grp": {int(u): gc[i] for i, u in enumerate(cnt["uh"])}, "marg": mc, "tail": tail}


def install_clq2():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    d = json.loads(FIT2.read_text(encoding="utf-8"))
    R = rows2()
    seasons = np.unique(R.season.to_numpy())
    tabs = {}
    for k, nm in enumerate(CLASSES2):
        if nm in d["classes"]:
            tabs[k] = tables2(R[R.cls == k].reset_index(drop=True), d["classes"][nm]["best"], seasons)
    ob = d["offset"]["best"]
    opm = offset_pmf(offset_rows(), int(ob["qs"]), float(ob["a"]))
    ocdf = {q: np.cumsum(p) for q, p in opm.items()}
    for v in ocdf.values():
        v[-1] = 1.0
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        if qtr not in (1, 3) or code not in CODES2:
            return drawn
        hs = clock_val - OFFSET[qtr]
        if hs <= 0:
            return drawn
        k = 6 if code == 6 else int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        tb = tabs.get(k)
        if tb is None:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT2, cseed)
        rng = st["rng"]
        cv = int(ck.cell([hs], [score_diff], [float(drawn["off_to_used"])], [float(drawn["def_to_used"])], tb["edges"], tb["sign"], tb["to"])[0])
        hv = int(ck.hbin([hs], tb["edges"])[0])
        if tb["qs"]:
            q3 = int(qtr == 3)
            cv, hv = cv * 2 + q3, hv * 2 + q3
        cdf = tb["cell"].get(cv)
        if cdf is None:
            cdf = tb["grp"].get(hv, tb["marg"])
        off = 0.0
        if clock_val == START[qtr]:
            oc = ocdf[qtr if len(ocdf) > 1 else 0]
            off = float(min(int(np.searchsorted(oc, rng.random(), side="right")), OFFB - 1, int(hs)))
        bn = int(np.searchsorted(cdf, rng.random(), side="right"))
        if bn >= NPB - 1:
            tl = tb["tail"]
            val = float(tl[int(rng.random() * len(tl))]) if len(tl) else float(NPB - 1)
        else:
            val = float(bn)
        new = off + float(min(val, hs - off))
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
