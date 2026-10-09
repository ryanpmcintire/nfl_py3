import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_cky as base

FIT = REPO / "artifacts" / "mod25e3" / "cky2" / "fit.json"
SALT = 4332
CLS = base.CLS
YS = tuple(float(y) for y in range(36, 46))
HPS = (45.0, 60.0, 75.0, 90.0)
HE = 90.0
NB = int(HE) + 2
AEDG = {"none": [], "w": [20.0], "ww": [10.0, 20.0]}
APAR = ("flat", "cku")
ASPL = ("pool", "five")
AS = base.AS
NKA = 30
BED = {"none": [], "w": [30.0], "ww": [20.0, 45.0]}
BAS = base.BAS
BSC = ("in", "all")
BQS = ("q4", "pool")
CIDX = np.array([0, -1, 1, 2, 3, 4])
NBOOT = 400


def enabled():
    return os.environ.get("CKY2") == "1"


def a_specs():
    return list(itertools.product(AEDG, APAR, ASPL))


def b_specs():
    out = [(mode, sc, qs, en) for mode in ("el", "rem") for sc in BSC for qs in BQS for en in BED]
    return out


def build_z():
    R = base.rows()
    D = base.arrays(R)
    D = base.attach_cku(D, R)
    D["game"] = R.game_id.astype(str).to_numpy()
    D["gsr"] = R.gsr.to_numpy(float)
    popA = (D["qtr"] == 4) & (D["hs"] <= HE)
    Z = {"D": D, "R": R, "popidx": np.flatnonzero(popA), "akeys": {}, "bll": {}, "bd": {}}
    pi = Z["popidx"]
    for nm in ("season", "cls", "hs", "yl", "cens", "game"):
        Z["p" + nm] = D[nm][pi]
    Z["ppcku"] = D["pcku"][pi]
    for c in CLS:
        gi = np.flatnonzero((D["cls"] == c) & (D["hs"] <= HE))
        b = {"gidx": gi}
        for nm in ("season", "qtr", "hs", "yl", "el", "cens", "game"):
            b[nm] = D[nm][gi]
        b["hs"] = b["hs"].astype(float)
        b["e"] = np.minimum(np.floor(b["el"]).astype(int), NB - 1)
        b["m"] = np.minimum(np.where(b["cens"] == 0, np.floor(b["el"]), np.ceil(b["hs"]) - 1).astype(int), NB - 1).clip(min=0)
        b["rem"] = np.clip(np.round(b["hs"] - b["el"]).astype(int), 0, NB - 1)
        b["L"] = np.clip(np.ceil(b["hs"]).astype(int) - 1, 0, NB - 1)
        b["lim"] = np.clip(np.round(b["hs"]).astype(int), 1, NB - 1)
        Z["bd"][c] = b
    return Z


def akey(Z, Y, HP, split, en, nocell):
    kk = (Y, HP, split, en, nocell)
    if kk in Z["akeys"]:
        return Z["akeys"][kk]
    hs, yl, cls = Z["phs"], Z["pyl"], Z["pcls"]
    inn = np.zeros(len(hs), int) if nocell else ((yl <= Y) & (hs <= HP)).astype(int)
    e = AEDG[en]
    nh = len(e) + 1
    h = base.hbin(hs, e)
    c = np.zeros(len(hs), int) if split == "pool" else CIDX[cls]
    key = (inn * 5 + c) * nh + h
    Z["akeys"][kk] = key
    return key


def a_pred(Z, key, par, excl, t):
    s = Z["pseason"]
    tr = ~np.isin(s, list(excl))
    te = s == t
    if not te.any():
        return None
    cn = Z["pcens"]
    n = np.bincount(key[tr], minlength=NKA).astype(float)
    k = np.bincount(key[tr], weights=cn[tr], minlength=NKA)
    g = float(cn[tr].mean())
    pp = Z["ppcku"][te]
    pp = np.where(np.isnan(pp), g, pp) if par == "cku" else np.full(int(te.sum()), g)
    kt = key[te]
    P = np.stack([(k[kt] + a * pp) / (n[kt] + a) for a in AS])
    return P, np.flatnonzero(te)


def km_cdf(b, tr, a):
    unc = b["cens"] == 0
    d = np.bincount(b["e"][tr & unc], minlength=NB).astype(float)
    r = np.bincount(b["m"][tr], minlength=NB).astype(float)[::-1].cumsum()[::-1]
    h0 = d.sum() / max(r.sum(), 1.0)
    h = np.clip((d + a * h0) / (r + a), 1e-9, 1 - 1e-9)
    surv = np.concatenate([[1.0], np.cumprod(1 - h)[:-1]])
    return np.cumsum(h * surv)


def b_train_mask(b, qs, sc, Y, HP, excl):
    tr = ~np.isin(b["season"], list(excl))
    if qs == "q4":
        tr = tr & (b["qtr"] == 4)
    if sc == "in":
        tr = tr & (b["yl"] <= Y) & (b["hs"] <= HP)
    return tr


def b_ll_one(b, sp, Y, HP, excl, te):
    mode, sc, qs, en = sp
    tr = b_train_mask(b, qs, sc, Y, HP, excl)
    idx = np.flatnonzero(te)
    ar = np.arange(len(idx))
    out = np.zeros((len(BAS), len(idx)))
    if mode == "el":
        e = BED[en]
        h = base.hbin(b["hs"], e)
        et, Lt, ht = b["e"][idx], b["L"][idx], h[idx]
        for i, a in enumerate(BAS):
            for z in range(len(e) + 1):
                sel = ht == z
                if not sel.any():
                    continue
                cdf = km_cdf(b, tr & (h == z), a)
                f = np.diff(np.concatenate([[0.0], cdf]))
                out[i][sel] = np.log(np.maximum(f[et[sel]], 1e-300)) - np.log(cdf[Lt[sel]])
    else:
        e = BED[en]
        h = base.hbin(b["hs"], e)
        unc = (b["cens"] == 0) & tr
        C = np.zeros((len(e) + 1, NB))
        np.add.at(C, (h[unc], b["rem"][unc]), 1.0)
        Cr = C[h[idx]]
        bt, lim = b["rem"][idx], b["lim"][idx]
        for i, a in enumerate(BAS):
            M = Cr + a / NB
            cs = np.cumsum(M[:, 1:], axis=1)
            out[i] = np.log(M[ar, np.maximum(bt, 1)]) - np.log(cs[ar, lim - 1])
    return out, idx


def b_all(Z, c, sp, Y, HP, excl, t, te):
    b = Z["bd"][c]
    ck = (c, sp, frozenset(excl), t)
    if ck not in Z["bll"]:
        Z["bll"][ck] = b_ll_one(b, sp, Y, HP, excl, te)
    return Z["bll"][ck]


BASE_SP = ("el", "all", "pool", "none")


def b_eval(Z, c, Y, HP, sp, excl, t):
    b = Z["bd"][c]
    te = (b["season"] == t) & (b["qtr"] == 4) & (b["cens"] == 0)
    if not te.any():
        return None
    mode, sc, qs, en = sp
    ll_base, idx = b_all(Z, c, BASE_SP, Y, HP, excl, t, te)
    inm = (b["yl"][idx] <= Y) & (b["hs"][idx] <= HP)
    if sc == "all":
        ll_spec, _ = b_all(Z, c, sp, Y, HP, excl, t, te)
    else:
        ll_spec, _ = b_ll_one(b, sp, Y, HP, excl, te)
    return np.where(inm[None, :], ll_spec, ll_base), idx


def a_tot(Z, cfg, excl_t_pairs):
    Y, HP = cfg
    res = {}
    for sp in a_specs():
        en, par, split = sp
        key = akey(Z, Y, HP, split, en, False)
        tot = np.zeros(len(AS))
        for ex, t in excl_t_pairs:
            r = a_pred(Z, key, par, ex, t)
            if r is None:
                continue
            P, idx = r
            y = Z["pcens"][idx]
            tot += base.bll(y[None, :], P).sum(axis=1)
        res[sp] = tot
    return res


def b_tot(Z, c, cfg, excl_t_pairs):
    Y, HP = cfg
    res = {}
    for sp in b_specs():
        tot = np.zeros(len(BAS))
        for ex, t in excl_t_pairs:
            r = b_eval(Z, c, Y, HP, sp, ex, t)
            if r is None:
                continue
            tot += r[0].sum(axis=1)
        res[sp] = tot
    return res


def best_of(res):
    bs, ba, bv = None, None, -1e18
    for sp, v in res.items():
        i = int(np.argmax(v))
        if v[i] > bv:
            bs, ba, bv = sp, i, float(v[i])
    return bs, ba, bv


def choose(Z, seasons, pairs):
    best = None
    detail = {}
    for cfg in itertools.product(YS, HPS):
        sa, ia, va = best_of(a_tot(Z, cfg, pairs))
        tot = va
        bb = {}
        for c in CLS:
            sb, ib, vb = best_of(b_tot(Z, c, cfg, pairs))
            tot += vb
            bb[c] = (sb, ib)
        detail[cfg] = tot
        if best is None or tot > best[0]:
            best = (tot, cfg, (sa, ia), bb)
    return best, detail


def boot_pos(gains, games, rng):
    u, inv = np.unique(games, return_inverse=True)
    per = np.bincount(inv, weights=gains, minlength=len(u))
    n = len(u)
    s = np.array([per[rng.integers(0, n, n)].sum() for _ in range(NBOOT)])
    return float((s > 0).mean()), float(np.percentile(s, 5)), float(np.percentile(s, 95))


def cky1_cell_ll(Z, s, f1):
    D = Z["D"]
    Y1, W1 = float(f1["Y"]), float(f1["W"])
    spA = f1["A_final"]["spec"]
    aA = float(f1["A_final"]["a"])
    cell = (D["qtr"] == 4) & (D["hs"] <= base.HP) & (D["yl"] <= Y1)
    tr = cell & (D["season"] != s)
    n, k = float(tr.sum()), float(D["cens"][tr].sum())
    outm = (D["qtr"] == 4) & (D["hs"] <= base.HP) & ~(D["yl"] <= Y1) & (D["season"] != s)
    g = float(D["cens"][outm].mean())
    pb = (D["cens"][outm].sum() + base.PRIOR * g) / (outm.sum() + base.PRIOR)
    te = cell & (D["season"] == s)
    idx = np.flatnonzero(te)
    pc = D["pcku"][idx]
    pp = np.where(np.isnan(pc), pb, pc) if spA[4] == "cku" else np.full(len(idx), pb)
    pe = (k + aA * pp) / (n + aA)
    tot = np.full(len(idx), np.nan)
    ta = np.full(len(idx), np.nan)
    pdeath = np.zeros(len(idx))
    for c in CLS:
        bs = f1["B_final"][str(c)]
        qs, mode, en, inr_only = bs["spec"]
        nb = int(W1) + 2
        e = base.BEDGES[en]
        ok = (D["cls"] == c) & (D["hs"] <= W1) & (D["cens"] == 0) & (D["season"] != s) & (((D["yl"] <= Y1) if inr_only else True)) & ((D["qtr"] == 4) if qs == "q4" else True)
        h = base.hbin(D["hs"], e)
        tg = base.target(D, mode, nb)
        C = np.zeros((len(e) + 1, nb))
        np.add.at(C, (h[ok], tg[ok]), 1.0)
        sel = np.flatnonzero(D["cls"][idx] == c)
        for j in sel:
            i = idx[j]
            lim = min(int(round(D["hs"][i])), nb - 1)
            M = C[h[i]][: lim + 1] + float(bs["a"]) / nb
            cs = np.cumsum(M)
            if D["cens"][i] == 1:
                pd = (M[lim] / cs[lim]) if mode == "el" else 0.0
                tot[j] = np.log(pe[j] + (1 - pe[j]) * pd)
                ta[j] = np.log(pe[j] + (1 - pe[j]) * pd)
            else:
                tot[j] = np.log(1 - pe[j]) + np.log(M[min(tg[i], lim)] / cs[lim])
                ta[j] = np.log(1 - pe[j])
    return idx, tot, ta


def fit():
    Z = build_z()
    D = Z["D"]
    f1 = json.loads(base.FIT.read_text(encoding="utf-8"))
    seasons = sorted(set(D["season"].tolist()))
    fgy = D["fgy"]
    ypcts = {str(int(y)): float((fgy <= y).mean()) for y in YS}
    out = {"seasons": [int(s) for s in seasons], "fg_yl_percentile_of_Y": ypcts, "fg_yl_pcts": {str(p): float(np.percentile(fgy, p)) for p in (50, 75, 90, 95, 99)}}
    nspec = len(a_specs()) * len(AS) + len(CLS) * len(b_specs()) * len(BAS)
    out["looks"] = len(YS) * len(HPS) * nspec
    rng = np.random.default_rng(SALT)
    fold = {}
    tot2 = np.full(len(D["season"]), np.nan)
    ta2 = np.full(len(D["season"]), np.nan)
    ta1 = np.full(len(D["season"]), np.nan)
    pe2 = np.full(len(D["season"]), np.nan)
    gainA = {c: [] for c in CLS}
    gainB = {c: [] for c in CLS}
    gameA = {c: [] for c in CLS}
    gameB = {c: [] for c in CLS}
    cls_fold = {c: {} for c in CLS}
    for s in seasons:
        inner = [t for t in seasons if t != s]
        pairs = [({s, t}, t) for t in inner]
        best, detail = choose(Z, seasons, pairs)
        tot, cfg, (sa, ia), bb = best
        Y, HP = cfg
        en, par, split = sa
        r = a_pred(Z, akey(Z, Y, HP, split, en, False), par, {s}, s)
        P, idx = r
        Pn, _ = a_pred(Z, akey(Z, Y, HP, split, en, True), par, {s}, s)
        y = Z["pcens"][idx]
        llm = base.bll(y, P[ia])
        lln = base.bll(y, Pn[ia])
        gi = Z["popidx"][idx]
        pe2[gi] = P[ia]
        tot2[gi] = llm
        ta2[gi] = llm
        for c in CLS:
            m = Z["pcls"][idx] == c
            gainA[c].append((llm - lln)[m])
            gameA[c].append(Z["pgame"][idx][m])
        bch = {}
        for c in CLS:
            sb, ib = bb[c]
            rb = b_eval(Z, c, Y, HP, sb, {s}, s)
            if rb is None:
                continue
            llb, bidx = rb
            rbase = b_eval(Z, c, Y, HP, BASE_SP, {s}, s)
            gb = llb[ib] - rbase[0][ib]
            gainB[c].append(gb)
            gameB[c].append(Z["bd"][c]["game"][bidx])
            gidx = Z["bd"][c]["gidx"][bidx]
            tot2[gidx] = tot2[gidx] + llb[ib]
            bch[c] = {"spec": list(sb), "a": BAS[ib]}
        fold[int(s)] = {"cfg": [Y, HP], "A": {"spec": list(sa), "a": AS[ia]}, "B": {str(c): v for c, v in bch.items()}, "inner_score": float(tot)}
        wins = sum(1 for v in detail.values() if v > tot - 1e-9)
        print("fold", s, "cfg", cfg, "A", sa, AS[ia], "B", {c: (v["spec"], v["a"]) for c, v in bch.items()}, flush=True)
    out["folds"] = fold
    out["cfg_counts"] = {}
    for v in fold.values():
        kk = f"{int(v['cfg'][0])}_{int(v['cfg'][1])}"
        out["cfg_counts"][kk] = out["cfg_counts"].get(kk, 0) + 1
    out["apply"] = {}
    for c in CLS:
        ga = np.concatenate(gainA[c]) if gainA[c] else np.zeros(0)
        gb = np.concatenate(gainB[c]) if gainB[c] else np.zeros(0)
        gma = np.concatenate(gameA[c]) if gameA[c] else np.zeros(0, str)
        gmb = np.concatenate(gameB[c]) if gameB[c] else np.zeros(0, str)
        g_all = np.concatenate([ga, gb])
        gm = np.concatenate([gma, gmb])
        pp, lo, hi = boot_pos(g_all, gm, rng)
        sfold = {}
        for k_, s in enumerate(seasons):
            sfold[s] = float(gainA[c][k_].sum() + (gainB[c][k_].sum() if k_ < len(gainB[c]) else 0.0)) if k_ < len(gainA[c]) else 0.0
        out["apply"][str(c)] = {"gain": float(g_all.sum()), "gain_A": float(ga.sum()), "gain_B": float(gb.sum()), "probability_positive": pp, "p5": lo, "p95": hi, "fold_wins": int(sum(1 for v in sfold.values() if v > 0)), "folds": len(sfold), "n_rows": int(len(g_all))}
        print("class", c, out["apply"][str(c)], flush=True)
    c1 = (D["qtr"] == 4) & (D["hs"] <= base.HP) & (D["yl"] <= float(f1["Y"]))
    rows1, rows2, per1 = [], [], {}
    t1 = np.full(len(D["season"]), np.nan)
    for s in seasons:
        idx, tt, aa = cky1_cell_ll(Z, s, f1)
        t1[idx] = tt
        ta1[idx] = aa
    both = c1 & ~np.isnan(t1) & ~np.isnan(tot2)
    d = tot2[both] - t1[both]
    gm = D["game"][both]
    pp, lo, hi = boot_pos(d, gm, rng)
    wins = sum(1 for s in seasons if (tot2[both & (D["season"] == s)] - t1[both & (D["season"] == s)]).sum() > 0)
    out["vs_cky1_A_only"] = {"ll_cky2": float(ta2[both].sum()), "ll_cky1": float(ta1[both].sum())}
    out["vs_cky1"] = {"rows": int(both.sum()), "deaths": int(D["cens"][both].sum()), "ll_cky2": float(tot2[both].sum()), "ll_cky1": float(t1[both].sum()), "gain": float(d.sum()), "probability_positive": pp, "p5": lo, "p95": hi, "fold_wins": int(wins), "folds": len(seasons)}
    out["vs_cky1_by_class"] = {str(c): {"rows": int((both & (D["cls"] == c)).sum()), "ll_cky2": float(tot2[both & (D["cls"] == c)].sum()), "ll_cky1": float(t1[both & (D["cls"] == c)].sum())} for c in CLS}
    print("vs cky1", out["vs_cky1"], out["vs_cky1_by_class"], flush=True)
    full = Z["popidx"]
    bins_y = [(0, 30), (30, 36), (36, 40), (40, 45), (45, 101)]
    bins_h = [(0, 10), (10, 20), (20, 45), (45, 60), (60, 90)]
    tab = []
    for (y0, y1) in bins_y:
        for (h0, h1) in bins_h:
            m = (D["qtr"] == 4) & (D["yl"] > y0) & (D["yl"] <= y1) & (D["hs"] > h0) & (D["hs"] <= h1) & ~np.isnan(pe2)
            if m.sum() == 0:
                continue
            tab.append({"yl": [y0, y1], "hs": [h0, h1], "n": int(m.sum()), "real_end": float(D["cens"][m].mean()), "pe_cky2_loso": float(pe2[m].mean())})
    out["pend_by_bin_loso"] = tab
    cfgsel = list(itertools.product(YS, HPS))
    allp = [({t}, t) for t in seasons]
    bestf, detf = choose(Z, seasons, allp)
    totf, cfgf, (saf, iaf), bbf = bestf
    out["cfg"] = {"Y": cfgf[0], "HP": cfgf[1]}
    out["A_final"] = {"spec": list(saf), "a": AS[iaf]}
    out["B_final"] = {}
    for c in CLS:
        sb, ib = bbf[c]
        out["B_final"][str(c)] = {"apply": bool(out["apply"][str(c)]["gain"] > 0), "spec": list(sb), "a": BAS[ib]}
    out["cfg_scores_loso"] = {f"{int(k[0])}_{int(k[1])}": float(v) for k, v in detf.items()}
    print("final", out["cfg"], out["A_final"], out["B_final"], "looks", out["looks"], flush=True)
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out, default=float), encoding="utf-8")


def build_plan(Z, fit_):
    D = Z["D"]
    Y, HP = float(fit_["cfg"]["Y"]), float(fit_["cfg"]["HP"])
    en, par, split = fit_["A_final"]["spec"]
    aA = float(fit_["A_final"]["a"])
    key = akey(Z, Y, HP, split, en, False)
    cn = Z["pcens"]
    n = np.bincount(key, minlength=NKA).astype(float)
    k = np.bincount(key, weights=cn, minlength=NKA)
    plan = {"Y": Y, "HP": HP, "en": en, "par": par, "split": split, "a": aA, "n": n, "k": k, "g": float(cn.mean()), "B": {}}
    for c in CLS:
        bf = fit_["B_final"][str(c)]
        if not bf["apply"]:
            continue
        mode, sc, qs, be = bf["spec"]
        a = float(bf["a"])
        b = Z["bd"][c]
        tr = b_train_mask(b, qs, sc, Y, HP, set())
        e = BED[be]
        h = base.hbin(b["hs"], e)
        if mode == "el":
            plan["B"][c] = {"mode": mode, "edges": e, "cdf": [km_cdf(b, tr & (h == z), a) for z in range(len(e) + 1)]}
        else:
            unc = (b["cens"] == 0) & tr
            C = np.zeros((len(e) + 1, NB))
            np.add.at(C, (h[unc], b["rem"][unc]), 1.0)
            plan["B"][c] = {"mode": mode, "C": C + a / NB, "edges": e}
    return plan


def p_end_cell(plan, k, hs, pp):
    c = 0 if plan["split"] == "pool" else int(CIDX[k])
    e = AEDG[plan["en"]]
    nh = len(e) + 1
    h = int(base.hbin(np.array([hs]), e)[0])
    kk = (1 * 5 + c) * nh + h
    return float((plan["k"][kk] + plan["a"] * pp) / (plan["n"][kk] + plan["a"]))


def draw_new(pl, hs, u):
    if pl["mode"] == "el":
        L = min(max(int(math.ceil(hs)) - 1, 0), NB - 1)
        cdf = pl["cdf"][int(base.hbin(np.array([hs]), pl["edges"])[0])][: L + 1]
        e = int(np.searchsorted(cdf / cdf[-1], u, side="left"))
        return float(min(e, L))
    h = int(base.hbin(np.array([hs]), pl["edges"])[0])
    lim = min(max(int(round(hs)), 1), NB - 1)
    M = pl["C"][h][1 : lim + 1]
    cdf = np.cumsum(M) / M.sum()
    b = int(min(np.searchsorted(cdf, u, side="left"), lim - 1)) + 1
    return float(max(hs - b, 0.0))


def install_cky2():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    fit_ = json.loads(FIT.read_text(encoding="utf-8"))
    Z = build_z()
    plan = build_plan(Z, fit_)
    Y, HP = plan["Y"], plan["HP"]
    E = None
    if plan["par"] == "cku":
        import mod25e_cku as cku

        _, E = cku.tables()
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base_pol = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base_pol(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        if qtr != 4 or code not in (0, 1, 4, 5):
            return drawn
        hs = float(clock_val)
        if hs <= 0 or hs > HP or float(yardline) > Y or not (base.SD_LO <= score_diff <= base.SD_HI):
            return drawn
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        if k not in CLS or k not in plan["B"]:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        rng = st["rng"]
        u1, u2 = rng.random(), rng.random()
        pp = plan["g"]
        if E is not None and k in E:
            import mod25e_cku as cku

            oto, dto = base.timeouts()
            pp = cku.p_end(E[k], hs, score_diff, qtr, oto, dto, cku.used_flag(drawn["off_to_used"], drawn["def_to_used"]))
        pe = p_end_cell(plan, k, hs, pp)
        new = float(hs) if u1 < pe else draw_new(plan["B"][k], hs, u2)
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
        os.environ["CKH"] = "1"
        fit()
