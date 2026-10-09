import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "cky" / "fit.json"
SALT = 4331
SD_LO = -3.0
SD_HI = 0.0
HP = 45.0
HMAX = 120.0
CLS = (0, 2, 3, 4, 5)
PCTS = (90.0, 95.0, 99.0)
QSS = ("q4", "pool")
AEDGES = {"none": [], "w": [20.0], "ww": [10.0, 20.0, 30.0]}
SPLITS = ("pool", "three", "five")
PARENTS = ("band", "cku")
AS = (0.5, 2.0, 8.0, 32.0, 128.0)
WS = (45.0, 60.0, 90.0, 120.0)
BEDGES = {"none": [], "w": [30.0], "ww": [20.0, 45.0]}
MODES = ("rem", "el")
BAS = (0.5, 2.0, 8.0, 32.0)
PRIOR = 2.0
NK = 20


def enabled():
    return os.environ.get("CKY") == "1"


def rows():
    import mod25e_clk as ck

    R = ck.real_rows()
    R = R[R.cls.isin(CLS) & R.qtr.isin([2, 4]) & R.sd.between(SD_LO, SD_HI) & (R.hs <= HMAX)].copy()
    R["cens"] = (R.el >= R.hs).astype(float)
    R["rem"] = (R.hs - R.el).clip(lower=0)
    return R.reset_index(drop=True)


def fg_attempts():
    import pandas as pd

    import mod25e_clk as ck

    F = pd.read_parquet(ck.REAL)
    F = F[(F.code == 3) & F.qtr.isin([2, 4]) & F.sd.between(SD_LO, SD_HI)].copy()
    hs = np.where(F.qtr == 2, F.gsr - 1800.0, F.gsr.astype(float))
    F = F[(hs > 0) & (hs <= HMAX)]
    return F.season.to_numpy(), F.yl.to_numpy(float)


def arrays(R):
    fs, fy = fg_attempts()
    D = {c: R[c].to_numpy() for c in ("season", "qtr", "cls", "hs", "yl", "el", "rem", "cens", "oto", "dto", "otu", "dtu")}
    D["hs"] = D["hs"].astype(float)
    D["AD"] = D["hs"] <= HP
    D["fgs"], D["fgy"] = fs, fy
    D["key"] = (R.game_id.astype(str) + "_" + R.play_id.astype(int).astype(str)).to_numpy()
    D["pcku"] = np.full(len(R), np.nan)
    return D


def attach_cku(D, R):
    import mod25e_cks as cks
    import mod25e_cku as cku

    cs = json.loads(cku.FIT.read_text(encoding="utf-8"))["classes"]
    U = cku.rows()
    pos = {k: i for i, k in enumerate(D["key"])}
    for nm, s in cs.items():
        sub = U[U.cls == s["k"]].reset_index(drop=True)
        p = cku.end_fit(sub, s["spec"], s["a1"], s["a2"], True)
        ks = (sub.game_id.astype(str) + "_" + sub.play_id.astype(int).astype(str)).to_numpy()
        for j, kk in enumerate(ks):
            i = pos.get(kk)
            if i is not None:
                D["pcku"][i] = p[j]
    return D


def ypct(D, pct, excl):
    m = ~np.isin(D["fgs"], list(excl))
    return float(np.percentile(D["fgy"][m], pct))


def hbin(hs, edges):
    return np.searchsorted(np.asarray(edges, float), hs, side="left") if len(edges) else np.zeros(len(hs), int)


def cellkey(cls, hs, split, edges):
    h = hbin(hs, edges)
    nh = len(edges) + 1
    if split == "pool":
        c = np.zeros(len(cls), int)
    elif split == "three":
        c = np.select([cls == 4, cls == 5], [1, 2], 0)
    else:
        c = np.select([cls == 0, cls == 2, cls == 3, cls == 4], [0, 1, 2, 3], 4)
    return c * nh + h


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def a_eval(D, spec, excl, te_season, cache):
    qs, pct, split, en, par = spec
    Y = ypct(D, pct, excl)
    kk = (split, en)
    if kk not in cache:
        cache[kk] = cellkey(D["cls"], D["hs"], split, AEDGES[en])
    key = cache[kk]
    q = (D["qtr"] == 4) if qs == "q4" else np.ones(len(key), bool)
    tr = ~np.isin(D["season"], list(excl)) & D["AD"] & q
    inr = D["yl"] <= Y
    mi, mo = tr & inr, tr & ~inr
    ni = np.bincount(key[mi], minlength=NK).astype(float)
    ki = np.bincount(key[mi], weights=D["cens"][mi], minlength=NK)
    no = np.bincount(key[mo], minlength=NK).astype(float)
    ko = np.bincount(key[mo], weights=D["cens"][mo], minlength=NK)
    g = float(D["cens"][mo].mean()) if mo.any() else 0.0
    te = (D["season"] == te_season) & D["AD"] & (D["qtr"] == 4) & inr
    idx = np.flatnonzero(te)
    if not len(idx):
        return None
    kt = key[idx]
    pb = (ko[kt] + PRIOR * g) / (no[kt] + PRIOR)
    pc = D["pcku"][idx]
    pp = np.where(np.isnan(pc), pb, pc) if par == "cku" else pb
    y = D["cens"][idx]
    out = {}
    for a in AS:
        p = (ki[kt] + a * pp) / (ni[kt] + a)
        out[a] = bll(y, p)
    return {"ll": out, "base": bll(y, pb), "cku": np.where(np.isnan(pc), np.nan, bll(y, np.nan_to_num(pc, nan=0.5))), "n": len(idx), "k": float(y.sum()), "pm": {a: float(((ki[kt] + a * pp) / (ni[kt] + a)).mean()) for a in AS}, "pcku_mean": float(np.nanmean(pc)) if np.isfinite(pc).any() else None, "pb_mean": float(pb.mean())}


def target(D, mode, nb):
    x = D["rem"] if mode == "rem" else D["el"]
    return np.minimum(np.round(x).astype(int), nb - 1)


def b_eval(D, c, W, spec, excl, te_season, pct_inr):
    qs, mode, en, inr_only = spec
    nb = int(W) + 2
    Y = ypct(D, pct_inr, excl)
    ok = (D["cls"] == c) & (D["hs"] <= W) & (D["cens"] == 0)
    q = (D["qtr"] == 4) if qs == "q4" else np.ones(len(ok), bool)
    ir = D["yl"] <= Y
    tr = ok & ~np.isin(D["season"], list(excl)) & q & (ir if inr_only else True)
    te = ok & (D["season"] == te_season) & (D["qtr"] == 4) & ir
    idx = np.flatnonzero(te)
    if not len(idx):
        return None
    e = BEDGES[en]
    h = hbin(D["hs"], e)
    tg = target(D, mode, nb)
    C = np.zeros((len(e) + 1, nb))
    np.add.at(C, (h[tr], tg[tr]), 1.0)
    lim = np.minimum(np.round(D["hs"][idx]).astype(int), nb - 1)
    Cr = C[h[idx]]
    ar = np.arange(len(idx))
    out = {}
    for a in BAS:
        M = Cr + a / nb
        cs = np.cumsum(M, axis=1)
        out[a] = np.log(M[ar, tg[idx]] / cs[ar, lim])
    return {"ll": out, "n": len(idx)}


def nested(seasons, specs, evalfn, avals):
    res = {}
    for s in seasons:
        best = None
        for sp in specs:
            tot = {a: 0.0 for a in avals}
            for t in seasons:
                if t == s:
                    continue
                r = evalfn(sp, {s, t}, t)
                if r is None:
                    continue
                for a in avals:
                    tot[a] += float(r["ll"][a].sum())
            a_ = max(tot, key=tot.get)
            if best is None or tot[a_] > best[0]:
                best = (tot[a_], sp, a_)
        r = evalfn(best[1], {s}, s)
        res[s] = {"spec": best[1], "a": best[2], "r": r}
    return res


def loso_best(seasons, specs, evalfn, avals):
    tot = {}
    for sp in specs:
        for t in seasons:
            r = evalfn(sp, {t}, t)
            if r is None:
                continue
            for a in avals:
                tot[(sp, a)] = tot.get((sp, a), 0.0) + float(r["ll"][a].sum())
    return max(tot, key=tot.get), tot


def fit():
    import mod25e_gfl as gf

    if gf.enabled():
        gf.patch()
    R = rows()
    D = arrays(R)
    D = attach_cku(D, R)
    seasons = sorted(set(D["season"].tolist()))
    out = {"looks": 0, "seasons": [int(s) for s in seasons]}
    cache = {}
    aspecs = list(itertools.product(QSS, PCTS, SPLITS, AEDGES, PARENTS))
    out["looks"] += len(aspecs) * len(AS)
    ev = lambda sp, ex, t: a_eval(D, sp, ex, t, cache)
    nest = nested(seasons, aspecs, ev, AS)
    tm, tb, tc, nn, kk = 0.0, 0.0, 0.0, 0, 0.0
    per = {}
    cm, cc = 0.0, 0.0
    ncku = 0
    for s, v in nest.items():
        r = v["r"]
        if r is None:
            continue
        lm = float(r["ll"][v["a"]].sum())
        lb = float(r["base"].sum())
        ok = ~np.isnan(r["cku"])
        lc = float(r["cku"][ok].sum())
        lmc = float(r["ll"][v["a"]][ok].sum())
        tm += lm
        tb += lb
        nn += r["n"]
        kk += r["k"]
        cm += lmc
        cc += lc
        ncku += int(ok.sum())
        per[int(s)] = {"spec": list(v["spec"]), "a": v["a"], "n": r["n"], "k": r["k"], "ll_mine": lm, "ll_band": lb, "ll_cku_rows": lc, "ll_mine_cku_rows": lmc, "p_mine": r["pm"][v["a"]], "p_cku_mean": r["pcku_mean"], "p_band_mean": r["pb_mean"]}
    out["A_nested"] = {"n": nn, "events": kk, "ll_mine": tm, "ll_band_parent": tb, "gain_vs_band": tm - tb, "n_cku_rows": ncku, "ll_mine_on_cku_rows": cm, "ll_cku_on_cku_rows": cc, "gain_vs_cku": cm - cc, "per_season": per, "fold_wins_vs_band": sum(1 for v in per.values() if v["ll_mine"] > v["ll_band"]), "fold_wins_vs_cku": sum(1 for v in per.values() if v["ll_mine_cku_rows"] > v["ll_cku_rows"]), "folds": len(per)}
    (sp_a, a_a), tot_a = loso_best(seasons, aspecs, ev, AS)
    out["A_final"] = {"spec": list(sp_a), "a": a_a, "ll_loso": tot_a[(sp_a, a_a)]}
    qsA, pctA, splitA, enA, parA = sp_a
    print("A nested", json.dumps({k: v for k, v in out["A_nested"].items() if k != "per_season"}), flush=True)
    print("A final", out["A_final"], flush=True)
    for s, v in per.items():
        print(" A fold", s, v["spec"], v["a"], "n", v["n"], "k", v["k"], "mine", round(v["ll_mine"], 3), "band", round(v["ll_band"], 3), "cku_rows", round(v["ll_cku_rows"], 3), "mine_cku_rows", round(v["ll_mine_cku_rows"], 3), flush=True)
    out["B"] = {}
    for W in WS:
        for c in CLS:
            bs_m = list(itertools.product(QSS, MODES, BEDGES, (True,)))
            bs_c = list(itertools.product(QSS, ("el",), BEDGES, (False,)))
            out["looks"] += (len(bs_m) + len(bs_c)) * len(BAS)
            tot_m, tot_c, per_b, choices = 0.0, 0.0, {}, {}
            for s in seasons:
                pct_s = nest[s]["spec"][1]
                evb = lambda sp, ex, t, pct_s=pct_s: b_eval(D, c, W, sp, ex, t, pct_s)
                n_m = nested_one(seasons, s, bs_m, evb)
                n_c = nested_one(seasons, s, bs_c, evb)
                if n_m is None or n_c is None:
                    continue
                lm = float(n_m["r"]["ll"][n_m["a"]].sum())
                lc = float(n_c["r"]["ll"][n_c["a"]].sum())
                per_b[int(s)] = (lm, lc, n_m["r"]["n"])
                choices[int(s)] = [list(n_m["spec"]), n_m["a"]]
                tot_m += lm
                tot_c += lc
            n_tot = sum(v[2] for v in per_b.values())
            wins = sum(1 for v in per_b.values() if v[0] > v[1])
            out["B"][f"{int(W)}_{c}"] = {"n": n_tot, "ll_mine": tot_m, "ll_cur": tot_c, "gain": tot_m - tot_c, "fold_wins": wins, "folds": len(per_b), "choices": choices}
            print("B W", W, "cls", c, "n", n_tot, "mine", round(tot_m, 2), "cur", round(tot_c, 2), "gain", round(tot_m - tot_c, 2), "wins", wins, "/", len(per_b), flush=True)
    final = {}
    for W in WS:
        gain = 0.0
        for c in CLS:
            r = out["B"][f"{int(W)}_{c}"]
            if r["gain"] > 0:
                gain += r["gain"]
        final[str(int(W))] = gain
    Wb = max(WS, key=lambda w: final[str(int(w))])
    out["W_gain"] = final
    out["W"] = Wb
    out["B_final"] = {}
    for c in CLS:
        r = out["B"][f"{int(Wb)}_{c}"]
        use = r["gain"] > 0
        bs = list(itertools.product(QSS, MODES, BEDGES, (True,))) if use else list(itertools.product(QSS, ("el",), BEDGES, (False,)))
        evb = lambda sp, ex, t, c=c: b_eval(D, c, Wb, sp, ex, t, pctA)
        (sp_b, a_b), _ = loso_best(seasons, bs, evb, BAS)
        out["B_final"][str(c)] = {"apply": use, "spec": list(sp_b), "a": a_b}
    out["pct"] = pctA
    out["Y"] = ypct(D, pctA, set())
    print("W gains", final, "W", Wb, "B_final", out["B_final"], "Y", out["Y"], "pct", pctA, "looks", out["looks"], flush=True)
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out, default=float), encoding="utf-8")


def nested_one(seasons, s, specs, evb):
    best = None
    for sp in specs:
        tot = {a: 0.0 for a in BAS}
        for t in seasons:
            if t == s:
                continue
            r = evb(sp, {s, t}, t)
            if r is None:
                continue
            for a in BAS:
                tot[a] += float(r["ll"][a].sum())
        a_ = max(tot, key=tot.get)
        if best is None or tot[a_] > best[0]:
            best = (tot[a_], sp, a_)
    r = evb(best[1], {s}, s)
    if r is None:
        return None
    return {"spec": best[1], "a": best[2], "r": r}


def timeouts():
    f = sys._getframe(2)
    for _ in range(16):
        if f is None:
            break
        loc = f.f_locals
        if "off_to" in loc and "def_to" in loc:
            return float(loc["off_to"]), float(loc["def_to"])
        f = f.f_back
    raise RuntimeError("timeouts not found in call chain")


def install_cky():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    fit_ = json.loads(FIT.read_text(encoding="utf-8"))
    R = rows()
    D = arrays(R)
    spA = fit_["A_final"]["spec"]
    aA = float(fit_["A_final"]["a"])
    qsA, pctA, splitA, enA, parA = spA
    Y = float(fit_["Y"])
    W = float(fit_["W"])
    eA = AEDGES[enA]
    key = cellkey(D["cls"], D["hs"], splitA, eA)
    q = (D["qtr"] == 4) if qsA == "q4" else np.ones(len(key), bool)
    tr = D["AD"] & q
    inr = D["yl"] <= Y
    mi, mo = tr & inr, tr & ~inr
    ni = np.bincount(key[mi], minlength=NK).astype(float)
    ki = np.bincount(key[mi], weights=D["cens"][mi], minlength=NK)
    no = np.bincount(key[mo], minlength=NK).astype(float)
    ko = np.bincount(key[mo], weights=D["cens"][mo], minlength=NK)
    g = float(D["cens"][mo].mean())
    band = (ko + PRIOR * g) / (no + PRIOR)
    E = None
    if parA == "cku":
        import mod25e_cku as cku

        _, E = cku.tables()
    plan = {}
    for c in CLS:
        b = fit_["B_final"][str(c)]
        qs, mode, en, inr_only = b["spec"]
        nb = int(W) + 2
        e = BEDGES[en]
        ok = (D["cls"] == c) & (D["hs"] <= W) & (D["cens"] == 0) & ((D["yl"] <= Y) if inr_only else True) & ((D["qtr"] == 4) if qs == "q4" else True)
        h = hbin(D["hs"], e)
        tg = target(D, mode, nb)
        C = np.zeros((len(e) + 1, nb))
        np.add.at(C, (h[ok], tg[ok]), 1.0)
        plan[c] = {"C": C, "mode": mode, "edges": e, "a": float(b["a"]), "nb": nb}
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if qtr not in (2, 4) or code not in (0, 1, 4, 5) or hs <= 0 or hs > W:
            return drawn
        if not (SD_LO <= score_diff <= SD_HI) or float(yardline) > Y:
            return drawn
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        if k not in CLS:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        rng = st["rng"]
        u1, u2 = rng.random(), rng.random()
        pe = 0.0
        if hs <= HP and (qsA == "pool" or qtr == 4):
            ck_ = int(cellkey(np.array([k]), np.array([hs]), splitA, eA)[0])
            pp = float(band[ck_])
            if E is not None and k in E:
                import mod25e_cku as cku

                oto, dto = timeouts()
                pp = cku.p_end(E[k], hs, score_diff, qtr, oto, dto, cku.used_flag(drawn["off_to_used"], drawn["def_to_used"]))
            pe = (ki[ck_] + aA * pp) / (ni[ck_] + aA)
            if u1 < pe:
                new = float(hs)
            else:
                new = None
        else:
            new = None
        if new is None:
            pl = plan.get(k)
            if pl is None:
                return drawn
            h = int(hbin(np.array([hs]), pl["edges"])[0])
            lim = min(int(round(hs)), pl["nb"] - 1)
            M = pl["C"][h][: lim + 1] + pl["a"] / pl["nb"]
            cdf = np.cumsum(M) / M.sum()
            b = int(min(np.searchsorted(cdf, u2, side="left"), lim))
            new = float(max(hs - b, 0.0)) if pl["mode"] == "rem" else float(min(b, hs))
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
