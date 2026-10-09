import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_urg as ug  # noqa: E402

FIT = REPO / "artifacts" / "mod25e3" / "urg2" / "fit.json"
FITB = REPO / "artifacts" / "mod25e3" / "urg2" / "fit_b.json"
SALT = 4341
SALT_B = 4342
LO = 15.0
HI = 180.0
HB = 120.0
GRIDS = (15.0, 30.0)
WINDOWS = (4, 2)
CLASSES = (0, 1, 2, 3)
NB = ug.NB
EV = ug.EV
MIN_WINS = ug.MIN_WINS
YLO, YHI = -12, 60
NY = YHI - YLO + 1
B_EDGES = {"none": [], "w": [60.0], "ww": [30.0, 60.0, 90.0]}
B_BANDS = ("none", "sign3", "trail5")
B_TOS = (0, 1)
B_AS = (2.0, 8.0, 32.0, 128.0, 512.0)
B_SCHEMES = [{"edges": e, "band": b, "to": t, "a": a} for e, b, t, a in itertools.product(B_EDGES, B_BANDS, B_TOS, B_AS)]


def enabled():
    return os.environ.get("URG2") == "1"


def nhof(w):
    return int(np.ceil((HI - LO) / w))


def hbin(hs, w):
    return np.clip(np.floor((np.asarray(hs, float) - LO - 1e-9) / w).astype(int), 0, nhof(w) - 1)


def parent_rows(C, a0, cell, used, e, loo, nh):
    CU = C.reshape(3, nh, NB).sum(1)
    Mp = (CU + 1.0) / (CU.sum(1, keepdims=True) + NB)
    num = C[cell] + a0 * Mp[used]
    n = C[cell].sum(1)
    if loo:
        num[np.arange(len(cell)), e] -= 1.0
        n = n - 1.0
    return num / (n + a0)[:, None]


def build(Rk, q, w):
    W = ((Rk.qtr == q) & (Rk.hs > LO) & (Rk.hs <= HI) & Rk.code.isin([0, 1])).to_numpy()
    D = Rk[W].reset_index(drop=True)
    hs = D.hs.to_numpy(float)
    nh = nhof(w)
    d = {"df": D, "w": w, "nh": nh, "season": D.season.to_numpy(), "e": np.minimum(np.round(D.el.to_numpy(float)).astype(int), NB - 1), "used": ug.used3(D.otu, D.dtu), "hs": hs, "sd": D.sd.to_numpy(float), "oto": D.oto.to_numpy(float), "gid": D.g.to_numpy(), "cens": D.el.to_numpy(float) >= hs - 1e-9}
    d["cell"] = d["used"] * nh + hbin(hs, w)
    d["CT"] = {}
    for s in np.unique(d["season"]):
        m = d["season"] == s
        T = np.zeros((3 * nh, NB))
        np.add.at(T, (d["cell"][m], d["e"][m]), 1.0)
        d["CT"][int(s)] = T
    d["G"] = {ug.skey(sch): ug.group(d["sd"], d["oto"], hs, sch) for sch in ug.SCHEMES}
    return d


def counts_of(d, seasons):
    return sum(d["CT"][int(s)] for s in seasons)


def sub_parent(d, C, a0, idx, loo):
    return parent_rows(C, a0, d["cell"][idx], d["used"][idx], d["e"][idx], loo, d["nh"])


def row_ll(P, e, cens):
    idx = np.arange(len(e))
    tail = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]
    return np.log(np.maximum(np.where(cens, tail[idx, e], P[idx, e]), 1e-300))


def row_mean(P, hs):
    return (P * np.minimum(EV[None, :], hs[:, None])).sum(1)


def score_scheme(d, tr, te, C, a0, sch):
    key = ug.skey(sch)
    P_tr = sub_parent(d, C, a0, tr, True)
    th = ug.fit_theta(P_tr, d["e"][tr], d["G"][key][tr], ug.ngroups(sch), sch["lam"])
    P = ug.tilt(sub_parent(d, C, a0, te, False), th[d["G"][key][te]])
    return row_ll(P, d["e"][te], d["cens"][te]), row_mean(P, d["hs"][te]), th


def free_scheme(d, te, C, a0):
    P = sub_parent(d, C, a0, te, False)
    return row_ll(P, d["e"][te], d["cens"][te]), row_mean(P, d["hs"][te])


def choose_a0(d, seasons):
    C_all = counts_of(d, seasons)
    best, bv = None, -1e18
    for a0 in ug.A0S:
        tot = 0.0
        for t in seasons:
            idx = np.where(d["season"] == t)[0]
            tot += free_scheme(d, idx, C_all - d["CT"][int(t)], a0)[0].sum()
        if tot > bv:
            best, bv = a0, tot
    return best


def clk_P(Rall, d, k, q):
    import mod25e_clk as ck

    Rk = Rall[Rall.cls == k].reset_index(drop=True)
    sp = ug.clk_spec(k)
    b = ck.ebin(Rk.el.to_numpy())
    c = ck.cell(Rk.hs.to_numpy(), Rk.sd.to_numpy(), Rk.oto.to_numpy(), Rk.dto.to_numpy(), sp["edges"], sp["sign"], sp["to"])
    hb = ck.hbin(Rk.hs.to_numpy(), sp["edges"])
    if sp["qs"]:
        q4 = (Rk.qtr.to_numpy() == 4).astype(int)
        c, hb = c * 2 + q4, hb * 2 + q4
    se = Rk.season.to_numpy()
    W = ((Rk.qtr == q) & (Rk.hs > LO) & (Rk.hs <= HI) & Rk.code.isin([0, 1])).to_numpy()
    out = np.zeros((len(d["e"]), ck.NBIN))
    for s in np.unique(se):
        tr = se != s
        te = (se == s) & W
        if not te.any():
            continue
        mh = np.zeros((int(hb.max()) + 1, ck.NBIN))
        np.add.at(mh, (hb[tr], b[tr]), 1.0)
        margh = (mh + 1.0) / (mh.sum(1, keepdims=True) + ck.NBIN)
        uc, inv = np.unique(c[tr], return_inverse=True)
        cnt = np.zeros((len(uc), ck.NBIN))
        np.add.at(cnt, (inv, b[tr]), 1.0)
        nn = cnt.sum(1)
        idx = np.minimum(np.searchsorted(uc, c[te]), len(uc) - 1)
        hit = uc[idx] == c[te]
        cc = np.where(hit[:, None], cnt[idx], 0.0)
        nh = np.where(hit, nn[idx], 0.0)
        out[d["season"] == s] = (cc + sp["a"] * margh[hb[te]]) / (nh + sp["a"])[:, None]
    return out


def cdw_P(d, k, q):
    import mod25e_cdw as cw

    b = json.loads(cw.FIT.read_text(encoding="utf-8"))["chosen"]
    spec, a = b["spec"], float(b["a"])
    R = cw.rows()
    out = np.zeros((len(d["e"]), NB))
    for s in np.unique(d["season"]):
        tr = (R.season != s) & (R.hs <= spec["hz"])
        te = (R.season == s) & (R.qtr == q) & (R.hs <= HI) & (R.cls == k)
        Rte = R[te]
        m = d["season"] == s
        if len(Rte) != int(m.sum()):
            raise RuntimeError("cdw row alignment")
        out[m] = cw.predict(R[tr], Rte, spec, a)
    return out


def hs_bias(d, preds, actual):
    rows = {}
    for nm, lo, hi in (("hs15_30", 15, 30), ("hs30_45", 30, 45), ("hs45_60", 45, 60), ("hs60_90", 60, 90), ("hs90_120", 90, 120), ("hs120_180", 120, 180)):
        m = (d["hs"] > lo) & (d["hs"] <= hi)
        if m.sum() >= 5:
            rows[nm] = {"n": int(m.sum()), "actual": float(actual[m].mean()), **{p: float(v[m].mean()) for p, v in preds.items()}}
    return rows


def run_class(Rall, q, k, rng):
    Rk = Rall[Rall.cls == k].reset_index(drop=True)
    ds = {w: build(Rk, q, w) for w in GRIDS}
    d0 = ds[GRIDS[0]]
    n = len(d0["e"])
    seasons = sorted(int(s) for s in np.unique(d0["season"]))
    if n < 200 or len(seasons) < 9:
        return None
    keys = [(w, ug.skey(s)) for w in GRIDS for s in ug.SCHEMES]
    sch_of = {(w, ug.skey(s)): s for w in GRIDS for s in ug.SCHEMES}
    llm = {kk: np.zeros(n) for kk in keys}
    mnm = {kk: np.zeros(n) for kk in keys}
    ll_f = {w: np.zeros(n) for w in GRIDS}
    mn_f = {w: np.zeros(n) for w in GRIDS}
    ll_n, mn_n = np.zeros(n), np.zeros(n)
    choices = {}
    for s in seasons:
        trs = [t for t in seasons if t != s]
        te = np.where(d0["season"] == s)[0]
        tr = np.where(d0["season"] != s)[0]
        inner = {}
        a0s = {}
        for w in GRIDS:
            d = ds[w]
            a0s[w] = choose_a0(d, trs)
            C_tr = counts_of(d, trs)
            ll_f[w][te], mn_f[w][te] = free_scheme(d, te, C_tr, a0s[w])
            for sch in ug.SCHEMES:
                kk = (w, ug.skey(sch))
                llm[kk][te], mnm[kk][te], _ = score_scheme(d, tr, te, C_tr, a0s[w], sch)
                tot = 0.0
                for t in trs:
                    itr = np.where((d["season"] != s) & (d["season"] != t))[0]
                    ite = np.where(d["season"] == t)[0]
                    tot += score_scheme(d, itr, ite, C_tr - d["CT"][int(t)], a0s[w], sch)[0].sum()
                inner[kk] = tot
        pick = max(inner, key=inner.get)
        ll_n[te], mn_n[te] = llm[pick][te], mnm[pick][te]
        choices[s] = {"a0": a0s[pick[0]], "w": pick[0], "scheme": pick[1]}
        print(q, ug.CLASS_NAMES[k], s, pick, a0s[pick[0]], flush=True)
    tot = {kk: float(v.sum()) for kk, v in llm.items()}
    fixed = max(tot, key=tot.get)
    wf, fsch = fixed[0], sch_of[fixed]
    df = ds[wf]
    a0f = choose_a0(df, seasons)
    C_all = counts_of(df, seasons)
    P_all = sub_parent(df, C_all, a0f, np.arange(n), True)
    th = ug.fit_theta(P_all, df["e"], df["G"][fixed[1]], ug.ngroups(fsch), fsch["lam"])
    Pc = clk_P(Rall, d0, k, q)
    ll_c, mn_c = row_ll(Pc, d0["e"], d0["cens"]), row_mean(Pc, d0["hs"])
    hi45 = d0["hs"] > 45.0
    cdw = None
    try:
        dsub = {"e": d0["e"][hi45], "season": d0["season"][hi45], "hs": d0["hs"][hi45]}
        Pw = cdw_P(dsub, k, q)
        cdw = (row_ll(Pw, dsub["e"], d0["cens"][hi45]), row_mean(Pw, dsub["hs"]))
    except Exception as ex:
        print("cdw unavailable", repr(ex), flush=True)
    actual = np.minimum(d0["df"].el.to_numpy(float), d0["hs"])
    out = {"n": n, "seasons": seasons, "cens_share": float(d0["cens"].mean()), "looks": len(keys), "choices": {str(s): v for s, v in choices.items()}, "fixed": {"w": wf, "scheme": fsch, "a0": a0f, "theta": [float(x) for x in th]}}
    out["ll"] = {"clk": float(ll_c.mean()), "free_w15": float(ll_f[15.0].mean()), "free_w30": float(ll_f[30.0].mean()), "nested": float(ll_n.mean()), "fixed_loso": float(llm[fixed].mean())}
    if cdw is not None:
        out["ll"]["cdw_hs45"] = float(cdw[0].mean())
        out["ll"]["nested_hs45"] = float(ll_n[hi45].mean())
        out["ll"]["clk_hs45"] = float(ll_c[hi45].mean())
    out["mean_ll_in_sample_gap"] = float(llm[fixed].mean() - ll_n.mean())
    out["fold_wins_grid"] = {str(w): sum(1 for v in choices.values() if v["w"] == w) for w in GRIDS}
    for nm, base, mask in (("clk", ll_c, None), ("free15", ll_f[15.0], None), ("free30", ll_f[30.0], None)):
        for tag, arr in (("nested", ll_n), ("fixed", llm[fixed])):
            diff = arr - base
            w_, per = ug.season_wins(diff, d0["season"])
            out.setdefault("gain_" + tag, {})[nm] = {**ug.boot(diff, d0["gid"], rng), "wins": w_, "per_season": per}
    if cdw is not None:
        diff = ll_n[hi45] - cdw[0]
        w_, per = ug.season_wins(diff, d0["season"][hi45])
        out["gain_nested"]["cdw_hs45"] = {**ug.boot(diff, d0["gid"][hi45], rng), "wins": w_, "per_season": per}
        diff = ll_c[hi45] - cdw[0]
        w_, per = ug.season_wins(diff, d0["season"][hi45])
        out["clk_vs_cdw_hs45"] = {**ug.boot(diff, d0["gid"][hi45], rng), "wins": w_, "per_season": per}
    preds = {"clk": mn_c, "urg2_nested": mn_n, "urg_free15": mn_f[15.0]}
    out["bias"] = ug.bias_table(d0, preds, actual)
    out["bias_hs"] = hs_bias(d0, preds, actual)
    out["by_scheme_ll"] = {"%s|%s" % kk: float(v.mean()) for kk, v in llm.items()}
    g = out["gain_nested"]["clk"]
    out["apply"] = bool(g["p_positive"] > 0.5 and g["wins"] >= MIN_WINS and g["mean"] > 0)
    c = out.get("clk_vs_cdw_hs45")
    out["clk_apply"] = bool((not out["apply"]) and c is not None and c["p_positive"] > 0.5 and c["wins"] >= MIN_WINS and c["mean"] > 0)
    out["source"] = "urg2" if out["apply"] else ("clk" if out["clk_apply"] else "none")
    return out


def fit():
    Rall = ug.load_rows()
    rng = np.random.default_rng(SALT)
    out = {"looks": 0, "lo": LO, "hi": HI, "grids": list(GRIDS), "windows": {}}
    for q in WINDOWS:
        out["windows"][str(q)] = {}
        for k in CLASSES:
            r = run_class(Rall, q, k, rng)
            if r is None:
                continue
            out["looks"] += r["looks"]
            out["windows"][str(q)][str(k)] = r
            g = r["gain_nested"]
            print("Q%d %s n=%d cens %.3f ll clk %.4f free15 %.4f free30 %.4f nested %.4f | vs clk %+.4f [%+.4f,%+.4f] P %.2f wins %d | grid folds %s | source %s" % (q, ug.CLASS_NAMES[k], r["n"], r["cens_share"], r["ll"]["clk"], r["ll"]["free_w15"], r["ll"]["free_w30"], r["ll"]["nested"], g["clk"]["mean"], g["clk"]["lo"], g["clk"]["hi"], g["clk"]["p_positive"], g["clk"]["wins"], r["fold_wins_grid"], r["source"]), flush=True)
            for nm, row in list(r["bias"].items()) + list(r["bias_hs"].items()):
                print("   ", nm, json.dumps({a: round(b, 2) if isinstance(b, float) else b for a, b in row.items()}), flush=True)
            FIT.parent.mkdir(parents=True, exist_ok=True)
            FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"])


def b_rows():
    import mod25e_clk as ck

    R = ck.real_rows()
    R = R[(R.code == 1) & R.cls.isin([0, 3]) & R.qtr.isin([2, 4])].copy()
    R["inc"] = (R.cls == 0).astype(float)
    R["yb"] = np.clip(np.round(R.yards.to_numpy(float)), YLO, YHI).astype(int) - YLO
    return R.reset_index(drop=True)


def b_key(hs, sd, oto, sch):
    h = np.searchsorted(np.asarray(B_EDGES[sch["edges"]], float), np.asarray(hs, float), side="left") if B_EDGES[sch["edges"]] else np.zeros(len(np.atleast_1d(hs)), int)
    sd = np.atleast_1d(np.asarray(sd, float))
    if sch["band"] == "sign3":
        b = (np.sign(sd) + 1).astype(int)
    elif sch["band"] == "trail5":
        b = ug.band(sd, "trail5")
    else:
        b = np.zeros(len(sd), int)
    t = (np.atleast_1d(np.asarray(oto, float)) > 0).astype(int) if sch["to"] else np.zeros(len(sd), int)
    return (h * 6 + b) * 2 + t


def b_fold(R, sch, a, tr, te, kind):
    late = R.hs.to_numpy() <= HB
    key = b_key(R.hs.to_numpy(), R.sd.to_numpy(), R.oto.to_numpy(), sch)
    nk = (len(B_EDGES[sch["edges"]]) + 1) * 12
    base = tr & ~late
    mt = tr & late
    if kind == "inc":
        c = float(R.inc.to_numpy()[base].mean())
        n = np.bincount(key[mt], minlength=nk).astype(float)
        kk = np.bincount(key[mt], weights=R.inc.to_numpy()[mt], minlength=nk)
        p = (kk[key[te]] + a * c) / (n[key[te]] + a)
        y = R.inc.to_numpy()[te]
        return np.log(np.where(y > 0, p, 1 - p)), np.full(int(te.sum()), c)
    cm = (R.cls.to_numpy() == 3)
    Pb = np.bincount(R.yb.to_numpy()[base & cm], minlength=NY).astype(float) + 1.0
    Pb[-YLO] = 0.0
    Pb = Pb / Pb.sum()
    cnt = np.zeros((nk, NY))
    np.add.at(cnt, (key[mt & cm], R.yb.to_numpy()[mt & cm]), 1.0)
    nn = cnt.sum(1)
    tm = te & cm
    P = (cnt[key[tm]] + a * Pb[None, :]) / (nn[key[tm]] + a)[:, None]
    return np.log(P[np.arange(int(tm.sum())), R.yb.to_numpy()[tm]]), np.log(Pb[R.yb.to_numpy()[tm]])


def fit_b():
    R = b_rows()
    rng = np.random.default_rng(SALT_B)
    late = R.hs.to_numpy() <= HB
    seasons = sorted(int(s) for s in np.unique(R.season))
    ss = R.season.to_numpy()
    out = {"looks": 0, "late_n": int(late.sum()), "comp_n": int((late & (R.cls.to_numpy() == 3)).sum()), "looks_per_kind": len(B_SCHEMES)}
    for kind in ("inc", "yards"):
        rowmask = late if kind == "inc" else late & (R.cls.to_numpy() == 3)
        idx_late = np.flatnonzero(rowmask)
        pos = {int(i): j for j, i in enumerate(idx_late)}
        tot = {}
        folds = {}
        for si, sch in enumerate(B_SCHEMES):
            ll = np.zeros(len(idx_late))
            for s in seasons:
                tr, te = ss != s, (ss == s) & late
                l, _ = b_fold(R, sch, sch["a"], tr, te, kind)
                te_idx = np.flatnonzero(te & (rowmask))
                ll[[pos[int(i)] for i in te_idx]] = l
            tot[si] = ll
        sums = {si: float(v.sum()) for si, v in tot.items()}
        nested = np.zeros(len(idx_late))
        choice = {}
        for s in seasons:
            best, bv = None, -1e18
            for si, sch in enumerate(B_SCHEMES):
                v = 0.0
                for t in seasons:
                    if t == s:
                        continue
                    tr = (ss != s) & (ss != t)
                    te = (ss == t) & late
                    v += b_fold(R, sch, sch["a"], tr, te, kind)[0].sum()
                if v > bv:
                    best, bv = si, v
            sel = (ss[idx_late] == s)
            nested[sel] = tot[best][sel]
            choice[str(s)] = B_SCHEMES[best]
            print(kind, s, B_SCHEMES[best], flush=True)
        basell = np.zeros(len(idx_late))
        for s in seasons:
            tr, te = ss != s, (ss == s) & late
            basell[[pos[int(i)] for i in np.flatnonzero(te & rowmask)]] = b_fold(R, B_SCHEMES[0], 1e12, tr, te, kind)[0]
        fixed = max(sums, key=sums.get)
        diff = nested - basell
        w_, per = ug.season_wins(diff, ss[idx_late])
        gain = {**ug.boot(diff, R.g.to_numpy()[idx_late], rng), "wins": w_, "per_season": per}
        dfx = tot[fixed] - basell
        wf_, perf = ug.season_wins(dfx, ss[idx_late])
        gain_fixed = {**ug.boot(dfx, R.g.to_numpy()[idx_late], rng), "wins": wf_}
        rec = {"n": int(len(idx_late)), "ll_base": float(basell.mean()), "ll_nested": float(nested.mean()), "ll_fixed": float(tot[fixed].mean()), "gain_nested": gain, "gain_fixed": gain_fixed, "fixed": B_SCHEMES[fixed], "choices": choice}
        rec["apply"] = bool(gain["p_positive"] > 0.5 and gain["wins"] >= MIN_WINS and gain["mean"] > 0)
        out[kind] = rec
        out["looks"] += len(B_SCHEMES)
        print(kind, "n", rec["n"], "base %.4f nested %.4f fixed %.4f" % (rec["ll_base"], rec["ll_nested"], rec["ll_fixed"]), "gain %+.4f [%+.4f,%+.4f] P %.2f wins %d" % (gain["mean"], gain["lo"], gain["hi"], gain["p_positive"], gain["wins"]), "apply", rec["apply"], flush=True)
    hsb = R.hs.to_numpy()
    shares = {}
    for nm, lo, hi in (("hs0_30", 0, 30), ("hs30_60", 30, 60), ("hs60_90", 60, 90), ("hs90_120", 90, 120), ("hs120_300", 120, 300), ("hs300_900", 300, 901)):
        m = (hsb > lo) & (hsb <= hi)
        shares[nm] = {"n": int(m.sum()), "inc_share": float(R.inc.to_numpy()[m].mean())}
    out["inc_share_real"] = shares
    FITB.parent.mkdir(parents=True, exist_ok=True)
    FITB.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"])


def a_tables():
    Rall = ug.load_rows()
    f = json.loads(FIT.read_text(encoding="utf-8"))["windows"]
    T = {}
    for qs, ks in f.items():
        q = int(qs)
        for kss, r in ks.items():
            if r["source"] != "urg2":
                continue
            k = int(kss)
            w = float(r["fixed"]["w"])
            d = build(Rall[Rall.cls == k].reset_index(drop=True), q, w)
            C = counts_of(d, r["seasons"])
            nh = d["nh"]
            a0 = float(r["fixed"]["a0"])
            CU = C.reshape(3, nh, NB).sum(1)
            Mp = (CU + 1.0) / (CU.sum(1, keepdims=True) + NB)
            P0 = (C + a0 * np.repeat(Mp, nh, axis=0)) / (C.sum(1) + a0)[:, None]
            T[(q, k)] = {"w": w, "nh": nh, "sch": r["fixed"]["scheme"], "th": np.array(r["fixed"]["theta"]), "P0": P0}
    return T


def clk_pools(ks):
    import mod25e_clk as ck

    R = ck.real_rows()
    el = R.el.to_numpy(float)
    kk = R.cls.to_numpy()
    q4all = (R.qtr.to_numpy() == 4).astype(int)
    sch, qs, table, marg, margh = {}, {}, {}, {}, {}
    for k in ks:
        sp = ug.clk_spec(k)
        sch[k] = (sp["edges"], sp["sign"], sp["to"], sp["a"])
        qs[k] = sp["qs"]
        mk = kk == k
        marg[k] = el[mk]
        hk = ck.hbin(R.hs.to_numpy()[mk], sch[k][0]) * 2 ** qs[k] + (q4all[mk] if qs[k] else 0)
        for hv in np.unique(hk):
            margh[(k, int(hv))] = el[mk][hk == hv]
        c = ck.cell(R.hs.to_numpy()[mk], R.sd.to_numpy()[mk], R.oto.to_numpy()[mk], R.dto.to_numpy()[mk], *sch[k][:3])
        if qs[k]:
            c = c * 2 + q4all[mk]
        for cv in np.unique(c):
            table[(k, int(cv))] = el[mk][c == cv]
    return sch, qs, table, marg, margh


def b_tables(fb):
    R = b_rows()
    late = R.hs.to_numpy() <= HB
    out = {}
    base = ~late
    out["c"] = float(R.inc.to_numpy()[base].mean())
    cm = R.cls.to_numpy() == 3
    Pb = np.bincount(R.yb.to_numpy()[base & cm], minlength=NY).astype(float) + 1.0
    Pb[-YLO] = 0.0
    out["Pb"] = Pb / Pb.sum()
    for kind in ("inc", "yards"):
        rec = fb[kind]
        sch = rec["fixed"]
        key = b_key(R.hs.to_numpy(), R.sd.to_numpy(), R.oto.to_numpy(), sch)
        nk = (len(B_EDGES[sch["edges"]]) + 1) * 12
        mt = late if kind == "inc" else late & cm
        if kind == "inc":
            out["inc_n"] = np.bincount(key[mt], minlength=nk).astype(float)
            out["inc_k"] = np.bincount(key[mt], weights=R.inc.to_numpy()[mt], minlength=nk)
        else:
            cnt = np.zeros((nk, NY))
            np.add.at(cnt, (key[mt], R.yb.to_numpy()[mt]), 1.0)
            out["y_cnt"] = cnt
        out[kind + "_sch"] = sch
        out[kind + "_apply"] = bool(rec["apply"])
    return out


def defer_region():
    import mod25e_cky as cy

    which = None
    if os.environ.get("CKY2") == "1":
        p = REPO / "artifacts" / "mod25e3" / "cky2" / "fit.json"
        if p.exists():
            which = p
        try:
            import mod25e_cky2 as cy2

            lo, hi = float(getattr(cy2, "SD_LO", cy.SD_LO)), float(getattr(cy2, "SD_HI", cy.SD_HI))
        except Exception:
            lo, hi = cy.SD_LO, cy.SD_HI
    elif os.environ.get("CKY") == "1":
        which = cy.FIT
        lo, hi = cy.SD_LO, cy.SD_HI
    if which is None:
        return None
    f = json.loads(Path(which).read_text(encoding="utf-8"))
    return lo, hi, float(f["Y"]), float(f["W"])


def install_urg2():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    T = a_tables()
    fa = json.loads(FIT.read_text(encoding="utf-8"))["windows"]
    clk_ks = sorted({int(k) for qs, ks in fa.items() for k, r in ks.items() if r["source"] == "clk"})
    cpool = clk_pools(clk_ks) if clk_ks else None
    clk_src = {(int(qs), int(k)) for qs, ks in fa.items() for k, r in ks.items() if r["source"] == "clk"}
    fb = json.loads(FITB.read_text(encoding="utf-8")) if FITB.exists() else None
    BT = b_tables(fb) if fb is not None and (fb["inc"]["apply"] or fb["yards"]["apply"]) else None
    region = defer_region()
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None, "rngb": None}
    cache = {}
    base = dv._G["pol"]

    def draw_b(drawn, k0, hs, yardline, score_diff, oto):
        rb = st["rngb"]
        u1, u2 = rb.random(), rb.random()
        want_inc = k0 == 0
        if BT["inc_apply"]:
            key = int(b_key([hs], [score_diff], [oto], BT["inc_sch"])[0])
            a = BT["inc_sch"]["a"]
            p = (BT["inc_k"][key] + a * BT["c"]) / (BT["inc_n"][key] + a)
            want_inc = u1 < p
        redraw_y = False
        if (not want_inc) and (k0 == 0 or BT["yards_apply"]):
            redraw_y = True
        if want_inc and k0 != 0:
            return dict(drawn, yards_gained=0.0, dist_gained=0.0, auto_first=False, repeat_down=False)
        if redraw_y:
            if BT["yards_apply"]:
                key = int(b_key([hs], [score_diff], [oto], BT["yards_sch"])[0])
                a = BT["yards_sch"]["a"]
                w = BT["y_cnt"][key] + a * BT["Pb"]
            else:
                w = BT["Pb"].copy()
            ymax = int(min(YHI, np.floor(float(yardline)) - 1)) - YLO
            w = np.array(w, float)
            w[max(ymax, 0) + 1:] = 0.0
            if w.sum() <= 0:
                return drawn
            yv = float(np.searchsorted(np.cumsum(w / w.sum()), u2, side="left") + YLO)
            return dict(drawn, yards_gained=yv, dist_gained=yv, auto_first=False, repeat_down=False)
        return drawn

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        while "offense" not in fr.f_locals:
            fr = fr.f_back
        off_to = fr.f_locals["off_to"]
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if code not in (0, 1) or qtr not in (2, 4) or hs <= LO:
            return drawn
        if region is not None and region[0] <= score_diff <= region[1] and float(yardline) <= region[2] and hs <= region[3]:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
            st["rngb"] = dv.task_rng(SALT_B, cseed)
        old = float(drawn["clock_elapsed"])
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        changed = False
        if BT is not None and code == 1 and hs <= HB and k in (0, 3):
            nd = draw_b(drawn, k, hs, yardline, score_diff, float(off_to))
            if nd is not drawn:
                drawn = nd
                changed = True
                k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        new = old
        if hs <= HI:
            t = T.get((int(qtr), k))
            if t is not None:
                u = st["rng"].random()
                u3 = int(ug.used3([float(drawn["off_to_used"])], [float(drawn["def_to_used"])])[0])
                hb_ = int(hbin([hs], t["w"])[0])
                g = int(ug.group([score_diff], [float(off_to)], [hs], t["sch"])[0])
                ck_ = (int(qtr), k, u3, hb_, g)
                cdf = cache.get(ck_)
                if cdf is None:
                    wv = t["P0"][u3 * t["nh"] + hb_] * np.exp(t["th"][g] * EV)
                    cdf = np.cumsum(wv / wv.sum())
                    cdf[-1] = 1.0
                    cache[ck_] = cdf
                new = float(min(int(np.searchsorted(cdf, u, side="left")), hs))
            elif cpool is not None and (int(qtr), k) in clk_src:
                sch, qs, table, marg, margh = cpool
                cv = int(ck.cell([hs], [score_diff], [float(off_to)], [float(fr.f_locals["def_to"])], *sch[k][:3])[0])
                q4 = int(qtr == 4) if qs[k] else 0
                if qs[k]:
                    cv = cv * 2 + q4
                a = sch[k][3]
                pool = table.get((k, cv))
                nn = 0 if pool is None else len(pool)
                u, v = st["rng"].random(), st["rng"].random()
                hv = int(ck.hbin([hs], sch[k][0])[0]) * 2 ** qs[k] + q4
                src = pool if (nn and u < nn / (nn + a)) else margh.get((k, hv), marg[k])
                new = float(min(src[int(v * len(src))], hs))
        if new == old and not changed:
            return drawn
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


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
    elif len(sys.argv) > 1 and sys.argv[1] == "fitb":
        fit_b()


if __name__ == "__main__":
    main()
