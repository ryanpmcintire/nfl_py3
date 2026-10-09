import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "urg" / "fit.json"
SALT = 4323
LO = 45.0
HI = 180.0
HSPLIT = 120.0
HW = 15.0
NH = 9
NB = 62
WINDOWS = (4, 2)
CLASSES = (0, 1, 2, 3)
CLASS_NAMES = {0: "inc", 1: "term", 2: "run", 3: "pass"}
A0S = (8.0, 32.0, 128.0, 512.0, 2048.0)
BANDS = ("sign3", "trail5", "full9")
NBAND = {"sign3": 3, "trail5": 6, "full9": 9}
LAMS = (1.0, 10000.0)
MIN_WINS = 5
NBOOT = 400
ITERS = 10
STEP = 0.1
EV = np.arange(NB, dtype=float)
SCHEMES = [{"band": b, "to": t, "h": h, "lam": l} for b, t, h, l in itertools.product(BANDS, (0, 1), (0, 1), LAMS)]
REPORT_BANDS = (("trail17+", -1e9, -17), ("trail9-16", -16, -9), ("trail4-8", -8, -4), ("trail1-3", -3, -1), ("tie", 0, 0), ("lead1-8", 1, 8), ("lead9+", 9, 1e9))


def enabled():
    return os.environ.get("URG") == "1"


def used3(otu, dtu):
    return np.where(np.asarray(otu) > 0, 1, np.where(np.asarray(dtu) > 0, 2, 0)).astype(int)


def hbin(hs):
    return np.clip(np.floor((np.asarray(hs, float) - LO - 1e-9) / HW).astype(int), 0, NH - 1)


def band(sd, kind):
    sd = np.asarray(sd, float)
    if kind == "sign3":
        return (np.sign(sd) + 1).astype(int)
    if kind == "trail5":
        return np.select([sd <= -17, sd <= -9, sd <= -4, sd < 0, sd == 0], [0, 1, 2, 3, 4], 5)
    return np.select([sd <= -17, sd <= -9, sd <= -4, sd < 0, sd == 0, sd <= 3, sd <= 8, sd <= 16], [0, 1, 2, 3, 4, 5, 6, 7], 8)


def group(sd, oto, hs, sch):
    b = band(sd, sch["band"])
    t = (np.asarray(oto) > 0).astype(int) if sch["to"] else np.zeros(len(b), int)
    h = (np.asarray(hs) > HSPLIT).astype(int) if sch["h"] else np.zeros(len(b), int)
    return (b * 2 + t) * 2 + h


def ngroups(sch):
    return NBAND[sch["band"]] * 4


def skey(sch):
    return "%s|to%d|h%d|l%g" % (sch["band"], sch["to"], sch["h"], sch["lam"])


def parent_rows(C, a0, cell, used, e, loo):
    CU = C.reshape(3, NH, NB).sum(1)
    Mp = (CU + 1.0) / (CU.sum(1, keepdims=True) + NB)
    num = C[cell] + a0 * Mp[used]
    n = C[cell].sum(1)
    if loo:
        num[np.arange(len(cell)), e] -= 1.0
        n = n - 1.0
    return num / (n + a0)[:, None]


def fit_theta(P, e, g, G, lam):
    th = np.zeros(G)
    ef = e.astype(float)
    for _ in range(ITERS):
        W = P * np.exp(th[g][:, None] * EV[None, :])
        Z = W.sum(1)
        m1 = (W * EV).sum(1) / Z
        m2 = (W * EV * EV).sum(1) / Z
        gr = np.bincount(g, weights=ef - m1, minlength=G) - lam * th
        he = np.bincount(g, weights=m2 - m1 * m1, minlength=G) + lam
        th = th + np.clip(gr / he, -STEP, STEP)
    return th


def tilt(P, th_rows):
    W = P * np.exp(th_rows[:, None] * EV[None, :])
    return W / W.sum(1, keepdims=True)


def load_rows():
    import mod25e_clk as ck

    return ck.real_rows()


def build(Rk, q):
    W = ((Rk.qtr == q) & (Rk.hs > LO) & (Rk.hs <= HI) & Rk.code.isin([0, 1])).to_numpy()
    D = Rk[W].reset_index(drop=True)
    d = {"df": D, "season": D.season.to_numpy(), "e": np.minimum(np.round(D.el.to_numpy(float)).astype(int), NB - 1), "used": used3(D.otu, D.dtu), "hs": D.hs.to_numpy(float), "sd": D.sd.to_numpy(float), "oto": D.oto.to_numpy(float), "gid": D.g.to_numpy()}
    d["cell"] = d["used"] * NH + hbin(d["hs"])
    d["CT"] = {}
    for s in np.unique(d["season"]):
        m = d["season"] == s
        T = np.zeros((3 * NH, NB))
        np.add.at(T, (d["cell"][m], d["e"][m]), 1.0)
        d["CT"][int(s)] = T
    d["G"] = {skey(sch): group(d["sd"], d["oto"], d["hs"], sch) for sch in SCHEMES}
    return d


def counts_of(d, seasons):
    return sum(d["CT"][int(s)] for s in seasons)


def sub_parent(d, C, a0, idx, loo):
    return parent_rows(C, a0, d["cell"][idx], d["used"][idx], d["e"][idx], loo)


def score_scheme(d, tr, te, C, a0, sch):
    key = skey(sch)
    P_tr = sub_parent(d, C, a0, tr, True)
    th = fit_theta(P_tr, d["e"][tr], d["G"][key][tr], ngroups(sch), sch["lam"])
    P0 = sub_parent(d, C, a0, te, False)
    P = tilt(P0, th[d["G"][key][te]])
    e = d["e"][te]
    ll = np.log(P[np.arange(len(e)), e])
    mean = (P * np.minimum(EV[None, :], d["hs"][te][:, None])).sum(1)
    return ll, mean, th


def free_scheme(d, te, C, a0):
    P = sub_parent(d, C, a0, te, False)
    e = d["e"][te]
    return np.log(P[np.arange(len(e)), e]), (P * np.minimum(EV[None, :], d["hs"][te][:, None])).sum(1)


def choose_a0(d, seasons):
    C_all = counts_of(d, seasons)
    best, bv = None, -1e18
    for a0 in A0S:
        tot = 0.0
        for t in seasons:
            idx = np.where(d["season"] == t)[0]
            tot += free_scheme(d, idx, C_all - d["CT"][int(t)], a0)[0].sum()
        if tot > bv:
            best, bv = a0, tot
    return best


def clk_spec(k):
    import mod25e_clk as ck

    s = json.loads(ck.FIT.read_text(encoding="utf-8"))["classes"][ck.CLASSES[k]]["best"]
    f2 = json.loads(ck.FIT2.read_text(encoding="utf-8"))["classes"][ck.CLASSES[k]]["best"]
    return {"edges": ck.EDGES[s["edges"]], "sign": s["sign"], "to": s["to"], "a": float(f2["a"]), "qs": int(f2["qs"])}


def clk_predict(Rall, d, k, q):
    import mod25e_clk as ck

    Rk = Rall[Rall.cls == k].reset_index(drop=True)
    sp = clk_spec(k)
    b = ck.ebin(Rk.el.to_numpy())
    c = ck.cell(Rk.hs.to_numpy(), Rk.sd.to_numpy(), Rk.oto.to_numpy(), Rk.dto.to_numpy(), sp["edges"], sp["sign"], sp["to"])
    hb = ck.hbin(Rk.hs.to_numpy(), sp["edges"])
    if sp["qs"]:
        q4 = (Rk.qtr.to_numpy() == 4).astype(int)
        c, hb = c * 2 + q4, hb * 2 + q4
    se = Rk.season.to_numpy()
    W = ((Rk.qtr == q) & (Rk.hs > LO) & (Rk.hs <= HI) & Rk.code.isin([0, 1])).to_numpy()
    ll = np.zeros(len(d["e"]))
    mean = np.zeros(len(d["e"]))
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
        P = (cc + sp["a"] * margh[hb[te]]) / (nh + sp["a"])[:, None]
        m = d["season"] == s
        eb = d["e"][m]
        ll[m] = np.log(P[np.arange(len(eb)), eb])
        mean[m] = (P * np.minimum(EV[None, :], d["hs"][m][:, None])).sum(1)
    return ll, mean


def cdw_predict(d, k, q):
    import mod25e_cdw as cw

    b = json.loads(cw.FIT.read_text(encoding="utf-8"))["chosen"]
    spec, a = b["spec"], float(b["a"])
    R = cw.rows()
    ll = np.zeros(len(d["e"]))
    mean = np.zeros(len(d["e"]))
    for s in np.unique(d["season"]):
        tr = (R.season != s) & (R.hs <= spec["hz"])
        te = (R.season == s) & (R.qtr == q) & (R.hs <= HI) & (R.cls == k)
        Rte = R[te]
        m = d["season"] == s
        if len(Rte) != int(m.sum()):
            raise RuntimeError("cdw row alignment")
        P = cw.predict(R[tr], Rte, spec, a)
        eb = d["e"][m]
        ll[m] = np.log(P[np.arange(len(eb)), eb])
        mean[m] = (P * np.minimum(EV[None, :], d["hs"][m][:, None])).sum(1)
    return ll, mean


def boot(diff, gid, rng):
    u, inv = np.unique(gid, return_inverse=True)
    sg = np.bincount(inv, weights=diff, minlength=len(u))
    ng = np.bincount(inv, minlength=len(u)).astype(float)
    reps = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(u), len(u))
        reps.append(sg[i].sum() / ng[i].sum())
    reps = np.array(reps)
    return {"mean": float(diff.mean()), "lo": float(np.percentile(reps, 5)), "hi": float(np.percentile(reps, 95)), "p_positive": float((reps > 0).mean())}


def season_wins(diff, season):
    r = {int(s): float(diff[season == s].mean()) for s in np.unique(season)}
    return sum(1 for v in r.values() if v > 0), r


def bias_table(d, preds, actual):
    rows = {}
    sd = d["sd"]
    hs = d["hs"]
    for name, lo, hi in REPORT_BANDS:
        m = (sd >= lo) & (sd <= hi)
        if m.sum() < 5:
            continue
        rows[name] = {"n": int(m.sum()), "actual": float(actual[m].mean()), **{p: float(v[m].mean()) for p, v in preds.items()}}
    m = (sd >= -8) & (sd <= 0) & (hs > 60) & (hs <= 120)
    if m.sum() >= 5:
        rows["contested_60_120"] = {"n": int(m.sum()), "actual": float(actual[m].mean()), **{p: float(v[m].mean()) for p, v in preds.items()}}
    return rows


def run_class(Rall, q, k, rng):
    Rk = Rall[Rall.cls == k].reset_index(drop=True)
    d = build(Rk, q)
    n = len(d["e"])
    seasons = sorted(int(s) for s in np.unique(d["season"]))
    if n < 200 or len(seasons) < 9:
        return None
    llm = {skey(s): np.zeros(n) for s in SCHEMES}
    mnm = {skey(s): np.zeros(n) for s in SCHEMES}
    ll_n, mn_n, ll_f, mn_f = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    choices = {}
    for s in seasons:
        trs = [t for t in seasons if t != s]
        a0 = choose_a0(d, trs)
        C_tr = counts_of(d, trs)
        te = np.where(d["season"] == s)[0]
        tr = np.where(d["season"] != s)[0]
        ll_f[te], mn_f[te] = free_scheme(d, te, C_tr, a0)
        inner = {}
        for sch in SCHEMES:
            key = skey(sch)
            llm[key][te], mnm[key][te], _ = score_scheme(d, tr, te, C_tr, a0, sch)
            tot = 0.0
            for t in trs:
                itr = np.where((d["season"] != s) & (d["season"] != t))[0]
                ite = np.where(d["season"] == t)[0]
                tot += score_scheme(d, itr, ite, C_tr - d["CT"][int(t)], a0, sch)[0].sum()
            inner[key] = tot
        pick = max(inner, key=inner.get)
        ll_n[te], mn_n[te] = llm[pick][te], mnm[pick][te]
        choices[s] = {"a0": a0, "scheme": pick}
        print(q, CLASS_NAMES[k], s, a0, pick, flush=True)
    tot = {key: float(v.sum()) for key, v in llm.items()}
    fixed = max(tot, key=tot.get)
    fsch = next(sch for sch in SCHEMES if skey(sch) == fixed)
    a0f = choose_a0(d, seasons)
    C_all = counts_of(d, seasons)
    P_all = sub_parent(d, C_all, a0f, np.arange(n), True)
    th = fit_theta(P_all, d["e"], d["G"][fixed], ngroups(fsch), fsch["lam"])
    ll_c, mn_c = clk_predict(Rall, d, k, q)
    ll_w, mn_w = cdw_predict(d, k, q)
    actual = np.minimum(d["df"].el.to_numpy(float), d["hs"])
    out = {"n": n, "seasons": seasons, "looks": len(SCHEMES), "choices": {str(s): v for s, v in choices.items()}, "fixed": {"scheme": fsch, "a0": a0f, "theta": [float(x) for x in th]}}
    out["ll"] = {"clk": float(ll_c.mean()), "cdw": float(ll_w.mean()), "free": float(ll_f.mean()), "nested": float(ll_n.mean()), "fixed_loso": float(llm[fixed].mean())}
    out["mean_ll_in_sample_gap"] = float(llm[fixed].mean() - ll_n.mean())
    for nm, base in (("clk", ll_c), ("cdw", ll_w), ("free", ll_f)):
        for tag, arr in (("nested", ll_n), ("fixed", llm[fixed])):
            diff = arr - base
            w, per = season_wins(diff, d["season"])
            out.setdefault("gain_" + tag, {})[nm] = {**boot(diff, d["gid"], rng), "wins": w, "per_season": per}
    preds = {"clk": mn_c, "cdw": mn_w, "urg_nested": mn_n, "urg_free": mn_f}
    out["bias"] = bias_table(d, preds, actual)
    out["by_scheme_ll"] = {key: float(v.mean()) for key, v in llm.items()}
    g = out["gain_nested"]["clk"]
    out["apply"] = bool(g["p_positive"] > 0.5 and g["wins"] >= MIN_WINS and g["mean"] > 0)
    return out


def shrink_diag(Rall):
    import mod25e_clk as ck

    out = {}
    for k in (0, 2, 3):
        sp = clk_spec(k)
        Rk = Rall[Rall.cls == k]
        c = ck.cell(Rk.hs.to_numpy(), Rk.sd.to_numpy(), Rk.oto.to_numpy(), Rk.dto.to_numpy(), sp["edges"], sp["sign"], sp["to"])
        if sp["qs"]:
            c = c * 2 + (Rk.qtr.to_numpy() == 4).astype(int)
        uc, inv, cn = np.unique(c, return_inverse=True, return_counts=True)
        W = ((Rk.qtr == 4) & (Rk.hs > LO) & (Rk.hs <= HI)).to_numpy()
        n = cn[inv][W]
        wgt = n / (n + sp["a"])
        out[CLASS_NAMES[k]] = {"a": sp["a"], "qs": sp["qs"], "sign": sp["sign"], "to": sp["to"], "cell_n_median": float(np.median(n)), "cell_weight_median": float(np.median(wgt)), "cell_weight_mean": float(wgt.mean())}
    return out


def fit():
    Rall = load_rows()
    rng = np.random.default_rng(SALT)
    out = {"looks": 0, "windows": {}, "shrink_clk": shrink_diag(Rall)}
    print(json.dumps(out["shrink_clk"]), flush=True)
    for q in WINDOWS:
        out["windows"][str(q)] = {}
        for k in CLASSES:
            r = run_class(Rall, q, k, rng)
            if r is None:
                continue
            out["looks"] += r["looks"]
            out["windows"][str(q)][str(k)] = r
            g = r["gain_nested"]
            print("Q%d %s n=%d ll clk %.4f cdw %.4f free %.4f nested %.4f | vs clk %+.4f [%+.4f,%+.4f] P %.2f wins %d | vs cdw %+.4f P %.2f wins %d | apply %s" % (q, CLASS_NAMES[k], r["n"], r["ll"]["clk"], r["ll"]["cdw"], r["ll"]["free"], r["ll"]["nested"], g["clk"]["mean"], g["clk"]["lo"], g["clk"]["hi"], g["clk"]["p_positive"], g["clk"]["wins"], g["cdw"]["mean"], g["cdw"]["p_positive"], g["cdw"]["wins"], r["apply"]), flush=True)
            for nm, row in r["bias"].items():
                print("   ", nm, json.dumps({a: round(b, 2) if isinstance(b, float) else b for a, b in row.items()}), flush=True)
            FIT.parent.mkdir(parents=True, exist_ok=True)
            FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"])


def tables():
    Rall = load_rows()
    f = json.loads(FIT.read_text(encoding="utf-8"))["windows"]
    T = {}
    for qs, ks in f.items():
        q = int(qs)
        for kss, r in ks.items():
            if not r["apply"]:
                continue
            k = int(kss)
            d = build(Rall[Rall.cls == k].reset_index(drop=True), q)
            C = counts_of(d, r["seasons"])
            a0 = float(r["fixed"]["a0"])
            CU = C.reshape(3, NH, NB).sum(1)
            Mp = (CU + 1.0) / (CU.sum(1, keepdims=True) + NB)
            P0 = (C + a0 * np.repeat(Mp, NH, axis=0)) / (C.sum(1) + a0)[:, None]
            T[(q, k)] = {"sch": r["fixed"]["scheme"], "th": np.array(r["fixed"]["theta"]), "P0": P0}
    return T


def install_urg():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    T = tables()
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    cache = {}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        while "offense" not in fr.f_locals:
            fr = fr.f_back
        off_to = fr.f_locals["off_to"]
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if code not in (0, 1) or qtr not in (2, 4) or hs <= LO or hs > HI:
            return drawn
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        t = T.get((int(qtr), k))
        if t is None:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        u = st["rng"].random()
        u3 = int(used3([float(drawn["off_to_used"])], [float(drawn["def_to_used"])])[0])
        cell = u3 * NH + int(hbin([hs])[0])
        g = int(group([score_diff], [float(off_to)], [hs], t["sch"])[0])
        ck_ = (int(qtr), k, cell, g)
        cdf = cache.get(ck_)
        if cdf is None:
            w = t["P0"][cell] * np.exp(t["th"][g] * EV)
            cdf = np.cumsum(w / w.sum())
            cdf[-1] = 1.0
            cache[ck_] = cdf
        new = float(min(int(np.searchsorted(cdf, u, side="left")), hs))
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
