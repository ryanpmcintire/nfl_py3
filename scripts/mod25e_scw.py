import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "scw" / "fit.json"
SCRATCH = REPO / "tests" / "scratch" / "e112s"
SEASONS = tuple(range(2009, 2018))
OUTER = (2.0, 4.0, 8.0, 16.0)
KS = (100, 200, 400, 800, 1600)
CUR = (8.0, 200)
ALPHA = 1.0
LAM = 1_000_000.0
YBINS = (-1, 0, 3, 6, 10, 19)
NBOOT = 1000
GROUPS = (("lead17p", 17.0, 99.0), ("lead9_16", 9.0, 16.9), ("trail9_16", -16.9, -9.0), ("trail17p", -99.0, -17.0))


def enabled():
    return os.environ.get("SCW") == "1"


def to_scaled(d, outer, inner, inner_scale):
    a = np.abs(d)
    return np.sign(d) * (np.minimum(a, inner) / inner_scale + np.maximum(a - inner, 0.0) / outer)


def from_scaled(m, inner, inner_scale, outer):
    a = np.abs(m)
    lim = inner / inner_scale
    return np.sign(m) * np.where(a <= lim, a * inner_scale, inner + (a - lim) * outer)


def outcome_class(tr):
    y = np.digitize(tr["yards_gained"].to_numpy(), YBINS, right=True)
    td = tr["points_off"].to_numpy() >= 6
    to = tr["possession_flip"].to_numpy() & ~td
    et = np.where(td, 1, np.where(to, 2, 0))
    ptc = np.clip(tr["play_type_code"].to_numpy().astype(int), 0, 6)
    nb = len(YBINS) + 1
    return (ptc * nb + y) * 3 + et, td.astype(float), to.astype(float)


def feats(eng, tr, outer):
    F = eng.feature_matrix(tr["dist_raw"].to_numpy(), tr["fp_raw"].to_numpy(), tr["sc_raw"].to_numpy(), tr["time_raw"].to_numpy(), tr["off_to_raw"].to_numpy(), tr["def_to_raw"].to_numpy(), tr["phase"].to_numpy())
    F[:, 2] = to_scaled(np.clip(tr["sc_raw"].to_numpy(), -eng.SCORE_CLIP, eng.SCORE_CLIP), outer, eng.SCORE_INNER, eng.SCORE_INNER_SCALE)
    return F


def run_phase(eng, tr, cls, td, to, h, phase, kmax):
    from scipy.spatial import cKDTree

    ncls = int(cls.max()) + 1
    ph = tr["phase"].to_numpy()
    dn = tr["down_i"].to_numpy()
    sc = tr["sc_raw"].to_numpy()
    season = tr["season"].to_numpy()
    orow, drow, hom = tr["off_row"].to_numpy(), tr["def_row"].to_numpy(), tr["is_home_off"].to_numpy()
    cfgs = [(s, k) for s in OUTER for k in KS]
    qmask = (ph == phase) & np.isin(dn, (1, 2, 3)) & (np.abs(sc) > eng.SCORE_INNER)
    qidx = np.flatnonzero(qmask)
    nq = len(qidx)
    pos = {int(r): i for i, r in enumerate(qidx)}
    LL = np.zeros((nq, len(cfgs)), dtype=np.float32)
    PTD = np.zeros((nq, len(cfgs)), dtype=np.float32)
    PTO = np.zeros((nq, len(cfgs)), dtype=np.float32)
    Fcache = {s: feats(eng, tr, s) for s in OUTER}
    for f in SEASONS:
        for d in (1, 2, 3):
            tmask = (season != f) & (dn == d) & eng.phase_pool_mask(ph, phase)
            pidx = np.flatnonzero(tmask)
            qi = qidx[(season[qidx] == f) & (dn[qidx] == d)]
            if len(qi) == 0:
                continue
            g = np.bincount(cls[pidx], minlength=ncls) + 0.5
            g = g / g.sum()
            kk = min(kmax, len(pidx))
            qrows = np.array([pos[int(r)] for r in qi])
            for si, s in enumerate(OUTER):
                F = Fcache[s]
                tree = cKDTree(F[pidx])
                _, ind = tree.query(F[qi], k=kk, workers=1)
                nbr = pidx[ind]
                d2 = (orow[nbr] - orow[qi][:, None]) ** 2 + (drow[nbr] - drow[qi][:, None]) ** 2
                w0 = np.exp(-d2 / (2.0 * h * h)) * np.where(hom[nbr] == hom[qi][:, None], LAM, 1.0)
                for ki, k in enumerate(KS):
                    w = w0[:, :k]
                    sm = w.sum(1, keepdims=True)
                    bad = ~np.isfinite(sm[:, 0]) | (sm[:, 0] <= 0)
                    w = np.where(bad[:, None], 1.0, w)
                    sm = w.sum(1, keepdims=True)
                    w = w / sm
                    ess = 1.0 / (w**2).sum(1)
                    nb = nbr[:, :k]
                    c = (w * (cls[nb] == cls[qi][:, None])).sum(1)
                    p = (ess * c + ALPHA * g[cls[qi]]) / (ess + ALPHA)
                    j = si * len(KS) + ki
                    LL[qrows, j] = np.log(p)
                    PTD[qrows, j] = (w * td[nb]).sum(1)
                    PTO[qrows, j] = (w * to[nb]).sum(1)
    return qidx, cfgs, LL, PTD, PTO


def boot(diff, games, seed=7):
    u, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnt = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(NBOOT, len(u)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(diff.mean()), float(np.percentile(m, 5)), float(np.percentile(m, 95)), float((m > 0).mean())


def analyse(tr, qidx, cfgs, LL, PTD, PTO, td, to, tag):
    season = tr["season"].to_numpy()[qidx]
    games = tr["game_id"].to_numpy()[qidx]
    sc = tr["sc_raw"].to_numpy()[qidx]
    ci = cfgs.index(CUR)
    n = len(qidx)
    mean_ll = LL.mean(0)
    L = np.array([[LL[season == f, j].sum() for j in range(len(cfgs))] for f in SEASONS])
    N = np.array([(season == f).sum() for f in SEASONS])
    out = {"tag": tag, "n_queries": int(n), "cfgs": [list(c) for c in cfgs], "mean_ll_all": mean_ll.tolist(), "cur_index": ci}
    chosen_f, oos = [], np.zeros(n)
    for fi, f in enumerate(SEASONS):
        inner = L[[i for i in range(len(SEASONS)) if i != fi]].sum(0) / N[[i for i in range(len(SEASONS)) if i != fi]].sum()
        j = int(np.argmax(inner))
        chosen_f.append(j)
        oos[season == f] = LL[season == f, j]
    cur = LL[:, ci]
    out["chosen_per_fold"] = {str(f): list(cfgs[j]) for f, j in zip(SEASONS, chosen_f)}
    out["oos_ll_per_play"] = float(oos.mean())
    out["cur_ll_per_play"] = float(cur.mean())
    out["oos_gain"] = boot(oos - cur, games)
    out["fold_gain"] = {str(f): float((oos[season == f] - cur[season == f]).mean()) for f in SEASONS}
    out["fold_wins"] = int(sum(1 for f in SEASONS if (oos[season == f] - cur[season == f]).mean() > 0))
    b = int(np.argmax(mean_ll))
    out["insample_best"] = {"cfg": list(cfgs[b]), "ll": float(mean_ll[b]), "gain_vs_cur": boot(LL[:, b] - cur, games), "fold_wins_vs_cur": int(sum(1 for f in SEASONS if (LL[season == f, b] - cur[season == f]).mean() > 0))}
    out["insample_minus_oos_gap"] = float(mean_ll[b] - oos.mean())
    out["fixed_best_fold_gain"] = {str(f): float((LL[season == f, b] - cur[season == f]).mean()) for f in SEASONS}
    best_s = {"%s,%s" % (c[0], c[1]): float(mean_ll[j]) for j, c in enumerate(cfgs)}
    out["grid_mean_ll"] = best_s
    for name, lo, hi in GROUPS:
        m = (sc >= lo) & (sc <= hi)
        if m.sum() == 0:
            continue
        tdq = td[qidx][m].mean()
        toq = to[qidx][m].mean()
        out.setdefault("groups", {})[name] = {"n": int(m.sum()), "real_td_play": float(tdq), "real_to_play": float(toq), "pred_td_cur": float(PTD[m, ci].mean()), "pred_td_best": float(PTD[m, b].mean()), "pred_to_cur": float(PTO[m, ci].mean()), "pred_to_best": float(PTO[m, b].mean()), "ll_gain_best_vs_cur": boot(LL[m, b] - cur[m], games[m])}
        oosg = oos[m] - cur[m]
        out["groups"][name]["ll_gain_oos_vs_cur"] = boot(oosg, games[m])
    lead = sc >= 17
    if lead.any():
        out["lead17p_td_ratio_best_over_cur"] = float(PTD[lead, b].mean() / PTD[lead, ci].mean())
    return out


FAMILIES = (("joint", lambda c: True), ("outer_only_k200", lambda c: c[1] == CUR[1]), ("k_only_outer8", lambda c: c[0] == CUR[0]))


def finish(tr, td, to, store):
    res = {"looks_per_phase": len(OUTER) * len(KS), "grid_outer": list(OUTER), "grid_k": list(KS), "current": list(CUR), "phases": {}}
    for phase, (qidx, cfgs, LL, PTD, PTO) in store.items():
        res["phases"][str(phase)] = {}
        for name, keep in FAMILIES:
            cols = [j for j, c in enumerate(cfgs) if keep(c)]
            sub = [cfgs[j] for j in cols]
            res["phases"][str(phase)][name] = analyse(tr, qidx, sub, LL[:, cols], PTD[:, cols], PTO[:, cols], td, to, "phase%d_%s" % (phase, name))
    ch = {}
    for p, fam in res["phases"].items():
        pick = {"outer": CUR[0], "k": CUR[1]}
        for name, idx in (("outer_only_k200", 0), ("k_only_outer8", 1)):
            r = fam[name]
            g = r["insample_best"]
            interior = idx == 0 or int(g["cfg"][1]) < max(KS)
            if interior and tuple(g["cfg"]) != CUR and r["oos_gain"][0] > 0 and r["fold_wins"] > len(SEASONS) // 2:
                pick["outer" if idx == 0 else "k"] = g["cfg"][idx] if idx == 0 else int(g["cfg"][idx])
        if (pick["outer"], pick["k"]) != CUR:
            ch[p] = pick
    res["chosen"] = ch
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(res), encoding="utf-8")
    for p, fam in res["phases"].items():
        for name, r in fam.items():
            print("phase", p, name, "n", r["n_queries"], "cur ll", round(r["cur_ll_per_play"], 4), "oos ll", round(r["oos_ll_per_play"], 4), "gain", [round(x, 5) for x in r["oos_gain"]], "wins", r["fold_wins"], "/9")
            print(" chosen per fold", r["chosen_per_fold"])
            print(" insample best", r["insample_best"]["cfg"], "gap", round(r["insample_minus_oos_gap"], 5))
            if name in ("outer_only_k200", "k_only_outer8"):
                for gname, gv in r.get("groups", {}).items():
                    print(" ", gname, "n", gv["n"], "real td/play %.4f pred cur %.4f best %.4f | to real %.4f cur %.4f best %.4f | oos ll gain %s" % (gv["real_td_play"], gv["pred_td_cur"], gv["pred_td_best"], gv["real_to_play"], gv["pred_to_cur"], gv["pred_to_best"], [round(x, 4) for x in gv["ll_gain_oos_vs_cur"]]))
                print(" lead17 td ratio best/cur", r.get("lead17p_td_ratio_best_over_cur"))
    print("chosen", ch)


def prep():
    import sim04_engine as eng
    import mod25e_kstate as ks

    tr = ks.load_trans()
    ok = tr["down_i"].isin([1, 2, 3]) & tr["season"].isin(SEASONS)
    tr = tr[ok].reset_index(drop=True)
    cls, td, to = outcome_class(tr)
    h = eng.TEAM_KERNEL_H_SCALE * float(np.nanstd(np.concatenate([tr["off_row"].to_numpy(), tr["off_row"].to_numpy()])))
    return eng, tr, cls, td, to, h


def fit():
    eng, tr, cls, td, to, h = prep()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    store = {}
    for phase in (2, 3):
        store[phase] = run_phase(eng, tr, cls, td, to, h, phase, max(KS))
        qidx, cfgs, LL, PTD, PTO = store[phase]
        np.savez_compressed(SCRATCH / ("scw_phase%d.npz" % phase), qidx=qidx, LL=LL, PTD=PTD, PTO=PTO)
    finish(tr, td, to, store)


def reanalyse():
    eng, tr, cls, td, to, h = prep()
    cfgs = [(s, k) for s in OUTER for k in KS]
    store = {}
    for phase in (2, 3):
        z = np.load(SCRATCH / ("scw_phase%d.npz" % phase))
        store[phase] = (z["qidx"], cfgs, z["LL"], z["PTD"], z["PTO"])
    finish(tr, td, to, store)


def install_scw():
    import mod25d_variance as dv
    from scipy.spatial import cKDTree

    spec = json.loads(FIT.read_text(encoding="utf-8")).get("chosen", {})
    eng = dv.sim
    t = dv._G["tables"]
    inner, inner_scale, cur_outer = eng.SCORE_INNER, eng.SCORE_INNER_SCALE, eng.SCORE_OUTER_SCALE
    lim = inner / inner_scale
    pend = {}
    for key, (tree, sub) in t["nn_trees_cond"].items():
        c = spec.get(str(key[1]))
        if c is None or isinstance(tree, ScoreTree):
            continue
        if not isinstance(tree, cKDTree):
            raise RuntimeError("SCW needs the plain neighbour trees; run with DKF and DK2 off")
        pend[key] = (ScoreTree(tree, sub, float(c["outer"]), int(c["k"]), inner, inner_scale, cur_outer, lim), sub)
    t["nn_trees_cond"].update(pend)
    t["nn_cache_cond"].clear()
    t["nn_weight_cache_cond"].clear()
    return len(pend)


class ScoreTree:
    def __init__(self, tree, sub, outer, k, inner, inner_scale, cur_outer, lim):
        from scipy.spatial import cKDTree

        self.base = tree
        self.k = min(int(k), len(sub))
        self.inner, self.inner_scale, self.cur_outer, self.lim = inner, inner_scale, cur_outer, lim
        self.outer = outer
        X = np.array(tree.data, copy=True)
        d = from_scaled(X[:, 2], inner, inner_scale, cur_outer)
        X[:, 2] = to_scaled(d, outer, inner, inner_scale)
        self.tree = cKDTree(X)

    def query(self, x, k=1, **kw):
        q = np.array(x, dtype=float, copy=True)
        one = q.ndim == 1
        q2 = np.atleast_2d(q)
        wide = np.abs(q2[:, 2]) > self.lim + 1e-9
        if one:
            if wide[0]:
                d = from_scaled(q2[0, 2], self.inner, self.inner_scale, self.cur_outer)
                q2[0, 2] = to_scaled(d, self.outer, self.inner, self.inner_scale)
                return self.tree.query(q2[0], k=self.k)
            return self.base.query(q, k=k)
        dd = np.zeros((len(q2), k))
        ii = np.zeros((len(q2), k), dtype=int)
        if wide.any():
            d = from_scaled(q2[wide, 2], self.inner, self.inner_scale, self.cur_outer)
            qw = q2[wide].copy()
            qw[:, 2] = to_scaled(d, self.outer, self.inner, self.inner_scale)
            a, b = self.tree.query(qw, k=k)
            dd[wide], ii[wide] = a, b
        if (~wide).any():
            a, b = self.base.query(q2[~wide], k=k)
            dd[~wide], ii[~wide] = a, b
        return dd, ii

    @property
    def data(self):
        return self.base.data


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
    if len(sys.argv) > 1 and sys.argv[1] == "reanalyse":
        reanalyse()
