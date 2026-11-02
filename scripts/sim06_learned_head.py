import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "sim06"
os.environ.setdefault("SIM10_SEASONS", "2018-2025")
os.environ.setdefault("SIM10_OUT", str(ART / "sim10"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import pandas as pd

import sim10_drive as s10

warnings.filterwarnings("ignore")
TRAIN = tuple(range(2018, 2024))
TEST = (2024, 2025)
ALL = TRAIN + TEST
QCHUNK = 400
NBOOT = 4000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def class_codes(seg, po, pd_, train_mask):
    cp = np.select([pd_ > 0, po == 0, po == 3, po == 6, po == 7, po == 8], [5, 0, 1, 2, 3, 4], 6)
    nyl = seg["nyl"].to_numpy()
    open_ = (cp == 0) & (seg["he"].to_numpy() == 0)
    t1, t2 = np.quantile(nyl[open_ & train_mask], [1 / 3, 2 / 3])
    cp = np.where(open_, 7 + (nyl > t1) + (nyl > t2), cp)
    el = seg["el"].to_numpy()
    qs = np.quantile(el[train_mask], np.linspace(0.1, 0.9, 9))
    return cp * 10 + np.searchsorted(qs, el)


def relabel(code, fit_mask):
    u = np.unique(code[fit_mask])
    idx = np.searchsorted(u, code)
    hit = (idx < len(u)) & (u[np.minimum(idx, len(u) - 1)] == code)
    return np.where(hit, idx, -1), u


def kernel_dist(Xq, lgq, Xp, clsp, hep, K, h, alpha, g, gc):
    oh = np.zeros((len(Xp), K))
    oh[np.arange(len(Xp)), clsp] = 1.0
    pri = (oh.sum(axis=0) + gc / gc.sum() * 0.5 * K) / (len(Xp) + 0.5 * K)
    dead = hep == 1
    dims = [(d, hd) for d, hd in zip(s10.FIT_DIMS, h, strict=True) if np.isfinite(hd)]
    for a in range(0, len(Xq), QCHUNK):
        b = min(a + QCHUNK, len(Xq))
        z = np.zeros((b - a, len(Xp)))
        for d, hd in dims:
            hq = hd * np.exp(g * (lgq[a:b] - s10.L0))[:, None] if d == 1 else hd
            z += ((Xq[a:b, d][:, None] - Xp[:, d][None, :]) / hq) ** 2
        W = np.exp(-0.5 * z)
        W *= ~(dead[None, :] & (Xp[:, 2][None, :] < Xq[a:b, 2][:, None]))
        yield a, b, (W @ oh + alpha * pri) / (W.sum(axis=1) + alpha)[:, None]


def kernel_fold(X, cls, half, home, he, lg, q_mask, p_mask, K, h, alpha, g, floor, full=False):
    gc = np.bincount(cls[p_mask & (cls >= 0)], minlength=K) + 0.5
    qall = np.flatnonzero(q_mask)
    out = np.zeros((len(qall), K)) if full else np.full(len(qall), np.nan)
    for hf in (1, 2):
        for hm in (0, 1):
            st = (half == hf) & (home == hm)
            qi = np.flatnonzero(st & q_mask)
            pi = np.flatnonzero(st & p_mask & (cls >= 0))
            if len(qi) == 0:
                continue
            pos = np.searchsorted(qall, qi)
            for a, b, D in kernel_dist(X[qi], lg[qi], X[pi], cls[pi], he[pi], K, h, alpha, g, gc):
                if full:
                    out[pos[a:b]] = D
                else:
                    c = cls[qi[a:b]]
                    out[pos[a:b]] = np.where(c >= 0, D[np.arange(b - a), np.maximum(c, 0)], floor)
    return out


def head_features(seg):
    return np.c_[
        seg["yl"], seg["sd"], np.log1p(seg["secs"].clip(lower=0.0)), seg["too"], seg["tod"],
        seg["week"], seg["half"] - 1.0, seg["home"],
    ].astype(np.float64)


def make_head(spec):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    if spec["kind"] == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(spec["hidden"]), alpha=spec["alpha"],
                early_stopping=True, random_state=0, max_iter=spec["max_iter"],
            ),
        )
    ct = ColumnTransformer(
        [("sp", SplineTransformer(), [0, 1, 2, 5]), ("pl", "passthrough", [3, 4, 6, 7])]
    )
    return make_pipeline(
        ct, StandardScaler(), LogisticRegression(C=spec["C"], max_iter=spec["max_iter"])
    )


def head_dist(spec, Xtr, ytr, Xte, K, floor):
    ok = ytr >= 0
    m = make_head(spec).fit(Xtr[ok], ytr[ok])
    P = np.zeros((len(Xte), K))
    P[:, m.classes_] = m.predict_proba(Xte)
    P = np.maximum(P, floor)
    return P / P.sum(axis=1, keepdims=True)


def pick(P, y, floor):
    return np.where(y >= 0, P[np.arange(len(y)), np.maximum(y, 0)], floor)


def reliability(P, y, bins=10):
    ok = y >= 0
    Pk, yk = P[ok], y[ok]
    Y = np.zeros_like(Pk)
    Y[np.arange(len(yk)), yk] = 1.0
    pf, yf = Pk.ravel(), Y.ravel()
    edges = np.quantile(pf, np.linspace(0, 1, bins + 1))
    bi = np.clip(np.searchsorted(edges, pf, side="right") - 1, 0, bins - 1)
    rows = [[float(pf[bi == b].mean()), float(yf[bi == b].mean()), int((bi == b).sum())] for b in range(bins)]
    ece = float(sum(abs(r[0] - r[1]) * r[2] for r in rows) / len(pf))
    return {"deciles": rows, "ece": ece}


def group_reliability(P, y, grp):
    ok = y >= 0
    Pk, yk = P[ok], y[ok]
    out = {}
    for gname in np.unique(grp):
        cols = np.flatnonzero(grp == gname)
        pm = Pk[:, cols].sum(axis=1)
        obs = np.isin(yk, cols).astype(float)
        out[int(gname)] = [float(pm.mean()), float(obs.mean()), int(obs.sum())]
    return out


def boot_games(d, gid, rng):
    ug, inv = np.unique(gid, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnts = np.bincount(inv).astype(float)
    n = len(ug)
    res = np.empty(NBOOT)
    for i in range(NBOOT):
        s = rng.integers(0, n, n)
        res[i] = sums[s].sum() / cnts[s].sum()
    return res


def boot_blocks(per_block, rng):
    n = len(per_block)
    return np.array([per_block[rng.integers(0, n, n)].mean() for _ in range(NBOOT)])


def summ(res):
    return {
        "mean": float(np.mean(res)),
        "lo95": float(np.quantile(res, 0.025)),
        "hi95": float(np.quantile(res, 0.975)),
        "probability_positive": float(np.mean(res > 0)),
    }


def main():
    seg = pd.read_parquet(s10.OUT / "segments.parquet")
    po, pd_ = s10.segment_points(seg)
    keep = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    seg = seg[keep].reset_index(drop=True)
    po, pd_ = po[keep], pd_[keep]
    season = seg["season"].to_numpy().astype(int)
    train = np.isin(season, TRAIN)
    test = np.isin(season, TEST)
    code = class_codes(seg, po, pd_, train)
    cls, uniq = relabel(code, train)
    K = len(uniq)
    floor = 1.0 / (int(train.sum()) + K)
    log("drives", len(seg), "train", int(train.sum()), "test", int(test.sum()), "K", K,
        "unseen in test", int(((cls < 0) & test).sum()))
    grp = uniq // 10
    X = s10.design(seg)
    half, home = seg["half"].to_numpy(), seg["home"].to_numpy()
    he = seg["he"].to_numpy()
    lg = s10.game_lt(seg)
    gid = seg["game_id"].to_numpy()
    F = head_features(seg)
    ti = np.flatnonzero(test)
    res = {"n_train": int(train.sum()), "n_test": int(test.sum()), "K": K,
           "unseen_test": int(((cls < 0) & test).sum()), "floor": floor}

    bw = json.loads((REPO / "artifacts" / "sim10" / "bandwidths_week.json").read_text())
    hA, alphaA, gA = list(bw["h"]), float(bw["alpha"]), float(bw["g_sd"])
    res["kernel_A"] = {"h": hA, "alpha": alphaA, "g": gA, "source": "artifacts/sim10/bandwidths_week.json fit on 2009-2017"}

    marg = np.bincount(cls[train], minlength=K) + 0.5
    marg = marg / marg.sum()
    llM = np.where(cls[ti] >= 0, np.log(marg[np.maximum(cls[ti], 0)]), np.log(floor))

    pA = kernel_fold(X, cls, half, home, he, lg, test, train, K, hA, alphaA, gA, floor)
    log("kernel A test ll", float(np.log(pA).mean()))

    rng = np.random.default_rng(0)
    qsub = np.sort(rng.choice(np.flatnonzero(train), size=3000, replace=False))
    qmask = np.zeros(len(seg), bool)
    qmask[qsub] = True

    def kern_cv(h, alpha, g):
        tot, n = 0.0, 0
        for s in TRAIN:
            qm = qmask & (season == s)
            p = kernel_fold(X, cls, half, home, he, lg, qm, train & (season != s), K, h, alpha, g, floor)
            tot += float(np.log(p).sum())
            n += int(qm.sum())
        return tot / n

    h, alpha, g = list(hA), alphaA, gA
    best = kern_cv(h, alpha, g)
    log("kernel B start train-CV ll", best)
    mult = [0.5, 0.71, 1.41, 2.0]
    for rnd in range(2):
        improved = False
        for i in range(len(h) + 2):
            if i < len(h):
                if not np.isfinite(h[i]):
                    continue
                cands = [h[i] * m for m in mult]
            elif i == len(h):
                cands = [alpha * m for m in mult]
            else:
                cands = [g + m - 1.0 for m in mult]
            for v in cands:
                hh, al, gg = list(h), alpha, g
                if i < len(h):
                    hh[i] = v
                elif i == len(h):
                    al = v
                else:
                    gg = v
                ll = kern_cv(hh, al, gg)
                if ll > best + 1e-6:
                    best, h, alpha, g, improved = ll, hh, al, gg, True
            log("kernel B round", rnd, "dim", i, "best", best)
        if not improved:
            break
    hB, alphaB, gB = h, alpha, g
    res["kernel_B"] = {"h": [float(v) for v in hB], "alpha": alphaB, "g": gB, "train_cv_ll": best}
    pB = kernel_fold(X, cls, half, home, he, lg, test, train, K, hB, alphaB, gB, floor)
    log("kernel B test ll", float(np.log(pB).mean()))

    specs = []
    for hid in ([32], [64, 64]):
        for a in (1e-4, 1e-2, 1.0):
            specs.append({"kind": "mlp", "hidden": hid, "alpha": a, "max_iter": 200})
    for c in (0.01, 0.1, 1.0):
        specs.append({"kind": "logit", "C": c, "max_iter": 300})
    blocks = [(2018, 2019), (2020, 2021), (2022, 2023)]
    cvs = []
    for sp in specs:
        tot, n = 0.0, 0
        for bl in blocks:
            te = np.isin(season, bl)
            P = head_dist(sp, F[train & ~te], cls[train & ~te], F[te], K, floor)
            tot += float(np.log(pick(P, cls[te], floor)).sum())
            n += int(te.sum())
        cvs.append(tot / n)
        log("head cv", sp, tot / n)
    spec = specs[int(np.argmax(cvs))]
    res["head_cv"] = [{"spec": s, "cv_ll": c} for s, c in zip(specs, cvs, strict=True)]
    res["head_selected"] = spec
    best_of = {k: max(((c, s) for c, s in zip(cvs, specs, strict=True) if s["kind"] == k), key=lambda t: t[0])[1] for k in ("mlp", "logit")}

    dists = {}
    for name, sp in (("selected", spec), ("mlp", best_of["mlp"]), ("logit", best_of["logit"])):
        dists[name] = head_dist(sp, F[train], cls[train], F[ti], K, floor)
        log("head", name, sp, "test ll", float(np.log(pick(dists[name], cls[ti], floor)).mean()))
    llH = np.log(pick(dists["selected"], cls[ti], floor))
    llA, llB = np.log(pA), np.log(pB)
    res["test_ll"] = {
        "learned_selected": float(llH.mean()),
        "learned_mlp": float(np.log(pick(dists["mlp"], cls[ti], floor)).mean()),
        "learned_logit": float(np.log(pick(dists["logit"], cls[ti], floor)).mean()),
        "kernel_A_locked_bw": float(llA.mean()), "kernel_B_train_refit": float(llB.mean()),
        "marginal": float(llM.mean()),
    }
    rb = np.random.default_rng(1)
    gt = gid[ti]
    comps = {
        "learned_minus_kernelA": llH - llA,
        "learned_minus_kernelB": llH - llB,
        "learned_minus_marginal": llH - llM,
        "kernelB_minus_kernelA": llB - llA,
    }
    res["test_diff_game_bootstrap"] = {k: summ(boot_games(v, gt, rb)) for k, v in comps.items()}
    res["test_diff_by_season"] = {
        str(s): {k: float(v[season[ti] == s].mean()) for k, v in comps.items()} for s in TEST
    }
    res["n_test_games"] = int(len(np.unique(gt)))

    PKA = kernel_fold(X, cls, half, home, he, lg, test, train, K, hA, alphaA, gA, floor, full=True)
    PKB = kernel_fold(X, cls, half, home, he, lg, test, train, K, hB, alphaB, gB, floor, full=True)
    for nm, P in (("learned", dists["selected"]), ("kernel_A", PKA), ("kernel_B", PKB)):
        res[f"reliability_{nm}"] = {**reliability(P, cls[ti]), "by_outcome_group": group_reliability(P, cls[ti], grp)}
    log("reliability done")

    per = {}
    for s in ALL:
        te = season == s
        tr = ~te
        pk = kernel_fold(X, cls, half, home, he, lg, te, tr, K, hB, alphaB, gB, floor)
        P = head_dist(spec, F[tr], cls[tr], F[te], K, floor)
        per[s] = (np.log(pick(P, cls[te], floor)), np.log(pk))
        log("LOSO season", s, "learned", float(per[s][0].mean()), "kernelB", float(per[s][1].mean()))
    ps = np.array([per[s][0].mean() - per[s][1].mean() for s in ALL])
    dall = np.concatenate([per[s][0] - per[s][1] for s in ALL])
    gall = np.concatenate([gid[season == s] for s in ALL])
    res["loso_all_seasons"] = {
        "per_season_learned_minus_kernelB": {str(s): float(v) for s, v in zip(ALL, ps, strict=True)},
        "season_block_bootstrap": summ(boot_blocks(ps, rb)),
        "game_bootstrap": summ(boot_games(dall, gall, rb)),
        "learned_ll": float(np.concatenate([per[s][0] for s in ALL]).mean()),
        "kernel_B_ll": float(np.concatenate([per[s][1] for s in ALL]).mean()),
        "n": int(len(dall)), "n_games": int(len(np.unique(gall))),
    }
    (ART / "results.json").write_text(json.dumps(res, indent=1))
    log("done")


if __name__ == "__main__":
    main()
