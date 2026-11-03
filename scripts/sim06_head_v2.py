import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["SIM10_SEASONS"] = "2009-2017"
os.environ["SIM10_OUT"] = str(REPO / "artifacts" / "sim10")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sim06_learned_head as h6  # noqa: E402
import sim10_drive as s10  # noqa: E402

warnings.filterwarnings("ignore")
NQ = int(os.environ.get("DIAG_NQ", "900"))
MAXIT = int(os.environ.get("V2_MAXIT", "80"))
OUT = REPO / "artifacts" / "sim06" / "results_v2.json"
T0 = time.time()
CANDS = ("masked", "masked_season")


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def feasibility(cls, half, secs, he, K, fit):
    anynon = np.zeros((3, K), bool)
    maxhe = np.full((3, K), -1.0)
    for hf in (1, 2):
        m = fit & (half == hf) & (cls >= 0)
        a = m & (he == 0)
        anynon[hf, np.unique(cls[a])] = True
        b = m & (he == 1)
        if b.any():
            np.maximum.at(maxhe[hf], cls[b], secs[b])
    return anynon, maxhe


def feas_mask(idx, cls, half, secs, anynon, maxhe, K):
    hf = half[idx].astype(int)
    hf = np.clip(hf, 1, 2)
    m = anynon[hf] | (maxhe[hf] >= secs[idx][:, None])
    ok = cls[idx] >= 0
    m[np.flatnonzero(ok), cls[idx][ok]] = True
    return m


class MaskedMLP:
    def __init__(self, hidden=(64, 64), alpha=1e-4, lr=1e-3, batch=200, seed=0):
        self.h, self.alpha, self.lr, self.batch, self.seed = hidden, alpha, lr, batch, seed

    def _fwd(self, A):
        acts = [A]
        for i, (w, b) in enumerate(zip(self.W, self.B, strict=True)):
            A = A @ w + b
            if i < len(self.W) - 1:
                A = np.maximum(A, 0.0)
            acts.append(A)
        return acts

    def _proba(self, Z, M):
        Z = np.where(M, Z, -np.inf)
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y, M, maxit):
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        K = M.shape[1]
        sizes = [X.shape[1], *self.h, K]
        self.W = [
            rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1]))
            for i in range(len(sizes) - 1)
        ]
        self.B = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        n = len(Xs)
        perm = rng.permutation(n)
        nv = n // 10
        vi, ti = perm[:nv], perm[nv:]
        params = self.W + self.B
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        t = 0
        best, bestp, bad = np.inf, None, 0
        for ep in range(maxit):  # noqa: B007
            o = rng.permutation(ti)
            for a in range(0, len(o), self.batch):
                bi = o[a : a + self.batch]
                acts = self._fwd(Xs[bi])
                P = self._proba(acts[-1], M[bi])
                P[np.arange(len(bi)), y[bi]] -= 1.0
                d = P / len(bi)
                gW, gB = [None] * len(self.W), [None] * len(self.W)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ d + self.alpha * self.W[i] / len(bi)
                    gB[i] = d.sum(0)
                    if i > 0:
                        d = (d @ self.W[i].T) * (acts[i] > 0)
                t += 1
                for j, (p, g) in enumerate(zip(params, gW + gB, strict=True)):
                    m1[j] = 0.9 * m1[j] + 0.1 * g
                    m2[j] = 0.999 * m2[j] + 0.001 * g * g
                    p -= self.lr * (m1[j] / (1 - 0.9**t)) / (np.sqrt(m2[j] / (1 - 0.999**t)) + 1e-8)
            Pv = self._proba(self._fwd(Xs[vi])[-1], M[vi])
            ll = -np.log(np.maximum(Pv[np.arange(nv), y[vi]], 1e-300)).mean()
            if ll < best - 1e-4:
                best, bad = ll, 0
                bestp = [p.copy() for p in params]
            else:
                bad += 1
                if bad >= 10:
                    break
        nW = len(self.W)
        self.W = bestp[:nW]
        self.B = bestp[nW:]
        self.epochs = ep + 1
        return self

    def predict(self, X, M):
        return self._proba(self._fwd((X - self.mu) / self.sd)[-1], M)


def feats(F, season, use_season):
    return np.c_[F, (season - 2013.0) / 4.0] if use_season else F


def fit_predict(use_season, Ftr, str_, ytr, Mtr, Fte, ste, Mte, maxit):
    ok = ytr >= 0
    mdl = MaskedMLP().fit(feats(Ftr[ok], str_[ok], use_season), ytr[ok], Mtr[ok], maxit)
    return mdl.predict(feats(Fte, ste, use_season), Mte), mdl.epochs


def lossp(P, y, floor):
    p = np.where(y >= 0, P[np.arange(len(y)), np.maximum(y, 0)], floor)
    return np.log(np.maximum(p, floor))


def main():
    seg = pd.read_parquet(s10.OUT / "segments.parquet")
    po, pd_ = s10.segment_points(seg)
    keep = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    seg = seg[keep].reset_index(drop=True)
    po, pd_ = po[keep], pd_[keep]
    season = seg["season"].to_numpy().astype(int)
    X = s10.design(seg)
    half, home = seg["half"].to_numpy(), seg["home"].to_numpy()
    he = seg["he"].to_numpy()
    secs = seg["secs"].clip(lower=0.0).to_numpy(float)
    lg = s10.game_lt(seg)
    F = h6.head_features(seg)
    spec = json.loads((REPO / "artifacts" / "sim06" / "results.json").read_text())["head_selected"]
    bw = json.loads((REPO / "artifacts" / "sim10" / "bandwidths_week.json").read_text())
    hA, alphaA, gA = list(bw["h"]), float(bw["alpha"]), float(bw["g_sd"])
    rng = np.random.default_rng(0)
    rng_in = np.random.default_rng(1)
    rows = []
    sel_log = {}
    for s in range(2009, 2018):
        fit = season != s
        code = h6.class_codes(seg, po, pd_, fit)
        cls, uniq = h6.relabel(code, fit)
        K = len(uniq)
        floor = 1.0 / (int(fit.sum()) + K)
        ok = fit & (cls >= 0)
        mpo = np.bincount(cls[ok], weights=po[ok], minlength=K) / np.maximum(
            np.bincount(cls[ok], minlength=K), 1
        )
        qi = np.sort(
            rng.choice(
                np.flatnonzero(season == s), size=min(NQ, int((season == s).sum())), replace=False
            )
        )
        ii = np.sort(rng_in.choice(np.flatnonzero(fit & (cls >= 0)), size=NQ, replace=False))
        qm = np.zeros(len(seg), bool)
        qm[qi] = True
        PK = h6.kernel_fold(
            X, cls, half, home, he, lg, qm, fit, K, hA, alphaA, gA, floor, full=True
        )
        anynon, maxhe = feasibility(cls, half, secs, he, K, fit)
        fi = np.flatnonzero(fit)
        Mtr = feas_mask(fi, cls, half, secs, anynon, maxhe, K)
        Mq = feas_mask(qi, cls, half, secs, anynon, maxhe, K)
        Mi = feas_mask(ii, cls, half, secs, anynon, maxhe, K)
        ytr = cls[fi]
        seasons_fit = sorted(set(season[fi]))
        blocks = [seasons_fit[b::3] for b in range(3)]
        inner = {c: [] for c in CANDS}
        for blk in blocks:
            te = np.isin(season[fi], blk)
            for c in CANDS:
                P, _ = fit_predict(
                    c == "masked_season",
                    F[fi][~te],
                    season[fi][~te],
                    ytr[~te],
                    Mtr[~te],
                    F[fi][te],
                    season[fi][te],
                    Mtr[te],
                    MAXIT,
                )
                inner[c].append(float(lossp(P, ytr[te], floor).mean()))
        score = {c: float(np.mean(v)) for c, v in inner.items()}
        pick = max(score, key=score.get)
        sel_log[s] = {"inner_ll": score, "chosen": pick}
        res = {}
        for c in CANDS:
            Pq, ep = fit_predict(  # noqa: RUF059
                c == "masked_season", F[fi], season[fi], ytr, Mtr, F[qi], season[qi], Mq, MAXIT
            )
            res[c] = Pq
        PH = h6.head_dist(spec, F[fit], cls[fit], F[qi], K, floor)
        PH_in = h6.head_dist(spec, F[fit], cls[fit], F[ii], K, floor)
        mdl_in = {}
        for c in CANDS:
            mdl = MaskedMLP().fit(
                feats(F[fi][ytr >= 0], season[fi][ytr >= 0], c == "masked_season"),
                ytr[ytr >= 0],
                Mtr[ytr >= 0],
                MAXIT,
            )
            mdl_in[c] = mdl.predict(feats(F[ii], season[ii], c == "masked_season"), Mi)
        mask_k = np.zeros((len(qi), K), bool)
        for j, i in enumerate(qi):
            st = (half == half[i]) & (home == home[i]) & fit & (cls >= 0)
            pi = np.flatnonzero(st)
            z = np.zeros(len(pi))
            for d_, hd in zip(s10.FIT_DIMS, hA, strict=True):
                if not np.isfinite(hd):
                    continue
                hq = hd * np.exp(gA * (lg[i] - s10.L0)) if d_ == 1 else hd
                z += ((X[pi, d_] - X[i, d_]) / hq) ** 2
            z[(he[pi] == 1) & (X[pi, 2] < X[i, 2])] = np.inf
            w = np.exp(-0.5 * (z - z.min()))
            nz = w > 1e-12
            mask_k[j] = np.bincount(cls[pi[nz]], weights=w[nz], minlength=K) > 0

        def kmask(P):
            pm = P * mask_k  # noqa: B023
            return pm / pm.sum(1, keepdims=True)

        models = {"kernel": PK, "head_v1": PH, "head_v1_kmask": kmask(PH)}
        for c in CANDS:
            models["v2_" + c] = res[c]
            models["v2_" + c + "_kmask"] = kmask(res[c])
        models["v2_chosen"] = res[pick]
        models["v2_chosen_kmask"] = kmask(res[pick])
        kmask_cov = float(mask_k[np.arange(len(qi)), np.maximum(cls[qi], 0)][cls[qi] >= 0].mean())
        for name, P in models.items():
            ept = P @ mpo
            ll = lossp(P, cls[qi], floor)
            for j, i in enumerate(qi):
                rows.append(
                    (
                        s,
                        "oos",
                        name,
                        ept[j],
                        ll[j],
                        po[i],
                        seg["yl"].iat[i],
                        seg["sd"].iat[i],
                        seg["secs"].iat[i],
                        half[i],
                    )
                )
        inm = {"head_v1": PH_in}
        for c in CANDS:
            inm["v2_" + c] = mdl_in[c]
        inm["v2_chosen"] = mdl_in[pick]
        for name, P in inm.items():
            ept = P @ mpo
            ll = lossp(P, cls[ii], floor)
            for j, i in enumerate(ii):
                rows.append(
                    (
                        s,
                        "ins",
                        name,
                        ept[j],
                        ll[j],
                        po[i],
                        seg["yl"].iat[i],
                        seg["sd"].iat[i],
                        seg["secs"].iat[i],
                        half[i],
                    )
                )
        sel_log[s]["kernel_support_covers_true_class"] = kmask_cov
        sel_log[s]["train_infeasible_true_class_fixed"] = 0
        log("season", s, "K", K, score, pick)
    d = pd.DataFrame(
        rows, columns=["season", "split", "model", "ep", "ll", "obs", "yl", "sd", "secs", "half"]
    )
    summ = {}
    for sp in ("oos", "ins"):
        x = d[d.split == sp]
        obs = x[x.model == "head_v1"].groupby("season").obs.mean().mean()  # noqa: F841
        tab = x.groupby("model").agg(ep=("ep", "mean"), ll=("ll", "mean"))
        tab["obs"] = (
            x[x.model == "head_v1"].obs.mean()  # noqa: RUF034
            if sp == "ins"
            else x[x.model == "head_v1"].obs.mean()
        )
        tab["ep_minus_obs"] = tab["ep"] - tab["obs"]
        summ[sp] = tab.round(5).to_dict("index")
    o = d[(d.split == "oos")]
    bins = {}
    for nm, edges in (
        ("yl", [0, 25, 50, 75, 100]),
        ("sd", [-100, -14, -7, -1, 0, 7, 14, 100]),
        ("half", [0, 1, 2]),
    ):
        ob = o[o.model == "head_v1"].reset_index(drop=True)
        b = pd.cut(ob[nm], edges)
        t = {"obs": ob.groupby(b).obs.mean()}
        for m in ("kernel", "head_v1_kmask", "v2_chosen_kmask", "v2_chosen"):
            xm = o[o.model == m].reset_index(drop=True)
            t[m + "_minus_obs"] = xm.groupby(b).ep.mean() - t["obs"]
        bins[nm] = pd.DataFrame(t).round(3).astype(float).rename(index=str).to_dict("index")
    per_season = {}
    for m in ("kernel", "head_v1", "v2_chosen", "head_v1_kmask", "v2_chosen_kmask"):
        xm = o[o.model == m]
        per_season[m] = {
            "ll": xm.groupby("season").ll.mean().round(4).to_dict(),
            "ep": xm.groupby("season").ep.mean().round(4).to_dict(),
        }
    base = o[o.model == "kernel"].groupby("season").ll.mean()
    diffs = {}
    for m in ("head_v1", "v2_chosen", "v2_masked", "v2_masked_season"):
        dd = (o[o.model == m].groupby("season").ll.mean() - base).to_numpy()
        br = np.random.default_rng(3)
        bs = np.array([dd[br.integers(0, len(dd), len(dd))].mean() for _ in range(4000)])
        diffs[m + "_minus_kernel_ll"] = {
            "mean": float(dd.mean()),
            "lo95": float(np.quantile(bs, 0.025)),
            "hi95": float(np.quantile(bs, 0.975)),
            "probability_positive": float((bs > 0).mean()),
            "seasons_positive": int((dd > 0).sum()),
        }
    dd = (
        o[o.model == "v2_chosen"].groupby("season").ll.mean()
        - o[o.model == "head_v1"].groupby("season").ll.mean()
    ).to_numpy()
    bs = np.array([dd[br.integers(0, len(dd), len(dd))].mean() for _ in range(4000)])
    diffs["v2_chosen_minus_head_v1_ll"] = {
        "mean": float(dd.mean()),
        "lo95": float(np.quantile(bs, 0.025)),
        "hi95": float(np.quantile(bs, 0.975)),
        "probability_positive": float((bs > 0).mean()),
        "seasons_positive": int((dd > 0).sum()),
    }
    out = {
        "protocol": "LOSO 2009-17, 900 random drives/season, kernel A, same qi as sim06_head_diagnosis.py; in-sample = 900 random fit-season drives",  # noqa: E501
        "candidates": list(CANDS),
        "selection": sel_log,
        "summary": summ,
        "bins_oos_ep_minus_obs": bins,
        "per_season": per_season,
        "ll_diffs": diffs,
        "looks": {"candidates": 2, "inner_blocks": 3, "outer_folds": 9},
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(pd.DataFrame(summ["oos"]).T.to_string())
    print(pd.DataFrame(summ["ins"]).T.to_string())
    print(json.dumps(diffs, indent=1))
    print(json.dumps({k: v["chosen"] for k, v in sel_log.items()}))
    for nm in bins:
        print(nm, pd.DataFrame(bins[nm]).T.to_string())


def final(use_season):
    seg = pd.read_parquet(s10.OUT / "segments.parquet")
    po, pd_ = s10.segment_points(seg)
    keep = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    seg = seg[keep].reset_index(drop=True)
    po, pd_ = po[keep], pd_[keep]
    season = seg["season"].to_numpy().astype(int)
    half = seg["half"].to_numpy()
    he = seg["he"].to_numpy()
    secs = seg["secs"].clip(lower=0.0).to_numpy(float)
    F = h6.head_features(seg)
    fit = np.ones(len(seg), bool)
    code = h6.class_codes(seg, po, pd_, fit)
    cls, uniq = h6.relabel(code, fit)
    K = len(uniq)
    anynon, maxhe = feasibility(cls, half, secs, he, K, fit)
    M = feas_mask(np.arange(len(seg)), cls, half, secs, anynon, maxhe, K)
    ok = cls >= 0
    mdl = MaskedMLP().fit(feats(F[ok], season[ok], use_season), cls[ok], M[ok], MAXIT)
    arrs = {
        "n": np.array(len(seg)),
        "k": np.array(K),
        "use_season": np.array(int(use_season)),
        "mean": mdl.mu,
        "scale": mdl.sd,
        "anynon": anynon,
        "maxhe": maxhe,
    }
    for i, (w_, b_) in enumerate(zip(mdl.W, mdl.B, strict=True)):
        arrs[f"w{i}"] = w_
        arrs[f"b{i}"] = b_
    np.savez(REPO / "artifacts" / "sim06" / "head_v2.npz", **arrs)
    log("saved head_v2.npz K", K, "epochs", mdl.epochs)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        final(sys.argv[2] == "season")
    else:
        main()
