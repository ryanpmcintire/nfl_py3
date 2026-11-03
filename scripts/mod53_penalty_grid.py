# ruff: noqa: E501
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U8 = Path(__file__).resolve().parent / "mod23_unit8_span.py"
NS = {"__file__": str(U8), "__name__": "mod23_unit8_span_import"}
exec(compile(U8.read_text().rsplit("\nmain()", 1)[0], str(U8), "exec"), NS)

from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from nfl_ats import clv  # noqa: E402
from nfl_ats import margin as margin_mod  # noqa: E402
from nfl_ats.mass_preserving_lattice import band_read  # noqa: E402

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
compare = NS["compare"]
OUT = u3.ROOT / "artifacts" / "mod53_unit2"
ORIG = margin_mod.make_margin_estimator
LOG10_START = -1.0
LOG10_STEP = 0.5
DOF_FLOOR = 1.0
MAX_COMPONENTS = 30
MIN_SEASONS = 3
EIG_REL_TOL = 1e-10
LOG = []
STATE = {"seasons": None, "arm": None, "results": None}
CACHE = {}
PROB_EPS = 1e-6
SEASON_MAP = None


def grid_for(gram_centered):
    d = np.clip(np.linalg.eigvalsh(gram_centered), 0.0, None)
    lams = []
    i = 0
    while True:
        lam = 10.0 ** (LOG10_START + LOG10_STEP * i)
        lams.append(lam)
        if float(np.sum(d / (d + lam))) < DOF_FLOOR:
            return np.array(lams), d
        i += 1


def dof_at(d, lam):
    return float(np.sum(d / (d + lam)))


def ridge_choice(Z, y, seasons):
    A = np.hstack([Z, np.ones((len(Z), 1))])
    uniq = np.unique(seasons)
    Zc = Z - Z.mean(axis=0)
    lams, d = grid_for(Zc.T @ Zc)
    blocks = {s: A[seasons == s] for s in uniq}
    yy = {s: y[seasons == s] for s in uniq}
    G = {s: blocks[s].T @ blocks[s] for s in uniq}
    c = {s: blocks[s].T @ yy[s] for s in uniq}
    Gt = sum(G.values())
    ct = sum(c.values())
    eye = 1e-12 * np.eye(A.shape[1])
    sse = np.zeros(len(lams))
    for j, lam in enumerate(lams):
        pen = np.full(A.shape[1], lam)
        pen[-1] = 0.0
        pen = np.diag(pen)
        for s in uniq:
            w = np.linalg.solve(Gt - G[s] + pen + eye, ct - c[s])
            sse[j] += float(np.sum((yy[s] - blocks[s] @ w) ** 2))
    return A, lams, d, sse


def cover_prob(pool_line, pool_margin, line, point):
    r = band_read(pool_line, pool_margin, float(line), float(point))
    total = r.cover + r.loss
    return r.cover / total if total > 0.0 else 0.5


def score_games(pool_line, pool_margin, lines, points, margins):
    ll = 0.0
    hit = 0
    n = 0
    for line, point, m in zip(lines, points, margins, strict=True):
        if m == line:
            continue
        p = min(max(cover_prob(pool_line, pool_margin, line, point), PROB_EPS), 1.0 - PROB_EPS)
        home = m > line
        ll += -np.log(p if home else 1.0 - p)
        hit += int((p >= 0.5) == home)
        n += 1
    return ll, hit, n


def decision_choice(Z, y, res, seasons):
    A = np.hstack([Z, np.ones((len(Z), 1))])
    uniq = np.unique(seasons)
    Zc = Z - Z.mean(axis=0)
    lams, d = grid_for(Zc.T @ Zc)
    line = res - y
    blocks = {s: A[seasons == s] for s in uniq}
    G = {s: blocks[s].T @ blocks[s] for s in uniq}
    c = {s: blocks[s].T @ y[seasons == s] for s in uniq}
    Gt = sum(G.values())
    ct = sum(c.values())
    eye = 1e-12 * np.eye(A.shape[1])
    ll = np.zeros(len(lams))
    hit = np.zeros(len(lams))
    cnt = np.zeros(len(lams))
    for s in uniq:
        m = seasons == s
        for j, lam in enumerate(lams):
            pen = np.full(A.shape[1], lam)
            pen[-1] = 0.0
            w = np.linalg.solve(Gt - G[s] + np.diag(pen) + eye, ct - c[s])
            loc = float(np.median(y[~m] - A[~m] @ w))
            pts = line[m] + blocks[s] @ w + loc
            a, b, n = score_games(line[~m], res[~m], line[m], pts, res[m])
            ll[j] += a
            hit[j] += b
            cnt[j] += n
    return A, lams, d, ll / cnt, hit / cnt, line


def insample_score(A, y, res, line, lam):
    pen = np.full(A.shape[1], lam)
    pen[-1] = 0.0
    w = np.linalg.solve(A.T @ A + np.diag(pen) + 1e-12 * np.eye(A.shape[1]), A.T @ y)
    loc = float(np.median(y - A @ w))
    a, b, n = score_games(line, res, line, line + A @ w + loc, res)
    return a / n, b / n


def pca_choice(Z, y, seasons):
    uniq = np.unique(seasons)
    n = len(Z)
    stats = {s: (Z[seasons == s], y[seasons == s]) for s in uniq}
    tot_s = Z.sum(axis=0)
    tot_g = Z.T @ Z
    tot_zy = Z.T @ y
    tot_y = float(y.sum())
    sse = np.zeros(min(MAX_COMPONENTS, Z.shape[1]))
    for s in uniq:
        zs, ys = stats[s]
        nt = n - len(zs)
        st = tot_s - zs.sum(axis=0)
        mu = st / nt
        gt = tot_g - zs.T @ zs
        yt = tot_y - float(ys.sum())
        ybar = yt / nt
        zy = tot_zy - zs.T @ ys
        cov = gt - nt * np.outer(mu, mu)
        cy = zy - mu * yt
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1][: len(sse)]
        w, v = w[order], v[:, order]
        ok = w > EIG_REL_TOL * w[0]
        b = np.where(ok, (v.T @ cy) / np.where(ok, w, 1.0), 0.0)
        contrib = ((zs - mu) @ v) * b
        pred = ybar + np.cumsum(contrib, axis=1)
        sse += np.sum((ys[:, None] - pred) ** 2, axis=0)
    return sse


class ArmRidge(BaseEstimator, RegressorMixin):
    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float)
        seasons = STATE["seasons"]
        arm = STATE["arm"]
        uniq = np.unique(seasons)
        rec = {"season": int(seasons.max()), "n": len(y)}
        if arm == "I":
            self.coef_ = np.zeros(Z.shape[1])
            self.intercept_ = float(y.mean())
            rec.update(dof=0.0)
        elif len(uniq) < MIN_SEASONS:
            r = Ridge(alpha=BASE_ALPHA).fit(Z, y)
            self.coef_, self.intercept_ = r.coef_, float(r.intercept_)
            rec.update(fallback=True)
        elif arm in ("L", "C"):
            S = int(seasons.max())
            res = STATE["results"]
            prior = seasons < S
            key = (arm, S, int(prior.sum()))
            if len(np.unique(seasons[prior])) < MIN_SEASONS:
                r = Ridge(alpha=BASE_ALPHA).fit(Z, y)
                self.coef_, self.intercept_ = r.coef_, float(r.intercept_)
                rec.update(fallback=True)
            else:
                if key not in CACHE:
                    A0, lams, d, ll, acc, line = decision_choice(
                        Z[prior], y[prior], res[prior], seasons[prior]
                    )
                    obj = -ll if arm == "L" else acc
                    j = int(np.argmax(obj))
                    i10 = int(np.argmin(np.abs(np.log10(lams) - np.log10(BASE_ALPHA))))
                    ins = insample_score(A0, y[prior], res[prior], line, lams[j])
                    ins10 = insample_score(A0, y[prior], res[prior], line, lams[i10])
                    CACHE[key] = {
                        "lam": float(lams[j]),
                        "idx": j,
                        "grid_len": len(lams),
                        "dof": dof_at(d, lams[j]),
                        "inner_ll": float(ll[j]),
                        "inner_acc": float(acc[j]),
                        "inner_ll_a10": float(ll[i10]),
                        "inner_acc_a10": float(acc[i10]),
                        "in_ll": ins[0],
                        "in_acc": ins[1],
                        "in_ll_a10": ins10[0],
                        "in_acc_a10": ins10[1],
                        "ll_curve": [float(v) for v in ll],
                        "acc_curve": [float(v) for v in acc],
                        "lams": [float(v) for v in lams],
                    }
                ch = CACHE[key]
                A = np.hstack([Z, np.ones((len(Z), 1))])
                pen = np.full(A.shape[1], ch["lam"])
                pen[-1] = 0.0
                w = np.linalg.solve(A.T @ A + np.diag(pen) + 1e-12 * np.eye(A.shape[1]), A.T @ y)
                self.coef_, self.intercept_ = w[:-1], float(w[-1])
                rec.update(
                    {k: v for k, v in ch.items() if k not in ("ll_curve", "acc_curve", "lams")}
                )
        elif arm == "A":
            A, lams, d, sse = ridge_choice(Z, y, seasons)
            j = int(np.argmin(sse))
            pen = np.full(A.shape[1], lams[j])
            pen[-1] = 0.0
            w = np.linalg.solve(A.T @ A + np.diag(pen) + 1e-12 * np.eye(A.shape[1]), A.T @ y)
            self.coef_, self.intercept_ = w[:-1], float(w[-1])
            i10 = int(np.argmin(np.abs(np.log10(lams) - np.log10(BASE_ALPHA))))
            rec.update(
                lam=float(lams[j]),
                idx=j,
                grid_len=len(lams),
                grid_top=float(lams[-1]),
                dof=dof_at(d, lams[j]),
                inner_mse=float(sse[j] / len(y)),
                inner_mse_base=float(sse[i10] / len(y)),
                curve=[float(v / len(y)) for v in sse],
                lams=[float(v) for v in lams],
                train_mse=float(np.mean((y - Z @ self.coef_ - self.intercept_) ** 2)),
            )
        elif arm == "P":
            sse = pca_choice(Z, y, seasons)
            k = int(np.argmin(sse)) + 1
            mu = Z.mean(axis=0)
            Zc = Z - mu
            w, v = np.linalg.eigh(Zc.T @ Zc)
            order = np.argsort(w)[::-1][:k]
            w, v = w[order], v[:, order]
            ok = w > EIG_REL_TOL * w[0]
            b = np.where(ok, (v.T @ (Zc.T @ y)) / np.where(ok, w, 1.0), 0.0)
            self.coef_ = v @ b
            self.intercept_ = float(y.mean() - mu @ self.coef_)
            rec.update(
                k=k,
                dof=float(k),
                inner_mse=float(sse[k - 1] / len(y)),
                curve=[float(x / len(y)) for x in sse],
                train_mse=float(np.mean((y - Z @ self.coef_ - self.intercept_) ** 2)),
            )
        LOG.append(rec)
        return self

    def predict(self, Z):
        return np.asarray(Z, dtype=float) @ self.coef_ + self.intercept_


class ArmPipe(Pipeline):
    def fit(self, X, y=None, **kw):
        cols = list(X.columns)
        keys = pd.util.hash_pandas_object(X[cols], index=False).to_numpy()
        looked = SEASON_MAP.reindex(keys)
        if looked["season"].isna().any():
            raise RuntimeError("season lookup failed")
        STATE["seasons"] = looked["season"].to_numpy().astype(int)
        STATE["results"] = looked["result"].to_numpy(dtype=float)
        return super().fit(X, y, **kw)


def install(arm):
    STATE["arm"] = arm

    def make(model_name, random_state=42, **kw):
        base = ORIG(model_name, random_state, **kw)
        if model_name != "ridge":
            return base
        return ArmPipe(steps=[*base.steps[:-1], ("regressor", ArmRidge())])

    margin_mod.make_margin_estimator = make


def selfcheck():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 6))
    y = Z[:, 0] + rng.normal(size=300)
    seasons = np.repeat(np.arange(2010, 2020), 30)
    A, lams, d, sse = ridge_choice(Z, y, seasons)
    lam = 10.0
    pen = np.full(7, lam)
    pen[-1] = 0.0
    w = np.linalg.solve(A.T @ A + np.diag(pen), A.T @ y)
    r = Ridge(alpha=lam).fit(Z, y)
    ok = np.allclose(w[:-1], r.coef_, atol=1e-8) and abs(w[-1] - r.intercept_) < 1e-8
    held = np.zeros(len(lams))
    for j, lm in enumerate(lams):
        for s in np.unique(seasons):
            m = seasons == s
            rr = Ridge(alpha=lm).fit(Z[~m], y[~m])
            held[j] += float(np.sum((y[m] - rr.predict(Z[m])) ** 2))
    ok2 = np.allclose(held, sse, rtol=1e-6)
    zc = Z - Z.mean(axis=0)
    tr = float(np.trace(zc @ np.linalg.inv(zc.T @ zc + lam * np.eye(6)) @ zc.T))
    dof_ok = abs(dof_at(d, lam) - tr) < 1e-8
    pk = pca_choice(Z, y, seasons)
    held_p = np.zeros(len(pk))
    for s in np.unique(seasons):
        m = seasons == s
        mu = Z[~m].mean(axis=0)
        _, _, vt = np.linalg.svd(Z[~m] - mu, full_matrices=False)
        for k in range(1, 7):
            xk = (Z[~m] - mu) @ vt[:k].T
            ols = np.linalg.lstsq(
                np.hstack([np.ones((int((~m).sum()), 1)), xk]), y[~m], rcond=None
            )[0]
            pred = ols[0] + ((Z[m] - mu) @ vt[:k].T) @ ols[1:]
            held_p[k - 1] += float(np.sum((y[m] - pred) ** 2))
    ok3 = np.allclose(held_p[:6], pk[:6], rtol=1e-6)
    print(
        "selfcheck ridge_equals_sklearn",
        bool(ok),
        "inner_loso_equals_sklearn",
        bool(ok2),
        "dof_equals_trace",
        bool(dof_ok),
        "pca_inner_equals_svd_ols",
        bool(ok3),
        flush=True,
    )
    pool_l = rng.normal(size=400) * 6.0
    pool_m = np.round(pool_l + rng.normal(size=400) * 13.0)
    ll_c, _hit, n_c = score_games(pool_l, pool_m, pool_l[:20], pool_l[:20], pool_m[:20])
    ok4 = n_c > 0 and bool(np.isfinite(ll_c))
    print("selfcheck decision_score_finite", ok4, flush=True)
    if not (ok and ok2 and dof_ok and ok3 and ok4):
        raise SystemExit("selfcheck failed")


def summarize_log(arm):
    df = pd.DataFrame([r for r in LOG if not r.get("fallback")])
    out = {}
    for s, g in df.groupby("season"):
        r = {
            "fits": len(g),
            "dof_median": float(g["dof"].median()),
            "dof_min_max": [float(g["dof"].min()), float(g["dof"].max())],
        }
        if arm == "A":
            r.update(
                lam_median=float(g["lam"].median()),
                lam_min_max=[float(g["lam"].min()), float(g["lam"].max())],
                at_grid_top_fits=int((g["idx"] == g["grid_len"] - 1).sum()),
                at_grid_bottom_fits=int((g["idx"] == 0).sum()),
                grid_top_median=float(g["grid_top"].median()),
                inner_mse_chosen=float(g["inner_mse"].mean()),
                inner_mse_alpha10=float(g["inner_mse_base"].mean()),
                train_mse=float(g["train_mse"].mean()),
            )
        if arm == "P":
            r.update(
                k_median=float(g["k"].median()),
                k_min_max=[int(g["k"].min()), int(g["k"].max())],
                at_k_max_fits=int((g["k"] == MAX_COMPONENTS).sum()),
                at_k_min_fits=int((g["k"] == 1).sum()),
                inner_mse_chosen=float(g["inner_mse"].mean()),
                train_mse=float(g["train_mse"].mean()),
            )
        out[int(s)] = r
    return out


def summarize_decision():
    out = {}
    for (_arm, S, _n), v in sorted(CACHE.items(), key=lambda kv: kv[0][1]):
        out[int(S)] = {k: v[k] for k in v if k not in ("ll_curve", "acc_curve", "lams")}
    return out


def main():
    global SEASON_MAP
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    selfcheck()
    features = pd.read_parquet(u3.FEATURES)
    keys = pd.util.hash_pandas_object(features[list(FULL)], index=False)
    sm = pd.DataFrame(
        {
            "season": features["season"].astype(int).to_numpy(),
            "result": pd.to_numeric(features["result"], errors="coerce").to_numpy(),
        },
        index=keys.to_numpy(),
    )
    dup = sm.index.duplicated(keep=False)
    if int(sm[dup].groupby(level=0).nunique(dropna=False).gt(1).any(axis=1).sum()):
        raise SystemExit("season key conflict")
    SEASON_MAP = sm[~sm.index.duplicated(keep="first")]
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    pop = HARNESS["proxy_population"](features)
    results, chosen = {}, {}
    arms = sys.argv[1:] or ["L", "C"]
    for tag in arms:
        LOG.clear()
        CACHE.clear()
        install(tag)
        try:
            late = HARNESS["true_eval"](features, tuple(FULL), BASE_ALPHA)
            late = late.loc[late["season"].ge(2020)]
            early = HARNESS["proxy_eval"](
                features, pop.loc[pop["season"].le(2019)], lambda s: (FULL, BASE_ALPHA)
            )
        finally:
            margin_mod.make_margin_estimator = ORIG
        run = pd.concat([early, late])
        run.to_parquet(OUT / f"{tag}_run.parquet")
        results[tag] = compare(run.loc[run.index.isin(base.index)], base)
        pd.DataFrame(
            [{k: v for k, v in r.items() if k not in ("curve", "lams")} for r in LOG]
        ).to_csv(OUT / f"{tag}_fits.csv", index=False)
        if tag in ("L", "C"):
            (OUT / f"{tag}_curves.json").write_text(json.dumps(list(CACHE.items()), default=str))
            chosen[tag] = summarize_decision()
        if tag in ("A", "P"):
            (OUT / f"{tag}_curves.json").write_text(
                json.dumps([r for r in LOG if "curve" in r], default=str)
            )
            chosen[tag] = summarize_log(tag)
        for era, c in results[tag].items():
            print(
                tag,
                era,
                c["record"],
                c["record_base"],
                c["d_acc"],
                c["d_acc_season_boot"],
                c["flips_W_L"],
                c["ll_arm"],
                c["ll_base"],
                c["ll_market"],
                c["brier_arm"],
                c["brier_base"],
                c["brier_market"],
                flush=True,
            )
            print(
                tag,
                era,
                "per_season",
                {k: round(v["d_acc"], 2) for k, v in c["per_season"].items()},
                flush=True,
            )
        print(tag, "chosen", json.dumps(chosen.get(tag)), flush=True)
        (OUT / "report.json").write_text(
            json.dumps({"results": results, "chosen": chosen}, indent=1, default=str)
        )
    cells = []
    names = {
        "L": "ridge_penalty_by_inner_loso_cover_log_loss",
        "C": "ridge_penalty_by_inner_loso_pick_accuracy",
        "A": "ridge_extended_grid_inner_loso",
        "P": "reduced_rank_pca_inner_loso",
        "I": "intercept_only_control",
    }
    for tag, r in results.items():
        for era, c in r.items():
            if era.startswith("from_") or era == "pooled":
                continue
            lo, hi = (2011, 2019) if era.startswith("sbr") else (2020, 2025)
            b = c["d_acc_season_boot"]
            cells.append(
                {
                    "source": "artifacts/mod53_unit2/report.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod53_penalty_grid_edge",
                    "category": "control" if tag == "I" else "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": "Base model, same 88 inputs, past-only weekly refits, opener input; selection by inner leave-one-season-out; penalty chosen once per outer season by inner leave-one-season-out over earlier seasons on the discrete opener cover probability; two arms (L cover log loss, C pick accuracy) by two eras, four looks.",
                    "name": f"mod53_unit2_{names[tag]}_{era}_accuracy_diff",
                    "description": f"Base model {names[tag]}, paired vs base ridge alpha 10, {lo}-{hi}; {c['record']} vs {c['record_base']}.",
                    "effect": c["d_acc"],
                    "interval_low": b["low"],
                    "interval_high": b["high"],
                    "probability_positive": b["pp"],
                    "sample_games": c["n"],
                    "sample_blocks": b["blocks"],
                    "plain_summary": "Choosing the base model's shrinkage by the decision metric inside each past-only window moved picks against the opener by an unresolved amount.",
                }
            )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
