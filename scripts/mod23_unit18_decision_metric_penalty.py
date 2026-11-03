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

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
compare = NS["compare"]
OUT = u3.ROOT / "artifacts" / "mod23_unit18"
ORIG = margin_mod.make_margin_estimator
GRID = tuple(10.0 ** (-1 + 0.5 * i) for i in range(13))
LOG = []
STATE = {"seasons": None, "shared_mask": None, "arm": None}
SEASON_MAP = None


def penalties(mask, a, b):
    lam = np.where(np.append(mask, False), a, b)
    lam[-1] = 0.0
    return lam


class MetricRidge(BaseEstimator, RegressorMixin):
    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float)
        seasons = STATE["seasons"]
        A = np.hstack([Z, np.ones((len(Z), 1))])
        uniq = np.unique(seasons)
        eye = 1e-12 * np.eye(A.shape[1])
        pen = np.ones(A.shape[1])
        pen[-1] = 0.0
        if len(uniq) < 3:
            best = BASE_ALPHA
        else:
            blocks = {s: A[seasons == s] for s in uniq}
            yy = {s: y[seasons == s] for s in uniq}
            G = {s: blocks[s].T @ blocks[s] for s in uniq}
            c = {s: blocks[s].T @ yy[s] for s in uniq}
            Gt = sum(G.values())
            ct = sum(c.values())
            best, best_score = None, np.inf
            for a in GRID:
                lam = np.diag(a * pen)
                score = 0.0
                for s in uniq:
                    w = np.linalg.solve(Gt - G[s] + lam + eye, ct - c[s])
                    p = blocks[s] @ w
                    m = yy[s] != 0
                    if STATE["arm"] == "ACC":
                        score -= float(np.sum(np.sign(p[m]) == np.sign(yy[s][m])))
                    else:
                        tr = np.ones(len(A), dtype=bool)
                        tr[seasons == s] = False
                        e = y[tr] - A[tr] @ w
                        e = e[y[tr] != 0]
                        es = np.sort(e)
                        n = len(es)
                        lo = np.searchsorted(es, -p[m], side="left")
                        hi = np.searchsorted(es, -p[m], side="right")
                        pc = ((n - hi) + 0.5 * (hi - lo) + 1.0) / (n + 2.0)
                        win = yy[s][m] > 0
                        score -= float(np.sum(np.where(win, np.log(pc), np.log(1.0 - pc))))
                if score < best_score:
                    best, best_score = a, score
        w = np.linalg.solve(A.T @ A + np.diag(best * pen) + eye, A.T @ y)
        self.coef_ = w[:-1]
        self.intercept_ = w[-1]
        LOG.append((int(seasons.max()), len(y), best, best))
        return self

    def predict(self, Z):
        return np.asarray(Z, dtype=float) @ self.coef_ + self.intercept_


class MetricPipe(Pipeline):
    def fit(self, X, y=None, **kw):
        cols = list(X.columns)
        keys = pd.util.hash_pandas_object(X[cols], index=False).to_numpy()
        seasons = SEASON_MAP.reindex(keys).to_numpy()
        if np.isnan(seasons).any():
            raise RuntimeError("season lookup failed")
        STATE["seasons"] = seasons.astype(int)
        return super().fit(X, y, **kw)


def install(arm):
    STATE["arm"] = arm

    def make(model_name, random_state=42, **kw):
        base = ORIG(model_name, random_state, **kw)
        if model_name != "ridge":
            return base
        return MetricPipe(steps=[*base.steps[:-1], ("regressor", MetricRidge())])

    margin_mod.make_margin_estimator = make


def selfcheck():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 6))
    y = Z[:, 0] + rng.normal(size=300)
    A = np.hstack([Z, np.ones((300, 1))])
    lam = np.full(7, 10.0)
    lam[-1] = 0.0
    w = np.linalg.solve(A.T @ A + np.diag(lam), A.T @ y)
    r = Ridge(alpha=10.0).fit(Z, y)
    ok = np.allclose(w[:-1], r.coef_, atol=1e-8) and abs(w[-1] - r.intercept_) < 1e-8
    print("closed_form_equals_sklearn_ridge", bool(ok), flush=True)
    if not ok:
        raise SystemExit("selfcheck failed")


def main():
    global SEASON_MAP
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    selfcheck()
    features = pd.read_parquet(u3.FEATURES)
    keys = pd.util.hash_pandas_object(features[list(FULL)], index=False)
    sm = pd.Series(features["season"].astype(int).to_numpy(), index=keys.to_numpy())
    dup = sm.index.duplicated(keep=False)
    conflict = int(sm[dup].groupby(level=0).nunique().gt(1).sum())
    print("dup_keys", int(dup.sum()), "season_conflicts", conflict, flush=True)
    if conflict:
        raise SystemExit("season key conflict")
    SEASON_MAP = sm[~sm.index.duplicated(keep="first")]
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    pop = HARNESS["proxy_population"](features)
    results = {}
    chosen = {}
    for tag in ("ACC", "LL"):
        LOG.clear()
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
        df = pd.DataFrame(LOG, columns=["season", "n", "a_shared", "a_nfl"])
        chosen[tag] = {
            int(s): {
                "fits": len(g),
                "shared_median": float(g["a_shared"].median()),
                "nfl_median": float(g["a_nfl"].median()),
                "shared_min_max": [float(g["a_shared"].min()), float(g["a_shared"].max())],
                "nfl_min_max": [float(g["a_nfl"].min()), float(g["a_nfl"].max())],
            }
            for s, g in df.groupby("season")
        }
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
        print(tag, "chosen", json.dumps(chosen[tag]), flush=True)
        (OUT / "report.json").write_text(
            json.dumps({"results": results, "chosen": chosen}, indent=1, default=str)
        )
    cells = []
    for tag, r in results.items():
        for era, c in r.items():
            lo, hi = (2011, 2019) if era.startswith("sbr") else (2020, 2025)
            b = c["d_acc_season_boot"]
            arm = "inner_loso_opener_accuracy" if tag == "ACC" else "inner_loso_cover_logloss"
            cells.append(
                {
                    "source": "artifacts/mod23_unit18/report.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit18_decision_metric_penalty",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": "Same 88 inputs, past-only weekly refits, penalties chosen by inner leave-one-season-out; arm ACC picks the penalty by inner leave-one-season-out cover accuracy, arm LL by discrete empirical-residual cover log loss; grid 10^(-1..5) step .5; four looks.",
                    "name": f"mod23_unit18_{arm}_{era}_accuracy_diff",
                    "description": f"{arm} ridge, paired vs base ridge alpha 10, {lo}-{hi}; {c['record']} vs {c['record_base']}.",
                    "effect": c["d_acc"],
                    "interval_low": b["low"],
                    "interval_high": b["high"],
                    "probability_positive": b["pp"],
                    "sample_games": c["n"],
                    "sample_blocks": b["blocks"],
                    "plain_summary": "Choosing the shrinkage strength inside each past-only window moved picks against the opener by an unresolved amount.",
                }
            )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
