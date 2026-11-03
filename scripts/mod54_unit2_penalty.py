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

from sklearn.pipeline import Pipeline  # noqa: E402

from nfl_ats import clv  # noqa: E402
from nfl_ats import margin as margin_mod  # noqa: E402

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
compare = NS["compare"]
OUT = u3.ROOT / "artifacts" / "mod54_unit2"
ARMS = {"A": ("M", 2 * BASE_ALPHA), "H": ("B", BASE_ALPHA / 2)}
ORIG = margin_mod.make_margin_estimator
HOME_VIEW = "home_view"
NEGATE = {"spread_line", "rest_diff", "elo_diff"}
SYMMETRIC = {"total_line", "neutral_site", "div_game", "temp", "wind", "week_sin", "week_cos"}
LOG = []
STATE = {"arm": None}


def mirror_plan(cols):
    plan = {}
    for c in cols:
        if c in NEGATE or c.startswith("diff_") or c.endswith("_diff"):
            plan[c] = ("neg", c)
        elif c in SYMMETRIC:
            plan[c] = ("same", c)
        elif c == "elo_home_win_prob":
            plan[c] = ("complement", c)
        elif c.startswith("home_"):
            plan[c] = ("swap", "away_" + c[5:])
        elif c.startswith("away_"):
            plan[c] = ("swap", "home_" + c[5:])
        elif c.endswith("_home"):
            plan[c] = ("swap", c[:-5] + "_away")
        elif c.endswith("_away"):
            plan[c] = ("swap", c[:-5] + "_home")
        else:
            raise SystemExit("unclassified column " + c)
    for kind, src in plan.values():
        if kind == "swap" and src not in cols:
            raise SystemExit("missing partner " + src)
    return plan


PLAN = mirror_plan(list(FULL))


def mirror(X):
    out = pd.DataFrame(index=X.index)
    for c, (kind, src) in PLAN.items():
        v = pd.to_numeric(X[src], errors="coerce")
        if kind == "neg":
            v = -v
        elif kind == "complement":
            v = 1.0 - v
        out[c] = v.to_numpy()
    return out


class MirrorPipe(Pipeline):
    def fit(self, X, y=None, **kw):
        y = np.asarray(y, dtype=float)
        X = X.reset_index(drop=True)
        if STATE["arm"] == "M":
            Xm = mirror(X)
            Xa = pd.concat(
                [X.assign(**{HOME_VIEW: 1.0}), Xm.assign(**{HOME_VIEW: -1.0})],
                ignore_index=True,
            )
            super().fit(Xa, np.concatenate([y, -y]), **kw)
        else:
            super().fit(X, y, **kw)
        pred = self.predict(X)
        nz = y != 0
        LOG.append(
            {
                "n": len(y),
                "train_mse": float(np.mean((y - pred) ** 2)),
                "train_acc": float(np.mean((pred[nz] > 0) == (y[nz] > 0))),
            }
        )
        return self

    def predict(self, X, **kw):
        X = X.reset_index(drop=True)
        if STATE["arm"] == "M":
            X = X.assign(**{HOME_VIEW: 1.0})
        return Pipeline.predict(self, X, **kw)


class MeanPipe(Pipeline):
    def fit(self, X, y=None, **kw):
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X, **kw):
        return np.full(len(X), self.mean_)


def install(arm):
    STATE["arm"] = arm

    def make(model_name, random_state=42, **kw):
        base = ORIG(model_name, random_state, **kw)
        if model_name != "ridge":
            return base
        if arm == "I":
            return MeanPipe(steps=[("m", "passthrough")])
        return MirrorPipe(steps=list(base.steps))

    margin_mod.make_margin_estimator = make


def identity_check(features):
    rng = np.random.default_rng(0)
    n = 400
    cols = ["diff_a", "diff_b", "rest_diff", "spread_line"]
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=cols)
    y = (X["diff_a"] * 2 - X["spread_line"] + rng.normal(size=n)).to_numpy()
    Xm = -X
    est = ORIG("ridge", 42, ridge_alpha=BASE_ALPHA).fit(X, y)
    p0 = est.predict(X)
    q0 = est.predict(Xm)
    Xa = pd.concat([X.assign(h=1.0), Xm.assign(h=-1.0)], ignore_index=True)
    est2 = ORIG("ridge", 42, ridge_alpha=BASE_ALPHA).fit(Xa, np.concatenate([y, -y]))
    p1 = est2.predict(X.assign(h=1.0))
    syn = {
        "pred_max_abs_diff_mirrored_vs_base": float(np.max(np.abs(p0 - p1))),
        "base_max_abs_antisym_violation": float(np.max(np.abs(p0 + q0))),
        "ridge_alpha_ratio_note": "mirroring doubles rows so the same alpha shrinks half as much",
    }
    tr = features.loc[features["season"].le(2019) & features["ats_margin"].notna()]
    Xt = tr.loc[:, list(FULL)].reset_index(drop=True)
    yt = pd.to_numeric(tr["ats_margin"]).to_numpy()
    e0 = ORIG("ridge", 42, ridge_alpha=BASE_ALPHA).fit(Xt, yt)
    pa = e0.predict(Xt)
    pb = e0.predict(mirror(Xt))
    real = {
        "n": len(Xt),
        "mean_pred_plus_mirror_pred": float(np.mean(pa + pb)),
        "sd_pred_plus_mirror_pred": float(np.std(pa + pb)),
        "sd_pred": float(np.std(pa)),
        "mean_abs_antisym_violation": float(np.mean(np.abs(pa + pb))),
        "mean_abs_pred": float(np.mean(np.abs(pa))),
        "flip_rate_sign_pred_vs_neg_mirror": float(np.mean((pa > 0) != (pb < 0))),
    }
    return {"synthetic": syn, "real_base_fit_through_2019": real}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    ident = identity_check(features)
    print("identity", json.dumps(ident), flush=True)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    pop = HARNESS["proxy_population"](features)
    results = {}
    fitlog = {}
    arms = sys.argv[1:] or ["A", "H"]
    for tag in arms:
        LOG.clear()
        install(ARMS[tag][0])
        alpha = ARMS[tag][1]
        try:
            late = HARNESS["true_eval"](features, tuple(FULL), alpha)
            late = late.loc[late["season"].ge(2020)]
            early = HARNESS["proxy_eval"](
                features, pop.loc[pop["season"].le(2019)], lambda s, a=alpha: (FULL, a)
            )
        finally:
            margin_mod.make_margin_estimator = ORIG
        run = pd.concat([early, late])
        run.to_parquet(OUT / f"{tag}_run.parquet")
        results[tag] = compare(run.loc[run.index.isin(base.index)], base)
        lg = pd.DataFrame(LOG)
        fitlog[tag] = (
            {
                "fits": len(lg),
                "train_mse": float(lg["train_mse"].mean()),
                "train_acc": float(lg["train_acc"].mean()),
            }
            if len(lg)
            else {}
        )
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
        print(tag, "fitlog", fitlog[tag], flush=True)
        (OUT / "report.json").write_text(
            json.dumps(
                {"identity": ident, "results": results, "fitlog": fitlog}, indent=1, default=str
            )
        )
    cells = []
    names = {"A": "mirrored_rows_alpha_doubled", "H": "base_alpha_halved"}
    for tag, r in results.items():
        if False:
            continue
        for era, c in r.items():
            if era.startswith("from_") or era == "pooled":
                continue
            lo, hi = (2011, 2019) if era.startswith("proxy") else (2020, 2025)
            b = c["d_acc_season_boot"]
            cells.append(
                {
                    "source": "artifacts/mod54_unit2/report.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod54_mirrored_rows",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": "Base model, same 90 inputs, past-only weekly refits, opener input; mirrored rows with a home-view column versus base ridge alpha 10; arms A mirrored with alpha 20 (row count doubled) and H base with alpha 5, by two eras, four looks.",
                    "name": f"mod54_unit2_{names[tag]}_{era}_accuracy_diff",
                    "description": f"Base model {names[tag]}, paired vs base ridge alpha 10, {lo}-{hi}; {c['record']} vs {c['record_base']}.",
                    "effect": c["d_acc"],
                    "interval_low": b["low"],
                    "interval_high": b["high"],
                    "probability_positive": b["pp"],
                    "sample_games": c["n"],
                    "sample_blocks": b["blocks"],
                    "plain_summary": "Separating row mirroring from penalty strength moved picks against the opener by an unresolved amount.",
                }
            )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
