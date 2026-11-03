# ruff: noqa: E501
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U8 = Path(__file__).resolve().parent / "mod23_unit8_span.py"
NS = {"__file__": str(U8), "__name__": "mod23_unit8_span_import"}
exec(compile(U8.read_text().rsplit("\nmain()", 1)[0], str(U8), "exec"), NS)

from sklearn.linear_model import ElasticNetCV, RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from nfl_ats import clv  # noqa: E402
from nfl_ats import margin as margin_mod  # noqa: E402

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
compare = NS["compare"]
OUT = u3.ROOT / "artifacts" / "mod23_unit16"
ORIG = margin_mod.make_margin_estimator
FOLDS = KFold(n_splits=5, shuffle=False)
ARMS = {
    "E_elastic_net": lambda: ElasticNetCV(
        l1_ratio=[0.2, 0.5, 0.8], alphas=20, cv=FOLDS, max_iter=5000
    ),
    "R_ridge_cv": lambda: RidgeCV(alphas=np.logspace(-1, 3, 9), cv=FOLDS),
}


def install(factory):
    def make(model_name, random_state=42, **kw):
        base = ORIG(model_name, random_state, **kw)
        if model_name != "ridge":
            return base
        return Pipeline(steps=[*base.steps[:-1], ("regressor", factory())])

    margin_mod.make_margin_estimator = make


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    pop = HARNESS["proxy_population"](features)
    results = {}
    for tag, factory in ARMS.items():
        install(factory)
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
                c["brier_arm"],
                c["brier_base"],
                flush=True,
            )
        (OUT / "report.json").write_text(json.dumps(results, indent=1, default=str))
    cells = []
    for tag, r in results.items():
        for era, c in r.items():
            lo, hi = (2011, 2019) if era.startswith("sbr") else (2020, 2025)
            b = c["d_acc_season_boot"]
            cells.append(
                {
                    "source": "artifacts/mod23_unit16/report.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit16_estimator_class",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": "Same 88 inputs, past-only weekly refits, estimator swapped (elastic net or ridge with inner contiguous-fold selection); two arms by two eras, four looks.",
                    "name": f"mod23_unit16_{tag}_{era}_accuracy_diff",
                    "description": f"Base inputs with {tag}, paired vs base ridge alpha 10, {lo}-{hi}; {c['record']} vs {c['record_base']}.",
                    "effect": c["d_acc"],
                    "interval_low": b["low"],
                    "interval_high": b["high"],
                    "probability_positive": b["pp"],
                    "sample_games": c["n"],
                    "sample_blocks": b["blocks"],
                    "plain_summary": "Swapping the fitting method on the same inputs moved picks against the opener by an unresolved amount.",
                }
            )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
