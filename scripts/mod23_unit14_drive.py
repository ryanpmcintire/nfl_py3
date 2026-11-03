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

from nfl_ats import clv  # noqa: E402
from nfl_ats.margin import (  # noqa: E402
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    _target_values,
    make_margin_estimator,
)

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
compare = NS["compare"]
OUT = u3.ROOT / "artifacts" / "mod23_unit14_drive"
ARMS = {"D_drive": "diff_drive_", "P_playquality": "diff_pbp_"}


def add_products(features, prefix):
    cols = tuple(
        c
        for c in features.columns
        if c.startswith(prefix) and not c.startswith("diff_pbp_matchup_")
    )
    f = features.copy()
    for c in cols:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f, cols


def coefficients(f, columns, new):
    prof = HARNESS["PROFILE"]
    supp = tuple(c for c in _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(prof, ()) if c in columns)
    reg = f.loc[f["game_type"].eq("REG") & f["ats_margin"].notna()]
    out = {}
    for season in range(2020, 2026):
        train = reg.loc[reg["season"].lt(season)]
        est = make_margin_estimator(
            "ridge", ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=supp
        )
        est.fit(train.loc[:, list(columns)], _target_values(train, "market_residual"))
        names = list(est[:-1].get_feature_names_out())
        coef = np.ravel(est[-1].coef_)
        out[season] = {
            n: float(c)
            for n, c in zip(names, coef, strict=False)
            if n.split("__", 1)[0] in new or n in new
        }
    return out


def reliability(df):
    bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
    d = df.assign(bin=pd.cut(df["p"], bins, include_lowest=True))
    g = d.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean")
    )
    return {str(k): {c: float(v) for c, v in r.items()} for k, r in g.iterrows()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    results = {}
    for tag, prefix in ARMS.items():
        f, new = add_products(features, prefix)
        print(tag, len(new), flush=True)
        run = HARNESS["true_eval"](f, tuple(FULL) + new, BASE_ALPHA)
        run = run.loc[run["season"].ge(2020)]
        run.to_parquet(OUT / f"{tag}_run.parquet")
        cmp = compare(run.loc[run.index.isin(base.index)], base)["tuesday_2020_2025"]
        d = run.join(base[["y"]], how="left") if "y" not in run else run
        results[tag] = {
            "compare": cmp,
            "reliability": reliability(d),
            "coefficients": coefficients(f, tuple(FULL) + new, new),
            "n_scored": len(run),
        }
        print(
            tag,
            cmp["record"],
            cmp["record_base"],
            cmp["d_acc"],
            cmp["d_acc_season_boot"],
            cmp["flips_W_L"],
            cmp["ll_arm"],
            cmp["ll_base"],
            cmp["brier_arm"],
            cmp["brier_base"],
            flush=True,
        )
    (OUT / "report.json").write_text(json.dumps(results, indent=1, default=str))
    cells = []
    for tag, r in results.items():
        c = r["compare"]
        b = c["d_acc_season_boot"]
        cells.append(
            {
                "source": "artifacts/mod23_unit14_drive/report.json",
                "effect_units": "accuracy_points",
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": 2020,
                "season_end": 2025,
                "family": "mod23_unit14_drive_playquality",
                "category": "modeling",
                "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                "notes": "Base ridge alpha 10 plus a block of drive-efficiency or play-quality differential inputs, leave-one-season-out at the opener; two arms, one era, two looks.",
                "name": f"mod23_unit14_drive_{tag}_accuracy_diff",
                "description": f"Base model plus input block ({tag}), paired vs base, 2020-2025; {c['record']} vs {c['record_base']}.",
                "effect": c["d_acc"],
                "interval_low": b["low"],
                "interval_high": b["high"],
                "probability_positive": b["pp"],
                "sample_games": c["n"],
                "sample_blocks": b["blocks"],
                "plain_summary": "Adding drive-efficiency or play-quality differentials moved picks against the opener by an unresolved amount.",
            }
        )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
