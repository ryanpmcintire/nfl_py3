# ruff: noqa: E501
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U8 = Path(__file__).resolve().parent / "mod23_unit8.py"
NS = {"__file__": str(U8), "__name__": "mod23_unit8_import"}
exec(compile(U8.read_text().rsplit("\nmain()", 1)[0], str(U8), "exec"), NS)

clv = NS["clv"]
u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
CTX = NS["CTX"]
compare = NS["compare"]
OUT = u3.ROOT / "artifacts" / "mod23_unit15"
ARMS = {"T": ("tot",), "S": ("spr",), "TS": ("tot", "spr")}


def pregame_strata(features):
    reg = features.drop_duplicates("game_id").set_index("game_id")
    tot = pd.to_numeric(reg["total_line"], errors="coerce")
    spr = pd.to_numeric(reg["spread_line"], errors="coerce").abs()
    out = pd.DataFrame(index=reg.index)
    out["tot"] = pd.qcut(tot, 3, labels=False)
    out["spr"] = pd.qcut(spr.rank(method="first"), 3, labels=False)
    return out.astype("float")


def stratum_key(strata, parts):
    s = strata[list(parts)]
    key = pd.Series("", index=s.index)
    for c in parts:
        key = key + "|" + s[c].map(lambda v: "na" if pd.isna(v) else str(int(v)))
    return key.where(s.notna().all(axis=1), "unknown")


NS["stratum_key"] = stratum_key


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(u3.FEATURES)
    strata = pregame_strata(features)
    oof = NS["oof_residuals"](features)
    pop = HARNESS["proxy_population"](features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base true", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    print("strata coverage", strata.notna().sum().to_dict(), flush=True)
    CTX["oof"] = oof
    CTX["strata"] = strata
    NS["install"]()
    tables = {}
    for s in range(2012, 2026):
        t = NS["variance_table"](oof, strata, ("tot", "spr"), s)
        if t is not None:
            tables[s] = {k: round(v, 2) for k, v in t[0].items()}
    results = {}
    for name, parts in ARMS.items():
        path = OUT / f"{name}_run.parquet"
        if path.exists():
            run = pd.read_parquet(path)
        else:
            CTX["mode"] = "weighted"
            CTX["parts"] = parts
            os.environ[clv.CACHE_DISABLED_ENV] = "1"
            run = HARNESS["run_arm"](features, pop, lambda s: (FULL, BASE_ALPHA))
            run.to_parquet(path)
        run = run.loc[run.index.isin(base.index)]
        results[name] = compare(run, base)
        for era, r in results[name].items():
            print(
                name,
                era,
                r["record"],
                "base",
                r["record_base"],
                "d_acc",
                round(r["d_acc"], 3),
                r["d_acc_season_boot"],
                "flips",
                r["flips"],
                r["flips_W_L"],
                flush=True,
            )
    (OUT / "report.json").write_text(
        json.dumps({"results": results, "variance_tables_TS": tables}, indent=1, default=str)
    )
    cells = []
    for name, res in results.items():
        for era, r in res.items():
            lo, hi = (2011, 2019) if era.startswith("sbr") else (2020, 2025)
            cells.append(
                {
                    "source": "artifacts/mod23_unit15/report.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit15_pregame_variance_weights",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": f"Ridge alpha 10 trained with weights of one over the prior-season out-of-fold residual variance of pregame strata (arm {name}: total-line tertile, absolute-spread tertile); three arms by two eras, six looks.",
                    "name": f"mod23_unit15_weights_{name}_{era}_accuracy_diff",
                    "description": f"Base model with pregame-variance training weights ({name}), paired vs base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                    "effect": r["d_acc"],
                    "interval_low": r["d_acc_season_boot"]["low"],
                    "interval_high": r["d_acc_season_boot"]["high"],
                    "probability_positive": r["d_acc_season_boot"]["pp"],
                    "sample_games": r["n"],
                    "sample_blocks": r["d_acc_season_boot"]["blocks"],
                    "plain_summary": "Giving games with noisier expected scoring less weight in training moved picks against the opener by an unresolved amount.",
                }
            )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
