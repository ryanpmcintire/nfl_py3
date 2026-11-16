from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hindsight_audit_team as hat
import v4_candidate as v4
from mod27_fenced_repairs_replay import paired, record

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/opener_parity"
REFERENCE = "QW"
CONSTANT = "opener_parity_const"
ARMS = ("OP", "OPr", "OPi")


def rebase(table, spreads, restrict):
    out = table.copy()
    hist = out["season"].lt(v4.LIVE_SEASON).to_numpy()
    regular = out["game_type"].eq("REG").to_numpy()
    opener = spreads.reindex(out["game_id"]).to_numpy(float)
    have = hist & ~np.isnan(opener)
    result = pd.to_numeric(out["result"], errors="coerce").to_numpy(float)
    close = out["spread_line"].to_numpy(float).copy()
    spread = close.copy()
    spread[have] = opener[have]
    target = out["ats_margin"].to_numpy(float).copy()
    target[have] = result[have] - opener[have]
    if restrict:
        target[hist & ~have] = np.nan
    out["spread_line"] = spread
    out["ats_margin"] = target
    done = hist & regular & ~np.isnan(result)
    coverage = {
        "completed_history_reg_rows": int(done.sum()),
        "rebased_to_opener": int((done & have).sum()),
        "fell_back_to_close": int((done & ~have).sum()),
        "dropped_from_training": int((done & ~have).sum()) if restrict else 0,
        "spread_changed_rows": int((np.abs(spread - close) > 0)[done].sum()),
        "mean_abs_spread_change": float(np.nanmean(np.abs(spread - close)[done & have])),
        "opener_pushes_in_training": int((target[done] == 0).sum()),
        "close_pushes_in_training": int((table["ats_margin"].to_numpy(float)[done] == 0).sum()),
        "fallback_by_season": {
            str(int(s)): int(n)
            for s, n in out.loc[done & ~have, "season"].value_counts().sort_index().items()
        },
    }
    return out, coverage


def main():
    parser = argparse.ArgumentParser(description="Grade opener-parity training against QW.")
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v4.OUT = OUT
    v4.CACHED = args.cached
    table = pd.read_parquet(v4.TABLE)
    table["game_id"] = table["game_id"].astype(str)
    q_table, _ = v4.qb_repaired(table)
    qw_table, _, _ = v4.forecast_weather(q_table)
    _, spreads = hat.tuesday_quotes(table)
    op_table, op_cov = rebase(qw_table, spreads, restrict=False)
    opr_table, opr_cov = rebase(qw_table, spreads, restrict=True)
    opi_table = op_table.assign(**{CONSTANT: 0.0})
    tables = {
        "S": (table, v4.FULL),
        "QW": (qw_table, v4.FULL),
        "OP": (op_table, v4.FULL),
        "OPr": (opr_table, v4.FULL),
        "OPi": (opi_table, (CONSTANT,)),
    }
    pop = pd.read_parquet(v4.PICK / "per_game.parquet")
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    graded = v4.grade(tables, pop)
    ids = pop["game_id"].to_numpy()
    y = pop["home_covered"].to_numpy(float)
    frame = pop.assign(week_block=pop["season"] * 100 + pop["week"])
    probs = {}
    for name in tables:
        raw = pd.read_parquet(OUT / f"opener_{name}.parquet")
        mp = raw["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
        probs[name], _, _, _ = v4.stage_two(pop, mp)
    ref = probs[REFERENCE]
    for name in ARMS:
        a = probs[name]
        changed = (a >= 0.5) != (ref >= 0.5)
        graded[name]["vs_QW"] = {
            "season_block": paired(frame, ref, a, "season"),
            "week_block": paired(frame, ref, a, "week_block"),
            "changed_picks": int(changed.sum()),
            "changed_record_arm": record(a[changed], y[changed]) if changed.any() else "0-0",
            "changed_record_reference": record(ref[changed], y[changed])
            if changed.any()
            else "0-0",
        }
    graded["coverage"] = {"OP": op_cov, "OPr": opr_cov, "sources": hat.SOURCES}
    (OUT / "results.json").write_text(json.dumps(graded, indent=2, default=str), encoding="utf-8")
    summary = {
        n: {
            "record": graded[n]["record"],
            "model_only_record": graded[n]["model_only_record"],
            "missing_model": graded[n]["missing_model"],
            "log_loss": graded[n]["log_loss"],
            "in_sample_log_loss": graded[n]["in_sample_log_loss"],
            "model_logit_by_fold": {
                k: v.get("model_logit") for k, v in graded[n]["fold_coefficients"].items()
            },
            "vs_QW": graded[n].get("vs_QW"),
        }
        for n in tables
    }
    summary["identity"] = graded["identity"]
    summary["coverage"] = graded["coverage"]
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
