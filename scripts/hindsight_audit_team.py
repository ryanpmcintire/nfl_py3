from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_candidate as v4  # noqa: E402
from mod27_fenced_repairs_replay import paired, record  # noqa: E402

from nfl_ats import clv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/hindsight_audit_team"
FAMILY = "hindsight_audit_team"
REFERENCE = "QW"
LIVE_SEASON = v4.LIVE_SEASON
SBR = ROOT / "data/processed/sbr_odds.parquet"
SOURCES = {}


def tuesday_quotes(table):
    pairing = clv.cached_pairing_table(
        v4.m22.MARKET_ROOT,
        labels=("tue_open",),
        schedule=table,
    )
    tue = pairing.loc[pairing["decision_label"].eq("tue_open")]
    tue = tue.drop_duplicates("game_id").set_index("game_id")
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id"]).drop_duplicates("game_id")
    sbr["game_id"] = sbr["game_id"].astype(str)
    sbr = sbr.set_index("game_id")
    close = table.set_index("game_id")["spread_line"]
    both = sbr["close_home_spread"].dropna().index.intersection(close.dropna().index)
    sign = float(np.sign(np.corrcoef(sbr.loc[both, "close_home_spread"], close.loc[both])[0, 1]))
    total = tue["total_line"].astype(float).dropna()
    spread = tue["home_spread"].astype(float).dropna()
    total = total.combine_first(sbr["open_total"].astype(float).dropna())
    spread = spread.combine_first(sign * sbr["open_home_spread"].astype(float).dropna())
    SOURCES.update(
        {
            "sbr_sign": sign,
            "tuesday_totals": int(tue["total_line"].notna().sum()),
            "sbr_totals": int(sbr["open_total"].notna().sum()),
        }
    )
    return total, spread


def as_of(table, column, values):
    out = table.copy()
    hist = out["season"].lt(LIVE_SEASON).to_numpy()
    new = values.reindex(out["game_id"]).to_numpy(float)
    have = hist & ~np.isnan(new)
    old = out[column].to_numpy(float).copy()
    out.loc[have, column] = new[have]
    regular = out["game_type"].eq("REG").to_numpy() if "game_type" in out else np.ones(len(out), bool)
    coverage = {
        "history_rows": int((hist & regular).sum()),
        "replaced": int((have & regular).sum()),
        "kept_close": int((hist & regular & ~have).sum()),
        "mean_abs_change": float(np.nanmean(np.abs(new[have] - old[have]))) if have.any() else 0.0,
        "changed_rows": int((np.abs(new[have] - old[have]) > 0).sum()),
    }
    return out, coverage


def main():
    parser = argparse.ArgumentParser(description="Grade as-of-deadline market columns against QW.")
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v4.OUT = OUT
    v4.CACHED = args.cached
    table = pd.read_parquet(v4.TABLE)
    table["game_id"] = table["game_id"].astype(str)
    q_table, _ = v4.qb_repaired(table)
    qw_table, _, _ = v4.forecast_weather(q_table)
    totals, spreads = tuesday_quotes(table)
    t_table, t_cov = as_of(qw_table, "total_line", totals)
    p_table, p_cov = as_of(qw_table, "spread_line", spreads)
    tp_table, _ = as_of(t_table, "spread_line", spreads)
    tables = {
        "S": (table, v4.FULL),
        "QW": (qw_table, v4.FULL),
        "T": (t_table, v4.FULL),
        "P": (p_table, v4.FULL),
        "TP": (tp_table, v4.FULL),
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
    for name in ("T", "P", "TP"):
        a = probs[name]
        changed = (a >= 0.5) != (ref >= 0.5)
        graded[name]["vs_QW"] = {
            "season_block": paired(frame, ref, a, "season"),
            "week_block": paired(frame, ref, a, "week_block"),
            "changed_picks": int(changed.sum()),
            "changed_record_arm": record(a[changed], y[changed]) if changed.any() else "0-0",
            "changed_record_reference": record(ref[changed], y[changed]) if changed.any() else "0-0",
            "changed_games": pop.loc[changed, "game_id"].tolist(),
        }
    graded["coverage"] = {"total_line": t_cov, "spread_line": p_cov, "sources": SOURCES}
    (OUT / "results.json").write_text(json.dumps(graded, indent=2, default=str), encoding="utf-8")
    summary = {
        n: {
            "record": graded[n]["record"],
            "log_loss": graded[n]["log_loss"],
            "vs_QW": {
                k: v for k, v in graded[n].get("vs_QW", {}).items() if k != "changed_games"
            },
        }
        for n in tables
    }
    summary["identity"] = graded["identity"]
    summary["coverage"] = graded["coverage"]
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
