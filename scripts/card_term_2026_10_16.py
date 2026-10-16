from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, loso, metrics, reliability, summary  # noqa: F401
from nfl_ats.clv import load_decision_quotes
from nfl_ats.pick_probability import FLAG_SUM_COLUMN, MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.sharp_book_movement_features import sharp_book_movement_features
from per_flag_weights_eval import ARTIFACT, SEASONS, SEED, bootstrap, prepare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/scratch/card_term_2026_10_16"
TERM = "total_move_margin_shift"
VARIANTS = {
    "four_term": BASE,
    "five_term": (*BASE, TERM),
    "model_only": ("model_logit",),
}


def term_table(data):
    q = load_decision_quotes(ROOT / "data/market/raw", seasons=SEASONS)
    q = q.loc[q["nflverse_game_id"].notna() & q["market"].eq("totals")].copy()
    q["line"] = pd.to_numeric(q["line"], errors="coerce")
    q = q.loc[q["outcome_name"].astype(str).str.lower().eq("over")]
    opens = q.loc[q["decision_label"].eq("tue_open")].groupby("nflverse_game_id")["line"].median()
    kick = q.groupby("nflverse_game_id")["commence_time_utc"].min()
    g = data[["game_id", "season", "week", "tue_open_home_spread"]].copy()
    g["commence_time_utc"] = pd.to_datetime(g["game_id"].map(kick), utc=True)
    g["week_first_commence_utc"] = g.groupby(["season", "week"])["commence_time_utc"].transform("min")
    g = g.dropna(subset=["commence_time_utc"])
    qq = q.assign(market="spreads", home_spread_line=q["line"])
    exp = sharp_book_movement_features(qq, g[["game_id", "commence_time_utc", "week_first_commence_utc"]])
    out = g.merge(exp[["game_id", "leader_median_net_move", "leader_books"]], on="game_id")
    out["total_open"] = out["game_id"].map(opens)
    seen = out["leader_books"].gt(0) & out["total_open"].notna()
    out[TERM] = np.where(
        seen, out["leader_median_net_move"] * (-out["tue_open_home_spread"] / out["total_open"]), np.nan
    )
    return out[["game_id", TERM, "leader_median_net_move", "total_open"]]


def main():
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data.season.between(2020, 2025)].reset_index(drop=True)
    t = term_table(data)
    data = data.merge(t, on="game_id", how="left")
    y = data["home_covered"].to_numpy(float)
    seen = data[TERM].notna()
    cov = {
        "games": int(len(data)),
        "term_observed": int(seen.sum()),
        "total_moved_nonzero": int(data["leader_median_net_move"].fillna(0).ne(0).sum()),
        "by_season": {str(s): [int(g[TERM].notna().sum()), int(len(g))] for s, g in data.groupby("season")},
        "term_sd": float(data[TERM].std()),
        "corr_with_spread_move": float(data[TERM].corr(data[MOVE_COLUMN])),
    }
    rng = np.random.default_rng(SEED)
    result = {"coverage": cov, "views": {}}
    for view in ("tuesday", "full"):
        frame = prepare(data, view)
        if view == "tuesday":
            frame[TERM] = 0.0
        probs, entry = {}, {}
        for name, feats in VARIANTS.items():
            oos, folds = loso(frame, feats)
            probs[name] = oos
            entry[name] = {
                "oos": metrics(oos, y),
                "reliability": reliability(oos, y),
                "fold_coefficients": folds,
                "fold_summary": summary(folds),
            }
        entry["market"] = {"oos": metrics(np.full(len(y), 0.5), y)}
        new, old = probs["five_term"], probs["four_term"]
        entry["paired_five_minus_four"] = bootstrap(data, new, old, rng)
        diff = (new >= 0.5) != (old >= 0.5)
        entry["decisive"] = {
            "changed": int(diff.sum()),
            "five_wins": int(((new >= 0.5) == (y == 1))[diff].sum()),
            "four_wins": int(((old >= 0.5) == (y == 1))[diff].sum()),
        }
        result["views"][view] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=1, default=str)
    (OUT / "results.json").write_text(text, encoding="utf-8")
    print(json.dumps({"coverage": cov, **{v: {k: result["views"][v][k] for k in ("decisive","paired_five_minus_four")} | {n: result["views"][v][n]["oos"] for n in ("four_term","five_term","model_only","market")} | {"fold": result["views"][v]["five_term"]["fold_summary"]} for v in result["views"]}}, indent=1, default=str))


if __name__ == "__main__":
    main()
