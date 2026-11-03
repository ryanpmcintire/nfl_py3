import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import pooled_screen as ps  # noqa: E402
import xlg14_mod41_cfb as u1  # noqa: E402

PER_GAME = REPO / "artifacts" / "opener_evaluation" / "20260929T192743Z" / "per_game.parquet"
NFL_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
OUT_DIR = REPO / "artifacts" / "xlg14"
WINDOW = u1.WINDOW
WEEK_DAYS = 7
BYE_DAYS = 13
NAMES = ["short_week", "bye", "prior_ats", "early_carryover", "road_trip"]
HYPOTHESIS = {
    "short_week": -1,
    "bye": 1,
    "prior_ats": -1,
    "early_carryover": 1,
    "road_trip": 1,
}


def family(g, home_key, away_key, open_col):
    g = g.copy()
    g["day"] = pd.to_datetime(g["gameday"])
    rows = []
    for side, key, sgn in (
        ("home", home_key, 1.0),
        ("away", away_key, -1.0),
    ):
        r = g[["game_id", "season", "day", "result", "neutral_site", open_col]].copy()
        r["team"] = g[key].to_numpy()
        r["side"] = side
        r["margin"] = sgn * g["result"].to_numpy(float)
        r["team_open"] = sgn * g[open_col].to_numpy(float)
        r["away"] = (side == "away") & ~g["neutral_site"].fillna(False).astype(bool).to_numpy()
        rows.append(r)
    t = pd.concat(rows, ignore_index=True)
    t = t.sort_values(["team", "season", "day", "game_id"]).reset_index(drop=True)
    grp = t.groupby(["team", "season"])
    t["rest"] = (t["day"] - grp["day"].shift(1)).dt.days
    t["short"] = (t["rest"] < WEEK_DAYS).astype(float).where(t["rest"].notna(), 0.0)
    t["bye"] = (t["rest"] >= BYE_DAYS).astype(float)
    t["ats_sign"] = np.sign(t["margin"] - t["team_open"])
    t["prior_ats"] = grp["ats_sign"].shift(1).fillna(0.0)
    t["prev_away"] = grp["away"].shift(1).astype(float).fillna(0.0)
    t["n_prior"] = grp.cumcount()
    sm = t.groupby(["team", "season"])["margin"].mean().rename("smean").reset_index()
    sm["season"] = sm["season"] + 1
    t = t.merge(sm, on=["team", "season"], how="left")
    t["carry"] = t["smean"].fillna(0.0)
    wide = {}
    for side in ("home", "away"):
        s = t[t["side"].eq(side)].set_index("game_id")
        wide[side] = s
    h, a = wide["home"], wide["away"]
    out = pd.DataFrame(index=h.index)
    out["short_week"] = h["short"] - a["short"].reindex(h.index)
    out["bye"] = h["bye"] - a["bye"].reindex(h.index)
    out["prior_ats"] = h["prior_ats"] - a["prior_ats"].reindex(h.index)
    early = (h["n_prior"] < WINDOW) & (a["n_prior"].reindex(h.index) < WINDOW)
    out["early_carryover"] = (h["carry"] - a["carry"].reindex(h.index)).where(early, 0.0)
    out["road_trip"] = a["prev_away"].reindex(h.index) - h["prev_away"]
    return out.reset_index()


def cfb_frame():
    full = pd.read_parquet(u1.FEATURES)
    fam = family(full, "home_id", "away_id", "spread_open")
    d = full[full["season"].isin(u1.SEASONS) & full["spread_open"].notna() & full["result"].notna()]
    d = d[d["result"].ne(d["spread_open"])].reset_index(drop=True)
    d["y"] = (d["result"] > d["spread_open"]).astype(int)
    cols = [c for c in ps.shared_columns() if c not in u1.CLOSE_COLUMNS] + ["spread_open"]
    d["logit_p"] = u1.base_logit(d, cols)
    d = d.merge(fam, on="game_id", how="left")
    d["block"] = d["season"] * 100 + pd.to_numeric(d["week"]).astype(int)
    return d


def nfl_frame():
    o = pd.read_parquet(PER_GAME)
    g = pd.read_parquet(NFL_FEATURES)
    g = g[g["season"].between(2019, 2025)].copy()
    g["neutral_site"] = g["neutral_site"].fillna(False)
    g = g.merge(o[["game_id", "tue_open_home_spread"]], on="game_id", how="left")
    fam = family(g, "home_team", "away_team", "tue_open_home_spread")
    d = o[o["season"].between(2020, 2025) & o["margin_vs_open"].ne(0)].copy()
    d["y"] = (d["margin_vs_open"] > 0).astype(int)
    d["logit_p"] = logit(d["home_cover_probability_at_open"].clip(1e-6, 1 - 1e-6))
    d = d.merge(fam, on="game_id", how="left")
    d["block"] = d["season"] * 100 + pd.to_numeric(d["week"]).astype(int)
    return d


def main():
    mode = sys.argv[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if mode == "cfb":
        d = cfb_frame()
        names = NAMES
    else:
        d = nfl_frame()
        names = sys.argv[2:]
    for n in names:
        d[n] = d[n].fillna(0.0)
    out = {"mode": mode, "games": len(d), "nonzero": {n: int(d[n].ne(0).sum()) for n in names}}
    for n in names:
        o = u1.analyze(d, ["logit_p", n], n)
        co = [v[-1] for v in o["coef_per_heldout_season"].values()]
        o["folds_hypothesis_sign"] = int(sum(np.sign(c) == HYPOTHESIS[n] for c in co))
        o["folds"] = len(co)
        out[n] = o
        print(
            n,
            "nonzero",
            out["nonzero"][n],
            "decisive",
            o["decisive_pick_changed_oos"],
            o["decisive_term_correct"],
            o["decisive_base_correct"],
            "ll",
            o["oos_log_loss_gain"],
            "coef",
            o["coef_full"][-1],
            "foldsign",
            o["folds_hypothesis_sign"],
            "/",
            o["folds"],
            "gap",
            o["in_minus_out_log_loss_gain"],
            flush=True,
        )
    out["intercept_only_oos"] = o["intercept_only_oos"]
    out["baseline_oos"] = o["baseline_oos"]
    (OUT_DIR / f"family_{mode}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
