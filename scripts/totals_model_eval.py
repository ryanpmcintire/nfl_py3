from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.joint_residual_model import (
    realised_residual_frame,
    walk_forward_joint_predictions,
)
from nfl_ats.served_total import BLEND_K01_WEIGHT, JOINT_TOTAL_BLEND_WEIGHT
from nfl_ats.tiebreaker import TOTAL_LOW_SIDE_SHADE_POINTS

REPO = Path(__file__).resolve().parents[1]
WAVE1 = REPO / "artifacts/totals_backtest/20260901T184010Z/predictions.parquet"
WAVE2 = REPO / "artifacts/totals_backtest_wave2/wp18run/screen/wave2_predictions.parquet"
FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
OUT = REPO / "tests/scratch/totals_model_eval"
FIRST_SEASON = 2013
LAST_SEASON = 2025
WEIGHTS = tuple(round(0.1 * i, 1) for i in range(11))
SAMPLES = 2000


def rnd(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def build():
    feats = pd.read_parquet(FEATURES)
    pop = realised_residual_frame(feats)
    joint = walk_forward_joint_predictions(pop)[["game_id", "predicted_total_residual"]].rename(
        columns={"predicted_total_residual": "res_joint"}
    )
    w1 = pd.read_parquet(WAVE1)[["game_id", "predicted_residual"]].rename(
        columns={"predicted_residual": "res_w1"}
    )
    w2 = pd.read_parquet(WAVE2)[["game_id", "predicted_residual"]].rename(
        columns={"predicted_residual": "res_w2"}
    )
    sched = pd.read_parquet(sorted(glob.glob(str(REPO / "data/raw/*/schedules.parquet")))[-1])
    base = pop[["game_id", "season", "week", "gameday", "market_total", "actual_total"]]
    d = base.merge(joint, on="game_id").merge(w1, on="game_id").merge(w2, on="game_id")
    d = d.merge(sched[["game_id", "gametime"]], on="game_id", how="left")
    d = d.loc[d["season"].between(FIRST_SEASON, LAST_SEASON)].dropna().reset_index(drop=True)
    d["key"] = d["gameday"].astype(str) + " " + d["gametime"].astype(str)
    last = d.sort_values("key").groupby(["season", "week"]).tail(1).index
    d["last"] = d.index.isin(last)
    return d


def loso_k(d, col, grid):
    out = np.empty(len(d))
    chosen = {}
    for s in sorted(d["season"].unique()):
        tr = d["season"] != s
        best, arg = 1e18, 0.0
        for k in grid:
            v = float(
                np.mean(
                    np.abs(
                        d.loc[tr, "market_total"] + k * d.loc[tr, col] - d.loc[tr, "actual_total"]
                    )
                )
            )
            if v < best:
                best, arg = v, k
        te = d["season"] == s
        out[te.to_numpy()] = (d.loc[te, "market_total"] + arg * d.loc[te, col]).to_numpy()
        chosen[int(s)] = arg
    return out, chosen


def paired(d, served_err, arm_err, label, mask):
    f = d.loc[mask, ["season", "week"]].copy()
    f["diff"] = (served_err - arm_err)[mask.to_numpy()]
    r = week_blocked_bootstrap(
        f,
        lambda x: {"mae_improvement": float(x["diff"].mean())},
        samples=SAMPLES,
        metric_columns=["season", "week", "diff"],
    ).iloc[0]
    seasons = f.assign(season=d.loc[mask, "season"]).groupby("season")["diff"].mean()
    return {
        "arm_vs_served": label,
        "n": int(mask.sum()),
        "diff": float(r["estimate"]),
        "lo": float(r["lower"]),
        "hi": float(r["upper"]),
        "probability_positive": float(r["probability_positive"]),
        "seasons_better": int((seasons > 0).sum()),
        "seasons": len(seasons),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    a = d["actual_total"].to_numpy()
    m = d["market_total"]
    sweep = {
        f"{k}": float(np.mean(np.abs(m + k * d["res_joint"] - d["actual_total"]))) for k in WEIGHTS
    }
    lk, chosen = loso_k(d, "res_joint", WEIGHTS)
    arms = {
        "market": m.to_numpy(),
        "wave1_k0.1": (m + BLEND_K01_WEIGHT * d["res_w1"]).to_numpy(),
        "wave2_k0.1": (m + BLEND_K01_WEIGHT * d["res_w2"]).to_numpy(),
        "joint_k0.1": (m + JOINT_TOTAL_BLEND_WEIGHT * d["res_joint"]).to_numpy(),
        "joint_loso_k": lk,
        "market_shade": (m + TOTAL_LOW_SIDE_SHADE_POINTS).to_numpy(),
        "wave2_k0.1_shade": (
            m + BLEND_K01_WEIGHT * d["res_w2"] + TOTAL_LOW_SIDE_SHADE_POINTS
        ).to_numpy(),
        "served_joint_k0.1_shade": (
            m + JOINT_TOTAL_BLEND_WEIGHT * d["res_joint"] + TOTAL_LOW_SIDE_SHADE_POINTS
        ).to_numpy(),
    }
    served = "served_joint_k0.1_shade"
    result = {"joint_k_sweep_mae_all": sweep, "loso_k_chosen": chosen, "mae": {}, "paired": []}
    masks = {"all_continuous": pd.Series(True, index=d.index), "last_game_integer": d["last"]}
    for pop_name, mask in masks.items():
        integer = pop_name == "last_game_integer"
        errs = {k: np.abs((rnd(v) if integer else v) - a) for k, v in arms.items()}
        result["mae"][pop_name] = {k: float(e[mask.to_numpy()].mean()) for k, e in errs.items()}
        for k, e in errs.items():
            if k != served:
                result["paired"].append(
                    {"population": pop_name, **paired(d, errs[served], e, k, mask)}
                )
    result["looks"] = len(result["paired"]) + len(WEIGHTS)
    result["games"] = len(d)
    result["last_games"] = int(d["last"].sum())
    (OUT / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    d.to_parquet(OUT / "rows.parquet")
    json.dump(result, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
