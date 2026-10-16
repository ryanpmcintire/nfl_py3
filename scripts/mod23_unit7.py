import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats.constants import FEATURE_SETS

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod23_unit7"
BASE = ROOT / "artifacts" / "opener_evaluation" / "20260929T192743Z" / "per_game.parquet"
SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
BOOT = 20000
SEED = 20260817


def openers():
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id", "open_home_spread"])
    sbr = sbr.drop_duplicates("game_id")
    sbr = sbr.loc[sbr["season"].le(2019), ["game_id", "open_home_spread"]]
    base = pd.read_parquet(BASE)[["game_id", "tue_open_home_spread"]]
    base = base.rename(columns={"tue_open_home_spread": "open_home_spread"})
    return pd.concat([sbr, base], ignore_index=True).drop_duplicates("game_id")


def finish(df):
    df = df.copy()
    p = df["p"].clip(1e-6, 1 - 1e-6)
    m = df["margin_vs_open"]
    pick = p >= 0.5
    df["correct"] = np.where(m.eq(0), np.nan, np.where(pick, m.gt(0), ~m.gt(0)))
    df = df.loc[df["correct"].notna()].copy()
    p = df["p"].clip(1e-6, 1 - 1e-6)
    y = (df["margin_vs_open"] > 0).astype(float)
    df["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    df["brier"] = (p - y) ** 2
    return df.set_index("game_id")[["season", "week", "p", "correct", "ll", "brier"]]


def boot(s, col):
    g = s.groupby("season")[col].agg(["sum", "count"])
    sums = g["sum"].to_numpy()
    sizes = g["count"].to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(sums), (BOOT, len(sums)))
    d = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())]


def compare(arm, base):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["d_ll"] = j["ll"] - j["ll_b"]
    j["d_brier"] = j["brier"] - j["brier_b"]
    wins = int(j["correct"].sum())
    out = {
        "n": len(j),
        "record": f"{wins}-{len(j) - wins}",
        "acc": float(j["correct"].mean()),
        "acc_base": float(j["correct_b"].mean()),
        "d_acc": float(j["d_acc"].mean()),
        "d_acc_ci_pp": boot(j, "d_acc"),
        "ll": float(j["ll"].mean()),
        "ll_base": float(j["ll_b"].mean()),
        "d_ll_ci_pp": boot(j, "d_ll"),
        "brier": float(j["brier"].mean()),
        "brier_base": float(j["brier_b"].mean()),
        "d_brier_ci_pp": boot(j, "d_brier"),
        "per_season": {
            int(s): {
                "n": len(g),
                "record": f"{int(g['correct'].sum())}-{len(g) - int(g['correct'].sum())}",
                "d_acc": float(g["d_acc"].mean()),
                "d_ll": float(g["d_ll"].mean()),
            }
            for s, g in j.groupby("season")
        },
    }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(u3.FEATURES)
    base_game = pd.read_parquet(BASE)
    base = finish(
        pd.DataFrame(
            {
                "game_id": base_game["game_id"],
                "season": base_game["season"],
                "week": base_game["week"],
                "p": base_game["home_cover_probability_at_open"],
                "margin_vs_open": base_game["margin_vs_open"],
            }
        )
    )
    print("base", int(base["correct"].sum()), len(base), flush=True)
    op = openers().set_index("game_id")["open_home_spread"]
    sub = features.loc[features["game_id"].isin(op.index)].copy()
    print("rows with opener", len(sub), "of", len(features), flush=True)
    arm_frame = sub.copy()
    arm_frame["spread_line"] = arm_frame["game_id"].map(op).to_numpy(dtype=float)
    arm_frame["ats_margin"] = pd.to_numeric(arm_frame["result"]) - arm_frame["spread_line"]
    results = {}
    for name, frame in (("A_open_trained", arm_frame), ("C_close_trained_same_rows", sub)):
        scored = u3.run_arm(frame, list(FEATURE_SETS[u3.SET_NAME]))
        scored = scored.loc[scored["season"].ge(2020)]
        scored["p"] = scored["home_cover_probability_at_open"]
        arm = finish(scored[["game_id", "season", "week", "p", "margin_vs_open"]])
        arm.to_parquet(OUT / f"{name}.parquet")
        print(name, int(arm["correct"].sum()), len(arm), flush=True)
        results[name] = compare(arm, base)
        (OUT / "report.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results[name], indent=1), flush=True)


main()
