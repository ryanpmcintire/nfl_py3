from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("F:/Repos/nfl_py3")
FTN_ROOT = ROOT / "data" / "raw" / "ftn_charting"
PBP = ROOT / "data" / "pbp" / "raw" / "20261027T220410Z"
OUT = ROOT / "artifacts" / "mod10_ftn_traits" / "result.json"
SEASONS = (2022, 2023, 2024, 2025)
GATE = 0.436
BOOT = 2000
SEED = 20261103
MIN_REC_HALF = {"drop_rate": 15, "created_rate": 25, "contested_rate": 25}
MIN_QB_HALF = 100
STAB_MULT = 2


def load():
    stamp = sorted(p for p in FTN_ROOT.iterdir() if p.is_dir())[-1]
    ftn = pd.concat(
        [pd.read_parquet(stamp / f"ftn_charting_{s}.parquet") for s in SEASONS], ignore_index=True
    )
    pbp = pd.concat(
        [
            pd.read_parquet(
                PBP / f"season={s}",
                columns=[
                    "game_id",
                    "play_id",
                    "season",
                    "season_type",
                    "week",
                    "posteam",
                    "passer_player_id",
                    "receiver_player_id",
                    "qb_dropback",
                    "pass_attempt",
                    "sack",
                ],
            )
            for s in SEASONS
        ],
        ignore_index=True,
    )
    pbp = pbp[pbp.season_type == "REG"].drop(columns=["season_type"])
    df = pbp.merge(
        ftn.drop(columns=["season", "week"]),
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "nflverse_play_id"],
        how="inner",
    )
    for c in (
        "is_catchable_ball",
        "is_contested_ball",
        "is_created_reception",
        "is_drop",
        "is_interception_worthy",
        "is_qb_fault_sack",
    ):
        df[c] = df[c].fillna(False).astype(bool)
    df["half"] = np.where(df.week % 2 == 1, "odd", "even")
    return df


def rec_table(df, num_col, den_col):
    t = df[(df.pass_attempt == 1) & df.receiver_player_id.notna()].copy()
    t["den"] = t[den_col].astype(float)
    t["num"] = (t[num_col] & t[den_col].astype(bool)).astype(float)
    g = t.groupby(["season", "half", "posteam", "receiver_player_id"], as_index=False)[
        ["num", "den"]
    ].sum()
    return g.rename(columns={"receiver_player_id": "pid"})


def qb_table(df, num_col, den_kind):
    t = df[(df.qb_dropback == 1) & df.passer_player_id.notna()].copy()
    if den_kind == "dropback":
        t["den"] = 1.0
    else:
        t = t[t.pass_attempt == 1]
        t["den"] = 1.0
    t["num"] = t[num_col].astype(float)
    g = t.groupby(["season", "half", "posteam", "passer_player_id"], as_index=False)[
        ["num", "den"]
    ].sum()
    return g.rename(columns={"passer_player_id": "pid"})


def add_rate(g, demean):
    g = g.copy()
    g["rate"] = g.num / g.den
    if demean:
        tn = g.groupby(["season", "half", "posteam"]).num.transform("sum") - g.num
        td = g.groupby(["season", "half", "posteam"]).den.transform("sum") - g.den
        g["rate"] = g.rate - np.where(td > 0, tn / td.where(td > 0, 1), np.nan)
    return g


def halves(g, minimum):
    w = g.pivot_table(
        index=["season", "pid"], columns="half", values=["rate", "den"], aggfunc="first"
    )
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna()
    w = w[(w.den_odd >= minimum) & (w.den_even >= minimum)]
    return w.reset_index()


def sb(x):
    return 2 * x / (1 + x)


def sp(a, b):
    if len(a) < 5:
        return np.nan
    return spearmanr(a, b).statistic


def boot_corr(frame, a, b, rng, transform):
    pids = frame.pid.unique()
    groups = {p: frame[frame.pid == p] for p in pids}
    pt = transform(sp(frame[a], frame[b]))
    draws = []
    for _ in range(BOOT):
        pick = rng.choice(pids, len(pids), replace=True)
        s = pd.concat([groups[p] for p in pick])
        draws.append(transform(sp(s[a], s[b])))
    d = np.array([x for x in draws if np.isfinite(x)])
    return {
        "point": float(pt),
        "lo": float(np.percentile(d, 2.5)),
        "hi": float(np.percentile(d, 97.5)),
        "p_positive": float((d > 0).mean()),
        "p_above_gate": float((d > GATE).mean()),
        "n": len(frame),
    }


def stability(g, minimum, rng):
    full = g.groupby(["season", "pid"], as_index=False)[["num", "den"]].sum()
    full = full[full.den >= STAB_MULT * minimum]
    full["rate"] = full.num / full.den
    nxt = full.copy()
    nxt["season"] = nxt.season - 1
    m = full.merge(nxt, on=["season", "pid"], suffixes=("_t", "_n"))
    if len(m) < 5:
        return {"n": len(m)}
    return boot_corr(m, "rate_t", "rate_n", rng, lambda x: x)


def main():
    df = load()
    rng = np.random.default_rng(SEED)
    results = []
    specs = []
    for name, (num, den) in {
        "rec_drop_rate": ("is_drop", "is_catchable_ball"),
        "rec_created_reception_rate": ("is_created_reception", "is_catchable_ball"),
        "rec_contested_rate": ("is_contested_ball", "is_catchable_ball"),
    }.items():
        key = {
            "rec_drop_rate": "drop_rate",
            "rec_created_reception_rate": "created_rate",
            "rec_contested_rate": "contested_rate",
        }[name]
        for demean in (False, True):
            specs.append((name, demean, rec_table(df, num, den), MIN_REC_HALF[key]))
    for name, (num, kind) in {
        "qb_interception_worthy_per_dropback": ("is_interception_worthy", "dropback"),
        "qb_fault_sack_per_dropback": ("is_qb_fault_sack", "dropback"),
        "qb_catchable_per_attempt": ("is_catchable_ball", "attempt"),
    }.items():
        specs.append((name, False, qb_table(df, num, kind), MIN_QB_HALF))
    for name, demean, g, minimum in specs:
        g = add_rate(g, demean)
        h = halves(g, minimum)
        row = {"trait": name, "team_context_removed": demean, "min_per_half": minimum}
        for s in SEASONS:
            hs = h[h.season == s]
            row[f"sb_{s}"] = (
                boot_corr(hs, "rate_odd", "rate_even", rng, sb) if len(hs) >= 5 else {"n": len(hs)}
            )
        row["sb_pooled"] = boot_corr(h, "rate_odd", "rate_even", rng, sb)
        row["stability_t_to_t1"] = stability(g.assign(rate=g.num / g.den), minimum, rng)
        results.append(row)
        print(name, demean, round(row["sb_pooled"]["point"], 3), row["sb_pooled"]["n"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"gate": GATE, "looks": len(results) * 6, "results": results}, indent=1)
    )


if __name__ == "__main__":
    main()
