import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PBP = ROOT / "data" / "pbp" / "raw" / "20261103T172043Z"
SCHED = ROOT / "data" / "raw" / "20261103T172043Z" / "schedules.parquet"
OUT = ROOT / "artifacts" / "mod37"
SEED = 20261103
PERMS = 5000
RANDOM_SPLITS = 500
SEASONS = range(2009, 2026)
COLS = [
    "game_id",
    "season",
    "season_type",
    "home_team",
    "away_team",
    "posteam",
    "penalty_team",
    "penalty_type",
    "play_type",
    "play_id",
]


def games():
    sched = pd.read_parquet(SCHED, columns=["game_id", "roof", "stadium_id", "referee"])
    parts = []
    for s in SEASONS:
        d = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=COLS)
        d = d.loc[d["season_type"] == "REG"]
        snap = d["play_type"].isin(["pass", "run", "no_play"])
        fsp = (d["penalty_type"] == "False Start") & (d["penalty_team"] == d["posteam"])
        away = d["posteam"] == d["away_team"]
        snaps = d.loc[snap & away].groupby("game_id").size().rename("snaps")
        fs = (
            d.loc[fsp & away]
            .drop_duplicates(["game_id", "play_id"])
            .groupby("game_id")
            .size()
            .rename("fs")
        )
        allsn = d.loc[snap].groupby("game_id").size().rename("allsnaps")
        allfs = (
            d.loc[fsp]
            .drop_duplicates(["game_id", "play_id"])
            .groupby("game_id")
            .size()
            .rename("allfs")
        )
        g = d.groupby("game_id").agg(
            season=("season", "first"), home=("home_team", "first"), away=("away_team", "first")
        )
        parts.append(g.join([snaps, fs, allsn, allfs]).fillna({"fs": 0, "allfs": 0}))
    g = pd.concat(parts).reset_index().merge(sched, on="game_id", how="left")
    g = g.rename(columns={"stadium_id": "stadium", "referee": "official_name"})
    return g.dropna(subset=["snaps"]).reset_index(drop=True)


def loo_rate(g, keys, num, den):
    sn = g.groupby(keys)[num].transform("sum") - g[num]
    sd = g.groupby(keys)[den].transform("sum") - g[den]
    return sn / sd.where(sd > 0)


def build():
    g = games()
    g["team_rate"] = loo_rate(g.assign(k=g["away"]), ["season", "away"], "fs", "snaps")
    g["crew_rate"] = loo_rate(g, ["season", "official_name"], "allfs", "allsnaps")
    lg = g.groupby("season")["allfs"].transform("sum") / g.groupby("season")["allsnaps"].transform(
        "sum"
    )
    g["crew_ratio"] = (g["crew_rate"] / lg).fillna(1.0)
    g["expected"] = g["snaps"] * g["team_rate"] * g["crew_ratio"]
    g = g.dropna(subset=["expected"]).copy()
    g["excess"] = g["fs"] - g["expected"]
    g["expected_nocrew"] = g["snaps"] * g["team_rate"]
    g["excess_nocrew"] = g["fs"] - g["expected_nocrew"]
    return g


def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    c = (w * (a - ma) * (b - mb)).sum()
    v = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
    return float(c / v) if v > 0 else float("nan")


def half_stat(g, col, half):
    t = g.assign(h=half)
    s = t.groupby(["stadium", "h"])[col].agg(["mean", "size"]).unstack("h")
    s = s.dropna()
    if len(s) < 5:
        return float("nan")
    n0, n1 = s[("size", 0)].to_numpy(float), s[("size", 1)].to_numpy(float)
    return wcorr(s[("mean", 0)].to_numpy(), s[("mean", 1)].to_numpy(), 2 * n0 * n1 / (n0 + n1))


def sb(r):
    return 2 * r / (1 + r)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = build()
    rng = np.random.default_rng(SEED)
    res = {
        "n_games": len(g),
        "n_venues": int(g["stadium"].nunique()),
        "fs_per_game_visitor": float(g["fs"].mean()),
    }
    for col in ("excess", "excess_nocrew"):
        oe = (g["season"] % 2).to_numpy()
        obs = half_stat(g, col, oe)
        null = []
        for _ in range(PERMS):
            p = g.assign(stadium=rng.permutation(g["stadium"].to_numpy()))
            null.append(half_stat(p, col, oe))
        null = np.array(null)
        rs = [half_stat(g, col, rng.integers(0, 2, len(g))) for _ in range(RANDOM_SPLITS)]
        res[col] = {
            "oddeven_r": obs,
            "oddeven_sb": sb(obs),
            "null_mean": float(np.nanmean(null)),
            "null_sd": float(np.nanstd(null)),
            "perm_p_ge": float((null >= obs).mean()),
            "random_split_r_mean": float(np.nanmean(rs)),
            "random_split_sb": sb(float(np.nanmean(rs))),
        }
    rf = g.groupby("stadium").agg(
        mean_excess=("excess", "mean"),
        n=("excess", "size"),
        roof=("roof", lambda s: s.mode().iat[0]),
        home=("home", lambda s: s.mode().iat[0]),
    )
    res["by_roof"] = g.groupby("roof")["excess"].agg(["mean", "size"]).round(4).to_dict("index")
    res["top5"] = rf.sort_values("mean_excess").tail(5).round(3).reset_index().to_dict("records")
    for r in ("dome", "outdoors"):
        sub = g.loc[g["roof"].isin([r, "closed"] if r == "dome" else [r])]
        oe = (sub["season"] % 2).to_numpy()
        res[f"oddeven_{r}"] = half_stat(sub, "excess", oe)
    g.to_parquet(OUT / "games.parquet")
    (OUT / "step12.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
