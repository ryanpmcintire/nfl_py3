import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import xlg10_addone as x  # noqa: E402

from nfl_ats.cfb_features import cfb_season_partitions  # noqa: E402

NFL_PBP_ROOT = ROOT / "data" / "pbp" / "raw"
CFB_ROOT = ROOT / "data" / "cfb"
CFB_PASS_TYPES = (
    "Pass Reception",
    "Pass Incompletion",
    "Passing Touchdown",
    "Interception Return",
    "Interception Return Touchdown",
)
CFB_INT_TYPES = ("Interception Return", "Interception Return Touchdown")
COUNTS = ["fum_lost", "fum_all", "ints", "passes"]
ARMS = {
    "fumble_lost_diff": ["fumble_lost_diff"],
    "int_rate_diff": ["int_rate_diff"],
    "turnover_bundle": ["fumble_lost_diff", "int_rate_diff"],
}


def nfl_counts(seasons):
    latest = {}
    for snap in sorted(p for p in NFL_PBP_ROOT.iterdir() if p.is_dir()):
        for f in snap.glob("season=*/plays.parquet"):
            latest[int(f.parent.name.split("=")[1])] = f
    cols = [
        "game_id",
        "posteam",
        "play_type",
        "fumbled_1_player_id",
        "fumble_lost",
        "interception",
        "pass_attempt",
    ]
    out = []
    for s in sorted(seasons):
        if s not in latest:
            continue
        d = pd.read_parquet(latest[s], columns=cols)
        d = d[d["posteam"].notna()]
        rp = d["play_type"].isin(["pass", "run"])
        fumbled = d["fumbled_1_player_id"].notna()
        d["fum_all"] = (rp & fumbled).astype(float)
        d["fum_lost"] = (rp & fumbled & (d["fumble_lost"] == 1)).astype(float)
        d["passes"] = (d["pass_attempt"] == 1).astype(float)
        d["ints"] = ((d["pass_attempt"] == 1) & (d["interception"] == 1)).astype(float)
        out.append(d.groupby(["game_id", "posteam"])[COUNTS].sum().reset_index())
    r = pd.concat(out).rename(columns={"posteam": "team"})
    return r


def cfb_counts(seasons):
    parts = cfb_season_partitions(CFB_ROOT, "pbp")
    cols = ["game_id", "pos_team_id", "type.text"]
    out = []
    for s in sorted(seasons):
        if s not in parts:
            continue
        d = pd.read_parquet(parts[s], columns=cols)
        d = d[d["pos_team_id"].notna()]
        t = d["type.text"]
        d["fum_all"] = t.str.startswith("Fumble Recovery").astype(float)
        d["fum_lost"] = t.str.startswith("Fumble Recovery (Opponent)").astype(float)
        d["passes"] = t.isin(CFB_PASS_TYPES).astype(float)
        d["ints"] = t.isin(CFB_INT_TYPES).astype(float)
        d["pos_team_id"] = d["pos_team_id"].astype("int64")
        out.append(d.groupby(["game_id", "pos_team_id"])[COUNTS].sum().reset_index())
    return pd.concat(out).rename(columns={"pos_team_id": "team"})


def rate(num, den):
    return np.where(den > 0, num / den.where(den > 0), np.nan)


def pregame(f, counts, home_key, away_key):
    f = f.copy()
    f["gameday"] = pd.to_datetime(f["gameday"])
    long = pd.concat(
        [
            f[["game_id", "season", "gameday", home_key]].rename(columns={home_key: "team"}),
            f[["game_id", "season", "gameday", away_key]].rename(columns={away_key: "team"}),
        ]
    )
    long = long.merge(counts, on=["game_id", "team"], how="left")
    long = long.sort_values(["team", "season", "gameday", "game_id"])
    g = long.groupby(["team", "season"], sort=False)
    for c in COUNTS:
        long["cum_" + c] = g[c].cumsum() - long[c]
    long["fumble_lost"] = rate(long["cum_fum_lost"], long["cum_fum_all"])
    long["int_rate"] = rate(long["cum_ints"], long["cum_passes"])
    out = f
    for side, key in (("home", home_key), ("away", away_key)):
        m = long.rename(
            columns={
                "team": key,
                "fumble_lost": f"{side}_fumble_lost",
                "int_rate": f"{side}_int_rate",
            }
        )[["game_id", key, f"{side}_fumble_lost", f"{side}_int_rate"]]
        out = out.merge(m.drop_duplicates(["game_id", key]), on=["game_id", key], how="left")
    out["fumble_lost_diff"] = out["home_fumble_lost"] - out["away_fumble_lost"]
    out["int_rate_diff"] = out["home_int_rate"] - out["away_int_rate"]
    return out


_cache = {}


def features(f):
    is_cfb = "home_id" in f.columns
    key = "cfb" if is_cfb else "nfl"
    if key not in _cache:
        seasons = set(f["season"].unique())
        _cache[key] = cfb_counts(seasons) if is_cfb else nfl_counts(seasons)
    hk, ak = ("home_id", "away_id") if is_cfb else ("home_team", "away_team")
    out = pregame(f, _cache[key], hk, ak)
    cols = ["fumble_lost_diff", "int_rate_diff"]
    print(key, len(out), {c: round(float(out[c].isna().mean()), 3) for c in cols}, flush=True)
    return out


x.ARMS = ARMS
x.schedule_features = features

if __name__ == "__main__":
    x.main()
