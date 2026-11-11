import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sim10_drive as eng  # noqa: E402

FC = (
    ROOT
    / "data"
    / "raw"
    / "forecast_archive"
    / "pool_decision_2009_2025_aware"
    / "forecasts.parquet"
)
OUT = ROOT / "artifacts" / "sim15"
OUTDOOR = ("outdoors", "open")
BAND = (31, 40)
DECISION_TYPES = ("pass", "run", "punt", "field_goal")
SLOPE_OUTCOME = "fg_31_40"
SLOPE_SPEC = "main_observed"
SLOPE_TERM = "wind10"


def wind_table():
    rg = pd.read_parquet(eng.OUT / "real_games.parquet")[["game_id", "season", "week"]]
    fc = pd.read_parquet(FC)
    fc = fc[fc.fetch_status == "ok"][["game_id", "roof", "actual_wind_mph"]].drop_duplicates(
        "game_id"
    )
    t = rg.merge(fc, on="game_id", how="left")
    outdoor = t["roof"].isin(OUTDOOR)
    t["outdoor"] = outdoor
    t["wind_mph"] = np.where(outdoor, t["actual_wind_mph"], np.nan)
    t["in_archive"] = t["roof"].notna()
    t = t.drop(columns=["actual_wind_mph"]).rename(columns={"roof": "roof_state"})
    t.to_parquet(OUT / "wind_table.parquet", index=False)
    cov = (
        t.groupby("season")
        .agg(
            games=("game_id", "size"),
            outdoor=("outdoor", "sum"),
            wind_known=("wind_mph", lambda s: int(s.notna().sum())),
            not_in_archive=("in_archive", lambda s: int((~s).sum())),
        )
        .assign(outdoor_missing_wind=lambda d: d["outdoor"] - d["wind_known"])
    )
    return t, cov


def segment_flags(p):
    games = set(pd.read_parquet(eng.OUT / "real_games.parquet")["game_id"])
    p = p[p["game_id"].isin(games)].reset_index(drop=True)
    gid = p["game_id"].to_numpy()
    same_next = np.r_[gid[1:] == gid[:-1], False]
    hs, as_ = p["hs"].to_numpy(float), p["as"].to_numpy(float)
    scored = (np.r_[hs[1:], 0.0] != hs) | (np.r_[as_[1:], 0.0] != as_)
    scored = scored & same_next
    half = p["half"].to_numpy()
    snap = p["down"].notna() & p["posteam"].notna() & p["fixed_drive"].notna()
    first = p[snap].groupby(["game_id", "fixed_drive"], sort=False).head(1).index.to_numpy()
    starts = np.sort(first)
    sg, sh = gid[starts], half[starts]
    nxt_same = np.r_[(sg[1:] == sg[:-1]) & (sh[1:] == sh[:-1]), False]
    hk = pd.Series(np.arange(len(p))).groupby([gid, half]).transform("max").to_numpy() + 1
    end = np.where(nxt_same, np.r_[starts[1:], 0], hk[starts])
    seg = pd.read_parquet(eng.OUT / "segments.parquet", columns=["game_id", "season", "yl"])
    assert len(seg) == len(starts)
    assert (seg["game_id"].to_numpy() == gid[starts]).all()
    assert np.allclose(seg["yl"].to_numpy(float), p["yardline_100"].to_numpy(float)[starts])
    in_band = p["yardline_100"].between(*BAND).to_numpy()
    kick = (p["play_type"] == "field_goal").to_numpy()
    made_long = (kick & in_band & scored).astype(np.int64)
    cs = np.r_[0, np.cumsum(made_long)]
    seg_flag = (cs[end] - cs[starts]) > 0
    decision = (
        p["play_type"].isin(DECISION_TYPES) & (p["down"] == 4) & (p["qtr"] <= 4) & in_band
    ).to_numpy()
    share = float(kick[decision].mean())
    return seg_flag.astype(np.int8), share, int(decision.sum()), int(made_long.sum())


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    t, cov = wind_table()
    print(cov.to_string())
    print("total games", len(t), "outdoor", int(t["outdoor"].sum()))
    print("outdoor without wind value", int((t["outdoor"] & t["wind_mph"].isna()).sum()))
    print("not in archive", int((~t["in_archive"]).sum()))
    flags, share, ndec, nmade = segment_flags(eng.load_pbp())
    pd.DataFrame({"long_fg_made": flags}).to_parquet(OUT / "seg_long_fg.parquet", index=False)
    lo = pd.read_csv(OUT / "loso.csv")
    lo = lo[(lo["outcome"] == SLOPE_OUTCOME) & (lo["spec"] == SLOPE_SPEC)]
    slopes = {str(int(r.season)): float(getattr(r, SLOPE_TERM)) for r in lo.itertuples()}
    hook = {
        "slope_per_10mph_by_season": slopes,
        "try_share_in_band": share,
        "band_yardline_100": list(BAND),
        "band_decisions": ndec,
        "segments_flagged": int(flags.sum()),
        "band_scoring_kicks": nmade,
    }
    (OUT / "wind_hook.json").write_text(json.dumps(hook, indent=1))
    print(json.dumps(hook, indent=1))


if __name__ == "__main__":
    run()
