import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402

OUT = REPO / "artifacts" / "mod25e8"


def load_real():
    res = {}
    for nm, ss in (("train", dv.TRAIN), ("eval", dv.EVAL)):
        f = OUT / f"real_{nm}.parquet"
        if f.exists():
            res[nm] = pd.read_parquet(f)
            continue
        P, M, meta, pbp = dv.real_load(ss)
        P.to_parquet(f)
        res[nm] = P
    return res


def load_sim(d):
    fs = sorted(glob.glob(str(Path(d) / "play_*_*.parquet")))
    P = pd.concat([pd.read_parquet(f).assign(f=i) for i, f in enumerate(fs) if int(Path(f).stem.split("_")[2]) >= 2], ignore_index=True)
    P["g"] = P["f"].astype(np.int64) * 100000 + P["g"].astype(np.int64)
    P["flip"] = P["flip"].astype(bool)
    nx = P.groupby("g")["offhome"].shift(-1)
    P["flip"] = (nx.notna() & (nx != P["offhome"])) | P["flip"]
    P["tov"] = 0
    return P.drop(columns=["f"])


def zone(P):
    q = P["qtr"].to_numpy()
    g = P["gsr"].to_numpy()
    return np.select([(q == 2) & (g <= 120) | (q == 4) & (g <= 120), (q == 4) & (g <= 300), q <= 3], ["2min", "q4_5min", "q1_3"], "q4_early")


def snap_table(P, name):
    R = P[P["qtr"] <= 4].copy()
    R["z"] = zone(R)
    pl = R["code"].isin([0, 1])
    o = {}
    o["el_all"] = R["el"].mean()
    for q in (1, 2, 3, 4):
        o[f"el_q{q}"] = R.loc[R["qtr"] == q, "el"].mean()
    for c, nm in ((0, "pass"), (1, "run"), (2, "punt"), (3, "fg")):
        o[f"el_{nm}"] = R.loc[R["code"] == c, "el"].mean()
        o[f"n_{nm}_pg"] = (R["code"] == c).sum() / R["g"].nunique()
    for z in ("q1_3", "q4_early", "q4_5min", "2min"):
        m = R["z"] == z
        o[f"el_{z}"] = R.loc[m, "el"].mean()
        o[f"el_pass_{z}"] = R.loc[m & (R["code"] == 0), "el"].mean()
        o[f"el_run_{z}"] = R.loc[m & (R["code"] == 1), "el"].mean()
    o["snaps_pg"] = len(R) / R["g"].nunique()
    o["el_sum_pg"] = R["el"].sum() / R["g"].nunique()
    o["el_q123_play"] = R.loc[(R["qtr"] <= 3) & pl, "el"].mean()
    o["el_q123_play_med"] = R.loc[(R["qtr"] <= 3) & pl, "el"].median()
    o["el_q123_play_lt10"] = (R.loc[(R["qtr"] <= 3) & pl, "el"] < 10).mean()
    o["el_q123_play_lt20"] = (R.loc[(R["qtr"] <= 3) & pl, "el"] < 20).mean()
    o["el_q123_play_gt40"] = (R.loc[(R["qtr"] <= 3) & pl, "el"] > 40).mean()
    return pd.Series(o, name=name)


def drive_stats(P, name):
    D, R = dv.drive_table(P)
    ng = D["g"].nunique()
    o = {"drives_pg": len(D) / ng, "plays_per_drive": D["n"].mean(), "secs_per_drive": D["secs"].mean()}
    o["snaps_pg"] = len(R) / ng
    for k in ("td", "fg", "fg_miss", "punt", "turnover", "downs", "def_score", "end_half"):
        o["n_" + k + "_pg"] = (D["oc"] == k).sum() / ng
    o["punt_pg"] = (R["code"] == 2).sum() / ng
    for z, m in (("q1_3", D["q0"] <= 3), ("q4", D["q0"] == 4)):
        d = D[m]
        o[f"drives_pg_{z}"] = len(d) / ng
        o[f"plays_{z}"] = d["n"].mean()
        o[f"secs_{z}"] = d["secs"].mean()
    d3 = D[D["oc"].isin(["punt"])]
    o["punt_drive_plays"] = d3["n"].mean()
    o["three_out"] = ((D["n"] <= 3) & (D["oc"] == "punt")).mean()
    o["n1_3_share"] = (D["n"] <= 3).mean()
    for k in (1, 2, 3, 4, 5, 6, 7, 8):
        o[f"len_{k}"] = (D["n"] == k).sum() / ng
    o["len_9p"] = (D["n"] >= 9).sum() / ng
    o["secs_end_gap"] = D["secs"].sum() / ng
    return pd.Series(o, name=name)


def main():
    real = load_real()
    sim = load_sim(sys.argv[1])
    tabs = [snap_table(real["train"], "real_train"), snap_table(real["eval"], "real_eval"), snap_table(sim, "sim")]
    ds = [drive_stats(real["train"], "real_train"), drive_stats(real["eval"], "real_eval"), drive_stats(sim, "sim")]
    pd.set_option("display.width", 200)
    print(pd.concat(tabs, axis=1).round(3).to_string())
    print(pd.concat(ds, axis=1).round(3).to_string())


if __name__ == "__main__":
    main()
