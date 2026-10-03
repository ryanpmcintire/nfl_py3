import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim04_engine as sim  # noqa: E402


def pool(seasons):
    pbp = sim.load_reg_seasons(tuple(seasons))
    tr = sim.build_transition_frame(pbp)
    keep = [c for c in ["game_id", "play_id", "desc", "penalty", "penalty_team", "penalty_type", "fumble", "fumble_lost", "fumble_recovery_1_team", "interception", "play_type", "posteam", "defteam", "qtr", "yardline_100", "touchdown", "safety", "return_yards", "kick_distance", "punt_blocked", "timeout"] if c in pbp.columns]
    m = tr.merge(pbp[keep].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left", suffixes=("", "_p"))
    return pbp, tr, m


def main():
    pbp, tr, m = pool(range(2009, 2018))
    print("rows", len(m), "cols", [c for c in m.columns if c.endswith("_p")])
    term = m["possession_flip"].astype(bool) | (m["points_off"] > 0) | (m["points_def"] > 0)
    neg = ~term & (m["yards_gained"] < -20)
    print("pool non-terminal yg<-20:", int(neg.sum()), "of", int((~term).sum()))
    z = m[neg].copy()
    nxt = m.groupby("game_id")["qtr_actual"].shift(-1)
    z["qchg"] = (nxt[neg] != z["qtr_actual"]).to_numpy()
    z["pen"] = z.get("penalty", pd.Series(0, index=z.index)).fillna(0).astype(int)
    z["fum"] = z.get("fumble", pd.Series(0, index=z.index)).fillna(0).astype(int)
    z["tdp"] = z.get("touchdown", pd.Series(0, index=z.index)).fillna(0).astype(int)
    z["pt"] = z["play_type"]
    print(z.groupby(["pt", "pen", "fum", "tdp", "qchg"]).agg(n=("yards_gained", "size"), yg=("yards_gained", "median"), yl=("fp_raw", "median"), nyl=("next_yardline", "median")).sort_values("n", ascending=False).head(25).to_string())
    print("next_down value counts", z["next_down"].value_counts().head(5).to_dict())
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    print(z[["game_id", "play_id", "pt", "down_i", "fp_raw", "next_yardline", "next_down", "yards_gained", "desc"]].sample(14, random_state=1).to_string())
    z.to_pickle(str(REPO / "artifacts" / "mod25e3" / "neg_pool.pkl"))


def enabled():
    import os

    return os.environ.get("NEG") == "1"


def install_neg():
    import mod25d_variance as dv

    t = dv._G["tables"]
    a = t["arrays"]
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = dv.sim.build_transition_frame(pbp)
    assert len(trans) == len(a["possession_flip"])
    gid = trans["game_id"].to_numpy()
    last = np.r_[gid[1:] != gid[:-1], True]
    qtr = trans["qtr_actual"].to_numpy()
    half_last = np.r_[(gid[1:] == gid[:-1]) & (qtr[:-1] == 2) & (qtr[1:] == 3), False]
    edge = last | half_last
    right = pbp[["game_id", "play_id", "yards_gained"]].drop_duplicates(["game_id", "play_id"])
    real = trans[["game_id", "play_id"]].merge(right, on=["game_id", "play_id"], how="left")["yards_gained"].fillna(0.0).to_numpy(dtype=float)
    nonterm = ~a["possession_flip"].astype(bool) & (a["points_off"] == 0) & (a["points_def"] == 0)
    fix = edge & nonterm
    yg = a["yards_gained"].astype(float).copy()
    before = int((nonterm & (yg < -20)).sum())
    yg[fix] = real[fix]
    a["yards_gained"] = yg
    af = a["auto_first"].copy()
    rd = a["repeat_down"].copy()
    af[fix] = False
    rd[fix] = False
    a["auto_first"] = af
    a["repeat_down"] = rd
    if "nn_weight_cache_cond" in t:
        t["nn_weight_cache_cond"].clear()
    print("neg rows", int(fix.sum()), "nonterminal yg<-20 before", before, "after", int((nonterm & (yg < -20)).sum()), flush=True)


def seqstats(D):
    term = (D["flip"] > 0) | (D["po"] > 0) | (D["pdf"] > 0)
    start = np.r_[True, term[:-1]]
    sid = np.cumsum(start)
    n = sid[-1]
    pts = np.bincount(sid - 1, weights=D["po"], minlength=n)
    ptsd = np.bincount(sid - 1, weights=D["pdf"], minlength=n)
    punt = np.bincount(sid - 1, weights=(D["code"] == 2).astype(float), minlength=n) > 0
    fp = D["yl"][start]
    return dict(drives=int(n), start_yl=float(fp.mean()), punt_share=float(punt.mean()), pts_per_drive=float(pts.mean()), def_pts_per_drive=float(ptsd.mean()), plays_per_drive=float(len(D["yl"]) / n))


def analyze(path):
    import glob

    cols = ["yl", "down", "qtr", "idx", "yg", "flip", "po", "pdf", "nyl", "ndown", "code", "pool_yg", "pool_flip", "pool_po", "pool_pdf", "pool_nyl", "x"]
    parts = [np.fromfile(f, dtype=np.float64).reshape(-1, len(cols)) for f in sorted(glob.glob(str(Path(path) / "e49_*.bin")))]
    X = np.vstack(parts)
    D = {c: X[:, i] for i, c in enumerate(cols)}
    term = (D["flip"] > 0) | (D["po"] > 0) | (D["pdf"] > 0)
    neg = ~term & (D["yg"] < -20)
    print("draws", len(X), "nonterminal yg<-20", int(neg.sum()), "share", float(neg.mean()))
    print("sim", seqstats({k: v for k, v in D.items()}))
    pbp = sim.load_reg_seasons(tuple(range(2009, 2018)))
    tr = sim.build_transition_frame(pbp)
    R = dict(yl=tr["fp_raw"].to_numpy(float), flip=tr["possession_flip"].to_numpy(float), po=tr["points_off"].to_numpy(float), pdf=tr["points_def"].to_numpy(float), code=tr["play_type_code"].to_numpy(float))
    print("real", seqstats(R))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "analyze":
        analyze(sys.argv[2])
    else:
        main()
