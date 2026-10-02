import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402

OUT = REPO / "artifacts" / "mod25e"
SB = [0, 20, 35, 50, 65, 80, 101]
ZB = [0, 10, 20, 35, 50, 65, 80, 101]


def prep(P):
    D, R = dv.drive_table(P)
    D = D.copy()
    D["prev"] = D.groupby("g")["oc"].shift(1)
    D["sb"] = pd.cut(D["yl0"], SB, right=False)
    return D, R


def plays_nt(R):
    R = R.sort_values(["g"], kind="stable").reset_index(drop=True)
    nxt_yl = R["yl"].shift(-1)
    same = (R["g"] == R["g"].shift(-1)) & ~R["flip"] & (R["po"] == 0) & (R["pdf"] == 0)
    R = R.assign(gain=(R["yl"] - nxt_yl).where(same))
    return R


def summarize(P, name):
    D, R = prep(P)
    ng = D["g"].nunique()
    o = {"name": name, "games": int(ng)}
    o["pts_game"] = float((D["pts"].sum() + D["pdf"].sum()) / ng)
    o["drive_pts_game"] = float(D["pts"].sum() / ng)
    o["drives_game"] = float(len(D) / ng)
    o["pts_drive"] = float(D["pts"].mean())
    o["plays_drive"] = float(D["n"].mean())
    o["start_yl"] = float(D["yl0"].mean())
    for k in ("td", "fg", "fg_miss", "punt", "turnover", "downs", "def_score", "end_half"):
        o["oc_" + k] = float((D["oc"] == k).mean())
    for pv in ("td", "fg", "punt", "turnover", "downs"):
        d = D[D["prev"] == pv]
        o[f"start_after_{pv}"] = float(d["yl0"].mean())
        o[f"n_after_{pv}"] = float(len(d) / ng)
    d = D[D["prev"].isin(["td", "fg"])]
    o["tb_share_kick"] = float((d["yl0"].isin([75.0, 80.0, 70.0])).mean())
    o["start_kick_sd"] = float(d["yl0"].std())
    o["start_punt_sd"] = float(D[D["prev"] == "punt"]["yl0"].std())
    o["rz_share"] = float(D["rz"].mean())
    rz = D[D["rz"]]
    o["rz_td"] = float((rz["oc"] == "td").mean())
    o["rz_fg"] = float((rz["oc"] == "fg").mean())
    o["rz_pts"] = float(rz["pts"].mean())
    o["nonrz_pts"] = float(D[~D["rz"]]["pts"].mean())
    o["nonrz_td"] = float((D[~D["rz"]]["oc"] == "td").mean())
    for b, g in D.groupby("sb", observed=True):
        o[f"drv_share_{b}"] = float(len(g) / len(D))
        o[f"drv_pts_{b}"] = float(g["pts"].mean())
        o[f"drv_td_{b}"] = float((g["oc"] == "td").mean())
        o[f"drv_rzr_{b}"] = float(g["rz"].mean())
    R = plays_nt(R)
    S = R[R["code"].isin([0, 1]) & R["gain"].notna()]
    o["ypp"] = float(S["gain"].mean())
    o["expl20"] = float((S["gain"] >= 20).mean())
    o["expl_yds_share"] = float(S.loc[S["gain"] >= 20, "gain"].sum() / S["gain"].sum())
    o["gain_sd"] = float(S["gain"].std())
    for dn in (1, 2, 3, 4):
        d = S[S["down"] == dn]
        o[f"ypp_d{dn}"] = float(d["gain"].mean())
        o[f"fd_d{dn}"] = float((d["gain"] >= d["dist"]).mean())
    S2 = S.assign(zb=pd.cut(S["yl"], ZB, right=False))
    for b, g in S2.groupby("zb", observed=True):
        o[f"ypp_z{b}"] = float(g["gain"].mean())
        o[f"expl_z{b}"] = float((g["gain"] >= 20).mean())
        o[f"nplay_z{b}"] = float(len(g) / ng)
    RP = R[R["code"].isin([0, 1])]
    o["tov_play"] = float((RP["tov"] > 0).mean())
    ztd = R[(R["yl"] <= 10) & R["code"].isin([0, 1])]
    o["td_play_inside10"] = float((ztd["po"] >= 6).mean())
    o["td_play_10_20"] = float((R[(R["yl"] > 10) & (R["yl"] <= 20) & R["code"].isin([0, 1])]["po"] >= 6).mean())
    o["td_play_20_35"] = float((R[(R["yl"] > 20) & (R["yl"] <= 35) & R["code"].isin([0, 1])]["po"] >= 6).mean())
    o["td_play_35_50"] = float((R[(R["yl"] > 35) & (R["yl"] <= 50) & R["code"].isin([0, 1])]["po"] >= 6).mean())
    o["td_play_50_65"] = float((R[(R["yl"] > 50) & (R["yl"] <= 65) & R["code"].isin([0, 1])]["po"] >= 6).mean())
    o["td_play_65_100"] = float((R[(R["yl"] > 65) & R["code"].isin([0, 1])]["po"] >= 6).mean())
    f = R[R["code"] == 3]
    o["fg_att_game"] = float(len(f) / ng)
    o["fg_make"] = float((f["po"] == 3).mean())
    for lo, hi in ((0, 25), (25, 35), (35, 45), (45, 60)):
        g = f[(f["yl"] > lo) & (f["yl"] <= hi)]
        o[f"fg_n_{lo}_{hi}"] = float(len(g) / ng)
        o[f"fg_make_{lo}_{hi}"] = float((g["po"] == 3).mean())
    f4 = R[(R["down"] == 4) & R["code"].isin([0, 1, 2, 3])]
    for lo, hi in ((0, 20), (20, 35), (35, 50), (50, 65), (65, 101)):
        g = f4[(f4["yl"] > lo) & (f4["yl"] <= hi)]
        o[f"go_4th_{lo}_{hi}"] = float(g["code"].isin([0, 1]).mean())
        o[f"fg_4th_{lo}_{hi}"] = float((g["code"] == 3).mean())
    o["_D"] = D
    return o


def decomp(r, s):
    Dr, Ds = r["_D"], s["_D"]
    sh_r = Dr.groupby("sb", observed=True)["pts"].agg(["mean", "size"])
    sh_s = Ds.groupby("sb", observed=True)["pts"].agg(["mean", "size"])
    wr = sh_r["size"] / sh_r["size"].sum()
    ws = sh_s["size"] / sh_s["size"].sum()
    comp = float(((ws - wr) * sh_r["mean"]).sum())
    rate = float((ws * (sh_s["mean"] - sh_r["mean"])).sum())
    dpg = s["drives_game"] - r["drives_game"]
    return {"d_pts_game_drive": s["drive_pts_game"] - r["drive_pts_game"], "drive_count_term": dpg * r["pts_drive"], "start_composition_term": comp * s["drives_game"], "within_start_rate_term": rate * s["drives_game"], "d_pts_drive": s["pts_drive"] - r["pts_drive"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="crk06")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default="e4a")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    P, M, meta, pbp = dv.real_load(dv.TRAIN)
    res["real_train"] = summarize(P, "real_train")
    P, M, meta, pbp = dv.real_load(dv.EVAL)
    res["real_eval"] = summarize(P, "real_eval")
    for v in args.variants.split(","):
        S = pd.read_parquet(Path(args.cache) / f"sim_{v}.parquet")
        nx = S.groupby("g")["offhome"].shift(-1)
        S["flip"] = (nx.notna() & (nx != S["offhome"])) | S["flip"].astype(bool)
        res[v] = summarize(S, v)
    dec = {}
    for v in args.variants.split(","):
        for ref in ("real_train", "real_eval"):
            dec[f"{v}_vs_{ref}"] = decomp(res[ref], res[v])
    keys = [k for k in res["real_train"] if k not in ("_D", "name")]
    lines = [f"{'metric':32s} " + " ".join(f"{n:>11s}" for n in res)]
    for k in keys:
        lines.append(f"{k:32s} " + " ".join(f"{res[n].get(k, float('nan')):11.4f}" for n in res))
    lines.append("")
    for k, v in dec.items():
        lines.append(k + " " + json.dumps({a: round(b, 3) for a, b in v.items()}))
    txt = "\n".join(lines)
    (OUT / f"deficit_{args.tag}.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
