import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import sim09_u3a as u3  # noqa: E402
import sim09_urgency as urg  # noqa: E402

OUT = REPO / "artifacts" / "sim09" / "u3d"
EXTRA = ["out_of_bounds", "incomplete_pass", "complete_pass", "qb_scramble", "timeout", "posteam_timeouts_remaining"]
u3.SRC = u3.SRC + EXTRA
dv.DV["crzk"] = dict(dv.DV["crz"], tfix=1)


def install_tfix():
    t = dv._G["tables"]
    a = t["arrays"]
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = dv.sim.build_transition_frame(pbp)
    assert len(trans) == len(a["possession_flip"])
    gid = trans["game_id"].to_numpy()
    assert int((gid[1:] != gid[:-1]).sum()) + 1 == len(np.unique(gid))
    last = np.r_[gid[1:] != gid[:-1], True]
    qtr = trans["qtr_actual"].to_numpy()
    half_last = np.r_[(gid[1:] == gid[:-1]) & (qtr[:-1] == 2) & (qtr[1:] == 3), False]
    edge = last | half_last
    tov = np.asarray(dv.c25.attrs_from(pbp, trans)["tov"], dtype=float)
    code = a["play_type_code"]
    fix = edge & (a["points_off"] == 0) & (a["points_def"] == 0) & (tov == 0) & np.isin(code, (0, 1, 4, 5)) & a["possession_flip"].astype(bool)
    flip = a["possession_flip"].copy()
    flip[fix] = False
    nd = a["next_down"].astype(float).copy()
    nd[fix | (edge & ~a["possession_flip"].astype(bool) & (a["points_off"] == 0) & (a["points_def"] == 0))] = np.nan
    a["possession_flip"] = flip
    a["next_down"] = nd
    t["nn_weight_cache_cond"].clear()
    print("tfix rows", int(fix.sum()), "half-last", int(half_last.sum()), "of game-last", int(last.sum()), flush=True)


def tfix_init(setting):
    import mod25e_scorestate as ss

    ss.s_init(setting)
    if dv._G["cfg"].get("tfix"):
        install_tfix()


def tfix_init_budget(setting):
    import mod25e_budget as bud

    bud.e_init(setting)
    if dv._G["cfg"].get("tfix"):
        install_tfix()


def real_frame():
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    at = dv.c25.attrs_from(pbp, tr)
    tr = tr.assign(tov=at["tov"], epa=at["epa"], yards=at["yards"]).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    gid, _ = pd.factorize(tr["game_id"])
    P = pd.DataFrame({
        "g": gid, "down": tr["down_i"].to_numpy(), "dist": tr["dist_raw"].to_numpy(), "yl": tr["fp_raw"].to_numpy(), "sd": tr["sc_raw"].to_numpy(),
        "gsr": tr["gsr_actual"].to_numpy(), "qtr": tr["qtr_actual"].to_numpy(), "code": tr["play_type_code"].to_numpy(), "po": tr["points_off"].to_numpy(),
        "pdf": tr["points_def"].to_numpy(), "flip": tr["possession_flip"].to_numpy().astype(bool), "el": tr["clock_elapsed"].to_numpy(),
        "epa": tr["epa"].to_numpy(), "yards": tr["yards"].to_numpy(), "tov": tr["tov"].to_numpy(),
    })
    cols = ["game_id", "play_id"] + u3.SRC
    nv = pd.concat([pd.read_parquet(urg.NV / f"pbp_{s}.parquet", columns=cols) for s in range(min(dv.TRAIN), max(dv.TRAIN) + 1)], ignore_index=True).drop_duplicates(["game_id", "play_id"])
    x = tr[["game_id", "play_id"]].merge(nv, on=["game_id", "play_id"], how="left")
    for c in u3.SRC:
        P["src_" + c] = x[c].to_numpy()
    P["w"] = 0
    return P


def pool_tov():
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    return np.asarray(dv.c25.attrs_from(pbp, tr)["tov"], dtype=float)


def sim_frame(minsid):
    P = u3.load_sim_frames(minsid)
    tv = pool_tov()
    P["tov"] = tv[P["idx"].to_numpy(np.int64)].astype(int)
    return P


def kind(P):
    s = lambda c: P["src_" + c].fillna(0).to_numpy()
    code = P["code"].to_numpy()
    k = np.full(len(P), "other", dtype=object)
    k[code == 2] = "punt"
    k[code == 3] = "kick"
    k[code == 4] = "kneel"
    k[code == 5] = "spike"
    run, ps = code == 0, code == 1
    k[run] = "run"
    k[ps] = "pass_other"
    k[ps & (s("complete_pass") == 1) & (s("out_of_bounds") == 0)] = "comp_in"
    k[ps & (s("complete_pass") == 1) & (s("out_of_bounds") == 1)] = "comp_oob"
    k[ps & (s("incomplete_pass") == 1)] = "incomplete"
    k[ps & (s("sack") == 1)] = "sack"
    k[ps & (s("interception") == 1)] = "int"
    k[run & (s("out_of_bounds") == 1)] = "run_oob"
    k[run & (s("qb_scramble") == 1)] = "scramble"
    return k


def anatomy(P, name, el_max=60):
    P = P[P["qtr"] <= 4].copy()
    P["k"] = kind(P)
    D, R = dv.drive_table(P)
    R = R.copy()
    R = R[R["el"] <= el_max]
    out = {}
    first = R.groupby("poss").first()
    last = R.groupby("poss").last()
    lastk = last["k"]
    D = D.set_index("poss")
    D["lastk"] = lastk
    D["tos"] = D["oc"]
    D.loc[(D["oc"] == "turnover") & (D["lastk"] == "int"), "tos"] = "int"
    D.loc[(D["oc"] == "turnover") & (~D["lastk"].isin(["int"])), "tos"] = "fumble"
    D.loc[D["lastk"] == "kneel", "tos"] = "kneel"
    sc = R[R["code"].isin([0, 1, 4, 5])]
    tor = R["src_posteam_timeouts_remaining"].groupby(R["poss"])
    D["to_used"] = (tor.first() - tor.last()).clip(lower=0).reindex(D.index) if name == "real" else np.nan
    D["nsc"] = sc.groupby("poss").size().reindex(D.index).fillna(0)
    D["spike"] = (R["k"] == "spike").groupby(R["poss"]).sum().reindex(D.index).fillna(0)
    D["oobn"] = R["k"].isin(["comp_oob", "run_oob"]).groupby(R["poss"]).sum().reindex(D.index).fillna(0)
    D["snapsecs"] = R["el"].groupby(R["poss"]).sum().reindex(D.index).fillna(0)
    D["gsr_end"] = (R["gsr"] - R["el"]).groupby(R["poss"]).last().reindex(D.index)
    pr = R["code"].isin([0]) & ~R["k"].isin(["spike"])
    ni = (R["k"] == "incomplete")
    c = ni.groupby(R["poss"]).cumsum()
    before = ((c == 0) & R["code"].isin([0, 1]) & ~ni).groupby(R["poss"]).sum()
    anyinc = ni.groupby(R["poss"]).sum()
    D["pre_inc"] = before.reindex(D.index)
    D["has_inc"] = anyinc.reindex(D.index) > 0
    rows = []
    for q, lo, hi in ((2, 1800, 1920), (4, 0, 120)):
        m = (D["q0"] == q) & (D["sd0"] < 0) & (D["gsr0"] > lo) & (D["gsr0"] <= hi)
        d = D[m]
        Rm = R[R["poss"].isin(d.index)]
        o = {"n_drives": len(d), "scr_plays": d["nsc"].mean(), "snaps": d["n"].mean(), "secs": d["snapsecs"].mean(), "start_yl": d["yl0"].mean(), "start_gsr": d["gsr0"].mean() - lo, "end_gsr_mean": d["gsr_end"].mean() - lo,
             "spikes": d["spike"].mean(), "to_used": d["to_used"].mean(), "oob_plays": d["oobn"].mean(), "pre_inc": d.loc[d["has_inc"], "pre_inc"].mean(), "no_inc_share": 1 - d["has_inc"].mean()}
        for k in ("td", "fg", "fg_miss", "punt", "int", "fumble", "downs", "end_half", "kneel", "def_score", "other"):
            o["end_" + k] = (d["tos"] == k).mean()
        db = Rm[Rm["code"] == 1]
        o["sack_rate"] = (db["k"] == "sack").mean()
        o["scr_rate"] = (db["k"] == "scramble").mean()
        o["comp_oob_share_of_comp"] = (db["k"] == "comp_oob").sum() / max(1, db["k"].isin(["comp_in", "comp_oob"]).sum())
        for kk in ("comp_in", "comp_oob", "incomplete", "run", "run_oob", "spike", "sack", "kneel", "int", "scramble", "pass_other"):
            o["el_" + kk] = Rm.loc[Rm["k"] == kk, "el"].mean()
            o["n_" + kk + "_pd"] = (Rm["k"] == kk).sum() / max(1, len(d))
        o["pass_share"] = (Rm["code"] == 1).sum() / max(1, Rm["code"].isin([0, 1]).sum())
        o["run_oob_share"] = (Rm["k"] == "run_oob").sum() / max(1, (Rm["code"] == 0).sum())
        sc4 = Rm[Rm["code"].isin([0, 1]) & (Rm["down"] == 4)]
        o["reach4_per_drive"] = len(sc4) / max(1, len(d))
        o["reach4_share"] = Rm[Rm["down"] == 4].groupby("poss").ngroups / max(1, len(d))
        a4 = Rm[(Rm["down"] == 4) & Rm["code"].isin([0, 1, 2, 3])]
        o["fourth_go_share"] = a4["code"].isin([0, 1]).mean()
        o["fourth_punt_share"] = (a4["code"] == 2).mean()
        o["fourth_fg_share"] = (a4["code"] == 3).mean()
        for dn in (1, 2, 3):
            sd_ = Rm[Rm["code"].isin([0, 1]) & (Rm["down"] == dn)]
            o[f"d{dn}_n_pd"] = len(sd_) / max(1, len(d))
            o[f"d{dn}_inc"] = (sd_["k"] == "incomplete").mean()
            o[f"d{dn}_gain"] = sd_["yards"].mean()
            o[f"d{dn}_gain_comp"] = sd_.loc[sd_["k"].isin(["comp_in", "comp_oob"]), "yards"].mean()
        o["plays_ge10_share"] = (d["nsc"] >= 10).mean()
        o["n_drives_per_game"] = len(d) / D["g"].nunique()
        for lo2, hi2 in ((0, 30), (30, 60), (60, 90), (90, 120)):
            m2 = (d["gsr0"] - lo > lo2) & (d["gsr0"] - lo <= hi2)
            o[f"b{hi2}_n_share"] = m2.mean()
            o[f"b{hi2}_scr"] = d.loc[m2, "nsc"].mean()
            o[f"b{hi2}_td"] = (d.loc[m2, "tos"] == "td").mean()
            o[f"b{hi2}_downs"] = (d.loc[m2, "tos"] == "downs").mean()
        for kk, v in o.items():
            rows.append((name, f"q{q}", kk, v))
    return pd.DataFrame(rows, columns=["src", "half", "metric", "value"]), D


def cmd_sim(args):
    import mod25_generator as gen
    import mod25e_budget as bud

    bud.OUT = Path(args.out_dir).resolve()
    os.environ["BUD_OUT"] = str(bud.OUT)
    bud.OUT.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = tfix_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    ss.s_init = tfix_init
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_analyze(args):
    import argparse as ap_

    u3.OUT = OUT / "analyze"
    u3.bud.OUT = Path(args.post_dir)
    u3.cmd_analyze(ap_.Namespace(minsid=2, boot=args.boot))


def cmd_anat(args):
    u3.bud.OUT = Path(args.dir)
    ra, _ = anatomy(real_frame(), "real")
    sa, _ = anatomy(sim_frame(2), "sim")
    t = pd.concat([ra, sa]).pivot_table(index=["half", "metric"], columns="src", values="value")[["real", "sim"]]
    t["diff"] = t["sim"] - t["real"]
    t.to_csv(OUT / f"anatomy_{Path(args.dir).name}.csv")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(t.round(4).to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("anat")
    a.add_argument("--dir", default=str(REPO / "artifacts" / "sim09" / "u3a_play"))
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crzk")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUT / "play_crzk"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzk")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    n = sub.add_parser("analyze")
    n.add_argument("--post-dir", dest="post_dir", default=str(OUT / "play_crzk"))
    n.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    {"anat": cmd_anat, "sim": cmd_sim, "e5": cmd_e5, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
