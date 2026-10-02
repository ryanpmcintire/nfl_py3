import argparse
import inspect
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
import sim09_u3d as u3d  # noqa: E402

OUT = REPO / "artifacts" / "sim09" / "u4a"
dv.DV["crzf"] = dict(dv.DV["crzk"], fat=1)
dv.DV["crzfc"] = dict(dv.DV["crzk"], fat=1, fatc=1)

RUN_RESET = "    plays = 0\n    possessions = 1\n"
RUN_PRE = "        phase = 4 if in_ot else compute_phase(qtr, gsr)\n"
RUN_POL = "drawn = policy(down, distance, yardline, score_diff, qtr, clock_val, drawn)"
RUN_HALF = "                possessions += 1\n                offense = second_half_receiver\n"
RUN_OT = "                possessions += 1\n                drive_start_qtr = 5\n"
RUN_MAIN = "            possessions += 1\n            if in_ot:\n                ot_possession_index += 1\n"
PICK_W = "weights = np.exp(-dist_sq / (2.0 * h * h))"
PICK_K = "wkey = (key, off_sim, def_sim, is_home_sim)"


def patch_run(src):
    for a in (RUN_RESET, RUN_PRE, RUN_POL, RUN_HALF, RUN_OT, RUN_MAIN):
        assert a in src, a
    src = src.replace(RUN_RESET, RUN_RESET + '    FAT_LAST["home"] = FAT_LAST["away"] = 0\n    FAT_CUM["home"] = FAT_CUM["away"] = 0\n    FAT_CUR[0] = 0\n')
    src = src.replace(RUN_PRE, '        _fd = "away" if offense == "home" else "home"\n        FAT_ST[0] = FAT_LAST[_fd]\n        FAT_ST[1] = FAT_CUM[_fd]\n' + RUN_PRE)
    src = src.replace(RUN_POL, RUN_POL + '\n        if int(drawn["play_type_code"]) in (0, 1):\n            FAT_CUR[0] += 1\n            FAT_CUM[_fd] += 1')
    end = '                FAT_LAST["away" if offense == "home" else "home"] = FAT_CUR[0]\n                FAT_CUR[0] = 0\n'
    src = src.replace(RUN_HALF, end + RUN_HALF)
    src = src.replace(RUN_OT, '                FAT_LAST["home"] = FAT_LAST["away"] = 0\n                FAT_CUR[0] = 0\n' + RUN_OT)
    end12 = '            FAT_LAST["away" if offense == "home" else "home"] = FAT_CUR[0]\n            FAT_CUR[0] = 0\n'
    src = src.replace(RUN_MAIN, end12 + RUN_MAIN)
    return src


def patch_pick(src):
    assert PICK_W in src and PICK_K in src
    src = src.replace(PICK_W, PICK_W + " * FATW(neighbors)")
    return src.replace(PICK_K, "wkey = (key, off_sim, def_sim, is_home_sim, FAT_ST[0], FAT_ST[1])")


def pool_fatigue():
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = dv.sim.build_transition_frame(pbp)
    d = pbp.drop_duplicates(["game_id", "play_id"]).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    pr = (d["play_type"].isin(["pass", "run"]) & (d["qb_kneel"] != 1) & (d["qb_spike"] != 1)).astype(int)
    d["pr"] = pr
    x = d[d["fixed_drive"].notna() & d["posteam"].notna()]
    t = x.groupby(["game_id", "fixed_drive"]).agg(defn=("defteam", "first"), first=("play_id", "min"), plays=("pr", "sum")).reset_index().sort_values(["game_id", "first"])
    t["prior"] = t.groupby(["game_id", "defn"])["plays"].shift(1)
    d = d.merge(t[["game_id", "fixed_drive", "prior"]], on=["game_id", "fixed_drive"], how="left")
    d["cum"] = d.groupby(["game_id", "defteam"])["pr"].cumsum() - d["pr"]
    m = trans[["game_id", "play_id"]].merge(d[["game_id", "play_id", "prior", "cum"]], on=["game_id", "play_id"], how="left")
    L = m["prior"].fillna(0).to_numpy(np.float64)
    C = m["cum"].fillna(0).to_numpy(np.float64)
    return L, C


def bandwidth(v):
    return float(1.06 * np.std(v) * dv.sim.K_STATE ** -0.2)


def install_fat(cfg):
    ns = dv._G["ns"]
    t = dv._G["tables"]
    L, C = pool_fatigue()
    assert len(L) == len(t["arrays"]["play_type_code"])
    hl = bandwidth(L) if cfg.get("fat") else 0.0
    hc = bandwidth(C) if cfg.get("fatc") else 0.0
    st = [0.0, 0.0]

    def fatw(nb):
        z2 = np.zeros(len(nb))
        if hl:
            z2 = z2 + ((L[nb] - st[0]) / hl) ** 2
        if hc:
            z2 = z2 + ((C[nb] - st[1]) / hc) ** 2
        return np.exp(-0.5 * (z2 - z2.min()))

    ns["FAT_ST"] = st
    ns["FAT_LAST"] = {"home": 0, "away": 0}
    ns["FAT_CUM"] = {"home": 0, "away": 0}
    ns["FAT_CUR"] = [0]
    ns["FATW"] = fatw
    assert "FATW" in ns["pick_index_nn_conditioned"].__code__.co_names
    t["nn_weight_cache_cond"].clear()
    print("fatigue kernel", "hl", hl, "hc", hc, "sdL", float(L.std()), "sdC", float(C.std()), flush=True)


def run_fat(base_init, setting):
    cfg = json.loads(setting["mech"])
    if not (cfg.get("fat") or cfg.get("fatc")):
        base_init(setting)
        return
    orig = inspect.getsource

    def gs(o):
        s = orig(o)
        if o is dv.sim.run_one_game:
            s = patch_run(s)
        elif o is dv.sim.pick_index_nn_conditioned:
            s = patch_pick(s)
        return s

    inspect.getsource = gs
    try:
        base_init(setting)
    finally:
        inspect.getsource = orig
    install_fat(dv._G["cfg"])


def fat_init_budget(setting):
    run_fat(u3d.tfix_init_budget, setting)


def fat_init(setting):
    run_fat(u3d.tfix_init, setting)


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
    gen.init_worker = fat_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    ss.s_init = fat_init
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_analyze(args):
    import argparse as ap_

    u3.OUT = OUT / "analyze"
    u3.bud.OUT = Path(args.post_dir)
    u3.cmd_analyze(ap_.Namespace(minsid=2, boot=args.boot))


def cmd_fit(args):
    L, C = pool_fatigue()
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = dv.sim.build_transition_frame(pbp)
    at = dv.c25.attrs_from(pbp, trans)
    code = trans["play_type_code"].to_numpy()
    m = np.isin(code, (0, 1))
    el = (3600 - trans["gsr_actual"].to_numpy()) / 3600
    dn = trans["down_i"].to_numpy()
    X = np.column_stack([np.ones(len(L)), L / 10, C / 100, el, el**2, dn == 2, dn == 3, dn == 4, np.log(np.maximum(trans["dist_raw"].to_numpy(), 1)), trans["fp_raw"].to_numpy() / 100])[m].astype(float)
    for nm, y in (("yards", np.asarray(at["yards"], dtype=float)), ("epa", np.asarray(at["epa"], dtype=float))):
        b = np.linalg.lstsq(X, y[m], rcond=None)[0]
        print(nm, "prior_len/10", round(b[1], 4), "cum/100", round(b[2], 4), "n", int(m.sum()))
    print("kernel hl", bandwidth(L), "hc", bandwidth(C), "sdL", L.std(), "sdC", C.std())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crzf")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUT / "play_crzf"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzf")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    n = sub.add_parser("analyze")
    n.add_argument("--post-dir", dest="post_dir", default=str(OUT / "play_crzf"))
    n.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    {"fit": cmd_fit, "sim": cmd_sim, "e5": cmd_e5, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
