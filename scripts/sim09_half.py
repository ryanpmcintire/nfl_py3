import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402

OUTD = REPO / "artifacts" / "sim09" / "u3c"
SECOND_HALF_PHASE = 5
dv.DV["crzh"] = dict(dv.DV["crz"], half=1)


def split_trees(sim, trans, trees):
    down_arr = trans["down_i"].to_numpy()
    phase_arr = trans["phase"].to_numpy()
    qtr = trans["qtr_actual"].to_numpy()
    feats = sim.feature_matrix(
        trans["dist_raw"].to_numpy(),
        trans["fp_raw"].to_numpy(),
        trans["sc_raw"].to_numpy(),
        trans["time_raw"].to_numpy(),
        trans["off_to_raw"].to_numpy(),
        trans["def_to_raw"].to_numpy(),
        phase_arr,
    )
    out = dict(trees)
    for down in (1, 2, 3, 4):
        base = (down_arr == down) & (phase_arr == 0)
        for ph, m in ((0, base & (qtr <= 2)), (SECOND_HALF_PHASE, base & (qtr == 3))):
            sub = np.flatnonzero(m)
            if len(sub):
                out[(down, ph)] = (cKDTree(feats[sub]), sub)
    return out


def install_half_index():
    sim = dv.sim
    if getattr(sim, "_HALF_INDEX", False):
        return
    orig = sim.build_neighbor_index_scipy

    def wrapped(trans):
        return split_trees(sim, trans, orig(trans))

    sim.build_neighbor_index_scipy = wrapped
    sim._HALF_INDEX = True


def install_half_pick():
    ns = dv._G["ns"]
    base = ns["pick_index_nn_conditioned"]
    assert all((d, SECOND_HALF_PHASE) in dv._G["tables"]["nn_trees_cond"] for d in (1, 2, 3, 4))

    def pick(rng, tbl, down, phase, *rest):
        if phase == 0 and int(sys._getframe(1).f_locals["qtr"]) == 3:
            phase = SECOND_HALF_PHASE
        return base(rng, tbl, down, phase, *rest)

    ns["pick_index_nn_conditioned"] = pick


def half_init_budget(setting):
    cfg = json.loads(setting["mech"])
    if cfg.get("half"):
        install_half_index()
    import mod25e_budget as bud

    bud.e_init(setting)
    if dv._G["cfg"].get("half"):
        install_half_pick()


def half_init_ss(setting):
    cfg = json.loads(setting["mech"])
    if cfg.get("half"):
        install_half_index()
    import mod25e_scorestate as ss

    ss.s_init(setting)
    if dv._G["cfg"].get("half"):
        install_half_pick()


def cmd_sim(args):
    import mod25_generator as gen
    import mod25e_budget as bud
    import sim09_u3a as u3

    bud.OUT = Path(args.out_dir).resolve()
    os.environ["BUD_OUT"] = str(bud.OUT)
    bud.OUT.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = half_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    ss.s_init = half_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_analyze(args):
    import argparse as ap_
    import mod25e_budget as bud
    import sim09_u3a as u3

    u3.OUT = OUTD / "analyze"
    u3.OUT.mkdir(parents=True, exist_ok=True)
    bud.OUT = Path(args.post_dir)
    u3.cmd_analyze(ap_.Namespace(minsid=2, boot=args.boot))


def cmd_cover(args):
    sim = dv.sim
    sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = sim.build_transition_frame(pbp).reset_index(drop=True)
    qtr = trans["qtr_actual"].to_numpy()
    ph = trans["phase"].to_numpy()
    dn = trans["down_i"].to_numpy()
    rows = []
    for d in (1, 2, 3, 4):
        base = (dn == d) & (ph == 0)
        rows.append({"down": d, "old_phase0_pool": int(base.sum()), "first_half_pool": int((base & (qtr <= 2)).sum()), "second_half_q3_pool": int((base & (qtr == 3)).sum()), "q2_early": int((base & (qtr == 2)).sum()), "q1": int((base & (qtr == 1)).sum())})
    for r in rows:
        print(r)
    print("K_STATE", sim.K_STATE, "min split pool", min(min(r["first_half_pool"], r["second_half_q3_pool"]) for r in rows), "cells with pool < K:", sum(1 for r in rows for k in ("first_half_pool", "second_half_q3_pool") if r[k] < sim.K_STATE))
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "coverage.json").write_text(json.dumps({"rows": rows, "K_STATE": sim.K_STATE}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crzh")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUTD / "play_crzh"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzh")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    a = sub.add_parser("analyze")
    a.add_argument("--post-dir", dest="post_dir", default=str(OUTD / "play_crzh"))
    a.add_argument("--boot", type=int, default=500)
    sub.add_parser("cover")
    args = ap.parse_args()
    {"sim": cmd_sim, "e5": cmd_e5, "analyze": cmd_analyze, "cover": cmd_cover}[args.cmd](args)


if __name__ == "__main__":
    main()
