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
import sim09_hk as hk  # noqa: E402

OUT = REPO / "artifacts" / "sim09" / "u4g"
WARN = OUT / "warn.json"
SALT = 4098
MIN_CELL = 40
WARN_AT = {2: 1920.0, 4: 120.0}
FLOOR = {2: 1800.0, 4: 0.0}
dv.DV["u4g"] = dict(dv.DV["crzhk"], tw=1)
ORIG_BUDGET = hk.hk_init_budget
ORIG_SS = hk.hk_init_ss


def cell_key(q, code, ends):
    return f"{q}|{code}|{int(ends)}"


def cmd_fit(args):
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    gid = tr["game_id"].to_numpy()
    gsr = tr["gsr_actual"].to_numpy(dtype=float)
    qtr = tr["qtr_actual"].to_numpy()
    code = tr["play_type_code"].to_numpy()
    ends = tr["possession_flip"].to_numpy().astype(bool) | ((tr["points_off"].to_numpy() + tr["points_def"].to_numpy()) > 0)
    same = np.r_[gid[1:] == gid[:-1], False]
    nxt = np.r_[gsr[1:], np.nan]
    tab = {}
    for q, t in WARN_AT.items():
        cross = same & (qtr == q) & (gsr > t) & (nxt <= t) & (nxt > FLOOR[q])
        stop = cross & (nxt == t)
        tab[f"{q}|all"] = [int(stop.sum()), int(cross.sum())]
        for e in (0, 1):
            m = cross & (ends == bool(e))
            tab[f"{q}|e{e}"] = [int((stop & m).sum()), int(m.sum())]
        for c in range(7):
            for e in (0, 1):
                m = cross & (code == c) & (ends == bool(e))
                tab[cell_key(q, c, e)] = [int((stop & m).sum()), int(m.sum())]
    OUT.mkdir(parents=True, exist_ok=True)
    WARN.write_text(json.dumps(tab, indent=1))
    for k, v in tab.items():
        print(k, v, round(v[0] / v[1], 3) if v[1] else None)


def load_probs():
    tab = json.loads(WARN.read_text())
    out = {}
    for k, (s, n) in tab.items():
        q = k.split("|")[0]
        if k.endswith("|all") or "|e" in k or n >= MIN_CELL:
            out[k] = s / n
        else:
            e = k.split("|")[2]
            out[k] = tab[f"{q}|e{e}"][0] / tab[f"{q}|e{e}"][1]
    return out


def install_tw():
    probs = load_probs()
    base = dv._G["pol"]
    st = {"k": None, "rng": None}
    cseed = int(dv._G["cfg"].get("seed", 3))

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        t = WARN_AT.get(qtr)
        if t is not None and clock_val > t and clock_val - float(drawn["clock_elapsed"]) < t:
            if st["k"] != dv._G.get("task_key"):
                st["k"] = dv._G.get("task_key")
                st["rng"] = dv.task_rng(SALT, cseed)
            ends = bool(drawn["flip"]) or (drawn["points_off"] + drawn["points_def"]) > 0
            if st["rng"].random() < probs[cell_key(qtr, int(drawn["play_type_code"]), ends)]:
                drawn = dict(drawn)
                drawn["clock_elapsed"] = float(clock_val - t)
        return drawn

    dv._G["pol"] = pol


def u4g_init_budget(setting):
    ORIG_BUDGET(setting)
    if dv._G["cfg"].get("tw"):
        install_tw()


def u4g_init_ss(setting):
    ORIG_SS(setting)
    if dv._G["cfg"].get("tw"):
        install_tw()


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
    gen.init_worker = u4g_init_budget
    gen.play_season = hk.u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    ss.s_init = u4g_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_check(args):
    import sim09_u3a as u3
    import sim09_u4e as u4e

    u3.bud.OUT = Path(args.post_dir)
    S = u3.load_sim_frames(2)
    pool = pd.read_parquet(OUT / "pool.parquet")
    for nm, P in (("real", pool.rename(columns={"gid": "g"})), ("sim", S)):
        P = P[P["qtr"] <= 4].reset_index(drop=True)
        P["nx"] = P.groupby("g")["gsr"].shift(-1)
        for q, t in WARN_AT.items():
            w = P[(P["qtr"] == q) & P["nx"].notna()]
            stop = int(((w["gsr"] > t) & ((w["nx"] - t).abs() < 1e-3)).sum())
            cross = int(((w["gsr"] > t) & (w["nx"] < t - 1e-3)).sum())
            f = P[(P["qtr"] == q) & (P["gsr"] <= t)].groupby("g")["gsr"].first()
            print(nm, "q", q, "crossing plays ending exactly at warning", round(stop / max(stop + cross, 1), 3), "first snap gsr at/after warning mean", round(float(f.mean()), 1), "share at warning", round(float(((f - t).abs() < 1e-3).mean()), 3), flush=True)
    sys.argv = ["u4e"]
    u4e.cmd_check(argparse.Namespace(post_dir=args.post_dir, modes="1"))


def cmd_analyze(args):
    hk.cmd_analyze(args)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="u4g")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUT / "play_u4g"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="u4g")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    n = sub.add_parser("analyze")
    n.add_argument("--post-dir", dest="post_dir", default=str(OUT / "play_u4g"))
    n.add_argument("--boot", type=int, default=500)
    c = sub.add_parser("check")
    c.add_argument("--post-dir", dest="post_dir", required=True)
    args = ap.parse_args()
    {"fit": cmd_fit, "sim": cmd_sim, "e5": cmd_e5, "analyze": cmd_analyze, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
