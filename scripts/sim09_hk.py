import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import sim09_half as hf  # noqa: E402
import sim09_u3a as u3  # noqa: E402
import sim09_u3d as kd  # noqa: E402

OUTD = REPO / "artifacts" / "sim09" / "u3e"
dv.DV["crzhk"] = dict(dv.DV["crz"], half=1, tfix=1)


def hk_init_budget(setting):
    hf.half_init_budget(setting)
    if dv._G["cfg"].get("tfix"):
        kd.install_tfix()


def hk_init_ss(setting):
    hf.half_init_ss(setting)
    if dv._G["cfg"].get("tfix"):
        kd.install_tfix()


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
    gen.init_worker = hk_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    ss.s_init = hk_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_analyze(args):
    u3.OUT = OUTD / "analyze_crzhk"
    u3.OUT.mkdir(parents=True, exist_ok=True)
    u3.bud.OUT = Path(args.post_dir)
    u3.cmd_analyze(argparse.Namespace(minsid=2, boot=args.boot))


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crzhk")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUTD / "play_crzhk"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzhk")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    n = sub.add_parser("analyze")
    n.add_argument("--post-dir", dest="post_dir", default=str(OUTD / "play_crzhk"))
    n.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    {"sim": cmd_sim, "e5": cmd_e5, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
