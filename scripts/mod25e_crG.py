import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

os.environ["PASS_MODE"] = "epa"

import mod25d_variance as dv  # noqa: E402
import mod25e_cov as cv  # noqa: E402
import mod25e_drill as dr  # noqa: E402
import mod25e_pass as ps  # noqa: E402
import sim09_f2 as f2  # noqa: E402
import sim09_f3 as f3  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3" / "crG"
PIECES = ("f2", "dr", "f3", "epa")
EPA_M = 0.03


def flags(off):
    off = set(off.split(",")) if off else set()
    on_f2 = "f2" not in off
    cfg = dict(dv.DV["crzhk"])
    cfg.update(tw=int(on_f2), f2=int(on_f2), dr=int("dr" not in off), f3=int("f3" not in off), pm=[0.0 if "epa" in off else EPA_M, 0.0])
    return cfg


def register(off=""):
    cv.patch_generator()
    dv.DV["crG"] = flags(off)


def crG_budget(setting):
    ps.run_init(dr.drl_init_budget, setting)
    if dv._G["cfg"].get("f3"):
        f3.f3_install()


def crG_ss(setting):
    ps.run_init(dr.drl_init_ss, setting)
    if dv._G["cfg"].get("f3"):
        f3.f3_install()


def crG_play_season(task):
    return dv.d_play_season(task)


def cmd_sim(a):
    import mod25_generator as gen
    import mod25e_budget as bud
    import sim09_u3a as u3

    register(a.off)
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bud.OUT = out
    os.environ["BUD_OUT"] = str(out)
    dv.c25.ensure_policies()
    v = dv.DV["crG"]
    cfgj = json.dumps(dict(v, seed=a.seed, name=f"crG_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = crG_budget
    gen.play_season = crG_play_season if v.get("f3") else u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, a.worlds, a.seasons, a.workers, a.seed, progress=False)
    games.to_parquet(out / "sim_games.parquet")
    arrs = {}
    for (w, sd), (weekly, qb_out) in latents.items():
        arrs[f"w_{w}_{sd}"] = weekly
        arrs[f"q_{w}_{sd}"] = qb_out
    np.savez(out / "latents.npz", **arrs)
    print("done", len(games), el, flush=True)


def cmd_e5(a):
    import mod25e_scorestate as ss

    register(a.off)
    outd = ss.OUT / f"e5_crG_s{a.seed}"
    outd.mkdir(parents=True, exist_ok=True)
    os.environ["BUD_OUT"] = str(outd)
    ss.DV["crG"] = dv.DV["crG"]
    ss.s_init = crG_ss
    a.variant = "crG"
    a.scale = 1.0
    ss.cmd_e5(a)


def cmd_ref(a):
    import mod25e_budget as bud

    cv.patch_generator()
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    a.variant = "f2" if a.piece == "f2" else "crzf3"
    if a.piece == "f3":
        f3.cmd_sim(a)
    else:
        f2.cmd_sim(a)


def cmd_cmp(a):
    import pandas as pd

    x = pd.read_parquet(Path(a.a) / "sim_games.parquet")
    y = pd.read_parquet(Path(a.b) / "sim_games.parquet")
    print(a.a, a.b, "rows", len(x), len(y), "equal", x.equals(y))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("sim", "e5", "ref"):
        s = sub.add_parser(nm)
        s.add_argument("--off", default="")
        s.add_argument("--worlds", type=int, default=1 if nm != "e5" else 8)
        s.add_argument("--seasons", type=int, default=1 if nm != "e5" else 8)
        s.add_argument("--workers", type=int, default=1)
        s.add_argument("--seed", type=int, default=31 if nm != "e5" else 11)
        if nm != "e5":
            s.add_argument("--out-dir", dest="out_dir", required=True)
        if nm == "ref":
            s.add_argument("--piece", choices=("f2", "f3"), required=True)
    c = sub.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    {"sim": cmd_sim, "e5": cmd_e5, "ref": cmd_ref, "cmp": cmd_cmp}[a.cmd](a)


if __name__ == "__main__":
    main()
