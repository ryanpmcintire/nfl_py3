import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

os.environ["PASS_MODE"] = "epa"

import mod25d_variance as dv  # noqa: E402
import mod25e_cov as cv  # noqa: E402
import mod25e_crG as crG  # noqa: E402
import mod25e_endgame as eg  # noqa: E402
import mod25e_pass as ps  # noqa: E402
import mod25e_rulestr as rs  # noqa: E402
import sim09_f2 as f2  # noqa: E402
import sim09_f3 as f3  # noqa: E402

PIECES = ("f2", "eg", "w", "f3", "epa")
PATCHED = {"w": False}


def flags(off):
    off = set(off.split(",")) if off else set()
    on_f2 = "f2" not in off
    cfg = dict(dv.DV["crzhk"])
    cfg.update(tw=int(on_f2), f2=int(on_f2), dr=0, eg=0 if ("eg" in off or not on_f2) else 2, f3=int("f3" not in off), pm=[0.0 if "epa" in off else crG.EPA_M, 0.0])
    os.environ["RS_W"] = "0" if ("w" in off or not on_f2) else "1"
    return cfg


def weight_once():
    if os.environ.get("RS_W") == "1" and not PATCHED["w"]:
        f2.install_f2 = rs.weighted_install()
        PATCHED["w"] = True


def H_budget(setting):
    weight_once()
    ps.run_init(eg.eg_init_budget, setting)
    if dv._G["cfg"].get("f3"):
        f3.f3_install()


def H_ss(setting):
    weight_once()
    ps.run_init(eg.eg_init_ss, setting)
    if dv._G["cfg"].get("f3"):
        f3.f3_install()


def install():
    crG.flags = flags
    crG.crG_budget = H_budget
    crG.crG_ss = H_ss


def cmd_sim(a):
    install()
    crG.cmd_sim(a)


def cmd_e5(a):
    import mod25e_scorestate as ss

    install()
    crG.register(a.off)
    dv.DV["crH"] = dict(dv.DV["crG"])
    outd = ss.OUT / f"e5_crH_s{a.seed}"
    outd.mkdir(parents=True, exist_ok=True)
    os.environ["BUD_OUT"] = str(outd)
    ss.DV["crH"] = dv.DV["crH"]
    ss.s_init = H_ss
    a.variant = "crH"
    a.scale = 1.0
    ss.cmd_e5(a)


def cmd_ref(a):
    cv.patch_generator()
    a.variant = "egd"
    a.worlds = 1
    a.seasons = 1
    a.workers = 1
    eg.cmd_sim(a)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("sim", "e5"):
        s = sub.add_parser(nm)
        s.add_argument("--off", default="")
        s.add_argument("--worlds", type=int, default=1 if nm == "sim" else 8)
        s.add_argument("--seasons", type=int, default=1 if nm == "sim" else 8)
        s.add_argument("--workers", type=int, default=1)
        s.add_argument("--seed", type=int, default=31 if nm == "sim" else 11)
        if nm == "sim":
            s.add_argument("--out-dir", dest="out_dir", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--seed", type=int, default=31)
    r.add_argument("--out-dir", dest="out_dir", required=True)
    a = ap.parse_args()
    {"sim": cmd_sim, "e5": cmd_e5, "ref": cmd_ref}[a.cmd](a)


if __name__ == "__main__":
    main()
