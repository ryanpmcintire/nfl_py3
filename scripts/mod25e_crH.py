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
import mod25e_ot as ot  # noqa: E402
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
    cfg.update(tw=int(on_f2), f2=int(on_f2), dr=0, eg=0 if ("eg" in off or not on_f2) else (4 if os.environ.get("EGT") == "1" and os.environ.get("EGH") == "1" else (3 if os.environ.get("EGT") == "1" else 2)), f3=int("f3" not in off), pm=[0.0 if "epa" in off else crG.EPA_M, 0.0])
    if os.environ.get("KICK") in ("0", "1"):
        cfg["kick"] = int(os.environ["KICK"])
    os.environ["RS_W"] = "0" if ("w" in off or not on_f2) else "1"
    return cfg


def weight_once():
    if os.environ.get("RS_W") == "1" and not PATCHED["w"]:
        f2.install_f2 = rs.weighted_install()
        PATCHED["w"] = True


def run_with_flags(init, setting):
    weight_once()
    if ot.enabled():
        orig, seen = ot.hook_source()
    try:
        ps.run_init(init, setting)
    finally:
        if ot.enabled():
            import inspect

            inspect.getsource = orig
    if ot.enabled():
        assert seen["n"] >= 1
        ot.install(dv)
    import mod25e_neg as mn

    if mn.enabled():
        mn.install_neg()
    import mod25e_gz as gz

    if gz.enabled() or gz.auditing():
        gz.install_gz()
    if os.environ.get("KICK") in ("0", "1"):
        import mod25e_kick as mk

        mk.install_kick()
    if dv._G["cfg"].get("f3"):
        f3.f3_install()


def H_budget(setting):
    run_with_flags(eg.eg_init_budget, setting)


def H_ss(setting):
    run_with_flags(eg.eg_init_ss, setting)


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
    label = "crHp" if os.environ.get("F2PR") == "1" and os.environ.get("EGH") == "1" and os.environ.get("EGT") == "1" else "crHh" if os.environ.get("EGH") == "1" and os.environ.get("EGT") == "1" else ("crHt" if os.environ.get("EGT") == "1" else "crH")
    if os.environ.get("QBC") == "1":
        label = label + "q"
    if os.environ.get("OTY"):
        label = label + "o"
    if os.environ.get("KICK") in ("0", "1"):
        label = label + "k"
    if os.environ.get("GZ") == "1":
        label = label + "g"
    if os.environ.get("NEG") == "1":
        label = label + "n"
    dv.DV[label] = dict(dv.DV["crG"])
    outd = ss.OUT / f"e5_{label}_s{a.seed}"
    outd.mkdir(parents=True, exist_ok=True)
    os.environ["BUD_OUT"] = str(outd)
    ss.DV[label] = dv.DV[label]
    ss.s_init = H_ss
    a.variant = label
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
