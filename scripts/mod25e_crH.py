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
    import mod25e_paty as py

    if py.enabled():
        py.install_paty()
    import mod25e_neg as mn

    if mn.enabled():
        mn.install_neg()
    import mod25e_gz as gz

    if gz.enabled() or gz.auditing():
        gz.install_gz()
    if os.environ.get("KICK") in ("0", "1"):
        import mod25e_kick as mk
        import mod25e_risk as rk

        if rk.kick_enabled():
            rk.install_kick()
        else:
            mk.install_kick()
    if dv._G["cfg"].get("f3"):
        f3.f3_install()
    import mod25e_dkern as dk

    if dk.enabled():
        dk.install_dk()
    import mod25e_dkern2 as dk2

    if dk2.enabled():
        dk2.install_dk2()
    import mod25e_clk as ck
    import mod25e_gfl as gf

    if gf.enabled():
        gf.patch()
    if ck.enabled():
        ck.install_ck()
    import mod25e_ckc as kc

    if kc.enabled():
        kc.install_ckc()
    if ck.kn_enabled():
        ck.install_kn()
    import mod25e_sel as sl

    if sl.enabled():
        sl.install_sel()
    import mod25e_kfit as kf

    if kf.enabled():
        kf.install_kf()
    import mod25e_fourth as fd

    if fd.enabled():
        fd.install_fd4()
    import mod25e_tdc as td

    if td.enabled():
        td.install_tdc()
    import mod25e_risk as rk2

    if rk2.tov_enabled():
        rk2.install_tov()
    import mod25e_adj as ad

    if ad.enabled():
        ad.install_adj()
    import mod25e_ylm as ym

    if ym.enabled():
        ym.install_ylm()
    import mod25e_stb as sb

    if sb.enabled():
        sb.install_stb()
    import mod25e_stf as sf

    if sf.enabled():
        sf.install_stf()
    import mod25e_wfg as wf

    if wf.enabled():
        wf.install_wfg()
    import mod25e_fgd as fg

    if fg.enabled():
        fg.install_fgd()
    import mod25e_cdr as cdr

    if cdr.enabled():
        cdr.install_cdr()
    import mod25e_cdt as cdt

    if cdt.enabled():
        cdt.install_cdt()
    import mod25e_cdw as cw

    if cw.enabled():
        cw.install_cdw()
    import mod25e_tpo as tp

    if tp.enabled():
        tp.install_tpo()
    import mod25e_clq as cq

    if cq.enabled():
        cq.install_clq()
    import mod25e_urg as ug

    if ug.enabled():
        ug.install_urg()
    import mod25e_cky as cy

    if cy.enabled():
        cy.install_cky()
    import mod25e_geo as ge

    if ge.enabled():
        ge.install_geo()
    import mod25e_rtd as rt

    if rt.enabled():
        rt.install_rtd()
    import mod25e_cky2 as cy2

    if cy2.enabled():
        cy2.install_cky2()
    import mod25e_urg2 as u2

    if u2.enabled():
        u2.install_urg2()
    install_log_sync(dv)


def install_log_sync(dv):
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        lg = dv._G.get("log")
        n = len(lg) if lg is not None else 0
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if lg and len(lg) == n + 1:
            row = list(lg[-1])
            row[6:11] = [int(drawn["play_type_code"]), float(drawn["points_off"]), float(drawn["points_def"]), bool(drawn["flip"]), float(drawn["clock_elapsed"])]
            lg[-1] = tuple(row)
        return drawn

    dv._G["pol"] = pol


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
    if os.environ.get("DKF") == "1":
        label = label + "d"
    if os.environ.get("DK2") == "1":
        label = label + "e"
    if os.environ.get("CLK") == "1":
        label = label + "c"
    if os.environ.get("SEL") == "1":
        label = label + "s"
    if os.environ.get("KFIT") == "1":
        label = label + "k2"
    if os.environ.get("KN") == "1":
        label = label + "m"
    if os.environ.get("FD4") == "1":
        label = label + "f"
    if os.environ.get("CLK2") == "1":
        label = label + "w"
    if os.environ.get("TDC") == "1":
        label = label + "t"
    if os.environ.get("KGZ") == "1":
        label = label + "j"
    if os.environ.get("OKK") == "1":
        label = label + "o2"
    if os.environ.get("ADJ") == "1":
        label = label + "a"
    if os.environ.get("STF") == "1":
        label = label + "s2"
    if os.environ.get("YLM") == "1":
        label = label + ("yp" if os.environ.get("YLM_CLASSES") == "punt" else "y")
    if os.environ.get("RISK") == "1":
        label = label + "r"
    if os.environ.get("WFG") == "1":
        label = label + "w2"
    if os.environ.get("PATY") == "1":
        label = label + "x"
    if os.environ.get("FGD") == "1":
        label = label + "fd"
    if os.environ.get("CDR") == "1":
        label = label + "cd"
    if os.environ.get("STB") == "1":
        label = label + "tb"
    if os.environ.get("CKC") == "1":
        label = label + "kc"
    if os.environ.get("CKH") == "1":
        label = label + "kh"
    if os.environ.get("GFL") == "1":
        label = label + "gf"
    if os.environ.get("GFL") == "2":
        label = label + "gh"
    if os.environ.get("CKS") == "1" and os.environ.get("CKH") == "1":
        label = label + "ke"
    if os.environ.get("CKU") == "1" and os.environ.get("CKS") == "1" and os.environ.get("CKH") == "1":
        label = label + "ku"
    if os.environ.get("CDW") == "1":
        label = label + "dw"
    if os.environ.get("CDW") == "2":
        label = label + "w2"
    if os.environ.get("TPO") == "1":
        label = label + "tp"
    if os.environ.get("CLQ") == "1":
        label = label + "ql"
    if os.environ.get("CLQ") == "2":
        label = label + "q2"
    if os.environ.get("URG") == "1":
        label = label + "ug"
    if os.environ.get("CKY") == "1":
        label = label + "ky"
    if os.environ.get("CDT") == "1":
        label = label + "dt"
    if os.environ.get("CDT") == "2":
        label = label + "dn"
    if os.environ.get("GEO") == "1":
        label = label + "go"
    if os.environ.get("GEO") == "2":
        label = label + "g2"
    if os.environ.get("RTD") == "1":
        label = label + "rt"
    if os.environ.get("F3W") == "1":
        label = label + "fw"
    if os.environ.get("F3W") == "2":
        label = label + "f2"
    if os.environ.get("GZT") == "1":
        label = label + "gt"
    if os.environ.get("CKY2") == "1":
        label = label + "y2"
    if os.environ.get("URG2") == "1":
        label = label + "u2"
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
