import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def enabled():
    return os.environ.get("STBK") == "1" and os.environ.get("STB") == "1"


def hook_shape(po, pdf):
    return (po >= 3 and pdf == 0) or (pdf >= 6 and po == 0) or (po >= 3 and pdf >= 6)


def install_stbk():
    import mod25d_variance as dv
    import mod25e_stb as sb

    cfg_kick = int(dv._G["cfg"].get("kick", 1))
    inner_base = dv._G["pol"]
    last = {"out": None, "drawn_by_kick": False}

    def inner(dn, distance, yardline, score_diff, qtr, clock_val, drawn):
        out = inner_base(dn, distance, yardline, score_diff, qtr, clock_val, drawn)
        last["out"] = out
        last["drawn_by_kick"] = cfg_kick != 0 and qtr <= 4 and bool(out["flip"]) and hook_shape(float(out["points_off"]), float(out["points_def"]))
        return out

    dv._G["pol"] = inner
    sb.install_stb()
    stb_pol = dv._G["pol"]

    def pol(dn, distance, yardline, score_diff, qtr, clock_val, drawn):
        res = stb_pol(dn, distance, yardline, score_diff, qtr, clock_val, drawn)
        if last["drawn_by_kick"]:
            return last["out"]
        return res

    dv._G["pol"] = pol
