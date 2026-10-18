import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from mod25e_rg import row_classes

FIT = REPO / "artifacts" / "mod25e3" / "rgt" / "fit.json"
COUNT = {"calls": 0}


def enabled():
    return os.environ.get("RGT") == "1"


def cell_of(spec, code, depth, gsr):
    nd = len(spec["lead_edges"])
    nt = len(spec["time_edges"]) + 1
    lb = int(np.digitize(depth, spec["lead_edges"])) - 1
    tb = int(np.digitize(gsr, spec["time_edges"]))
    return code * nd * nt + lb * nt + tb


def install_rgt():
    import mod25d_variance as dv
    import mod25e_geo as ge
    import sim04_engine as sim

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    arrays = dv._G["tables"]["arrays"]
    ratio = np.asarray(spec["ratio"])
    ctx = {"tcell": None, "dist": 0.0}
    orig_fix = ge.Geo.fix

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        tc = ctx["tcell"]
        code = int(drawn["play_type_code"])
        if tc is None or code not in (0, 1):
            return orig_fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)
        row = ratio[tc[code]]

        def wf(near):
            COUNT["calls"] += 1
            return weight_fn(near) * row[row_classes(arrays, near, q, ctx["dist"])]

        return orig_fix(self, drawn, i, down, q, f3q, net_sim, wf, rng, force)

    ge.Geo.fix = fix
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        if sim.compute_phase(qtr, clock_val) == 3 and score_diff <= -1:
            tf = sim.continuous_time_feature(qtr, clock_val)
            ctx["tcell"] = (cell_of(spec, 0, -score_diff, tf), cell_of(spec, 1, -score_diff, tf))
            ctx["dist"] = float(distance)
        else:
            ctx["tcell"] = None
        try:
            return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        finally:
            ctx["tcell"] = None

    dv._G["pol"] = pol
