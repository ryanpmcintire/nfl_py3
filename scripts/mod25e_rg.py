import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "rg" / "fit.json"
YARD_EDGES = np.array([0.5, 2.5, 5.5, 9.5])
NCLS = 12
COUNT = {"calls": 0}


def enabled():
    return os.environ.get("RGR") == "1"


def row_classes(a, ids, q, dist):
    ids = np.asarray(ids)
    y = a["yards_gained"][ids].astype(float)
    td = a["points_off"][ids] >= 6
    fl = a["possession_flip"][ids].astype(bool)
    fr = (fl | (a["points_def"][ids] > 0)) & ~td
    srvtd = td | (~td & ~fr & (y >= q))
    conv = srvtd | a["auto_first"][ids].astype(bool) | ((~a["repeat_down"][ids].astype(bool)) & (y >= dist) & ~fl)
    turn = ~srvtd & (fr | fl)
    return np.where(srvtd, 10, np.where(turn, 11, np.searchsorted(YARD_EDGES, y) * 2 + (conv & ~srvtd).astype(int)))


def cell_of(spec, lead, gsr):
    lb = int(np.digitize(lead, spec["lead_edges"])) - 1
    tb = int(np.digitize(gsr, spec["time_edges"]))
    return lb * (len(spec["time_edges"]) + 1) + tb


def install_rg():
    import mod25d_variance as dv
    import mod25e_geo as ge
    import sim04_engine as sim

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    arrays = dv._G["tables"]["arrays"]
    ratio = np.asarray(spec["ratio"])
    ctx = {"cell": None, "dist": 0.0}
    orig_fix = ge.Geo.fix

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        cell = ctx["cell"]
        if cell is None or int(drawn["play_type_code"]) != 0:
            return orig_fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)

        def wf(near):
            COUNT["calls"] += 1
            return weight_fn(near) * ratio[cell][row_classes(arrays, near, q, ctx["dist"])]

        return orig_fix(self, drawn, i, down, q, f3q, net_sim, wf, rng, force)

    ge.Geo.fix = fix
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        if sim.compute_phase(qtr, clock_val) == 3 and score_diff >= 1:
            tf = sim.continuous_time_feature(qtr, clock_val)
            ctx["cell"] = cell_of(spec, score_diff, tf)
            ctx["dist"] = float(distance)
        else:
            ctx["cell"] = None
        try:
            return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        finally:
            ctx["cell"] = None

    dv._G["pol"] = pol
