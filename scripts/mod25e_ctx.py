import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from mod25e_rg import row_classes

FITS = {"1": REPO / "artifacts" / "mod25e3" / "ctx" / "fit.json", "2": REPO / "artifacts" / "mod25e3" / "ctx" / "fit_F.json"}
FIT = FITS["1"]
COUNT = {"calls": 0}


def enabled():
    return os.environ.get("CTX") in FITS


def cell_index(spec, code, sc, qtr, tm, dn, dist=None):
    code = np.asarray(code)
    sc = np.asarray(sc, dtype=float)
    qtr = np.asarray(qtr)
    tm = np.asarray(tm, dtype=float)
    dn = np.asarray(dn)
    smap = np.asarray(spec["score_map"])
    sb = smap[np.digitize(sc, spec["score_edges"])]
    ns = int(smap.max()) + 1
    np_ = len(spec["time_edges"]) + 4
    nd = 3
    late = 3 + np.digitize(tm, spec["time_edges"])
    per = np.where(qtr <= 2, 0, np.where(qtr == 3, 1, np.where((qtr == 4) & (tm > spec["phase_edge"]), 2, late)))
    d3 = np.where(dn <= 2, 0, np.where(dn == 3, 1, 2))
    cell = ((code * ns + sb) * np_ + per) * nd + d3
    edges = spec.get("dist_edges")
    if edges:
        cell = cell * (len(edges) + 1) + np.digitize(np.asarray(dist, dtype=float), edges, right=True)
    return cell


def install_ctx():
    import mod25d_variance as dv
    import mod25e_geo as ge
    import sim04_engine as sim

    if os.environ.get("RGT") == "1":
        raise RuntimeError("CTX replaces RGT; unset RGT")
    path = FITS[os.environ["CTX"]] if FIT is FITS["1"] else FIT
    spec = json.loads(path.read_text(encoding="utf-8"))
    arrays = dv._G["tables"]["arrays"]
    ratio = np.asarray(spec["ratio"])
    guard = spec["goal_guard"]
    ctx = {"cells": None, "dist": 0.0, "lead_late": False}
    orig_fix = ge.Geo.fix

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        cells = ctx["cells"]
        code = int(drawn["play_type_code"])
        if cells is None or code not in (0, 1):
            return orig_fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)
        if (ctx["lead_late"] and code == 0) or int(down) == 4 or q <= guard:
            return orig_fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)
        row = ratio[cells[code]]

        def wf(near):
            COUNT["calls"] += 1
            return weight_fn(near) * row[row_classes(arrays, near, q, ctx["dist"])]

        return orig_fix(self, drawn, i, down, q, f3q, net_sim, wf, rng, force)

    ge.Geo.fix = fix
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        tf = sim.continuous_time_feature(qtr, clock_val)
        rgt_terr = sim.compute_phase(qtr, clock_val) == 3 and score_diff <= -1
        ctx["cells"] = tuple(int(cell_index(spec, c, score_diff, qtr, tf, down, distance if c == 0 and not rgt_terr else 0.0)) for c in (0, 1))
        ctx["dist"] = float(distance)
        ctx["lead_late"] = sim.compute_phase(qtr, clock_val) == 3 and score_diff >= 1
        try:
            return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        finally:
            ctx["cells"] = None

    dv._G["pol"] = pol
