import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_ctx as cx

FIT = REPO / "artifacts" / "mod25e3" / "ctxscr" / "fit.json"
COUNT = {"calls": 0}
RUN_CODE = 0


def enabled():
    return os.environ.get("CTXSCR") == "1"


def scramble_flags(arrays):
    import sim04_engine as sim

    gid = arrays["game_id"]
    seasons = tuple(sorted({int(str(g)[:4]) for g in np.unique(gid)}))
    pbp = sim.load_reg_seasons(seasons)
    key = pbp[["game_id", "play_id", "play_type", "qb_dropback"]].copy()
    key["scr"] = ((key["play_type"] == "run") & (key["qb_dropback"] == 1)).astype(np.int8)
    key = key.drop_duplicates(["game_id", "play_id"])[["game_id", "play_id", "scr"]]
    base = pd.DataFrame({"game_id": gid, "play_id": arrays["play_id"]})
    merged = base.merge(key, on=["game_id", "play_id"], how="left")
    assert len(merged) == len(gid)
    return merged["scr"].fillna(0).to_numpy().astype(np.int8)


def install_ctxscr():
    import mod25d_variance as dv
    import mod25e_geo as ge
    import sim04_engine as sim

    if not cx.enabled():
        raise RuntimeError("CTXSCR requires CTX")
    path = cx.FITS[os.environ["CTX"]] if cx.FIT is cx.FITS["1"] else cx.FIT
    cspec = json.loads(path.read_text(encoding="utf-8"))
    if not cspec.get("dist_edges"):
        raise RuntimeError("CTXSCR requires the run-distance CTX fit")
    spec = json.loads(FIT.read_text(encoding="utf-8"))
    rho_s = np.asarray(spec["rho_scramble"])
    rho_d = np.asarray(spec["rho_designed"])
    guard = cspec["goal_guard"]
    arrays = dv._G["tables"]["arrays"]
    flags = scramble_flags(arrays)
    state = {"cell": None, "lead_late": False}
    prev_fix = ge.Geo.fix

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        cell = state["cell"]
        if cell is None or int(drawn["play_type_code"]) != RUN_CODE or state["lead_late"] or int(down) == 4 or q <= guard:
            return prev_fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)

        def wf(near):
            COUNT["calls"] += 1
            return weight_fn(near) * np.where(flags[near] == 1, rho_s[cell], rho_d[cell])

        return prev_fix(self, drawn, i, down, q, f3q, net_sim, wf, rng, force)

    ge.Geo.fix = fix
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        tf = sim.continuous_time_feature(qtr, clock_val)
        rgt_terr = sim.compute_phase(qtr, clock_val) == 3 and score_diff <= -1
        state["cell"] = int(cx.cell_index(cspec, RUN_CODE, score_diff, qtr, tf, down, 0.0 if rgt_terr else distance))
        state["lead_late"] = sim.compute_phase(qtr, clock_val) == 3 and score_diff >= 1
        try:
            return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        finally:
            state["cell"] = None

    dv._G["pol"] = pol
