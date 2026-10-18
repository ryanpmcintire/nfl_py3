import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_ctx as cx

SCR = REPO / "tests" / "scratch"
POOL = SCR / "e177" / "ctx_pool.npz"
META = SCR / "e162" / "gp_meta.npz"
LOG = SCR / "e177" / "ctx_an_D.log"
NCL = 12
TD_CLASS = 10
SCORE_EDGES = [-16.5, -8.5, -0.5, 0.5, 8.5, 16.5]
SCORE_MAP = [0, 0, 1, 2, 3, 4, 4]
TIME_EDGES = [60.0, 120.0, 180.0]
PHASE_EDGE = 300.0
GOAL_GUARD = 10.0
SEASONS = list(range(2009, 2018))
MERGE = "S5"
LAM = 5.0
TGT = 1
SMOOTH = 0.5
PARENT_PRIOR = 1.0
SPEC = dict(score_edges=SCORE_EDGES, score_map=SCORE_MAP, time_edges=TIME_EDGES, phase_edge=PHASE_EDGE)


def load():
    P = np.load(POOL)
    M = np.load(META, allow_pickle=True)
    q = P["q"]
    H = P["H"]
    RC = P["RC"]
    ssn = P["season"]
    code = P["code"]
    good = np.ones(len(q), bool)
    for s in range(len(SEASONS)):
        good &= H[s].sum(axis=1) > 0
    q = q[good]
    H = H[:, good]
    RC = RC[good]
    sidx = ssn[good] - SEASONS[0]
    code = code[good]
    d = dict(
        q=q, H=H, RC=RC, sidx=sidx, code=code,
        sc=M["sc"][q], tm=M["tm"][q], qt=M["qtr"][q], dn=M["down"][q], fpq=M["fpq"][q],
    )
    d["cell"] = cx.cell_index(SPEC, code, d["sc"], d["qt"], d["tm"], d["dn"])
    d["par"] = code * 3 + np.where(d["dn"] <= 2, 0, np.where(d["dn"] == 3, 1, 2))
    d["inC"] = (d["dn"] != 4) & (d["fpq"] > GOAL_GUARD)
    return d


def fit(d, train, hsel):
    nc = 2 * (max(SCORE_MAP) + 1) * (len(TIME_EDGES) + 4) * 3
    npar = 6
    R = np.zeros((nc, NCL))
    N = np.zeros((nc, NCL))
    RP = np.zeros((npar, NCL))
    NP_ = np.zeros((npar, NCL))
    for t in train:
        m = np.flatnonzero((d["sidx"] == t) & d["inC"])
        h = d["H"][hsel(t)][m]
        pm = (h + SMOOTH) / (h.sum(axis=1, keepdims=True) + SMOOTH * NCL)
        cell = d["cell"][m]
        par = d["par"][m]
        rc = d["RC"][m]
        R += np.bincount(cell * NCL + rc, minlength=nc * NCL).reshape(nc, NCL)
        RP += np.bincount(par * NCL + rc, minlength=npar * NCL).reshape(npar, NCL)
        for c in range(NCL):
            N[:, c] += np.bincount(cell, weights=pm[:, c], minlength=nc)
            NP_[:, c] += np.bincount(par, weights=pm[:, c], minlength=npar)
    pc = np.zeros(nc, int)
    pc[d["cell"]] = d["par"]
    rp = (RP + PARENT_PRIOR) / (NP_ + PARENT_PRIOR)
    o = (R + LAM * rp[pc]) / (N + LAM)
    o[:, TD_CLASS] = 1.0
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", default=str(cx.FIT))
    a = ap.parse_args()
    log = LOG.read_text(encoding="utf-8")
    modal = ast.literal_eval(re.search(r"modal combo (\(.*\))", log).group(1))
    chosen = ast.literal_eval(re.search(r"^chosen (\[.*\])", log, re.M).group(1))
    assert modal == (MERGE, LAM, TGT), modal
    d = load()
    if a.fold is not None:
        si = a.fold - SEASONS[0]
        rest = [u for u in range(len(SEASONS)) if u != si]
        o = fit(d, rest, lambda t: si)
        np.save(a.out, o)
        return
    o = fit(d, range(len(SEASONS)), lambda t: t)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = dict(SPEC)
    spec.update(
        ratio=o.tolist(), goal_guard=GOAL_GUARD, lam=LAM, merge=MERGE, parent_target=bool(TGT),
        td_class=TD_CLASS, seasons=SEASONS,
        provenance=dict(source="tests/scratch/e177/ctx_anD.py", log="tests/scratch/e177/ctx_an_D.log", fold_chosen=[list(c) for c in chosen], recorded="e177-ctx-ratios-D-td-frozen-rg-kept", pool_queries=int(len(d["q"]))),
    )
    out.write_text(json.dumps(spec), encoding="utf-8")
    print("wrote", out, "cells", o.shape, "queries", len(d["q"]))


if __name__ == "__main__":
    main()
