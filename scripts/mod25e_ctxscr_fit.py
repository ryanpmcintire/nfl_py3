import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_ctx as cx
import mod25e_ctx_fit as cf
import sim04_engine as sim

POOL_SCR = cf.SCR / "e177" / "ctx_pool_scr.npz"
FIT = REPO / "artifacts" / "mod25e3" / "ctxscr" / "fit.json"
LOG = REPO / "tests" / "scratch" / "e178" / "ctx_an_SCR.log"
ANS = "tests/scratch/e178/ctx_anSCR.py"
RECORDED = "e178-ctxscr-scramble-weight-on-run-queries"
LAM = cf.LAM
PARENT_PRIOR = cf.PARENT_PRIOR
NDB = 3
LEAD_PHASE = 3
CLIP_LO = 0.0
CLIP_HI = 1.0
FLOOR = 1e-9


def scramble_flag_rows(q):
    pbp = sim.load_reg_seasons(tuple(cf.SEASONS))
    live = pbp[pbp["play_type"].isin(sim.LIVE_TYPES)].dropna(
        subset=["down", "ydstogo", "yardline_100", "score_differential", "qtr", "game_seconds_remaining"]
    )
    live = live.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    flag = ((live["play_type"] == "run") & (live["qb_dropback"] == 1)).to_numpy().astype(float)
    return flag[q]


def load():
    d = cf.load(True)
    P = np.load(POOL_SCR)
    M = np.load(cf.META, allow_pickle=True)
    good = np.ones(len(P["q"]), bool)
    for s in range(len(cf.SEASONS)):
        good &= P["H"][s].sum(axis=1) > 0
    d["HS"] = P["HS"][:, good]
    d["scr"] = scramble_flag_rows(d["q"])
    d["ph"] = M["PH"][d["q"]]
    terr_rg = (d["ph"] == LEAD_PHASE) & (d["code"] == 0) & (d["sc"] >= 1)
    d["mk"] = (d["code"] == 0) & d["inC"] & ~terr_rg
    return d


def fit_rho(d, r, train, hsel):
    nc = r.shape[0]
    idx = np.flatnonzero(np.isin(d["sidx"], list(train)) & d["mk"])
    hs = hsel(d["sidx"][idx])
    cell = d["cell"][idx]
    rf = r[cell]
    hq = d["H"][hs, idx]
    hsq = d["HS"][hs, idx]
    ps = np.clip((rf * hsq).sum(1) / np.maximum((rf * hq).sum(1), FLOOR), CLIP_LO, CLIP_HI)
    sc = d["scr"][idx]
    par = cell % NDB
    n_s = np.bincount(cell, weights=ps, minlength=nc)
    n_d = np.bincount(cell, weights=1 - ps, minlength=nc)
    r_s = np.bincount(cell, weights=sc, minlength=nc)
    r_d = np.bincount(cell, weights=1 - sc, minlength=nc)
    p_s = np.bincount(par, weights=ps, minlength=NDB)
    p_d = np.bincount(par, weights=1 - ps, minlength=NDB)
    q_s = np.bincount(par, weights=sc, minlength=NDB)
    q_d = np.bincount(par, weights=1 - sc, minlength=NDB)
    rp_s = (q_s + PARENT_PRIOR) / (p_s + PARENT_PRIOR)
    rp_d = (q_d + PARENT_PRIOR) / (p_d + PARENT_PRIOR)
    pc = np.arange(nc) % NDB
    rho_s = (r_s + LAM * rp_s[pc]) / (n_s + LAM)
    rho_d = (r_d + LAM * rp_d[pc]) / (n_d + LAM)
    rho_s[nc // 2:] = 1.0
    rho_d[nc // 2:] = 1.0
    return rho_s, rho_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = load()
    seasons = range(len(cf.SEASONS))
    if a.fold is not None:
        si = a.fold - cf.SEASONS[0]
        rest = [u for u in seasons if u != si]
        r = cf.fit(d, rest, lambda t: si)
        rho_s, rho_d = fit_rho(d, r, rest, lambda s_: np.full(len(s_), si))
        np.save(a.out, np.stack([rho_s, rho_d]))
        return
    r = cf.fit(d, seasons, lambda t: t)
    rho_s, rho_d = fit_rho(d, r, seasons, lambda s_: s_)
    spec = dict(
        rho_scramble=rho_s.tolist(), rho_designed=rho_d.tolist(), lam=LAM, parent="run_to_go_bucket",
        ctx_mode="2", seasons=cf.SEASONS,
        provenance=dict(source=ANS, log=str(LOG.relative_to(REPO)).replace("\\", "/"), recorded=RECORDED, pool_queries=int(len(d["q"])), affected_queries=int(d["mk"].sum()), real_scramble_share=float(d["scr"][d["mk"]].mean())),
    )
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    print("wrote", FIT, "cells", len(rho_s), "affected", int(d["mk"].sum()))


if __name__ == "__main__":
    main()
