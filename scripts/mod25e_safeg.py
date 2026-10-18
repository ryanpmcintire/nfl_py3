import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "artifacts" / "mod25e3" / "safeg"
LOSS = OUT / "loss.npy"
BANDS = ((1, 79), (80, 89), (90, 94), (95, 99))
COUNT = {"calls": 0, "viol": 0, "safe_pick": 0, "nosafe_pick": 0, "none": 0}
ACTIVE = {"on": True, "mode": None}
ORIG = {}


def enabled():
    return os.environ.get("SAFEG") == "1"


def build():
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(tuple(range(2009, 2018)))
    trans = sim.build_transition_frame(pbp).reset_index(drop=True)
    key = pbp[["game_id", "play_id", "yards_gained"]].rename(columns={"yards_gained": "pbp_yards"}).drop_duplicates(["game_id", "play_id"])
    m = trans[["game_id", "play_id"]].merge(key, on=["game_id", "play_id"], how="left")
    loss = -m["pbp_yards"].to_numpy(float)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(LOSS, loss)
    s = (trans["points_def"].to_numpy() == 2) & np.isin(trans["play_type_code"].to_numpy(), (0, 1))
    fp = trans["fp_raw"].to_numpy(float)
    need = 100.0 - fp
    res = {
        "rows": int(len(trans)),
        "scrimmage_safety_rows": int(s.sum()),
        "pbp_missing": int(np.isnan(loss[s]).sum()),
        "transition_yards_mean": float(trans["yards_gained"].to_numpy(float)[s].mean()),
        "pbp_loss_mean": float(np.nanmean(loss[s])),
        "pbp_loss_reaches_goal_line": float((loss[s] >= need[s]).mean()),
        "pbp_loss_ge_need_minus_2": float((loss[s] >= need[s] - 2).mean()),
    }
    print(json.dumps(res, indent=1))
    (OUT / "build.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def pdf2_mask(geo, P):
    m = P.get("pdf2")
    if m is None:
        m = geo.a["points_def"][P["ids"]] == 2
        P["pdf2"] = m
    return m


def install_safeg(loss=None):
    import mod25e_geo as ge

    if getattr(ge.Geo.fix, "_safeg", False):
        return
    if loss is None:
        loss = np.load(LOSS)
    ORIG["violates"], ORIG["candidates"], ORIG["fix"] = ge.Geo.violates, ge.Geo.candidates, ge.Geo.fix
    ORIG["loss"] = loss
    violates0, candidates0, fix0 = ge.Geo.violates, ge.Geo.candidates, ge.Geo.fix

    def reach(i, q):
        return float(loss[int(i)]) >= 100.0 - float(q)

    def is_safety(drawn):
        return float(drawn["points_def"]) == 2.0 and float(drawn["points_off"]) < 6.0

    def violates(self, drawn, i, q):
        if ACTIVE["on"] and is_safety(drawn):
            return not reach(i, q)
        return violates0(self, drawn, i, q)

    def candidates(self, P, q, net_sim, dexact=None):
        mode = ACTIVE["mode"]
        if mode is None or not ACTIVE["on"]:
            return candidates0(self, P, q, net_sim, dexact)
        m = pdf2_mask(self, P)
        if mode == "safe":
            ok = m & (loss[P["ids"]] >= 100.0 - float(q))
            if dexact is not None:
                ok = ok & (np.abs(P["f3"][:, 0] - dexact) < 1e-9)
            return np.flatnonzero(ok)
        pos = candidates0(self, P, q, net_sim, dexact)
        return pos[~m[pos]]

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        if not ACTIVE["on"] or force or not (is_safety(drawn) and violates(self, drawn, i, q)):
            return fix0(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)
        COUNT["calls"] += 1
        COUNT["viol"] += 1
        ACTIVE["mode"] = "safe"
        try:
            out = fix0(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)
            if out is not None:
                COUNT["safe_pick"] += 1
                new, j, shift = out
                new = dict(new)
                new["next_yardline"] = self.a["next_yardline"][int(j)]
                return new, j, shift
            ACTIVE["mode"] = "nosafe"
            out = fix0(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force)
            COUNT["nosafe_pick" if out is not None else "none"] += 1
            return out
        finally:
            ACTIVE["mode"] = None

    fix._safeg = True
    ge.Geo.violates = violates
    ge.Geo.candidates = candidates
    ge.Geo.fix = fix


def check():
    import mod25e_geo as ge
    import sim04_engine as sim

    t0 = time.time()
    pbp = sim.load_reg_seasons(tuple(range(2009, 2018)))
    trans = sim.build_transition_frame(pbp).reset_index(drop=True)
    loss = np.load(LOSS)
    cols = ("down_i", "play_type_code", "fp_raw", "points_off", "points_def", "possession_flip", "yards_gained", "next_down", "next_distance", "next_yardline", "dist_gained", "clock_elapsed", "off_to_used", "def_to_used", "auto_first", "repeat_down", "dist_raw", "sc_raw", "time_raw", "off_to_raw", "def_to_raw", "phase")
    a = {k: trans[k].to_numpy() for k in cols}
    a["off_row"] = np.zeros(len(trans))
    a["def_row"] = np.zeros(len(trans))
    f = sim.feature_matrix(a["dist_raw"], a["fp_raw"], a["sc_raw"], a["time_raw"], a["off_to_raw"], a["def_to_raw"], a["phase"])
    ids = np.flatnonzero(np.isin(a["down_i"], (1, 2, 3, 4)))
    geo = ge.Geo(a, ids, f[ids], 0, sim.K_STATE, sim.TEAM_RATING_YARD_GAIN)
    install_safeg(loss)
    trees = sim.build_neighbor_index_scipy(trans)
    qs = np.flatnonzero(np.isin(a["play_type_code"], (0, 1)) & (a["fp_raw"] <= 99) & np.isin(a["down_i"], (1, 2, 3, 4)))
    ph = a["phase"]
    nb_of = {}
    for (d, p), (tree, sub) in trees.items():
        sel = qs[(a["down_i"][qs] == d) & (ph[qs] == p)]
        if len(sel) == 0:
            continue
        _, ind = tree.query(f[sel], k=min(sim.K_STATE, len(sub)))
        for r, row in zip(sel.tolist(), sub[ind].tolist()):
            nb_of[r] = row
    ones = lambda near: np.ones(len(near))
    band_of = np.full(len(trans), -1)
    for b, (lo, hi) in enumerate(BANDS):
        band_of[(a["fp_raw"] >= lo) & (a["fp_raw"] <= hi)] = b
    n = len(qs)
    arm = {k: np.zeros(n, bool) for k in ("real", "drawn", "base", "safeg")}
    same_nonsafety = 0
    diff_nonsafety = 0
    for u, r in enumerate(qs.tolist()):
        arm["real"][u] = a["points_def"][r] == 2
        nb = nb_of[r]
        pick = np.random.default_rng([7, r]).integers(len(nb))
        j0 = int(nb[pick])
        q = float(a["fp_raw"][r])
        drawn = {k: a[k][j0] for k in ge.REPICK_KEYS}
        drawn["flip"] = bool(a["possession_flip"][j0])
        drawn["auto_first"] = bool(a["auto_first"][j0])
        drawn["repeat_down"] = bool(a["repeat_down"][j0])
        drawn["yards_gained"] = float(a["yards_gained"][j0])
        drawn["idx"] = j0
        arm["drawn"][u] = drawn["points_def"] == 2
        res = {}
        for tag, on in (("base", False), ("safeg", True)):
            ACTIVE["on"] = on
            out = geo.fix(drawn, j0, int(a["down_i"][r]), q, f[r][[0, 2, 3]], 0.0, ones, np.random.default_rng([9, r]))
            dr = out[0] if out is not None else drawn
            res[tag] = dr
            arm[tag][u] = float(dr["points_def"]) == 2
        if drawn["points_def"] != 2:
            if res["base"]["idx"] == res["safeg"]["idx"]:
                same_nonsafety += 1
            else:
                diff_nonsafety += 1
    ACTIVE["on"] = True
    rng = np.random.default_rng(5)
    bq = band_of[qs]
    out = {"queries": int(n), "seconds": round(time.time() - t0), "counts": COUNT, "nonsafety_draws_same_idx": same_nonsafety, "nonsafety_draws_diff_idx": diff_nonsafety, "bands": {}}
    for b, (lo, hi) in enumerate(BANDS):
        m = np.flatnonzero(bq == b)
        row = {"n": int(len(m))}
        for k, v in arm.items():
            x = v[m].astype(float)
            reps = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(300)]
            row[k] = [float(x.mean()), float(np.quantile(reps, 0.05)), float(np.quantile(reps, 0.95))]
        out["bands"][f"yl{lo}-{hi}"] = row
    tot = {}
    for k, v in arm.items():
        tot[k] = float(v.mean())
    out["all_per_play"] = tot
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
