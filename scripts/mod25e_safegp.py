import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "artifacts" / "mod25e3" / "safegp"
SALT = 8137
PUNT = 2
COUNT = {"punts": 0, "safeties": 0, "viol": 0, "repick": 0, "none": 0}


def enabled():
    return os.environ.get("SAFEGP") == "1"


def punt_pool(a, k):
    ok = (a["play_type_code"] == PUNT) & (a["points_def"] == 0) & (a["points_off"] < 6)
    ids = np.flatnonzero(ok)
    o = np.argsort(a["fp_raw"][ids], kind="stable")
    ids = ids[o]
    return {"ids": ids, "fp": a["fp_raw"][ids].astype(float), "k": int(k)}


def reaches(a, i, q):
    return float(a["fp_raw"][int(i)]) <= float(q)


def is_punt_safety(drawn):
    return int(drawn["play_type_code"]) == PUNT and float(drawn["points_def"]) == 2.0 and float(drawn["points_off"]) < 6.0


def replacement(pool, q, rng):
    fp = pool["fp"]
    k = min(pool["k"], len(fp))
    pos = int(np.searchsorted(fp, q))
    lo = max(0, pos - k)
    hi = min(len(fp), pos + k)
    cand = np.arange(lo, hi)
    d = np.abs(fp[cand] - q)
    near = cand[np.argpartition(d, k - 1)[:k]] if k < len(cand) else cand
    return int(pool["ids"][near[int(rng.integers(len(near)))]])


def serve(a, pool, drawn, i, q, rng):
    COUNT["punts"] += 1
    if not is_punt_safety(drawn):
        return drawn
    COUNT["safeties"] += 1
    if reaches(a, i, q):
        return drawn
    COUNT["viol"] += 1
    j = replacement(pool, q, rng)
    if j < 0:
        COUNT["none"] += 1
        return drawn
    COUNT["repick"] += 1
    new = dict(drawn)
    keep = ("clock_elapsed", "off_to_used", "def_to_used")
    for key in ("points_off", "points_def", "next_down", "next_distance", "next_yardline", "dist_gained", "play_type_code"):
        new[key] = a[key][j]
    for key in keep:
        new[key] = drawn[key]
    new["flip"] = bool(a["possession_flip"][j])
    new["auto_first"] = bool(a["auto_first"][j])
    new["repeat_down"] = bool(a["repeat_down"][j])
    new["yards_gained"] = float(a["yards_gained"][j]) + (float(drawn["yards_gained"]) - float(a["yards_gained"][int(i)]))
    new["idx"] = j
    return new


def install_safegp():
    import mod25d_variance as dv
    import sim04_engine as sim

    tables = dv._G["tables"]
    a = tables["arrays"]
    pool = punt_pool(a, sim.K_STATE)
    seed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        i = drawn.get("idx")
        if i is not None and is_punt_safety(drawn):
            if st["k"] != dv._G.get("task_key"):
                st["k"] = dv._G.get("task_key")
                st["rng"] = dv.task_rng(SALT, seed)
            qi = float(np.floor(float(yardline) + 0.5))
            drawn = serve(a, pool, drawn, i, qi, st["rng"])
        return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)

    dv._G["pol"] = pol


def check():
    import sim04_engine as sim

    t0 = time.time()
    pbp = sim.load_reg_seasons(tuple(range(2009, 2018)))
    trans = sim.build_transition_frame(pbp).reset_index(drop=True)
    cols = ("down_i", "play_type_code", "fp_raw", "points_off", "points_def", "possession_flip", "yards_gained", "next_down", "next_distance", "next_yardline", "dist_gained", "clock_elapsed", "off_to_used", "def_to_used", "auto_first", "repeat_down", "dist_raw", "sc_raw", "time_raw", "off_to_raw", "def_to_raw", "phase")
    a = {k: trans[k].to_numpy() for k in cols}
    f = sim.feature_matrix(a["dist_raw"], a["fp_raw"], a["sc_raw"], a["time_raw"], a["off_to_raw"], a["def_to_raw"], a["phase"])
    pool = punt_pool(a, sim.K_STATE)
    trees = sim.build_neighbor_index_scipy(trans)
    qs = np.flatnonzero((a["play_type_code"] == PUNT) & (a["down_i"] == 4))
    ph = a["phase"]
    nb_of = {}
    for (d, p), (tree, sub) in trees.items():
        if d != 4:
            continue
        sel = qs[ph[qs] == p]
        if len(sel) == 0:
            continue
        _, ind = tree.query(f[sel], k=min(sim.K_STATE, len(sub)))
        for r, row in zip(sel.tolist(), sub[ind].tolist()):
            nb_of[r] = row
    punt_nb = {r: np.asarray([x for x in nb if a["play_type_code"][x] == PUNT]) for r, nb in nb_of.items()}
    keys = [r for r in qs.tolist() if len(punt_nb.get(r, ())) > 0]
    rg = np.random.default_rng(11)
    perm = rg.permutation(len(keys))
    bins = ((1, 49), (50, 79), (80, 89), (90, 94), (95, 99))
    n = len(keys)
    arm = {k: np.zeros(n, bool) for k in ("real", "own_drawn", "own_served", "far_drawn", "far_served")}
    changed_nonsafety = 0
    q_of = np.zeros(n)
    for u, r in enumerate(keys):
        q = float(a["fp_raw"][r])
        q_of[u] = q
        arm["real"][u] = a["points_def"][r] == 2
        for tag, src_r in (("own", r), ("far", keys[perm[u]])):
            nb = punt_nb[src_r]
            j0 = int(nb[np.random.default_rng([7, r]).integers(len(nb))])
            drawn = {k: a[k][j0] for k in ("points_off", "points_def", "clock_elapsed", "next_down", "next_distance", "next_yardline", "dist_gained", "off_to_used", "def_to_used", "play_type_code")}
            drawn["flip"] = bool(a["possession_flip"][j0])
            drawn["auto_first"] = bool(a["auto_first"][j0])
            drawn["repeat_down"] = bool(a["repeat_down"][j0])
            drawn["yards_gained"] = float(a["yards_gained"][j0])
            drawn["idx"] = j0
            arm[tag + "_drawn"][u] = drawn["points_def"] == 2
            out = serve(a, pool, drawn, j0, q, np.random.default_rng([9, r]))
            arm[tag + "_served"][u] = float(out["points_def"]) == 2
            if drawn["points_def"] != 2 and out["idx"] != j0:
                changed_nonsafety += 1
    rng = np.random.default_rng(5)
    res = {"punts": int(n), "seconds": round(time.time() - t0), "counts": COUNT, "nonsafety_changed": changed_nonsafety, "per1000": {}}
    for lo, hi in bins:
        m = np.flatnonzero((q_of >= lo) & (q_of <= hi))
        row = {"n": int(len(m))}
        for k, v in arm.items():
            x = v[m].astype(float) * 1000.0
            reps = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(300)]
            row[k] = [round(float(x.mean()), 2), round(float(np.quantile(reps, 0.05)), 2), round(float(np.quantile(reps, 0.95)), 2)]
        res["per1000"][f"yl{lo}-{hi}"] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
