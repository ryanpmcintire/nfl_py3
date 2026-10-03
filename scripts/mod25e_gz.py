import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "mod25e3" / "gz"
RUN_PASS = (0, 1)


def enabled():
    return os.environ.get("GZ") == "1"


def auditing():
    return bool(os.environ.get("GZA"))


def install_gz():
    import mod25d_variance as dv
    import sim04_engine as sim

    ns = dv._G["ns"]
    tables = dv._G["tables"]
    arrays = tables["arrays"]
    seed = int(dv._G["cfg"].get("seed", 3))
    fix = enabled()
    rows = {}
    for tree, sub in tables["nn_trees_cond"].values():
        rows.update(zip(sub.tolist(), tree.data))
    ids = np.array(sorted(rows), dtype=np.int64)
    dat = np.array([rows[i] for i in ids.tolist()])
    dist = dat[:, 0] * sim.SCALE_YDSTOGO
    fp = arrays["fp_raw"][ids]
    gtg = arrays["dist_raw"][ids] >= fp
    down_i = arrays["down_i"][ids]
    code = arrays["play_type_code"][ids]
    pools = {}
    for d in (1, 2, 3, 4):
        for c in RUN_PASS:
            for y in np.unique(fp[gtg & (down_i == d) & (code == c)]):
                m = gtg & (down_i == d) & (code == c) & (fp == y)
                pools[(d, c, float(y))] = (ids[m], dat[m])
    del dist
    st = {"k": None, "rng": None}
    stats = {}
    tag = os.environ.get("GZA", "fix" if fix else "base")
    base = dv._G["pol"]

    def bump(y, key, v=1.0):
        s = stats.setdefault(str(int(y)), {})
        s[key] = s.get(key, 0.0) + v

    def snap(yardline, drawn):
        y = str(int(yardline))
        c = int(drawn["play_type_code"])
        s = stats.setdefault("S", {}).setdefault(y, {})
        s[f"n{c}"] = s.get(f"n{c}", 0) + 1
        if float(drawn["points_off"]) >= 6 and c in RUN_PASS:
            s[f"td{c}"] = s.get(f"td{c}", 0) + 1

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        i = drawn.get("idx")
        if i is not None:
            drawn = pol_inner(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if auditing() and yardline <= 15.0:
            snap(yardline, drawn)
        return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)

    def pol_inner(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        i = drawn.get("idx")
        if i is None or distance < yardline or int(drawn["play_type_code"]) not in RUN_PASS:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(8128, seed)
        rng = st["rng"]
        i0 = i
        y = float(yardline)
        d = down if down in (1, 2, 3, 4) else 4
        pool = pools.get((d, int(drawn["play_type_code"]), float(round(y))))
        if fix and pool is not None:
            pid, pdat = pool
            tf = sim.continuous_time_feature(qtr, clock_val)
            f = sim.feature_matrix(
                np.array([distance]),
                np.array([y]),
                np.array([score_diff]),
                np.array([tf]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([sim.compute_phase(qtr, clock_val)]),
            )[0]
            dd = ((pdat[:, [0, 2, 3]] - f[[0, 2, 3]]) ** 2).sum(axis=1)
            k = min(sim.K_STATE, len(pid))
            near = pid[np.argpartition(dd, k - 1)[:k]] if k < len(pid) else pid
            fr = sys._getframe(1)
            while fr is not None and "off_sim" not in fr.f_locals:
                fr = fr.f_back
            L = fr.f_locals
            h = tables["team_kernel_h"]
            d2 = (arrays["off_row"][near] - L["off_sim"]) ** 2 + (arrays["def_row"][near] - L["def_sim"]) ** 2
            w = np.exp(-d2 / (2.0 * h * h)) * np.where(arrays["is_home_off"][near] == L["is_home_sim"], ns["TEAM_KERNEL_LAMBDA"], 1.0)
            if "IPW" in ns:
                w = w * np.asarray(ns["IPW"])[near]
            if "PASS_W" in ns:
                w = w * ns["PASS_W"](None, near, L["off_sim"], L["def_sim"])
            c = np.cumsum(w)
            if not np.isfinite(c[-1]) or c[-1] <= 0.0:
                j = int(near[int(rng.integers(len(near)))])
            else:
                j = int(near[min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), len(near) - 1)])
            shift = float(drawn["yards_gained"]) - float(arrays["yards_gained"][int(i)])
            new = dict(drawn)
            for key in ("points_off", "points_def", "clock_elapsed", "next_down", "next_distance", "next_yardline", "dist_gained", "off_to_used", "def_to_used", "play_type_code"):
                new[key] = arrays[key][j]
            new["flip"] = bool(arrays["possession_flip"][j])
            new["auto_first"] = bool(arrays["auto_first"][j])
            new["repeat_down"] = bool(arrays["repeat_down"][j])
            new["yards_gained"] = arrays["yards_gained"][j] + shift
            new["idx"] = int(i0)
            st["shift"] = shift
            if auditing():
                bump(y, "repicked")
            drawn = new
            i = j
        if auditing():
            bump(y, "n")
            bump(y, "src_fp_diff", float(arrays["fp_raw"][int(i)] != round(y)))
            bump(y, "src_fp_gap", float(abs(arrays["fp_raw"][int(i)] - y)))
            scored = float(drawn["points_off"]) > 0 or float(drawn["points_def"]) > 0
            if not scored and not drawn["flip"]:
                bump(y, "nonscore")
                bump(y, "clipped", float(float(drawn["yards_gained"]) > y - 1.0))
                bump(y, "lost_yards", max(0.0, float(drawn["yards_gained"]) - (y - 1.0)))
            if float(drawn["points_off"]) >= 6:
                bump(y, "td")
        return drawn

    class PL(list):
        def append(self, v):
            sh = st.pop("shift", None)
            super().append(v if sh is None else (v[0], v[1], sh))

    if fix:
        ns["PLAYLOG"] = PL(ns["PLAYLOG"])
    dv._G["pol"] = pol
    if auditing():
        orig = ns["run_one_game"]
        OUT.mkdir(parents=True, exist_ok=True)

        def wrapped(*a, **k):
            r = orig(*a, **k)
            stats["games"] = stats.get("games", 0) + 1
            (OUT / f"audit_{tag}_{os.getpid()}.json").write_text(json.dumps(stats))
            return r

        ns["run_one_game"] = wrapped
