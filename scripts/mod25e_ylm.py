import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "e94" / "fit.json"
ALL_CLASSES = ("punt", "fgmiss", "to_run", "to_pass", "downs_run", "downs_pass")
CLASSES = tuple(c for c in os.environ.get("YLM_CLASSES", ",".join(ALL_CLASSES)).split(",") if c in ALL_CLASSES)


def enabled():
    return os.environ.get("YLM") == "1"


def class_of(code, down):
    if code == 2:
        return "punt"
    if code == 3:
        return "fgmiss"
    if code in (0, 1):
        return ("downs_" if down >= 4 else "to_") + ("run" if code == 0 else "pass")
    return None


def bounds(cum, c, m):
    if m == 0:
        return 1, 99
    w = 0
    while True:
        lo, hi = max(c - w, 1), min(c + w, 99)
        if cum[hi] - cum[lo - 1] >= m or (lo == 1 and hi == 99):
            return lo, hi
        w += 1


def install_ylm():
    import mod25d_variance as dv

    fit = json.loads(FIT.read_text(encoding="utf-8"))
    a = dv._G["tables"]["arrays"]
    flip = np.asarray(a["possession_flip"], bool) & (np.asarray(a["points_off"]) == 0) & (np.asarray(a["points_def"]) == 0)
    code = np.asarray(a["play_type_code"]).astype(int)
    down = np.asarray(a["down_i"]).astype(int)
    fp = np.clip(np.round(np.asarray(a["fp_raw"], float)).astype(int), 1, 99)
    nyl = np.asarray(a["next_yardline"], float)
    valid = flip & np.isfinite(nyl)
    tabs = {}
    for c in CLASSES:
        m = valid & (code == 2 if c == "punt" else code == 3 if c == "fgmiss" else np.isin(code, (0, 1)) & ((down >= 4) == c.startswith("downs_")) & (code == (0 if c.endswith("run") else 1)))
        rows = np.flatnonzero(m)
        rows = rows[np.argsort(fp[rows], kind="stable")]
        cum = np.cumsum(np.bincount(fp[rows], minlength=100))
        tabs[c] = {"rows": rows, "cum": cum, "edges": np.r_[0, cum]}
    state = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(dn, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        while fr is not None and "offense" not in fr.f_locals:
            fr = fr.f_back
        L = fr.f_locals
        offense, off_to, def_to = L["offense"], L["off_to"], L["def_to"]  # noqa: F841
        qtr_l, gsr, possessions = L.get("qtr"), L.get("gsr"), L.get("possessions")  # noqa: F841
        drawn = base(dn, distance, yardline, score_diff, qtr, clock_val, drawn)
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(9002, int(dv._G["cfg"].get("seed", 3)))
        if not bool(drawn["flip"]) or qtr > 4:
            return drawn
        if float(drawn["points_off"]) != 0.0 or float(drawn["points_def"]) != 0.0:
            return drawn
        c = class_of(int(drawn.get("play_type_code", -1)), int(dn))
        if c is None or c not in tabs:
            return drawn
        T = tabs[c]
        y = int(min(max(round(float(yardline)), 1), 99))
        lo, hi = bounds(T["cum"], y, int(fit[c]))
        a0, b0 = T["edges"][lo], T["edges"][hi + 1]
        if b0 <= a0:
            return drawn
        j = T["rows"][a0 + min(int(state["rng"].random() * (b0 - a0)), b0 - a0 - 1)]
        new = dict(drawn)
        new["next_yardline"] = float(a["next_yardline"][j])
        new["next_down"] = a["next_down"][j]
        new["next_distance"] = a["next_distance"][j]
        return new

    dv._G["pol"] = pol
