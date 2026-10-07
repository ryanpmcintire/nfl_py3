import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
OUTD = REPO / "artifacts" / "mod25e3" / "fgd"
FIT = OUTD / "fit.json"
TRAIN = tuple(range(2009, 2018))
SALT = 9801
FG_CODE = 3
KICK_SNAP_YARDS = 17.0
STATS = {"eligible": 0, "swapped": 0}


def enabled():
    return os.environ.get("FGD") == "1"


def fg_rows(t):
    d = t[(t.play_type_code == FG_CODE)]
    return d


def cmd_fit():
    import sim04_engine as sim
    from sklearn.linear_model import LogisticRegression

    pbp = sim.load_reg_seasons(TRAIN)
    t = sim.build_transition_frame(pbp)
    d = fg_rows(t)
    print(d.groupby([d.points_off, d.points_def, d.possession_flip]).size())
    made = (d.points_off >= 3).to_numpy().astype(int)
    kd = d.fp_raw.to_numpy() + KICK_SNAP_YARDS
    m = LogisticRegression(C=1e6, max_iter=1000).fit(kd[:, None], made)
    fit = dict(train=list(TRAIN), n=int(len(d)), made=int(made.sum()), coef=float(m.coef_[0][0]), icpt=float(m.intercept_[0]), snap=KICK_SNAP_YARDS)
    OUTD.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(fit, indent=1))
    print(fit)


def install_fgd():
    import mod25d_variance as dv

    fit = json.loads(FIT.read_text(encoding="utf-8"))
    a = dv._G["tables"]["arrays"]
    code = np.asarray(a["play_type_code"]).astype(int)
    nyl = np.asarray(a["next_yardline"], float)
    fp = np.clip(np.round(np.asarray(a["fp_raw"], float)).astype(int), 1, 99)
    pts = np.asarray(a["points_off"], float)
    ok = (code == FG_CODE) & np.isfinite(nyl) & (np.asarray(a["clock_elapsed"], float) >= 0)
    tabs = {}
    for made in (True, False):
        rows = np.flatnonzero(ok & ((pts >= 3) == made))
        rows = rows[np.argsort(fp[rows], kind="stable")]
        edges = np.r_[0, np.cumsum(np.bincount(fp[rows], minlength=100))]
        tabs[made] = (rows, edges, np.flatnonzero(np.bincount(fp[rows], minlength=100)))
    state = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(dn, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(dn, distance, yardline, score_diff, qtr, clock_val, drawn)
        if int(drawn.get("play_type_code", -1)) != FG_CODE:
            return drawn
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(SALT, int(dv._G["cfg"].get("seed", 3)))
        STATS["eligible"] += 1
        z = fit["icpt"] + fit["coef"] * (float(yardline) + fit["snap"])
        made = bool(state["rng"].random() < 1.0 / (1.0 + np.exp(-z)))
        if made == (float(drawn["points_off"]) >= 3.0):
            return drawn
        rows, edges, present = tabs[made]
        y = int(min(max(round(float(yardline)), 1), 99))
        yb = int(present[np.argmin(np.abs(present - y))])
        lo, hi = edges[yb], edges[yb + 1]
        j = rows[lo + min(int(state["rng"].random() * (hi - lo)), hi - lo - 1)]
        STATS["swapped"] += 1
        new = dict(drawn)
        for k in ("points_off", "points_def", "clock_elapsed", "next_down", "next_distance", "next_yardline", "dist_gained", "off_to_used", "def_to_used", "play_type_code"):
            new[k] = a[k][j]
        new["flip"] = bool(a["possession_flip"][j])
        new["auto_first"] = bool(a["auto_first"][j])
        new["repeat_down"] = bool(a["repeat_down"][j])
        new["yards_gained"] = a["yards_gained"][j]
        lg = dv._G["log"]
        r = list(lg[-1])
        r[6], r[7], r[8], r[9], r[10] = int(new["play_type_code"]), float(new["points_off"]), float(new["points_def"]), bool(new["flip"]), float(new["clock_elapsed"])
        lg[-1] = tuple(r)
        return new

    dv._G["pol"] = pol


if __name__ == "__main__":
    cmd_fit()
