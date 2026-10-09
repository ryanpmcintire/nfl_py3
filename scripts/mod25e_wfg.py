import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUTD = REPO / "artifacts" / "mod25e3" / "wfg"
FIT = OUTD / "fit.json"
SALT = 9701
FIRST_MODIFIED_SEASON = 2012
LAST_POOL_SEASON = 2019
MAX_FIT_YARDLINE = 45
STATS = {"eligible": 0, "attempt": 0}


def enabled():
    return os.environ.get("WFG") == "1"


def cmd_fit():
    import mod25e_e97 as e97
    from sklearn.linear_model import LogisticRegression

    seasons = tuple(s for s in range(FIRST_MODIFIED_SEASON, LAST_POOL_SEASON + 1) if (e97.gen.PBP_DIR / f"season={s}").exists())
    e97.REAL_SEASONS = seasons
    _, pr = e97.real_ot()
    d = pr[(pr.idx >= 1) & pr.down.isin([1, 2, 3]) & (pr.yl <= MAX_FIT_YARDLINE) & (pr.sd >= 0)]
    X = np.column_stack([d.yl.to_numpy(), (d.down == 2).to_numpy(), (d.down == 3).to_numpy()]).astype(float)
    y = d.isfg.astype(int).to_numpy()
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    fit = dict(seasons=list(seasons), n=int(len(d)), fg=int(y.sum()), yl=float(m.coef_[0][0]), d2=float(m.coef_[0][1]), d3=float(m.coef_[0][2]), icpt=float(m.intercept_[0]), max_yl=MAX_FIT_YARDLINE)
    OUTD.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(fit, indent=1))
    print(fit)


def install_wfg():
    import mod25d_variance as dv

    fit = json.loads(FIT.read_text(encoding="utf-8"))
    a = dv._G["tables"]["arrays"]
    code = np.asarray(a["play_type_code"]).astype(int)
    nyl = np.asarray(a["next_yardline"], float)
    fp = np.clip(np.round(np.asarray(a["fp_raw"], float)).astype(int), 1, 99)
    ok = (code == 3) & np.isfinite(nyl) & (np.asarray(a["clock_elapsed"], float) >= 0)
    rows = np.flatnonzero(ok)
    rows = rows[np.argsort(fp[rows], kind="stable")]
    cum = np.cumsum(np.bincount(fp[rows], minlength=100))
    edges = np.r_[0, cum]
    present = np.flatnonzero(np.bincount(fp[rows], minlength=100))
    state = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(dn, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        while fr is not None and "ot_possession_index" not in fr.f_locals:
            fr = fr.f_back
        L = fr.f_locals
        in_ot, idx_ot = bool(L.get("in_ot")), int(L.get("ot_possession_index", 0))
        drawn = base(dn, distance, yardline, score_diff, qtr, clock_val, drawn)
        if not in_ot or idx_ot < 1 or score_diff < 0 or dn not in (1, 2, 3) or yardline > fit["max_yl"]:
            return drawn
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(SALT, int(dv._G["cfg"].get("seed", 3)))
        z = fit["icpt"] + fit["yl"] * float(yardline) + fit["d2"] * (dn == 2) + fit["d3"] * (dn == 3)
        p = 1.0 / (1.0 + np.exp(-z))
        STATS["eligible"] += 1
        if state["rng"].random() >= p:
            return drawn
        y = int(min(max(round(float(yardline)), 1), 99))
        yb = int(present[np.argmin(np.abs(present - y))])
        lo, hi = edges[yb], edges[yb + 1]
        j = rows[lo + min(int(state["rng"].random() * (hi - lo)), hi - lo - 1)]
        STATS["attempt"] += 1
        new = dict(drawn)
        for k in ("points_off", "points_def", "clock_elapsed", "next_down", "next_distance", "next_yardline", "dist_gained", "off_to_used", "def_to_used", "play_type_code"):
            new[k] = a[k][j]
        new["flip"] = bool(a["possession_flip"][j])
        new["auto_first"] = bool(a["auto_first"][j])
        new["repeat_down"] = bool(a["repeat_down"][j])
        new["yards_gained"] = a["yards_gained"][j]
        return new

    dv._G["pol"] = pol


if __name__ == "__main__":
    cmd_fit()
