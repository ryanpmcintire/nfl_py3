import os
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["g", "k", "pn", "oto_pre", "dto_pre", "oto_used", "dto_used", "oto_applied", "dto_applied"]
ST = {"g": -1, "k": 0, "rows": [], "pend": None}


def enabled():
    return os.environ.get("TOLOG") == "1"


def frame(rows):
    if not rows:
        return pd.DataFrame({c: np.empty(0, dtype=np.int64) for c in COLS})
    d = pd.DataFrame(np.asarray(rows, dtype=np.float64), columns=COLS)
    return d.astype({c: np.int64 for c in ("g", "k", "pn", "oto_pre", "dto_pre", "oto_applied", "dto_applied")})


def install_tolog():
    import mod25d_variance as dv

    ns = dv._G["ns"]
    top = ns["DECIDE"]
    game = ns["run_one_game"]
    base = dv._G["pol"]

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        ST["pend"] = (ST["g"], ST["k"], len(ns["PLAYLOG"]), int(off_to), int(def_to))
        ST["k"] += 1
        return top(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        p = ST["pend"]
        if p is not None:
            ou = float(drawn["off_to_used"])
            du = float(drawn["def_to_used"])
            ST["rows"].append((p[0], p[1], p[2], p[3], p[4], ou, du, p[3] - max(0, p[3] - ou), p[4] - max(0, p[4] - du)))
            ST["pend"] = None
        return drawn

    def run_one_game(*a, **k):
        ST["g"] += 1
        ST["k"] = 0
        ST["pend"] = None
        return game(*a, **k)

    ns["DECIDE"] = decide
    ns["run_one_game"] = run_one_game
    dv._G["pol"] = pol
    orig = dv.d_play_season

    def d_play_season(task):
        ST["g"] = -1
        ST["k"] = 0
        ST["rows"] = []
        ST["pend"] = None
        res = orig(task)
        outd = Path(dv._G["cfg"]["outdir"])
        outd.mkdir(parents=True, exist_ok=True)
        frame(ST["rows"]).to_parquet(outd / f"to_{task[0]}_{task[1]}.parquet")
        ST["rows"] = []
        return res

    dv.d_play_season = d_play_season
