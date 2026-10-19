import os
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["g", "k", "pn", "idx_in", "idx_out", "ptype_in", "ptype_out", "down", "dist", "yl", "sd", "qtr", "phase", "clock"]
ST = {"g": -1, "k": 0, "rows": []}


def enabled():
    return os.environ.get("DECLOG") == "1"


def frame(rows):
    if not rows:
        return pd.DataFrame({c: np.empty(0) for c in COLS}).assign(changed=np.empty(0, dtype=bool), ptype_changed=np.empty(0, dtype=bool))
    d = pd.DataFrame(np.asarray(rows, dtype=np.float64), columns=COLS)
    for c in ("g", "k", "pn", "idx_in", "idx_out", "ptype_in", "ptype_out", "down", "qtr", "phase"):
        d[c] = d[c].astype(np.int64)
    d["changed"] = d["idx_in"] != d["idx_out"]
    d["ptype_changed"] = d["ptype_in"] != d["ptype_out"]
    return d


def install_declog():
    import mod25d_variance as dv

    ns = dv._G["ns"]
    code = np.asarray(dv._G["tables"]["arrays"]["play_type_code"])
    top = ns["DECIDE"]
    game = ns["run_one_game"]

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        out = top(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
        ST["rows"].append((ST["g"], ST["k"], len(ns["PLAYLOG"]), int(idx), int(out), int(code[int(idx)]), int(code[int(out)]), int(down), float(distance), float(yardline), float(score_diff), int(qtr), int(phase), float(clock_val)))
        ST["k"] += 1
        return out

    def run_one_game(*a, **k):
        ST["g"] += 1
        ST["k"] = 0
        return game(*a, **k)

    ns["DECIDE"] = decide
    ns["run_one_game"] = run_one_game
    orig = dv.d_play_season

    def d_play_season(task):
        ST["g"] = -1
        ST["k"] = 0
        ST["rows"] = []
        res = orig(task)
        outd = Path(dv._G["cfg"]["outdir"])
        outd.mkdir(parents=True, exist_ok=True)
        frame(ST["rows"]).to_parquet(outd / f"dec_{task[0]}_{task[1]}.parquet")
        ST["rows"] = []
        return res

    dv.d_play_season = d_play_season
