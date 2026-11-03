import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim08_evaluate as ev
from sim08_negcoef_cells import slope

base = ev.base
OUT = ev.RUN_ROOT / "main" / "negcoef_cells2.json"


def main():
    frame = base.load_games()
    outcome = frame["outcome"].to_numpy()
    line = frame["line"].to_numpy(dtype=float)
    seasons = frame["season"].to_numpy()
    S = ev.load_arm("main", "primary", frame)
    M = S["margins"]
    c, _, l_ = ev.sim_counts(M, line)
    simp = (c + 0.5) / (c + l_ + 1.0)
    sim = base.safe_logit(simp)
    mean = M.mean(1)
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)
    out = {"per_season": {}, "cells": {}}
    for s_ in np.unique(seasons):
        m = (seasons == s_) & nonpush
        out["per_season"][int(s_)] = slope(sim[m], y[m])
    a = np.abs(line)
    for name, mask in (
        ("home_fav_3_to_4", (line > 2.5) & (line <= 4)),
        ("road_fav_3_to_4", (line < -2.5) & (line >= -4)),
        ("home_fav_other", (line > 4) | ((line > 0) & (line <= 2.5))),
        ("road_fav_other", (line < -4) | ((line < 0) & (line >= -2.5))),
        ("abs_2.5_4", (a > 2.5) & (a <= 4)),
        ("abs_not_2.5_4", ~((a > 2.5) & (a <= 4))),
    ):
        k = mask & nonpush
        if k.sum() == 0:
            continue
        out["cells"][name] = {
            "n": int(k.sum()),
            "real_cover_rate": float(y[k].mean()),
            "sim_cover_cond_mean": float(simp[k].mean()),
            "sim_mean_margin_minus_line": float((mean - line)[k].mean()),
            "sim_logit_sd": float(sim[k].std()),
            "slope": slope(sim[k], y[k]),
        }
    ins = (a > 2.5) & (a <= 4) & nonpush
    out["abs_2.5_4_by_sim_tercile"] = []
    order = np.argsort(sim[ins])
    for part in np.array_split(order, 3):
        idx = np.where(ins)[0][part]
        out["abs_2.5_4_by_sim_tercile"].append(
            {
                "n": len(idx),
                "sim_p": float(simp[idx].mean()),
                "real_cover": float(y[idx].mean()),
                "mean_minus_line": float((mean - line)[idx].mean()),
            }
        )
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
