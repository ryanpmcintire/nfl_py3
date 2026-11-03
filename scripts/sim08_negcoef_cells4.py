import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim08_evaluate as ev
from sim08_negcoef_cells import slope

base = ev.base
OUT = ev.RUN_ROOT / "main" / "negcoef_cells4.json"


def main():
    frame = base.load_games()
    outcome = frame["outcome"].to_numpy()
    line = frame["line"].to_numpy(dtype=float)
    S = ev.load_arm("main", "primary", frame)
    M = S["margins"]
    c, _, l_ = ev.sim_counts(M, line)
    sim = base.safe_logit((c + 0.5) / (c + l_ + 1.0))
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)
    mean = M.mean(1)
    se = M.std(1, ddof=1) / np.sqrt(M.shape[1])
    tot = S["totals"].mean(1)
    xe = S["x_env"]
    xd = S["x_diff"]
    df = pd.DataFrame({"line": line})
    within = sim - pd.Series(sim).groupby(df["line"]).transform("mean").to_numpy()
    miss = mean - line
    miss_w = miss - pd.Series(miss).groupby(df["line"]).transform("mean").to_numpy()
    A = np.column_stack([np.ones(len(sim)), tot, xe, xd, np.abs(line)])
    b, *_ = np.linalg.lstsq(A, within, rcond=None)
    pred = A @ b
    out = {
        "within_sim_logit_sd": float(within.std()),
        "within_miss_pts_sd": float(miss_w.std()),
        "mc_se_of_mean_margin_pts_median": float(np.median(se)),
        "within_miss_sd_minus_mc": float(np.sqrt(max(miss_w.var() - np.mean(se**2), 0.0))),
        "corr_within_sim_with_miss": float(np.corrcoef(within, miss_w)[0, 1]),
        "corr_miss_with_total": float(np.corrcoef(miss, tot)[0, 1]),
        "corr_miss_with_x_env": float(np.corrcoef(miss, xe)[0, 1]),
        "corr_miss_with_x_diff": float(np.corrcoef(miss, xd)[0, 1]),
        "within_sim_R2_on_design": float(1 - (within - pred).var() / within.var()),
        "y_on_miss_pts_within_line": slope(miss_w[nonpush], y[nonpush]),
        "y_on_predicted_part": slope(pred[nonpush], y[nonpush]),
        "y_on_unpredicted_part": slope((within - pred)[nonpush], y[nonpush]),
        "y_on_total": slope(tot[nonpush], y[nonpush]),
    }
    hi = se > np.median(se)
    out["y_on_within_sim_low_mc_se_half"] = slope(within[nonpush & ~hi], y[nonpush & ~hi])
    out["y_on_within_sim_high_mc_se_half"] = slope(within[nonpush & hi], y[nonpush & hi])
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
