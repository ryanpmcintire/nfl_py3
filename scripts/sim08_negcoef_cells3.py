import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim08_evaluate as ev
from sim08_negcoef_cells import slope

base = ev.base
OUT = ev.RUN_ROOT / "main" / "negcoef_cells3.json"


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
    df = pd.DataFrame({"line": line, "sim": sim, "y": y})
    g = df.groupby("line")["sim"]
    between = g.transform("mean").to_numpy()
    within = sim - between
    n_line = g.transform("size").to_numpy()
    rng = np.random.default_rng(0)
    half = np.sqrt(4.0 / M.shape[1])
    out = {
        "n_distinct_lines": int(df["line"].nunique()),
        "share_sim_variance_between_line": float(between.var() / sim.var()),
        "within_line_sd": float(within.std()),
        "mc_noise_sd_expected_logit": float(half),
        "between_line_component": slope(between[nonpush], y[nonpush]),
        "within_line_component": slope(within[nonpush], y[nonpush]),
        "within_line_component_lines_n_ge_20": slope(
            within[nonpush & (n_line >= 20)], y[nonpush & (n_line >= 20)]
        ),
    }
    lines_ok = nonpush & (n_line >= 20)
    out["between_line_component_lines_n_ge_20"] = slope(between[lines_ok], y[lines_ok])
    rate = df.loc[nonpush].groupby("line")["y"].agg(["mean", "size"])
    simm = df.loc[nonpush].groupby("line")["sim"].mean()
    big = rate[rate["size"] >= 30].index
    out["line_table_n_ge_30"] = [
        {
            "line": float(k),
            "n": int(rate.loc[k, "size"]),
            "real_cover": float(rate.loc[k, "mean"]),
            "sim_logit_mean": float(simm.loc[k]),
        }
        for k in big
    ]
    perm = []
    for _ in range(2000):
        yp = y.copy()
        idx = np.where(nonpush)[0]
        yp[idx] = rng.permutation(y[idx])
        r = slope(within[nonpush], yp[nonpush])
        perm.append(r["coef"])
    perm = np.array(perm)
    obs = out["within_line_component"]["coef"]
    out["within_line_permutation_p_le_observed"] = float((perm <= obs).mean())
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "line_table_n_ge_30"}, indent=1))
    for r in out["line_table_n_ge_30"]:
        print(r)


if __name__ == "__main__":
    main()
