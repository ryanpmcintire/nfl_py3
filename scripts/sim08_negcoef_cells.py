import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim08_evaluate as ev

base = ev.base
OUT = ev.RUN_ROOT / "main" / "negcoef_cells.json"


def slope(x, y):
    from sklearn.linear_model import LogisticRegression

    if len(y) < 40 or y.min() == y.max():
        return None
    m = LogisticRegression(C=1e10, max_iter=2000).fit(x[:, None], y)
    p = m.predict_proba(x[:, None])[:, 1]
    A = np.column_stack([np.ones(len(x)), x])
    H = (A * (p * (1 - p))[:, None]).T @ A
    se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    return {"n": len(y), "coef": float(m.coef_[0][0]), "se": se, "cover_rate": float(y.mean())}


def main():
    frame = base.load_games()
    outcome = frame["outcome"].to_numpy()
    res = frame["result"].to_numpy(dtype=float)
    line = frame["line"].to_numpy(dtype=float)
    S = ev.load_arm("main", "primary", frame)
    M = S["margins"]
    c, _, l_ = ev.sim_counts(M, line)
    sim = base.safe_logit((c + 0.5) / (c + l_ + 1.0))
    pc = c / M.shape[1]
    pl = l_ / M.shape[1]
    mean = M.mean(1)
    med = np.median(M, 1)
    tot = S["totals"].mean(1)
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)
    out = {}
    hand = {}
    for name, idx in (
        ("home_fav", int(np.where((line >= 5) & nonpush)[0][0])),
        ("road_fav", int(np.where((line <= -5) & nonpush)[0][0])),
        ("pickem", int(np.where(line == 0)[0][0]) if (line == 0).any() else int(np.argmin(np.abs(line)))),
    ):
        hand[name] = {
            "game_id": str(frame["game_id"].iloc[idx]),
            "home_spread_line": float(line[idx]),
            "result_home_minus_away": float(res[idx]),
            "label": str(outcome[idx]),
            "label_recomputed": "cover" if res[idx] > line[idx] else ("loss" if res[idx] < line[idx] else "push"),
            "sim_mean_margin": float(mean[idx]),
            "sim_p_cover": float(pc[idx]),
            "sim_p_loss": float(pl[idx]),
            "sim_p_cover_from_margin_gt_line": float((M[idx] > line[idx]).mean()),
        }
    out["hand"] = hand
    out["signs"] = {
        "corr_mean_margin_minus_line_vs_result_minus_line": float(np.corrcoef(mean - line, res - line)[0, 1]),
        "corr_simlogit_vs_mean_margin_minus_line": float(np.corrcoef(sim, mean - line)[0, 1]),
        "corr_simlogit_vs_mean_minus_median": float(np.corrcoef(sim, mean - med)[0, 1]),
        "corr_simlogit_vs_median_minus_line": float(np.corrcoef(sim, med - line)[0, 1]),
        "corr_result_minus_line_vs_line": float(np.corrcoef(res - line, line)[0, 1]),
        "corr_median_minus_line_vs_line": float(np.corrcoef(med - line, line)[0, 1]),
        "mean_margin_minus_line_mean": float((mean - line).mean()),
        "mean_margin_minus_line_sd": float((mean - line).std()),
        "simlogit_sd": float(sim.std()),
    }
    out["signs"]["sampling_sd_logit_expected"] = float(np.sqrt(4.0 / M.shape[1]))
    cells = {}

    def add(name, mask):
        cells[name] = slope(sim[mask & nonpush], y[mask & nonpush])

    add("all", np.ones(len(y), bool))
    a = np.abs(line)
    add("line_home_fav", line > 0)
    add("line_road_fav", line < 0)
    add("line_pickem", line == 0)
    for lo, hi in ((0, 2.5), (2.5, 3.01), (3.01, 4), (4, 6), (6, 6.51), (6.51, 7.01), (7.01, 10), (10, 99)):
        add(f"absline_{lo}_{hi}", (a > lo) & (a <= hi) if lo > 0 else (a <= hi))
    for q in range(3):
        lo, hi = np.quantile(tot, [q / 3, (q + 1) / 3])
        add(f"total_tercile{q}", (tot >= lo) & (tot <= hi))
    anchor = np.abs(S["x_diff"])
    for q in range(3):
        lo, hi = np.quantile(anchor, [q / 3, (q + 1) / 3])
        add(f"absxdiff_tercile{q}", (anchor >= lo) & (anchor <= hi))
    ae = np.abs(mean - line)
    for q in range(3):
        lo, hi = np.quantile(ae, [q / 3, (q + 1) / 3])
        add(f"abs_anchor_resid_tercile{q}", (ae >= lo) & (ae <= hi))
    skew = mean - med
    for q in range(3):
        lo, hi = np.quantile(skew, [q / 3, (q + 1) / 3])
        add(f"mean_minus_median_tercile{q}", (skew >= lo) & (skew <= hi))
    out["cells"] = cells
    comp = {}
    r1 = slope((mean - line)[nonpush], y[nonpush])
    r2 = slope((med - line)[nonpush], y[nonpush])
    r3 = slope((mean - med)[nonpush], y[nonpush])
    comp["y_on_mean_minus_line"] = r1
    comp["y_on_median_minus_line"] = r2
    comp["y_on_mean_minus_median"] = r3
    out["components"] = comp
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
