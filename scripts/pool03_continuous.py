import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from nfl_ats.evidence_conventions import probability_positive_from_draws

spec = importlib.util.spec_from_file_location("p3", Path("scripts/pool03_coinflip_ownership.py"))
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)
p01 = p3.p01
WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def side_flip(g, w):
    lead = p3.logit(g["p"]) - w * p3.logit(g["pred"]) * 4 * g["p"] * (1 - g["p"])
    return (lead < 0).to_numpy()


def main():
    ap = argparse.ArgumentParser(description="POOL-03 continuous ownership term")
    ap.add_argument("--paths", type=int, default=600)
    ap.add_argument("--weeks", type=int, default=10)
    ap.add_argument("--points", type=int, default=65)
    ap.add_argument("--rank", type=int, default=19)
    ap.add_argument("--entries", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()
    g = p3.games_frame()
    df = p01.load_games()
    rng = np.random.default_rng(a.seed)
    E = a.entries - 1
    table = p01.rival_table(a.points)
    kp, kb, kd, have = [], [], [], 0
    while have < a.paths:
        pp, bb, dd, dsd, our_bp = p01.past_state(df, rng, E, 2000, a.points, a.rank, table)
        kp.append(pp)
        kb.append(bb)
        kd.append(dd)
        have += len(pp)
    state = (*tuple(np.concatenate(x)[: a.paths] for x in (kp, kb, kd)), our_bp)
    folds, diffs = [], []
    for wk in sorted(g["week"].unique()):
        tr = g[g["week"] != wk].reset_index(drop=True)
        te = g[g["week"] == wk].reset_index(drop=True)
        Ftr = np.array([side_flip(tr, w) for w in WEIGHTS])
        m = p3.evaluate(tr, Ftr, state, a.seed + int(wk), a.weeks, E, dsd, a.points).mean(axis=0)
        best = int(np.argmax(m))
        Fte = np.array([side_flip(te, 0.0), side_flip(te, WEIGHTS[best])])
        r = p3.evaluate(te, Fte, state, a.seed + 100 + int(wk), a.weeks, E, dsd, a.points)
        diffs.append(r[:, 1] - r[:, 0])
        folds.append(
            {
                "week": int(wk),
                "weight": WEIGHTS[best],
                "changed": int(Fte[1].sum()),
                "p_first_served": float(r[:, 0].mean()),
                "p_first_term": float(r[:, 1].mean()),
            }
        )
    D = np.array(diffs)
    br = np.random.default_rng(a.seed + 7)
    nw, npth = D.shape
    reps = np.array(
        [D[np.ix_(br.integers(0, nw, nw), br.integers(0, npth, npth))].mean() for _ in range(5000)]
    )
    weekly = D.mean(axis=1)
    wreps = np.array([weekly[br.integers(0, nw, nw)].mean() for _ in range(5000)])
    rep = []
    for wk in sorted(g["week"].unique()):
        te = g[g["week"] == wk].reset_index(drop=True)
        w = next(f for f in folds if f["week"] == wk)["weight"]
        fl = side_flip(te, w)
        for i in np.where(fl)[0]:
            rep.append(
                {
                    "game_id": te.loc[i, "game_id"],
                    "week": int(wk),
                    "p": float(te.loc[i, "p"]),
                    "share_served": float(te.loc[i, "share"]),
                    "served_covered": bool(te.loc[i, "cov_o"]),
                    "changed_covered": bool(not te.loc[i, "cov_o"]),
                }
            )
    pts = {"served": 0, "term": 0}
    for wk in sorted(g["week"].unique()):
        te = g[g["week"] == wk]
        w = next(f for f in folds if f["week"] == wk)["weight"]
        fl = side_flip(te.reset_index(drop=True), w)
        c = te["cov_o"].to_numpy()
        pts["served"] += int(c.sum())
        pts["term"] += int(np.where(fl, ~c, c).sum())
    out = {
        "looks": len(WEIGHTS) * nw,
        "folds": folds,
        "gain_pp": float(D.mean() * 100),
        "interval_paths_and_weeks": [
            float(np.quantile(reps, 0.025) * 100),
            float(np.quantile(reps, 0.975) * 100),
        ],
        "p_pos_paths_and_weeks": float(probability_positive_from_draws(reps)),
        "interval_weeks_only": [
            float(np.quantile(wreps, 0.025) * 100),
            float(np.quantile(wreps, 0.975) * 100),
        ],
        "p_pos_weeks_only": float(probability_positive_from_draws(wreps)),
        "changed_games": rep,
        "real_points_all_weeks": pts,
    }
    Path("artifacts/pool_rank_replay/pool03_continuous.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
