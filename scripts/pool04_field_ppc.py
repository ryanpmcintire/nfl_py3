import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pool01_forward_sim import WEEKLY_FILE, load_games, rival_table
from pool01_weekly_points import split_half

OUT = Path("artifacts/pool_rank_replay")
LAST_WEEK = int(re.search(r"w01-(\d+)", WEEKLY_FILE.name).group(1))
MODES = ("model_cover", "fixed_cover", "fixed_cover_skill")


def stats(m):
    return m.mean(axis=1), m.std(axis=1, ddof=1), m.max(axis=1)


def pct(sim, obs):
    return float((sim < obs).mean() + 0.5 * (sim == obs).mean())


def cell(obs, sim):
    lo, hi = np.quantile(sim, [0.05, 0.95])
    return [float(obs), float(sim.mean()), pct(sim, obs), float(lo), float(hi)]


def week_corr(m):
    c = np.corrcoef(m.T)
    return float(c[np.triu_indices_from(c, k=1)].mean())


def tied_at_max(m):
    return (m == m.max(axis=-1, keepdims=True)).sum(axis=-1)


def block(o, s):
    sm, ss, sx = stats(s)
    thr = float(np.quantile(o, 0.9, method="higher"))
    return {
        "mean": cell(o.mean(), sm),
        "sd": cell(o.std(ddof=1), ss),
        "max": cell(o.max(), sx),
        "top_decile_score": thr,
        "share_at_or_above_top_decile": cell((o >= thr).mean(), (s >= thr).mean(axis=1)),
    }


def simulate(df, rng, S, E, mode, dsd, weeks):
    wk = []
    for w in range(1, weeks + 1):
        g = df[df["week"] == w]
        n = len(g)
        p, sh = g["p"].to_numpy(), g["share"].to_numpy()
        if mode == "model_cover":
            cover = rng.random((S, n)) < p[None, :]
        else:
            cover = np.tile(g["cov_o"].to_numpy(), (S, 1))
        wk.append(np.where(cover, sh[None, :], 1.0 - sh[None, :]))
    skill = dsd * rng.standard_normal((S, E)) if mode == "fixed_cover_skill" else np.zeros((S, E))
    pts = []
    for thr in wk:
        tp = np.clip(thr[:, None, :] + skill[:, :, None], 0.0, 1.0)
        pts.append((rng.random(tp.shape) < tp).sum(axis=2).astype(float))
    return np.stack(pts, axis=2)


def main():
    ap = argparse.ArgumentParser(description="Observed field scores against the simulated field")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--weeks", type=int, default=LAST_WEEK, help="last week to score")
    ap.add_argument("--points", type=int, required=True, help="our season points on the page")
    ap.add_argument("--out", default="pool04_field_ppc.json", help="results file name")
    a = ap.parse_args()
    df = load_games()
    w = pd.read_parquet(WEEKLY_FILE)
    obs = w[[f"w{k}" for k in range(1, a.weeks + 1)]].to_numpy(float)
    n_games = df[df["week"] <= a.weeks].groupby("week").size().to_numpy(float)
    E = obs.shape[0]
    _, _, dsd = rival_table(a.points)
    rng = np.random.default_rng(a.seed)
    ot = obs.sum(axis=1)
    looks = 4 * a.weeks + 5 + 2
    res = {
        "weekly_file": WEEKLY_FILE.name,
        "weeks": a.weeks,
        "entries": E,
        "sims": a.sims,
        "dsd": dsd,
        "looks_per_mode": looks,
        "looks": looks * len(MODES),
        "cell_layout": ["observed", "sim_mean", "percentile_of_observed", "sim_q05", "sim_q95"],
        "observed_split_half_spearman_brown": split_half(obs / n_games)[1],
        "modes": {},
    }
    for mode in MODES:
        sim = simulate(df, rng, a.sims, E, mode, dsd, a.weeks)
        cells = {f"w{k + 1}": block(obs[:, k], sim[:, :, k]) for k in range(a.weeks)}
        tot = sim.sum(axis=2)
        ss = tot.std(axis=1, ddof=1)
        cells["total"] = block(ot, tot)
        cells["total"]["tied_at_max"] = cell(tied_at_max(ot), tied_at_max(tot))
        cells["week_to_week_correlation"] = cell(
            week_corr(obs), np.array([week_corr(sim[i]) for i in range(a.sims)])
        )
        cells["split_half_spearman_brown"] = cell(
            res["observed_split_half_spearman_brown"],
            np.array([split_half(sim[i] / n_games)[1] for i in range(a.sims)]),
        )
        wsd = np.mean([cells[f"w{k + 1}"]["sd"][0] for k in range(a.weeks)])
        wsd_sim = np.mean([cells[f"w{k + 1}"]["sd"][1] for k in range(a.weeks)])
        ratio = ot.std(ddof=1) / ss - 1.0
        cells["avg_weekly_sd_ratio_obs_over_sim"] = float(wsd / wsd_sim)
        cells["total_sd_ratio_obs_over_sim"] = float(ot.std(ddof=1) / ss.mean())
        cells["total_variance_ratio"] = float(ot.var(ddof=1) / (ss**2).mean())
        cells["total_sd_ratio_minus_one"] = {
            "effect": float(ot.std(ddof=1) / ss.mean() - 1.0),
            "interval_95": [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))],
            "standard_error": float(ratio.std(ddof=1)),
            "probability_positive": float((ratio > 0).mean()),
        }
        outside = [
            f"w{k + 1}.{s}"
            for k in range(a.weeks)
            for s in ("mean", "sd", "max", "share_at_or_above_top_decile")
            if not cells[f"w{k + 1}"][s][3] <= cells[f"w{k + 1}"][s][0] <= cells[f"w{k + 1}"][s][4]
        ]
        cells["weekly_cells_outside_90_band"] = outside
        res["modes"][mode] = cells
    (OUT / a.out).write_text(json.dumps(res, indent=1) + "\n")
    for mode, c in res["modes"].items():
        print(mode)
        for k in [f"w{j + 1}" for j in range(a.weeks)] + ["total"]:
            row = {
                s: [round(v, 3) for v in c[k][s]]
                for s in ("mean", "sd", "max", "share_at_or_above_top_decile")
            }
            print(" ", k, c[k]["top_decile_score"], row)
        for k in (
            "week_to_week_correlation",
            "split_half_spearman_brown",
            "total_sd_ratio_minus_one",
            "avg_weekly_sd_ratio_obs_over_sim",
            "weekly_cells_outside_90_band",
        ):
            print(" ", k, c[k])
        print("  tied_at_max", c["total"]["tied_at_max"])


if __name__ == "__main__":
    main()
