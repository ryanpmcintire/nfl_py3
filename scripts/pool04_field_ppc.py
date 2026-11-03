import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pool01_forward_sim import load_games, rival_table

OUT = Path("artifacts/pool_rank_replay")
WEEKLY_FILE = Path("data/splash/field/2026_weekly_points_w01-08.parquet")
WEEKS = 8


def stats(m):
    return m.mean(axis=1), m.std(axis=1, ddof=1), m.max(axis=1)


def pct(sim, obs):
    return float((sim < obs).mean() + 0.5 * (sim == obs).mean())


def simulate(df, rng, S, E, mode, dsd):
    wk = []
    for w in range(1, WEEKS + 1):
        g = df[df["week"] == w]
        games = {k: g[k].to_numpy() for k in ("p", "share", "u_o", "u_t")}
        n = len(g)
        p, sh = games["p"], games["share"]
        if mode == "fixed_cover":
            cover = np.tile(g["cov_o"].to_numpy(), (S, 1))
        else:
            cover = rng.random((S, n)) < p[None, :]
        thr = np.where(cover, sh[None, :], 1.0 - sh[None, :])
        wk.append(thr)
    skill = dsd * rng.standard_normal((S, E)) if mode == "fixed_cover_skill" else np.zeros((S, E))
    pts = []
    for thr in wk:
        tp = np.clip(thr[:, None, :] + skill[:, :, None], 0.0, 1.0)
        pts.append((rng.random(tp.shape) < tp).sum(axis=2).astype(float))
    return np.stack(pts, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    df = load_games()
    w = pd.read_parquet(WEEKLY_FILE)
    obs = w[[f"w{k}" for k in range(1, WEEKS + 1)]].to_numpy(float)
    E = obs.shape[0]
    _, _, dsd = rival_table(65)
    rng = np.random.default_rng(a.seed)
    res = {"entries": E, "sims": a.sims, "dsd": dsd, "looks_per_mode": 28, "modes": {}}
    for mode in ("model_cover", "fixed_cover", "fixed_cover_skill"):
        sim = simulate(df, rng, a.sims, E, mode, dsd)
        cells = {}
        for k in range(WEEKS):
            sm, ss, sx = stats(sim[:, :, k])
            om, os_, ox = obs[:, k].mean(), obs[:, k].std(ddof=1), obs[:, k].max()
            cells[f"w{k+1}"] = {
                "mean": [float(om), float(sm.mean()), pct(sm, om)],
                "sd": [float(os_), float(ss.mean()), pct(ss, os_)],
                "max": [float(ox), float(sx.mean()), pct(sx, ox)],
            }
        tot = sim.sum(axis=2)
        sm, ss, sx = stats(tot)
        ot = obs.sum(axis=1)
        cells["total"] = {
            "mean": [float(ot.mean()), float(sm.mean()), pct(sm, ot.mean())],
            "sd": [float(ot.std(ddof=1)), float(ss.mean()), pct(ss, ot.std(ddof=1))],
            "max": [float(ot.max()), float(sx.mean()), pct(sx, ot.max())],
        }
        wsd = np.mean([cells[f"w{k+1}"]["sd"][0] for k in range(WEEKS)])
        wsd_sim = np.mean([cells[f"w{k+1}"]["sd"][1] for k in range(WEEKS)])
        cells["avg_weekly_sd_ratio_obs_over_sim"] = float(wsd / wsd_sim)
        cells["total_sd_ratio_obs_over_sim"] = float(ot.std(ddof=1) / ss.mean())
        cells["total_variance_ratio"] = float(ot.var(ddof=1) / (ss**2).mean())
        res["modes"][mode] = cells
    (OUT / "pool04_field_ppc.json").write_text(json.dumps(res, indent=1) + "\n")
    for mode, c in res["modes"].items():
        keys = ("total_sd_ratio_obs_over_sim", "total_variance_ratio")
        print(mode, c["total"], {k: c[k] for k in keys})

if __name__ == "__main__":
    main()
