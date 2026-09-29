from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "artifacts/totals_backtest/20260901T184010Z/predictions.parquet"
EVAL = range(2013, 2026)
WS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SS = tuple(np.arange(-3.0, 1.01, 0.5))
FIELD_N = 30
DRAWS = 200
SEED = 20260929


def load():
    p = pd.read_parquet(PRED)
    p = p.loc[p["game_type"].astype(str).eq("REG")].copy()
    sched = pd.read_parquet(sorted(glob.glob(str(REPO / "data/raw/*/schedules.parquet")))[-1])
    p = p.merge(sched[["game_id", "gametime"]], on="game_id", how="left")
    p["key"] = p["gameday"].astype(str) + " " + p["gametime"].astype(str).fillna("")
    p = p.dropna(subset=["market_total", "actual_total", "predicted_residual"]).reset_index(
        drop=True
    )
    idx = p.sort_values("key").groupby(["season", "week"]).tail(1).index
    p["last"] = p.index.isin(idx)
    return p


def rnd(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def mae(g, a):
    return float(np.mean(np.abs(g - a)))


def fit_ws(tr, grid_w, grid_s):
    best, arg = 1e9, (0.0, 0.0)
    for w in grid_w:
        for s in grid_s:
            v = mae(rnd(tr.market_total + w * tr.predicted_residual + s), tr.actual_total)
            if v < best:
                best, arg = v, (w, s)
    return arg


def mode_round(m, tr):
    mt = tr.market_total.to_numpy()
    at = tr.actual_total.to_numpy()
    out = np.empty(len(m))
    cache = {}
    for i, x in enumerate(np.asarray(m)):
        c = float(rnd(x))
        if c not in cache:
            near = np.abs(mt - c) <= 3
            cand = np.arange(c - 2, c + 3)
            counts = [(at[near] == k).sum() for k in cand]
            cache[c] = cand[int(np.argmax(counts))]
        out[i] = cache[c]
    return out


def field_win(g, a, m, sd, rng):
    n = len(g)
    field = np.floor(m[:, None, None] + rng.normal(1.5, sd, (n, DRAWS, FIELD_N)) + 0.5)
    fe = np.abs(field - a[:, None, None]).min(axis=2)
    ours = np.abs(g - a)[:, None]
    win = (ours < fe) + 0.5 * (ours == fe)
    return float(win.mean())


def main():
    d = load()
    seasons = list(EVAL)
    arms = ["M0", "M1", "M2", "M3", "M4", "R1", "R2", "R3", "R4"]
    guesses = {a: np.full(len(d), np.nan) for a in arms}
    folds = []
    for s in seasons:
        te = d.season.eq(s).to_numpy()
        tr = d.loc[~te]
        t = d.loc[te]
        m, r = t.market_total.to_numpy(), t.predicted_residual.to_numpy()
        w1, _ = fit_ws(tr, WS, (0.0,))
        _, s2 = fit_ws(tr, (0.0,), SS)
        w3, s3 = fit_ws(tr, WS, SS)
        folds.append({"season": s, "w1": w1, "s2": s2, "w3": w3, "s3": s3})
        guesses["M0"][te] = rnd(m)
        guesses["M1"][te] = rnd(m + w1 * r)
        guesses["M2"][te] = rnd(m + s2)
        guesses["M3"][te] = rnd(m + w3 * r + s3)
        guesses["M4"][te] = rnd(m + 0.1 * r - 1.0)
        guesses["R1"][te] = np.floor(m)
        guesses["R2"][te] = np.ceil(m)
        guesses["R3"][te] = mode_round(m, tr)
        guesses["R4"][te] = mode_round(m + w3 * r + s3, tr)
    ev = d.season.isin(seasons).to_numpy()
    last = d["last"].to_numpy()
    a_all = d.actual_total.to_numpy()
    mk_all = d.market_total.to_numpy()
    out = {"n_A": int(ev.sum()), "n_B": int((ev & last).sum()), "arms": {}}
    for name in arms:
        g = guesses[name]
        row = {"mae_A": mae(g[ev], a_all[ev]), "mae_B": mae(g[ev & last], a_all[ev & last])}
        per = []
        for s in seasons:
            mk = d.season.eq(s).to_numpy()
            per.append(mae(g[mk], a_all[mk]) - mae(guesses["M0"][mk], a_all[mk]))
        per = np.array(per)
        row["season_diff_vs_M0"] = [round(float(x), 3) for x in per]
        row["seasons_better_of_13"] = int((per < 0).sum())
        rng = np.random.default_rng(1)
        boots = np.array([-per[rng.integers(0, len(per), len(per))].mean() for _ in range(4000)])
        row["mae_reduction_mean"] = float(-per.mean())
        row["reduction_ci"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        row["prob_positive_reduction"] = float(np.mean(boots > 0))
        mb = ev & last
        for sd in (4, 6, 8):
            rng = np.random.default_rng(SEED)
            row[f"win_share_B_sd{sd}"] = field_win(g[mb], a_all[mb], mk_all[mb], sd, rng)
        out["arms"][name] = row
    ins = fit_ws(d.loc[ev], WS, SS)
    out["in_sample_M3_ws"] = ins
    gi = rnd(d.market_total + ins[0] * d.predicted_residual + ins[1])
    out["in_sample_M3_mae_A"] = mae(gi[ev], a_all[ev])
    out["oos_M3_mae_A"] = out["arms"]["M3"]["mae_A"]
    out["gap_oos_minus_in"] = out["oos_M3_mae_A"] - out["in_sample_M3_mae_A"]
    out["folds"] = folds
    out["mean_actual_minus_market_A"] = float((d.actual_total - d.market_total)[ev].mean())
    json.dump(out, sys.stdout, indent=1, default=float)


if __name__ == "__main__":
    main()
