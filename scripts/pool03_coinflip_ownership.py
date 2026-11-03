import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from nfl_ats.evidence_conventions import probability_positive_from_draws

FAMILY = "pool03_coinflip_ownership"
OUT = Path("artifacts/pool_rank_replay")
spec = importlib.util.spec_from_file_location("p01", Path("scripts/pool01_forward_sim.py"))
p01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p01)
K = p01.K
BANDS = [0.01, 0.02, 0.03, 0.05, 0.08]
WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_share(x, n_o, n_t):
    def nll(b):
        q = np.clip(sig(b[0] + b[1] * x), 1e-9, 1 - 1e-9)
        return -(n_o * np.log(q) + n_t * np.log(1 - q)).sum()

    return minimize(nll, np.zeros(2), method="BFGS").x


def games_frame():
    raw = pd.read_parquet(p01.FIELD_FILE)
    rows = []
    for gid, g in raw.groupby("game_id"):
        o = g.loc[g["model_p"].idxmax()]
        t = g.drop(o.name).iloc[0]
        rows.append(
            {
                "game_id": gid,
                "week": int(o["week"]),
                "p": float(o["model_p"]),
                "x": float(-o["line"]),
                "n_o": float(o["picks"]),
                "n_t": float(t["picks"]),
                "cov_o": bool(o["covered"]),
            }
        )
    g = pd.DataFrame(rows)
    g["share"] = g["n_o"] / (g["n_o"] + g["n_t"])
    pred = np.zeros(len(g))
    for w in sorted(g["week"].unique()):
        tr, te = g["week"] != w, g["week"] == w
        b = fit_share(
            g.loc[tr, "x"].to_numpy(), g.loc[tr, "n_o"].to_numpy(), g.loc[tr, "n_t"].to_numpy()
        )
        pred[te.to_numpy()] = sig(b[0] + b[1] * g.loc[te, "x"].to_numpy())
    g["pred"] = pred
    return g


def proxy_report(g):
    n = g["n_o"] + g["n_t"]
    ll_m = float((g["n_o"] * np.log(g["pred"]) + g["n_t"] * np.log(1 - g["pred"])).sum())
    ll_c = float((g["n_o"] * np.log(0.5) + g["n_t"] * np.log(0.5)).sum())
    ms = g["n_o"].sum() / n.sum()
    ll_s = float((g["n_o"] * np.log(ms) + g["n_t"] * np.log(1 - ms)).sum())
    q = pd.qcut(g["pred"], 5, duplicates="drop")
    rel = [
        {
            "bin": str(k),
            "games": len(v),
            "predicted": float(np.average(v["pred"], weights=v["n_o"] + v["n_t"])),
            "observed": float(v["n_o"].sum() / (v["n_o"] + v["n_t"]).sum()),
        }
        for k, v in g.groupby(q, observed=True)
    ]
    sl = np.polyfit(logit(g["pred"]), logit(g["share"]), 1)
    pooled = spearmanr(g["pred"], g["share"])
    per_week = [
        float(spearmanr(v["pred"], v["share"])[0]) for _, v in g.groupby("week") if len(v) > 3
    ]
    rng = np.random.default_rng(3)
    boots = []
    for _ in range(4000):
        s = g.iloc[rng.integers(0, len(g), len(g))]
        boots.append(spearmanr(s["pred"], s["share"])[0])
    return {
        "games": len(g),
        "entries_per_game": float(n.mean()),
        "heldout_loglik_per_pick": ll_m / float(n.sum()),
        "loglik_per_pick_coinflip": ll_c / float(n.sum()),
        "loglik_per_pick_mean_share_insample": ll_s / float(n.sum()),
        "calibration_slope_logit": float(sl[0]),
        "calibration_intercept_logit": float(sl[1]),
        "reliability": rel,
        "spearman_pooled": float(pooled[0]),
        "spearman_pooled_interval_95": [
            float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)),
        ],
        "spearman_within_week_mean": float(np.mean(per_week)),
        "spearman_within_week": per_week,
        "rmse_share": float(np.sqrt(np.mean((g["pred"] - g["share"]) ** 2))),
    }


def flips(g, b, w):
    inband = (g["p"] - 0.5).abs() <= b
    return (inband & (w * logit(g["pred"]) > logit(g["p"]))).to_numpy()


def cells():
    out = [(0.0, 0.0)]
    out += [(b, w) for b in BANDS for w in WEIGHTS]
    return out


def evaluate(pool, F, state, rng_seed, weeks, E, dsd, our_points):
    pts0, bp0, dm0, our_bp0 = state
    rng = np.random.default_rng(rng_seed)
    n = len(pool)
    games = {k: pool[k].to_numpy() for k in ("p", "share")}
    games["u_o"] = np.zeros(n)
    games["u_t"] = np.zeros(n)
    C = F.shape[0]
    res = np.zeros((len(pts0), C))
    for i in range(len(pts0)):
        delta = dm0[i] + dsd * rng.standard_normal(E)
        rp, rb = pts0[i].copy(), bp0[i].copy()
        op = np.full(C, float(our_points))
        ob = np.full(C, float(our_bp0))
        for _ in range(weeks):
            idx = rng.integers(0, n, size=(1, 15))
            d = p01.draw_week(games, rng, 1, E, idx=idx, delta=delta)
            cov = d["cover"][0]
            f = F[:, idx[0]]
            ours = np.where(f, ~cov[None, :], cov[None, :])
            op += ours.sum(axis=1)
            s = int(d["served"][0])
            ob += ours[:, s]
            rp = rp + d["pts"][0]
            rb = rb + d["bp"][0]
        fk = rp * K + rb
        for c in range(C):
            res[i, c] = p01.share_given_key(fk[None, :], np.array([op[c] * K + ob[c]]))[0]
    return res


def main():
    ap = argparse.ArgumentParser(description="POOL-03 coin-flip band ownership term")
    ap.add_argument("--paths", type=int, default=300)
    ap.add_argument("--weeks", type=int, default=10)
    ap.add_argument("--points", type=int, default=65)
    ap.add_argument("--rank", type=int, default=19)
    ap.add_argument("--entries", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--proxy-only", action="store_true")
    a = ap.parse_args()
    g = games_frame()
    out = {"family": FAMILY, "proxy": proxy_report(g)}
    if not a.proxy_only:
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
        state = (
            np.concatenate(kp)[: a.paths],
            np.concatenate(kb)[: a.paths],
            np.concatenate(kd)[: a.paths],
            our_bp,
        )
        cl = cells()
        folds = []
        diffs = []
        for w in sorted(g["week"].unique()):
            tr = g[g["week"] != w].reset_index(drop=True)
            te = g[g["week"] == w].reset_index(drop=True)
            Ftr = np.array([flips(tr, b, wt) for b, wt in cl])
            rtr = evaluate(tr, Ftr, state, a.seed + int(w), a.weeks, E, dsd, a.points)
            m = rtr.mean(axis=0)
            best = int(np.argmax(m))
            Fte = np.array([flips(te, cl[0][0], cl[0][1]), flips(te, *cl[best])])
            rte = evaluate(te, Fte, state, a.seed + 100 + int(w), a.weeks, E, dsd, a.points)
            d = rte[:, 1] - rte[:, 0]
            diffs.append(d)
            folds.append(
                {
                    "held_out_week": int(w),
                    "chosen_band": cl[best][0],
                    "chosen_weight": cl[best][1],
                    "train_p_first_served": float(m[0]),
                    "train_p_first_chosen": float(m[best]),
                    "games_in_chosen_band": int(((te["p"] - 0.5).abs() <= cl[best][0]).sum()),
                    "sides_changed": int(Fte[1].sum()),
                    "heldout_p_first_served": float(rte[:, 0].mean()),
                    "heldout_p_first_term": float(rte[:, 1].mean()),
                    "train_cells": {
                        f"{b}_{wt}": float(x) for (b, wt), x in zip(cl, m, strict=True)
                    },
                }
            )
        D = np.array(diffs)
        pathmean = D.mean(axis=0)
        brng = np.random.default_rng(a.seed + 7)
        reps = np.array(
            [pathmean[brng.integers(0, len(pathmean), len(pathmean))].mean() for _ in range(10000)]
        )
        out["folds"] = folds
        out["paths"] = int(a.paths)
        out["looks"] = {
            "cells_per_fold": len(cl),
            "folds": len(folds),
            "total": len(cl) * len(folds),
        }
        out["gain_term_minus_served_pp"] = {
            "mean": float(pathmean.mean() * 100),
            "interval_95": [
                float(np.quantile(reps, 0.025) * 100),
                float(np.quantile(reps, 0.975) * 100),
            ],
            "probability_positive": float(probability_positive_from_draws(reps)),
        }
        out["p_first_served_heldout_mean"] = float(
            np.mean([f["heldout_p_first_served"] for f in folds])
        )
        out["p_first_term_heldout_mean"] = float(
            np.mean([f["heldout_p_first_term"] for f in folds])
        )
        out["sides_changed_per_week"] = {str(f["held_out_week"]): f["sides_changed"] for f in folds}
    (OUT / "pool03_coinflip_ownership.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1)[:5000])


if __name__ == "__main__":
    main()
