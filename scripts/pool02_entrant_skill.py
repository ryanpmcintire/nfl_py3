import json

import numpy as np
import pandas as pd
from scipy.stats import norm

d = pd.read_parquet("data/splash/field/2026_weekly_points_w01-08.parquet")
Y = d[[f"w{i}" for i in range(1, 9)]].values.astype(float)
G = np.array([16, 16, 16, 16, 15, 14, 14, 14.0])
A = Y / G
splits = {"odd_even": ([0, 2, 4, 6], [1, 3, 5, 7]), "w1-4_v_w5-8": ([0, 1, 2, 3], [4, 5, 6, 7])}


def half(Yv, idx):
    return Yv[:, idx].sum(1) / G[idx].sum()


def rel(Yv, s):
    a, b = splits[s]
    r = np.corrcoef(half(Yv, a), half(Yv, b))[0, 1]
    return 2 * r / (1 + r)


rng = np.random.default_rng(20261103)
B = 4000
pw = Y.sum(0) / (G * len(Y))
out = {}
for s in splits:
    obs = rel(Y, s)
    perm = []
    for _ in range(B):
        P = np.array([rng.permutation(r) for r in A]) * G
        perm.append(rel(P, s))
    p50 = [rel(rng.binomial(G.astype(int), 0.5, size=Y.shape).astype(float), s) for _ in range(B)]
    pm = [rel(rng.binomial(G.astype(int), pw, size=Y.shape).astype(float), s) for _ in range(B)]
    idx = rng.integers(0, len(Y), (B, len(Y)))
    bs = [rel(Y[i], s) for i in idx]
    ci = np.percentile(bs, [2.5, 97.5])

    def f(x, observed=obs):
        return (
            float(np.mean(x)),
            float(np.percentile(x, 2.5)),
            float(np.percentile(x, 97.5)),
            float(np.mean(np.array(x) >= observed)),
        )

    out[s] = {
        "obs": obs,
        "boot_ci": ci.tolist(),
        "boot_pplus": float(np.mean(np.array(bs) > 0)),
        "perm": f(perm),
        "coin50": f(p50),
        "market_week": f(pm),
    }
print(json.dumps(out, indent=1))
tot = Y.sum(1) / G.sum()
N = G.sum()
m = tot.mean()
vb = np.mean(tot * (1 - tot) / N)
print("rate mean", m, "var obs", tot.var(ddof=1), "var binom", vb)
OUR_POINTS_THROUGH_WEEK8 = 65
res = {}
for lab, r in [("pt", out["odd_even"]["obs"]), ("hi", out["odd_even"]["boot_ci"][1]), ("lo", 0)]:
    t2 = max(r * tot.var(ddof=1), 0)
    k = t2 / (t2 + vb)
    mu = k * (OUR_POINTS_THROUGH_WEEK8 / N - m)
    sd = np.sqrt(k * vb)
    res[lab] = {
        "tau": float(np.sqrt(t2)),
        "k": float(k),
        "post_mean_pp": float(mu * 100),
        "post_sd_pp": float(sd * 100),
        "p_gt_median": float(norm.cdf(mu / sd)) if sd > 0 else 0.5,
    }
print(res, "median rate", np.median(tot), "ours", OUR_POINTS_THROUGH_WEEK8 / N)
print(
    "rank19 pct",
    (tot * N > OUR_POINTS_THROUGH_WEEK8).sum(),
    (tot * N == OUR_POINTS_THROUGH_WEEK8).sum(),
)
