import argparse
import json
from math import erf, sqrt

import numpy as np
import sky03_matchups as m
from sky04_lead_lag import fit_b, logit

ap = argparse.ArgumentParser()
ap.add_argument("--multipliers", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
ap.add_argument("--reps", type=int, default=200)
ap.add_argument(
    "--out", default="F:/Repos/nfl_py3/artifacts/sky03/def_traits/positive_control.json"
)
args = ap.parse_args()
seasons = m.EVAL_SEASONS
import sky03_def_traits as dt  # noqa: E402
import sky03_pfr_traits as p  # noqa: E402

t, _ = dt.load()
rr = dt.build(t)
z, r, w = dt.CAND["B_oN_x_completions_per_target_allowed"]
_, sub, _x = dt.term(rr, z, r, w, seasons, np.random.default_rng(m.SEED))
pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
off = logit(pr)
xs = sub.x.values
se = sub.season.values
y0 = sub.home_covered.astype(float).values
fc = np.array([fit_b(off[se != s], xs[se != s], y0[se != s]) for s in seasons])
bobs = float(fit_b(off, xs, y0))


def phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def oos(y):
    pooled = np.zeros(len(y))
    for s in seasons:
        te = se == s
        b = fit_b(off[~te], xs[~te], y[~te])
        pooled[te] = 1 / (1 + np.exp(-(off[te] + b * xs[te])))
    d = (pr - y) ** 2 - (pooled - y) ** 2
    return d.mean(), d.mean() / (d.std(ddof=1) / sqrt(len(d)))


rng = np.random.default_rng(20261103)
REPS = args.reps
out = {
    "script": "scripts/sky03_def_positive_control.py",
    "n": len(sub),
    "x_sd": float(xs.std()),
    "pooled_obs": bobs,
    "x_corr_logit_resid": 0,
}
sd_b = float(np.std(fc))
out["fold_coef_sd"] = sd_b
for mult in args.multipliers:
    b0 = mult * abs(bobs) if mult else 0.0
    gains, pp, det = [], [], 0
    for _ in range(REPS):
        p = 1 / (1 + np.exp(-(off + b0 * xs)))  # noqa: F811
        y = (rng.random(len(p)) < p).astype(float)
        gm, z = oos(y)
        gains.append(gm)
        pp.append(phi(z))
    pp = np.array(pp)
    out[f"mult_{mult}"] = {
        "b0": b0,
        "mean_gain": float(np.mean(gains)),
        "frac_Ppos_ge_0.95": float((pp >= 0.95).mean()),
        "frac_Ppos_ge_0.5": float((pp >= 0.5).mean()),
        "mean_Ppos": float(pp.mean()),
    }
    print(mult, out[f"mult_{mult}"], flush=True)
json.dump(out, open(args.out, "w"), indent=1)  # noqa: SIM115
