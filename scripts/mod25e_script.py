import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod25d_variance as dv
import mod25e_budget as b
from mod25e_analysis import drives_games, per_game


def tabs(T, G):
    D, X = per_game(T, G)
    mr = np.where(np.isnan(X["mreg"]), X["margin"], X["mreg"])
    M = np.c_[np.zeros(len(X)), X["m1"].fillna(pd.Series(mr)), X["m2"].fillna(pd.Series(mr)), X["m3"].fillna(pd.Series(mr)), mr]
    return D, X, M


def slopes(M, sid, rng, boot):
    ids = np.unique(sid)
    out = {}
    def one(sel):
        r = {}
        for q in (1, 2, 3, 4):
            x, inc = M[sel, q - 1], M[sel, q] - M[sel, q - 1]
            r[f"slope_q{q}_on_lead"] = np.polyfit(x, inc, 1)[0] if q > 1 else np.nan
        inc = [M[sel, q] - M[sel, q - 1] for q in (1, 2, 3, 4)]
        for i in range(4):
            for j in range(i + 1, 4):
                r[f"cov_q{i+1}q{j+1}"] = np.cov(inc[i], inc[j])[0, 1]
        r["cross_sum"] = sum(r[f"cov_q{i+1}q{j+1}"] for i in range(4) for j in range(i + 1, 4)) * 2
        return r
    pt = one(np.ones(len(sid), bool))
    idx = {k: np.flatnonzero(sid == k) for k in ids}
    bs = []
    for _ in range(boot):
        pick = rng.choice(ids, len(ids))
        sel = np.concatenate([idx[k] for k in pick])
        bs.append(one(sel))
    return pt, pd.DataFrame(bs)


rng = np.random.default_rng(3)
T, G = b.build_real(dv.EVAL)
D, X, M = tabs(T, G)
rp, rb = slopes(M, X["sid"].to_numpy(), rng, 400)
T, G = b.build_sim(2)
D, X, M = tabs(T, G)
sp, sb = slopes(M, X["sid"].to_numpy(), rng, 400)
for k in [k for k in rp if np.isfinite(rp[k])]:
    d = sb[k] - rb[k]
    d = d[np.isfinite(d)]
    print(f"{k:22s} real {rp[k]:8.3f} sim {sp[k]:8.3f} diff {sp[k]-rp[k]:8.3f} [{np.percentile(d,2.5):.3f},{np.percentile(d,97.5):.3f}] P+ {np.mean(d>0):.3f}")
