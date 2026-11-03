import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

R = "F:/Repos/nfl_py3/artifacts/"
sc = pd.read_parquet(R + "sim10_wf/pipeline_joint/scored_hgb_nomkt.parquet")
bp = pd.read_parquet(R + "opener_evaluation/20260929T192743Z/per_game.parquet")


def fr(pg):
    d = pg.loc[
        pg["margin_vs_open"].ne(0)
        & pg["correct_at_open_probability_rule"].notna()
        & pg["season"].between(2020, 2025)
    ]
    y = np.where(
        d["pick_home_at_open_probability_rule"],
        d["correct_at_open_probability_rule"],
        1 - d["correct_at_open_probability_rule"],
    )
    return pd.DataFrame(
        {
            "season": d["season"].to_numpy(),
            "p": d["home_cover_probability_at_open"].to_numpy(),
            "y": y,
        },
        index=d["game_id"].to_numpy(),
    )


a = fr(sc)
b = fr(bp)
i = a.index.intersection(b.index)
a, b = a.loc[i], b.loc[i]
assert (a.y == b.y).all()
y = a.y.to_numpy()
s = a.season.to_numpy()


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def tfit(z, y):
    return minimize_scalar(lambda t: ll(expit(t * z), y).sum(), bounds=(-3, 6), method="bounded").x


def boot(num_by_s, n_by_s, D=10000):
    rng = np.random.default_rng(20261103)
    k = len(num_by_s)
    ix = rng.integers(0, k, (D, k))
    bb = num_by_s[ix].sum(1) / n_by_s[ix].sum(1)
    return np.percentile(bb, [2.5, 97.5]), (bb > 0).mean()


def seasonstat(d):
    ss = np.unique(s)
    return np.array([d[s == q].sum() for q in ss]), np.array([(s == q).sum() for q in ss])


out = {}
for nm, p in (("base", b.p.to_numpy()), ("student", a.p.to_numpy())):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    slope_all = tfit(z, y)
    ss = np.unique(s)
    sl = {int(q): float(tfit(z[s == q], y[s == q])) for q in ss}
    loso = np.zeros(len(y))
    for q in ss:
        m = s == q
        loso[m] = expit(tfit(z[~m], y[~m]) * z[m])
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(2000):
        pick = rng.choice(ss, len(ss))
        idx = np.concatenate([np.where(s == q)[0] for q in pick])
        bs.append(tfit(z[idx], y[idx]))
    out[nm] = {
        "raw_ll": float(ll(p, y).mean()),
        "loso_recal_ll": float(ll(loso, y).mean()),
        "slope_pooled": float(slope_all),
        "slope_ci": np.percentile(bs, [2.5, 97.5]).tolist(),
        "slope_P_lt1": float((np.array(bs) < 1).mean()),
        "slope_by_season": sl,
        "sd_logit": float(z.std()),
        "mean_p_home": float(p.mean()),
        "home_cover_rate": float(y.mean()),
    }
    q = pd.qcut(p, 5, duplicates="drop")
    t = (
        pd.DataFrame({"p": p, "y": y, "q": q})
        .groupby("q", observed=True)
        .agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
    )
    out[nm]["reliability"] = t.round(4).reset_index(drop=True).to_dict("list")
rawgain = ll(b.p.to_numpy(), y) - ll(a.p.to_numpy(), y)
n_, d_ = seasonstat(rawgain)
ci, pp = boot(n_, d_)
out["raw_ll_gain"] = {"mean": float(rawgain.mean()), "ci": ci.tolist(), "P+": float(pp)}
sd_ratio = out["student"]["sd_logit"] / out["base"]["sd_logit"]
out["sd_ratio"] = float(sd_ratio)
f = pd.read_parquet(R + "sim10_wf/forward_2026_joint/scored_hgb_nomkt_open.parquet")
for nm in ("p_base", "p_arm"):
    z = logit(np.clip(f[nm].to_numpy(), 1e-6, 1 - 1e-6))
    yy = f.y_home.to_numpy().astype(float)
    rng = np.random.default_rng(2)
    bs = [tfit(z[j], yy[j]) for j in (rng.integers(0, len(f), len(f)) for _ in range(2000))]
    out["f26_" + nm] = {
        "n": len(f),
        "raw_ll": float(ll(f[nm].to_numpy(), yy).mean()),
        "slope": float(tfit(z, yy)),
        "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
        "sd_logit": float(z.std()),
    }
print(json.dumps(out, indent=1))
