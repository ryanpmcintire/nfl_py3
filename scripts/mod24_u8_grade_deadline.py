import glob
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nfl_ats.mass_preserving_lattice import band_read

rng = np.random.default_rng(20261020)
sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])
sch = sch[sch.game_type == "REG"]
pool = sch[(sch.season < 2025) & sch.result.notna() & sch.spread_line.notna()]
pl = pool.spread_line.to_numpy(float)
pr = pool.result.to_numpy(float)
kick = pd.to_datetime(sch.gameday + " " + sch.gametime.fillna("13:00"))
kick = kick.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
kick = pd.Series(kick.values, index=sch.game_id.values)
pg = pd.read_parquet(sorted(glob.glob("artifacts/pick_probability/*/per_game.parquet"))[-1])
pg = pg[pg.season == 2025][
    ["game_id", "tue_open_home_spread", "result", "out_of_season_tuesday_home_probability"]
]
pg = pg.rename(columns={"out_of_season_tuesday_home_probability": "p_served"})
pm = pd.read_parquet("data/processed/prediction_markets.parquet")
pm = pm[(pm.point == "deadline") & (pm.season == 2025) & pm.implied_home_spread.notna()]
pm = pm.assign(ts=pd.to_datetime(pm.point_utc, utc=True))
late = pm.ml_last_ts.notna() & (pd.to_datetime(pm.ml_last_ts, unit="s", utc=True) > pm.ts)
assert not late.any()
assert (pm.ts.values <= kick.reindex(pm.game_id).values).all()
looks = 0


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(X, y):
    def nl(b):
        return ll(y, 1 / (1 + np.exp(-X @ b))) * len(y)

    return minimize(nl, np.zeros(X.shape[1]), method="BFGS").x


def boot(v, byweek, n=4000):
    ks = np.unique(byweek)
    idx = {k: np.where(byweek == k)[0] for k in ks}
    out = []
    for _ in range(n):
        pick = rng.choice(ks, len(ks))
        out.append(np.concatenate([v[idx[k]] for k in pick]).mean())
    out = np.array(out)
    return v.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), float((out > 0).mean())


res = {}
for src in ["kalshi", "poly"]:
    f = pm[pm.source == src].merge(pg, on="game_id")
    L = f.tue_open_home_spread.to_numpy(float)
    sgn = np.sign(np.corrcoef(f.implied_home_spread, L)[0, 1])
    corr = abs(np.corrcoef(f.implied_home_spread, L)[0, 1])
    assert corr > 0.9
    M = sgn * f.implied_home_spread.to_numpy(float)
    pc = []
    for line, m in zip(L, M):
        r = band_read(pl, pr, float(line), float(m))
        pc.append(r.cover / max(r.cover + r.loss, 1e-9))
    f["p_mkt"] = pc
    f["d"] = M - L
    dec = f[(f.result != f.tue_open_home_spread)].reset_index(drop=True)
    y = (dec.result > dec.tue_open_home_spread).astype(float).to_numpy()
    wk = dec.week.to_numpy()
    ms = (dec.p_mkt > 0.5).to_numpy()
    ss = (dec.p_served > 0.5).to_numpy()
    mk_ok = (ms == (y == 1)).astype(float)
    sv_ok = (ss == (y == 1)).astype(float)
    dis = ms != ss
    print(f"== {src} n_deadline_games {f.shape[0]} decisive {len(dec)} corr_with_opener {corr:.3f}")
    print(
        f"decisive disagreement games {int(dis.sum())}: market side {int(mk_ok[dis].sum())}-{int((1 - mk_ok[dis]).sum())}, served side {int(sv_ok[dis].sum())}-{int((1 - sv_ok[dis]).sum())}"
    )
    a = boot((mk_ok - 0.5) * 100, wk)
    b = boot((mk_ok - sv_ok) * 100, wk)
    c = boot((sv_ok - 0.5) * 100, wk)
    print(
        f"market side acc pts v 50: {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] P+ {a[3]:.3f}; served side {c[0]:.2f} [{c[1]:.2f},{c[2]:.2f}]; market-minus-served {b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}] P+ {b[3]:.3f}"
    )
    X1 = lg(dec.p_served.to_numpy())[:, None]
    X2 = lg(dec.p_mkt.to_numpy())[:, None]
    X3 = np.column_stack([X1[:, 0], X2[:, 0]])
    ps = {k: np.zeros(len(y)) for k in ["const", "served", "mkt", "comb"]}
    coefs = []
    for w in np.unique(wk):
        te = wk == w
        tr = ~te
        ps["const"][te] = y[tr].mean()
        ps["served"][te] = 1 / (1 + np.exp(-X1[te, 0]))
        b2 = fit(np.column_stack([np.ones(tr.sum()), X2[tr]]), y[tr])
        ps["mkt"][te] = 1 / (1 + np.exp(-(b2[0] + b2[1] * X2[te, 0])))
        b3 = fit(np.column_stack([np.ones(tr.sum()), X3[tr]]), y[tr])
        coefs.append(b3)
        ps["comb"][te] = 1 / (1 + np.exp(-(b3[0] + b3[1] * X3[te, 0] + b3[2] * X3[te, 1])))
    coefs = np.array(coefs)
    for k, p in ps.items():
        print(
            f"  {k}: logloss {ll(y, p):.4f} brier {np.mean((p - y) ** 2):.4f} acc {np.mean((p > 0.5) == (y == 1)) * 100:.2f}"
        )
    cm = ((ps["comb"] > 0.5) == (y == 1)).astype(float)
    d2 = boot((cm - sv_ok) * 100, wk)
    print(
        f"  combined-minus-served acc pts {d2[0]:.2f} [{d2[1]:.2f},{d2[2]:.2f}] P+ {d2[3]:.3f}; changed sides {int(((ps['comb'] > 0.5) != ss).sum())}"
    )
    print(
        f"  fold coefs (a, served, mkt) mean {coefs.mean(0).round(3)} sd {coefs.std(0).round(3)} min_mkt {coefs[:, 2].min():.3f} max_mkt {coefs[:, 2].max():.3f} folds {len(coefs)}"
    )
    res[src] = dict(n=len(dec), market=a, mm_s=b, comb_s=d2)
print(json.dumps(res, default=float))
