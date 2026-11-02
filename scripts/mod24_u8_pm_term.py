import glob
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nfl_ats.mass_preserving_lattice import band_read

G26 = "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/352ecbdc-246f-44c0-9f81-c287a6ef4436/scratchpad/agentB/g26.parquet"
rng = np.random.default_rng(20261102)
sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])
sch = sch[sch.game_type == "REG"]
pool = sch[(sch.season < 2025) & sch.result.notna() & sch.spread_line.notna()]
pl = pool.spread_line.to_numpy(float)
pr = pool.result.to_numpy(float)
kick = pd.to_datetime(sch.gameday + " " + sch.gametime.fillna("13:00")).dt.tz_localize("America/New_York")
gd = pd.to_datetime(sch.gameday)
off = np.where(gd.dt.weekday <= 1, -((gd.dt.weekday + 1) % 7), (6 - gd.dt.weekday) % 7)
sun = (gd + pd.to_timedelta(off, unit="D") + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York")
dl = pd.concat([kick, sun], axis=1).min(axis=1).dt.tz_convert("UTC")
dl = pd.Series(dl.values, index=sch.game_id.astype(str).values)

pg = pd.read_parquet(sorted(glob.glob("artifacts/pick_probability/20261101T171318Z/per_game.parquet"))[-1])
pg = pg[pg.season == 2025]
a25 = pd.DataFrame(
    dict(
        game_id=pg.game_id.astype(str),
        season=2025,
        week=pg.week,
        L=pg.tue_open_home_spread,
        move=pg.market_move_toward_home.fillna(0.0) * pg.market_move_available.fillna(0.0).clip(0, 1),
        p_served=pg.out_of_season_tuesday_home_probability,
        y=(pg.result > pg.tue_open_home_spread).astype(float),
        decisive=pg.result != pg.tue_open_home_spread,
    )
)
g = pd.read_parquet(G26)
g = g[g.settle_margin != 0]
ph = np.where((g.picked_side == "AWAY") & (g.final_probability > 0.5), 1 - g.final_probability, g.final_probability)
a26 = pd.DataFrame(
    dict(
        game_id=g.game_id.astype(str).values,
        season=2026,
        week=g.week.values,
        L=g.market_line.values,
        move=(g.market_move_toward_home * g.market_move_available).values,
        p_served=ph,
        y=g.home_covered.values,
        decisive=True,
    )
)
a26["deadline"] = pd.to_datetime(g.deadline.values, utc=True)
print("served-vs-discrete corr 2026", np.corrcoef(ph, g.discrete)[0, 1])
A = pd.concat([a25, a26], ignore_index=True)
A = A[A.decisive]
pm = pd.read_parquet("data/processed/prediction_markets.parquet")
pm = pm[(pm.point == "deadline") & (pm.season >= 2025) & pm.implied_home_spread.notna()]
pm = pm.assign(ts=pd.to_datetime(pm.point_utc, utc=True))
late = pm.ml_last_ts.notna() & (pd.to_datetime(pm.ml_last_ts, unit="s", utc=True) > pm.ts)
assert not late.any()
dd = dl.reindex(pm.game_id.astype(str)).values
ok = pm.season.values == 2025
assert (pm.ts.values[ok] <= dd[ok]).all(), "price after deadline"
d26 = A.set_index("game_id").deadline.reindex(pm.game_id.astype(str)).values
m26 = pm.season.values == 2026
chk = m26 & ~pd.isna(d26)
assert (pm.ts.values[chk] <= pd.to_datetime(d26[chk], utc=True).values).all(), "2026 price after deadline"


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(X, y):
    return minimize(lambda b: ll(y, 1 / (1 + np.exp(-X @ b))) * len(y), np.zeros(X.shape[1]), method="BFGS").x


def boot(v, wk, n=4000):
    ks = np.unique(wk)
    idx = {k: np.where(wk == k)[0] for k in ks}
    o = np.array([np.concatenate([idx[k] for k in rng.choice(ks, len(ks))]) for _ in range(n)], dtype=object)
    m = np.array([v[i.astype(int)].mean() for i in o])
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), float((m > 0).mean())


def pc(L, tilt):
    r = band_read(pl, pr, float(L), float(tilt))
    return r.cover / max(r.cover + r.loss, 1e-9)


out = {}
for src in ["kalshi", "poly"]:
    f = A.merge(pm[pm.source == src][["game_id", "implied_home_spread"]].assign(game_id=lambda d: d.game_id.astype(str)), on="game_id")
    c = np.corrcoef(f.implied_home_spread, f.L)[0, 1]
    assert abs(c) > 0.9
    M = np.sign(c) * f.implied_home_spread.to_numpy(float)
    f["z"] = [lg(pc(L, m)) - lg(pc(L, L + mv)) for L, m, mv in zip(f.L, M, f.move)]
    f = f.reset_index(drop=True)
    print(f"== {src} n {len(f)} 2025 {int((f.season == 2025).sum())} 2026 {int((f.season == 2026).sum())} corr {c:.3f} z sd {f.z.std():.3f}")
    r = {}
    for name, te_mask, mode in [("lowo2025", f.season == 2025, "lowo"), ("fwd2026", f.season == 2026, "fwd")]:
        t = f[te_mask].reset_index(drop=True)
        y = t.y.to_numpy()
        xs = lg(t.p_served.to_numpy())
        pc_ = np.zeros(len(t))
        coefs = []
        if mode == "lowo":
            for w in np.unique(t.week):
                te = (t.week == w).to_numpy()
                b = fit(np.column_stack([np.ones((~te).sum()), xs[~te], t.z[~te]]), y[~te])
                coefs.append(b)
                pc_[te] = 1 / (1 + np.exp(-(b[0] + b[1] * xs[te] + b[2] * t.z[te])))
        else:
            tr = f[f.season == 2025]
            b = fit(np.column_stack([np.ones(len(tr)), lg(tr.p_served.to_numpy()), tr.z]), tr.y.to_numpy())
            coefs.append(b)
            pc_ = 1 / (1 + np.exp(-(b[0] + b[1] * xs + b[2] * t.z.to_numpy())))
        ps = t.p_served.to_numpy()
        sv = ((ps > 0.5) == (y == 1)).astype(float)
        cb = ((pc_ > 0.5) == (y == 1)).astype(float)
        dis = (pc_ > 0.5) != (ps > 0.5)
        bt = boot((cb - sv) * 100, t.week.to_numpy())
        co = np.array(coefs)
        print(
            f"  {name} n {len(t)} served acc {sv.mean() * 100:.2f} comb acc {cb.mean() * 100:.2f} LL served {ll(y, ps):.4f} comb {ll(y, pc_):.4f} Brier served {np.mean((ps - y) ** 2):.4f} comb {np.mean((pc_ - y) ** 2):.4f}"
        )
        print(
            f"    comb-minus-served acc pts {bt[0]:.2f} [{bt[1]:.2f},{bt[2]:.2f}] P+ {bt[3]:.3f}; flipped {int(dis.sum())}: comb {int(cb[dis].sum())}-{int((1 - cb[dis]).sum())} served {int(sv[dis].sum())}-{int((1 - sv[dis]).sum())}; coef z mean {co[:, 2].mean():.3f} min {co[:, 2].min():.3f} max {co[:, 2].max():.3f} folds {len(co)} served-coef {co[:, 1].mean():.3f}"
        )
        r[name] = dict(n=len(t), acc=bt, flipped=int(dis.sum()))
    out[src] = r
print(json.dumps(out, default=float))
