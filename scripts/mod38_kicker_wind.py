import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

fs = sorted(
    glob.glob("F:/Repos/nfl_py3/data/raw/nflverse_pbp_full/20261103T131922Z/pbp_20*.parquet")
)
cols = [
    "game_id",
    "season",
    "season_type",
    "game_date",
    "field_goal_attempt",
    "field_goal_result",
    "kick_distance",
    "kicker_player_id",
    "roof",
    "temp",
    "wind",
    "posteam",
]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs])
d = d[
    (d.field_goal_attempt == 1)
    & d.field_goal_result.isin(["made", "missed", "blocked"])
    & d.kick_distance.notna()
    & d.season.between(2009, 2025)
].copy()
d["y"] = (d.field_goal_result == "made").astype(float)
d["indoor"] = d.roof.isin(["dome", "closed"]).astype(int)
d["wind"] = d.wind.fillna(0.0)
d.loc[d.indoor == 1, "wind"] = 0.0
d["temp"] = d.temp.fillna(d.temp.median())
d.loc[d.indoor == 1, "temp"] = d.temp.median()
d = d.sort_values(["game_date", "game_id"]).reset_index(drop=True)
print(len(d), d.kicker_player_id.nunique(), d.y.mean())
dm = d.kick_distance.mean()
d["dc"] = (d.kick_distance - dm) / 10
d["wd"] = d.wind / 10 * d.dc
d["w"] = d.wind / 10
X = np.c_[np.ones(len(d)), d.dc, d.dc**2, d.w, d.wd, (d.temp - d.temp.mean()) / 10, d.indoor]


def newton(X, y, off=None, ridge=1e-4, it=30):
    b = np.zeros(X.shape[1])
    off = 0 if off is None else off
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b + off)))
        g = X.T @ (y - p) - ridge * b
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(b))
        s = np.linalg.solve(H, g)
        b += s
        if abs(s).max() < 1e-8:
            break
    return b


bg = newton(X, d.y.values)
GLOBAL_NAMES = ["c", "dc", "dc2", "w", "wd", "temp", "indoor"]
print("global", dict(zip(GLOBAL_NAMES, bg.round(3), strict=False)))
d["off"] = X @ bg


def kslope(sub):
    Z = np.c_[np.ones(len(sub)), sub.wd.values]
    return newton(Z, sub.y.values, sub.off.values, ridge=1e-2)[1]


out = d[d.indoor == 0]
cnt = out[out.w > 0].groupby("kicker_player_id").size()
ks = cnt[cnt >= 40].index
print("kickers", len(ks))
groups = {k: g for k, g in out.groupby("kicker_player_id") if k in ks}


def split(rng=None, perm=False):
    a = []
    b = []
    for _k, g in groups.items():
        g = g.copy()
        if perm:
            g["wd"] = rng.permutation(g.wd.values)
        a.append(kslope(g.iloc[0::2]))
        b.append(kslope(g.iloc[1::2]))
    a = np.array(a)
    b = np.array(b)
    return np.corrcoef(a, b)[0, 1], a, b


r, a, b = split()
sb = 2 * r / (1 + r)
rng = np.random.default_rng(0)
nul = np.array([split(rng, True)[0] for _ in range(300)])
print("split-half r", r, "SB", sb, "null mean", nul.mean(), "sd", nul.std(), "p", (nul >= r).mean())
for _lab, _sub in [("dome", d[d.indoor == 1])]:
    pass
print("slope sd half a", a.std(), "b", b.std())
np.save("F:/Repos/nfl_py3/tests/scratch/mod38/bg.npy", bg)
d.to_pickle("F:/Repos/nfl_py3/tests/scratch/mod38/fg.pkl")

print("spearman", spearmanr(a, b))


def kslope2(sub):
    return newton(sub.wd.values[:, None], sub.y.values, sub.off.values, ridge=1.0)[0]


def split2(by):
    A = []
    B = []
    for _k, g in groups.items():
        m = (g[by] % 2 == 0).values if by != "i" else (np.arange(len(g)) % 2 == 0)
        A.append(kslope2(g[m]))
        B.append(kslope2(g[~m]))
    A = np.array(A)
    B = np.array(B)
    return np.corrcoef(A, B)[0, 1], spearmanr(A, B)[0], A.std(), B.std()


print("noint alt", split2("i"))
print("noint season parity", split2("season"))
