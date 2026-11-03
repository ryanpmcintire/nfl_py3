import os

os.environ["OMP_NUM_THREADS"] = "1"
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H

rng = np.random.default_rng(14)
d = pd.read_parquet("data/pbp/team_style/raw_pbp_narrow.parquet")
d = d[
    (d.down == 4)
    & d.play_type.isin(["pass", "run", "punt", "field_goal"])
    & (d.qtr <= 4)
    & d.season_type.eq("REG")
]
d = d[~d.qb_kneel.fillna(0).astype(bool) & ~d.qb_spike.fillna(0).astype(bool)]
d = d.dropna(
    subset=["yardline_100", "ydstogo", "score_differential", "game_seconds_remaining"]
).copy()
d["go"] = d.play_type.isin(["pass", "run"]).astype(int)
s = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[
    ["game_id", "home_team", "home_coach", "away_coach"]
]
d = d.merge(s, on="game_id", suffixes=("", "_s"))
d["coach"] = np.where(d.posteam == d.home_team, d.home_coach, d.away_coach)
d = d.dropna(subset=["coach"])
X = d[["yardline_100", "ydstogo", "score_differential", "game_seconds_remaining", "qtr"]].values
d["p"] = np.nan
for sn in sorted(d.season.unique()):
    m = (d.season == sn).values
    clf = H(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=12, min_samples_leaf=60, random_state=0
    )
    clf.fit(X[~m], d.go.values[~m])
    d.loc[m, "p"] = clf.predict_proba(X[m])[:, 1]
d["res"] = d.go - d.p
d["res"] -= d.groupby("season").res.transform("mean")


def ll(y, p):
    return -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))


print(
    "plays",
    len(d),
    "go rate",
    d.go.mean().round(4),
    "LOSO logloss",
    ll(d.go, d.p).round(4),
    "base",
    ll(d.go, d.go.mean()).round(4),
)
d["wk"] = d.week
gk = (
    d[["season", "coach", "game_id", "wk"]].drop_duplicates().sort_values(["season", "coach", "wk"])
)
gk["half"] = gk.groupby(["season", "coach"]).cumcount() % 2
d = d.merge(gk[["game_id", "coach", "half"]], on=["game_id", "coach"])
d["cs"] = d.season.astype(str) + "|" + d.coach


def sh(res, cs, half, season, minn=5):
    t = pd.DataFrame({"r": res, "cs": cs, "h": half})
    g = t.groupby(["cs", "h"]).r.agg(["mean", "size"]).unstack()
    ok = (g["size"].fillna(0) >= minn).all(axis=1)
    a, b = g["mean"][0][ok], g["mean"][1][ok]
    return np.corrcoef(a, b)[0, 1], int(ok.sum())


r, n = sh(d.res.values, d.cs.values, d.half.values, d.season.values)
nul = []
for _ in range(1000):
    pr = d.res.values.copy()
    for _sn, ix in d.groupby("season").indices.items():
        pr[ix] = rng.permutation(pr[ix])
    nul.append(sh(pr, d.cs.values, d.half.values, d.season.values)[0])
nul = np.array(nul)
print(
    "L1 split-half r",
    round(r, 3),
    "n coach-seasons",
    n,
    "SB",
    round(2 * r / (1 + r), 3),
    "null mean",
    nul.mean().round(3),
    "p95",
    np.quantile(nul, 0.95).round(3),
    "perm p",
    (nul >= r).mean(),
)
cs = d.groupby(["season", "coach"]).res.agg(["mean", "size"]).reset_index()
a = cs.merge(cs.assign(season=cs.season - 1), on=["season", "coach"], suffixes=("0", "1"))
a = a[(a.size0 >= 10) & (a.size1 >= 10)]
rc = np.corrcoef(a.mean0, a.mean1)[0, 1]
nc = np.array([np.corrcoef(a.mean0, rng.permutation(a.mean1.values))[0, 1] for _ in range(2000)])
print(
    "L2 cross-season r",
    round(rc, 3),
    "pairs",
    len(a),
    "null p95",
    np.quantile(nc, 0.95).round(3),
    "perm p",
    (nc >= rc).mean(),
)
print(
    "sd of coach-season residual",
    cs[cs["size"] >= 10]["mean"].std().round(4),
    "mean n",
    cs["size"].mean().round(1),
)
bs = []
for _ in range(1000):
    ix = rng.integers(0, len(a), len(a))
    bs.append(np.corrcoef(a.mean0.values[ix], a.mean1.values[ix])[0, 1])
print(
    "L2 boot interval",
    np.quantile(bs, [0.025, 0.975]).round(3),
    "P(r>0)",
    np.mean(np.array(bs) > 0),
)
bs = []
co = d.groupby("cs").indices
sub = d[["cs", "half", "res", "season"]].reset_index(drop=True)
units = sub.cs.unique()
for _ in range(300):
    samp = rng.choice(units, len(units))
    pass


def pair_stats(x, y, label, nb=2000, seed=0):
    r_ = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r0 = np.corrcoef(x, y)[0, 1]
    nu = np.array([np.corrcoef(x, r_.permutation(y))[0, 1] for _ in range(nb)])
    bb = []
    for _ in range(nb):
        ix = r_.integers(0, len(x), len(x))
        bb.append(np.corrcoef(x[ix], y[ix])[0, 1])
    bb = np.array(bb)
    print(
        label,
        "n",
        len(x),
        "r",
        round(r0, 3),
        "boot",
        np.nanquantile(bb, [0.025, 0.975]).round(3),
        "P+",
        round(float(np.nanmean(bb > 0)), 3),
        "null p95",
        np.quantile(nu, 0.95).round(3),
        "perm p",
        round(float((nu >= r0).mean()), 4),
    )
    return r0, bb


cts = d.groupby(["season", "posteam", "coach"]).res.agg(["mean", "size"]).reset_index()
cts = cts[cts["size"] >= 10]
nxt = cts.assign(season=cts.season - 1)
pc = cts.merge(nxt, on="season", suffixes=("0", "1"))
mv = pc[(pc.coach0 == pc.coach1) & (pc.posteam0 != pc.posteam1)]
tc = pc[(pc.posteam0 == pc.posteam1) & (pc.coach0 != pc.coach1)]
st = pc[(pc.posteam0 == pc.posteam1) & (pc.coach0 == pc.coach1)]
r4, b4 = pair_stats(mv.mean0, mv.mean1, "L4 coach movers", seed=4)
r5, b5 = pair_stats(tc.mean0, tc.mean1, "L5 team across coach change", seed=5)
r6, b6 = pair_stats(st.mean0, st.mean1, "ref same coach same team", seed=6)
dd = b4 - np.random.default_rng(7).permutation(b5)
print(
    "L4-L5",
    round(r4 - r5, 3),
    "boot",
    np.quantile(dd, [0.025, 0.975]).round(3),
    "P(L4>L5)",
    round(float((dd > 0).mean()), 3),
)

ts = d.groupby(["season", "posteam"]).res.agg(["mean", "size"]).reset_index()
ts = ts[ts["size"] >= 10]
ss = pd.read_parquet(sorted(glob.glob("artifacts/staff_state/*/staff_state.parquet"))[-1])
ss = ss[ss.state.eq("filled")]
who = (
    ss.sort_values("first_observed_at")
    .groupby(["season", "team", "role"])
    .person.first()
    .unstack()
    .reset_index()
)
tp = ts.merge(ts.assign(season=ts.season - 1), on=["season", "posteam"], suffixes=("0", "1"))
tp = tp.merge(who.rename(columns={"team": "posteam"}), on=["season", "posteam"]).merge(
    who.rename(columns={"team": "posteam"}).assign(season=lambda z: z.season - 1),
    on=["season", "posteam"],
    suffixes=("", "_n"),
)
tp = tp.dropna(subset=["HC", "OC", "HC_n", "OC_n"])
sameh = tp.HC_n == tp.HC
sameo = tp.OC_n == tp.OC
for lab, m, sd in [
    ("OC: same HC same OC", sameh & sameo, 11),
    ("OC: same HC new OC", sameh & ~sameo, 12),
    ("OC: new HC same OC", ~sameh & sameo, 13),
    ("OC: new HC new OC", ~sameh & ~sameo, 14),
]:
    q = tp[m]
    if len(q) >= 8:
        pair_stats(q.mean0, q.mean1, lab, seed=sd)
    else:
        print(lab, "n", len(q))

NVD = Path("data/raw/nflverse_pbp_full/20261103T131922Z")
cols = [
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "home_team",
    "home_coach",
    "away_coach",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "two_point_attempt",
    "extra_point_attempt",
]
f = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(NVD.glob("pbp_20*.parquet"))])
f = f[f.season_type.eq("REG") & ((f.two_point_attempt == 1) | (f.extra_point_attempt == 1))]
f = f.dropna(subset=["score_differential", "game_seconds_remaining", "posteam"]).copy()
f["coach"] = np.where(f.posteam == f.home_team, f.home_coach, f.away_coach)
f = f.dropna(subset=["coach"]).reset_index(drop=True)
f["go"] = (f.two_point_attempt == 1).astype(int)
Xf = f[["score_differential", "game_seconds_remaining", "qtr"]].values
f["p"] = np.nan
for sn in sorted(f.season.unique()):
    m = (f.season == sn).values
    c = H(max_iter=60, learning_rate=0.08, max_leaf_nodes=6, min_samples_leaf=80, random_state=0)
    c.fit(Xf[~m], f.go.values[~m])
    f.loc[m, "p"] = c.predict_proba(Xf[m])[:, 1]
f["res"] = f.go - f.p
f["res"] -= f.groupby("season").res.transform("mean")
print("L3 tries", len(f), "2pt rate", f.go.mean().round(4))
gk = (
    f[["coach", "season", "week", "game_id"]]
    .drop_duplicates()
    .sort_values(["coach", "season", "week"])
)
gk["half"] = gk.groupby("coach").cumcount() % 2
f = f.merge(gk[["game_id", "coach", "half"]], on=["game_id", "coach"])
g = f.groupby(["coach", "half"]).res.agg(["mean", "size"]).unstack()
ok = (g["size"].fillna(0) >= 15).all(axis=1)
a3, b3 = g["mean"][0][ok].values, g["mean"][1][ok].values
r3, bb3 = pair_stats(a3, b3, "L3 two-point split-half (coach, 2009-2017, >=15 tries/half)", seed=3)
print("L3 SB", round(2 * r3 / (1 + r3), 3))


cl = d.groupby(["season", "posteam", "coach"]).res.agg(["mean", "size"]).reset_index()
cl = cl[cl["size"] >= 5].sort_values(["coach", "season"])
rows = []
for _c, g_ in cl.groupby("coach"):
    g_ = g_.reset_index(drop=True)
    for i in range(len(g_) - 1):
        a_, b_ = g_.iloc[i], g_.iloc[i + 1]
        if a_.posteam != b_.posteam and b_.season - a_.season <= 3:
            rows.append((a_["mean"], b_["mean"]))
mvr = np.array(rows)
pair_stats(mvr[:, 0], mvr[:, 1], "L4b coach movers gap<=3 min5", seed=41)
rows = []
for _t, g_ in cl.groupby("posteam"):
    g_ = g_.sort_values("season").reset_index(drop=True)
    for i in range(len(g_) - 1):
        a_, b_ = g_.iloc[i], g_.iloc[i + 1]
        if a_.coach != b_.coach and b_.season - a_.season <= 3:
            rows.append((a_["mean"], b_["mean"]))
tcr = np.array(rows)
pair_stats(tcr[:, 0], tcr[:, 1], "L5b team across coach change gap<=3 min5", seed=51)
