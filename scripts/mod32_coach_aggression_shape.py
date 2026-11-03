import os

os.environ["OMP_NUM_THREADS"] = "1"
import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.linear_model import LogisticRegression

FAMILY = "mod32_coach_aggression_shape"
KEYS = (3, 7)
BASE = "artifacts/opener_evaluation/20261101T151614Z/per_game.parquet"
NBOOT = 2000
rng = np.random.default_rng(32)

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
sched = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])
sched = sched[sched.game_type.eq("REG") & sched.result.notna() & sched.spread_line.notna()]
sched = sched[sched.season <= 2025]
d = d.merge(sched[["game_id", "home_coach", "away_coach"]], on="game_id")
d["coach"] = np.where(d.posteam == d.home_team, d.home_coach, d.away_coach)
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
smean = d.groupby("season").res.mean()
d["res_c"] = d.res - d.season.map(smean.shift(1)).fillna(0.0)
tg = (
    d.groupby(["game_id", "posteam", "coach", "season"])
    .agg(S=("res_c", "sum"), n=("res_c", "size"))
    .reset_index()
)
tg = tg.merge(sched[["game_id", "week"]], on="game_id").sort_values(
    ["posteam", "coach", "season", "week"]
)
tg["cS"] = tg.groupby(["posteam", "coach"]).S.cumsum() - tg.S
tg["cn"] = tg.groupby(["posteam", "coach"]).n.cumsum() - tg.n
cs = tg.groupby(["posteam", "coach", "season"]).agg(S=("S", "sum"), n=("n", "sum")).reset_index()
cs = cs[cs.n >= 10]
cs["m"] = cs.S / cs.n
kfold = {}
for sn in sorted(tg.season.unique()):
    t = cs[cs.season != sn]
    ve = d[d.season != sn].res_c.var()
    vu = max(t.m.var() - (ve / t.n).mean(), 1e-9)
    kfold[sn] = ve / vu
print(
    "k per play (LOSO method of moments) min/median/max",
    round(min(kfold.values()), 1),
    round(float(np.median(list(kfold.values()))), 1),
    round(max(kfold.values()), 1),
)
tg["k"] = tg.season.map(kfold)
tg["idx"] = tg.cS / (tg.cn + tg.k)
gm = sched[
    ["game_id", "season", "week", "home_team", "away_team", "result", "spread_line"]
].merge(
    tg[["game_id", "posteam", "idx", "cn"]].rename(
        columns={"posteam": "home_team", "idx": "ih", "cn": "nh"}
    ),
    on=["game_id", "home_team"],
).merge(
    tg[["game_id", "posteam", "idx", "cn"]].rename(
        columns={"posteam": "away_team", "idx": "ia", "cn": "na"}
    ),
    on=["game_id", "away_team"],
)
gm = gm[gm.season >= 2010].copy()
gm["A"] = gm.ih + gm.ia
gm["Z"] = gm.A / gm.A.std()
gm["absS"] = gm.spread_line.abs()
gm["ae"] = (gm.result - gm.spread_line).abs()
gm["key"] = gm.result.abs().isin(KEYS).astype(int)
print(
    "games",
    len(gm),
    "seasons",
    gm.season.min(),
    gm.season.max(),
    "A sd",
    round(gm.A.std(), 4),
    "share both spells with prior plays",
    round(((gm.nh > 0) & (gm.na > 0)).mean(), 3),
)


def ols(Xm, y):
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return b


def logit_fit(Xm, y):
    m = LogisticRegression(penalty=None, max_iter=500).fit(Xm, y)
    return np.r_[m.intercept_, m.coef_[0]]


def sig(z):
    return 1 / (1 + np.exp(-z))


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


for out in ("ae", "key"):
    insamp, oos, coefs = [], [], []
    for sn in sorted(gm.season.unique()):
        tr, te = gm[gm.season != sn], gm[gm.season == sn]
        X0t, X1t = tr[["absS"]].values, tr[["absS", "Z"]].values
        X0e, X1e = te[["absS"]].values, te[["absS", "Z"]].values
        if out == "ae":
            b0 = ols(np.c_[np.ones(len(tr)), X0t], tr.ae.values)
            b1 = ols(np.c_[np.ones(len(tr)), X1t], tr.ae.values)
            coefs.append(b1[2])
            e0 = np.mean((te.ae.values - np.c_[np.ones(len(te)), X0e] @ b0) ** 2)
            e1 = np.mean((te.ae.values - np.c_[np.ones(len(te)), X1e] @ b1) ** 2)
            i0 = np.mean((tr.ae.values - np.c_[np.ones(len(tr)), X0t] @ b0) ** 2)
            i1 = np.mean((tr.ae.values - np.c_[np.ones(len(tr)), X1t] @ b1) ** 2)
        else:
            b0 = logit_fit(X0t, tr.key.values)
            b1 = logit_fit(X1t, tr.key.values)
            coefs.append(b1[2])
            e0 = ll(te.key.values, sig(b0[0] + X0e @ b0[1:]))
            e1 = ll(te.key.values, sig(b1[0] + X1e @ b1[1:]))
            i0 = ll(tr.key.values, sig(b0[0] + X0t @ b0[1:]))
            i1 = ll(tr.key.values, sig(b1[0] + X1t @ b1[1:]))
        oos.append(e0 - e1)
        insamp.append(i0 - i1)
    coefs = np.array(coefs)
    full = gm[["absS", "Z"]].values
    if out == "ae":
        bf = ols(np.c_[np.ones(len(gm)), full], gm.ae.values)[2]
    else:
        bf = logit_fit(full, gm.key.values)[2]
    bs = []
    for _ in range(NBOOT):
        g = gm.iloc[rng.integers(0, len(gm), len(gm))]
        if out == "ae":
            bs.append(ols(np.c_[np.ones(len(g)), g[["absS", "Z"]].values], g.ae.values)[2])
        else:
            bs.append(logit_fit(g[["absS", "Z"]].values, g.key.values)[2])
    bs = np.array(bs)
    print(
        out,
        "pooled coef on Z",
        round(bf, 4),
        "CI95",
        np.quantile(bs, [0.025, 0.975]).round(4),
        "P(coef>0)",
        round(float((bs > 0).mean()), 3),
        "| per-fold coef>0:",
        int((coefs > 0).sum()),
        "of",
        len(coefs),
        "| held-out gain mean",
        round(float(np.mean(oos)), 5),
        "seasons gain>0",
        int((np.array(oos) > 0).sum()),
        "| in-sample gain",
        round(float(np.mean(insamp)), 5),
        "gap",
        round(float(np.mean(insamp) - np.mean(oos)), 5),
    )

b = pd.read_parquet(BASE)
b = b.merge(gm[["game_id", "Z"]], on="game_id", how="left")
b = b[b.margin_vs_open != 0].dropna(
    subset=["Z", "home_cover_probability_excluding_push_at_open"]
).copy()
b = b.reset_index(drop=True)
b["y"] = (b.margin_vs_open > 0).astype(int)
b["q"] = b.home_cover_probability_excluding_push_at_open.clip(1e-4, 1 - 1e-4)
b["lq"] = np.log(b.q / (1 - b.q))
print("cover games (no push)", len(b), "seasons", sorted(b.season.unique()))


def nll(y, lg):
    return -np.sum(y * -np.log1p(np.exp(-lg)) + (1 - y) * -np.log1p(np.exp(lg)))


def fit_arm(arm, tr):
    y, lq, z = tr.y.values, tr.lq.values, tr.Z.values
    if arm == "base":
        return (0.0, 0.0)
    if arm == "intercept":
        return (minimize_scalar(lambda a: nll(y, a + lq)).x, 0.0)
    if arm == "shape":
        r = minimize_scalar(lambda c: nll(y, lq * (1 + c * z)), bounds=(-2, 2), method="bounded")
        return (0.0, r.x)
    a0 = 0.0
    for _ in range(30):
        c = minimize_scalar(
            lambda c, a0=a0: nll(y, a0 + lq * (1 + c * z)), bounds=(-2, 2), method="bounded"
        ).x
        a0 = minimize_scalar(lambda a, c=c: nll(y, a + lq * (1 + c * z))).x
    return (a0, c)


def pred(par, te):
    return sig(par[0] + te.lq.values * (1 + par[1] * te.Z.values))


seasons = sorted(b.season.unique())
res = {}
for arm in ("base", "intercept", "shape", "intercept+shape"):
    p = pd.Series(index=b.index, dtype=float)
    ins, cf = [], []
    for sn in seasons:
        tr, te = b[b.season != sn], b[b.season == sn]
        par = fit_arm(arm, tr)
        cf.append(par)
        p[te.index] = pred(par, te)
        ins.append(ll(tr.y.values, pred(par, tr)))
    res[arm] = (p, cf, np.mean(ins))
pb = res["base"][0]
for arm, (p, cf, ins) in res.items():
    pick = (p > 0.5).astype(int)
    flips = pick != (pb > 0.5).astype(int)
    dec = int(((pick == b.y) & flips).sum()), int(((pick != b.y) & flips).sum())
    oos_ll = ll(b.y.values, p.values)
    print(
        arm,
        "acc",
        round((pick == b.y).mean(), 4),
        "logloss",
        round(oos_ll, 5),
        "brier",
        round(np.mean((p - b.y) ** 2), 5),
        "in-sample LL",
        round(ins, 5),
        "gap(oos-in)",
        round(oos_ll - ins, 5),
        "decisive right/wrong",
        dec,
        "per-fold (a,c)",
        [(round(a, 3), round(c, 3)) for a, c in cf],
    )


def pl(p):
    return -(b.y * np.log(p) + (1 - b.y) * np.log(1 - p))


ps, pi = res["shape"][0], res["intercept"][0]
for name, ref in (("base", pb), ("intercept", pi)):
    dl = pl(ref) - pl(ps)
    bsd = np.array([dl.values[rng.integers(0, len(dl), len(dl))].mean() for _ in range(NBOOT)])
    print(
        "shape vs",
        name,
        "per-game logloss gain",
        round(dl.mean(), 6),
        "CI95",
        np.quantile(bsd, [0.025, 0.975]).round(6),
        "P(gain>0)",
        round(float((bsd > 0).mean()), 3),
    )
