import glob
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H

SEED = 22
NBOOT = 2000
NPERM = 2000
FAMILY = "mkt22_move_follows_traits"
OPENER = "artifacts/opener_evaluation/20261101T151614Z/per_game.parquet"
OUT = "artifacts/mkt22/cells.json"
rng = np.random.default_rng(SEED)

sched = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])
sched = sched[sched.game_type.eq("REG") & sched.result.notna() & (sched.season <= 2025)]
d = pd.read_parquet("data/pbp/team_style/raw_pbp_narrow.parquet")

f = d[
    (d.down == 4)
    & d.play_type.isin(["pass", "run", "punt", "field_goal"])
    & (d.qtr <= 4)
    & d.season_type.eq("REG")
]
f = f[~f.qb_kneel.fillna(0).astype(bool) & ~f.qb_spike.fillna(0).astype(bool)]
f = f.dropna(
    subset=["yardline_100", "ydstogo", "score_differential", "game_seconds_remaining"]
).copy()
f["go"] = f.play_type.isin(["pass", "run"]).astype(int)
f = f.merge(sched[["game_id", "home_coach", "away_coach"]], on="game_id")
f["coach"] = np.where(f.posteam == f.home_team, f.home_coach, f.away_coach)
X = f[["yardline_100", "ydstogo", "score_differential", "game_seconds_remaining", "qtr"]].values
f["p"] = np.nan
for sn in sorted(f.season.unique()):
    m = (f.season == sn).values
    clf = H(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=12, min_samples_leaf=60, random_state=0
    )
    clf.fit(X[~m], f.go.values[~m])
    f.loc[m, "p"] = clf.predict_proba(X[m])[:, 1]
f["res"] = f.go - f.p
smean = f.groupby("season").res.mean()
f["res_c"] = f.res - f.season.map(smean.shift(1)).fillna(0.0)
tg = (
    f.groupby(["game_id", "posteam", "coach", "season"])
    .agg(S=("res_c", "sum"), n=("res_c", "size"))
    .reset_index()
    .merge(sched[["game_id", "week"]], on="game_id")
    .sort_values(["posteam", "coach", "season", "week"])
)
tg["cS"] = tg.groupby(["posteam", "coach"]).S.cumsum() - tg.S
tg["cn"] = tg.groupby(["posteam", "coach"]).n.cumsum() - tg.n
cs = tg.groupby(["posteam", "coach", "season"]).agg(S=("S", "sum"), n=("n", "sum")).reset_index()
cs = cs[cs.n >= 10]
cs["m"] = cs.S / cs.n
kf = {}
for sn in sorted(tg.season.unique()):
    t = cs[cs.season != sn]
    ve = f[f.season != sn].res_c.var()
    vu = max(t.m.var() - (ve / t.n).mean(), 1e-9)
    kf[sn] = ve / vu
tg["idx"] = tg.cS / (tg.cn + tg.season.map(kf))
agg = tg[["game_id", "posteam", "idx"]]

ahead = d[(d.qtr == 4) & d.posteam.notna() & d.fixed_drive.notna()].sort_values(
    ["game_id", "play_id"]
)
ahead = ahead[(ahead.qb_kneel != 1) & (ahead.qb_spike != 1) & (ahead.aborted_play != 1)]
ahead = ahead[(ahead.pass_attempt == 1) | (ahead.rush_attempt == 1)]
first = ahead.groupby(["game_id", "posteam", "fixed_drive"]).first().reset_index()
lead = first.loc[first.score_differential >= 8, ["game_id", "posteam", "fixed_drive"]]
ahead = ahead.merge(lead, on=["game_id", "posteam", "fixed_drive"])
ahead = ahead.dropna(subset=["pass_oe"])
ag = (
    ahead.groupby(["game_id", "posteam"])
    .agg(oe=("pass_oe", "sum"), n=("pass_oe", "size"))
    .reset_index()
)
tl = pd.concat(
    [
        sched[["game_id", "gameday", "home_team"]].rename(columns={"home_team": "posteam"}),
        sched[["game_id", "gameday", "away_team"]].rename(columns={"away_team": "posteam"}),
    ]
)
ag = tl.merge(ag[["game_id", "posteam", "oe", "n"]], on=["game_id", "posteam"], how="left")
ag[["oe", "n"]] = ag[["oe", "n"]].fillna(0.0)
ag = ag.sort_values(["posteam", "gameday"])
ag["co"] = ag.groupby("posteam").oe.cumsum() - ag.oe
ag["cn"] = ag.groupby("posteam").n.cumsum() - ag.n
ag["tr"] = ag.co / ag.cn.replace(0, np.nan)


def pair(src, col, name):
    h = sched[["game_id", "home_team", "away_team"]]
    a = src.rename(columns={"posteam": "home_team", col: "h"})[["game_id", "home_team", "h"]]
    b = src.rename(columns={"posteam": "away_team", col: "a"})[["game_id", "away_team", "a"]]
    o = h.merge(a, on=["game_id", "home_team"]).merge(b, on=["game_id", "away_team"])
    o[name] = o.h - o.a
    return o[["game_id", name]]


m37 = pd.read_parquet("artifacts/mod37/games.parquet")[["game_id", "stadium", "excess"]]
m37 = m37.merge(sched[["game_id", "gameday"]], on="game_id").sort_values(["stadium", "gameday"])
m37["ce"] = m37.groupby("stadium").excess.cumsum() - m37.excess
m37["cn"] = m37.groupby("stadium").cumcount()
m37["fs"] = m37.ce / m37.cn.replace(0, np.nan)

traits = (
    sched[["game_id", "season", "week"]]
    .merge(m37[["game_id", "fs"]], on="game_id", how="left")
    .merge(pair(agg, "idx", "agg"), on="game_id", how="left")
    .merge(pair(ag.rename(columns={"tr": "trv"}), "trv", "pass_ahead"), on="game_id", how="left")
)

o = pd.read_parquet(OPENER)[
    ["game_id", "tue_open_home_spread", "close_home_spread", "margin_vs_open"]
]
o = o.rename(columns={"tue_open_home_spread": "open", "close_home_spread": "close"})
o["era"] = "tue_2020_25"
s = pd.read_parquet("data/processed/sbr_odds.parquet")
s = s[
    (s.season >= 2011)
    & (s.season <= 2019)
    & s.game_id.notna()
    & s.open_home_spread.notna()
    & s.close_home_spread.notna()
]
s = s.rename(columns={"open_home_spread": "open", "close_home_spread": "close"})
s["margin_vs_open"] = (s.home_score - s.away_score) - s.open
s = s[["game_id", "open", "close", "margin_vs_open"]].assign(era="sbr_2011_19")
sc = sched[["game_id", "result", "spread_line"]]
chk = s.merge(sc, on="game_id")
print(
    "sbr sign check corr(close, spread_line)", np.corrcoef(chk.close, chk.spread_line)[0, 1], len(s)
)
lines = pd.concat([o, s]).merge(traits, on="game_id")
lines["move"] = lines.close - lines.open


def design(df, tcol):
    W = pd.get_dummies(df.season.astype(str) + "_" + df.week.astype(str), drop_first=True).to_numpy(
        float
    )
    return np.column_stack([np.ones(len(df)), df[tcol].to_numpy(float), df.open.to_numpy(float), W])


def fit(Xm, y):
    return np.linalg.lstsq(Xm, y, rcond=None)[0]


def cell(df, tcol, ycol, extra):
    df = df.dropna(subset=[tcol, ycol]).reset_index(drop=True)
    if extra:
        df = df[df.margin_vs_open != 0].reset_index(drop=True)
    sd = df[tcol].std()
    df[tcol] = df[tcol] / sd
    Xm = design(df, tcol)
    if extra:
        Xm = np.column_stack([Xm, df.move.to_numpy(float)])
    y = df[ycol].to_numpy(float)
    est = fit(Xm, y)[1]
    n = len(df)
    bs = np.empty(NBOOT)
    for i in range(NBOOT):
        ix = rng.integers(0, n, n)
        bs[i] = fit(Xm[ix], y[ix])[1]
    blk = df.groupby(["season", "week"]).indices
    nl = np.empty(NPERM)
    t0 = df[tcol].to_numpy(float)
    for i in range(NPERM):
        t = t0.copy()
        for ix in blk.values():
            t[ix] = t0[rng.permutation(ix)]
        Xp = Xm.copy()
        Xp[:, 1] = t
        nl[i] = fit(Xp, y)[1]
    return {
        "n": n,
        "slope_per_sd": float(est),
        "se": float(bs.std()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
        "p_pos": float((bs > 0).mean()),
        "perm_p_two": float((np.abs(nl) >= abs(est)).mean()),
        "perm_sd": float(nl.std()),
        "trait_sd": float(sd),
    }


res = {}
for era, df in lines.groupby("era"):
    for tcol in ("fs", "agg", "pass_ahead"):
        for ycol, extra, tag in (
            ("move", False, "move"),
            ("margin_vs_open", True, "cover_beyond_move"),
        ):
            k = f"{era}|{tcol}|{tag}"
            res[k] = cell(df.copy(), tcol, ycol, extra)
            print(k, json.dumps({a: round(b, 4) for a, b in res[k].items()}))
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
