import glob
import os

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

R = "F:/Repos/nfl_py3/data/"
PART = R + "players/participation/raw/20260813T131635Z/season={s}/participation.parquet"
PBP = R + "pbp/raw/20260925T202544Z/season={s}/plays.parquet"
OPENER = "F:/Repos/nfl_py3/artifacts/opener_evaluation/20260929T192743Z/per_game.parquet"
OUT = "F:/Repos/nfl_py3/tests/scratch/mod42"
OLP = {"T", "G", "C", "OL"}
SEASONS = range(2020, 2026)
BOOT = 5000
os.makedirs(OUT, exist_ok=True)
ros = pd.read_parquet(
    max(glob.glob(R + "players/raw/*/weekly_rosters.parquet")),
    columns=["season", "position", "gsis_id", "week"],
)
ros = (
    ros.dropna(subset=["gsis_id"])
    .sort_values("week")
    .groupby(["season", "gsis_id"])
    .position.last()
)
rows = []
for s in SEASONS:
    pt = pd.read_parquet(PART.format(s=s), columns=["game_id", "play_id", "offense_players"])
    pb = pd.read_parquet(PBP.format(s=s))
    pb = pb[(pb.season_type == "REG") & pb.posteam.notna()]
    d = pb.merge(pt, on=["game_id", "play_id"])
    d = d[d.play_type.isin(["pass", "run"]) & d.offense_players.notna()].sort_values(
        ["game_id", "play_id"],
    )
    for (gid, team), g in d.groupby(["game_id", "posteam"], sort=False):
        ids = g.offense_players.iloc[0].split(";")
        ol = sorted(p for p in ids if ros.get((s, p), "?") in OLP)
        rows.append(
            {
                "game_id": gid,
                "season": s,
                "week": int(g.week.iloc[0]),
                "team": team,
                "opp": g.defteam.iloc[0],
                "five": ";".join(ol) if len(ol) == 5 else None,
                "db": g.qb_dropback.fillna(0).sum(),
                "pr": (g.sack.fillna(0) + g.qb_hit.fillna(0)).sum(),
            },
        )
tg = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
opp = tg[["game_id", "team", "db", "pr"]].rename(
    columns={"team": "opp", "db": "opp_db", "pr": "opp_pr"},
)
tg = tg.merge(opp, on=["game_id", "opp"], how="left")
streak, newst, press = {}, {}, {}
for (_s, t), g in tg.groupby(["season", "team"]):
    g = g.sort_values("week")
    fives = g.five.tolist()
    gids = g.game_id.tolist()
    cdb = np.concatenate([[0], np.cumsum(g.opp_db.values)[:-1]])
    cpr = np.concatenate([[0], np.cumsum(g.opp_pr.values)[:-1]])
    for i, gid in enumerate(gids):
        prior = [f for f in fives[:i] if isinstance(f, str)]
        k = 0
        for f in reversed(prior):
            if f != prior[-1]:
                break
            k += 1
        streak[(gid, t)] = k
        newst[(gid, t)] = (
            len(set(prior[-1].split(";")) - set(prior[-2].split(";"))) if len(prior) > 1 else 0
        )
        press[(gid, t)] = cpr[i] / cdb[i] if i >= 2 and cdb[i] > 0 else np.nan
o = pd.read_parquet(OPENER)
o = o[o.season.isin(list(SEASONS)) & o.margin_vs_open.ne(0)].copy()
gm = pd.concat(
    [
        pd.read_parquet(PBP.format(s=s), columns=["game_id", "home_team", "away_team"])
        .drop_duplicates("game_id")
        .assign(s=s)
        for s in SEASONS
    ],
)
o = o.merge(gm[["game_id", "home_team", "away_team"]], on="game_id")
for side in ("home", "away"):
    keys = list(zip(o.game_id, o[side + "_team"], strict=True))
    o[side + "_streak"] = [streak.get(k, np.nan) for k in keys]
    o[side + "_new"] = [newst.get(k, np.nan) for k in keys]
for side, oth in (("home", "away"), ("away", "home")):
    keys = list(zip(o.game_id, o[oth + "_team"], strict=True))
    o[side + "_opp_press"] = [press.get(k, np.nan) for k in keys]
for c in ("home_opp_press", "away_opp_press"):
    o[c] = (o[c] - o[c].mean()).fillna(0)
o["ls_h"] = np.log1p(o.home_streak.fillna(0))
o["ls_a"] = np.log1p(o.away_streak.fillna(0))
o["d_cont"] = o.ls_h - o.ls_a
o["d_new"] = -(o.home_new.fillna(0) - o.away_new.fillna(0))
o["d_inter"] = o.ls_h * o.home_opp_press - o.ls_a * o.away_opp_press
o["d_press"] = o.home_opp_press - o.away_opp_press
for c in ("d_cont", "d_new", "d_inter", "d_press"):
    o[c] = o[c] / o[c].std()
o["y"] = (o.margin_vs_open > 0).astype(int)
o["z"] = logit(o.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6))
d = o.reset_index(drop=True)
arms = {
    "intercept": [],
    "served": ["z"],
    "served+cont": ["z", "d_cont"],
    "served+new": ["z", "d_new"],
    "served+cont+press": ["z", "d_cont", "d_press"],
    "served+cont+inter": ["z", "d_cont", "d_inter"],
    "served+cont+new+inter": ["z", "d_cont", "d_new", "d_inter"],
}
P, coef, ins = {}, {}, {}
for a, cols in arms.items():
    p = pd.Series(0.5, index=d.index)
    coef[a] = {}
    for s in SEASONS:
        tr = d[d.season != s]
        te = d.season == s
        if not cols:
            p[te] = tr.y.mean()
            continue
        m = LogisticRegression(C=1e6, max_iter=1000).fit(tr[cols], tr.y)
        p[te] = m.predict_proba(d.loc[te, cols])[:, 1]
        coef[a][s] = np.round(m.coef_[0], 3).tolist()
    P[a] = p.clip(1e-6, 1 - 1e-6)
    if cols:
        ins[a] = (
            LogisticRegression(C=1e6, max_iter=1000).fit(d[cols], d.y).predict_proba(d[cols])[:, 1]
        )
    else:
        ins[a] = np.full(len(d), d.y.mean())


def ll(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return -(d.y.values * np.log(p) + (1 - d.y.values) * np.log(1 - p))


rng = np.random.default_rng(1)
sv = (P["served"] > 0.5).values
yv = (d.y == 1).values
print(
    "opener rows",
    len(o),
    "streak known both",
    int((o.home_streak.notna() & o.away_streak.notna()).sum()),
)
print("streak>0 share", round(float((o.home_streak > 0).mean()), 3))
print("streak dist", o.home_streak.value_counts().sort_index().head(10).to_dict())
print("new dist", o.home_new.value_counts().sort_index().to_dict())
print("n", len(d), "looks", len(arms) - 2)
for a in arms:
    pa = P[a].values
    w = int(((pa > 0.5) == yv).sum())
    print(
        a,
        f"W-L {w}-{len(d) - w}",
        f"LL {ll(pa).mean():.5f}",
        f"Brier {((pa - d.y.values) ** 2).mean():.5f}",
        f"insample LL {ll(ins[a]).mean():.5f} gap {ll(pa).mean() - ll(ins[a]).mean():.5f}",
    )
base = ll(P["served"])
for a in list(arms)[2:]:
    dv = base - ll(P[a])
    b = np.array([dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(BOOT)])
    sb = pd.Series(dv).groupby(d.season.values).mean()
    pa = (P[a] > 0.5).values
    fl = pa != sv
    dw = int((pa[fl] == yv[fl]).sum())
    sw = int((sv[fl] == yv[fl]).sum())
    print(
        a,
        "flipped",
        int(fl.sum()),
        f"model {dw}-{int(fl.sum()) - dw} served {sw}-{int(fl.sum()) - sw}",
    )
    print(
        "  dLL {:.5f} se {:.5f} CI[{:.5f},{:.5f}] P+ {:.3f}".format(
            dv.mean(), b.std(), *np.percentile(b, [2.5, 97.5]), (b > 0).mean()
        ),
        "seasons+",
        int((sb > 0).sum()),
        "/6",
    )
    print("  coefs/fold", coef[a])
print("served vs intercept dLL", float((ll(P["intercept"]) - base).mean()))
res = d.y.values - P["served"].values
tt = pd.concat(
    [
        pd.DataFrame(
            {
                "season": d.season,
                "team": d[side + "_team"],
                "week": d.week,
                "x": d["ls_" + side[0]] - d["ls_" + side[0]].mean(),
                "r": sgn * res,
            }
        )
        for side, sgn in (("home", 1), ("away", -1))
    ]
)
tt["prod"] = tt.x * tt.r
tt["half"] = tt.week % 2
ag = tt.pivot_table(
    index=["season", "team"], columns="half", values="prod", aggfunc="mean"
).dropna()
rho = float(np.corrcoef(ag[0], ag[1])[0, 1])
perm = np.array([np.corrcoef(ag[0], rng.permutation(ag[1].values))[0, 1] for _ in range(BOOT)])
print(
    "split-half corr",
    round(rho, 3),
    "n",
    len(ag),
    "SB",
    round(2 * rho / (1 + rho), 3),
    "perm sd",
    round(float(perm.std()), 3),
    "P(null>=obs)",
    float((perm >= rho).mean()),
)
