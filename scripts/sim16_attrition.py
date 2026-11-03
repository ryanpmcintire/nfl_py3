import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="tests/scratch/sim16")
ap.add_argument("--reps", type=int, default=200)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = "F:/Repos/nfl_py3/"
EFFECT = 0.038
OLP = {"T", "G", "C", "OL"}
rng = np.random.default_rng(16)


def latest(name):
    return sorted(glob.glob(R + f"data/players/raw/*/{name}.parquet"))[-1]


hz = pd.read_parquet(R + "tests/scratch/sim12/hz.parquet")
hz = hz[hz.grp == "OL"].copy()
hz["wk"] = hz.game.str.split("_").str[1].astype(int)
hz = hz.sort_values(["season", "team", "player", "wk"])
hz["prev_exit"] = hz.groupby(["season", "team", "player"]).exit.shift(1).fillna(0)
rn = pd.read_parquet(
    latest("weekly_rosters"), columns=["season", "team", "gsis_id", "full_name", "years_exp"]
)
rn = rn.dropna(subset=["gsis_id"]).drop_duplicates(["season", "team", "gsis_id"])
hz = hz.merge(rn.rename(columns={"gsis_id": "player"}), on=["season", "team", "player"], how="left")
inj = pd.read_parquet(latest("injuries"))
inj = inj[inj.game_type == "REG"]
q = (
    inj.assign(quest=(inj.report_status == "Questionable").astype(float))
    .groupby(["season", "team", "week", "gsis_id"])
    .quest.max()
    .reset_index()
    .rename(columns={"week": "wk", "gsis_id": "player"})
)
hz = hz.merge(q, on=["season", "team", "wk", "player"], how="left")
hz["quest"] = hz.quest.fillna(0)
sc = pd.read_parquet(latest("snap_counts"))
sc = sc[sc.game_type == "REG"]
sc = sc.groupby(["season", "week", "team", "player", "position"]).offense_snaps.sum().reset_index()
sc = sc.sort_values(["season", "team", "player", "week"])
sc["cum_prior"] = sc.groupby(["season", "team", "player"]).offense_snaps.cumsum() - sc.offense_snaps
sc["games_prior"] = sc.groupby(["season", "team", "player"]).cumcount()
sc["ppg"] = np.where(sc.games_prior > 0, sc.cum_prior / sc.games_prior.clip(lower=1), np.nan)
pp = sc[["season", "team", "player", "week", "ppg"]].rename(
    columns={"player": "full_name", "week": "wk"}
)
hz = hz.merge(pp, on=["season", "team", "full_name", "wk"], how="left")
ol = sc[sc.position.isin(OLP)]
names = hz.groupby(["season", "team", "wk"]).full_name.apply(set).to_dict()
rows = []
for (s, t, w), g in ol.groupby(["season", "team", "week"]):
    st = names.get((s, t, w), set())
    b = g[~g.player.isin(st)].ppg.dropna()
    top = b.max() if len(b) else 0.0
    sp = g[g.player.isin(st)].ppg.dropna()
    rows.append((s, t, w, top, sp.mean() if len(sp) else np.nan))
gp = pd.DataFrame(rows, columns=["season", "team", "wk", "bk_ppg", "st_ppg"])
gp["gap"] = (1 - gp.bk_ppg / gp.st_ppg).clip(0, 1).fillna(1.0)
hz = hz.merge(gp[["season", "team", "wk", "gap"]], on=["season", "team", "wk"], how="left")
hz["gap"] = hz.gap.fillna(1.0)
hz["load"] = hz.ppg.fillna(hz.ppg.mean())
hz["yrs"] = hz.years_exp.fillna(hz.years_exp.mean())
FEAT = ["prev_exit", "quest", "yrs", "load"]
CN = ["c_full", "c_flatgap", "c_hazonly"]


def fit(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (y - p) - 1e-6 * b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b, np.linalg.inv(H)


def pr(X, b):
    return 1 / (1 + np.exp(-X @ b))


def met(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (
        float(((p > 0.5) == (y > 0.5)).mean()),
        float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        float(((p - y) ** 2).mean()),
    )


mu, sd = hz[FEAT].mean(), hz[FEAT].std()


def hazmat(d):
    return np.column_stack([np.ones(len(d)), ((d[FEAT] - mu) / sd).values])


hz["h_full"] = np.nan
hz["h_flat"] = np.nan
hcoef = {}
for s in sorted(hz.season.unique()):
    tr, te = hz.season != s, hz.season == s
    b, _ = fit(hazmat(hz[tr]), hz.exit[tr].values.astype(float))
    hz.loc[te, "h_full"] = pr(hazmat(hz[te]), b)
    hz.loc[te, "h_flat"] = hz.exit[tr].mean()
    hcoef[int(s)] = b[1:].round(3).tolist()
bfull, cov = fit(hazmat(hz), hz.exit.values.astype(float))
print("hazard z-coefs", FEAT, bfull[1:].round(3), "se", np.sqrt(np.diag(cov))[1:].round(3))
print("exit rate", round(hz.exit.mean(), 4), "slots", len(hz), "mean gap", round(hz.gap.mean(), 3))
hz["c_full"] = hz.h_full * hz.gap
hz["c_flatgap"] = hz.h_flat * hz.gap
hz["c_hazonly"] = hz.h_full
tc = hz.groupby(["season", "team", "wk"])[CN].sum().reset_index()
print("mean team exp EPA loss/play", round(tc.c_full.mean() * EFFECT, 4))

pg = pd.read_parquet(R + "artifacts/opener_evaluation/20261101T151614Z/per_game.parquet")
pg["home"] = pg.game_id.str.split("_").str[3]
pg["away"] = pg.game_id.str.split("_").str[2]
for side in ("home", "away"):
    m = tc.rename(columns={"team": side, "wk": "week"})
    m = m.rename(columns={c: f"{side}_{c}" for c in CN})
    pg = pg.merge(
        m[["season", side, "week", *[f"{side}_{c}" for c in CN]]],
        on=["season", side, "week"],
        how="left",
    )
pg = pg[(pg.margin_vs_open != 0) & pg.home_cover_probability_excluding_push_at_open.notna()]
pg = pg.dropna(subset=["home_c_full", "away_c_full"]).reset_index(drop=True)
pg["y"] = (pg.margin_vs_open > 0).astype(float)
pb = pg.home_cover_probability_excluding_push_at_open.clip(1e-4, 1 - 1e-4)
pg["lb"] = np.log(pb / (1 - pb))
SE = list(range(2020, 2026))


def design(df, d):
    return np.column_stack([np.ones(len(df)), df.lb.values] + ([d] if d is not None else []))


def loso(df, d):
    out = np.zeros(len(df))
    co = []
    X = design(df, d)
    for s in SE:
        tr, te = (df.season != s).values, (df.season == s).values
        b, _ = fit(X[tr], df.y.values[tr])
        out[te] = pr(X[te], b)
        co.append(b)
    return out, np.array(co)


def insamp(df, d):
    X = design(df, d)
    b, cv = fit(X, df.y.values)
    return met(pr(X, b), df.y.values), b, np.sqrt(np.diag(cv))


def ll(p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


y = pg.y.values
p0, _ = loso(pg, None)
m0 = met(p0, y)
i0 = insamp(pg, None)[0]
print("n", len(pg), "control oos acc/ll/brier", np.round(m0, 4), "ins ll", round(i0[1], 4))
ll0 = ll(p0)
grp = pg.season.values * 100 + pg.week.values
cells = {}
for name in CN:
    H, A = pg["home_" + name].values, pg["away_" + name].values
    d = H - A
    d = d / d.std()
    p1, co = loso(pg, d)
    m1 = met(p1, y)
    ims, b, se = insamp(pg, d)
    dec = (p1 > 0.5) != (p0 > 0.5)
    dl = ll0 - ll(p1)
    bs = np.array([dl[rng.integers(0, len(y), len(y))].mean() for _ in range(500)])
    pl = []
    for _ in range(args.reps):
        pool = np.concatenate([H, A])
        gg = np.concatenate([grp, grp])
        out = pool.copy()
        for k in np.unique(gg):
            ix = np.flatnonzero(gg == k)
            out[ix] = pool[rng.permutation(ix)]
        dp = out[: len(pg)] - out[len(pg) :]
        dp = dp / dp.std()
        pp1, _ = loso(pg, dp)
        pl.append(float((ll0 - ll(pp1)).mean()))
    pl = np.array(pl)
    obs = float(dl.mean())
    cells[name] = {
        "coef_ins": float(b[2]),
        "se": float(se[2]),
        "p_coef_pos": float(0.5 * (1 + math.erf(b[2] / se[2] / 2**0.5))),
        "fold_coefs": co[:, 2].round(3).tolist(),
        "oos": m1,
        "ins": ims,
        "gap_ll": m1[1] - ims[1],
        "ll_improve": obs,
        "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "p_improve_positive": float((bs > 0).mean()),
        "decisive_n_term_ctrl": [
            int(dec.sum()),
            int(((p1 > 0.5) == (y > 0.5))[dec].sum()),
            int(((p0 > 0.5) == (y > 0.5))[dec].sum()),
        ],
        "placebo_mean": float(pl.mean()),
        "placebo_p_ge_obs": float((pl >= obs).mean()),
    }
    print(name, json.dumps(cells[name]))
av = pd.read_parquet(R + "data/processed/game_features_player_learned_availability.parquet")
ac = [
    c
    for c in av.columns
    if c.startswith("diff_")
    and any(k in c for k in ("injury", "qb_start_probability", "continuity", "roster_cont"))
]
j = pg.merge(av[["game_id", *ac]], on="game_id")
j["d"] = j.home_c_full - j.away_c_full
print(len(j), {a: round(float(j.d.corr(j[a], method="spearman")), 3) for a in ac})
with open(args.out + "/cells.json", "w") as fh:
    json.dump({"cells": cells, "hazard_folds": hcoef}, fh, default=float)
