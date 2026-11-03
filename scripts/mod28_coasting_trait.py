import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = "F:/Repos/nfl_py3/"
SEED = 28
LEAD = 8
BOOT = 2000
PERM = 300
REL_START = 2012
EXT_START = 2009
LOSO_FIRST = 2020
LOSO_LAST = 2025
MIN_PRIOR_SEASONS = 3
MIN_GAMES_PER_SEASON = 2
CI_LO = 5
CI_HI = 95
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
OUT = ap.parse_args().out
rng = np.random.default_rng(SEED)
pb = pd.read_parquet(
    ROOT + "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=[
        "game_id",
        "season",
        "posteam",
        "fixed_drive",
        "qtr",
        "yardline_100",
        "game_seconds_remaining",
        "qb_kneel",
        "qb_spike",
        "pass_oe",
        "pass_attempt",
        "rush_attempt",
        "score_differential",
        "posteam_score",
        "posteam_score_post",
        "play_id",
        "aborted_play",
    ],
)
pb = pb[(pb.qtr == 4) & pb.posteam.notna() & pb.fixed_drive.notna()].sort_values(
    ["game_id", "play_id"]
)
pb = pb[(pb.qb_kneel != 1) & (pb.qb_spike != 1) & (pb.aborted_play != 1)]
pb = pb[(pb.pass_attempt == 1) | (pb.rush_attempt == 1)]
rows = []
for (gid, team, dr), d in pb.groupby(["game_id", "posteam", "fixed_drive"], sort=False):
    f = d.iloc[0]
    if not f.score_differential >= LEAD:
        continue
    pts = d.posteam_score_post.iloc[-1] - f.posteam_score
    gs = d.game_seconds_remaining.values
    el = float(gs[0] - gs[-1]) if len(d) > 1 else np.nan
    rows.append(
        (
            gid,
            int(f.season),
            team,
            dr,
            f.yardline_100,
            pts,
            d.pass_oe.mean(),
            el,
            len(d) - 1,
            int(d.pass_oe.notna().sum()),
        )
    )
dv = pd.DataFrame(
    rows, columns=["game_id", "season", "team", "drive", "yl", "pts", "oe", "el", "nint", "noe"]
)
dv = dv.dropna(subset=["yl", "pts"])
b = np.polyfit(dv.yl, dv.pts, 1)
dv["ppd"] = dv.pts - np.polyval(b, dv.yl)
dv["oe_w"] = dv.oe.fillna(0) * dv.noe
dv["el"] = dv.el.fillna(0)
g = (
    dv.groupby(["game_id", "season", "team"])
    .agg(
        ppd=("ppd", "mean"),
        oe_s=("oe_w", "sum"),
        noe=("noe", "sum"),
        el=("el", "sum"),
        nint=("nint", "sum"),
        ndr=("drive", "size"),
    )
    .reset_index()
)
g["oe"] = g.oe_s / g.noe.replace(0, np.nan)
g["sec"] = g.el / g.nint.replace(0, np.nan)
g = g.dropna(subset=["oe", "sec", "ppd"])
TR = {"oe": 1.0, "sec": -1.0, "ppd": 1.0}
pr = pd.read_parquet(
    ROOT + "artifacts/margins/20261103T172959Z/predictions.parquet",
    columns=[
        "game_id",
        "season",
        "week",
        "kickoff",
        "home_team",
        "away_team",
        "spread_line",
        "ats_margin",
        "result",
        "home_cover_probability_excluding_push",
    ],
)
pr = pr[pr.ats_margin.notna() & (pr.ats_margin != 0)].copy()
pr["y"] = (pr.ats_margin > 0).astype(float)
print(
    "sign check",
    np.corrcoef(pr.ats_margin, pr.result - pr.spread_line)[0, 1],
    np.corrcoef(pr.ats_margin, pr.result + pr.spread_line)[0, 1],
)
gk = pr[["game_id", "kickoff"]].drop_duplicates()


def sbf(r):
    return 2 * r / (1 + r)


def reliability(frame, order, start):
    parts = []
    for _, d in frame.groupby("season"):
        d = d.sort_values(["team", order]).copy()
        d["i"] = d.groupby("team").cumcount()
        parts.append(d.assign(h=d.i % 2))
    rl = pd.concat(parts)
    out = {}
    for t in TR:
        a = rl.groupby(["season", "team", "h"])[t].mean().unstack("h").dropna()
        a = a[a.index.get_level_values("season") >= start]
        r = np.corrcoef(a[0], a[1])[0, 1]
        n = len(a)
        bs = []
        for _ in range(BOOT):
            ix = rng.integers(0, n, n)
            bs.append(np.corrcoef(a[0].values[ix], a[1].values[ix])[0, 1])
        ss = a.index.get_level_values("season").values
        nullr = []
        for _ in range(BOOT):
            e = a[1].values.copy()
            for s in np.unique(ss):
                m = ss == s
                e[m] = rng.permutation(e[m])
            nullr.append(np.corrcoef(a[0].values, e)[0, 1])
        out[t] = {
            "n": n,
            "r_half": float(r),
            "sb": float(sbf(r)),
            "sb_ci": [float(sbf(np.percentile(bs, CI_LO))), float(sbf(np.percentile(bs, CI_HI)))],
            "p_pos_boot": float((np.array(bs) > 0).mean()),
            "perm_p": float((np.array(nullr) >= r).mean()),
            "null_sd": float(np.std(nullr)),
        }
        print(t, out[t])
    return out


g_all = g.copy()
g = g.merge(gk, on="game_id", how="inner").sort_values("kickoff")
print("team-games with leading drives", len(g), "seasons", g.season.min(), g.season.max())
relrows = reliability(g, "kickoff", REL_START)


def vc(d, t):
    x = d.set_index(["season", "team"])[t]
    k = x.groupby(level=[0, 1]).size()
    x = x[x.index.map(k) >= MIN_GAMES_PER_SEASON]
    k = x.groupby(level=[0, 1]).size()
    gm = x.groupby(level=[0, 1]).transform("mean")
    sw2 = ((x - gm) ** 2).sum() / (len(x) - len(k))
    N = len(x)
    n0 = (N - (k**2).sum() / N) / (len(k) - 1)
    msb = (k * (x.groupby(level=[0, 1]).mean() - x.mean()) ** 2).sum() / (len(k) - 1)
    return sw2, (msb - sw2) / n0, x.mean()


pr = pr.sort_values("kickoff").reset_index(drop=True)
feat = {t: np.zeros((len(pr), 2)) for t in TR}
gt = {tm: d for tm, d in g.groupby("team")}  # noqa: C416
for s in sorted(pr.season.unique()):
    prior = g[g.season < s]
    if prior.season.nunique() < MIN_PRIOR_SEASONS:
        continue
    par = {t: vc(prior, t) for t in TR}
    for pos in np.where(pr.season.values == s)[0]:
        row = pr.iloc[pos]
        for c, tm in enumerate([row.home_team, row.away_team]):
            w = gt.get(tm)
            if w is None:
                continue
            h = w[w.season.isin([s - 1, s]) & (w.kickoff < row.kickoff)]
            if len(h) == 0:
                continue
            for t in TR:
                sw2, sb2, mu = par[t]
                if sb2 <= 0:
                    continue
                k = sw2 / sb2
                feat[t][pos, c] = TR[t] * ((h[t] - mu).sum() / (len(h) + k)) / np.sqrt(sb2)
    print("as-of", s, {t: round(par[t][0] / max(par[t][1], 1e-9), 1) for t in TR})
for t in TR:
    pr["h_" + t] = feat[t][:, 0]
    pr["a_" + t] = feat[t][:, 1]
for pre in ("h_", "a_"):
    pr[pre + "comp"] = (pr[pre + "oe"] + pr[pre + "sec"] + pr[pre + "ppd"]) / 3
fav_home = pr.spread_line > 0
for t in [*list(TR), "comp"]:
    pr["x_" + t] = np.where(fav_home, pr["h_" + t], -pr["a_" + t])
p0 = pr.home_cover_probability_excluding_push
pr["off"] = np.log(p0 / (1 - p0))


def fit(x, off, y):
    def f(bb):
        return np.sum(np.logaddexp(0, off + bb * x) - y * (off + bb * x))

    return minimize_scalar(f, bounds=(-3, 3), method="bounded").x


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def loso(df, col, perm=False):
    df = df.copy()
    if perm:
        for s in df.season.unique():
            m = (df.season == s).values
            df.loc[m, col] = rng.permutation(df.loc[m, col].values)
    res = []
    bs = []
    for s in range(LOSO_FIRST, LOSO_LAST + 1):
        tr = df[df.season != s]
        te = df[df.season == s]
        if len(te) == 0:
            continue
        sd = tr[col].std()
        bb = fit(tr[col].values / sd, tr.off.values, tr.y.values)
        pm = 1 / (1 + np.exp(-(te.off.values + bb * te[col].values / sd)))
        pb0 = 1 / (1 + np.exp(-te.off.values))
        res.append(
            (
                ll(pm, te.y.values) - ll(pb0, te.y.values),
                ll(pm, te.y.values) - np.log(0.5),
                ll(pb0, te.y.values) - np.log(0.5),
            )
        )
        bs.append(bb)
    return res, bs


cells = []
valid = pr[(pr.season >= REL_START) & pr.off.notna()]
for t in [*list(TR), "comp"]:
    col = "x_" + t
    for name, sub in (("abs_spread_ge7", valid[valid.spread_line.abs() >= 7]), ("all", valid)):
        res, bs = loso(sub, col)
        d = np.concatenate([r[0] for r in res])
        n = len(d)
        bt = np.array([d[rng.integers(0, n, n)].sum() for _ in range(BOOT)])
        nulls = np.array(
            [
                np.concatenate([r[0] for r in loso(sub, col, perm=True)[0]]).sum()
                for _ in range(PERM)
            ]
        )
        c = {
            "trait": t,
            "subset": name,
            "n": int(n),
            "coefs": [round(float(x), 4) for x in bs],
            "dLL_per_game": float(d.mean()),
            "dLL_total": float(d.sum()),
            "p_pos_boot": float((bt > 0).mean()),
            "ci90_per_game": [
                float(np.percentile(bt, CI_LO) / n),
                float(np.percentile(bt, CI_HI) / n),
            ],
            "perm_p": float((nulls >= d.sum()).mean()),
            "model_vs_market_ll": float(np.concatenate([r[1] for r in res]).mean()),
            "base_vs_market_ll": float(np.concatenate([r[2] for r in res]).mean()),
        }
        cells.append(c)
        print(c)
g_all["gorder"] = g_all.game_id
ext = reliability(g_all[g_all.season <= LOSO_LAST], "gorder", EXT_START)
with open(OUT, "w") as fh:
    json.dump({"reliability": relrows, "cells": cells, "reliability_ext": ext}, fh, indent=1)
