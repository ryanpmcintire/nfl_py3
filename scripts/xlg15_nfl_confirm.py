import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier as H

ROOT = Path(__file__).resolve().parents[1]
PBP = ROOT / "data" / "pbp" / "raw" / "20261117T221807Z"
PICK = ROOT / "artifacts" / "pick_probability" / "20261120T171725Z" / "per_game.parquet"
OUT = ROOT / "artifacts" / "xlg15"
SERVED = "out_of_season_home_probability"
SEED = 1515
BOOT = 2000
PERMS = 2000
NULL_DRAWS = 200
LEAD = 8
MIN_PLAYS_HALF = 5
NEWTON_STEPS = 50
NEWTON_TOL = 1e-10
SEASONS = range(2009, 2026)
ERAS = {"2009-2019": (2009, 2019), "2020-2025": (2020, 2025)}
COLS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "qtr",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "play_type",
    "qb_kneel",
    "pass_oe",
]
GBM = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=12, min_samples_leaf=60, random_state=0)
rng = np.random.default_rng(SEED)


def load():
    parts = []
    for s in SEASONS:
        d = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=COLS)
        parts.append(d[d.season_type == "REG"])
    p = pd.concat(parts, ignore_index=True)
    p["kneel"] = p.qb_kneel.fillna(0).astype(bool)
    p["scrim"] = p.play_type.isin(["run", "pass"])
    return p


def sb(r):
    return 2 * r / (1 + r)


def traits(p):
    a = p[(p.qtr == 4) & p.scrim & ~p.kneel & (p.score_differential >= LEAD) & p.pass_oe.notna()]
    t1 = (
        a.groupby(["season", "week", "game_id", "posteam"])
        .agg(v=("pass_oe", "sum"), n=("pass_oe", "size"))
        .reset_index()
    )
    f = p[(p.down == 4) & (p.qtr <= 4) & ~p.kneel].copy()
    f["go"] = f.scrim.astype(int)
    f["kick"] = f.play_type.isin(["punt", "field_goal"]).astype(int)
    f = f[(f.go + f.kick) == 1].dropna(
        subset=["ydstogo", "yardline_100", "score_differential", "game_seconds_remaining"]
    )
    x = f[["yardline_100", "ydstogo", "score_differential", "game_seconds_remaining", "qtr"]].values
    f["pr"] = np.nan
    f["pr_prior"] = np.nan
    for s in sorted(f.season.unique()):
        m = (f.season == s).values
        f.loc[m, "pr"] = H(**GBM).fit(x[~m], f.go.values[~m]).predict_proba(x[m])[:, 1]
        pm = (f.season < s).values
        if pm.any():
            f.loc[m, "pr_prior"] = H(**GBM).fit(x[pm], f.go.values[pm]).predict_proba(x[m])[:, 1]
    f["res"] = f.go - f.pr
    f["res"] -= f.groupby("season").res.transform("mean")
    f["res_prior"] = f.go - f.pr_prior
    t2 = (
        f.groupby(["season", "week", "game_id", "posteam"])
        .agg(v=("res", "sum"), n=("res", "size"), vp=("res_prior", "sum"), np_=("res_prior", "count"))
        .reset_index()
    )
    return t1, t2


def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    c = (w * (a - ma) * (b - mb)).sum()
    v = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
    return float(c / v) if v > 0 else float("nan")


def split_rel(tg, era):
    lo, hi = era
    d = tg[(tg.season >= lo) & (tg.season <= hi)].sort_values(["season", "posteam", "week", "game_id"]).copy()
    d["h"] = d.groupby(["season", "posteam"]).cumcount() % 2
    s = d.groupby(["season", "posteam", "h"]).agg(v=("v", "sum"), n=("n", "sum")).unstack("h")
    s = s[(s["n"].fillna(0) >= MIN_PLAYS_HALF).all(axis=1)]
    a = (s["v"][0] / s["n"][0]).values
    b = (s["v"][1] / s["n"][1]).values
    return a, b, s.index.get_level_values("season").values, s.index.get_level_values("posteam").values


def rel_stats(a, b, seasons, teams):
    w = np.ones(len(a))
    r = wcorr(a, b, w)
    ut = np.unique(teams)
    idx = {t: np.flatnonzero(teams == t) for t in ut}
    bs = []
    for _ in range(BOOT):
        ix = np.concatenate([idx[t] for t in rng.choice(ut, len(ut))])
        bs.append(wcorr(a[ix], b[ix], w[ix]))
    bs = np.array(bs)
    nl = []
    for _ in range(PERMS):
        e = b.copy()
        for gv in np.unique(seasons):
            m = seasons == gv
            e[m] = rng.permutation(e[m])
        nl.append(wcorr(a, e, w))
    nl = np.array(nl)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {
        "n_units": int(len(a)),
        "n_teams": int(len(ut)),
        "r_half": r,
        "r_ci": [float(lo), float(hi)],
        "sb": sb(r),
        "sb_ci": [sb(float(lo)), sb(float(hi))],
        "p_pos": float(np.nanmean(bs > 0)),
        "perm_p": float((nl >= r).mean()),
        "perm_95": float(np.percentile(nl, 95)),
    }


def asof(tg, num, den):
    d = tg.sort_values(["posteam", "season", "week", "game_id"]).copy()
    d["cn"] = d.groupby("posteam")[num].cumsum() - d[num]
    d["cd"] = d.groupby("posteam")[den].cumsum() - d[den]
    d["tr"] = d.cn / d.cd.where(d.cd > 0)
    return d[["game_id", "posteam", "tr"]]


def fit(x, off, y):
    b = np.zeros(x.shape[1])
    for _ in range(NEWTON_STEPS):
        p = expit(off + x @ b)
        g = x.T @ (y - p)
        h = (x * (p * (1 - p))[:, None]).T @ x
        st = np.linalg.lstsq(h, g, rcond=None)[0]
        b = b + st
        if np.abs(st).max() < NEWTON_TOL:
            break
    return b


def ll(e, y):
    return np.logaddexp(0, e) - y * e


def loso(z, off, y, season):
    out = np.zeros((len(y), 2))
    coef = {}
    for s in np.unique(season):
        tr, te = season != s, season == s
        mu, sd = z[tr].mean(), z[tr].std()
        zt = (z - mu) / sd
        x1 = np.column_stack([np.ones(len(y)), zt])
        b1 = fit(x1[tr], off[tr], y[tr])
        b0 = fit(x1[tr, :1], off[tr], y[tr])
        out[te, 0] = off[te] + b0[0]
        out[te, 1] = off[te] + x1[te] @ b1
        coef[int(s)] = round(float(b1[1]), 4)
    return out, coef


def block_boot(v, season):
    us = np.unique(season)
    sums = np.array([v[season == s].sum() for s in us])
    cnts = np.array([(season == s).sum() for s in us])
    bs = []
    for _ in range(BOOT):
        ix = rng.integers(0, len(us), len(us))
        bs.append(sums[ix].sum() / cnts[ix].sum())
    bs = np.array(bs)
    return {
        "mean": float(v.mean()),
        "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "p_pos": float((bs > 0).mean()),
    }


def opener(m, col):
    d = m.dropna(subset=[col]).reset_index(drop=True)
    y = d.home_covered.astype(float).values
    p0 = d[SERVED].clip(1e-4, 1 - 1e-4).values
    off = np.log(p0 / (1 - p0))
    z = d[col].values.astype(float)
    season = d.season.values
    e, coef = loso(z, off, y, season)
    gain = ll(e[:, 0], y) - ll(e[:, 1], y)
    null = []
    for _ in range(NULL_DRAWS):
        zs = z.copy()
        for s in np.unique(season):
            k = season == s
            zs[k] = rng.permutation(zs[k])
        en, _ = loso(zs, off, y, season)
        null.append((ll(en[:, 0], y) - ll(en[:, 1], y)).mean())
    null = np.array(null)
    sb_, sa = off > 0, e[:, 1] > 0
    fl = sb_ != sa
    return {
        "n_games": int(len(y)),
        "ll_gain_vs_icpt": block_boot(gain, season),
        "ll_gain_vs_base": block_boot(ll(off, y) - ll(e[:, 1], y), season),
        "null_percentile": float((null < gain.mean()).mean() * 100),
        "acc_base": float((sb_ == (y == 1)).mean()),
        "acc_arm": float((sa == (y == 1)).mean()),
        "decisive_flips": int(fl.sum()),
        "flip_arm_correct": int((sa[fl] == (y[fl] == 1)).sum()),
        "coef_by_fold": coef,
    }


def main():
    p = load()
    t1, t2 = traits(p)
    res = {"family": "xlg15_cfb_trait_reliability", "nfl_reliability": {}, "nfl_opener": {}}
    for name, tg in (("pass_oe_when_ahead", t1), ("fourth_down_go_residual", t2)):
        res["nfl_reliability"][name] = {}
        for en, era in ERAS.items():
            a, b, ss, tt = split_rel(tg, era)
            res["nfl_reliability"][name][en] = rel_stats(a, b, ss, tt)
            print("REL", name, en, res["nfl_reliability"][name][en], flush=True)
    sched = p.groupby("game_id").agg(season=("season", "first")).reset_index()
    pg = pd.read_parquet(PICK, columns=["game_id", "season", "home_team", "away_team", "margin_vs_open", "home_covered", SERVED])
    pg = pg[pg.margin_vs_open.ne(0) & pg[SERVED].notna()].copy()
    for name, tg, num, den in (
        ("pass_oe_when_ahead", t1, "v", "n"),
        ("fourth_down_go_residual", t2, "vp", "np_"),
    ):
        full = (
            p[p.posteam.notna()][["season", "week", "game_id", "posteam"]]
            .drop_duplicates()
            .merge(tg[["game_id", "posteam", num, den]], on=["game_id", "posteam"], how="left")
            .fillna({num: 0, den: 0})
        )
        ao = asof(full, num, den)
        grid = ao.merge(sched, on="game_id")
        h = grid.rename(columns={"posteam": "home_team", "tr": "tr_h"})[["game_id", "home_team", "tr_h"]]
        w = grid.rename(columns={"posteam": "away_team", "tr": "tr_a"})[["game_id", "away_team", "tr_a"]]
        m = pg.merge(h, on=["game_id", "home_team"], how="left").merge(w, on=["game_id", "away_team"], how="left")
        m[name] = m.tr_h - m.tr_a
        res["nfl_opener"][name] = opener(m, name)
        print("OPENER", name, res["nfl_opener"][name], flush=True)
    (OUT / "nfl_confirm.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
