import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from eng53_units import emit, piece
from sklearn.ensemble import HistGradientBoostingClassifier as H

ROOT = Path(__file__).resolve().parents[1]
SEED = 15
BOOT = 2000
PERMS = 2000
LEAD = 8
MIN_PLAYS_HALF = 5
SEASONS = range(2006, 2026)
ERAS = {"2014-2019": (2014, 2019), "2023-2025": (2023, 2025)}
COLS = [
    "game_id",
    "season",
    "week",
    "seasonType",
    "pos_team_id",
    "def_pos_team_id",
    "homeTeamId",
    "period",
    "down",
    "distance",
    "text",
    "rush",
    "pass",
    "type.text",
    "start.yardsToEndzone",
    "clock.minutes",
    "clock.seconds",
    "pos_score_diff_start",
    "kneel_down",
    "pass_oe",
]
KICK_PREFIX = ("Punt", "Blocked Punt", "Field Goal", "Blocked Field Goal", "Missed Field Goal")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
OUT = Path(ap.parse_args().out)
rng = np.random.default_rng(SEED)


def load():
    parts = []
    for s in SEASONS:
        fs = sorted(glob.glob(str(ROOT / f"data/cfb/pbp/raw/*/season={s}/plays.parquet")))
        d = pd.read_parquet(fs[-1], columns=COLS)
        parts.append(d[d.seasonType == 2])
    return pd.concat(parts, ignore_index=True)


def sb(r):
    return 2 * r / (1 + r)


def team_games(p):
    a = p[p.period == 4].copy()
    a = a[(a.rush | a["pass"]) & ~a.kneel_down.fillna(False).astype(bool)]
    a = a[(a.pos_score_diff_start >= LEAD) & a.pass_oe.notna()]
    t1 = (
        a.groupby(["season", "week", "game_id", "pos_team_id"])
        .agg(v=("pass_oe", "sum"), n=("pass_oe", "size"))
        .reset_index()
    )
    f = p[(p.down == 4) & (p.period <= 4) & ~p.kneel_down.fillna(False).astype(bool)].copy()
    f["go"] = (f.rush | f["pass"]).astype(int)
    f["kick"] = f["type.text"].fillna("").str.startswith(KICK_PREFIX).astype(int)
    f = f[(f.go + f.kick) == 1].copy()
    f["secs"] = (4 - f.period) * 900 + f["clock.minutes"] * 60 + f["clock.seconds"]
    f = f.dropna(subset=["distance", "start.yardsToEndzone", "pos_score_diff_start", "secs"])
    x = f[["start.yardsToEndzone", "distance", "pos_score_diff_start", "secs", "period"]].values
    f["pr"] = np.nan
    for s in sorted(f.season.unique()):
        m = (f.season == s).values
        clf = H(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=12,
            min_samples_leaf=60,
            random_state=0,
        )
        clf.fit(x[~m], f.go.values[~m])
        f.loc[m, "pr"] = clf.predict_proba(x[m])[:, 1]
    f["res"] = f.go - f.pr
    f["res"] -= f.groupby("season").res.transform("mean")
    t2 = (
        f.groupby(["season", "week", "game_id", "pos_team_id"])
        .agg(v=("res", "sum"), n=("res", "size"))
        .reset_index()
    )
    return t1, t2


def venue_games(p):
    sn = p[p.rush | p["pass"]]
    away = sn.pos_team_id != sn.homeTeamId
    snaps = sn[away].groupby("game_id").size().rename("snaps")
    fsp = p[
        p.text.fillna("").str.contains("False Start", case=False)
        & (p["type.text"] == "Penalty")
        & (p.pos_team_id != p.homeTeamId)
    ]
    fs = fsp.drop_duplicates(["game_id", "text"]).groupby("game_id").size().rename("fs")
    at = sn[away].groupby("game_id").pos_team_id.first().rename("away_id")
    g = (
        p.groupby("game_id")
        .agg(season=("season", "first"), week=("week", "first"), home_id=("homeTeamId", "first"))
        .join([snaps, fs, at])
        .fillna({"fs": 0})
        .dropna(subset=["snaps", "away_id"])
        .reset_index()
    )
    return g


def split_rel(tg, era, name=None, sink=None):
    lo, hi = era
    d = (
        tg[(tg.season >= lo) & (tg.season <= hi)]
        .sort_values(["season", "pos_team_id", "week", "game_id"])
        .copy()
    )
    d["h"] = d.groupby(["season", "pos_team_id"]).cumcount() % 2
    s = d.groupby(["season", "pos_team_id", "h"]).agg(v=("v", "sum"), n=("n", "sum")).unstack("h")
    ok = (s["n"].fillna(0) >= MIN_PLAYS_HALF).all(axis=1)
    s = s[ok]
    a = (s["v"][0] / s["n"][0]).values
    b = (s["v"][1] / s["n"][1]).values
    ss = s.index.get_level_values("season").values
    if sink is not None:
        k = d[pd.MultiIndex.from_frame(d[["season", "pos_team_id"]]).isin(s.index)]
        sink.append(
            piece(
                name,
                k.season.astype(str) + "|" + k.pos_team_id.astype(str),
                k.pos_team_id,
                k.season,
                k.game_id,
                k.h,
                k.v / k.n,
                k.n,
            )
        )
    return a, b, ss


def venue_rel(g, era, name=None, sink=None):
    lo, hi = era
    d = g[(g.season >= lo) & (g.season <= hi)].copy()
    d["h"] = d.season % 2
    sm = d.groupby(["home_id", "h"]).excess.agg(["mean", "size"]).unstack("h").dropna()
    w = 2 * sm[("size", 0)] * sm[("size", 1)] / (sm[("size", 0)] + sm[("size", 1)])
    if sink is not None:
        k = d[d.home_id.isin(sm.index)]
        sink.append(
            piece(name, k.home_id, k.home_id, k.season, k.game_id, k.h, k.excess, np.ones(len(k)))
        )
    return sm[("mean", 0)].values, sm[("mean", 1)].values, w.values, d


def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    c = (w * (a - ma) * (b - mb)).sum()
    v = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
    return float(c / v) if v > 0 else float("nan")


def rel_stats(a, b, groups, w=None):
    w = np.ones(len(a)) if w is None else w
    r = wcorr(a, b, w)
    n = len(a)
    bs = []
    for _ in range(BOOT):
        ix = rng.integers(0, n, n)
        bs.append(wcorr(a[ix], b[ix], w[ix]))
    bs = np.array(bs)
    nl = []
    for _ in range(PERMS):
        e = b.copy()
        for gv in np.unique(groups):
            m = groups == gv
            e[m] = rng.permutation(e[m])
        nl.append(wcorr(a, e, w))
    nl = np.array(nl)
    return {
        "n_units": int(n),
        "r_half": r,
        "sb": sb(r),
        "sb_ci": [sb(float(np.nanpercentile(bs, 2.5))), sb(float(np.nanpercentile(bs, 97.5)))],
        "p_pos_boot": float(np.nanmean(bs > 0)),
        "perm_p": float((nl >= r).mean()),
    }


def asof(tg, grid, num, den):
    d = grid.merge(
        tg[["game_id", "pos_team_id", num, den]], on=["game_id", "pos_team_id"], how="left"
    ).fillna({num: 0, den: 0})
    d = d.sort_values(["pos_team_id", "season", "week", "game_id"])
    d["cn"] = d.groupby("pos_team_id")[num].cumsum() - d[num]
    d["cd"] = d.groupby("pos_team_id")[den].cumsum() - d[den]
    d["tr"] = d.cn / d.cd.where(d.cd > 0)
    return d[["game_id", "pos_team_id", "tr"]]


def price(frame, y, x, era):
    lo, hi = era
    d = (
        frame[(frame.season >= lo) & (frame.season <= hi)]
        .dropna(subset=[y, x])
        .reset_index(drop=True)
    )
    sdx = d[x].std()
    z = ((d[x] - d[x].mean()) / sdx).values
    yy = d[y].values
    n = len(d)

    def slope(ix):
        zz, y2 = z[ix], yy[ix]
        return np.cov(zz, y2, ddof=1)[0, 1] / zz.var(ddof=1)

    est = slope(np.arange(n))
    bs = np.array([slope(rng.integers(0, n, n)) for _ in range(BOOT)])
    return {
        "n_games": int(n),
        "slope_ats_points_per_sd": float(est),
        "se": float(bs.std()),
        "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "p_pos": float((bs > 0).mean()),
    }


p = load()
t1, t2 = team_games(p)
vg = venue_games(p)
g = pd.read_parquet(ROOT / "data/processed/cfb_game_features.parquet")
g = g[["game_id", "season", "home_id", "away_id", "result", "spread_open", "neutral_site"]]
g = g.dropna(subset=["spread_open", "result"]).copy()
g["ats_open"] = g.result - g.spread_open

gw = pd.read_parquet(ROOT / "data/processed/cfb_game_features.parquet")[
    ["game_id", "season", "week", "home_id", "away_id"]
]
grid = pd.concat(
    [
        gw.rename(columns={"home_id": "pos_team_id"})[["game_id", "season", "week", "pos_team_id"]],
        gw.rename(columns={"away_id": "pos_team_id"})[["game_id", "season", "week", "pos_team_id"]],
    ]
)
res = {"family": "xlg15_cfb_trait_reliability", "reliability": {}, "pricing": {}}
unit_pieces = []
for name, tg in (("pass_oe_when_ahead", t1), ("fourth_down_go_residual", t2)):
    res["reliability"][name] = {}
    for en, era in ERAS.items():
        a, b, ss = split_rel(tg, era, f"{name}|{en}", unit_pieces)
        res["reliability"][name][en] = rel_stats(a, b, ss)
        print("REL", name, en, res["reliability"][name][en], flush=True)
    ao = asof(tg, grid, "v", "n")
    h = ao.rename(columns={"pos_team_id": "home_id", "tr": "tr_h"})
    aw = ao.rename(columns={"pos_team_id": "away_id", "tr": "tr_a"})
    fr = g.merge(h, on=["game_id", "home_id"]).merge(aw, on=["game_id", "away_id"])
    fr["diff"] = fr.tr_h - fr.tr_a
    res["pricing"][name] = {}
    for en, era in ERAS.items():
        res["pricing"][name][en] = price(fr, "ats_open", "diff", era)
        print("PRICE", name, en, res["pricing"][name][en], flush=True)

vg = vg.sort_values(["away_id", "season", "week", "game_id"]).copy()
for k, c in (("fs", "fs"), ("snaps", "snaps")):
    vg["c" + k] = vg.groupby("away_id")[c].cumsum() - vg[c]
vg["lg_f"] = vg.groupby("season").fs.transform("sum")
vg["lg_s"] = vg.groupby("season").snaps.transform("sum")
vg["team_f"] = vg.groupby(["season", "away_id"]).fs.transform("sum") - vg.fs
vg["team_s"] = vg.groupby(["season", "away_id"]).snaps.transform("sum") - vg.snaps
vg["rate_loo"] = vg.team_f / vg.team_s.where(vg.team_s > 0)
vg["excess"] = vg.fs - vg.snaps * vg.rate_loo
vg = vg.dropna(subset=["excess"])
vg["rate_asof"] = vg.cfs / vg.csnaps.where(vg.csnaps > 0)
vg["excess_asof"] = vg.fs - vg.snaps * vg.rate_asof
vn = vg.merge(g[["game_id", "neutral_site"]], on="game_id", how="left")
vn = vn[vn.neutral_site.fillna(0) == 0]
res["reliability"]["venue_false_starts"] = {}
for en, era in ERAS.items():
    a, b, w, _ = venue_rel(vn, era, f"venue_false_starts|{en}", unit_pieces)
    res["reliability"]["venue_false_starts"][en] = rel_stats(a, b, np.zeros(len(a)), w)
    print("REL venue", en, res["reliability"]["venue_false_starts"][en], flush=True)
vv = vn.dropna(subset=["excess_asof"]).sort_values(["home_id", "season", "week", "game_id"]).copy()
vv["cx"] = vv.groupby("home_id").excess_asof.cumsum() - vv.excess_asof
vv["ck"] = vv.groupby("home_id").excess_asof.cumcount()
vv["venue_tr"] = vv.cx / vv.ck.where(vv.ck > 0)
fr = g.merge(vv[["game_id", "venue_tr"]], on="game_id")
res["pricing"]["venue_false_starts"] = {}
for en, era in ERAS.items():
    res["pricing"]["venue_false_starts"][en] = price(fr, "ats_open", "venue_tr", era)
    print("PRICE venue", en, res["pricing"]["venue_false_starts"][en], flush=True)
emit("xlg15", unit_pieces)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
