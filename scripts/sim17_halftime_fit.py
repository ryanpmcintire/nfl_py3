import glob
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path

import numpy as np
import pandas as pd

WP_LO, WP_HI = 0.05, 0.95
MIN_CELL_GAMES = 6
N_BOOT = 1000
N_PERM = 2000
FIRST_PRED = 2010
PERIODS = {"h2": (3, 4), "q3": (3, 3)}
UNITS = (
    "EPA per play, second half minus first half; off = offense OC spell, "
    "def = defense DC spell allowed; estimated from seasons before the game season only, "
    "empirical-Bayes shrunk; absent = 0"
)
OUT = Path("artifacts/sim17")
rng = np.random.default_rng(1720)

d = pd.read_parquet(
    "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=[
        "game_id",
        "season",
        "season_type",
        "week",
        "posteam",
        "defteam",
        "play_type",
        "epa",
        "wp",
        "qtr",
        "qb_kneel",
        "qb_spike",
        "score_differential",
    ],
)
d = d[d.season_type.eq("REG") & d.play_type.isin(["pass", "run"]) & d.qtr.between(1, 4)]
d = d[~d.qb_kneel.fillna(0).astype(bool) & ~d.qb_spike.fillna(0).astype(bool)]
d = d.dropna(subset=["epa", "wp", "posteam", "defteam"])
d = d[d.wp.between(WP_LO, WP_HI)].copy()

margin = (
    d[d.qtr == 2]
    .groupby(["game_id", "posteam"])
    .score_differential.last()
    .rename("margin")
    .reset_index()
    .rename(columns={"posteam": "team"})
)

keys = ["game_id", "season", "week"]


def period_frame(period, lo, hi):
    h1 = d[d.qtr <= 2]
    h2 = d[d.qtr.between(lo, hi)]
    out = {}
    for side, (own, opp) in {"off": ("posteam", "defteam"), "def": ("defteam", "posteam")}.items():
        a = h1.groupby([*keys, own, opp]).epa.agg(["mean", "size"])
        b = h2.groupby([*keys, own, opp]).epa.agg(["mean", "size"])
        m = a.join(b, lsuffix="1", rsuffix="2", how="inner").reset_index()
        m = m.rename(columns={own: "team", opp: "opp"})
        m[f"{side}_delta"] = m.mean2 - m.mean1
        m[f"{side}_w"] = 1.0 / (1.0 / m.size1 + 1.0 / m.size2)
        m[f"{side}_n2"] = m.size2
        out[side] = m[[*keys, "team", "opp", f"{side}_delta", f"{side}_w", f"{side}_n2"]]
    g = out["off"].merge(out["def"], on=[*keys, "team", "opp"])
    return g


sched = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[
    ["game_id", "gameday", "home_team", "home_coach", "away_coach"]
]
ch = pd.read_parquet(
    [
        p
        for p in sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))
        if "coordinator_history" in p
    ][-1]
)
ch["effective_observed_at"] = pd.to_datetime(ch.effective_observed_at, utc=True).astype(
    "datetime64[ns, UTC]"
)


def attach_keys(g):
    g = g.merge(margin, on=["game_id", "team"], how="left")
    g["margin"] = g.margin.fillna(0.0)
    g = g.merge(sched, on="game_id")
    g["hc"] = np.where(g.team == g.home_team, g.home_coach, g.away_coach)
    g["t"] = (pd.to_datetime(g.gameday, utc=True) + pd.Timedelta(days=1)).astype(
        "datetime64[ns, UTC]"
    )
    g["season"] = g.season.astype("int64")
    g = g.sort_values("t").reset_index(drop=True)
    for role, name in [("OC", "oc"), ("DC", "dc")]:
        c = ch[ch.role == role].sort_values("effective_observed_at").astype({"season": "int64"})
        m = pd.merge_asof(
            g[["game_id", "team", "season", "t"]],
            c[["team", "season", "effective_observed_at", "person"]],
            left_on="t",
            right_on="effective_observed_at",
            by=["team", "season"],
            direction="backward",
        )
        first = c.drop_duplicates(["team", "season"])[["team", "season", "person"]].rename(
            columns={"person": "first_person"}
        )
        m = m.merge(first, on=["team", "season"], how="left")
        g[name] = m.person.fillna(m.first_person).values
    g = g.dropna(subset=["hc"]).sort_values(["season", "week"]).reset_index(drop=True)
    g["margin_abs"] = g.margin.abs()
    for k in ["hc", "oc", "dc"]:
        g["sp_" + k] = g[k].astype("string") + "|" + g.team
        g.loc[g[k].isna(), "sp_" + k] = pd.NA
    for col in ["off_delta", "def_delta"]:
        tot = g.groupby(["season", "team"])[col].transform("sum")
        n = g.groupby(["season", "team"])[col].transform("size")
        g[col + "_loo"] = ((tot - g[col]) / (n - 1).replace(0, np.nan)).fillna(0.0)
    loo = g.set_index(["game_id", "team"])
    kk = list(zip(g.game_id, g.opp, strict=True))
    g["opp_off_loo"] = loo.off_delta_loo.reindex(kk).fillna(0.0).values
    g["opp_def_loo"] = loo.def_delta_loo.reindex(kk).fillna(0.0).values
    return g


def loso_resid(frame, y, w, xcols):
    out = np.zeros(len(frame))
    X = np.column_stack([np.ones(len(frame))] + [frame[c].values for c in xcols])
    yv, wv = frame[y].values, frame[w].values
    for sn in sorted(frame.season.unique()):
        m = (frame.season == sn).values
        sw = np.sqrt(wv[~m])
        b = np.linalg.lstsq(X[~m] * sw[:, None], yv[~m] * sw, rcond=None)[0]
        out[m] = yv[m] - X[m] @ b
    return out


def cells(frame, key):
    sp = frame["sp_" + key]
    ok = sp.notna().to_numpy()
    cid = pd.factorize(sp.astype("string") + "#" + frame.season.astype(str))[0]
    cid = np.where(ok, cid, -1)
    return cid, ok


def pair_index(frame, key):
    cid, _ok = cells(frame, key)
    first = {}
    for i, c in enumerate(cid):
        if c >= 0 and c not in first:
            first[c] = i
    sp = frame["sp_" + key].to_numpy()
    se = frame.season.to_numpy()
    lookup = {(sp[i], se[i]): c for c, i in first.items()}
    a, b = [], []
    for (s_, y_), c in lookup.items():
        c2 = lookup.get((s_, y_ + 1))
        if c2 is not None:
            a.append(c)
            b.append(c2)
    return cid, np.array(a), np.array(b), len(first)


def cell_means(cid, ncell, res, rows):
    sel = cid[rows] >= 0
    num = np.bincount(cid[rows][sel], weights=res[rows][sel], minlength=ncell)
    den = np.bincount(cid[rows][sel], minlength=ncell)
    return num, den


def pair_r(cid, ncell, a, b, res, rows, base_den):
    num, den = cell_means(cid, ncell, res, rows)
    use = (
        (base_den[a] >= MIN_CELL_GAMES)
        & (base_den[b] >= MIN_CELL_GAMES)
        & (den[a] > 0)
        & (den[b] > 0)
    )
    if use.sum() < 10:
        return np.nan, int(use.sum())
    x = num[a][use] / den[a][use]
    y = num[b][use] / den[b][use]
    return float(np.corrcoef(x, y)[0, 1]), int(use.sum())


def reliability(frame, key, res):
    cid, a, b, ncell = pair_index(frame, key)
    allrows = np.arange(len(frame))
    base_den = np.bincount(cid[cid >= 0], minlength=ncell)
    r0, npairs = pair_r(cid, ncell, a, b, res, allrows, base_den)
    gid = frame.game_id.to_numpy()
    ug, inv = np.unique(gid, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ug))]
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(ug), len(ug))
        rows = np.concatenate([members[k] for k in pick])
        r, _ = pair_r(cid, ncell, a, b, res, rows, base_den)
        boots.append(r)
    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    null = []
    gsign = rng.choice([-1.0, 1.0], size=(N_PERM, len(ug)))
    for k in range(N_PERM):
        r, _ = pair_r(cid, ncell, a, b, res * gsign[k][inv], allrows, base_den)
        null.append(r)
    null = np.array(null)
    return {
        "r": r0,
        "pairs": npairs,
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "probability_positive": float((boots > 0).mean()),
        "null_mean": float(np.nanmean(null)),
        "null_p95": float(np.nanpercentile(null, 95)),
        "perm_p": float((np.nansum(null >= r0) + 1) / (len(null) + 1)),
    }


def shrunk_table(train, key, col):
    gk = train[train["sp_" + key].notna()]
    grp = gk.groupby("sp_" + key)[col]
    m, n = grp.mean(), grp.size()
    dev = gk[col] - gk["sp_" + key].map(m)
    sig2 = float((dev**2).sum() / max(len(gk) - len(m), 1))
    grand = float(gk[col].mean())
    tau2 = max(float(m.var(ddof=1)) - float((sig2 / n).mean()), 0.0)
    b = tau2 / (tau2 + sig2 / n)
    return grand + b * (m - grand), float(tau2), sig2, len(m)


def heldout(frame, key, col, wcol):
    pred = pd.Series(np.nan, index=frame.index)
    tabs = {}
    for s in range(FIRST_PRED, int(frame.season.max()) + 1):
        tab, tau2, sig2, nsp = shrunk_table(frame[frame.season < s], key, col)
        te = frame.season == s
        pred[te] = frame.loc[te, "sp_" + key].map(tab)
        tabs[s] = {"tau2": tau2, "sig2": sig2, "spells": nsp, "table": tab}
    ok = pred.notna().to_numpy()
    x, y = pred.to_numpy()[ok], frame[col].to_numpy()[ok]
    w = frame[wcol].to_numpy()[ok]
    gid = frame.game_id.to_numpy()[ok]
    ug, inv = np.unique(gid, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ug))]

    def slope(ix):
        xx, yy, ww = x[ix], y[ix], w[ix]
        xc = xx - np.average(xx, weights=ww)
        return float((ww * xc * (yy - np.average(yy, weights=ww))).sum() / (ww * xc**2).sum())

    s0 = slope(np.arange(len(x)))
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(ug), len(ug))
        bs.append(slope(np.concatenate([members[k] for k in pick])))
    bs = np.array(bs)
    return (
        pred,
        tabs,
        {
            "games": int(ok.sum()),
            "slope": s0,
            "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "probability_positive": float((bs > 0).mean()),
            "pred_sd_epa_per_play": float(
                np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))
            ),
        },
    )


result = {"filters": "pass/run, no kneel/spike, qtr1-4, wp in [0.05,0.95], REG 2009-2025"}
tables = {}
for period, (lo, hi) in PERIODS.items():
    g = attach_keys(period_frame(period, lo, hi))
    g["off_res"] = loso_resid(g, "off_delta", "off_w", ["margin", "margin_abs", "opp_def_loo"])
    g["def_res"] = loso_resid(g, "def_delta", "def_w", ["margin", "margin_abs", "opp_off_loo"])
    plays_off = float(g.groupby("season").off_n2.mean().mean())
    res_p = {"plays_per_team_game_in_period": plays_off, "games": len(g)}
    for col, wcol, ks in [("off_res", "off_w", ["hc", "oc"]), ("def_res", "def_w", ["hc", "dc"])]:
        for k in ks:
            rel = reliability(g, k, g[col].to_numpy())
            pred, tabs, ho = heldout(g, k, col, wcol)
            ho["points_per_game_sd"] = ho["pred_sd_epa_per_play"] * plays_off
            res_p[f"{col}_{k}"] = {"reliability_season_t_t1": rel, "heldout": ho}
            tables[(period, col, k)] = g.assign(pred=pred)[["game_id", "team", "pred"]]
    result[period] = res_p
    print(period, json.dumps(res_p, indent=1), flush=True)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "halftime_fit.json").write_text(json.dumps(result, indent=1))

by_game = {}
for (period, col, k), t in tables.items():
    if period != "h2" or k not in ("oc", "dc"):
        continue
    side = "off" if col == "off_res" else "def"
    for gid, team, pr in t.itertuples(index=False):
        if pr == pr:
            by_game.setdefault(gid, {}).setdefault(team, {})[side] = float(pr)
(OUT / "halftime_shift_h2.json").write_text(
    json.dumps(
        {
            "units": UNITS,
            "response": {
                "off": result["h2"]["off_res_oc"]["heldout"],
                "def": result["h2"]["def_res_dc"]["heldout"],
            },
            "by_game_team": by_game,
        }
    )
)
