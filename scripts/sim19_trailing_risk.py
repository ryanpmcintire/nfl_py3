import glob
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

N_BOOT = 500
N_PERM = 1000
MIN_SPELL_GAMES = 16
MIN_SEASON_GAMES = 6
N_QUANT = 4
rng = np.random.default_rng(19)

d = pd.read_parquet(
    "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=[
        "game_id",
        "season",
        "season_type",
        "week",
        "posteam",
        "play_type",
        "down",
        "ydstogo",
        "yardline_100",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
        "qb_dropback",
        "qb_kneel",
        "qb_spike",
        "sack",
        "interception",
        "pass_attempt",
        "air_yards",
    ],
)
d = d[d.season_type.eq("REG") & d.qtr.between(1, 4) & d.posteam.notna()]
d = d[~d.qb_kneel.fillna(0).astype(bool) & ~d.qb_spike.fillna(0).astype(bool)]
d = d.dropna(subset=["score_differential", "down", "ydstogo", "game_seconds_remaining"])
d = d[d.play_type.isin(["pass", "run", "punt", "field_goal"])].copy()
d["scrim"] = d.play_type.isin(["pass", "run"])
d["trail"] = d.score_differential < 0
d["deficit"] = -d.score_differential
d["db"] = d.qb_dropback.fillna(0).astype(float)
d["att"] = d.pass_attempt.fillna(0).astype(float)
d["sk"] = d.sack.fillna(0).astype(float)
d["ic"] = d.interception.fillna(0).astype(float)
yb = pd.qcut(d.ydstogo[~d.trail], N_QUANT, retbins=True, duplicates="drop")[1]
yl = pd.qcut(d.yardline_100[~d.trail], N_QUANT, retbins=True, duplicates="drop")[1]


def cell(df, with_yl):
    c = df.down.astype(int).astype(str) + "_" + np.digitize(df.ydstogo, yb[1:-1]).astype(str)
    if with_yl:
        c = c + "_" + np.digitize(df.yardline_100, yl[1:-1]).astype(str)
    return c


sc = d[d.scrim].copy()
sc["cell"] = cell(sc, False)
ref = sc[~sc.trail]
sc["ref_pass"] = sc.cell.map(ref.groupby("cell").db.mean())
att = sc[(sc.att == 1) & sc.air_yards.notna()].copy()
refd = ref[(ref.att == 1) & ref.air_yards.notna()]
att["ref_depth"] = att.cell.map(refd.groupby("cell").air_yards.mean())
dbk = sc[sc.db == 1].copy()
dbk["ref_sack"] = dbk.cell.map(ref[ref.db == 1].groupby("cell").sk.mean())
pa = sc[sc.att == 1].copy()
pa["ref_int"] = pa.cell.map(ref[ref.att == 1].groupby("cell").ic.mean())
f4 = d[d.down.eq(4)].copy()
f4["cell"] = cell(f4, True)
f4["go"] = f4.scrim.astype(float)
f4["ref_go"] = f4.cell.map(f4[~f4.trail].groupby("cell").go.mean())
SETS = {
    "depth": (att, "air_yards", "ref_depth"),
    "pass_rate": (sc, "db", "ref_pass"),
    "sack_rate": (dbk, "sk", "ref_sack"),
    "int_rate": (pa, "ic", "ref_int"),
    "go_4th": (f4, "go", "ref_go"),
}
T = {}
NT = {}
for k, (df, y, r) in SETS.items():
    t = df[df.trail & df[r].notna()].copy()
    t["raw"] = t[y]
    t["ex"] = t[y] - t[r]
    T[k] = t
    nt = df[~df.trail & df[r].notna()].copy()
    nt["ex"] = nt[y] - nt[r]
    NT[k] = nt
alltr = pd.concat([T[k][["deficit", "game_seconds_remaining"]] for k in T])
qs = np.linspace(0, 1, N_QUANT + 1)[1:-1]
de = np.unique(np.quantile(alltr.deficit, qs))
te = np.unique(np.quantile(alltr.game_seconds_remaining, qs))
for t in T.values():
    t["db_bin"] = np.digitize(t.deficit, de)
    t["tb_bin"] = np.digitize(t.game_seconds_remaining, te)
    t["cellid"] = t.db_bin * (len(te) + 1) + t.tb_bin
games = {k: pd.factorize(T[k].game_id)[0] for k in T}


def lab(i, e):
    lo = "min" if i == 0 else f"{e[i - 1]:.0f}"
    hi = "max" if i == len(e) else f"{e[i]:.0f}"
    return f"{lo}-{hi}"


NC = (len(de) + 1) * (len(te) + 1)
res = {"deficit_edges": de.tolist(), "secs_edges": te.tolist(), "cells": {}, "slopes": {}}
for k, t in T.items():
    g = games[k]
    G = g.max() + 1
    out = {}
    sy = np.zeros((G, NC))
    sx = np.zeros((G, NC))
    sr = np.zeros((G, NC))
    np.add.at(sy, (g, t.cellid.values), t.raw.values)
    np.add.at(sx, (g, t.cellid.values), 1.0)
    np.add.at(sr, (g, t.cellid.values), t.ex.values)
    cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(float)
    n = sx.sum(0)
    est_raw = sy.sum(0) / np.maximum(n, 1)
    est_ex = sr.sum(0) / np.maximum(n, 1)
    bx = (cnt @ sr) / np.maximum(cnt @ sx, 1)
    for c in range(NC):
        if n[c] == 0:
            continue
        di, ti = divmod(c, len(te) + 1)
        out[f"deficit {lab(di, de)} | secs {lab(ti, te)}"] = {
            "n": int(n[c]),
            "raw": float(est_raw[c]),
            "excess": float(est_ex[c]),
            "excess_ci95": np.percentile(bx[:, c], [2.5, 97.5]).tolist(),
            "p_pos": float((bx[:, c] > 0).mean()),
        }
    res["cells"][k] = out
    X = np.column_stack([np.ones(len(t)), t.deficit.values, t.game_seconds_remaining.values / 60.0])
    yv = t.ex.values

    def fit(idx, X=X, yv=yv):
        return np.linalg.lstsq(X[idx], yv[idx], rcond=None)[0]

    b0 = fit(np.arange(len(t)))
    order = np.argsort(g, kind="stable")
    starts = np.searchsorted(g[order], np.arange(G))
    ends = np.append(starts[1:], len(t))
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([order[starts[i] : ends[i]] for i in pick])
        bs.append(fit(idx))
    bs = np.array(bs)
    res["slopes"][k] = {
        "per_point_of_deficit": float(b0[1]),
        "ci95_deficit": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
        "p_pos_deficit": float((bs[:, 1] > 0).mean()),
        "per_minute_remaining": float(b0[2]),
        "ci95_minutes": np.percentile(bs[:, 2], [2.5, 97.5]).tolist(),
        "p_pos_minutes": float((bs[:, 2] > 0).mean()),
    }
    seas = t.season.values
    se_lin = se_bin = 0.0
    for sn in np.unique(seas):
        m = seas == sn
        bl = np.linalg.lstsq(X[~m], yv[~m], rcond=None)[0]
        cm = pd.Series(yv[~m]).groupby(t.cellid.values[~m]).mean()
        pr = pd.Series(t.cellid.values[m]).map(cm).fillna(yv[~m].mean()).values
        se_lin += ((yv[m] - X[m] @ bl) ** 2).sum()
        se_bin += ((yv[m] - pr) ** 2).sum()
    res["slopes"][k]["loso_mse_linear"] = se_lin / len(t)
    res["slopes"][k]["loso_mse_cellmean"] = se_bin / len(t)

sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[
    ["game_id", "gameday", "home_team", "home_coach", "away_coach"]
]
tg = (
    d[["game_id", "season", "week", "posteam"]]
    .drop_duplicates()
    .rename(columns={"posteam": "team"})
)
tg = tg.merge(sch, on="game_id")
tg["hc"] = np.where(tg.team == tg.home_team, tg.home_coach, tg.away_coach)
tg["t"] = (pd.to_datetime(tg.gameday, utc=True) + pd.Timedelta(days=1)).astype(
    "datetime64[ns, UTC]"
)
tg["season"] = tg.season.astype("int64")
ch = pd.read_parquet(sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))[-1])
ch["effective_observed_at"] = pd.to_datetime(ch.effective_observed_at, utc=True).astype(
    "datetime64[ns, UTC]"
)
tg = tg.sort_values("t").reset_index(drop=True)
c = ch[ch.role == "OC"].sort_values("effective_observed_at").astype({"season": "int64"})
m = pd.merge_asof(
    tg[["game_id", "team", "season", "t"]],
    c[["team", "season", "effective_observed_at", "person"]],
    left_on="t",
    right_on="effective_observed_at",
    by=["team", "season"],
    direction="backward",
)
first = c.drop_duplicates(["team", "season"])[["team", "season", "person"]].rename(
    columns={"person": "fp"}
)
m = m.merge(first, on=["team", "season"], how="left")
tg["oc"] = m.person.fillna(m.fp).values
tg = tg.sort_values(["season", "week"]).reset_index(drop=True)
for kk in ("hc", "oc"):
    tg["sp_" + kk] = tg[kk].astype("string") + "|" + tg.team
    tg.loc[tg[kk].isna(), "sp_" + kk] = pd.NA
    tg["par_" + kk] = tg.groupby("sp_" + kk).cumcount() % 2


def boot_r(a, b):
    nn = len(a)
    rs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, nn, nn)
        rs.append(np.corrcoef(a[i], b[i])[0, 1])
    rs = np.array(rs)
    return np.percentile(rs, [2.5, 97.5]).tolist(), float((rs > 0).mean())


rel = {}
for k, mode in [(k, m_) for k in T for m_ in ("style_incl", "shift")]:
    t = T[k]
    gx = t.groupby(["game_id", "posteam"]).agg(ex=("ex", "sum"), n=("ex", "size")).reset_index()
    gx = gx.rename(columns={"posteam": "team"})
    if mode == "shift":
        nt = NT[k].rename(columns={"posteam": "team"})
        ng_ = nt.groupby(["game_id", "season", "team"]).ex.agg(["sum", "size"]).reset_index()
        ng_["S"] = ng_.groupby(["season", "team"])["sum"].transform("sum")
        ng_["N"] = ng_.groupby(["season", "team"])["size"].transform("sum")
        ng_["loo"] = (ng_.S - ng_["sum"]) / (ng_.N - ng_["size"]).replace(0, np.nan)
        gx = gx.merge(ng_[["game_id", "team", "loo"]], on=["game_id", "team"], how="left")
        gx = gx.dropna(subset=["loo"])
        gx["ex"] = gx.ex - gx.n * gx.loo
    base = tg.merge(gx, on=["game_id", "team"])
    for key in ("hc", "oc"):
        sk = "sp_" + key
        gk = base[base[sk].notna()]
        sp = gk.groupby(sk).size()
        gk = gk[gk[sk].isin(sp[sp >= MIN_SPELL_GAMES].index)].copy()
        gk["code"] = pd.factorize(gk[sk])[0]
        pr = gk.groupby(["code", "par_" + key])[["ex", "n"]].sum().unstack()
        nmin = pr["n"].median().min()
        ok = ((pr["n"] >= nmin).all(axis=1)).values
        a = (pr["ex"][0] / pr["n"][0]).values[ok]
        b = (pr["ex"][1] / pr["n"][1]).values[ok]
        r = float(np.corrcoef(a, b)[0, 1])
        ci, pp = boot_r(a, b)
        ss = gk.groupby([sk, "season"])[["ex", "n"]].sum().reset_index()
        ng = gk.groupby([sk, "season"]).size().rename("ng").reset_index()
        ss = ss.merge(ng, on=[sk, "season"])
        ss = ss[(ss.ng >= MIN_SEASON_GAMES) & (ss.n >= ss.n.median())]
        ss["v"] = ss.ex / ss.n
        nx = ss.copy()
        nx["season"] = nx.season - 1
        pm = ss.merge(nx, on=[sk, "season"], suffixes=("_t", "_t1"))
        r2 = float(np.corrcoef(pm.v_t, pm.v_t1)[0, 1])
        ci2, pp2 = boot_r(pm.v_t.values, pm.v_t1.values)
        gk = gk.reset_index(drop=True)
        nr = []
        cd = gk.code.values
        par = gk["par_" + key].values.astype(np.int64)
        ex_, n_ = gk.ex.values, gk.n.values
        seasons = gk.season.values
        grp = [np.flatnonzero(seasons == s_) for s_ in np.unique(seasons)]
        keep_codes = np.flatnonzero(ok)
        size = 2 * len(ok)
        for _ in range(N_PERM):
            pi = np.arange(len(gk))
            for ix in grp:
                pi[ix] = rng.permutation(ix)
            sa = np.bincount(cd * 2 + par, ex_[pi], minlength=size).reshape(-1, 2)
            sn = np.bincount(cd * 2 + par, n_[pi], minlength=size).reshape(-1, 2)
            with np.errstate(all="ignore"):
                va = sa[keep_codes, 0] / sn[keep_codes, 0]
                vb = sa[keep_codes, 1] / sn[keep_codes, 1]
            f = np.isfinite(va) & np.isfinite(vb)
            nr.append(np.corrcoef(va[f], vb[f])[0, 1])
        nr = np.array(nr)
        rel[f"{k}|{key}|{mode}"] = {
            "n_spells": int(ok.sum()),
            "odd_even_r": r,
            "odd_even_ci95": ci,
            "odd_even_p_pos": pp,
            "spearman_brown": 2 * r / (1 + r),
            "odd_even_perm_p": float((np.sum(nr >= r) + 1) / (len(nr) + 1)),
            "t_t1_pairs": len(pm),
            "t_t1_r": r2,
            "t_t1_ci95": ci2,
            "t_t1_p_pos": pp2,
        }
res["reliability"] = rel
res["looks"] = {
    "cells": sum(len(v) for v in res["cells"].values()),
    "slopes": 2 * len(T),
    "reliability": 2 * 2 * 2 * len(T),
}
res["filters"] = (
    "REG, qtr 1-4, no kneel/spike, trailing = score_differential<0; excess = outcome minus "
    "cell mean of non-trailing plays in same down x ydstogo quartile (x yardline for 4th); "
    "bin edges are quartiles of trailing plays"
)
os.makedirs("artifacts/sim19", exist_ok=True)
with open("artifacts/sim19/trailing_risk.json", "w") as fh:
    json.dump(res, fh, indent=1)
print("done", res["looks"])
