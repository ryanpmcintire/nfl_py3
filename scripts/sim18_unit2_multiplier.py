import glob
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

MIN_SPELL_GAMES = 16
MIN_SEASON_GAMES = 6
N_PERM = 500
N_BOOT = 2000
TIME_EDGES = [0, 300, 600, 900, 1200, 1500, 1800]
LEAD_EDGES = [1, 9, 17, 10**6]
LEAD_NAMES = ["lead_1_8", "lead_9_16", "lead_17_plus"]
rng = np.random.default_rng(1802)
OUT = "artifacts/sim18"


def tbin(s):
    return np.clip(np.searchsorted(TIME_EDGES, s, side="right") - 1, 0, len(TIME_EDGES) - 2)


def lbin(a):
    return np.where(a >= 1, np.searchsorted(LEAD_EDGES, a, side="right") - 1, -1)


d = pd.read_parquet(
    "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=[
        "game_id",
        "season",
        "season_type",
        "week",
        "home_team",
        "posteam",
        "defteam",
        "fixed_drive",
        "play_type",
        "pass_attempt",
        "qb_kneel",
        "qb_spike",
        "epa",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
        "play_id",
    ],
)
d = d[d.season_type.eq("REG") & d.posteam.notna()].copy()
d = d.sort_values(["game_id", "play_id"]).reset_index(drop=True)

sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])
sch = sch[sch.game_type.eq("REG")][
    [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_coach",
        "away_coach",
        "gameday",
        "result",
    ]
].copy()
sch["t"] = (pd.to_datetime(sch.gameday, utc=True) + pd.Timedelta(days=1)).astype(
    "datetime64[ns, UTC]"
)
tgl = pd.concat(
    [
        sch.assign(team=sch.home_team, opp=sch.away_team, coach=sch.home_coach, home=1),
        sch.assign(team=sch.away_team, opp=sch.home_team, coach=sch.away_coach, home=0),
    ]
)[["game_id", "season", "week", "team", "opp", "coach", "home", "t"]]
tgl["season"] = tgl.season.astype("int64")
tgl = tgl.sort_values("t").reset_index(drop=True)
cdir = sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))[-1]
ch = pd.read_parquet(cdir)
ch["effective_observed_at"] = pd.to_datetime(ch.effective_observed_at, utc=True).astype(
    "datetime64[ns, UTC]"
)
oc = ch[ch.role == "OC"].sort_values("effective_observed_at").astype({"season": "int64"})
m = pd.merge_asof(
    tgl[["game_id", "team", "season", "t"]],
    oc[["team", "season", "effective_observed_at", "person"]],
    left_on="t",
    right_on="effective_observed_at",
    by=["team", "season"],
    direction="backward",
)
first = oc.drop_duplicates(["team", "season"])[["team", "season", "person"]].rename(
    columns={"person": "first_person"}
)
m = m.merge(first, on=["team", "season"], how="left")
tgl["oc"] = m.person.fillna(m.first_person).values
tgl = tgl.sort_values(["season", "t", "game_id"]).reset_index(drop=True)
for k, c in [("hc", "coach"), ("oc", "oc")]:
    tgl["sp_" + k] = tgl[c].astype("string") + "|" + tgl.team
    tgl.loc[tgl[c].isna(), "sp_" + k] = pd.NA

e = d[d.play_type.isin(["run", "pass"]) & d.epa.notna()]
og = e.groupby(["game_id", "posteam"]).epa.agg(["sum", "size"]).reset_index()
og.columns = ["game_id", "team", "off_sum", "off_n"]
dg = e.groupby(["game_id", "defteam"]).epa.agg(["sum", "size"]).reset_index()
dg.columns = ["game_id", "team", "def_sum", "def_n"]
q = tgl[["game_id", "season", "team", "opp", "t"]].merge(og, on=["game_id", "team"], how="left")
q = q.merge(dg, on=["game_id", "team"], how="left").fillna(
    {"off_sum": 0, "off_n": 0, "def_sum": 0, "def_n": 0}
)
q = q.sort_values(["team", "t"]).reset_index(drop=True)


def prior_rating(frame, sum_col, n_col):
    out = np.full(len(frame), np.nan)
    for _, g in frame.groupby("team"):
        idx = g.index.values
        ss = g.season.values
        sv = g[sum_col].values
        nv = g[n_col].values
        for i in range(len(g)):
            m_ = (ss == ss[i]) | (ss == ss[i] - 1)
            m_ = m_ & (np.arange(len(g)) < i)
            den = nv[m_].sum()
            out[idx[i]] = sv[m_].sum() / den if den > 0 else np.nan
    return out


q["off_r"] = prior_rating(q, "off_sum", "off_n")
q["def_r"] = prior_rating(q, "def_sum", "def_n")
rating = q[["game_id", "team", "off_r", "def_r"]]
tgl = tgl.merge(rating, on=["game_id", "team"], how="left")
orat = rating.rename(columns={"team": "opp", "off_r": "opp_off_r", "def_r": "opp_def_r"})
tgl = tgl.merge(orat, on=["game_id", "opp"], how="left")
for c in ["off_r", "def_r", "opp_off_r", "opp_def_r"]:
    tgl[c] = tgl[c].fillna(0.0)

dd = d[d.qtr.isin([3, 4])].dropna(
    subset=["game_seconds_remaining", "score_differential", "fixed_drive"]
)
dv = (
    dd.groupby(["game_id", "fixed_drive"], sort=False)
    .agg(
        season=("season", "first"),
        posteam=("posteam", "first"),
        start=("game_seconds_remaining", "first"),
        lead=("score_differential", "first"),
    )
    .reset_index()
)
dv["nxt"] = dv.groupby("game_id").start.shift(-1).fillna(0.0)
dv["el"] = np.minimum(dv.start - dv.nxt, dv.start)
dv["tb"] = tbin(dv.start.values)
dv["lb"] = lbin(dv.lead.abs().values)

p = dd[dd.play_type.isin(["run", "pass"]) | dd.qb_kneel.fillna(0).astype(bool)].copy()
p["kneel"] = p.qb_kneel.fillna(0).astype(float)
p["run"] = (p.play_type.eq("run") & p.kneel.eq(0)).astype(float)
p = p[p.qb_spike.fillna(0).eq(0)].copy()
p["nxt_gsr"] = p.groupby(["game_id", "fixed_drive"]).game_seconds_remaining.shift(-1)
p["nxt_qtr"] = p.groupby(["game_id", "fixed_drive"]).qtr.shift(-1)
p["sec"] = np.where(p.nxt_qtr.eq(p.qtr), p.game_seconds_remaining - p.nxt_gsr, np.nan)
p["tb"] = tbin(p.game_seconds_remaining.values)
p["lb"] = lbin(p.score_differential.abs().values)
seasons = sorted(p.season.unique())

cell = p[(p.score_differential > 0) & (p.lb >= 0)].copy()
cell["cell"] = cell.lb * 10 + cell.tb
for col in ["sec", "run"]:
    base = np.full(len(cell), np.nan)
    for s in seasons:
        mk = (cell.season == s).values
        mu = cell[~mk & cell[col].notna().values].groupby("cell")[col].mean()
        base[mk] = cell.loc[mk, "cell"].map(mu).values
    cell["res_" + col] = cell[col] - base


def team_game(col):
    sub = cell[cell["res_" + col].notna()]
    return (
        sub.groupby(["game_id", "posteam"])["res_" + col]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"posteam": "team", "mean": "res_" + col, "size": "n_" + col})
    )


tg = tgl.merge(team_game("sec"), on=["game_id", "team"], how="left").merge(
    team_game("run"), on=["game_id", "team"], how="left"
)
QCOLS = ["off_r", "def_r", "opp_off_r", "opp_def_r"]


def quality_adjust(frame, col, wcol):
    out = np.full(len(frame), np.nan)
    ok = frame[col].notna().values
    X = np.column_stack([np.ones(len(frame)), frame[QCOLS].values])
    y = frame[col].values
    w = frame[wcol].fillna(0).values
    for s in seasons:
        te = (frame.season == s).values & ok
        trm = (frame.season != s).values & ok
        sw = np.sqrt(w[trm])
        b = np.linalg.lstsq(X[trm] * sw[:, None], y[trm] * sw, rcond=None)[0]
        out[te] = y[te] - X[te] @ b
    sw = np.sqrt(w[ok])
    bf = np.linalg.lstsq(X[ok] * sw[:, None], y[ok] * sw, rcond=None)[0]
    pred = X[ok] @ bf
    wv = w[ok]
    ybar = np.average(y[ok], weights=wv)
    r2 = 1 - np.average((y[ok] - pred) ** 2, weights=wv) / np.average(
        (y[ok] - ybar) ** 2, weights=wv
    )
    return out, bf, float(r2)


qinfo = {}
for col, wcol in [("res_sec", "n_sec"), ("res_run", "n_run")]:
    tg["adj_" + col], bf, r2 = quality_adjust(tg, col, wcol)
    qinfo[col] = {
        "quality_coef": dict(zip(["const", *QCOLS], map(float, bf), strict=True)),
        "weighted_r2": r2,
    }
for k in ["hc", "oc"]:
    tg["par_" + k] = tg.groupby("sp_" + k).cumcount() % 2


def split_stat(frame, sk, pk, col, wcol, flips=None):
    v = frame[col].values * (flips if flips is not None else 1.0)
    fr = pd.DataFrame(
        {
            sk: frame[sk].values,
            pk: frame[pk].values,
            "vw": v * frame[wcol].values,
            "w": frame[wcol].values,
        }
    )
    g = fr.groupby([sk, pk]).agg(vw=("vw", "sum"), w=("w", "sum")).reset_index()
    g["mu"] = g.vw / g.w
    pv = g.pivot(index=sk, columns=pk, values="mu").dropna()
    r = float(np.corrcoef(pv[0].values, pv[1].values)[0, 1])
    wh = g.pivot(index=sk, columns=pk, values="w").dropna().values.mean()
    return r, float(wh), len(pv)


def season_stat(frame, sk, col, wcol, flips=None):
    v = frame[col].values * (flips if flips is not None else 1.0)
    fr = pd.DataFrame(
        {
            sk: frame[sk].values,
            "season": frame.season.values,
            "vw": v * frame[wcol].values,
            "w": frame[wcol].values,
        }
    )
    g = (
        fr.groupby([sk, "season"])
        .agg(vw=("vw", "sum"), w=("w", "sum"), ng=("w", "size"))
        .reset_index()
    )
    g = g[g.ng >= MIN_SEASON_GAMES]
    g["mu"] = g.vw / g.w
    nx = g[[sk, "season", "mu"]].copy()
    nx["season"] = nx.season - 1
    pr = g.merge(nx, on=[sk, "season"], suffixes=("_t", "_t1"))
    return float(np.corrcoef(pr.mu_t, pr.mu_t1)[0, 1]), len(pr)


rel = {}
for key in ["hc", "oc"]:
    sk, pk = "sp_" + key, "par_" + key
    for col, wcol in [("res_sec", "n_sec"), ("res_run", "n_run")]:
        for kind, c in [("raw", col), ("adj", "adj_" + col)]:
            gk = tg[tg[sk].notna() & tg[c].notna() & tg[wcol].notna()]
            sz = gk.groupby(sk).size()
            gk = gk[gk[sk].isin(sz[sz >= MIN_SPELL_GAMES].index)].copy()
            gk["par"] = gk.groupby(sk).cumcount() % 2
            obs_r, _, nsp = split_stat(gk, sk, "par", c, wcol)
            obs_s, npairs = season_stat(gk, sk, c, wcol)
            pa, pb = [], []
            for _ in range(N_PERM):
                fl = rng.choice([-1.0, 1.0], len(gk))
                pa.append(split_stat(gk, sk, "par", c, wcol, fl)[0])
                pb.append(season_stat(gk, sk, c, wcol, fl)[0])
            pa, pb = np.array(pa), np.array(pb)
            rel[f"{key}|{kind}|{col}"] = {
                "n_spells": nsp,
                "split_half_sb": 2 * obs_r / (1 + obs_r),
                "split_half_null_q95": float(np.quantile(2 * pa / (1 + pa), 0.95)),
                "split_half_p": float((pa >= obs_r).mean()),
                "n_season_pairs": npairs,
                "season_t_t1_r": obs_s,
                "season_null_q95": float(np.quantile(pb, 0.95)),
                "season_p": float((pb >= obs_s).mean()),
            }
for key in ["hc", "oc"]:
    for col in ["res_sec", "res_run"]:
        r_ = rel[f"{key}|raw|{col}"]
        a_ = rel[f"{key}|adj|{col}"]
        a_["sb_ratio_adj_over_raw"] = a_["split_half_sb"] / r_["split_half_sb"]
        a_["season_ratio_adj_over_raw"] = a_["season_t_t1_r"] / r_["season_t_t1_r"]

dleader = dv[(dv.lead > 0) & (dv.lb >= 0)].copy()
dleader["team"] = dleader.posteam
dleader = dleader.merge(
    tgl[["game_id", "team", "sp_oc", "t"]].dropna(subset=["sp_oc"]), on=["game_id", "team"]
)
dleader["cell"] = dleader.lb * 10 + dleader.tb
base = np.full(len(dleader), np.nan)
for s in seasons:
    mk = (dleader.season == s).values
    mu = dleader[~mk].groupby("cell").el.mean()
    base[mk] = dleader.loc[mk, "cell"].map(mu).values
dleader["ratio"] = dleader.el / base
dleader = dleader.dropna(subset=["ratio"]).sort_values(
    ["sp_oc", "t", "start"], ascending=[True, True, False]
)
dleader["par"] = dleader.groupby("sp_oc").cumcount() % 2


def spell_agg(frame, keys):
    g = frame.groupby(keys).ratio.agg(["sum", "size"]).reset_index()
    g.columns = [*keys, "s", "n"]
    return g


def split_half_drive(frame):
    g = frame.groupby(["sp_oc", "par"]).ratio.agg(["mean", "size"]).reset_index()
    pv = g.pivot(index="sp_oc", columns="par", values="mean").dropna()
    nh = g.pivot(index="sp_oc", columns="par", values="size").dropna()
    if len(pv) < 3:
        return 0.0, 1.0
    r = float(np.corrcoef(pv[0].values, pv[1].values)[0, 1])
    return r, float(nh.values.mean())


def multiplier_test(frame, train_rule):
    rows = []
    for s in seasons:
        ho = frame[frame.season == s]
        tr = frame[train_rule(frame.season.values, s)]
        r_, nh = split_half_drive(tr)
        kk = nh * (1 - r_) / r_ if r_ > 0 else np.inf
        trg = spell_agg(tr, ["sp_oc"]).rename(columns={"s": "s_tr", "n": "n_tr"})
        hog = spell_agg(ho, ["sp_oc"]).rename(columns={"s": "s_ho", "n": "n_ho"})
        mg = hog.merge(trg, on="sp_oc")
        mg["raw_m"] = mg.s_tr / mg.n_tr
        mg["shrunk_m"] = 1 + (mg.n_tr / (mg.n_tr + kk)) * (mg.raw_m - 1)
        mg["y"] = mg.s_ho / mg.n_ho
        mg["season"] = s
        mg["K"] = kk
        mg["rel_half"] = r_
        rows.append(mg)
    return pd.concat(rows, ignore_index=True)


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    c = np.average((x - mx) * (y - my), weights=w)
    return float(
        c / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))
    )


def wslope(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.average((x - mx) ** 2, weights=w))


def summarize_mult(mg, col):
    x = (mg[col] - 1).values
    y = (mg.y - 1).values
    w = mg.n_ho.values.astype(float)
    n = len(mg)
    bs, bc, bd = [], [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        if np.std(x[ix]) == 0:
            continue
        bs.append(wslope(x[ix], y[ix], w[ix]))
        bc.append(wcorr(x[ix], y[ix], w[ix]))
        bd.append(
            np.average(y[ix] ** 2, weights=w[ix]) - np.average((y[ix] - x[ix]) ** 2, weights=w[ix])
        )
    bs, bc, bd = map(np.array, (bs, bc, bd))
    return {
        "n_spell_seasons": n,
        "slope": wslope(x, y, w),
        "slope_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "slope_p_positive": float((bs > 0).mean()),
        "corr": wcorr(x, y, w),
        "corr_ci": [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
        "corr_p_positive": float((bc > 0).mean()),
        "mse_gain_vs_constant": float(
            np.average(y**2, weights=w) - np.average((y - x) ** 2, weights=w)
        ),
        "mse_gain_ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
        "mse_gain_p_positive": float((bd > 0).mean()),
    }


mult = {}
variants = {
    "loso_other_seasons": lambda ss, s: ss != s,
    "prior_season_only": lambda ss, s: ss == s - 1,
}
bins = {"all": dleader, **{LEAD_NAMES[i]: dleader[dleader.lb == i] for i in range(3)}}
for bn, fr in bins.items():
    for vn, rule in variants.items():
        mg = multiplier_test(fr, rule)
        mult[f"{bn}|{vn}"] = {
            "median_K_drives": float(mg.K.replace(np.inf, np.nan).median()),
            "mean_half_reliability": float(mg.rel_half.mean()),
            "shrunk": summarize_mult(mg, "shrunk_m"),
            "unshrunk": summarize_mult(mg, "raw_m"),
        }

K_BY_FOLD = {}


def trait_frame(col, wcol):
    sk = "sp_oc"
    out = []
    for s in seasons:
        gk = tg[(tg.season != s) & tg[sk].notna() & tg[col].notna() & tg[wcol].notna()]
        sz = gk.groupby(sk).size()
        gk = gk[gk[sk].isin(sz[sz >= MIN_SPELL_GAMES].index)].copy()
        gk["par"] = gk.groupby(sk).cumcount() % 2
        r_, wh, _ = split_stat(gk, sk, "par", col, wcol)
        kk = wh * (1 - r_) / r_ if r_ > 0 else np.inf
        K_BY_FOLD[f"{col}|{s}"] = [float(r_), float(kk)]
        fr = tg[tg.season == s].copy()
        allr = tg[tg[sk].notna() & tg[col].notna() & tg[wcol].notna()].sort_values("t")
        allr = allr.assign(vw=allr[col] * allr[wcol])
        allr["cvw"] = allr.groupby(sk).vw.cumsum() - allr.vw
        allr["cw"] = allr.groupby(sk)[wcol].cumsum() - allr[wcol]
        allr = allr[["game_id", "team", "cvw", "cw"]]
        fr = fr.merge(allr, on=["game_id", "team"], how="left")
        fr["cvw"] = fr.cvw.fillna(0.0)
        fr["cw"] = fr.cw.fillna(0.0)
        fr = (
            tg[tg.season == s][["game_id", "team", sk, "t"]]
            .merge(
                tg[[sk, "t", "game_id", "team", wcol, col]].dropna(subset=[sk]),
                on=[sk, "t", "game_id", "team"],
                how="left",
            )[["game_id", "team"]]
            .merge(allr, on=["game_id", "team"], how="left")
        )
        fr["cvw"] = fr.cvw.fillna(0.0)
        fr["cw"] = fr.cw.fillna(0.0)
        fr["trait"] = 0.0 if not np.isfinite(kk) else fr.cvw / (fr.cw + kk)
        fr.loc[fr.cw == 0, "trait"] = 0.0
        out.append(fr[["game_id", "team", "trait"]])
    return pd.concat(out, ignore_index=True)


tg_prior_cols = {}
for name, col, wcol in [
    ("sec_adj", "adj_res_sec", "n_sec"),
    ("run_adj", "adj_res_run", "n_run"),
    ("sec_raw", "res_sec", "n_sec"),
]:
    tf = trait_frame(col, wcol)
    tf = tf.drop_duplicates(["game_id", "team"]).rename(columns={"trait": name})
    tg_prior_cols[name] = tf
tgt = tgl[["game_id", "team"]].drop_duplicates(["game_id", "team"])
for name, tf in tg_prior_cols.items():
    tgt = tgt.merge(tf, on=["game_id", "team"], how="left")
    tgt[name] = tgt[name].fillna(0.0)

o = pd.read_parquet(sorted(glob.glob("artifacts/opener_evaluation/*/per_game.parquet"))[-1])
o = o.merge(sch[["game_id", "home_team", "away_team"]], on="game_id")
for name in tg_prior_cols:
    h = tgt.rename(columns={"team": "home_team", name: "h_" + name})[
        ["game_id", "home_team", "h_" + name]
    ]
    a = tgt.rename(columns={"team": "away_team", name: "a_" + name})[
        ["game_id", "away_team", "a_" + name]
    ]
    o = o.merge(h, on=["game_id", "home_team"], how="left").merge(
        a, on=["game_id", "away_team"], how="left"
    )
    o["gap_" + name] = o["h_" + name] - o["a_" + name]
o = o[o.season.between(2020, 2025) & o.margin_vs_open.ne(0)].copy()
o["y"] = (o.margin_vs_open > 0).astype(int)
o["z"] = logit(o.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6))
for name in tg_prior_cols:
    o["gap_" + name] = o["gap_" + name] / o["gap_" + name].std()
arms = {
    "intercept": [],
    "served": ["z"],
    "served+sec_adj": ["z", "gap_sec_adj"],
    "served+run_adj": ["z", "gap_run_adj"],
    "served+sec_raw": ["z", "gap_sec_raw"],
}
P, coef = {}, {}
for a, cols in arms.items():
    pr = pd.Series(0.5, index=o.index)
    coef[a] = {}
    for s in range(2020, 2026):
        tr = o[o.season != s]
        te = o.season == s
        if not cols:
            pr[te] = tr.y.mean()
            continue
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(tr[cols], tr.y)
        pr[te] = mdl.predict_proba(o.loc[te, cols])[:, 1]
        coef[a][s] = np.round(mdl.coef_[0], 4).tolist()
    P[a] = pr.clip(1e-6, 1 - 1e-6)


def ll(pv):
    return -(o.y * np.log(pv) + (1 - o.y) * np.log(1 - pv))


pick = {
    "n": len(o),
    "arms": {},
    "coef_per_fold": {k: v for k, v in coef.items() if v},
    "vs_served": {},
}
for a in arms:
    w = int(((P[a] > 0.5) == (o.y == 1)).sum())
    pick["arms"][a] = {
        "W": w,
        "L": len(o) - w,
        "logloss": float(ll(P[a]).mean()),
        "brier": float(((P[a] - o.y) ** 2).mean()),
    }
for a in list(arms)[2:]:
    dvv = (ll(P["served"]) - ll(P[a])).values
    bt = np.array([dvv[rng.integers(0, len(dvv), len(dvv))].mean() for _ in range(5000)])
    flips = (P[a] > 0.5) != (P["served"] > 0.5)
    fw = int(((P[a] > 0.5)[flips] == (o.y[flips] == 1)).sum())
    cf = np.array([v[1] for v in coef[a].values()])
    pick["vs_served"][a] = {
        "dLL": float(dvv.mean()),
        "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
        "p_positive": float((bt > 0).mean()),
        "seasons_positive": int((pd.Series(dvv).groupby(o.season.values).mean() > 0).sum()),
        "dBrier": float(((P["served"] - o.y) ** 2).mean() - ((P[a] - o.y) ** 2).mean()),
        "flips": int(flips.sum()),
        "flip_wins": fw,
        "flip_losses": int(flips.sum()) - fw,
        "coef_mean": float(cf.mean()),
        "coef_sd": float(cf.std()),
        "coef_sign_consistent": int((np.sign(cf) == np.sign(cf.mean())).sum()),
    }
pick["served_vs_intercept_dLL"] = float(ll(P["intercept"]).mean() - ll(P["served"]).mean())

q4 = dd[dd.qtr.eq(4)].groupby("game_id").first()[["posteam", "home_team", "score_differential"]]
q4["home_q4"] = np.where(q4.posteam == q4.home_team, q4.score_differential, -q4.score_differential)
cg = sch[["game_id", "season", "home_team", "away_team", "result"]].merge(
    q4[["home_q4"]], left_on="game_id", right_index=True
)
for name in ["sec_adj", "run_adj", "sec_raw"]:
    h = tgt.rename(columns={"team": "home_team", name: "h"})[["game_id", "home_team", "h"]]
    a = tgt.rename(columns={"team": "away_team", name: "a"})[["game_id", "away_team", "a"]]
    cg = cg.merge(h, on=["game_id", "home_team"]).merge(a, on=["game_id", "away_team"])
    cg["gap_" + name] = cg.h - cg.a
    cg = cg.drop(columns=["h", "a"])
cg = cg[cg.home_q4 != 0].copy()
sgn = np.sign(cg.home_q4)
cg["lead_q4"] = cg.home_q4.abs()
cg["comp"] = sgn * (cg.result - cg.home_q4)
comp = {}
for name in ["sec_adj", "run_adj", "sec_raw"]:
    cg["ltr"] = sgn * cg["gap_" + name]
    for scope, fr in [("2012_2025", cg[cg.season >= 2012]), ("2020_2025", cg[cg.season >= 2020])]:
        x = fr.ltr.values
        yv = fr.comp.values
        ctrl = np.column_stack([np.ones(len(fr)), fr.lead_q4.values])
        res_y = yv - ctrl @ np.linalg.lstsq(ctrl, yv, rcond=None)[0]
        res_x = x - ctrl @ np.linalg.lstsq(ctrl, x, rcond=None)[0]
        b = float((res_x * res_y).sum() / (res_x**2).sum())
        bt = []
        for _ in range(N_BOOT):
            ix = rng.integers(0, len(fr), len(fr))
            bt.append(float((res_x[ix] * res_y[ix]).sum() / (res_x[ix] ** 2).sum()))
        bt = np.array(bt)
        ss = fr.season.values
        per = [
            float((res_x[ss == s_] * res_y[ss == s_]).sum() / (res_x[ss == s_] ** 2).sum())
            for s_ in sorted(set(ss))
        ]
        comp[f"{name}|{scope}"] = {
            "n": len(fr),
            "slope_points_per_trait_unit": b,
            "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
            "p_positive": float((bt > 0).mean()),
            "corr": float(np.corrcoef(res_x, res_y)[0, 1]),
            "seasons_positive": int(sum(v > 0 for v in per)),
            "n_seasons": len(per),
        }

res = {
    "looks": {
        "reliability_cells": len(rel),
        "multiplier_cells": len(mult) * 2,
        "pick_arms": len(arms) - 2,
        "compression_cells": len(comp),
        "family": "lead-protection trait, quality, drive multiplier, pick, compression terms",
    },
    "quality_adjustment": qinfo,
    "reliability": rel,
    "multiplier": mult,
    "pick": pick,
    "compression": comp,
    "trait_fold_reliability_K": K_BY_FOLD,
}
Path(f"{OUT}/unit2.json").write_text(json.dumps(res, indent=1, default=float))
print("done", res["looks"])
