import glob
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nfl_ats.home_side_location import TRAILING_SEASONS
from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    BAND_STEP,
    MAX_BAND,
    MIN_BAND_GAMES,
    tilted_atoms,
)
from nfl_ats.schedule_flag_features import default_opener_lines

K_GRID = [4, 8, 16, 32, 64]
FIRST_SCORED = 2012
OPENER_FIRST = 2020
N_BOOT = 1000
WITHIN = 3.0
EDGES = [0.0, 1.5, 3.0]
LINES = [-10.0, -7.0, -3.0, 3.0, 7.0, 10.0]
SCALES = [0.9, 1.0, 1.1]
rng = np.random.default_rng(4501)
OUT = Path("artifacts/mod45")
OUT.mkdir(parents=True, exist_ok=True)

sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])
sch = sch[sch.game_type.eq("REG")].copy()
sch["season"] = sch.season.astype("int64")
sch["day"] = pd.to_datetime(sch.gameday)
op = default_opener_lines(sch)
sch = sch.merge(op, on="game_id", how="left")

d = pd.read_parquet(
    "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=["game_id", "season_type", "posteam", "play_type", "qb_kneel", "qb_spike"],
)
d = d[d.season_type.eq("REG") & d.posteam.notna() & d.play_type.isin(["run", "pass"])]
d = d[d.qb_kneel.fillna(0).eq(0) & d.qb_spike.fillna(0).eq(0)]
pl = d.groupby(["game_id", "posteam"]).size().rename("plays").reset_index()

tg = pd.concat(
    [
        sch.assign(team=sch.home_team, coach=sch.home_coach, home=1),
        sch.assign(team=sch.away_team, coach=sch.away_coach, home=0),
    ]
)[["game_id", "season", "week", "day", "team", "coach", "home"]]
tg["t"] = (tg.day + pd.Timedelta(days=1)).dt.tz_localize("UTC").astype("datetime64[ns, UTC]")
tg = tg.merge(pl.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left")
tg = tg[tg.plays.notna()].copy()

cdir = sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))[-1]
ch = pd.read_parquet(cdir)
ch["effective_observed_at"] = pd.to_datetime(ch.effective_observed_at, utc=True).astype(
    "datetime64[ns, UTC]"
)
oc = ch[ch.role == "OC"].sort_values("effective_observed_at").astype({"season": "int64"})
tg = tg.sort_values("t").reset_index(drop=True)
m = pd.merge_asof(
    tg[["game_id", "team", "season", "t"]],
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
tg["oc"] = m.person.fillna(m.first_person).values
tg["sp_oc"] = tg.oc.astype("string") + "|" + tg.team
tg["sp_hc"] = tg.coach.astype("string") + "|" + tg.team
tg = tg.sort_values(["day", "game_id"]).reset_index(drop=True)


def traits(k):
    sums = {"oc": {}, "hc": {}}
    cnts = {"oc": {}, "hc": {}}
    ls, ln = 0.0, 0
    out_oc = np.full(len(tg), np.nan)
    out_hc = np.full(len(tg), np.nan)
    out_mu = np.full(len(tg), np.nan)
    for _, g in tg.groupby("day", sort=True):
        mu = ls / ln if ln else np.nan
        for i, r in zip(g.index, g.itertuples(), strict=True):
            out_mu[i] = mu
            for role, key, out in (("oc", r.sp_oc, out_oc), ("hc", r.sp_hc, out_hc)):
                n = cnts[role].get(key, 0)
                s = sums[role].get(key, 0.0)
                out[i] = (s - n * mu) / (n + k) if n and ln else 0.0
        for r in g.itertuples():
            ls += r.plays
            ln += 1
            for role, key in (("oc", r.sp_oc), ("hc", r.sp_hc)):
                sums[role][key] = sums[role].get(key, 0.0) + r.plays
                cnts[role][key] = cnts[role].get(key, 0) + 1
    return out_mu, out_oc, out_hc


per_k = {k: traits(k) for k in K_GRID}


def game_frame(k):
    mu, t_oc, t_hc = per_k[k]
    f = tg[["game_id", "season", "home", "plays"]].copy()
    f["mu"] = mu
    f["oc"] = t_oc
    f["hc"] = t_hc
    h = f[f.home == 1].set_index("game_id")
    a = f[f.home == 0].set_index("game_id")
    j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
    r = pd.DataFrame(
        {
            "season": j.season_h,
            "plays": j.plays_h + j.plays_a,
            "mu2": j.mu_h + j.mu_a,
            "oc": j.oc_h + j.oc_a,
            "hc": j.hc_h + j.hc_a,
        }
    )
    r["mix"] = 0.5 * (r.oc + r.hc)
    return r.reset_index()


gm = sch[
    [
        "game_id",
        "result",
        "spread_line",
        "total_line",
        "tue_open_home_spread",
        "tue_open_total_line",
    ]
]
base = {}
for k in K_GRID:
    v = game_frame(k).merge(gm, on="game_id")
    base[k] = v[v.season >= FIRST_SCORED].dropna(subset=["mu2"]).reset_index(drop=True)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def loso_pred(df, cols):
    pred = np.full(len(df), np.nan)
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    for s in sorted(df.season.unique()):
        te = (df.season == s).values
        b = ols(X[~te], df.plays.values[~te])
        pred[te] = X[te] @ b
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def boot_diff(e_a, e_b):
    n = len(e_a)
    diff = np.asarray(e_a) - np.asarray(e_b)
    bs = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return [
        float(diff.mean()),
        [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        float((bs > 0).mean()),
    ]


res = {"step1": {}}
seasons = sorted(base[K_GRID[0]].season.unique())
pick = {}
for s in seasons:
    best = None
    for k in K_GRID:
        dk = base[k]
        tr = dk[dk.season != s]
        sse = ((tr.plays - tr.mu2 - tr.oc) ** 2).sum()
        if best is None or sse < best[1]:
            best = (k, sse)
    pick[s] = best[0]
res["k_choice_loso"] = {int(s): int(k) for s, k in pick.items()}
G = pd.concat([base[pick[s]][base[pick[s]].season == s] for s in seasons]).reset_index(drop=True)
G["exp_oc"] = G.mu2 + G.oc
G["exp_hc"] = G.mu2 + G.hc
G["exp_mix"] = G.mu2 + G.mix

sources = {
    "close_2012_25": ("total_line", G[G.total_line.notna()]),
    "open_2020_25": (
        "tue_open_total_line",
        G[G.season.ge(OPENER_FIRST) & G.tue_open_total_line.notna()],
    ),
}
for name, (tc, sub) in sources.items():
    sub = sub.reset_index(drop=True)
    out = {"n": len(sub)}
    p0 = loso_pred(sub, ["mu2"])
    pt = loso_pred(sub, [tc])
    out["r2_league_mean_only"] = float(r2(sub.plays, p0))
    out["r2_total_only"] = float(r2(sub.plays, pt))
    out["corr_total_plays"] = float(np.corrcoef(sub[tc], sub.plays)[0, 1])
    for arm in ["exp_oc", "exp_hc", "exp_mix"]:
        pa = loso_pred(sub, [arm])
        pb = loso_pred(sub, [tc, arm])
        out[arm] = {
            "r2_alone": float(r2(sub.plays, pa)),
            "r2_with_total": float(r2(sub.plays, pb)),
            "corr_exp_plays": float(np.corrcoef(sub[arm], sub.plays)[0, 1]),
            "corr_exp_total": float(np.corrcoef(sub[arm], sub[tc])[0, 1]),
            "sse_gain_vs_total_only": boot_diff((sub.plays - pt) ** 2, (sub.plays - pb) ** 2),
        }
    res["step1"][name] = out


def z(x):
    return (x - x.mean()) / x.std()


def fit_disp(sub, line_col, tot_col, arm):
    sub = sub.dropna(subset=[line_col, tot_col]).copy()
    sub["margin"] = sub.result - sub[line_col]
    sub["a"] = sub.margin.abs()
    sub["w3"] = (sub.a <= WITHIN).astype(int)
    sub["zx"] = z(sub[arm])
    sub["zt"] = z(sub[tot_col])
    sub["zs"] = z(sub[line_col].abs())
    sub = sub.reset_index(drop=True)
    X = np.column_stack([np.ones(len(sub)), sub.zx, sub.zt, sub.zs])
    out = {"n": len(sub)}
    b = ols(X, sub.a.values)
    bs = []
    bl = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(sub), len(sub))
        bs.append(ols(X[ix], sub.a.values[ix])[1])
        lr = LogisticRegression(C=1e6, max_iter=200).fit(X[ix][:, 1:], sub.w3.values[ix])
        bl.append(lr.coef_[0][0])
    bs = np.array(bs)
    bl = np.array(bl)
    out["abs_resid_slope_per_sd"] = {
        "coef": float(b[1]),
        "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "p_positive": float((bs > 0).mean()),
        "coef_total": float(b[2]),
        "coef_abs_spread": float(b[3]),
    }
    lr = LogisticRegression(C=1e6, max_iter=200).fit(X[:, 1:], sub.w3.values)
    out["within3_logit_per_sd"] = {
        "coef": float(lr.coef_[0][0]),
        "ci": [float(np.quantile(bl, 0.025)), float(np.quantile(bl, 0.975))],
        "p_negative": float((bl < 0).mean()),
        "base_rate": float(sub.w3.mean()),
    }
    per = {}
    sse_full, sse_red = 0.0, 0.0
    cols = [0, 2, 3]
    for s in sorted(sub.season.unique()):
        te = (sub.season == s).values
        bf = ols(X[~te], sub.a.values[~te])
        br = ols(X[~te][:, cols], sub.a.values[~te])
        sse_full += ((sub.a.values[te] - X[te] @ bf) ** 2).sum()
        sse_red += ((sub.a.values[te] - X[te][:, cols] @ br) ** 2).sum()
        per[int(s)] = float(ols(X[te], sub.a.values[te])[1])
    out["loso_mse_gain_vs_no_tempo"] = float((sse_red - sse_full) / len(sub))
    out["per_season_slope"] = per
    out["seasons_positive"] = int(sum(v > 0 for v in per.values()))
    return out, sub


res["step2"] = {}
opener_sub = {}
for arm in ["exp_oc", "exp_hc", "exp_mix"]:
    o, s_ = fit_disp(
        G[G.season >= OPENER_FIRST], "tue_open_home_spread", "tue_open_total_line", arm
    )
    res["step2"]["open_2020_25|" + arm] = o
    opener_sub[arm] = s_
    o, _ = fit_disp(G, "spread_line", "total_line", arm)
    res["step2"]["close_2012_25|" + arm] = o
res["looks"] = {
    "step1": len(sources) * 4,
    "step2": len(res["step2"]) * 2,
    "step3": len(LINES) * len(EDGES) * len(SCALES),
}

pool = sch[sch.result.notna() & sch.spread_line.notna()][
    ["game_id", "season", "day", "result", "spread_line", "tue_open_home_spread"]
].copy()
pool["line"] = pool.tue_open_home_spread.fillna(pool.spread_line)
pool["result"] = np.rint(pool.result.astype(float))


def scaled_atoms(margins, line, point, s):
    v, n = np.unique(margins, return_counts=True)
    n = n.astype(float)
    dev = np.abs(v - line)
    if abs(s - 1.0) > 1e-12:
        target = s * (n / n.sum() * dev).sum()
        lo, hi = -2.0, 2.0
        for _ in range(80):
            g = 0.5 * (lo + hi)
            w = n * np.exp(-g * dev)
            w = w / w.sum()
            if (w * dev).sum() > target:
                lo = g
            else:
                hi = g
        n = n * np.exp(-0.5 * (lo + hi) * dev)
    mass, _ = tilted_atoms(v, n, line, point)
    return v, mass


def pool_margins(prior, line):
    band = BAND_HALF_WIDTH
    while True:
        sel = np.abs(prior.line.values - line) <= band
        if sel.sum() >= MIN_BAND_GAMES or band >= MAX_BAND:
            break
        band = min(band + BAND_STEP, MAX_BAND)
    return prior.result.values[sel]


def cover_non_push(v, mass, line):
    c = mass[v > line + 1e-9].sum()
    lo = mass[v < line - 1e-9].sum()
    return c / (c + lo)


tab = []
prior_t = pool[(pool.season < OPENER_FIRST) & (pool.season >= OPENER_FIRST - TRAILING_SEASONS)]
for line in LINES:
    mm = pool_margins(prior_t, line)
    for edge in EDGES:
        row = {"line": line, "edge": edge}
        for s in SCALES:
            v, mass = scaled_atoms(mm, line, line + edge, s)
            row[f"scale_{s:.1f}"] = float(cover_non_push(v, mass, line))
        tab.append(row)
res["step3_table"] = tab

o = pd.read_parquet(sorted(glob.glob("artifacts/opener_evaluation/*/per_game.parquet"))[-1])
o = o[["game_id", "residual_at_open", "home_cover_probability_at_open"]]
A = opener_sub["exp_oc"].merge(o, on="game_id")
A = A.merge(sch[["game_id", "day"]], on="game_id")
A = A[A.margin.ne(0)].reset_index(drop=True)
A["y"] = (A.margin > 0).astype(int)
A["line"] = A.tue_open_home_spread
A["point"] = A.line + A.residual_at_open
A["lla"] = np.log(A.a.clip(lower=0.5))
Xs = np.column_stack([np.ones(len(A)), A.zx, A.zt, A.zs])
A["pb"] = np.nan
A["ps"] = np.nan
A["pi"] = np.nan
betas = {}
for s in sorted(A.season.unique()):
    te = (A.season == s).values
    beta = ols(Xs[~te], A.lla.values[~te])[1]
    betas[int(s)] = float(beta)
    A.loc[te, "pi"] = A.y[~te].mean()
    for i in A.index[te]:
        r = A.loc[i]
        prior = pool[
            (pool.day < r.day) & (pool.season >= s - TRAILING_SEASONS) & (pool.game_id != r.game_id)
        ]
        mm = pool_margins(prior, r.line)
        v, mass = scaled_atoms(mm, r.line, r.point, 1.0)
        A.loc[i, "pb"] = cover_non_push(v, mass, r.line)
        v, mass = scaled_atoms(mm, r.line, r.point, float(np.exp(beta * r.zx)))
        A.loc[i, "ps"] = cover_non_push(v, mass, r.line)


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


y = A.y.values
l_i = logloss(y, A.pi.values)
l_b = logloss(y, A.pb.values)
l_s = logloss(y, A.ps.values)
l_g = logloss(y, A.home_cover_probability_at_open.values)
g = {
    "n": len(A),
    "ll_intercept": float(l_i.mean()),
    "ll_discrete_base": float(l_b.mean()),
    "ll_discrete_scaled": float(l_s.mean()),
    "ll_served_file": float(l_g.mean()),
    "scaled_minus_base_gain": boot_diff(l_b, l_s),
    "scaled_gain_vs_intercept": boot_diff(l_i, l_s),
    "base_gain_vs_intercept": boot_diff(l_i, l_b),
}
hb = A.pb.values > 0.5
hs = A.ps.values > 0.5
flip = hb != hs
win = y == 1
g["pick_base_W_L"] = [int((hb == win).sum()), int((hb != win).sum())]
g["pick_scaled_W_L"] = [int((hs == win).sum()), int((hs != win).sum())]
g["flips"] = int(flip.sum())
g["flips_scaled_W_L"] = [int((hs[flip] == win[flip]).sum()), int((hs[flip] != win[flip]).sum())]
g["mean_abs_prob_change"] = float(np.abs(A.ps - A.pb).mean())
g["max_abs_prob_change"] = float(np.abs(A.ps - A.pb).max())
g["beta_log_abs_by_heldout_season"] = betas
res["step3_graded"] = g
(OUT / "unit1.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
