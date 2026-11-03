import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import glob

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

WP_LO, WP_HI = 0.05, 0.95
MIN_SPELL_GAMES = 16
MIN_SEASON_GAMES = 6
N_BOOT = 2000
N_PERM = 0
rng = np.random.default_rng(1702)

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
        "qb_dropback",
        "passer_player_id",
    ],
)
d = d[d.season_type.eq("REG") & d.play_type.isin(["pass", "run"]) & d.qtr.between(1, 4)]
d = d[~d.qb_kneel.fillna(0).astype(bool) & ~d.qb_spike.fillna(0).astype(bool)]
d = d.dropna(subset=["epa", "wp", "posteam", "defteam"])
d = d[d.wp.between(WP_LO, WP_HI)].copy()
d["half"] = np.where(d.qtr <= 2, 1, 2)

h = (
    d[d.qtr == 2]
    .groupby(["game_id", "posteam"])
    .score_differential.last()
    .rename("margin")
    .reset_index()
    .rename(columns={"posteam": "team"})
)


def side(group_col, other_col, name):
    g = d.groupby(["game_id", "season", "week", group_col, other_col, "half"]).epa.agg(
        ["mean", "size"]
    )
    g = g.unstack("half")
    g.columns = [f"{a}{b}" for a, b in g.columns]
    g = g.dropna().reset_index().rename(columns={group_col: "team", other_col: "opp"})
    g[name] = g.mean2 - g.mean1
    g["w_" + name] = 1.0 / (1.0 / g.size1 + 1.0 / g.size2)
    return g[["game_id", "season", "week", "team", "opp", name, "w_" + name]]


off = side("posteam", "defteam", "off_delta")
de = side("defteam", "posteam", "def_delta")
g = off.merge(de, on=["game_id", "season", "week", "team", "opp"])
g = g.merge(h, on=["game_id", "team"], how="left")
g["margin"] = g.margin.fillna(0.0)

s = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[
    ["game_id", "home_team", "home_coach", "away_coach"]
]
g = g.merge(s, on="game_id")
g["coach"] = np.where(g.team == g.home_team, g.home_coach, g.away_coach)
g = g.dropna(subset=["coach"]).sort_values(["season", "week"]).reset_index(drop=True)
g["spell"] = g.coach + "|" + g.team

for col in ["off_delta", "def_delta"]:
    tot = g.groupby(["season", "team"])[col].transform("sum")
    n = g.groupby(["season", "team"])[col].transform("size")
    g[col + "_loo"] = ((tot - g[col]) / (n - 1).replace(0, np.nan)).fillna(0.0)
loo = g.set_index(["game_id", "team"])
key = list(zip(g.game_id, g.opp, strict=True))
g["opp_off_loo"] = loo.off_delta_loo.reindex(key).fillna(0.0).values
g["opp_def_loo"] = loo.def_delta_loo.reindex(key).fillna(0.0).values
g["margin_abs"] = g.margin.abs()


def loso_resid(frame, y, xcols):
    out = np.zeros(len(frame))
    X = np.column_stack([np.ones(len(frame))] + [frame[c].values for c in xcols])
    yv = frame[y].values
    w = frame["w_" + y].values
    coefs = []
    for sn in sorted(frame.season.unique()):
        m = (frame.season == sn).values
        sw = np.sqrt(w[~m])
        b = np.linalg.lstsq(X[~m] * sw[:, None], yv[~m] * sw, rcond=None)[0]
        out[m] = yv[m] - X[m] @ b
        coefs.append(b)
    return out, np.array(coefs)


g["off_res"], co = loso_resid(g, "off_delta", ["margin", "margin_abs", "opp_def_loo"])
g["def_res"], cd = loso_resid(g, "def_delta", ["margin", "margin_abs", "opp_off_loo"])

dbk = d[d.qb_dropback.fillna(0).astype(bool) & d.passer_player_id.notna()]
qbn = dbk.groupby(["game_id", "posteam", "passer_player_id"]).size().rename("n").reset_index()
qbn = qbn.sort_values(["game_id", "posteam", "n"], ascending=[True, True, False])
qbn = qbn.drop_duplicates(["game_id", "posteam"]).rename(
    columns={"posteam": "team", "passer_player_id": "qb"}
)
g = g.merge(qbn[["game_id", "team", "qb"]], on=["game_id", "team"], how="left")

sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[["game_id", "gameday"]]
g = g.merge(sch, on="game_id")
g["t"] = pd.to_datetime(g.gameday, utc=True) + pd.Timedelta(days=1)
cdir = [
    p
    for p in sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))
    if "coordinator_history" in p
][-1]
ch = pd.read_parquet(cdir)
ch["effective_observed_at"] = pd.to_datetime(ch.effective_observed_at, utc=True).astype(
    "datetime64[ns, UTC]"
)
g["t"] = g.t.astype("datetime64[ns, UTC]")
g["season"] = g.season.astype("int64")
g = g.sort_values("t").reset_index(drop=True)
for role, name in [("HC", "hc_p"), ("OC", "oc"), ("DC", "dc")]:
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
    g[name] = (m.person.fillna(m.first_person)).values
g = g.sort_values(["season", "week"]).reset_index(drop=True)
for k, col in [("hc", "coach"), ("oc", "oc"), ("dc", "dc"), ("qb", "qb")]:
    g["sp_" + k] = g[col].astype("string") + "|" + g.team
    g.loc[g[col].isna(), "sp_" + k] = pd.NA
    g["par_" + k] = g.groupby("sp_" + k).cumcount() % 2


OPENER = sorted(glob.glob("artifacts/opener_evaluation/*/per_game.parquet"))[-1]
FIRST_TEST, LAST_TEST = 2020, 2025
CLIP = 1e-6

g = g.sort_values(["season", "week"]).reset_index(drop=True)


def shrunk_table(train, key, col):
    gk = train[train["sp_" + key].notna()]
    grp = gk.groupby("sp_" + key)[col]
    m, n = grp.mean(), grp.size()
    dev = gk[col] - gk["sp_" + key].map(m)
    sig2 = float((dev**2).sum() / max(len(gk) - len(m), 1))
    grand = float(gk[col].mean())
    tau2 = max(float(m.var(ddof=1)) - float((sig2 / n).mean()), 0.0)
    b = tau2 / (tau2 + sig2 / n)
    return grand + b * (m - grand), float(tau2), sig2


def predict_all(frame, first_season):
    out = {}
    taus = {}
    for col, keys in [("off_res", ["hc", "oc"]), ("def_res", ["hc", "dc"])]:
        for k in keys:
            pred = pd.Series(np.nan, index=frame.index)
            for s in range(first_season, int(frame.season.max()) + 1):
                tr = frame[frame.season < s]
                tab, tau2, sig2 = shrunk_table(tr, k, col)
                taus[f"{col}_{k}_{s}"] = [tau2, sig2]
                te = frame.season == s
                pred[te] = frame.loc[te, "sp_" + k].map(tab)
            out[f"{col}_{k}"] = pred
    return out, taus


FIRST_PRED = 2010
preds, taus = predict_all(g, FIRST_PRED)
for k, v in preds.items():
    g["p_" + k] = v


def corr_slope(x, y):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc**2).sum()), float(np.corrcoef(x, y)[0, 1])


def boot(frame, xcol, ycol, n_boot=N_BOOT):
    x, y = frame[xcol].values, frame[ycol].values
    idx = [rng.integers(0, len(x), len(x)) for _ in range(n_boot)]
    sl = np.array([corr_slope(x[i], y[i]) for i in idx])
    return sl


part1 = {}
for col, keys in [("off_res", ("oc", "hc")), ("def_res", ("dc", "hc"))]:
    pc = ["p_" + col + "_" + k for k in keys]
    cm = g.dropna(subset=[*pc, col]).reset_index(drop=True)
    res = {"games": len(cm)}
    bs = {}
    for k, p in zip(keys, pc, strict=True):
        sl, r = corr_slope(cm[p].values, cm[col].values)
        b = boot(cm, p, col)
        bs[k] = b
        mse0 = float((cm[col] ** 2).mean())
        mse1 = float(((cm[col] - cm[p]) ** 2).mean())
        res[k] = {
            "slope": sl,
            "slope_ci95": np.percentile(b[:, 0], [2.5, 97.5]).tolist(),
            "slope_prob_positive": float((b[:, 0] > 0).mean()),
            "corr": r,
            "corr_ci95": np.percentile(b[:, 1], [2.5, 97.5]).tolist(),
            "corr_prob_positive": float((b[:, 1] > 0).mean()),
            "mse_reduction_vs_zero": mse0 - mse1,
            "pred_sd": float(cm[p].std()),
        }
    dr = bs[keys[0]][:, 1] - bs[keys[1]][:, 1]
    res["corr_diff_first_minus_second"] = {
        "mean": float(dr.mean()),
        "ci95": np.percentile(dr, [2.5, 97.5]).tolist(),
        "prob_positive": float((dr > 0).mean()),
    }
    part1[col] = res

s2 = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[
    ["game_id", "home_team", "away_team"]
]
tm = g[["game_id", "team", "p_off_res_oc", "p_def_res_dc"]]
hm = s2.merge(
    tm.rename(columns={"team": "home_team", "p_off_res_oc": "h_oc", "p_def_res_dc": "h_dc"}),
    on=["game_id", "home_team"],
    how="left",
).merge(
    tm.rename(columns={"team": "away_team", "p_off_res_oc": "a_oc", "p_def_res_dc": "a_dc"}),
    on=["game_id", "away_team"],
    how="left",
)
hm[["h_oc", "a_oc", "h_dc", "a_dc"]] = hm[["h_oc", "a_oc", "h_dc", "a_dc"]].fillna(0.0)
hm["f_off"] = hm.h_oc - hm.a_oc
hm["f_def"] = -(hm.h_dc - hm.a_dc)
hm["f_tot"] = hm.f_off + hm.f_def

o = pd.read_parquet(OPENER)
d = o.merge(hm[["game_id", "f_off", "f_def", "f_tot"]], on="game_id")
d = d[d.season.between(FIRST_TEST, LAST_TEST) & d.margin_vs_open.ne(0)].reset_index(drop=True)
d["y"] = (d.margin_vs_open > 0).astype(int)
d["z"] = logit(d.home_cover_probability_at_open.clip(CLIP, 1 - CLIP))
arms = {
    "intercept": [],
    "served": ["z"],
    "served+f_tot": ["z", "f_tot"],
    "served+f_off": ["z", "f_off"],
    "served+f_def": ["z", "f_def"],
}
P, P_in, coef = {}, {}, {}
for a, cols in arms.items():
    p = pd.Series(0.5, index=d.index)
    pin = pd.Series(0.5, index=d.index)
    coef[a] = {}
    for s in range(FIRST_TEST, LAST_TEST + 1):
        tr = d[d.season != s]
        te = d.season == s
        if not cols:
            p[te] = tr.y.mean()
            pin[te] = d.y.mean()
            continue
        sd = tr[cols[1:]].std() if len(cols) > 1 else None
        X = d[cols].copy()
        if sd is not None:
            X[cols[1:]] = X[cols[1:]] / sd
        m = LogisticRegression(C=1e6, max_iter=1000).fit(X[~te.values], tr.y)
        p[te] = m.predict_proba(X[te.values])[:, 1]
        coef[a][s] = np.round(m.coef_[0], 4).tolist()
    mi = LogisticRegression(C=1e6, max_iter=1000)
    if cols:
        sd = d[cols[1:]].std() if len(cols) > 1 else None
        X = d[cols].copy()
        if sd is not None:
            X[cols[1:]] = X[cols[1:]] / sd
        mi.fit(X, d.y)
        pin = pd.Series(mi.predict_proba(X)[:, 1], index=d.index)
    P[a] = p.clip(CLIP, 1 - CLIP)
    P_in[a] = pin.clip(CLIP, 1 - CLIP)


def ll(p):
    return -(d.y * np.log(p) + (1 - d.y) * np.log(1 - p))


part2 = {"n": len(d), "opener_file": OPENER.replace("\\", "/"), "arms": {}}
pick = {a: P[a] > 0.5 for a in arms}
win = d.y == 1
for a in arms:
    w = int((pick[a] == win).sum())
    flips = pick[a] != pick["served"]
    fw = int(((pick[a] == win) & flips).sum())
    part2["arms"][a] = {
        "W": w,
        "L": len(d) - w,
        "decisive_flips_vs_served": int(flips.sum()),
        "flips_won_by_arm": fw,
        "flips_lost_by_arm": int(flips.sum()) - fw,
        "ll_oos": float(ll(P[a]).mean()),
        "ll_in_sample": float(ll(P_in[a]).mean()),
        "brier_oos": float(((P[a] - d.y) ** 2).mean()),
        "coef_per_fold": coef[a],
    }
    part2["arms"][a]["oos_minus_in_sample_ll_gap"] = (
        part2["arms"][a]["ll_oos"] - part2["arms"][a]["ll_in_sample"]
    )
base = ll(P["served"]).values
for a in list(arms)[2:]:
    diff = base - ll(P[a]).values
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)])
    sb = pd.Series(diff).groupby(d.season.values).mean()
    part2["arms"][a]["dLL_vs_served"] = {
        "mean": float(diff.mean()),
        "se": float(bs.std()),
        "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
        "prob_positive": float((bs > 0).mean()),
        "seasons_positive": int((sb > 0).sum()),
    }
    bb = ((P["served"] - d.y) ** 2).values - ((P[a] - d.y) ** 2).values
    part2["arms"][a]["dBrier_vs_served"] = float(bb.mean())
dI = ll(P["intercept"]).values - base
part2["served_vs_intercept_dLL"] = float(dI.mean())
part2["looks"] = {"part1": 4 + 2, "part2": 3, "intercept_control": 1}
part2["looks"]["total"] = sum(part2["looks"].values())
os.makedirs("artifacts/sim17", exist_ok=True)
out = {"part1": part1, "part2": part2}
with open("artifacts/sim17/unit3_heldout.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
