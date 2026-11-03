import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import glob

import numpy as np
import pandas as pd

WP_LO, WP_HI = 0.05, 0.95
MIN_SPELL_GAMES = 16
MIN_SEASON_GAMES = 6
N_BOOT = 2000
N_PERM = 2000
rng = np.random.default_rng(17)

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

g["ord"] = g.groupby("spell").cumcount()
g["parity"] = g.ord % 2
res = {
    "games": len(g),
    "wp_filter": [WP_LO, WP_HI],
    "coef_mean_off": co.mean(0).tolist(),
    "coef_sd_off": co.std(0).tolist(),
    "coef_mean_def": cd.mean(0).tolist(),
    "coef_sd_def": cd.std(0).tolist(),
}
res["league_raw"] = {}
for c in ["off_delta", "def_delta"]:
    v = g[c].values
    w = g["w_" + c].values
    res["league_raw"][c] = {
        "wmean": float(np.average(v, weights=w)),
        "mean": float(v.mean()),
        "se": float(v.std() / np.sqrt(len(v))),
    }

sp = g.groupby("spell").size()
keep = sp[sp >= MIN_SPELL_GAMES].index
gk = g[g.spell.isin(keep)].copy()
res["spells_kept"] = len(keep)
res["spells_total"] = len(sp)


def boot_ci(a, b):
    n = len(a)
    rs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        rs.append(np.corrcoef(a[i], b[i])[0, 1])
    rs = np.array(rs)
    return np.percentile(rs, [2.5, 97.5]).tolist(), float(np.mean(rs > 0))


out = {}
for col in ["off_res", "def_res"]:
    p = gk.pivot_table(index="spell", columns="parity", values=col, aggfunc="mean").dropna()
    a, b = p[0].values, p[1].values
    r = float(np.corrcoef(a, b)[0, 1])
    ci, ppos = boot_ci(a, b)
    sm = gk.groupby(["spell", "season"])[col].agg(["mean", "size"]).reset_index()
    sm = sm[sm["size"] >= MIN_SEASON_GAMES]
    nxt = sm.copy()
    nxt["season"] = nxt.season - 1
    pr = sm.merge(nxt, on=["spell", "season"], suffixes=("_t", "_t1"))
    x, y = pr.mean_t.values, pr.mean_t1.values
    r_ts = float(np.corrcoef(x, y)[0, 1])
    ci_ts, ppos_ts = boot_ci(x, y)
    codes = pd.factorize(gk.spell)[0]
    vals = gk[col].values
    par = gk.parity.values
    cnt = np.bincount(codes)
    ca = np.maximum(np.bincount(codes, par == 0), 1)
    cb = np.maximum(np.bincount(codes, par == 1), 1)
    obs_bv = float((np.bincount(codes, vals) / cnt).var(ddof=1))
    null_bv, null_r = [], []
    for _ in range(N_PERM):
        v = vals * rng.choice([-1.0, 1.0], len(vals))
        null_bv.append((np.bincount(codes, v) / cnt).var(ddof=1))
        pa = np.bincount(codes, v * (par == 0)) / ca
        pb = np.bincount(codes, v * (par == 1)) / cb
        null_r.append(np.corrcoef(pa, pb)[0, 1])
    null_bv, null_r = np.array(null_bv), np.array(null_r)
    out[col] = {
        "n_spells": len(a),
        "spell_mean_sd": float(np.sqrt(obs_bv)),
        "odd_even_r": r,
        "odd_even_ci95": ci,
        "odd_even_prob_positive": ppos,
        "spearman_brown": 2 * r / (1 + r),
        "season_t_t1_r": r_ts,
        "season_pairs": len(pr),
        "season_t_t1_ci95": ci_ts,
        "season_t_t1_prob_positive": ppos_ts,
        "between_var_obs": obs_bv,
        "between_var_null_mean": float(null_bv.mean()),
        "between_var_null_p95": float(np.percentile(null_bv, 95)),
        "between_var_perm_p": float((np.sum(null_bv >= obs_bv) + 1) / (N_PERM + 1)),
        "excess_var_share": float(max(obs_bv - null_bv.mean(), 0) / obs_bv),
        "null_r_sd": float(null_r.std()),
        "r_perm_p": float((np.sum(null_r >= r) + 1) / (N_PERM + 1)),
        "resid_wmean": float(np.average(g[col], weights=g["w_" + col.replace("_res", "_delta")])),
    }
res["metrics"] = out
os.makedirs("artifacts/sim17", exist_ok=True)
with open("artifacts/sim17/halftime_adjustment.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
