import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import glob

import numpy as np
import pandas as pd
from eng53_units import emit, piece

WP_LO, WP_HI = 0.05, 0.95
MIN_SPELL_GAMES = 16
MIN_SEASON_GAMES = 6
N_BOOT = 2000
N_PERM = 2000
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


def boot_ci(a, b):
    n = len(a)
    rs = np.array(
        [np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    )
    return np.percentile(rs, [2.5, 97.5]).tolist(), float(np.mean(rs > 0))


def reliab(frame, key, col, sink=None):
    gk = frame[frame["sp_" + key].notna()]
    sp = gk.groupby("sp_" + key).size()
    gk = gk[gk["sp_" + key].isin(sp[sp >= MIN_SPELL_GAMES].index)].copy()
    sk = "sp_" + key
    if sink is not None:
        sink.append(
            piece(
                f"{col}|{key}",
                gk[sk],
                gk[sk],
                gk.season,
                gk.game_id,
                gk["par_" + key],
                gk[col],
                np.ones(len(gk)),
            )
        )
    p = gk.pivot_table(index=sk, columns="par_" + key, values=col, aggfunc="mean").dropna()
    a, b = p[0].values, p[1].values
    r = float(np.corrcoef(a, b)[0, 1])
    ci, ppos = boot_ci(a, b)
    sm = gk.groupby([sk, "season"])[col].agg(["mean", "size"]).reset_index()
    sm = sm[sm["size"] >= MIN_SEASON_GAMES]
    nxt = sm.copy()
    nxt["season"] = nxt.season - 1
    pr = sm.merge(nxt, on=[sk, "season"], suffixes=("_t", "_t1"))
    x, y = pr.mean_t.values, pr.mean_t1.values
    r_ts = float(np.corrcoef(x, y)[0, 1])
    ci_ts, ppos_ts = boot_ci(x, y)
    codes = pd.factorize(gk[sk])[0]
    vals = gk[col].values
    par = gk["par_" + key].values
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
    return {
        "n_games": len(gk),
        "n_spells": len(a),
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
        "between_var_perm_p": float((np.sum(null_bv >= obs_bv) + 1) / (N_PERM + 1)),
        "excess_var_share": float(max(obs_bv - null_bv.mean(), 0) / obs_bv),
        "r_perm_p": float((np.sum(null_r >= r) + 1) / (N_PERM + 1)),
    }


def dummies(frame, key):
    return pd.get_dummies(frame["sp_" + key], dtype=float).values


def basis(X):
    q, r = np.linalg.qr(X)
    keep = np.abs(np.diag(r)) > 1e-9 * np.abs(r).max()
    return q[:, keep]


def incremental(frame, col, key_new, key_base):
    y = frame[col].values
    y = y - y.mean()
    ones = np.ones((len(frame), 1))
    Qb = basis(np.hstack([ones, dummies(frame, key_base)]))
    Qf = basis(np.hstack([ones, dummies(frame, key_base), dummies(frame, key_new)]))
    df = Qf.shape[1] - Qb.shape[1]

    def stat(v):
        return float(((Qf.T @ v) ** 2).sum() - ((Qb.T @ v) ** 2).sum())

    obs = stat(y)
    s2 = float(((y - Qf @ (Qf.T @ y)) ** 2).sum() / (len(y) - Qf.shape[1]))
    nul = np.array([stat(y * rng.choice([-1.0, 1.0], len(y))) for _ in range(N_PERM)])
    n = len(y)
    return {
        "df": int(df),
        "inc_ss": obs,
        "adj_inc_var": float((obs - df * s2) / n),
        "null_adj_inc_var_mean": float((nul.mean() - df * s2) / n),
        "perm_p": float((np.sum(nul >= obs) + 1) / (N_PERM + 1)),
        "resid_var": s2,
    }


def keepmask(frame, key):
    sp = frame.groupby("sp_" + key)["sp_" + key].transform("size")
    return frame["sp_" + key].notna() & (sp >= MIN_SPELL_GAMES)


res = {"games": len(g), "coord_file": cdir.replace("\\", "/")}
res["coverage"] = {k: int(g["sp_" + k].notna().sum()) for k in ["hc", "oc", "dc", "qb"]}
res["spells_kept_own"] = {
    k: int(g.loc[keepmask(g, k), "sp_" + k].nunique()) for k in ["hc", "oc", "dc", "qb"]
}
res["reliability_own"] = {}
unit_pieces = []
for col, keys in [("off_res", ["hc", "oc", "qb"]), ("def_res", ["hc", "dc"])]:
    res["reliability_own"][col] = {k: reliab(g, k, col, unit_pieces) for k in keys}
emit("sim17_unit2", unit_pieces)
res["reliability_common"] = {}
res["decomposition"] = {}
for col, keys in [("off_res", ["hc", "oc", "qb"]), ("def_res", ["hc", "dc"])]:
    msk = np.ones(len(g), bool)
    for k in keys:
        msk &= keepmask(g, k).values
    cm = g[msk].copy()
    res["reliability_common"][col] = {"games": len(cm), **{k: reliab(cm, k, col) for k in keys}}
    dec = {}
    for a in keys:
        for b in keys:
            if a != b:
                dec[f"{a}_given_{b}"] = incremental(cm, col, a, b)
        dec[f"{a}_alone"] = incremental(cm.assign(sp_none="x"), col, a, "none")
    res["decomposition"][col] = dec
res["looks"] = {
    "reliability_own": 5 * 3,
    "reliability_common": 5 * 3,
    "decomposition": 9 + 4,
}
res["looks"]["total"] = sum(res["looks"].values())
os.makedirs("artifacts/sim17", exist_ok=True)
with open("artifacts/sim17/unit2_carriers.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
