import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="tests/scratch/mod33")
ap.add_argument("--boots", type=int, default=500)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = "F:/Repos/nfl_py3/"
GRPS = ["OL", "QB", "WR1", "DB1"]
SEASONS = list(range(2020, 2026))
rng = np.random.default_rng(33)

hz = pd.read_parquet(R + "tests/scratch/sim12/hz.parquet")
hz["wk"] = hz.game.str.split("_").str[1].astype(int)
ex = hz.groupby(["season", "team", "wk", "grp"]).exit.sum().unstack(fill_value=0).reset_index()
ex = ex.sort_values(["season", "team", "wk"])
for g in GRPS:
    ex["prev_" + g] = ex.groupby(["season", "team"])[g].shift(1).fillna(0)

pg = pd.read_parquet(R + "artifacts/opener_evaluation/20261101T151614Z/per_game.parquet")
pg["home"] = pg.game_id.str.split("_").str[3]
pg["away"] = pg.game_id.str.split("_").str[2]
for side in ("home", "away"):
    m = ex.rename(columns={"team": side, "wk": "week"})[
        ["season", side, "week"] + ["prev_" + g for g in GRPS]
    ]
    m = m.rename(columns={"prev_" + g: f"{side}_prev_{g}" for g in GRPS})
    pg = pg.merge(m, on=["season", side, "week"], how="left")
for g in GRPS:
    pg["d_" + g] = pg[f"home_prev_{g}"].fillna(0) - pg[f"away_prev_{g}"].fillna(0)
pg = pg[(pg.margin_vs_open != 0) & pg.home_cover_probability_excluding_push_at_open.notna()].copy()
pg = pg.reset_index(drop=True)
pg["y"] = (pg.margin_vs_open > 0).astype(float)
pb = pg.home_cover_probability_excluding_push_at_open.clip(1e-4, 1 - 1e-4)
pg["lb"] = np.log(pb / (1 - pb))


def fit(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b, np.linalg.inv(H)


def pr(X, b):
    return 1 / (1 + np.exp(-X @ b))


def mats(df, cols):
    return np.column_stack([np.ones(len(df)), df.lb.values] + [df[c].values for c in cols])


def loso(df, cols):
    out = np.zeros(len(df))
    coefs = []
    for s in SEASONS:
        tr, te = (df.season != s).values, (df.season == s).values
        b, _ = fit(mats(df[tr], cols), df.y.values[tr])
        out[te] = pr(mats(df[te], cols), b)
        coefs.append(b)
    return out, np.array(coefs)


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (
        float(((p > 0.5) == (y > 0.5)).mean()),
        float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        float(((p - y) ** 2).mean()),
    )


def insample(df, cols):
    b, cov = fit(mats(df, cols), df.y.values)
    return metrics(pr(mats(df, cols), b), df.y.values), b, np.sqrt(np.diag(cov))


y = pg.y.values
p0, c0 = loso(pg, [])
p_base = pg.home_cover_probability_excluding_push_at_open.values
rows = [("served_base_raw", *metrics(p_base, y), np.nan, np.nan, np.nan, len(pg))]
m0 = metrics(p0, y)
i0 = insample(pg, [])[0]
rows.append(("intercept_only_control", *m0, i0[1], m0[1] - i0[1], np.nan, len(pg)))
fams = {g: [f"d_{g}"] for g in GRPS}
fams["all4"] = [f"d_{g}" for g in GRPS]
cells = []
ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
for name, cols in fams.items():
    p1, c1 = loso(pg, cols)
    m1 = metrics(p1, y)
    ims, b, se = insample(pg, cols)
    dec = (p1 > 0.5) != (p0 > 0.5)
    dw = int(((p1 > 0.5) == (y > 0.5))[dec].sum())
    dw0 = int(((p0 > 0.5) == (y > 0.5))[dec].sum())
    n = len(y)
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    d = ll0 - ll1
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(args.boots)])
    z = b[2:] / se[2:]
    cells.append(
        {
            "name": name,
            "coef_insample": b[2:].tolist(),
            "se": se[2:].tolist(),
            "p_coef_positive": [0.5 * (1 + math.erf(v / 2**0.5)) for v in z],
            "fold_coefs": c1[:, 2:].round(4).tolist(),
            "oos": m1,
            "ins": ims,
            "gap_logloss": m1[1] - ims[1],
            "ll_improve_vs_control": float(d.mean()),
            "ll_improve_ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "p_improve_positive": float((bs > 0).mean()),
            "decisive_n": int(dec.sum()),
            "decisive_term_wins": dw,
            "decisive_control_wins": dw0,
            "nonzero_games": int((pg[cols] != 0).any(axis=1).sum()),
        }
    )
    rows.append((name + "_term", *m1, ims[1], m1[1] - ims[1], int(dec.sum()), len(pg)))
tab = pd.DataFrame(
    rows, columns=["arm", "acc", "logloss", "brier", "ins_logloss", "gap", "decisive", "n"]
)
print(tab.round(4).to_string())
for c in cells:
    print(json.dumps(c, default=float))

av = pd.read_parquet(R + "data/processed/game_features_player_learned_availability.parquet")
acols = [
    c
    for c in av.columns
    if c.startswith("diff_")
    and any(k in c for k in ("injury", "qb_start_probability", "lineup_continuity", "roster_cont"))
]
j = pg.merge(av[["game_id", *acols]], on="game_id", how="inner")
cdf = pd.DataFrame({g: {a: j[f"d_{g}"].corr(j[a], method="spearman") for a in acols} for g in GRPS})
print(len(j), "games with availability features")
print(cdf.round(3).to_string())
icols = [a for a in acols if "injury" in a]
X = np.column_stack([np.ones(len(j)), j.lb.values] + [j[a].fillna(0).values for a in icols])
b, _ = fit(X, j.y.values)
j["rj"] = j.y - pr(X, b)
print(
    "corr of term with residual of base+served-injury logit",
    {g: round(float(j[f"d_{g}"].corr(j.rj)), 4) for g in GRPS},
)

inj = pd.read_parquet(sorted(glob.glob(R + "data/players/raw/*/injuries.parquet"))[-1])
inj = inj[inj.game_type == "REG"]
st = (
    inj.groupby(["season", "team", "week", "gsis_id"])
    .report_status.agg(lambda s: set(s.dropna()))
    .reset_index()
)
st["des_out"] = st.report_status.apply(lambda s: bool(s & {"Out", "Doubtful"}))
st["des_any"] = st.report_status.apply(lambda s: bool(s & {"Out", "Doubtful", "Questionable"}))
rn = pd.read_parquet(
    sorted(glob.glob(R + "data/players/raw/*/weekly_rosters.parquet"))[-1],
    columns=["season", "team", "week", "gsis_id", "full_name"],
)
rn = rn.dropna().drop_duplicates(["season", "team", "gsis_id"])[
    ["season", "team", "gsis_id", "full_name"]
]
sc = pd.read_parquet(sorted(glob.glob(R + "data/players/raw/*/snap_counts.parquet"))[-1])
sc = sc[sc.game_type == "REG"]
sc["snaps"] = sc.offense_snaps.fillna(0) + sc.defense_snaps.fillna(0)
sc = (
    sc.groupby(["season", "week", "team", "player"])
    .snaps.sum()
    .reset_index()
    .rename(columns={"player": "full_name"})
)
h = hz.merge(
    rn, left_on=["season", "team", "player"], right_on=["season", "team", "gsis_id"], how="left"
)
nxt = ex[["season", "team", "wk"]].copy()
nxt["next_wk"] = nxt.groupby(["season", "team"]).wk.shift(-1)
h = h.merge(nxt, on=["season", "team", "wk"])
h = h[h.next_wk.notna()]
h = h.merge(
    st.rename(columns={"week": "next_wk", "gsis_id": "player"})[
        ["season", "team", "next_wk", "player", "des_out", "des_any"]
    ],
    on=["season", "team", "next_wk", "player"],
    how="left",
)
h["des_out"] = h.des_out.fillna(False).astype(bool)
h["des_any"] = h.des_any.fillna(False).astype(bool)
h = h.merge(
    sc.rename(columns={"week": "next_wk"}),
    on=["season", "team", "next_wk", "full_name"],
    how="left",
)
h["name_matched"] = h.full_name.notna()
h["absent"] = h.snaps.fillna(0) == 0
h["absent_undes"] = h.absent & ~h.des_any
t = (
    h.groupby(["grp", "exit"])
    .agg(
        n=("player", "size"),
        des_out=("des_out", "mean"),
        des_any=("des_any", "mean"),
        name_matched=("name_matched", "mean"),
        absent=("absent", "mean"),
        absent_undes=("absent_undes", "mean"),
    )
    .reset_index()
)
print(t.round(3).to_string())
t.to_csv(args.out + "/timing.csv", index=False)
with open(args.out + "/cells.json", "w") as fh:
    json.dump(cells, fh, default=float)
q = h[(h.grp == "QB") & (h.exit > 0)].copy()
q["k_des"] = q.des_any.astype(float)
q["k_und"] = (~q.des_any).astype(float)
qk = q.groupby(["season", "team", "next_wk"])[["k_des", "k_und"]].max().reset_index()
qk = qk.rename(columns={"next_wk": "week"})
for side in ("home", "away"):
    m = qk.rename(columns={"team": side, "k_des": f"{side}_des", "k_und": f"{side}_und"})
    pg = pg.merge(m, on=["season", side, "week"], how="left")
pg["d_QBdes"] = pg.home_des.fillna(0) - pg.away_des.fillna(0)
pg["d_QBund"] = pg.home_und.fillna(0) - pg.away_und.fillna(0)
sets = {
    "QB_split": ["d_QBdes", "d_QBund"],
    "QB_undes_only": ["d_QBund"],
    "QB_des_only": ["d_QBdes"],
}
ucells = []
for name, cols in sets.items():
    p1, c1 = loso(pg, cols)
    m1 = metrics(p1, y)
    ims, b, se = insample(pg, cols)
    dec = (p1 > 0.5) != (p0 > 0.5)
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    d = ll0 - ll1
    bs = np.array([d[rng.integers(0, len(y), len(y))].mean() for _ in range(args.boots)])
    cell = {
        "name": name,
        "coef": b[2:].tolist(),
        "se": se[2:].tolist(),
        "p_coef_positive": [0.5 * (1 + math.erf(v / 2**0.5)) for v in b[2:] / se[2:]],
        "fold_coefs": c1[:, 2:].round(3).tolist(),
        "oos": m1,
        "ins": ims,
        "ll_improve": float(d.mean()),
        "p_improve_positive": float((bs > 0).mean()),
        "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "decisive": [
            int(dec.sum()),
            int(((p1 > 0.5) == (y > 0.5))[dec].sum()),
            int(((p0 > 0.5) == (y > 0.5))[dec].sum()),
        ],
        "nonzero": {c: int((pg[c] != 0).sum()) for c in cols},
    }
    ucells.append(cell)
    print(json.dumps(cell, default=float))
with open(args.out + "/cells_unit2.json", "w") as fh:
    json.dump(ucells, fh, default=float)
