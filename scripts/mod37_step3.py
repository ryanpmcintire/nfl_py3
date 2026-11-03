import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PBP = ROOT / "data" / "pbp" / "raw" / "20261103T172043Z"
SCHED = ROOT / "data" / "raw" / "20261103T172043Z" / "schedules.parquet"
G = ROOT / "artifacts" / "mod37" / "games.parquet"
PER_GAME = ROOT / "artifacts" / "opener_evaluation" / "20261101T151614Z" / "per_game.parquet"
OUT = ROOT / "artifacts" / "mod37"
SEED = 20261103
DRAWS = 5000
FIRST = 2011
MIN_PRIOR_GAMES = 8


def epa_table():
    rows = []
    for s in range(2009, 2026):
        d = pd.read_parquet(
            PBP / f"season={s}" / "plays.parquet",
            columns=[
                "game_id",
                "season_type",
                "home_team",
                "away_team",
                "posteam",
                "play_type",
                "epa",
            ],
        )
        d = d.loc[
            (d["season_type"] == "REG")
            & d["play_type"].isin(["pass", "run", "no_play"])
            & d["epa"].notna()
        ]
        a = (
            d.loc[d["posteam"] == d["away_team"]]
            .groupby("game_id")["epa"]
            .mean()
            .rename("away_off")
        )
        h = (
            d.loc[d["posteam"] == d["home_team"]]
            .groupby("game_id")["epa"]
            .mean()
            .rename("home_off")
        )
        rows.append(pd.concat([a, h], axis=1))
    return pd.concat(rows)


def asof_premium(g):
    out = pd.Series(0.0, index=g.index)
    for s in sorted(g["season"].unique()):
        prior = g.loc[g["season"] < s]
        cur = g["season"] == s
        if prior.empty:
            continue
        v = prior.groupby("stadium")["excess"].agg(["mean", "size"])
        within = float(prior["excess"].var(ddof=1))
        between = max(float(v["mean"].var(ddof=1)) - float((within / v["size"]).mean()), 0.0)
        if between <= 0:
            continue
        k = within / between
        shr = v["mean"] * v["size"] / (v["size"] + k)
        out.loc[cur] = g.loc[cur, "stadium"].map(shr).fillna(0.0).to_numpy()
    return out


def team_asof(g, e):
    long = []
    for side, other in (("away", "home"), ("home", "away")):
        t = g[["game_id", "season", "order", side]].rename(columns={side: "team"})
        t["off"] = e[f"{side}_off"].reindex(g["game_id"]).to_numpy()
        t["allow"] = e[f"{other}_off"].reindex(g["game_id"]).to_numpy()
        t["side"] = side
        long.append(t)
    L = pd.concat(long).sort_values(["team", "order"])
    for c in ("off", "allow"):
        L[c + "_asof"] = L.groupby("team")[c].transform(
            lambda x: x.shift(1).expanding(min_periods=MIN_PRIOR_GAMES).mean()
        )
    return L


def block_boot(vals, blocks, rng):
    df = pd.DataFrame({"v": vals, "b": blocks}).groupby("b")["v"].agg(["sum", "count"])
    s, c = df["sum"].to_numpy(), df["count"].to_numpy()
    idx = rng.integers(0, len(s), size=(DRAWS, len(s)))
    est = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {
        "estimate": float(s.sum() / c.sum()),
        "se": float(est.std(ddof=1)),
        "low": float(lo),
        "high": float(hi),
        "probability_positive": float((est > 0).mean()),
    }


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def step_epa(g, rng):
    e = epa_table()
    g = g.merge(pd.read_parquet(SCHED, columns=["game_id", "gameday"]), on="game_id", how="left")
    g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    g["order"] = np.arange(len(g))
    g["prem"] = asof_premium(g)
    L = team_asof(g, e)
    ax = L.loc[L["side"] == "away"].set_index("game_id")
    hx = L.loc[L["side"] == "home"].set_index("game_id")
    g["y"] = e["away_off"].reindex(g["game_id"]).to_numpy()
    g["a_off"] = ax["off_asof"].reindex(g["game_id"]).to_numpy()
    g["h_allow"] = hx["allow_asof"].reindex(g["game_id"]).to_numpy()
    g["dome"] = g["roof"].isin(["dome", "closed"]).astype(float)
    d = g.loc[(g["season"] >= FIRST) & g[["y", "a_off", "h_allow"]].notna().all(axis=1)].copy()
    cols = ["a_off", "h_allow", "dome", "prem"]
    folds, gains = [], []
    for s in sorted(d["season"].unique()):
        tr, te = d.loc[d["season"] != s], d.loc[d["season"] == s]

        def one(f, c):
            return np.column_stack([np.ones(len(f)), f[c].to_numpy(float)])

        b1 = ols(one(tr, cols), tr["y"].to_numpy(float))
        b0 = ols(one(tr, cols[:-1]), tr["y"].to_numpy(float))
        yt = te["y"].to_numpy(float)
        e1 = (yt - one(te, cols) @ b1) ** 2
        e0 = (yt - one(te, cols[:-1]) @ b0) ** 2
        gains.append(pd.DataFrame({"s": s, "d": e0 - e1}))
        folds.append({"season": int(s), "coef_prem": round(float(b1[-1]), 4)})
    rg = pd.concat(gains)
    Xa = np.column_stack([np.ones(len(d)), d[cols].to_numpy(float)])
    ya = d["y"].to_numpy(float)
    beta = ols(Xa, ya)
    seasons = d["season"].to_numpy()
    uniq = np.unique(seasons)
    est = []
    for _ in range(DRAWS):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(seasons == p)[0] for p in pick])
        est.append(ols(Xa[idx], ya[idx])[-1])
    est = np.array(est)
    return {
        "n_games": len(d),
        "coef_prem_pooled": float(beta[-1]),
        "coef_season_boot_se": float(est.std(ddof=1)),
        "coef_interval": [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))],
        "probability_negative_coef": float((est < 0).mean()),
        "per_fold_coef": folds,
        "folds_negative": int(sum(f["coef_prem"] < 0 for f in folds)),
        "loso_sq_error_gain": block_boot(rg["d"].to_numpy(), rg["s"].to_numpy(), rng),
        "prem_sd": float(d["prem"].std()),
        "epa_effect_per_1sd": float(beta[-1] * d["prem"].std()),
    }, g


def nll(b, o, yy, zz, use):
    eta = o + b[0] + (b[1] * zz if use else 0)
    return float(np.sum(np.logaddexp(0, eta) - yy * eta))


def step_prob(g, rng):
    pg = pd.read_parquet(PER_GAME)
    m = pg.merge(g[["game_id", "prem"]], on="game_id", how="left")
    m = (
        m.loc[m["result"].ne(m["tue_open_home_spread"])]
        .dropna(subset=["home_cover_probability_excluding_push_at_open", "prem"])
        .copy()
    )
    y = (m["result"] > m["tue_open_home_spread"]).astype(float).to_numpy()
    p0 = m["home_cover_probability_excluding_push_at_open"].clip(1e-4, 1 - 1e-4).to_numpy()
    off = np.log(p0 / (1 - p0))
    z = m["prem"].to_numpy(float)

    def ll(e, yy):
        return np.logaddexp(0, e) - yy * e

    rows = []
    for s in sorted(m["season"].unique()):
        tr, te = (m["season"] != s).to_numpy(), (m["season"] == s).to_numpy()
        b1 = minimize(nll, np.zeros(2), args=(off[tr], y[tr], z[tr], True), method="BFGS").x
        b0 = minimize(nll, np.zeros(1 + 1), args=(off[tr], y[tr], z[tr], False), method="BFGS").x
        e1 = off[te] + b1[0] + b1[1] * z[te]
        e0 = off[te] + b0[0]
        rows.append(
            pd.DataFrame(
                {
                    "season": s,
                    "ll_base": ll(off[te], y[te]),
                    "ll_icpt": ll(e0, y[te]),
                    "ll_arm": ll(e1, y[te]),
                    "side_base": off[te] > 0,
                    "side_arm": e1 > 0,
                    "y": y[te],
                    "coef": b1[1],
                }
            )
        )
    r = pd.concat(rows)
    fl = r.loc[r["side_base"] != r["side_arm"]]
    return {
        "n_games": len(r),
        "decisive_flips": len(fl),
        "flip_arm_correct": int((fl["side_arm"].astype(float) == fl["y"]).sum()),
        "flip_base_correct": int((fl["side_base"].astype(float) == fl["y"]).sum()),
        "acc_base": float((r["side_base"].astype(float) == r["y"]).mean()),
        "acc_arm": float((r["side_arm"].astype(float) == r["y"]).mean()),
        "ll_base": float(r["ll_base"].mean()),
        "ll_icpt": float(r["ll_icpt"].mean()),
        "ll_arm": float(r["ll_arm"].mean()),
        "ll_gain_vs_base": block_boot(
            (r["ll_base"] - r["ll_arm"]).to_numpy(), r["season"].to_numpy(), rng
        ),
        "ll_gain_vs_icpt": block_boot(
            (r["ll_icpt"] - r["ll_arm"]).to_numpy(), r["season"].to_numpy(), rng
        ),
        "coef_by_fold": r.groupby("season")["coef"].first().round(4).to_dict(),
    }


def main():
    rng = np.random.default_rng(SEED)
    g = pd.read_parquet(G)
    epa, g = step_epa(g, rng)
    prob = step_prob(g, rng)
    res = {"epa": epa, "prob": prob}
    (OUT / "step3.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
