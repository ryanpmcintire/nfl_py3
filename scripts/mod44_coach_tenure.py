import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

OPENER = "artifacts/opener_evaluation/20260929T192743Z/per_game.parquet"
PROXY = "artifacts/mod24_u1/base_per_game.parquet"
SCHED = sorted(glob.glob("data/raw/*/schedules.parquet"))[-1]
BOOT = 5000
OUT = "artifacts/mod44_unit1"
STATES = ["first_year", "interim", "established"]


def team_games():
    s = pd.read_parquet(SCHED)
    s = s.dropna(subset=["home_coach", "away_coach"]).sort_values(["gameday", "game_id"])
    cols = ["game_id", "season", "week", "gameday", "team", "coach"]
    h = s[["game_id", "season", "week", "gameday", "home_team", "home_coach"]]
    h.columns = cols
    a = s[["game_id", "season", "week", "gameday", "away_team", "away_coach"]]
    a.columns = cols
    t = pd.concat([h, a]).sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    t["new"] = t.groupby("team")["coach"].transform(lambda c: c.ne(c.shift()))
    t["stint"] = t.groupby("team")["new"].cumsum()
    t["games_into"] = t.groupby(["team", "stint"]).cumcount()
    g = t.groupby(["team", "stint"])
    t["s_season"] = g["season"].transform("first")
    t["s_week"] = g["week"].transform("first")
    first_stint = t["stint"].eq(t.groupby("team")["stint"].transform("min"))
    same_season = t["season"].eq(t["s_season"])
    t["first_year"] = (~first_stint) & same_season & t["s_week"].le(1)
    t["interim"] = (~first_stint) & same_season & t["s_week"].gt(1)
    t["state"] = np.where(
        t["first_year"], "first_year", np.where(t["interim"], "interim", "established")
    )
    t["tenure_feat"] = np.where(first_stint, np.nan, -np.log1p(t["games_into"]))
    return t


def population():
    o = pd.read_parquet(OPENER)
    o = o[o.season.between(2020, 2025) & o.margin_vs_open.ne(0)].copy()
    o["y"] = (o.margin_vs_open > 0).astype(float)
    o["p"] = o["home_cover_probability_at_open"]
    cur = o[["game_id", "season", "week", "y", "p"]].assign(era="2020-25")
    b = pd.read_parquet(PROXY)
    b = b[b.season <= 2019].reset_index()
    b["y"] = np.where(b.pick_home, b.correct, 1 - b.correct)
    pro = b[["game_id", "season", "week", "y", "p"]].assign(era="2011-19")
    return pd.concat([pro, cur], ignore_index=True)


def attach(pop, t):
    cols = ["state", "tenure_feat", "games_into", "first_year", "interim"]
    parts = t[["game_id", "team", *cols]]
    h = parts.rename(columns={c: "h_" + c for c in [*cols, "team"]})
    a = parts.rename(columns={c: "a_" + c for c in [*cols, "team"]})
    d = pop.merge(h, on="game_id")
    d = d.merge(a, on="game_id")
    d = d[d.h_team.ne(d.a_team)]
    gid = d["game_id"].str.split("_")
    d = d[d.h_team.eq(gid.str[3]) & d.a_team.eq(gid.str[2])]
    return d.reset_index(drop=True)


def eb(m, v):
    w = 1 / v
    mu0 = (w * m).sum() / w.sum()
    q = (w * (m - mu0) ** 2).sum()
    tau2 = max(0.0, (q - (len(m) - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = (ws * m).sum() / ws.sum()
    return mu, np.sqrt(1 / ws.sum()), tau2


def cell_row(era, name, n, units, mu, se, tau2, raw):
    return {
        "era": era,
        "cell": name,
        "teamgames": n,
        "units": units,
        "resid_pp": 100 * mu,
        "se_pp": 100 * se,
        "lo_pp": 100 * (mu - 1.96 * se),
        "hi_pp": 100 * (mu + 1.96 * se),
        "p_pos": float(norm.cdf(mu / se)),
        "tau_pp": 100 * np.sqrt(tau2),
        "raw_mean_pp": 100 * raw,
    }


def residual_cells(d):
    rows = []
    for era, e in d.groupby("era"):
        long = []
        for side, sgn, team in (("h", 1, "h_team"), ("a", -1, "a_team")):
            x = e[["season", "y", "p", side + "_state", side + "_games_into"]].copy()
            x.columns = ["season", "y", "p", "state", "games_into"]
            x["r"] = sgn * (x["y"] - x["p"])
            x["unit"] = x["season"].astype(str) + "_" + e[team]
            long.append(x)
        L = pd.concat(long)
        for state in STATES:
            S = L[L.state == state]
            u = S.groupby("unit")["r"].agg(["mean", "count"])
            v = S["r"].var() / u["count"]
            mu, se, tau2 = eb(u["mean"].to_numpy(), v.to_numpy())
            rows.append(cell_row(era, state, len(S), len(u), mu, se, tau2, S["r"].mean()))
        S = L[L.state.isin(["first_year", "interim"])]
        X = S["games_into"].to_numpy(float)
        Y = S["r"].to_numpy()
        xc = X - X.mean()
        b = (xc * (Y - Y.mean())).sum() / (xc**2).sum()
        res = Y - Y.mean() - b * xc
        seb = np.sqrt((res**2).sum() / (len(Y) - 2) / (xc**2).sum())
        rows.append(cell_row(era, "games_slope_per_game", len(S), 0, b, seb, 0.0, b))
    return pd.DataFrame(rows)


def loso(e):
    e = e.copy()
    e["z"] = logit(e["p"].clip(1e-6, 1 - 1e-6))
    e["fy"] = e.h_first_year.astype(float) - e.a_first_year.astype(float)
    e["im"] = e.h_interim.astype(float) - e.a_interim.astype(float)
    e["tf"] = e.h_tenure_feat.fillna(0) - e.a_tenure_feat.fillna(0)
    arms = {
        "intercept": [],
        "served": ["z"],
        "served+first_year": ["z", "fy"],
        "served+interim": ["z", "im"],
        "served+both_states": ["z", "fy", "im"],
        "served+games_into": ["z", "tf"],
    }
    yv = e.y.to_numpy()

    def ll(p):
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        return -(yv * np.log(p) + (1 - yv) * np.log(1 - p))

    P, coef, insamp = {}, {}, {}
    for a, cols in arms.items():
        p = pd.Series(0.5, index=e.index)
        coef[a] = {}
        for s in sorted(e.season.unique()):
            tr, te = e[e.season != s], e.season == s
            if not cols:
                p[te] = tr.y.mean()
                continue
            m = LogisticRegression(C=1e6, max_iter=1000).fit(tr[cols], tr.y)
            p[te] = m.predict_proba(e.loc[te, cols])[:, 1]
            coef[a][int(s)] = np.round(m.coef_[0], 3).tolist()
        P[a] = p.clip(1e-6, 1 - 1e-6).to_numpy()
        if cols:
            m = LogisticRegression(C=1e6, max_iter=1000).fit(e[cols], e.y)
            insamp[a] = m.predict_proba(e[cols])[:, 1]
        else:
            insamp[a] = np.full(len(e), e.y.mean())

    rng = np.random.default_rng(44)
    base_pick = P["served"] > 0.5
    out = []
    for a in arms:
        pk = P[a] > 0.5
        flips = pk != base_pick
        won = pk == (yv == 1)
        dv = ll(P["served"]) - ll(P[a])
        b = np.array([dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(BOOT)])
        out.append(
            {
                "arm": a,
                "flips_won": int((flips & won).sum()),
                "flips_lost": int((flips & ~won).sum()),
                "wins": int(won.sum()),
                "n": len(e),
                "ll": ll(P[a]).mean(),
                "brier": float(((P[a] - yv) ** 2).mean()),
                "d_ll_vs_served": dv.mean(),
                "ci_lo": np.percentile(b, 2.5),
                "ci_hi": np.percentile(b, 97.5),
                "p_pos": float((b > 0).mean()),
                "ll_insample": ll(insamp[a]).mean(),
                "gap_oos_minus_in": ll(P[a]).mean() - ll(insamp[a]).mean(),
            }
        )
    return pd.DataFrame(out), coef


def main():
    d = attach(population(), team_games())
    os.makedirs(OUT, exist_ok=True)
    print("rows", d.groupby("era").size().to_dict())
    for era, e in d.groupby("era"):
        for s in ["first_year", "interim"]:
            print(era, s, "team-games", int((e.h_state == s).sum() + (e.a_state == s).sum()))
    cells = residual_cells(d)
    pd.set_option("display.width", 250)
    print(cells.round(3).to_string())
    cells.to_csv(f"{OUT}/cells.csv", index=False)
    res = {}
    for era, e in d.groupby("era"):
        tab, coef = loso(e)
        print(era)
        print(tab.round(5).to_string())
        for a in ["served+first_year", "served+interim", "served+both_states", "served+games_into"]:
            print(a, coef[a])
        res[era] = tab.to_dict("records")
    with open(f"{OUT}/loso.json", "w") as f:
        json.dump(res, f, default=float)


main()
