import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

o = pd.read_parquet("artifacts/opener_evaluation/20260929T192743Z/per_game.parquet")
n = pd.read_parquet("data/processed/game_features_ngs.parquet")
c1 = "diff_ngs_passing__completion_percentage_above_expectation"
c2 = "diff_ngs_rushing__rush_yards_over_expected_per_att"
d = o.merge(n[["game_id", c1, c2]], on="game_id")
d = d[d.season.between(2020, 2025) & d.margin_vs_open.ne(0)].copy()
d["y"] = (d.margin_vs_open > 0).astype(int)
d["z"] = logit(d.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6))
d["g1"] = -d[c1].fillna(0)
d["g2"] = -d[c2].fillna(0)
for c in ["g1", "g2"]:
    d[c] = d[c] / d[c].std()
d["gc"] = d.g1 + d.g2
arms = {
    "intercept": [],
    "served": ["z"],
    "served+cpoe_gap": ["z", "g1"],
    "served+ryoe_gap": ["z", "g2"],
    "served+both": ["z", "g1", "g2"],
    "served+sum": ["z", "gc"],
}
P = {}
coef = {}
for a, cols in arms.items():
    p = pd.Series(0.5, index=d.index)
    coef[a] = {}
    for s in range(2020, 2026):
        tr = d[d.season != s]
        te = d.season == s
        if not cols:
            p[te] = tr.y.mean()
            continue
        m = LogisticRegression(C=1e6, max_iter=1000).fit(tr[cols], tr.y)
        p[te] = m.predict_proba(d.loc[te, cols])[:, 1]
        coef[a][s] = np.round(m.coef_[0], 3).tolist()
    P[a] = p.clip(1e-6, 1 - 1e-6)


def ll(p):
    return -(d.y * np.log(p) + (1 - d.y) * np.log(1 - p))


rng = np.random.default_rng(1)
print("n", len(d))
for a in arms:
    w = ((P[a] > 0.5) == (d.y == 1)).sum()
    print(
        a,
        f"W-L {w}-{len(d) - w}",
        f"LL {ll(P[a]).mean():.5f}",
        f"Brier {((P[a] - d.y) ** 2).mean():.5f}",
    )
base = ll(P["served"])
for a in [*list(arms)[2:], "served"]:
    for ref, rn in (
        [(P["intercept"], "vs_intercept")] if a == "served" else [(P["served"], "vs_served")]
    ):
        diff = (ll(ref) if a == "served" else base) - ll(P[a])
        dv = diff.values
        b = np.array([dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(5000)])
        sb = diff.groupby(d["season"]).mean().values
        print(
            a,
            rn,
            "dLL {:.5f} se {:.5f} CI[{:.5f},{:.5f}] P+ {:.3f}".format(
                dv.mean(), b.std(), *np.percentile(b, [2.5, 97.5]), (b > 0).mean()
            ),
            "seasons+",
            (sb > 0).sum(),
            "/6",
        )
        sw = ((P[a] > 0.5) == (d.y == 1)).sum() - (
            (P["served" if a != "served" else "intercept"] > 0.5) == (d.y == 1)
        ).sum()
        print("  dWins", sw, "flips", ((P[a] > 0.5) != (P["served"] > 0.5)).sum())
for a in ["served+cpoe_gap", "served+ryoe_gap", "served+both"]:
    print(a, coef[a])
