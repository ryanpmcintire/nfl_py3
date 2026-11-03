import glob

import numpy as np
import pandas as pd

d = sorted(glob.glob("data/raw/ngs/*/"))[-1]
p = pd.read_parquet(d + "ngs_passing.parquet")
r = pd.read_parquet(d + "ngs_rushing.parquet")
p = p[(p.season_type == "REG") & (p.week >= 1) & (p.season <= 2025)].dropna(
    subset=["completion_percentage_above_expectation"]
)
r = r[(r.season_type == "REG") & (r.week >= 1) & (r.season <= 2025)].dropna(
    subset=["rush_yards_over_expected"]
)
p["v"] = p.completion_percentage_above_expectation * p.attempts
pw = (
    p.groupby(["season", "team_abbr", "week"])
    .agg(v=("v", "sum"), n=("attempts", "sum"))
    .reset_index()
)
rw = (
    r.groupby(["season", "team_abbr", "week"])
    .agg(v=("rush_yards_over_expected", "sum"), n=("rush_attempts", "sum"))
    .reset_index()
)
rng = np.random.default_rng(0)


def half(w):
    w = w.assign(h=w.week % 2)
    g = w.groupby(["season", "team_abbr", "h"]).agg(v=("v", "sum"), n=("n", "sum")).reset_index()
    g["m"] = g.v / g.n
    a = g[g.h == 1].set_index(["season", "team_abbr"]).m
    b = g[g.h == 0].set_index(["season", "team_abbr"]).m
    j = pd.concat([a, b], axis=1, keys=["o", "e"]).dropna()
    return j


for name, w in [("cpoe", pw), ("ryoe", rw)]:
    j = half(w)
    j["o"] = j.o - j.groupby("season").o.transform("mean")
    j["e"] = j.e - j.groupby("season").e.transform("mean")
    rho = np.corrcoef(j.o, j.e)[0, 1]
    null = []
    for _ in range(5000):
        e = j.groupby("season").e.transform(lambda s: rng.permutation(s.values))
        null.append(np.corrcoef(j.o, e)[0, 1])
    null = np.array(null)
    sb = 2 * rho / (1 + rho)
    per = [np.corrcoef(g.o, g.e)[0, 1] for _, g in j.groupby("season")]
    print(
        name,
        len(j),
        f"r_half={rho:.3f} SB={sb:.3f} perm_p={(null >= rho).mean():.4f} null_sd={null.std():.3f}",
        "per-season",
        np.round(per, 2),
        "pos",
        sum(x > 0 for x in per),
    )
