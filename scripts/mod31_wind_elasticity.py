import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
PBP = ROOT / "data" / "pbp" / "raw" / "20260929T191306Z"
FC = (
    ROOT
    / "data"
    / "raw"
    / "forecast_archive"
    / "pool_decision_2009_2025_aware"
    / "forecasts.parquet"
)
OUT = ROOT / "artifacts" / "mod31"
FAMILY = "mod31_wind_elasticity"
OUTCOMES = ["pass_rate", "epa_play"]
NPERM = 5000
NRAND = 1000
SEED = 20261103
MIN_GAMES = 12


def panel():
    fc = pd.read_parquet(FC)
    fc = fc[(fc.fetch_status == "ok") & fc.roof.isin(["outdoors", "open"])][
        ["game_id", "actual_wind_mph", "forecast_wind_mph", "kickoff_utc"]
    ].drop_duplicates("game_id")
    cols = [
        "game_id",
        "season",
        "season_type",
        "posteam",
        "play_type",
        "epa",
        "qb_kneel",
        "qb_spike",
        "score_differential",
    ]
    d = pd.concat(
        [
            pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
            for s in range(2009, 2026)
        ],
        ignore_index=True,
    )
    d = d[
        (d.season_type == "REG")
        & d.posteam.notna()
        & d.play_type.isin(["pass", "run"])
        & (d.qb_kneel != 1)
        & (d.qb_spike != 1)
        & d.epa.notna()
    ]
    d = d.merge(fc, on="game_id", how="inner")
    d = d[d.actual_wind_mph.notna()]
    d["is_pass"] = (d.play_type == "pass").astype(float)
    g = d.groupby(["game_id", "posteam", "season"], as_index=False).agg(
        n=("is_pass", "size"),
        pass_rate=("is_pass", "mean"),
        epa_play=("epa", "mean"),
        script=("score_differential", "mean"),
        wind=("actual_wind_mph", "first"),
        fwind=("forecast_wind_mph", "first"),
        kick=("kickoff_utc", "first"),
    )
    g["wind10"] = g.wind / 10.0
    g["script10"] = g.script / 10.0
    return g.sort_values("kick").reset_index(drop=True)


def slope(df, y):
    if len(df) < MIN_GAMES:
        return np.nan, np.nan
    x = np.column_stack([np.ones(len(df)), df.wind10, df.script10])
    w = df.n.to_numpy(float)
    yy = df[y].to_numpy(float)
    xtw = x.T * w
    a = xtw @ x
    try:
        ai = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    b = ai @ xtw @ yy
    r = yy - x @ b
    s2 = float((w * r * r).sum() / (len(df) - 3))
    return float(b[1]), float(np.sqrt(s2 * ai[1, 1]))


def team_slopes(g, y):
    return {t: slope(df, y)[0] for t, df in g.groupby("posteam")}


def corr_perm(a, b, rng):
    k = sorted(set(a) & set(b))
    u = np.array([a[t] for t in k])
    v = np.array([b[t] for t in k])
    ok = ~(np.isnan(u) | np.isnan(v))
    u, v = u[ok], v[ok]
    r = float(np.corrcoef(u, v)[0, 1])
    null = np.array([np.corrcoef(u, rng.permutation(v))[0, 1] for _ in range(NPERM)])
    p = float((null >= r).mean())
    return r, 2 * r / (1 + r) if r > -1 else np.nan, p, len(u)


def reliability(g):
    rng = np.random.default_rng(SEED)
    rows = []
    for y in OUTCOMES:
        a = team_slopes(g[g.season % 2 == 0], y)
        b = team_slopes(g[g.season % 2 == 1], y)
        r, sb, p, n = corr_perm(a, b, rng)
        rows.append(
            {
                "outcome": y,
                "scheme": "odd_even_seasons",
                "r": r,
                "spearman_brown": sb,
                "perm_p": p,
                "teams": n,
            }
        )
        rs = []
        for _ in range(NRAND):
            flag = rng.random(len(g)) < 0.5
            ra = team_slopes(g[flag], y)
            rb = team_slopes(g[~flag], y)
            k = sorted(set(ra) & set(rb))
            u = np.array([ra[t] for t in k])
            v = np.array([rb[t] for t in k])
            ok = ~(np.isnan(u) | np.isnan(v))
            rs.append(np.corrcoef(u[ok], v[ok])[0, 1])
        rs = np.array(rs)
        rows.append(
            {
                "outcome": y,
                "scheme": "random_game_halves",
                "r": float(rs.mean()),
                "spearman_brown": float(2 * rs.mean() / (1 + rs.mean())),
                "r_q025": float(np.quantile(rs, 0.025)),
                "r_q975": float(np.quantile(rs, 0.975)),
                "prob_positive": float((rs > 0).mean()),
                "teams": len(ra),
            }
        )
    return rows


def asof_elasticity(g, y):
    out = np.full(len(g), np.nan)
    league = np.full(len(g), np.nan)
    seasons = sorted(g.season.unique())
    for s in seasons:
        prior = g[g.season < s]
        if prior.posteam.nunique() < 20:
            continue
        ps = {t: slope(df, y) for t, df in prior.groupby("posteam")}
        sl = np.array([v[0] for v in ps.values() if not np.isnan(v[0])])
        se2 = np.array([v[1] ** 2 for v in ps.values() if not np.isnan(v[0])])
        mu = float(np.average(sl, weights=1 / (se2 + 1e-12)))
        tau2 = max(float(sl.var(ddof=1) - se2.mean()), 0.0)
        cur = g.index[g.season == s]
        for i in cur:
            row = g.loc[i]
            hist = g[(g.posteam == row.posteam) & (g.kick < row.kick)]
            b, se = slope(hist, y)
            if np.isnan(b):
                out[i] = mu
            else:
                w = tau2 / (tau2 + se * se) if tau2 > 0 else 0.0
                out[i] = mu + w * (b - mu)
            league[i] = tau2
    return out, league


def main():
    OUT.mkdir(exist_ok=True)
    g = panel()
    print("team-games", len(g), "teams", g.posteam.nunique())
    if sys.argv[1] == "rel":
        rows = reliability(g)
        (OUT / "reliability.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(r)


main()
