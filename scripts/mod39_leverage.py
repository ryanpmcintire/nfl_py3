import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "artifacts" / "opener_evaluation" / "20261103T173420Z" / "per_game.parquet"
TW = ROOT / "artifacts" / "sim09" / "stakes_teamweek.csv"
FEAT = ROOT / "data" / "processed" / "game_features.parquet"
OUT = ROOT / "artifacts" / "mod39"
ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
DRAWS = 5000
RNG = np.random.default_rng(20261103)
COLS = ["lev", "elim", "clinch"]


def games():
    f = pd.read_parquet(
        FEAT,
        columns=[
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "result",
            "spread_line",
            "game_type",
        ],
    )
    f = f[(f.game_type == "REG") & f.week.between(10, 16) & (f.season <= 2025)].copy()
    for c in ["home_team", "away_team"]:
        f[c] = f[c].replace(ALIAS)
    t = pd.read_csv(TW)[["season", "week", "team", *COLS]]
    for s, k in [("h", "home_team"), ("a", "away_team")]:
        f = f.merge(
            t.rename(columns={"team": k, **{c: f"{c}_{s}" for c in COLS}}),
            on=["season", "week", k],
            how="inner",
        )
    f["levdiff"] = f.lev_h - f.lev_a
    f["elimdiff"] = f.elim_h - f.elim_a
    f["clinchdiff"] = f.clinch_h - f.clinch_a
    f["y_close"] = f.result - f.spread_line
    o = pd.read_parquet(OPEN)[
        [
            "game_id",
            "tue_open_home_spread",
            "margin_vs_open",
            "home_cover_probability_excluding_push_at_open",
        ]
    ]
    return f.merge(o, on="game_id", how="left")


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def boot_slope(y, x, extra=None):
    X = np.column_stack([np.ones(len(y)), x] + ([] if extra is None else [extra]))
    b = ols(y, X)[1]
    n = len(y)
    bs = np.empty(DRAWS)
    for i in range(DRAWS):
        ix = RNG.integers(0, n, n)
        bs[i] = ols(y[ix], X[ix])[1]
    return {
        "slope": float(b),
        "se": float(bs.std()),
        "lo": float(np.quantile(bs, 0.025)),
        "hi": float(np.quantile(bs, 0.975)),
        "p_pos": float((bs > 0).mean()),
        "n": int(n),
    }


def logit(p):
    return np.log(p / (1 - p))


def fit(off, X, y):
    def nll(b):
        z = off + X @ b
        return np.sum(np.logaddexp(0, z) - y * z) + 1e-6 * np.sum(b**2)

    return minimize(nll, np.zeros(X.shape[1]), method="BFGS").x


def loso(d, cols):
    d = d.reset_index(drop=True)
    off = logit(d.home_cover_probability_excluding_push_at_open.clip(0.01, 0.99).values)
    y = (d.margin_vs_open > 0).astype(float).values
    p_full = np.zeros(len(d))
    p_ctl = np.zeros(len(d))
    coefs = {}
    for s in sorted(d.season.unique()):
        tr = (d.season != s).values
        te = ~tr
        Xf = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        b = fit(off[tr], Xf[tr], y[tr])
        coefs[int(s)] = [float(v) for v in b]
        p_full[te] = 1 / (1 + np.exp(-(off[te] + Xf[te] @ b)))
        c = fit(off[tr], Xf[tr][:, :1], y[tr])
        p_ctl[te] = 1 / (1 + np.exp(-(off[te] + c[0])))
    return d, y, p_full, p_ctl, coefs


def score(y, p):
    return (
        -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        (p - y) ** 2,
        ((p > 0.5) == (y == 1)).astype(float),
    )


def loso_report(d, cols):
    d, y, pf, pc, coefs = loso(d, cols)
    lf, bf, af = score(y, pf)
    lc, bc, ac = score(y, pc)
    n = len(y)
    dd = {"logloss_gain": lc - lf, "brier_gain": bc - bf, "acc_gain": af - ac}
    out = {"n": int(n), "coefs_by_heldout_season": coefs}
    for k, v in dd.items():
        bs = np.array([v[RNG.integers(0, n, n)].mean() for _ in range(DRAWS)])
        out[k] = {
            "mean": float(v.mean()),
            "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)),
            "p_pos": float((bs > 0).mean()),
        }
    out["acc_full"] = float(af.mean())
    out["acc_ctl"] = float(ac.mean())
    out["logloss_full"] = float(lf.mean())
    out["logloss_ctl"] = float(lc.mean())
    dis = (pf > 0.5) != (pc > 0.5)
    out["decisive_games"] = int(dis.sum())
    out["decisive_full_wins"] = int(af[dis].sum())
    out["decisive_ctl_wins"] = int(ac[dis].sum())
    ins = d.copy()
    off = logit(ins.home_cover_probability_excluding_push_at_open.clip(0.01, 0.99).values)
    X = np.column_stack([np.ones(n)] + [ins[c].values for c in cols])
    b = fit(off, X, y)
    pin = 1 / (1 + np.exp(-(off + X @ b)))
    c0 = fit(off, X[:, :1], y)
    pin0 = 1 / (1 + np.exp(-(off + c0[0])))
    out["insample_logloss_gain"] = float(score(y, pin0)[0].mean() - score(y, pin)[0].mean())
    out["insample_coefs"] = [float(v) for v in b]
    out["gap_logloss"] = out["insample_logloss_gain"] - out["logloss_gain"]["mean"]
    return out


def main():
    g = games()
    OUT.mkdir(exist_ok=True)
    res = {"n_games_wk10_16_2009_25": len(g), "looks": 7, "family": "mod39_playoff_leverage"}
    op = g[g.season >= 2020].dropna(
        subset=["margin_vs_open", "home_cover_probability_excluding_push_at_open"]
    )
    res["levdiff_sd"] = float(g.levdiff.std())
    res["opener_slope_2020_25"] = boot_slope(op.margin_vs_open.values, op.levdiff.values)
    res["opener_slope_ctl_open_spread"] = boot_slope(
        op.margin_vs_open.values, op.levdiff.values, op.tue_open_home_spread.values
    )
    cl = g[g.season <= 2019]
    res["close_slope_2009_19"] = boot_slope(cl.y_close.values, cl.levdiff.values)
    cl2 = g[g.season >= 2020]
    res["close_slope_2020_25"] = boot_slope(cl2.y_close.values, cl2.levdiff.values)
    halves = {}
    for name, ss in [
        ("odd_seasons", [s for s in range(2009, 2026) if s % 2]),
        ("even_seasons", [s for s in range(2009, 2026) if s % 2 == 0]),
    ]:
        h = g[g.season.isin(ss)]
        halves[name] = float(
            ols(h.y_close.values, np.column_stack([np.ones(len(h)), h.levdiff.values]))[1]
        )
    res["split_half_close_slopes"] = halves
    res["loso_levdiff_opener_2020_25"] = loso_report(op, ["levdiff"])
    res["overlap"] = {
        "corr_levdiff_elimdiff": float(op.levdiff.corr(op.elimdiff)),
        "corr_levdiff_clinchdiff": float(op.levdiff.corr(op.clinchdiff)),
        "n_elim_or_clinch_games": int(
            ((op.elim_h + op.elim_a + op.clinch_h + op.clinch_a) > 0).sum()
        ),
    }
    res["loso_stakes_only"] = loso_report(op, ["elimdiff", "clinchdiff"])
    res["loso_levdiff_plus_stakes"] = loso_report(op, ["levdiff", "elimdiff", "clinchdiff"])
    (OUT / "mod39_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("loso")}, indent=1))
    for k in [k for k in res if k.startswith("loso")]:
        r = res[k]
        print(k, {a: r[a] for a in r if a not in ("coefs_by_heldout_season", "insample_coefs")})


main()
