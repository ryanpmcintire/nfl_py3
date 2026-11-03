import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from xlg10_addone import schedule_features  # noqa: E402

OPENER = ROOT / "artifacts/opener_evaluation/20260929T192743Z/per_game.parquet"
SBR = ROOT / "artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet"
FEATURES = ROOT / "data/processed/game_features.parquet"
BOOT = 2000
EPS = 1e-6


def fit(x, y):
    return LogisticRegression(C=1e6, max_iter=2000).fit(x, y)


def load(era):
    f = schedule_features(
        pd.read_parquet(
            FEATURES, columns=["game_id", "season", "week", "gameday", "home_team", "away_team"]
        )
    )
    med = float(pd.concat([f.home_rest_days, f.away_rest_days]).dropna().median())
    f["min_rest"] = f[["home_rest_days", "away_rest_days"]].min(axis=1, skipna=False)
    f["short"] = (f.min_rest < med).astype(int)
    if era == "opener":
        o = pd.read_parquet(OPENER)
        d = o.rename(columns={"home_cover_probability_at_open": "p"})[
            ["game_id", "p", "margin_vs_open"]
        ].rename(columns={"margin_vs_open": "m"})
    else:
        o = pd.read_parquet(SBR)
        d = o.rename(columns={"home_cover_probability_at_open_proxy": "p"})[
            ["game_id", "p", "margin_vs_open_proxy"]
        ].rename(columns={"margin_vs_open_proxy": "m"})
    d = d.merge(f[["game_id", "season", "min_rest", "short"]], on="game_id")
    d = d[d.m.ne(0) & d.min_rest.notna()].copy()
    d["y"] = (d.m > 0).astype(int)
    d["z"] = logit(d.p.clip(EPS, 1 - EPS))
    return d.reset_index(drop=True), med


def slope_int(g, rng):
    m = fit(g[["z"]].to_numpy(), g.y.to_numpy())
    pt = (float(m.coef_[0, 0]), float(m.intercept_[0]))
    bs = []
    for _ in range(BOOT):
        b = g.iloc[rng.integers(0, len(g), len(g))]
        if b.y.nunique() < 2:
            continue
        mm = fit(b[["z"]].to_numpy(), b.y.to_numpy())
        bs.append((mm.coef_[0, 0], mm.intercept_[0]))
    bs = np.array(bs)
    return pt, np.percentile(bs, [2.5, 97.5], axis=0), (bs[:, 0] < 1).mean()


def loso(d, arms):
    seasons = sorted(d.season.unique())
    P = {a: pd.Series(0.5, index=d.index) for a in arms}
    coef = {a: {} for a in arms}
    for s in seasons:
        tr, te = d[d.season != s], d.season == s
        for a, cols in arms.items():
            if not cols:
                P[a][te] = tr.y.mean()
                continue
            m = fit(tr[cols].to_numpy(), tr.y.to_numpy())
            P[a][te] = m.predict_proba(d.loc[te, cols].to_numpy())[:, 1]
            coef[a][s] = np.round(m.coef_[0], 3).tolist()
    return {a: p.clip(EPS, 1 - EPS) for a, p in P.items()}, coef


def ll(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def report(d, era, rng):
    print(f"\n=== {era} n={len(d)} short={d.short.sum()} normal={(1 - d.short).sum()}")
    for name, g in (("short", d[d.short == 1]), ("normal", d[d.short == 0])):
        (b, a), ci, pl = slope_int(g, rng)
        w = ((g.p > 0.5) == (g.y == 1)).sum()
        print(
            f"{name} n={len(g)} W-L {w}-{len(g) - w} slope {b:.3f} "
            f"[{ci[0, 0]:.3f},{ci[1, 0]:.3f}] P(slope<1) {pl:.3f} "
            f"intercept {a:.3f} [{ci[0, 1]:.3f},{ci[1, 1]:.3f}]"
        )
        g = g.assign(bin=pd.qcut(g.p, 4, duplicates="drop"))
        print(
            g.groupby("bin", observed=True)
            .agg(n=("y", "size"), pmean=("p", "mean"), obs=("y", "mean"))
            .round(3)
            .to_string()
        )
    d["zs"] = d.z * d.short
    arms = {"intercept": [], "served": ["z"], "inter": ["z", "short", "zs"]}
    P, coef = loso(d, arms)
    for a in arms:
        w = ((P[a] > 0.5) == (d.y == 1)).sum()
        print(
            f"{a} W-L {w}-{len(d) - w} LL {ll(P[a], d.y).mean():.5f} "
            f"Brier {((P[a] - d.y) ** 2).mean():.5f}"
        )
    print("coef per fold [z,short,zs]", coef["inter"])
    flip = (P["inter"] > 0.5) != (P["served"] > 0.5)
    dec = d[flip]
    cw = ((P["inter"][flip] > 0.5) == (dec.y == 1)).sum()
    sw = ((P["served"][flip] > 0.5) == (dec.y == 1)).sum()
    print(f"decisive (side differs) n={flip.sum()} interaction wins {cw} served wins {sw}")
    for ref in ("served", "intercept"):
        diff = (ll(P[ref], d.y) - ll(P["inter"], d.y)).to_numpy()
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOT)])
        print(
            f"dLL inter vs {ref} {diff.mean():.5f} CI[{np.percentile(bs, 2.5):.5f},"
            f"{np.percentile(bs, 97.5):.5f}] P+ {(bs > 0).mean():.3f}"
        )
    for sub, mask in (("short", d.short == 1), ("normal", d.short == 0)):
        diff = (ll(P["served"], d.y) - ll(P["inter"], d.y))[mask].to_numpy()
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOT)])
        print(
            f"dLL inter vs served within {sub} {diff.mean():.5f} CI[{np.percentile(bs, 2.5):.5f},"
            f"{np.percentile(bs, 97.5):.5f}] P+ {(bs > 0).mean():.3f}"
        )
    insamp = fit(d[["z", "short", "zs"]].to_numpy(), d.y.to_numpy())
    pi = pd.Series(insamp.predict_proba(d[["z", "short", "zs"]].to_numpy())[:, 1])
    print(
        f"in-sample LL {ll(pi, d.y).mean():.5f} oos LL {ll(P['inter'], d.y).mean():.5f}"
        f" gap {ll(P['inter'], d.y).mean() - ll(pi, d.y).mean():.5f}"
    )


def main():
    rng = np.random.default_rng(47)
    for era in ("opener", "sbr"):
        d, med = load(era)
        d = d[d.season.between(2020, 2025)] if era == "opener" else d[d.season <= 2019]
        print("median team rest days", med)
        report(d, "2020-25 opener" if era == "opener" else "2011-19 SBR proxy", rng)


main()
