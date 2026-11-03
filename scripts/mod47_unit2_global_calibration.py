from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
PICKPROB = ROOT / "artifacts/pick_probability/20261103T195411Z/per_game.parquet"
LIVE = ROOT / "artifacts/prospective_scorecard/20261103T192211Z/predictions.csv"
WEEK9 = ROOT / "artifacts/margin_predictions/2026-week-09-20261103T194941Z/pool_card.csv"
BOOT = 2000
EPS = 1e-6
BINS = 5
SEED = 47


def fit(x, y):
    return LogisticRegression(C=1e6, max_iter=2000).fit(x, y)


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def slope_boot(z, y, rng):
    m = fit(z[:, None], y)
    pt = (float(m.coef_[0, 0]), float(m.intercept_[0]))
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        mm = fit(z[i][:, None], y[i])
        bs.append((mm.coef_[0, 0], mm.intercept_[0]))
    bs = np.array(bs)
    return pt, np.percentile(bs, [2.5, 97.5], axis=0), (bs[:, 0] < 1).mean()


def paired(a, b, rng):
    d = a - b
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return d.mean(), np.percentile(bs, [2.5, 97.5]), (bs > 0).mean()


def reliability(p, y):
    t = pd.DataFrame({"p": p, "y": y})
    t["bin"] = pd.qcut(t.p, BINS, duplicates="drop")
    return (
        t.groupby("bin", observed=True)
        .agg(n=("y", "size"), pmean=("p", "mean"), obs=("y", "mean"))
        .round(4)
    )


def main():
    rng = np.random.default_rng(SEED)
    d = pd.read_parquet(PICKPROB)
    d = d[d.margin_vs_open.ne(0) & d.season.between(2020, 2025)].reset_index(drop=True)
    y = d.home_covered.to_numpy(dtype=int)
    cols = {
        "raw_model": d.model_probability.to_numpy(),
        "served_oos": d.out_of_season_home_probability.to_numpy(),
        "served_in_sample": d.in_sample_home_probability.to_numpy(),
    }
    print("n", len(d), "seasons", sorted(d.season.unique()))
    print(
        "max |raw - home_cover_probability_at_open|",
        float(np.abs(cols["raw_model"] - d.home_cover_probability_at_open).max()),
    )
    for name, p in cols.items():
        z = logit(np.clip(p, EPS, 1 - EPS))
        (b, a), ci, pl = slope_boot(z, y, rng)
        print(
            f"{name} slope {b:.3f} [{ci[0, 0]:.3f},{ci[1, 0]:.3f}] P(slope<1) {pl:.3f} "
            f"intercept {a:.3f} [{ci[0, 1]:.3f},{ci[1, 1]:.3f}] LL {ll(p, y).mean():.5f}"
        )
    print("coin/market-neutral LL", float(ll(d.neutral_market_probability.to_numpy(), y).mean()))
    print("reliability served_oos (home side)")
    print(reliability(cols["served_oos"], y).to_string())
    pick = np.where(cols["served_oos"] >= 0.5, cols["served_oos"], 1 - cols["served_oos"])
    ypick = np.where(cols["served_oos"] >= 0.5, y, 1 - y)
    print("reliability served_oos (pick side)")
    print(reliability(pick, ypick).to_string())
    zs = logit(np.clip(cols["served_oos"], EPS, 1 - EPS))
    shr = np.zeros(len(d))
    coefs = {}
    for s in sorted(d.season.unique()):
        te = (d.season == s).to_numpy()
        m = fit(zs[~te][:, None], y[~te])
        shr[te] = m.predict_proba(zs[te][:, None])[:, 1]
        coefs[int(s)] = (round(float(m.coef_[0, 0]), 3), round(float(m.intercept_[0]), 3))
    print("LOSO shrink coef per fold (slope, intercept)", coefs)
    base = ll(cols["served_oos"], y)
    new = ll(shr, y)
    print(f"LL served {base.mean():.5f} shrunk {new.mean():.5f} coin {np.log(2):.5f}")
    for nm, ref in (("served", base), ("coin", np.full(len(y), np.log(2)))):
        m, ci, pp = paired(ref, new, rng)
        print(f"dLL shrunk vs {nm} {m:.5f} CI[{ci[0]:.5f},{ci[1]:.5f}] P+ {pp:.3f}")
    m, ci, pp = paired(np.full(len(y), np.log(2)), base, rng)
    print(f"dLL served vs coin {m:.5f} CI[{ci[0]:.5f},{ci[1]:.5f}] P+ {pp:.3f}")
    m, ci, pp = paired(ll(cols["raw_model"], y), base, rng)
    print(f"dLL served vs raw model {m:.5f} CI[{ci[0]:.5f},{ci[1]:.5f}] P+ {pp:.3f}")
    print("side changes under shrink", int(((shr >= 0.5) != (cols["served_oos"] >= 0.5)).sum()))
    for s in sorted(d.season.unique()):
        k = (d.season == s).to_numpy()
        print(s, f"dLL {np.mean(base[k] - new[k]):.5f}")

    lv = pd.read_csv(LIVE)
    lv = lv[lv.y.isin([0, 1])]
    p = lv.p_served.to_numpy()
    yl = lv.y.to_numpy(dtype=int)
    (b, a), ci, pl = slope_boot(logit(np.clip(p, EPS, 1 - EPS)), yl, rng)
    print(
        f"LIVE 2026 n={len(lv)} slope {b:.3f} [{ci[0, 0]:.3f},{ci[1, 0]:.3f}] "
        f"intercept {a:.3f} [{ci[0, 1]:.3f},{ci[1, 1]:.3f}] LL {ll(p, yl).mean():.5f} "
        f"coin {np.log(2):.5f} mean p {p.mean():.4f} obs {yl.mean():.4f}"
    )
    print(reliability(p, yl).to_string())

    full = fit(zs[:, None], y)
    b_full = float(full.coef_[0, 0])
    w = pd.read_csv(WEEK9)
    pw = w.pick_probability.to_numpy()
    w["after"] = expit(b_full * logit(pw))
    print("full-sample shrink slope", round(b_full, 3))
    print(
        w[["away_team", "home_team", "pool_pick", "pick_probability", "after"]]
        .round(3)
        .head(8)
        .to_string()
    )


main()
