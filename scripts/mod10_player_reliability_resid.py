from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_ratings_on_production as base

from nfl_ats.participation import PARTICIPATION_RATING_EPA_CLIP, PARTICIPATION_RATING_RIDGE_ALPHA

ALPHAS = tuple(PARTICIPATION_RATING_RIDGE_ALPHA * 10.0**e for e in np.arange(-2.0, 3.01, 0.5))
SPLITS = {
    "S1": ((2016, 2018), (2017, 2019)),
    "S2": ((2020, 2022), (2021, 2023)),
    "S3": ((2024,), (2025,)),
}
OUT = Path("F:/Repos/nfl_py3/artifacts/mod10_graph_reliability/resid.json")
BOOT = 2000
SEED = 20261103
GATE = 0.436
MIN_PLAYS = base.RELIABILITY_MIN_PLAYS


def sb(x):
    return 2 * x / (1 + x)


def player_matrix(table):
    vec = DictVectorizer(sparse=True, sort=True)
    gen = (
        {
            **{f"offense_player::{p}": 1.0 for p in base._player_ids(r.offense_players)},
            **{f"defense_player::{p}": -1.0 for p in base._player_ids(r.defense_players)},
        }
        for r in table.itertuples(index=False)
    )
    X = vec.fit_transform(gen).tocsr()
    return X, vec.get_feature_names_out()


def ts_matrix(table):
    o = table["posteam"].astype(str) + table["season"].astype(int).astype(str)
    d = table["defteam"].astype(str) + table["season"].astype(int).astype(str)
    oc, ou = pd.factorize(o)
    dc, du = pd.factorize(d)
    n = len(table)
    ar = np.arange(n)
    A = sparse.csr_matrix((np.ones(n), (ar, oc)), shape=(n, len(ou)))
    B = sparse.csr_matrix((np.ones(n), (ar, dc)), shape=(n, len(du)))
    return sparse.hstack([A, B]).tocsr()


def residual(T, y, mask):
    Tm = T[mask]
    sol = lsqr(Tm, y[mask], atol=1e-8, btol=1e-8, iter_lim=500)[0]
    return y[mask] - Tm @ sol


def fit(X, T, y, mask, alpha):
    r = residual(T, y, mask)
    est = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, tol=1e-6).fit(X[mask], r)
    cnt = np.asarray(abs(X[mask]).sum(axis=0)).ravel()
    return est.coef_, cnt


def boot(l, r, rng):  # noqa: E741
    n = len(l)
    d = np.empty(BOOT)
    for i in range(BOOT):
        s = rng.integers(0, n, n)
        d[i] = np.corrcoef(l[s], r[s])[0, 1]
    return d


def summarize(l, r, rng):  # noqa: E741
    p = float(np.corrcoef(l, r)[0, 1])
    d = boot(l, r, rng)
    s = sb(d)
    return {
        "n": len(l),
        "pearson": p,
        "sb": sb(p),
        "sb_lo": float(np.percentile(s, 2.5)),
        "sb_hi": float(np.percentile(s, 97.5)),
        "p_pos": float(np.mean(d > 0)),
        "p_gate": float(np.mean(s > GATE)),
        "draws": BOOT,
    }


def main():
    base.PBP_SNAPSHOT = base.REPO_ROOT / "data" / "pbp" / "raw" / "20261027T220410Z"
    table = base._play_table().reset_index(drop=True)
    season = table["season"].astype(int)
    y = (
        pd.to_numeric(table["epa"])
        .clip(-PARTICIPATION_RATING_EPA_CLIP, PARTICIPATION_RATING_EPA_CLIP)
        .to_numpy(float)
    )
    cov = {}
    for s in sorted(season.unique()):
        m = (season == s).to_numpy()
        has = table.loc[m, "offense_players"].map(lambda v: len(base._player_ids(v))).to_numpy()
        cov[int(s)] = {
            "plays": int(m.sum()),
            "with_players": int((has > 0).sum()),
            "mean_offense_players": float(has.mean()),
        }
        print("coverage", s, cov[int(s)], flush=True)
    X, names = player_matrix(table)
    T = ts_matrix(table)
    chans = {
        ch: np.array([n.startswith(f"{ch}_player::") for n in names])
        for ch in ("offense", "defense")
    }
    seasons_all = sorted({s for v in SPLITS.values() for h in v for s in h})
    ps = {}
    for s in seasons_all:
        m = (season == s).to_numpy()
        for a in ALPHAS:
            ps[(s, a)] = fit(X, T, y, m, a)
        print("season done", s, flush=True)

    def half(seasons, a, per_season):
        if per_season:
            num = np.zeros(X.shape[1])
            den = np.zeros(X.shape[1])
            for s in seasons:
                c, n = ps[(s, a)]
                num += c * n
                den += n
            return num / np.maximum(den, 1e-9), den
        m = season.isin(seasons).to_numpy()
        return fit(X, T, y, m, a)

    rng = np.random.default_rng(SEED)
    cells = {}
    for dname, per in (("pooled_resid", False), ("perseason_resid", True)):
        for sp, (ha, hb) in SPLITS.items():
            for a in ALPHAS:
                ca, na = half(ha, a, per)
                cb, nb = half(hb, a, per)
                for ch, cm in chans.items():
                    idx = np.where(cm & (na >= MIN_PLAYS) & (nb >= MIN_PLAYS))[0]
                    cells[f"{dname}|{sp}|{a}|{ch}"] = (
                        summarize(ca[idx], cb[idx], rng) if len(idx) > 20 else None
                    )
            print("cells", dname, sp, flush=True)
    lopo = {}
    for dname in ("pooled_resid", "perseason_resid"):
        for ev in SPLITS:
            others = [s for s in SPLITS if s != ev]

            def score(split, a):
                v = [cells[f"{dname}|{split}|{a}|{ch}"] for ch in ("offense", "defense")]  # noqa: B023
                return np.nan if any(x is None for x in v) else np.mean([x["pearson"] for x in v])

            a_sel = max(
                ALPHAS,
                key=lambda a: (
                    np.nanmean([score(o, a) for o in others])
                    if not np.all(np.isnan([score(o, a) for o in others]))
                    else -9
                ),
            )
            a_in = max(ALPHAS, key=lambda a: -9 if np.isnan(score(ev, a)) else score(ev, a))
            rec = {
                "selected_on": others,
                "alpha_selected": a_sel,
                "alpha_in_sample_best": a_in,
                "alpha_index": [i for i, a in enumerate(ALPHAS) if a == a_sel][0],  # noqa: RUF015
                "grid_size": len(ALPHAS),
            }
            for ch in ("offense", "defense"):
                oos = cells[f"{dname}|{ev}|{a_sel}|{ch}"]
                ins = cells[f"{dname}|{ev}|{a_in}|{ch}"]
                rec[ch] = (
                    None
                    if oos is None
                    else {"oos": oos, "in_sample_sb": ins["sb"], "gap": ins["sb"] - oos["sb"]}
                )
            lopo[f"{dname}|{ev}"] = rec
            print(dname, ev, json.dumps(rec), flush=True)
    OUT.write_text(
        json.dumps(
            {
                "alphas": list(ALPHAS),
                "splits": {k: [list(v[0]), list(v[1])] for k, v in SPLITS.items()},
                "coverage": cov,
                "looks": 2 * len(ALPHAS) * len(SPLITS),
                "lopo": lopo,
                "cells": cells,
            },
            indent=1,
            default=float,
        )
    )


main()
