from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_ratings_on_production as base

from nfl_ats.participation import (
    PARTICIPATION_RATING_EPA_CLIP,
    PARTICIPATION_RATING_RIDGE_ALPHA,
    PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
)

ALPHAS = tuple(
    PARTICIPATION_RATING_RIDGE_ALPHA * m for m in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
)
SPLITS = {"S1": ((2016, 2018), (2017, 2019)), "S2": ((2022, 2024), (2023, 2025))}
OUT = Path("F:/Repos/nfl_py3/artifacts/mod10_graph_reliability/lopo_wide.json")
BOOT = 2000
SEED = 20261103
SCALE = PARTICIPATION_RATING_TEAM_FEATURE_SCALE


def rows(table, team_season):
    for r in table.itertuples(index=False):
        v = {}
        for p in base._player_ids(r.offense_players):
            v[f"offense_player::{p}"] = 1.0
        for p in base._player_ids(r.defense_players):
            v[f"defense_player::{p}"] = -1.0
        v[f"offense_team::{r.posteam}"] = SCALE
        v[f"defense_team::{r.defteam}"] = -SCALE
        if team_season:
            v[f"offense_ts::{r.posteam}{int(r.season)}"] = SCALE
            v[f"defense_ts::{r.defteam}{int(r.season)}"] = -SCALE
        yield v


def build(table, ts):
    vec = DictVectorizer(sparse=True, sort=True)
    X = vec.fit_transform(rows(table, ts)).tocsr()
    return X, vec.get_feature_names_out()


def fit(X, y, mask, alpha):
    est = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, tol=1e-6).fit(X[mask], y[mask])
    cnt = np.asarray(abs(X[mask]).sum(axis=0)).ravel()
    return est.coef_, cnt


def boot(l, r, rng):  # noqa: E741
    n = len(l)
    d = np.empty(BOOT)
    for i in range(BOOT):
        s = rng.integers(0, n, n)
        d[i] = np.corrcoef(l[s], r[s])[0, 1]
    return d


def sb(x):
    return 2 * x / (1 + x)


def summarize(l, r, rng):  # noqa: E741
    p = float(np.corrcoef(l, r)[0, 1])
    d = boot(l, r, rng)
    return {
        "n": len(l),
        "pearson": p,
        "sb": sb(p),
        "sb_lo": sb(float(np.percentile(d, 2.5))),
        "sb_hi": sb(float(np.percentile(d, 97.5))),
        "p_pos": float(np.mean(d > 0)),
        "draws": BOOT,
    }


def channel_idx(names, cnt_a, cnt_b):
    out = {}
    for ch in ("offense", "defense"):
        pre = f"{ch}_player::"
        out[ch] = [
            i
            for i, n in enumerate(names)
            if n.startswith(pre)
            and cnt_a[i] >= base.RELIABILITY_MIN_PLAYS
            and cnt_b[i] >= base.RELIABILITY_MIN_PLAYS
        ]
    return out


def half_coefs(design, X, y, season, seasons, alpha, per_season):
    if not per_season:
        m = season.isin(seasons).to_numpy()
        c, n = fit(X, y, m, alpha)
        return c, n
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for s in seasons:
        m = (season == s).to_numpy()
        c, n = fit(X, y, m, alpha)
        num += c * n
        den += n
    return num / np.maximum(den, 1e-9), den


def split_scores(designs, y, season, split):
    sa, sb_ = SPLITS[split]
    res = {}
    for dname, (X, names, per_season) in designs.items():  # noqa: B007
        res[dname] = {}
        for a in ALPHAS:
            ca, na = half_coefs(dname, X, y, season, sa, a, per_season)
            cb, nb = half_coefs(dname, X, y, season, sb_, a, per_season)
            res[dname][a] = (ca, cb, na, nb)
    return res


def main():
    base.PBP_SNAPSHOT = base.REPO_ROOT / "data" / "pbp" / "raw" / "20261027T220410Z"
    table = base._play_table().reset_index(drop=True)
    season = table["season"].astype(int)
    y = (
        pd.to_numeric(table["epa"])
        .clip(-PARTICIPATION_RATING_EPA_CLIP, PARTICIPATION_RATING_EPA_CLIP)
        .to_numpy(float)
    )
    Xp, names = build(table, False)
    Xt, names_t = build(table, True)
    designs = {
        "plain": (Xp, names, False),
        "teamseason": (Xt, names_t, False),
        "perseason_weighted": (Xp, names, True),
    }
    raw = {s: split_scores(designs, y, season, s) for s in SPLITS}
    repair = {}
    odd = season.isin(base.ODD_SEASONS).to_numpy()
    even = season.isin(base.EVEN_SEASONS).to_numpy()
    for key, (X, nm), a in (
        ("teamseason_a1000", (Xt, names_t), PARTICIPATION_RATING_RIDGE_ALPHA),
        ("alpha3000", (Xp, names), 3.0 * PARTICIPATION_RATING_RIDGE_ALPHA),
    ):
        ca, na = fit(X, y, odd, a)
        cb, nb = fit(X, y, even, a)
        idx = channel_idx(nm, na, nb)["offense"]
        repair[key] = summarize(ca[idx], cb[idx], np.random.default_rng(SEED))
        print("repair", key, json.dumps(repair[key]), flush=True)
    rng = np.random.default_rng(SEED)
    out = {
        "repair": repair,
        "alphas": list(ALPHAS),
        "splits": {k: [list(v[0]), list(v[1])] for k, v in SPLITS.items()},
        "looks": len(designs) * len(ALPHAS),
        "cells": {},
        "lopo": {},
    }
    for dname, (X, nm, ps) in designs.items():  # noqa: B007
        for s in SPLITS:
            for a in ALPHAS:
                ca, cb, na, nb = raw[s][dname][a]
                ids = channel_idx(nm, na, nb)
                for ch, idx in ids.items():
                    out["cells"][f"{dname}|{s}|{a}|{ch}"] = summarize(ca[idx], cb[idx], rng)
    for dname in designs:
        for sel, ev in (("S1", "S2"), ("S2", "S1")):

            def score(split, a):
                return np.mean(
                    [
                        out["cells"][f"{dname}|{split}|{a}|{ch}"]["pearson"]  # noqa: B023
                        for ch in ("offense", "defense")
                    ]
                )

            a_sel = max(ALPHAS, key=lambda a: score(sel, a))
            a_in = max(ALPHAS, key=lambda a: score(ev, a))
            rec = {
                "selected_on": sel,
                "evaluated_on": ev,
                "alpha_selected": a_sel,
                "alpha_in_sample_best": a_in,
            }
            for ch in ("offense", "defense"):
                oos = out["cells"][f"{dname}|{ev}|{a_sel}|{ch}"]
                ins = out["cells"][f"{dname}|{ev}|{a_in}|{ch}"]
                rec[ch] = {
                    "oos_sb": oos["sb"],
                    "oos_lo": oos["sb_lo"],
                    "oos_hi": oos["sb_hi"],
                    "oos_p_pos": oos["p_pos"],
                    "n": oos["n"],
                    "in_sample_sb": ins["sb"],
                    "gap": ins["sb"] - oos["sb"],
                }
            out["lopo"][f"{dname}|{sel}->{ev}"] = rec
            print(dname, sel, ev, json.dumps(rec), flush=True)
    OUT.write_text(json.dumps(out, indent=1))


main()
