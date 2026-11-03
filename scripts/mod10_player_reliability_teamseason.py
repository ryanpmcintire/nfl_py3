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

ALPHAS = (PARTICIPATION_RATING_RIDGE_ALPHA, 3.0 * PARTICIPATION_RATING_RIDGE_ALPHA)
OUT = Path("F:/Repos/nfl_py3/artifacts/mod10_graph_reliability/result.json")
BOOT = 2000
SEED = 20261103


def rows(table: pd.DataFrame, team_season: bool):
    scale = PARTICIPATION_RATING_TEAM_FEATURE_SCALE
    for r in table.itertuples(index=False):
        v = {}
        for p in base._player_ids(r.offense_players):
            v[f"offense_player::{p}"] = 1.0
        for p in base._player_ids(r.defense_players):
            v[f"defense_player::{p}"] = -1.0
        v[f"offense_team::{r.posteam}"] = scale
        v[f"defense_team::{r.defteam}"] = -scale
        if team_season:
            v[f"offense_ts::{r.posteam}{int(r.season)}"] = scale
            v[f"defense_ts::{r.defteam}{int(r.season)}"] = -scale
        yield v


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(table, team_season, alpha, half_a_mask, half_b_mask):
    vec = DictVectorizer(sparse=True, sort=True)
    X = vec.fit_transform(rows(table, team_season)).tocsr()
    names = vec.get_feature_names_out()
    y = (
        pd.to_numeric(table["epa"])
        .clip(-PARTICIPATION_RATING_EPA_CLIP, PARTICIPATION_RATING_EPA_CLIP)
        .to_numpy(float)
    )
    res = {}
    coefs, counts = [], []
    for mask in (half_a_mask, half_b_mask):
        est = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, tol=1e-6).fit(X[mask], y[mask])
        coefs.append(dict(zip(names, est.coef_)))  # noqa: B905
        counts.append(np.asarray(abs(X[mask]).sum(axis=0)).ravel())
    cdict = [dict(zip(names, c)) for c in counts]  # noqa: B905, F841
    rng = np.random.default_rng(SEED)
    for ch, pre in (("offense", "offense_player::"), ("defense", "defense_player::")):
        idx = [
            i
            for i, n in enumerate(names)
            if n.startswith(pre)
            and counts[0][i] >= base.RELIABILITY_MIN_PLAYS
            and counts[1][i] >= base.RELIABILITY_MIN_PLAYS
        ]
        l = np.array([coefs[0][names[i]] for i in idx])  # noqa: E741
        r = np.array([coefs[1][names[i]] for i in idx])
        pr = corr(l, r)
        draws = []
        for _ in range(BOOT):
            s = rng.integers(0, len(idx), len(idx))
            draws.append(corr(l[s], r[s]))
        sb = lambda x: 2 * x / (1 + x)  # noqa: E731
        d = np.array(draws)
        res[ch] = {
            "players": len(idx),
            "pearson": pr,
            "sb": sb(pr),
            "sb_lo": sb(float(np.percentile(d, 2.5))),
            "sb_hi": sb(float(np.percentile(d, 97.5))),
        }
    return res


def main():
    base.PBP_SNAPSHOT = base.REPO_ROOT / "data" / "pbp" / "raw" / "20261027T220410Z"
    table = base._play_table()
    table = table.reset_index(drop=True)
    season = table["season"].astype(int)
    odd = season.isin(base.ODD_SEASONS).to_numpy()
    even = season.isin(base.EVEN_SEASONS).to_numpy()
    out = {"design": "odd v even seasons, raw ridge coefs, >=500 plays both halves", "arms": {}}
    for ts in (False, True):
        for a in ALPHAS:
            key = f"team_season={ts},alpha={a}"
            out["arms"][key] = evaluate(table, ts, a, odd, even)
            print(key, json.dumps(out["arms"][key]), flush=True)
    OUT.write_text(json.dumps(out, indent=1))


main()
