import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.pool_hook import POOL_OFFSET, is_whole, tuesday_market_lines

FAMILY = "pool08_hook_field_share"
FIELD_FILE = Path("data/splash/field/2026_field_best_picks_w01-08.parquet")
DATA_ROOT = Path("data")
OUT = Path("artifacts/pool_rank_replay/pool08_hook_field_share.json")
SEASON = 2026
WEEKS = range(1, 9)
BOOT_DRAWS = 4000
PERM_DRAWS = 20000
SEED = 8
INTERVAL = (2.5, 97.5)
NEWTON_STEPS = 50
RIDGE = 1e-9


def fit_logit(X, n1, n0):
    b = np.zeros(X.shape[1])
    for _ in range(NEWTON_STEPS):
        q = expit(X @ b)
        w = (n1 + n0) * q * (1 - q) + RIDGE
        g = X.T @ (n1 - (n1 + n0) * q)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + RIDGE * np.eye(X.shape[1]), g)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return b


def load():
    raw = pd.read_parquet(FIELD_FILE)
    rows = []
    for gid, g in raw.groupby("game_id"):
        h = g[g["home"]].iloc[0]
        a = g[~g["home"]].iloc[0]
        rows.append(
            {
                "game_id": gid,
                "week": int(h["week"]),
                "home": h["team"],
                "away": a["team"],
                "x_home": float(-h["line"]),
                "n_home": float(h["picks"]),
                "n_away": float(a["picks"]),
                "home_covered": bool(h["covered"]),
                "p_home": float(h["model_p"]),
            }
        )
    df = pd.DataFrame(rows)
    market = {}
    for w in WEEKS:
        ids = df.loc[df["week"] == w, "game_id"].tolist()
        market.update(tuesday_market_lines(DATA_ROOT, SEASON, w, ids))
    df["market"] = df["game_id"].map(market)
    off = df["x_home"] - df["market"]
    df["hook_home"] = np.where(
        df["market"].map(lambda v: isinstance(v, float) and is_whole(v))
        & np.isclose(off.abs(), POOL_OFFSET),
        -np.sign(off),
        0.0,
    )
    return df


def lowo(df, hook):
    pred = np.zeros(len(df))
    coef = []
    for w in sorted(df["week"].unique()):
        tr = df["week"] != w
        cols = [np.ones(len(df)), df["x_home"].to_numpy()]
        if hook:
            cols.append(df["hook_home"].to_numpy())
        X = np.column_stack(cols)
        b = fit_logit(X[tr], df.loc[tr, "n_home"].to_numpy(), df.loc[tr, "n_away"].to_numpy())
        pred[~tr.to_numpy()] = expit(X[~tr.to_numpy()] @ b)
        coef.append(b.tolist())
    return pred, coef


def logloss(df, pred, idx):
    d = df.iloc[idx]
    p = pred[idx]
    n = d["n_home"].to_numpy() + d["n_away"].to_numpy()
    return (
        -(d["n_home"].to_numpy() * np.log(p) + d["n_away"].to_numpy() * np.log(1 - p)).sum()
        / n.sum()
    )


def ci(x):
    lo, hi = np.percentile(x, INTERVAL)
    return [float(lo), float(hi)]


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    p0, c0 = lowo(df, False)
    p1, c1 = lowo(df, True)
    df["pred_home"] = p0
    df["pred_home_hook"] = p1
    df["share_home"] = df["n_home"] / (df["n_home"] + df["n_away"])
    hk = df[df["hook_home"] != 0].reset_index(drop=True)
    hidx = np.flatnonzero(df["hook_home"].to_numpy() != 0)
    s = hk["hook_home"].to_numpy()
    resid_home = (hk["share_home"] - hk["pred_home"]).to_numpy()
    hook_share = np.where(s > 0, hk["share_home"], 1 - hk["share_home"])
    hook_pred = np.where(s > 0, hk["pred_home"], 1 - hk["pred_home"])
    stat = float(np.mean(s * resid_home))
    boots = np.array(
        [np.mean((s * resid_home)[i]) for i in rng.integers(0, len(hk), (BOOT_DRAWS, len(hk)))]
    )
    weeks = hk["week"].to_numpy()
    perm = np.empty(PERM_DRAWS)
    for k in range(PERM_DRAWS):
        sp = s.copy()
        for w in np.unique(weeks):
            m = weeks == w
            sp[m] = rng.permutation(s[m])
        perm[k] = np.mean(sp * resid_home)
    p_perm = float((np.abs(perm) >= abs(stat)).mean())
    wmean = float(
        np.average(hook_share - hook_pred, weights=(hk["n_home"] + hk["n_away"]).to_numpy())
    )
    ll0 = logloss(df, p0, hidx)
    ll1 = logloss(df, p1, hidx)
    all_idx = np.arange(len(df))
    imp_all = logloss(df, p0, all_idx) - logloss(df, p1, all_idx)
    gains = []
    for i in rng.integers(0, len(df), (BOOT_DRAWS, len(df))):
        gains.append(logloss(df, p0, i) - logloss(df, p1, i))
    gains = np.array(gains)
    hook_cov = np.where(s > 0, hk["home_covered"], ~hk["home_covered"]).astype(float)
    hook_p = np.where(s > 0, hk["p_home"], 1 - hk["p_home"])
    diff = hook_cov - hook_p
    dboot = np.array([diff[i].mean() for i in rng.integers(0, len(hk), (BOOT_DRAWS, len(hk)))])
    pnull = []
    for _ in range(PERM_DRAWS):
        pnull.append((rng.random(len(hk)) < hook_p).mean() - hook_p.mean())
    pnull = np.array(pnull)
    obs = float(hook_cov.mean() - hook_p.mean())
    p_cov_perm = float((np.abs(pnull) >= abs(obs)).mean())
    out = {
        "family": FAMILY,
        "games_total": len(df),
        "games_hook": len(hk),
        "hook_by_week": hk.groupby("week").size().to_dict(),
        "hook_side_home": int((s > 0).sum()),
        "decisive_record_hook_side_covered": [int(hook_cov.sum()), len(hk)],
        "mean_model_p_hook_side": float(hook_p.mean()),
        "cover_minus_model_p": obs,
        "cover_minus_model_p_ci": ci(dboot),
        "cover_minus_model_p_prob_positive": probability_positive_from_draws(dboot),
        "cover_perm_two_sided_p": p_cov_perm,
        "mean_field_share_hook_side": float(hook_share.mean()),
        "mean_proxy_share_hook_side": float(hook_pred.mean()),
        "share_resid_pp": 100 * stat,
        "share_resid_ci_pp": [100 * v for v in ci(boots)],
        "share_resid_prob_positive": probability_positive_from_draws(boots),
        "share_resid_perm_two_sided_p": p_perm,
        "share_resid_entry_weighted_pp": 100 * wmean,
        "weeks_resid_pp": {
            int(w): float(100 * np.mean((s * resid_home)[weeks == w])) for w in np.unique(weeks)
        },
        "hook_games_logloss_proxy": ll0,
        "hook_games_logloss_with_hook": ll1,
        "all_games_logloss_improvement": imp_all,
        "all_games_improvement_ci": ci(gains),
        "all_games_improvement_prob_positive": probability_positive_from_draws(gains),
        "fold_coef_proxy": c0,
        "fold_coef_with_hook": c1,
        "looks": {"share_resid": 1, "logloss_hook_term": 1, "cover_vs_model_p": 1},
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("fold")}, indent=1))
    print("hook coefs by fold", [round(c[2], 3) for c in c1])


main()
