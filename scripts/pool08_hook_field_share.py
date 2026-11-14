import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.forecast_cold_visitor_tilt_overlay import tuesday_noon_et_cutoff_utc
from nfl_ats.market_data import load_quote_history
from nfl_ats.pool_hook import POOL_OFFSET, is_whole, tuesday_market_lines
from nfl_ats.splash_lines import load_splash_capture

FAMILY = "pool08_hook_field_share"
FIELD = Path("data/splash/field")
DATA_ROOT = Path("data")
OUT = Path("artifacts/pool_rank_replay/pool08_hook_field_share.json")
SEASON = 2026
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-week", type=int, default=None)
    parser.add_argument("--subset", action="append", default=[])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--anchor", choices=("served", "first_snapshot"), default="served")
    return parser.parse_args()


def first_snapshot_lines(week, game_ids):
    capture = load_splash_capture(DATA_ROOT, SEASON, week)
    if capture is None:
        return {}
    opened = tuesday_noon_et_cutoff_utc(pd.Timestamp(capture.captured_at_et).tz_convert("UTC"))
    quotes = load_quote_history(DATA_ROOT / "market" / "raw", since=opened)
    if quotes.empty:
        return {}
    quotes = quotes.loc[quotes["market"].eq("spreads") & quotes["nflverse_game_id"].isin(game_ids)]
    observed = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    commence = pd.to_datetime(quotes["commence_time_utc"], utc=True)
    quotes = quotes.loc[observed.ge(opened) & observed.lt(commence)]
    if quotes.empty:
        return {}
    snapshots = (
        quotes.groupby(["nflverse_game_id", "observed_at_utc"])["home_spread_line"]
        .median()
        .reset_index()
        .sort_values("observed_at_utc")
    )
    first = snapshots.groupby("nflverse_game_id").first()["home_spread_line"]
    return {str(game): float(value) for game, value in first.items()}


def field_file(last_week):
    if last_week is None:
        return sorted(FIELD.glob(f"{SEASON}_field_best_picks_w01-*.parquet"))[-1]
    return FIELD / f"{SEASON}_field_best_picks_w01-{last_week:02d}.parquet"


def subset_cell(s, resid, weeks, lo, hi, rng):
    m = (weeks >= lo) & (weeks <= hi)
    sm, rm, wm = s[m], resid[m], weeks[m]
    n = int(m.sum())
    stat = float(np.mean(sm * rm))
    boots = np.array([np.mean((sm * rm)[i]) for i in rng.integers(0, n, (BOOT_DRAWS, n))])
    perm = np.empty(PERM_DRAWS)
    arrangements = 1
    for w in np.unique(wm):
        g = wm == w
        arrangements *= comb(int(g.sum()), int((sm[g] > 0).sum()))
    for k in range(PERM_DRAWS):
        sp = sm.copy()
        for w in np.unique(wm):
            g = wm == w
            sp[g] = rng.permutation(sm[g])
        perm[k] = np.mean(sp * rm)
    return {
        "weeks": [lo, hi],
        "games_hook": n,
        "hook_side_home": int((sm > 0).sum()),
        "share_resid_pp": 100 * stat,
        "share_resid_ci_pp": [100 * v for v in ci(boots)],
        "share_resid_prob_positive": probability_positive_from_draws(boots),
        "share_resid_perm_two_sided_p": float((np.abs(perm) >= abs(stat)).mean()),
        "perm_distinct_arrangements": arrangements,
    }


def load(path, anchor):
    raw = pd.read_parquet(path)
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
    for w in sorted(df["week"].unique()):
        ids = df.loc[df["week"] == w, "game_id"].tolist()
        if anchor == "first_snapshot":
            market.update(first_snapshot_lines(int(w), ids))
        else:
            market.update(tuesday_market_lines(DATA_ROOT, SEASON, int(w), ids))
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
    args = parse_args()
    rng = np.random.default_rng(SEED)
    path = field_file(args.last_week)
    df = load(path, args.anchor)
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
    subsets = []
    for spec in args.subset:
        lo, hi = (int(v) for v in spec.split("-"))
        subsets.append(subset_cell(s, resid_home, weeks, lo, hi, rng))
    X = np.column_stack([np.ones(len(df)), df["x_home"].to_numpy(), df["hook_home"].to_numpy()])
    n1 = df["n_home"].to_numpy()
    n0 = df["n_away"].to_numpy()
    full = fit_logit(X, n1, n0)
    cboot = np.array(
        [fit_logit(X[i], n1[i], n0[i])[2] for i in rng.integers(0, len(df), (BOOT_DRAWS, len(df)))]
    )
    out = {
        "family": FAMILY,
        "field_file": path.as_posix(),
        "anchor": args.anchor,
        "weeks": [int(df["week"].min()), int(df["week"].max())],
        "hook_games": {
            str(g): [float(a), float(p), float(h)]
            for g, a, p, h in zip(
                hk["game_id"], hk["market"], hk["x_home"], hk["hook_home"], strict=True
            )
        },
        "subsets": subsets,
        "hook_coef_full_fit": float(full[2]),
        "hook_coef_ci": ci(cboot),
        "hook_coef_prob_positive": probability_positive_from_draws(cboot),
        "hook_coef_share_shift_at_even_pp": float(100 * (expit(full[2]) - 0.5)),
        "full_fit_coef_with_hook": full.tolist(),
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
        "looks": {
            "share_resid": 1,
            "share_resid_subsets": len(subsets),
            "logloss_hook_term": 1,
            "hook_coefficient": 1,
            "cover_vs_model_p": 1,
        },
    }
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("fold")}, indent=1))
    print("hook coefs by fold", [round(c[2], 3) for c in c1])


main()
