import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from nfl_ats.board_content import _pool_hook_effect
from nfl_ats.card_view import resolve_card_probabilities
from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.pick_probability import load_pick_probability_model
from nfl_ats.pool_hook import POOL_OFFSET, crossed_push_mass, is_whole, tuesday_market_lines

FAMILY = "pool09_hook_value_slope"
FIELD = Path("data/splash/field")
DATA_ROOT = Path("data")
OUT = {
    "served_effect": Path("artifacts/pool_rank_replay/pool09_hook_value_slope.json"),
    "push_mass": Path("artifacts/pool_rank_replay/pool09_hook_push_mass_slope.json"),
}
ARTIFACTS = Path("artifacts")
SEASON = 2026
FIRST_SERVED_WEEK = 3
SPLIT_WEEK = 6
BOOT_DRAWS = 4000
PERM_DRAWS = 20000
SEED = 9
INTERVAL = (2.5, 97.5)
NEWTON_STEPS = 50
RIDGE = 1e-9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-week", type=int, default=None)
    parser.add_argument("--covariate", choices=tuple(OUT), default="served_effect")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def field_file(last_week):
    if last_week is None:
        return sorted(FIELD.glob(f"{SEASON}_field_best_picks_w01-*.parquet"))[-1]
    return FIELD / f"{SEASON}_field_best_picks_w01-{last_week:02d}.parquet"


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


def load(path):
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


def served_effects(model, last_week):
    out = {}
    for w in range(FIRST_SERVED_WEEK, last_week + 1):
        d = next(
            x
            for x in sorted((ARTIFACTS / "margin_predictions").glob(f"2026-week-{w:02d}-*"))
            if (x / "pool_card.csv").exists()
        )
        pred = pd.read_csv(d / "recommendations.csv")
        card = resolve_card_probabilities(
            pred,
            data_root=DATA_ROOT,
            require_fresh_arrest_overlay=False,
            artifacts_root=ARTIFACTS,
        )
        fp = card.predictions
        for _, row in fp.iterrows():
            gid = str(row["game_id"])
            out[gid] = (
                _pool_hook_effect(model, row, True),
                float(row["pool_hook"]),
                float(row["market_line_home_cover_probability"]),
            )
    return out


def ols_slope(x, y):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc**2).sum())


def within_slope(x, y, weeks):
    xw = x - pd.Series(x).groupby(weeks).transform("mean").to_numpy()
    yw = y - pd.Series(y).groupby(weeks).transform("mean").to_numpy()
    return float((xw * yw).sum() / (xw**2).sum())


def ci(a):
    lo, hi = np.percentile(a, INTERVAL)
    return [float(lo), float(hi)]


def main():
    args = parse_args()
    rng = np.random.default_rng(SEED)
    model = load_pick_probability_model(ARTIFACTS)
    path = field_file(args.last_week)
    df = load(path)
    p0, _ = lowo(df, False)
    df["pred_home"] = p0
    df["share_home"] = df["n_home"] / (df["n_home"] + df["n_away"])
    hook = df[df["hook_home"] != 0]
    extra = {}
    if args.covariate == "push_mass":
        table = dict(model.hook_push_mass)
        mass = np.array(
            [
                crossed_push_mass(float(a), float(p), table)
                for a, p in zip(hook["market"], hook["x_home"], strict=True)
            ]
        )
        hk = hook[mass > 0].reset_index(drop=True)
        s = hk["hook_home"].to_numpy()
        x = 100 * mass[mass > 0]
        extra = {
            "games_without_table_entry": int((mass <= 0).sum()),
            "push_mass_pp_by_anchor": {
                str(a): [int(n), float(100 * table[float(a)])]
                for a, n in hk["market"].value_counts().sort_index().items()
            },
            "half_point_cover_value_pp_mean": float(x.mean() / 2),
        }
    else:
        eff = served_effects(model, int(df["week"].max()))
        hk = hook[hook["game_id"].isin(eff)].reset_index(drop=True)
        s = hk["hook_home"].to_numpy()
        home_eff = np.array([eff[g][0] for g in hk["game_id"]])
        served_sign = np.array([np.sign(eff[g][1]) for g in hk["game_id"]])
        x = 100 * s * home_eff
        extra = {"served_sign_agrees_with_hook_side": int((served_sign == s).sum())}
    y = 100 * s * (hk["share_home"] - hk["pred_home"]).to_numpy()
    weeks = hk["week"].to_numpy()
    n = len(hk)
    slope = ols_slope(x, y)
    boots = np.array([ols_slope(x[i], y[i]) for i in rng.integers(0, n, (BOOT_DRAWS, n))])
    perm = np.empty(PERM_DRAWS)
    perm_within = np.empty(PERM_DRAWS)
    for k in range(PERM_DRAWS):
        xp = x.copy()
        for w in np.unique(weeks):
            m = weeks == w
            xp[m] = rng.permutation(x[m])
        perm[k] = ols_slope(xp, y)
        perm_within[k] = within_slope(xp, y, weeks)
    p_perm = float((np.abs(perm) >= abs(slope)).mean())
    p_one = float((perm >= slope).mean())
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho_boot = np.array(
        [np.corrcoef(rx[i], ry[i])[0, 1] for i in rng.integers(0, n, (BOOT_DRAWS, n))]
    )
    half = weeks < SPLIT_WEEK
    slope_within = within_slope(x, y, weeks)
    within_boot = np.array(
        [within_slope(x[i], y[i], weeks[i]) for i in rng.integers(0, n, (BOOT_DRAWS, n))]
    )
    out = {
        "family": FAMILY,
        "covariate": args.covariate,
        "field_file": path.as_posix(),
        "weeks": [int(weeks.min()), int(weeks.max())],
        "games_hook_all": len(hook),
        "games_hook": n,
        "hook_by_week": {int(w): int((weeks == w).sum()) for w in np.unique(weeks)},
        **extra,
        "effect_pp_mean": float(x.mean()),
        "effect_pp_min_max": [float(x.min()), float(x.max())],
        "effect_pp_sd": float(x.std(ddof=1)),
        "resid_pp_mean": float(y.mean()),
        "slope_pp_per_pp": slope,
        "slope_ci": ci(boots),
        "slope_prob_positive": probability_positive_from_draws(boots),
        "slope_perm_two_sided_p": p_perm,
        "slope_perm_one_sided_p": p_one,
        "permuted_covariate_slope_mean": float(perm.mean()),
        "permuted_covariate_slope_ci": ci(perm),
        "slope_perm_centred_two_sided_p": float(
            (np.abs(perm - perm.mean()) >= abs(slope - perm.mean())).mean()
        ),
        "slope_within_week": slope_within,
        "slope_within_week_ci": ci(within_boot),
        "slope_within_week_prob_positive": probability_positive_from_draws(within_boot),
        "slope_within_week_perm_two_sided_p": float(
            (np.abs(perm_within) >= abs(slope_within)).mean()
        ),
        "permuted_covariate_within_week_slope_ci": ci(perm_within),
        "spearman": rho,
        "spearman_ci": ci(rho_boot),
        "spearman_prob_positive": probability_positive_from_draws(rho_boot),
        "slope_early": ols_slope(x[half], y[half]),
        "slope_late": ols_slope(x[~half], y[~half]),
        "n_early": int(half.sum()),
        "looks": {
            "slope": 1,
            "slope_within_week": 1,
            "spearman_check": 1,
            "half_split_reliability": 2,
        },
    }
    target = args.out or OUT[args.covariate]
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


main()
