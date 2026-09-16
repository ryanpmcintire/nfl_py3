import argparse
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap

EVAL_PATH = Path("artifacts/opener_evaluation/20260916T161257Z/per_game.parquet")
FEATURES_PATH = Path("data/processed/game_features.parquet")
ARTIFACT_ROOT = Path("artifacts/market_derived_ratings")
SEED = 20260916
SAMPLES = 2000
DECAY = 0.9
RIDGE = 3.0
HFA_WEEK_ONE = 2.5
LOGIT_L2 = 1e-3
LOGIT_ITERS = 50
RELIABILITY_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LOOKS = []


def record_look(label):
    LOOKS.append(label)


def load_graded():
    evalu = pd.read_parquet(EVAL_PATH)
    feats = pd.read_parquet(FEATURES_PATH)
    keep = ["game_id", "kickoff", "home_team", "away_team", "home_score", "away_score"]
    joined = evalu.merge(feats[keep], on="game_id", how="left")
    assert joined["kickoff"].notna().all(), "kickoff join failed"
    assert joined["home_team"].notna().all(), "team join failed"
    graded = joined.loc[joined["correct_at_open_probability_rule"].notna()].copy()
    graded = graded.reset_index(drop=True)
    n_push = int((graded["margin_vs_open"] == 0.0).sum())
    assert n_push == 0, "graded set must exclude pushes"
    graded["open_line"] = graded["tue_open_home_spread"].astype(float)
    graded["close_line"] = graded["close_home_spread"].astype(float)
    graded["home_covered"] = (graded["margin_vs_open"].astype(float) > 0.0).astype(float)
    served_pick = graded["pick_home_at_open_probability_rule"].astype(bool)
    served_correct = graded["correct_at_open_probability_rule"].astype(float)
    check = np.where(served_pick, served_correct, 1.0 - served_correct)
    assert np.allclose(check, graded["home_covered"].to_numpy()), "cover identity failed"
    graded["served_pick_home"] = served_pick
    graded["served_correct"] = served_correct
    graded["served_p"] = graded["home_cover_probability_at_open"].astype(float)
    graded["market_pick_home"] = graded["open_line"] > 0.0
    graded["market_correct"] = (
        graded["market_pick_home"] == graded["home_covered"].astype(bool)
    ).astype(float)
    return graded


def fit_week_ratings(prior, teams, hfa_fallback):
    n_teams = len(teams)
    if len(prior) == 0:
        return np.zeros(n_teams), float(hfa_fallback)
    w_now = float(prior["fit_week"].iloc[0])
    y = prior["close_line"].astype(float).to_numpy()
    decay_w = np.power(DECAY, (w_now - prior["week"].astype(float).to_numpy()))
    n = len(prior)
    mat = np.zeros((n, n_teams + 1))
    rows = np.arange(n)
    mat[rows, prior["home_idx"].to_numpy()] = 1.0
    mat[rows, prior["away_idx"].to_numpy()] = -1.0
    mat[rows, n_teams] = 1.0
    sw = np.sqrt(decay_w)
    mat_w = mat * sw[:, None]
    y_w = y * sw
    pen = np.zeros((n_teams + 1, n_teams + 1))
    pen[np.arange(n_teams), np.arange(n_teams)] = RIDGE
    lhs = mat_w.T @ mat_w + pen
    rhs = mat_w.T @ y_w
    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return sol[:n_teams], float(sol[n_teams])


def walk_forward_ratings(graded):
    graded = graded.copy()
    graded["implied_line"] = np.nan
    graded["used_hfa"] = np.nan
    for season, season_games in graded.groupby("season"):
        teams = sorted(set(season_games["home_team"]) | set(season_games["away_team"]))
        index = {t: i for i, t in enumerate(teams)}
        weeks = sorted(season_games["week"].unique().tolist())
        for w in weeks:
            test_mask = (graded["season"] == season) & (graded["week"] == w)
            prior = graded.loc[(graded["season"] == season) & (graded["week"] < w)].copy()
            if len(prior):
                kick_min = pd.to_datetime(graded.loc[test_mask, "kickoff"]).min()
                kick_max_used = pd.to_datetime(prior["kickoff"]).max()
                assert kick_max_used < kick_min, "future line in rating fit"
                assert bool((prior["week"] < w).all()), "future week in rating fit"
                prior["fit_week"] = float(w)
                prior["home_idx"] = prior["home_team"].map(index).astype(int)
                prior["away_idx"] = prior["away_team"].map(index).astype(int)
                ratings, hfa = fit_week_ratings(prior, teams, HFA_WEEK_ONE)
            else:
                ratings = np.zeros(len(teams))
                hfa = float(HFA_WEEK_ONE)
            for row_id in graded.loc[test_mask].index:
                home_i = index[graded.loc[row_id, "home_team"]]
                away_i = index[graded.loc[row_id, "away_team"]]
                graded.loc[row_id, "implied_line"] = ratings[home_i] - ratings[away_i] + hfa
                graded.loc[row_id, "used_hfa"] = hfa
    record_look("rating_walkforward_point_in_time")
    assert graded["implied_line"].notna().all(), "missing implied lines"
    graded["resid"] = graded["open_line"] - graded["implied_line"]
    graded["move"] = graded["close_line"] - graded["open_line"]
    return graded


def fit_line_slope(train):
    x = train["resid"].astype(float).to_numpy()
    y = train["move"].astype(float).to_numpy()
    mat = np.column_stack([np.ones(len(train)), x])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_logit(train, cols, target):
    means = {c: float(train[c].astype(float).mean()) for c in cols}
    stds = {c: float(train[c].astype(float).std(ddof=0)) or 1.0 for c in cols}
    mat = np.column_stack(
        [np.ones(len(train))]
        + [((train[c].astype(float) - means[c]) / stds[c]).to_numpy() for c in cols]
    )
    y = train[target].astype(float).to_numpy()
    beta = np.zeros(mat.shape[1])
    for _ in range(LOGIT_ITERS):
        z = np.clip(mat @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = mat.T @ (y - p) - LOGIT_L2 * beta
        hess = (mat.T * w) @ mat + LOGIT_L2 * np.eye(mat.shape[1])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta = beta + step
    return beta, means, stds


def apply_logit(df, cols, beta, means, stds):
    mat = np.column_stack(
        [np.ones(len(df))] + [((df[c].astype(float) - means[c]) / stds[c]).to_numpy() for c in cols]
    )
    z = np.clip(mat @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def natural_coefs(beta, cols, means, stds):
    out = {c: float(beta[i + 1] / stds[c]) for i, c in enumerate(cols)}
    intercept = float(beta[0])
    for i, c in enumerate(cols):
        intercept -= float(beta[i + 1]) * means[c] / stds[c]
    out["intercept"] = intercept
    return out


def loso_predict(graded):
    seasons = sorted(graded["season"].unique().tolist())
    graded = graded.copy()
    graded["slope_a"] = np.nan
    graded["slope_b"] = np.nan
    graded["move_pred"] = np.nan
    graded["r1_p"] = np.nan
    graded["r2_p"] = np.nan
    slope_folds = {}
    r1_folds = {}
    r2_folds = {}
    r1_in_acc = []
    r2_in_acc = []
    for held in seasons:
        train = graded.loc[graded["season"] != held]
        test_idx = graded.loc[graded["season"] == held].index
        a, b = fit_line_slope(train)
        slope_folds[str(held)] = {"intercept": a, "slope": b, "n_train": len(train)}
        graded.loc[test_idx, "slope_a"] = a
        graded.loc[test_idx, "slope_b"] = b
        graded.loc[test_idx, "move_pred"] = a + b * graded.loc[test_idx, "resid"].astype(float)
        beta1, m1, s1 = fit_logit(train, ["resid"], "home_covered")
        beta2, m2, s2 = fit_logit(train, ["served_logit", "resid"], "home_covered")
        r1_folds[str(held)] = dict(natural_coefs(beta1, ["resid"], m1, s1))
        r2_folds[str(held)] = dict(natural_coefs(beta2, ["served_logit", "resid"], m2, s2))
        r1_folds[str(held)]["n_train"] = len(train)
        r2_folds[str(held)]["n_train"] = len(train)
        graded.loc[test_idx, "r1_p"] = apply_logit(graded.loc[test_idx], ["resid"], beta1, m1, s1)
        graded.loc[test_idx, "r2_p"] = apply_logit(
            graded.loc[test_idx], ["served_logit", "resid"], beta2, m2, s2
        )
        in1 = (apply_logit(train, ["resid"], beta1, m1, s1) >= 0.5).astype(float)
        in2 = (apply_logit(train, ["served_logit", "resid"], beta2, m2, s2) >= 0.5).astype(float)
        r1_in_acc.append(float((in1 == train["home_covered"].astype(float).to_numpy()).mean()))
        r2_in_acc.append(float((in2 == train["home_covered"].astype(float).to_numpy()).mean()))
    record_look("loso_move_slope_six_folds")
    record_look("loso_r1_resid_only_six_folds")
    record_look("loso_r2_served_plus_resid_six_folds")
    graded["r1_pick_home"] = graded["r1_p"] >= 0.5
    graded["r2_pick_home"] = graded["r2_p"] >= 0.5
    graded["r1_correct"] = (graded["r1_pick_home"] == graded["home_covered"].astype(bool)).astype(
        float
    )
    graded["r2_correct"] = (graded["r2_pick_home"] == graded["home_covered"].astype(bool)).astype(
        float
    )
    beta1_full, m1_full, s1_full = fit_logit(graded, ["resid"], "home_covered")
    beta2_full, m2_full, s2_full = fit_logit(graded, ["served_logit", "resid"], "home_covered")
    graded["r1_p_in"] = apply_logit(graded, ["resid"], beta1_full, m1_full, s1_full)
    graded["r2_p_in"] = apply_logit(graded, ["served_logit", "resid"], beta2_full, m2_full, s2_full)
    graded["r1_correct_in"] = (
        (graded["r1_p_in"] >= 0.5).astype(float) == graded["home_covered"]
    ).astype(float)
    graded["r2_correct_in"] = (
        (graded["r2_p_in"] >= 0.5).astype(float) == graded["home_covered"]
    ).astype(float)
    record_look("in_sample_r1_r2_full_fit")
    info = {
        "slope_folds": slope_folds,
        "r1_folds": r1_folds,
        "r2_folds": r2_folds,
        "r1_full": natural_coefs(beta1_full, ["resid"], m1_full, s1_full),
        "r2_full": natural_coefs(beta2_full, ["served_logit", "resid"], m2_full, s2_full),
        "r1_in_acc_mean": float(np.mean(r1_in_acc)),
        "r2_in_acc_mean": float(np.mean(r2_in_acc)),
    }
    return graded, info


def paired_effect(df, cand_col, base_col, scale):
    def metric_fn(sub):
        return {
            "effect": float(
                (sub[cand_col].astype(float).mean() - sub[base_col].astype(float).mean()) * scale
            )
        }

    result = week_blocked_bootstrap(df, metric_fn, block="season", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def reliability_table(df, p_col):
    rows = []
    for lo, hi in pairwise(RELIABILITY_EDGES):
        top = hi if hi < 1.0 else 1.0 + 1e-9
        mask = df[p_col].ge(lo) & df[p_col].lt(top)
        n = int(mask.sum())
        mean_p = float(df.loc[mask, p_col].mean()) if n else float("nan")
        actual = float(df.loc[mask, "home_covered"].mean()) if n else float("nan")
        rows.append(
            {
                "bin": str(lo) + "_to_" + str(hi),
                "n": n,
                "mean_p": mean_p,
                "actual": actual,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default="")
    args = parser.parse_args()
    graded = load_graded()
    point_margin = (graded["home_score"] - graded["away_score"]).astype(float)
    var_ratio = float(graded["close_line"].var(ddof=0) / point_margin.var(ddof=0))
    var_close = float(graded["close_line"].var(ddof=0))
    var_margin = float(point_margin.var(ddof=0))
    graded = walk_forward_ratings(graded)
    graded["served_logit"] = np.log(
        graded["served_p"].clip(1e-6, 1.0 - 1e-6)
        / (1.0 - graded["served_p"].clip(1e-6, 1.0 - 1e-6))
    )
    graded, info = loso_predict(graded)
    graded["move_null_se"] = graded["move"] ** 2
    graded["move_model_se"] = (graded["move"] - graded["move_pred"]) ** 2
    graded["mse_gain"] = graded["move_null_se"] - graded["move_model_se"]
    record_look("a_move_mse_gain_vs_null")
    a_mse = paired_effect(graded, "move_model_se", "move_null_se", -1.0)
    pooled_slope = dict(info["slope_folds"])
    oos_corr = float(graded[["resid", "move"]].corr().loc["resid", "move"])
    r2_vs_served = paired_effect(graded, "r2_correct", "served_correct", 100.0)
    record_look("b_r2_vs_served_accuracy")
    r1_vs_served = paired_effect(graded, "r1_correct", "served_correct", 100.0)
    record_look("b_r1_vs_served_accuracy")
    r2_vs_market = paired_effect(graded, "r2_correct", "market_correct", 100.0)
    record_look("b_r2_vs_market_accuracy")
    served_vs_market = paired_effect(graded, "served_correct", "market_correct", 100.0)
    record_look("b_served_vs_market_accuracy")
    graded["served_brier"] = (graded["served_p"] - graded["home_covered"]) ** 2
    graded["r2_brier"] = (graded["r2_p"] - graded["home_covered"]) ** 2
    clip_r2 = graded["r2_p"].clip(1e-6, 1.0 - 1e-6)
    clip_s = graded["served_p"].clip(1e-6, 1.0 - 1e-6)
    graded["served_logloss"] = -(
        graded["home_covered"] * np.log(clip_s)
        + (1.0 - graded["home_covered"]) * np.log(1.0 - clip_s)
    )
    graded["r2_logloss"] = -(
        graded["home_covered"] * np.log(clip_r2)
        + (1.0 - graded["home_covered"]) * np.log(1.0 - clip_r2)
    )
    r2_vs_served_brier = paired_effect(graded, "served_brier", "r2_brier", 1.0)
    record_look("b_r2_vs_served_brier")
    r2_vs_served_logloss = paired_effect(graded, "served_logloss", "r2_logloss", 1.0)
    record_look("b_r2_vs_served_logloss")
    r2_in_vs_served = paired_effect(graded, "r2_correct_in", "served_correct", 100.0)
    record_look("b_r2_in_sample_vs_served")
    season_rows = []
    for season, sub in graded.groupby("season"):
        season_rows.append(
            {
                "season": int(season),
                "n": len(sub),
                "served_acc": float(sub["served_correct"].mean()),
                "market_acc": float(sub["market_correct"].mean()),
                "r1_oos_acc": float(sub["r1_correct"].mean()),
                "r2_oos_acc": float(sub["r2_correct"].mean()),
                "slope_b": float(sub["slope_b"].iloc[0]),
                "resid_move_corr": float(sub[["resid", "move"]].corr().loc["resid", "move"]),
                "mse_gain": float(sub["mse_gain"].mean()),
            }
        )
    record_look("season_split_descriptive")
    rel_r2 = reliability_table(graded, "r2_p")
    record_look("reliability_r2_five_bins_descriptive")
    rel_served = reliability_table(graded, "served_p")
    record_look("reliability_served_five_bins_descriptive")
    diff_mask = graded["r2_pick_home"].ne(graded["served_pick_home"])
    decisive = {
        "n_diff": int(diff_mask.sum()),
        "served_record_on_diff": str(int(graded.loc[diff_mask, "served_correct"].sum()))
        + "-"
        + str(int(diff_mask.sum() - graded.loc[diff_mask, "served_correct"].sum())),
        "r2_record_on_diff": str(int(graded.loc[diff_mask, "r2_correct"].sum()))
        + "-"
        + str(int(diff_mask.sum() - graded.loc[diff_mask, "r2_correct"].sum())),
    }
    summary = {
        "eval_population": str(EVAL_PATH),
        "graded_games": len(graded),
        "seasons": sorted(graded["season"].unique().tolist()),
        "seed": SEED,
        "samples": SAMPLES,
        "bootstrap_block": "season",
        "predeclared": {"decay": DECAY, "ridge": RIDGE, "hfa_week_one": HFA_WEEK_ONE},
        "variance": {"var_close": var_close, "var_margin": var_margin, "ratio": var_ratio},
        "a_move": {
            "mse_gain_vs_null": a_mse,
            "oos_corr_resid_move": oos_corr,
            "slope_folds": pooled_slope,
        },
        "b_cover": {
            "records": {
                "served": str(int(graded["served_correct"].sum()))
                + "-"
                + str(len(graded) - int(graded["served_correct"].sum())),
                "market": str(int(graded["market_correct"].sum()))
                + "-"
                + str(len(graded) - int(graded["market_correct"].sum())),
                "r1_oos": str(int(graded["r1_correct"].sum()))
                + "-"
                + str(len(graded) - int(graded["r1_correct"].sum())),
                "r2_oos": str(int(graded["r2_correct"].sum()))
                + "-"
                + str(len(graded) - int(graded["r2_correct"].sum())),
            },
            "r2_oos_acc": float(graded["r2_correct"].mean()),
            "r1_oos_acc": float(graded["r1_correct"].mean()),
            "served_acc": float(graded["served_correct"].mean()),
            "market_acc": float(graded["market_correct"].mean()),
            "r2_in_acc_mean": info["r2_in_acc_mean"],
            "r1_in_acc_mean": info["r1_in_acc_mean"],
            "r2_vs_served": r2_vs_served,
            "r1_vs_served": r1_vs_served,
            "r2_vs_market": r2_vs_market,
            "served_vs_market": served_vs_market,
            "r2_vs_served_brier": r2_vs_served_brier,
            "r2_vs_served_logloss": r2_vs_served_logloss,
            "r2_in_vs_served": r2_in_vs_served,
            "decisive_games": decisive,
        },
        "r1_full": info["r1_full"],
        "r2_full": info["r2_full"],
        "r1_folds": info["r1_folds"],
        "r2_folds": info["r2_folds"],
        "season_split": season_rows,
        "reliability_r2_oos": rel_r2,
        "reliability_served": rel_served,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
    }
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ARTIFACT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    export_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "open_line",
        "close_line",
        "implied_line",
        "resid",
        "move",
        "move_pred",
        "served_p",
        "served_pick_home",
        "served_correct",
        "market_pick_home",
        "market_correct",
        "r1_p",
        "r1_correct",
        "r2_p",
        "r2_correct",
        "home_covered",
    ]
    graded[export_cols].to_csv(out_dir / "per_game.csv", index=False)
    print("artifact_dir=" + str(out_dir))
    print("graded_games=" + str(len(graded)))
    print(
        "var_close="
        + str(var_close)
        + " var_margin="
        + str(var_margin)
        + " ratio="
        + str(var_ratio)
    )
    print("a_mse_gain_vs_null=" + json.dumps(a_mse))
    print("a_oos_corr=" + str(oos_corr))
    print("b_r2_vs_served=" + json.dumps(r2_vs_served))
    print("b_r1_vs_served=" + json.dumps(r1_vs_served))
    print("b_r2_vs_market=" + json.dumps(r2_vs_market))
    print("b_served_vs_market=" + json.dumps(served_vs_market))
    print("b_r2_vs_served_brier=" + json.dumps(r2_vs_served_brier))
    print("b_r2_vs_served_logloss=" + json.dumps(r2_vs_served_logloss))
    print("b_r2_in_vs_served=" + json.dumps(r2_in_vs_served))
    print("records=" + json.dumps(summary["b_cover"]["records"]))
    print("decisive=" + json.dumps(decisive))
    print("season_split=" + json.dumps(season_rows))
    print("r2_full=" + json.dumps(info["r2_full"]))
    print("r1_full=" + json.dumps(info["r1_full"]))
    print("look_count=" + str(len(LOOKS)) + " looks=" + json.dumps(LOOKS))


if __name__ == "__main__":
    main()
