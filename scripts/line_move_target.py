import argparse
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from nfl_ats.pick_probability import (
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    PROBABILITY_EPSILON,
    signed_composition_flags,
)
from nfl_ats.pick_probability_fit import (
    MARKET_MOVE_COLUMN,
    _arrest_incidents,
    _design,
    _fit_logit,
    _forecast_temperatures,
    _market_move_table,
    _predict,
    _protection_back_side,
    _standardisers,
)
from nfl_ats.snapshots import latest_snapshot, load_snapshot

OPENER_EVALUATION = "artifacts/opener_evaluation/20260915T171638Z/per_game.parquet"
MOVE_FEATURES = ("tue_open_home_spread", "residual_at_open", "model_logit", "week")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260916
FIT_RIDGE = 1e-3
RIDGE_ALPHA = 1.0
HGB_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "random_state": 7,
}
RELIABILITY_EDGES = (0.5, 0.55, 0.6, 0.65, 0.7, 1.01)
LOOKS = 8
LOOK_FAMILY = "2 move candidates by 4 metrics against the served baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description="MKT-20 line-move target: predict close minus open from opener-time info, LOSO"
    )
    parser.add_argument("--out-root", default="artifacts/line_move_target")
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def season_block_bootstrap(frame, value_column, season_column, draws, seed):
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    seasons = frame[season_column].to_numpy()
    unique = sorted(set(seasons))
    rng = np.random.default_rng(seed)
    index_by_season = {s: np.flatnonzero(seasons == s) for s in unique}
    draws_out = np.empty(draws, dtype=float)
    for b in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        pooled = np.concatenate([index_by_season[s] for s in picked])
        draws_out[b] = float(np.mean(values[pooled]))
    estimate = float(np.mean(values))
    return {
        "estimate": estimate,
        "lower": float(np.quantile(draws_out, 0.025)),
        "upper": float(np.quantile(draws_out, 0.975)),
        "probability_positive": float(np.mean(draws_out > 0.0)),
        "draws": int(draws),
    }


def reliability_table(probabilities, outcomes):
    rows = []
    shown = np.where(probabilities >= 0.5, probabilities, 1.0 - probabilities)
    picked_home = probabilities >= 0.5
    for lower, upper in pairwise(RELIABILITY_EDGES):
        mask = (shown >= lower) & (shown < upper)
        games = int(mask.sum())
        if games == 0:
            rows.append({"lower": lower, "upper": upper, "games": 0})
            continue
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "games": games,
                "mean_probability": float(np.mean(shown[mask])),
                "accuracy": float(np.mean(picked_home[mask].astype(float) == outcomes[mask])),
            }
        )
    return rows


def brier_score(probabilities, outcomes):
    return float(np.mean((probabilities - outcomes) ** 2))


def log_loss(probabilities, outcomes):
    clipped = np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return float(-np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped)))


def accuracy_of(probabilities, outcomes):
    return float(np.mean((probabilities >= 0.5).astype(float) == outcomes))


def per_game_log_loss(probabilities, outcomes):
    clipped = np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return -(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))


def log_loss_gain(candidate, baseline, outcomes):
    return per_game_log_loss(baseline, outcomes) - per_game_log_loss(candidate, outcomes)


def main():
    args = parse_args()
    artifacts_root = Path("artifacts")
    data_root = Path("data")
    per_game = pd.read_parquet(OPENER_EVALUATION)
    for column in MOVE_FEATURES:
        assert "close" not in column, column
    margin = pd.to_numeric(per_game["margin_vs_open"], errors="coerce")
    graded = per_game.loc[margin.notna() & margin.ne(0.0)].reset_index(drop=True)
    move = pd.to_numeric(graded["close_home_spread"], errors="coerce") - pd.to_numeric(
        graded["tue_open_home_spread"], errors="coerce"
    )
    check = pd.to_numeric(graded["open_move"], errors="coerce")
    assert int(move.isna().sum()) == 0
    assert float((move - check).abs().max()) == 0.0
    graded["move"] = move.to_numpy(dtype=float)
    stated = pd.to_numeric(graded["home_cover_probability_at_open"], errors="coerce").clip(
        PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    graded["model_probability"] = stated.to_numpy(dtype=float)
    graded["model_logit"] = np.log(stated / (1.0 - stated)).to_numpy(dtype=float)
    graded["home_covered"] = (pd.to_numeric(graded["margin_vs_open"], errors="coerce") > 0).astype(
        float
    )
    needed = [*MOVE_FEATURES, "move", "home_covered", "model_probability"]
    before = len(graded)
    graded = graded.loc[graded[needed].notna().all(axis=1)].reset_index(drop=True)
    dropped = before - len(graded)
    label_std = float(np.std(graded["move"].to_numpy(dtype=float), ddof=1))
    margin_values = pd.to_numeric(graded["margin_vs_open"], errors="coerce").to_numpy(dtype=float)
    margin_std = float(np.std(margin_values, ddof=1))

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    schedule_names = (
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
    )
    schedule_columns = [column for column in schedule_names if column in schedules.columns]
    joined = graded.merge(
        schedules[schedule_columns].drop_duplicates("game_id"),
        on=["game_id", "season"],
        how="left",
        suffixes=("", "_schedule"),
    )
    if "week_schedule" in joined.columns:
        joined = joined.drop(columns=["week_schedule"])
    assert int(joined["home_team"].isna().sum()) == 0
    flags = signed_composition_flags(
        joined,
        schedules,
        incidents=_arrest_incidents(data_root),
        forecasts_tuesday_noon=_forecast_temperatures(data_root),
        protection_back_side=_protection_back_side(schedules, data_root),
    )
    joined = joined.merge(flags, on="game_id", how="left", validate="one_to_one")
    market, market_artifact = _market_move_table(artifacts_root)
    joined = joined.merge(market, on="game_id", how="left")
    if MARKET_MOVE_COLUMN in joined.columns:
        raw_move = pd.to_numeric(joined[MARKET_MOVE_COLUMN], errors="coerce")
    else:
        raw_move = pd.Series(np.nan, index=joined.index, dtype=float)
    joined[MOVE_AVAILABLE_COLUMN] = raw_move.notna().astype(float)
    joined[MOVE_COLUMN] = raw_move.fillna(0.0)
    served_oos = pd.Series(np.nan, index=joined.index, dtype=float)
    seasons = sorted(int(value) for value in joined["season"].unique())
    for held in seasons:
        train = joined.loc[joined["season"].ne(held)]
        test = joined.loc[joined["season"].eq(held)]
        fold_means, fold_stds = _standardisers(train)
        fold_beta = _fit_logit(
            _design(train, fold_means, fold_stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        served_oos.loc[test.index] = _predict(test, fold_beta, fold_means, fold_stds)
    assert int(served_oos.isna().sum()) == 0
    joined["served_probability"] = served_oos.to_numpy(dtype=float)
    served_record_wins = int(
        ((served_oos.ge(0.5).astype(float) == joined["home_covered"]).astype(float)).sum()
    )
    served_record = f"{served_record_wins}-{len(joined) - served_record_wins}"

    feature_matrix = joined[list(MOVE_FEATURES)].astype(float).to_numpy()
    label = joined["move"].astype(float).to_numpy()
    cover = joined["home_covered"].astype(float).to_numpy()
    residual = pd.to_numeric(joined["residual_at_open"], errors="coerce").to_numpy(dtype=float)
    opener = pd.to_numeric(joined["tue_open_home_spread"], errors="coerce").to_numpy(dtype=float)
    ridge_oos = np.empty(len(joined), dtype=float)
    gbr_oos = np.empty(len(joined), dtype=float)
    sigma_oos = np.empty(len(joined), dtype=float)
    for held in seasons:
        mask_test = joined["season"].eq(held).to_numpy()
        mask_train = ~mask_test
        train_x = feature_matrix[mask_train]
        train_y = label[mask_train]
        means = train_x.mean(axis=0)
        stds = train_x.std(axis=0, ddof=0)
        stds[stds == 0.0] = 1.0
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit((train_x - means) / stds, train_y)
        ridge_oos[mask_test] = model.predict((feature_matrix[mask_test] - means) / stds)
        booster = HistGradientBoostingRegressor(**HGB_PARAMS)
        booster.fit(train_x, train_y)
        gbr_oos[mask_test] = booster.predict(feature_matrix[mask_test])
        train_err = (
            pd.to_numeric(joined.loc[mask_train, "margin_vs_open"], errors="coerce").to_numpy(
                dtype=float
            )
            - residual[mask_train]
        )
        sigma_oos[mask_test] = float(np.std(train_err, ddof=1))
    all_means = feature_matrix.mean(axis=0)
    all_stds = feature_matrix.std(axis=0, ddof=0)
    all_stds[all_stds == 0.0] = 1.0
    ridge_is_model = Ridge(alpha=RIDGE_ALPHA)
    ridge_is_model.fit((feature_matrix - all_means) / all_stds, label)
    ridge_is = ridge_is_model.predict((feature_matrix - all_means) / all_stds)
    gbr_is_model = HistGradientBoostingRegressor(**HGB_PARAMS)
    gbr_is_model.fit(feature_matrix, label)
    gbr_is = gbr_is_model.predict(feature_matrix)
    sigma_is = float(
        np.std(
            pd.to_numeric(joined["margin_vs_open"], errors="coerce").to_numpy(dtype=float)
            - residual,
            ddof=1,
        )
    )

    def implied_probability(pred_move, sigma):
        return norm.cdf((residual - pred_move) / sigma)

    ridge_prob_oos = implied_probability(ridge_oos, sigma_oos)
    gbr_prob_oos = implied_probability(gbr_oos, sigma_oos)
    ridge_prob_is = implied_probability(ridge_is, np.full(len(joined), sigma_is))
    gbr_prob_is = implied_probability(gbr_is, np.full(len(joined), sigma_is))
    served_prob = joined["served_probability"].to_numpy(dtype=float)
    model_prob = joined["model_probability"].to_numpy(dtype=float)

    abs_move = np.abs(label)
    mae_zero = float(np.mean(abs_move))
    mae_ridge_oos = float(np.mean(np.abs(label - ridge_oos)))
    mae_gbr_oos = float(np.mean(np.abs(label - gbr_oos)))
    mae_ridge_is = float(np.mean(np.abs(label - ridge_is)))
    mae_gbr_is = float(np.mean(np.abs(label - gbr_is)))
    diff_frame = pd.DataFrame(
        {
            "season": joined["season"].to_numpy(),
            "ridge_mae_gain": abs_move - np.abs(label - ridge_oos),
            "gbr_mae_gain": abs_move - np.abs(label - gbr_oos),
            "ridge_acc_gain": ((ridge_prob_oos >= 0.5).astype(float) == cover).astype(float)
            - ((served_prob >= 0.5).astype(float) == cover).astype(float),
            "gbr_acc_gain": ((gbr_prob_oos >= 0.5).astype(float) == cover).astype(float)
            - ((served_prob >= 0.5).astype(float) == cover).astype(float),
            "ridge_brier_gain": (served_prob - cover) ** 2 - (ridge_prob_oos - cover) ** 2,
            "gbr_brier_gain": (served_prob - cover) ** 2 - (gbr_prob_oos - cover) ** 2,
            "ridge_ll_gain": log_loss_gain(ridge_prob_oos, served_prob, cover),
            "gbr_ll_gain": log_loss_gain(gbr_prob_oos, served_prob, cover),
        }
    )
    boot = {}
    for column in [c for c in diff_frame.columns if c != "season"]:
        boot[column] = season_block_bootstrap(diff_frame, column, "season", args.draws, args.seed)

    season_rows = []
    for season in seasons:
        mask = joined["season"].eq(season).to_numpy()
        games = int(mask.sum())
        row = {
            "season": season,
            "games": games,
            "mae_zero": float(np.mean(abs_move[mask])),
            "mae_ridge_oos": float(np.mean(np.abs(label[mask] - ridge_oos[mask]))),
            "mae_gbr_oos": float(np.mean(np.abs(label[mask] - gbr_oos[mask]))),
            "acc_served": accuracy_of(served_prob[mask], cover[mask]),
            "acc_model": accuracy_of(model_prob[mask], cover[mask]),
            "acc_ridge_oos": accuracy_of(ridge_prob_oos[mask], cover[mask]),
            "acc_gbr_oos": accuracy_of(gbr_prob_oos[mask], cover[mask]),
            "brier_served": brier_score(served_prob[mask], cover[mask]),
            "brier_ridge_oos": brier_score(ridge_prob_oos[mask], cover[mask]),
            "brier_gbr_oos": brier_score(gbr_prob_oos[mask], cover[mask]),
            "logloss_served": log_loss(served_prob[mask], cover[mask]),
            "logloss_ridge_oos": log_loss(ridge_prob_oos[mask], cover[mask]),
            "logloss_gbr_oos": log_loss(gbr_prob_oos[mask], cover[mask]),
        }
        season_rows.append(row)

    per_game_out = pd.DataFrame(
        {
            "game_id": joined["game_id"].astype(str),
            "season": joined["season"].astype(int),
            "week": joined["week"].astype(int),
            "tue_open_home_spread": opener,
            "close_home_spread": pd.to_numeric(
                joined["close_home_spread"], errors="coerce"
            ).to_numpy(dtype=float),
            "move": label,
            "residual_at_open": residual,
            "home_covered": cover,
            "served_probability": served_prob,
            "model_probability": model_prob,
            "ridge_move_oos": ridge_oos,
            "gbr_move_oos": gbr_oos,
            "ridge_probability_oos": ridge_prob_oos,
            "gbr_probability_oos": gbr_prob_oos,
            "ridge_move_is": ridge_is,
            "gbr_move_is": gbr_is,
        }
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    per_game_out.to_parquet(out_dir / "per_game.parquet", index=False)
    summary = {
        "design": "MKT-20 line-move target, predeclared 2026-09-16",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "opener_evaluation": "opener_evaluation/20260915T171638Z",
        "market_move_artifact": market_artifact,
        "games": len(joined),
        "seasons": seasons,
        "rows_dropped_missing": dropped,
        "served_record_reproduced": served_record,
        "served_record_anchor": "841-662",
        "label_std_move": label_std,
        "label_std_margin": margin_std,
        "move_features": list(MOVE_FEATURES),
        "leakage_guard": "features contain no close columns; label is close minus open only",
        "ridge_alpha": RIDGE_ALPHA,
        "hgb_params": HGB_PARAMS,
        "fit_ridge_served": FIT_RIDGE,
        "bootstrap": {"draws": int(args.draws), "seed": int(args.seed), "block": "season"},
        "looks": LOOKS,
        "look_family": LOOK_FAMILY,
        "mae_zero": mae_zero,
        "mae_ridge_oos": mae_ridge_oos,
        "mae_gbr_oos": mae_gbr_oos,
        "mae_ridge_is": mae_ridge_is,
        "mae_gbr_is": mae_gbr_is,
        "mae_gap_ridge_is_minus_oos": float(mae_ridge_is - mae_ridge_oos),
        "mae_gap_gbr_is_minus_oos": float(mae_gbr_is - mae_gbr_oos),
        "mae_gain_intervals": {
            "ridge": boot["ridge_mae_gain"],
            "gbr": boot["gbr_mae_gain"],
        },
        "levels_oos": {
            "acc_served": accuracy_of(served_prob, cover),
            "acc_model": accuracy_of(model_prob, cover),
            "acc_ridge": accuracy_of(ridge_prob_oos, cover),
            "acc_gbr": accuracy_of(gbr_prob_oos, cover),
            "brier_served": brier_score(served_prob, cover),
            "brier_model": brier_score(model_prob, cover),
            "brier_ridge": brier_score(ridge_prob_oos, cover),
            "brier_gbr": brier_score(gbr_prob_oos, cover),
            "logloss_served": log_loss(served_prob, cover),
            "logloss_model": log_loss(model_prob, cover),
            "logloss_ridge": log_loss(ridge_prob_oos, cover),
            "logloss_gbr": log_loss(gbr_prob_oos, cover),
        },
        "levels_is": {
            "acc_ridge": accuracy_of(ridge_prob_is, cover),
            "acc_gbr": accuracy_of(gbr_prob_is, cover),
            "brier_ridge": brier_score(ridge_prob_is, cover),
            "brier_gbr": brier_score(gbr_prob_is, cover),
            "logloss_ridge": log_loss(ridge_prob_is, cover),
            "logloss_gbr": log_loss(gbr_prob_is, cover),
        },
        "gain_vs_served_intervals": {
            "acc_ridge": boot["ridge_acc_gain"],
            "acc_gbr": boot["gbr_acc_gain"],
            "brier_ridge": boot["ridge_brier_gain"],
            "brier_gbr": boot["gbr_brier_gain"],
            "logloss_ridge": boot["ridge_ll_gain"],
            "logloss_gbr": boot["gbr_ll_gain"],
        },
        "reliability_oos": {
            "served": reliability_table(served_prob, cover),
            "ridge": reliability_table(ridge_prob_oos, cover),
            "gbr": reliability_table(gbr_prob_oos, cover),
        },
        "seasons_table": season_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(out_dir), "games": len(joined)}))


if __name__ == "__main__":
    main()
