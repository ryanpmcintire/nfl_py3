from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.pick_probability import (
    FLAG_SUM_COLUMN,
    MARKET_MOVE_FEATURE_LEGACY,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    market_move_toward_home,
    signed_composition_flags_fail_open,
)
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, build_fit_population
from nfl_ats.snapshots import latest_snapshot, load_snapshot

REPO = Path(__file__).resolve().parents[1]
EXTENDED_POPULATION = REPO / "artifacts/extended_fit_population/20260923T205910Z/population.parquet"
WATERFALL_FEED = REPO / "artifacts/waterfall_feed/20260924T161758Z/feed.json"
SEED = 20260924
SAMPLES = 4000
HELD_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
PRE2020_INDICATOR = "pre_2020_indicator"
PRE2020_INTERACTION = "pre_2020_x_model_logit"
C_FEATURES = [*FIT_FEATURES, PRE2020_INDICATOR, PRE2020_INTERACTION]
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def design(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    cols = [np.ones(len(df))]
    for c in feature_cols:
        cols.append(df[c].astype(float).to_numpy())
    return np.column_stack(cols)


def standardize(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    means = {c: float(df[c].mean()) for c in feature_cols}
    stds = {c: float(df[c].std(ddof=0)) or 1.0 for c in feature_cols}
    return means, stds


def design_standardised(
    df: pd.DataFrame, feature_cols: list[str], means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    cols = [np.ones(len(df))]
    for c in feature_cols:
        cols.append(((df[c].astype(float) - means[c]) / stds[c]).to_numpy())
    return np.column_stack(cols)


def fit_logit_beta(
    x: np.ndarray, y: np.ndarray, l2: float = FIT_RIDGE, iters: int = 50
) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        z = np.clip(x @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = x.T @ (y - p) - l2 * beta
        hessian = (x.T * w) @ x + l2 * np.eye(x.shape[1])
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        beta = beta + step
    return beta


def predict_logit(
    df: pd.DataFrame, feature_cols: list[str], beta: np.ndarray, means: dict, stds: dict
) -> np.ndarray:
    z = np.clip(design_standardised(df, feature_cols, means, stds) @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def loso_fixed_test_set(
    train_population: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    held_seasons: tuple[int, ...],
    arm_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    predictions: dict[str, np.ndarray] = {}
    fold_betas: dict[str, dict[str, Any]] = {}
    for held in held_seasons:
        train = train_population.loc[train_population["season"].ne(held)]
        test = train_population.loc[train_population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = standardize(train, feature_cols)
        beta = fit_logit_beta(
            design_standardised(train, feature_cols, means, stds),
            train[target_col].astype(float).to_numpy(),
        )
        p = predict_logit(test, feature_cols, beta, means, stds)
        predictions[str(held)] = np.column_stack([test["game_id"].to_numpy(), p])
        fold_betas[str(held)] = {
            "intercept": float(beta[0]),
            **{name: float(beta[idx + 1]) for idx, name in enumerate(feature_cols)},
            "n_train": len(train),
            "n_test": len(test),
        }
    record_look(f"loso_{arm_name}")
    return predictions, fold_betas


def merge_predictions(
    served: pd.DataFrame, predictions: dict[str, np.ndarray], column: str
) -> pd.DataFrame:
    rows = []
    for block in predictions.values():
        rows.append(pd.DataFrame({"game_id": block[:, 0], column: block[:, 1].astype(float)}))
    stacked = pd.concat(rows, ignore_index=True)
    out = served.merge(stacked, on="game_id", how="left", validate="one_to_one")
    return out


def prob_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pc = np.clip(p, 1e-9, 1.0 - 1e-9)
    return {
        "n": len(y),
        "accuracy": float(np.mean((pc >= 0.5) == y)),
        "brier": float(np.mean((pc - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))),
    }


def games_record(correct: pd.Series) -> str:
    wins = int(pd.to_numeric(correct, errors="coerce").fillna(0.0).sum())
    return f"{wins}-{len(correct) - wins}"


def calibration_table(df: pd.DataFrame, p_col: str, target_col: str) -> list[dict[str, Any]]:
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for lo, hi in pairwise(edges):
        mask = df[p_col].ge(lo) & df[p_col].lt(hi if hi < 1.0 else 1.0 + 1e-9)
        n = int(mask.sum())
        rows.append(
            {
                "bin": f"{lo:.1f}_to_{hi:.1f}",
                "n": n,
                "mean_predicted": float(df.loc[mask, p_col].mean()) if n else None,
                "actual_rate": float(df.loc[mask, target_col].mean()) if n else None,
            }
        )
    record_look(f"calibration_table_{p_col}")
    return rows


def paired_accuracy_effect(
    df: pd.DataFrame, candidate_correct: str, baseline_correct: str
) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return {
            "effect": float(
                (
                    sub[candidate_correct].astype(float).mean()
                    - sub[baseline_correct].astype(float).mean()
                )
                * 100.0
            )
        }

    result = week_blocked_bootstrap(df, metric_fn, block="season", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect_accuracy_points": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def paired_brier_logloss_effect(
    df: pd.DataFrame, a_col: str, b_col: str, target_col: str
) -> dict[str, Any]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        pa = sub[a_col].to_numpy(dtype=float).clip(1e-9, 1.0 - 1e-9)
        pb = sub[b_col].to_numpy(dtype=float).clip(1e-9, 1.0 - 1e-9)
        y = sub[target_col].to_numpy(dtype=float)
        return {
            "brier_effect": float(np.mean((pb - y) ** 2) - np.mean((pa - y) ** 2)),
            "log_loss_effect": float(
                -np.mean(y * np.log(pb) + (1.0 - y) * np.log1p(-pb))
                - -np.mean(y * np.log(pa) + (1.0 - y) * np.log1p(-pa))
            ),
        }

    result = week_blocked_bootstrap(df, metric_fn, block="season", samples=SAMPLES, seed=SEED)
    out = {}
    for _, row in result.iterrows():
        out[row["metric"]] = {
            "estimate": float(row["estimate"]),
            "interval_low": float(row["lower"]),
            "interval_high": float(row["upper"]),
            "probability_positive": float(row["probability_positive"]),
        }
    record_look("paired_brier_logloss_effect_b_vs_a")
    return out


def decisive_report(
    scored: pd.DataFrame, a_pick: str, a_correct: str, b_pick: str, b_correct: str
) -> dict[str, Any]:
    decisive = scored.loc[scored[a_pick].ne(scored[b_pick])]
    return {
        "n_decisive": len(decisive),
        "a_record_on_decisive": games_record(decisive[a_correct]),
        "b_record_on_decisive": games_record(decisive[b_correct]),
    }


def load_extended_population() -> pd.DataFrame:
    extended = pd.read_parquet(EXTENDED_POPULATION).copy()
    extended["game_id"] = extended["game_id"].astype(str)
    extended[PRE2020_INDICATOR] = extended["season"].lt(2020).astype(float)
    extended[PRE2020_INTERACTION] = extended[PRE2020_INDICATOR] * extended["model_logit"].astype(
        float
    )
    return extended


def run_arm(
    served: pd.DataFrame,
    train_population: pd.DataFrame,
    feature_cols: list[str],
    arm_name: str,
    p_column: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    predictions, fold_betas = loso_fixed_test_set(
        train_population, feature_cols, "home_covered", HELD_SEASONS, arm_name
    )
    merged = merge_predictions(served[["game_id"]], predictions, p_column)
    return merged, fold_betas


def week3_2026_pick_check(
    a_beta: np.ndarray,
    a_means: dict,
    a_stds: dict,
    b_beta: np.ndarray,
    b_means: dict,
    b_stds: dict,
) -> dict[str, Any]:
    feed = json.loads(WATERFALL_FEED.read_text(encoding="utf-8"))
    games = pd.DataFrame(feed["games"])
    games["game_id"] = games["game_id"].astype(str)
    games["season"] = int(feed["season"])
    games["week"] = int(feed["week"])
    p = games["discrete_base_home_cover_probability"].astype(float).clip(1e-9, 1.0 - 1e-9)
    games["model_logit"] = np.log(p / (1.0 - p))
    games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True, errors="coerce")
    games["gameday"] = games["kickoff"].dt.tz_convert("US/Eastern").dt.date.astype(str)

    data_root = REPO / "data"
    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    schedules = schedules[
        ["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team"]
    ].drop_duplicates("game_id")
    schedules["game_id"] = schedules["game_id"].astype(str)
    matched = schedules.merge(games[["game_id"]], on="game_id", how="inner")
    if len(matched) != len(games):
        raise RuntimeError(
            f"schedule snapshot matched {len(matched)}/{len(games)} week-3-2026 feed games; "
            "cannot build predictions frame for the composition-flag/market-move call"
        )
    predictions_frame = matched.merge(
        games[["game_id", "model_logit", "market_line", "picked_side"]], on="game_id", how="left"
    )
    predictions_frame["kickoff"] = (
        games.set_index("game_id").loc[predictions_frame["game_id"], "kickoff"].to_numpy()
    )

    flags = signed_composition_flags_fail_open(predictions_frame, schedules)
    move = market_move_toward_home(
        predictions_frame, data_root, feature_version=MARKET_MOVE_FEATURE_LEGACY
    )
    frame = predictions_frame.merge(flags, on="game_id", how="left").merge(
        move, on="game_id", how="left"
    )
    frame[MOVE_AVAILABLE_COLUMN] = pd.to_numeric(
        frame[MOVE_AVAILABLE_COLUMN], errors="coerce"
    ).fillna(0.0)
    frame[MOVE_COLUMN] = pd.to_numeric(frame[MOVE_COLUMN], errors="coerce").fillna(0.0)
    frame[FLAG_SUM_COLUMN] = pd.to_numeric(frame[FLAG_SUM_COLUMN], errors="coerce").fillna(0.0)

    frame["a_home_probability"] = predict_logit(frame, list(FIT_FEATURES), a_beta, a_means, a_stds)
    frame["b_home_probability"] = predict_logit(frame, list(FIT_FEATURES), b_beta, b_means, b_stds)
    frame["a_pick_home"] = frame["a_home_probability"].ge(0.5)
    frame["b_pick_home"] = frame["b_home_probability"].ge(0.5)
    frame["flips"] = frame["a_pick_home"].ne(frame["b_pick_home"])
    record_look("week3_2026_pick_check_full_sample_refit")
    cols = [
        "game_id",
        "home_team",
        "away_team",
        FLAG_SUM_COLUMN,
        MOVE_COLUMN,
        MOVE_AVAILABLE_COLUMN,
        "a_home_probability",
        "b_home_probability",
        "a_pick_home",
        "b_pick_home",
        "flips",
    ]
    return {
        "n_games": len(frame),
        "n_flips": int(frame["flips"].sum()),
        "note": (
            "model_logit read from waterfall_feed discrete_base_home_cover_probability at the "
            "currently tracked line, not confirmed identical to the true Tuesday-open value used "
            "for 2020-2025 training rows; pick-flip check only, not an accuracy comparison "
            "(week 3 2026 is ungraded)"
        ),
        "rows": frame[cols].to_dict(orient="records"),
    }


def main() -> None:
    served, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    served = served.copy()
    served["game_id"] = served["game_id"].astype(str)
    extended = load_extended_population()

    arm_a_p, arm_a_betas = run_arm(served, served, list(FIT_FEATURES), "arm_a_served", "a_oos_p")
    arm_b_p, arm_b_betas = run_arm(
        served, extended, list(FIT_FEATURES), "arm_b_extended", "b_oos_p"
    )
    arm_c_p, arm_c_betas = run_arm(
        served, extended, C_FEATURES, "arm_c_extended_pre2020_interaction", "c_oos_p"
    )

    scored = served.merge(arm_a_p, on="game_id", how="left")
    scored = scored.merge(arm_b_p[["game_id", "b_oos_p"]], on="game_id", how="left")
    scored = scored.merge(arm_c_p[["game_id", "c_oos_p"]], on="game_id", how="left")
    scored = scored.loc[
        scored["a_oos_p"].notna() & scored["b_oos_p"].notna() & scored["c_oos_p"].notna()
    ].copy()
    if len(scored) != 1503:
        raise RuntimeError(f"expected 1503 identically-scored games, got {len(scored)}")

    for label in ("a", "b", "c"):
        col = f"{label}_oos_p"
        scored[f"{label}_pick_home"] = scored[col].ge(0.5)
        scored[f"{label}_correct"] = (
            scored[f"{label}_pick_home"].astype(float).eq(scored["home_covered"])
        )

    a_metrics = prob_metrics(scored["a_oos_p"].to_numpy(), scored["home_covered"].to_numpy())
    b_metrics = prob_metrics(scored["b_oos_p"].to_numpy(), scored["home_covered"].to_numpy())
    c_metrics = prob_metrics(scored["c_oos_p"].to_numpy(), scored["home_covered"].to_numpy())

    b_vs_a_accuracy_effect = paired_accuracy_effect(scored, "b_correct", "a_correct")
    c_vs_a_accuracy_effect = paired_accuracy_effect(scored, "c_correct", "a_correct")
    c_vs_b_accuracy_effect = paired_accuracy_effect(scored, "c_correct", "b_correct")
    b_vs_a_prob_effect = paired_brier_logloss_effect(scored, "a_oos_p", "b_oos_p", "home_covered")
    c_vs_a_prob_effect = paired_brier_logloss_effect(scored, "a_oos_p", "c_oos_p", "home_covered")

    b_vs_a_decisive = decisive_report(
        scored, "a_pick_home", "a_correct", "b_pick_home", "b_correct"
    )
    c_vs_a_decisive = decisive_report(
        scored, "a_pick_home", "a_correct", "c_pick_home", "c_correct"
    )

    a_calibration = calibration_table(scored, "a_oos_p", "home_covered")
    b_calibration = calibration_table(scored, "b_oos_p", "home_covered")
    c_calibration = calibration_table(scored, "c_oos_p", "home_covered")

    a_means, a_stds = standardize(served, list(FIT_FEATURES))
    a_full_beta = fit_logit_beta(
        design_standardised(served, list(FIT_FEATURES), a_means, a_stds),
        served["home_covered"].astype(float).to_numpy(),
    )
    b_means, b_stds = standardize(extended, list(FIT_FEATURES))
    b_full_beta = fit_logit_beta(
        design_standardised(extended, list(FIT_FEATURES), b_means, b_stds),
        extended["home_covered"].astype(float).to_numpy(),
    )
    record_look("full_sample_refit_a_and_b_for_week3_2026")
    week3_2026 = week3_2026_pick_check(a_full_beta, a_means, a_stds, b_full_beta, b_means, b_stds)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts/four_term_extended_training" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "question": "does training FIT_FEATURES on 2011-2025 (minus holdout) beat training on "
        "2020-2025 (minus holdout) alone, scored on the identical 1503 2020-2025 opener-graded "
        "games",
        "fit_features": list(FIT_FEATURES),
        "c_features": C_FEATURES,
        "ridge": FIT_RIDGE,
        "seed": SEED,
        "samples": SAMPLES,
        "held_seasons": list(HELD_SEASONS),
        "n_scored": len(scored),
        "a_served_2020_2025_training": {
            "metrics": a_metrics,
            "record": games_record(scored["a_correct"]),
            "calibration": a_calibration,
            "fold_betas": arm_a_betas,
        },
        "b_extended_2011_2025_training": {
            "metrics": b_metrics,
            "record": games_record(scored["b_correct"]),
            "calibration": b_calibration,
            "fold_betas": arm_b_betas,
        },
        "c_extended_pre2020_interaction_training": {
            "metrics": c_metrics,
            "record": games_record(scored["c_correct"]),
            "calibration": c_calibration,
            "fold_betas": arm_c_betas,
        },
        "b_vs_a_accuracy_effect": b_vs_a_accuracy_effect,
        "c_vs_a_accuracy_effect": c_vs_a_accuracy_effect,
        "c_vs_b_accuracy_effect": c_vs_b_accuracy_effect,
        "b_vs_a_brier_logloss_effect": b_vs_a_prob_effect,
        "c_vs_a_brier_logloss_effect": c_vs_a_prob_effect,
        "b_vs_a_decisive": b_vs_a_decisive,
        "c_vs_a_decisive": c_vs_a_decisive,
        "week3_2026_pick_check": week3_2026,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
        "served_population_provenance": provenance,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    export_cols = [
        "game_id",
        "season",
        "week",
        "home_covered",
        "a_oos_p",
        "b_oos_p",
        "c_oos_p",
        "a_correct",
        "b_correct",
        "c_correct",
    ]
    scored[export_cols].to_csv(out_dir / "per_game.csv", index=False)
    print(str(out_dir))


if __name__ == "__main__":
    main()
