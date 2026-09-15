from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap

POPULATION_PATH = Path("artifacts/sharp_weighted_follow/20260909T233606Z/per_game.parquet")
SEED = 20260914
SAMPLES = 20000
CONFIDENCE_BANDS = [
    ("le_0_52", -np.inf, 0.52),
    ("0_52_to_0_55", 0.52, 0.55),
    ("0_55_to_0_58", 0.55, 0.58),
    ("gt_0_58", 0.58, np.inf),
]
CALIBRATION_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def load_population() -> pd.DataFrame:
    raw = pd.read_parquet(POPULATION_PATH)
    graded = raw.loc[raw.correct_at_open_probability_rule.notna()].copy()
    graded = graded.reset_index(drop=True)
    p_home = graded.home_cover_probability_at_open.astype(float)
    p_clipped = p_home.clip(1e-6, 1.0 - 1e-6)
    card_pick_home = graded.pick_home_at_open_probability_rule.astype(bool)
    card_correct = graded.correct_at_open_probability_rule.astype(float)
    home_covered = np.where(card_pick_home, card_correct, 1.0 - card_correct)
    move = graded.leader_median_net.astype(float)
    move_thresholded = np.where(move.abs().ge(0.5), move, 0.0)
    graded["p_home"] = p_home
    graded["x1_model_logit"] = np.log(p_clipped / (1.0 - p_clipped))
    graded["card_pick_home"] = card_pick_home
    graded["card_correct"] = card_correct
    graded["home_covered"] = home_covered.astype(float)
    graded["move"] = move
    graded["move_thresholded"] = move_thresholded
    graded["confidence"] = np.maximum(p_home, 1.0 - p_home)
    graded["c1_pick_home"] = graded.s1_leader_only_pick.astype(bool)
    graded["c1_correct"] = graded.s1_leader_only_correct.astype(float)
    return graded


def fit_logit(x: np.ndarray, y: np.ndarray, *, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
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


def standardize_fit(
    train: pd.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    means = {c: float(train[c].mean()) for c in feature_cols}
    stds = {c: float(train[c].std(ddof=0)) or 1.0 for c in feature_cols}
    cols = [np.ones(len(train))]
    for c in feature_cols:
        cols.append(((train[c].astype(float) - means[c]) / stds[c]).to_numpy())
    x = np.column_stack(cols)
    y = train[target_col].astype(float).to_numpy()
    beta = fit_logit(x, y)
    return beta, means, stds


def predict_p(
    df: pd.DataFrame,
    feature_cols: list[str],
    beta: np.ndarray,
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    cols = [np.ones(len(df))]
    for c in feature_cols:
        cols.append(((df[c].astype(float) - means[c]) / stds[c]).to_numpy())
    x = np.column_stack(cols)
    z = np.clip(x @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def natural_coefficients(
    beta: np.ndarray, feature_cols: list[str], means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    out = {}
    for i, c in enumerate(feature_cols):
        out[c] = float(beta[i + 1] / stds[c])
    intercept = float(beta[0])
    for i, c in enumerate(feature_cols):
        intercept -= float(beta[i + 1]) * means[c] / stds[c]
    out["intercept"] = intercept
    return out


def loso_arm(df: pd.DataFrame, feature_cols: list[str], arm_name: str) -> dict[str, Any]:
    seasons = sorted(df.season.unique().tolist())
    fold_coefficients: dict[str, Any] = {}
    oos_p = pd.Series(index=df.index, dtype=float)
    for held_out in seasons:
        train = df.loc[df.season.ne(held_out)]
        test = df.loc[df.season.eq(held_out)]
        beta, means, stds = standardize_fit(train, feature_cols, "home_covered")
        oos_p.loc[test.index] = predict_p(test, feature_cols, beta, means, stds)
        coefs = natural_coefficients(beta, feature_cols, means, stds)
        coefs["n_train"] = len(train)
        fold_coefficients[str(held_out)] = coefs
    record_look(f"loso_{arm_name}_fit_three_folds")
    oos_pick_home = oos_p.ge(0.5)
    oos_correct = oos_pick_home.eq(df.home_covered.ge(0.5)).astype(float)

    beta_full, means_full, stds_full = standardize_fit(df, feature_cols, "home_covered")
    in_p = pd.Series(predict_p(df, feature_cols, beta_full, means_full, stds_full), index=df.index)
    in_pick_home = in_p.ge(0.5)
    in_correct = in_pick_home.eq(df.home_covered.ge(0.5)).astype(float)
    in_sample_coefficients = natural_coefficients(beta_full, feature_cols, means_full, stds_full)
    record_look(f"in_sample_{arm_name}_fit_all_799")

    return {
        "feature_cols": feature_cols,
        "fold_coefficients": fold_coefficients,
        "in_sample_coefficients": in_sample_coefficients,
        "oos_p": oos_p,
        "oos_pick_home": oos_pick_home,
        "oos_correct": oos_correct,
        "in_p": in_p,
        "in_pick_home": in_pick_home,
        "in_correct": in_correct,
    }


def paired_effect(df: pd.DataFrame, candidate_col: str, baseline_col: str) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return {
            "effect": float(
                (sub[candidate_col].astype(float).mean() - sub[baseline_col].astype(float).mean())
                * 100.0
            )
        }

    result = week_blocked_bootstrap(df, metric_fn, block="week", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def games_record(correct: pd.Series) -> str:
    wins = int(correct.sum())
    losses = int(len(correct) - wins)
    return f"{wins}-{losses}"


def diff_report(df: pd.DataFrame, arm_pick_col: str, arm_correct_col: str) -> dict[str, Any]:
    diff = df[arm_pick_col].ne(df.card_pick_home)
    n = int(diff.sum())
    card_rec = games_record(df.loc[diff, "card_correct"])
    arm_rec = games_record(df.loc[diff, arm_correct_col])
    return {"n_diff_from_card": n, "card_record_on_diff": card_rec, "arm_record_on_diff": arm_rec}


def calibration_table(df: pd.DataFrame, p_col: str) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in pairwise(CALIBRATION_BIN_EDGES):
        mask = df[p_col].ge(lo) & df[p_col].lt(hi if hi < 1.0 else 1.0 + 1e-9)
        n = int(mask.sum())
        mean_p = float(df.loc[mask, p_col].mean()) if n else float("nan")
        actual_rate = float(df.loc[mask, "home_covered"].mean()) if n else float("nan")
        rows.append(
            {
                "bin": f"{lo:.1f}_to_{hi:.1f}",
                "n": n,
                "mean_predicted_p_home": mean_p,
                "actual_home_cover_rate": actual_rate,
                "gap": (actual_rate - mean_p) if n else float("nan"),
            }
        )
    record_look("calibration_table_5_bins")
    return rows


def confidence_band_behaviour(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for name, lo, hi in CONFIDENCE_BANDS:
        in_band = df.confidence.gt(lo) & df.confidence.le(hi)
        n_games = int(in_band.sum())
        n_diff = int((in_band & df.c2_oos_pick_home.ne(df.card_pick_home)).sum())
        df_local = df.copy()
        col = f"policy_c2_band_{name}"
        df_local[col] = np.where(in_band, df.c2_oos_correct, df.card_correct)
        effect = paired_effect(df_local, col, "card_correct")
        record_look(f"confidence_band_c2_policy:{name}")
        rows.append(
            {
                "band": name,
                "n_games": n_games,
                "n_c2_diff_from_card": n_diff,
                "effect_of_c2_in_band_vs_card": effect,
            }
        )
    return rows


def season_split(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for season, sub in df.groupby("season"):
        rows.append(
            {
                "season": int(season),
                "n_games": len(sub),
                "c0_accuracy": float(sub.card_correct.mean()),
                "c1_accuracy": float(sub.c1_correct.mean()),
                "c2_oos_accuracy": float(sub.c2_oos_correct.mean()),
                "c3_oos_accuracy": float(sub.c3_oos_correct.mean()),
            }
        )
    record_look("season_split_descriptive")
    return rows


def main() -> None:
    df = load_population()

    c2 = loso_arm(df, ["x1_model_logit", "move"], "c2")
    c3 = loso_arm(df, ["x1_model_logit", "move_thresholded"], "c3")
    df["c2_oos_p"] = c2["oos_p"]
    df["c2_oos_pick_home"] = c2["oos_pick_home"]
    df["c2_oos_correct"] = c2["oos_correct"]
    df["c2_in_correct"] = c2["in_correct"]
    df["c3_oos_pick_home"] = c3["oos_pick_home"]
    df["c3_oos_correct"] = c3["oos_correct"]
    df["c3_in_correct"] = c3["in_correct"]

    records = {
        "c0": games_record(df.card_correct),
        "c1": games_record(df.c1_correct),
        "c2_oos": games_record(df.c2_oos_correct),
        "c3_oos": games_record(df.c3_oos_correct),
    }

    diffs = {
        "c1_vs_c0": diff_report(df, "c1_pick_home", "c1_correct"),
        "c2_oos_vs_c0": diff_report(df, "c2_oos_pick_home", "c2_oos_correct"),
        "c3_oos_vs_c0": diff_report(df, "c3_oos_pick_home", "c3_oos_correct"),
    }

    record_look("c1_vs_c0")
    c1_vs_c0 = paired_effect(df, "c1_correct", "card_correct")
    record_look("c2_oos_vs_c0")
    c2_oos_vs_c0 = paired_effect(df, "c2_oos_correct", "card_correct")
    record_look("c2_oos_vs_c1")
    c2_oos_vs_c1 = paired_effect(df, "c2_oos_correct", "c1_correct")
    record_look("c2_in_sample_vs_c0")
    c2_in_vs_c0 = paired_effect(df, "c2_in_correct", "card_correct")
    record_look("c3_oos_vs_c0")
    c3_oos_vs_c0 = paired_effect(df, "c3_oos_correct", "card_correct")
    record_look("c3_oos_vs_c1")
    c3_oos_vs_c1 = paired_effect(df, "c3_oos_correct", "c1_correct")
    record_look("c3_oos_vs_c2_oos")
    c3_oos_vs_c2_oos = paired_effect(df, "c3_oos_correct", "c2_oos_correct")
    record_look("c3_in_sample_vs_c0")
    c3_in_vs_c0 = paired_effect(df, "c3_in_correct", "card_correct")

    c2_diff_mask = df.c2_oos_pick_home.ne(df.card_pick_home)
    df_pc = df.copy()
    df_pc["foresight_on_c2_diff"] = np.where(
        c2_diff_mask & df.card_correct.eq(0.0), 1.0, df.card_correct
    )
    record_look("positive_control_foresight_on_c2_diff_games")
    positive_control = paired_effect(df_pc, "foresight_on_c2_diff", "card_correct")
    positive_control_detail = {
        "n_c2_diff_games": int(c2_diff_mask.sum()),
        "n_card_wrong_on_diff": int((c2_diff_mask & df.card_correct.eq(0.0)).sum()),
        "card_record_on_diff": games_record(df.loc[c2_diff_mask, "card_correct"]),
        "foresight_record_on_diff": games_record(df_pc.loc[c2_diff_mask, "foresight_on_c2_diff"]),
        "effect_vs_card": positive_control,
    }

    calibration = calibration_table(df, "c2_oos_p")
    confidence_bands = confidence_band_behaviour(df)
    seasons = season_split(df)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("artifacts/market_updated_model") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "population": str(POPULATION_PATH),
        "graded_games": len(df),
        "seed": SEED,
        "samples": SAMPLES,
        "records": records,
        "diffs": diffs,
        "c1_vs_c0": c1_vs_c0,
        "c2_oos_vs_c0": c2_oos_vs_c0,
        "c2_oos_vs_c1": c2_oos_vs_c1,
        "c2_in_sample_vs_c0": c2_in_vs_c0,
        "c2_gap_in_minus_out": float(c2_in_vs_c0["effect"] - c2_oos_vs_c0["effect"]),
        "c3_oos_vs_c0": c3_oos_vs_c0,
        "c3_oos_vs_c1": c3_oos_vs_c1,
        "c3_oos_vs_c2_oos": c3_oos_vs_c2_oos,
        "c3_in_sample_vs_c0": c3_in_vs_c0,
        "c3_gap_in_minus_out": float(c3_in_vs_c0["effect"] - c3_oos_vs_c0["effect"]),
        "positive_control_foresight_on_c2_diff": positive_control_detail,
        "c2_fold_coefficients": c2["fold_coefficients"],
        "c2_in_sample_coefficients": c2["in_sample_coefficients"],
        "c3_fold_coefficients": c3["fold_coefficients"],
        "c3_in_sample_coefficients": c3["in_sample_coefficients"],
        "calibration_c2_out_of_sample": calibration,
        "confidence_band_behaviour_c2": confidence_bands,
        "season_split": seasons,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    export_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "tue_open_home_spread",
        "p_home",
        "confidence",
        "move",
        "card_pick_home",
        "card_correct",
        "c1_pick_home",
        "c1_correct",
        "c2_oos_p",
        "c2_oos_pick_home",
        "c2_oos_correct",
        "c3_oos_pick_home",
        "c3_oos_correct",
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(out_dir / "per_game.csv", index=False)

    print(f"artifact_dir={out_dir}")
    print("records", json.dumps(records))
    print("look_count", len(LOOKS))
    print("c1_vs_c0", json.dumps(c1_vs_c0))
    print("c2_oos_vs_c0", json.dumps(c2_oos_vs_c0))
    print("c2_oos_vs_c1", json.dumps(c2_oos_vs_c1))
    print("c2_in_vs_c0", json.dumps(c2_in_vs_c0), "gap", summary["c2_gap_in_minus_out"])
    print("c3_oos_vs_c0", json.dumps(c3_oos_vs_c0))
    print("c3_oos_vs_c1", json.dumps(c3_oos_vs_c1))
    print("c3_oos_vs_c2_oos", json.dumps(c3_oos_vs_c2_oos))
    print("c3_in_vs_c0", json.dumps(c3_in_vs_c0), "gap", summary["c3_gap_in_minus_out"])
    print("positive_control", json.dumps(positive_control_detail))
    print("diffs", json.dumps(diffs))
    print("c2_fold_coefficients", json.dumps(c2["fold_coefficients"]))
    print("c3_fold_coefficients", json.dumps(c3["fold_coefficients"]))


if __name__ == "__main__":
    main()
