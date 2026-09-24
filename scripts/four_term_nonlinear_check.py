from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
WATERFALL_FEED = REPO / "artifacts/waterfall_feed/20260924T161758Z/feed.json"
SEED = 20260924
SAMPLES = 4000
N1_PARAMS = {
    "max_depth": 2,
    "max_iter": 100,
    "learning_rate": 0.05,
    "min_samples_leaf": 50,
    "random_state": SEED,
}
INTERACTION_TERMS = [
    ("model_logit", FLAG_SUM_COLUMN),
    ("model_logit", MOVE_COLUMN),
    ("model_logit", MOVE_AVAILABLE_COLUMN),
    (FLAG_SUM_COLUMN, MOVE_COLUMN),
    (FLAG_SUM_COLUMN, MOVE_AVAILABLE_COLUMN),
    (MOVE_COLUMN, MOVE_AVAILABLE_COLUMN),
]
N2_INTERACTION_COLUMNS = [f"int_{a}_x_{b}" for a, b in INTERACTION_TERMS]
N2_FEATURES = [*FIT_FEATURES, *N2_INTERACTION_COLUMNS]
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for (a, b), name in zip(INTERACTION_TERMS, N2_INTERACTION_COLUMNS, strict=True):
        out[name] = out[a].astype(float) * out[b].astype(float)
    return out


def standardize(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    means = {c: float(df[c].mean()) for c in feature_cols}
    stds = {c: float(df[c].std(ddof=0)) or 1.0 for c in feature_cols}
    return means, stds


def design(
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
    z = np.clip(design(df, feature_cols, means, stds) @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def loso_logit(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, arm_name: str
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for held in sorted(int(v) for v in df["season"].unique()):
        train = df.loc[df["season"].ne(held)]
        test = df.loc[df["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = standardize(train, feature_cols)
        beta = fit_logit_beta(
            design(train, feature_cols, means, stds), train[target_col].astype(float).to_numpy()
        )
        out.loc[test.index] = predict_logit(test, feature_cols, beta, means, stds)
    record_look(f"loso_{arm_name}")
    return out


def loso_hgb(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for held in sorted(int(v) for v in df["season"].unique()):
        train = df.loc[df["season"].ne(held)]
        test = df.loc[df["season"].eq(held)]
        if train.empty or test.empty:
            continue
        clf = HistGradientBoostingClassifier(**N1_PARAMS)
        clf.fit(
            train[feature_cols].astype(float).to_numpy(),
            train[target_col].astype(float).to_numpy(),
        )
        positive_index = int(np.where(clf.classes_ == 1.0)[0][0])
        out.loc[test.index] = clf.predict_proba(test[feature_cols].astype(float).to_numpy())[
            :, positive_index
        ]
    record_look("loso_n1_hgb")
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


def week3_2026_pick_check(
    base_beta: np.ndarray,
    base_means: dict,
    base_stds: dict,
    n1_model: HistGradientBoostingClassifier,
    n2_beta: np.ndarray,
    n2_means: dict,
    n2_stds: dict,
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
            f"schedule snapshot matched {len(matched)}/{len(games)} week-3-2026 feed games"
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
    frame = add_interactions(frame)

    frame["base_home_probability"] = predict_logit(
        frame, list(FIT_FEATURES), base_beta, base_means, base_stds
    )
    n1_positive_index = int(np.where(n1_model.classes_ == 1.0)[0][0])
    frame["n1_home_probability"] = n1_model.predict_proba(
        frame[list(FIT_FEATURES)].astype(float).to_numpy()
    )[:, n1_positive_index]
    frame["n2_home_probability"] = predict_logit(
        frame, N2_FEATURES, n2_beta, n2_means, n2_stds
    )
    frame["base_pick_home"] = frame["base_home_probability"].ge(0.5)
    frame["n1_pick_home"] = frame["n1_home_probability"].ge(0.5)
    frame["n2_pick_home"] = frame["n2_home_probability"].ge(0.5)
    frame["n1_flips"] = frame["base_pick_home"].ne(frame["n1_pick_home"])
    frame["n2_flips"] = frame["base_pick_home"].ne(frame["n2_pick_home"])
    record_look("week3_2026_pick_check_full_sample_refit")
    cols = [
        "game_id",
        "home_team",
        "away_team",
        FLAG_SUM_COLUMN,
        MOVE_COLUMN,
        MOVE_AVAILABLE_COLUMN,
        "base_home_probability",
        "n1_home_probability",
        "n2_home_probability",
        "base_pick_home",
        "n1_pick_home",
        "n2_pick_home",
        "n1_flips",
        "n2_flips",
    ]
    return {
        "n_games": len(frame),
        "n_flips_n1_vs_base": int(frame["n1_flips"].sum()),
        "n_flips_n2_vs_base": int(frame["n2_flips"].sum()),
        "note": (
            "model_logit read from waterfall_feed discrete_base_home_cover_probability at the "
            "currently tracked line, not confirmed identical to the true Tuesday-open value used "
            "for 2020-2025 training rows; pick-flip check only, week 3 2026 is ungraded"
        ),
        "rows": frame[cols].to_dict(orient="records"),
    }


def main() -> None:
    served, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    served = served.copy()
    served["game_id"] = served["game_id"].astype(str)
    served = add_interactions(served)

    base_oos = loso_logit(served, list(FIT_FEATURES), "home_covered", "base")
    n1_oos = loso_hgb(served, list(FIT_FEATURES), "home_covered")
    n2_oos = loso_logit(served, N2_FEATURES, "home_covered", "n2")

    served["base_oos_p"] = base_oos
    served["n1_oos_p"] = n1_oos
    served["n2_oos_p"] = n2_oos
    scored = served.loc[base_oos.notna() & n1_oos.notna() & n2_oos.notna()].copy()

    for label in ("base", "n1", "n2"):
        col = f"{label}_oos_p"
        scored[f"{label}_pick_home"] = scored[col].ge(0.5)
        scored[f"{label}_correct"] = scored[f"{label}_pick_home"].astype(float).eq(
            scored["home_covered"]
        )

    base_metrics = prob_metrics(scored["base_oos_p"].to_numpy(), scored["home_covered"].to_numpy())
    n1_metrics = prob_metrics(scored["n1_oos_p"].to_numpy(), scored["home_covered"].to_numpy())
    n2_metrics = prob_metrics(scored["n2_oos_p"].to_numpy(), scored["home_covered"].to_numpy())

    n1_vs_base_accuracy = paired_accuracy_effect(scored, "n1_correct", "base_correct")
    n2_vs_base_accuracy = paired_accuracy_effect(scored, "n2_correct", "base_correct")
    n1_vs_base_prob = paired_brier_logloss_effect(scored, "base_oos_p", "n1_oos_p", "home_covered")
    n2_vs_base_prob = paired_brier_logloss_effect(scored, "base_oos_p", "n2_oos_p", "home_covered")

    n1_vs_base_decisive = decisive_report(
        scored, "base_pick_home", "base_correct", "n1_pick_home", "n1_correct"
    )
    n2_vs_base_decisive = decisive_report(
        scored, "base_pick_home", "base_correct", "n2_pick_home", "n2_correct"
    )

    base_calibration = calibration_table(scored, "base_oos_p", "home_covered")
    n1_calibration = calibration_table(scored, "n1_oos_p", "home_covered")
    n2_calibration = calibration_table(scored, "n2_oos_p", "home_covered")

    base_means, base_stds = standardize(served, list(FIT_FEATURES))
    base_full_beta = fit_logit_beta(
        design(served, list(FIT_FEATURES), base_means, base_stds),
        served["home_covered"].astype(float).to_numpy(),
    )
    n1_full_model = HistGradientBoostingClassifier(**N1_PARAMS)
    n1_full_model.fit(
        served[list(FIT_FEATURES)].astype(float).to_numpy(),
        served["home_covered"].astype(float).to_numpy(),
    )
    n2_means, n2_stds = standardize(served, N2_FEATURES)
    n2_full_beta = fit_logit_beta(
        design(served, N2_FEATURES, n2_means, n2_stds),
        served["home_covered"].astype(float).to_numpy(),
    )
    record_look("full_sample_refit_base_n1_n2_for_week3_2026")
    week3_2026 = week3_2026_pick_check(
        base_full_beta,
        base_means,
        base_stds,
        n1_full_model,
        n2_full_beta,
        n2_means,
        n2_stds,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts/four_term_nonlinear_check" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "question": "does nonlinear structure in the same four served terms "
        "(model_logit, flag_sum, move_toward_home, move_available) beat the served ridge "
        "logistic out of season, on the identical 1503 2020-2025 opener-graded games",
        "fit_features": list(FIT_FEATURES),
        "n1_hgb_params": N1_PARAMS,
        "n2_interaction_columns": N2_INTERACTION_COLUMNS,
        "ridge": FIT_RIDGE,
        "seed": SEED,
        "samples": SAMPLES,
        "n_scored": len(scored),
        "base_served_ridge_logistic": {
            "metrics": base_metrics,
            "record": games_record(scored["base_correct"]),
            "calibration": base_calibration,
        },
        "n1_hgb_trees": {
            "metrics": n1_metrics,
            "record": games_record(scored["n1_correct"]),
            "calibration": n1_calibration,
        },
        "n2_logistic_pairwise_interactions": {
            "metrics": n2_metrics,
            "record": games_record(scored["n2_correct"]),
            "calibration": n2_calibration,
        },
        "n1_vs_base_accuracy_effect": n1_vs_base_accuracy,
        "n2_vs_base_accuracy_effect": n2_vs_base_accuracy,
        "n1_vs_base_brier_logloss_effect": n1_vs_base_prob,
        "n2_vs_base_brier_logloss_effect": n2_vs_base_prob,
        "n1_vs_base_decisive": n1_vs_base_decisive,
        "n2_vs_base_decisive": n2_vs_base_decisive,
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
        "base_oos_p",
        "n1_oos_p",
        "n2_oos_p",
        "base_correct",
        "n1_correct",
        "n2_correct",
    ]
    scored[export_cols].to_csv(out_dir / "per_game.csv", index=False)
    print(str(out_dir))


if __name__ == "__main__":
    main()
