from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, build_fit_population

REPO = Path(__file__).resolve().parents[1]
EXTENDED_POPULATION = REPO / "artifacts/extended_fit_population/20260923T205910Z/population.parquet"
SBR_SCORED_ARTIFACT = REPO / "artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet"
SPREAD_BUCKET_EDGE_SMALL = 7.0
SPREAD_BUCKET_EDGE_LARGE = 7.5
CANDIDATE_TERMS = ["abs_open_spread", "abs_open_spread_x_model_logit"]
CANDIDATE_FEATURES = [*FIT_FEATURES, *CANDIDATE_TERMS]
SEED = 20260924
SAMPLES = 4000
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def spread_bucket(abs_spread: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [abs_spread.le(SPREAD_BUCKET_EDGE_SMALL), abs_spread.ge(SPREAD_BUCKET_EDGE_LARGE)],
            ["small_le7", "large_ge7p5"],
            default="between_7_7p5",
        ),
        index=abs_spread.index,
    )


def design(
    df: pd.DataFrame, feature_cols: list[str], means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    cols = [np.ones(len(df))]
    for c in feature_cols:
        cols.append(((df[c].astype(float) - means[c]) / stds[c]).to_numpy())
    return np.column_stack(cols)


def standardize(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    means = {c: float(df[c].mean()) for c in feature_cols}
    stds = {c: float(df[c].std(ddof=0)) or 1.0 for c in feature_cols}
    return means, stds


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
) -> tuple[pd.Series, dict[str, Any]]:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    fold_betas: dict[str, Any] = {}
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
        fold_betas[str(held)] = {
            "intercept": float(beta[0]),
            **{name: float(beta[idx + 1]) for idx, name in enumerate(feature_cols)},
            "n_train": len(train),
            "n_test": len(test),
        }
    record_look(f"loso_{arm_name}")
    return out, fold_betas


def in_sample_fit(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, arm_name: str
) -> dict[str, Any]:
    means, stds = standardize(df, feature_cols)
    beta = fit_logit_beta(
        design(df, feature_cols, means, stds), df[target_col].astype(float).to_numpy()
    )
    p = predict_logit(df, feature_cols, beta, means, stds)
    record_look(f"in_sample_fit_{arm_name}")
    betas = {
        "intercept": float(beta[0]),
        **{name: float(beta[idx + 1]) for idx, name in enumerate(feature_cols)},
    }
    metrics = prob_metrics(p, df[target_col].astype(float).to_numpy())
    return {"betas": betas, "metrics": metrics, "n": len(df)}


def prob_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pc = np.clip(p, 1e-9, 1.0 - 1e-9)
    return {
        "accuracy": float(np.mean((pc >= 0.5) == y)),
        "brier": float(np.mean((pc - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))),
    }


def games_record(correct: pd.Series) -> str:
    wins = int(correct.sum())
    return f"{wins}-{len(correct) - wins}"


def bucket_report(df: pd.DataFrame, p_col: str, target_col: str) -> list[dict[str, Any]]:
    rows = []
    for label in ("small_le7", "between_7_7p5", "large_ge7p5"):
        mask = df["bucket"].eq(label)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": label, "n": 0})
            continue
        sub = df.loc[mask]
        picked = sub[p_col].ge(0.5)
        correct = picked.astype(float).eq(sub[target_col]).astype(float)
        rows.append(
            {
                "bucket": label,
                "n": n,
                "accuracy": float(correct.mean()),
                "record": games_record(correct),
                "mean_predicted_home_prob": float(sub[p_col].mean()),
                "actual_home_cover_rate": float(sub[target_col].mean()),
                "brier": float(np.mean((sub[p_col].clip(1e-9, 1 - 1e-9) - sub[target_col]) ** 2)),
            }
        )
    record_look(f"bucket_report_{p_col}")
    return rows


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


def diff_report(
    df: pd.DataFrame, a_pick: str, a_correct: str, b_pick: str, b_correct: str
) -> dict[str, Any]:
    diff = df[a_pick].ne(df[b_pick])
    return {
        "n_diff": int(diff.sum()),
        "a_record_on_diff": games_record(df.loc[diff, a_correct]),
        "b_record_on_diff": games_record(df.loc[diff, b_correct]),
    }


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


def load_served_population() -> pd.DataFrame:
    population, _provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    population = population.copy()
    population["game_id"] = population["game_id"].astype(str)
    population["abs_open_spread"] = population["tue_open_home_spread"].abs()
    return population


def load_extended_population(served_2020_2025: pd.DataFrame) -> pd.DataFrame:
    extended = pd.read_parquet(EXTENDED_POPULATION).copy()
    extended["game_id"] = extended["game_id"].astype(str)
    sbr = pd.read_parquet(SBR_SCORED_ARTIFACT)[["game_id", "proxy_open_home_spread"]].copy()
    sbr["game_id"] = sbr["game_id"].astype(str)
    pre = extended.loc[extended["season"].le(2019)].merge(sbr, on="game_id", how="left")
    pre["abs_open_spread"] = pre["proxy_open_home_spread"].abs()
    pre = pre.drop(columns=["proxy_open_home_spread"])
    served_spread = served_2020_2025[["game_id", "abs_open_spread"]].drop_duplicates("game_id")
    post = extended.loc[extended["season"].ge(2020)].merge(served_spread, on="game_id", how="left")
    combined = pd.concat([pre, post], ignore_index=True)
    missing = int(combined["abs_open_spread"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} extended-population rows have no matched opener spread")
    return combined.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def run_population(tag: str, population: pd.DataFrame) -> dict[str, Any]:
    population = population.copy()
    population["abs_open_spread_x_model_logit"] = (
        population["abs_open_spread"] * population["model_logit"]
    )
    population["bucket"] = spread_bucket(population["abs_open_spread"])

    base_oos, base_fold_betas = loso_logit(
        population, list(FIT_FEATURES), "home_covered", f"{tag}_base_4term"
    )
    candidate_oos, candidate_fold_betas = loso_logit(
        population, CANDIDATE_FEATURES, "home_covered", f"{tag}_candidate_6term"
    )
    population["base_oos_p"] = base_oos
    population["candidate_oos_p"] = candidate_oos
    scored = population.loc[base_oos.notna() & candidate_oos.notna()].copy()

    scored["base_pick"] = scored["base_oos_p"].ge(0.5)
    scored["base_correct"] = scored["base_pick"].astype(float).eq(scored["home_covered"])
    scored["candidate_pick"] = scored["candidate_oos_p"].ge(0.5)
    scored["candidate_correct"] = scored["candidate_pick"].astype(float).eq(scored["home_covered"])

    base_metrics = prob_metrics(scored["base_oos_p"].to_numpy(), scored["home_covered"].to_numpy())
    candidate_metrics = prob_metrics(
        scored["candidate_oos_p"].to_numpy(), scored["home_covered"].to_numpy()
    )
    base_in_sample = in_sample_fit(population, list(FIT_FEATURES), "home_covered", f"{tag}_base")
    candidate_in_sample = in_sample_fit(
        population, CANDIDATE_FEATURES, "home_covered", f"{tag}_candidate"
    )

    base_bucket = bucket_report(scored, "base_oos_p", "home_covered")
    candidate_bucket = bucket_report(scored, "candidate_oos_p", "home_covered")

    effect = paired_accuracy_effect(scored, "candidate_correct", "base_correct")
    diff = diff_report(scored, "candidate_pick", "candidate_correct", "base_pick", "base_correct")
    record_look(f"{tag}_candidate_vs_base_paired_bootstrap")

    return {
        "tag": tag,
        "n_games": len(population),
        "n_scored": len(scored),
        "seasons": sorted(int(v) for v in population["season"].unique()),
        "base_bucket_report": base_bucket,
        "candidate_bucket_report": candidate_bucket,
        "base_overall_metrics": base_metrics,
        "candidate_overall_metrics": candidate_metrics,
        "base_records": games_record(scored["base_correct"]),
        "candidate_records": games_record(scored["candidate_correct"]),
        "base_in_sample_fit": base_in_sample,
        "candidate_in_sample_fit": candidate_in_sample,
        "base_in_sample_vs_oos_gap": {
            "accuracy": base_in_sample["metrics"]["accuracy"] - base_metrics["accuracy"],
            "brier": base_in_sample["metrics"]["brier"] - base_metrics["brier"],
            "log_loss": base_in_sample["metrics"]["log_loss"] - base_metrics["log_loss"],
        },
        "candidate_in_sample_vs_oos_gap": {
            "accuracy": candidate_in_sample["metrics"]["accuracy"] - candidate_metrics["accuracy"],
            "brier": candidate_in_sample["metrics"]["brier"] - candidate_metrics["brier"],
            "log_loss": candidate_in_sample["metrics"]["log_loss"] - candidate_metrics["log_loss"],
        },
        "candidate_vs_base_effect": effect,
        "candidate_vs_base_diff": diff,
        "candidate_calibration": calibration_table(scored, "candidate_oos_p", "home_covered"),
        "base_calibration": calibration_table(scored, "base_oos_p", "home_covered"),
        "base_fold_betas": base_fold_betas,
        "candidate_fold_betas": candidate_fold_betas,
    }, scored


def week3_pick_flips(population_2020_2025: pd.DataFrame) -> dict[str, Any]:
    week3 = population_2020_2025.loc[
        population_2020_2025["season"].eq(2025) & population_2020_2025["week"].eq(3)
    ].copy()
    if week3.empty:
        return {"note": "no season=2025 week=3 games in the fit population (ungraded/current card)"}
    means, stds = standardize(population_2020_2025, list(FIT_FEATURES))
    base_beta = fit_logit_beta(
        design(population_2020_2025, list(FIT_FEATURES), means, stds),
        population_2020_2025["home_covered"].astype(float).to_numpy(),
    )
    week3["base_full_sample_p"] = predict_logit(week3, list(FIT_FEATURES), base_beta, means, stds)

    population_2020_2025 = population_2020_2025.copy()
    population_2020_2025["abs_open_spread_x_model_logit"] = (
        population_2020_2025["abs_open_spread"] * population_2020_2025["model_logit"]
    )
    week3["abs_open_spread_x_model_logit"] = week3["abs_open_spread"] * week3["model_logit"]
    cmeans, cstds = standardize(population_2020_2025, CANDIDATE_FEATURES)
    candidate_beta = fit_logit_beta(
        design(population_2020_2025, CANDIDATE_FEATURES, cmeans, cstds),
        population_2020_2025["home_covered"].astype(float).to_numpy(),
    )
    week3["candidate_full_sample_p"] = predict_logit(
        week3, CANDIDATE_FEATURES, candidate_beta, cmeans, cstds
    )
    week3["base_pick_home"] = week3["base_full_sample_p"].ge(0.5)
    week3["candidate_pick_home"] = week3["candidate_full_sample_p"].ge(0.5)
    week3["flips"] = week3["base_pick_home"].ne(week3["candidate_pick_home"])
    record_look("week3_full_sample_refit_pick_check")
    cols = [
        "game_id",
        "home_team",
        "away_team",
        "abs_open_spread",
        "base_full_sample_p",
        "candidate_full_sample_p",
        "base_pick_home",
        "candidate_pick_home",
        "flips",
    ]
    cols = [c for c in cols if c in week3.columns]
    return {
        "n_week3_games": len(week3),
        "n_flips": int(week3["flips"].sum()),
        "rows": week3[cols].to_dict(orient="records"),
    }


def main() -> None:
    served = load_served_population()
    extended = load_extended_population(served)

    result_2020_2025, scored_2020_2025 = run_population("served_2020_2025", served)
    result_2011_2025, scored_2011_2025 = run_population("extended_2011_2025", extended)
    week3 = week3_pick_flips(served)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts/mod18_spread_regime" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "unit": 1,
        "predeclared_candidate_terms": CANDIDATE_TERMS,
        "predeclared_bucket_edges": {
            "small_le": SPREAD_BUCKET_EDGE_SMALL,
            "large_ge": SPREAD_BUCKET_EDGE_LARGE,
        },
        "ridge": FIT_RIDGE,
        "seed": SEED,
        "samples": SAMPLES,
        "served_2020_2025": result_2020_2025,
        "extended_2011_2025": result_2011_2025,
        "week3_pick_check": week3,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    export_cols_2020 = [
        c
        for c in [
            "game_id",
            "season",
            "week",
            "abs_open_spread",
            "bucket",
            "home_covered",
            "base_oos_p",
            "candidate_oos_p",
        ]
        if c in scored_2020_2025.columns
    ]
    scored_2020_2025[export_cols_2020].to_csv(out_dir / "per_game_2020_2025.csv", index=False)
    export_cols_ext = [
        c
        for c in [
            "game_id",
            "season",
            "week",
            "opener_source",
            "abs_open_spread",
            "bucket",
            "home_covered",
            "base_oos_p",
            "candidate_oos_p",
        ]
        if c in scored_2011_2025.columns
    ]
    scored_2011_2025[export_cols_ext].to_csv(out_dir / "per_game_2011_2025.csv", index=False)

    print(f"artifact_dir={out_dir}")
    print("served_2020_2025 base_bucket", json.dumps(result_2020_2025["base_bucket_report"]))
    print(
        "served_2020_2025 candidate_vs_base",
        json.dumps(result_2020_2025["candidate_vs_base_effect"]),
        result_2020_2025["candidate_vs_base_diff"],
    )
    print("extended_2011_2025 base_bucket", json.dumps(result_2011_2025["base_bucket_report"]))
    print(
        "extended_2011_2025 candidate_vs_base",
        json.dumps(result_2011_2025["candidate_vs_base_effect"]),
        result_2011_2025["candidate_vs_base_diff"],
    )
    print("week3_pick_check", json.dumps(week3))
    print("look_count", len(LOOKS))


if __name__ == "__main__":
    main()
