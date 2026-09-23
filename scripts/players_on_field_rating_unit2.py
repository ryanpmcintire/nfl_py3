from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from players_on_field_rating_eval import (
    ARTIFACTS_ROOT,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DATA_ROOT,
    INTERVAL_LEVEL,
    RATINGS_TABLE,
    RELIABILITY_EDGES,
    REPO_ROOT,
    design_matrix,
    in_sample_fit,
    loso,
    natural_coefficients,
    predict,
    standardisers,
)

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, build_fit_population
from nfl_ats.pick_probability_fit import _fit_logit as fit_logit
from nfl_ats.signal_atlas import _cell as signal_cell
from nfl_ats.signal_atlas import _metrics as signal_metrics

OUTPUT_ROOT = ARTIFACTS_ROOT / "players_on_field_rating_unit2"


def residualized_oos_term(
    population: pd.DataFrame,
    base_features: tuple[str, ...],
    predictor_col: str,
    construct_col: str,
) -> tuple[pd.Series, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    term_name = "residual_term"
    features = (*base_features, term_name)
    seasons = sorted(int(value) for value in population["season"].unique())
    out = pd.Series(np.nan, index=population.index, dtype=float)
    fit_folds: dict[str, dict[str, float]] = {}
    residual_folds: dict[str, dict[str, float]] = {}
    for held in seasons:
        train_mask = population["season"].ne(held)
        test_mask = population["season"].eq(held)
        train = population.loc[train_mask].copy()
        test = population.loc[test_mask].copy()
        if train.empty or test.empty:
            continue
        x_train = train[predictor_col].to_numpy(dtype=float)
        y_train = train[construct_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_train, y_train, 1)
        train[term_name] = y_train - (intercept + slope * x_train)
        x_test = test[predictor_col].to_numpy(dtype=float)
        y_test = test[construct_col].to_numpy(dtype=float)
        test[term_name] = y_test - (intercept + slope * x_test)
        means, stds = standardisers(train, features)
        beta = fit_logit(
            design_matrix(train, features, means, stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        out.loc[test.index] = predict(test, features, beta, means, stds)
        fit_folds[str(held)] = natural_coefficients(beta, features, means, stds)
        residual_folds[str(held)] = {"slope": float(slope), "intercept": float(intercept)}
    return out, fit_folds, residual_folds


def residualized_is_term(
    population: pd.DataFrame,
    base_features: tuple[str, ...],
    predictor_col: str,
    construct_col: str,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    x_all = population[predictor_col].to_numpy(dtype=float)
    y_all = population[construct_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_all, y_all, 1)
    frame = population.copy()
    frame["residual_term"] = y_all - (intercept + slope * x_all)
    features = (*base_features, "residual_term")
    predicted, coefficients = in_sample_fit(frame, features)
    return predicted, coefficients, {"slope": float(slope), "intercept": float(intercept)}


def evaluate_candidate(
    population: pd.DataFrame,
    candidate_oos: pd.Series,
    candidate_is: np.ndarray,
    candidate_folds: dict[str, dict[str, float]],
    base_folds: dict[str, dict[str, float]],
    base_is: np.ndarray,
    declaration: dict[str, object],
) -> dict[str, object]:
    frame = population.copy()
    frame["rating_full"] = candidate_oos
    frame["rating_reduced"] = frame["base_oos_probability"]
    scored = frame.loc[frame["rating_full"].notna() & frame["rating_reduced"].notna()].copy()
    cell = signal_cell(scored, "rating", "overall", declaration)
    candidate_oos_metrics = signal_metrics(
        scored["rating_full"].to_numpy(dtype=float),
        scored["home_covered"].to_numpy(dtype=float),
        RELIABILITY_EDGES,
    )
    base_oos_metrics = signal_metrics(
        scored["rating_reduced"].to_numpy(dtype=float),
        scored["home_covered"].to_numpy(dtype=float),
        RELIABILITY_EDGES,
    )
    candidate_is_metrics = signal_metrics(
        candidate_is, population["home_covered"].to_numpy(dtype=float), RELIABILITY_EDGES
    )
    base_is_metrics = signal_metrics(
        base_is, population["home_covered"].to_numpy(dtype=float), RELIABILITY_EDGES
    )
    return {
        "coverage_games": len(scored),
        "candidate_oos_metrics": {
            key: candidate_oos_metrics[key] for key in ("accuracy", "brier", "log_loss")
        },
        "base_oos_metrics": {
            key: base_oos_metrics[key] for key in ("accuracy", "brier", "log_loss")
        },
        "candidate_reliability_table": candidate_oos_metrics["reliability"],
        "is_vs_oos_gap": {
            "candidate_in_sample_accuracy": candidate_is_metrics["accuracy"],
            "candidate_oos_accuracy": candidate_oos_metrics["accuracy"],
            "candidate_accuracy_gap": (
                candidate_is_metrics["accuracy"] - candidate_oos_metrics["accuracy"]
            ),
            "base_in_sample_accuracy": base_is_metrics["accuracy"],
            "base_oos_accuracy": base_oos_metrics["accuracy"],
            "base_accuracy_gap": base_is_metrics["accuracy"] - base_oos_metrics["accuracy"],
        },
        "fold_coefficients": candidate_folds,
        "base_fold_coefficients": base_folds,
        "paired_cell": cell,
    }


def main() -> None:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    graded, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    ratings = pd.read_parquet(RATINGS_TABLE)[["game_id", "diff_lineup_total"]].copy()
    merged = graded.merge(ratings, on="game_id", how="left")
    coverage_total = len(merged)
    population = merged.loc[
        merged["diff_lineup_total"].notna() & merged["tue_open_home_spread"].notna()
    ].reset_index(drop=True)
    coverage_matched = len(population)

    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": RELIABILITY_EDGES,
    }

    base_oos, base_folds = loso(population, FIT_FEATURES)
    base_is, _base_is_coefficients = in_sample_fit(population, FIT_FEATURES)
    population = population.assign(base_oos_probability=base_oos, base_is_probability=base_is)

    population["term_interaction_unseen"] = population["diff_lineup_total"] * (
        1.0 - population[MOVE_AVAILABLE_COLUMN]
    )
    interaction_oos, interaction_folds = loso(
        population, (*FIT_FEATURES, "term_interaction_unseen")
    )
    interaction_is, interaction_is_coefficients = in_sample_fit(
        population, (*FIT_FEATURES, "term_interaction_unseen")
    )
    result_interaction = evaluate_candidate(
        population,
        interaction_oos,
        interaction_is,
        interaction_folds,
        base_folds,
        base_is,
        declaration,
    )
    result_interaction["in_sample_coefficients"] = interaction_is_coefficients

    residual_oos, residual_fit_folds, residual_regression_folds = residualized_oos_term(
        population, FIT_FEATURES, "tue_open_home_spread", "diff_lineup_total"
    )
    residual_is, residual_is_coefficients, residual_is_regression = residualized_is_term(
        population, FIT_FEATURES, "tue_open_home_spread", "diff_lineup_total"
    )
    result_residual = evaluate_candidate(
        population,
        residual_oos,
        residual_is,
        residual_fit_folds,
        base_folds,
        base_is,
        declaration,
    )
    result_residual["in_sample_coefficients"] = residual_is_coefficients
    result_residual["residualization_fold_regressions"] = residual_regression_folds
    result_residual["residualization_in_sample_regression"] = residual_is_regression

    summary = {
        "schema_version": 1,
        "created_at_utc": now.isoformat(),
        "command": "python scripts/players_on_field_rating_unit2.py",
        "provenance": provenance,
        "ratings_table": str(RATINGS_TABLE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "coverage_total_games": coverage_total,
        "coverage_matched_games": coverage_matched,
        "base_features": list(FIT_FEATURES),
        "predeclared_looks": [
            (
                "look_a_unpriced_residual: diff_lineup_total residualised on "
                "tue_open_home_spread, regression fit on LOSO training seasons only and "
                "applied to the held-out season, residual added as a 5th fitted term"
            ),
            (
                "look_b_unseen_interaction: diff_lineup_total * (1 - move_available) "
                "added as a 5th fitted term"
            ),
        ],
        "look_a_unpriced_residual": result_residual,
        "look_b_unseen_interaction": result_interaction,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    population.to_parquet(output_dir / "per_game.parquet", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
