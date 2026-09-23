from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability import FLAG_SUM_COLUMN, MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, build_fit_population
from nfl_ats.pick_probability_fit import _fit_logit as fit_logit
from nfl_ats.signal_atlas import _cell as signal_cell

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = ARTIFACTS_ROOT / "lead65_gated_flag_in_fit"

EARLY_WINDOW_MAX_WEEK = 4
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
INTERVAL_LEVEL = 0.95


def design_matrix(
    frame: pd.DataFrame, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    columns = [np.ones(len(frame))]
    for name in features:
        columns.append(((frame[name].astype(float) - means[name]) / stds[name]).to_numpy())
    return np.column_stack(columns)


def standardisers(
    train: pd.DataFrame, features: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    means = {name: float(train[name].mean()) for name in features}
    stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in features}
    return means, stds


def natural_coefficients(
    beta: np.ndarray, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    natural = {name: float(beta[index + 1] / stds[name]) for index, name in enumerate(features)}
    intercept = float(beta[0])
    for index, name in enumerate(features):
        intercept -= float(beta[index + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def predict(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    beta: np.ndarray,
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    z = np.clip(design_matrix(frame, features, means, stds) @ beta, -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def loso(
    population: pd.DataFrame, features: tuple[str, ...]
) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    seasons = sorted(int(value) for value in population["season"].unique())
    out = pd.Series(np.nan, index=population.index, dtype=float)
    folds: dict[str, dict[str, float]] = {}
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = standardisers(train, features)
        beta = fit_logit(
            design_matrix(train, features, means, stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        out.loc[test.index] = predict(test, features, beta, means, stds)
        folds[str(held)] = natural_coefficients(beta, features, means, stds)
    return out, folds


def in_sample_fit(
    population: pd.DataFrame, features: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, float]]:
    means, stds = standardisers(population, features)
    beta = fit_logit(
        design_matrix(population, features, means, stds),
        population["home_covered"].astype(float).to_numpy(),
        FIT_RIDGE,
    )
    return predict(population, features, beta, means, stds), natural_coefficients(
        beta, features, means, stds
    )


def accuracy(probability: pd.Series, target: pd.Series) -> float:
    picked_home = probability.ge(0.5)
    correct = picked_home.astype(float).eq(target).astype(float)
    return float(correct.mean())


def main() -> None:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    population, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    week = population["week"].astype(int)
    early = week.le(EARLY_WINDOW_MAX_WEEK)
    protection = population["flag_protection"].astype(float)
    served_flag_sum = population[FLAG_SUM_COLUMN].astype(float)

    population["gated_flag_sum"] = served_flag_sum - protection * (~early).astype(float)
    population["flag_sum_no_protection"] = served_flag_sum - protection
    population["protection_early_interaction"] = protection * early.astype(float)

    features_served = FIT_FEATURES
    features_gated = ("model_logit", "gated_flag_sum", MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)
    features_interaction = (
        "model_logit",
        "flag_sum_no_protection",
        "protection_early_interaction",
        MOVE_COLUMN,
        MOVE_AVAILABLE_COLUMN,
    )

    served_oos, served_folds = loso(population, features_served)
    served_is, served_is_coefficients = in_sample_fit(population, features_served)
    population["served_oos_probability"] = served_oos

    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": [0.0, 0.5, 1.0],
    }

    looks: dict[str, object] = {}
    for look_name, features in (
        ("gated_flag_sum_v1", features_gated),
        ("protection_early_interaction_v1", features_interaction),
    ):
        variant_oos, variant_folds = loso(population, features)
        variant_is, variant_is_coefficients = in_sample_fit(population, features)
        frame = population.copy()
        frame["rating_full"] = variant_oos
        frame["rating_reduced"] = frame["served_oos_probability"]
        scored = frame.loc[frame["rating_full"].notna() & frame["rating_reduced"].notna()].copy()

        cell = signal_cell(scored, "rating", "overall", declaration)
        variant_oos_accuracy = accuracy(scored["rating_full"], scored["home_covered"].astype(float))
        served_oos_accuracy = accuracy(
            scored["rating_reduced"], scored["home_covered"].astype(float)
        )
        variant_is_accuracy = accuracy(
            pd.Series(variant_is, index=population.index), population["home_covered"].astype(float)
        )
        served_is_accuracy = accuracy(
            pd.Series(served_is, index=population.index), population["home_covered"].astype(float)
        )
        looks[look_name] = {
            "features": list(features),
            "coverage_games": len(scored),
            "paired_cell": cell,
            "is_vs_oos_gap": {
                "variant_in_sample_accuracy": variant_is_accuracy,
                "variant_oos_accuracy": variant_oos_accuracy,
                "variant_accuracy_gap": variant_is_accuracy - variant_oos_accuracy,
                "served_in_sample_accuracy": served_is_accuracy,
                "served_oos_accuracy": served_oos_accuracy,
                "served_accuracy_gap": served_is_accuracy - served_oos_accuracy,
            },
            "fold_coefficients": variant_folds,
            "in_sample_coefficients": variant_is_coefficients,
        }

    summary = {
        "schema_version": 1,
        "created_at_utc": now.isoformat(),
        "command": "python scripts/lead65_gated_flag_in_fit.py",
        "provenance": provenance,
        "early_window_max_week": EARLY_WINDOW_MAX_WEEK,
        "served_features": list(features_served),
        "served_fold_coefficients": served_folds,
        "served_in_sample_coefficients": served_is_coefficients,
        "looks": looks,
        "look_count": len(looks),
        "predeclared_family": "pbp08_protection_mismatch_gated_flag_sum_fit",
        "mechanism": (
            "The protection-mismatch flag's whole-card edge sits in weeks 1-4, where its "
            "four-game pressure window still reaches into the prior season's unpriced tail; "
            "from week 5 on the window is entirely in-season and public, so it is a probable "
            "drag once folded into flag_sum at equal weight to the other members. Both looks "
            "test whether gating or reweighting that one member inside the served four-term "
            "fitted probability, rather than the served card's unconditional composition, "
            "recovers value without a hard flip rule."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    population.to_parquet(output_dir / "per_game.parquet", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
