from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import CLOSE_LABEL_PRIORITY, build_pairing_table, close_reference_table
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    build_fit_population,
)
from nfl_ats.pick_probability_fit import _fit_logit as fit_logit
from nfl_ats.signal_atlas import _cell as signal_cell
from nfl_ats.signal_atlas import _metrics as signal_metrics
from nfl_ats.snapshots import latest_snapshot, load_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
MARKET_ROOT = REPO_ROOT / "data" / "market" / "raw"
RATINGS_TABLE = (
    ARTIFACTS_ROOT / "latent_ratings_on_production" / "expected_lineup_ratings.parquet"
)
OUTPUT_ROOT = ARTIFACTS_ROOT / "players_on_field_rating"

CANDIDATE_TERMS = ("diff_lineup_total", "diff_divergence")
RELIABILITY_EDGES = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
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


def season_block_bootstrap_slope(
    frame: pd.DataFrame, x_column: str, y_column: str
) -> dict[str, object]:
    x = frame[x_column].to_numpy(dtype=float)
    y = frame[y_column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    seasons = sorted(int(value) for value in frame["season"].unique())
    season_array = frame["season"].to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = rng.choice(seasons, size=len(seasons), replace=True)
        idx = np.concatenate([np.where(season_array == season)[0] for season in picked])
        bx, by = x[idx], y[idx]
        if np.std(bx) == 0:
            continue
        bs, _ = np.polyfit(bx, by, 1)
        draws.append(bs)
    boots = np.asarray(draws, dtype=float)
    tail = (1.0 - INTERVAL_LEVEL) / 2.0
    lower, upper = np.quantile(boots, [tail, 1.0 - tail])
    probability_positive = float((boots > 0).mean() + 0.5 * (boots == 0).mean())
    return {
        "games": int(len(frame)),
        "seasons": seasons,
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": correlation,
        "bootstrap_draws": int(len(boots)),
        "bootstrap_interval": [float(lower), float(upper)],
        "probability_positive": probability_positive,
    }


def main() -> None:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    graded, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    ratings = pd.read_parquet(RATINGS_TABLE)[
        [
            "game_id",
            "diff_lineup_total",
            "diff_divergence",
            "home_full_strength_total",
            "away_full_strength_total",
        ]
    ].copy()
    ratings["diff_full_strength_total"] = (
        ratings["home_full_strength_total"] - ratings["away_full_strength_total"]
    )
    merged = graded.merge(ratings, on="game_id", how="left")
    coverage_total = len(merged)
    coverage_matched = int(merged["diff_lineup_total"].notna().sum())
    population = merged.loc[merged["diff_lineup_total"].notna()].reset_index(drop=True)

    schedules, _team_stats = load_snapshot(latest_snapshot(DATA_ROOT / "raw"))
    schedule = (
        population[["game_id", "season", "week"]]
        .drop_duplicates("game_id")
        .merge(
            schedules[["game_id", "spread_line"]].drop_duplicates("game_id"),
            on="game_id",
            how="left",
        )
    )
    pairing = build_pairing_table(
        MARKET_ROOT,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=schedule,
    )
    line_move_result: dict[str, object] = {"available": False}
    if not pairing.empty:
        close = close_reference_table(pairing, schedule)
        move_frame = population[
            [
                "game_id",
                "season",
                "tue_open_home_spread",
                "diff_lineup_total",
                "diff_divergence",
                "diff_full_strength_total",
            ]
        ].merge(close[["game_id", "close_home_spread"]], on="game_id", how="inner")
        move_frame["target_close_minus_open"] = (
            move_frame["close_home_spread"] - move_frame["tue_open_home_spread"]
        )
        move_frame = move_frame.dropna(subset=["target_close_minus_open"]).reset_index(drop=True)
        line_move_result = {"available": True, "games": len(move_frame)}
        for term in (*CANDIDATE_TERMS, "diff_full_strength_total"):
            line_move_result[term] = season_block_bootstrap_slope(
                move_frame, term, "target_close_minus_open"
            )

    base_oos, base_folds = loso(population, FIT_FEATURES)
    base_is, base_is_coefficients = in_sample_fit(population, FIT_FEATURES)
    population = population.assign(base_oos_probability=base_oos, base_is_probability=base_is)

    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": RELIABILITY_EDGES,
    }
    fit_results: dict[str, object] = {}
    for term in CANDIDATE_TERMS:
        features = (*FIT_FEATURES, term)
        candidate_oos, candidate_folds = loso(population, features)
        candidate_is, candidate_is_coefficients = in_sample_fit(population, features)
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
        fit_results[term] = {
            "coverage_games": int(len(scored)),
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
            "in_sample_coefficients": candidate_is_coefficients,
            "paired_cell": cell,
        }

    summary = {
        "schema_version": 1,
        "created_at_utc": now.isoformat(),
        "command": "python scripts/players_on_field_rating_eval.py",
        "provenance": provenance,
        "ratings_table": str(RATINGS_TABLE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "coverage_total_games": coverage_total,
        "coverage_matched_games": coverage_matched,
        "base_features": list(FIT_FEATURES),
        "candidate_terms": list(CANDIDATE_TERMS),
        "line_move_look": line_move_result,
        "fit_results": fit_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    population.to_parquet(output_dir / "per_game.parquet", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
