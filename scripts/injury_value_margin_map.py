from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, build_fit_population
from nfl_ats.pick_probability_fit import _fit_logit as fit_logit
from nfl_ats.signal_atlas import _cell as signal_cell

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = ARTIFACTS_ROOT / "injury_value_margin_map"

VALUE_LOST_DIFF_COLUMNS = (
    "diff_injury_skill_epa_value_lost",
    "diff_injury_defense_disruption_value_lost",
)
POINT_IN_TIME_COLUMN = "home_injury_observed_at"
FIRST_SEASON = 2020
LAST_SEASON = 2024
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
INTERVAL_LEVEL = 0.95
CANDIDATE_FEATURES = (*FIT_FEATURES, "value_lost_diff")


def load_injury_value(data_root: Path) -> pd.DataFrame:
    table = pd.read_parquet(
        data_root / "processed" / "game_features_player_value.parquet",
        columns=["game_id", *VALUE_LOST_DIFF_COLUMNS, POINT_IN_TIME_COLUMN],
    )
    total = pd.Series(0.0, index=table.index, dtype=float)
    for column in VALUE_LOST_DIFF_COLUMNS:
        total = total + pd.to_numeric(table[column], errors="raise")
    table["value_lost_diff"] = total
    table["point_in_time"] = table[POINT_IN_TIME_COLUMN].notna()
    return table[["game_id", "value_lost_diff", "point_in_time"]]


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


def predict(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    beta: np.ndarray,
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    z = np.clip(design_matrix(frame, features, means, stds) @ beta, -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def natural_coefficients(
    beta: np.ndarray, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    natural = {name: float(beta[index + 1] / stds[name]) for index, name in enumerate(features)}
    intercept = float(beta[0])
    for index, name in enumerate(features):
        intercept -= float(beta[index + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def loso_logistic(
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


def loso_margin_ols(population: pd.DataFrame) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    seasons = sorted(int(value) for value in population["season"].unique())
    out = pd.Series(np.nan, index=population.index, dtype=float)
    folds: dict[str, dict[str, float]] = {}
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        x = train["value_lost_diff"].astype(float).to_numpy()
        y = train["margin_vs_open"].astype(float).to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        test_x = test["value_lost_diff"].astype(float).to_numpy()
        out.loc[test.index] = intercept + slope * test_x
        folds[str(held)] = {
            "slope_points_per_unit": float(slope),
            "intercept": float(intercept),
            "train_games": len(train),
            "test_games": len(test),
        }
    return out, folds


def week_blocks(frame: pd.DataFrame, columns: list[str]) -> list[list[np.ndarray]]:
    blocks = []
    for season in sorted(frame["season"].unique()):
        weeks = []
        season_frame = frame.loc[frame["season"] == season]
        for week in sorted(season_frame["week"].unique()):
            mask = season_frame["week"] == week
            weeks.append(season_frame.loc[mask, columns].to_numpy(dtype=float))
        blocks.append(weeks)
    return blocks


def slope_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> np.ndarray:
    blocks = week_blocks(frame, ["value_lost_diff", "margin_vs_open"])
    sums = []
    for season_weeks in blocks:
        week_sums = []
        for values in season_weeks:
            x = values[:, 0]
            y = values[:, 1]
            week_sums.append(
                np.array([len(x), x.sum(), y.sum(), float((x * y).sum()), float((x * x).sum())])
            )
        sums.append(np.asarray(week_sums, dtype=float))
    rng = np.random.default_rng(seed)
    slopes = np.zeros(draws)
    for draw in range(draws):
        total = np.zeros(5)
        for index in rng.integers(0, len(sums), size=len(sums)):
            block = sums[index]
            total += block[rng.integers(0, len(block), size=len(block))].sum(axis=0)
        n, sx, sy, sxy, sxx = total
        denominator = n * sxx - sx * sx
        slopes[draw] = (n * sxy - sx * sy) / denominator if denominator else np.nan
    return slopes


def generic_block_bootstrap(
    frame: pd.DataFrame, difference: np.ndarray, draws: int, seed: int
) -> np.ndarray:
    width = difference.shape[1]
    blocks = []
    for season in sorted(frame["season"].unique()):
        weeks = []
        for week in sorted(frame.loc[frame["season"] == season, "week"].unique()):
            mask = ((frame["season"] == season) & (frame["week"] == week)).to_numpy()
            weeks.append(np.append(difference[mask].sum(axis=0), mask.sum()))
        blocks.append(np.asarray(weeks, dtype=float))
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros((draws, width))
    for draw in range(draws):
        total = np.zeros(width + 1)
        for index in rng.integers(0, len(blocks), size=len(blocks)):
            block = blocks[index]
            total += block[rng.integers(0, len(block), size=len(block))].sum(axis=0)
        bootstrap[draw] = total[:width] / total[width]
    return bootstrap


def error_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> np.ndarray:
    residual = frame["margin_vs_open"].astype(float).to_numpy()
    predicted = frame["margin_oos_prediction"].astype(float).to_numpy()
    baseline_abs = np.abs(residual)
    candidate_abs = np.abs(residual - predicted)
    baseline_sq = residual**2
    candidate_sq = (residual - predicted) ** 2
    difference = np.column_stack(
        [baseline_abs - candidate_abs, baseline_sq - candidate_sq]
    )
    return generic_block_bootstrap(frame, difference, draws, seed)


def classify(probability_positive: float) -> str:
    if probability_positive >= 0.975 or probability_positive <= 0.025:
        return "resolved_directional"
    return "unresolved_below_power"


def main() -> None:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    population, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    injury = load_injury_value(DATA_ROOT)
    population = population.merge(injury, on="game_id", how="left", validate="one_to_one")

    scoped = population.loc[
        population["season"].between(FIRST_SEASON, LAST_SEASON)
        & population["point_in_time"].fillna(False)
        & population["value_lost_diff"].notna()
    ].reset_index(drop=True)

    margin_oos, margin_folds = loso_margin_ols(scoped)
    scoped["margin_oos_prediction"] = margin_oos

    slopes = np.array([fold["slope_points_per_unit"] for fold in margin_folds.values()])
    baseline_mae = float(scoped["margin_vs_open"].abs().mean())
    candidate_mae = float((scoped["margin_vs_open"] - scoped["margin_oos_prediction"]).abs().mean())
    baseline_rmse = float(np.sqrt((scoped["margin_vs_open"] ** 2).mean()))
    candidate_rmse = float(
        np.sqrt(((scoped["margin_vs_open"] - scoped["margin_oos_prediction"]) ** 2).mean())
    )

    slope_draws = slope_bootstrap(scoped, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    pooled_slope = float(np.nanmean(slope_draws))
    slope_positive = float((slope_draws > 0).mean() + 0.5 * (slope_draws == 0).mean())
    slope_tail = (1.0 - INTERVAL_LEVEL) / 2.0
    slope_interval = np.nanquantile(slope_draws, [slope_tail, 1.0 - slope_tail]).tolist()

    error_draws = error_bootstrap(scoped, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    error_tail = (1.0 - INTERVAL_LEVEL) / 2.0
    mae_positive = float((error_draws[:, 0] > 0).mean() + 0.5 * (error_draws[:, 0] == 0).mean())
    rmse_positive = float((error_draws[:, 1] > 0).mean() + 0.5 * (error_draws[:, 1] == 0).mean())
    mae_interval = np.quantile(error_draws[:, 0], [error_tail, 1.0 - error_tail]).tolist()
    rmse_interval = np.quantile(error_draws[:, 1], [error_tail, 1.0 - error_tail]).tolist()

    baseline_oos, baseline_folds = loso_logistic(scoped, FIT_FEATURES)
    candidate_oos, candidate_folds = loso_logistic(scoped, CANDIDATE_FEATURES)

    scored = scoped.copy()
    scored["rating_full"] = candidate_oos
    scored["rating_reduced"] = baseline_oos
    scored = scored.loc[scored["rating_full"].notna() & scored["rating_reduced"].notna()].copy()

    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": [0.0, 0.5, 1.0],
    }
    probability_cell = signal_cell(scored, "rating", "overall", declaration)

    summary = {
        "predeclaration": {
            "predictor": "value_lost_diff = diff_injury_skill_epa_value_lost + "
            "diff_injury_defense_disruption_value_lost",
            "target": "margin_vs_open (final margin residual vs the opening spread)",
            "seasons": [FIRST_SEASON, LAST_SEASON],
            "point_in_time_definition": f"{POINT_IN_TIME_COLUMN} not null",
            "opener_evaluation": provenance["opener_evaluation"],
            "scoped_games": len(scoped),
        },
        "margin_regression": {
            "loso_folds": margin_folds,
            "pooled_bootstrap_slope_points_per_unit": pooled_slope,
            "slope_probability_positive": slope_positive,
            "slope_interval": slope_interval,
            "slope_sign_consistency": float(np.mean(slopes > 0))
            if np.all(slopes != 0)
            else float(np.mean(slopes >= 0)),
            "fold_slopes": slopes.tolist(),
            "opener_alone_oos_mae": baseline_mae,
            "candidate_oos_mae": candidate_mae,
            "mae_improvement": baseline_mae - candidate_mae,
            "mae_improvement_probability_positive": mae_positive,
            "mae_improvement_interval": mae_interval,
            "opener_alone_oos_rmse": baseline_rmse,
            "candidate_oos_rmse": candidate_rmse,
            "rmse_improvement_meansq": (baseline_rmse**2 - candidate_rmse**2),
            "rmse_improvement_probability_positive": rmse_positive,
            "rmse_improvement_interval_meansq": rmse_interval,
            "classification": classify(slope_positive),
        },
        "cover_probability_paired_look": {
            "baseline_features": list(FIT_FEATURES),
            "candidate_features": list(CANDIDATE_FEATURES),
            "baseline_folds": baseline_folds,
            "candidate_folds": candidate_folds,
            "cell": probability_cell,
            "classification": classify(probability_cell["probability_positive"]),
        },
    }

    scoped.to_parquet(output_dir / "scoped_population.parquet", index=False)
    scored.to_parquet(output_dir / "paired_probability_population.parquet", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["predeclaration"], indent=2))
    print("fold slopes:", margin_folds)
    print("pooled bootstrap slope:", pooled_slope, "probability_positive:", slope_positive)
    print("opener-alone MAE:", baseline_mae, "candidate OOS MAE:", candidate_mae)
    print("opener-alone RMSE:", baseline_rmse, "candidate OOS RMSE:", candidate_rmse)
    print(
        "probability cell accuracy_delta_points:",
        probability_cell["accuracy_delta_points"],
        "probability_positive:",
        probability_cell["probability_positive"],
    )
    print(
        "decisive games:",
        probability_cell["decisive_games"],
        "full(candidate) wins:",
        probability_cell["full_decisive_wins"],
        "reduced(baseline) wins:",
        probability_cell["reduced_decisive_wins"],
    )
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
