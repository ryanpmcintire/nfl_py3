from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_ats.experiment_runner as experiment_runner
import nfl_ats.officials_archive as officials_archive
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, build_fit_population
from nfl_ats.pick_probability_fit import _fit_logit as fit_logit
from nfl_ats.signal_atlas import _cell as signal_cell

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = ARTIFACTS_ROOT / "lead59_type_trait_bins"

BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20260923
INTERVAL_LEVEL = 0.95
TOP_QUARTILE = 4
BOTTOM_QUARTILE = 1


def build_type_trait_archive_on(repo_root: Path, penalty_type: str) -> tuple[pd.DataFrame, dict]:
    officials_path, _game_penalties_path, officials_snapshot_id = (
        experiment_runner._latest_officials_snapshot(repo_root)
    )
    officials = officials_archive.load_officials(
        repo_root, officials_path=officials_path, include_archive=True
    )
    refs = officials.loc[
        (officials["position"] == experiment_runner._REFEREE_POSITION)
        & (officials["season_type"] == experiment_runner._REFEREE_SEASON_TYPE)
    ].copy()
    schedules_path = experiment_runner._latest_schedules_snapshot(repo_root)
    schedules = pd.read_parquet(schedules_path).loc[:, ["game_id", "old_game_id"]]
    refs = refs.merge(
        schedules, left_on="game_id", right_on="old_game_id", how="inner", suffixes=("_legacy", "")
    )
    refs = refs.loc[:, ["game_id", "official_name", "season"]]

    penalty_type_path, penalty_type_snapshot_id = experiment_runner._latest_penalty_type_snapshot(
        repo_root
    )
    game_penalty_types = pd.read_parquet(penalty_type_path)
    covered = set(game_penalty_types["game_id"].astype(str))
    n_before_coverage_filter = len(refs)
    refs = refs.loc[refs["game_id"].astype(str).isin(covered)].copy()
    n_dropped_uncovered = n_before_coverage_filter - len(refs)

    type_counts = game_penalty_types.loc[
        game_penalty_types["penalty_type"] == penalty_type, ["game_id", "penalties_total"]
    ]
    merged_games = refs.merge(type_counts, on="game_id", how="left")
    merged_games["penalties_total"] = merged_games["penalties_total"].fillna(0.0)

    name_season = (
        merged_games.groupby(["official_name", "season"])
        .agg(mean_total=("penalties_total", "mean"))
        .reset_index()
    )
    reliability, reliability_pairs = experiment_runner._referee_year_over_year_reliability(
        name_season, "mean_total"
    )

    lag = name_season.sort_values(["official_name", "season"]).copy()
    lag["prev_total"] = lag.groupby("official_name")["mean_total"].shift(1)
    lag["prev_season"] = lag.groupby("official_name")["season"].shift(1)
    lagged = lag.loc[lag["season"] - lag["prev_season"] == 1].copy()
    lagged["lag_type_quartile"] = pd.qcut(lagged["prev_total"], 4, labels=[1, 2, 3, 4]).astype(int)

    game_trait = merged_games[["game_id", "official_name", "season"]].merge(
        lagged[["official_name", "season", "lag_type_quartile"]],
        on=["official_name", "season"],
        how="left",
    )
    diagnostics = {
        "penalty_type": penalty_type,
        "officials_snapshot_id": officials_snapshot_id,
        "penalty_type_snapshot_id": penalty_type_snapshot_id,
        "reliability": reliability,
        "reliability_pairs": reliability_pairs,
        "n_officials": int(name_season["official_name"].nunique()),
        "referee_games_before_coverage_filter": int(n_before_coverage_filter),
        "referee_games_dropped_uncovered": int(n_dropped_uncovered),
    }
    return game_trait, diagnostics


def archive_on_vs_off_diagnostic(
    repo_root: Path, penalty_type: str, archive_on_trait: pd.DataFrame
) -> dict:
    production = experiment_runner._build_referee_type_trait_data(
        repo_root, penalty_type
    ).game_trait
    left = archive_on_trait.drop_duplicates("game_id").set_index("game_id")["lag_type_quartile"]
    right = production.drop_duplicates("game_id").set_index("game_id")["lag_type_quartile"]
    shared = left.index.intersection(right.index)
    a = left.loc[shared]
    b = right.loc[shared]
    changed = ~(a.eq(b) | (a.isna() & b.isna()))
    return {
        "games_archive_on": len(left),
        "games_production_archive_off": len(right),
        "games_shared": len(shared),
        "games_only_archive_on": len(left.index.difference(right.index)),
        "games_only_production": len(right.index.difference(left.index)),
        "n_changed_on_shared": int(changed.sum()),
    }


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
    population = experiment_runner._merge_home_pass_rate_quartile(population, REPO_ROOT)

    dpi_trait, dpi_diagnostics = build_type_trait_archive_on(
        REPO_ROOT, experiment_runner._DPI_PENALTY_TYPE
    )
    holding_trait, holding_diagnostics = build_type_trait_archive_on(
        REPO_ROOT, experiment_runner._HOLDING_PENALTY_TYPE
    )
    dpi_archive_diff = archive_on_vs_off_diagnostic(
        REPO_ROOT, experiment_runner._DPI_PENALTY_TYPE, dpi_trait
    )
    holding_archive_diff = archive_on_vs_off_diagnostic(
        REPO_ROOT, experiment_runner._HOLDING_PENALTY_TYPE, holding_trait
    )

    dpi_lookup = dpi_trait.drop_duplicates("game_id").set_index("game_id")["lag_type_quartile"]
    holding_lookup = (
        holding_trait.drop_duplicates("game_id").set_index("game_id")["lag_type_quartile"]
    )
    population["dpi_lag_quartile"] = population["game_id"].map(dpi_lookup)
    population["holding_lag_quartile"] = population["game_id"].map(holding_lookup)

    home_favorite = population["tue_open_home_spread"].astype(float) > 0.0
    pass_heavy_top = population["home_pass_rate_quartile"].eq(TOP_QUARTILE)
    run_heavy_bottom = population["home_pass_rate_quartile"].eq(BOTTOM_QUARTILE)
    dpi_top = population["dpi_lag_quartile"].eq(TOP_QUARTILE)
    holding_top = population["holding_lag_quartile"].eq(TOP_QUARTILE)

    population["dpi_tilt_pass_heavy_favorite"] = (
        home_favorite & pass_heavy_top & dpi_top.fillna(False)
    ).astype(float)
    population["holding_tilt_run_heavy"] = (
        run_heavy_bottom & holding_top.fillna(False)
    ).astype(float)

    features_served = FIT_FEATURES
    served_oos, served_folds = loso(population, features_served)
    served_is, served_is_coefficients = in_sample_fit(population, features_served)
    population["served_oos_probability"] = served_oos

    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": [0.0, 0.5, 1.0],
    }

    bin_flags = {
        "dpi_tilt_pass_heavy_favorite": "dpi_tilt_pass_heavy_favorite",
        "holding_tilt_run_heavy": "holding_tilt_run_heavy",
    }

    looks: dict[str, object] = {}
    for look_name, flag_column in bin_flags.items():
        features = (*FIT_FEATURES, flag_column)
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
            "flag_positive_games": int(population[flag_column].sum()),
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
        "command": "python scripts/lead59_type_trait_bins.py",
        "provenance": provenance,
        "served_features": list(features_served),
        "served_fold_coefficients": served_folds,
        "served_in_sample_coefficients": served_is_coefficients,
        "type_trait_build_diagnostics": {
            "dpi": dpi_diagnostics,
            "holding": holding_diagnostics,
        },
        "archive_on_vs_production_archive_off": {
            "dpi": dpi_archive_diff,
            "holding": holding_archive_diff,
        },
        "looks": looks,
        "look_count": len(looks),
        "predeclared_family": "lead59_type_trait_archive_bins_fit_v1",
        "predeclaration": (
            "docs/lanes/lead59-archive-battery.md#"
            "type-trait-binning-predeclaration-2026-09-23-before-measuring"
        ),
        "mechanism": (
            "Both bins mirror the two live prospective crew-tilt challenger flags "
            "(experiment_runner._flag_referee_dpi_tilt_pass_heavy_favorite and "
            "_flag_referee_holding_tilt_run_heavy): a penalty-type-heavy referee crossed "
            "with a home offensive tendency that draws that type of flag. This unit tests "
            "whether folding either flag into the served four-term fitted probability as "
            "its own term, rather than leaving it as a standalone rule, recovers value, "
            "using the archive-on referee population under the narrower coverage filter "
            "already pinned in crew_tilt_refresh_overlay._referee_name_season (drop games "
            "absent from the penalty-type snapshot instead of fabricating a zero)."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    population.to_parquet(output_dir / "per_game.parquet", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
