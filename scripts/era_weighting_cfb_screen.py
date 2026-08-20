"""MOD-14 CFB screen: rolling training windows vs. time-decayed sample weights.

Predeclaration: ``docs/era_weighting_screen.md`` (frozen before any arm's
accuracy sign was computed). Rotation registry: untouched -- CFB is free
under rule 8, no NFL confirmation window is spent here.

Walks the frozen XLG-03 CFB market-residual benchmark
(``nfl_ats.cfb_benchmark``) forward week by week, fitting seven arms per
week from strictly-earlier training rows: ``baseline`` (uniform weight, all
history -- must reproduce the frozen benchmark's own ``market_residual``
arm) plus six candidates (``half_life_{2,4,8,16}`` exponential season decay,
``rolling_{6,10}`` truncated training windows). Only the row
weighting/selection differs between arms; ``ridge_alpha=10.0`` and every
other setting is held fixed throughout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from era_weighting_lib import (
    BASELINE_ARM_NAME,
    CANDIDATE_ARM_NAMES,
    ERA_WEIGHTING_ARMS,
    fit_weighted_ridge_margin,
    half_life_weights,
    rolling_window_floor_season,
)

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_DISTRIBUTION_FRACTION,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
    cfb_walk_forward_benchmark,
    summarize_cfb_benchmark,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.cfb_opponent_adjustment import paired_margin_error_comparison
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import MarginModel, fit_market_baseline
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data/processed/cfb_game_features.parquet"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819

_PASSTHROUGH = (
    "game_id",
    "season",
    "week",
    "gameday",
    "source_regime",
    "spread_line",
    "spread_book_count",
    "spread_dispersion",
    "home_spread_odds",
    "away_spread_odds",
    "result",
    "ats_margin",
    "home_cover",
)


def _score_week(weekly_games: pd.DataFrame, models: dict[str, MarginModel]) -> list[pd.DataFrame]:
    batches: list[pd.DataFrame] = []
    for method, model in models.items():
        batch = weekly_games.loc[:, list(_PASSTHROUGH)].copy()
        forecasts = model.predict(weekly_games)
        for column in forecasts:
            batch[column] = forecasts[column]
        batch["method"] = method
        batch["feature_set"] = method
        batch["model_name"] = model.model_name
        batch["train_rows"] = model.training_rows
        batch["distribution_rows"] = model.distribution_rows
        # The screen is a forced-pick instrument, not a wagering policy --
        # matches nfl_ats.cfb_benchmark._score_week's own convention.
        batch["bet_side"] = "PASS"
        batch["bet_odds"] = np.nan
        batches.append(batch)
    return batches


def run_screen(
    features: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    min_train_games: int,
    ridge_alpha: float,
    distribution_fraction: float,
) -> pd.DataFrame:
    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {start_season} to {end_season}")

    skip_counts: dict[str, int] = {arm.name: 0 for arm in ERA_WEIGHTING_ARMS}
    batches: list[pd.DataFrame] = []
    for (season, _week), weekly_games in test.groupby(["season", "week"], sort=True):
        predict_season = int(str(season))
        cutoff = weekly_games["gameday"].min()
        training_full = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training_full) < min_train_games:
            continue

        target_full = pd.to_numeric(training_full["ats_margin"], errors="raise").to_numpy(
            dtype=float
        )
        seasons_full = training_full["season"].to_numpy(dtype=float)

        models: dict[str, MarginModel] = {"market": fit_market_baseline(training_full)}
        for arm in ERA_WEIGHTING_ARMS:
            if arm.kind == "uniform":
                arm_frame, arm_target = training_full, target_full
                weights = np.ones(len(arm_frame), dtype=float)
            elif arm.kind == "half_life":
                arm_frame, arm_target = training_full, target_full
                weights = half_life_weights(seasons_full, predict_season, arm.parameter or 1.0)
            else:  # rolling
                floor_season = rolling_window_floor_season(predict_season, arm.parameter or 1.0)
                arm_frame = training_full.loc[training_full["season"].ge(floor_season)]
                if len(arm_frame) < min_train_games:
                    skip_counts[arm.name] += 1
                    continue
                arm_target = pd.to_numeric(arm_frame["ats_margin"], errors="raise").to_numpy(
                    dtype=float
                )
                weights = np.ones(len(arm_frame), dtype=float)
            try:
                models[arm.name] = fit_weighted_ridge_margin(
                    arm_frame,
                    target=arm_target,
                    feature_columns=CFB_MODEL_FEATURE_COLUMNS,
                    weights=weights,
                    ridge_alpha=ridge_alpha,
                    distribution_fraction=distribution_fraction,
                    min_distribution_rows=10,
                    random_state=42,
                )
            except ValueError:
                skip_counts[arm.name] += 1
                continue
        batches.extend(_score_week(weekly_games, models))

    if not batches:
        raise ValueError("No CFB week had enough prior training games")
    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)
    predictions.attrs["skip_counts"] = skip_counts
    return predictions


def paired_report(predictions: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    keep = predictions.loc[
        predictions["method"].isin((BASELINE_ARM_NAME, *CANDIDATE_ARM_NAMES))
    ].copy()
    for window in ("clean_core", "all"):
        subset = keep if window == "all" else keep.loc[keep["evaluation_window"].eq(window)]
        if subset.empty:
            continue
        scored = subset.loc[subset["home_cover"].notna()]
        for block in ("week", "season"):
            if block == "season" and scored["season"].nunique() < 2:
                continue
            probability = paired_feature_comparisons(
                scored,
                baseline_feature_set=BASELINE_ARM_NAME,
                samples=samples,
                block=block,
                seed=seed,
            )
            probability["evaluation_window"] = window
            probability["family"] = "probability"
            rows.append(probability)

            margin_rows = []
            for arm_name in CANDIDATE_ARM_NAMES:
                if arm_name not in set(subset["method"]):
                    continue
                margin_rows.append(
                    paired_margin_error_comparison(
                        subset,
                        baseline_method=BASELINE_ARM_NAME,
                        candidate_method=arm_name,
                        samples=samples,
                        block=block,
                        seed=seed,
                    )
                )
            if margin_rows:
                margin = pd.concat(margin_rows, ignore_index=True)
                margin = margin.rename(columns={"candidate_method": "candidate_feature_set"})
                margin["evaluation_window"] = window
                margin["family"] = "margin_error"
                rows.append(margin)
    return pd.concat(rows, ignore_index=True)


def self_check(
    predictions: pd.DataFrame, features: pd.DataFrame, *, args: argparse.Namespace
) -> dict[str, Any]:
    """Verify `baseline` reproduces the frozen cfb_walk_forward_benchmark's own arm."""

    frozen = cfb_walk_forward_benchmark(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        min_train_games=args.min_train_games,
        ridge_alpha=args.ridge_alpha,
    )
    frozen_clean = frozen.predictions.loc[
        frozen.predictions["evaluation_window"].eq("clean_core")
        & frozen.predictions["method"].eq("market_residual")
    ]
    own_clean = predictions.loc[
        predictions["evaluation_window"].eq("clean_core")
        & predictions["method"].eq(BASELINE_ARM_NAME)
    ]
    frozen_accuracy = float(
        (frozen_clean["home_cover_probability"].ge(0.5) == frozen_clean["home_cover"]).mean()
    )
    own_accuracy = float(
        (own_clean["home_cover_probability"].ge(0.5) == own_clean["home_cover"]).mean()
    )
    return {
        "frozen_games": len(frozen_clean),
        "own_games": len(own_clean),
        "frozen_accuracy": frozen_accuracy,
        "own_accuracy": own_accuracy,
        "accuracy_diff": own_accuracy - frozen_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=CFB_BENCHMARK_START_SEASON)
    parser.add_argument("--end-season", type=int, default=CFB_BENCHMARK_END_SEASON)
    parser.add_argument("--min-train-games", type=int, default=CFB_BENCHMARK_MIN_TRAIN_GAMES)
    parser.add_argument("--ridge-alpha", type=float, default=CFB_BENCHMARK_RIDGE_ALPHA)
    parser.add_argument(
        "--distribution-fraction", type=float, default=CFB_BENCHMARK_DISTRIBUTION_FRACTION
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features)
    print(f"features: {len(features)} games")

    predictions = run_screen(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        min_train_games=args.min_train_games,
        ridge_alpha=args.ridge_alpha,
        distribution_fraction=args.distribution_fraction,
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)
    print(f"scored rows: {len(predictions)}; skip counts: {predictions.attrs['skip_counts']}")

    summary, season_summary = summarize_cfb_benchmark(predictions)
    summary.to_csv(output / "summary.csv", index=False)
    season_summary.to_csv(output / "season_summary.csv", index=False)

    checks = self_check(predictions, features, args=args)
    print(f"self-check (baseline vs frozen benchmark, clean core): {checks}")

    paired = paired_report(predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    paired.to_csv(output / "paired_comparisons.csv", index=False)

    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/era_weighting_screen.md",
        "hypothesis_frozen_before_scoring": True,
        "rotation_registry_touched": False,
        "ridge_alpha": args.ridge_alpha,
        "min_train_games": args.min_train_games,
        "distribution_fraction": args.distribution_fraction,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "skip_counts": predictions.attrs["skip_counts"],
        "self_check": checks,
        "arms": [arm.name for arm in ERA_WEIGHTING_ARMS],
    }
    configuration = {
        "command": "era-weighting-cfb-screen",
        "features": str(args.features),
        "start_season": args.start_season,
        "end_season": args.end_season,
        "min_train_games": args.min_train_games,
        "ridge_alpha": args.ridge_alpha,
        "distribution_fraction": args.distribution_fraction,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    diagnostics["provenance"] = artifact_provenance(configuration, args.features, project_root=REPO)
    write_experiment_artifact(
        output,
        "diagnostics.json",
        diagnostics,
        command="era-weighting-cfb-screen",
        metrics=diagnostics,
        notes=(
            "MOD-14 CFB screen: rolling-window/half-life era-weighting arms vs. the "
            "uniform-weight baseline, clean-core week-blocked paired accuracy."
        ),
    )

    print("\n=== headline: clean core, week-blocked, accuracy, paired vs baseline ===")
    headline = paired.loc[
        paired["evaluation_window"].eq("clean_core")
        & paired["block"].eq("week")
        & paired["family"].eq("probability")
        & paired["metric"].eq("accuracy_improvement")
    ]
    print(
        headline.loc[
            :,
            [
                "candidate_feature_set",
                "estimate",
                "lower",
                "upper",
                "probability_positive",
                "paired_games",
            ],
        ].to_string(index=False)
    )
    print(f"\nartifacts: {output}")


if __name__ == "__main__":
    main()
