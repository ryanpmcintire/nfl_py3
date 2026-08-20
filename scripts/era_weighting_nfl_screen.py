"""MOD-14 NFL close-grade analog: era weighting on the production weak_stack recipe.

Predeclaration: ``docs/era_weighting_screen.md`` Section 6. Only meant to be
run after the CFB screen (``scripts/era_weighting_cfb_screen.py``) shows the
best arm leaning positive on accuracy -- this is a second, disclosed look at
the same hypothesis family, not an independent blind test. A below-power
screen: read-only, no pick is played and no promotion decision is made here.

Mirrors ``scripts/smooth_cdf_mapping_measurement.py``'s own protocol: the
production ``weak_stack``/``ridge``/``ridge_alpha=10.0``/``market_residual``
recipe, ``data/processed/game_features_weak_stack.parquet``, CLOSE grade
(native ``spread_line``, not an opener archive), restricted to seasons no
rotation-registry family has reserved (``nfl_ats.rotation.season_usage``) so
no window is spent (rule 8). Rotation registry: untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from era_weighting_lib import (
    BASELINE_ARM_NAME,
    ERA_WEIGHTING_ARMS,
    fit_weighted_ridge_margin,
    half_life_weights,
    rolling_window_floor_season,
)

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import MarginModel, margin_feature_columns
from nfl_ats.modeling import regular_season_rows
from nfl_ats.outcomes import walk_forward_outcomes
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.rotation import load_registry, season_usage

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"

FEATURE_PROFILE = "weak_stack"
RIDGE_ALPHA = 10.0
TARGET = "market_residual"
DISTRIBUTION_FRACTION = 0.20

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819

_PASSTHROUGH = (
    "game_id",
    "season",
    "week",
    "gameday",
    "spread_line",
    "result",
    "ats_margin",
    "home_cover",
)


def non_reserved_seasons(*, start_season: int, end_season: int) -> tuple[int, ...]:
    """Every season in [start_season, end_season] no family has reserved.

    Identical logic to scripts/smooth_cdf_mapping_measurement.py's own
    helper (reused by re-implementation, not import, to keep this script
    standalone) -- rule 8 makes iteration on these seasons free.
    """

    registry = load_registry()
    reserved = {int(season) for season in season_usage(registry)}
    return tuple(season for season in range(start_season, end_season + 1) if season not in reserved)


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
        batches.append(batch)
    return batches


def run_screen(
    features: pd.DataFrame,
    *,
    eval_seasons: tuple[int, ...],
    start_season: int,
    end_season: int,
    min_train_games: int,
    ridge_alpha: float,
    feature_profile: str,
    distribution_fraction: float,
) -> pd.DataFrame:
    feature_columns = margin_feature_columns(TARGET, feature_profile)  # type: ignore[arg-type]
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[pd.to_numeric(frame["ats_margin"], errors="coerce").notna()].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError("No completed NFL games in the requested window")

    eval_set = set(eval_seasons)
    skip_counts: dict[str, int] = {arm.name: 0 for arm in ERA_WEIGHTING_ARMS}
    batches: list[pd.DataFrame] = []
    for (season, _week), weekly_games in test.groupby(["season", "week"], sort=True):
        if int(str(season)) not in eval_set:
            continue
        predict_season = int(str(season))
        cutoff = weekly_games["gameday"].min()
        training_full = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training_full) < min_train_games:
            continue

        target_full = pd.to_numeric(training_full["ats_margin"], errors="raise").to_numpy(
            dtype=float
        )
        seasons_full = training_full["season"].to_numpy(dtype=float)

        models: dict[str, MarginModel] = {}
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
                    feature_columns=feature_columns,
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
        raise ValueError("No evaluation week had enough prior training games")
    predictions = pd.concat(batches, ignore_index=True)
    predictions = predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)
    predictions.attrs["skip_counts"] = skip_counts
    return predictions


def self_check(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    eval_seasons: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Verify `baseline` reproduces the frozen production recipe's own fit."""

    frozen = walk_forward_outcomes(
        features,
        start_season=min(eval_seasons),
        end_season=max(eval_seasons),
        feature_profile=FEATURE_PROFILE,  # type: ignore[arg-type]
        methods=("market_residual",),
        ridge_alpha=args.ridge_alpha,
        min_train_games=args.min_train_games,
    )
    frozen_scored = frozen.predictions.loc[
        frozen.predictions["method"].eq("market_residual")
        & frozen.predictions["season"].isin(eval_seasons)
        & frozen.predictions["home_cover"].notna()
    ]
    own_scored = predictions.loc[
        predictions["method"].eq(BASELINE_ARM_NAME) & predictions["home_cover"].notna()
    ]
    frozen_accuracy = float(
        (frozen_scored["home_cover_probability"].ge(0.5) == frozen_scored["home_cover"]).mean()
    )
    own_accuracy = float(
        (own_scored["home_cover_probability"].ge(0.5) == own_scored["home_cover"]).mean()
    )
    return {
        "frozen_games": len(frozen_scored),
        "own_games": len(own_scored),
        "frozen_accuracy": frozen_accuracy,
        "own_accuracy": own_accuracy,
        "accuracy_diff": own_accuracy - frozen_accuracy,
    }


def paired_report(predictions: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    scored = predictions.loc[predictions["home_cover"].notna()].copy()
    rows: list[pd.DataFrame] = []
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
        probability["family"] = "probability"
        rows.append(probability)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--feature-profile", default=FEATURE_PROFILE)
    parser.add_argument("--distribution-fraction", type=float, default=DISTRIBUTION_FRACTION)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    eval_seasons = non_reserved_seasons(start_season=args.start_season, end_season=args.end_season)
    print(f"Non-reserved evaluation seasons: {eval_seasons}")

    features = pd.read_parquet(args.features)
    predictions = run_screen(
        features,
        eval_seasons=eval_seasons,
        start_season=args.start_season,
        end_season=args.end_season,
        min_train_games=args.min_train_games,
        ridge_alpha=args.ridge_alpha,
        feature_profile=args.feature_profile,
        distribution_fraction=args.distribution_fraction,
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)
    print(f"scored rows: {len(predictions)}; skip counts: {predictions.attrs['skip_counts']}")

    checks = self_check(predictions, features, eval_seasons=eval_seasons, args=args)
    print(f"self-check (baseline vs frozen production recipe): {checks}")

    paired = paired_report(predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    paired.to_csv(output / "paired_comparisons.csv", index=False)

    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/era_weighting_screen.md",
        "rotation_registry_touched": False,
        "grade": "close",
        "multiplicity_disclosure": (
            "Six candidate arms vs one baseline; this run is gated on a positive CFB lean, "
            "so it is a second, disclosed look at the same direction, not an independent "
            "blind draw -- see docs/era_weighting_screen.md Section 6."
        ),
        "eval_seasons": list(eval_seasons),
        "recipe": {
            "target": TARGET,
            "regressor": "ridge",
            "ridge_alpha": args.ridge_alpha,
            "feature_profile": args.feature_profile,
            "min_train_games": args.min_train_games,
            "feature_table": str(args.features),
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "skip_counts": predictions.attrs["skip_counts"],
        "self_check": checks,
        "arms": [arm.name for arm in ERA_WEIGHTING_ARMS],
    }
    configuration = {
        "command": "era-weighting-nfl-screen",
        "features": str(args.features),
        "start_season": args.start_season,
        "end_season": args.end_season,
        "min_train_games": args.min_train_games,
        "ridge_alpha": args.ridge_alpha,
        "feature_profile": args.feature_profile,
        "distribution_fraction": args.distribution_fraction,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    diagnostics["provenance"] = artifact_provenance(configuration, args.features, project_root=REPO)
    write_experiment_artifact(
        output,
        "diagnostics.json",
        diagnostics,
        command="era-weighting-nfl-screen",
        metrics=diagnostics,
        notes=(
            "MOD-14 NFL close-grade analog: era-weighting arms on the production "
            "weak_stack recipe, restricted to non-rotation-reserved seasons."
        ),
    )

    print("\n=== week-blocked, accuracy, paired vs baseline (NFL close grade) ===")
    headline = paired.loc[paired["block"].eq("week") & paired["metric"].eq("accuracy_improvement")]
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
