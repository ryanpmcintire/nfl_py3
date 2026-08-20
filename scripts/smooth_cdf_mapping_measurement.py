"""MOD-08 confirmation measurement: Gaussian CDF mapping vs. the production ECDF.

Predeclaration: ``docs/smooth_cdf_mapping.md``. This script never calls
``nfl_ats.rotation.assign_window``/``record_look`` -- per rule 8 ("CFB and
non-reserved seasons stay free"), it restricts the NFL evaluation to seasons
no rotation-registry family has reserved (``nfl_ats.rotation.season_usage``),
so no registry window is spent or implied.

Walks forward the PRODUCTION recipe exactly (``market_residual`` target,
``ridge`` regressor, ridge_alpha=10.0, ``weak_stack`` feature profile,
min_train_games=500 -- ``artifacts/active_ats_model.json`` as of 2026-08-19)
on the standard ``game_features_weak_stack.parquet`` table (the CLOSE grade:
nflverse's own spread_line, not an archived Tuesday-opener snapshot). Every
non-reserved test week is scored twice from the SAME fitted model and the
SAME residual sample: the production `ecdf` control
(``margin._smoothed_probability``, reproduced via
``nfl_ats.calibration.smoothed_home_cover_probability(method="ecdf")``) and
the `gaussian` candidate (``method="gaussian"``). Reserved-season weeks are
still walked over (their games enter the training pool for later cutoffs,
exactly as they would for the real production model) but are never fit-and-
scored as evaluation rows, so no registry-governed season contributes any
evidence to the reported comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.key_numbers import line_bucket
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.rotation import load_registry, season_usage

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"

BASELINE_ARM = "ecdf"
CANDIDATE_ARM = "gaussian"

# Production recipe, artifacts/active_ats_model.json (2026-08-19).
FEATURE_PROFILE = "weak_stack"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
MIN_TRAIN_GAMES = 500
TARGET = "market_residual"


def non_reserved_seasons(*, start_season: int, end_season: int) -> tuple[int, ...]:
    """Every season in ``[start_season, end_season]`` no family has reserved.

    ``nfl_ats.rotation.season_usage`` reports every season any family has
    SPENT a window over, across the whole registry -- not just one family's
    chain. Per rule 8, iterating on a season nothing has reserved needs no
    registry entry of its own.
    """

    registry = load_registry()
    reserved = {int(season) for season in season_usage(registry)}
    return tuple(season for season in range(start_season, end_season + 1) if season not in reserved)


def run_walk_forward(
    features: pd.DataFrame,
    *,
    eval_seasons: tuple[int, ...],
    start_season: int,
    end_season: int,
    min_train_games: int = MIN_TRAIN_GAMES,
    ridge_alpha: float = RIDGE_ALPHA,
    feature_profile: str = FEATURE_PROFILE,
) -> pd.DataFrame:
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["ats_margin"].notna()].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError("No completed NFL games in the requested window")

    passthrough = (
        "game_id",
        "season",
        "week",
        "gameday",
        "spread_line",
        "result",
        "ats_margin",
        "home_cover",
    )

    eval_set = set(eval_seasons)
    batches: list[pd.DataFrame] = []
    for (season, _week), weekly_games in test.groupby(["season", "week"], sort=True):
        if int(season) not in eval_set:
            continue
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target=TARGET,  # type: ignore[arg-type]
            model_name=REGRESSOR,
            ridge_alpha=ridge_alpha,
            feature_profile=feature_profile,  # type: ignore[arg-type]
        )
        predicted = model.predict(weekly_games)
        centers = predicted["predicted_margin"].to_numpy(dtype=float)
        spread = weekly_games["spread_line"].to_numpy(dtype=float)
        base = weekly_games.loc[:, list(passthrough)].copy()
        base["predicted_margin"] = centers
        base["distribution_rows"] = model.distribution_rows
        base["train_rows"] = model.training_rows

        for method in (BASELINE_ARM, CANDIDATE_ARM):
            batch = base.copy()
            if method == BASELINE_ARM:
                probability = predicted["home_cover_probability"].to_numpy(dtype=float)
            else:
                probability = smoothed_home_cover_probability(
                    model.residuals, centers, spread, method=method
                )
            batch["home_cover_probability"] = probability
            batch["feature_set"] = method
            batches.append(batch)

    if not batches:
        raise ValueError("No evaluation week had enough prior training games")
    predictions = pd.concat(batches, ignore_index=True)
    return predictions.sort_values(["gameday", "game_id", "feature_set"]).reset_index(drop=True)


def flip_report(predictions: pd.DataFrame) -> dict[str, Any]:
    baseline = predictions.loc[predictions["feature_set"].eq(BASELINE_ARM)].set_index("game_id")
    candidate = predictions.loc[predictions["feature_set"].eq(CANDIDATE_ARM)].set_index("game_id")
    common = baseline.index.intersection(candidate.index)
    baseline = baseline.loc[common]
    candidate = candidate.loc[common]
    base_side = baseline["home_cover_probability"].ge(0.5)
    cand_side = candidate["home_cover_probability"].ge(0.5)
    flipped = base_side.ne(cand_side)

    flipped_games = baseline.loc[flipped].copy()
    flipped_games["line_bucket"] = line_bucket(flipped_games["spread_line"])
    all_games = baseline.copy()
    all_games["line_bucket"] = line_bucket(all_games["spread_line"])
    flip_rate_by_bucket = (
        flipped_games.groupby("line_bucket").size() / all_games.groupby("line_bucket").size()
    ).fillna(0.0)

    return {
        "games": len(common),
        "flipped": int(flipped.sum()),
        "flip_rate": float(flipped.mean()),
        "flip_rate_by_line_bucket": {str(k): float(v) for k, v in flip_rate_by_bucket.items()},
        "flipped_game_ids": sorted(str(g) for g in flipped.loc[flipped].index),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    eval_seasons = non_reserved_seasons(start_season=args.start_season, end_season=args.end_season)
    print(f"Non-reserved evaluation seasons: {eval_seasons}")

    features = pd.read_parquet(args.features)
    predictions = run_walk_forward(
        features,
        eval_seasons=eval_seasons,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)

    scored = predictions.loc[predictions["home_cover"].notna()].copy()
    paired = pd.concat(
        [
            paired_feature_comparisons(
                scored,
                baseline_feature_set=BASELINE_ARM,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                scored,
                baseline_feature_set=BASELINE_ARM,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    paired.to_csv(output / "paired_comparisons.csv", index=False)

    flips = flip_report(predictions)
    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/smooth_cdf_mapping.md",
        "rotation_registry_touched": False,
        "eval_seasons": list(eval_seasons),
        "recipe": {
            "target": TARGET,
            "regressor": REGRESSOR,
            "ridge_alpha": RIDGE_ALPHA,
            "feature_profile": FEATURE_PROFILE,
            "min_train_games": MIN_TRAIN_GAMES,
            "feature_table": str(args.features),
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "scored_games": int(scored["game_id"].nunique()),
        "flip_report": flips,
        "distribution_rows_range": [
            int(predictions["distribution_rows"].min()),
            int(predictions["distribution_rows"].max()),
        ],
    }
    configuration = {
        "command": "smooth-cdf-mapping-measurement",
        "features": str(args.features),
        "start_season": args.start_season,
        "end_season": args.end_season,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    diagnostics["provenance"] = artifact_provenance(configuration, args.features, project_root=REPO)
    write_experiment_artifact(
        output,
        "diagnostics.json",
        diagnostics,
        command="smooth-cdf-mapping-measurement",
        metrics=diagnostics,
        notes=(
            "MOD-08 confirmation measurement: Gaussian CDF mapping vs. the production "
            "ECDF, close grade, restricted to non-rotation-reserved seasons."
        ),
    )

    print("\n=== week-blocked, paired (positive = gaussian better) ===")
    week_rows = paired.loc[paired["block"].eq("week")]
    print(
        week_rows.loc[
            :,
            ["metric", "estimate", "lower", "upper", "probability_positive", "paired_games"],
        ].to_string(index=False)
    )
    print(f"\nFlip rate: {flips['flipped']}/{flips['games']} ({flips['flip_rate']:.4%})")
    print(f"\nartifacts: {output}")


if __name__ == "__main__":
    main()
