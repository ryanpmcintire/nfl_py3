"""Screen: does smoothing the out-of-time residual ECDF help cover calibration?

Predeclaration: ``docs/ecdf_smoothing.md`` (frozen before any confirmation
run). Rotation registry: untouched here -- this script never calls
``nfl_ats.rotation.assign_window``/``record_look``. Per rule 8, CFB and
non-reserved seasons are free; the CFB run below is the primary evidence,
mirroring how MOD-16 (``docs/margin_variance.md``) screened its own
distribution-work candidate on CFB before any NFL predeclaration. The NFL
walk-forward is a SECONDARY, INFORMATIONAL run only -- it re-scores the
already-mined 2018-2025 era (and free 2009-2017) purely to quantify how many
picks would move and whether that movement looks systematic; it is not a
registry "look" and must never be read as a confirmation result.

``margin.MarginModel`` builds its predictive distribution from a fixed
out-of-time residual SAMPLE (the trailing 20% chronological holdout in
``fit_margin_model``/``fit_cfb_residual_model``) and reads cover probability
off the discretized empirical CDF (``margin._smoothed_probability``). This
script re-scores identical weeks with the SAME residual draws and mean model,
changing only how the residual distribution is read: the raw ECDF (control,
reproduces production exactly) vs three smoothed alternatives fit with
``nfl_ats.calibration.fit_residual_smoother`` -- Gaussian, Gaussian KDE, and
skew-normal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.calibration import RESIDUAL_SMOOTHING_METHODS, smoothed_home_cover_probability
from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
    fit_cfb_residual_model,
)
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.key_numbers import line_bucket
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"
DEFAULT_NFL_FEATURES = REPO / "data/processed/game_features_player.parquet"

BASELINE_ARM = "ecdf"
CANDIDATE_METHODS: tuple[str, ...] = tuple(
    method for method in RESIDUAL_SMOOTHING_METHODS if method != BASELINE_ARM
)

_CFB_PASSTHROUGH = (
    "game_id",
    "season",
    "week",
    "gameday",
    "spread_line",
    "result",
    "ats_margin",
    "home_cover",
)
_NFL_PASSTHROUGH = (*_CFB_PASSTHROUGH, "predicted_market_residual")


def _score_week_arms(
    weekly_games: pd.DataFrame,
    model: Any,
    *,
    methods: tuple[str, ...],
    passthrough: tuple[str, ...],
) -> pd.DataFrame:
    """One row per (game, method): every arm shares the same mean model."""

    predicted = model.predict(weekly_games)
    centers = predicted["predicted_margin"].to_numpy(dtype=float)
    spread = weekly_games["spread_line"].to_numpy(dtype=float)
    base = weekly_games.loc[:, [c for c in passthrough if c in weekly_games.columns]].copy()
    base["predicted_margin"] = centers
    if "predicted_market_residual" not in base.columns:
        base["predicted_market_residual"] = predicted["predicted_market_residual"].to_numpy(
            dtype=float
        )

    batches: list[pd.DataFrame] = []
    for method in methods:
        batch = base.copy()
        if method == BASELINE_ARM:
            probability = predicted["home_cover_probability"].to_numpy(dtype=float)
        else:
            probability = smoothed_home_cover_probability(
                model.residuals, centers, spread, method=method
            )
        batch["home_cover_probability"] = probability
        batch["feature_set"] = method
        batch["distribution_rows"] = model.distribution_rows
        batch["train_rows"] = model.training_rows
        batches.append(batch)
    return pd.concat(batches, ignore_index=True)


def run_cfb_screen(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    methods: tuple[str, ...] = (BASELINE_ARM, *CANDIDATE_METHODS),
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
        raise ValueError("No completed CFB games in the requested window")

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_cfb_residual_model(training, ridge_alpha=ridge_alpha)
        batches.append(
            _score_week_arms(weekly_games, model, methods=methods, passthrough=_CFB_PASSTHROUGH)
        )
    if not batches:
        raise ValueError("No CFB week had enough prior training games")
    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    return predictions.sort_values(["gameday", "game_id", "feature_set"]).reset_index(drop=True)


def run_nfl_walk_forward(
    features: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    min_train_games: int,
    ridge_alpha: float = 10.0,
    feature_profile: str = "player",
    methods: tuple[str, ...] = (BASELINE_ARM, *CANDIDATE_METHODS),
) -> pd.DataFrame:
    """Informational only -- see module docstring. Not a registry look."""

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["ats_margin"].notna()].copy()
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError("No completed NFL games in the requested window")

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name="ridge",
            ridge_alpha=ridge_alpha,
            feature_profile=feature_profile,  # type: ignore[arg-type]
        )
        batches.append(
            _score_week_arms(weekly_games, model, methods=methods, passthrough=_NFL_PASSTHROUGH)
        )
    if not batches:
        raise ValueError("No NFL week had enough prior training games")
    predictions = pd.concat(batches, ignore_index=True)
    return predictions.sort_values(["gameday", "game_id", "feature_set"]).reset_index(drop=True)


def paired_evidence(predictions: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    windows = (
        list(predictions["evaluation_window"].unique())
        if "evaluation_window" in predictions.columns
        else [None]
    )
    for window in windows:
        subset = (
            predictions
            if window is None
            else predictions.loc[predictions["evaluation_window"].eq(window)]
        )
        if subset.empty:
            continue
        for block in ("week", "season"):
            if block == "season" and subset["season"].nunique() < 2:
                continue
            paired = paired_feature_comparisons(
                subset, baseline_feature_set=BASELINE_ARM, samples=samples, block=block, seed=seed
            )
            paired["evaluation_window"] = window if window is not None else "all"
            rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def flip_report(predictions: pd.DataFrame, *, candidate_method: str) -> dict[str, Any]:
    """How many picks a candidate method flips vs the ecdf control, and where."""

    baseline = predictions.loc[predictions["feature_set"].eq(BASELINE_ARM)].set_index("game_id")
    candidate = predictions.loc[predictions["feature_set"].eq(candidate_method)].set_index(
        "game_id"
    )
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

    # Distance-to-threshold proxy: |predicted_market_residual| for flipped vs all games,
    # when that column is available (NFL run only).
    magnitude_summary: dict[str, float] = {}
    if "predicted_market_residual" in baseline.columns:
        flipped_abs = flipped_games["predicted_market_residual"].abs()
        all_abs = all_games["predicted_market_residual"].abs()
        magnitude_summary = {
            "flipped_mean_abs_predicted_residual": float(flipped_abs.mean())
            if len(flipped_abs)
            else float("nan"),
            "all_mean_abs_predicted_residual": float(all_abs.mean()),
            "flipped_median_abs_predicted_residual": float(flipped_abs.median())
            if len(flipped_abs)
            else float("nan"),
            "all_median_abs_predicted_residual": float(all_abs.median()),
        }

    return {
        "candidate_method": candidate_method,
        "games": len(common),
        "flipped": int(flipped.sum()),
        "flip_rate": float(flipped.mean()),
        "flip_rate_by_line_bucket": {str(k): float(v) for k, v in flip_rate_by_bucket.items()},
        **magnitude_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfb-features", type=Path, default=DEFAULT_CFB_FEATURES)
    parser.add_argument("--nfl-features", type=Path, default=DEFAULT_NFL_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--skip-nfl", action="store_true", help="CFB-only run")
    parser.add_argument("--nfl-start-season", type=int, default=2009)
    parser.add_argument("--nfl-end-season", type=int, default=2025)
    parser.add_argument("--nfl-min-train-games", type=int, default=500)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    print("CFB screen: walking forward the ecdf control and three smoothed candidates")
    cfb_features = pd.read_parquet(args.cfb_features)
    cfb_predictions = run_cfb_screen(cfb_features)
    cfb_predictions.to_parquet(output / "cfb_predictions.parquet", index=False)
    cfb_paired = paired_evidence(
        cfb_predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    cfb_paired.to_csv(output / "cfb_paired_comparisons.csv", index=False)
    cfb_flips = [
        flip_report(
            cfb_predictions.loc[cfb_predictions["evaluation_window"].eq("clean_core")],
            candidate_method=method,
        )
        for method in CANDIDATE_METHODS
    ]

    print("\n=== CFB clean-core, week-blocked, paired (positive = smoothed better) ===")
    headline = cfb_paired.loc[
        cfb_paired["evaluation_window"].eq("clean_core") & cfb_paired["block"].eq("week")
    ]
    print(
        headline.loc[
            :,
            [
                "candidate_feature_set",
                "metric",
                "estimate",
                "lower",
                "upper",
                "probability_positive",
                "paired_games",
            ],
        ].to_string(index=False)
    )
    print("\n=== CFB clean-core pick flips vs the ecdf control ===")
    for report in cfb_flips:
        print(
            f"{report['candidate_method']:>12}: {report['flipped']}/{report['games']} flipped "
            f"({report['flip_rate']:.4%})"
        )

    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/ecdf_smoothing.md",
        "rotation_registry_touched": False,
        "cfb_flip_reports": cfb_flips,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }

    if not args.skip_nfl:
        print(
            "\nNFL walk-forward (INFORMATIONAL ONLY -- not a registry look): "
            "quantifying pick movement"
        )
        nfl_features = pd.read_parquet(args.nfl_features)
        nfl_predictions = run_nfl_walk_forward(
            nfl_features,
            start_season=args.nfl_start_season,
            end_season=args.nfl_end_season,
            min_train_games=args.nfl_min_train_games,
        )
        nfl_predictions.to_parquet(output / "nfl_predictions.parquet", index=False)
        nfl_predictions_scored = nfl_predictions.loc[nfl_predictions["home_cover"].notna()].copy()
        nfl_paired = pd.concat(
            [
                paired_feature_comparisons(
                    nfl_predictions_scored.assign(evaluation_window="nfl_informational"),
                    baseline_feature_set=BASELINE_ARM,
                    samples=args.bootstrap_samples,
                    block="week",
                    seed=args.bootstrap_seed,
                ).assign(evaluation_window="nfl_informational"),
                paired_feature_comparisons(
                    nfl_predictions_scored.assign(evaluation_window="nfl_informational"),
                    baseline_feature_set=BASELINE_ARM,
                    samples=args.bootstrap_samples,
                    block="season",
                    seed=args.bootstrap_seed,
                ).assign(evaluation_window="nfl_informational"),
            ],
            ignore_index=True,
        )
        nfl_paired.to_csv(output / "nfl_paired_comparisons.csv", index=False)
        nfl_flips = [
            flip_report(nfl_predictions, candidate_method=method) for method in CANDIDATE_METHODS
        ]
        diagnostics["nfl_flip_reports"] = nfl_flips
        diagnostics["nfl_distribution_rows_range"] = [
            int(nfl_predictions["distribution_rows"].min()),
            int(nfl_predictions["distribution_rows"].max()),
        ]
        diagnostics["nfl_games_with_distribution_rows_518"] = int(
            (
                nfl_predictions.loc[
                    nfl_predictions["feature_set"].eq(BASELINE_ARM), "distribution_rows"
                ]
                == 518
            ).sum()
        )

        print("\n=== NFL (informational), week-blocked, paired (positive = smoothed better) ===")
        nfl_headline = nfl_paired.loc[nfl_paired["block"].eq("week")]
        print(
            nfl_headline.loc[
                :,
                [
                    "candidate_feature_set",
                    "metric",
                    "estimate",
                    "lower",
                    "upper",
                    "probability_positive",
                    "paired_games",
                ],
            ].to_string(index=False)
        )
        print("\n=== NFL pick flips vs the ecdf control ===")
        for report in nfl_flips:
            print(
                f"{report['candidate_method']:>12}: {report['flipped']}/{report['games']} flipped "
                f"({report['flip_rate']:.4%})"
            )

    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    print(f"\nartifacts: {output}")


if __name__ == "__main__":
    main()
