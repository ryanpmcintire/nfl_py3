"""Screen recency-weighted TRAINING ROWS for the CFB mean model (not the ECDF reader).

Research item: ``docs/residual_location.md`` sec 5, "Predeclaration a future
session would freeze (recency-weighted mean model, not the ECDF reader)".
That document screened recency-weighting and shrinkage of the out-of-time
*residual* sample (the ECDF ``ecdf_smoothing`` calibration draws) and found
every candidate leaned or resolved negative -- but flagged a different,
untested lever: the deployed mean model's own Ridge fit is an unweighted
average over its entire, ever-growing training history, i.e. recency-blind
in a way the residual-reader screen never touched. This script screens THAT
lever: fit the temporary (leading-80%) and final (100%) Ridge estimators
``fit_cfb_residual_model`` (``src/nfl_ats/cfb_benchmark.py``) builds each
walk-forward week with a per-row exponential recency weight,
``0.5 ** (games_ago / half_life_games)``, instead of the current implicit
uniform weight. The out-of-time residual sample and how it is READ (the
unweighted ECDF) are left exactly as production has them -- unchanged --
isolating this as an independent candidate from the already-screened reader
family.

This is a SCREEN, not a confirmation: it never calls
``nfl_ats.rotation.assign_window``/``record_look`` and runs CFB-only (free
per rule 8 of the rotation registry, ``docs/rotation_registry.md``). Per the
predeclaration, an NFL rotation-registry window is only ever predeclared --
not spent -- if the CFB read here clears ``probability_positive >= 0.75`` on
at least one half-life; this script does not touch the registry either way.

``src/`` is intentionally left unmodified: ``fit_cfb_residual_model_recency_
weighted`` below duplicates ``nfl_ats.cfb_benchmark.fit_cfb_residual_model``'s
body (rather than adding a ``sample_weight`` parameter to the frozen
function) so the production recipe stays byte-for-byte untouched while this
research-only script varies one thing about it.

Grade: ``nflverse_spread`` per the predeclaration -- ``cfb_game_features.
parquet``'s own ``spread_line``, not an archived opener snapshot, matching
``ecdf_smoothing``'s and ``residual_location_screen.py``'s reasoning (the
opener pool's three NFL windows are too scarce to spend confirming an
unscreened candidate).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_DISTRIBUTION_FRACTION,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_REGRESSOR,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.data import DataContractError
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import MarginModel, make_margin_estimator
from nfl_ats.residual_location import recency_weights

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"

BASELINE_ARM = "ecdf"
# Frozen exactly as docs/residual_location.md sec 5 predeclared -- no arm
# additions (the document's own text notes a wider/longer-half-life grid
# would be "worth adding" given distribution_rows now runs to 2,499, but the
# orchestrator running this predeclaration instructed running it frozen).
HALF_LIFE_SWEEP: tuple[float, ...] = (100.0, 200.0, 400.0, 800.0)

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


def _method_label(half_life_games: float) -> str:
    return f"train_recency_hl{int(half_life_games)}"


def fit_cfb_residual_model_recency_weighted(
    training: pd.DataFrame,
    *,
    half_life_games: float,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    distribution_fraction: float = CFB_BENCHMARK_DISTRIBUTION_FRACTION,
    min_distribution_rows: int = 10,
    random_state: int = 42,
    feature_columns: tuple[str, ...] = CFB_MODEL_FEATURE_COLUMNS,
) -> MarginModel:
    """``fit_cfb_residual_model``'s exact recipe, with recency-weighted Ridge fits.

    Duplicated from ``nfl_ats.cfb_benchmark.fit_cfb_residual_model`` rather
    than parameterizing that frozen function, so the production path is
    untouched by this research-only screen. The only change: both the
    temporary (leading-``1 - distribution_fraction``) and final (100% of
    ``training``) Ridge fits take a per-row exponential recency weight,
    ``nfl_ats.residual_location.recency_weights`` -- the most recent row in
    EACH fit's own training set gets weight 1, halving every
    ``half_life_games`` games back. The out-of-time residual sample (and how
    it is later read into a probability) is computed identically to
    production and is NOT reweighted -- that lever was already screened
    (and left unweighted, i.e. rejected) by ``docs/residual_location.md``
    sec 4.
    """

    required = {"game_id", "gameday", "ats_margin", *feature_columns}
    missing = sorted(required.difference(training.columns))
    if missing:
        raise DataContractError(f"CFB margin training is missing columns: {', '.join(missing)}")
    if not 0.10 <= distribution_fraction < 0.5:
        raise ValueError("distribution_fraction must be in [0.10, 0.5)")

    frame = training.loc[pd.to_numeric(training["ats_margin"], errors="coerce").notna()].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError("At least 50 completed games are required for a CFB margin model")
    distribution_rows = int(len(frame) * distribution_fraction)
    if distribution_rows < min_distribution_rows or len(frame) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time residual distribution")

    split = len(frame) - distribution_rows
    target = pd.to_numeric(frame["ats_margin"], errors="raise").to_numpy(dtype=float)

    temporary_weights = recency_weights(split, half_life_games=half_life_games)
    temporary = make_margin_estimator(
        CFB_BENCHMARK_REGRESSOR, random_state, ridge_alpha=ridge_alpha
    )
    temporary.fit(
        frame.iloc[:split].loc[:, list(feature_columns)],
        target[:split],
        regressor__sample_weight=temporary_weights,
    )
    calibration_prediction = np.asarray(
        temporary.predict(frame.iloc[split:].loc[:, list(feature_columns)]), dtype=float
    )
    residuals = np.asarray(target[split:] - calibration_prediction, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < min_distribution_rows:
        raise ValueError("Out-of-time residual distribution has too few finite values")

    final_weights = recency_weights(len(frame), half_life_games=half_life_games)
    estimator = make_margin_estimator(
        CFB_BENCHMARK_REGRESSOR, random_state, ridge_alpha=ridge_alpha
    )
    estimator.fit(
        frame.loc[:, list(feature_columns)],
        target,
        regressor__sample_weight=final_weights,
    )
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name=CFB_BENCHMARK_REGRESSOR,
        ridge_alpha=ridge_alpha,
        target="market_residual",
        feature_columns=feature_columns,
        training_rows=len(frame),
        distribution_rows=len(residuals),
        training_max_gameday=frame["gameday"].max().date().isoformat(),
    )


def _score_arm(weekly_games: pd.DataFrame, model: MarginModel, label: str) -> pd.DataFrame:
    predicted = model.predict(weekly_games)
    batch = weekly_games.loc[:, [c for c in _PASSTHROUGH if c in weekly_games.columns]].copy()
    batch["predicted_margin"] = predicted["predicted_margin"].to_numpy(dtype=float)
    batch["home_cover_probability"] = predicted["home_cover_probability"].to_numpy(dtype=float)
    batch["feature_set"] = label
    batch["distribution_rows"] = model.distribution_rows
    batch["train_rows"] = model.training_rows
    return batch


def run_cfb(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    half_life_sweep: tuple[float, ...] = HALF_LIFE_SWEEP,
) -> pd.DataFrame:
    """One walk-forward pass scoring the unweighted baseline plus every recency-
    weighted-training candidate on identical weeks/games, mirroring
    ``scripts/residual_location_screen.py``'s ``run_cfb`` structure.
    """

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

    prediction_batches: list[pd.DataFrame] = []
    for (_season, _week), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        baseline_model = fit_cfb_residual_model(training, ridge_alpha=ridge_alpha)
        week_batches = [_score_arm(weekly_games, baseline_model, BASELINE_ARM)]
        for half_life in half_life_sweep:
            weighted_model = fit_cfb_residual_model_recency_weighted(
                training, half_life_games=half_life, ridge_alpha=ridge_alpha
            )
            week_batches.append(_score_arm(weekly_games, weighted_model, _method_label(half_life)))
        prediction_batches.append(pd.concat(week_batches, ignore_index=True))
    if not prediction_batches:
        raise ValueError("No CFB week had enough prior training games")
    predictions = pd.concat(prediction_batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "feature_set"]).reset_index(
        drop=True
    )
    return predictions


def paired_evidence(predictions: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for window in predictions["evaluation_window"].unique():
        subset = predictions.loc[predictions["evaluation_window"].eq(window)]
        if subset.empty:
            continue
        for block in ("week", "season"):
            if block == "season" and subset["season"].nunique() < 2:
                continue
            paired = paired_feature_comparisons(
                subset, baseline_feature_set=BASELINE_ARM, samples=samples, block=block, seed=seed
            )
            paired["evaluation_window"] = window
            rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfb-features", type=Path, default=DEFAULT_CFB_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=CFB_BENCHMARK_START_SEASON)
    parser.add_argument("--end-season", type=int, default=CFB_BENCHMARK_END_SEASON)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print(
        f"CFB walk-forward ({args.start_season}-{args.end_season}): screening recency-weighted "
        "training rows against the unweighted-training-sample production model"
    )
    cfb_features = pd.read_parquet(args.cfb_features)
    predictions = run_cfb(cfb_features, start_season=args.start_season, end_season=args.end_season)
    predictions.to_parquet(output / "cfb_predictions.parquet", index=False)
    fit_elapsed = time.monotonic() - started
    print(f"walk-forward fitting complete in {fit_elapsed:.1f}s")

    paired = paired_evidence(predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    paired.to_csv(output / "cfb_paired_comparisons.csv", index=False)

    print(
        "\n=== CFB clean-core, season-blocked (primary), "
        "paired (positive = candidate beats ecdf) ==="
    )
    season_headline = paired.loc[
        paired["evaluation_window"].eq("clean_core") & paired["block"].eq("season")
    ]
    print(
        season_headline.loc[
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

    print("\n=== CFB clean-core, week-blocked (corroborating), paired ===")
    week_headline = paired.loc[
        paired["evaluation_window"].eq("clean_core") & paired["block"].eq("week")
    ]
    print(
        week_headline.loc[
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

    total_elapsed = time.monotonic() - started
    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/residual_location.md sec 5",
        "lever": "recency-weighted training rows on the temporary+final Ridge estimators; "
        "the out-of-time residual sample and the ECDF reader stay unweighted/unchanged",
        "rotation_registry_touched": False,
        "half_life_sweep": list(HALF_LIFE_SWEEP),
        "start_season": args.start_season,
        "end_season": args.end_season,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "fit_elapsed_seconds": fit_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "n_weeks": int(predictions.loc[predictions["feature_set"].eq(BASELINE_ARM)].shape[0]),
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    print(f"\ntotal elapsed: {total_elapsed:.1f}s")
    print(f"artifacts: {output}")


if __name__ == "__main__":
    main()
