"""Characterize the residual ECDF's location and screen recency/shrinkage on CFB.

Research item: ``docs/residual_location.md``. This is a SCREEN, not a
confirmation: it never calls ``nfl_ats.rotation.assign_window``/``record_look``,
runs CFB-only (free per rule 8 of the rotation registry), and reuses the exact
``fit_cfb_residual_model`` recipe the frozen XLG-03/MOD-16 CFB benchmark and
``docs/ecdf_smoothing.md`` already used, so every comparison shares the same
mean model and residual draws -- only the probability *reader* differs between
arms.

Two things happen in one walk-forward pass, to avoid fitting each week's model
twice:

1. **Characterization.** For every week, record the out-of-time residual
   sample's size, mean, median, std, and a cheap decomposition check: the raw
   (model-free) drift in ``ats_margin`` between the chronological early 80%
   (what the temporary calibration-split model was fit on) and the late 20%
   (what the residual sample IS) -- this is the model-free upper bound on how
   much of the residual location a pure "recent games differ from older
   games" story can explain, since a linear model with an unpenalized
   intercept trained only on the early slice reproduces close to zero mean
   residual on ITS OWN training data by construction.
2. **Screening.** Every week's games are scored under the production ``ecdf``
   control plus a sweep of recency-weighted and location-shrunk alternatives
   (``nfl_ats.residual_location``), producing a long ``feature_set``-tagged
   prediction table scored with the same ``paired_feature_comparisons``
   machinery ``scripts/ecdf_smoothing.py`` used.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
    fit_cfb_residual_model,
)
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.residual_location import (
    recency_weighted_home_cover_probability,
    shrunk_home_cover_probability,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"
DEFAULT_NFL_FEATURES = REPO / "data/processed/game_features_player.parquet"

BASELINE_ARM = "ecdf"
HALF_LIFE_SWEEP: tuple[float, ...] = (100.0, 200.0, 400.0, 800.0)
SHRINK_SWEEP: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

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


def _method_label(kind: str, value: float) -> str:
    if kind == "recency":
        return f"recency_hl{int(value)}"
    return f"shrink_{round(value * 100):03d}"


def _score_week_arms(weekly_games: pd.DataFrame, model: Any) -> pd.DataFrame:
    predicted = model.predict(weekly_games)
    centers = predicted["predicted_margin"].to_numpy(dtype=float)
    spread = weekly_games["spread_line"].to_numpy(dtype=float)
    base = weekly_games.loc[:, [c for c in _PASSTHROUGH if c in weekly_games.columns]].copy()
    base["predicted_margin"] = centers

    batches: list[pd.DataFrame] = []

    def _emit(label: str, probability: np.ndarray) -> None:
        batch = base.copy()
        batch["home_cover_probability"] = probability
        batch["feature_set"] = label
        batch["distribution_rows"] = model.distribution_rows
        batch["train_rows"] = model.training_rows
        batches.append(batch)

    _emit(BASELINE_ARM, predicted["home_cover_probability"].to_numpy(dtype=float))
    for half_life in HALF_LIFE_SWEEP:
        probability = recency_weighted_home_cover_probability(
            model.residuals, centers, spread, half_life_games=half_life
        )
        _emit(_method_label("recency", half_life), probability)
    for fraction in SHRINK_SWEEP:
        probability = shrunk_home_cover_probability(
            model.residuals, centers, spread, shrink_fraction=fraction
        )
        _emit(_method_label("shrink", fraction), probability)
    return pd.concat(batches, ignore_index=True)


def _raw_drift(training: pd.DataFrame, distribution_fraction: float = 0.20) -> dict[str, float]:
    """Model-free early/late ats_margin mean split, replicating the fit's own sort/split."""

    frame = training.loc[pd.to_numeric(training["ats_margin"], errors="coerce").notna()].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    target = pd.to_numeric(frame["ats_margin"], errors="raise").to_numpy(dtype=float)
    distribution_rows = int(len(frame) * distribution_fraction)
    split = len(frame) - distribution_rows
    early = target[:split]
    late = target[split:]
    return {
        "early_mean": float(np.mean(early)) if len(early) else float("nan"),
        "late_mean": float(np.mean(late)) if len(late) else float("nan"),
        "raw_drift": (float(np.mean(late)) - float(np.mean(early)))
        if len(early) and len(late)
        else float("nan"),
    }


def run_cfb(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    weekly_capture: Callable[[int, int, pd.DataFrame, Any], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``weekly_capture(season, week, weekly_games, model)`` is called once per

    scored week, after the week's model is fit and before it is scored, purely
    as an optional side-channel hook -- added so a re-measurement script (e.g.
    ``scripts/residual_location_reaudit.py``) can cache each week's out-of-time
    residual sample / predicted centres for an honest, mechanism-targeted
    bootstrap without duplicating this walk-forward loop or refitting anything
    a second time. ``None`` (the default) reproduces this function's original
    behaviour exactly -- no caller is required to pass it.
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
    location_rows: list[dict[str, Any]] = []
    for (season, week), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_cfb_residual_model(training, ridge_alpha=ridge_alpha)
        if weekly_capture is not None:
            weekly_capture(int(str(season)), int(str(week)), weekly_games, model)
        drift = _raw_drift(training)
        location_rows.append(
            {
                "league": "cfb",
                "season": int(str(season)),
                "week": int(str(week)),
                "train_rows": model.training_rows,
                "distribution_rows": model.distribution_rows,
                "residual_mean": float(np.mean(model.residuals)),
                "residual_median": float(np.median(model.residuals)),
                "residual_std": float(np.std(model.residuals, ddof=1)),
                **drift,
            }
        )
        prediction_batches.append(_score_week_arms(weekly_games, model))
    if not prediction_batches:
        raise ValueError("No CFB week had enough prior training games")
    predictions = pd.concat(prediction_batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "feature_set"]).reset_index(
        drop=True
    )
    location = pd.DataFrame(location_rows).sort_values(["season", "week"]).reset_index(drop=True)
    return predictions, location


def run_nfl_location(
    features: pd.DataFrame,
    *,
    start_season: int = 2009,
    end_season: int = 2025,
    min_train_games: int = 500,
    ridge_alpha: float = 10.0,
    feature_profile: str = "player",
) -> pd.DataFrame:
    """Informational-only NFL location characterization (no prediction scoring).

    Never a registry look: this only reads ``MarginModel.residuals`` off the
    already-frozen ``fit_margin_model`` recipe, exactly as
    ``scripts/ecdf_smoothing.py``'s NFL walk-forward does. No accuracy
    comparison is produced here -- only the location table, since the NFL
    ledger is scarce and this repository's own convention (docs/ecdf_smoothing.md)
    is to screen candidates on CFB and use NFL only informationally.
    """

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["ats_margin"].notna()].copy()
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError("No completed NFL games in the requested window")

    rows: list[dict[str, Any]] = []
    for (season, week), weekly_games in test.groupby(["season", "week"], sort=True):
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
        drift = _raw_drift(training)
        rows.append(
            {
                "league": "nfl",
                "season": int(str(season)),
                "week": int(str(week)),
                "train_rows": model.training_rows,
                "distribution_rows": model.distribution_rows,
                "residual_mean": float(np.mean(model.residuals)),
                "residual_median": float(np.median(model.residuals)),
                "residual_std": float(np.std(model.residuals, ddof=1)),
                **drift,
            }
        )
    if not rows:
        raise ValueError("No NFL week had enough prior training games")
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def season_location_table(location: pd.DataFrame) -> pd.DataFrame:
    grouped = location.groupby(["league", "season"])
    summary = grouped.agg(
        weeks=("week", "count"),
        mean_of_weekly_means=("residual_mean", "mean"),
        mean_of_weekly_medians=("residual_median", "mean"),
        min_distribution_rows=("distribution_rows", "min"),
        max_distribution_rows=("distribution_rows", "max"),
        mean_raw_drift=("raw_drift", "mean"),
    ).reset_index()
    return summary


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


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfb-features", type=Path, default=DEFAULT_CFB_FEATURES)
    parser.add_argument("--nfl-features", type=Path, default=DEFAULT_NFL_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument("--skip-nfl-location", action="store_true")
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    print("CFB walk-forward: characterizing residual location + screening recency/shrinkage arms")
    cfb_features = pd.read_parquet(args.cfb_features)
    cfb_predictions, cfb_location = run_cfb(cfb_features)
    cfb_predictions.to_parquet(output / "cfb_predictions.parquet", index=False)
    cfb_location.to_csv(output / "cfb_location.csv", index=False)

    season_table = season_location_table(cfb_location)
    season_table.to_csv(output / "season_location_table.csv", index=False)
    print("\n=== CFB residual location by season ===")
    print(season_table.to_string(index=False))

    correlation = float(np.corrcoef(cfb_location["residual_mean"], cfb_location["raw_drift"])[0, 1])
    slope, intercept = np.polyfit(cfb_location["raw_drift"], cfb_location["residual_mean"], 1)
    print(
        f"\nresidual_mean vs raw_drift (model-free early/late ats_margin split): "
        f"corr={correlation:.4f}, slope={slope:.4f}, intercept={intercept:.4f}"
    )

    cfb_paired = paired_evidence(
        cfb_predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    cfb_paired.to_csv(output / "cfb_paired_comparisons.csv", index=False)
    print("\n=== CFB clean-core, week-blocked, paired (positive = candidate better than ecdf) ===")
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

    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/residual_location.md",
        "rotation_registry_touched": False,
        "half_life_sweep": list(HALF_LIFE_SWEEP),
        "shrink_sweep": list(SHRINK_SWEEP),
        "residual_mean_vs_raw_drift_correlation": correlation,
        "residual_mean_vs_raw_drift_slope": float(slope),
        "residual_mean_vs_raw_drift_intercept": float(intercept),
        "cfb_distribution_rows_range": [
            int(cfb_location["distribution_rows"].min()),
            int(cfb_location["distribution_rows"].max()),
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }

    if not args.skip_nfl_location:
        print("\nNFL walk-forward (INFORMATIONAL location table only -- no NFL registry look)")
        nfl_features = pd.read_parquet(args.nfl_features)
        nfl_location = run_nfl_location(nfl_features)
        nfl_location.to_csv(output / "nfl_location.csv", index=False)
        nfl_season_table = season_location_table(nfl_location)
        nfl_season_table.to_csv(output / "nfl_season_location_table.csv", index=False)
        print("\n=== NFL residual location by season (informational) ===")
        print(nfl_season_table.to_string(index=False))
        diagnostics["nfl_distribution_rows_range"] = [
            int(nfl_location["distribution_rows"].min()),
            int(nfl_location["distribution_rows"].max()),
        ]

    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    print(f"\nartifacts: {output}")


if __name__ == "__main__":
    main()
