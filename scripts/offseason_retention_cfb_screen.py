"""Screen the offseason_retention correction on the free CFB benchmark.

Predeclared BEFORE any arm was scored (see ``docs/offseason_retention.md``):
the retention grid, ``min_train_games``, bootstrap samples/seed, evaluation
windows, and the comparison baseline (0.67, the currently shipped default).
Rotation registry: untouched -- rule 8 makes CFB and its benchmark free, and
no NFL confirmation window is spent by this script.

CFB has far more roster turnover than the NFL (more scholarship players,
shorter careers, transfer portal attrition), so whatever retention value is
CFB-optimal is NOT the NFL answer and this script does not claim otherwise.
What it CAN establish: does moving retention from 0.67 toward the NFL-side
estimate (~0.35-0.40) help, hurt, or make no detectable difference on a large,
independent walk-forward benchmark that uses the identical mechanism --
``cfb_features.py`` mirrors ``features.py``'s formula verbatim ("NFL
parameters taken verbatim", per its own module docstring). That makes this a
direction-and-sanity check on the functional form, not a transfer estimate of
the correct NFL value.

Every arm reuses the frozen XLG-03 harness (``cfb_benchmark.py``): same
regressor, ridge alpha, training floor, and residual-distribution recipe;
only ``offseason_retention`` differs across feature builds. Each grid point's
``market_residual`` predictions are tagged with a retention-specific method
name and compared, pairwise, against the 0.67 arm on identical weeks and
identical games.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/offseason_retention_cfb_screen.py \\
        --output <scratch dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_START_SEASON,
    cfb_walk_forward_benchmark,
)
from nfl_ats.cfb_features import build_cfb_game_features, load_cfb_benchmark_inputs
from nfl_ats.cfb_opponent_adjustment import paired_margin_error_comparison
from nfl_ats.experiments import paired_feature_comparisons

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_ROOT = REPO / "data" / "cfb"

# Predeclared grid: spans the shipped default (0.67) and the three NFL-side
# route estimates measured in scripts/offseason_retention_routes.py (route1
# horizon-4 median ~0.37, route2 ~0.379, route3 full-season ~0.400), a
# no-retention control, and a symmetric point above 0.67.
RETENTION_GRID: tuple[float, ...] = (0.0, 0.20, 0.337, 0.40, 0.50, 0.67, 0.75)
BASELINE_RETENTION = 0.67
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260817
EVALUATION_WINDOWS = ("clean_core", "thin_2006_2011", "regime_2020", "all")


def _method_name(retention: float) -> str:
    return f"retention_{retention:.3f}".replace(".", "_")


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfb-root", type=Path, default=DEFAULT_CFB_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=CFB_BENCHMARK_START_SEASON)
    parser.add_argument("--end-season", type=int, default=CFB_BENCHMARK_END_SEASON)
    parser.add_argument("--min-train-games", type=int, default=CFB_BENCHMARK_MIN_TRAIN_GAMES)
    parser.add_argument("--grid", type=float, nargs="*", default=list(RETENTION_GRID))
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid = sorted(set(args.grid) | {BASELINE_RETENTION})

    print(f"grid: {grid}")
    print("loading CFB schedules/lines/pbp ...")
    schedules, lines, pbp = load_cfb_benchmark_inputs(
        args.cfb_root, args.start_season, args.end_season
    )

    market_predictions: pd.DataFrame | None = None
    method_predictions: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for retention in grid:
        print(f"\nbuilding CFB features at offseason_retention={retention}")
        features, _audit = build_cfb_game_features(
            schedules,
            lines,
            pbp,
            start_season=args.start_season,
            end_season=args.end_season,
            offseason_retention=retention,
        )
        result = cfb_walk_forward_benchmark(
            features,
            start_season=args.start_season,
            end_season=args.end_season,
            min_train_games=args.min_train_games,
        )
        if market_predictions is None:
            market_predictions = result.predictions.loc[
                result.predictions["method"].eq("market")
            ].copy()
        candidate = result.predictions.loc[
            result.predictions["method"].eq("market_residual")
        ].copy()
        method = _method_name(retention)
        candidate["method"] = method
        candidate["feature_set"] = method
        candidate["retention"] = retention
        method_predictions.append(candidate)
        summary = result.summary.copy()
        summary["retention"] = retention
        summaries.append(summary)
        headline = summary.loc[summary["evaluation_window"].eq("clean_core")]
        print(
            headline.loc[
                :, ["method", "games", "cover_accuracy", "cover_brier_score", "margin_mae"]
            ].to_string(index=False)
        )

    assert market_predictions is not None
    market_predictions["feature_set"] = "market"
    predictions = pd.concat([market_predictions, *method_predictions], ignore_index=True)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all.to_csv(args.output / "summary_by_retention.csv", index=False)

    baseline_method = _method_name(BASELINE_RETENTION)
    method_names = [_method_name(r) for r in grid]
    accuracy_rows: list[pd.DataFrame] = []
    margin_rows: list[pd.DataFrame] = []
    for window in EVALUATION_WINDOWS:
        subset = (
            predictions
            if window == "all"
            else predictions.loc[predictions["evaluation_window"].eq(window)]
        )
        candidate_subset = subset.loc[subset["method"].isin(method_names)]
        if candidate_subset.empty:
            continue
        for block in ("week", "season"):
            if block == "season" and candidate_subset["season"].nunique() < 2:
                continue
            accuracy = paired_feature_comparisons(
                candidate_subset,
                baseline_feature_set=baseline_method,
                samples=args.bootstrap_samples,
                block=block,
                seed=args.bootstrap_seed,
            )
            accuracy["window"] = window
            accuracy["block"] = block
            accuracy_rows.append(accuracy)
            for retention in grid:
                if retention == BASELINE_RETENTION:
                    continue
                candidate_method = _method_name(retention)
                margin = paired_margin_error_comparison(
                    candidate_subset,
                    baseline_method=baseline_method,
                    candidate_method=candidate_method,
                    samples=args.bootstrap_samples,
                    block=block,
                    seed=args.bootstrap_seed,
                )
                margin["retention"] = retention
                margin["window"] = window
                margin_rows.append(margin)

    accuracy_all = pd.concat(accuracy_rows, ignore_index=True)
    margin_all = pd.concat(margin_rows, ignore_index=True)
    accuracy_all.to_csv(args.output / "paired_accuracy_brier.csv", index=False)
    margin_all.to_csv(args.output / "paired_margin.csv", index=False)

    print("\n=== headline: clean core, week-blocked, each retention vs 0.67 ===")
    headline_acc = accuracy_all.loc[
        accuracy_all["window"].eq("clean_core") & accuracy_all["block"].eq("week")
    ]
    print(
        headline_acc.loc[
            :,
            [
                "candidate_feature_set",
                "metric",
                "estimate",
                "lower",
                "upper",
                "probability_positive",
            ],
        ].to_string(index=False)
    )
    headline_margin = margin_all.loc[
        margin_all["window"].eq("clean_core") & margin_all["block"].eq("week")
    ]
    print(
        headline_margin.loc[
            :, ["retention", "metric", "estimate", "lower", "upper", "probability_positive"]
        ].to_string(index=False)
    )

    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/offseason_retention.md",
        "hypothesis_frozen_before_scoring": True,
        "rotation_registry_touched": False,
        "grid": grid,
        "baseline_retention": BASELINE_RETENTION,
        "start_season": args.start_season,
        "end_season": args.end_season,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (args.output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"\nartifacts written to {args.output}")


if __name__ == "__main__":
    main()
