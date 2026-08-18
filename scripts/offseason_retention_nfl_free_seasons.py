"""Ad hoc sweep of offseason_retention on the two genuinely unspent NFL seasons.

NOT a rotation-registry confirmation look: no family is declared and no window
is assigned or recorded (``nfl_ats.rotation`` is never called here). Per rule
8, iteration on seasons no family has reserved needs no registry entry, and
the binding instruction for this task is explicit: never spend a rotation
confirmation window.

``registry/rotation_registry.json`` as of 2026-08-17 records ``season_usage``
only for 2013-2017 and 2020-2021 (spent by ``pbp_drive_bundle``,
``player_qb_continuity``, ``best_pick_ranker``, ``best_pick_ranker_opener``,
and ``mod07_weak_signal_stack``). 2018-2025 carries a separate, undeclared
~130-150-look prose multiplicity discount (``docs/rotation_registry.md``) and
is avoided here per the task brief even though the JSON ledger does not
formally reserve it. That leaves **2011-2012** as the only NFL seasons no
family has ever scored against and that fall outside the broad 2018-2025
mining era. Two seasons is thin (~530 games) -- this is a secondary sanity
check, not a confirmation-grade look, and its low power is reported honestly
rather than dressed up.

RESOLVED CONFOUND (corrected 2026-08-18; no longer applies to any run of this
script): ``features.py``'s ``_build_features_pass`` used to call
``add_elo_features(games)`` without forwarding its own ``offseason_retention``
parameter, so every arm built through
``build_game_features(..., offseason_retention=r)`` would have kept its Elo
feature's offseason regression fixed at whatever
``DEFAULT_OFFSEASON_RETENTION`` equalled in ``constants.py``, regardless of
``r``. That call site was fixed in the same commit that added this script
(``7455d62``, ``features.py:492`` now reads
``add_elo_features(games, offseason_retention=offseason_retention)``), so the
confound described here never actually affected this script's own measured
grid -- it was disclosed defensively before the fix landed and the disclosure
was never removed. See ``docs/offseason_retention.md`` "Defect 1" for the
resolved history.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/offseason_retention_nfl_free_seasons.py \\
        --output <scratch dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.features import build_game_features
from nfl_ats.outcomes import walk_forward_outcomes
from nfl_ats.snapshots import latest_snapshot, load_snapshot

REPO = Path(__file__).resolve().parents[1]

RETENTION_GRID: tuple[float, ...] = (0.0, 0.337, 0.40, 0.67, 0.75)
BASELINE_RETENTION = 0.67
TEST_START_SEASON = 2011
TEST_END_SEASON = 2012
MIN_TRAIN_GAMES = 500
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260817


def _method_name(retention: float) -> str:
    return f"retention_{retention:.3f}".replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--grid", type=float, nargs="*", default=list(RETENTION_GRID))
    parser.add_argument("--start-season", type=int, default=TEST_START_SEASON)
    parser.add_argument("--end-season", type=int, default=TEST_END_SEASON)
    parser.add_argument("--min-train-games", type=int, default=MIN_TRAIN_GAMES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid = sorted(set(args.grid) | {BASELINE_RETENTION})

    print(
        f"grid: {grid}; test window {args.start_season}-{args.end_season} "
        "(unspent per rotation ledger, not a registry look)"
    )
    snapshot = latest_snapshot(args.raw_root)
    schedules, team_stats = load_snapshot(snapshot)

    predictions_by_method: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    market: pd.DataFrame | None = None
    for retention in grid:
        print(f"\nbuilding NFL 'base' features at offseason_retention={retention}")
        features = build_game_features(schedules, team_stats, offseason_retention=retention)
        result = walk_forward_outcomes(
            features,
            start_season=args.start_season,
            end_season=args.end_season,
            min_train_games=args.min_train_games,
            feature_profile="base",
            methods=("market", "market_residual"),
        )
        candidate = result.predictions.loc[
            result.predictions["method"].eq("market_residual")
        ].copy()
        method = _method_name(retention)
        candidate["method"] = method
        candidate["feature_set"] = method
        candidate["retention"] = retention
        predictions_by_method.append(candidate)
        if market is None:
            market = result.predictions.loc[result.predictions["method"].eq("market")].copy()
        summary = result.summary.copy()
        summary["retention"] = retention
        summaries.append(summary)
        print(
            summary.loc[
                :, ["method", "games", "cover_accuracy", "cover_brier_score", "margin_mae"]
            ].to_string(index=False)
        )

    assert market is not None
    market["feature_set"] = "market"
    predictions = pd.concat([market, *predictions_by_method], ignore_index=True)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.output / "summary_by_retention.csv", index=False
    )

    baseline_method = _method_name(BASELINE_RETENTION)
    method_names = [_method_name(r) for r in grid]
    candidate_only = predictions.loc[predictions["method"].isin(method_names)]
    accuracy_rows: list[pd.DataFrame] = []
    for block in ("week", "season"):
        if block == "season" and candidate_only["season"].nunique() < 2:
            continue
        comparison = paired_feature_comparisons(
            candidate_only,
            baseline_feature_set=baseline_method,
            samples=args.bootstrap_samples,
            block=block,
            seed=args.bootstrap_seed,
        )
        comparison["block"] = block
        accuracy_rows.append(comparison)
    accuracy = pd.concat(accuracy_rows, ignore_index=True)
    accuracy.to_csv(args.output / "paired_accuracy_brier.csv", index=False)

    print(
        f"\n=== week-blocked, each retention vs {BASELINE_RETENTION}, "
        f"{args.start_season}-{args.end_season} ==="
    )
    headline = accuracy.loc[accuracy["block"].eq("week")]
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
        "predeclaration": "docs/offseason_retention.md",
        "rotation_registry_touched": False,
        "grid": grid,
        "baseline_retention": BASELINE_RETENTION,
        "start_season": args.start_season,
        "end_season": args.end_season,
        "min_train_games": args.min_train_games,
        "known_confound": (
            "none. features.py:492 forwards offseason_retention to "
            "add_elo_features; a prior draft of this script's docstring "
            "described an elo-not-threaded confound, but that call site was "
            "fixed in the same commit that added this script (7455d62), "
            "before this script was ever run, so no diagnostics artifact from "
            "this script was ever affected. Corrected 2026-08-18 -- see "
            "docs/offseason_retention.md 'Defect 1'."
        ),
    }
    (args.output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"\nartifacts written to {args.output}")


if __name__ == "__main__":
    main()
