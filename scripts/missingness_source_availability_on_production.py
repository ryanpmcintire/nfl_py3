"""MOD-13: source-era missingness, scored against the production incumbent.

The predeclaration is ``docs/missingness_audit.md`` (written 2026-09-01
before this script was scored).  The candidate preserves the seven measured
lineup-continuity values, suppresses only their implicit per-column imputer
indicators, and supplies one explicit ``roster_continuity_data_available``
flag.  Production ``weak_stack`` is the unchanged comparator.

Run ``null``, then ``positive-control``, then ``screen`` exactly once on the
rotation-assigned opener window.  The screen writes both close and opener
prediction-level rows, but the opener is the decision grade.  The control is
deliberately leaky: it places realized ATS margin in the explicit availability
slot, proving the paired evaluator can detect an injected effect.  It is never
promotable and is not a claim about the real source-availability feature.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import per13_durability_on_production as evaluator  # noqa: E402

from nfl_ats.constants import SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS  # noqa: E402
from nfl_ats.io import atomic_parquet  # noqa: E402
from nfl_ats.margin import MarginFeatureProfile  # noqa: E402
from nfl_ats.missingness_availability import (  # noqa: E402
    ROSTER_CONTINUITY_DATA_AVAILABLE,
    add_roster_continuity_availability,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BASELINE_PROFILE: MarginFeatureProfile = "weak_stack"
CANDIDATE_PROFILE: MarginFeatureProfile = "weak_stack_source_availability"
ROTATION_FAMILY = "missingness_availability_flags"
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"
DEFAULT_OPENER_ARTIFACT = (
    REPO_ROOT / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
)


def _configure_evaluator() -> None:
    """Reuse the frozen paired evaluator without duplicating its mechanics."""

    evaluator.BASELINE_PROFILE = BASELINE_PROFILE
    evaluator.CANDIDATE_PROFILE = CANDIDATE_PROFILE
    evaluator.CONTROL_COLUMN = ROSTER_CONTINUITY_DATA_AVAILABLE


def _summary(fitted: pd.DataFrame, *, samples: int, seed: int, permutations: int) -> dict[str, Any]:
    graded = evaluator.grade(fitted)
    return {
        "home_pick_rate": {
            arm: float(graded[column].ge(0.5).mean())
            for arm, column in evaluator.ARM_PROBABILITY.items()
        },
        "permutation_null": evaluator.null_distribution(
            fitted, permutations=permutations, seed=seed
        ),
        "candidate_vs_baseline": evaluator.summarize_pair(graded, samples=samples, seed=seed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--seasons", required=True, help="assigned inclusive rotation range")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--opener-artifact", type=Path, default=DEFAULT_OPENER_ARTIFACT)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=evaluator.BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=evaluator.SEED)
    parser.add_argument("--out", default="", help="artifact directory name override")
    args = parser.parse_args()
    start, _, end = args.seasons.partition("-")
    seasons = tuple(range(int(start), int(end or start) + 1))
    if not args.opener_artifact.is_file():
        raise FileNotFoundError(f"Opener pairing artifact is absent: {args.opener_artifact}")

    _configure_evaluator()
    started = time.time()
    features = add_roster_continuity_availability(pd.read_parquet(args.features))
    opener = pd.read_parquet(args.opener_artifact)
    leak = args.mode == "positive-control"

    # Instrument checks run on the opener grade, which is the stated decision
    # grade.  The single screen reuses the same fitted sequence for its
    # required transparent close-grade secondary report.
    opener_fitted = evaluator.run_window(
        features,
        seasons,
        leak_treatment=leak,
        grade_name="opener",
        opener_frame=opener,
    )
    if opener_fitted.empty:
        raise RuntimeError("The assigned opener window produced no scored games")

    if args.mode == "null":
        result: dict[str, Any] = {
            "opener": {
                "status": "scored",
                "null": evaluator.null_distribution(
                    opener_fitted, permutations=args.permutations, seed=args.seed
                ),
            }
        }
        outputs = {"opener": opener_fitted}
    else:
        close_fitted = evaluator.run_window(
            features,
            seasons,
            leak_treatment=leak,
            grade_name="close",
        )
        result = {
            "opener": _summary(
                opener_fitted,
                samples=args.bootstrap_samples,
                seed=args.seed,
                permutations=args.permutations,
            ),
            "close": _summary(
                close_fitted,
                samples=args.bootstrap_samples,
                seed=args.seed,
                permutations=args.permutations,
            ),
        }
        outputs = {"opener": opener_fitted, "close": close_fitted}

    configuration = {
        "mode": args.mode,
        "seasons": list(seasons),
        "rotation_family": ROTATION_FAMILY,
        "decision_grade": "opener",
        "secondary_grade": "close",
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": CANDIDATE_PROFILE,
        "source_columns": list(SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS),
        "availability_flag": ROSTER_CONTINUITY_DATA_AVAILABLE,
        "positive_control": "realized ats_margin in availability flag (deliberate leak)",
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "predeclaration": "docs/missingness_audit.md",
        "features_path": str(args.features),
        "opener_artifact": str(args.opener_artifact),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = (
        REPO_ROOT / "artifacts" / (args.out or "missingness_availability_flags") / timestamp
    )
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="missingness-source-availability-on-production",
        metrics={"mode": args.mode, "decision_grade": "opener"},
        notes="MOD-13 source-era availability candidate vs the production weak_stack incumbent.",
    )
    for grade_name, frame in outputs.items():
        atomic_parquet(frame, output_dir / f"predictions_{grade_name}.parquet")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
