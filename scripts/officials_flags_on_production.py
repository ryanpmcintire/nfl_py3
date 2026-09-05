"""Run one predeclared, rotation-assigned opener confirmation for an
officiating-crew flag stacked on PRODUCTION (LEAD-34 crew second-meeting
favorite, LEAD-31 rookie-crew underdog).

Predeclared in ``docs/officials_crew_leads.md`` before either candidate was
scored. LEAD-32 (``crew_home_bias_on_production``) is deliberately absent
here: its own predeclared Stage-1 reliability gate (within-season split-half
``probability_positive`` > 0.5) measured 0.325 and was not cleared, so per
that doc's own pre-committed rule its Stage-2 screen is not run this
session (``nfl_ats.officials_flag_features.attach_crew_home_bias_features``
remains built and tested for any future re-run of the gate).

Both candidates read the Tuesday-OPENER consensus spread/total (never the
nflverse schedule's own closing line) and are built at runtime from the
already-captured local ``data/raw/officials/*`` snapshot -- no new fetch, no
precomputed candidate-specific parquet. Referee crew assignments are only
published Wednesday-Thursday of game week (after the Tuesday lock, per
``docs/referee_assignments_capture.md``), so both are late-week REFRESH-
channel candidates (``nfl_ats.crew_tilt_refresh_overlay`` is the existing
live vehicle for this pattern); this script builds and screens them at the
frozen Tuesday line like every other refresh channel already screened.

This is a thin wrapper around ``scripts/on_production_opener_confirmation.py``
(imported, never edited), identical in shape to
``scripts/schedule_flag_on_production.py`` / ``scripts/qb_identity_on_production.py``:
it reuses that module's ``profile_identity``, ``scoped_window_frame``,
``run_arm``, ``paired_frame``, ``summarize``, and ``null_distribution``
verbatim. ``null`` and ``positive-control`` are instrument checks; only
``screen`` is the single outcome look.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import on_production_opener_confirmation as confirmation  # noqa: E402

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.officials_flag_features import (  # noqa: E402
    ROOKIE_CREW_UNDERDOG_COLUMN,
    SECOND_MEETING_FAVORITE_COLUMN,
    attach_rookie_crew_underdog_features,
    attach_second_meeting_favorite_features,
    describe_crew_familiarity,
    describe_referee_left_censoring,
)
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)
from nfl_ats.rotation import load_registry  # noqa: E402
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot  # noqa: E402

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
DEFAULT_MARKET_ROOT = confirmation.DEFAULT_MARKET_ROOT
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"


@dataclass(frozen=True)
class OfficialsCandidate:
    """Duck-compatible with ``on_production_opener_confirmation.Candidate``."""

    family: str
    profile: str
    column: str
    attach: Any
    predeclaration: str
    artifact_dir: str


CANDIDATES: dict[str, OfficialsCandidate] = {
    "second_meeting": OfficialsCandidate(
        family="crew_second_meeting_favorite_on_production",
        profile="weak_stack_crew_second_meeting_favorite",
        column=SECOND_MEETING_FAVORITE_COLUMN,
        attach=attach_second_meeting_favorite_features,
        predeclaration="docs/officials_crew_leads.md#crew_second_meeting_favorite_on_production-lead-34",
        artifact_dir="officials_flags_on_production/second_meeting",
    ),
    "rookie_underdog": OfficialsCandidate(
        family="rookie_crew_underdog_on_production",
        profile="weak_stack_rookie_crew_underdog",
        column=ROOKIE_CREW_UNDERDOG_COLUMN,
        attach=attach_rookie_crew_underdog_features,
        predeclaration="docs/officials_crew_leads.md#rookie_crew_underdog_on_production-lead-31",
        artifact_dir="officials_flags_on_production/rookie_underdog",
    ),
}


def build_candidate_features(
    base_features: pd.DataFrame, candidate: OfficialsCandidate, schedule: pd.DataFrame
) -> pd.DataFrame:
    """Merge the one candidate officiating-crew flag onto the PRODUCTION table."""

    return candidate.attach(base_features, schedule=schedule)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=confirmation.NULL_PERMUTATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=confirmation.BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=confirmation.BOOTSTRAP_SEED)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    args = parser.parse_args()
    candidate = CANDIDATES[args.candidate]

    features_path = args.features or DEFAULT_FEATURES
    schedules_path = args.schedules or latest_schedules_snapshot(REPO_ROOT)
    base_features = pd.read_parquet(features_path)
    schedule = pd.read_parquet(schedules_path)
    features = build_candidate_features(base_features, candidate, schedule)

    identity = confirmation.profile_identity(candidate, features)
    scoped, seasons = confirmation.scoped_window_frame(
        features, load_registry(args.registry), candidate.family
    )
    started = time.time()
    baseline = confirmation.run_arm(
        scoped,
        candidate,
        market_root=args.market_root,
        profile=BASELINE_PROFILE,
        seasons=seasons,
        min_train_games=args.min_train_games,
        leak=False,
    )
    treatment = confirmation.run_arm(
        scoped,
        candidate,
        market_root=args.market_root,
        profile=candidate.profile,
        seasons=seasons,
        min_train_games=args.min_train_games,
        leak=args.mode == "positive-control",
    )
    paired = confirmation.paired_frame(baseline, treatment)
    if paired.empty:
        raise RuntimeError("No paired opener-grade games were scored")

    result: dict[str, Any] = {
        "status": "scored",
        "profile_identity": identity,
        "paired_games": len(paired),
        "paired_weeks": int(paired.groupby(["season", "week"]).ngroups),
    }
    if args.mode == "null":
        result["null_production_rule"] = confirmation.null_distribution(
            paired, probability_rule=True, permutations=args.permutations, seed=args.seed
        )
        result["null_sign_rule"] = confirmation.null_distribution(
            paired, probability_rule=False, permutations=args.permutations, seed=args.seed
        )
    else:
        for label, reference, treatment_col in (
            ("opener_production_rule", "baseline_correct_open_pr", "candidate_correct_open_pr"),
            ("opener_sign_rule", "baseline_correct_open", "candidate_correct_open"),
            ("close_production_rule", "baseline_correct_close_pr", "candidate_correct_close_pr"),
            ("close_sign_rule", "baseline_correct_close", "candidate_correct_close"),
        ):
            result[label] = confirmation.summarize(
                paired, reference, treatment_col, args.bootstrap_samples, args.seed
            )
        result["permutation_null_production_rule"] = confirmation.null_distribution(
            paired, probability_rule=True, permutations=args.permutations, seed=args.seed
        )
        result["baseline_metrics"] = confirmation.opener_evaluation_metrics(baseline)
        result["candidate_metrics"] = confirmation.opener_evaluation_metrics(treatment)
        result["picks_disagreeing_production_rule"] = int(
            (paired.baseline_pick_home_pr != paired.candidate_pick_home_pr).sum()
        )
        flagged = features[candidate.column]
        result["flag_summary"] = {
            "n_games_total": len(features),
            "n_games_nonzero_flag": int((flagged.fillna(0) != 0).sum()),
            "n_games_positive_flag": int((flagged.fillna(0) > 0).sum()),
            "n_games_negative_flag": int((flagged.fillna(0) < 0).sum()),
            "n_games_missing_flag": int(flagged.isna().sum()),
            "by_season_nonzero": {
                str(season): int((group.fillna(0) != 0).sum())
                for season, group in flagged.groupby(features["season"])
            },
        }
        if candidate.family == "crew_second_meeting_favorite_on_production":
            result["crew_familiarity_diagnostic"] = describe_crew_familiarity()
        if candidate.family == "rookie_crew_underdog_on_production":
            result["referee_left_censoring_diagnostic"] = describe_referee_left_censoring()

    configuration = {
        "candidate": args.candidate,
        "mode": args.mode,
        "family": candidate.family,
        "window_seasons": list(seasons),
        "grade": "opener",
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": candidate.profile,
        "candidate_column": candidate.column,
        "predeclaration": candidate.predeclaration,
        "features_path": str(features_path),
        "schedules_path": str(schedules_path),
        "market_root": str(args.market_root),
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, features_path, project_root=REPO_ROOT),
        "schedule_snapshot": {
            "path": str(schedules_path),
            "sha256": sha256_file(schedules_path),
        },
    }
    output = (
        REPO_ROOT
        / "artifacts"
        / candidate.artifact_dir
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="officials-flags-on-production",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Rotation-assigned opener confirmation for an officiating-crew "
            "late-week-refresh candidate (LEAD-34/LEAD-31); prediction-level "
            "paired output retained; the candidate column is computed at "
            "runtime from the local officials/game_penalties/schedules "
            "snapshots, never read from a precomputed parquet."
        ),
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
