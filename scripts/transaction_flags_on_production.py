"""Run one predeclared, rotation-assigned opener confirmation for a
transaction-wire candidate stacked on PRODUCTION (Wave 6: LEAD-12 holdout
slow-start fade, LEAD-23 trade-deadline integration drag, LEAD-14
suspension-return rust).

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 6" before any of the
three candidates below was scored. Every candidate column is built from the
newest local ``schedules.parquet`` snapshot plus the newest local PFR
transaction-wire index and snap-count history
(``nfl_ats.transaction_flag_features``) and merged onto the PRODUCTION
feature table (``data/processed/game_features_weak_stack.parquet``, the
same table every sibling on-production candidate profile is pinned to) by
``game_id`` at runtime -- no precomputed candidate-specific parquet is
written.

This is a thin wrapper around ``scripts/on_production_opener_confirmation.py``
(imported, never edited): it reuses that module's ``profile_identity``,
``scoped_window_frame``, ``run_arm``, ``paired_frame``, ``summarize``, and
``null_distribution`` verbatim -- the same estimator as the played
``weak_stack`` chain, the same week-blocked bootstrap, the same
within-week permutation null, the same positive-control leak (the
candidate column replaced by the REALIZED margin), and the same refusal to
infer a confirmation window from the command line. ``null`` and
``positive-control`` are instrument checks; only ``screen`` is the single
outcome look.

A sibling script rather than an extension of
``scripts/schedule_flag_on_production.py``'s or
``scripts/qb_identity_on_production.py``'s own ``CANDIDATES`` map, for the
same reason each of those gives for not extending an earlier one: this
candidate's own inputs (the PFR transaction-wire index plus snap-count
history) do not match either file's own loader signature, and both were
(or may still be) concurrently edited by other fleet lanes this same
session.
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
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.rotation import load_registry  # noqa: E402
from nfl_ats.transaction_flag_features import (  # noqa: E402
    DEADLINE_INTEGRATION_DRAG_COLUMN,
    HOLDOUT_SLOW_START_COLUMN,
    SUSPENSION_RETURN_RUST_COLUMN,
    attach_deadline_integration_drag_features,
    attach_holdout_slow_start_features,
    attach_suspension_return_rust_features,
    default_schedule,
    default_snap_counts,
    default_transactions_index,
    describe_deadline_acquisition_population,
    describe_holdout_population,
    describe_suspension_return_population,
    latest_pfr_transactions_snapshot,
)

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"


@dataclass(frozen=True)
class TransactionCandidate:
    """Duck-compatible with ``on_production_opener_confirmation.Candidate``:
    carries the same ``family``/``profile``/``column`` attribute names, so
    the template's ``profile_identity``/``scoped_window_frame``/``run_arm``
    accept it unmodified."""

    family: str
    profile: str
    column: str
    predeclaration: str
    artifact_dir: str


CANDIDATES: dict[str, TransactionCandidate] = {
    "holdout_slow_start": TransactionCandidate(
        family="holdout_slow_start_on_production",
        profile="weak_stack_holdout_slow_start",
        column=HOLDOUT_SLOW_START_COLUMN,
        predeclaration="docs/schedule_flag_battery.md#section-17----lead-12-holdout-slow-start-fade",
        artifact_dir="transaction_flags_on_production/holdout_slow_start",
    ),
    "deadline_drag": TransactionCandidate(
        family="deadline_integration_drag_on_production",
        profile="weak_stack_deadline_drag",
        column=DEADLINE_INTEGRATION_DRAG_COLUMN,
        predeclaration="docs/schedule_flag_battery.md#section-18----lead-23-trade-deadline-integration-drag",
        artifact_dir="transaction_flags_on_production/deadline_drag",
    ),
    "suspension_rust": TransactionCandidate(
        family="suspension_return_rust_on_production",
        profile="weak_stack_suspension_rust",
        column=SUSPENSION_RETURN_RUST_COLUMN,
        predeclaration="docs/schedule_flag_battery.md#section-19----lead-14-suspension-return-rust",
        artifact_dir="transaction_flags_on_production/suspension_rust",
    ),
}


def build_candidate_features(
    base_features: pd.DataFrame,
    candidate: TransactionCandidate,
    *,
    schedule: pd.DataFrame,
    transactions_index: pd.DataFrame,
    snap_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the one candidate transaction-wire column onto the PRODUCTION
    table."""

    if candidate.family == "holdout_slow_start_on_production":
        return attach_holdout_slow_start_features(
            base_features,
            schedule=schedule,
            transactions_index=transactions_index,
            snap_counts=snap_counts,
        )
    if candidate.family == "deadline_integration_drag_on_production":
        return attach_deadline_integration_drag_features(
            base_features,
            schedule=schedule,
            transactions_index=transactions_index,
            snap_counts=snap_counts,
        )
    if candidate.family == "suspension_return_rust_on_production":
        return attach_suspension_return_rust_features(
            base_features,
            schedule=schedule,
            transactions_index=transactions_index,
            snap_counts=snap_counts,
        )
    raise ValueError(f"unrecognized candidate family: {candidate.family}")


def population_diagnostic(
    candidate: TransactionCandidate,
    *,
    schedule: pd.DataFrame,
    transactions_index: pd.DataFrame,
    snap_counts: pd.DataFrame,
) -> dict[str, Any]:
    if candidate.family == "holdout_slow_start_on_production":
        return describe_holdout_population(transactions_index, snap_counts)
    if candidate.family == "deadline_integration_drag_on_production":
        return describe_deadline_acquisition_population(transactions_index, snap_counts)
    if candidate.family == "suspension_return_rust_on_production":
        return describe_suspension_return_population(transactions_index, snap_counts, schedule)
    raise ValueError(f"unrecognized candidate family: {candidate.family}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--pfr-snapshot", type=Path, default=None)
    parser.add_argument("--market-root", type=Path, default=confirmation.DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=confirmation.NULL_PERMUTATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=confirmation.BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=confirmation.BOOTSTRAP_SEED)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    args = parser.parse_args()
    candidate = CANDIDATES[args.candidate]

    features_path = args.features or DEFAULT_FEATURES
    base_features = pd.read_parquet(features_path)
    schedule = pd.read_parquet(args.schedules) if args.schedules is not None else default_schedule()
    pfr_snapshot_path = (
        args.pfr_snapshot / "index.parquet"
        if args.pfr_snapshot is not None
        else latest_pfr_transactions_snapshot()
    )
    transactions_index = default_transactions_index(snapshot=pfr_snapshot_path)
    snap_counts = default_snap_counts()

    features = build_candidate_features(
        base_features,
        candidate,
        schedule=schedule,
        transactions_index=transactions_index,
        snap_counts=snap_counts,
    )

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
        raw = features[candidate.column]
        result["flag_summary"] = {
            "n_games_total": len(features),
            "n_games_nonzero_flag": int((raw.fillna(0) != 0).sum()),
            "n_games_positive_flag": int((raw.fillna(0) > 0).sum()),
            "n_games_negative_flag": int((raw.fillna(0) < 0).sum()),
            "n_games_missing_flag": int(raw.isna().sum()),
            "by_season_nonzero": {
                str(season): int((group.fillna(0) != 0).sum())
                for season, group in raw.groupby(features["season"])
            },
        }
        result["population_diagnostic"] = population_diagnostic(
            candidate,
            schedule=schedule,
            transactions_index=transactions_index,
            snap_counts=snap_counts,
        )

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
        "pfr_snapshot": str(pfr_snapshot_path),
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
        command="transaction-flags-on-production",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Rotation-assigned opener confirmation for a PFR transaction-wire "
            "candidate (LEAD-12/LEAD-23/LEAD-14, Wave 6); prediction-level "
            "paired output retained; the candidate column is computed at "
            "runtime from the local transaction-wire index/snap-count "
            "snapshots, never read from a precomputed parquet."
        ),
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
