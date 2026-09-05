"""Run the predeclared, rotation-assigned opener confirmation for LEAD-15
(backup tenure-gap valuation) stacked on PRODUCTION.

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 9" before any outcome
was scored. The one candidate column, ``backup_tenure_gap_flag``, is built
ONLY from the newest ``data/raw/*/schedules.parquet`` snapshot's listed
starters plus the PINNED
``data/players/raw/20260817T184901Z/weekly_rosters.parquet``
(``nfl_ats.backup_tenure_flag_features``) -- never lane AB's out-of-scope
depth-chart archive -- and merged onto the PRODUCTION feature table
(``data/processed/game_features_weak_stack.parquet``, the same table every
sibling on-production candidate profile is pinned to) by ``game_id`` at
runtime -- no precomputed candidate-specific parquet is written.

This is a thin wrapper around ``scripts/on_production_opener_confirmation.py``
(imported, never edited), mirroring ``scripts/schedule_flag_on_production.py``'s
own thin-wrapper shape: it reuses that module's ``profile_identity``,
``scoped_window_frame``, ``run_arm``, ``paired_frame``, ``summarize``, and
``null_distribution`` verbatim -- the same estimator as the played
``weak_stack`` chain, the same week-blocked bootstrap, the same within-week
permutation null, the same positive-control leak (the candidate column
replaced by the REALIZED margin), and the same refusal to infer a
confirmation window from the command line. ``null`` and ``positive-control``
are instrument checks; only ``screen`` is the single outcome look.
"""

from __future__ import annotations

import argparse
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

import on_production_opener_confirmation as confirmation  # noqa: E402

from nfl_ats.backup_tenure_flag_features import (  # noqa: E402
    BACKUP_TENURE_GAP_COLUMN,
    DEFAULT_WEEKLY_ROSTERS_PATH,
    attach_backup_tenure_gap_features,
    default_weekly_rosters,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)
from nfl_ats.rotation import load_registry  # noqa: E402

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
DEFAULT_MARKET_ROOT = confirmation.DEFAULT_MARKET_ROOT
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"

FAMILY = "backup_tenure_gap_on_production"
PROFILE = "weak_stack_backup_tenure_gap"
PREDECLARATION = "docs/schedule_flag_battery.md#section-22----backup_tenure_gap_on_production"
ARTIFACT_DIR = "backup_tenure_flag_on_production"


class _Candidate:
    """Duck-compatible with ``on_production_opener_confirmation.Candidate``:
    carries the same ``family``/``profile``/``column`` attribute names, so
    the template's ``profile_identity``/``scoped_window_frame``/``run_arm``
    accept it unmodified."""

    family = FAMILY
    profile = PROFILE
    column = BACKUP_TENURE_GAP_COLUMN


CANDIDATE = _Candidate()


def _latest_schedule_snapshot() -> Path:
    """Newest ``data/raw/*/schedules.parquet``, duplicated from
    ``nfl_ats.backup_tenure_flag_features.default_schedule``'s identical glob
    so this script always has a concrete path for provenance metadata."""

    candidates = sorted((REPO_ROOT / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {REPO_ROOT}")
    return candidates[-1]


def build_candidate_features(
    base_features: pd.DataFrame, schedule: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    """Merge ``backup_tenure_gap_flag`` onto the PRODUCTION table."""

    return attach_backup_tenure_gap_features(base_features, schedule=schedule, rosters=rosters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--rosters", type=Path, default=None)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=confirmation.NULL_PERMUTATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=confirmation.BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=confirmation.BOOTSTRAP_SEED)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    args = parser.parse_args()

    features_path = args.features or DEFAULT_FEATURES
    schedules_path = args.schedules or _latest_schedule_snapshot()
    rosters_path = args.rosters or DEFAULT_WEEKLY_ROSTERS_PATH
    schedule = pd.read_parquet(schedules_path)
    rosters = default_weekly_rosters(rosters_path)
    base_features = pd.read_parquet(features_path)
    features = build_candidate_features(base_features, schedule, rosters)

    identity = confirmation.profile_identity(CANDIDATE, features)
    scoped, seasons = confirmation.scoped_window_frame(
        features, load_registry(args.registry), CANDIDATE.family
    )
    started = time.time()
    baseline = confirmation.run_arm(
        scoped,
        CANDIDATE,
        market_root=args.market_root,
        profile=BASELINE_PROFILE,
        seasons=seasons,
        min_train_games=args.min_train_games,
        leak=False,
    )
    treatment = confirmation.run_arm(
        scoped,
        CANDIDATE,
        market_root=args.market_root,
        profile=CANDIDATE.profile,
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
        flagged = features[CANDIDATE.column]
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

    configuration = {
        "mode": args.mode,
        "family": CANDIDATE.family,
        "window_seasons": list(seasons),
        "grade": "opener",
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": CANDIDATE.profile,
        "candidate_column": CANDIDATE.column,
        "predeclaration": PREDECLARATION,
        "features_path": str(features_path),
        "schedules_path": str(schedules_path),
        "rosters_path": str(rosters_path),
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
        "schedule_snapshot": {"path": str(schedules_path), "sha256": sha256_file(schedules_path)},
    }

    output = REPO_ROOT / "artifacts" / ARTIFACT_DIR / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="backup-tenure-flags-on-production",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Rotation-assigned opener confirmation for LEAD-15 (backup "
            "tenure-gap valuation); prediction-level paired output "
            "retained; the candidate column is computed at runtime from "
            "the schedule + pinned weekly-rosters snapshots, never read "
            "from a precomputed parquet."
        ),
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
