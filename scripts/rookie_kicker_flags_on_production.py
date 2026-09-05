"""Run one predeclared, rotation-assigned opener confirmation for LEAD-24
stage 2 (rookie-wall dependence fade) or LEAD-16 (midweek kicker change),
each stacked on PRODUCTION ("Wave 8" of ``docs/schedule_flag_battery.md``).

Every flag is built from already-captured local snapshots -- the LEAD-24
Stage 1 rookie-wall panel/dependence metric
(:mod:`nfl_ats.rookie_wall`/:mod:`nfl_ats.rookie_kicker_flag_features`), the
local PFR transaction-wire index plus the pinned local snap-count snapshot
(LEAD-16), and the Tuesday-opener consensus spread from the local market
archive (LEAD-16 only) -- and merged onto the PRODUCTION feature table
(``data/processed/game_features_weak_stack.parquet``) by ``game_id`` at
runtime -- no precomputed candidate-specific parquet is written for LEAD-16.

This is a thin wrapper around ``scripts/on_production_opener_confirmation.py``
(imported, never edited), the same reuse pattern
``scripts/schedule_flag_on_production.py`` already established: it reuses
that module's ``profile_identity``, ``scoped_window_frame``, ``run_arm``,
``paired_frame``, ``summarize``, and ``null_distribution`` verbatim -- the
same estimator as the played ``weak_stack`` chain, the same week-blocked
bootstrap, the same within-week permutation null, the same positive-control
leak (the candidate column replaced by the REALIZED margin), and the same
refusal to infer a confirmation window from the command line. ``null`` and
``positive-control`` are instrument checks; only ``screen`` is the single
outcome look.

The LEAD-24 stage-2 candidate's dependence table (the expensive part: it
rebuilds the full age-curves player panel) is computed ONCE per process and
cached across the ``--mode`` runs a single invocation performs; pass
``--dependence-table`` to reuse an already-written parquet across SEPARATE
invocations (three ``--mode`` runs) instead of rebuilding it three times.
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
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)
from nfl_ats.rookie_kicker_flag_features import (  # noqa: E402
    KICKER_CHANGE_COLUMN,
    ROOKIE_WALL_DEPENDENCE_COLUMN,
    attach_kicker_change_underdog_features,
    attach_rookie_wall_dependence_fade_features,
    describe_kicker_change_population,
    rookie_wall_dependence_table,
)
from nfl_ats.rotation import load_registry  # noqa: E402
from nfl_ats.schedule_flag_features import DEFAULT_MARKET_ROOT  # noqa: E402
from nfl_ats.transaction_flag_features import (  # noqa: E402
    default_snap_counts,
    default_transactions_index,
)
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot  # noqa: E402

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"


@dataclass(frozen=True)
class Candidate:
    """Duck-compatible with ``on_production_opener_confirmation.Candidate``:
    carries the same ``family``/``profile``/``column`` attribute names, so
    the template's ``profile_identity``/``scoped_window_frame``/``run_arm``
    accept it unmodified."""

    family: str
    profile: str
    column: str
    predeclaration: str
    artifact_dir: str


CANDIDATES: dict[str, Candidate] = {
    "rookie_wall_dependence": Candidate(
        family="rookie_wall_dependence_on_production",
        profile="weak_stack_rookie_wall_dependence",
        column=ROOKIE_WALL_DEPENDENCE_COLUMN,
        predeclaration="docs/schedule_flag_battery.md#section-12----lead-24-stage-2-rookie-wall-dependence-fade",
        artifact_dir="rookie_kicker_flags_on_production/rookie_wall_dependence",
    ),
    "kicker_change": Candidate(
        family="kicker_change_underdog_on_production",
        profile="weak_stack_kicker_change_underdog",
        column=KICKER_CHANGE_COLUMN,
        predeclaration="docs/schedule_flag_battery.md#section-13----lead-16-midweek-kicker-change",
        artifact_dir="rookie_kicker_flags_on_production/kicker_change",
    ),
}


def build_candidate_features(
    base_features: pd.DataFrame,
    candidate_key: str,
    schedule: pd.DataFrame,
    *,
    dependence_table: pd.DataFrame | None,
    market_root: Path,
) -> pd.DataFrame:
    """Merge the one candidate flag onto the PRODUCTION table."""

    if candidate_key == "rookie_wall_dependence":
        return attach_rookie_wall_dependence_fade_features(
            base_features, schedule=schedule, dependence_table=dependence_table
        )
    if candidate_key == "kicker_change":
        return attach_kicker_change_underdog_features(
            base_features, schedule=schedule, market_root=market_root
        )
    raise ValueError(f"unknown candidate {candidate_key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--dependence-table",
        type=Path,
        default=None,
        help="rookie_wall_dependence only: reuse an already-written "
        "team-week dependence parquet instead of rebuilding the LEAD-24 "
        "Stage 1 panel from raw snapshots (expensive; identical across "
        "--mode invocations for the same snapshots)",
    )
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

    dependence_table = None
    if args.candidate == "rookie_wall_dependence":
        if args.dependence_table is not None:
            dependence_table = pd.read_parquet(args.dependence_table)
        else:
            dependence_table, _ = rookie_wall_dependence_table()

    features = build_candidate_features(
        base_features,
        args.candidate,
        schedule,
        dependence_table=dependence_table,
        market_root=args.market_root,
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
        if args.candidate == "kicker_change":
            result["population"] = describe_kicker_change_population(
                default_transactions_index(), default_snap_counts()
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
        command="rookie-kicker-flags-on-production",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Rotation-assigned opener confirmation for LEAD-24 stage 2 "
            "(rookie-wall dependence fade) or LEAD-16 (midweek kicker "
            "change); prediction-level paired output retained."
        ),
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
