"""Run one predeclared, rotation-assigned opener confirmation for a
pure-schedule flag stacked on PRODUCTION (LEAD-21/LEAD-22/LEAD-40).

Predeclared in ``docs/schedule_flag_battery.md`` before any of the three was
scored: LEAD-21 post-overtime fatigue, LEAD-22 Monday-night-road short week,
LEAD-40 home-Thursday rest compound. Every flag is built ONLY from the
newest ``data/raw/*/schedules.parquet`` snapshot
(``nfl_ats.schedule_flag_features``) and merged onto the PRODUCTION feature
table (``data/processed/game_features_weak_stack.parquet``, the same table
every sibling on-production candidate profile is pinned to) by ``game_id``
at runtime -- no precomputed candidate-specific parquet is written, since
every input is a deterministic schedule fact.

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
from nfl_ats.rotation import load_registry  # noqa: E402
from nfl_ats.schedule_flag_features import (  # noqa: E402
    attach_home_thursday_features,
    attach_mnf_road_short_week_features,
    attach_post_ot_fatigue_features,
)
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot  # noqa: E402

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
DEFAULT_MARKET_ROOT = confirmation.DEFAULT_MARKET_ROOT
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"


@dataclass(frozen=True)
class ScheduleCandidate:
    """Duck-compatible with ``on_production_opener_confirmation.Candidate``:
    carries the same ``family``/``profile``/``column`` attribute names, so
    the template's ``profile_identity``/``scoped_window_frame``/``run_arm``
    accept it unmodified."""

    family: str
    profile: str
    column: str
    attach: Any
    predeclaration: str
    artifact_dir: str


CANDIDATES: dict[str, ScheduleCandidate] = {
    "post_ot": ScheduleCandidate(
        family="post_ot_fatigue_on_production",
        profile="weak_stack_post_ot",
        column="post_ot_fatigue_flag",
        attach=attach_post_ot_fatigue_features,
        predeclaration="docs/schedule_flag_battery.md#lead-21-post-overtime-fatigue",
        artifact_dir="schedule_flag_on_production/post_ot",
    ),
    "mnf_road": ScheduleCandidate(
        family="mnf_road_short_week_on_production",
        profile="weak_stack_mnf_road",
        column="mnf_road_short_week_flag",
        attach=attach_mnf_road_short_week_features,
        predeclaration="docs/schedule_flag_battery.md#lead-22-monday-night-road-short-week",
        artifact_dir="schedule_flag_on_production/mnf_road",
    ),
    "home_thursday": ScheduleCandidate(
        family="home_thursday_on_production",
        profile="weak_stack_home_thursday",
        column="home_thursday_flag",
        attach=attach_home_thursday_features,
        predeclaration="docs/schedule_flag_battery.md#lead-40-home-thursday-rest-compound",
        artifact_dir="schedule_flag_on_production/home_thursday",
    ),
}


def build_candidate_features(
    base_features: pd.DataFrame, candidate: ScheduleCandidate, schedule: pd.DataFrame
) -> pd.DataFrame:
    """Merge the one candidate schedule flag onto the PRODUCTION table."""

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
        command="schedule-flag-on-production",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Rotation-assigned opener confirmation for a pure-schedule flag "
            "(LEAD-21/22/40); prediction-level paired output retained; the "
            "candidate column is computed at runtime from the schedules "
            "snapshot, never read from a precomputed parquet."
        ),
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
