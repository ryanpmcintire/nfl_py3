"""Run one predeclared, rotation-assigned opener confirmation for a
quarterback-identity candidate stacked on PRODUCTION (Wave 5: LEAD-20
rookie-QB debut fade, LEAD-25 quarterback revenge game).

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 5" before either of
the two candidates below was scored. Every candidate column is built from
the newest local ``schedules.parquet`` snapshot's listed starters plus
already-captured local roster/combine data (``nfl_ats.qb_identity_features``)
and merged onto the PRODUCTION feature table
(``data/processed/game_features_weak_stack.parquet``, the same table every
sibling on-production candidate profile is pinned to) by ``game_id`` at
runtime -- no precomputed candidate-specific parquet is written.

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

A sibling script rather than an extension of either
``scripts/schedule_flag_on_production.py``'s or
``scripts/pbp_trait_on_production.py``'s own ``CANDIDATES`` map, for the
same reason the latter gives for not extending the former: each of those
files was being concurrently edited by a different fleet lane this same
session, and this candidate's own inputs (listed schedule starters plus
local roster/combine snapshots) do not match either file's own loader
signature. Zero risk of colliding with a concurrent edit to either file.
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
from nfl_ats.qb_identity_features import (  # noqa: E402
    DEFAULT_COMBINE_RAW_ROOT,
    DEFAULT_PLAYERS_RAW_ROOT,
    QB_REVENGE_COLUMN,
    ROOKIE_QB_DEBUT_FADE_COLUMN,
    attach_qb_revenge_features,
    attach_rookie_qb_debut_fade_features,
    default_combine,
    default_schedule,
    default_weekly_rosters,
    describe_rookie_qb_debut_population,
    draft_team_by_gsis_id,
    qb_revenge_join_diagnostics,
)
from nfl_ats.rotation import load_registry  # noqa: E402

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"


@dataclass(frozen=True)
class QbIdentityCandidate:
    """Duck-compatible with ``on_production_opener_confirmation.Candidate``:
    carries the same ``family``/``profile``/``column`` attribute names, so
    the template's ``profile_identity``/``scoped_window_frame``/``run_arm``
    accept it unmodified."""

    family: str
    profile: str
    column: str
    predeclaration: str
    artifact_dir: str


CANDIDATES: dict[str, QbIdentityCandidate] = {
    "rookie_debut": QbIdentityCandidate(
        family="rookie_qb_debut_fade_on_production",
        profile="weak_stack_rookie_qb_debut_fade",
        column=ROOKIE_QB_DEBUT_FADE_COLUMN,
        predeclaration="docs/schedule_flag_battery.md#section-15----lead-20-rookie-qb-debut-fade",
        artifact_dir="qb_identity_on_production/rookie_debut",
    ),
    "qb_revenge": QbIdentityCandidate(
        family="qb_revenge_on_production",
        profile="weak_stack_qb_revenge",
        column=QB_REVENGE_COLUMN,
        predeclaration="docs/schedule_flag_battery.md#section-16----lead-25-quarterback-revenge-game",
        artifact_dir="qb_identity_on_production/qb_revenge",
    ),
}


def build_candidate_features(
    base_features: pd.DataFrame,
    candidate: QbIdentityCandidate,
    *,
    schedule: pd.DataFrame,
    rosters: pd.DataFrame,
    combine: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the one candidate QB-identity column onto the PRODUCTION table."""

    if candidate.family == "rookie_qb_debut_fade_on_production":
        return attach_rookie_qb_debut_fade_features(
            base_features, schedule=schedule, rosters=rosters
        )
    if candidate.family == "qb_revenge_on_production":
        lookup = draft_team_by_gsis_id(combine, rosters)
        return attach_qb_revenge_features(
            base_features, schedule=schedule, draft_team_lookup=lookup
        )
    raise ValueError(f"unrecognized candidate family: {candidate.family}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--players-raw-root", type=Path, default=DEFAULT_PLAYERS_RAW_ROOT)
    parser.add_argument("--combine-raw-root", type=Path, default=DEFAULT_COMBINE_RAW_ROOT)
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
    if args.players_raw_root == DEFAULT_PLAYERS_RAW_ROOT:
        rosters = default_weekly_rosters()
    else:
        from nfl_ats.players import canonicalize_rosters, latest_player_snapshot

        snapshot = latest_player_snapshot(args.players_raw_root)
        rosters = canonicalize_rosters(pd.read_parquet(snapshot.rosters_path))
    combine = (
        default_combine()
        if args.combine_raw_root == DEFAULT_COMBINE_RAW_ROOT
        else pd.read_parquet(sorted(args.combine_raw_root.glob("*/combine.parquet"))[-1])
    )
    features = build_candidate_features(
        base_features, candidate, schedule=schedule, rosters=rosters, combine=combine
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
        if candidate.family == "rookie_qb_debut_fade_on_production":
            result["rookie_debut_population_diagnostic"] = describe_rookie_qb_debut_population(
                schedule, rosters
            )
        if candidate.family == "qb_revenge_on_production":
            lookup = draft_team_by_gsis_id(combine, rosters)
            result["qb_revenge_join_diagnostic"] = qb_revenge_join_diagnostics(schedule, lookup)

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
        command="qb-identity-on-production",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Rotation-assigned opener confirmation for a quarterback-identity "
            "candidate (LEAD-20/LEAD-25 Wave 5); prediction-level paired "
            "output retained; the candidate column is computed at runtime "
            "from the local schedule/roster/combine snapshots, never read "
            "from a precomputed parquet."
        ),
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
