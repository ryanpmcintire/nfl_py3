"""Run one predeclared, rotation-assigned opener confirmation for a
PBP coaching-trait candidate stacked on PRODUCTION (Wave 4: LEAD-26 opening-
drive EPA, LEAD-27 third-quarter point differential, LEAD-30's fourth-down
aggression x opener-spread interaction).

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 4" before any of the
three candidates below was scored, gated on lane J's split-half reliability
measurement (``docs/pbp_trait_reliability.md``) for the underlying traits.
Every candidate column is built entirely from the local PBP snapshot (plus,
for the fourth-down interaction only, the Tuesday-OPENER consensus spread
from the local market archive) and merged onto the PRODUCTION feature table
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

**Why a sibling script rather than an extension of
``scripts/schedule_flag_on_production.py``'s own ``CANDIDATES`` map**
(the way lane F and lane H each extended it in place): that script's
``ScheduleCandidate.attach`` interface is hard-wired to a ``schedule``
keyword (a ``schedules.parquet`` frame) and its ``main()`` always loads and
passes one -- correct for a pure-schedule flag, but wrong for a PBP-derived
trait, which needs a play-by-play snapshot (and, for LEAD-30, the opener
market store) instead. Extending that file's ``main()`` to branch between a
schedule input and a PBP input is a structural change to a file lane H was
editing concurrently this same session, not an additive one; this sibling
script reuses the actual shared estimator
(``on_production_opener_confirmation``) at the same level every other
on-production wrapper in this repo already does, with zero risk of
colliding with a concurrent edit to the schedule-flag file.
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
from nfl_ats.pbp_trait_on_production_features import (  # noqa: E402
    DEFAULT_MARKET_ROOT,
    FOURTH_DOWN_INTERACTION_COLUMN,
    attach_fourth_down_interaction_features,
    attach_opening_drive_epa_features,
    attach_q3_point_diff_features,
    load_pbp_panel,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.rotation import load_registry  # noqa: E402

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"
DEFAULT_PBP_ROOT = REPO_ROOT / "data/pbp/raw"


@dataclass(frozen=True)
class PbpTraitCandidate:
    """Duck-compatible with ``on_production_opener_confirmation.Candidate``:
    carries the same ``family``/``profile``/``column`` attribute names, so
    the template's ``profile_identity``/``scoped_window_frame``/``run_arm``
    accept it unmodified."""

    family: str
    profile: str
    column: str
    predeclaration: str
    artifact_dir: str


CANDIDATES: dict[str, PbpTraitCandidate] = {
    "opening_drive": PbpTraitCandidate(
        family="opening_drive_script_on_production",
        profile="weak_stack_opening_drive_epa",
        column="opening_drive_epa",
        predeclaration="docs/schedule_flag_battery.md#section-12----lead-26-opening-drive-script-efficiency",
        artifact_dir="pbp_trait_on_production/opening_drive",
    ),
    "q3_diff": PbpTraitCandidate(
        family="q3_adjustment_on_production",
        profile="weak_stack_q3_point_diff",
        column="q3_point_diff",
        predeclaration="docs/schedule_flag_battery.md#section-13----lead-27-third-quarter-adjustments",
        artifact_dir="pbp_trait_on_production/q3_diff",
    ),
    "fourth_down_interaction": PbpTraitCandidate(
        family="fourth_down_aggression_interaction_on_production",
        profile="weak_stack_fourth_down_interaction",
        column=FOURTH_DOWN_INTERACTION_COLUMN,
        predeclaration="docs/schedule_flag_battery.md#section-14----lead-30-fourth-down-aggression-x-opener-spread-interaction",
        artifact_dir="pbp_trait_on_production/fourth_down_interaction",
    ),
}


def build_candidate_features(
    base_features: pd.DataFrame,
    candidate: PbpTraitCandidate,
    *,
    pbp: pd.DataFrame,
    market_root: Path,
) -> pd.DataFrame:
    """Merge the one candidate PBP-trait column onto the PRODUCTION table."""

    if candidate.family == "opening_drive_script_on_production":
        return attach_opening_drive_epa_features(base_features, pbp=pbp)
    if candidate.family == "q3_adjustment_on_production":
        return attach_q3_point_diff_features(base_features, pbp=pbp)
    if candidate.family == "fourth_down_aggression_interaction_on_production":
        return attach_fourth_down_interaction_features(
            base_features, pbp=pbp, market_root=market_root
        )
    raise ValueError(f"unrecognized candidate family: {candidate.family}")


def fourth_down_side_split(features: pd.DataFrame, paired: pd.DataFrame) -> dict[str, Any]:
    """Diagnostic-only split of LEAD-30's single interaction column into its
    two predeclared sides (aggressive-dog games, interaction > 0; aggressive-
    favourite games, interaction < 0). Never a separate registry cell -- the
    interaction IS the family (docs/schedule_flag_battery.md "Wave 4"); this
    exists only to make the claimed asymmetry visible in the write-up.
    """

    raw = features[["game_id", FOURTH_DOWN_INTERACTION_COLUMN]].copy()
    raw["game_id"] = raw["game_id"].astype(str)
    joined = paired.merge(raw, on="game_id", how="left", validate="one_to_one")

    def _side_summary(mask: pd.Series) -> dict[str, Any]:
        subset = joined.loc[mask].dropna(
            subset=["baseline_correct_open_pr", "candidate_correct_open_pr"]
        )
        if subset.empty:
            return {"n_games": 0, "delta_accuracy": None}
        delta = float(
            (subset["candidate_correct_open_pr"] - subset["baseline_correct_open_pr"]).mean()
        )
        return {
            "n_games": len(subset),
            "candidate_accuracy": float(subset["candidate_correct_open_pr"].mean()),
            "baseline_accuracy": float(subset["baseline_correct_open_pr"].mean()),
            "delta_accuracy": delta,
        }

    column = joined[FOURTH_DOWN_INTERACTION_COLUMN]
    return {
        "aggressive_dog_games": _side_summary(column > 0),
        "aggressive_favorite_games": _side_summary(column < 0),
        "neutral_or_missing_games": _side_summary((column == 0) | column.isna()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--pbp-root", type=Path, default=DEFAULT_PBP_ROOT)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=confirmation.NULL_PERMUTATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=confirmation.BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=confirmation.BOOTSTRAP_SEED)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    args = parser.parse_args()
    candidate = CANDIDATES[args.candidate]

    features_path = args.features or DEFAULT_FEATURES
    base_features = pd.read_parquet(features_path)
    pbp = load_pbp_panel(args.pbp_root)
    features = build_candidate_features(
        base_features, candidate, pbp=pbp, market_root=args.market_root
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
        result["candidate_summary"] = {
            "n_games_total": len(features),
            "n_games_known": int(raw.notna().sum()),
            "n_games_missing": int(raw.isna().sum()),
            "n_games_positive": int((raw.fillna(0) > 0).sum()),
            "n_games_negative": int((raw.fillna(0) < 0).sum()),
            "n_games_zero": int((raw.fillna(float("nan")) == 0).sum()),
            "mean": float(raw.mean()) if raw.notna().any() else None,
            "std": float(raw.std()) if raw.notna().any() else None,
        }
        if candidate.column == FOURTH_DOWN_INTERACTION_COLUMN:
            result["fourth_down_side_split_diagnostic"] = fourth_down_side_split(features, paired)

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
        "pbp_root": str(args.pbp_root),
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
        command="pbp-trait-on-production",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Rotation-assigned opener confirmation for a PBP coaching-trait "
            "candidate (LEAD-26/27/30 Wave 4); prediction-level paired output "
            "retained; the candidate column is computed at runtime from the "
            "local PBP snapshot (plus, for the fourth-down interaction, the "
            "opener market store), never read from a precomputed parquet."
        ),
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
