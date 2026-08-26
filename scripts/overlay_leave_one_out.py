"""Leave-one-out test of the PLAYED four-member overlay policy.

Predeclaration: see ``docs/overlay_leave_one_out.md``, written and committed
before this script produced any output. Four variants only, each dropping
exactly one member from the played four-member union:

    drop coach_fade
    drop division_revenge_tilt
    drop player_arrests_back_side_policy
    drop spread_gap_zone_fade

Same predeclared split as ``scripts/overlay_subset_holdout_v2.py``: choose
(i.e. look) on 2020-2022, primary/decision-grade read on 2023-2025 unchanged.
Unlike that study, there is no search here -- four variants are fixed by
design, not selected from a larger pool -- so "choose" names only which season
range is the primary read, not a subset search.

This module adapts ``overlay_subset_holdout_v2.py`` BY IMPORT: the flip-set
construction, the bootstrap machinery, and the season split below are the
literal functions/constants from that module, not a re-implementation.

**Binding closing-grounds taxonomy (verbatim).** An interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. At this
evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a
real small signal. Only two grounds ever close a line of work: (1) refuted
mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of zero)
or zero split-half reliability; (2) bounded by a positive control proven able
to detect an effect that size. Everything else is `unresolved_below_power`:
record it with `nfl-ats weak-signals record`, report `probability_positive`,
never the binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the validator.

**Decision rule.** Forced-pick pool: 285 cards get submitted either way, so the
decision is expected value, not a threshold. `probability_positive` above 0.5
on the PRIMARY (holdout) read favours dropping that member; below 0.5 favours
keeping it. Never used as a gate on whether to report a number.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/overlay_leave_one_out.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from overlay_stack_backtest import (  # noqa: E402
    OVERLAY_NAMES,
    build_predictions_frame,
    load_inputs,
    run_overlays,
)
from overlay_subset_composition import reconstruct_arrest_flip_set  # noqa: E402
from overlay_subset_holdout_v2 import (  # noqa: E402
    ARREST_MEMBER,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    DEFAULT_PER_GAME,
    EVALUATION_SEASONS,
    PLAYED_UNION,
    SAMPLES,
    SEED,
    SELECTION_SEASONS,
    bootstrap,
    evaluate,
    extra_flip_sets,
    union_delta,
)

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO / "artifacts/overlay_leave_one_out"

#: The four variants, predeclared in docs/overlay_leave_one_out.md. Each drops
#: exactly one member from PLAYED_UNION.
VARIANTS: dict[str, str] = {
    "drop_coach_fade": "coach_fade_overlay",
    "drop_division_revenge_tilt": "division_revenge_tilt_overlay",
    "drop_player_arrests_back_side_policy": ARREST_MEMBER,
    "drop_spread_gap_zone_fade": "spread_gap_zone_fade_overlay",
}


def build_member_flip_sets() -> tuple[dict[str, set[str]], pd.DataFrame, str]:
    per_game, schedules, player_features, snapshot_name, _ = load_inputs(
        DEFAULT_PER_GAME, DEFAULT_DATA_ROOT
    )
    predictions = build_predictions_frame(per_game, schedules)
    results = run_overlays(predictions, schedules, player_features)
    member_flip_sets = {name: {f.game_id for f in results[name].flips} for name in OVERLAY_NAMES}
    member_flip_sets.update(extra_flip_sets(predictions, schedules))
    arrest_ids, _ = reconstruct_arrest_flip_set(per_game, DEFAULT_FEATURES, DEFAULT_INCIDENTS)
    member_flip_sets[ARREST_MEMBER] = arrest_ids

    missing_played = [m for m in PLAYED_UNION if m not in member_flip_sets]
    if missing_played:
        raise ValueError(f"PLAYED_UNION members missing from flip sets: {missing_played}")

    frame = predictions[["game_id", "season", "week"]].merge(
        per_game[["game_id", "correct_at_open_probability_rule"]], on="game_id", how="left"
    )
    frame["correct"] = pd.to_numeric(frame["correct_at_open_probability_rule"], errors="coerce")
    frame = frame.loc[frame["correct"].notna()].reset_index(drop=True)
    return member_flip_sets, frame, snapshot_name


def paired_delta(
    variant_correct: np.ndarray,
    incumbent_correct: np.ndarray,
) -> np.ndarray:
    """Per-game (variant - incumbent) accuracy delta, in [-1, 0, +1]."""

    return variant_correct - incumbent_correct


def evaluate_variant(
    variant_subset: tuple[str, ...],
    incumbent_subset: tuple[str, ...],
    correct: np.ndarray,
    game_ids: pd.Series,
    member_flip_sets: dict[str, set[str]],
    blocks: pd.DataFrame,
) -> dict[str, Any]:
    variant_deltas = union_delta(correct, game_ids, member_flip_sets, variant_subset)
    incumbent_deltas = union_delta(correct, game_ids, member_flip_sets, incumbent_subset)
    variant_correct = correct + variant_deltas
    incumbent_correct = correct + incumbent_deltas
    paired = paired_delta(variant_correct, incumbent_correct)

    week_stats = bootstrap(paired, blocks, "week")
    season_stats = bootstrap(paired, blocks, "season")

    return {
        "variant_members": list(variant_subset),
        "incumbent_members": list(incumbent_subset),
        "n_games": len(correct),
        "games_changed": int(np.count_nonzero(paired != 0.0)),
        "incumbent_accuracy": float(incumbent_correct.mean()),
        "variant_accuracy": float(variant_correct.mean()),
        "paired_delta_accuracy_points": float(paired.mean() * 100.0),
        "week_blocked": week_stats,
        "season_blocked": season_stats,
    }


def main(argv: list[str] | None = None) -> int:
    member_flip_sets, frame, snapshot_name = build_member_flip_sets()

    print("PLAYED_UNION members and flip counts:", file=sys.stderr)
    for name in PLAYED_UNION:
        print(f"  {name:<48} flips={len(member_flip_sets[name])}", file=sys.stderr)

    def half(seasons: tuple[int, ...]) -> pd.DataFrame:
        return frame.loc[frame["season"].isin(seasons)].reset_index(drop=True)

    report: dict[str, Any] = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "schedule_snapshot": snapshot_name,
        "predeclaration": "docs/overlay_leave_one_out.md",
        "played_union": list(PLAYED_UNION),
        "selection_seasons": list(SELECTION_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "bootstrap_samples": SAMPLES,
        "bootstrap_seed": SEED,
        "variants": {},
    }

    # Sanity anchor: reproduce the played union's full-archive accuracy against
    # docs/overlay_subset_holdout_v2.md's reported 55.4225% (vs raw 53.3599%)
    # on 1,503 opener games.
    all_correct = frame["correct"].to_numpy(dtype=float)
    played_full_archive = evaluate(
        PLAYED_UNION,
        all_correct,
        frame["game_id"],
        member_flip_sets,
        frame[["season", "week"]],
        with_intervals=True,
    )
    report["played_union_full_archive_sanity_anchor"] = played_full_archive
    report["sanity_anchor_reference"] = {
        "source": "docs/overlay_subset_holdout_v2.md",
        "reported_played_accuracy": 0.554225,
        "reported_raw_accuracy": 0.533599,
        "reported_n_games": 1503,
    }

    sel_frame, ev_frame = half(SELECTION_SEASONS), half(EVALUATION_SEASONS)
    sel_correct = sel_frame["correct"].to_numpy(dtype=float)
    ev_correct = ev_frame["correct"].to_numpy(dtype=float)

    selection_deltas: dict[str, float] = {}
    holdout_deltas: dict[str, float] = {}

    for variant_name, dropped_member in VARIANTS.items():
        variant_subset = tuple(sorted(m for m in PLAYED_UNION if m != dropped_member))
        selection = evaluate_variant(
            variant_subset,
            PLAYED_UNION,
            sel_correct,
            sel_frame["game_id"],
            member_flip_sets,
            sel_frame[["season", "week"]],
        )
        holdout = evaluate_variant(
            variant_subset,
            PLAYED_UNION,
            ev_correct,
            ev_frame["game_id"],
            member_flip_sets,
            ev_frame[["season", "week"]],
        )
        full_archive = evaluate_variant(
            variant_subset,
            PLAYED_UNION,
            all_correct,
            frame["game_id"],
            member_flip_sets,
            frame[["season", "week"]],
        )
        report["variants"][variant_name] = {
            "dropped_member": dropped_member,
            "selection_half_2020_2022": selection,
            "holdout_half_2023_2025_PRIMARY": holdout,
            "full_archive_bonus": full_archive,
        }
        selection_deltas[variant_name] = selection["paired_delta_accuracy_points"]
        holdout_deltas[variant_name] = holdout["paired_delta_accuracy_points"]

    # Shrinkage/rank-stability diagnostics across the four variants, exactly as
    # requested, explicitly flagged as under-powered at n=4 (see predeclaration).
    sel_vals = np.array([selection_deltas[name] for name in VARIANTS])
    hold_vals = np.array([holdout_deltas[name] for name in VARIANTS])
    slope = float(np.polyfit(sel_vals, hold_vals, 1)[0]) if len(set(sel_vals)) > 1 else float("nan")
    rho = float(pd.Series(sel_vals).corr(pd.Series(hold_vals), method="spearman"))
    report["shrinkage_rank_stability_n4_UNDERPOWERED"] = {
        "note": (
            "Computed across only the 4 predeclared variants for comparability "
            "with docs/overlay_subset_holdout_v2.md's reporting convention. "
            "n=4 estimates a slope/rank-correlation very noisily -- NOT "
            "decision-relevant on its own, disclosed rather than hidden."
        ),
        "selection_half_deltas": selection_deltas,
        "holdout_half_deltas": holdout_deltas,
        "ols_slope_holdout_on_selection": slope,
        "spearman_rho": rho,
    }

    out_dir = DEFAULT_OUTPUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "command": "overlay_leave_one_out",
        "created_at_utc": report["computed_at_utc"],
        "provenance": artifact_provenance(
            {
                "played_union": list(PLAYED_UNION),
                "variants": dict(VARIANTS),
                "selection_seasons": list(SELECTION_SEASONS),
                "evaluation_seasons": list(EVALUATION_SEASONS),
            },
            DEFAULT_FEATURES,
            project_root=REPO,
        ),
        "report": report,
    }
    write_experiment_artifact(
        out_dir,
        "result.json",
        metadata,
        command="overlay_leave_one_out",
        metrics={
            f"{name}_holdout_paired_delta_accuracy_points": holdout_deltas[name]
            for name in VARIANTS
        },
        notes=(
            "Leave-one-out test of the played four-member overlay policy; "
            "see docs/overlay_leave_one_out.md"
        ),
        source="scripts/overlay_leave_one_out.py",
        project_root=REPO,
    )

    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
