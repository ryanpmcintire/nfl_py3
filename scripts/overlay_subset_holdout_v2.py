"""Which subset of pick-flipping overlays should actually be PLAYED?

Predeclaration (written before any output of this script existed)
----------------------------------------------------------------
The 2026-08-21 study (`docs/overlay_subset_composition.md`,
`docs/overlay_selection_holdout.md`) searched all 127 non-empty subsets of
SEVEN members and ran a split-half holdout. Since then five more pick-flipping
overlays were registered. This re-runs the same design over **twelve** members
(4,095 subsets) so the newer overlays can compete, and keeps the same honest
decider.

**The deciding test is the FORWARD split: choose the subset using seasons
2020-2022 ONLY, then apply it unchanged to 2023-2025 and report that.** That is
the only direction that is actually deployable -- you can only ever select on
the past. The reverse split (choose on 2023-2025, apply to 2020-2022) is run
too and reported, but it is a stability check, NOT a second chance to pick a
winner. Reporting both and then choosing whichever looks better would be a
second selection on top of the first, which is exactly the error the holdout
exists to avoid.

Secondary reads, all predeclared here:

* the in-sample full-slate maximum (an upper bound, reported so the gap to the
  holdout number is visible rather than hidden);
* the shrinkage factor (OLS slope of holdout delta on selection delta across
  all subsets) and Spearman rank correlation between halves -- how much of a
  selection-half advantage survives, on average;
* the PLAYED four-member union scored the same way, so the comparison is
  against what is actually submitted, not against the raw model, and the
  FORMER coach->arrests chain as a second reference arm;
* the naive all-twelve stack, as the control for "just play everything".

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
decision is expected value, not a threshold. The subset with the better holdout
expectation is the one to play. `probability_positive` is reported for every
arm and never used as a gate.

**Attribution caveat, mandatory.** Every overlay here was registered on windows
that overlap this archive, so even the holdout half is not virgin data for the
COMPONENTS -- it is virgin only for the SUBSET CHOICE. The holdout number is an
honest estimate of the selection procedure's value, not a fresh confirmation of
any component. No rotation-registry window is spent.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/overlay_subset_holdout_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from itertools import combinations
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

from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.forecast_cold_visitor_tilt_overlay import (  # noqa: E402
    apply_forecast_cold_visitor_tilt_overlay,
)
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (  # noqa: E402
    apply_precip_high_total_tilt_overlay,
)
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (  # noqa: E402
    apply_warm_team_cold_late_tilt_overlay,
)
from nfl_ats.interim_hc_first_game_tilt_overlay import (  # noqa: E402
    apply_interim_hc_first_game_tilt_overlay,
)
from nfl_ats.pbp08_matchup_flags import build_flag_table  # noqa: E402
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (  # noqa: E402
    apply_pbp08_protection_mismatch_tilt,
)

DEFAULT_PER_GAME = REPO / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
DEFAULT_DATA_ROOT = REPO / "data"
DEFAULT_FEATURES = REPO / "data/processed/game_features_pbp.parquet"
DEFAULT_INCIDENTS = (
    REPO / "data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet"
)
DEFAULT_FORECASTS = REPO / "data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet"
DEFAULT_PBP = REPO / "data/pbp/raw/20260817T184927Z"
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/overlay_subset_holdout_v2"

ARREST_MEMBER = "player_arrests_back_side_policy"
SELECTION_SEASONS = (2020, 2021, 2022)
EVALUATION_SEASONS = (2023, 2024, 2025)
SAMPLES = 20_000
SEED = 20260825

#: The FORMER production chain, kept as a reference arm only.
#:
#: CORRECTED 2026-08-25: an earlier version of this file called this "what is
#: submitted today". It is not. Production resolves
#: ``nfl_ats.clv._FOUR_OVERLAY_POLICY_ID`` --
#: ``overlay_union_coach_division_revenge_player_arrests_spread_gap_v1`` --
#: whenever ``require_fresh_arrest_overlay=True``, which
#: ``cli._cmd_publish_predictions`` always passes. Verified by running
#: ``record_paper_decisions`` itself against the real active model, not by
#: reading a study's label for its baseline. See
#: ``docs/overlay_subset_holdout_v2.md``.
FORMER_CHAIN = (ARREST_MEMBER, "coach_fade_overlay")

#: What is actually submitted: the four-member OR union.
PLAYED_UNION = tuple(
    sorted(
        (
            "coach_fade_overlay",
            "division_revenge_tilt_overlay",
            ARREST_MEMBER,
            "spread_gap_zone_fade_overlay",
        )
    )
)


def extra_flip_sets(predictions: pd.DataFrame, schedules: pd.DataFrame) -> dict[str, set[str]]:
    """Flip sets for the five overlays the 2026-08-21 study predates.

    Each is called exactly as its production recorder calls it. A member whose
    inputs are unavailable raises rather than silently contributing an empty
    flip set -- an empty set would look like a well-behaved overlay that never
    fires, and would quietly bias every subset containing it toward the
    baseline.
    """

    forecasts = pd.read_parquet(DEFAULT_FORECASTS)
    flags = build_flag_table(
        schedules.loc[schedules["game_type"].astype(str).eq("REG")], DEFAULT_PBP
    )

    # The precip overlay reads ``total_line``, which production's own card
    # carries (it arrives from the schedules snapshot through the feature
    # table). The opener archive does not, so it is merged from the same
    # source production uses rather than invented.
    #
    # DISCLOSED: the schedules snapshot's ``total_line`` is the CLOSING total,
    # not a Tuesday-opener total. The spread leg of this study is opener-graded
    # and the total leg is not. That mismatch is inherited from the registered
    # signal itself, which reads the same field -- it is not introduced here,
    # but any subset containing this member carries it.
    augmented = predictions.merge(
        schedules[["game_id", "total_line"]].drop_duplicates("game_id"),
        on="game_id",
        how="left",
    )

    results = {
        "interim_hc_first_game_tilt_overlay": apply_interim_hc_first_game_tilt_overlay(
            predictions, REPO
        ),
        "forecast_cold_visitor_tilt": apply_forecast_cold_visitor_tilt_overlay(
            predictions, schedules, forecasts
        ),
        "forecast_weather_kn_warm_team_cold_late_tilt": apply_warm_team_cold_late_tilt_overlay(
            predictions, schedules, forecasts
        ),
        "forecast_weather_kn_precip_high_total_tilt": apply_precip_high_total_tilt_overlay(
            augmented, schedules, forecasts
        ),
        "pbp08_protection_mismatch_tilt_overlay": apply_pbp08_protection_mismatch_tilt(
            predictions, flags
        ),
    }
    return {name: {flip.game_id for flip in result.flips} for name, result in results.items()}


def union_delta(
    correct_baseline: np.ndarray,
    game_ids: pd.Series,
    member_flip_sets: dict[str, set[str]],
    subset: tuple[str, ...],
) -> np.ndarray:
    """Per-game accuracy delta of flipping every game ANY member fires on.

    The OR convention, identical to the 2026-08-21 study: a game flips once no
    matter how many members fire on it, so two overlays agreeing never
    double-flip back to the original pick.
    """

    fired = set().union(*(member_flip_sets[name] for name in subset))
    flipped = game_ids.isin(fired).to_numpy()
    return np.where(flipped, (1.0 - correct_baseline) - correct_baseline, 0.0)


def bootstrap(deltas: np.ndarray, blocks: pd.DataFrame, block: str) -> dict[str, float]:
    frame = blocks.copy()
    frame["delta"] = deltas
    stats = week_blocked_bootstrap(
        frame,
        lambda f: {"delta": float(f["delta"].mean() * 100.0)},
        samples=SAMPLES,
        seed=SEED,
        block=block,
    )
    row = stats.iloc[0]
    return {
        "estimate_accuracy_points": float(row["estimate"]),
        "lower_accuracy_points": float(row["lower"]),
        "upper_accuracy_points": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
        "block": block,
    }


def evaluate(
    subset: tuple[str, ...],
    correct: np.ndarray,
    game_ids: pd.Series,
    member_flip_sets: dict[str, set[str]],
    blocks: pd.DataFrame,
    *,
    with_intervals: bool = False,
) -> dict[str, Any]:
    deltas = union_delta(correct, game_ids, member_flip_sets, subset)
    payload: dict[str, Any] = {
        "members": list(subset),
        "n_games": len(correct),
        "flips": int(np.count_nonzero(deltas != 0.0)),
        "baseline_accuracy": float(correct.mean()),
        "candidate_accuracy": float((correct + deltas).mean()),
        "delta_accuracy_points": float(deltas.mean() * 100.0),
    }
    if with_intervals:
        payload["week_blocked"] = bootstrap(deltas, blocks, "week")
        payload["season_blocked"] = bootstrap(deltas, blocks, "season")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    per_game, schedules, player_features, snapshot_name, _ = load_inputs(
        DEFAULT_PER_GAME, DEFAULT_DATA_ROOT
    )
    predictions = build_predictions_frame(per_game, schedules)
    results = run_overlays(predictions, schedules, player_features)
    member_flip_sets = {name: {f.game_id for f in results[name].flips} for name in OVERLAY_NAMES}
    member_flip_sets.update(extra_flip_sets(predictions, schedules))
    arrest_ids, _ = reconstruct_arrest_flip_set(per_game, DEFAULT_FEATURES, DEFAULT_INCIDENTS)
    member_flip_sets[ARREST_MEMBER] = arrest_ids

    members = tuple(sorted(member_flip_sets))
    print(f"members ({len(members)}):", file=sys.stderr)
    for name in members:
        print(f"  {name:<48} flips={len(member_flip_sets[name])}", file=sys.stderr)

    frame = predictions[["game_id", "season", "week"]].merge(
        per_game[["game_id", "correct_at_open_probability_rule"]], on="game_id", how="left"
    )
    frame["correct"] = pd.to_numeric(frame["correct_at_open_probability_rule"], errors="coerce")
    frame = frame.loc[frame["correct"].notna()].reset_index(drop=True)

    subsets = [
        tuple(sorted(combo))
        for size in range(1, len(members) + 1)
        for combo in combinations(members, size)
    ]
    print(f"searching {len(subsets)} subsets on {len(frame)} scored games", file=sys.stderr)

    def half(seasons: tuple[int, ...]) -> pd.DataFrame:
        return frame.loc[frame["season"].isin(seasons)].reset_index(drop=True)

    report: dict[str, Any] = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "schedule_snapshot": snapshot_name,
        "n_members": len(members),
        "members": list(members),
        "member_flip_counts": {k: len(v) for k, v in member_flip_sets.items()},
        "n_subsets": len(subsets),
        "n_scored_games": len(frame),
        "selection_seasons": list(SELECTION_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "bootstrap_samples": SAMPLES,
        "bootstrap_seed": SEED,
        "former_chain_reference": list(FORMER_CHAIN),
        "played_union": list(PLAYED_UNION),
    }

    def run_split(sel: tuple[int, ...], ev: tuple[int, ...], label: str) -> dict[str, Any]:
        sel_frame, ev_frame = half(sel), half(ev)
        sel_correct = sel_frame["correct"].to_numpy(dtype=float)
        ev_correct = ev_frame["correct"].to_numpy(dtype=float)
        sel_blocks = sel_frame[["season", "week"]]
        ev_blocks = ev_frame[["season", "week"]]

        sel_deltas = np.array(
            [
                union_delta(sel_correct, sel_frame["game_id"], member_flip_sets, s).mean()
                for s in subsets
            ]
        )
        ev_deltas = np.array(
            [
                union_delta(ev_correct, ev_frame["game_id"], member_flip_sets, s).mean()
                for s in subsets
            ]
        )
        best = subsets[int(np.argmax(sel_deltas))]

        slope = float(np.polyfit(sel_deltas, ev_deltas, 1)[0])
        rho = float(pd.Series(sel_deltas).corr(pd.Series(ev_deltas), method="spearman"))

        return {
            "label": label,
            "selection_seasons": list(sel),
            "evaluation_seasons": list(ev),
            "frozen_subset": best,
            "selection_half": evaluate(
                best, sel_correct, sel_frame["game_id"], member_flip_sets, sel_blocks
            ),
            "holdout": evaluate(
                best,
                ev_correct,
                ev_frame["game_id"],
                member_flip_sets,
                ev_blocks,
                with_intervals=True,
            ),
            "former_chain_on_holdout": evaluate(
                tuple(sorted(FORMER_CHAIN)),
                ev_correct,
                ev_frame["game_id"],
                member_flip_sets,
                ev_blocks,
                with_intervals=True,
            ),
            "all_members_on_holdout": evaluate(
                members, ev_correct, ev_frame["game_id"], member_flip_sets, ev_blocks
            ),
            "shrinkage": {
                "ols_slope_holdout_on_selection": slope,
                "spearman_rho": rho,
                "note": (
                    "slope < 1 is the fraction of a selection-half advantage expected to "
                    "survive out-of-sample, averaged over all subsets"
                ),
            },
        }

    report["forward_split_DECIDER"] = run_split(SELECTION_SEASONS, EVALUATION_SEASONS, "forward")
    report["reverse_split_stability_check"] = run_split(
        EVALUATION_SEASONS, SELECTION_SEASONS, "reverse"
    )

    # In-sample maximum over the whole archive: an upper bound, reported so the
    # gap to the holdout number is visible rather than hidden.
    all_correct = frame["correct"].to_numpy(dtype=float)
    full_deltas = np.array(
        [union_delta(all_correct, frame["game_id"], member_flip_sets, s).mean() for s in subsets]
    )
    in_sample_best = subsets[int(np.argmax(full_deltas))]
    report["in_sample_maximum_UPPER_BOUND"] = evaluate(
        in_sample_best,
        all_correct,
        frame["game_id"],
        member_flip_sets,
        frame[["season", "week"]],
        with_intervals=True,
    )
    report["played_union_full_archive"] = evaluate(
        PLAYED_UNION,
        all_correct,
        frame["game_id"],
        member_flip_sets,
        frame[["season", "week"]],
        with_intervals=True,
    )

    # The decision-relevant question, and a far smaller selection space than
    # 4,095: does adding ONE more member to the PLAYED union help? Ranked on
    # the selection half only, then the frozen choice is scored on the holdout.
    # Every candidate's holdout marginal is reported for completeness and is
    # explicitly NOT used to choose -- picking the best holdout number would be
    # selecting on the holdout, the exact error this design prevents.
    def marginal(sub: pd.DataFrame, name: str) -> np.ndarray:
        correct = sub["correct"].to_numpy(dtype=float)
        base = union_delta(correct, sub["game_id"], member_flip_sets, PLAYED_UNION)
        with_member = union_delta(
            correct, sub["game_id"], member_flip_sets, tuple(sorted((*PLAYED_UNION, name)))
        )
        return with_member - base

    candidates = [name for name in members if name not in PLAYED_UNION]
    sel_frame, ev_frame = half(SELECTION_SEASONS), half(EVALUATION_SEASONS)
    ranked = sorted(
        ((float(marginal(sel_frame, name).mean() * 100.0), name) for name in candidates),
        reverse=True,
    )
    frozen = ranked[0][1]
    frozen_marginal = marginal(ev_frame, frozen)
    report["single_addition_to_played_union"] = {
        "candidates": len(candidates),
        "selection_half_ranking": [
            {"member": name, "marginal_accuracy_points": est} for est, name in ranked
        ],
        "frozen_choice": frozen,
        "holdout": {
            "member": frozen,
            "games_changed": int(np.count_nonzero(frozen_marginal != 0.0)),
            "marginal_accuracy_points": float(frozen_marginal.mean() * 100.0),
            "week_blocked": bootstrap(frozen_marginal, ev_frame[["season", "week"]], "week"),
            "season_blocked": bootstrap(frozen_marginal, ev_frame[["season", "week"]], "season"),
        },
        "all_holdout_marginals_NOT_USED_TO_CHOOSE": {
            name: bootstrap(marginal(ev_frame, name), ev_frame[["season", "week"]], "week")
            for name in candidates
        },
    }

    out_dir = args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    fwd = report["forward_split_DECIDER"]
    print(json.dumps({k: v for k, v in fwd.items() if k != "label"}, indent=2, default=str))
    print(f"\nWrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
