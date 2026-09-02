"""Pace-mismatch dog tilt overlay, stacked on PRODUCTION -- opener-grade back-test.

Measures what adding the pace-mismatch dog tilt overlay
(``nfl_ats.pace_mismatch_dog_tilt_overlay``, registry cell
``team_style_pace_mismatch_dog_cover``) would have done ON TOP OF the picks
the pool actually plays today, not on top of a bare, un-flipped baseline.
"Production" here is the frozen, live-wired policy
(``nfl_ats.four_overlay_composition.PLAYED_UNION`` /
``scripts/overlay_subset_holdout_v2.py:PLAYED_UNION``): the joint-OR union of
``coach_fade_overlay``, ``division_revenge_tilt_overlay``,
``player_arrests_back_side_policy``, and ``spread_gap_zone_fade_overlay``,
each evaluated independently against the same raw active-model card and
complemented once. This module adds a FIFTH member, the pace-mismatch dog
tilt, and measures the marginal delta -- exactly the "single addition to the
played union" question ``scripts/overlay_subset_holdout_v2.py`` already asks
of other candidates, applied here to this one candidate alone (no search, no
selection: this is a single frozen addition, not a subset search).

Baseline archive: ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``,
the SAME frozen baseline every other overlay back-test in this repository
uses -- the active weak_stack model's 1,537 REG games 2020-2025, graded at
the Tuesday opener under the production probability rule
(``home_cover_probability_at_open >= 0.5``), baseline accuracy 53.36% on
1,503 scored games.

**Spread column used for favourite/underdog: ``spread_line``, itself seeded
from ``tue_open_home_spread`` by ``overlay_stack_backtest.build_predictions_frame``
(``scripts/overlay_stack_backtest.py:183-188``) -- the Tuesday-OPENER decision
line, not the close.** This matches (a) the pool's own primary goal (beating
the OPENING line -- AGENTS.md, "A promotion bar is not a decision bar": grade
the decision at the opener), (b) this archive's own opener grading (every
column in ``per_game.parquet`` used here is opener-graded), and (c) the exact
field every sibling overlay recorder already reads for
``decision_home_spread``. The pace-mismatch overlay's OWN flag
(:func:`nfl_ats.pace_mismatch_dog_tilt_overlay.pace_mismatch_flag_by_game`)
never reads ``spread_line`` at all -- it is a pure function of the schedule
and the prior-season pace cache -- so the opener-vs-close choice only affects
which side (favourite/underdog) a flagged game's flip lands on, via
``apply_pace_mismatch_dog_tilt_overlay``.

**MINED-SEASONS READ, not a fresh confirmation.** The registry cell itself was
measured on REG 2009-2025 (``scripts/team_style_screen.py``); this archive's
2020-2025 span is a subset of that same mined window, and three of the four
PLAYED_UNION members were themselves screened or tuned on overlapping windows
(see ``scripts/overlay_stack_backtest.py``'s own caveat, reused verbatim in
spirit here). This is CONTINUOUS EVIDENCE on an already-looked-at window, a
diagnostic, not a fresh confirmation, and it spends no rotation-registry
window.

**Binding closing-grounds taxonomy (verbatim, per AGENTS.md/CLAUDE.md).** An
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero, on BOTH blockings) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``: record it with
``nfl-ats weak-signals record``, report ``probability_positive``, never the
binary "contains zero". The registry code hard-rejects inadmissible closures;
if a record command errors, the verdict is wrong, not the validator.

Machinery reused BY IMPORT, not re-implemented, per
``scripts/overlay_subset_composition.py``'s own import convention (both files
run as ``python scripts/<name>.py``, which puts ``scripts/`` on ``sys.path``
automatically -- confirmed by that script's own bare
``from overlay_stack_backtest import (...)``):

* ``scripts/overlay_stack_backtest.py``: ``load_inputs``, ``build_predictions_frame``,
  ``build_eval_frame`` (used here for its established push-handling
  ``correct_baseline`` extraction only -- its own six-overlay
  ``OVERLAY_NAMES`` solo/loo/combined columns are a byproduct of reusing that
  logic, not this study's subject: "production" here is the played
  FOUR-member union, not all six prospective overlays), ``run_both_blockings``,
  ``run_overlays``, ``OVERLAY_NAMES``, ``DEFAULT_PER_GAME_ARTIFACT``.
* ``scripts/overlay_subset_composition.py``: ``reconstruct_arrest_flip_set``
  (the player-arrests member has no ``apply_*`` overlay module -- it is
  reconstructed from the same frozen policy machinery
  ``scripts/overlay_subset_holdout_v2.py`` already uses for the identical
  purpose).
* ``scripts/overlay_subset_holdout_v2.py``: ``PLAYED_UNION``, ``ARREST_MEMBER``,
  ``DEFAULT_FEATURES``, ``DEFAULT_INCIDENTS`` (the exact reference arm and
  reconstruction inputs the 2026-08-25 holdout study already established as
  "what is actually submitted" -- see that module's own corrected
  ``FORMER_CHAIN`` docstring note).

Usage (from the repo root, per AGENTS.md environment conventions)::

    .\\.tools\\uv.exe run --no-sync python scripts/pace_mismatch_dog_stacked_backtest.py
"""

from __future__ import annotations

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
    DEFAULT_PER_GAME_ARTIFACT,
    OVERLAY_NAMES,
    build_eval_frame,
    build_predictions_frame,
    load_inputs,
    run_both_blockings,
    run_overlays,
)
from overlay_subset_composition import reconstruct_arrest_flip_set  # noqa: E402
from overlay_subset_holdout_v2 import (  # noqa: E402
    ARREST_MEMBER,
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    PLAYED_UNION,
)

from nfl_ats.coach_fade_overlay import apply_coach_fade_overlay  # noqa: E402
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay  # noqa: E402
from nfl_ats.pace_mismatch_dog_tilt_overlay import (  # noqa: E402
    CHALLENGER_ID,
    apply_pace_mismatch_dog_tilt_overlay,
    pace_mismatch_flag_by_game,
    team_season_style_path,
)
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)
from nfl_ats.spread_gap_zone_fade_overlay import apply_spread_gap_zone_fade_overlay  # noqa: E402

DEFAULT_DATA_ROOT = REPO / "data"
DEFAULT_TEAM_SEASON_STYLE = team_season_style_path(DEFAULT_DATA_ROOT)
DEFAULT_OUTPUT_ROOT = REPO / "artifacts" / "pace_mismatch_dog_stacked"
SAMPLES = 20_000
#: A fresh, recorded seed for this study -- not reused from a sibling
#: overlay-stack study, but fixed here so a re-run reproduces exactly.
SEED = 20260901
CONFIDENCE = 0.95
SPREAD_GAP_MEMBER = "spread_gap_zone_fade_overlay"


def build_production_flip_sets(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    per_game: pd.DataFrame,
    features_path: Path,
    incidents_path: Path,
) -> dict[str, set[str]]:
    """The four PLAYED_UNION members' own flip sets, each evaluated
    independently against the raw (un-flipped) baseline card -- exactly how
    production evaluates every member (joint OR, complement once)."""

    coach = apply_coach_fade_overlay(predictions, schedules)
    division_revenge = apply_division_revenge_tilt_overlay(predictions, schedules)
    spread_gap = apply_spread_gap_zone_fade_overlay(predictions)
    arrest_ids, _arrest_scored = reconstruct_arrest_flip_set(
        per_game, features_path, incidents_path
    )
    return {
        "coach_fade_overlay": {flip.game_id for flip in coach.flips},
        "division_revenge_tilt_overlay": {flip.game_id for flip in division_revenge.flips},
        ARREST_MEMBER: arrest_ids,
        SPREAD_GAP_MEMBER: {flip.game_id for flip in spread_gap.flips},
    }


def run_backtest(
    *,
    per_game_path: Path = DEFAULT_PER_GAME_ARTIFACT,
    data_root: Path = DEFAULT_DATA_ROOT,
    team_season_style_path_: Path = DEFAULT_TEAM_SEASON_STYLE,
    features_path: Path = DEFAULT_FEATURES,
    incidents_path: Path = DEFAULT_INCIDENTS,
    samples: int = SAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    per_game, schedules, player_features, snapshot_name, _player_feature_path = load_inputs(
        per_game_path, data_root
    )
    predictions = build_predictions_frame(per_game, schedules)

    # Established push-handling correct_baseline extraction, reused rather
    # than re-implemented (see module docstring). The six-overlay solo/loo/
    # combined columns this also computes are unused here.
    six_overlay_results = run_overlays(predictions, schedules, player_features)
    six_flip_sets = {
        name: {flip.game_id for flip in six_overlay_results[name].flips} for name in OVERLAY_NAMES
    }
    baseline_eval_frame = build_eval_frame(predictions, per_game, six_flip_sets)

    team_season_style = pd.read_parquet(team_season_style_path_)
    flags = pace_mismatch_flag_by_game(schedules, team_season_style)
    mine = apply_pace_mismatch_dog_tilt_overlay(predictions, flags)
    my_flip_set = {flip.game_id for flip in mine.flips}

    production_flip_sets = build_production_flip_sets(
        predictions, schedules, per_game, features_path, incidents_path
    )
    production_flip_sets[CHALLENGER_ID] = my_flip_set

    production_ids = set().union(*(production_flip_sets[name] for name in PLAYED_UNION))
    candidate_members = (*PLAYED_UNION, CHALLENGER_ID)
    candidate_ids = set().union(*(production_flip_sets[name] for name in candidate_members))

    spread_gap_ids = production_flip_sets[SPREAD_GAP_MEMBER]
    overlap_with_spread_gap_zone_fade = sorted(my_flip_set & spread_gap_ids)

    # Games where the pace-mismatch flag fires but the PLAYED union has
    # already flipped the same game for another reason: adding this member
    # cannot move accuracy on those games (the pick is already flipped), so
    # only the NET-NEW games determine the candidate-vs-production delta.
    net_new_ids = sorted(my_flip_set - production_ids)

    frame = baseline_eval_frame[["game_id", "season", "week", "correct_baseline"]].copy()
    frame["correct_production"] = np.where(
        frame["game_id"].isin(production_ids),
        1.0 - frame["correct_baseline"],
        frame["correct_baseline"],
    )
    frame["correct_candidate"] = np.where(
        frame["game_id"].isin(candidate_ids),
        1.0 - frame["correct_baseline"],
        frame["correct_baseline"],
    )

    def stacked_metric(df: pd.DataFrame) -> dict[str, float]:
        valid = df.dropna(subset=["correct_production", "correct_candidate"])
        production_acc = float(valid["correct_production"].mean()) if len(valid) else float("nan")
        candidate_acc = float(valid["correct_candidate"].mean()) if len(valid) else float("nan")
        return {
            "production_accuracy": production_acc,
            "candidate_accuracy": candidate_acc,
            "candidate_minus_production": candidate_acc - production_acc,
        }

    blocking = run_both_blockings(frame, stacked_metric, samples=samples, seed=seed)

    n_games = len(frame)
    n_pushes = int(frame["correct_baseline"].isna().sum())
    n_scored = n_games - n_pushes
    week_block_count = int(frame[["season", "week"]].drop_duplicates().shape[0])
    season_block_count = int(frame["season"].nunique())

    week_delta = blocking["week_blocked"]["candidate_minus_production"]
    season_delta = blocking["season_blocked"]["candidate_minus_production"]

    return {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration_note": (
            "MINED-SEASONS READ, not a fresh confirmation. The registry cell "
            "(team_style_pace_mismatch_dog_cover) was measured on REG "
            "2009-2025; this archive's 2020-2025 span is a subset of that "
            "same window, and three of the four PLAYED_UNION members were "
            "themselves screened/tuned on overlapping windows. Continuous "
            "evidence on an already-looked-at window; no rotation-registry "
            "window spent."
        ),
        "source_artifact": str(per_game_path),
        "source_artifact_sha256": sha256_file(per_game_path),
        "team_season_style_source": str(team_season_style_path_),
        "team_season_style_sha256": sha256_file(team_season_style_path_),
        "schedule_snapshot": snapshot_name,
        "spread_column_used": (
            "spread_line, seeded from tue_open_home_spread by "
            "build_predictions_frame (scripts/overlay_stack_backtest.py:183-188) "
            "-- the Tuesday-OPENER decision line, matching this archive's own "
            "opener grading and the pool's opener-graded primary goal "
            "(AGENTS.md: 'grade the decision at the opener'). The pace-mismatch "
            "flag itself never reads spread_line; only the favourite/underdog "
            "side determination does."
        ),
        "grading_rule": (
            "production probability rule (home_cover_probability >= 0.5) at the "
            "Tuesday-opener decision line, matching pool.py/backtest.py."
        ),
        "played_union_members": list(PLAYED_UNION),
        "candidate_members": list(candidate_members),
        "challenger_id": CHALLENGER_ID,
        "n_games": n_games,
        "n_pushes": n_pushes,
        "n_scored_games": n_scored,
        "week_block_count": week_block_count,
        "season_block_count": season_block_count,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "n_flipped_by_pace_mismatch_total": len(my_flip_set),
        "n_flipped_by_pace_mismatch_net_new_over_production": len(net_new_ids),
        "pace_mismatch_flipped_game_ids": sorted(my_flip_set),
        "pace_mismatch_net_new_flipped_game_ids": net_new_ids,
        "n_production_flips_played_union": len(production_ids),
        "spread_gap_zone_fade_overlap": {
            "n_spread_gap_zone_fade_flips": len(spread_gap_ids),
            "n_pace_mismatch_flips": len(my_flip_set),
            "n_overlap": len(overlap_with_spread_gap_zone_fade),
            "overlap_game_ids": overlap_with_spread_gap_zone_fade,
            "note": (
                "Both overlays key off the card's own spread_line (spread_gap_zone_fade "
                "unconditionally inside its frozen [7.5, 10.0] zone; pace_mismatch_dog "
                "conditionally, only when the model holds the favourite). Measured overlap "
                "on this same 1,537-game archive, not assumed."
            ),
        },
        "production_vs_baseline": {
            "production_accuracy": stacked_metric(frame)["production_accuracy"],
            "baseline_accuracy": float(frame["correct_baseline"].dropna().mean()),
        },
        "candidate_vs_production": {
            "week_blocked": blocking["week_blocked"],
            "season_blocked": blocking["season_blocked"],
        },
        "verdict_handling": (
            "Registered regardless of sign UNLESS the whole 95% interval sits below "
            "zero on BOTH blockings (resolved wrong sign for the STACKED form -> "
            "refuted_mechanism / wrong_sign_resolved, new family "
            "team_style_pace_mismatch_stacked_on_production, registration NOT "
            "proposed). Every other case, including an interval crossing zero and "
            "including a negative point estimate whose interval crosses zero, is "
            "unresolved_below_power in that same new family, and registration of "
            "the un-stacked cell (team_style_pace_mismatch_dog_cover, already "
            "measured pregame-safe on its own) IS still proposed -- a stacked read "
            "is context, not a gate on the underlying signal."
        ),
        "week_blocked_summary": (
            f"{week_delta['estimate'] * 100:+.4f} pts, 95% "
            f"[{week_delta['lower'] * 100:+.4f}, {week_delta['upper'] * 100:+.4f}], "
            f"P+ {week_delta['probability_positive']:.4f}"
        ),
        "season_blocked_summary": (
            f"{season_delta['estimate'] * 100:+.4f} pts, 95% "
            f"[{season_delta['lower'] * 100:+.4f}, {season_delta['upper'] * 100:+.4f}], "
            f"P+ {season_delta['probability_positive']:.4f}"
        ),
    }


def main() -> int:
    result = run_backtest()

    configuration = {
        "command": "pace-mismatch-dog-stacked-backtest",
        "per_game_artifact": str(DEFAULT_PER_GAME_ARTIFACT),
        "data_root": str(DEFAULT_DATA_ROOT),
        "team_season_style_source": str(DEFAULT_TEAM_SEASON_STYLE),
        "played_union_members": list(PLAYED_UNION),
        "challenger_id": CHALLENGER_ID,
        "bootstrap_samples": result["bootstrap_samples"],
        "bootstrap_seed": result["bootstrap_seed"],
    }
    payload = {
        **configuration,
        "result": result,
        "provenance": artifact_provenance(
            configuration, DEFAULT_PER_GAME_ARTIFACT, project_root=REPO
        ),
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = DEFAULT_OUTPUT_ROOT / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="pace-mismatch-dog-stacked-backtest",
        metrics={
            "n_games": result["n_games"],
            "n_scored_games": result["n_scored_games"],
            "n_flipped_total": result["n_flipped_by_pace_mismatch_total"],
            "n_flipped_net_new": result["n_flipped_by_pace_mismatch_net_new_over_production"],
            "week_blocked_delta_accuracy_points": result["candidate_vs_production"]["week_blocked"][
                "candidate_minus_production"
            ]["estimate"]
            * 100.0,
            "week_blocked_probability_positive": result["candidate_vs_production"]["week_blocked"][
                "candidate_minus_production"
            ]["probability_positive"],
        },
        notes=(
            "Pace-mismatch dog tilt overlay (team_style_pace_mismatch_dog_cover) stacked "
            "ON TOP OF the PLAYED four-member overlay union, not a bare baseline -- see "
            "docs/pace_mismatch_dog_tilt_overlay.md. Mined-seasons read, context not a gate."
        ),
    )
    print(f"Wrote {output_dir / 'results.json'}")
    print(f"Week-blocked candidate vs production: {result['week_blocked_summary']}")
    print(f"Season-blocked candidate vs production: {result['season_blocked_summary']}")
    print(
        f"n_flipped total={result['n_flipped_by_pace_mismatch_total']} "
        f"net_new={result['n_flipped_by_pace_mismatch_net_new_over_production']} "
        f"overlap_with_spread_gap_zone_fade="
        f"{result['spread_gap_zone_fade_overlap']['n_overlap']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
