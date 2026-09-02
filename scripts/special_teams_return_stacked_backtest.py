"""Special-teams return tilt overlay: stacked-on-production back-test.

Measures ``nfl_ats.special_teams_return_tilt_overlay``'s flip set applied on
top of the SAME frozen baseline every other prospective tilt overlay
back-test uses: ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``,
the active ``weak_stack``/ridge-alpha-10 model's own 1,537 REG-season games
2020-2025, graded at the Tuesday opener under the PRODUCTION probability rule
(``home_cover_probability >= 0.5``) -- baseline 53.36% on 1,503 scored games
(``correct_at_open_probability_rule``). This is the CURRENT production
model's own card, not a synthetic or hypothetical baseline -- the same
archive ``scripts/overlay_stack_backtest.py`` and
``scripts/overlay_subset_composition.py`` already backtest every live
pick-level tilt overlay (surface-switch, interim-HC-first-game,
PBP-08-protection-mismatch) against.

Machinery is IMPORTED, not re-implemented, from ``scripts/overlay_stack_backtest.py``
(``load_inputs``, ``build_predictions_frame``, ``build_eval_frame``,
``run_both_blockings``, ``DEFAULT_PER_GAME_ARTIFACT``, ``OVERLAY_NAMES``),
following the bare-module import convention
``scripts/overlay_subset_composition.py`` already uses (both scripts live in
``scripts/``, so ``python scripts/<this file>.py`` puts the script's own
directory on ``sys.path`` and the sibling import resolves without any
``sys.path`` manipulation). ``build_eval_frame`` requires a ``flip_sets``
entry for each of the six ``OVERLAY_NAMES`` it iterates internally; none of
those six overlays is applied here (this overlay is tracked INDEPENDENTLY,
never stacked on the other prospective overlays -- see every sibling
``docs/*_overlay.md``'s "not stacked" note), so each contributes an EMPTY
flip set and the function's own ``correct_baseline`` column is the only
piece of it this script actually consumes.

The candidate's own flip set comes from calling
``nfl_ats.special_teams_return_tilt_overlay.apply_special_teams_return_tilt_overlay``
directly against the archive's own reshaped predictions frame and the
archive's own schedule snapshot -- exactly how
``scripts/overlay_stack_backtest.py::run_overlays`` calls every sibling
overlay, never a re-derivation of the flip logic.

Uncertainty: ``nfl_ats.clv.week_blocked_bootstrap`` (via ``run_both_blockings``),
20,000 samples, a fixed recorded seed (matching the seed
``scripts/overlay_stack_backtest.py`` and
``scripts/overlay_subset_composition.py`` already use for this same archive),
week-blocked primary and season-blocked secondary, paired candidate-vs-
baseline deltas in accuracy points.

**Verdict handling (read the module docstring's "Verdict handling" note in
the caller before treating this as anything but a mined-seasons read).** This
is CONTEXT, not a gate: the challenger is registered regardless of sign
unless the WHOLE interval sits on the wrong side of zero on BOTH blockings.
An interval crossing zero, or a negative point estimate whose interval
crosses zero, is NOT grounds to decline -- AGENTS.md, binding.

This rule flags roughly a quarter of team-games league-wide (the registry
cell's ``fraction_of_slate`` is 0.2335), so EXPECT a large flip count on this
archive -- that is a real, disclosed property of the rule, not a defect.

Writes ``artifacts/special_teams_return_stacked/<UTC timestamp>/results.json``.

Usage (from the repo root, per AGENTS.md environment conventions)::

    .\\.tools\\uv.exe run --no-sync python scripts/special_teams_return_stacked_backtest.py
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from overlay_stack_backtest import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PER_GAME_ARTIFACT,
    OVERLAY_NAMES,
    build_eval_frame,
    build_predictions_frame,
    load_inputs,
    run_both_blockings,
)

from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact
from nfl_ats.special_teams_return_tilt_overlay import (
    CHALLENGER_ID,
    apply_special_teams_return_tilt_overlay,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("artifacts/special_teams_return_stacked")
DEFAULT_SAMPLES = 20_000
DEFAULT_SEED = 20260819


def candidate_metric(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["correct_baseline", "correct_candidate"])
    baseline_acc = float(valid["correct_baseline"].mean()) if len(valid) else float("nan")
    candidate_acc = float(valid["correct_candidate"].mean()) if len(valid) else float("nan")
    return {
        "baseline_accuracy": baseline_acc,
        "candidate_accuracy": candidate_acc,
        "candidate_minus_baseline": candidate_acc - baseline_acc,
    }


def run_backtest(
    per_game_path: Path,
    data_root: Path,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    per_game, schedules, _player_features, snapshot_name, _player_feature_path = load_inputs(
        per_game_path, data_root
    )
    predictions = build_predictions_frame(per_game, schedules)

    tilt = apply_special_teams_return_tilt_overlay(predictions, schedules, data_root)
    my_flip_ids = {flip.game_id for flip in tilt.flips}

    # build_eval_frame iterates OVERLAY_NAMES internally to add each sibling
    # overlay's own solo/leave-one-out columns; none of the six is applied in
    # this script (see module docstring), so every sibling flip set is empty
    # and only its `correct_baseline` column is actually consumed below.
    empty_sibling_flip_sets: dict[str, set[str]] = {name: set() for name in OVERLAY_NAMES}
    eval_frame = build_eval_frame(predictions, per_game, empty_sibling_flip_sets)

    eval_frame["special_teams_return_flip"] = eval_frame["game_id"].isin(my_flip_ids)
    eval_frame["correct_candidate"] = np.where(
        eval_frame["special_teams_return_flip"],
        1.0 - eval_frame["correct_baseline"],
        eval_frame["correct_baseline"],
    )

    n_games = len(eval_frame)
    n_pushes = int(eval_frame["correct_baseline"].isna().sum())
    n_scored = n_games - n_pushes
    week_block_count = int(eval_frame[["season", "week"]].drop_duplicates().shape[0])
    season_block_count = int(eval_frame["season"].nunique())

    bootstrap = run_both_blockings(eval_frame, candidate_metric, samples=samples, seed=seed)

    valid = eval_frame.dropna(subset=["correct_baseline"])
    baseline_accuracy = float(valid["correct_baseline"].mean())
    candidate_accuracy = float(valid["correct_candidate"].mean())

    return {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "challenger_id": CHALLENGER_ID,
        "mined_seasons_read_disclosure": (
            "This is a MINED-SEASONS read on an already-looked-at archive (2020-2025 opener "
            "grades), CONTEXT for the registration decision, not a gate. Per AGENTS.md, the "
            "challenger is registered regardless of sign unless the WHOLE interval sits below "
            "zero on BOTH blockings."
        ),
        "source_artifact": str(per_game_path),
        "source_artifact_sha256": sha256_file(per_game_path),
        "schedule_snapshot": snapshot_name,
        "grading_rule": (
            "production probability rule (home_cover_probability >= 0.5) at the "
            "Tuesday-opener decision line, matching pool.py/backtest.py."
        ),
        "baseline_description": (
            "The active model's own opener-graded picks (CURRENT production, not a bare/"
            "synthetic baseline) -- the special-teams return flip set is applied ON TOP of "
            "this exact card, never on a hypothetical stripped-down baseline."
        ),
        "seasons": [int(eval_frame["season"].min()), int(eval_frame["season"].max())],
        "n_games": n_games,
        "n_pushes": n_pushes,
        "n_scored_games": n_scored,
        "week_block_count": week_block_count,
        "season_block_count": season_block_count,
        "n_flipped": len(my_flip_ids),
        "n_both_flagged_untouched": len(tilt.both_flagged_games),
        "flipped_game_ids": sorted(my_flip_ids),
        "both_flagged_game_ids": sorted(tilt.both_flagged_games),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "candidate_minus_baseline_accuracy": candidate_accuracy - baseline_accuracy,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "bootstrap": bootstrap,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path, default=DEFAULT_PER_GAME_ARTIFACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    result = run_backtest(
        args.per_game_artifact, args.data_root, samples=args.samples, seed=args.seed
    )

    configuration = {
        "command": "special-teams-return-stacked-backtest",
        "challenger_id": CHALLENGER_ID,
        "per_game_artifact": str(args.per_game_artifact),
        "data_root": str(args.data_root),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
    }
    payload = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(
            configuration, args.per_game_artifact, project_root=REPO_ROOT
        ),
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = REPO_ROOT / args.output_root / timestamp
    week = result["bootstrap"]["week_blocked"]["candidate_minus_baseline"]
    season = result["bootstrap"]["season_blocked"]["candidate_minus_baseline"]
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="special-teams-return-stacked-backtest",
        metrics={
            "n_flipped": result["n_flipped"],
            "candidate_minus_baseline_accuracy": result["candidate_minus_baseline_accuracy"],
            "week_blocked_probability_positive": week["probability_positive"],
            "season_blocked_probability_positive": season["probability_positive"],
        },
        notes=(
            "special_teams_return_top_quartile tilt overlay (registry cell, reliability "
            "0.109-0.158) stacked on the active model's own opener-graded production picks "
            "-- a MINED-SEASONS read, context for the special_teams_return_stacked_on_production "
            "registration decision, not a gate. See "
            "docs/special_teams_return_tilt_overlay.md."
        ),
    )

    output_path = output_dir / "results.json"
    print(f"Wrote {output_path}")
    print(
        f"n_flipped={result['n_flipped']} of {result['n_games']} games "
        f"(n_both_flagged_untouched={result['n_both_flagged_untouched']})"
    )
    print(
        f"baseline_accuracy={result['baseline_accuracy'] * 100:.4f}% "
        f"candidate_accuracy={result['candidate_accuracy'] * 100:.4f}%"
    )
    print(
        "week-blocked candidate-vs-baseline: "
        f"{week['estimate'] * 100:+.4f} pts [{week['lower'] * 100:+.4f}, "
        f"{week['upper'] * 100:+.4f}] P+ {week['probability_positive']:.4f}"
    )
    print(
        "season-blocked candidate-vs-baseline: "
        f"{season['estimate'] * 100:+.4f} pts [{season['lower'] * 100:+.4f}, "
        f"{season['upper'] * 100:+.4f}] P+ {season['probability_positive']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
