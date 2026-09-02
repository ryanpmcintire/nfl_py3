"""Turnover-luck rebound tilt, stacked ON TOP OF the tracked overlay chain.

Answers the "stacked on production" question the challenger registration
cannot answer on its own: does flipping onto a bottom-quartile
prior-season-turnover team move accuracy when it is layered on top of what
is ALREADY being tracked, not just against a bare, unflipped baseline?

**What "production picks" means here, stated plainly.** ``publishing.py``'s
actual composed card (``nfl_ats.four_overlay_composition``) is not one of the
modules this script was told to reuse, so "production" in this script is the
SAME proxy ``scripts/overlay_stack_backtest.py`` itself already builds and
calls the "combined stack": the active model's own bare opener-graded picks
(``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``), with
every one of the six already-registered ``ACTIVE_PROSPECTIVE`` pick-flipping
overlays (``coach_fade_overlay``, ``injury_value_lost_tilt_overlay``,
``division_revenge_tilt_overlay``, ``backup_qb_fade_overlay``,
``surface_switch_tilt_overlay``, ``spread_gap_zone_fade_overlay``) OR-combined
on top, exactly as that script's own docstring frames it -- a diagnostic, not
a literal reproduction of the live composed card. This is the closest
already-built "what's actually being tracked" proxy available without
re-deriving ``four_overlay_composition.py`` from scratch, and the
simplification is stated here rather than silently assumed.

This backtest then adds the turnover-luck rebound tilt
(``nfl_ats.turnover_luck_rebound_tilt_overlay.apply_turnover_luck_rebound_tilt_overlay``)
as a SEVENTH member, using the exact same OR-combination rule
``scripts/overlay_stack_backtest.py`` verifies empirically for its own six:
every overlay's flip, wherever it fires, sets ``home_cover_probability`` to
the complement of the SAME unflipped baseline probability, so two overlays
firing on the same game agree by construction and the union can be applied
with a single complement.

Baseline: ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``,
the tracked, real (non-scratch) run of ``nfl-ats opener-evaluation`` for the
active ``weak_stack``/ridge-alpha-10 model, 1,537 REG-season games 2020-2025,
graded with the PRODUCTION probability rule (``correct_at_open_probability_rule``),
baseline accuracy 53.36% on 1,503 scored games -- the SAME frozen baseline
every other overlay back-test in this repository uses.

**This is a MINED-SEASONS read, not a fresh confirmation, and it is CONTEXT,
not a gate** (per this session's binding taxonomy, restated verbatim below).
The verdict-recording rule this script's caller follows:

* whole 95% interval BELOW zero on BOTH the week-blocked and season-blocked
  bootstrap => resolved wrong sign FOR THE STACKED FORM => record
  ``refuted_mechanism`` / ``wrong_sign_resolved`` in a NEW family
  ``close_game_luck_turnover_stacked_on_production``, and do NOT propose
  registering this challenger;
* every other outcome (an interval that crosses zero, INCLUDING a negative
  point estimate whose interval crosses zero) => record
  ``unresolved_below_power`` in that same NEW family, and DO propose
  registering the challenger. An interval containing zero is never grounds
  to decline.

Binding closing-ground taxonomy (verbatim, per AGENTS.md / CLAUDE.md, for any
process that scores or adjudicates this artifact): an interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. Only
two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED
wrong sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is ``unresolved_below_power``: record it,
report ``probability_positive``, never the binary "contains zero".

Usage (from the repo root, per AGENTS.md environment conventions)::

    .\\.tools\\uv.exe run --no-sync python scripts/turnover_luck_rebound_stacked_backtest.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from overlay_stack_backtest import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PER_GAME_ARTIFACT,
    OVERLAY_NAMES,
    build_eval_frame,
    build_predictions_frame,
    load_inputs,
    run_both_blockings,
    run_overlays,
    verify_no_direction_conflicts,
)

from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.turnover_luck_rebound_tilt_overlay import (  # noqa: E402
    CHALLENGER_ID,
    TURNOVER_UNDER_Q25_THRESHOLD,
    apply_turnover_luck_rebound_tilt_overlay,
)

DEFAULT_SAMPLES = 20_000
DEFAULT_SEED = 20260821
DEFAULT_CONFIDENCE = 0.95
DEFAULT_OUTPUT_ROOT = Path("artifacts/turnover_luck_rebound_stacked")

NEW_ROTATION_FAMILY = "close_game_luck_turnover_stacked_on_production"


def marginal_metric(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["correct_production", "correct_stacked_with_mine"])
    production_acc = float(valid["correct_production"].mean()) if len(valid) else float("nan")
    stacked_acc = float(valid["correct_stacked_with_mine"].mean()) if len(valid) else float("nan")
    return {
        "production_accuracy": production_acc,
        "stacked_with_mine_accuracy": stacked_acc,
        "delta": stacked_acc - production_acc,
    }


def baseline_metric(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["correct_baseline", "correct_stacked_with_mine"])
    baseline_acc = float(valid["correct_baseline"].mean()) if len(valid) else float("nan")
    stacked_acc = float(valid["correct_stacked_with_mine"].mean()) if len(valid) else float("nan")
    return {
        "baseline_accuracy": baseline_acc,
        "stacked_with_mine_accuracy": stacked_acc,
        "delta": stacked_acc - baseline_acc,
    }


def run_backtest(
    per_game_path: Path,
    data_root: Path,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    per_game, schedules, player_features, snapshot_name, _player_feature_path = load_inputs(
        per_game_path, data_root
    )
    predictions = build_predictions_frame(per_game, schedules)

    # The six already-registered ACTIVE_PROSPECTIVE overlays, OR-combined --
    # the "production proxy" this script stacks on top of (see module
    # docstring for why this is a proxy, not a literal publishing.py replay).
    results = run_overlays(predictions, schedules, player_features)
    flip_sets = {name: {flip.game_id for flip in result.flips} for name, result in results.items()}
    verify_no_direction_conflicts(predictions, results, flip_sets)
    eval_frame = build_eval_frame(predictions, per_game, flip_sets)
    eval_frame = eval_frame.rename(columns={"correct_combined": "correct_production"})

    # The turnover-luck rebound tilt: independently evaluated against the
    # SAME unflipped baseline, exactly like every member of the six-overlay
    # stack.
    pbp_snapshot = latest_pbp_snapshot(data_root / "pbp" / "raw")
    pbp = load_pbp_snapshot(pbp_snapshot)
    my_result = apply_turnover_luck_rebound_tilt_overlay(predictions, schedules, pbp)
    my_flip_ids = {flip.game_id for flip in my_result.flips}

    # Direction-agreement check, mirroring verify_no_direction_conflicts:
    # every game MY overlay flips must set home_cover_probability to exactly
    # 1 - baseline, the same complement convention the six-member stack uses,
    # so the OR-union below is well-defined.
    baseline_probability = predictions.set_index("game_id")["home_cover_probability"]
    if my_flip_ids:
        overlaid = my_result.overlaid_predictions.set_index("game_id")["home_cover_probability"]
        ids = sorted(my_flip_ids)
        actual = overlaid.loc[ids].to_numpy(dtype=float)
        expected = 1.0 - baseline_probability.loc[ids].to_numpy(dtype=float)
        if not np.allclose(actual, expected, atol=1e-9):
            raise AssertionError(
                "turnover_luck_rebound_tilt_overlay flipped a game to something other than "
                "the complement of the baseline pick -- the OR-combination rule's premise is "
                "violated"
            )

    production_flip_ids = set(eval_frame.loc[eval_frame["flipped_combined"], "game_id"])
    new_flip_ids = my_flip_ids - production_flip_ids
    stacked_ids = production_flip_ids | my_flip_ids

    eval_frame["flipped_mine"] = eval_frame["game_id"].isin(my_flip_ids)
    eval_frame["flipped_stacked_with_mine"] = eval_frame["game_id"].isin(stacked_ids)
    eval_frame["correct_stacked_with_mine"] = np.where(
        eval_frame["flipped_stacked_with_mine"],
        1.0 - eval_frame["correct_baseline"],
        eval_frame["correct_baseline"],
    )

    n_games = len(eval_frame)
    n_pushes = int(eval_frame["correct_baseline"].isna().sum())
    n_scored = n_games - n_pushes
    week_block_count = int(eval_frame[["season", "week"]].drop_duplicates().shape[0])
    season_block_count = int(eval_frame["season"].nunique())

    marginal_on_production = run_both_blockings(
        eval_frame, marginal_metric, samples=samples, seed=seed
    )
    marginal_on_baseline = run_both_blockings(
        eval_frame, baseline_metric, samples=samples, seed=seed
    )

    valid_baseline = eval_frame.dropna(subset=["correct_baseline"])
    valid_production = eval_frame.dropna(subset=["correct_production"])
    valid_stacked = eval_frame.dropna(subset=["correct_stacked_with_mine"])

    return {
        "computed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "challenger_id": CHALLENGER_ID,
        "new_rotation_family": NEW_ROTATION_FAMILY,
        "frozen_threshold": TURNOVER_UNDER_Q25_THRESHOLD,
        "predeclaration_note": (
            "Not a predeclared, window-spending measurement -- a diagnostic layering the "
            "ACTIVE_PROSPECTIVE turnover_luck_rebound_tilt_overlay onto the SAME six-member "
            "OR-combined overlay stack scripts/overlay_stack_backtest.py already tracks, "
            "against the same already-frozen, already-published opener archive. MINED-SEASONS "
            "read; CONTEXT for the challenger registration, not a gate on it -- per AGENTS.md, "
            "an interval crossing zero is never grounds to decline registering. No "
            "rotation-registry window is spent by this run."
        ),
        "production_proxy_note": (
            "'production' here is the six already-registered ACTIVE_PROSPECTIVE overlays "
            f"({', '.join(OVERLAY_NAMES)}) OR-combined on the bare baseline -- the closest "
            "already-built proxy for what is being tracked, NOT a literal replay of "
            "publishing.py's four_overlay_composition-composed card, which was out of scope "
            "for this script's reuse list."
        ),
        "source_artifact": str(per_game_path),
        "schedule_snapshot": snapshot_name,
        "pbp_snapshot": pbp_snapshot.snapshot_id,
        "grading_rule": "production probability rule (home_cover_probability >= 0.5) at the "
        "Tuesday-opener decision line, matching pool.py/backtest.py -- not the sign rule "
        "docs/opener_evaluation.md originally predeclared (see that doc's 2026-08-19 addendum).",
        "seasons": [int(eval_frame["season"].min()), int(eval_frame["season"].max())],
        "n_games": n_games,
        "n_pushes": n_pushes,
        "n_scored_games": n_scored,
        "week_block_count": week_block_count,
        "season_block_count": season_block_count,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "n_flipped_vs_baseline": len(my_flip_ids),
        "n_flipped_ids_vs_baseline": sorted(my_flip_ids),
        "n_new_flips_not_already_in_production": len(new_flip_ids),
        "n_flipped_already_covered_by_production": len(my_flip_ids & production_flip_ids),
        "n_both_flagged_games": len(my_result.both_flagged_games),
        "accuracy": {
            "baseline_accuracy": float(valid_baseline["correct_baseline"].mean()),
            "production_accuracy": float(valid_production["correct_production"].mean()),
            "stacked_with_mine_accuracy": float(valid_stacked["correct_stacked_with_mine"].mean()),
        },
        "marginal_delta_vs_production": marginal_on_production,
        "marginal_delta_vs_bare_baseline": marginal_on_baseline,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path, default=DEFAULT_PER_GAME_ARTIFACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    per_game_path = (REPO_ROOT / args.per_game_artifact).resolve()
    data_root = (REPO_ROOT / args.data_root).resolve()

    started = time.time()
    result = run_backtest(per_game_path, data_root, samples=args.samples, seed=args.seed)

    configuration = {
        "per_game_artifact": str(args.per_game_artifact),
        "data_root": str(args.data_root),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "confidence": DEFAULT_CONFIDENCE,
        "challenger_id": CHALLENGER_ID,
        "overlay_stack_members": list(OVERLAY_NAMES),
        "frozen_threshold": TURNOVER_UNDER_Q25_THRESHOLD,
        "predeclaration": "docs/turnover_luck_rebound_tilt_overlay.md",
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, per_game_path, project_root=REPO_ROOT),
    }

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO_ROOT / args.output_root / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="turnover-luck-rebound-stacked-backtest",
        metrics={
            "delta_vs_production_week_blocked": result["marginal_delta_vs_production"][
                "week_blocked"
            ]["delta"]["estimate"],
            "n_flipped_vs_baseline": result["n_flipped_vs_baseline"],
        },
        notes=(
            "turnover_luck_rebound_tilt_overlay (bottom-quartile prior-season centered "
            "turnover differential rebound) stacked on top of the six-member ACTIVE_PROSPECTIVE "
            "overlay OR-union, against the frozen opener archive. Mined-seasons read; context "
            "for the challenger registration, not a gate -- see "
            "docs/turnover_luck_rebound_tilt_overlay.md."
        ),
    )
    print(f"wrote {output_dir / 'results.json'}")

    delta_week = result["marginal_delta_vs_production"]["week_blocked"]["delta"]
    delta_season = result["marginal_delta_vs_production"]["season_blocked"]["delta"]
    print(
        "Marginal delta vs production (week-blocked): "
        f"{delta_week['estimate'] * 100:+.3f} pts "
        f"[{delta_week['lower'] * 100:+.3f}, {delta_week['upper'] * 100:+.3f}] "
        f"P+ {delta_week['probability_positive']:.4f}"
    )
    print(
        "Marginal delta vs production (season-blocked): "
        f"{delta_season['estimate'] * 100:+.3f} pts "
        f"[{delta_season['lower'] * 100:+.3f}, {delta_season['upper'] * 100:+.3f}] "
        f"P+ {delta_season['probability_positive']:.4f}"
    )
    n_flipped = result["n_flipped_vs_baseline"]
    n_new = result["n_new_flips_not_already_in_production"]
    print(f"n_flipped vs baseline: {n_flipped}")
    print(f"n_new flips not already covered by production: {n_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
