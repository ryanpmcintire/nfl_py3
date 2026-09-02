"""Bye-edge fade overlay: stacked-on-production back-test.

Measures what applying ``bye_edge_fade_overlay`` ALONE would have scored
historically on top of the active model's own opener-graded picks
(production probability rule: ``home_cover_probability >= 0.5``) -- exactly
the same "solo vs. baseline" question ``scripts/overlay_stack_backtest.py``
already answers for its own six pick-flipping overlays, extended to this
seventh, separately-registered challenger. This is CONTEXT for the
registration decision, not a gate on it (see the module docstring of
``src/nfl_ats/bye_edge_fade_overlay.py`` and ``docs/bye_edge_fade_overlay.md``
for the verdict-handling rule this script's output feeds).

Baseline: ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``,
the tracked, real (non-scratch) run of ``nfl-ats opener-evaluation`` for the
active ``weak_stack``/ridge-alpha-10 model, 1,537 REG-season games 2020-2025,
graded with the PRODUCTION probability rule
(``correct_at_open_probability_rule`` / ``home_cover_probability_at_open``) --
the SAME frozen artifact ``scripts/overlay_stack_backtest.py`` and
``scripts/overlay_subset_composition.py`` both use, baseline accuracy 53.36%
on 1,503 scored games.

Machinery reused, not reimplemented, from ``scripts/overlay_stack_backtest.py``
(``load_inputs``, ``build_predictions_frame``, ``build_eval_frame``,
``run_both_blockings``, ``DEFAULT_PER_GAME_ARTIFACT``, ``OVERLAY_NAMES``),
imported the same way ``scripts/overlay_subset_composition.py`` does (a bare
``from overlay_stack_backtest import ...`` -- resolvable because running this
script directly puts ``scripts/`` on ``sys.path``). ``build_eval_frame``
expects one flip-set per name in ``OVERLAY_NAMES`` (the six existing
overlays); this script passes EMPTY sets for all six and this overlay's own
flip-set under a seventh key
(``bye_edge_fade_overlay``, not a member of ``OVERLAY_NAMES``), so
``correct_combined`` reflects ONLY this overlay's own flips on top of the
unflipped baseline -- never stacked on the six already-registered overlays.

Uncertainty: ``nfl_ats.clv.week_blocked_bootstrap`` (via
``run_both_blockings``, the same helper the reference artifacts use), 20,000
samples, a fixed recorded seed, week-blocked primary plus a season-blocked
secondary.

Verdict handling (restated here so this script stands on its own; the
closing-grounds taxonomy is binding per AGENTS.md/CLAUDE.md and must be
pasted verbatim into any adjudicating subagent prompt):

    An interval or CI that contains zero is NEVER grounds to reject, fail,
    or close an experiment. Only two grounds ever close a line of work:
    (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
    wrong side of zero) or zero split-half reliability; (2) bounded by a
    positive control proven able to detect an effect that size. Everything
    else is `unresolved_below_power`: record it with `nfl-ats weak-signals
    record`, report `probability_positive`, never the binary "contains
    zero".

Applied to this script's own output: if the whole week-blocked AND
season-blocked 95% interval sits BELOW zero, the stacked form is a resolved
wrong sign (record ``refuted_mechanism`` / ``wrong_sign_resolved`` in a new
family ``bye_overval_fade_stacked_on_production``, and do not propose
registering the challenger). Every other case -- including an interval that
crosses zero, and including a negative point estimate whose interval crosses
zero -- is ``unresolved_below_power`` in that same family, and the
challenger registration proceeds regardless.

Usage (from the repo root, per AGENTS.md environment conventions)::

    .\\.tools\\uv.exe run --no-sync python scripts/bye_edge_fade_stacked_backtest.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from overlay_stack_backtest import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PER_GAME_ARTIFACT,
    OVERLAY_NAMES,
    build_eval_frame,
    build_predictions_frame,
    load_inputs,
    run_both_blockings,
)

from nfl_ats.bye_edge_fade_overlay import CHALLENGER_ID, apply_bye_edge_fade_overlay
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = Path("artifacts/bye_edge_fade_stacked")
DEFAULT_SAMPLES = 20_000
#: Recorded, fixed seed -- matches the bye-overvaluation screen family's own
#: bootstrap-seed convention (scripts/bye_overvaluation_screen.py,
#: scripts/overlay_subset_composition.py both use 20260821).
DEFAULT_SEED = 20260821
CONFIDENCE = 0.95


def bye_edge_fade_metric(df: Any) -> dict[str, float]:
    """Mirrors ``overlay_stack_backtest.combined_metric`` exactly, renamed."""

    valid = df.dropna(subset=["correct_baseline", "correct_combined"])
    baseline_acc = float(valid["correct_baseline"].mean()) if len(valid) else float("nan")
    candidate_acc = float(valid["correct_combined"].mean()) if len(valid) else float("nan")
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
    result = apply_bye_edge_fade_overlay(predictions, schedules)
    flip_ids = sorted({flip.game_id for flip in result.flips})

    # build_eval_frame requires a flip-set for every name in OVERLAY_NAMES
    # (the six existing overlays); pass empty sets for all six so this
    # overlay is scored ALONE, never stacked on any of them.
    flip_sets: dict[str, set[str]] = {name: set() for name in OVERLAY_NAMES}
    flip_sets[CHALLENGER_ID] = set(flip_ids)
    eval_frame = build_eval_frame(predictions, per_game, flip_sets)

    stats = run_both_blockings(eval_frame, bye_edge_fade_metric, samples=samples, seed=seed)

    n_games = len(eval_frame)
    n_pushes = int(eval_frame["correct_baseline"].isna().sum())
    n_scored = n_games - n_pushes
    week_block_count = int(eval_frame[["season", "week"]].drop_duplicates().shape[0])
    season_block_count = int(eval_frame["season"].nunique())

    return {
        "source_artifact": str(per_game_path),
        "source_artifact_sha256": sha256_file(per_game_path),
        "schedule_snapshot": snapshot_name,
        "grading_rule": (
            "production probability rule (home_cover_probability >= 0.5) at the "
            "Tuesday-opener decision line, matching pool.py/backtest.py and "
            "scripts/overlay_stack_backtest.py -- not the sign rule "
            "docs/opener_evaluation.md originally predeclared."
        ),
        "combination_note": (
            "bye_edge_fade_overlay applied ALONE on top of the raw, unflipped "
            "active-model opener picks (home_cover_probability_at_open) -- the SAME "
            "'production picks' baseline every existing overlay's solo read uses in "
            "scripts/overlay_stack_backtest.py's solo_vs_baseline; not stacked on any "
            "of the six already-registered overlays that script tracks."
        ),
        "seasons": [int(eval_frame["season"].min()), int(eval_frame["season"].max())],
        "n_games": n_games,
        "n_pushes": n_pushes,
        "n_scored_games": n_scored,
        "n_flipped": len(flip_ids),
        "flipped_game_ids": flip_ids,
        "week_block_count": week_block_count,
        "season_block_count": season_block_count,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "week_blocked": stats["week_blocked"],
        "season_blocked": stats["season_blocked"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path, default=DEFAULT_PER_GAME_ARTIFACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    started = time.time()
    result = run_backtest(
        args.per_game_artifact, args.data_root, samples=args.samples, seed=args.seed
    )

    configuration = {
        "per_game_artifact": str(args.per_game_artifact),
        "data_root": str(args.data_root),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "grading_rule": result["grading_rule"],
        "combination_note": result["combination_note"],
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        "predeclaration_note": (
            "Not a predeclared, window-spending measurement -- a diagnostic applying "
            "the already-registered ACTIVE_PROSPECTIVE bye_edge_fade_overlay's "
            "parameter-free rule to an already-frozen, already-published opener "
            "archive (mirrors scripts/overlay_stack_backtest.py's own "
            "predeclaration_note). This is a MINED-SEASONS read, CONTEXT for the "
            "registration decision, not a gate on it -- see AGENTS.md's crossing-zero "
            "invariant. No rotation-registry window is spent by this run."
        ),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(
            configuration,
            args.per_game_artifact,
            project_root=REPO_ROOT,
        ),
    }

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_root / timestamp
    week_delta = result["week_blocked"]["candidate_minus_baseline"]
    season_delta = result["season_blocked"]["candidate_minus_baseline"]
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="bye-edge-fade-stacked-backtest",
        metrics={
            "week_blocked_delta_accuracy_points": week_delta["estimate"] * 100.0,
            "week_blocked_probability_positive": week_delta["probability_positive"],
            "season_blocked_delta_accuracy_points": season_delta["estimate"] * 100.0,
            "season_blocked_probability_positive": season_delta["probability_positive"],
            "n_flipped": result["n_flipped"],
        },
        notes=(
            "Stacked-on-production back-test of bye_edge_fade_overlay "
            "(registry/weak_signals.json:bye_overval_fade_full_slate_post2011) applied "
            "alone on top of the frozen active-model opener baseline "
            "(artifacts/opener_evaluation/20260819T174244Z/per_game.parquet); a "
            "mined-seasons read, not a fresh confirmation, spending no "
            "rotation-registry window."
        ),
    )

    print(f"Wrote {output_dir / 'results.json'}")
    print(f"n_flipped: {result['n_flipped']} of {result['n_games']} games")
    print(
        "Bye-edge fade vs baseline (week-blocked): "
        f"{week_delta['estimate'] * 100:+.4f} pts "
        f"[{week_delta['lower'] * 100:+.4f}, {week_delta['upper'] * 100:+.4f}] "
        f"P+ {week_delta['probability_positive']:.4f} "
        f"n_blocks={result['week_block_count']}"
    )
    print(
        "Bye-edge fade vs baseline (season-blocked): "
        f"{season_delta['estimate'] * 100:+.4f} pts "
        f"[{season_delta['lower'] * 100:+.4f}, {season_delta['upper'] * 100:+.4f}] "
        f"P+ {season_delta['probability_positive']:.4f} "
        f"n_blocks={result['season_block_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
