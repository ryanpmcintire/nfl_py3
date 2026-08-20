"""Combined overlay-stack backtest at the opener grade.

Measures what the STACK of all six pick-flipping prospective overlays would
have scored historically, applied jointly to the frozen active model's own
opener-graded picks (production probability rule: ``home_cover_probability
>= 0.5``). This answers a question no single challenger registration answers:
where do the live challengers put the pool relative to the 55% goal if all of
them fire together, not one at a time against the un-flipped card the way
``nfl-ats prospective-score`` tracks each challenger independently
(docs/*_overlay.md, "tracked INDEPENDENTLY ... not stacked on the other
overlays").

Baseline: ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``,
the tracked, real (non-scratch) run of ``nfl-ats opener-evaluation`` for the
active `weak_stack`/ridge-alpha-10 model, 1,537 REG-season games 2020-2025,
graded with the PRODUCTION probability rule
(``correct_at_open_probability_rule`` / ``home_cover_probability_at_open``) --
not the sign rule docs/opener_evaluation.md originally predeclared. See that
document's 2026-08-19 addendum for why the probability rule is production's
actual pick rule.

Six pick-flipping overlays are applied JOINTLY (the seventh and eighth live
challengers are excluded and the exclusion is recorded in the output, not
silently dropped):

* ``coach_fade_overlay``           -- nfl_ats.coach_fade_overlay (weeks 1-8 only)
* ``injury_value_lost_tilt_overlay`` -- nfl_ats.injury_value_tilt_overlay
* ``division_revenge_tilt_overlay``  -- nfl_ats.division_revenge_tilt_overlay
* ``backup_qb_fade_overlay``         -- nfl_ats.backup_qb_fade_overlay
* ``surface_switch_tilt_overlay``    -- nfl_ats.surface_switch_tilt_overlay
* ``spread_gap_zone_fade_overlay``   -- nfl_ats.spread_gap_zone_fade_overlay

Excluded, and why: ``mod07_weak_signal_stack`` IS the active model itself
(this backtest already starts from its own opener-graded picks as the
baseline -- it is not a pick-flipping overlay ON TOP of anything).
``best_pick_nomination_v2`` only chooses which already-picked game gets the
week's Best-Pick bonus marker; it never touches ``home_cover_probability`` or
which side is picked, so it cannot move ATS accuracy and is out of scope for
a pick-flipping stack.

Combination rule (derived from the code, then verified empirically, not
assumed): every one of the six overlays, when it fires, sets
``home_cover_probability`` to exactly ``1 - baseline_probability`` -- the
complement of the model's OWN original pick, computed independently against
that same unflipped baseline (this mirrors exactly how each challenger is
scored in production: independently, against the un-flipped card). Since
there are only two sides, any two overlays that both fire on the same game
necessarily agree on the resulting side -- there is no construction under
which this stack can produce a genuine direction conflict. This script
verifies that empirically (``verify_no_direction_conflicts``) rather than
just asserting it. The combined-stack pick is therefore well-defined with a
simple OR across the six independent flip conditions: flip a game if ANY
overlay's condition fires on it, complementing the baseline probability.

Uncertainty: ``nfl_ats.clv.week_blocked_bootstrap`` (the same block-bootstrap
tool -- whole-week or whole-season resampling, ``probability_positive`` as
the continuous read, not a binary "contains zero" verdict) that produced the
reference artifact's own ``uncertainty.csv``, run at 20,000 samples with a
fixed seed for: the combined stack vs baseline, each overlay's marginal
(leave-one-out) contribution inside the stack, and each overlay's own solo
(unstacked) delta vs baseline for context.

Caveat, stated in the output, not just here: several of the six overlays
were themselves screened or tuned on windows this archive re-touches (e.g.
the surface-switch construct's NFL era split covers 2018-2025; the coach-fade
construct's registered effect lives inside 2018-2025; several bias-battery
cells were opener-re-screened on 2020-2025 -- the same 2020-2025 span this
archive covers). This combined read is CONTINUOUS EVIDENCE on already-looked
-at windows, a diagnostic, not a fresh confirmation, and it spends no
rotation-registry window.

Usage (from the repo root, per AGENTS.md environment conventions)::

    .\\.tools\\uv.exe run --no-sync python scripts/overlay_stack_backtest.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.backup_qb_fade_overlay import apply_backup_qb_fade_overlay
from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.coach_fade_overlay import apply_coach_fade_overlay
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
from nfl_ats.injury_value_tilt_overlay import apply_injury_value_tilt_overlay
from nfl_ats.provenance import sha256_file
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.spread_gap_zone_fade_overlay import apply_spread_gap_zone_fade_overlay
from nfl_ats.surface_switch_tilt_overlay import apply_surface_switch_tilt_overlay
from nfl_ats.surgical_gating import VALUE_LOST_DIFF_COLUMNS

DEFAULT_PER_GAME_ARTIFACT = Path("artifacts/opener_evaluation/20260819T174244Z/per_game.parquet")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_ROOT = Path("artifacts/overlay_stack_backtest")
DEFAULT_SAMPLES = 20_000
DEFAULT_SEED = 20260819
DEFAULT_CONFIDENCE = 0.95

OVERLAY_NAMES: tuple[str, ...] = (
    "coach_fade_overlay",
    "injury_value_lost_tilt_overlay",
    "division_revenge_tilt_overlay",
    "backup_qb_fade_overlay",
    "surface_switch_tilt_overlay",
    "spread_gap_zone_fade_overlay",
)

EXCLUDED_CHALLENGERS: dict[str, str] = {
    "mod07_weak_signal_stack": (
        "IS the active production model itself (weak_stack/market_residual/ridge "
        "alpha 10) -- this backtest already starts from ITS OWN opener-graded picks "
        "as the baseline. It is not a pick-flipping overlay layered on top of anything."
    ),
    "best_pick_nomination_v2": (
        "Chooses which ALREADY-PICKED game gets the week's Best-Pick bonus marker. "
        "It never changes home_cover_probability or which side is picked, so it "
        "cannot move ATS accuracy and has no place in a pick-flipping overlay stack."
    ),
}


def load_inputs(
    per_game_path: Path, data_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Path]:
    per_game = pd.read_parquet(per_game_path)
    snapshot = latest_snapshot(data_root / "raw")
    schedules, _team_stats = load_snapshot(snapshot)
    player_feature_path = data_root / "processed" / "game_features_player.parquet"
    player_features = pd.read_parquet(
        player_feature_path, columns=["game_id", *VALUE_LOST_DIFF_COLUMNS]
    )
    return per_game, schedules, player_features, snapshot.root.name, player_feature_path


def build_predictions_frame(per_game: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """The 1,537-game opener archive, reshaped into the pick-level card schema
    every overlay's ``apply_*`` function expects (``game_id``/``season``/``week``/
    ``home_team``/``away_team``/``game_type``/``spread_line``/``home_cover_probability``).

    ``home_cover_probability`` is seeded from ``home_cover_probability_at_open``
    -- production's own probability rule at the opener, not the sign rule --
    and ``spread_line`` from ``tue_open_home_spread``, the decision line every
    pick in this archive was actually formed at, matching the exact field the
    sibling overlays' own recorders read for ``decision_home_spread``.
    """

    sched_cols = schedules[["game_id", "home_team", "away_team", "game_type"]].drop_duplicates(
        "game_id"
    )
    predictions = per_game[
        [
            "game_id",
            "season",
            "week",
            "tue_open_home_spread",
            "home_cover_probability_at_open",
        ]
    ].merge(sched_cols, on="game_id", how="left", validate="one_to_one")

    missing_meta = (
        predictions["home_team"].isna()
        | predictions["away_team"].isna()
        | predictions["game_type"].isna()
    )
    if missing_meta.any():
        raise ValueError(
            f"{int(missing_meta.sum())} archived games have no matching schedule row: "
            f"{predictions.loc[missing_meta, 'game_id'].tolist()}"
        )
    non_reg = predictions.loc[predictions["game_type"].ne("REG")]
    if not non_reg.empty:
        raise ValueError(
            f"opener-evaluation archive contains non-REG games: {non_reg['game_id'].tolist()}"
        )

    predictions = predictions.rename(
        columns={
            "tue_open_home_spread": "spread_line",
            "home_cover_probability_at_open": "home_cover_probability",
        }
    )
    return predictions.reset_index(drop=True)


def run_overlays(
    predictions: pd.DataFrame, schedules: pd.DataFrame, player_features: pd.DataFrame
) -> dict[str, Any]:
    return {
        "coach_fade_overlay": apply_coach_fade_overlay(predictions, schedules),
        "injury_value_lost_tilt_overlay": apply_injury_value_tilt_overlay(
            predictions, player_features
        ),
        "division_revenge_tilt_overlay": apply_division_revenge_tilt_overlay(
            predictions, schedules
        ),
        "backup_qb_fade_overlay": apply_backup_qb_fade_overlay(predictions, schedules),
        "surface_switch_tilt_overlay": apply_surface_switch_tilt_overlay(predictions, schedules),
        "spread_gap_zone_fade_overlay": apply_spread_gap_zone_fade_overlay(predictions),
    }


def verify_no_direction_conflicts(
    predictions: pd.DataFrame, results: dict[str, Any], flip_sets: dict[str, set[str]]
) -> None:
    """Every overlay's flip must equal ``1 - baseline`` on every game it flips.

    This is the empirical check behind the module docstring's combination-rule
    claim: if it held only "by construction", a future edit to one overlay
    (e.g. a partial-magnitude flip instead of a full complement) would silently
    break the OR-combination logic below without this script noticing. Raises
    if any overlay ever disagrees with its own baseline complement.
    """

    baseline = predictions.set_index("game_id")["home_cover_probability"]
    for name, result in results.items():
        ids = sorted(flip_sets[name])
        if not ids:
            continue
        overlaid = result.overlaid_predictions.set_index("game_id")["home_cover_probability"]
        actual = overlaid.loc[ids].to_numpy(dtype=float)
        expected = 1.0 - baseline.loc[ids].to_numpy(dtype=float)
        if not np.allclose(actual, expected, atol=1e-9):
            raise AssertionError(
                f"{name} flipped a game to something other than the complement of the "
                "baseline pick -- the OR-combination rule's premise is violated"
            )


def bootstrap_to_dict(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        out[str(row["metric"])] = {
            "estimate": float(row["estimate"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "probability_positive": float(row["probability_positive"]),
            "confidence": float(row["confidence"]),
            "block": str(row["block"]),
            "bootstrap_samples": int(row["samples"]),
        }
    return out


def run_both_blockings(
    frame: pd.DataFrame,
    metric_fn: Any,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    week_result = week_blocked_bootstrap(
        frame, metric_fn, block="week", samples=samples, confidence=DEFAULT_CONFIDENCE, seed=seed
    )
    season_result = week_blocked_bootstrap(
        frame,
        metric_fn,
        block="season",
        samples=samples,
        confidence=DEFAULT_CONFIDENCE,
        seed=seed,
    )
    return {
        "week_blocked": bootstrap_to_dict(week_result),
        "season_blocked": bootstrap_to_dict(season_result),
    }


def combined_metric(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["correct_baseline", "correct_combined"])
    baseline_acc = float(valid["correct_baseline"].mean()) if len(valid) else float("nan")
    combined_acc = float(valid["correct_combined"].mean()) if len(valid) else float("nan")
    return {
        "baseline_accuracy": baseline_acc,
        "combined_accuracy": combined_acc,
        "combined_minus_baseline": combined_acc - baseline_acc,
    }


def make_marginal_metric(name: str) -> Any:
    loo_col = f"correct_loo_{name}"

    def _metric(df: pd.DataFrame) -> dict[str, float]:
        valid = df.dropna(subset=["correct_combined", loo_col])
        full_acc = float(valid["correct_combined"].mean()) if len(valid) else float("nan")
        loo_acc = float(valid[loo_col].mean()) if len(valid) else float("nan")
        return {
            "full_stack_accuracy": full_acc,
            "leave_one_out_accuracy": loo_acc,
            "marginal_delta": full_acc - loo_acc,
        }

    return _metric


def make_solo_metric(name: str) -> Any:
    solo_col = f"correct_solo_{name}"

    def _metric(df: pd.DataFrame) -> dict[str, float]:
        valid = df.dropna(subset=["correct_baseline", solo_col])
        baseline_acc = float(valid["correct_baseline"].mean()) if len(valid) else float("nan")
        solo_acc = float(valid[solo_col].mean()) if len(valid) else float("nan")
        return {
            "baseline_accuracy": baseline_acc,
            "solo_accuracy": solo_acc,
            "solo_minus_baseline": solo_acc - baseline_acc,
        }

    return _metric


def build_eval_frame(
    predictions: pd.DataFrame,
    per_game: pd.DataFrame,
    flip_sets: dict[str, set[str]],
) -> pd.DataFrame:
    eval_frame = predictions[["game_id", "season", "week"]].merge(
        per_game[["game_id", "correct_at_open_probability_rule"]], on="game_id", how="left"
    )
    eval_frame = eval_frame.rename(columns={"correct_at_open_probability_rule": "correct_baseline"})
    eval_frame["correct_baseline"] = pd.to_numeric(eval_frame["correct_baseline"], errors="coerce")

    combined_ids = set().union(*flip_sets.values()) if flip_sets else set()
    eval_frame["flipped_combined"] = eval_frame["game_id"].isin(combined_ids)
    eval_frame["correct_combined"] = np.where(
        eval_frame["flipped_combined"],
        1.0 - eval_frame["correct_baseline"],
        eval_frame["correct_baseline"],
    )

    for name in OVERLAY_NAMES:
        solo_ids = flip_sets[name]
        eval_frame[f"correct_solo_{name}"] = np.where(
            eval_frame["game_id"].isin(solo_ids),
            1.0 - eval_frame["correct_baseline"],
            eval_frame["correct_baseline"],
        )
        other_ids: set[str] = set()
        for other_name in OVERLAY_NAMES:
            if other_name != name:
                other_ids |= flip_sets[other_name]
        eval_frame[f"correct_loo_{name}"] = np.where(
            eval_frame["game_id"].isin(other_ids),
            1.0 - eval_frame["correct_baseline"],
            eval_frame["correct_baseline"],
        )

    return eval_frame


def overlap_diagnostics(eval_frame: pd.DataFrame, flip_sets: dict[str, set[str]]) -> dict[str, Any]:
    membership = pd.DataFrame(
        {name: eval_frame["game_id"].isin(flip_sets[name]) for name in OVERLAY_NAMES}
    )
    touch_count = membership.sum(axis=1)
    multiplicity_hist = {
        str(int(k)): int(v) for k, v in touch_count.value_counts().sort_index().items()
    }
    pairwise: dict[str, int] = {}
    for i, a in enumerate(OVERLAY_NAMES):
        for b in OVERLAY_NAMES[i + 1 :]:
            pairwise[f"{a}__{b}"] = int((membership[a] & membership[b]).sum())
    touched_game_ids = {
        name: sorted(eval_frame.loc[membership[name], "game_id"]) for name in OVERLAY_NAMES
    }
    return {
        "pairwise_overlap_counts": pairwise,
        "flip_multiplicity_histogram": multiplicity_hist,
        "games_touched_by_at_least_one_overlay": int((touch_count > 0).sum()),
        "games_touched_by_multiple_overlays": int((touch_count > 1).sum()),
        "sum_of_solo_flip_counts": int(sum(len(s) for s in flip_sets.values())),
        "union_flip_count": int((touch_count > 0).sum()),
        "conflict_note": (
            "Every overlay's flip sets home_cover_probability to exactly "
            "1 - baseline_probability (verified programmatically for every flipped "
            "row of every overlay against the SAME unflipped baseline pick -- see "
            "verify_no_direction_conflicts in scripts/overlay_stack_backtest.py). Since "
            "there are only two sides, any two overlays that both fire on the same game "
            "necessarily agree on the resulting side: 'overlap' here means redundant "
            "coverage, never a contradictory recommendation. Zero direction conflicts "
            "are possible by this construction, and the check above found zero."
        ),
        "touched_game_ids_by_overlay": touched_game_ids,
    }


def run_backtest(
    per_game_path: Path,
    data_root: Path,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    per_game, schedules, player_features, snapshot_name, player_feature_path = load_inputs(
        per_game_path, data_root
    )
    predictions = build_predictions_frame(per_game, schedules)
    results = run_overlays(predictions, schedules, player_features)
    flip_sets = {name: {flip.game_id for flip in result.flips} for name, result in results.items()}
    verify_no_direction_conflicts(predictions, results, flip_sets)

    eval_frame = build_eval_frame(predictions, per_game, flip_sets)
    n_games = len(eval_frame)
    n_pushes = int(eval_frame["correct_baseline"].isna().sum())
    n_scored = n_games - n_pushes
    week_block_count = int(eval_frame[["season", "week"]].drop_duplicates().shape[0])
    season_block_count = int(eval_frame["season"].nunique())

    combined_stack = run_both_blockings(eval_frame, combined_metric, samples=samples, seed=seed)
    marginal_contributions = {
        name: run_both_blockings(eval_frame, make_marginal_metric(name), samples=samples, seed=seed)
        for name in OVERLAY_NAMES
    }
    solo_vs_baseline = {
        name: run_both_blockings(eval_frame, make_solo_metric(name), samples=samples, seed=seed)
        for name in OVERLAY_NAMES
    }
    overlap = overlap_diagnostics(eval_frame, flip_sets)

    return {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration_note": (
            "Not a predeclared, window-spending measurement -- a diagnostic combining "
            "already-registered ACTIVE_PROSPECTIVE overlays' rules against an "
            "already-frozen, already-published opener archive. Several of the six "
            "overlays were themselves screened/tuned on windows this archive re-touches "
            "(e.g. the surface-switch NFL era split covers 2018-2025; the coach-fade "
            "registered effect lives inside 2018-2025; several bias-battery cells were "
            "opener-re-screened on 2020-2025, the same 2020-2025 span this archive "
            "covers). Treat this as CONTINUOUS EVIDENCE on already-looked-at windows, "
            "not a fresh confirmation. No rotation-registry window is spent by this run."
        ),
        "source_artifact": str(per_game_path),
        "source_artifact_sha256": sha256_file(per_game_path),
        "schedule_snapshot": snapshot_name,
        "player_feature_table": str(player_feature_path),
        "player_feature_table_sha256": sha256_file(player_feature_path),
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
        "excluded_challengers": EXCLUDED_CHALLENGERS,
        "overlay_flip_counts": {name: results[name].flip_count for name in OVERLAY_NAMES},
        "overlap": overlap,
        "combined_stack": combined_stack,
        "marginal_contributions": marginal_contributions,
        "solo_vs_baseline": solo_vs_baseline,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path, default=DEFAULT_PER_GAME_ARTIFACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    payload = run_backtest(
        args.per_game_artifact, args.data_root, samples=args.samples, seed=args.seed
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    combined_week = payload["combined_stack"]["week_blocked"]["combined_minus_baseline"]
    combined_season = payload["combined_stack"]["season_blocked"]["combined_minus_baseline"]
    print(f"Wrote {output_path}")
    print(
        "Combined stack vs baseline (week-blocked): "
        f"{combined_week['estimate'] * 100:+.3f} pts "
        f"[{combined_week['lower'] * 100:+.3f}, {combined_week['upper'] * 100:+.3f}] "
        f"P+ {combined_week['probability_positive']:.4f}"
    )
    print(
        "Combined stack vs baseline (season-blocked): "
        f"{combined_season['estimate'] * 100:+.3f} pts "
        f"[{combined_season['lower'] * 100:+.3f}, {combined_season['upper'] * 100:+.3f}] "
        f"P+ {combined_season['probability_positive']:.4f}"
    )
    for name in OVERLAY_NAMES:
        print(f"  {name}: {payload['overlay_flip_counts'][name]} triggers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
