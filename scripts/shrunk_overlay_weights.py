"""Structural de-overfit of overlay-subset selection: ridge-logistic shrunk weights.

The discrete approach chose ONE subset out of 127 correlated candidates by
hunting the maximum full-slate delta (best subset +2.06 pts,
artifacts/overlay_subset_composition); the split-half holdout measured the
resulting selection shrinkage at slope ~0.64 (artifacts/
overlay_selection_holdout/20260821T195512Z/result.json, rank-stability OLS
slope 0.6356). This script replaces max-hunting with structure: each game's
seven binary flip indicators (the six prospective overlays plus the
reconstructed player-arrests back-side policy) become FEATURES in an
L2-regularized logistic model of baseline pick correctness, so the weights
are continuous, shrunk, and cross-validated instead of selected.

PREDECLARED READS (stated here before this script was ever executed):

1. ALPHA SELECTION ON LOG LOSS ONLY. Ridge penalty sweep over
   {3, 10, 30, 100, 300} (sklearn LogisticRegression C = 1/alpha, lbfgs,
   intercept unpenalized). Alpha is chosen by LEAVE-ONE-SEASON-OUT cross-
   validation minimizing POOLED held-out LOG LOSS, averaged over all scored
   games across folds. No accuracy number is consulted until alpha is frozen.

2. ATTRIBUTION ESTIMATE (UPPER BOUND). Refit the chosen-alpha model on ALL
   1,503 scored games, flip where predicted P(baseline correct) < 0.5, score
   the flipped card against the unflipped baseline on the SAME games. This is
   IN-SAMPLE and therefore a fit-inflated upper bound.

3. NESTED WALK-FORWARD ESTIMATE (DEPLOYABLE EXPECTATION). Expanding window:
   for each season t after the first, fit on strictly PRIOR seasons only,
   predict season t, flip by the same 0.5 rule. Out-of-fold picks are scored
   chronologically against the unflipped baseline restricted to the same
   covered games. Season 2020 has no prior training data and is excluded from
   the nested estimate. THIS is the honest number.

4. THREE-ARM COMPARISON. Incumbent production chain (coach fade -> player-
   arrests back-side policy, applied sequentially, as shipped), best discrete
   subset (recomputed here as the argmax over all 127 non-empty subsets --
   selection-inflated reference, NOT a prospect), and the shrunk-weight
   policy. Paired deltas vs the unflipped baseline, week-blocked AND
   season-blocked bootstrap (20,000 samples, seed 20260822), probability_
   positive reported as the continuous read.

Attribution on already-scored archive data only (the frozen opener per-game
artifact); no rotation-registry window is spent.

Machinery (flip conditions, delta construction, blocked bootstrap) is reused
unchanged from scripts/overlay_subset_composition.py /
scripts/overlay_stack_backtest.py, whose fast bootstrap was verified exactly
equivalent to ``nfl_ats.clv.week_blocked_bootstrap`` (and is re-verified here
for the walk-forward column).

Usage (from the repo root)::

    .\\.tools\\uv.exe run python scripts/shrunk_overlay_weights.py
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from overlay_stack_backtest import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PER_GAME_ARTIFACT,
    OVERLAY_NAMES,
    build_predictions_frame,
    load_inputs,
    run_overlays,
    verify_no_direction_conflicts,
)
from overlay_subset_composition import (
    ARREST_MEMBER_NAME,
    CONFIDENCE,
    blocked_bootstrap_matrix,
    build_delta_matrix,
    reconstruct_arrest_flip_set,
)
from sklearn.linear_model import LogisticRegression

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact

ALPHA_GRID: tuple[float, ...] = (3.0, 10.0, 30.0, 100.0, 300.0)
DEFAULT_INCIDENTS = Path("data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet")
DEFAULT_FEATURES = Path("data/processed/game_features_pbp.parquet")
DEFAULT_OUTPUT_ROOT = Path("artifacts/shrunk_overlay_weights")
DEFAULT_SAMPLES = 20_000
DEFAULT_SEED = 20260822
FLIP_THRESHOLD = 0.5


def fit_ridge_logistic(
    features: np.ndarray, outcome: np.ndarray, alpha: float
) -> LogisticRegression:
    model = LogisticRegression(C=1.0 / alpha, solver="lbfgs", max_iter=10_000)
    model.fit(features, outcome)
    return model


def coefficient_dict(model: LogisticRegression, members: tuple[str, ...]) -> dict[str, float]:
    values = {"intercept": float(model.intercept_[0])}
    for name, value in zip(members, model.coef_[0], strict=True):
        values[f"beta_{name}"] = float(value)
    return values


def deltas_from_flips(flips: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    candidate = np.where(flips, 1.0 - outcome, outcome)
    return candidate - outcome


def select_alpha_loso(
    features: np.ndarray, outcome: np.ndarray, seasons: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """Leave-one-season-out CV on pooled held-out log loss. Declared metric."""
    unique_seasons = sorted(int(s) for s in np.unique(seasons))
    losses = np.empty((len(ALPHA_GRID), len(unique_seasons)), dtype=float)
    for alpha_index, alpha in enumerate(ALPHA_GRID):
        for season_index, season in enumerate(unique_seasons):
            test = seasons == season
            model = fit_ridge_logistic(features[~test], outcome[~test], alpha)
            prob = np.clip(predict_correct_probability(model, features[test]), 1e-12, 1 - 1e-12)
            y = outcome[test]
            losses[alpha_index, season_index] = float(
                -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))
            )
    pooled = losses.mean(axis=1)
    best_index = int(np.argmin(pooled))
    table = [
        {
            "alpha": float(alpha),
            "pooled_loso_log_loss": float(pooled[index]),
            "per_season_log_loss": {
                str(season): float(losses[index, season_index])
                for season_index, season in enumerate(unique_seasons)
            },
        }
        for index, alpha in enumerate(ALPHA_GRID)
    ]
    return float(ALPHA_GRID[best_index]), {"folds": unique_seasons, "table": table}


def predict_correct_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def run_walk_forward(
    features: np.ndarray,
    outcome: np.ndarray,
    seasons: np.ndarray,
    alpha: float,
    members: tuple[str, ...],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    """Expanding-window walk-forward. Train strictly prior seasons, predict next.

    Returns out-of-fold predicted P(correct) (NaN for the first season, which
    has no prior training data), per-fold diagnostics, and the coefficients of
    the final fold's fit.
    """
    unique_seasons = sorted(int(s) for s in np.unique(seasons))
    predicted = np.full(len(outcome), np.nan)
    folds: list[dict[str, Any]] = []
    final_coefficients: dict[str, float] = {}
    for season in unique_seasons[1:]:
        train = seasons < season
        test = seasons == season
        model = fit_ridge_logistic(features[train], outcome[train], alpha)
        predicted[test] = predict_correct_probability(model, features[test])
        final_coefficients = coefficient_dict(model, members)
        folds.append(
            {
                "target_season": season,
                "train_seasons": sorted(int(s) for s in np.unique(seasons[train])),
                "n_train_games": int(train.sum()),
                "n_test_games": int(test.sum()),
                "n_flips": int(np.count_nonzero(predicted[test] < FLIP_THRESHOLD)),
                "coefficients": coefficient_dict(model, members),
            }
        )
    return predicted, folds, final_coefficients


def bootstrap_scope(
    deltas: np.ndarray, blocks: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for block in ("week", "season"):
        stats = blocked_bootstrap_matrix(deltas, blocks, block=block, samples=samples, seed=seed)
        payload[f"{block}_blocked"] = {
            "block": block,
            "blocks": int(stats["block_count"]),
            "estimate_accuracy_points": (stats["estimate"] * 100.0).tolist(),
            "lower_accuracy_points": (stats["lower"] * 100.0).tolist(),
            "upper_accuracy_points": (stats["upper"] * 100.0).tolist(),
            "probability_positive": stats["probability_positive"].tolist(),
            "standard_error_accuracy_points": (stats["standard_error"] * 100.0).tolist(),
            "bootstrap_samples": samples,
            "confidence": CONFIDENCE,
        }
    return payload


def verify_equivalence(
    deltas_column: np.ndarray, blocks: pd.DataFrame, *, samples: int, seed: int
) -> bool:
    frame = blocks.copy()
    frame["delta"] = deltas_column

    def metric(df: pd.DataFrame) -> dict[str, float]:
        return {"delta": float(df["delta"].mean())}

    reference = week_blocked_bootstrap(
        frame, metric, block="week", samples=samples, confidence=CONFIDENCE, seed=seed
    )
    row = reference.iloc[0]
    fast = blocked_bootstrap_matrix(
        deltas_column.reshape(-1, 1), blocks, block="week", samples=samples, seed=seed
    )
    return bool(
        np.isclose(row["estimate"], fast["estimate"][0], atol=1e-12)
        and np.isclose(row["lower"], fast["lower"][0], atol=1e-12)
        and np.isclose(row["upper"], fast["upper"][0], atol=1e-12)
        and np.isclose(row["probability_positive"], fast["probability_positive"][0])
    )


def production_chain_deltas(
    per_game: pd.DataFrame,
    eval_frame: pd.DataFrame,
    valid_mask: np.ndarray,
    coach_flip_ids: set[str],
    arrest_scored: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Incumbent production chain: coach fade, then player-arrests back-side policy.

    Reconstructed exactly as scripts/overlay_subset_composition.py does it
    against the frozen baseline card. Returns per-game correctness of the
    sequentially flipped card (paired deltas are taken by the caller) and the
    count of arrest flips that survived the coach pass.
    """
    game_pick_home = per_game.set_index("game_id")["pick_home_at_open_probability_rule"].astype(
        bool
    )
    game_correct = per_game.set_index("game_id")["correct_at_open_probability_rule"]
    flags_indexed = arrest_scored.set_index("game_id")
    home_flag = flags_indexed["home_incident_flag"].astype(bool)
    away_flag = flags_indexed["away_incident_flag"].astype(bool)
    exactly_one = home_flag ^ away_flag
    sequential_pick = game_pick_home.copy()
    coach_index = sequential_pick.index.isin(coach_flip_ids)
    sequential_pick.loc[coach_index] = ~sequential_pick.loc[coach_index]
    sequential_opposes = sequential_pick.ne(home_flag)
    sequential_arrest_flip = exactly_one & sequential_opposes
    final_pick = sequential_pick.where(~sequential_arrest_flip, home_flag)
    base_aligned = game_correct.reindex(final_pick.index)
    sequential_correct = pd.Series(
        np.where(final_pick.eq(game_pick_home), base_aligned, 1.0 - base_aligned),
        index=final_pick.index,
    )
    by_game = sequential_correct.reindex(eval_frame["game_id"]).to_numpy(dtype=float)[valid_mask]
    arrest_flips_after_coach = (
        sequential_arrest_flip.reindex(eval_frame["game_id"]).fillna(False).to_numpy(dtype=bool)
    )[valid_mask]
    return by_game, int(arrest_flips_after_coach.sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path, default=DEFAULT_PER_GAME_ARTIFACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    per_game, schedules, player_features, snapshot_name, player_feature_path = load_inputs(
        args.per_game_artifact, args.data_root
    )
    predictions = build_predictions_frame(per_game, schedules)
    results = run_overlays(predictions, schedules, player_features)
    flip_sets = {name: {flip.game_id for flip in result.flips} for name, result in results.items()}
    verify_no_direction_conflicts(predictions, results, flip_sets)

    arrest_flip_ids, arrest_scored = reconstruct_arrest_flip_set(
        per_game, args.features, args.incidents
    )

    members: tuple[str, ...] = (*OVERLAY_NAMES, ARREST_MEMBER_NAME)
    member_flip_sets: dict[str, set[str]] = {name: flip_sets[name] for name in OVERLAY_NAMES}
    member_flip_sets[ARREST_MEMBER_NAME] = arrest_flip_ids

    eval_frame = predictions[["game_id", "season", "week"]].merge(
        per_game[["game_id", "correct_at_open_probability_rule"]], on="game_id", how="left"
    )
    eval_frame = eval_frame.rename(columns={"correct_at_open_probability_rule": "correct_baseline"})
    eval_frame["correct_baseline"] = pd.to_numeric(eval_frame["correct_baseline"], errors="coerce")
    valid_mask = eval_frame["correct_baseline"].notna().to_numpy()
    valid_blocks = eval_frame.loc[valid_mask, ["season", "week"]].reset_index(drop=True)

    outcome = eval_frame.loc[valid_mask, "correct_baseline"].to_numpy(dtype=float)
    seasons = valid_blocks["season"].to_numpy(dtype=int)
    indicator_matrix = np.column_stack(
        [
            eval_frame["game_id"].isin(member_flip_sets[name]).to_numpy(dtype=float)[valid_mask]
            for name in members
        ]
    )
    design = indicator_matrix
    n_scored = int(valid_mask.sum())
    baseline_accuracy = float(outcome.mean())

    chosen_alpha, alpha_cv = select_alpha_loso(design, outcome, seasons)

    full_model = fit_ridge_logistic(design, outcome, chosen_alpha)
    p_correct_full = predict_correct_probability(full_model, design)
    flips_full = p_correct_full < FLIP_THRESHOLD
    delta_attr = deltas_from_flips(flips_full, outcome)

    predicted_wf, folds, final_fold_coefficients = run_walk_forward(
        design, outcome, seasons, chosen_alpha, members
    )
    nested_mask = ~np.isnan(predicted_wf)
    flips_wf = predicted_wf[nested_mask] < FLIP_THRESHOLD
    delta_wf = deltas_from_flips(flips_wf, outcome[nested_mask])

    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(members) + 1):
        subsets.extend(tuple(sorted(combo)) for combo in combinations(members, size))
    delta_matrix_full = build_delta_matrix(
        eval_frame["correct_baseline"], eval_frame["game_id"], member_flip_sets, members, subsets
    )
    delta_matrix = delta_matrix_full[valid_mask]
    subset_means = delta_matrix.mean(axis=0)
    best_column = int(np.argmax(subset_means))
    best_subset = subsets[best_column]
    delta_best_subset = delta_matrix[:, best_column]

    coach_flip_ids = flip_sets[OVERLAY_NAMES[0]]
    seq_correct, _arrest_flips = production_chain_deltas(
        per_game, eval_frame, valid_mask, coach_flip_ids, arrest_scored
    )
    seq_deltas = seq_correct - outcome

    all_arms = np.column_stack([seq_deltas, delta_best_subset, delta_attr])
    all_arm_names = [
        "production_chain_coach_then_arrest",
        "best_discrete_subset_selection_inflated",
        "shrunk_weights_in_sample_attribution",
    ]
    nested_rows_all = np.column_stack(
        [seq_deltas[nested_mask], delta_best_subset[nested_mask], delta_attr[nested_mask], delta_wf]
    )
    nested_arm_names = [*all_arm_names, "shrunk_weights_walkforward_nested"]

    bootstrap_all = bootstrap_scope(all_arms, valid_blocks, samples=args.samples, seed=args.seed)
    nested_blocks = valid_blocks.loc[nested_mask].reset_index(drop=True)
    bootstrap_nested = bootstrap_scope(
        nested_rows_all, nested_blocks, samples=args.samples, seed=args.seed
    )
    equivalence_check = verify_equivalence(
        delta_wf, nested_blocks, samples=args.samples, seed=args.seed
    )

    week_stats = bootstrap_nested["week_blocked"]
    wf_index = len(nested_arm_names) - 1
    wf_estimate = float(week_stats["estimate_accuracy_points"][wf_index])
    wf_lower = float(week_stats["lower_accuracy_points"][wf_index])
    wf_upper = float(week_stats["upper_accuracy_points"][wf_index])
    wf_pplus = float(week_stats["probability_positive"][wf_index])
    wf_se = float(week_stats["standard_error_accuracy_points"][wf_index])
    season_stats = bootstrap_nested["season_blocked"]
    wf_season_lower = float(season_stats["lower_accuracy_points"][wf_index])
    wf_season_upper = float(season_stats["upper_accuracy_points"][wf_index])

    if wf_upper < 0.0:
        classification = "refuted_mechanism"
        closing_ground = "wrong_sign_resolved"
    else:
        classification = "unresolved_below_power"
        closing_ground = None

    payload = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration_note": (
            "Alpha selection by leave-one-season-out CV on pooled held-out LOG LOSS "
            "(grid {3, 10, 30, 100, 300}) was declared BEFORE any accuracy number "
            "existed; the in-sample attribution estimate is an upper bound and the "
            "expanding-window walk-forward is the honest deployable expectation. "
            "Attribution on already-scored archive data only; no rotation-registry "
            "window is spent."
        ),
        "source_artifact": str(args.per_game_artifact),
        "source_artifact_sha256": sha256_file(args.per_game_artifact),
        "schedule_snapshot": snapshot_name,
        "player_feature_table": str(player_feature_path),
        "incidents_table": str(args.incidents),
        "grading_rule": (
            "production probability rule (home_cover_probability >= 0.5) at the "
            "Tuesday-opener decision line; paired candidate-vs-unflipped-baseline"
        ),
        "flip_threshold": FLIP_THRESHOLD,
        "n_games": len(eval_frame),
        "n_scored_games": n_scored,
        "n_pushes": int((~valid_mask).sum()),
        "baseline_accuracy": baseline_accuracy,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "member_flip_counts": {name: len(member_flip_sets[name]) for name in members},
        "alpha_selection": {
            "declared_metric": "pooled_loso_log_loss",
            **alpha_cv,
            "chosen_alpha": chosen_alpha,
        },
        "weights_full_sample_fit": coefficient_dict(full_model, members),
        "weights_walkforward_final_fold_trained_through_prior_season": final_fold_coefficients,
        "walkforward_folds": folds,
        "attribution_in_sample": {
            "label": "UPPER BOUND: model refit on all scored games it is scored on",
            "n_games": n_scored,
            "n_flips": int(flips_full.sum()),
            "candidate_accuracy": float((outcome + delta_attr).mean()),
            "delta_accuracy_points": float(delta_attr.mean() * 100.0),
            "bootstrap": bootstrap_all,
        },
        "walkforward_nested": {
            "label": "DEPLOYABLE EXPECTATION: expanding-window out-of-fold picks",
            "covered_seasons": sorted(int(s) for s in np.unique(seasons[nested_mask])),
            "excluded_seasons": sorted(int(s) for s in np.unique(seasons[~nested_mask])),
            "n_games": int(nested_mask.sum()),
            "n_flips": int(flips_wf.sum()),
            "baseline_accuracy_on_covered_games": float(outcome[nested_mask].mean()),
            "candidate_accuracy_on_covered_games": float((outcome[nested_mask] + delta_wf).mean()),
            "delta_accuracy_points": float(delta_wf.mean() * 100.0),
            "bootstrap": bootstrap_nested,
        },
        "production_chain_reference": {
            "arrest_flips_after_coach": _arrest_flips,
            "note": (
                "Incumbent decision_policy_id coach_fade_then_player_arrests_v1, "
                "reconstructed against the frozen baseline card."
            ),
        },
        "best_discrete_subset_reference": {
            "members": list(best_subset),
            "full_slate_delta_accuracy_points": float(subset_means[best_column] * 100.0),
            "note": (
                "Selection-inflated reference: argmax over 127 correlated subsets on the "
                "same archive; not a prospect."
            ),
        },
        "equivalence_check_vs_nfl_ats_week_blocked_bootstrap": equivalence_check,
        "record_line_suggestion": {
            "name": "shrunk_overlay_policy_walkforward",
            "effect_accuracy_points": wf_estimate,
            "interval_low_accuracy_points": wf_lower,
            "interval_high_accuracy_points": wf_upper,
            "probability_positive": wf_pplus,
            "standard_error_accuracy_points": wf_se,
            "classification": classification,
            "closing_ground": closing_ground,
            "sample_games": int(nested_mask.sum()),
            "sample_blocks_week": int(week_stats["blocks"]),
            "sample_games_season_blocked": int(nested_mask.sum()),
            "sample_blocks_season": int(season_stats["blocks"]),
        },
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    metadata = {
        "created_at_utc": payload["computed_at_utc"],
        "command": "scripts/shrunk_overlay_weights.py",
        **payload,
        "provenance": artifact_provenance(
            configuration={
                "script": "scripts/shrunk_overlay_weights.py",
                "per_game_artifact": str(args.per_game_artifact),
                "alpha_grid": list(ALPHA_GRID),
                "alpha_selection_metric": "pooled_loso_log_loss",
                "flip_threshold": FLIP_THRESHOLD,
                "bootstrap_samples": args.samples,
                "bootstrap_seed": args.seed,
                "members": list(members),
            },
            feature_path=args.features,
            project_root=Path.cwd(),
        ),
    }
    write_experiment_artifact(
        output_dir,
        "result.json",
        metadata,
        command="shrunk-overlay-weights",
        metrics={
            "n_scored_games_attribution": float(n_scored),
            "n_games_walkforward_nested": float(int(nested_mask.sum())),
            "baseline_accuracy": baseline_accuracy,
            "chosen_alpha": chosen_alpha,
            "attribution_delta_accuracy_points": float(delta_attr.mean() * 100.0),
            "walkforward_delta_accuracy_points": wf_estimate,
            "interval_low": wf_lower,
            "interval_high": wf_upper,
            "probability_positive": wf_pplus,
            "sample_blocks": float(week_stats["blocks"]),
            "effect_units": "accuracy_points",
            "classification": classification,
        },
        notes=(
            "Structural de-overfit: ridge-logistic shrunk overlay weights; the "
            "walk-forward nested estimate is the deployable expectation, the "
            "in-sample attribution row an upper bound; recorded separately via "
            "`nfl-ats weak-signals record` (record lines in "
            "registry/experiments/shrunk-overlay-weights/record_lines.txt)."
        ),
        source="docs/shrunk_overlay_weights.md",
        project_root=Path.cwd(),
    )

    print(f"Wrote {output_path}")
    print(f"Chosen alpha (LOSO log-loss CV): {chosen_alpha}")
    for entry in alpha_cv["table"]:
        marker = " <-- chosen" if entry["alpha"] == chosen_alpha else ""
        print(
            f"  alpha {entry['alpha']:>6}: pooled LOSO log loss "
            f"{entry['pooled_loso_log_loss']:.6f}{marker}"
        )
    print("Weights (full-sample fit, intercept + seven indicators):")
    for key, value in payload["weights_full_sample_fit"].items():
        print(f"  {key}: {value:+.4f}")
    print(
        f"Attribution (in-sample upper bound, {n_scored} games): "
        f"{payload['attribution_in_sample']['delta_accuracy_points']:+.4f} pts, "
        f"{int(flips_full.sum())} flips"
    )
    print(
        f"Walk-forward (nested deployable, {int(nested_mask.sum())} games): "
        f"{float(delta_wf.mean() * 100.0):+.4f} pts, {int(flips_wf.sum())} flips | "
        f"week-blocked {wf_estimate:+.4f} [{wf_lower:+.4f}, {wf_upper:+.4f}] "
        f"P+ {wf_pplus:.4f} | season-blocked [{wf_season_lower:+.4f}, {wf_season_upper:+.4f}]"
    )
    print(f"Equivalence check vs nfl_ats.clv.week_blocked_bootstrap: {equivalence_check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
