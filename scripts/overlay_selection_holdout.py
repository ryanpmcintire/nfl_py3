"""Holdout de-biasing of the 127-subset overlay-composition selection.

Quantifies how much of the composition study's headline (+2.06 pts for
coach_fade+division_revenge+player_arrests_back_side_policy+
spread_gap_zone_fade, artifacts/overlay_subset_composition/20260821T174356Z)
survives when the subset is chosen on one set of seasons and scored on
disjoint seasons. This is ATTRIBUTION ON ALREADY-SCORED DATA: both halves sit
inside the same frozen opener archive, so no rotation-registry window is
spent and nothing here is a fresh confirmation.

PREDECLARED READS (stated before this script was ever executed):

1. SPLIT-HALF HOLDOUT. Selection half = seasons {2020, 2021, 2022}. Choose the
   subset maximizing the FULL-SLATE accuracy point estimate (mean paired
   delta vs unflipped baseline) on SELECTION-HALF GAMES ONLY. Freeze it.
   Evaluate that exact frozen subset on evaluation half = seasons
   {2023, 2024, 2025}: paired candidate-vs-unflipped-baseline accuracy delta,
   week-blocked AND season-blocked bootstrap (20,000 samples, seed 20260821),
   probability_positive reported as the continuous read. Selection-half delta
   and holdout delta are reported side by side.

2. REVERSE SPLIT. Identical protocol with selection on {2023, 2024, 2025} and
   evaluation on {2020, 2021, 2022}, so no conclusion rests on one arbitrary
   cut of the calendar.

3. RANK STABILITY. Across ALL 127 non-empty subsets: Spearman rank
   correlation of per-subset deltas between the two halves, and the OLS
   slope of holdout delta on selection-half delta (slope < 1 is the
   shrinkage factor attributable to selection luck). Also reported: where
   each split's winning subset ranks on the OTHER half (its out-of-sample
   rank among all 127), and where the full-slate global-max subset ranks
   within each half.

Reference anchors are computed INSIDE each half so the reader can calibrate:
naive all-seven stack, arrest-only-on-baseline, and the true production
chain (coach fade -> player-arrests back-side policy, applied sequentially),
each against the same unflipped-baseline accuracy for that half.

Machinery (flip conditions, delta construction, blocked bootstrap) is reused
unchanged from scripts/overlay_subset_composition.py /
scripts/overlay_stack_backtest.py, which were verified exactly equivalent to
``nfl_ats.clv.week_blocked_bootstrap``.

Usage (from the repo root)::

    .\\.tools\\uv.exe run python scripts/overlay_selection_holdout.py
"""

from __future__ import annotations

import argparse
import json
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
from scipy.stats import spearmanr

from nfl_ats.provenance import sha256_file

DEFAULT_INCIDENTS = Path("data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet")
DEFAULT_FEATURES = Path("data/processed/game_features_pbp.parquet")
DEFAULT_OUTPUT_ROOT = Path("artifacts/overlay_selection_holdout")
DEFAULT_SAMPLES = 20_000
DEFAULT_SEED = 20260821
FORWARD_SELECTION_SEASONS = (2020, 2021, 2022)
FORWARD_EVALUATION_SEASONS = (2023, 2024, 2025)


def _rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _bootstrap_read(
    deltas_column: np.ndarray,
    seasons: pd.Series,
    weeks: pd.Series,
    mask: np.ndarray,
    n_pushes: int,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    column = deltas_column[mask].reshape(-1, 1)
    blocks = pd.DataFrame(
        {"season": seasons.to_numpy()[mask], "week": weeks.to_numpy()[mask]}
    ).reset_index(drop=True)
    payload: dict[str, Any] = {
        "estimate_accuracy_points": float(column.mean() * 100.0),
        "n_games": int(mask.sum()),
        "n_pushes_in_half": int(n_pushes),
    }
    for block in ("week", "season"):
        stats = blocked_bootstrap_matrix(column, blocks, block=block, samples=samples, seed=seed)
        payload[f"{block}_blocked"] = {
            "estimate_accuracy_points": float(stats["estimate"][0] * 100.0),
            "lower_accuracy_points": float(stats["lower"][0] * 100.0),
            "upper_accuracy_points": float(stats["upper"][0] * 100.0),
            "probability_positive": float(stats["probability_positive"][0]),
            "standard_error_accuracy_points": float(stats["standard_error"][0] * 100.0),
            "block": block,
            "blocks": int(stats["block_count"]),
            "bootstrap_samples": samples,
            "confidence": CONFIDENCE,
        }
    return payload


def _holdout_direction(
    label: str,
    selection_seasons: tuple[int, ...],
    evaluation_seasons: tuple[int, ...],
    deltas: np.ndarray,
    subsets: list[tuple[str, ...]],
    columns: dict[tuple[str, ...], int],
    seasons: pd.Series,
    weeks: pd.Series,
    base_valid: np.ndarray,
    global_max_subset: tuple[str, ...],
    n_pushes_by_season: dict[int, int],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    selection_mask = seasons.isin(selection_seasons).to_numpy()
    evaluation_mask = seasons.isin(evaluation_seasons).to_numpy()
    selection_deltas = deltas[selection_mask]
    means = selection_deltas.mean(axis=0)
    best_column = int(np.argmax(means))
    frozen_subset = subsets[best_column]
    evaluation_deltas = deltas[evaluation_mask]
    eval_ranks = _rank_descending(evaluation_deltas.mean(axis=0))
    selection_ranks = _rank_descending(means)

    return {
        "label": label,
        "selection_seasons": list(selection_seasons),
        "evaluation_seasons": list(evaluation_seasons),
        "n_selection_games": int(selection_mask.sum()),
        "n_evaluation_games": int(evaluation_mask.sum()),
        "baseline_accuracy_selection_half": float(base_valid[selection_mask].mean()),
        "baseline_accuracy_evaluation_half": float(base_valid[evaluation_mask].mean()),
        "frozen_subset": {
            "members": list(frozen_subset),
            "union_flip_count_selection_half": int(
                np.count_nonzero(deltas[selection_mask][:, best_column])
            ),
            "union_flip_count_evaluation_half": int(
                np.count_nonzero(evaluation_deltas[:, best_column])
            ),
            "selection_half_delta_accuracy_points": float(means[best_column] * 100.0),
            "selection_half_rank_of_127": int(selection_ranks[best_column]),
            "holdout": _bootstrap_read(
                deltas[:, best_column],
                seasons,
                weeks,
                evaluation_mask,
                n_pushes=int(sum(n_pushes_by_season.get(int(s), 0) for s in evaluation_seasons)),
                samples=samples,
                seed=seed,
            ),
        },
        "global_max_subset_out_of_sample": {
            "members": list(global_max_subset),
            "column": columns[global_max_subset],
            "selection_half_rank_of_127": int(selection_ranks[columns[global_max_subset]]),
            "evaluation_half_rank_of_127": int(eval_ranks[columns[global_max_subset]]),
            "evaluation_half_delta_accuracy_points": float(
                evaluation_deltas[:, columns[global_max_subset]].mean() * 100.0
            ),
        },
    }


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


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

    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(members) + 1):
        subsets.extend(tuple(sorted(combo)) for combo in combinations(members, size))
    columns = {subset: index for index, subset in enumerate(subsets)}

    eval_frame = predictions[["game_id", "season", "week"]].merge(
        per_game[["game_id", "correct_at_open_probability_rule"]], on="game_id", how="left"
    )
    eval_frame = eval_frame.rename(columns={"correct_at_open_probability_rule": "correct_baseline"})
    eval_frame["correct_baseline"] = pd.to_numeric(eval_frame["correct_baseline"], errors="coerce")
    valid_mask = eval_frame["correct_baseline"].notna().to_numpy()

    delta_matrix_full = build_delta_matrix(
        eval_frame["correct_baseline"], eval_frame["game_id"], member_flip_sets, members, subsets
    )
    deltas = delta_matrix_full[valid_mask]
    seasons = eval_frame.loc[valid_mask, "season"].reset_index(drop=True)
    weeks = eval_frame.loc[valid_mask, "week"].reset_index(drop=True)
    n_pushes_by_season: dict[int, int] = {}
    for season_value in eval_frame["season"].unique():
        season_rows = eval_frame["season"].eq(season_value)
        pushes = int((season_rows & ~eval_frame["correct_baseline"].notna()).sum())
        if pushes:
            n_pushes_by_season[int(season_value)] = pushes
    base_valid = eval_frame.loc[valid_mask, "correct_baseline"].to_numpy(dtype=float)

    full_slate_means = deltas.mean(axis=0)
    global_max_column = int(np.argmax(full_slate_means))
    global_max_subset = subsets[global_max_column]

    forward = _holdout_direction(
        "split_half_holdout",
        FORWARD_SELECTION_SEASONS,
        FORWARD_EVALUATION_SEASONS,
        deltas,
        subsets,
        columns,
        seasons,
        weeks,
        base_valid,
        global_max_subset,
        n_pushes_by_season,
        samples=args.samples,
        seed=args.seed,
    )
    reverse = _holdout_direction(
        "reverse_split",
        FORWARD_EVALUATION_SEASONS,
        FORWARD_SELECTION_SEASONS,
        deltas,
        subsets,
        columns,
        seasons,
        weeks,
        base_valid,
        global_max_subset,
        n_pushes_by_season,
        samples=args.samples,
        seed=args.seed,
    )

    selection_forward = seasons.isin(FORWARD_SELECTION_SEASONS).to_numpy()
    selection_means = deltas[selection_forward].mean(axis=0) * 100.0
    evaluation_means = deltas[~selection_forward].mean(axis=0) * 100.0
    rho = float(spearmanr(selection_means, evaluation_means).statistic)
    slope = float(np.polyfit(selection_means, evaluation_means, 1)[0])

    coach_flip_ids = flip_sets[OVERLAY_NAMES[0]]
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
    sequential_by_game = sequential_correct.reindex(eval_frame["game_id"]).to_numpy(dtype=float)[
        valid_mask
    ]
    sequential_flip_by_game = (
        sequential_arrest_flip.reindex(eval_frame["game_id"]).fillna(False).to_numpy(dtype=bool)
    )[valid_mask]

    references: dict[str, Any] = {}
    for half_name, season_tuple in (
        ("selection_2020_2022", FORWARD_SELECTION_SEASONS),
        ("evaluation_2023_2025", FORWARD_EVALUATION_SEASONS),
    ):
        half_mask = seasons.isin(season_tuple).to_numpy()
        half_base = base_valid[half_mask]
        all_seven_column = columns[tuple(sorted(members))]
        arrest_only_column = columns[(ARREST_MEMBER_NAME,)]
        seq_valid = ~np.isnan(sequential_by_game[half_mask])
        references[half_name] = {
            "seasons": list(season_tuple),
            "n_games": int(half_mask.sum()),
            "baseline_accuracy": float(half_base.mean()),
            "naive_all_seven": {
                "members": sorted(members),
                "delta_accuracy_points": float(
                    deltas[half_mask][:, all_seven_column].mean() * 100.0
                ),
                "candidate_accuracy": float(
                    (half_base + deltas[half_mask][:, all_seven_column]).mean()
                ),
                "union_flips_in_half": int(
                    np.count_nonzero(deltas[half_mask][:, all_seven_column])
                ),
            },
            "arrest_only_on_baseline": {
                "delta_accuracy_points": float(
                    deltas[half_mask][:, arrest_only_column].mean() * 100.0
                ),
                "candidate_accuracy": float(
                    (half_base + deltas[half_mask][:, arrest_only_column]).mean()
                ),
                "union_flips_in_half": int(
                    np.count_nonzero(deltas[half_mask][:, arrest_only_column])
                ),
            },
            "production_chain_coach_then_arrest": {
                "candidate_accuracy": float(sequential_by_game[half_mask][seq_valid].mean()),
                "arrest_flips_after_coach_in_half": int(sequential_flip_by_game[half_mask].sum()),
            },
        }

    payload = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration_note": (
            "The three reads (split-half holdout 2020-2022 -> 2023-2025, reverse "
            "split 2023-2025 -> 2020-2022, and the 127-subset rank-stability/"
            "shrinkage analysis) were stated verbatim in this script's docstring "
            "BEFORE any of their outputs existed. Attribution on already-scored "
            "archive data only; no rotation-registry window is spent."
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
        "combination_rule": (
            "Identical to scripts/overlay_subset_composition.py: a subset flips a "
            "game if ANY member fires, complementing the unflipped baseline pick."
        ),
        "seasons": sorted(int(s) for s in seasons.unique()),
        "n_scored_games": int(valid_mask.sum()),
        "n_subsets": len(subsets),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "full_slate_global_max": {
            "members": list(global_max_subset),
            "delta_accuracy_points": float(full_slate_means[global_max_column] * 100.0),
        },
        "split_half_holdout": forward,
        "reverse_split": reverse,
        "rank_stability": {
            "spearman_rho_selection_vs_holdout": rho,
            "ols_slope_holdout_on_selection_accuracy_points": slope,
            "interpretation": (
                "slope < 1 is the shrinkage factor: the fraction of a selection-"
                "half advantage expected to survive out-of-sample, on average"
            ),
        },
        "references_per_half": references,
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {output_path}")
    for direction in (forward, reverse):
        frozen = direction["frozen_subset"]
        holdout = frozen["holdout"]
        week = holdout["week_blocked"]
        season = holdout["season_blocked"]
        print(
            f"{direction['label']}: frozen {'+'.join(m for m in frozen['members'])} | "
            f"selection-half {frozen['selection_half_delta_accuracy_points']:+.4f} pts -> "
            f"holdout {holdout['estimate_accuracy_points']:+.4f} pts | week-blocked "
            f"{week['estimate_accuracy_points']:+.4f} [{week['lower_accuracy_points']:+.4f}, "
            f"{week['upper_accuracy_points']:+.4f}] P+ {week['probability_positive']:.4f} | "
            f"season-blocked P+ {season['probability_positive']:.4f}"
        )
    forward_rank = forward["global_max_subset_out_of_sample"]["evaluation_half_rank_of_127"]
    reverse_rank = reverse["global_max_subset_out_of_sample"]["evaluation_half_rank_of_127"]
    print(
        f"rank stability: Spearman rho {rho:.4f}, shrinkage slope {slope:.4f}, "
        f"global max ({'+'.join(global_max_subset)}) out-of-sample ranks "
        f"{forward_rank} (forward eval) / {reverse_rank} (reverse eval)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
