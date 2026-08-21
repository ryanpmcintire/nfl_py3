"""Composed-subset backtest of the pick-flipping overlays at the opener grade.

The all-six joint stack was measured resolvably WORSE than baseline
(artifacts/overlay_stack_backtest/20260819T191534Z/result.json: combined
50.37% vs baseline 53.36%, season-blocked P+ 0.00145). This script enumerates
ALL non-empty subsets of the six prospective overlays plus the reconstructed
player-arrests back-side flip (127 subsets), applies each jointly to the same
frozen baseline per-game picks (same complement-flip OR rule), scores each
under the production probability rule, and runs week-blocked AND season-blocked
paired bootstrap (candidate subset vs unflipped baseline; 20,000 samples,
seed 20260821) for every subset.

Baseline: artifacts/opener_evaluation/20260819T174244Z/per_game.parquet, the
active weak_stack model's 1,537 REG games 2020-2025 graded at the Tuesday
opener with the production probability rule (home_cover_probability >= 0.5),
baseline accuracy 53.36% on 1,503 scored games.

The arrest flip is reconstructed from scripts/player_arrests_policy_eval.py's
own frozen machinery (broad_incident_game_flags + apply_frozen_policy) against
the point-in-time incidents snapshot that script predeclared. The published
production-chain reference figures (candidate 53.7591% vs production 53.3599%)
are reproduced in-line as a reconstruction check.

Subset selection on the same archive this enumeration scores is MINING: the
top subset's figure is an upper bound inflated by selection, not a prospect.
The greedy forward-selection ordering is POST-HOC ATTRIBUTION -- it proposes,
it does not conclude.

Usage (from the repo root)::

    .\\.tools\\uv.exe run python scripts/overlay_subset_composition.py
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
from player_arrests_policy_eval import (
    apply_frozen_policy,
    broad_incident_game_flags,
)

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.provenance import sha256_file

DEFAULT_INCIDENTS = Path("data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet")
DEFAULT_FEATURES = Path("data/processed/game_features_pbp.parquet")
DEFAULT_OUTPUT_ROOT = Path("artifacts/overlay_subset_composition")
DEFAULT_SAMPLES = 20_000
DEFAULT_SEED = 20260821
ARREST_MEMBER_NAME = "player_arrests_back_side_policy"
CONFIDENCE = 0.95


def reconstruct_arrest_flip_set(
    per_game: pd.DataFrame,
    features_path: Path,
    incidents_path: Path,
) -> tuple[set[str], pd.DataFrame]:
    identity = pd.read_parquet(
        features_path,
        columns=["game_id", "gameday", "home_team", "away_team"],
    )
    identity = identity.loc[identity["game_id"].isin(per_game["game_id"])].copy()
    if len(identity) != len(per_game):
        raise ValueError(
            f"arrest identity join covers {len(identity)} of {len(per_game)} baseline rows"
        )
    incidents = pd.read_parquet(incidents_path, columns=["record_id", "incident_date", "team"])
    flags, _coverage = broad_incident_game_flags(identity, incidents)
    scored = apply_frozen_policy(per_game, flags)
    flips = set(scored.loc[scored["policy_flip"], "game_id"].astype(str))
    return flips, scored


def build_delta_matrix(
    correct_baseline: pd.Series,
    game_ids: pd.Series,
    member_flip_sets: dict[str, set[str]],
    members: tuple[str, ...],
    subsets: list[tuple[str, ...]],
) -> np.ndarray:
    base = correct_baseline.to_numpy(dtype=float)
    membership = {name: game_ids.isin(member_flip_sets[name]).to_numpy() for name in members}
    matrix = np.empty((len(base), len(subsets)), dtype=float)
    for column, subset in enumerate(subsets):
        flipped = np.zeros(len(base), dtype=bool)
        for name in subset:
            flipped |= membership[name]
        candidate = np.where(flipped, 1.0 - base, base)
        matrix[:, column] = candidate - base
    return matrix


def blocked_bootstrap_matrix(
    deltas: np.ndarray,
    blocks_frame: pd.DataFrame,
    *,
    block: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    group_columns = ["season", "week"] if block == "week" else ["season"]
    grouped_indices = list(
        blocks_frame.groupby(group_columns, sort=False, dropna=False).indices.values()
    )
    block_sums = np.vstack([deltas[idx].sum(axis=0) for idx in grouped_indices])
    block_counts = np.array([len(idx) for idx in grouped_indices], dtype=float)
    generator = np.random.default_rng(seed)
    draws = np.empty((samples, deltas.shape[1]), dtype=float)
    for sample_index in range(samples):
        selected = generator.integers(0, len(grouped_indices), size=len(grouped_indices))
        draws[sample_index] = block_sums[selected].sum(axis=0) / block_counts[selected].sum()
    tail = (1.0 - CONFIDENCE) / 2.0
    return {
        "estimate": deltas.mean(axis=0),
        "lower": np.quantile(draws, tail, axis=0),
        "upper": np.quantile(draws, 1.0 - tail, axis=0),
        "probability_positive": np.mean(draws > 0.0, axis=0),
        "standard_error": draws.std(axis=0, ddof=1),
        "block_count": len(grouped_indices),
    }


def verify_against_week_blocked_bootstrap(
    deltas: np.ndarray,
    columns: dict[str, int],
    blocks_frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}

    def metric(df: pd.DataFrame) -> dict[str, float]:
        return {"delta": float(df["delta"].mean())}

    for name, column in columns.items():
        frame = blocks_frame.copy()
        frame["delta"] = deltas[:, column]
        for block in ("week", "season"):
            reference = week_blocked_bootstrap(
                frame, metric, block=block, samples=samples, confidence=CONFIDENCE, seed=seed
            )
            row = reference.iloc[0]
            fast = blocked_bootstrap_matrix(
                deltas[:, [column]], blocks_frame, block=block, samples=samples, seed=seed
            )
            key = f"{name}_{block}"
            checks[key] = bool(
                np.isclose(row["estimate"], fast["estimate"][0], atol=1e-12)
                and np.isclose(row["lower"], fast["lower"][0], atol=1e-12)
                and np.isclose(row["upper"], fast["upper"][0], atol=1e-12)
                and np.isclose(row["probability_positive"], fast["probability_positive"][0])
            )
    return checks


def greedy_forward_selection(
    deltas: np.ndarray,
    columns: dict[tuple[str, ...], int],
    members: tuple[str, ...],
) -> list[dict[str, Any]]:
    chosen: list[str] = []
    remaining = list(members)
    steps: list[dict[str, Any]] = []
    while remaining:
        best_name = None
        best_delta = -np.inf
        for name in remaining:
            trial = tuple(sorted([*chosen, name]))
            delta = float(deltas[:, columns[trial]].mean())
            if delta > best_delta:
                best_delta = delta
                best_name = name
        assert best_name is not None
        chosen.append(best_name)
        remaining.remove(best_name)
        trial = tuple(sorted(chosen))
        column = columns[trial]
        steps.append(
            {
                "step": len(steps) + 1,
                "added": best_name,
                "members_so_far": sorted(chosen),
                "point_estimate_accuracy_points": float(deltas[:, column].mean() * 100.0),
            }
        )
    return steps


def accuracy_of_flips(correct_baseline: pd.Series, flipped: pd.Series) -> float:
    valid = correct_baseline.notna()
    base = correct_baseline[valid].to_numpy(dtype=float)
    candidate = np.where(flipped[valid].to_numpy(dtype=bool), 1.0 - base, base)
    return float(candidate.mean())


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
    valid_blocks = eval_frame.loc[valid_mask, ["season", "week"]].reset_index(drop=True)

    delta_matrix_full = build_delta_matrix(
        eval_frame["correct_baseline"], eval_frame["game_id"], member_flip_sets, members, subsets
    )
    deltas = delta_matrix_full[valid_mask]

    week_stats = blocked_bootstrap_matrix(
        deltas, valid_blocks, block="week", samples=args.samples, seed=args.seed
    )
    season_stats = blocked_bootstrap_matrix(
        deltas, valid_blocks, block="season", samples=args.samples, seed=args.seed
    )

    check_columns = {
        tuple(sorted([OVERLAY_NAMES[0], ARREST_MEMBER_NAME])): columns[
            tuple(sorted([OVERLAY_NAMES[0], ARREST_MEMBER_NAME]))
        ],
        tuple(sorted(members)): columns[tuple(sorted(members))],
    }
    equivalence_checks = verify_against_week_blocked_bootstrap(
        deltas, check_columns, valid_blocks, samples=args.samples, seed=args.seed
    )

    base_valid = eval_frame.loc[valid_mask, "correct_baseline"].to_numpy(dtype=float)
    baseline_accuracy = float(base_valid.mean())

    subset_records: list[dict[str, Any]] = []
    for subset in subsets:
        column = columns[subset]
        candidate_accuracy = float((base_valid + deltas[:, column]).mean())
        subset_records.append(
            {
                "members": list(subset),
                "n_members": len(subset),
                "union_flip_count": int(np.count_nonzero(deltas[:, column] != 0.0)),
                "baseline_accuracy": baseline_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "delta_estimate_accuracy_points": float(deltas[:, column].mean() * 100.0),
                "week_blocked": {
                    "estimate_accuracy_points": float(week_stats["estimate"][column] * 100.0),
                    "lower_accuracy_points": float(week_stats["lower"][column] * 100.0),
                    "upper_accuracy_points": float(week_stats["upper"][column] * 100.0),
                    "probability_positive": float(week_stats["probability_positive"][column]),
                    "standard_error_accuracy_points": float(
                        week_stats["standard_error"][column] * 100.0
                    ),
                    "block": "week",
                    "blocks": int(week_stats["block_count"]),
                    "bootstrap_samples": args.samples,
                    "confidence": CONFIDENCE,
                },
                "season_blocked": {
                    "estimate_accuracy_points": float(season_stats["estimate"][column] * 100.0),
                    "lower_accuracy_points": float(season_stats["lower"][column] * 100.0),
                    "upper_accuracy_points": float(season_stats["upper"][column] * 100.0),
                    "probability_positive": float(season_stats["probability_positive"][column]),
                    "standard_error_accuracy_points": float(
                        season_stats["standard_error"][column] * 100.0
                    ),
                    "block": "season",
                    "blocks": int(season_stats["block_count"]),
                    "bootstrap_samples": args.samples,
                    "confidence": CONFIDENCE,
                },
            }
        )
    subset_records.sort(
        key=lambda record: (-record["delta_estimate_accuracy_points"], record["members"])
    )

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
    sequential_pick.loc[sequential_pick.index.isin(coach_flip_ids)] = ~sequential_pick.loc[
        sequential_pick.index.isin(coach_flip_ids)
    ]
    sequential_opposes = sequential_pick.ne(home_flag)
    sequential_arrest_flip = exactly_one & sequential_opposes
    final_pick = sequential_pick.where(~sequential_arrest_flip, home_flag)
    base_aligned = game_correct.reindex(final_pick.index)
    sequential_correct = pd.Series(
        np.where(final_pick.eq(game_pick_home), base_aligned, 1.0 - base_aligned),
        index=final_pick.index,
    )
    sequential_valid = sequential_correct.notna()

    production_chain_reference = {
        "published_reference": {
            "source": "docs/player_arrests_policy_eval.md",
            "production_accuracy": 0.533599,
            "candidate_accuracy": 0.537591,
            "note": (
                "Published chain composes raw model -> coach fade -> player-arrests "
                "back-side policy (decision_policy_id coach_fade_then_player_arrests_v1); "
                "the historical 53.76% figure was measured by applying the arrest policy "
                "directly to the frozen 53.36% opener baseline."
            ),
        },
        "arrest_only_on_baseline": {
            "flips": len(arrest_flip_ids),
            "candidate_accuracy": accuracy_of_flips(
                eval_frame["correct_baseline"], eval_frame["game_id"].isin(arrest_flip_ids)
            ),
        },
        "coach_then_arrest_sequential": {
            "coach_flips": len(coach_flip_ids),
            "arrest_flips_after_coach": int(sequential_arrest_flip.sum()),
            "candidate_accuracy": float(sequential_correct[sequential_valid].mean()),
        },
    }

    payload = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration_note": (
            "Not a predeclared, window-spending measurement. The four identities recorded "
            "to the weak-signal registry from this artifact were declared BEFORE results "
            "were seen; the full 127-subset enumeration and the greedy ordering are "
            "post-hoc mining on already-looked-at windows and are attribution only."
        ),
        "selection_caveat": (
            "The top subset's figure is an upper bound inflated by selecting the maximum "
            "over 127 correlated candidates scored on the same 2020-2025 archive this "
            "enumeration re-uses. It is not a prospective expectation."
        ),
        "combination_rule": (
            "Identical to scripts/overlay_stack_backtest.py: each member's flip sets "
            "home_cover_probability to exactly 1 - baseline probability (verified "
            "programmatically for the six overlays via verify_no_direction_conflicts); a "
            "subset flips a game if ANY member fires, complementing the unflipped baseline "
            "pick. Arrest-member conditions are computed against the unflipped baseline, "
            "matching how the published arrest evaluation scored it."
        ),
        "source_artifact": str(args.per_game_artifact),
        "source_artifact_sha256": sha256_file(args.per_game_artifact),
        "schedule_snapshot": snapshot_name,
        "player_feature_table": str(player_feature_path),
        "player_feature_table_sha256": sha256_file(player_feature_path),
        "incidents_table": str(args.incidents),
        "incidents_table_sha256": sha256_file(args.incidents),
        "grading_rule": (
            "production probability rule (home_cover_probability >= 0.5) at the "
            "Tuesday-opener decision line"
        ),
        "seasons": [int(eval_frame["season"].min()), int(eval_frame["season"].max())],
        "n_games": len(eval_frame),
        "n_pushes": int((~valid_mask).sum()),
        "n_scored_games": int(valid_mask.sum()),
        "week_block_count": int(valid_blocks.drop_duplicates().shape[0]),
        "season_block_count": int(eval_frame["season"].nunique()),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "member_flip_counts": {name: len(member_flip_sets[name]) for name in members},
        "equivalence_check_vs_nfl_ats_week_blocked_bootstrap": equivalence_checks,
        "production_chain_reference": production_chain_reference,
        "greedy_forward_selection": {
            "label": "POST-HOC ATTRIBUTION: greedy forward selection maximizing the "
            "full-sample point estimate at each step, run on the same data the subsets "
            "were scored on. It proposes an ordering; it does not conclude anything.",
            "steps": greedy_forward_selection(deltas, columns, members),
        },
        "subsets": subset_records,
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Equivalence checks vs nfl_ats.clv.week_blocked_bootstrap: {equivalence_checks}")
    print("Production-chain reproduction:")
    for key, value in production_chain_reference.items():
        if isinstance(value, dict) and "candidate_accuracy" in value:
            print(f"  {key}: {value['candidate_accuracy'] * 100:.4f}%")
    print("Top five subsets by point estimate:")
    for record in subset_records[:5]:
        season = record["season_blocked"]
        print(
            f"  {'+'.join(record['members'])}: {record['candidate_accuracy'] * 100:.4f}% "
            f"(delta {record['delta_estimate_accuracy_points']:+.4f} pts, season-blocked "
            f"{season['estimate_accuracy_points']:+.4f} [{season['lower_accuracy_points']:+.4f}, "
            f"{season['upper_accuracy_points']:+.4f}] P+ {season['probability_positive']:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
