"""Recompute opener-graded overlay subsets using the original frozen policies.

Extracted from scripts/overlay_subset_composition.py. The input adapters and
arrest policy preserve the sibling research scripts' algorithms so this module
works independently of the repository's scripts directory. The 127-subset
ranking remains archive attribution, not a new policy selection.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.backup_qb_fade_overlay import apply_backup_qb_fade_overlay
from nfl_ats.clv import pick_correct, week_blocked_bootstrap
from nfl_ats.coach_fade_overlay import apply_coach_fade_overlay
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
from nfl_ats.injury_value_tilt_overlay import apply_injury_value_tilt_overlay
from nfl_ats.provenance import sha256_file, write_stamped_artifact
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.spread_gap_zone_fade_overlay import apply_spread_gap_zone_fade_overlay
from nfl_ats.surface_switch_tilt_overlay import apply_surface_switch_tilt_overlay
from nfl_ats.surgical_gating import VALUE_LOST_DIFF_COLUMNS

OVERLAY_NAMES: tuple[str, ...] = (
    "coach_fade_overlay",
    "injury_value_lost_tilt_overlay",
    "division_revenge_tilt_overlay",
    "backup_qb_fade_overlay",
    "surface_switch_tilt_overlay",
    "spread_gap_zone_fade_overlay",
)


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


WINDOW_DAYS = 14

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "JAC": "JAX",
    "IN": "IND",
}


class PolicyEvaluationError(ValueError):
    """Frozen input or policy contract was violated."""


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PolicyEvaluationError(f"{label} is missing columns: {', '.join(missing)}")


def broad_incident_game_flags(
    games: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    window_days: int = WINDOW_DAYS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one broad, point-in-time incident flag for each game side."""

    if window_days != WINDOW_DAYS:
        raise PolicyEvaluationError(
            f"Frozen player-arrest policy requires window_days={WINDOW_DAYS}"
        )
    _require_columns(
        games,
        {"game_id", "gameday", "home_team", "away_team"},
        "game identity table",
    )
    _require_columns(
        incidents,
        {"record_id", "incident_date", "team"},
        "safe incident index",
    )
    if games["game_id"].duplicated().any():
        raise PolicyEvaluationError("game identity table contains duplicate game_id rows")
    if incidents["record_id"].duplicated().any():
        raise PolicyEvaluationError("safe incident index contains duplicate record_id rows")

    safe = incidents[["record_id", "incident_date", "team"]].copy()
    safe["incident_date"] = pd.to_datetime(safe["incident_date"], errors="coerce")
    if safe["incident_date"].isna().any():
        raise PolicyEvaluationError("safe incident index contains invalid incident dates")
    safe["team"] = safe["team"].astype("string").str.strip().replace(TEAM_ALIASES).astype(object)

    identity = games[["game_id", "gameday", "home_team", "away_team"]].copy()
    for team_column in ("home_team", "away_team"):
        identity[team_column] = (
            identity[team_column].astype("string").str.strip().replace(TEAM_ALIASES).astype(object)
        )
    identity["gameday"] = pd.to_datetime(identity["gameday"], errors="coerce")
    if identity["gameday"].isna().any():
        raise PolicyEvaluationError("game identity table contains invalid gameday values")
    days_since_tuesday = (identity["gameday"].dt.weekday - 1) % 7
    identity["decision_date"] = (
        identity["gameday"] - pd.to_timedelta(days_since_tuesday, unit="D")
    ).dt.normalize()

    schedule_teams = set(identity["home_team"]) | set(identity["away_team"])
    mapped = safe.loc[safe["team"].isin(schedule_teams)].copy()
    mapped = mapped.sort_values(["incident_date", "team", "record_id"])

    flags = identity[["game_id"]].copy()
    for side, team_column in (("home", "home_team"), ("away", "away_team")):
        team_games = identity[["game_id", "decision_date", team_column]].rename(
            columns={team_column: "team"}
        )
        team_games = team_games.sort_values(["decision_date", "team", "game_id"])
        joined = pd.merge_asof(
            team_games,
            mapped,
            by="team",
            left_on="decision_date",
            right_on="incident_date",
            direction="backward",
            allow_exact_matches=False,
        )
        age = (joined["decision_date"] - joined["incident_date"]).dt.days
        joined[f"{side}_incident_flag"] = age.between(1, window_days, inclusive="both")
        flags = flags.merge(
            joined[["game_id", f"{side}_incident_flag"]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    return flags, {
        "source_incidents": len(safe),
        "schedule_mapped_incidents": len(mapped),
    }


def apply_frozen_policy(opener: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Apply the predeclared flip rule and grade both arms at the opener."""

    _require_columns(
        opener,
        {
            "game_id",
            "season",
            "week",
            "margin_vs_open",
            "pick_home_at_open_probability_rule",
            "correct_at_open_probability_rule",
        },
        "frozen opener evaluation",
    )
    _require_columns(flags, {"game_id", "home_incident_flag", "away_incident_flag"}, "flags")
    if opener["game_id"].duplicated().any() or flags["game_id"].duplicated().any():
        raise PolicyEvaluationError("opener and flag tables must each have unique game_id rows")

    scored = opener.merge(flags, on="game_id", how="left", validate="one_to_one")
    if scored[["home_incident_flag", "away_incident_flag"]].isna().any().any():
        raise PolicyEvaluationError("every frozen opener game must receive both incident flags")
    scored["home_incident_flag"] = scored["home_incident_flag"].astype(bool)
    scored["away_incident_flag"] = scored["away_incident_flag"].astype(bool)

    production_pick = scored["pick_home_at_open_probability_rule"].astype(bool)
    exactly_one = scored["home_incident_flag"] ^ scored["away_incident_flag"]
    production_opposes = production_pick.ne(scored["home_incident_flag"])
    scored["policy_flip"] = exactly_one & production_opposes
    scored["candidate_pick_home"] = production_pick.where(
        ~scored["policy_flip"], scored["home_incident_flag"]
    )

    margin = pd.to_numeric(scored["margin_vs_open"], errors="coerce")
    recomputed_baseline = pick_correct(production_pick, margin).where(margin.notna())
    archived_baseline = pd.to_numeric(scored["correct_at_open_probability_rule"], errors="coerce")
    if not np.allclose(recomputed_baseline, archived_baseline, equal_nan=True):
        raise PolicyEvaluationError(
            "archived production correctness does not match its pick and opener margin"
        )
    scored["candidate_correct_at_open"] = pick_correct(scored["candidate_pick_home"], margin).where(
        margin.notna()
    )
    if not scored.loc[margin.eq(0.0), "candidate_correct_at_open"].isna().all():
        raise PolicyEvaluationError("candidate policy did not preserve opener pushes")
    return scored


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
    columns: dict[tuple[str, ...], int],
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


def run_overlay_composition(
    *,
    per_game_artifact: Path,
    data_root: Path,
    features: Path,
    incidents: Path = DEFAULT_INCIDENTS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Recompute archive composition without selecting a new policy."""
    started = perf_counter()
    if samples < 2:
        raise ValueError("samples must be at least 2")
    per_game_artifact = per_game_artifact.resolve()
    metadata = json.loads(per_game_artifact.with_name("metadata.json").read_text(encoding="utf-8"))
    per_game, schedules, player_features, snapshot_name, player_feature_path = load_inputs(
        per_game_artifact, data_root
    )
    predictions = build_predictions_frame(per_game, schedules)
    results = run_overlays(predictions, schedules, player_features)
    flip_sets = {name: {flip.game_id for flip in result.flips} for name, result in results.items()}
    verify_no_direction_conflicts(predictions, results, flip_sets)

    arrest_flip_ids, arrest_scored = reconstruct_arrest_flip_set(per_game, features, incidents)

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
        deltas, valid_blocks, block="week", samples=samples, seed=seed
    )
    season_stats = blocked_bootstrap_matrix(
        deltas, valid_blocks, block="season", samples=samples, seed=seed
    )

    check_columns = {
        tuple(sorted([OVERLAY_NAMES[0], ARREST_MEMBER_NAME])): columns[
            tuple(sorted([OVERLAY_NAMES[0], ARREST_MEMBER_NAME]))
        ],
        tuple(sorted(members)): columns[tuple(sorted(members))],
    }
    equivalence_checks = verify_against_week_blocked_bootstrap(
        deltas, check_columns, valid_blocks, samples=samples, seed=seed
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
                    "bootstrap_samples": samples,
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
                    "bootstrap_samples": samples,
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

    payload: dict[str, Any] = {
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
        "source_artifact": str(per_game_artifact),
        "active_model_id": metadata.get("active_model_id"),
        "active_model_config": metadata.get("active_model_config"),
        "probability_method": metadata.get("probability_method"),
        "feature_table_sha256": metadata.get("feature_table_sha256"),
        "source_artifact_sha256": sha256_file(per_game_artifact),
        "schedule_snapshot": snapshot_name,
        "player_feature_table": str(player_feature_path),
        "player_feature_table_sha256": sha256_file(player_feature_path),
        "incidents_table": str(incidents),
        "incidents_table_sha256": sha256_file(incidents),
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
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
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

    payload["timing"] = {"total_seconds": perf_counter() - started}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = output_root / timestamp
    stamped = write_stamped_artifact(payload, output_dir / "result.json")
    return {**stamped, "artifact_directory": str(output_dir)}
