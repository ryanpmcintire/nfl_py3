"""Deterministic multi-entry allocation for forced-pick ATS pools.

The primary entry maximises expected correct picks game by game. Additional
entries diversify by flipping selected games while respecting a hard pairwise
overlap ceiling. This module only constructs paper pool cards; it has no wager
placement, settlement, or empirical model-selection behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultiEntryPlan:
    """Allocated card rows, pairwise overlap audit, and portfolio-level metrics."""

    entries: pd.DataFrame
    overlap: pd.DataFrame
    metrics: dict[str, Any]


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing multi-entry columns: {', '.join(missing)}")
    if predictions.empty:
        raise ValueError("Multi-entry planning requires at least one game")
    frame = predictions.copy()
    for column in ("season", "week"):
        if column in frame and (frame[column].isna().any() or frame[column].nunique() != 1):
            raise ValueError(f"Multi-entry planning requires exactly one {column}")
    if frame["gameday"].isna().any():
        raise ValueError("Multi-entry gameday values must be present")
    game_ids = frame["game_id"].astype(str).str.strip()
    if frame["game_id"].isna().any() or game_ids.eq("").any():
        raise ValueError("Multi-entry game_id values must be non-empty")
    if game_ids.duplicated().any():
        raise ValueError("Multi-entry game_id values must be unique")
    frame["game_id"] = game_ids
    for column in ("home_team", "away_team"):
        values = frame[column].astype(str).str.strip().str.upper()
        if frame[column].isna().any() or values.eq("").any() or values.eq("NAN").any():
            raise ValueError(f"Multi-entry {column} values must be non-empty")
        frame[column] = values
    if frame["home_team"].eq(frame["away_team"]).any():
        raise ValueError("Multi-entry home_team and away_team must differ")
    scheduled_teams = pd.concat([frame["home_team"], frame["away_team"]], ignore_index=True)
    if scheduled_teams.duplicated().any():
        duplicate = str(scheduled_teams.loc[scheduled_teams.duplicated()].iloc[0])
        raise ValueError(f"Multi-entry schedule includes team {duplicate} more than once")
    probability = pd.to_numeric(frame["home_cover_probability"], errors="coerce")
    if probability.isna().any() or not np.isfinite(probability).all():
        raise ValueError("Multi-entry cover probabilities must be finite numbers")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("Multi-entry cover probabilities must lie in [0, 1]")
    frame["home_cover_probability"] = probability
    spread = pd.to_numeric(frame["spread_line"], errors="coerce")
    if spread.isna().any() or not np.isfinite(spread).all():
        raise ValueError("Multi-entry spread lines must be finite numbers")
    frame["spread_line"] = spread
    return frame.sort_values("game_id").reset_index(drop=True)


def _candidate_flip_sets(
    game_ids: tuple[str, ...],
    flip_cost: np.ndarray,
    *,
    max_flips: int,
    max_expected_loss: float,
) -> list[frozenset[int]]:
    candidates: list[tuple[float, int, tuple[str, ...], frozenset[int]]] = []
    for count in range(1, max_flips + 1):
        for positions in combinations(range(len(game_ids)), count):
            loss = float(flip_cost[list(positions)].sum())
            if loss > max_expected_loss + 1e-12:
                continue
            candidates.append(
                (
                    loss,
                    count,
                    tuple(game_ids[position] for position in positions),
                    frozenset(positions),
                )
            )
    candidates.sort(key=lambda item: item[:3])
    return [item[3] for item in candidates]


def _select_entries(
    candidates: list[frozenset[int]],
    *,
    entry_count: int,
    min_disagreements: int,
) -> list[frozenset[int]]:
    selected: list[frozenset[int]] = [frozenset()]
    for candidate in candidates:
        if all(
            len(candidate.symmetric_difference(existing)) >= min_disagreements
            for existing in selected
        ):
            selected.append(candidate)
            if len(selected) == entry_count:
                return selected
    if len(selected) != entry_count:
        raise ValueError(
            "No multi-entry allocation satisfies the overlap, flip, and expected-loss constraints"
        )
    return selected


def build_multi_entry_plan(
    predictions: pd.DataFrame,
    *,
    entry_count: int,
    max_pairwise_overlap: float = 0.875,
    max_flips_from_primary: int | None = None,
    max_expected_correct_loss: float | None = None,
) -> MultiEntryPlan:
    """Allocate correlated ATS entries under an explicit overlap ceiling.

    The first card is the per-game maximum-probability card. Candidate variants
    are ordered by expected-correct loss, number of flips, then ``game_id``.
    Each subsequent card is the first candidate satisfying the overlap ceiling
    against every already-selected card. The deterministic greedy rule protects
    expected score while making its non-global nature explicit in ``metrics``.
    """

    if isinstance(entry_count, bool) or not isinstance(entry_count, int) or entry_count < 1:
        raise ValueError("entry_count must be a positive integer")
    if not math.isfinite(max_pairwise_overlap) or not 0.0 <= max_pairwise_overlap <= 1.0:
        raise ValueError("max_pairwise_overlap must lie in [0, 1]")
    if entry_count > 1 and max_pairwise_overlap == 1.0:
        raise ValueError("Multiple entries require max_pairwise_overlap below 1")
    frame = _validate_predictions(predictions)
    games = len(frame)
    if games > 18:
        raise ValueError("Multi-entry planning is weekly and supports at most 18 games")
    max_flips = games if max_flips_from_primary is None else max_flips_from_primary
    if isinstance(max_flips, bool) or not isinstance(max_flips, int) or not 0 <= max_flips <= games:
        raise ValueError("max_flips_from_primary must be an integer between 0 and the game count")
    max_loss = games if max_expected_correct_loss is None else float(max_expected_correct_loss)
    if not math.isfinite(max_loss) or max_loss < 0.0:
        raise ValueError("max_expected_correct_loss must be finite and non-negative")

    probability = frame["home_cover_probability"].to_numpy(dtype=float)
    primary_home = probability >= 0.5
    primary_probability = np.where(primary_home, probability, 1.0 - probability)
    flip_cost = 2.0 * np.abs(probability - 0.5)
    game_ids = tuple(frame["game_id"].astype(str))
    allowed_agreements = math.floor(max_pairwise_overlap * games + 1e-12)
    min_disagreements = games - allowed_agreements

    selected: list[frozenset[int]]
    if entry_count == 1:
        selected = [frozenset()]
    else:
        candidates = _candidate_flip_sets(
            game_ids,
            flip_cost,
            max_flips=max_flips,
            max_expected_loss=max_loss,
        )
        selected = _select_entries(
            candidates,
            entry_count=entry_count,
            min_disagreements=min_disagreements,
        )

    baseline_expected = float(primary_probability.sum())
    entry_rows: list[dict[str, Any]] = []
    for entry_index, flipped in enumerate(selected, start=1):
        expected_loss = float(flip_cost[list(flipped)].sum()) if flipped else 0.0
        expected_correct = baseline_expected - expected_loss
        for game_index, row in enumerate(frame.itertuples(index=False)):
            is_flipped = game_index in flipped
            pick_home = bool(primary_home[game_index]) != is_flipped
            pick_probability = (
                probability[game_index] if pick_home else 1.0 - probability[game_index]
            )
            entry_rows.append(
                {
                    "entry_id": entry_index,
                    "is_primary_entry": entry_index == 1,
                    "game_id": str(row.game_id),
                    "gameday": row.gameday,
                    "away_team": str(row.away_team),
                    "home_team": str(row.home_team),
                    "pool_side": "HOME" if pick_home else "AWAY",
                    "pool_pick": str(row.home_team) if pick_home else str(row.away_team),
                    "pick_line": (
                        -float(cast(Any, row.spread_line))
                        if pick_home
                        else float(cast(Any, row.spread_line))
                    ),
                    "pick_probability": float(pick_probability),
                    "flipped_from_primary": is_flipped,
                    "expected_correct_cost": float(flip_cost[game_index]) if is_flipped else 0.0,
                    "entry_expected_correct": expected_correct,
                    "entry_expected_loss_vs_primary": expected_loss,
                    "entry_flips_from_primary": len(flipped),
                }
            )
    entries = pd.DataFrame(entry_rows)

    overlap_rows: list[dict[str, Any]] = []
    for left, right in combinations(range(entry_count), 2):
        disagreements = len(selected[left].symmetric_difference(selected[right]))
        agreements = games - disagreements
        overlap_rows.append(
            {
                "entry_a": left + 1,
                "entry_b": right + 1,
                "agreements": agreements,
                "disagreements": disagreements,
                "overlap_rate": agreements / games,
                "within_limit": agreements <= allowed_agreements,
            }
        )
    overlap = pd.DataFrame(
        overlap_rows,
        columns=[
            "entry_a",
            "entry_b",
            "agreements",
            "disagreements",
            "overlap_rate",
            "within_limit",
        ],
    )
    observed_overlap = float(overlap["overlap_rate"].max()) if not overlap.empty else 1.0
    metrics: dict[str, Any] = {
        "paper_only": True,
        "method": "deterministic_expected_score_greedy",
        "entry_count": entry_count,
        "games": games,
        "baseline_expected_correct": baseline_expected,
        "total_expected_correct": float(
            entries.groupby("entry_id", sort=True)["entry_expected_correct"].first().sum()
        ),
        "max_pairwise_overlap_allowed": max_pairwise_overlap,
        "allowed_pairwise_agreements": allowed_agreements,
        "minimum_pairwise_disagreements": min_disagreements,
        "observed_max_pairwise_overlap": observed_overlap,
        "max_flips_from_primary": max_flips,
        "max_expected_correct_loss": max_loss,
        "candidate_order": "expected_loss_then_flip_count_then_game_id",
    }
    return MultiEntryPlan(entries=entries, overlap=overlap, metrics=metrics)


def multi_entry_plan_markdown(plan: MultiEntryPlan) -> str:
    """Render one audit row per entry plus its pairwise overlap table."""

    summary = (
        plan.entries.groupby("entry_id", sort=True)
        .agg(
            picks=("game_id", "size"),
            flips=("entry_flips_from_primary", "first"),
            expected_correct=("entry_expected_correct", "first"),
            expected_loss=("entry_expected_loss_vs_primary", "first"),
        )
        .reset_index()
    )
    summary["expected_correct"] = summary["expected_correct"].map(lambda value: f"{value:.3f}")
    summary["expected_loss"] = summary["expected_loss"].map(lambda value: f"{value:.3f}")
    overlap = plan.overlap.copy()
    if not overlap.empty:
        overlap["overlap_rate"] = overlap["overlap_rate"].map(lambda value: f"{value:.1%}")
    result = (
        "# Multi-entry ATS pool plan\n\n"
        "Paper pool cards only. Entry 1 maximizes expected correct picks; later entries "
        "trade stated expected score for controlled overlap.\n\n"
        + summary.to_markdown(index=False)
        + "\n"
    )
    if not overlap.empty:
        result += "\n## Pairwise overlap\n\n" + overlap.to_markdown(index=False) + "\n"
    return result
