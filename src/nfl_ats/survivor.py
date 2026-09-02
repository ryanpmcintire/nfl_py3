"""Deterministic survivor-pool planning from straight-up win probabilities.

A survivor entry needs one winner per week and may not reuse a team.  Maximising
this week's probability greedily can consume a strong favourite that is much
more valuable in a later week, so the planner solves the complete week-to-team
assignment in log-probability space.  It is decision support for a pool, not a
wagering interface and not an evaluator of whether the probabilities are true.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Candidate:
    week: int
    team: str
    opponent: str
    game_id: str
    gameday: object
    side: str
    probability: float


def _team(value: object, *, context: str) -> str:
    team = str(value).strip().upper()
    if not team or team == "NAN":
        raise ValueError(f"{context} contains an empty team")
    return team


def _hungarian_assignment(
    week_options: Sequence[Sequence[_Candidate]],
) -> tuple[tuple[_Candidate, ...], float]:
    """Return the maximum-log-probability injective assignment.

    This is the rectangular Hungarian algorithm.  Rows and columns are sorted,
    and strict comparisons retain the first equal-cost path, making exact ties
    deterministic without perturbing the probabilities being optimized.
    """

    if not week_options:
        return (), 0.0
    teams = sorted({candidate.team for options in week_options for candidate in options})
    if len(teams) < len(week_options):
        raise ValueError("No survivor plan exists: fewer available teams than planning weeks")
    team_index = {team: index for index, team in enumerate(teams)}
    candidates = {
        (row, item.team): item for row, options in enumerate(week_options) for item in options
    }

    invalid_cost = 1.0e12
    zero_cost = 1.0e6
    cost = np.full((len(week_options), len(teams)), invalid_cost, dtype=float)
    for row, options in enumerate(week_options):
        for candidate in options:
            cost[row, team_index[candidate.team]] = (
                -math.log(candidate.probability) if candidate.probability > 0.0 else zero_cost
            )

    # One-indexed implementation of the shortest augmenting-path form.
    rows, columns = cost.shape
    row_potential = np.zeros(rows + 1, dtype=float)
    column_potential = np.zeros(columns + 1, dtype=float)
    matched_row = np.zeros(columns + 1, dtype=int)
    predecessor = np.zeros(columns + 1, dtype=int)
    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = np.full(columns + 1, np.inf, dtype=float)
        used = np.zeros(columns + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, columns + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    cost[active_row - 1, candidate_column - 1]
                    - row_potential[active_row]
                    - column_potential[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            if not np.isfinite(delta):  # pragma: no cover - rectangular matrix prevents this
                raise ValueError("No survivor plan exists for the requested weeks")
            for candidate_column in range(columns + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assigned_columns = np.full(rows, -1, dtype=int)
    for column in range(1, columns + 1):
        if matched_row[column] != 0:
            assigned_columns[matched_row[column] - 1] = column - 1
    assignment: list[_Candidate] = []
    log_probability = 0.0
    for row, column in enumerate(assigned_columns):
        key = (row, teams[int(column)])
        if key not in candidates:
            week = min(option.week for option in week_options[row])
            raise ValueError(f"No survivor plan exists for week {week} without reusing a team")
        selected_candidate = candidates[key]
        assignment.append(selected_candidate)
        if selected_candidate.probability == 0.0:
            log_probability = -math.inf
        elif np.isfinite(log_probability):
            log_probability += math.log(selected_candidate.probability)
    return tuple(assignment), log_probability


def _probability(log_probability: float) -> float:
    return 0.0 if log_probability == -math.inf else math.exp(log_probability)


def _validate_and_expand(
    predictions: pd.DataFrame,
    *,
    method: str,
    weeks: Sequence[int] | None,
) -> tuple[int, tuple[int, ...], dict[int, tuple[_Candidate, ...]]]:
    required = {
        "season",
        "week",
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "method",
        "home_win_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing survivor columns: {', '.join(missing)}")
    frame = predictions.loc[predictions["method"].eq(method)].copy()
    if frame.empty:
        raise ValueError(f"No survivor predictions found for method {method!r}")
    if frame["game_id"].duplicated().any():
        raise ValueError(f"Method {method!r} contains duplicate game predictions")

    season_values = pd.to_numeric(frame["season"], errors="coerce")
    if season_values.isna().any() or not np.equal(season_values, np.floor(season_values)).all():
        raise ValueError("Survivor prediction seasons must be integers")
    seasons = sorted(set(season_values.astype(int)))
    if len(seasons) != 1:
        raise ValueError("Survivor planning requires predictions from exactly one season")
    season = seasons[0]

    week_values = pd.to_numeric(frame["week"], errors="coerce")
    if week_values.isna().any() or not np.equal(week_values, np.floor(week_values)).all():
        raise ValueError("Survivor prediction weeks must be positive integers")
    frame["week"] = week_values.astype(int)
    if (frame["week"] <= 0).any():
        raise ValueError("Survivor prediction weeks must be positive integers")
    available_weeks = tuple(sorted(frame["week"].unique().tolist()))
    selected_weeks = available_weeks if weeks is None else tuple(int(week) for week in weeks)
    if not selected_weeks:
        raise ValueError("Survivor planning requires at least one week")
    if (
        len(set(selected_weeks)) != len(selected_weeks)
        or tuple(sorted(selected_weeks)) != selected_weeks
    ):
        raise ValueError("Planning weeks must be unique and strictly increasing")
    missing_weeks = sorted(set(selected_weeks).difference(available_weeks))
    if missing_weeks:
        raise ValueError(f"Survivor predictions are missing weeks: {missing_weeks}")
    if any(right != left + 1 for left, right in pairwise(selected_weeks)):
        raise ValueError("Planning weeks must be consecutive")
    frame = frame.loc[frame["week"].isin(selected_weeks)].copy()

    probability = pd.to_numeric(frame["home_win_probability"], errors="coerce")
    if probability.isna().any() or not np.isfinite(probability).all():
        raise ValueError("Survivor win probabilities must be finite numbers")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("Survivor win probabilities must lie in [0, 1]")
    frame["home_win_probability"] = probability

    by_week: dict[int, tuple[_Candidate, ...]] = {}
    for week, group in frame.groupby("week", sort=True):
        week_number = int(cast(Any, week))
        candidates: list[_Candidate] = []
        seen: set[str] = set()
        for row in group.sort_values("game_id").itertuples(index=False):
            home = _team(row.home_team, context=f"week {week_number}")
            away = _team(row.away_team, context=f"week {week_number}")
            if home == away:
                raise ValueError(
                    f"Week {week_number} game {row.game_id!r} has the same team on both sides"
                )
            duplicate = sorted({home, away}.intersection(seen))
            if duplicate:
                raise ValueError(f"Week {week_number} schedules team {duplicate[0]} more than once")
            seen.update((home, away))
            home_probability = float(cast(Any, row.home_win_probability))
            candidates.extend(
                (
                    _Candidate(
                        week=week_number,
                        team=home,
                        opponent=away,
                        game_id=str(row.game_id),
                        gameday=row.gameday,
                        side="HOME",
                        probability=home_probability,
                    ),
                    _Candidate(
                        week=week_number,
                        team=away,
                        opponent=home,
                        game_id=str(row.game_id),
                        gameday=row.gameday,
                        side="AWAY",
                        probability=1.0 - home_probability,
                    ),
                )
            )
        by_week[week_number] = tuple(sorted(candidates, key=lambda item: item.team))
    return season, selected_weeks, by_week


def build_survivor_plan(
    predictions: pd.DataFrame,
    *,
    method: str = "straight_up",
    weeks: Sequence[int] | None = None,
    used_teams: Iterable[str] = (),
    locked_picks: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Plan one non-reusable winner per week for maximum joint survival.

    ``predictions`` may contain several outcome methods; only ``method`` is
    used.  ``used_teams`` represents picks consumed before this horizon.
    ``locked_picks`` preserves already-submitted choices inside the horizon.

    The audit columns separate immediate win probability from future team
    opportunity cost.  The latter is the reduction in optimal future survival
    caused by removing the chosen team, before accounting for this week's win.
    """

    season, selected_weeks, by_week = _validate_and_expand(predictions, method=method, weeks=weeks)
    normalized_used = tuple(_team(team, context="used_teams") for team in used_teams)
    if len(set(normalized_used)) != len(normalized_used):
        raise ValueError("used_teams contains a duplicate team")
    locked = {
        int(week): _team(team, context=f"locked pick for week {week}")
        for week, team in (locked_picks or {}).items()
    }
    unknown_locked_weeks = sorted(set(locked).difference(selected_weeks))
    if unknown_locked_weeks:
        raise ValueError(f"Locked picks fall outside the planning weeks: {unknown_locked_weeks}")
    all_consumed = [*normalized_used, *locked.values()]
    if len(set(all_consumed)) != len(all_consumed):
        raise ValueError("A survivor team cannot be reused across used teams and locked picks")

    unavailable = set(normalized_used)
    options: list[tuple[_Candidate, ...]] = []
    for week in selected_weeks:
        week_options = tuple(
            candidate for candidate in by_week[week] if candidate.team not in unavailable
        )
        if week in locked:
            week_options = tuple(
                candidate for candidate in week_options if candidate.team == locked[week]
            )
            if not week_options:
                raise ValueError(f"Locked team {locked[week]} is not available in week {week}")
        if not week_options:
            raise ValueError(f"No available survivor teams remain in week {week}")
        options.append(week_options)

    assignment, horizon_log_probability = _hungarian_assignment(options)
    horizon_probability = _probability(horizon_log_probability)
    rows: list[dict[str, object]] = []
    consumed = set(normalized_used)
    cumulative_probability = 1.0
    for index, chosen in enumerate(assignment):
        week_options = tuple(
            candidate for candidate in options[index] if candidate.team not in consumed
        )
        future_options = [
            tuple(candidate for candidate in later if candidate.team not in consumed)
            for later in options[index + 1 :]
        ]
        _, future_without_log = _hungarian_assignment(future_options)
        future_after_options = [
            tuple(candidate for candidate in later if candidate.team != chosen.team)
            for later in future_options
        ]
        _, future_after_log = _hungarian_assignment(future_after_options)
        future_without = _probability(future_without_log)
        future_after = _probability(future_after_log)
        cumulative_probability *= chosen.probability
        from_week = chosen.probability * future_after
        rows.append(
            {
                "season": season,
                "week": chosen.week,
                "gameday": chosen.gameday,
                "team": chosen.team,
                "opponent": chosen.opponent,
                "side": chosen.side,
                "game_id": chosen.game_id,
                "method": method,
                "is_locked": chosen.week in locked,
                "pick_probability": chosen.probability,
                "best_current_probability": max(item.probability for item in week_options),
                "current_probability_sacrifice": max(item.probability for item in week_options)
                - chosen.probability,
                "future_survival_probability": future_after,
                "future_team_opportunity_cost": future_without - future_after,
                "survival_probability_from_week": from_week,
                "cumulative_survival_probability": cumulative_probability,
                "horizon_survival_probability": horizon_probability,
                "used_teams_before": ",".join(sorted(consumed)),
            }
        )
        consumed.add(chosen.team)
    return pd.DataFrame(rows)


def survivor_plan_markdown(plan: pd.DataFrame) -> str:
    """Render the auditable plan without implying probabilities are outcomes."""

    if plan.empty:
        raise ValueError("Survivor plan is empty")
    display = plan.loc[
        :,
        [
            "week",
            "team",
            "opponent",
            "pick_probability",
            "current_probability_sacrifice",
            "future_team_opportunity_cost",
            "survival_probability_from_week",
            "is_locked",
        ],
    ].copy()
    for column in (
        "pick_probability",
        "current_probability_sacrifice",
        "future_team_opportunity_cost",
        "survival_probability_from_week",
    ):
        display[column] = display[column].map(lambda value: f"{float(value):.2%}")
    season = int(plan["season"].iloc[0])
    method = str(plan["method"].iloc[0])
    horizon = float(plan["horizon_survival_probability"].iloc[0])
    return (
        f"# Survivor plan: {season}\n\n"
        f"Straight-up method: `{method}`. Optimized horizon survival: {horizon:.2%}. "
        "Probabilities are model estimates, not guarantees. Each team is used at most once.\n\n"
        + display.to_markdown(index=False)
        + "\n"
    )
