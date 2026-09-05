"""Construction, sign-convention, restriction and leakage tests for the CFB
free-screen wave 1 (LEAD-48, LEAD-50, LEAD-46; ``docs/cfb_lead_screens_wave1.md``).

Per AGENTS.md's "add a leakage regression test for every new feature family"
rule: every candidate column here is proved to be a pure function of pregame
schedule/identity facts by shuffling the outcome columns (``result``,
``ats_margin``, ``home_cover``, ``home_points``, ``away_points``) and
asserting the flag is bit-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cfb_lead_screens_wave1 import (
    ALTITUDE_HOME_TEAMS,
    CANDIDATE_COLUMNS,
    attach_altitude_cold_home_flag,
    attach_post_bye_flag,
    attach_rivalry_home_dog_flag,
    compute_rivalry_pairs,
)

POST_BYE_COLUMN = CANDIDATE_COLUMNS["post_bye"]
RIVALRY_COLUMN = CANDIDATE_COLUMNS["rivalry_home_dog"]
ALTITUDE_COLUMN = CANDIDATE_COLUMNS["altitude_cold"]

_OUTCOME_COLUMNS = ("result", "ats_margin", "home_cover", "home_points", "away_points")


def _with_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    n = len(frame)
    frame["result"] = np.arange(n, dtype=float) - 3.0
    frame["ats_margin"] = np.arange(n, dtype=float) + 1.5
    frame["home_cover"] = np.where(np.arange(n) % 2 == 0, 1.0, 0.0)
    frame["home_points"] = np.arange(n, dtype=float) + 20.0
    frame["away_points"] = np.arange(n, dtype=float) + 17.0
    return frame


def _shuffle_outcomes(frame: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """A different, internally-consistent-looking outcome world, same games."""

    frame = frame.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    for column in _OUTCOME_COLUMNS:
        frame[column] = frame[column].to_numpy()[order]
    return frame


# ---------------------------------------------------------------------------
# LEAD-48: post_bye
# ---------------------------------------------------------------------------

#: (start_date, home_id, away_id) for the FULL local schedule (rest is derived
#: from this, never from the benchmark subset alone).
_POST_BYE_SCHEDULE_ROWS = [
    ("2024-09-07", 1, 2),  # both first appearances
    ("2024-09-14", 2, 3),  # T2 2nd (rest 7 from 09-07); T3 1st
    ("2024-09-21", 1, 3),  # game A: T1 2nd (rest 14, off bye); T3 2nd (rest 7)
    ("2024-09-21", 4, 5),  # game B: T4 1st; T5 1st -- both undefined
    ("2024-09-28", 1, 2),  # game C: T1 3rd (rest 7); T2 3rd (rest 14, off bye)
    ("2024-09-28", 4, 3),  # game D: T4 2nd (rest 7); T3 3rd (rest 7) -- neither
    ("2024-09-14", 6, 7),  # T6, T7 first appearances
    ("2024-09-28", 6, 7),  # game E: both 2nd (rest 14 each) -- BOTH off bye
]


def _post_bye_schedules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * len(_POST_BYE_SCHEDULE_ROWS),
            "season_type": ["regular"] * len(_POST_BYE_SCHEDULE_ROWS),
            "completed": [True] * len(_POST_BYE_SCHEDULE_ROWS),
            "start_date": [f"{row[0]}T18:00:00.000Z" for row in _POST_BYE_SCHEDULE_ROWS],
            "home_id": [row[1] for row in _POST_BYE_SCHEDULE_ROWS],
            "away_id": [row[2] for row in _POST_BYE_SCHEDULE_ROWS],
        }
    )


def _post_bye_games() -> pd.DataFrame:
    # Benchmark rows are the five lettered games (A-E, schedule indices
    # 2, 3, 4, 5, 7 above), numbered sequentially 101-105 for readability.
    benchmark_indices = [2, 3, 4, 5, 7]
    rows = [
        (101 + position, _POST_BYE_SCHEDULE_ROWS[index])
        for position, index in enumerate(benchmark_indices)
    ]
    frame = pd.DataFrame(
        {
            "game_id": [game_id for game_id, _ in rows],
            "season": [2024] * len(rows),
            "week": [3, 3, 4, 4, 4],
            "gameday": pd.to_datetime([row[0] for _, row in rows]),
            "home_id": [row[1] for _, row in rows],
            "away_id": [row[2] for _, row in rows],
        }
    )
    return _with_outcome_columns(frame)


def test_post_bye_signed_column_matches_hand_computed_cases() -> None:
    features = _post_bye_games()
    schedules = _post_bye_schedules()
    attached = attach_post_bye_flag(features, schedules=schedules).set_index("game_id")

    assert attached.loc[101, POST_BYE_COLUMN] == 1.0  # home (T1) off bye, away (T3) not
    assert np.isnan(attached.loc[102, POST_BYE_COLUMN])  # both sides' first game
    assert attached.loc[103, POST_BYE_COLUMN] == -1.0  # away (T2) off bye, home (T1) not
    assert attached.loc[104, POST_BYE_COLUMN] == 0.0  # neither side off bye
    assert attached.loc[105, POST_BYE_COLUMN] == 0.0  # BOTH sides off bye -> 0, not NaN


def test_post_bye_column_never_takes_a_third_value() -> None:
    features = _post_bye_games()
    values = attach_post_bye_flag(features, schedules=_post_bye_schedules())[POST_BYE_COLUMN]
    allowed = {-1.0, 0.0, 1.0}
    assert set(values.dropna().unique()).issubset(allowed)


def test_post_bye_flag_is_pregame_safe_under_outcome_permutation() -> None:
    features = _post_bye_games()
    schedules = _post_bye_schedules()
    before = attach_post_bye_flag(features, schedules=schedules)[POST_BYE_COLUMN].to_numpy()
    shuffled = _shuffle_outcomes(features)
    after = attach_post_bye_flag(shuffled, schedules=schedules)[POST_BYE_COLUMN].to_numpy()
    np.testing.assert_array_equal(before, after)


def test_post_bye_flag_unaffected_by_dropping_outcome_columns() -> None:
    features = _post_bye_games()
    schedules = _post_bye_schedules()
    with_outcomes = attach_post_bye_flag(features, schedules=schedules)[POST_BYE_COLUMN]
    stripped = features.drop(columns=list(_OUTCOME_COLUMNS))
    without_outcomes = attach_post_bye_flag(stripped, schedules=schedules)[POST_BYE_COLUMN]
    np.testing.assert_array_equal(with_outcomes.to_numpy(), without_outcomes.to_numpy())


# ---------------------------------------------------------------------------
# LEAD-50: rivalry_home_dog
# ---------------------------------------------------------------------------


def _rivalry_schedules() -> pd.DataFrame:
    """Alpha-Beta meet in 8 straight seasons (a rivalry); Gamma-Delta meet in
    only 3 (not a rivalry, below the 8-consecutive-season bar)."""

    rows = []
    for season in range(2010, 2018):  # 2010..2017 inclusive: 8 consecutive seasons
        rows.append((season, "Alpha", "Beta"))
    for season in (2015, 2016, 2018):  # not consecutive/enough
        rows.append((season, "Gamma", "Delta"))
    return pd.DataFrame(
        {
            "season": [row[0] for row in rows],
            "season_type": ["regular"] * len(rows),
            "completed": [True] * len(rows),
            "home_team": [row[1] for row in rows],
            "away_team": [row[2] for row in rows],
        }
    )


def _rivalry_games() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "game_id": [201, 202, 203, 204],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "gameday": pd.to_datetime(["2024-09-07", "2024-09-14", "2024-09-21", "2024-09-28"]),
            "home_team": ["Alpha", "Alpha", "Gamma", "Beta"],
            "away_team": ["Beta", "Beta", "Delta", "Alpha"],
            # rivalry & home underdog -> 1
            # rivalry & home FAVORED -> 0 (restriction test)
            # home underdog but NOT a rivalry pair -> 0 (restriction test)
            # rivalry pair but a pick'em (spread_line == 0) -> 0, not underdog
            "spread_line": [-3.0, 3.0, -5.0, 0.0],
        }
    )
    return _with_outcome_columns(frame)


def test_compute_rivalry_pairs_finds_the_long_pairing_only() -> None:
    pairs = compute_rivalry_pairs(_rivalry_schedules(), min_consecutive_seasons=8)
    assert frozenset({"Alpha", "Beta"}) in pairs
    assert frozenset({"Gamma", "Delta"}) not in pairs


def test_rivalry_home_dog_requires_both_rivalry_and_underdog() -> None:
    attached = attach_rivalry_home_dog_flag(
        _rivalry_games(), schedules=_rivalry_schedules()
    ).set_index("game_id")

    assert attached.loc[201, RIVALRY_COLUMN] == 1.0  # rivalry + home underdog
    assert attached.loc[202, RIVALRY_COLUMN] == 0.0  # rivalry but home favored
    assert attached.loc[203, RIVALRY_COLUMN] == 0.0  # home underdog but not a rivalry pair
    assert attached.loc[204, RIVALRY_COLUMN] == 0.0  # rivalry pair but a pick'em, not underdog


def test_rivalry_home_dog_flag_never_one_when_home_is_not_underdog() -> None:
    attached = attach_rivalry_home_dog_flag(_rivalry_games(), schedules=_rivalry_schedules())
    not_underdog = attached["spread_line"].ge(0.0)
    assert (attached.loc[not_underdog, RIVALRY_COLUMN] == 0.0).all()


def test_rivalry_home_dog_flag_is_pregame_safe_under_outcome_permutation() -> None:
    games = _rivalry_games()
    schedules = _rivalry_schedules()
    before = attach_rivalry_home_dog_flag(games, schedules=schedules)[RIVALRY_COLUMN].to_numpy()
    shuffled = _shuffle_outcomes(games)
    after = attach_rivalry_home_dog_flag(shuffled, schedules=schedules)[RIVALRY_COLUMN].to_numpy()
    np.testing.assert_array_equal(before, after)


# ---------------------------------------------------------------------------
# LEAD-46: altitude_cold
# ---------------------------------------------------------------------------


def _altitude_games() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "game_id": [301, 302, 303, 304, 305],
            "season": [2024] * 5,
            "week": [1, 6, 10, 14, 10],
            "gameday": pd.to_datetime(
                [
                    "2024-09-15",  # Colorado State home, BEFORE October -> 0
                    "2024-10-05",  # Colorado State home, October -> 1
                    "2024-11-01",  # Ohio State home (not altitude), October -> 0
                    "2024-12-01",  # Wyoming home, December -> 1
                    "2024-11-01",  # Ohio State home, Air Force AWAY -> 0 (home-only)
                ]
            ),
            "home_team": [
                "Colorado State",
                "Colorado State",
                "Ohio State",
                "Wyoming",
                "Ohio State",
            ],
            "away_team": ["Ohio State", "Ohio State", "Utah", "Ohio State", "Air Force"],
        }
    )
    return _with_outcome_columns(frame)


def test_altitude_teams_frozen_list_matches_roadmap() -> None:
    assert frozenset({"Colorado State", "Wyoming", "Air Force", "Utah"}) == ALTITUDE_HOME_TEAMS


def test_altitude_cold_home_requires_home_team_and_october_on() -> None:
    attached = attach_altitude_cold_home_flag(_altitude_games()).set_index("game_id")

    assert attached.loc[301, ALTITUDE_COLUMN] == 0.0  # altitude home, before October
    assert attached.loc[302, ALTITUDE_COLUMN] == 1.0  # altitude home, October
    assert attached.loc[303, ALTITUDE_COLUMN] == 0.0  # October, but not an altitude team
    assert attached.loc[304, ALTITUDE_COLUMN] == 1.0  # altitude home, December
    assert attached.loc[305, ALTITUDE_COLUMN] == 0.0  # altitude team is AWAY, not home


def test_altitude_cold_home_flag_is_pregame_safe_under_outcome_permutation() -> None:
    games = _altitude_games()
    before = attach_altitude_cold_home_flag(games)[ALTITUDE_COLUMN].to_numpy()
    after = attach_altitude_cold_home_flag(_shuffle_outcomes(games))[ALTITUDE_COLUMN].to_numpy()
    np.testing.assert_array_equal(before, after)


def test_altitude_cold_home_flag_never_nan() -> None:
    values = attach_altitude_cold_home_flag(_altitude_games())[ALTITUDE_COLUMN]
    assert values.notna().all()
    assert set(values.unique()).issubset({0.0, 1.0})
