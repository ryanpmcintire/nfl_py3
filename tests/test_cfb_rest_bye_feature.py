"""Leakage regression + known-answer tests for the CFB rest/bye replication
(``docs/cfb_rest_bye_replication.md``), per AGENTS.md's "add a leakage
regression test for every new feature family" rule.

What is proved here:

* **Leakage.** Every candidate column is a pure function of pregame schedule
  facts (season, dates, team ids, completion, season type) and is bit-identical
  after ``result`` / ``ats_margin`` / ``home_points`` / ``away_points`` are
  shuffled. Not a smell test -- the shuffle is applied to the frame the feature
  builder is handed.
* **Known answers, one hand-computed fixture covering every cell**, including a
  season opener (undefined rest), a true open date, a both-sides-off-bye game
  that ``bye_edge_home`` must NOT flag, a one-side-undefined game, and a game
  whose correct rest value is only reachable from the FULL schedule (a
  schedule-only opponent the benchmark table never carries).
* **The first-game rule**: a team's first game of a season has no defined rest,
  so every cell needing that side is NaN -- never 0, never "not off bye".
* **The derivation reproduces the frozen benchmark ``rest_diff``** wherever both
  are defined, on the real table (skipped when the gitignored local snapshots
  are absent, as in a fresh clone).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_rest_bye_feature import (
    CFB_AWAY_OFF_BYE_COLUMN,
    CFB_AWAY_OFF_BYE_GAP12_COLUMN,
    CFB_BYE_EDGE_HOME_COLUMN,
    CFB_HOME_OFF_BYE_COLUMN,
    CFB_HOME_OFF_BYE_GAP12_COLUMN,
    CFB_REST_BYE_FEATURE_COLUMNS,
    CFB_SHORT_WEEK_ROAD_COLUMN,
    CFB_SHORT_WEEK_ROAD_LE6_COLUMN,
    CFB_SIDE_REST_COLUMNS,
    attach_cfb_rest_bye_features,
    build_cfb_rest_team_panel,
    derive_cfb_rest_bye_features,
    derive_side_rest,
)
from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
CFB_DATA_ROOT = REPO_ROOT / "data" / "cfb"

# ---------------------------------------------------------------------------
# The hand-built fixture. Ten benchmark games plus ONE schedule-only game.
#
# Season 2024. Per-team appearance dates, and therefore the hand-computed rest:
#   T1: 09-07, 09-21, 10-05, 10-11         -> NaN, 14, 14, 6
#   T2: 09-07, 09-21*, 09-28, 10-10        -> NaN, 14,  7, 12   (* schedule-only)
#   T3: 09-14, 09-21, 10-05, 10-11         -> NaN,  7, 14, 6
#   T4: 09-14, 09-28, 10-05, 10-10         -> NaN, 14,  7, 5
#   T5: 10-05                              -> NaN
#   T6: 09-14, 09-28                       -> NaN, 14
#   T7: 09-14, 09-28                       -> NaN, 14
#   T99: 09-21 (schedule-only opponent)    -> NaN
# ---------------------------------------------------------------------------

_SCHEDULE_ROWS = [
    # (date, home_id, away_id, in_benchmark, week)
    ("2024-09-07", 1, 2, True, 1),
    ("2024-09-14", 3, 4, True, 2),
    ("2024-09-14", 6, 7, True, 2),
    ("2024-09-21", 1, 3, True, 3),
    ("2024-09-21", 2, 99, False, 3),
    ("2024-09-28", 2, 4, True, 4),
    ("2024-09-28", 6, 7, True, 4),
    ("2024-10-05", 1, 4, True, 5),
    ("2024-10-05", 3, 5, True, 5),
    ("2024-10-10", 2, 4, True, 6),
    ("2024-10-11", 3, 1, True, 6),
]

#: Hand-computed expected values, keyed by ``game_id``. ``None`` means NaN.
_EXPECTED: dict[int, dict[str, float | None]] = {
    # season openers on BOTH sides -- every cell undefined
    1: {"home_rest": None, "away_rest": None},
    2: {"home_rest": None, "away_rest": None},
    3: {"home_rest": None, "away_rest": None},
    # T1 off a true open date (14) vs T3 on a normal week (7)
    4: {"home_rest": 14.0, "away_rest": 7.0},
    # T2's 7 days is only reachable from the FULL schedule (its 09-21 game is
    # not in the benchmark table); the benchmark subset alone would say 21.
    6: {"home_rest": 7.0, "away_rest": 14.0},
    # BOTH sides off a true open date -- bye_edge_home must NOT fire
    7: {"home_rest": 14.0, "away_rest": 14.0},
    8: {"home_rest": 14.0, "away_rest": 7.0},
    # away side is playing its FIRST game of the season -- away cells undefined
    9: {"home_rest": 14.0, "away_rest": None},
    # home on the 12-day shoulder (not >=13), away on a 5-day short week
    10: {"home_rest": 12.0, "away_rest": 5.0},
    # both sides on a 6-day short week (the <=6 sensitivity arm, not <=5)
    11: {"home_rest": 6.0, "away_rest": 6.0},
}


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * len(_SCHEDULE_ROWS),
            "season_type": ["regular"] * len(_SCHEDULE_ROWS),
            "completed": [True] * len(_SCHEDULE_ROWS),
            "start_date": [f"{row[0]}T18:00:00.000Z" for row in _SCHEDULE_ROWS],
            "home_id": [row[1] for row in _SCHEDULE_ROWS],
            "away_id": [row[2] for row in _SCHEDULE_ROWS],
        }
    )


def _games() -> pd.DataFrame:
    """The benchmark-table subset: the schedule minus the schedule-only game."""

    rows = [(index + 1, row) for index, row in enumerate(_SCHEDULE_ROWS) if row[3]]
    return pd.DataFrame(
        {
            "game_id": [game_id for game_id, _ in rows],
            "season": [2024] * len(rows),
            "week": [row[4] for _, row in rows],
            "gameday": pd.to_datetime([row[0] for _, row in rows]),
            "home_id": [row[1] for _, row in rows],
            "away_id": [row[2] for _, row in rows],
            # Outcome columns exist so the leakage test can shuffle them.
            "result": np.arange(len(rows), dtype=float) - 3.0,
            "ats_margin": np.arange(len(rows), dtype=float) + 1.5,
            "home_points": np.arange(len(rows), dtype=float) + 20.0,
            "away_points": np.arange(len(rows), dtype=float) + 17.0,
        }
    )


def _derived() -> pd.DataFrame:
    derived, _ = derive_cfb_rest_bye_features(_games(), schedules=_schedules())
    return derived.set_index("game_id")


def _expected_cells(home: float | None, away: float | None) -> dict[str, float | None]:
    """The frozen cell definitions, restated independently of the module."""

    home_known = home is not None
    away_known = away is not None
    both = home_known and away_known
    return {
        CFB_HOME_OFF_BYE_COLUMN: float(home >= 13) if home_known else None,
        CFB_AWAY_OFF_BYE_COLUMN: float(away >= 13) if away_known else None,
        CFB_BYE_EDGE_HOME_COLUMN: (
            float(home >= 12 and away < 12) if both else None  # type: ignore[operator]
        ),
        CFB_SHORT_WEEK_ROAD_COLUMN: float(away <= 5) if away_known else None,
        CFB_HOME_OFF_BYE_GAP12_COLUMN: float(home >= 12) if home_known else None,
        CFB_AWAY_OFF_BYE_GAP12_COLUMN: float(away >= 12) if away_known else None,
        CFB_SHORT_WEEK_ROAD_LE6_COLUMN: float(away <= 6) if away_known else None,
    }


# ---------------------------------------------------------------------------
# Known answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game_id", sorted(_EXPECTED))
def test_known_answer_per_side_rest(game_id: int) -> None:
    derived = _derived()
    expected = _EXPECTED[game_id]
    for side, column in zip(("home_rest", "away_rest"), CFB_SIDE_REST_COLUMNS, strict=True):
        actual = derived.loc[game_id, column]
        if expected[side] is None:
            assert pd.isna(actual), f"game {game_id} {column} should be NaN, got {actual}"
        else:
            assert actual == pytest.approx(expected[side]), f"game {game_id} {column}"


@pytest.mark.parametrize("game_id", sorted(_EXPECTED))
def test_known_answer_every_cell(game_id: int) -> None:
    derived = _derived()
    expected = _expected_cells(_EXPECTED[game_id]["home_rest"], _EXPECTED[game_id]["away_rest"])
    for column, value in expected.items():
        actual = derived.loc[game_id, column]
        if value is None:
            assert pd.isna(actual), f"game {game_id} {column} should be NaN, got {actual}"
        else:
            assert actual == pytest.approx(value), f"game {game_id} {column}"


def test_both_sides_off_bye_is_not_a_bye_edge() -> None:
    """Game 7: T6 and T7 both come off a true 14-day open date.

    ``bye_edge_home`` is "home off strict bye AND opponent NOT off bye", so it
    must read 0 here even though BOTH off-bye cells read 1. This is the cell's
    whole point -- the NFL construct isolates the side holding the rest EDGE.
    """

    row = _derived().loc[7]
    assert row[CFB_HOME_OFF_BYE_COLUMN] == 1.0
    assert row[CFB_AWAY_OFF_BYE_COLUMN] == 1.0
    assert row[CFB_BYE_EDGE_HOME_COLUMN] == 0.0


def test_rest_uses_the_full_schedule_not_the_benchmark_subset() -> None:
    """Game 6's home team played a schedule-only game the benchmark never carries.

    Computed from the benchmark subset alone T2 would show 21 days (a bye) and
    ``home_off_bye`` would misfire; from the full schedule it is 7 days.
    """

    row = _derived().loc[6]
    assert row["cfb_home_rest_days"] == 7.0
    assert row[CFB_HOME_OFF_BYE_COLUMN] == 0.0
    assert row[CFB_BYE_EDGE_HOME_COLUMN] == 0.0


def test_the_thirteen_day_threshold_excludes_the_twelve_day_shoulder() -> None:
    """Game 10: home rest is exactly 12 -- inside the strict-bye gap, outside
    the NFL off-bye threshold of 13. The primary cell and its declared
    sensitivity arm must disagree here, which is the entire reason the
    sensitivity arm is declared."""

    row = _derived().loc[10]
    assert row["cfb_home_rest_days"] == 12.0
    assert row[CFB_HOME_OFF_BYE_COLUMN] == 0.0
    assert row[CFB_HOME_OFF_BYE_GAP12_COLUMN] == 1.0
    assert row[CFB_BYE_EDGE_HOME_COLUMN] == 1.0


def test_short_week_thresholds_separate_five_from_six() -> None:
    five = _derived().loc[10]
    six = _derived().loc[11]
    assert five["cfb_away_rest_days"] == 5.0
    assert five[CFB_SHORT_WEEK_ROAD_COLUMN] == 1.0
    assert five[CFB_SHORT_WEEK_ROAD_LE6_COLUMN] == 1.0
    assert six["cfb_away_rest_days"] == 6.0
    assert six[CFB_SHORT_WEEK_ROAD_COLUMN] == 0.0
    assert six[CFB_SHORT_WEEK_ROAD_LE6_COLUMN] == 1.0


# ---------------------------------------------------------------------------
# The first-game rule
# ---------------------------------------------------------------------------


def test_first_game_of_a_season_has_no_defined_rest_and_is_nan_not_zero() -> None:
    """Games 1-3 are both teams' season openers; game 9's AWAY side is one.

    Every affected cell is NaN, never 0. 0 would assert "not off bye", which is
    a claim the schedule cannot support for a team with no previous game.
    """

    derived = _derived()
    for game_id in (1, 2, 3):
        for column in CFB_REST_BYE_FEATURE_COLUMNS:
            assert pd.isna(derived.loc[game_id, column]), f"game {game_id} {column}"
    row = derived.loc[9]
    assert row[CFB_HOME_OFF_BYE_COLUMN] == 1.0
    assert pd.isna(row[CFB_AWAY_OFF_BYE_COLUMN])
    assert pd.isna(row[CFB_BYE_EDGE_HOME_COLUMN])
    assert pd.isna(row[CFB_SHORT_WEEK_ROAD_COLUMN])


def test_first_game_rows_are_counted_in_the_diagnostics_not_dropped() -> None:
    _, diagnostics = derive_cfb_rest_bye_features(_games(), schedules=_schedules())
    assert diagnostics["n_games"] == 10
    assert diagnostics["n_home_rest_missing"] == 3
    assert diagnostics["n_away_rest_missing"] == 4
    assert diagnostics["n_either_rest_missing"] == 4
    assert diagnostics["missing_by_column"][CFB_HOME_OFF_BYE_COLUMN] == 3
    assert diagnostics["missing_by_column"][CFB_BYE_EDGE_HOME_COLUMN] == 4


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome_column", ["result", "ats_margin", "home_points", "away_points"])
def test_every_cell_is_invariant_to_shuffling_an_outcome_column(outcome_column: str) -> None:
    """The leakage regression AGENTS.md requires for every new feature family.

    If any candidate column read an outcome -- directly, or through a join that
    happened to be ordered by one -- permuting that outcome would move it.
    """

    schedules = _schedules()
    baseline, _ = derive_cfb_rest_bye_features(_games(), schedules=schedules)
    shuffled_games = _games()
    rng = np.random.default_rng(20260901)
    shuffled_games[outcome_column] = rng.permutation(shuffled_games[outcome_column].to_numpy())
    shuffled, _ = derive_cfb_rest_bye_features(shuffled_games, schedules=schedules)
    pd.testing.assert_frame_equal(baseline, shuffled)


def test_cells_are_invariant_to_shuffling_every_outcome_column_at_once() -> None:
    schedules = _schedules()
    baseline, _ = derive_cfb_rest_bye_features(_games(), schedules=schedules)
    games = _games()
    rng = np.random.default_rng(7)
    for column in ("result", "ats_margin", "home_points", "away_points"):
        games[column] = rng.permutation(games[column].to_numpy())
    shuffled, _ = derive_cfb_rest_bye_features(games, schedules=schedules)
    pd.testing.assert_frame_equal(baseline, shuffled)


def test_derivation_reads_only_pregame_schedule_columns() -> None:
    """Dropping every outcome column from BOTH inputs changes nothing.

    The strongest form of the pregame guarantee: the builder cannot depend on a
    column that is not there.
    """

    baseline, _ = derive_cfb_rest_bye_features(_games(), schedules=_schedules())
    stripped = _games().drop(columns=["result", "ats_margin", "home_points", "away_points"])
    without_outcomes, _ = derive_cfb_rest_bye_features(stripped, schedules=_schedules())
    pd.testing.assert_frame_equal(baseline, without_outcomes)


def test_a_future_game_never_changes_an_earlier_games_rest() -> None:
    """Point-in-time safety in the time direction: rest looks backward only."""

    baseline = _derived()
    extended = pd.concat(
        [
            _schedules(),
            pd.DataFrame(
                {
                    "season": [2024],
                    "season_type": ["regular"],
                    "completed": [True],
                    "start_date": ["2024-11-30T18:00:00.000Z"],
                    "home_id": [1],
                    "away_id": [3],
                }
            ),
        ],
        ignore_index=True,
    )
    with_future, _ = derive_cfb_rest_bye_features(_games(), schedules=extended)
    pd.testing.assert_frame_equal(baseline.reset_index(), with_future)


# ---------------------------------------------------------------------------
# Contract / plumbing
# ---------------------------------------------------------------------------


def test_attach_is_purely_additive() -> None:
    games = _games()
    merged, _ = attach_cfb_rest_bye_features(games, schedules=_schedules())
    pd.testing.assert_frame_equal(merged.loc[:, games.columns], games)
    assert set(merged.columns) - set(games.columns) == {
        *CFB_SIDE_REST_COLUMNS,
        *CFB_REST_BYE_FEATURE_COLUMNS,
    }


def test_attach_refuses_a_column_collision() -> None:
    games = _games()
    games[CFB_HOME_OFF_BYE_COLUMN] = 0.0
    with pytest.raises(DataContractError, match=CFB_HOME_OFF_BYE_COLUMN):
        attach_cfb_rest_bye_features(games, schedules=_schedules())


def test_missing_required_column_is_refused() -> None:
    with pytest.raises(DataContractError, match="home_id"):
        derive_side_rest(_games().drop(columns=["home_id"]), _schedules())


def test_team_panel_stacks_both_sides_and_marks_undefined_rest() -> None:
    rested = derive_side_rest(_games(), _schedules())
    panel = build_cfb_rest_team_panel(rested)
    assert len(panel) == 2 * len(rested)
    # T5 appears once, in its own season opener -- every propensity undefined.
    t5 = panel.loc[panel["team_id"].eq(5)]
    assert len(t5) == 1
    assert bool(t5["own_rest_days"].isna().all())
    assert bool(t5["own_off_bye_13"].isna().all())
    # T4 in game 10 arrives on 5 days rest against a 12-day-rested opponent.
    t4 = panel.loc[panel["team_id"].eq(4) & panel["week"].eq(6) & ~panel["is_home"]]
    assert t4["own_rest_days"].iloc[0] == 5.0
    assert t4["own_short_week_5"].iloc[0] == 1.0
    assert t4["own_strict_bye_edge"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# The known-answer check against the real frozen table
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not FEATURES_PATH.is_file() or not (CFB_DATA_ROOT / "schedules" / "raw").is_dir(),
    reason="local CFB snapshots are gitignored and absent in a fresh clone",
)
def test_derived_side_rest_reproduces_the_frozen_rest_diff() -> None:
    """``home_rest - away_rest`` must equal the benchmark's own ``rest_diff``
    on every row where both are defined, with an identical missingness pattern.

    This is the known-answer test for the derivation as a whole: it pins the
    schedules source, the season range, the regular-season/completed filters and
    the within-(team, season) grouping against a column this session did not
    build.
    """

    features = pd.read_parquet(FEATURES_PATH)
    _, diagnostics = derive_cfb_rest_bye_features(features)
    reconstruction = diagnostics["rest_diff_reconstruction"]
    assert reconstruction["frozen_rest_diff_column_present"] is True
    assert reconstruction["n_both_defined"] > 10_000
    assert reconstruction["n_exact_match"] == reconstruction["n_both_defined"]
    assert reconstruction["n_missingness_pattern_mismatch"] == 0
    assert reconstruction["max_abs_difference"] == 0.0
