"""Leakage and join-correctness proofs for the CFB graph ``team_stat`` feature.

Predeclared in ``docs/graph_team_stat_cfb_replication.md`` section 7. These are
release-blocking: a pregame feature that can see a future week's statistic is a
leak, and a feature joined back by position rather than by ``game_id`` silently
attaches one game's rating to another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.data import DataContractError
from nfl_ats.graph_team_stat_cfb_feature import (
    CFB_GRAPH_CELLS,
    CFB_GRAPH_FROZEN_STRUCTURE,
    add_cfb_graph_team_stat_feature,
    cfb_cell_columns,
    cfb_graph_column,
)

CELL = "def_epa_per_play"
TEAM_COUNT = 16
WEEK_COUNT = 6


def _cfb_like_games(
    *,
    cell: str = CELL,
    team_count: int = TEAM_COUNT,
    weeks: int = WEEK_COUNT,
) -> pd.DataFrame:
    """A synthetic CFB-shaped schedule: ids, rotating pairings, pregame stats.

    ``team_count`` teams play ``team_count // 2`` games a week for ``weeks``
    weeks, with the pairing rotated each week so the graph is connected rather
    than a set of isolated pairs. Team ids are ESPN-like integers and each team
    carries a fixed pregame statistic, so the edge weight of every game is
    deterministic and a rating change can only come from the graph.
    """

    home_column, away_column = cfb_cell_columns(cell)
    ids = [2000 + index for index in range(team_count)]
    statistic = {team_id: 0.05 + 0.01 * position for position, team_id in enumerate(ids)}

    rows: list[dict[str, object]] = []
    game_id = 500_000
    for week in range(1, weeks + 1):
        rotated = ids[week % team_count :] + ids[: week % team_count]
        for slot in range(team_count // 2):
            home_id = rotated[slot]
            away_id = rotated[team_count - 1 - slot]
            game_id += 1
            rows.append(
                {
                    "game_id": game_id,
                    "season": 2015,
                    "week": week,
                    "gameday": pd.Timestamp("2015-09-05") + pd.Timedelta(weeks=week - 1),
                    "home_id": home_id,
                    "away_id": away_id,
                    "home_team": f"Team {home_id}",
                    "away_team": f"Team {away_id}",
                    "result": float((home_id - away_id) % 7) - 3.0,
                    "spread_line": 1.5,
                    home_column: statistic[home_id],
                    away_column: statistic[away_id],
                }
            )
    return pd.DataFrame(rows)


def _rated_weeks(frame: pd.DataFrame, column: str) -> pd.Series:
    """Weeks that actually carry a rating (the ``min_games`` warm-up gate)."""

    return frame.loc[frame[column].notna(), "week"].drop_duplicates().sort_values()


# ---------------------------------------------------------------------------
# 1. Leakage: week w reads only games through w-1
# ---------------------------------------------------------------------------


def test_future_cell_values_cannot_change_prior_week_ratings() -> None:
    """Violently perturbing a future week's statistic leaves every prior week's
    graph column byte-identical."""

    games = _cfb_like_games()
    column = cfb_graph_column(CELL)
    home_column, away_column = cfb_cell_columns(CELL)
    baseline = add_cfb_graph_team_stat_feature(games, CELL)
    assert baseline[column].notna().any(), "warm-up gate left nothing rated"

    changed = games.copy()
    future = changed["week"].eq(WEEK_COUNT)
    changed.loc[future, home_column] = 99.0
    changed.loc[future, away_column] = -99.0
    rerun = add_cfb_graph_team_stat_feature(changed, CELL)

    cutoff = WEEK_COUNT - 1
    expected = baseline.loc[baseline["week"].le(cutoff), ["game_id", column]]
    actual = rerun.loc[rerun["week"].le(cutoff), ["game_id", column]]
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))


def test_current_week_statistics_cannot_change_the_current_weeks_own_ratings() -> None:
    """A week's own stat differentials are folded in only AFTER that week has
    been assigned, so week w's ratings cannot move when week w's stats do --
    but the NEXT week's must."""

    games = _cfb_like_games()
    column = cfb_graph_column(CELL)
    home_column, away_column = cfb_cell_columns(CELL)
    baseline = add_cfb_graph_team_stat_feature(games, CELL)

    rated = _rated_weeks(baseline, column)
    target_week = int(rated.iloc[0])
    assert target_week < WEEK_COUNT, "need a rated week with a rated successor"

    changed = games.copy()
    current = changed["week"].eq(target_week)
    changed.loc[current, home_column] = 12.0
    changed.loc[current, away_column] = -12.0
    rerun = add_cfb_graph_team_stat_feature(changed, CELL)

    same_week = baseline["week"].eq(target_week)
    np.testing.assert_allclose(
        baseline.loc[same_week, column].to_numpy(dtype=float),
        rerun.loc[same_week, column].to_numpy(dtype=float),
    )
    later = baseline["week"].gt(target_week) & baseline[column].notna()
    assert later.any(), "no later rated week to check propagation against"
    assert not np.allclose(
        baseline.loc[later, column].to_numpy(dtype=float),
        rerun.loc[later, column].to_numpy(dtype=float),
    )


def test_a_removed_future_week_cannot_change_prior_ratings() -> None:
    """Truncating the corpus at week w-1 reproduces weeks 1..w-1 exactly: the
    walk-forward never reaches forward for context."""

    games = _cfb_like_games()
    column = cfb_graph_column(CELL)
    baseline = add_cfb_graph_team_stat_feature(games, CELL)
    truncated = add_cfb_graph_team_stat_feature(
        games.loc[games["week"].lt(WEEK_COUNT)].reset_index(drop=True), CELL
    )

    expected = baseline.loc[baseline["week"].lt(WEEK_COUNT), ["game_id", column]]
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        truncated[["game_id", column]].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 2. Join correctness
# ---------------------------------------------------------------------------


def test_join_is_by_game_id_and_survives_a_shuffled_input() -> None:
    """Every game keeps its OWN rating when the caller's rows arrive in a
    different order, and the caller's order is what comes back."""

    games = _cfb_like_games()
    column = cfb_graph_column(CELL)
    ordered = add_cfb_graph_team_stat_feature(games, CELL)
    shuffled_input = games.sample(frac=1.0, random_state=17)
    shuffled = add_cfb_graph_team_stat_feature(shuffled_input, CELL)

    assert list(shuffled["game_id"]) == list(shuffled_input["game_id"])
    assert shuffled.index.equals(shuffled_input.index)
    expected = ordered.set_index("game_id")[column]
    actual = shuffled.set_index("game_id")[column]
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())


def test_no_rows_are_added_or_dropped_and_only_one_column_is_added() -> None:
    games = _cfb_like_games()
    result = add_cfb_graph_team_stat_feature(games, CELL)
    assert len(result) == len(games)
    assert set(result.columns) - set(games.columns) == {cfb_graph_column(CELL)}


def test_the_callers_frame_is_never_mutated() -> None:
    games = _cfb_like_games()
    before = games.copy()
    add_cfb_graph_team_stat_feature(games, CELL)
    pd.testing.assert_frame_equal(games, before)


def test_duplicate_game_ids_are_refused_rather_than_fanned_out() -> None:
    games = pd.concat([_cfb_like_games(), _cfb_like_games()], ignore_index=True)
    with pytest.raises(DataContractError, match="unique game_id"):
        add_cfb_graph_team_stat_feature(games, CELL)


def test_missing_columns_are_named() -> None:
    games = _cfb_like_games().drop(columns=["home_id"])
    with pytest.raises(DataContractError, match="home_id"):
        add_cfb_graph_team_stat_feature(games, CELL)


# ---------------------------------------------------------------------------
# 3. Adaptation A1: node identity is the ESPN id, not the team name
# ---------------------------------------------------------------------------


def test_a_rebranded_program_stays_one_graph_node() -> None:
    """One id, two name strings across the corpus -- ratings must be identical
    to the run where the names never change, because names are not the key."""

    games = _cfb_like_games()
    column = cfb_graph_column(CELL)
    baseline = add_cfb_graph_team_stat_feature(games, CELL)

    renamed = games.copy()
    late = renamed["week"].ge(4)
    renamed.loc[late & renamed["home_id"].eq(2000), "home_team"] = "Rebranded"
    renamed.loc[late & renamed["away_id"].eq(2000), "away_team"] = "Rebranded"
    rerun = add_cfb_graph_team_stat_feature(renamed, CELL)

    pd.testing.assert_series_equal(baseline[column], rerun[column])


def test_changing_the_id_does_change_the_graph() -> None:
    """The mirror of the test above: identity is load-bearing, not ignored."""

    games = _cfb_like_games()
    column = cfb_graph_column(CELL)
    baseline = add_cfb_graph_team_stat_feature(games, CELL)

    split = games.copy()
    late = split["week"].ge(4)
    split.loc[late & split["home_id"].eq(2000), "home_id"] = 9999
    split.loc[late & split["away_id"].eq(2000), "away_id"] = 9999
    rerun = add_cfb_graph_team_stat_feature(split, CELL)

    both = baseline[column].notna() & rerun[column].notna()
    assert both.any()
    assert not np.allclose(
        baseline.loc[both, column].to_numpy(dtype=float),
        rerun.loc[both, column].to_numpy(dtype=float),
    )


def test_float_and_integer_ids_resolve_to_the_same_node() -> None:
    games = _cfb_like_games()
    column = cfb_graph_column(CELL)
    baseline = add_cfb_graph_team_stat_feature(games, CELL)
    as_float = games.copy()
    as_float["home_id"] = as_float["home_id"].astype(float)
    as_float["away_id"] = as_float["away_id"].astype(float)
    rerun = add_cfb_graph_team_stat_feature(as_float, CELL)
    pd.testing.assert_series_equal(baseline[column], rerun[column])


# ---------------------------------------------------------------------------
# 4. The frozen contract
# ---------------------------------------------------------------------------


def test_only_the_three_predeclared_cells_are_accepted() -> None:
    assert CFB_GRAPH_CELLS == ("def_epa_per_play", "off_epa_per_play", "off_success_rate")
    with pytest.raises(ValueError, match="predeclared CFB cells"):
        add_cfb_graph_team_stat_feature(_cfb_like_games(), "off_sack_rate")
    with pytest.raises(ValueError, match="predeclared CFB cells"):
        cfb_graph_column("def_yards_per_play")


def test_every_declared_cell_exists_in_the_frozen_cfb_feature_contract() -> None:
    """A cell whose home/away pair is not in the CFB benchmark's own contract
    would make the comparator incoherent: the point of the primary comparison
    is that the raw statistic is ALREADY in the reference arm."""

    for cell in CFB_GRAPH_CELLS:
        home_column, away_column = cfb_cell_columns(cell)
        assert home_column in CFB_MODEL_FEATURE_COLUMNS
        assert away_column in CFB_MODEL_FEATURE_COLUMNS
        assert f"diff_{cell}" in CFB_MODEL_FEATURE_COLUMNS


def test_the_structural_configuration_is_the_nfl_frozen_one() -> None:
    """Retuning any of these on CFB would answer a different question than
    'the same transform on new football'."""

    assert CFB_GRAPH_FROZEN_STRUCTURE == {
        "alpha": 0.85,
        "half_life_weeks": 8.0,
        "max_row_l1": 1.0,
        "prior_weight": 1.0,
        "min_games": 16,
        "propagation": "signed_katz",
        "injury_beta": 0.0,
    }


def test_column_names_are_namespaced_per_cell() -> None:
    names = {cfb_graph_column(cell) for cell in CFB_GRAPH_CELLS}
    assert len(names) == len(CFB_GRAPH_CELLS)
    assert cfb_graph_column(CELL) == "graph_v2_team_stat_def_epa_per_play_katz_diff"


def test_the_graph_column_is_not_a_copy_of_the_raw_differential() -> None:
    """If the transform returned the raw differential the whole comparison
    would be vacuous."""

    games = _cfb_like_games()
    column = cfb_graph_column(CELL)
    home_column, away_column = cfb_cell_columns(CELL)
    rated = add_cfb_graph_team_stat_feature(games, CELL)
    usable = rated.loc[rated[column].notna()]
    raw = usable[home_column] - usable[away_column]
    assert not np.allclose(usable[column].to_numpy(dtype=float), raw.to_numpy(dtype=float))
