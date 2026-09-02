"""Leakage regression + wiring tests for the ``weak_stack_graph_off_rush_epa``
candidate profile's one new column
(docs/graph_team_stat_off_rush_epa_on_production.md), per AGENTS.md's "a leakage
regression test for every new feature family" rule. Mirrors
``tests/test_graph_team_stat_def_ypp_production_feature.py`` (the
def_yards_per_play sibling) exactly, substituting the ``off_rush_epa_per_play``
team_stat column.

The underlying propagation engine (``add_graph_ratings_v2_features``) already
carries an exhaustive leak-safety proof for the ``team_stat`` arm in
``tests/test_graph_ratings_v2.py``
(``test_future_team_stat_values_cannot_change_prior_ratings``). These tests
cover the thin wrapper this module adds on top of it: that the additive merge
does not disturb pre-existing columns, and that the wrapper's own
``game_id``-keyed join does not reintroduce a leak the engine itself avoids.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.constants import (
    FEATURE_SETS,
    GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.graph_team_stat_off_rush_epa_production_feature import (
    GRAPH_OFF_RUSH_EPA_COLUMN,
    attach_graph_off_rush_epa_feature,
    derive_graph_off_rush_epa_feature,
)
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, margin_feature_columns


def test_weak_stack_graph_off_rush_epa_profile_is_registered_and_disjoint() -> None:
    assert "weak_stack_graph_off_rush_epa" in MARGIN_FEATURE_PROFILES
    assert set(GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS)
    for name in ("football_weak_stack", "full_weak_stack", "football", "full"):
        assert set(FEATURE_SETS[name]).isdisjoint(GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS)
    # weak_stack_graph_off_rush_epa = weak_stack plus exactly the one new column
    # -- never used by the active model.
    graph_off_rush_epa_columns = set(
        margin_feature_columns("market_residual", "weak_stack_graph_off_rush_epa")
    )
    weak_stack_columns = set(margin_feature_columns("market_residual", "weak_stack"))
    assert graph_off_rush_epa_columns - weak_stack_columns == set(
        GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS
    )
    assert len(GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS) == 1


# ---------------------------------------------------------------------------
# Synthetic schedule: 8 teams, 4 games/week. games_seen reaches the frozen
# min_games=16 threshold exactly at the end of week 4 (4 games/week x 4 weeks
# = 16), so week 5 is the first week with a non-NaN graph rating -- enough to
# exercise both the "still warming up" (NaN) and "rated" branches without a
# large fixture.
# ---------------------------------------------------------------------------


def _schedule(n_weeks: int) -> pd.DataFrame:
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    pairs = [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]
    rows: list[dict[str, object]] = []
    for week in range(1, n_weeks + 1):
        for index, (home, away) in enumerate(pairs):
            rows.append(
                {
                    "game_id": f"2021_{week:02d}_{home}_{away}",
                    "season": 2021,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "result": 3.0 if (week + index) % 2 == 0 else -3.0,
                    # Vary the raw off_rush_epa_per_play differential so the
                    # graph has a real signal to propagate rather than an
                    # all-zero matrix.
                    "home_off_rush_epa_per_play": -0.05 + 0.02 * index,
                    "away_off_rush_epa_per_play": 0.04 - 0.02 * index,
                }
            )
    frame = pd.DataFrame(rows)
    frame["gameday"] = pd.to_datetime("2021-09-01") + pd.to_timedelta(
        (frame["week"] - 1) * 7, unit="D"
    )
    return frame


def test_derive_graph_off_rush_epa_feature_is_nan_before_min_games_and_rated_after() -> None:
    schedule = _schedule(n_weeks=5)
    derived = derive_graph_off_rush_epa_feature(schedule).set_index("game_id")

    early = derived.loc[derived.index.str.startswith("2021_01_")]
    assert early[GRAPH_OFF_RUSH_EPA_COLUMN].isna().all()

    late = derived.loc[derived.index.str.startswith("2021_05_")]
    assert late[GRAPH_OFF_RUSH_EPA_COLUMN].notna().all()


def test_attach_graph_off_rush_epa_feature_is_purely_additive() -> None:
    schedule = _schedule(n_weeks=5)
    base = schedule.copy()
    base["some_pre_existing_column"] = range(len(base))

    widened = attach_graph_off_rush_epa_feature(base)

    pd.testing.assert_frame_equal(widened[base.columns.tolist()], base, check_exact=True)
    new_columns = set(widened.columns) - set(base.columns)
    assert new_columns == set(GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS)


def test_attach_graph_off_rush_epa_feature_refuses_a_collision() -> None:
    schedule = _schedule(n_weeks=1)
    schedule[GRAPH_OFF_RUSH_EPA_COLUMN] = 0.0
    with pytest.raises(DataContractError):
        attach_graph_off_rush_epa_feature(schedule)


def test_attach_graph_off_rush_epa_feature_is_leak_safe_across_a_future_week() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    Week 7's rating is read from the graph as accumulated through week 6, so
    perturbing week 6's own result and raw off_rush_epa_per_play columns (a
    violent, deliberate leak if the wrapper's merge were somehow order- or
    index-dependent) must leave weeks 1-6's OWN already-computed graph feature
    values byte-identical -- they are assigned before week 6's own values are
    folded into the accumulator -- while changing week 7's, which reads the
    graph through the now-perturbed week 6. This mirrors
    ``test_graph_ratings_v2.test_future_team_stat_values_cannot_change_prior_ratings``
    at this wrapper's own public entry point rather than only at the engine's.
    """

    schedule = _schedule(n_weeks=7)
    baseline = attach_graph_off_rush_epa_feature(schedule).set_index("game_id")

    mutated = schedule.copy()
    perturbed_week = mutated["game_id"].str.startswith("2021_06_")
    mutated.loc[perturbed_week, "result"] = 45.0
    mutated.loc[perturbed_week, "home_off_rush_epa_per_play"] = 0.60
    mutated.loc[perturbed_week, "away_off_rush_epa_per_play"] = -0.40
    changed = attach_graph_off_rush_epa_feature(mutated).set_index("game_id")

    # Only the DERIVED graph column is compared here -- week 6's own raw
    # pass-through columns (result, home/away_off_rush_epa_per_play) differ from
    # baseline by construction, since those are exactly what was mutated.
    unaffected = baseline.index[~baseline.index.str.startswith("2021_07_")]
    pd.testing.assert_series_equal(
        changed.loc[unaffected, GRAPH_OFF_RUSH_EPA_COLUMN],
        baseline.loc[unaffected, GRAPH_OFF_RUSH_EPA_COLUMN],
        check_exact=True,
    )
    # The perturbation is not simply inert: week 7 (downstream of week 6's
    # now-perturbed values) changes.
    following_week = baseline.index.str.startswith("2021_07_")
    assert not changed.loc[following_week, GRAPH_OFF_RUSH_EPA_COLUMN].equals(
        baseline.loc[following_week, GRAPH_OFF_RUSH_EPA_COLUMN]
    )
