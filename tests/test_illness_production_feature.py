"""Leakage, additivity and join contracts for the on-production illness columns.

Predeclared in ``docs/illness_on_production.md``. Every fixture is built in
memory: these tests must pass in a fresh clone with no local data snapshots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.illness_production_feature import (
    ILLNESS_AWAY_ACTIVE_GE1_COLUMN,
    ILLNESS_HOME_GE2_COLUMN,
    ILLNESS_ON_PRODUCTION_FEATURE_COLUMNS,
    attach_illness_features,
    derive_illness_features,
)


def _games() -> pd.DataFrame:
    """Two Sunday games in the same week, both kicking off at 13:00 ET.

    The pick deadline is ``min(own kickoff, that week's Sunday 16:00 ET)``, so
    for a 13:00 Sunday kickoff the deadline is the kickoff itself.
    """

    return pd.DataFrame(
        {
            "game_id": ["2012_05_AAA_BBB", "2012_05_CCC_DDD"],
            "season": [2012, 2012],
            "week": [5, 5],
            "gameday": ["2012-10-07", "2012-10-07"],
            "gametime": ["13:00", "13:00"],
            "home_team": ["BBB", "DDD"],
            "away_team": ["AAA", "CCC"],
        }
    )


def _injury_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """A minimal nflverse-injuries frame already through ``load_injuries``'s
    normalisation (REG-filtered, aliased teams, tz-aware ``date_modified``,
    ``is_illness`` computed)."""

    columns = ["season", "week", "team", "gsis_id", "date_modified", "is_illness", "report_status"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["team"] = frame["team"].astype("string")
    frame["date_modified"] = pd.to_datetime(frame["date_modified"], utc=True)
    return frame


def _row(
    team: str,
    gsis_id: str,
    date_modified: str,
    *,
    is_illness: bool = True,
    report_status: str = "Questionable",
) -> dict[str, object]:
    return {
        "season": 2012,
        "week": 5,
        "team": team,
        "gsis_id": gsis_id,
        "date_modified": date_modified,
        "is_illness": is_illness,
        "report_status": report_status,
    }


# ---------------------------------------------------------------------------
# Leakage: the as-of pick-deadline cutoff is binding
# ---------------------------------------------------------------------------


def test_report_filed_after_the_pick_deadline_never_reaches_the_column() -> None:
    """A revision filed AFTER kickoff must be invisible.

    The away column would fire on this row if the cutoff leaked; instead the
    team-week has zero visible revisions and resolves to MISSING (NaN), which
    is the documented treatment -- never a zero count.
    """

    games = _games()
    late = _injury_rows([_row("AAA", "P1", "2012-10-07 20:00:00")])
    derived = derive_illness_features(games, injuries=late).set_index("game_id")
    assert np.isnan(derived.loc["2012_05_AAA_BBB", ILLNESS_AWAY_ACTIVE_GE1_COLUMN])


def test_report_filed_before_the_pick_deadline_does_reach_the_column() -> None:
    """The mirror of the test above: the identical row filed BEFORE the
    deadline must fire, so the leakage test is proving a cutoff and not merely
    a builder that never returns anything."""

    games = _games()
    early = _injury_rows([_row("AAA", "P1", "2012-10-05 12:00:00")])
    derived = derive_illness_features(games, injuries=early).set_index("game_id")
    assert derived.loc["2012_05_AAA_BBB", ILLNESS_AWAY_ACTIVE_GE1_COLUMN] == 1.0


def test_only_the_latest_visible_revision_per_entity_counts() -> None:
    """Two revisions for the same player: the later one, still before the
    deadline, downgrades him to not-ill. The as-of state is the LATEST
    surviving revision, so the column must not fire."""

    games = _games()
    revised = _injury_rows(
        [
            _row("AAA", "P1", "2012-10-03 12:00:00", is_illness=True),
            _row("AAA", "P1", "2012-10-05 12:00:00", is_illness=False),
        ]
    )
    derived = derive_illness_features(games, injuries=revised).set_index("game_id")
    assert derived.loc["2012_05_AAA_BBB", ILLNESS_AWAY_ACTIVE_GE1_COLUMN] == 0.0


def test_a_later_revision_after_the_deadline_cannot_undo_a_visible_one() -> None:
    """The complement of the previous test, and the sharper leakage claim: a
    post-deadline downgrade must be invisible, so the column keeps the state
    that was actually knowable at pick time."""

    games = _games()
    revised = _injury_rows(
        [
            _row("AAA", "P1", "2012-10-05 12:00:00", is_illness=True),
            _row("AAA", "P1", "2012-10-07 20:00:00", is_illness=False),
        ]
    )
    derived = derive_illness_features(games, injuries=revised).set_index("game_id")
    assert derived.loc["2012_05_AAA_BBB", ILLNESS_AWAY_ACTIVE_GE1_COLUMN] == 1.0


def test_other_games_are_unaffected_by_one_games_reports() -> None:
    """A team-week with no visible rows at all stays MISSING rather than
    borrowing another game's state."""

    games = _games()
    only_first = _injury_rows([_row("AAA", "P1", "2012-10-05 12:00:00")])
    derived = derive_illness_features(games, injuries=only_first).set_index("game_id")
    assert np.isnan(derived.loc["2012_05_CCC_DDD", ILLNESS_AWAY_ACTIVE_GE1_COLUMN])


# ---------------------------------------------------------------------------
# The two columns' own thresholds
# ---------------------------------------------------------------------------


def test_active_excludes_players_not_expected_to_play() -> None:
    """ "Active" illness excludes Out/Doubtful, inherited unchanged from the
    frozen battery: a single ruled-out ill player must not fire the away
    column, while the team-week is still visible (0.0, not NaN)."""

    games = _games()
    ruled_out = _injury_rows([_row("AAA", "P1", "2012-10-05 12:00:00", report_status="Out")])
    derived = derive_illness_features(games, injuries=ruled_out).set_index("game_id")
    assert derived.loc["2012_05_AAA_BBB", ILLNESS_AWAY_ACTIVE_GE1_COLUMN] == 0.0


def test_home_column_needs_two_ill_players() -> None:
    """``illness_home_ge2`` fires at a count of two, not one."""

    games = _games()
    one = _injury_rows([_row("BBB", "P1", "2012-10-05 12:00:00")])
    two = _injury_rows(
        [
            _row("BBB", "P1", "2012-10-05 12:00:00"),
            _row("BBB", "P2", "2012-10-05 12:00:00"),
        ]
    )
    with_one = derive_illness_features(games, injuries=one).set_index("game_id")
    with_two = derive_illness_features(games, injuries=two).set_index("game_id")
    assert with_one.loc["2012_05_AAA_BBB", ILLNESS_HOME_GE2_COLUMN] == 0.0
    assert with_two.loc["2012_05_AAA_BBB", ILLNESS_HOME_GE2_COLUMN] == 1.0


# ---------------------------------------------------------------------------
# Additivity and join contracts
# ---------------------------------------------------------------------------


def test_attach_is_purely_additive() -> None:
    """Every pre-existing column comes back bit-identical and exactly the two
    named columns are added -- the discipline every candidate feature module in
    this family shares."""

    games = _games()
    games["some_existing_feature"] = [1.5, -2.25]
    injuries = _injury_rows([_row("AAA", "P1", "2012-10-05 12:00:00")])

    widened = attach_illness_features(games, injuries=injuries)

    new_columns = sorted(set(widened.columns) - set(games.columns))
    assert new_columns == sorted(ILLNESS_ON_PRODUCTION_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(games, widened[games.columns], check_exact=True)
    assert list(widened.index) == list(games.index)


def test_attach_requires_the_join_key() -> None:
    games = _games().drop(columns=["game_id"])
    with pytest.raises(DataContractError, match="game_id"):
        attach_illness_features(games, injuries=_injury_rows([]))


def test_attach_refuses_to_overwrite_an_existing_column() -> None:
    games = _games()
    games[ILLNESS_HOME_GE2_COLUMN] = 0.0
    with pytest.raises(DataContractError, match=ILLNESS_HOME_GE2_COLUMN):
        attach_illness_features(games, injuries=_injury_rows([]))


def test_derive_requires_every_input_column() -> None:
    games = _games().drop(columns=["gametime"])
    with pytest.raises(DataContractError, match="gametime"):
        derive_illness_features(games, injuries=_injury_rows([]))


def test_join_is_one_to_one() -> None:
    """A duplicated game_id must raise rather than silently fanning out rows --
    ``validate="one_to_one"`` on the merge is what enforces it.

    Two exception types are accepted because which one surfaces is a pandas
    implementation detail, not a contract: merging on a Series ``left_on`` with
    duplicate keys trips an internal index-name check (``TypeError``) before
    ``validate`` reports its own ``MergeError``. The claim under test is that
    it RAISES, never that it raises a particular class.
    """

    games = pd.concat([_games(), _games().iloc[[0]]], ignore_index=True)
    with pytest.raises((pd.errors.MergeError, TypeError)):
        attach_illness_features(games, injuries=_injury_rows([]))
