"""Leakage regression + join-correctness tests for the CFB FluView replication
(``docs/fluview_cfb_replication.md``), per AGENTS.md's "add a leakage
regression test for every new feature family" rule.

The as-of engine this module reuses (``build_checkpoint_tables`` /
``attach_asof_ili`` / ``compute_state_thresholds``) already carries an
exhaustive leak-safety proof in ``tests/test_fluview_battery_leakage.py``.
These tests cover what the CFB replication ADDS on top of it:

* the CFB decision cutoff never lands after kickoff -- which matters more here
  than in the NFL, because college football plays Tuesday and Wednesday games
  in November and the NFL population never does;
* a revision released after a game's cutoff is invisible to that game;
* the ``(season, team_id)`` school -> venue-state join is an id join, resolves
  every row, and is per-season so a venue move is carried;
* a neutral-site game comes back NaN (inapplicable), never 0 ("not elevated");
* a state with no resolvable checkpoint resolves to NaN, never a leaked or
  defaulted value;
* a two-school state contributes ONE state-week panel row per week, not one
  per game, so the frozen top-decile threshold is not silently re-weighted by
  how many programs a state happens to host.

Every test passes ``fluview_raw=``/``team_states=``/``thresholds=``
explicitly, so none of them depends on the local, gitignored FluView or
``team_info`` snapshots being present (a fresh clone may have neither).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.fluview_cfb_feature import (
    CFB_FLUVIEW_AWAY_ELEVATED_COLUMN,
    CFB_FLUVIEW_FEATURE_COLUMNS,
    CFB_FLUVIEW_HOME_ELEVATED_COLUMN,
    attach_cfb_fluview_features,
    attach_cfb_market_states,
    build_cfb_state_week_panel,
    cutoff_dates,
    derive_cfb_fluview_features,
)

THRESHOLDS = {"oh": 4.0, "tx": 4.0, "wy": 4.0}

#: Two schools in the SAME state (ids 1 and 2 both in Ohio) plus one in Texas,
#: mirroring the real two-program states (OH hosts Ohio State and Cincinnati,
#: CA hosts four FBS programs, ...).
TEAM_STATES = pd.DataFrame(
    {
        "season": [2018, 2018, 2018, 2019, 2019, 2019],
        "team_id": pd.array([1, 2, 3, 1, 2, 3], dtype="Int64"),
        "state": ["oh", "oh", "tx", "oh", "oh", "wy"],
    }
)


def _games() -> pd.DataFrame:
    """Four 2018 games: a Saturday, a NOVEMBER TUESDAY (the CFB-specific case),
    a second Saturday sharing the same week as the first, and a neutral-site
    game."""

    return pd.DataFrame(
        {
            "game_id": [101, 102, 103, 104],
            "season": [2018, 2018, 2018, 2018],
            "week": [1, 2, 1, 3],
            "gameday": pd.to_datetime(["2018-01-13", "2018-11-13", "2018-01-13", "2018-05-19"]),
            "home_id": pd.array([1, 3, 2, 1], dtype="Int64"),
            "away_id": pd.array([3, 1, 3, 3], dtype="Int64"),
            "neutral_site": [0, 0, 0, 1],
        }
    )


def _fluview_raw() -> pd.DataFrame:
    """``oh`` is revised from 1.0 (not elevated) to 99.0 (elevated) by a
    release on 2018-05-18 -- one day AFTER game 104's cutoff and long after
    game 101's, so neither may see it."""

    return pd.DataFrame(
        {
            "region": ["oh", "oh", "tx", "tx", "wy"],
            "epiweek": [201801, 201801, 201801, 201801, 201801],
            "issue": [201802, 201820, 201802, 201820, 201802],
            "lag": [1, 19, 1, 19, 1],
            "release_date": pd.to_datetime(
                ["2018-01-09", "2018-05-18", "2018-01-09", "2018-05-18", "2018-01-09"]
            ),
            "ili": [1.0, 99.0, 5.0, 5.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# 1. Cutoff arithmetic -- the CFB-specific weekday cases
# ---------------------------------------------------------------------------


def test_cutoff_is_the_tuesday_on_or_before_gameday_for_every_weekday() -> None:
    """The frozen NFL formula ``(weekday - 1) % 7`` must land on the Tuesday on
    or before the gameday for all seven weekdays, and must NEVER be after
    kickoff. 2018-11-13 is a Tuesday (a real CFB MACtion slot): its cutoff is
    that same day, which is still pregame, not the following week."""

    gamedays = pd.to_datetime([f"2018-11-{day:02d}" for day in range(12, 19)])
    cutoffs = cutoff_dates(pd.Series(gamedays))

    assert (cutoffs <= gamedays).all()
    assert (cutoffs.dt.weekday == 1).all()  # Monday=0, so Tuesday=1
    assert (gamedays - cutoffs).dt.days.max() <= 6
    # Tuesday 2018-11-13 -> itself; Monday 2018-11-12 -> the PREVIOUS Tuesday.
    assert cutoffs.iloc[1] == pd.Timestamp("2018-11-13")
    assert cutoffs.iloc[0] == pd.Timestamp("2018-11-06")


def test_cutoff_never_lands_after_kickoff_on_the_working_frame() -> None:
    frame = attach_cfb_market_states(_games(), TEAM_STATES)
    assert (frame["cutoff_date"] <= frame["gameday"]).all()


# ---------------------------------------------------------------------------
# 2. Leakage
# ---------------------------------------------------------------------------


def test_a_revision_released_after_the_cutoff_is_invisible() -> None:
    """Game 104's cutoff is Tuesday 2018-05-15 (gameday Saturday 2018-05-19),
    strictly before the 2018-05-18 release that revises ``oh``'s ILI from 1.0
    to 99.0. No game in this fixture may read the revised value; a game whose
    cutoff falls on/after the release date must."""

    games = _games()
    derived, _ = derive_cfb_fluview_features(
        games, fluview_raw=_fluview_raw(), team_states=TEAM_STATES, thresholds=THRESHOLDS
    )
    derived = derived.set_index("game_id")
    # 101/103 are Ohio home games at a January cutoff: the OLD, not-elevated value.
    assert derived.loc[101, CFB_FLUVIEW_HOME_ELEVATED_COLUMN] == 0.0
    assert derived.loc[103, CFB_FLUVIEW_HOME_ELEVATED_COLUMN] == 0.0

    later = games.copy()
    later["gameday"] = pd.to_datetime(["2018-05-26", "2018-11-13", "2018-05-26", "2018-05-19"])
    later["neutral_site"] = [0, 0, 0, 1]
    derived_later, _ = derive_cfb_fluview_features(
        later, fluview_raw=_fluview_raw(), team_states=TEAM_STATES, thresholds=THRESHOLDS
    )
    derived_later = derived_later.set_index("game_id")
    assert derived_later.loc[101, CFB_FLUVIEW_HOME_ELEVATED_COLUMN] == 1.0


def test_a_late_reissue_of_an_older_epiweek_never_overwrites_a_newer_one() -> None:
    """A stale revision of epiweek 201801, released after epiweek 201810 was
    already known, must not pull the as-of value backwards -- the running-max
    checkpoint rule. Without it, a game after the late release would read the
    old week's elevated value."""

    fluview = pd.DataFrame(
        {
            "region": ["oh", "oh", "oh"],
            "epiweek": [201801, 201810, 201801],
            "issue": [201802, 201811, 201830],
            "lag": [1, 1, 29],
            "release_date": pd.to_datetime(["2018-01-09", "2018-03-06", "2018-07-24"]),
            "ili": [99.0, 1.0, 99.0],
        }
    )
    games = _games().assign(
        gameday=pd.to_datetime(["2018-08-04", "2018-08-04", "2018-08-04", "2018-08-04"]),
        neutral_site=[0, 0, 0, 0],
    )
    derived, _ = derive_cfb_fluview_features(
        games, fluview_raw=fluview, team_states=TEAM_STATES, thresholds=THRESHOLDS
    )
    derived = derived.set_index("game_id")
    # The freshest EPIWEEK known as of 2018-07-31 is 201810 (ili 1.0), not the
    # late-reissued 201801 (ili 99.0).
    assert derived.loc[101, CFB_FLUVIEW_HOME_ELEVATED_COLUMN] == 0.0


def test_a_state_with_no_resolvable_checkpoint_is_nan_not_defaulted() -> None:
    """The measured ``ny``-style upstream gap (docs/fluview_battery.md section
    1) must resolve to missing, never to 0.0 ("not elevated") -- a defaulted 0
    would be evidence the data does not support."""

    empty = pd.DataFrame(columns=["region", "epiweek", "issue", "lag", "release_date", "ili"])
    derived, diagnostics = derive_cfb_fluview_features(
        _games(), fluview_raw=empty, team_states=TEAM_STATES, thresholds=THRESHOLDS
    )
    assert derived[CFB_FLUVIEW_HOME_ELEVATED_COLUMN].isna().all()
    assert derived[CFB_FLUVIEW_AWAY_ELEVATED_COLUMN].isna().all()
    assert diagnostics["n_home_missing"] == len(derived)


# ---------------------------------------------------------------------------
# 3. Join correctness
# ---------------------------------------------------------------------------


def test_state_join_is_by_season_and_team_id_and_carries_a_venue_move() -> None:
    """Team 3 sits in ``tx`` in 2018 and ``wy`` in 2019 in the fixture map. A
    per-season id join must give each season its own state; a season-invariant
    map (or a name join) would give both the same one."""

    games = pd.DataFrame(
        {
            "game_id": [201, 202],
            "season": [2018, 2019],
            "week": [1, 1],
            "gameday": pd.to_datetime(["2018-01-13", "2019-01-12"]),
            "home_id": pd.array([3, 3], dtype="Int64"),
            "away_id": pd.array([1, 1], dtype="Int64"),
            "neutral_site": [0, 0],
        }
    )
    frame = attach_cfb_market_states(games, TEAM_STATES)
    assert list(frame["home_state"]) == ["tx", "wy"]
    assert list(frame["away_state"]) == ["oh", "oh"]


def test_unmapped_team_is_counted_not_silently_dropped() -> None:
    games = _games()
    games.loc[0, "home_id"] = 999
    _, diagnostics = derive_cfb_fluview_features(
        games, fluview_raw=_fluview_raw(), team_states=TEAM_STATES, thresholds=THRESHOLDS
    )
    assert diagnostics["n_unmapped_state"] == 1


def test_neutral_site_game_is_nan_on_both_columns() -> None:
    """The CFB mirror of the NFL ``location == "Home"`` restriction
    (docs/fluview_battery.md section 2): at a neutral site the home-market
    mechanism does not apply, so the feature is missing, never 0."""

    derived, diagnostics = derive_cfb_fluview_features(
        _games(), fluview_raw=_fluview_raw(), team_states=TEAM_STATES, thresholds=THRESHOLDS
    )
    derived = derived.set_index("game_id")
    assert pd.isna(derived.loc[104, CFB_FLUVIEW_HOME_ELEVATED_COLUMN])
    assert pd.isna(derived.loc[104, CFB_FLUVIEW_AWAY_ELEVATED_COLUMN])
    assert not pd.isna(derived.loc[101, CFB_FLUVIEW_HOME_ELEVATED_COLUMN])
    assert diagnostics["n_neutral_site"] == 1


def test_two_schools_in_one_state_share_a_single_panel_row_per_week() -> None:
    """Games 101 and 103 are both Ohio home games in 2018 week 1. The state-week
    panel the frozen top-decile threshold is computed on must carry ONE ``oh``
    row for that week, not two -- otherwise a state's decile would be weighted
    by how many programs it hosts."""

    frame = attach_cfb_market_states(_games(), TEAM_STATES)
    frame["home_ili"] = [1.0, 5.0, 1.0, 1.0]
    frame["away_ili"] = [5.0, 1.0, 5.0, 5.0]
    panel = build_cfb_state_week_panel(frame)

    oh_week_1 = panel.loc[panel["state"].eq("oh") & panel["week"].eq(1)]
    assert len(oh_week_1) == 1
    # The neutral-site game (104) contributes no panel row at all.
    assert not (panel["season"].eq(2018) & panel["week"].eq(3)).any()


# ---------------------------------------------------------------------------
# 4. Additive-merge / contract discipline
# ---------------------------------------------------------------------------


def test_attach_is_purely_additive() -> None:
    base = _games()
    base["some_pre_existing_column"] = np.arange(len(base))
    widened, _ = attach_cfb_fluview_features(
        base, fluview_raw=_fluview_raw(), team_states=TEAM_STATES, thresholds=THRESHOLDS
    )
    pd.testing.assert_frame_equal(widened[base.columns.tolist()], base, check_exact=True)
    assert set(widened.columns) - set(base.columns) == set(CFB_FLUVIEW_FEATURE_COLUMNS)


def test_attach_refuses_a_column_collision() -> None:
    games = _games()
    games[CFB_FLUVIEW_HOME_ELEVATED_COLUMN] = 0.0
    with pytest.raises(DataContractError):
        attach_cfb_fluview_features(
            games, fluview_raw=_fluview_raw(), team_states=TEAM_STATES, thresholds=THRESHOLDS
        )


def test_missing_required_column_is_refused() -> None:
    games = _games().drop(columns=["neutral_site"])
    with pytest.raises(DataContractError):
        attach_cfb_market_states(games, TEAM_STATES)


def test_thresholds_are_applied_not_recomputed_when_supplied() -> None:
    """With a floor low enough that every reading clears it, every resolvable
    game must flag elevated -- proving the caller's frozen thresholds are used
    rather than a decile re-derived from whatever games were passed in."""

    derived, _ = derive_cfb_fluview_features(
        _games(),
        fluview_raw=_fluview_raw(),
        team_states=TEAM_STATES,
        thresholds={"oh": 0.5, "tx": 0.5, "wy": 0.5},
    )
    derived = derived.set_index("game_id")
    assert derived.loc[101, CFB_FLUVIEW_HOME_ELEVATED_COLUMN] == 1.0
    assert derived.loc[101, CFB_FLUVIEW_AWAY_ELEVATED_COLUMN] == 1.0
