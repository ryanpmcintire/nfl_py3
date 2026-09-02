"""Leakage, orientation, additivity and join contracts for the on-production
third-down mean-reversion fade column.

Predeclared in ``docs/redzone_reversion_on_production.md``. Every fixture is
built in memory: these tests must pass in a fresh clone with no local data
snapshots (no PBP snapshot is ever read).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.redzone_reversion_production_feature import (
    REDZONE_REVERSION_ON_PRODUCTION_FEATURE_COLUMNS,
    REDZONE_THIRD_DOWN_OVER_FADE_COLUMN,
    TRAIT_COLUMN,
    attach_redzone_third_down_features,
    derive_redzone_third_down_features,
    expanding_top_quartile_thresholds,
)

COLUMN = REDZONE_THIRD_DOWN_OVER_FADE_COLUMN

#: Four teams is the smallest panel where a 75th percentile is not degenerate:
#: exactly one of the four sits at or above the cut in a given season.
TEAMS = ("AAA", "BBB", "CCC", "DDD")


def _panel(values_by_season: dict[int, dict[str, float]]) -> pd.DataFrame:
    """A synthetic offensive efficiency panel: one centred third-down
    conversion rate per (season, team)."""

    rows = [
        {"season": season, "team": team, TRAIT_COLUMN: value}
        for season, by_team in values_by_season.items()
        for team, value in by_team.items()
    ]
    return pd.DataFrame(rows, columns=["season", "team", TRAIT_COLUMN])


def _flat(season: int, *, high: str = "AAA") -> dict[str, float]:
    """One clear over-performer, three at the floor."""

    return {team: (0.10 if team == high else -0.02) for team in TEAMS}


def _games(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    """``(game_id, season, home_team, away_team)`` rows."""

    return pd.DataFrame(rows, columns=["game_id", "season", "home_team", "away_team"])


def _value(games: pd.DataFrame, panel: pd.DataFrame, game_id: str) -> float:
    derived = derive_redzone_third_down_features(games, offense=panel).set_index("game_id")
    return float(derived.loc[game_id, COLUMN])


# ---------------------------------------------------------------------------
# Leakage: a season-S value uses ONLY seasons strictly before S
# ---------------------------------------------------------------------------


def test_only_strictly_prior_seasons_reach_a_value() -> None:
    """The binding leakage claim, both halves at once.

    A 2012 game's value must be a function of the 2011 panel row and a
    threshold estimated on 2009-2011 only. Injecting an extreme into 2012 --
    the game's OWN season -- or into 2013 must leave every 2012 value
    untouched. If either the prior-season lookup or the expanding threshold
    leaked, one of these injections would move the value.
    """

    base = {season: _flat(season) for season in (2009, 2010, 2011)}
    games = _games([("2012_01_BBB_AAA", 2012, "BBB", "AAA")])

    reference = _value(games, _panel(base), "2012_01_BBB_AAA")

    for injected_season in (2012, 2013, 2014):
        polluted = dict(base)
        # An extreme value large enough to drag any pooled quantile upward and
        # to flag every team it touches.
        polluted[injected_season] = dict.fromkeys(TEAMS, 99.0)
        assert _value(games, _panel(polluted), "2012_01_BBB_AAA") == reference


def test_the_threshold_itself_uses_only_strictly_prior_seasons() -> None:
    """Isolates the threshold half of the claim from the lookup half.

    The 2012 threshold is the 75th percentile over 2009-2011. Replacing the
    2012 and later panel rows entirely must not change it, while replacing a
    2010 row must.
    """

    base = {season: _flat(season) for season in (2009, 2010, 2011, 2012, 2013)}
    thresholds = expanding_top_quartile_thresholds(_panel(base))

    later_changed = dict(base)
    later_changed[2012] = dict.fromkeys(TEAMS, 50.0)
    later_changed[2013] = dict.fromkeys(TEAMS, 50.0)
    assert expanding_top_quartile_thresholds(_panel(later_changed))[2012] == thresholds[2012]

    earlier_changed = dict(base)
    earlier_changed[2010] = dict.fromkeys(TEAMS, 50.0)
    assert expanding_top_quartile_thresholds(_panel(earlier_changed))[2012] != thresholds[2012]


def test_the_first_panel_season_has_no_threshold_and_no_value() -> None:
    """No strictly-prior season exists for the panel's first season, so it
    carries neither a threshold nor a column value -- NaN, never 0."""

    panel = _panel({2009: _flat(2009), 2010: _flat(2010)})
    assert 2009 not in expanding_top_quartile_thresholds(panel).index
    assert 2010 in expanding_top_quartile_thresholds(panel).index

    games = _games([("2009_01_BBB_AAA", 2009, "BBB", "AAA")])
    derived = derive_redzone_third_down_features(games, offense=panel).set_index("game_id")
    assert np.isnan(derived.loc["2009_01_BBB_AAA", COLUMN])


def test_a_team_with_no_prior_season_row_is_missing_not_zero() -> None:
    """An expansion team with no season-S-1 panel row leaves the game NaN: "no
    prior-season information" and "an unflagged team" are different states."""

    panel = _panel({2009: _flat(2009), 2010: _flat(2010)})
    games = _games([("2011_01_ZZZ_AAA", 2011, "ZZZ", "AAA")])
    derived = derive_redzone_third_down_features(games, offense=panel).set_index("game_id")
    assert np.isnan(derived.loc["2011_01_ZZZ_AAA", COLUMN])


# ---------------------------------------------------------------------------
# Sign and orientation
# ---------------------------------------------------------------------------


def test_sign_orientation_covers_all_three_states() -> None:
    """``int(home flagged) - int(away flagged)``: a home over-performer to fade
    is +1, an away over-performer is -1, both-flagged and neither-flagged are
    both 0. This is deviation 1's whole content, pinned."""

    panel = _panel(
        {
            2010: _flat(2010),
            # 2011 is the prior season for the 2012 games below: AAA and BBB
            # are the over-performers, CCC and DDD are not.
            2011: {"AAA": 0.10, "BBB": 0.10, "CCC": -0.02, "DDD": -0.02},
        }
    )
    games = _games(
        [
            ("home_flagged", 2012, "AAA", "CCC"),
            ("away_flagged", 2012, "CCC", "AAA"),
            ("both_flagged", 2012, "AAA", "BBB"),
            ("neither_flagged", 2012, "CCC", "DDD"),
        ]
    )
    derived = derive_redzone_third_down_features(games, offense=panel).set_index("game_id")
    assert derived.loc["home_flagged", COLUMN] == 1.0
    assert derived.loc["away_flagged", COLUMN] == -1.0
    assert derived.loc["both_flagged", COLUMN] == 0.0
    assert derived.loc["neither_flagged", COLUMN] == 0.0


def test_the_flag_is_at_or_above_the_cut() -> None:
    """The frozen screen's C3 flag is ``>= q75``, inherited unchanged: a team
    sitting exactly ON the threshold is flagged."""

    prior = {"AAA": 0.10, "BBB": 0.04, "CCC": 0.00, "DDD": -0.02}
    panel = _panel({2010: prior, 2011: prior})
    cut = float(expanding_top_quartile_thresholds(_panel({2010: prior}))[2011])
    on_the_cut = min(t for t, v in prior.items() if v >= cut)

    games = _games([("g", 2011, on_the_cut, "DDD")])
    derived = derive_redzone_third_down_features(games, offense=panel).set_index("game_id")
    assert derived.loc["g", COLUMN] == 1.0


def test_team_aliases_are_canonicalised_on_both_sides() -> None:
    """``TEAM_ABBREVIATION_ALIASES`` is applied to the feature table's team
    codes as well as the panel's, matching the frozen screen. A game carrying
    the retired ``SD`` code must find the panel's ``LAC`` row."""

    panel = _panel(
        {
            2010: {"LAC": 0.10, "BBB": -0.02, "CCC": -0.02, "DDD": -0.02},
            2011: {"LAC": 0.10, "BBB": -0.02, "CCC": -0.02, "DDD": -0.02},
        }
    )
    games = _games([("g", 2012, "SD", "BBB")])
    derived = derive_redzone_third_down_features(games, offense=panel).set_index("game_id")
    assert derived.loc["g", COLUMN] == 1.0


# ---------------------------------------------------------------------------
# Additivity and join contracts
# ---------------------------------------------------------------------------


def test_attach_is_purely_additive() -> None:
    """Every pre-existing column comes back bit-identical and exactly the one
    named column is added -- the discipline every candidate feature module in
    this family shares."""

    panel = _panel({2010: _flat(2010), 2011: _flat(2011)})
    games = _games(
        [
            ("g1", 2012, "AAA", "CCC"),
            ("g2", 2012, "CCC", "DDD"),
        ]
    )
    games["some_existing_feature"] = [1.5, -2.25]

    widened = attach_redzone_third_down_features(games, offense=panel)

    new_columns = sorted(set(widened.columns) - set(games.columns))
    assert new_columns == sorted(REDZONE_REVERSION_ON_PRODUCTION_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(games, widened[games.columns], check_exact=True)
    assert list(widened.index) == list(games.index)


def test_attach_requires_the_join_key() -> None:
    panel = _panel({2010: _flat(2010), 2011: _flat(2011)})
    games = _games([("g1", 2012, "AAA", "CCC")]).drop(columns=["game_id"])
    with pytest.raises(DataContractError, match="game_id"):
        attach_redzone_third_down_features(games, offense=panel)


def test_attach_refuses_to_overwrite_an_existing_column() -> None:
    panel = _panel({2010: _flat(2010), 2011: _flat(2011)})
    games = _games([("g1", 2012, "AAA", "CCC")])
    games[COLUMN] = 0.0
    with pytest.raises(DataContractError, match=COLUMN):
        attach_redzone_third_down_features(games, offense=panel)


def test_derive_requires_every_input_column() -> None:
    panel = _panel({2010: _flat(2010), 2011: _flat(2011)})
    games = _games([("g1", 2012, "AAA", "CCC")]).drop(columns=["away_team"])
    with pytest.raises(DataContractError, match="away_team"):
        derive_redzone_third_down_features(games, offense=panel)


def test_a_duplicated_panel_row_raises() -> None:
    """Two rows for the same (team, season) would silently fan the join out;
    the builder refuses instead."""

    panel = pd.concat([_panel({2010: _flat(2010), 2011: _flat(2011)})] * 2, ignore_index=True)
    games = _games([("g1", 2012, "AAA", "CCC")])
    with pytest.raises(DataContractError, match="duplicate"):
        derive_redzone_third_down_features(games, offense=panel)


def test_join_is_one_to_one() -> None:
    """A duplicated game_id must raise rather than silently fanning out rows --
    ``validate="one_to_one"`` on the merge is what enforces it.

    Two exception types are accepted because which one surfaces is a pandas
    implementation detail, not a contract: merging on a Series ``left_on`` with
    duplicate keys trips an internal index-name check (``TypeError``) before
    ``validate`` reports its own ``MergeError``. The claim under test is that
    it RAISES, never that it raises a particular class.
    """

    panel = _panel({2010: _flat(2010), 2011: _flat(2011)})
    games = _games([("g1", 2012, "AAA", "CCC"), ("g1", 2012, "CCC", "DDD")])
    with pytest.raises((pd.errors.MergeError, TypeError)):
        attach_redzone_third_down_features(games, offense=panel)
