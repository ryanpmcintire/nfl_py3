"""Leakage, additivity and join contracts for the on-production pace column.

Predeclared in ``docs/team_style_pace_on_production.md``. Every fixture is
built in memory: these tests must pass in a fresh clone with no local data
snapshots (the team-season style panel lives under ``data/pbp/**``, which is
gitignored).

The leakage claim under test has TWO halves, because this column has two
pregame-safety obligations rather than one:

1. a season-S game's flag reads only season ``< S`` **pace values**, and
2. its top-quartile **threshold** is estimated only from games in seasons
   ``< S`` -- the declared deviation from ``scripts/team_style_screen.py``,
   whose single cut is estimated over the whole 2009-2025 panel.

Each half is proved in both directions: an injection at or after season S
cannot move a season-S flag, and the mirror injection strictly before season S
does move it, so the tests prove a cutoff rather than a builder that never
returns anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.team_style_pace_production_feature import (
    PACE_CENTERED_COLUMN,
    TEAM_STYLE_PACE_MISMATCH_COLUMN,
    TEAM_STYLE_PACE_MISMATCH_ON_PRODUCTION_FEATURE_COLUMNS,
    attach_team_style_pace_features,
    derive_team_style_pace_features,
)

TEAMS = ("AAA", "BBB", "CCC", "DDD")
SEASONS = (2009, 2010, 2011, 2012, 2013)


def _style(overrides: dict[tuple[int, str], float] | None = None) -> pd.DataFrame:
    """A (season, team) centred-pace panel with a deterministic spread.

    Team k in every season carries centred pace ``k`` (0, 1, 2, 3), so the
    absolute gaps between the six team pairs are 1, 2, 3, 1, 2, 1 in every
    season -- a stable distribution whose 75th percentile is 2.0.
    """

    rows = [
        {"season": season, "team": team, PACE_CENTERED_COLUMN: float(index)}
        for season in SEASONS
        for index, team in enumerate(TEAMS)
    ]
    frame = pd.DataFrame(rows)
    for (season, team), value in (overrides or {}).items():
        mask = frame["season"].eq(season) & frame["team"].eq(team)
        if mask.any():
            frame.loc[mask, PACE_CENTERED_COLUMN] = value
        else:
            frame = pd.concat(
                [
                    frame,
                    pd.DataFrame([{"season": season, "team": team, PACE_CENTERED_COLUMN: value}]),
                ],
                ignore_index=True,
            )
    return frame


def _games() -> pd.DataFrame:
    """Every ordered team pair in every season: 12 games a season, 60 total."""

    rows = []
    for season in SEASONS:
        for home in TEAMS:
            for away in TEAMS:
                if home == away:
                    continue
                rows.append(
                    {
                        "game_id": f"{season}_{away}_{home}",
                        "season": season,
                        "week": 1,
                        "home_team": home,
                        "away_team": away,
                    }
                )
    return pd.DataFrame(rows)


def _flags(games: pd.DataFrame, style: pd.DataFrame) -> pd.Series:
    derived = derive_team_style_pace_features(games, team_season=style)
    return derived.set_index("game_id")[TEAM_STYLE_PACE_MISMATCH_COLUMN]


# ---------------------------------------------------------------------------
# Leakage half 1: the pace VALUES a season-S flag may read
# ---------------------------------------------------------------------------


def test_pace_values_from_season_s_or_later_cannot_change_a_season_s_flag() -> None:
    """The core leakage regression: an extreme pace value injected into season
    S (or any later season) must leave every season-S flag untouched, because
    a season-S flag reads only season S-1 pace."""

    games = _games()
    baseline = _flags(games, _style())
    injected = _flags(
        games,
        _style({(2012, "AAA"): 999.0, (2013, "BBB"): -999.0}),
    )
    season_2012 = games.loc[games["season"].eq(2012), "game_id"]
    pd.testing.assert_series_equal(baseline.loc[season_2012], injected.loc[season_2012])


def test_pace_values_from_the_prior_season_do_change_the_flag() -> None:
    """The mirror of the test above: the same extreme value one season EARLIER
    must move the season-2012 flags, so the leakage test is proving a cutoff
    and not a builder that ignores its inputs."""

    games = _games()
    baseline = _flags(games, _style())
    injected = _flags(games, _style({(2011, "AAA"): 999.0}))
    season_2012 = games.loc[games["season"].eq(2012), "game_id"]
    assert not baseline.loc[season_2012].equals(injected.loc[season_2012])


def test_a_team_with_no_prior_season_pace_is_missing_not_zero() -> None:
    """Season 2009 has no prior season at all, so every 2009 flag is NaN --
    never a 0 that would read to the model as "a measured small gap"."""

    games = _games()
    flags = _flags(games, _style())
    season_2009 = games.loc[games["season"].eq(2009), "game_id"]
    assert flags.loc[season_2009].isna().all()


# ---------------------------------------------------------------------------
# Leakage half 2: the THRESHOLD a season-S flag may be cut against
# ---------------------------------------------------------------------------


def test_the_threshold_ignores_gaps_from_season_s_and_later() -> None:
    """A run of extreme gaps introduced in season S and later must not move the
    season-S threshold. ``EEE`` carries a huge centred pace in every season, but
    only plays games in 2012 and 2013, so it creates extreme gaps in the 2012+
    games only; the 2012 flags computed with and without those extra games must
    agree on the games common to both."""

    games = _games()
    baseline = _flags(games, _style())

    extra_style = _style(
        {(season, team): value for season in SEASONS for team, value in (("EEE", 500.0),)}
    )
    extra_games = pd.concat(
        [
            games,
            pd.DataFrame(
                [
                    {
                        "game_id": f"{season}_EEE_AAA",
                        "season": season,
                        "week": 1,
                        "home_team": "AAA",
                        "away_team": "EEE",
                    }
                    for season in (2012, 2013)
                ]
            ),
        ],
        ignore_index=True,
    )
    injected = _flags(extra_games, extra_style)
    season_2012 = games.loc[games["season"].eq(2012), "game_id"]
    pd.testing.assert_series_equal(baseline.loc[season_2012], injected.loc[season_2012])


def test_the_threshold_does_move_when_prior_season_gaps_change() -> None:
    """The mirror: the same extreme games placed in seasons STRICTLY BEFORE
    2013 raise the prior-season 75th percentile, so 2013 flags that fired under
    the tighter cut stop firing. Proves the expanding threshold is real and not
    a constant."""

    games = _games()
    baseline = _flags(games, _style())

    wide_style = _style({(season, "EEE"): 500.0 for season in SEASONS})
    wide_games = pd.concat(
        [
            games,
            pd.DataFrame(
                [
                    {
                        "game_id": f"{season}_EEE_{home}",
                        "season": season,
                        "week": 1,
                        "home_team": home,
                        "away_team": "EEE",
                    }
                    for season in (2010, 2011, 2012)
                    for home in TEAMS
                ]
            ),
        ],
        ignore_index=True,
    )
    injected = _flags(wide_games, wide_style)
    season_2013 = games.loc[games["season"].eq(2013), "game_id"]
    assert baseline.loc[season_2013].sum() > 0
    assert injected.loc[season_2013].sum() < baseline.loc[season_2013].sum()


def test_the_first_season_with_pace_but_no_prior_gap_is_missing() -> None:
    """Season 2010 has prior-season pace for every team but no strictly-prior
    season carrying a defined gap, so its threshold is undefined and every 2010
    flag is NaN -- the coverage floor declared in section 6.1."""

    games = _games()
    flags = _flags(games, _style())
    season_2010 = games.loc[games["season"].eq(2010), "game_id"]
    assert flags.loc[season_2010].isna().all()


def test_the_flag_is_the_top_quartile_of_the_prior_season_gap() -> None:
    """The construct itself: with per-season gaps of 1, 2, 3, 1, 2, 1 in each
    direction the prior-season 75th percentile is 2.0, so a gap of 2 or 3 fires
    and a gap of 1 does not."""

    games = _games()
    flags = _flags(games, _style())
    assert flags.loc["2012_AAA_BBB"] == 0.0  # gap 1
    assert flags.loc["2012_AAA_CCC"] == 1.0  # gap 2
    assert flags.loc["2012_AAA_DDD"] == 1.0  # gap 3


def test_the_flag_is_symmetric_in_home_and_away() -> None:
    """The mechanism is a variance mechanism, symmetric in the sign of the gap;
    swapping home and away must not change the flag. This is the property a
    linear ridge cannot form from ``home`` and ``away`` columns."""

    games = _games()
    flags = _flags(games, _style())
    for home, away in (("AAA", "DDD"), ("BBB", "CCC")):
        assert flags.loc[f"2012_{away}_{home}"] == flags.loc[f"2012_{home}_{away}"]


# ---------------------------------------------------------------------------
# Additivity and join contracts
# ---------------------------------------------------------------------------


def test_attach_is_purely_additive() -> None:
    """Every pre-existing column comes back bit-identical and exactly the one
    named column is added -- the discipline every candidate feature module in
    this family shares."""

    games = _games()
    games["some_existing_feature"] = np.arange(len(games), dtype=float)
    widened = attach_team_style_pace_features(games, team_season=_style())

    new_columns = sorted(set(widened.columns) - set(games.columns))
    assert new_columns == sorted(TEAM_STYLE_PACE_MISMATCH_ON_PRODUCTION_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(games, widened[games.columns], check_exact=True)
    assert list(widened.index) == list(games.index)


def test_attach_requires_the_join_key() -> None:
    games = _games().drop(columns=["game_id"])
    with pytest.raises(DataContractError, match="game_id"):
        attach_team_style_pace_features(games, team_season=_style())


def test_attach_refuses_to_overwrite_an_existing_column() -> None:
    games = _games()
    games[TEAM_STYLE_PACE_MISMATCH_COLUMN] = 0.0
    with pytest.raises(DataContractError, match=TEAM_STYLE_PACE_MISMATCH_COLUMN):
        attach_team_style_pace_features(games, team_season=_style())


def test_derive_requires_every_input_column() -> None:
    games = _games().drop(columns=["away_team"])
    with pytest.raises(DataContractError, match="away_team"):
        derive_team_style_pace_features(games, team_season=_style())


def test_derive_requires_the_style_panel_columns() -> None:
    style = _style().drop(columns=[PACE_CENTERED_COLUMN])
    with pytest.raises(DataContractError, match=PACE_CENTERED_COLUMN):
        derive_team_style_pace_features(_games(), team_season=style)


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
        attach_team_style_pace_features(games, team_season=_style())


def test_a_duplicated_team_season_row_raises() -> None:
    """The style panel is one row per (season, team); a duplicate would fan the
    game table out silently, so ``validate="many_to_one"`` must refuse it."""

    style = pd.concat([_style(), _style().iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        derive_team_style_pace_features(_games(), team_season=style)
