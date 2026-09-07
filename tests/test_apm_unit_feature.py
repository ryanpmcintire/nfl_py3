"""Leakage and additive-profile contracts for PER-09."""

import numpy as np
import pandas as pd
import pytest

from nfl_ats.apm_unit_feature import APM_UNIT_COLUMNS, attach_apm_unit_features, fit_unit_ratings
from nfl_ats.data import DataContractError
from nfl_ats.margin import margin_feature_columns


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = pd.DataFrame(
        [
            ["early", 2021, "A", "B", "2021-09-07T12:00Z"],
            ["later", 2021, "A", "B", "2021-09-21T12:00Z"],
        ],
        columns=["game_id", "season", "home_team", "away_team", "decision_timestamp"],
    )
    records = []
    for game, year, week, completed, shift in (
        ("seed", 2020, 17, "2021-01-04T12:00Z", 0),
        ("future", 2021, 1, "2021-09-14T12:00Z", 2),
    ):
        for i in range(24):
            home = i % 2 == 0
            records.append(
                {
                    "game_id": game,
                    "season": year,
                    "week": week,
                    "completed_at": completed,
                    "posteam": "A" if home else "B",
                    "defteam": "B" if home else "A",
                    "offense_players": "a;b" if home else "c;d",
                    "defense_players": "e;f" if home else "g;h",
                    "epa": (1.0 if home else -1.0) + shift * (i % 3),
                }
            )
    rosters = pd.DataFrame(
        [
            {"gsis_id": p, "season": y, "week": w, "position": "QB" if p < "e" else "CB"}
            for y, w in [(2020, 17), (2021, 1), (2021, 10)]
            for p in "abcdefgh"
        ]
    )
    return games, pd.DataFrame(records), rosters


def test_future_plays_and_future_rosters_cannot_move_earlier_rating() -> None:
    games, plays, rosters = fixture()
    base = attach_apm_unit_features(games, plays, rosters)
    plays.loc[plays.game_id.eq("future"), "epa"] = -4.9
    rosters.loc[rosters.week.eq(10), "position"] = "OL"
    changed = attach_apm_unit_features(games, plays, rosters)
    pd.testing.assert_series_equal(base.iloc[0], changed.iloc[0])
    assert not np.allclose(
        base.loc[1, list(APM_UNIT_COLUMNS)].astype(float),
        changed.loc[1, list(APM_UNIT_COLUMNS)].astype(float),
    )
    pd.testing.assert_frame_equal(base[games.columns], games)


def test_seed_uses_only_completed_prior_season_games_and_strict_boundary() -> None:
    games, plays, rosters = fixture()
    seed = plays.loc[plays.season.eq(2020)].copy()
    expected = attach_apm_unit_features(games.iloc[:1], seed, rosters)
    late = seed.assign(game_id="unfinished", epa=4.9, completed_at=games.decision_timestamp.iloc[0])
    unknown = late.assign(game_id="unknown", completed_at=None)
    actual = attach_apm_unit_features(games.iloc[:1], pd.concat([seed, late, unknown]), rosters)
    pd.testing.assert_frame_equal(expected, actual)
    assert actual.home_apm_off_rating.iloc[0] == pytest.approx(
        fit_unit_ratings(seed, rosters)["A", "off"][0]
    )


def test_missing_history_determinism_and_diffs() -> None:
    games, plays, rosters = fixture()
    empty = attach_apm_unit_features(games, plays.iloc[:0], rosters)
    assert empty[list(APM_UNIT_COLUMNS)].isna().all().all()
    a = attach_apm_unit_features(games, plays, rosters)
    b = attach_apm_unit_features(games, plays, rosters)
    pd.testing.assert_frame_equal(a, b)
    assert a.apm_off_rating_diff.tolist() == pytest.approx(
        (a.home_apm_off_rating - a.away_apm_off_rating).tolist()
    )


def test_invalid_timestamps_and_duplicate_games_fail_closed() -> None:
    games, plays, rosters = fixture()
    with pytest.raises(DataContractError, match="timestamp"):
        attach_apm_unit_features(games.assign(decision_timestamp=None), plays, rosters)
    with pytest.raises(DataContractError, match="unique"):
        attach_apm_unit_features(pd.concat([games, games]), plays, rosters)


def test_profile_is_exactly_incumbent_plus_six_columns() -> None:
    assert tuple(margin_feature_columns("market_residual", "weak_stack_apm_unit")) == (
        *margin_feature_columns("market_residual", "weak_stack"),
        *APM_UNIT_COLUMNS,
    )
