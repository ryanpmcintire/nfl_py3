"""Pregame and exact-season leakage contracts for PER-14."""

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.unit_prior_features import UNIT_PRIOR_COLUMNS, attach_unit_prior_features


def fixture():
    games = pd.DataFrame(
        {
            "season": [2021, 2022],
            "home_team": ["BUF", "BUF"],
            "away_team": ["KC", "KC"],
            "prediction_timestamp": ["2021-09-01", "2022-09-01"],
        },
        index=[7, 3],
    )
    ratings = pd.DataFrame(
        [
            {
                "season": 2020,
                "team": team,
                "unit": unit,
                "rating": rating,
                "finalized_at": "2021-03-01",
            }
            for team, rating in (("BUF", 3.0), ("KC", 1.0))
            for unit in ("OFF_OL", "OFF_SKILL")
        ]
    )
    return games, ratings


def test_exact_previous_season_and_index():
    games, ratings = fixture()
    result = attach_unit_prior_features(games, ratings)
    assert result.index.tolist() == [7, 3]
    assert result.loc[7, list(UNIT_PRIOR_COLUMNS)].tolist() == [2.0, 2.0]
    assert result.loc[3, list(UNIT_PRIOR_COLUMNS)].isna().all()
    pd.testing.assert_frame_equal(result[games.columns], games)


def test_current_and_future_season_cannot_enter():
    games, ratings = fixture()
    games = games.iloc[:1]
    original = attach_unit_prior_features(games, ratings)
    for season in (2021, 2022, 2026):
        poison = ratings.assign(season=season, rating=999999.0)
        changed = attach_unit_prior_features(games, pd.concat([ratings, poison]))
        pd.testing.assert_frame_equal(original, changed)


@pytest.mark.parametrize("finalized", ["2021-09-01", "2021-09-02", "2025-01-01"])
def test_unavailable_previous_season_rating_cannot_enter(finalized):
    games, ratings = fixture()
    ratings.loc[ratings.team.eq("BUF"), "finalized_at"] = finalized
    result = attach_unit_prior_features(games, ratings)
    assert result[list(UNIT_PRIOR_COLUMNS)].isna().all().all()


def test_missing_team_and_empty_ratings_stay_missing():
    games, ratings = fixture()
    for source in (ratings.iloc[:0], ratings[ratings.team.eq("BUF")]):
        result = attach_unit_prior_features(games, source)
        assert result[list(UNIT_PRIOR_COLUMNS)].isna().all().all()


def test_duplicate_rating_rejected():
    games, ratings = fixture()
    with pytest.raises(DataContractError, match="duplicate"):
        attach_unit_prior_features(games, pd.concat([ratings, ratings.iloc[:1]]))


@pytest.mark.parametrize(
    "column,value",
    [
        ("season", 2020.5),
        ("rating", float("inf")),
        ("finalized_at", "invalid"),
        ("team", None),
    ],
)
def test_invalid_rating_rejected(column, value):
    games, ratings = fixture()
    ratings[column] = value
    with pytest.raises(DataContractError):
        attach_unit_prior_features(games, ratings)


def test_prediction_timestamp_required():
    games, ratings = fixture()
    with pytest.raises(DataContractError, match="missing columns"):
        attach_unit_prior_features(games.drop(columns="prediction_timestamp"), ratings)
