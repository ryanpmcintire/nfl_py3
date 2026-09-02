"""ENV-04 deterministic rest-context and leakage contracts."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nfl_ats.constants import FEATURE_FAMILIES, FEATURE_SETS, REST_CONTEXT_FEATURE_COLUMNS
from nfl_ats.data import DataContractError
from nfl_ats.rest_context import add_rest_context_features, build_rest_context_features


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("g1", 2021, "2021-09-05", "A", "B", "Home"),
            ("g2", 2021, "2021-09-09", "C", "A", "Home"),
            ("g3", 2021, "2021-09-19", "D", "A", "Home"),
            ("g4", 2021, "2021-09-26", "E", "A", "Home"),
            ("g5", 2021, "2021-10-10", "A", "B", "Home"),
            ("g6", 2021, "2021-10-17", "A", "C", "Neutral"),
            ("g7", 2021, "2021-10-24", "F", "A", "Home"),
            ("g8", 2022, "2022-09-11", "A", "B", "Home"),
        ],
        columns=["game_id", "season", "gameday", "home_team", "away_team", "location"],
    ).assign(
        result=[7.0, -3.0, 10.0, -6.0, 2.0, math.nan, math.nan, math.nan],
        spread_line=[3.0, -1.0, 2.0, 4.0, -2.5, math.nan, math.nan, math.nan],
        temp=[72.0, 80.0, 60.0, 65.0, 55.0, math.nan, math.nan, math.nan],
    )


def _by_game(features: pd.DataFrame, game_id: str) -> pd.Series:
    return features.loc[features["game_id"].eq(game_id)].iloc[0]


def test_family_is_registered_but_not_admitted_to_any_model_profile() -> None:
    assert FEATURE_FAMILIES["rest_context"] == REST_CONTEXT_FEATURE_COLUMNS
    for profile, columns in FEATURE_SETS.items():
        assert set(columns).isdisjoint(REST_CONTEXT_FEATURE_COLUMNS), profile


def test_known_rest_bye_short_week_mini_bye_and_road_streak_values() -> None:
    features = build_rest_context_features(_schedules())

    opener = _by_game(features, "g1")
    assert pd.isna(opener["rest_home_days"])
    assert pd.isna(opener["rest_away_days"])
    assert pd.isna(opener["rest_days_diff"])
    assert pd.isna(opener["rest_home_off_bye"])

    short = _by_game(features, "g2")
    assert short["rest_away_days"] == 4.0
    assert short["rest_away_short_week"] == 1.0
    assert short["rest_away_mini_bye"] == 0.0
    assert short["rest_away_consecutive_road_games"] == 1.0

    mini_bye = _by_game(features, "g3")
    assert mini_bye["rest_away_days"] == 10.0
    assert mini_bye["rest_away_mini_bye"] == 1.0
    assert mini_bye["rest_away_short_week"] == 0.0
    assert mini_bye["rest_away_consecutive_road_games"] == 2.0

    third_road = _by_game(features, "g4")
    assert third_road["rest_away_consecutive_road_games"] == 3.0

    home_bye = _by_game(features, "g5")
    assert home_bye["rest_home_days"] == 14.0
    assert home_bye["rest_home_off_bye"] == 1.0
    assert home_bye["rest_home_mini_bye"] == 0.0
    assert home_bye["rest_days_diff"] == -21.0


def test_neutral_game_breaks_true_road_streak_and_season_resets_history() -> None:
    schedules = _schedules()
    schedules.loc[schedules["game_id"].eq("g5"), ["home_team", "away_team"]] = ["F", "A"]
    features = build_rest_context_features(schedules)

    assert _by_game(features, "g5")["rest_away_consecutive_road_games"] == 4.0
    assert _by_game(features, "g6")["rest_away_consecutive_road_games"] == 0.0
    assert _by_game(features, "g7")["rest_away_consecutive_road_games"] == 1.0
    assert pd.isna(_by_game(features, "g8")["rest_home_days"])
    assert pd.isna(_by_game(features, "g8")["rest_away_days"])


def test_postgame_and_future_row_mutations_cannot_change_decision_row() -> None:
    """Required leakage regression for the ENV-04 feature family."""

    schedules = _schedules()
    baseline = _by_game(build_rest_context_features(schedules), "g4")

    changed = schedules.copy()
    changed.loc[changed["game_id"].eq("g4"), ["result", "spread_line", "temp"]] = [
        -1000.0,
        1000.0,
        -1000.0,
    ]
    future = changed["game_id"].eq("g6")
    changed.loc[future, ["result", "spread_line", "temp"]] = [1000.0, -1000.0, 1000.0]
    changed.loc[future, ["gameday", "home_team", "away_team", "location"]] = [
        "2021-11-28",
        "Z",
        "Y",
        "Home",
    ]
    mutated = _by_game(build_rest_context_features(changed), "g4")

    pd.testing.assert_series_equal(
        mutated[[*REST_CONTEXT_FEATURE_COLUMNS]],
        baseline[[*REST_CONTEXT_FEATURE_COLUMNS]],
        check_names=False,
        check_exact=True,
    )


def test_input_order_cannot_change_features() -> None:
    schedules = _schedules()
    baseline = build_rest_context_features(schedules)
    shuffled = build_rest_context_features(
        schedules.sample(frac=1.0, random_state=19).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(shuffled, baseline)


def test_attach_preserves_rows_and_records_provenance() -> None:
    games = pd.DataFrame({"game_id": ["g4", "g2"], "existing": [4, 2]})
    result = add_rest_context_features(games, _schedules())

    assert result["game_id"].tolist() == ["g4", "g2"]
    assert result["existing"].tolist() == [4, 2]
    assert result.attrs["rest_context_provenance"]["outcome_columns_read"] == []
    assert result.attrs["rest_context_provenance"]["schedule_columns_read"] == [
        "game_id",
        "season",
        "gameday",
        "home_team",
        "away_team",
        "location",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"game_id": ""}, "blank game_id"),
        ({"season": 2021.5}, "invalid season"),
        ({"gameday": "not-a-date"}, "invalid gameday"),
        ({"location": "Somewhere"}, "invalid location values"),
        ({"away_team": "A"}, "identical home and away teams"),
    ],
)
def test_schedule_contract_rejects_invalid_rows(mutation: dict[str, object], message: str) -> None:
    schedules = _schedules()
    for column, value in mutation.items():
        schedules[column] = schedules[column].astype(object)
        schedules.loc[0, column] = value
    with pytest.raises(DataContractError, match=message):
        build_rest_context_features(schedules)


def test_schedule_contract_rejects_duplicate_team_date() -> None:
    schedules = pd.concat([_schedules(), _schedules().iloc[[0]]], ignore_index=True)
    schedules.loc[len(schedules) - 1, "game_id"] = "other"
    with pytest.raises(DataContractError, match="multiple games for one team"):
        build_rest_context_features(schedules)
