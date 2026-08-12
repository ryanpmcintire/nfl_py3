from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.constants import (
    GRAPH_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    STATE_METRICS,
)
from nfl_ats.data import DataContractError
from nfl_ats.features import (
    add_ats_outcomes,
    attach_team_states,
    build_game_features,
    build_team_game_metrics,
)


def test_ats_target_sign_and_push() -> None:
    frame = pd.DataFrame({"result": [7, -1, 3], "spread_line": [3, 2, 3]})
    result = add_ats_outcomes(frame)
    assert result["ats_margin"].tolist() == [4.0, -3.0, 0.0]
    assert result["home_cover"].iloc[:2].tolist() == [1.0, 0.0]
    assert pd.isna(result["home_cover"].iloc[2])


def test_feature_table_has_one_row_per_game_and_no_label_features(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    schedules = schedules.assign(gametime="20:20", weekday="Sunday")
    features = build_game_features(schedules, stats, span=3, min_periods=1)
    assert features["game_id"].is_unique
    assert len(features) == len(schedules)
    assert set(MODEL_FEATURE_COLUMNS).issubset(features.columns)
    assert set(GRAPH_FEATURE_COLUMNS).issubset(features.columns)
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(OUTCOME_COLUMNS)
    assert features.loc[0, "elo_diff"] == pytest.approx(55.0)
    assert features.loc[0, "kickoff"] == pd.Timestamp("2022-09-12 00:20:00+00:00")


def test_current_game_stats_cannot_change_current_pregame_features(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    baseline = build_game_features(schedules, stats, span=3, min_periods=1)
    changed_stats = stats.copy()
    second_game = schedules.loc[1, "game_id"]
    mask = changed_stats["game_id"].eq(second_game) & changed_stats["team"].eq("A")
    changed_stats.loc[mask, "passing_epa"] = 1_000.0
    changed = build_game_features(schedules, changed_stats, span=3, min_periods=1)

    column = "home_off_pass_epa_per_play"
    assert changed.loc[1, column] == pytest.approx(baseline.loc[1, column])
    assert changed.loc[2, column] != pytest.approx(baseline.loc[2, column])


def test_team_metric_builder_requires_two_teams(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    malformed = stats.loc[
        ~((stats["game_id"] == schedules.loc[0, "game_id"]) & (stats["team"] == "B"))
    ]
    with pytest.raises(DataContractError, match="Expected two"):
        build_team_game_metrics(schedules, malformed)


def test_historical_franchise_abbreviations_share_current_team_state(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    historical = schedules.copy()
    historical.loc[:, "home_team"] = "OAK"
    historical.loc[:, "away_team"] = "STL"
    current_stats = stats.copy()
    current_stats.loc[current_stats["team"].eq("A"), "team"] = "LV"
    current_stats.loc[current_stats["team"].eq("B"), "team"] = "LA"

    features = build_game_features(
        historical,
        current_stats,
        span=3,
        min_periods=1,
        graph_min_games=2,
    )
    assert features["home_team"].eq("LV").all()
    assert features["away_team"].eq("LA").all()
    assert features.loc[1:, "home_off_epa_per_play"].notna().all()
    assert features.loc[1:, "away_off_epa_per_play"].notna().all()


def test_offseason_state_regresses_and_season_game_count_resets() -> None:
    rows = []
    for team, state_value in (("A", 10.0), ("B", 0.0)):
        row: dict[str, object] = {
            "game_id": f"2022_18_{team}",
            "season": 2022,
            "gameday": pd.Timestamp("2023-01-08"),
            "team": team,
            "team_games": 17,
        }
        for metric in STATE_METRICS:
            row[f"state_{metric}"] = state_value
            row[f"league_mean_{metric}"] = 5.0
        rows.append(row)
    states = pd.DataFrame(rows)
    game = pd.DataFrame(
        {
            "game_id": ["2023_01_B_A"],
            "season": [2023],
            "gameday": [pd.Timestamp("2023-09-10")],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )
    attached = attach_team_states(game, states, offseason_retention=0.5)
    assert attached.loc[0, "home_team_games"] == 0
    assert attached.loc[0, "away_team_games"] == 0
    assert attached.loc[0, "home_off_epa_per_play"] == pytest.approx(7.5)
    assert attached.loc[0, "away_off_epa_per_play"] == pytest.approx(2.5)
    assert attached.loc[0, "diff_off_epa_per_play"] == pytest.approx(5.0)
