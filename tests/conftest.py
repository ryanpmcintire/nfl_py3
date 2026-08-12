from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import GRAPH_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS


@pytest.fixture
def schedules_and_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2022-09-11", periods=5, freq="7D")
    schedules = pd.DataFrame(
        {
            "game_id": [f"2022_{week:02d}_B_A" for week in range(1, 6)],
            "season": 2022,
            "game_type": "REG",
            "week": range(1, 6),
            "gameday": dates,
            "away_team": "B",
            "home_team": "A",
            "away_score": [17, 24, 14, 27, 20],
            "home_score": [20, 20, 21, 25, 20],
            "result": [3, -4, 7, -2, 0],
            "spread_line": [1.0, 2.0, 3.0, -3.0, 0.0],
            "total_line": [42.5, 44.0, 41.5, 47.0, 43.0],
            "home_rest": 7,
            "away_rest": 7,
            "location": "Home",
            "div_game": 1,
            "temp": 65.0,
            "wind": 8.0,
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
        }
    )
    rows: list[dict[str, object]] = []
    for game_index, game in schedules.iterrows():
        for team, direction in (("A", 1.0), ("B", -1.0)):
            rows.append(
                {
                    "game_id": game["game_id"],
                    "season": 2022,
                    "season_type": "REG",
                    "week": game["week"],
                    "team": team,
                    "attempts": 30 + game_index,
                    "carries": 25 - game_index,
                    "passing_epa": direction * (3.0 + game_index),
                    "rushing_epa": direction * (1.0 + game_index / 2),
                    "passing_cpoe": direction * (2.0 + game_index),
                    "passing_yards": 220 + direction * 10 + game_index,
                    "rushing_yards": 105 + direction * 5 + game_index,
                    "sacks_suffered": 2,
                    "passing_interceptions": int(direction < 0),
                    "sack_fumbles_lost": 0,
                    "rushing_fumbles_lost": 0,
                    "receiving_fumbles_lost": 0,
                }
            )
    return schedules, pd.DataFrame(rows)


@pytest.fixture
def model_frame() -> pd.DataFrame:
    rows = 160
    start = date(2018, 9, 1)
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "game_id": [f"game_{value:03d}" for value in index],
            "season": np.where(index < 100, 2019, 2020),
            "week": np.where(index < 100, (index // 10) + 1, ((index - 100) // 15) + 1),
            "gameday": [start + timedelta(days=int(value)) for value in index],
            "away_team": "AWY",
            "home_team": "HME",
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
        }
    )
    all_features = (*MODEL_FEATURE_COLUMNS, *GRAPH_FEATURE_COLUMNS)
    for feature_index, column in enumerate(all_features, start=1):
        frame[column] = np.sin(index / feature_index) + (index % 5) / 10.0
    frame["spread_line"] = np.where(index % 2 == 0, 2.5, -2.5)
    frame["home_cover"] = (index % 3 != 0).astype(float)
    frame["ats_margin"] = np.where(frame["home_cover"].eq(1), 3.0, -3.0)
    frame["result"] = frame["spread_line"] + frame["ats_margin"]
    return frame
