"""Pins the bucket arithmetic of ``scripts/big_spread_diagnosis.py`` on a synthetic frame."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from big_spread_diagnosis import (
    build_frame,
    cell_row,
    cut_table,
    residual_information_table,
    week_blocked_mean,
    week_blocked_slope,
)


def synthetic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Six 10.5+ games over three weeks plus one 0-3 game and one push.

    Game / home line / served residual / result / pick(home) / correct:
      a  +13.0  -2.0  +10  road  road covers (home short by 3)      -> 1
      b  +13.0  -2.0  +20  road  home covers                          -> 0
      c  +10.5  +3.0  +14  home  home covers                          -> 1
      d  -10.5  -1.0  -20  road  road (home dog loses by 20)          -> 1 (road favourite covers)
      e  -11.0  +2.0   -3  home  home dog covers                      -> 1
      f  +14.0  +1.0  +14  home  push                                 -> excluded
      g   +3.0  +1.0   +7  home  home covers                          -> 1 (0-3 bucket)
    """

    per_game = pd.DataFrame(
        {
            "game_id": list("abcdefg"),
            "season": [2024] * 7,
            "week": [1, 1, 2, 2, 3, 3, 3],
            "tue_open_home_spread": [13.0, 13.0, 10.5, -10.5, -11.0, 14.0, 3.0],
            "close_home_spread": [13.5, 12.0, 10.5, -11.0, -11.0, 14.0, 2.5],
            "result": [10.0, 20.0, 14.0, -20.0, -3.0, 14.0, 7.0],
            "residual_at_open": [-2.5, -2.5, 2.5, -1.5, 1.5, 0.5, 1.0],
            "residual_at_open_served": [-2.0, -2.0, 3.0, -1.0, 2.0, 1.0, 1.0],
            "home_side_offset_at_open": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
            "home_cover_probability_at_open": [0.42, 0.42, 0.61, 0.46, 0.57, 0.53, 0.54],
            "pick_home_at_open_probability_rule": [False, False, True, False, True, True, True],
            "correct_at_open_probability_rule": [1.0, 0.0, 1.0, 1.0, 1.0, np.nan, 1.0],
            "pick_home_at_open_probability_rule_raw": [False, False, True, False, True, True, True],
            "correct_at_open_probability_rule_raw": [1.0, 0.0, 1.0, 1.0, 1.0, np.nan, 1.0],
        }
    )
    per_game["margin_vs_open"] = per_game["result"] - per_game["tue_open_home_spread"]
    per_game["open_move"] = per_game["close_home_spread"] - per_game["tue_open_home_spread"]
    features = pd.DataFrame(
        {
            "game_id": list("abcdefg"),
            "gameday": pd.Timestamp("2024-09-08"),
            "weekday": "Sunday",
            "gametime": ["13:00", "20:20", "13:00", "16:25", "13:00", "13:00", "13:00"],
            "location": ["Home"] * 7,
            "div_game": [0, 1, 0, 0, 1, 0, 0],
            "neutral_site": [0] * 7,
            "elo_diff": [150.0, 150.0, 120.0, -120.0, -130.0, 160.0, 30.0],
            "home_qb_start_probability": [1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0],
            "away_qb_start_probability": [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    schedules = pd.DataFrame(
        {
            "game_id": list("abcdefg"),
            "home_rest": [7, 7, 10, 7, 6, 7, 7],
            "away_rest": [7, 6, 7, 7, 7, 7, 7],
            "roof": ["outdoors", "dome", "outdoors", "closed", "outdoors", "open", "outdoors"],
            "surface": ["grass"] * 7,
            "home_team": ["KC", "KC", "BUF", "NYJ", "CAR", "SF", "DAL"],
            "away_team": ["DEN", "LV", "MIA", "SF", "DAL", "LA", "NYG"],
        }
    )
    return per_game, features, schedules


def test_build_frame_derives_the_cut_columns_with_the_nflverse_sign_convention() -> None:
    frame = build_frame(*synthetic()).set_index("game_id")
    assert frame.loc["a", "bucket"] == "10.5+" and frame.loc["g", "bucket"] == "0-3"
    assert frame.loc["a", "home_side"] == "home favourite"
    assert frame.loc["d", "home_side"] == "home underdog"
    assert frame.loc["a", "pick_side"] == "picked underdog"
    assert frame.loc["d", "pick_side"] == "picked favourite"
    assert frame.loc["d", "pick_location"] == "picked the road team"
    assert frame.loc["a", "point_served"] == pytest.approx(11.0)
    assert frame.loc["a", "error_served"] == pytest.approx(-1.0)
    assert frame.loc["a", "error_raw"] == pytest.approx(-0.5)
    assert frame.loc["d", "error_served"] == pytest.approx(-8.5)
    assert frame.loc["d", "error_served_favourite"] == pytest.approx(8.5)
    assert frame.loc["d", "error_served_pick"] == pytest.approx(8.5)
    assert frame.loc["a", "move_band"] == "market moved against the model"
    assert frame.loc["b", "move_band"] == "market moved with the model"
    assert frame.loc["c", "move_band"] == "no move"
    assert frame.loc["c", "favourite_rest_band"] == "long (8 or more)"
    assert frame.loc["d", "favourite_qb_band"] == "starter expected (95%+)"
    assert frame.loc["d", "underdog_qb_band"] == "starter in doubt (under 95%)"
    assert frame.loc["b", "favourite_rest_edge"] == "favourite rested more"
    assert frame.loc["d", "favourite_travel"] == "two or more zones"
    assert frame.loc["a", "favourite_travel"] == "none"
    assert frame.loc["b", "primetime"] == "primetime" and frame.loc["b", "roof_band"] == "indoors"
    assert frame.loc["a", "line_band"] == "13-14" and frame.loc["c", "line_band"] == "10.5-12.5"
    assert frame.loc["a", "key_band"] == "between (nearest 14)"
    assert frame.loc["f", "key_band"] == "on or beside 14"
    assert frame.loc["a", "residual_lean"] == "leans underdog"
    assert frame.loc["d", "residual_lean"] == "leans favourite"
    assert bool(frame.loc["f", "push"]) is True


def test_cell_row_excludes_pushes_from_accuracy_but_not_from_the_point_error() -> None:
    frame = build_frame(*synthetic())
    big = frame.loc[frame["bucket"].eq("10.5+")]
    row = cell_row(big, draws=200)
    assert row["games"] == 6 and row["decided"] == 5
    assert row["accuracy"] == pytest.approx(4 / 5)
    assert row["error_served"] == pytest.approx(5.0 / 6.0)
    assert row["favourite_pick_rate"] == pytest.approx(2 / 5)
    assert row["home_cover_rate"] == pytest.approx(3 / 5)
    assert row["stated_confidence"] == pytest.approx((0.58 + 0.58 + 0.61 + 0.54 + 0.57) / 5)
    assert row["error_served_lower"] <= row["error_served"] <= row["error_served_upper"]


def test_cut_table_reports_every_level_and_the_bucket_total() -> None:
    frame = build_frame(*synthetic())
    cells = cut_table(frame, ("10.5+",), (("pick_location", "home / road"),), draws=100)
    assert list(cells["level"]) == ["all", "picked the home team", "picked the road team"]
    road = cells.set_index("level").loc["picked the road team"]
    assert road["decided"] == 3 and road["accuracy"] == pytest.approx(2 / 3)


def test_week_blocked_helpers_are_seeded_and_degenerate_on_constants() -> None:
    values = pd.Series([2.0, 2.0, 2.0, 2.0])
    blocks = pd.Series(["2024-1", "2024-1", "2024-2", "2024-3"])
    first = week_blocked_mean(values, blocks, draws=50, seed=1)
    assert first["estimate"] == first["lower"] == first["upper"] == 2.0
    assert first["probability_positive"] == 1.0
    varied = pd.Series([1.0, -1.0, 3.0, 5.0])
    assert week_blocked_mean(varied, blocks, draws=50, seed=7) == week_blocked_mean(
        varied, blocks, draws=50, seed=7
    )
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    slope = week_blocked_slope(
        x, 2.0 * x, pd.Series(["w1", "w1", "w2", "w2", "w3", "w3"]), draws=50, seed=3
    )
    assert slope["estimate"] == pytest.approx(2.0)
    assert slope["probability_positive"] == 1.0


def test_residual_information_table_uses_decided_games_only() -> None:
    frame = build_frame(*synthetic())
    info = residual_information_table(frame, ("10.5+", "0-3"), draws=100).set_index("bucket")
    assert info.loc["10.5+", "decided"] == 5
    assert info.loc["10.5+", "always_home_cover_rate"] == pytest.approx(3 / 5)
    assert info.loc["10.5+", "model_minus_always_home"] == pytest.approx(0.8 - 0.6)
    assert np.isnan(info.loc["0-3", "slope_served"])
