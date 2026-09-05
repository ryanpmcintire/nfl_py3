"""Quarter-boundary correctness and schedule-only flag leakage contract."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.altitude_split_features import denver_home_flag, quarter_margins


def games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2020, 2020],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "home_team": ["DEN", "KC"],
            "result": [10, -3],
        }
    )


def plays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["a", "a", "b", "b"],
            "play_id": [1, 2, 1, 2],
            "season_type": ["REG"] * 4,
            "qtr": [4, 4, 4, 5],
            "posteam": ["KC", "DEN", "KC", "DEN"],
            "home_team": ["DEN", "DEN", "KC", "KC"],
            "away_team": ["KC", "KC", "DEN", "DEN"],
            "score_differential": [-3, 10, 7, 0],
            "game_seconds_remaining": [900, 0, 900, 0],
        }
    )


def test_boundary_sign_and_overtime_exclusion() -> None:
    result = quarter_margins(plays(), games()).set_index("game_id")
    assert result.loc["a", "first_three_margin"] == 3
    assert result.loc["a", "fourth_margin"] == 7
    assert result.loc["b", "fourth_margin"] == -7  # OT final -3 is not regulation.


def test_missing_boundary_is_not_a_later_score() -> None:
    pbp = plays()
    pbp.loc[0, "game_seconds_remaining"] = 895
    assert quarter_margins(pbp, games()).game_id.tolist() == ["b"]


def test_missing_ot_score_does_not_use_final() -> None:
    pbp = plays()
    pbp.loc[3, "score_differential"] = float("nan")
    assert quarter_margins(pbp, games()).game_id.tolist() == ["a"]


def test_leakage_outcomes_and_future_rows_cannot_change_flag() -> None:
    original = games()
    expected = denver_home_flag(original)
    changed = original.assign(result=[-99, 99], fourth_margin=[999, -999])
    changed = pd.concat(
        [changed, original.assign(game_id=["future_a", "future_b"], season=2027)], ignore_index=True
    )
    pd.testing.assert_series_equal(denver_home_flag(changed).iloc[:2], expected)
    assert expected.tolist() == [True, False]


def test_missing_team_fails_closed() -> None:
    with pytest.raises(ValueError, match="Missing home"):
        denver_home_flag(pd.DataFrame({"home_team": [None]}))


def test_overlay_preserves_production_and_excludes_pushes() -> None:
    spec = importlib.util.spec_from_file_location(
        "cx15", Path("scripts/coaching_altitude_leads_on_production.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2020, 2020],
            "week": [1, 2],
            "margin_vs_open": [3, -1],
            "pick_home_at_open_probability_rule": [False, True],
        }
    )
    result = module.paired_overlay(predictions, games())
    assert result.candidate_pick_home.tolist() == [True, True]
    assert result.delta.tolist() == [100, 0]
    predictions.loc[0, "margin_vs_open"] = -3
    result = module.paired_overlay(predictions, games())
    assert result.candidate_pick_home.tolist() == [True, True]
    assert result.oracle_pick_home.tolist() == [False, True]
    assert result.delta.tolist() == [-100, 0]
    predictions.loc[0, "margin_vs_open"] = 0
    assert module.paired_overlay(predictions, games()).game_id.tolist() == ["b"]
