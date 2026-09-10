import numpy as np
import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.constants import FEATURE_FAMILIES, FEATURE_SETS
from nfl_ats.data import DataContractError
from nfl_ats.home_dog_location import (
    HOME_DOG_HINGE_COLUMNS,
    HOME_DOG_POINTS_COLUMNS,
    attach_home_dog_location,
)
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, fit_margin_model, margin_feature_columns

PROFILES = {
    "weak_stack_home_dog_points": HOME_DOG_POINTS_COLUMNS,
    "weak_stack_home_dog_hinge_7": HOME_DOG_HINGE_COLUMNS,
}


def test_home_dog_points_and_hinge_follow_the_home_line_orientation() -> None:
    frame = pd.DataFrame({"spread_line": [3.0, 0.0, -3.5, -7.0, -10.5, np.nan]})
    actual = attach_home_dog_location(frame)
    assert actual.home_dog_points.tolist()[:5] == [0.0, 0.0, 3.5, 7.0, 10.5]
    assert actual.home_dog_hinge_7.tolist()[:5] == [0.0, 0.0, 0.0, 0.0, 3.5]
    assert actual.loc[5, list(HOME_DOG_HINGE_COLUMNS)].isna().all()
    assert frame.columns.tolist() == ["spread_line"], "input frame is never mutated"


def test_future_games_and_results_cannot_move_an_earlier_row() -> None:
    frame = pd.DataFrame(
        {
            "game_id": [f"game_{i}" for i in range(6)],
            "gameday": pd.date_range("2024-09-05", periods=6, freq="7D"),
            "spread_line": [-9.5, 4.0, -1.5, -12.0, 6.5, -7.5],
            "result": [-3.0, 10.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    expected = attach_home_dog_location(frame)
    changed = pd.concat(
        [frame, pd.DataFrame({"game_id": ["future"], "spread_line": [-100.0], "result": [999.0]})],
        ignore_index=True,
    )
    changed.loc[3:, ["spread_line", "result"]] = [[50.0, -40.0]] * 4
    pd.testing.assert_frame_equal(
        expected.iloc[:3].reset_index(drop=True),
        attach_home_dog_location(changed).iloc[:3][expected.columns].reset_index(drop=True),
    )


@pytest.mark.parametrize(("profile", "columns"), PROFILES.items())
def test_profiles_are_additive_on_weak_stack_and_leave_the_incumbent_alone(
    profile: str, columns: tuple[str, ...]
) -> None:
    assert profile in MARGIN_FEATURE_PROFILES
    assert margin_feature_columns("market_residual", profile) == (
        *margin_feature_columns("market_residual", "weak_stack"),
        *columns,
    )
    for prefix in ("football", "full"):
        assert FEATURE_SETS[f"{prefix}_{profile}"] == (
            *FEATURE_SETS[f"{prefix}_weak_stack"],
            *columns,
        )
        assert set(FEATURE_SETS[f"{prefix}_weak_stack"]).isdisjoint(HOME_DOG_HINGE_COLUMNS)
    assert FEATURE_FAMILIES["home_dog_location_points"] == HOME_DOG_POINTS_COLUMNS
    assert FEATURE_FAMILIES["home_dog_location_hinge"] == HOME_DOG_HINGE_COLUMNS[1:]


def test_fit_requires_the_attached_columns(model_frame: pd.DataFrame) -> None:
    with pytest.raises(DataContractError, match="home_dog_points"):
        fit_margin_model(
            model_frame, target="market_residual", feature_profile="weak_stack_home_dog_points"
        )


@pytest.mark.parametrize("command", ["margin-predict", "margin-backtest", "opener-evaluation"])
@pytest.mark.parametrize("profile", PROFILES)
def test_cli_accepts_the_research_profiles(command: str, profile: str) -> None:
    argv = [command, "--feature-profile", profile]
    if command == "margin-predict":
        argv += ["--season", "2026", "--week", "1"]
    assert cli.build_parser().parse_args(argv).feature_profile == profile
