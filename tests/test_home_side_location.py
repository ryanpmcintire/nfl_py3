import numpy as np
import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.constants import FEATURE_FAMILIES, FEATURE_SETS
from nfl_ats.data import DataContractError
from nfl_ats.home_side_location import (
    HOME_SIDE_HINGE_COLUMNS,
    PRIOR_WEIGHT_GAMES,
    attach_home_side_location,
    fit_home_side_offsets,
    gaussian_median_cover_probability,
    prior_games_for_week,
    walk_forward_home_offsets,
)
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, fit_margin_model, margin_feature_columns

PROFILE = "weak_stack_home_side_hinge_7"


def test_home_side_hinge_is_symmetric_in_the_spread() -> None:
    frame = pd.DataFrame({"spread_line": [3.0, 0.0, -3.5, 7.0, -7.0, 10.5, -10.5, np.nan]})
    actual = attach_home_side_location(frame)
    assert actual.home_side_hinge_7.tolist()[:7] == [0.0, 0.0, 0.0, 0.0, 0.0, 3.5, 3.5]
    assert np.isnan(actual.loc[7, "home_side_hinge_7"])
    assert frame.columns.tolist() == ["spread_line"], "input frame is never mutated"


def test_future_games_and_results_cannot_move_an_earlier_feature_row() -> None:
    frame = pd.DataFrame(
        {
            "game_id": [f"game_{i}" for i in range(6)],
            "gameday": pd.date_range("2024-09-05", periods=6, freq="7D"),
            "spread_line": [-9.5, 4.0, -1.5, -12.0, 6.5, -7.5],
            "result": [-3.0, 10.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    expected = attach_home_side_location(frame)
    changed = pd.concat(
        [frame, pd.DataFrame({"game_id": ["future"], "spread_line": [-100.0], "result": [999.0]})],
        ignore_index=True,
    )
    changed.loc[3:, ["spread_line", "result"]] = [[50.0, -40.0]] * 4
    pd.testing.assert_frame_equal(
        expected.iloc[:3].reset_index(drop=True),
        attach_home_side_location(changed).iloc[:3][expected.columns].reset_index(drop=True),
    )


def _stream() -> pd.DataFrame:
    rows = []
    game = 0
    for season in (2020, 2021):
        for week in (1, 2, 3):
            sunday = pd.Timestamp(f"{season}-09-13") + pd.Timedelta(weeks=week - 1)
            for j in range(4):
                spread = [-12.0, 11.0, 2.5, -8.0][j]
                rows.append(
                    {
                        "game_id": f"g{game}",
                        "season": season,
                        "week": week,
                        "gameday": sunday - pd.Timedelta(days=3) if j == 0 else sunday,
                        "spread_line": spread,
                        "point_incumbent": spread + 1.0,
                        "result": spread + 4.0 * (1 if j % 2 == 0 else -1),
                    }
                )
                game += 1
    return pd.DataFrame(rows)


def test_offsets_are_shrunken_bucket_means_of_prior_home_error() -> None:
    prior = pd.DataFrame(
        {
            "spread_line": [-12.0, 11.0, 14.0, 2.0, np.nan],
            "point_incumbent": [-10.0, 12.0, 15.0, 3.0, 1.0],
            "result": [-4.0, 20.0, 12.0, 0.0, 7.0],
        }
    )
    fitted = fit_home_side_offsets(prior)
    assert fitted.prior_games["10.5+"] == 3
    assert fitted.offsets["10.5+"] == pytest.approx(11.0 / (3 + PRIOR_WEIGHT_GAMES))
    assert fitted.prior_games["0-3"] == 1
    assert fitted.offsets["0-3"] == 0.0
    replay = fit_home_side_offsets(prior, all_buckets=True)
    assert replay.offsets["0-3"] == pytest.approx(-3.0 / (1 + PRIOR_WEIGHT_GAMES))
    assert replay.offsets["10.5+"] == fitted.offsets["10.5+"]
    assert fitted.offsets["7.5-10"] == 0.0 and fitted.prior_games["7.5-10"] == 0
    applied = fitted.offset_for(pd.Series([-13.0, 9.0, np.nan]))
    assert applied.iloc[0] == pytest.approx(fitted.offsets["10.5+"])
    assert applied.iloc[1] == 0.0
    assert np.isnan(applied.iloc[2])


def test_prior_games_exclude_the_whole_target_week_and_the_completion_allowance() -> None:
    stream = _stream()
    prior = prior_games_for_week(stream, 2020, 2)
    assert set(prior.week) == {1} and set(prior.season) == {2020}
    tight = stream.copy()
    tight.loc[tight.game_id.eq("g3"), "gameday"] = pd.Timestamp("2020-09-16")
    assert "g3" not in set(prior_games_for_week(tight, 2020, 2).game_id)
    old = stream.copy()
    old.loc[old.season.eq(2020), "season"] = 2014
    assert prior_games_for_week(old, 2021, 1).empty


def test_future_games_and_results_cannot_move_an_earlier_offset() -> None:
    stream = _stream()
    expected = walk_forward_home_offsets(stream)
    changed = pd.concat(
        [
            stream,
            pd.DataFrame(
                {
                    "game_id": ["future"],
                    "season": [2021],
                    "week": [4],
                    "gameday": [pd.Timestamp("2021-10-10")],
                    "spread_line": [-14.0],
                    "point_incumbent": [-13.0],
                    "result": [40.0],
                }
            ),
        ],
        ignore_index=True,
    )
    later = changed.season.eq(2021) & changed.week.ge(2)
    changed.loc[later, "result"] = 60.0
    changed.loc[later, "point_incumbent"] = -30.0
    earlier = stream.season.lt(2021) | stream.week.lt(2)
    pd.testing.assert_frame_equal(
        expected.loc[earlier].reset_index(drop=True),
        walk_forward_home_offsets(changed).loc[earlier.index[earlier]].reset_index(drop=True),
    )
    first = stream.season.eq(2020) & stream.week.eq(1)
    assert (expected.loc[first, "home_side_offset"] == 0.0).all()
    assert (expected.loc[first, "prior_games_in_bucket"] == 0).all()


def test_gaussian_median_probability_matches_the_residual_smoother() -> None:
    from nfl_ats.calibration import smoothed_home_cover_probability

    rng = np.random.default_rng(3)
    residuals = rng.normal(0.4, 13.0, 400)
    lines = np.array([-3.0, 7.5, 10.0])
    points = np.array([1.0, 4.0, 14.5])
    expected = smoothed_home_cover_probability(residuals, points, lines, method="gaussian_median")
    actual = gaussian_median_cover_probability(
        lines, points, float(np.median(residuals)), float(np.std(residuals, ddof=1))
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    assert (gaussian_median_cover_probability(lines, points + 2.0, 0.4, 13.0) > actual).all()


def test_profile_is_additive_on_weak_stack_and_leaves_the_incumbent_alone() -> None:
    assert PROFILE in MARGIN_FEATURE_PROFILES
    assert margin_feature_columns("market_residual", PROFILE) == (
        *margin_feature_columns("market_residual", "weak_stack"),
        *HOME_SIDE_HINGE_COLUMNS,
    )
    for prefix in ("football", "full"):
        assert FEATURE_SETS[f"{prefix}_{PROFILE}"] == (
            *FEATURE_SETS[f"{prefix}_weak_stack"],
            *HOME_SIDE_HINGE_COLUMNS,
        )
        assert set(FEATURE_SETS[f"{prefix}_weak_stack"]).isdisjoint(HOME_SIDE_HINGE_COLUMNS)
    assert FEATURE_FAMILIES["home_side_location_hinge"] == HOME_SIDE_HINGE_COLUMNS


def test_fit_requires_the_attached_column(model_frame: pd.DataFrame) -> None:
    with pytest.raises(DataContractError, match="home_side_hinge_7"):
        fit_margin_model(model_frame, target="market_residual", feature_profile=PROFILE)


@pytest.mark.parametrize("command", ["margin-predict", "margin-backtest", "opener-evaluation"])
def test_cli_accepts_the_research_profile(command: str) -> None:
    argv = [command, "--feature-profile", PROFILE]
    if command == "margin-predict":
        argv += ["--season", "2026", "--week", "1"]
    assert cli.build_parser().parse_args(argv).feature_profile == PROFILE
