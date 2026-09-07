"""MOD-18 fixed mapping, feature, and chronological leakage contracts."""

import numpy as np
import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.constants import FEATURE_FAMILIES, FEATURE_SETS
from nfl_ats.margin import MARGIN_FEATURE_PROFILES
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.spread_regime import (
    SPREAD_REGIME_COLUMNS,
    attach_spread_regime,
    calibrate_spread_stream,
    spread_bucket,
)


def test_row_local_feature_orientation_and_future_invariance() -> None:
    frame = pd.DataFrame({"spread_line": [3, 3.5, 7, 7.5, 10, 10.5, -8, np.nan]})
    actual = attach_spread_regime(frame)
    assert actual.regime_fav_7p5_10.tolist()[:7] == [0, 0, 0, 1, 1, 0, -1]
    assert actual.regime_distance_7.iloc[6] == -1
    assert actual.regime_signed_absolute.iloc[6] == -8
    assert spread_bucket(frame.spread_line).tolist()[:7] == [
        "0-3",
        "3.5-6.5",
        "7",
        "7.5-10",
        "7.5-10",
        "10.5+",
        "7.5-10",
    ]
    changed = pd.concat(
        [frame, pd.DataFrame({"spread_line": [100], "result": [999]})], ignore_index=True
    )
    pd.testing.assert_frame_equal(
        actual, attach_spread_regime(changed).iloc[: len(frame)][actual.columns]
    )
    assert actual.loc[7, list(SPREAD_REGIME_COLUMNS)].isna().all()


def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"game_{i}" for i in range(25)],
            "season": [2020] * 24 + [2021],
            "week": [*range(1, 25), 1],
            "gameday": pd.date_range("2020-01-01", periods=25, freq="7D"),
            "spread_line": [8.0] * 25,
            "home_cover_probability": [0.56] * 25,
            "home_cover": [0.0] * 24 + [np.nan],
        }
    )


def test_calibration_uses_prior_seasons_and_flips_only_at_half() -> None:
    raw = history()
    result = calibrate_spread_stream(raw)
    assert result.home_cover_probability.iloc[0] == 0.56
    assert result.home_cover_probability.iloc[-1] == pytest.approx(20 * 0.56 / 44)
    assert result.calibration_rows.iloc[-1] == 24
    assert result.home_cover_probability.iloc[-1] < 0.5
    assert raw.home_cover_probability.eq(0.56).all()


def test_future_outcomes_lines_and_probabilities_cannot_move_earlier_calibration() -> None:
    raw = history()
    expected = calibrate_spread_stream(raw)
    changed = raw.copy()
    changed.loc[20:, ["spread_line", "home_cover_probability", "home_cover"]] = [-100, 0.99, 1]
    pd.testing.assert_frame_equal(expected.iloc[:20], calibrate_spread_stream(changed).iloc[:20])


def test_same_week_and_pushes_are_excluded_from_calibration() -> None:
    raw = history().iloc[:3].copy()
    raw["week"] = [1, 2, 2]
    raw["home_cover"] = [np.nan, 0.0, 1.0]
    result = calibrate_spread_stream(raw)
    assert result.calibration_rows.eq(0).all()
    assert result.home_cover_probability.eq(0.56).all()


def test_discrete_mapping_integer_push_and_half_point_mass() -> None:
    residuals = np.array([-7, -3, 0, 0, 0, 0, 0, 3, 7, 14], dtype=float)
    centers = np.array([3.0, 3.0, 3.0, 7.0, 10.0, 14.0])
    lines = np.array([2.5, 3.0, 3.5, 7.0, 10.0, 14.0])
    actual = smoothed_home_cover_probability(residuals, centers, lines, method="discrete_residual")
    expected = ((np.round(centers[:, None] + residuals) > lines[:, None]).sum(axis=1) + 0.5) / 11
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] > actual[1]
    assert actual[1] == actual[2]


def test_profile_is_additive() -> None:
    assert "weak_stack_spread_regime" in MARGIN_FEATURE_PROFILES
    assert FEATURE_FAMILIES["spread_regime"] == SPREAD_REGIME_COLUMNS
    for prefix in ("football", "full"):
        assert FEATURE_SETS[f"{prefix}_weak_stack_spread_regime"] == (
            *FEATURE_SETS[f"{prefix}_weak_stack"],
            *SPREAD_REGIME_COLUMNS,
        )


@pytest.mark.parametrize("command", ["margin-predict", "margin-backtest", "opener-evaluation"])
def test_cli_accepts_additive_mapping(command: str) -> None:
    argv = [command, "--probability-method", "discrete_residual"]
    if command == "margin-predict":
        argv += ["--season", "2026", "--week", "1"]
    assert cli.build_parser().parse_args(argv).probability_method == "discrete_residual"


def test_mapping_does_not_change_fitted_margin(model_frame: pd.DataFrame) -> None:
    target, models = fit_margin_models_for_week(
        model_frame,
        season=2020,
        week=4,
        regressor="ridge",
        min_train_games=100,
        feature_profile="base",
        ridge_alpha=10.0,
        methods=("market_residual",),
    )
    model = models["market_residual"]
    incumbent = model.predict(target, probability_method="gaussian_median")
    candidate = model.predict(target, probability_method="discrete_residual")
    other = incumbent.columns.drop("home_cover_probability")
    pd.testing.assert_frame_equal(incumbent[other], candidate[other], check_exact=True)
