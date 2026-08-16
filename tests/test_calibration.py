from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_ats.calibration import (
    COVER_CALIBRATION_METHODS,
    calibrate_cover_prediction_stream,
)
from nfl_ats.prediction_safety import (
    PredictionSafetyError,
    validate_outcome_prediction_card,
)


def _raw_prediction_stream() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    game_index = 0
    for season in (2018, 2019, 2020):
        start = pd.Timestamp(f"{season}-09-01")
        for week in range(1, 13):
            gameday = start + timedelta(days=7 * (week - 1))
            for game in range(4):
                outcome = float((game_index + game_index // 5) % 2)
                raw_probability = 0.44 + 0.08 * outcome + 0.01 * ((game_index % 3) - 1)
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{game}",
                        "season": season,
                        "week": week,
                        "gameday": gameday,
                        "away_team": "AWY",
                        "home_team": "HME",
                        "spread_line": 0.0,
                        "away_spread_odds": -110.0,
                        "home_spread_odds": -110.0,
                        "method": "market_residual",
                        "home_cover": outcome,
                        "home_cover_probability": raw_probability,
                        "home_cover_probability_excluding_push": raw_probability,
                        "push_probability": 0.0,
                        "home_loss_probability": 1.0 - raw_probability,
                        "home_win_probability": 0.5,
                        "predicted_margin": 0.0,
                        "fair_spread": 0.0,
                        "predicted_market_residual": 0.0,
                        "margin_lower_50": -3.0,
                        "margin_upper_50": 3.0,
                        "margin_lower_80": -10.0,
                        "margin_upper_80": 10.0,
                        "bet_side": "PASS",
                        "edge": 0.0,
                        "bet_odds": np.nan,
                        "break_even_probability": np.nan,
                        "train_rows": 500,
                        "train_max_gameday": (gameday - timedelta(days=1)).date().isoformat(),
                    }
                )
                game_index += 1
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", COVER_CALIBRATION_METHODS)
def test_calibration_uses_only_prior_prediction_weeks(method: str) -> None:
    calibrated = calibrate_cover_prediction_stream(
        _raw_prediction_stream(),
        method=method,
        evaluation_start_season=2020,
        min_calibration_games=40,
        min_edge=0.02,
    )
    assert len(calibrated) == 48
    assert calibrated["season"].eq(2020).all()
    assert calibrated["calibration_method"].eq(method).all()
    assert calibrated["home_cover_probability"].between(0.0, 1.0).all()

    if method == "none":
        assert calibrated["calibration_rows"].eq(0).all()
        assert calibrated["calibration_max_gameday"].isna().all()
        assert np.allclose(
            calibrated["home_cover_probability"],
            calibrated["raw_home_cover_probability"],
        )
    else:
        first_week = calibrated.loc[calibrated["week"].eq(1)]
        assert first_week["calibration_rows"].eq(96).all()
        assert (
            pd.to_datetime(first_week["calibration_max_gameday"])
            .lt(pd.to_datetime(first_week["gameday"]))
            .all()
        )

    for (season, week), weekly_rows in calibrated.groupby(["season", "week"]):
        audit = validate_outcome_prediction_card(
            weekly_rows,
            min_edge=0.02,
            expected_methods=("market_residual",),
            expected_season=int(season),
            expected_week=int(week),
        )
        assert "calibration_cutoff" in audit.checks_passed


def test_calibration_and_safety_guards() -> None:
    raw = _raw_prediction_stream()
    with pytest.raises(ValueError, match="Unknown cover calibration"):
        calibrate_cover_prediction_stream(
            raw,
            method="spline",
            evaluation_start_season=2020,
            min_calibration_games=40,
        )
    with pytest.raises(ValueError, match="need 400"):
        calibrate_cover_prediction_stream(
            raw,
            method="platt",
            evaluation_start_season=2020,
            min_calibration_games=400,
        )
    with pytest.raises(ValueError, match="at least 2"):
        calibrate_cover_prediction_stream(
            raw,
            method="platt",
            evaluation_start_season=2020,
            min_calibration_games=1,
        )
    with pytest.raises(ValueError, match="missing columns"):
        calibrate_cover_prediction_stream(
            raw.drop(columns="week"),
            method="none",
            evaluation_start_season=2020,
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        calibrate_cover_prediction_stream(
            raw.iloc[0:0],
            method="none",
            evaluation_start_season=2020,
        )
    wrong_method = raw.copy()
    wrong_method["method"] = "direct_ats"
    with pytest.raises(ValueError, match="only market_residual"):
        calibrate_cover_prediction_stream(
            wrong_method,
            method="none",
            evaluation_start_season=2020,
        )
    duplicate = pd.concat([raw, raw.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="one raw prediction"):
        calibrate_cover_prediction_stream(
            duplicate,
            method="none",
            evaluation_start_season=2020,
        )
    invalid_probability = raw.copy()
    invalid_probability.loc[0, "home_cover_probability"] = 1.1
    with pytest.raises(ValueError, match="finite and between"):
        calibrate_cover_prediction_stream(
            invalid_probability,
            method="none",
            evaluation_start_season=2020,
        )
    with pytest.raises(ValueError, match="No calibration evaluation"):
        calibrate_cover_prediction_stream(
            raw,
            method="none",
            evaluation_start_season=2030,
        )
    one_class = raw.copy()
    one_class["home_cover"] = 1.0
    with pytest.raises(ValueError, match="both ATS outcome classes"):
        calibrate_cover_prediction_stream(
            one_class,
            method="beta",
            evaluation_start_season=2020,
            min_calibration_games=40,
        )

    calibrated = calibrate_cover_prediction_stream(
        raw,
        method="platt",
        evaluation_start_season=2020,
        min_calibration_games=40,
    )
    first_week = calibrated.loc[calibrated["week"].eq(1)].copy()
    first_week["calibration_max_gameday"] = first_week["gameday"]
    with pytest.raises(PredictionSafetyError, match="calibration_cutoff"):
        validate_outcome_prediction_card(
            first_week,
            min_edge=0.02,
            expected_methods=("market_residual",),
        )
