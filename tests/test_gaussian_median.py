"""MOD-06: additive location choice, unchanged scale and incumbent defaults."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nfl_ats import cli
from nfl_ats.calibration import fit_residual_smoother, smoothed_home_cover_probability
from nfl_ats.outcomes import fit_margin_models_for_week, score_outcome_week
from nfl_ats.spread_explorer import compute_spread_explorer_params, widget_home_cover_probability


def test_median_location_boundary_and_unchanged_scale() -> None:
    residuals = np.array([-7, -4, -2, 0, 1, 3, 4, 6, 9, 50], dtype=float)
    median = float(np.median(residuals))
    centers = np.array([-median - 1e-6, -median, -median + 1e-6])
    actual = smoothed_home_cover_probability(
        residuals, centers, np.zeros(3), method="gaussian_median"
    )
    expected = stats.norm.sf(-centers, loc=median, scale=np.std(residuals, ddof=1))
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] < 0.5
    assert actual[1] == 0.5
    assert actual[2] > 0.5
    assert (
        fit_residual_smoother(residuals, "gaussian_median").std
        == fit_residual_smoother(residuals, "gaussian").std
    )


def test_incumbent_week_fixture_numeric_bytes_unchanged(model_frame: pd.DataFrame) -> None:
    card = score_outcome_week(
        model_frame,
        season=2020,
        week=4,
        min_train_games=100,
        feature_profile="base",
        probability_method="gaussian",
    )
    # Captured from the existing model_frame fixture before MOD-06 source edits.
    assert hashlib.sha256(
        card.select_dtypes(include="number").to_numpy().tobytes()
    ).hexdigest() == ("daa3008c0f4289e62a80a7b78b6a3d22083cde1457b7187d79acad1a7c3bb0ca")


@pytest.mark.parametrize("command", ["margin-predict", "margin-backtest", "opener-evaluation"])
def test_cli_accepts_gaussian_median(command: str) -> None:
    argv = [command, "--probability-method", "gaussian_median"]
    if command == "margin-predict":
        argv.extend(["--season", "2026", "--week", "1"])
    assert cli.build_parser().parse_args(argv).probability_method == "gaussian_median"


def test_median_card_refit_and_widget(model_frame: pd.DataFrame) -> None:
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
    median = model.predict(target, probability_method="gaussian_median")
    mean = model.predict(target, probability_method="gaussian")
    other = [c for c in median.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(median[other], mean[other], check_exact=True)
    card = target.copy()
    card["home_cover_probability"] = median["home_cover_probability"]
    params = compute_spread_explorer_params(
        card,
        model_frame,
        regressor="ridge",
        ridge_alpha=10.0,
        feature_profile="base",
        min_train_games=100,
        probability_method="gaussian_median",
    )
    for item in params.values():
        assert item.residual_mean == float(np.median(model.residuals))
        assert widget_home_cover_probability(
            item.card_line, item.center, item.residual_mean, item.residual_std
        ) == pytest.approx(item.card_home_cover_probability, abs=2e-7)


def test_board_reads_median_card_method(model_frame: pd.DataFrame, tmp_path: Path) -> None:
    from nfl_ats.board_content import _load_spread_explorer_params

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
    card = target.copy()
    card["home_cover_probability"] = models["market_residual"].predict(
        target, probability_method="gaussian_median"
    )["home_cover_probability"]
    path = tmp_path / "features.parquet"
    model_frame.to_parquet(path)
    metadata = {
        "probability_method": "gaussian_median",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "feature_profile": "base",
        "min_train_games": 100,
        "provenance": {"feature_table": {"path": str(path)}},
    }
    assert set(_load_spread_explorer_params(metadata, card, tmp_path)) == set(card.game_id)
