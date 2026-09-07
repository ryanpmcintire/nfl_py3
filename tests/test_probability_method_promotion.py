"""Production median promotion, with explicit mean/ECDF compatibility.

Pin weekly and CLI defaults, unchanged historical ECDF defaults, and artifact
identity separation for each probability mapping. See docs/gaussian_median_promotion.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.active_model import activate_matching_ats_model
from nfl_ats.margin import fit_margin_model
from nfl_ats.outcomes import score_outcome_week, walk_forward_outcomes

_FEATURE_PROFILE = "base"
_MIN_TRAIN_GAMES = 100
_SEASON = 2020
_WEEK = 4


# ---------------------------------------------------------------------------
# 1. MarginModel.predict: default unchanged, explicit gaussian differs
# ---------------------------------------------------------------------------


def test_predict_default_is_ecdf_and_matches_explicit_ecdf(model_frame: pd.DataFrame) -> None:
    model = fit_margin_model(
        model_frame.loc[model_frame["gameday"] < model_frame["gameday"].quantile(0.8)],
        target="market_residual",
        model_name="ridge",
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        ridge_alpha=10.0,
    )
    target = model_frame.tail(10)
    implicit = model.predict(target)
    explicit = model.predict(target, probability_method="ecdf")
    pd.testing.assert_frame_equal(implicit, explicit, check_exact=True)


def test_predict_gaussian_changes_only_home_cover_probability(model_frame: pd.DataFrame) -> None:
    model = fit_margin_model(
        model_frame.loc[model_frame["gameday"] < model_frame["gameday"].quantile(0.8)],
        target="market_residual",
        model_name="ridge",
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        ridge_alpha=10.0,
    )
    target = model_frame.tail(10)
    ecdf = model.predict(target, probability_method="ecdf")
    gaussian = model.predict(target, probability_method="gaussian")
    assert list(ecdf.columns) == list(gaussian.columns)
    other_columns = [c for c in ecdf.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        ecdf[other_columns].reset_index(drop=True),
        gaussian[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    assert not np.allclose(
        ecdf["home_cover_probability"].to_numpy(), gaussian["home_cover_probability"].to_numpy()
    )


# ---------------------------------------------------------------------------
# 2. outcomes.py: production defaults to gaussian, backtests stay ecdf
# ---------------------------------------------------------------------------


def test_score_outcome_week_defaults_to_gaussian_median(model_frame: pd.DataFrame) -> None:
    """The sole production weekly-forecast entry point's promoted default."""

    default_run = score_outcome_week(
        model_frame,
        season=_SEASON,
        week=_WEEK,
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
    )
    explicit_gaussian = score_outcome_week(
        model_frame,
        season=_SEASON,
        week=_WEEK,
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        probability_method="gaussian_median",
    )
    explicit_ecdf = score_outcome_week(
        model_frame,
        season=_SEASON,
        week=_WEEK,
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        probability_method="ecdf",
    )
    pd.testing.assert_frame_equal(
        default_run.reset_index(drop=True), explicit_gaussian.reset_index(drop=True)
    )
    ats = default_run.loc[default_run["method"].eq("market_residual")]
    ats_ecdf = explicit_ecdf.loc[explicit_ecdf["method"].eq("market_residual")]
    assert not np.allclose(
        ats["home_cover_probability"].to_numpy(dtype=float),
        ats_ecdf["home_cover_probability"].to_numpy(dtype=float),
    )
    # "market" method is unaffected by probability_method (it reads book
    # odds, never the residual sample) -- an invariant, not a loophole.
    market = default_run.loc[default_run["method"].eq("market")]
    market_ecdf = explicit_ecdf.loc[explicit_ecdf["method"].eq("market")]
    pd.testing.assert_series_equal(
        market["home_cover_probability"].reset_index(drop=True),
        market_ecdf["home_cover_probability"].reset_index(drop=True),
    )


def test_walk_forward_outcomes_default_is_still_ecdf(model_frame: pd.DataFrame) -> None:
    """Every historical/research backtest must stay bit-for-bit unaffected."""

    default_run = walk_forward_outcomes(
        model_frame,
        start_season=_SEASON,
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        methods=("market_residual",),
    )
    explicit_ecdf = walk_forward_outcomes(
        model_frame,
        start_season=_SEASON,
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        methods=("market_residual",),
        probability_method="ecdf",
    )
    pd.testing.assert_frame_equal(
        default_run.predictions.reset_index(drop=True),
        explicit_ecdf.predictions.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 3. CLI defaults
# ---------------------------------------------------------------------------


def test_margin_predict_cli_default_is_gaussian_median() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["margin-predict", "--season", "2026", "--week", "1"])
    assert args.probability_method == "gaussian_median"


def test_margin_backtest_cli_default_is_ecdf() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["margin-backtest"])
    assert args.probability_method == "ecdf"


# ---------------------------------------------------------------------------
# 4. active_model identity: probability_method must match to synchronize
# ---------------------------------------------------------------------------


def _forecast_metadata(probability_method: str | None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "created_at_utc": "2026-08-19T20:00:00+00:00",
        "season": 2026,
        "week": 1,
        "feature_profile": "weak_stack",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "ats_method": "market_residual",
        "provenance": {
            "configuration_sha256": "config",
            "feature_table": {"sha256": "features"},
        },
    }
    if probability_method is not None:
        metadata["probability_method"] = probability_method
    return metadata


def _evaluation(root: Path, name: str, probability_method: str | None) -> Path:
    evaluation = root / "margins" / name
    evaluation.mkdir(parents=True)
    (evaluation / "metadata.json").write_text(
        json.dumps(_forecast_metadata(probability_method)), encoding="utf-8"
    )
    pd.DataFrame(
        {"method": ["market_residual"], "cover_accuracy": [0.53], "cover_games": [1000]}
    ).to_csv(evaluation / "summary.csv", index=False)
    return evaluation


@pytest.mark.parametrize("method", ["gaussian", "gaussian_median"])
def test_gaussian_forecast_does_not_match_a_legacy_ecdf_evaluation(
    tmp_path: Path, method: str
) -> None:
    """The guard against the exact 'silently revert' failure mode: a forecast
    built with probability_method="ecdf" (the pre-promotion default,
    explicit or via an old artifact lacking the field) must never
    synchronize against an evaluation recorded under "gaussian", and vice
    versa -- each probability method needs its OWN matching evaluation."""

    _evaluation(tmp_path, "legacy_ecdf", probability_method=None)
    forecast = tmp_path / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)

    gaussian_forecast = _forecast_metadata(method)
    assert activate_matching_ats_model(tmp_path, forecast, gaussian_forecast) is None


@pytest.mark.parametrize("method", ["gaussian", "gaussian_median"])
def test_gaussian_forecast_matches_a_gaussian_evaluation(tmp_path: Path, method: str) -> None:
    _evaluation(tmp_path, "legacy_ecdf", probability_method=None)
    _evaluation(tmp_path, "gaussian_promoted", probability_method=method)
    forecast = tmp_path / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)

    manifest = activate_matching_ats_model(tmp_path, forecast, _forecast_metadata(method))

    assert manifest is not None
    assert manifest["status"] == "SYNCHRONIZED"
    assert manifest["probability_method"] == method


def test_legacy_ecdf_forecast_still_matches_the_legacy_evaluation(tmp_path: Path) -> None:
    """Backward compatibility: metadata written before this field existed
    (both sides) keeps matching exactly as it did before this promotion."""

    _evaluation(tmp_path, "legacy_ecdf", probability_method=None)
    forecast = tmp_path / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)

    manifest = activate_matching_ats_model(tmp_path, forecast, _forecast_metadata(None))

    assert manifest is not None
    assert manifest["status"] == "SYNCHRONIZED"
    assert manifest["probability_method"] == "ecdf"


@pytest.mark.parametrize(
    ("evaluation_method", "forecast_method"),
    [("gaussian", "gaussian_median"), ("gaussian_median", "gaussian")],
)
def test_gaussian_location_estimators_have_distinct_artifact_identity(
    tmp_path: Path, evaluation_method: str, forecast_method: str
) -> None:
    _evaluation(tmp_path, "evaluation", probability_method=evaluation_method)
    forecast = tmp_path / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    assert (
        activate_matching_ats_model(tmp_path, forecast, _forecast_metadata(forecast_method)) is None
    )
