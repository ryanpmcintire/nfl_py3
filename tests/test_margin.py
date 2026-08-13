from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.margin import (
    fit_margin_model,
    fit_market_baseline,
    make_margin_estimator,
    margin_feature_columns,
    margin_model_metadata,
)


def test_independent_margin_and_residual_models(model_frame: pd.DataFrame) -> None:
    fair = fit_margin_model(model_frame, target="margin", model_name="ridge")
    residual = fit_margin_model(model_frame, target="market_residual", model_name="ridge")
    target = model_frame.tail(5)
    fair_predictions = fair.predict(target)
    residual_predictions = residual.predict(target)

    for predictions in (fair_predictions, residual_predictions):
        assert len(predictions) == 5
        assert predictions["home_win_probability"].between(0.0, 1.0).all()
        assert predictions["home_cover_probability"].between(0.0, 1.0).all()
        assert predictions["margin_lower_80"].le(predictions["margin_upper_80"]).all()
    assert np.allclose(
        residual_predictions["fair_spread"],
        target["spread_line"] + residual_predictions["predicted_market_residual"],
    )
    assert "spread_line" not in margin_feature_columns("margin")
    assert "spread_line" in margin_feature_columns("market_residual")
    assert "graph_pagerank_diff" in margin_feature_columns("margin", "graph")
    assert "schedule_rating_diff" in margin_feature_columns("market_residual", "graph")
    assert "diff_pbp_matchup_epa_per_play" in margin_feature_columns("margin", "pbp_adjusted")
    assert "diff_drive_points_per_drive" in margin_feature_columns("margin", "drive")
    assert "diff_injury_skill_unavailability" in margin_feature_columns(
        "market_residual", "player_injuries"
    )
    assert "diff_qb_start_probability" not in margin_feature_columns(
        "market_residual", "player_injuries"
    )
    assert "diff_active_roster_continuity" in margin_feature_columns(
        "margin", "player_injuries_continuity"
    )
    participation_columns = margin_feature_columns("market_residual", "player_participation")
    assert "diff_injury_offense_participation_value_lost" in participation_columns
    assert "diff_injury_skill_epa_value_lost" in participation_columns
    assert margin_model_metadata(fair)["distribution_rows"] == 32
    regularized = fit_margin_model(
        model_frame,
        target="market_residual",
        model_name="ridge",
        ridge_alpha=1.0,
    )
    assert margin_model_metadata(regularized)["ridge_alpha"] == 1.0


def test_market_baseline_is_centered_on_spread(model_frame: pd.DataFrame) -> None:
    market = fit_market_baseline(model_frame.iloc[:100])
    predictions = market.predict(model_frame.tail(3))
    assert np.allclose(predictions["predicted_margin"], model_frame.tail(3)["spread_line"])
    assert predictions["predicted_market_residual"].eq(0.0).all()
    assert predictions["home_cover_probability"].eq(0.5).all()


def test_margin_hgb_and_guards(model_frame: pd.DataFrame) -> None:
    model = fit_margin_model(model_frame, target="margin", model_name="hgb")
    assert len(model.predict(model_frame.tail(2))) == 2
    with pytest.raises(ValueError, match="Unknown margin model"):
        make_margin_estimator("forest")
    with pytest.raises(ValueError, match="ridge_alpha"):
        make_margin_estimator("ridge", ridge_alpha=0.0)
    with pytest.raises(ValueError, match="Unknown margin target"):
        margin_feature_columns("score")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="At least 50"):
        fit_margin_model(model_frame.head(20))
    with pytest.raises(ValueError, match="distribution_fraction"):
        fit_margin_model(model_frame, distribution_fraction=0.05)
    missing = model_frame.tail(2).copy()
    missing.loc[:, "spread_line"] = np.nan
    with pytest.raises(ValueError, match="requires a spread"):
        model.predict(missing)
