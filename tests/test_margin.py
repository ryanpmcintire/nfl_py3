from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import FEATURE_FAMILIES
from nfl_ats.data import DataContractError
from nfl_ats.margin import (
    DEFAULT_LINE_SWEEP_OFFSETS,
    MarginModel,
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
    # MOD-07 candidate profile: the player-value composite plus the bias family,
    # with no duplicated columns and the frozen player profile left untouched.
    weak_stack_columns = margin_feature_columns("market_residual", "weak_stack")
    player_columns = margin_feature_columns("market_residual", "player")
    assert len(weak_stack_columns) == len(set(weak_stack_columns))
    assert set(player_columns).issubset(weak_stack_columns)
    assert set(FEATURE_FAMILIES["bias"]).issubset(weak_stack_columns)
    assert set(FEATURE_FAMILIES["bias"]).isdisjoint(player_columns)
    assert set(FEATURE_FAMILIES["bias"]).issubset(margin_feature_columns("margin", "weak_stack"))
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


def _reference_smoothed_cover_probability(
    center: float, residuals: np.ndarray, line: float
) -> float:
    """Independent re-implementation of the pre-existing smoothed formula.

    Deliberately duplicated rather than imported, so this test fails if the
    ``home_cover_probability`` column's values or semantics ever drift --
    the regression guard required for the three-way split to be additive.
    """

    distribution = center + residuals
    successes = np.count_nonzero(distribution > line)
    return (successes + 0.5) / (len(residuals) + 1.0)


@pytest.mark.parametrize("target", ["margin", "market_residual"])
def test_home_cover_probability_is_bit_identical_to_reference_formula(
    model_frame: pd.DataFrame, target: str
) -> None:
    model = fit_margin_model(model_frame, target=target, model_name="ridge")  # type: ignore[arg-type]
    rows = model_frame.tail(10)
    predictions = model.predict(rows)
    for index in rows.index:
        center = float(predictions.loc[index, "predicted_margin"])
        line = float(rows.loc[index, "spread_line"])
        expected = _reference_smoothed_cover_probability(center, model.residuals, line)
        assert predictions.loc[index, "home_cover_probability"] == expected
    # New columns are additive: the smoothed value is unaffected by their presence.
    assert "home_cover_probability_excluding_push" in predictions.columns
    assert "push_probability" in predictions.columns
    assert "home_loss_probability" in predictions.columns


def test_three_way_split_sums_to_one_and_excludes_push_from_cover(
    model_frame: pd.DataFrame,
) -> None:
    model = fit_margin_model(model_frame, target="margin", model_name="ridge")
    predictions = model.predict(model_frame.tail(10))
    total = (
        predictions["home_cover_probability_excluding_push"]
        + predictions["push_probability"]
        + predictions["home_loss_probability"]
    )
    assert np.allclose(total, 1.0)
    assert predictions["home_cover_probability_excluding_push"].between(0.0, 1.0).all()
    assert predictions["push_probability"].between(0.0, 1.0).all()
    assert predictions["home_loss_probability"].between(0.0, 1.0).all()


def test_push_probability_is_nonzero_only_at_integer_lines() -> None:
    model = MarginModel(
        estimator=None,
        residuals=np.array([0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0]),
        model_name="market",
        ridge_alpha=None,
        target="market",
        feature_columns=(),
        training_rows=100,
        distribution_rows=9,
        training_max_gameday="2020-01-01",
    )
    frame = pd.DataFrame(
        {
            "game_id": ["integer_line", "half_point_line"],
            "spread_line": [3.0, 3.5],
            "home_spread_odds": [-110.0, -110.0],
            "away_spread_odds": [-110.0, -110.0],
        }
    )
    predictions = model.predict(frame)
    integer_row = predictions.loc[frame["game_id"].eq("integer_line").idxmax()]
    half_point_row = predictions.loc[frame["game_id"].eq("half_point_line").idxmax()]
    assert integer_row["push_probability"] > 0.0
    assert half_point_row["push_probability"] == 0.0
    for row in (integer_row, half_point_row):
        total = (
            row["home_cover_probability_excluding_push"]
            + row["push_probability"]
            + row["home_loss_probability"]
        )
        assert total == pytest.approx(1.0)


def test_line_sweep_default_grid_sums_to_one_and_holds_belief_fixed_for_market(
    model_frame: pd.DataFrame,
) -> None:
    market = fit_market_baseline(model_frame)
    target = model_frame.tail(2)
    sweep = market.line_sweep(target)
    assert len(DEFAULT_LINE_SWEEP_OFFSETS) == 17
    assert min(DEFAULT_LINE_SWEEP_OFFSETS) == -4.0
    assert max(DEFAULT_LINE_SWEEP_OFFSETS) == 4.0
    assert len(sweep) == len(target) * len(DEFAULT_LINE_SWEEP_OFFSETS)
    assert set(sweep["game_id"]) == set(target["game_id"])
    total = (
        sweep["home_cover_probability_excluding_push"]
        + sweep["push_probability"]
        + sweep["home_loss_probability"]
    )
    assert np.allclose(total, 1.0)
    # The predictive distribution is conditioned on the actually quoted line
    # and stays fixed across the sweep; only the settlement threshold moves.
    # A higher home line is therefore never easier to cover, and a friendlier
    # line at -4.0 must genuinely beat a tougher one at +4.0.
    for _, group in sweep.groupby("game_id"):
        ordered = group.sort_values("line_offset")["home_cover_probability"].to_numpy()
        assert np.all(np.diff(ordered) <= 1e-12)
        assert ordered[0] > ordered[-1]
        at_quoted = group.loc[np.isclose(group["line_offset"], 0.0), "home_cover_probability"]
        assert ordered[-1] < float(at_quoted.iloc[0]) < ordered[0]


def test_line_sweep_varies_with_alternative_line_for_fair_margin(
    model_frame: pd.DataFrame,
) -> None:
    model = fit_margin_model(model_frame, target="margin", model_name="ridge")
    target = model_frame.tail(1)
    sweep = model.line_sweep(target, offsets=(-0.5, 0.0, 0.5)).sort_values("line_offset")
    cover = sweep["home_cover_probability"].tolist()
    # A higher home line is harder to cover, so cover probability is non-increasing.
    assert cover[0] >= cover[1] >= cover[2]


def test_line_sweep_shows_a_confidence_gap_across_a_half_point_hook(
    model_frame: pd.DataFrame,
) -> None:
    # fair_margin's center does not use spread_line as a feature, so its
    # cover probability moves purely because the threshold moves -- the
    # cleanest demonstration that a half-point hook (here, +2.5 vs +3) can
    # change the pick's confidence without changing the model's opinion of
    # the game itself.
    model = fit_margin_model(model_frame, target="margin", model_name="ridge")
    target = model_frame.tail(1).copy()
    target.loc[:, "spread_line"] = 3.0
    sweep = model.line_sweep(target, offsets=(-0.5, 0.0))
    at_2_5 = sweep.loc[np.isclose(sweep["alternative_line"], 2.5)].iloc[0]
    at_3_0 = sweep.loc[np.isclose(sweep["alternative_line"], 3.0)].iloc[0]
    assert at_2_5["push_probability"] == 0.0
    assert at_2_5["home_cover_probability"] >= at_3_0["home_cover_probability"]


def test_line_sweep_requires_columns_and_at_least_one_offset(model_frame: pd.DataFrame) -> None:
    market = fit_market_baseline(model_frame)
    with pytest.raises(ValueError, match="at least one offset"):
        market.line_sweep(model_frame.tail(1), offsets=())
    with pytest.raises(DataContractError, match="missing columns"):
        market.line_sweep(model_frame.tail(1).drop(columns="game_id"))
