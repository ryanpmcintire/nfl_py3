from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from nfl_ats.backtest import score_week
from nfl_ats.outcomes import OUTCOME_METHODS, score_outcome_week
from nfl_ats.prediction_safety import (
    PredictionSafetyError,
    validate_outcome_prediction_card,
    validate_prediction_card,
    validate_three_way_split,
)


def _direct_card(model_frame: pd.DataFrame) -> pd.DataFrame:
    predictions, _ = score_week(
        model_frame,
        season=2020,
        week=1,
        min_train_games=80,
    )
    return predictions


def test_direct_card_passes_independent_safety_audit(model_frame: pd.DataFrame) -> None:
    audit = validate_prediction_card(
        _direct_card(model_frame),
        min_edge=0.02,
        expected_season=2020,
        expected_week=1,
    )
    assert audit.status == "PASS"
    assert {
        "training_cutoff",
        "probabilities",
        "decision_policy",
        "market_math",
    }.issubset(audit.checks_passed)


@pytest.mark.parametrize(
    ("column", "value", "failed_check"),
    [
        ("home_cover_probability", 1.01, "probabilities"),
        ("pick", "SIDEWAYS", "decision_policy"),
        ("edge", 0.99, "decision_policy"),
        ("market_home_no_vig_probability", 0.75, "market_math"),
        ("home_spread_odds", -1.10, "market_inputs"),
    ],
)
def test_direct_card_fails_closed_on_corruption(
    model_frame: pd.DataFrame,
    column: str,
    value: object,
    failed_check: str,
) -> None:
    predictions = _direct_card(model_frame)
    predictions.loc[predictions.index[0], column] = value
    with pytest.raises(PredictionSafetyError, match=failed_check):
        validate_prediction_card(predictions, min_edge=0.02)


def test_direct_card_rejects_duplicate_games_and_leaky_cutoff(
    model_frame: pd.DataFrame,
) -> None:
    predictions = _direct_card(model_frame)
    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(PredictionSafetyError, match="unique_predictions"):
        validate_prediction_card(duplicate, min_edge=0.02)

    predictions["train_max_gameday"] = predictions["gameday"].astype(str)
    with pytest.raises(PredictionSafetyError, match="training_cutoff"):
        validate_prediction_card(predictions, min_edge=0.02)


def test_direct_card_rejects_malformed_model_inputs(model_frame: pd.DataFrame) -> None:
    predictions = _direct_card(model_frame)
    predictions["elo_diff"] = predictions["elo_diff"].astype(object)
    predictions.loc[predictions.index[0], "elo_diff"] = "not-a-number"
    with pytest.raises(PredictionSafetyError, match="model_inputs"):
        validate_prediction_card(
            predictions,
            min_edge=0.02,
            feature_columns=("elo_diff",),
        )


def test_direct_card_schema_identity_scope_and_cutoff_guards(
    model_frame: pd.DataFrame,
) -> None:
    predictions = _direct_card(model_frame)
    corruptions = (
        (predictions.iloc[0:0], "nonempty"),
        (predictions.drop(columns="edge"), "schema"),
        (predictions.assign(home_team=predictions["away_team"]), "game_identity"),
        (predictions.assign(season=2021), "card_scope"),
        (predictions.assign(train_rows=0), "training_cutoff"),
        (predictions.assign(train_max_gameday=pd.NaT), "training_cutoff"),
    )
    for corrupted, failed_check in corruptions:
        with pytest.raises(PredictionSafetyError, match=failed_check):
            validate_prediction_card(
                corrupted,
                min_edge=0.02,
                expected_season=2020,
            )


def test_direct_card_market_and_decision_math_guards(model_frame: pd.DataFrame) -> None:
    predictions = _direct_card(model_frame)
    first = predictions.index[0]
    corruptions: tuple[tuple[str, object, str], ...] = (
        ("spread_line", 41.0, "market_inputs"),
        ("home_cover_probability", float("nan"), "probabilities"),
        ("bet_side", "MAYBE", "decision_policy"),
        ("bet_odds", -105.0, "decision_policy"),
        ("break_even_probability", 0.25, "decision_policy"),
        ("market_hold", 0.25, "market_math"),
    )
    for column, value, failed_check in corruptions:
        corrupted = predictions.copy()
        corrupted.loc[first, column] = value
        with pytest.raises(PredictionSafetyError, match=failed_check):
            validate_prediction_card(corrupted, min_edge=0.02)

    with pytest.raises(PredictionSafetyError, match="decision_policy"):
        validate_prediction_card(predictions, min_edge=-0.01)

    missing_prices = predictions.copy()
    missing_prices.loc[first, ["home_spread_odds", "away_spread_odds"]] = np.nan
    audit = validate_prediction_card(missing_prices, min_edge=0.02)
    assert audit.status == "PASS_WITH_WARNINGS"
    assert "-110 fallback" in audit.warnings[0]


def test_direct_card_model_input_coverage_guards(model_frame: pd.DataFrame) -> None:
    predictions = _direct_card(model_frame)
    first = predictions.index[0]
    features = ("elo_diff", "temp", "wind", "rest_diff")

    infinite = predictions.copy()
    infinite.loc[first, "elo_diff"] = np.inf
    with pytest.raises(PredictionSafetyError, match="model_inputs"):
        validate_prediction_card(
            infinite,
            min_edge=0.02,
            feature_columns=features,
        )

    mostly_missing = predictions.copy()
    mostly_missing.loc[first, list(features[:3])] = np.nan
    with pytest.raises(PredictionSafetyError, match="more than half"):
        validate_prediction_card(
            mostly_missing,
            min_edge=0.02,
            feature_columns=features,
        )

    partly_missing = predictions.copy()
    partly_missing.loc[first, list(features[:2])] = np.nan
    audit = validate_prediction_card(
        partly_missing,
        min_edge=0.02,
        feature_columns=features,
    )
    assert audit.status == "PASS_WITH_WARNINGS"
    assert "model inputs" in audit.warnings[0]


def test_prospective_card_enforces_timing_outcome_and_price_embargoes(
    model_frame: pd.DataFrame,
) -> None:
    created_at = datetime(2018, 12, 9, tzinfo=UTC)
    predictions = _direct_card(model_frame)
    predictions["kickoff"] = pd.to_datetime(predictions["gameday"], utc=True) + pd.Timedelta(
        hours=12
    )
    for column in ("home_cover", "result", "ats_margin"):
        predictions[column] = np.nan

    audit = validate_prediction_card(
        predictions,
        min_edge=0.02,
        prospective=True,
        created_at=created_at,
    )
    assert audit.status == "PASS_WITH_WARNINGS"
    assert "line freshness" in audit.warnings[0]

    first = predictions.index[0]
    missing_price = predictions.copy()
    missing_price.loc[first, "home_spread_odds"] = np.nan
    with pytest.raises(PredictionSafetyError, match="market_inputs"):
        validate_prediction_card(
            missing_price,
            min_edge=0.02,
            prospective=True,
            created_at=created_at,
        )

    started = predictions.copy()
    started.loc[first, "kickoff"] = created_at
    with pytest.raises(PredictionSafetyError, match="pregame_timing"):
        validate_prediction_card(
            started,
            min_edge=0.02,
            prospective=True,
            created_at=created_at,
        )

    known_outcome = predictions.copy()
    known_outcome.loc[first, "home_cover"] = 1.0
    with pytest.raises(PredictionSafetyError, match="outcome_embargo"):
        validate_prediction_card(
            known_outcome,
            min_edge=0.02,
            prospective=True,
            created_at=created_at,
        )

    timestamped = predictions.copy()
    timestamped["market_observed_at_utc"] = created_at
    timestamped_audit = validate_prediction_card(
        timestamped,
        min_edge=0.02,
        prospective=True,
        created_at=created_at,
    )
    assert timestamped_audit.status == "PASS"
    assert "market_timing" in timestamped_audit.checks_passed

    timestamped.loc[first, "market_observed_at_utc"] = timestamped.loc[first, "kickoff"]
    with pytest.raises(PredictionSafetyError, match="market_timing"):
        validate_prediction_card(
            timestamped,
            min_edge=0.02,
            prospective=True,
            created_at=created_at,
        )


def test_outcome_card_enforces_method_and_margin_identities(
    model_frame: pd.DataFrame,
) -> None:
    predictions = score_outcome_week(
        model_frame,
        season=2020,
        week=1,
        min_train_games=80,
    )
    audit = validate_outcome_prediction_card(
        predictions,
        min_edge=0.02,
        expected_methods=OUTCOME_METHODS,
    )
    assert audit.status == "PASS"

    corrupted = predictions.copy()
    fair = corrupted["method"].eq("fair_margin")
    corrupted.loc[fair, "fair_spread"] = corrupted.loc[fair, "fair_spread"] + 1.0
    with pytest.raises(PredictionSafetyError, match="margin_identity"):
        validate_outcome_prediction_card(
            corrupted,
            min_edge=0.02,
            expected_methods=OUTCOME_METHODS,
        )

    incomplete = predictions.loc[predictions["method"].ne("direct_ats")]
    with pytest.raises(PredictionSafetyError, match="method_coverage"):
        validate_outcome_prediction_card(
            incomplete,
            min_edge=0.02,
            expected_methods=OUTCOME_METHODS,
        )


def test_outcome_card_rejects_invalid_distribution_baseline_and_probability(
    model_frame: pd.DataFrame,
) -> None:
    predictions = score_outcome_week(
        model_frame,
        season=2020,
        week=1,
        min_train_games=80,
    )
    first_fair = predictions.index[predictions["method"].eq("fair_margin")][0]
    first_market = predictions.index[predictions["method"].eq("market")][0]
    first_straight = predictions.index[predictions["method"].eq("straight_up")][0]

    bad_interval = predictions.copy()
    bad_interval.loc[first_fair, "margin_lower_80"] = bad_interval.loc[
        first_fair, "margin_upper_80"
    ]
    with pytest.raises(PredictionSafetyError, match="margin_distribution"):
        validate_outcome_prediction_card(
            bad_interval,
            min_edge=0.02,
            expected_methods=OUTCOME_METHODS,
        )

    bad_market = predictions.copy()
    bad_market.loc[first_market, "predicted_margin"] += 1.0
    bad_market.loc[first_market, "fair_spread"] += 1.0
    bad_market.loc[first_market, "predicted_market_residual"] += 1.0
    with pytest.raises(PredictionSafetyError, match="market_baseline"):
        validate_outcome_prediction_card(
            bad_market,
            min_edge=0.02,
            expected_methods=OUTCOME_METHODS,
        )

    bad_probability = predictions.copy()
    bad_probability.loc[first_straight, "home_win_probability"] = 1.1
    with pytest.raises(PredictionSafetyError, match="probabilities"):
        validate_outcome_prediction_card(
            bad_probability,
            min_edge=0.02,
            expected_methods=OUTCOME_METHODS,
        )

    bad_straight_wager = predictions.copy()
    bad_straight_wager.loc[first_straight, "bet_side"] = "HOME"
    with pytest.raises(PredictionSafetyError, match="decision_policy"):
        validate_outcome_prediction_card(
            bad_straight_wager,
            min_edge=0.02,
            expected_methods=OUTCOME_METHODS,
        )


def test_outcome_card_validates_three_way_split(model_frame: pd.DataFrame) -> None:
    predictions = score_outcome_week(model_frame, season=2020, week=1, min_train_games=80)
    audit = validate_outcome_prediction_card(
        predictions, min_edge=0.02, expected_methods=OUTCOME_METHODS
    )
    assert audit.status in ("PASS", "PASS_WITH_WARNINGS")
    assert {"three_way_probabilities", "three_way_sum", "push_half_point"}.issubset(
        audit.checks_passed
    )


def test_outcome_card_fails_closed_on_corrupt_three_way_fields(model_frame: pd.DataFrame) -> None:
    predictions = score_outcome_week(model_frame, season=2020, week=1, min_train_games=80)
    margin_row = predictions.index[predictions["method"].eq("fair_margin")][0]
    # The fixture's spread_line is always a half-point (2.5 or -2.5), so any
    # fair_margin row exercises the push-half-point invariant.

    out_of_bounds = predictions.copy()
    out_of_bounds.loc[margin_row, "push_probability"] = 1.5
    with pytest.raises(PredictionSafetyError, match="three_way_probabilities"):
        validate_outcome_prediction_card(
            out_of_bounds, min_edge=0.02, expected_methods=OUTCOME_METHODS
        )

    bad_sum = predictions.copy()
    bad_sum.loc[margin_row, "home_cover_probability_excluding_push"] += 0.2
    with pytest.raises(PredictionSafetyError, match="three_way_sum"):
        validate_outcome_prediction_card(bad_sum, min_edge=0.02, expected_methods=OUTCOME_METHODS)

    corrupted_push = predictions.copy()
    corrupted_push.loc[margin_row, "push_probability"] = 0.1
    corrupted_push.loc[margin_row, "home_cover_probability_excluding_push"] -= 0.1
    with pytest.raises(PredictionSafetyError, match="push_half_point"):
        validate_outcome_prediction_card(
            corrupted_push, min_edge=0.02, expected_methods=OUTCOME_METHODS
        )

    missing = predictions.drop(columns="push_probability")
    with pytest.raises(PredictionSafetyError, match="schema"):
        validate_outcome_prediction_card(missing, min_edge=0.02, expected_methods=OUTCOME_METHODS)


def test_validate_three_way_split_standalone() -> None:
    frame = pd.DataFrame(
        {
            "spread_line": [3.0, 3.5, -7.0],
            "home_cover_probability_excluding_push": [0.4, 0.5, 0.6],
            "push_probability": [0.2, 0.0, 0.0],
            "home_loss_probability": [0.4, 0.5, 0.4],
        }
    )
    checks = validate_three_way_split(frame)
    assert checks == ("three_way_probabilities", "three_way_sum", "push_half_point")

    out_of_bounds = frame.copy()
    out_of_bounds.loc[0, "push_probability"] = -0.1
    with pytest.raises(PredictionSafetyError, match="three_way_probabilities"):
        validate_three_way_split(out_of_bounds)

    bad_sum = frame.copy()
    bad_sum.loc[0, "home_loss_probability"] = 0.9
    with pytest.raises(PredictionSafetyError, match="three_way_sum"):
        validate_three_way_split(bad_sum)

    bad_half_point = frame.copy()
    bad_half_point.loc[1, "push_probability"] = 0.1
    bad_half_point.loc[1, "home_cover_probability_excluding_push"] = 0.4
    with pytest.raises(PredictionSafetyError, match="push_half_point"):
        validate_three_way_split(bad_half_point)

    missing = frame.drop(columns="push_probability")
    with pytest.raises(PredictionSafetyError, match="three-way split"):
        validate_three_way_split(missing)

    empty = frame.iloc[0:0]
    assert validate_three_way_split(empty) == (
        "three_way_probabilities",
        "three_way_sum",
        "push_half_point",
    )
