from __future__ import annotations

import pandas as pd

from nfl_ats.model_card import build_model_card, model_card_markdown


def test_model_card_captures_use_history_and_limitations(model_frame: pd.DataFrame) -> None:
    predictions = model_frame.assign(
        home_cover_probability=0.55,
        bet_side="PASS",
        bet_odds=-110.0,
    )
    metrics = {
        "model_name": "logistic",
        "feature_set": "market_context",
        "first_test_gameday": "2019-09-01",
        "last_test_gameday": "2020-12-31",
        "games_evaluated": len(predictions),
        "accuracy": 0.51,
        "brier_score": 0.249,
        "log_loss": 0.691,
        "expected_calibration_error": 0.01,
        "bets": 10,
        "net_profit_units": 1.0,
        "roi": 0.10,
    }
    card = build_model_card(metrics, {"configuration_sha256": "abc"}, predictions)
    assert card["model"]["feature_set"] == "market_context"
    assert len(card["evaluation"]["season_history"]) == 2
    assert card["known_limitations"]
    markdown = model_card_markdown(card)
    assert "## Intended use" in markdown
    assert "Flat-stake paper ROI: 10.0%" in markdown
