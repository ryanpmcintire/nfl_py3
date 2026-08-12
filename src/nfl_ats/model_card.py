"""Serializable model cards for evaluated backtest artifacts."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from nfl_ats.reporting import season_scorecard


def build_model_card(
    metrics: dict[str, Any],
    provenance: dict[str, Any],
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Describe intended use, evaluation history, and known limitations."""

    season_history = json.loads(
        season_scorecard(predictions).to_json(orient="records", date_format="iso")
    )
    return {
        "schema_version": 1,
        "model": {
            "name": metrics.get("model_name"),
            "feature_set": metrics.get("feature_set"),
        },
        "intended_use": [
            "Estimate pregame NFL home-team cover probabilities for research.",
            "Compare feature and model hypotheses with leakage-safe walk-forward tests.",
            "Support paper-only ATS and pool decision experiments.",
        ],
        "out_of_scope": [
            "A claim of profitable betting performance.",
            "Automated or real-money wagering.",
            "Use when the available line or price differs from the stored quote.",
        ],
        "evaluation": {
            "first_test_gameday": metrics.get("first_test_gameday"),
            "last_test_gameday": metrics.get("last_test_gameday"),
            "games_evaluated": metrics.get("games_evaluated"),
            "training_policy": (
                "Expanding walk-forward refit before every NFL week; all training dates are "
                "strictly earlier than the first game date in the scored week."
            ),
            "headline_metrics": {
                key: metrics.get(key)
                for key in (
                    "accuracy",
                    "brier_score",
                    "log_loss",
                    "expected_calibration_error",
                    "bets",
                    "net_profit_units",
                    "roi",
                )
            },
            "season_history": season_history,
        },
        "known_limitations": [
            "Historical nflverse spreads are closing lines, not archived decision-time quotes.",
            "The current feature table uses weekly team aggregates rather than player or "
            "play-by-play state.",
            "Injury, depth-chart, and forecast-time weather histories are not yet modeled.",
            "Repeated development-season experiments can overfit model and feature selection.",
            "A historical confidence interval does not replace frozen prospective evaluation.",
        ],
        "provenance": provenance,
    }


def model_card_markdown(card: dict[str, Any]) -> str:
    model = card["model"]
    evaluation = card["evaluation"]
    metrics = evaluation["headline_metrics"]
    intended = "\n".join(f"- {item}" for item in card["intended_use"])
    out_of_scope = "\n".join(f"- {item}" for item in card["out_of_scope"])
    limitations = "\n".join(f"- {item}" for item in card["known_limitations"])
    return (
        f"# Model card: {model['name']} / {model['feature_set']}\n\n"
        "## Intended use\n\n"
        f"{intended}\n\n"
        "## Out of scope\n\n"
        f"{out_of_scope}\n\n"
        "## Walk-forward evaluation\n\n"
        f"- Period: {evaluation['first_test_gameday']} through "
        f"{evaluation['last_test_gameday']}\n"
        f"- Games evaluated: {evaluation['games_evaluated']}\n"
        f"- Accuracy: {float(metrics['accuracy']):.1%}\n"
        f"- Brier score: {float(metrics['brier_score']):.4f}\n"
        f"- Log loss: {float(metrics['log_loss']):.4f}\n"
        f"- Calibration gap: {float(metrics['expected_calibration_error']):.1%}\n"
        f"- Flat-stake paper ROI: {float(metrics['roi']):.1%}\n\n"
        f"{evaluation['training_policy']}\n\n"
        "## Known limitations\n\n"
        f"{limitations}\n"
    )
