"""Leakage and copy-isolation tests for the Section 5 historical proxy."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inactives_channel_historical_screen import (
    _experiment_metrics,
    _identity_roles,
    apply_increments,
)


def test_historical_proxy_roles_use_only_strictly_earlier_snaps() -> None:
    snaps = pd.DataFrame(
        {
            "season": [2020, 2020],
            "week": [1, 2],
            "gsis_id": ["p1", "p1"],
            "position": ["WR", "WR"],
            "offense_pct": [0.8, 0.9],
            "defense_pct": [0.0, 0.0],
            "st_pct": [0.2, 0.2],
        }
    )
    assert _identity_roles(snaps, 2020, 1) == {}
    assert _identity_roles(snaps, 2020, 2)["p1"]["offense_pct"] == 0.8


def test_apply_increments_isolated_to_named_game() -> None:
    features = pd.DataFrame(
        {
            "game_id": ["target", "training"],
            "home_injury_offense_unavailability": [0.1, 0.2],
            "away_injury_offense_unavailability": [0.3, 0.4],
            "diff_injury_offense_unavailability": [-0.2, -0.2],
        }
    )
    adjusted = apply_increments(
        features,
        {
            "target": {
                "home": {"injury_offense_unavailability": 0.5},
                "away": {"injury_offense_unavailability": 0.0},
            }
        },
    )
    assert adjusted.loc[0, "home_injury_offense_unavailability"] == 0.6
    assert adjusted.loc[0, "diff_injury_offense_unavailability"] == 0.3
    pd.testing.assert_series_equal(adjusted.loc[1], features.loc[1], check_names=False)


def test_experiment_metrics_retain_screen_headlines() -> None:
    result = {
        "status": "scored",
        "population": {"paired_games": 429, "paired_weeks": 33},
        "candidate_vs_tuesday": {"week": {"estimate": -0.014, "probability_positive": 0.0418}},
        "positive_control_oracle": {"week": {"estimate": 0.459}},
    }
    assert _experiment_metrics(result) == {
        "status": "scored",
        "paired_games": 429,
        "paired_weeks": 33,
        "candidate_week_estimate": -0.014,
        "candidate_week_probability_positive": 0.0418,
        "positive_control_week_estimate": 0.459,
    }
