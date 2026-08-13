from __future__ import annotations

import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nfl_ats.dashboard import (
    _load_weekly_ats_forecast,
    _matching_historical_ats_benchmark,
    _matchup_signal_table,
    _readable_recommendations,
)


def test_readable_forecast_names_the_ats_team_line_and_signal() -> None:
    predictions = pd.DataFrame(
        {
            "gameday": ["2026-09-10", "2026-09-13"],
            "away_team": ["NE", "BAL"],
            "home_team": ["SEA", "IND"],
            "spread_line": [3.5, -3.5],
            "home_cover_probability": [0.498, 0.46],
            "bet_side": ["PASS", "AWAY"],
            "edge": [-0.02, 0.03],
        }
    )

    readable = _readable_recommendations(predictions)

    assert readable["ATS prediction"].tolist() == ["NE +3.5", "BAL -3.5"]
    assert readable["Signal"].tolist() == ["No separation", "Moderate lean"]
    assert readable["Paper action"].tolist() == ["PASS", "Paper pick: BAL"]


def test_weekly_forecast_uses_declared_outcome_method(tmp_path) -> None:
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    (forecast / "metadata.json").write_text(
        json.dumps({"command": "margin-predict", "ats_method": "market_residual"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "game_id": ["game", "game"],
            "method": ["direct_ats", "market_residual"],
            "home_cover_probability": [0.503, 0.616],
        }
    ).to_csv(forecast / "predictions.csv", index=False)

    recommendations, metadata = _load_weekly_ats_forecast(forecast)

    assert metadata["ats_method"] == "market_residual"
    assert recommendations["method"].tolist() == ["market_residual"]
    assert recommendations["home_cover_probability"].tolist() == [0.616]


def test_weekly_forecast_benchmark_matches_profile_and_method(tmp_path) -> None:
    benchmark = tmp_path / "margins" / "run"
    benchmark.mkdir(parents=True)
    (benchmark / "metadata.json").write_text(
        json.dumps(
            {
                "feature_profile": "player",
                "regressor": "ridge",
                "ridge_alpha": 10.0,
                "calibration_method": "none",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "method": ["direct_ats", "market_residual"],
            "cover_accuracy": [0.50, 0.5205],
            "cover_games": [2075, 2075],
        }
    ).to_csv(benchmark / "summary.csv", index=False)
    pd.DataFrame(
        {
            "method": ["market_residual"],
            "metric": ["cover_accuracy"],
            "block": ["week"],
            "lower": [0.4985],
            "upper": [0.5425],
        }
    ).to_csv(benchmark / "uncertainty.csv", index=False)

    result = _matching_historical_ats_benchmark(
        tmp_path,
        {
            "feature_profile": "player",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "calibration_method": "none",
            "ats_method": "market_residual",
        },
    )

    assert result is not None
    assert result["accuracy"] == pytest.approx(0.5205)
    assert result["games"] == 2075
    assert result["lower"] == pytest.approx(0.4985)


def test_matchup_signals_translate_home_away_directions() -> None:
    signals = _matchup_signal_table(
        pd.Series(
            {
                "home_team": "HOME",
                "away_team": "AWAY",
                "elo_diff": 25.0,
                "diff_def_epa_per_play": 0.05,
                "diff_injury_offense_unavailability": -0.2,
            }
        )
    ).set_index("Pregame signal")

    assert signals.loc["Pregame Elo difference", "Direction"] == "Leans HOME"
    assert signals.loc["Defensive EPA allowed difference", "Direction"] == "Leans AWAY"
    assert signals.loc["Offensive injury unavailability difference", "Direction"] == "Leans HOME"


def test_dashboard_pages_have_graceful_empty_states(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    app = AppTest.from_file("src/nfl_ats/dashboard.py")
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "NFL ATS Research Lab"
    for page in (
        "This week's predictions",
        "Does the model work?",
        "Why these picks?",
        "Data & system health",
        "Advanced research",
    ):
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
    app.sidebar.radio[0].set_value("Advanced research").run(timeout=30)
    app.selectbox[0].set_value("Player availability experiments").run(timeout=30)
    assert not app.exception
    assert any("No player-family comparison" in info.value for info in app.info)


def test_player_selection_gate_explains_failed_promotion(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    gate = artifacts / "player_model_selection" / "run"
    gate.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "player_qb_continuity|ridge_alpha=1|calibration=none",
                "feature_profile": "player_qb_continuity",
                "ridge_alpha": 1.0,
                "calibration_method": "none",
                "cover_games": 2075,
                "cover_accuracy": 0.5263,
                "cover_brier_score": 0.2531,
                "cover_ece": 0.0405,
            },
            {
                "candidate_id": "base|ridge_alpha=10|calibration=none",
                "feature_profile": "base",
                "ridge_alpha": 10.0,
                "calibration_method": "none",
                "cover_games": 2075,
                "cover_accuracy": 0.5108,
                "cover_brier_score": 0.2521,
                "cover_ece": 0.0456,
            },
        ]
    ).to_csv(gate / "candidate_summary.csv", index=False)
    pd.DataFrame([{"cover_accuracy": 0.5070}]).to_csv(gate / "nested_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric": "accuracy_improvement",
                "block": "week",
                "estimate": -0.0019,
                "lower": -0.0248,
                "upper": 0.0221,
            }
        ]
    ).to_csv(gate / "nested_paired_comparisons.csv", index=False)
    pd.DataFrame(
        [
            {
                "test_season": 2025,
                "candidate_id": "base|ridge_alpha=10|calibration=none",
                "validation_accuracy": 0.53,
                "test_accuracy": 0.49,
            }
        ]
    ).to_csv(gate / "nested_fold_summary.csv", index=False)

    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts))
    app = AppTest.from_file("src/nfl_ats/dashboard.py")
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("Advanced research").run(timeout=30)
    app.selectbox[0].set_value("Player availability experiments").run(timeout=30)

    assert not app.exception
    assert any("Promotion verdict: NO" in warning.value for warning in app.warning)


def test_participation_gate_retains_negative_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    experiment = artifacts / "participation_experiments" / "run"
    experiment.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "feature_profile": "player_value",
                "cover_accuracy": 0.5214,
                "cover_brier_score": 0.2530,
            },
            {
                "feature_profile": "player_participation",
                "cover_accuracy": 0.5171,
                "cover_brier_score": 0.2538,
            },
        ]
    ).to_csv(experiment / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric": "accuracy_improvement",
                "block": "week",
                "estimate": -0.0043,
                "lower": -0.0153,
                "upper": 0.0063,
            }
        ]
    ).to_csv(experiment / "paired_comparisons.csv", index=False)
    pd.DataFrame(
        [
            {"feature_profile": "player_value", "season": 2025, "cover_accuracy": 0.4982},
            {
                "feature_profile": "player_participation",
                "season": 2025,
                "cover_accuracy": 0.4908,
            },
        ]
    ).to_csv(experiment / "season_summary.csv", index=False)

    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts))
    app = AppTest.from_file("src/nfl_ats/dashboard.py")
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("Advanced research").run(timeout=30)
    app.selectbox[0].set_value("Player availability experiments").run(timeout=30)

    assert not app.exception
    assert any("Participation verdict: NO" in warning.value for warning in app.warning)
    assert any(metric.label == "With participation" for metric in app.metric)


def test_learned_availability_gate_explains_unresolved_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    experiment = artifacts / "availability_experiments" / "run"
    experiment.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "availability_method": "fixed",
                "cover_accuracy": 0.5214,
                "cover_brier_score": 0.2531,
            },
            {
                "availability_method": "learned",
                "cover_accuracy": 0.5224,
                "cover_brier_score": 0.2530,
            },
        ]
    ).to_csv(experiment / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric": "accuracy_improvement",
                "block": "week",
                "estimate": 0.0010,
                "lower": -0.0063,
                "upper": 0.0078,
            }
        ]
    ).to_csv(experiment / "paired_comparisons.csv", index=False)
    pd.DataFrame(
        [
            {"availability_method": "fixed", "season": 2025, "cover_accuracy": 0.4982},
            {"availability_method": "learned", "season": 2025, "cover_accuracy": 0.4908},
        ]
    ).to_csv(experiment / "season_summary.csv", index=False)
    (experiment / "metadata.json").write_text(
        json.dumps(
            {
                "learned_provenance": {
                    "feature_table": {"manifest": {"availability_brier_improvement": 0.00444}}
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts))
    app = AppTest.from_file("src/nfl_ats/dashboard.py")
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("Advanced research").run(timeout=30)
    app.selectbox[0].set_value("Player availability experiments").run(timeout=30)

    assert not app.exception
    assert any("Availability verdict: PROMISING BUT UNRESOLVED" in w.value for w in app.warning)
    assert any(metric.label == "Learned rates ATS" for metric in app.metric)
