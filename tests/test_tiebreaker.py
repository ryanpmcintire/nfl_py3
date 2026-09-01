"""Tests for the pool tiebreaker guess (owner request, 2026-09-01).

The sign-convention conversions are the part most worth pinning: schedules
``spread_line`` is positive-home-favored, an odds snapshot's HOME outcome
line is negative-home-favored, and one wrong sign silently swaps the two
teams' scores.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.tiebreaker import (
    MODEL_RESIDUAL_WEIGHT,
    MarketConsensus,
    ModelView,
    active_model_view,
    build_report,
    last_game_of_week,
    lined_finals,
    market_implied_scores,
    snapshot_consensus,
    tiebreaker_report,
    upcoming_week,
)


def _schedules() -> pd.DataFrame:
    # Three finished historical games plus one upcoming week with the
    # tiebreaker game LAST by (gameday, gametime).
    return pd.DataFrame(
        {
            "game_id": [
                "2024_01_A_B",
                "2024_01_C_D",
                "2024_02_E_F",
                "2026_01_X_Y",
                "2026_01_DEN_KC",
            ],
            "season": [2024, 2024, 2024, 2026, 2026],
            "week": [1, 1, 2, 1, 1],
            "game_type": ["REG"] * 5,
            "gameday": ["2024-09-08", "2024-09-08", "2024-09-15", "2026-09-13", "2026-09-14"],
            "gametime": ["13:00", "16:25", "13:00", "13:00", "20:15"],
            "home_team": ["B", "D", "F", "Y", "KC"],
            "away_team": ["A", "C", "E", "X", "DEN"],
            "home_score": [24.0, 20.0, 30.0, None, None],
            "away_score": [20.0, 23.0, 13.0, None, None],
            "spread_line": [3.0, 2.5, 7.0, 1.0, 2.5],
            "total_line": [43.5, 44.0, 41.0, 40.0, 43.0],
        }
    )


def test_market_implied_scores_positive_margin_favors_home() -> None:
    home, away = market_implied_scores(2.5, 43.0)
    assert home == pytest.approx(22.75)
    assert away == pytest.approx(20.25)
    assert home + away == pytest.approx(43.0)
    assert home - away == pytest.approx(2.5)


def test_last_game_of_week_uses_gametime_within_the_day() -> None:
    game = last_game_of_week(_schedules(), 2024, 1)
    assert game["game_id"] == "2024_01_C_D"  # 16:25 beats 13:00 on the same day
    assert last_game_of_week(_schedules(), 2026, 1)["game_id"] == "2026_01_DEN_KC"


def test_upcoming_week_finds_the_next_regular_week() -> None:
    assert upcoming_week(_schedules(), date(2026, 9, 1)) == (2026, 1)
    assert upcoming_week(_schedules(), date(2024, 9, 10)) == (2024, 2)


def test_snapshot_consensus_negates_the_home_line_and_takes_medians(tmp_path: Path) -> None:
    quotes = pd.DataFrame(
        {
            "nflverse_game_id": ["2026_01_DEN_KC"] * 6,
            "market": ["spreads", "spreads", "spreads", "totals", "totals", "totals"],
            "outcome_side": ["HOME", "HOME", "HOME", "OVER", "OVER", "OVER"],
            "line": [-2.5, -2.5, -3.0, 43.0, 43.5, 43.5],
            "bookmaker_key": ["book1", "book2", "book3", "book1", "book2", "book3"],
        }
    )
    snap = tmp_path / "market" / "raw" / "20260831T230102Z"
    snap.mkdir(parents=True)
    quotes.to_parquet(snap / "quotes.parquet")

    consensus = snapshot_consensus("2026_01_DEN_KC", tmp_path)
    assert consensus is not None
    assert consensus.home_expected_margin == pytest.approx(2.5)  # negated HOME median
    assert consensus.total_line == pytest.approx(43.5)
    assert "3 books" in consensus.source

    # A newer snapshot without the game falls back to the older one.
    newer = tmp_path / "market" / "raw" / "20260901T120000Z"
    newer.mkdir(parents=True)
    quotes.assign(nflverse_game_id="2026_01_X_Y").to_parquet(newer / "quotes.parquet")
    fallback = snapshot_consensus("2026_01_DEN_KC", tmp_path)
    assert fallback is not None
    assert "20260831T230102Z" in fallback.source


def test_build_report_guess_is_margin_consistent_and_sums_to_the_total() -> None:
    schedules = _schedules()
    finals = lined_finals(schedules)
    assert len(finals) == 3  # the two 2026 games have no scores
    game = schedules.iloc[4]
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC",
        home_expected_margin=2.5,
        total_line=43.0,
        source="test",
    )
    report = build_report(game, consensus, finals)
    assert report.guess_home + report.guess_away == round(report.median_total)
    assert report.home == "KC" and report.away == "DEN"
    assert report.neighborhood_games >= 1
    assert report.common_scores  # at least one exact final reported
    assert report.total_mae > 0


def test_tiebreaker_report_falls_back_to_schedules_when_no_snapshots(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")

    report = tiebreaker_report(tmp_path, season=2026, week=1)
    assert report.game_id == "2026_01_DEN_KC"
    assert "schedules" in report.consensus.source
    assert report.consensus.home_expected_margin == pytest.approx(2.5)
    assert report.implied_home == pytest.approx((43.0 + 2.5) / 2)


def _artifacts_tree(tmp_path: Path) -> Path:
    artifacts = tmp_path / "artifacts"
    forecast = artifacts / "margin_predictions" / "2026-week-01-test"
    forecast.mkdir(parents=True)
    (artifacts / "active_ats_model.json").write_text(
        '{"method": "market_residual"}', encoding="utf-8"
    )
    pd.DataFrame(
        {
            "game_id": ["2026_01_DEN_KC", "2026_01_DEN_KC"],
            "method": ["market", "market_residual"],
            "spread_line": [3.0, 3.0],
            "predicted_margin": [3.0, 4.31],
            "predicted_market_residual": [0.0, 1.31],
        }
    ).to_csv(forecast / "predictions.csv", index=False)
    return artifacts


def test_active_model_view_reads_the_active_method_row(tmp_path: Path) -> None:
    artifacts = _artifacts_tree(tmp_path)
    view = active_model_view("2026_01_DEN_KC", artifacts)
    assert view is not None
    assert view.predicted_margin == pytest.approx(4.31)  # market_residual, not market
    assert view.residual == pytest.approx(1.31)
    assert active_model_view("2026_01_NO_SUCH", artifacts) is None
    assert active_model_view("2026_01_DEN_KC", tmp_path / "empty") is None


def test_build_report_blends_the_model_residual_at_the_measured_weight() -> None:
    schedules = _schedules()
    finals = lined_finals(schedules)
    game = schedules.iloc[4]
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC",
        home_expected_margin=2.5,
        total_line=43.0,
        source="test",
    )
    view = ModelView(predicted_margin=4.31, forecast_line=3.0, residual=1.31, source="test")
    report = build_report(game, consensus, finals, view)
    assert report.guess_margin == pytest.approx(2.5 + MODEL_RESIDUAL_WEIGHT * 1.31)
    assert report.implied_home - report.implied_away == pytest.approx(report.guess_margin)
    # Without a model view the guess margin is the market's alone.
    market_only = build_report(game, consensus, finals)
    assert market_only.guess_margin == pytest.approx(2.5)
    assert market_only.model_view is None


def test_tiebreaker_report_unknown_game_id_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")
    with pytest.raises(ValueError, match="not in schedules"):
        tiebreaker_report(tmp_path, game_id="2026_01_NO_SUCH")
