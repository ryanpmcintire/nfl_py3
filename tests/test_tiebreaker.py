from __future__ import annotations

import json
import math
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nfl_ats.score_lattice as score_lattice_module
from nfl_ats.tiebreaker import (
    _MIN_NEIGHBORHOOD,
    _NEIGHBORHOOD_WINDOWS,
    MODEL_RESIDUAL_WEIGHT,
    TOTAL_LOW_SIDE_SHADE_POINTS,
    MarketConsensus,
    ModelView,
    TiebreakerConsistencyError,
    _neighborhood,
    active_model_view,
    build_report,
    effective_sample_size,
    format_report,
    kernel_weights,
    last_game_of_week,
    lined_finals,
    market_implied_scores,
    snapshot_consensus,
    tiebreaker_report,
    upcoming_week,
    weighted_median,
    weighted_score_counts,
)

_BASE_MARGIN_BANDWIDTH, _BASE_TOTAL_BANDWIDTH = 1.0, 1.5


def _schedules() -> pd.DataFrame:
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
    assert game["game_id"] == "2024_01_C_D"
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
    assert consensus.home_expected_margin == pytest.approx(2.5)
    assert consensus.total_line == pytest.approx(43.5)
    assert "3 books" in consensus.source

    newer = tmp_path / "market" / "raw" / "20260901T120000Z"
    newer.mkdir(parents=True)
    quotes.assign(nflverse_game_id="2026_01_X_Y").to_parquet(newer / "quotes.parquet")
    fallback = snapshot_consensus("2026_01_DEN_KC", tmp_path)
    assert fallback is not None
    assert "20260831T230102Z" in fallback.source


def test_build_report_guess_is_margin_consistent_and_sums_to_the_total() -> None:
    schedules = _schedules()
    finals = lined_finals(schedules)
    assert len(finals) == 3
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
    assert report.common_scores
    assert report.total_mae > 0


def test_tiebreaker_report_falls_back_to_schedules_when_no_snapshots(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")

    report = tiebreaker_report(tmp_path, season=2026, week=1)
    assert report.game_id == "2026_01_DEN_KC"
    assert "schedules" in report.consensus.source
    assert report.consensus.home_expected_margin == pytest.approx(2.5)
    assert report.implied_home == pytest.approx((43.0 + TOTAL_LOW_SIDE_SHADE_POINTS + 2.5) / 2)


def _artifacts_tree(tmp_path: Path) -> Path:
    artifacts = tmp_path / "artifacts"
    forecast = artifacts / "margin_predictions" / "2026-week-01-test"
    forecast.mkdir(parents=True)
    (artifacts / "active_ats_model.json").write_text(
        json.dumps(
            {
                "method": "market_residual",
                "model_id": "verified",
                "weekly_forecast": {"artifact": "margin_predictions/2026-week-01-test"},
            }
        ),
        encoding="utf-8",
    )
    (forecast / "metadata.json").write_text(json.dumps({"active_model_id": "verified"}))
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
    assert view.predicted_margin == pytest.approx(4.31)
    assert view.residual == pytest.approx(1.31)
    assert active_model_view("2026_01_NO_SUCH", artifacts) is None
    assert active_model_view("2026_01_DEN_KC", tmp_path / "empty") is None


def test_build_report_blends_the_model_residual_at_the_measured_weight() -> None:
    game, consensus, finals = _den_kc_game_with_dense_finals()
    view = ModelView(predicted_margin=4.31, forecast_line=3.0, residual=1.31, source="test")
    report = build_report(game, consensus, finals, view)
    assert report.guess_margin == pytest.approx(2.5 + MODEL_RESIDUAL_WEIGHT * 1.31)
    assert report.implied_home - report.implied_away == pytest.approx(report.guess_margin)
    market_only = build_report(game, consensus, finals)
    assert market_only.guess_margin == pytest.approx(2.5)
    assert market_only.model_view is None
    assert market_only.pick_side is None
    assert market_only.consistency_note == ""


def _den_kc_game_and_consensus() -> tuple[pd.Series, MarketConsensus, pd.DataFrame]:
    schedules = _schedules()
    finals = lined_finals(schedules)
    game = schedules.iloc[4]
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC", home_expected_margin=2.5, total_line=43.0, source="test"
    )
    return game, consensus, finals


def _dense_lattice_finals() -> pd.DataFrame:

    home = [23, 24, 20, 27, 17, 30, 24, 20, 27, 13, 21, 22, 21, 24, 23, 25, 26]
    away = [20, 17, 23, 20, 24, 13, 24, 17, 24, 20, 22, 21, 20, 20, 19, 18, 17]
    return pd.DataFrame(
        {
            "game_id": [f"lattice-fixture-{i}" for i in range(len(home))],
            "home_score": [float(value) for value in home],
            "away_score": [float(value) for value in away],
            "spread_line": [3.0] * len(home),
            "total_line": [43.0] * len(home),
        }
    )


def _den_kc_game_with_dense_finals() -> tuple[pd.Series, MarketConsensus, pd.DataFrame]:
    game = pd.Series({"game_id": "2026_01_DEN_KC", "home_team": "KC", "away_team": "DEN"})
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC", home_expected_margin=2.5, total_line=43.0, source="test"
    )
    return game, consensus, _dense_lattice_finals()


def test_build_report_never_produces_a_push_against_a_home_favorite_pick() -> None:

    game, consensus, finals = _den_kc_game_with_dense_finals()
    view = ModelView(predicted_margin=3.19, forecast_line=3.0, residual=0.19, source="test")
    report = build_report(game, consensus, finals, view)
    assert report.pick_side == "HOME"
    assert report.pick_spread_line == pytest.approx(3.0)
    margin = report.guess_home - report.guess_away
    assert margin > 3.0
    total = report.guess_home + report.guess_away
    assert abs(total - report.guess_total_line) <= 1.0
    assert f"consistent with the {report.home}" in report.consistency_note
    assert report.pick_cover_probability is not None
    assert 0.0 <= report.pick_cover_probability <= 1.0
    assert report.pick_push_probability is not None


def test_build_report_dog_pick_selects_the_away_side_consistently() -> None:

    game, consensus, finals = _den_kc_game_with_dense_finals()
    view = ModelView(predicted_margin=-1.0, forecast_line=3.0, residual=-4.0, source="test")
    report = build_report(game, consensus, finals, view)
    assert report.pick_side == "AWAY"
    margin = report.guess_home - report.guess_away
    assert margin < 3.0
    assert f"consistent with the {report.away}" in report.consistency_note


def test_build_report_a_pickem_residual_never_triggers_lattice_consistency() -> None:

    game, consensus, finals = _den_kc_game_and_consensus()
    view = ModelView(predicted_margin=3.0, forecast_line=3.0, residual=0.0, source="test")
    report = build_report(game, consensus, finals, view)
    assert report.pick_side is None
    assert report.consistency_note == ""


def test_build_report_raises_when_the_lattice_has_no_consistent_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(score_lattice_module, "pick_consistent_top_score", lambda *a, **k: None)
    game, consensus, finals = _den_kc_game_and_consensus()
    view = ModelView(predicted_margin=3.19, forecast_line=3.0, residual=0.19, source="test")
    with pytest.raises(TiebreakerConsistencyError):
        build_report(game, consensus, finals, view)


def test_build_report_raises_when_the_lattice_score_drifts_from_the_served_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        score_lattice_module, "pick_consistent_top_score", lambda *a, **k: (100, 50, 0.5, 2.0)
    )
    game, consensus, finals = _den_kc_game_and_consensus()
    view = ModelView(predicted_margin=3.19, forecast_line=3.0, residual=0.19, source="test")
    with pytest.raises(TiebreakerConsistencyError):
        build_report(game, consensus, finals, view)


def test_build_report_raises_a_consistency_error_when_the_lattice_itself_cannot_be_built() -> None:

    game, consensus, finals = _den_kc_game_and_consensus()
    view = ModelView(predicted_margin=-1.0, forecast_line=3.0, residual=-4.0, source="test")
    with pytest.raises(TiebreakerConsistencyError):
        build_report(game, consensus, finals, view)


def test_tiebreaker_report_has_no_totals_view_without_a_feature_table(tmp_path: Path) -> None:

    raw = tmp_path / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")

    report = tiebreaker_report(tmp_path, season=2026, week=1)
    assert report.totals_view is None
    assert report.guess_total_line == pytest.approx(
        report.consensus.total_line + TOTAL_LOW_SIDE_SHADE_POINTS
    )
    assert report.implied_home + report.implied_away == pytest.approx(
        43.0 + TOTAL_LOW_SIDE_SHADE_POINTS
    )


def test_tiebreaker_report_uses_the_wave2_totals_view_when_the_pbp_table_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    from nfl_ats.totals import TotalsView

    raw = tmp_path / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({"game_id": []}).to_parquet(processed / "game_features_pbp.parquet")

    sentinel = TotalsView(
        predicted_total=43.4,
        market_total=43.0,
        residual=0.4,
        train_games=4_630,
        source="totals ridge wave 2 (65 cols, drive pace, alpha=10) trained on 4630 games",
    )
    calls: list[str] = []

    def _fake_wave2(game_id: str, *_args: object, **_kwargs: object) -> TotalsView:
        calls.append(game_id)
        return sentinel

    monkeypatch.setattr("nfl_ats.tiebreaker.model_total_view_wave2", _fake_wave2)
    monkeypatch.setattr(
        "nfl_ats.tiebreaker.model_total_view",
        lambda *a, **k: pytest.fail(
            "wave-1 model_total_view must not be called when wave 2's feature table is present"
        ),
    )

    report = tiebreaker_report(tmp_path, season=2026, week=1)
    assert calls == ["2026_01_DEN_KC"]
    assert report.totals_view is sentinel
    assert "wave 2" in report.totals_view.source
    assert report.guess_total_line == pytest.approx(43.0 + 0.1 * 0.4 + TOTAL_LOW_SIDE_SHADE_POINTS)


def test_tiebreaker_report_falls_back_to_the_wave1_totals_view_when_the_pbp_table_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    from nfl_ats.totals import TotalsView

    raw = tmp_path / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({"game_id": []}).to_parquet(processed / "game_features.parquet")

    sentinel = TotalsView(
        predicted_total=43.1,
        market_total=43.0,
        residual=0.1,
        train_games=500,
        source="totals ridge(alpha=10) trained on 500 games before 2026 week 1",
    )

    def _fake_wave1(game_id: str, *_args: object, **_kwargs: object) -> TotalsView:
        return sentinel

    def _fail_wave2(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "wave-2 model_total_view_wave2 must not be called when its feature table is absent"
        )

    monkeypatch.setattr("nfl_ats.tiebreaker.model_total_view", _fake_wave1)
    monkeypatch.setattr("nfl_ats.tiebreaker.model_total_view_wave2", _fail_wave2)

    report = tiebreaker_report(tmp_path, season=2026, week=1)
    assert report.totals_view is not None
    assert report.totals_view.residual == pytest.approx(0.1)
    assert "wave 1 fallback" in report.totals_view.source
    assert "PBP table absent" in report.totals_view.source
    assert "totals ridge(alpha=10)" in report.totals_view.source


def test_tiebreaker_report_fails_closed_when_wave2_input_is_misaligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    from nfl_ats.totals import TotalsDataError

    raw = tmp_path / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({"game_id": []}).to_parquet(processed / "game_features_pbp.parquet")

    def _fail_wave2(*_args: object, **_kwargs: object) -> None:
        raise TotalsDataError("wave-2 table is stale")

    monkeypatch.setattr("nfl_ats.tiebreaker.model_total_view_wave2", _fail_wave2)
    monkeypatch.setattr(
        "nfl_ats.tiebreaker.model_total_view",
        lambda *a, **k: pytest.fail("stale wave-2 data must not fall back to wave 1"),
    )

    report = tiebreaker_report(tmp_path, season=2026, week=1)
    assert report.totals_view is None
    assert report.guess_total_line == pytest.approx(
        report.consensus.total_line + TOTAL_LOW_SIDE_SHADE_POINTS
    )


def test_tiebreaker_report_unknown_game_id_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")
    with pytest.raises(ValueError, match="not in schedules"):
        tiebreaker_report(tmp_path, game_id="2026_01_NO_SUCH")


def _bucket_history(buckets: list[tuple[float, int, int]]) -> pd.DataFrame:

    rows = []
    index = 0
    for total_line, actual_total, count in buckets:
        for _ in range(count):
            rows.append(
                {
                    "game_id": f"2020_01_{index:05d}",
                    "season": 2020,
                    "week": 1,
                    "game_type": "REG",
                    "gameday": "2020-09-10",
                    "gametime": "13:00",
                    "home_team": "H",
                    "away_team": "A",
                    "home_score": float(actual_total - 10),
                    "away_score": 10.0,
                    "spread_line": 2.5,
                    "total_line": float(total_line),
                }
            )
            index += 1
    return pd.DataFrame(rows)


def _upcoming_game(total_line: float) -> pd.Series:
    return pd.Series(
        {
            "game_id": "2026_01_DEN_KC",
            "season": 2026,
            "week": 1,
            "game_type": "REG",
            "gameday": "2026-09-14",
            "gametime": "20:15",
            "home_team": "KC",
            "away_team": "DEN",
            "home_score": None,
            "away_score": None,
            "spread_line": 2.5,
            "total_line": total_line,
        }
    )


def _hard_window_rows(finals: pd.DataFrame, margin: float, total: float) -> pd.DataFrame:

    for window in _NEIGHBORHOOD_WINDOWS:
        if window is None:
            break
        margin_width, total_width = window
        rows = finals.loc[
            (finals["spread_line"] - margin).abs().le(margin_width)
            & (finals["total_line"] - total).abs().le(total_width)
        ]
        if len(rows) >= _MIN_NEIGHBORHOOD:
            return rows
    return finals


def _hard_window_median(finals: pd.DataFrame, margin: float, total: float) -> float:
    rows = _hard_window_rows(finals, margin, total)
    return float((rows["home_score"] + rows["away_score"]).median())


def test_base_bandwidths_are_inherited_from_the_first_schedule_entry() -> None:

    assert _NEIGHBORHOOD_WINDOWS[0] == (_BASE_MARGIN_BANDWIDTH, _BASE_TOTAL_BANDWIDTH)
    assert _NEIGHBORHOOD_WINDOWS[-1] is None
    assert _MIN_NEIGHBORHOOD == 150


def test_kernel_weights_are_one_at_the_centre_zero_beyond_the_bandwidth() -> None:

    finals = pd.DataFrame(
        {
            "spread_line": [2.5, 2.5, 2.5, 3.0, 3.5, 4.5, 2.5],
            "total_line": [43.0, 43.75, 44.5, 43.0, 43.0, 43.0, 100.0],
        }
    )
    weights = kernel_weights(finals, 2.5, 43.0, _BASE_MARGIN_BANDWIDTH, _BASE_TOTAL_BANDWIDTH)
    assert weights.min() >= 0.0 and weights.max() <= 1.0
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5)
    assert weights[2] == pytest.approx(0.0)
    assert weights[3] == pytest.approx(0.5)
    assert weights[4] == pytest.approx(0.0)
    assert weights[5] == pytest.approx(0.0)
    assert weights[6] == pytest.approx(0.0)


def test_effective_sample_size_equals_the_count_for_equal_weights() -> None:

    assert effective_sample_size(np.ones(37)) == pytest.approx(37.0)
    assert effective_sample_size(np.full(37, 0.25)) == pytest.approx(37.0)
    assert effective_sample_size(np.zeros(5)) == 0.0
    mixed = np.array([1.0, 1.0, 0.5, 0.5])
    assert 3.0 < effective_sample_size(mixed) < 4.0


def test_weighted_median_reproduces_pandas_for_uniform_weights() -> None:

    for values in ([41.0, 43.0], [41.0, 43.0, 47.0], [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]):
        array = np.array(values, dtype=float)
        assert weighted_median(array, np.ones(len(array))) == pytest.approx(
            float(pd.Series(values).median())
        )
    assert math.isnan(weighted_median(np.array([]), np.array([])))
    assert math.isnan(weighted_median(np.array([1.0, 2.0]), np.zeros(2)))
    assert weighted_median(np.array([10.0, 50.0, 60.0]), np.array([9.0, 1.0, 1.0])) == 10.0


def test_weighted_score_counts_are_weighted_and_sum_to_the_total_weight() -> None:

    finals = lined_finals(_bucket_history([(43.0, 43, 200), (44.0, 47, 200)]))
    hood = _neighborhood(finals, 2.5, 43.0)
    counts = weighted_score_counts(hood.frame, hood.weights)
    assert set(counts) == {(33, 10), (37, 10)}
    assert sum(counts.values()) == pytest.approx(float(hood.weights.sum()))
    assert counts[(33, 10)] == pytest.approx(200.0)
    assert counts[(37, 10)] == pytest.approx(200.0 / 3.0)

    report = build_report(
        _upcoming_game(43.0),
        MarketConsensus(
            game_id="2026_01_DEN_KC", home_expected_margin=2.5, total_line=43.0, source="test"
        ),
        finals,
    )
    shaded_hood = _neighborhood(finals, 2.5, 43.0 + TOTAL_LOW_SIDE_SHADE_POINTS)
    shaded_counts = weighted_score_counts(shaded_hood.frame, shaded_hood.weights)
    ranked = sorted(shaded_counts.items(), key=lambda item: (-item[1], item[0]))
    assert [(home, away) for (home, away), _ in ranked[:3]] == [
        (home, away) for home, away, _ in report.common_scores
    ]
    assert report.common_scores[0][2] == pytest.approx(200.0 / 3.0)
    assert "(66.7x)" in format_report(report)


def test_neighborhood_does_not_widen_when_the_base_bandwidth_already_clears() -> None:

    finals = lined_finals(_bucket_history([(43.0, 43, 400)]))
    hood = _neighborhood(finals, 2.5, 43.0)
    assert hood.label == "±1.00 margin, ±1.50 total"
    assert hood.effective_size == pytest.approx(400.0)


def test_neighborhood_widens_to_the_effective_size_floor_and_stops_there() -> None:

    finals = lined_finals(_bucket_history([(43.0, 43, 100), (45.0, 49, 400)]))
    base = kernel_weights(finals, 2.5, 43.0, _BASE_MARGIN_BANDWIDTH, _BASE_TOTAL_BANDWIDTH)
    assert effective_sample_size(base) == pytest.approx(100.0)

    hood = _neighborhood(finals, 2.5, 43.0)
    assert hood.effective_size >= _MIN_NEIGHBORHOOD
    assert hood.effective_size == pytest.approx(float(_MIN_NEIGHBORHOOD), abs=0.5)
    margin_bandwidth = float(hood.label.split("±")[1].split(" ")[0])
    total_bandwidth = float(hood.label.split("±")[2].split(" ")[0])
    assert _BASE_MARGIN_BANDWIDTH < margin_bandwidth < 1.5
    assert _BASE_TOTAL_BANDWIDTH < total_bandwidth < 2.5


def test_neighborhood_falls_back_to_all_history_when_the_schedule_cannot_reach_the_floor() -> None:

    finals = lined_finals(_schedules())
    hood = _neighborhood(finals, 2.5, 43.0)
    assert hood.label == "all history"
    assert len(hood.frame) == len(finals) == 3
    assert hood.weights.tolist() == [1.0, 1.0, 1.0]
    assert hood.effective_size == pytest.approx(3.0)


def test_kernel_neighborhood_does_not_flip_the_guess_on_a_sub_quantum_nudge() -> None:

    finals = lined_finals(_bucket_history([(41.5, 41, 160), (43.0, 43, 160), (44.0, 47, 160)]))
    game = _upcoming_game(43.0)
    consensus = MarketConsensus(
        game_id="2026_01_DEN_KC", home_expected_margin=2.5, total_line=43.0, source="test"
    )

    assert _hard_window_median(finals, 2.5, 43.0) == 43.0
    assert _hard_window_median(finals, 2.5, 43.0421) == 45.0

    before = build_report(game, consensus, finals)
    after = build_report(
        game,
        MarketConsensus(
            game_id="2026_01_DEN_KC",
            home_expected_margin=2.5,
            total_line=43.0421,
            source="test",
        ),
        finals,
    )
    assert after.guess_total_line > before.guess_total_line
    assert after.median_total == before.median_total == 41.0
    assert (after.guess_home, after.guess_away) == (before.guess_home, before.guess_away)
    assert [(home, away) for home, away, _ in after.common_scores] == [
        (home, away) for home, away, _ in before.common_scores
    ]


def _dense_history() -> pd.DataFrame:

    lines = [30.0 + 0.5 * step for step in range(53)]
    return _bucket_history([(line, math.floor(line), 50) for line in lines])


def test_weighted_median_total_is_monotone_and_gentle_in_the_centre() -> None:

    finals = lined_finals(_dense_history())
    game = _upcoming_game(43.0)

    centres = [40.0 + 0.05 * step for step in range(121)]
    medians = []
    for centre in centres:
        report = build_report(
            game,
            MarketConsensus(
                game_id="2026_01_DEN_KC",
                home_expected_margin=2.5,
                total_line=centre,
                source="test",
            ),
            finals,
        )
        assert report.neighborhood_games >= _MIN_NEIGHBORHOOD
        medians.append(report.median_total)

    for previous, current, centre in zip(medians[:-1], medians[1:], centres[1:], strict=True):
        assert current >= previous, f"weighted median moved DOWN at centre {centre}"
        assert current - previous <= 1.0, f"weighted median jumped >1 point at centre {centre}"
    assert medians[-1] > medians[0]

    hard_counts = [len(_hard_window_rows(finals, 2.5, centre)) for centre in centres]
    assert max(abs(current - previous) for previous, current in pairwise(hard_counts)) >= 50

    sizes = [_neighborhood(finals, 2.5, centre).effective_size for centre in centres]
    assert max(abs(current - previous) for previous, current in pairwise(sizes)) < 10.0


def test_published_flip_overrides_raw_residual_sign() -> None:
    game, consensus, finals = _den_kc_game_with_dense_finals()
    view = ModelView(predicted_margin=3.19, forecast_line=3.0, residual=0.19, source="test")
    report = build_report(game, consensus, finals, view, published_pick_side="AWAY")
    assert report.pick_side == "AWAY"
    assert report.guess_home - report.guess_away < 3.0


def test_newer_unpromoted_forecast_cannot_change_active_margin(tmp_path: Path) -> None:
    artifacts = _artifacts_tree(tmp_path)
    newer = artifacts / "margin_predictions" / "2099-unpromoted"
    newer.mkdir()
    pd.DataFrame(
        [
            {
                "game_id": "2026_01_DEN_KC",
                "method": "market_residual",
                "spread_line": 3,
                "predicted_margin": -99,
                "predicted_market_residual": -102,
            }
        ]
    ).to_csv(newer / "predictions.csv", index=False)
    view = active_model_view("2026_01_DEN_KC", artifacts)
    assert view is not None
    assert view.predicted_margin == 4.31


def test_handed_forecast_must_match_verified_model(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "fixture"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")
    row = pd.Series(
        {
            "game_id": "2026_01_DEN_KC",
            "spread_line": 3,
            "predicted_margin": 3.19,
            "predicted_market_residual": 0.19,
        }
    )
    with pytest.raises(TiebreakerConsistencyError, match="model ID"):
        tiebreaker_report(
            tmp_path,
            season=2026,
            week=1,
            forecast_row=row,
            forecast_model_id="unpromoted",
            model_id="active",
            published_pick_side="AWAY",
            frozen_spread=3,
        )


def test_published_forecast_bypasses_disk_scan_and_preserves_flipped_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nfl_ats.tiebreaker as module

    raw = tmp_path / "raw" / "fixture"
    raw.mkdir(parents=True)
    _schedules().to_parquet(raw / "schedules.parquet")

    def refuse_scan(*args):
        pytest.fail("Published tiebreaker must not scan disk for a different forecast")

    monkeypatch.setattr(module, "active_model_view", refuse_scan)
    monkeypatch.setattr(module, "lined_finals", lambda _: _dense_lattice_finals())
    row = pd.Series(
        {
            "game_id": "2026_01_DEN_KC",
            "spread_line": 3,
            "predicted_margin": 3.19,
            "predicted_market_residual": 0.19,
        }
    )
    report = tiebreaker_report(
        tmp_path,
        artifacts_root=tmp_path / "artifacts",
        season=2026,
        week=1,
        forecast_row=row,
        forecast_model_id="active",
        model_id="active",
        published_pick_side="AWAY",
        frozen_spread=3,
    )
    assert report.model_view is not None
    assert report.model_view.predicted_margin == 3.19
    assert report.pick_side == "AWAY"
    assert report.pick_spread_line == 3
    assert report.guess_home - report.guess_away < 3
