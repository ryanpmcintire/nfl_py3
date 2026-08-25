from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Reuse the synthetic historical-backfill store builders from the clv test
# module so both suites exercise the same on-disk snapshot contract.
from test_clv import _event, _spread_book, _store_snapshot

from nfl_ats.estimation_variance import BootstrapDegeneracyError, BootstrapDegeneracyWarning
from nfl_ats.reporting import (
    CLV_STATUS_CAPTURE_UNAVAILABLE,
    CLV_STATUS_MEASURED,
    CLV_STATUS_NO_PAIRED_GAMES,
    artifact_directories,
    bankroll_curve,
    block_bootstrap_intervals,
    calibration_table,
    feature_coverage,
    read_json,
    season_scorecard,
)


def test_artifact_discovery_and_json(tmp_path) -> None:
    older = tmp_path / "20220101"
    newer = tmp_path / "20220102"
    older.mkdir()
    newer.mkdir()
    (older / "metrics.json").write_text('{"value": 1}', encoding="utf-8")
    (newer / "metrics.json").write_text('{"value": 2}', encoding="utf-8")
    assert artifact_directories(tmp_path, "metrics.json") == [newer, older]
    assert read_json(newer / "metrics.json") == {"value": 2}
    (newer / "bad.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_json(newer / "bad.json")


def test_probability_and_season_summaries(model_frame: pd.DataFrame) -> None:
    predictions = model_frame.copy()
    predictions["home_cover_probability"] = 0.55
    predictions["bet_side"] = "HOME"
    predictions["bet_odds"] = -110.0
    calibration = calibration_table(predictions, bins=5)
    assert calibration["games"].sum() == len(predictions)
    scorecard = season_scorecard(predictions)
    assert scorecard["games"].sum() == len(predictions)
    assert "bet_coverage" in scorecard
    assert "log_loss" in scorecard
    assert "expected_calibration_error" in scorecard

    intervals = block_bootstrap_intervals(predictions, samples=50, block="week", seed=7)
    assert {"accuracy", "brier_score", "roi"}.issubset(set(intervals["metric"]))
    assert intervals["lower"].le(intervals["upper"]).all()
    season_intervals = block_bootstrap_intervals(predictions, samples=50, block="season", seed=7)
    assert season_intervals["block"].eq("season").all()


def test_bankroll_curve_and_feature_coverage(model_frame: pd.DataFrame) -> None:
    ledger = model_frame.head(4).copy()
    ledger["bankroll_before_week"] = 100.0
    ledger["bankroll_after_week"] = [101.0, 101.0, 99.0, 99.0]
    curve = bankroll_curve(ledger)
    assert curve.iloc[0]["season_week"] == "start"
    assert curve["drawdown"].min() < 0.0
    model_frame.loc[0, "temp"] = float("nan")
    coverage = feature_coverage(model_frame, ("temp", "wind"))
    assert coverage.iloc[0]["feature"] == "temp"
    assert coverage.iloc[0]["missing"] == 1


def test_empty_reporting_results(tmp_path) -> None:
    assert artifact_directories(tmp_path / "missing", "x") == []
    assert calibration_table(pd.DataFrame({"home_cover": []})).empty
    empty_scorecard = season_scorecard(pd.DataFrame({"home_cover": []}))
    assert empty_scorecard.empty
    assert list(empty_scorecard.columns)[-3:] == ["clv_points", "clv_status", "clv_games"]
    assert bankroll_curve(pd.DataFrame()).empty


def test_bootstrap_validation(model_frame: pd.DataFrame) -> None:
    predictions = model_frame.assign(home_cover_probability=0.5)
    with pytest.raises(ValueError, match="samples"):
        block_bootstrap_intervals(predictions, samples=9)
    with pytest.raises(ValueError, match="confidence"):
        block_bootstrap_intervals(predictions, confidence=1.0)
    with pytest.raises(ValueError, match="block"):
        block_bootstrap_intervals(predictions, block="game")  # type: ignore[arg-type]


def test_block_bootstrap_intervals_flags_a_degenerate_block_count() -> None:
    """REGRESSION TEST for D4 (``docs/estimation_variance.md`` sec 13): this
    single-arm estimator reports levels, not paired deltas, but is exactly as
    vulnerable to a low block count as
    ``experiments.paired_feature_comparisons`` and must carry the same guard.
    """

    rows = []
    for game in range(24):
        rows.append(
            {
                "season": 2020 + game // 6,
                "week": 1 + game % 6,
                "home_cover": float(game % 2),
                "home_cover_probability": 0.9 if game % 2 else 0.1,
            }
        )
    predictions = pd.DataFrame(rows)

    with pytest.warns(BootstrapDegeneracyWarning, match="bootstrap blocks"):
        season_blocked = block_bootstrap_intervals(predictions, samples=50, block="season", seed=7)
    assert season_blocked["blocks"].eq(4).all()
    assert season_blocked["degenerate_blocks"].all(), (
        "a 4-block interval must be flagged; leaving it unflagged is the D4 defect"
    )

    # Same games, same estimates -- only the blocking choice differs. Week
    # blocking gives 24 blocks and is not flagged.
    week_blocked = block_bootstrap_intervals(predictions, samples=50, block="week", seed=7)
    assert week_blocked["blocks"].eq(24).all()
    assert not week_blocked["degenerate_blocks"].any()
    assert week_blocked["estimate"].to_numpy() == pytest.approx(
        season_blocked["estimate"].to_numpy()
    )

    with pytest.raises(BootstrapDegeneracyError):
        block_bootstrap_intervals(
            predictions,
            samples=50,
            block="season",
            seed=7,
            on_degenerate="raise",
        )


# ---------------------------------------------------------------------------
# Optional CLV columns: season_scorecard against a market-capture archive
# ---------------------------------------------------------------------------


def _clv_predictions() -> pd.DataFrame:
    """Two completed games, one per season, with known forced picks.

    KC-CIN: model takes HOME (probability 0.7). SEA-NE: model takes AWAY
    (probability 0.3). ``spread_line`` is the nflverse schedule close used as
    the clv pipeline's fallback close.
    """

    return pd.DataFrame(
        {
            "game_id": ["2024_02_CIN_KC", "2023_02_NE_SEA"],
            "season": [2024, 2023],
            "week": [2, 2],
            "home_team": ["KC", "SEA"],
            "away_team": ["CIN", "NE"],
            "kickoff": [
                pd.Timestamp("2024-09-13T00:20:00Z"),
                pd.Timestamp("2023-09-15T17:00:00Z"),
            ],
            "spread_line": [1.5, 3.75],
            "home_cover_probability": [0.7, 0.3],
            "home_cover": [1.0, 0.0],
        }
    )


def _clv_store(root: Path) -> Path:
    """Archive with tue_open + sun_late_close snapshots for both games.

    Hand-computed pairing (median across 2 identical books):
    - KC-CIN tue_open home_spread = +1.5, sun_late_close = +4.0, both captured
      before the Thursday 00:20Z kickoff.
    - SEA-NE tue_open home_spread = +2.5, sun_late_close = +1.0, both captured
      before the Sunday 17:00Z kickoff.
    """

    schedule = _clv_predictions()
    snapshots = [
        (2024, "tue_open", "2024-09-10T13:00:00Z", 1.5),
        (2024, "sun_late_close", "2024-09-12T20:15:00Z", 4.0),
        (2023, "tue_open", "2023-09-10T13:00:00Z", 2.5),
        (2023, "sun_late_close", "2023-09-14T20:15:00Z", 1.0),
    ]
    for season, label, snapshot_time, spread in snapshots:
        if season == 2024:
            events = [
                _event(
                    "kc-cin",
                    "Kansas City Chiefs",
                    "Cincinnati Bengals",
                    "2024-09-13T00:20:00Z",
                    [_spread_book("book_a", spread), _spread_book("book_b", spread)],
                )
            ]
        else:
            events = [
                _event(
                    "sea-ne",
                    "Seattle Seahawks",
                    "New England Patriots",
                    "2023-09-15T17:00:00Z",
                    [_spread_book("book_a", spread), _spread_book("book_b", spread)],
                )
            ]
        _store_snapshot(
            root,
            schedule,
            season=season,
            week=2,
            label=label,
            snapshot_time=snapshot_time,
            events=events,
        )
    return root


def test_season_scorecard_marks_clv_unavailable_without_capture_root(
    model_frame: pd.DataFrame,
) -> None:
    """No archive supplied => explicit marker column, never a silent NaN."""

    predictions = model_frame.copy()
    predictions["home_cover_probability"] = 0.55
    scorecard = season_scorecard(predictions)
    assert scorecard["clv_status"].eq(CLV_STATUS_CAPTURE_UNAVAILABLE).all()
    assert scorecard["clv_points"].isna().all()
    assert scorecard["clv_games"].eq(0).all()
    # Same marker when a root is supplied but the directory does not exist.
    missing_root = season_scorecard(predictions, market_capture_root=Path("Z:/no/such/dir"))
    assert missing_root["clv_status"].eq(CLV_STATUS_CAPTURE_UNAVAILABLE).all()


def test_season_scorecard_measures_clv_from_fixture_archive(tmp_path: Path) -> None:
    """Hand-computed CLV through the real build_pairing_table/close_reference_table/score_clv path.

    KC-CIN: HOME pick at decision spread +1.5, store close +4.0 =>
    +1 * (4.0 - 1.5) = +2.5. SEA-NE: AWAY pick at decision spread +2.5,
    store close +1.0 => -1 * (1.0 - 2.5) = +1.5.
    """

    scorecard = season_scorecard(
        _clv_predictions(), market_capture_root=_clv_store(tmp_path / "raw")
    )
    by_season = scorecard.set_index("season")
    assert by_season.loc[2024, "clv_status"] == CLV_STATUS_MEASURED
    assert by_season.loc[2024, "clv_points"] == pytest.approx(2.5)
    assert by_season.loc[2024, "clv_games"] == 1
    assert by_season.loc[2023, "clv_status"] == CLV_STATUS_MEASURED
    assert by_season.loc[2023, "clv_points"] == pytest.approx(1.5)
    assert by_season.loc[2023, "clv_games"] == 1


def test_season_scorecard_flags_seasons_the_archive_does_not_pair(tmp_path: Path) -> None:
    """Partial archive coverage stays visible per season, not collapsed to NaN."""

    import json
    import shutil

    full_store = _clv_store(tmp_path / "full" / "raw")
    partial_store = tmp_path / "partial" / "raw"
    shutil.copytree(full_store, partial_store)
    for entry in sorted(partial_store.iterdir()):
        manifest_path = entry / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("request") or {}).get("season") == 2024:
            shutil.rmtree(entry)
    scorecard = season_scorecard(_clv_predictions(), market_capture_root=partial_store)
    by_season = scorecard.set_index("season")
    assert by_season.loc[2023, "clv_status"] == CLV_STATUS_MEASURED
    assert by_season.loc[2023, "clv_points"] == pytest.approx(1.5)
    assert by_season.loc[2024, "clv_status"] == CLV_STATUS_NO_PAIRED_GAMES
    assert pd.isna(by_season.loc[2024, "clv_points"])
    assert by_season.loc[2024, "clv_games"] == 0
