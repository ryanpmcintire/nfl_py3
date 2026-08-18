from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.estimation_variance import BootstrapDegeneracyError, BootstrapDegeneracyWarning
from nfl_ats.reporting import (
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
    assert season_scorecard(pd.DataFrame({"home_cover": []})).empty
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
