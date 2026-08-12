from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from nfl_ats.prospective import (
    FROZEN_PREDICTION_COLUMNS,
    freeze_forecast,
    verify_frozen_forecast,
)
from nfl_ats.provenance import sha256_file


def _predictions(created_at: datetime) -> pd.DataFrame:
    rows = 2
    frame = pd.DataFrame({column: [None] * rows for column in FROZEN_PREDICTION_COLUMNS})
    frame["game_id"] = ["2026_01_A_B", "2026_01_C_D"]
    frame["season"] = 2026
    frame["week"] = 1
    frame["gameday"] = [created_at.date() + timedelta(days=2)] * rows
    frame["kickoff"] = [created_at + timedelta(days=2), created_at + timedelta(days=3)]
    frame["away_team"] = ["A", "C"]
    frame["home_team"] = ["B", "D"]
    frame["spread_line"] = [2.5, -1.5]
    frame["away_spread_odds"] = -110.0
    frame["home_spread_odds"] = -110.0
    frame["home_cover_probability"] = [0.55, 0.48]
    frame["pick"] = ["HOME", "AWAY"]
    frame["bet_side"] = ["HOME", "PASS"]
    break_even = 110 / 210
    frame["edge"] = [0.55 - break_even, 0.52 - break_even]
    frame["bet_odds"] = [-110.0, float("nan")]
    frame["break_even_probability"] = [break_even, float("nan")]
    frame["market_home_no_vig_probability"] = 0.5
    frame["market_hold"] = (2 * break_even) - 1
    frame["train_rows"] = 4000
    frame["train_max_gameday"] = created_at.date() - timedelta(days=1)
    frame["home_cover"] = pd.NA
    return frame


def test_freeze_forecast_writes_immutable_record(tmp_path) -> None:
    created_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
    frozen = freeze_forecast(
        _predictions(created_at),
        {"model_name": "logistic", "feature_set": "market_context"},
        tmp_path,
        created_at=created_at,
    )
    assert frozen.games == 2
    manifest_path = frozen.directory / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["forecast_id"] == frozen.forecast_id
    assert manifest["schema_version"] == 2
    assert manifest["prediction_safety"]["status"] == "PASS_WITH_WARNINGS"
    assert len(manifest["predictions_sha256"]) == 64
    stored = pd.read_parquet(frozen.directory / "predictions.parquet")
    assert stored["forecast_id"].eq(frozen.forecast_id).all()
    assert str(stored["kickoff"].dt.tz) == "UTC"
    assert verify_frozen_forecast(frozen.directory)["games"] == 2

    # Revalidation catches internally corrupt data even if an attacker or bug
    # also recomputes the file digest in the manifest.
    stored.loc[0, "edge"] = 0.99
    prediction_path = frozen.directory / "predictions.parquet"
    stored.to_parquet(prediction_path, index=False)
    manifest["predictions_sha256"] = sha256_file(prediction_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="decision_policy"):
        verify_frozen_forecast(frozen.directory)

    with pytest.raises(ValueError, match="already exists"):
        freeze_forecast(_predictions(created_at), {}, tmp_path, created_at=created_at)


def test_freeze_forecast_rejects_unverifiable_or_retrospective_rows(tmp_path) -> None:
    created_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
    predictions = _predictions(created_at)
    predictions.loc[0, "kickoff"] = pd.NaT
    with pytest.raises(ValueError, match="kickoff is missing"):
        freeze_forecast(predictions, {}, tmp_path, created_at=created_at)

    predictions = _predictions(created_at)
    predictions.loc[0, "kickoff"] = created_at
    with pytest.raises(ValueError, match="at or after kickoff"):
        freeze_forecast(predictions, {}, tmp_path, created_at=created_at)

    predictions = _predictions(created_at)
    predictions.loc[0, "home_cover"] = 1.0
    with pytest.raises(ValueError, match="before outcomes"):
        freeze_forecast(predictions, {}, tmp_path, created_at=created_at)

    predictions = _predictions(created_at)
    predictions.loc[0, "home_spread_odds"] = pd.NA
    with pytest.raises(ValueError, match="missing lines or prices"):
        freeze_forecast(predictions, {}, tmp_path, created_at=created_at)


def test_verify_frozen_forecast_rejects_manifest_corruption(tmp_path) -> None:
    created_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
    frozen = freeze_forecast(
        _predictions(created_at),
        {"model_name": "logistic", "min_edge": 0.02},
        tmp_path,
        created_at=created_at,
    )
    manifest_path = frozen.directory / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    corruptions = (
        ({**original, "predictions_sha256": "0" * 64}, "digest mismatch"),
        ({**original, "games": 999}, "row-count mismatch"),
        ({**original, "forecast_id": "wrong"}, "identity mismatch"),
        ({key: value for key, value in original.items() if key != "prediction_safety"}, "safety"),
    )
    for corrupted, message in corruptions:
        manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            verify_frozen_forecast(frozen.directory)

    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    assert verify_frozen_forecast(frozen.directory)["forecast_id"] == frozen.forecast_id

    empty = tmp_path / "incomplete"
    empty.mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        verify_frozen_forecast(empty)
