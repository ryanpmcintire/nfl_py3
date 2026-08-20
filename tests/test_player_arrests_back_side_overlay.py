"""Contracts for the prospective-only player-arrest back-side overlay."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.data import DataContractError
from nfl_ats.player_arrests_back_side_overlay import (
    CHALLENGER_ID,
    MAX_SNAPSHOT_AGE,
    WINDOW_DAYS,
    apply_player_arrests_back_side_overlay,
    load_latest_complete_arrest_snapshot,
    record_player_arrests_back_side_challenger_decisions,
)
from nfl_ats.prospective_scoring import load_challenger_decisions

_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "data/processed/game_features_weak_stack.parquet",
}


def _safe_incidents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "incident_date": [
                "2024-09-03",  # JAX: 14 days before Tuesday, included.
                "2024-09-02",  # BUF: 15 days before Tuesday, excluded.
                "2024-09-17",  # IND: same Tuesday, excluded.
                "2024-09-18",  # CHI: after Tuesday, excluded.
                "2024-09-10",  # IND: seven days before Tuesday, included.
                "2024-09-12",  # PIT: five days before Tuesday, included.
            ],
            "team": ["JAC", "BUF", "IN", "CHI", "IN", "PIT"],
            # Retrospective source fields are intentionally present. The
            # transformer must neither require nor inspect them.
            "outcome_archive_only": ["x"] * 6,
            "description_archive_only": ["private retrospective text"] * 6,
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["sole_flip", "sole_hold", "both", "neither"],
            "season": [2024] * 4,
            "week": [3] * 4,
            "gameday": ["2024-09-22"] * 4,
            "kickoff": ["2024-09-22T17:00:00+00:00"] * 4,
            "home_team": ["BUF", "IND", "PIT", "CHI"],
            "away_team": ["JAX", "BUF", "IND", "BUF"],
            "spread_line": [3.0, -2.5, 1.5, 2.0],
            # sole_flip opposes affected away JAX; sole_hold already backs
            # affected home IND; both and neither are frozen no-ops.
            "home_cover_probability": [0.60, 0.60, 0.55, 0.45],
        }
    )


def _write_snapshot(
    data_root: Path,
    snapshot_id: str,
    *,
    fetched_at_utc: str,
    complete: bool = True,
    incidents: pd.DataFrame | None = None,
) -> Path:
    directory = data_root / "raw" / "player_arrests" / snapshot_id
    directory.mkdir(parents=True)
    safe_path = directory / "incidents_point_in_time.parquet"
    (incidents if incidents is not None else _safe_incidents()).to_parquet(safe_path, index=False)
    digest = hashlib.sha256(safe_path.read_bytes()).hexdigest()
    manifest = {
        "snapshot_id": snapshot_id,
        "fetched_at_utc": fetched_at_utc,
        "complete": complete,
        "rows_cached": len(incidents if incidents is not None else _safe_incidents()),
        "point_in_time_policy": {"safe_index": safe_path.name},
        "files": {safe_path.name: digest},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_frozen_policy_uses_the_predeclared_fourteen_day_window() -> None:
    assert WINDOW_DAYS == 14
    assert pd.Timedelta(hours=36) == MAX_SNAPSHOT_AGE


def test_overlay_applies_exact_sole_flagged_back_side_rule() -> None:
    result = apply_player_arrests_back_side_overlay(_predictions(), _safe_incidents())
    overlaid = result.overlaid_predictions.set_index("game_id")

    assert result.home_flags.tolist() == [False, True, True, False]
    assert result.away_flags.tolist() == [True, False, True, False]
    assert [flip.game_id for flip in result.flips] == ["sole_flip"]
    assert overlaid.loc["sole_flip", "home_cover_probability"] == pytest.approx(0.40)
    assert overlaid.loc["sole_hold", "home_cover_probability"] == pytest.approx(0.60)
    assert overlaid.loc["both", "home_cover_probability"] == pytest.approx(0.55)
    assert overlaid.loc["neither", "home_cover_probability"] == pytest.approx(0.45)


def test_overlay_ignores_retrospective_fields_and_preserves_other_columns() -> None:
    predictions = _predictions()
    incidents = _safe_incidents()
    first = apply_player_arrests_back_side_overlay(predictions, incidents)
    incidents["outcome_archive_only"] = "mutated outcome"
    incidents["description_archive_only"] = "mutated description"
    second = apply_player_arrests_back_side_overlay(predictions, incidents)

    pd.testing.assert_frame_equal(first.overlaid_predictions, second.overlaid_predictions)
    other = [column for column in predictions if column != "home_cover_probability"]
    pd.testing.assert_frame_equal(first.overlaid_predictions[other], predictions[other])


def test_latest_complete_snapshot_requires_hash_verified_fresh_safe_index(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_snapshot(
        data_root,
        "20261004T120000Z",
        fetched_at_utc="2026-10-04T12:00:00+00:00",
    )
    snapshot = load_latest_complete_arrest_snapshot(
        data_root, now=datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    )

    assert snapshot.snapshot_id == "20261004T120000Z"
    assert snapshot.age_hours == pytest.approx(24.0)
    assert snapshot.rows_cached == 6


@pytest.mark.parametrize(
    ("fetched_at", "message"),
    [
        ("2026-10-03T23:59:59+00:00", "stale"),
        ("2026-10-05T12:00:01+00:00", "future-dated"),
    ],
)
def test_snapshot_freshness_refuses_stale_or_future_sources(
    tmp_path: Path, fetched_at: str, message: str
) -> None:
    data_root = tmp_path / "data"
    _write_snapshot(data_root, "20261005T000000Z", fetched_at_utc=fetched_at)

    with pytest.raises(DataContractError, match=message):
        load_latest_complete_arrest_snapshot(
            data_root, now=datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
        )


def test_newest_incomplete_snapshot_refuses_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_snapshot(
        data_root,
        "20261005T100000Z",
        fetched_at_utc="2026-10-05T10:00:00+00:00",
        complete=True,
    )
    _write_snapshot(
        data_root,
        "20261005T110000Z",
        fetched_at_utc="2026-10-05T11:00:00+00:00",
        complete=False,
    )

    with pytest.raises(DataContractError, match=r"[Nn]ewest.*incomplete"):
        load_latest_complete_arrest_snapshot(
            data_root, now=datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
        )


def test_snapshot_hash_mismatch_refuses(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    directory = _write_snapshot(
        data_root,
        "20261005T100000Z",
        fetched_at_utc="2026-10-05T10:00:00+00:00",
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["incidents_point_in_time.parquet"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataContractError, match="hash mismatch"):
        load_latest_complete_arrest_snapshot(
            data_root, now=datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
        )


def _write_registry(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    model = dict(_MODEL_CONFIG)
    model["ridge_alpha"] = ridge_alpha
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {"challenger_id": CHALLENGER_ID, "status": "ACTIVE_PROSPECTIVE", "model": model}
        ],
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_active_model_and_card(artifacts: Path) -> None:
    forecast = artifacts / "margin_predictions" / "2026-week-05-forecast"
    forecast.mkdir(parents=True)
    card = pd.DataFrame(
        {
            "game_id": ["2026_05_JAX_BUF", "2026_05_IND_CHI"],
            "season": [2026, 2026],
            "week": [5, 5],
            "gameday": ["2026-10-11", "2026-10-11"],
            "kickoff": ["2026-10-11T17:00:00+00:00"] * 2,
            "away_team": ["JAX", "IND"],
            "home_team": ["BUF", "CHI"],
            "spread_line": [3.0, 2.0],
            "home_cover_probability": [0.60, 0.55],
        }
    )
    card.to_csv(forecast / "recommendations.csv", index=False)
    metadata = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "created_at_utc": "2026-10-01T15:00:00+00:00",
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {
            "feature_table": {
                "path": "data/processed/game_features_weak_stack.parquet",
                "sha256": "abc123",
            }
        },
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    active = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "model_id": "model-xyz",
        "method": "market_residual",
        "feature_profile": "weak_stack",
        "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
        "weekly_forecast": {
            "artifact": "margin_predictions/2026-week-05-forecast",
            "season": 2026,
            "week": 5,
        },
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")


def test_recorder_writes_overlay_arm_once_with_snapshot_provenance(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    incidents = pd.DataFrame(
        {
            "record_id": ["fresh"],
            "incident_date": ["2026-09-29"],  # Seven days before Tuesday 2026-10-06.
            "team": ["JAX"],
        }
    )
    _write_snapshot(
        data_root,
        "20261005T120000Z",
        fetched_at_utc="2026-10-05T12:00:00+00:00",
        incidents=incidents,
    )
    now = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)

    first = record_player_arrests_back_side_challenger_decisions(artifacts, data_root, now=now)
    second = record_player_arrests_back_side_challenger_decisions(artifacts, data_root, now=now)
    ledger = load_challenger_decisions(artifacts)

    assert first["recorded"] == 2
    assert first["flip_count"] == 1
    assert first["flipped_game_ids"] == ["2026_05_JAX_BUF"]
    assert first["arrest_snapshot_id"] == "20261005T120000Z"
    assert second["recorded"] == 0
    assert second["already_recorded"] == 2
    assert len(ledger) == 2
    picks = ledger.set_index("game_id")["pick_side"].to_dict()
    assert picks == {"2026_05_JAX_BUF": "AWAY", "2026_05_IND_CHI": "HOME"}
    assert (
        pd.to_datetime(ledger["recorded_at_utc"], utc=True)
        < pd.to_datetime(ledger["kickoff"], utc=True)
    ).all()


def test_recorder_refuses_stale_source_before_creating_ledger(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    _write_snapshot(
        data_root,
        "20261003T000000Z",
        fetched_at_utc="2026-10-03T00:00:00+00:00",
    )

    with pytest.raises(DataContractError, match="stale"):
        record_player_arrests_back_side_challenger_decisions(
            artifacts,
            data_root,
            now=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
        )
    assert not (artifacts / "prospective" / "challenger_decisions.parquet").exists()


def test_recorder_refuses_active_config_drift(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts, ridge_alpha=99.0)
    _write_active_model_and_card(artifacts)
    _write_snapshot(
        data_root,
        "20261005T120000Z",
        fetched_at_utc="2026-10-05T12:00:00+00:00",
    )

    with pytest.raises(DataContractError, match="fingerprint"):
        record_player_arrests_back_side_challenger_decisions(
            artifacts,
            data_root,
            now=datetime(2026, 10, 5, 13, 0, tzinfo=UTC),
        )


def test_recorder_refuses_before_the_weekly_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    _write_snapshot(
        data_root,
        "20261001T120000Z",
        fetched_at_utc="2026-10-01T12:00:00+00:00",
    )

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_player_arrests_back_side_challenger_decisions(
            artifacts,
            data_root,
            now=datetime(2026, 10, 1, 13, 0, tzinfo=UTC),
        )
