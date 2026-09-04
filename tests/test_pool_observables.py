"""LEAD-52 capture path: manual pool observables, fail-closed validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nfl_ats.pool_observables import (
    DistributionObservation,
    FieldObservation,
    PoolObservableError,
    latest_snapshots,
    record_distribution,
    record_field_observation,
)

_OBSERVER = "lock-day reader"


def _field(**overrides) -> FieldObservation:
    values: dict = {
        "season": 2026,
        "week": 1,
        "entries": 100,
        "paid_places": 15,
        "prize_notes": "top 15 paid, winner-take-most",
        "observed_at_utc": "2026-09-08T13:00:00+00:00",
        "observer": _OBSERVER,
    }
    values.update(overrides)
    return FieldObservation(**values)


def _distribution(**overrides) -> DistributionObservation:
    values: dict = {
        "season": 2026,
        "week": 1,
        "game_id": "2026_01_MIA_LV",
        "home_share": 0.38,
        "away_share": 0.62,
        "unlocked_at_utc": "2026-09-13T13:00:00-04:00",
        "observed_at_utc": "2026-09-13T13:05:00-04:00",
        "observer": _OBSERVER,
    }
    values.update(overrides)
    return DistributionObservation(**values)


def test_field_record_writes_manifest_with_hashes(tmp_path: Path) -> None:
    result = record_field_observation(tmp_path, _field())
    directory = Path(result["directory"])
    assert (directory / "observations.json").is_file()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["season"] == 2026 and manifest["week"] == 1
    assert len(manifest["files"][0]["sha256"]) == 64
    payload = json.loads((directory / "observations.json").read_text(encoding="utf-8"))
    assert payload["observation"]["entries"] == 100
    assert payload["recorded_at_utc"]


def test_field_record_rejects_bad_rows() -> None:
    with pytest.raises(PoolObservableError):
        record_field_observation(Path("."), _field(entries=1))
    with pytest.raises(PoolObservableError):
        record_field_observation(Path("."), _field(entries=10, paid_places=11))
    with pytest.raises(PoolObservableError):
        record_field_observation(Path("."), _field(prize_notes="  "))
    with pytest.raises(PoolObservableError):
        record_field_observation(Path("."), _field(observed_at_utc="2026-09-08"))


def test_distribution_record_rejects_pre_unlock_and_bad_shares() -> None:
    with pytest.raises(PoolObservableError, match="precedes unlocked"):
        record_distribution(
            Path("."),
            _distribution(observed_at_utc="2026-09-13T12:00:00-04:00"),
        )
    with pytest.raises(PoolObservableError):
        record_distribution(Path("."), _distribution(home_share=0.9, away_share=0.9))
    with pytest.raises(PoolObservableError, match="canonical game id"):
        record_distribution(Path("."), _distribution(game_id="MIA-LV"))


def test_distribution_rounding_tolerance_accepts_whole_percents(tmp_path: Path) -> None:
    result = record_distribution(tmp_path, _distribution(home_share=0.38, away_share=0.61))
    assert Path(result["directory"]).is_dir()


def test_snapshots_list_is_read_only_and_empty_safe(tmp_path: Path) -> None:
    assert latest_snapshots(tmp_path) == []
    record_field_observation(tmp_path, _field())
    rows = latest_snapshots(tmp_path)
    assert len(rows) == 1 and rows[0]["season"] == 2026


def test_snapshot_directories_are_write_once(tmp_path: Path) -> None:
    fixed = datetime(2026, 9, 8, 13, 0, 0, tzinfo=UTC)
    record_field_observation(tmp_path, _field(), now=fixed)
    with pytest.raises(PoolObservableError, match="already exists"):
        record_field_observation(tmp_path, _field(), now=fixed)
