from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from nfl_ats.snapshots import (
    describe_snapshot,
    latest_snapshot,
    load_snapshot,
    utc_snapshot_id,
    write_snapshot,
)


def test_snapshot_round_trip_and_latest(
    tmp_path,
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    first = write_snapshot(schedules, stats, [2022], tmp_path, "20220101T000000Z")
    second = write_snapshot(schedules, stats, [2022], tmp_path, "20220102T000000Z")

    assert latest_snapshot(tmp_path) == second
    loaded = load_snapshot(first)
    pd.testing.assert_frame_equal(loaded[0], schedules)
    description = describe_snapshot(first)
    assert description["root"] == str(first.root)
    assert description["manifest"]["files"]["schedules.parquet"]["rows"] == 5
    assert len(description["manifest"]["files"]["schedules.parquet"]["sha256"]) == 64

    with pytest.raises(FileExistsError):
        write_snapshot(schedules, stats, [2022], tmp_path, first.snapshot_id)


def test_snapshot_helpers_reject_missing_data(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No complete snapshots"):
        latest_snapshot(tmp_path)


def test_latest_snapshot_ignores_foreign_source_directories(
    tmp_path,
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # data/raw also hosts named source directories (e.g. injury_news) carrying their
    # own manifest.json; sorting alphabetically after the timestamped snapshots must
    # not make them the "latest" snapshot.
    schedules, stats = schedules_and_stats
    real = write_snapshot(schedules, stats, [2022], tmp_path, "20220101T000000Z")

    foreign = tmp_path / "injury_news"
    foreign.mkdir()
    (foreign / "manifest.json").write_text("{}", encoding="utf-8")

    assert latest_snapshot(tmp_path) == real


def test_snapshot_id_normalizes_to_utc() -> None:
    timestamp = datetime(2022, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert utc_snapshot_id(timestamp) == "20220102T030405Z"
