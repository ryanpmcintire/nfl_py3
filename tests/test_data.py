from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from nfl_ats.data import (
    DataContractError,
    fetch_nflverse,
    validate_schedules,
    validate_team_stats,
)
from nfl_ats.snapshots import describe_snapshot, load_snapshot


def test_data_contracts_accept_valid_frames(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    validate_schedules(schedules)
    validate_team_stats(stats)


def test_schedule_contract_rejects_missing_and_duplicate_keys(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, _ = schedules_and_stats
    with pytest.raises(DataContractError, match="missing required columns"):
        validate_schedules(schedules.drop(columns="result"))
    duplicated = pd.concat([schedules, schedules.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate game_id"):
        validate_schedules(duplicated)


def test_team_stats_contract_rejects_duplicate_keys(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, stats = schedules_and_stats
    duplicated = pd.concat([stats, stats.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate game_id/team"):
        validate_team_stats(duplicated)


def test_fetch_uses_nflreadpy_and_writes_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    requested: dict[str, list[int]] = {}

    def load_schedules(seasons: list[int]) -> pd.DataFrame:
        requested["schedules"] = seasons
        return schedules

    def load_team_stats(seasons: list[int], summary_level: str) -> pd.DataFrame:
        assert summary_level == "week"
        requested["team_stats"] = seasons
        return stats

    fake = SimpleNamespace(load_schedules=load_schedules, load_team_stats=load_team_stats)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    snapshot = fetch_nflverse([2022, 2023], tmp_path, team_stat_seasons=[2022])
    loaded_schedules, loaded_stats = load_snapshot(snapshot)
    pd.testing.assert_frame_equal(loaded_schedules, schedules)
    pd.testing.assert_frame_equal(loaded_stats, stats)
    assert requested == {"schedules": [2022, 2023], "team_stats": [2022]}
    manifest = describe_snapshot(snapshot)["manifest"]
    assert manifest["schedule_seasons"] == [2022, 2023]
    assert manifest["team_stat_seasons"] == [2022]


def test_fetch_rejects_bad_season_lists(tmp_path) -> None:
    with pytest.raises(ValueError, match="At least one"):
        fetch_nflverse([], tmp_path)
    with pytest.raises(ValueError, match="unique and sorted"):
        fetch_nflverse([2022, 2021], tmp_path)
    with pytest.raises(ValueError, match="included in schedule"):
        fetch_nflverse([2022], tmp_path, team_stat_seasons=[2021])
