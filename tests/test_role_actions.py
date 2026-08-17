from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.role_actions import (
    ROLE_ACTIONS_REQUIRED_COLUMNS,
    canonicalize_role_actions,
    latest_role_actions_snapshot,
    load_role_actions_snapshot,
    role_actions_snapshot_from_root,
    write_role_actions_snapshot,
)


def _raw_frame(sacks_column: str = "sacks_suffered") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["QB-A", "RB-A", "WR-A", "QB-B"],
            "season": [2022, 2022, 2022, 2022],
            "week": [1, 1, 1, 1],
            "season_type": ["REG", "REG", "REG", "POST"],
            "game_id": ["2022_01_A_B", "2022_01_A_B", "2022_01_A_B", "2022_20_A_B"],
            "team": ["A", "A", "A", "B"],
            "position": ["qb", "rb", "wr", "qb"],
            "attempts": [30, 0, 0, 25],
            "carries": [2, 18, 1, 3],
            "receptions": [0, 3, 6, 0],
            "targets": [0, 4, 8, 0],
            sacks_column: [3, 0, 0, 2],
        }
    )


def test_canonicalize_keeps_and_renames_sacks_suffered() -> None:
    frame = _raw_frame("sacks_suffered")
    result = canonicalize_role_actions(frame)
    assert list(result.columns) == list(ROLE_ACTIONS_REQUIRED_COLUMNS)
    # POST row dropped by the REG filter.
    assert set(result["season_type"]) == {"REG"}
    qb_row = result.loc[result["player_id"].eq("QB-A")].iloc[0]
    assert qb_row["sacks_taken"] == 3


def test_canonicalize_keeps_and_renames_older_sacks_column() -> None:
    frame = _raw_frame("sacks")
    result = canonicalize_role_actions(frame)
    assert "sacks_taken" in result.columns
    assert "sacks" not in result.columns
    qb_row = result.loc[result["player_id"].eq("QB-A")].iloc[0]
    assert qb_row["sacks_taken"] == 3


def test_canonicalize_prefers_sacks_suffered_when_both_present() -> None:
    frame = _raw_frame("sacks_suffered")
    frame["sacks"] = [99, 99, 99, 99]
    result = canonicalize_role_actions(frame)
    qb_row = result.loc[result["player_id"].eq("QB-A")].iloc[0]
    assert qb_row["sacks_taken"] == 3


def test_canonicalize_raises_when_neither_sacks_column_exists() -> None:
    frame = _raw_frame("sacks_suffered").drop(columns=["sacks_suffered"])
    with pytest.raises(DataContractError):
        canonicalize_role_actions(frame)


def test_canonicalize_filters_to_regular_season_only() -> None:
    result = canonicalize_role_actions(_raw_frame())
    assert result["game_id"].tolist() == ["2022_01_A_B"] * 3


def _recorded_scope(manifest_path: Path) -> bool:
    return bool(json.loads(manifest_path.read_text(encoding="utf-8"))["include_postseason"])


def test_canonicalize_can_keep_the_postseason_on_request() -> None:
    frame = _raw_frame()
    expected = canonicalize_role_actions(frame)
    widened = canonicalize_role_actions(frame, include_postseason=True)
    assert set(widened["season_type"]) == {"REG", "POST"}
    assert len(widened) == 4
    pd.testing.assert_frame_equal(
        widened.loc[widened["season_type"].eq("REG")].reset_index(drop=True), expected
    )


def test_canonicalize_rejects_unknown_season_codes_when_widened() -> None:
    frame = _raw_frame()
    frame.loc[3, "season_type"] = "POSTSEASON"
    with pytest.raises(DataContractError, match="unrecognized season codes"):
        canonicalize_role_actions(frame, include_postseason=True)
    # The default path keeps its historical, silent "== REG" comparison.
    assert len(canonicalize_role_actions(frame)) == 3


def test_snapshot_records_scope_and_reads_back_regular_season_only(tmp_path: Path) -> None:
    regular = write_role_actions_snapshot(_raw_frame(), tmp_path / "reg", [2022], snapshot_id="reg")
    widened = write_role_actions_snapshot(
        _raw_frame(), tmp_path / "post", [2022], snapshot_id="post", include_postseason=True
    )
    assert _recorded_scope(regular.manifest_path) is False
    assert _recorded_scope(widened.manifest_path) is True
    pd.testing.assert_frame_equal(
        load_role_actions_snapshot(widened), load_role_actions_snapshot(regular)
    )
    assert len(load_role_actions_snapshot(widened, include_postseason=True)) == 4


def test_snapshot_without_the_scope_key_still_loads(tmp_path: Path) -> None:
    snapshot = write_role_actions_snapshot(_raw_frame(), tmp_path, [2022], snapshot_id="legacy")
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    del manifest["include_postseason"]
    snapshot.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reloaded = role_actions_snapshot_from_root(snapshot.root)
    assert reloaded == snapshot
    assert len(load_role_actions_snapshot(reloaded)) == 3


def test_canonicalize_raises_on_duplicate_game_team_player_rows() -> None:
    frame = _raw_frame()
    # Same (game_id, team, player_id) key but a differing count so it is not
    # collapsed by the initial exact-row de-duplication (mirrors
    # canonicalize_player_stats in nfl_ats.players).
    duplicate = frame.iloc[[0]].copy()
    duplicate["attempts"] = duplicate["attempts"] + 1
    frame = pd.concat([frame, duplicate], ignore_index=True)
    with pytest.raises(DataContractError):
        canonicalize_role_actions(frame)


def test_canonicalize_coerces_numeric_counts_and_fills_missing() -> None:
    frame = _raw_frame()
    frame["attempts"] = ["30", np.nan, "0", "25"]
    result = canonicalize_role_actions(frame)
    assert result["attempts"].dtype.kind == "f"
    assert result.loc[result["player_id"].eq("RB-A"), "attempts"].iloc[0] == 0.0


def test_canonicalize_requires_columns() -> None:
    frame = _raw_frame().drop(columns=["carries"])
    with pytest.raises(DataContractError):
        canonicalize_role_actions(frame)


def test_write_and_round_trip_snapshot(tmp_path: Path) -> None:
    root = tmp_path
    snapshot = write_role_actions_snapshot(
        _raw_frame(), root, [2022], snapshot_id="20220101T000000Z"
    )
    reloaded = role_actions_snapshot_from_root(snapshot.root)
    assert reloaded.snapshot_id == "20220101T000000Z"
    assert reloaded.seasons == (2022,)

    latest = latest_role_actions_snapshot(root)
    assert latest.snapshot_id == snapshot.snapshot_id

    loaded = load_role_actions_snapshot(latest)
    assert list(loaded.columns) == list(ROLE_ACTIONS_REQUIRED_COLUMNS)
    assert len(loaded) == 3


def test_write_snapshot_refuses_to_overwrite(tmp_path: Path) -> None:
    root = tmp_path
    write_role_actions_snapshot(_raw_frame(), root, [2022], snapshot_id="dup")
    with pytest.raises(FileExistsError):
        write_role_actions_snapshot(_raw_frame(), root, [2022], snapshot_id="dup")
