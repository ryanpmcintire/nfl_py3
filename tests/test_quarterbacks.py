from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS
from nfl_ats.quarterbacks import (
    canonicalize_depth_charts,
    depth_snapshot_from_root,
    enrich_with_qb_features,
    latest_depth_snapshot,
    latest_starting_qbs,
    load_depth_snapshot,
    write_depth_snapshot,
)


def _depth_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dt": [
                "2022-09-09T12:00:00Z",
                "2022-09-09T12:00:00Z",
                "2022-09-09T12:00:00Z",
                "2022-09-16T12:00:00Z",
                "2022-09-16T12:00:00Z",
                None,
            ],
            "team": ["A", "A", "B", "A", "B", "A"],
            "player_name": ["QB A", "Backup A", "QB B", "QB A", "QB B", "Old A"],
            "gsis_id": ["QB-A", "QB-A2", "QB-B", "QB-A", "QB-B", "OLD-A"],
            "pos_abb": ["QB", "QB", "QB", "QB", "QB", "QB"],
            "pos_rank": [1, 2, 1, 1, 1, 1],
            "espn_id": [1, 2, 3, 1, 3, 4],
            "pos_slot": 9,
        }
    )


def _qb_pbp() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 4):
        game_id = f"2022_{week:02d}_B_A"
        for team, opponent, player, direction in (
            ("A", "B", "QB-A", 1.0),
            ("B", "A", "QB-B", -1.0),
        ):
            for play_number in range(1, 7):
                row = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
                row.update(
                    {
                        "play_id": play_number + (100 if team == "B" else 0),
                        "game_id": game_id,
                        "season": 2022,
                        "season_type": "REG",
                        "week": week,
                        "home_team": "A",
                        "away_team": "B",
                        "posteam": team,
                        "defteam": opponent,
                        "fixed_drive": 1 if team == "A" else 2,
                        "down": 1,
                        "play_type": "pass",
                        "yards_gained": 8,
                        "pass_attempt": 1,
                        "rush_attempt": 0,
                        "qb_dropback": 1,
                        "qb_kneel": 0,
                        "qb_spike": 0,
                        "aborted_play": 0,
                        "sack": 0,
                        "qb_hit": 0,
                        "interception": int(play_number == 6 and team == "B"),
                        "epa": direction * week + play_number / 100,
                        "success": int(direction > 0),
                        "wp": 0.5,
                        "passer_player_id": player,
                        "passer_player_name": f"QB {team}",
                        "cpoe": direction * 2,
                        "pass_oe": 0.1,
                        "yardline_100": 60,
                        "play": 1,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"2022_{week:02d}_B_A" for week in range(1, 4)],
            "season": 2022,
            "week": range(1, 4),
            "gameday": pd.date_range("2022-09-11", periods=3, freq="7D"),
            "kickoff": pd.date_range("2022-09-11 17:00Z", periods=3, freq="7D"),
            "away_team": "B",
            "home_team": "A",
        }
    )


def test_depth_snapshot_keeps_only_timestamped_qbs(tmp_path: Path) -> None:
    snapshot = write_depth_snapshot(_depth_history(), tmp_path, [2022], "fixed")
    stored = pd.read_parquet(snapshot.data_path)
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert len(stored) == 5
    assert manifest["rows"] == 5
    assert manifest["sha256"]


def test_asof_depth_uses_latest_rank_one_and_rejects_stale() -> None:
    depth = canonicalize_depth_charts(_depth_history())
    starters = latest_starting_qbs(depth, pd.Timestamp("2022-09-17T00:00Z"))
    assert starters.set_index("team").loc["A", "gsis_id"] == "QB-A"
    assert starters.set_index("team").loc["B", "gsis_id"] == "QB-B"
    stale = latest_starting_qbs(depth, pd.Timestamp("2022-10-20T00:00Z"), max_age_days=14)
    assert stale.empty


def test_current_qb_game_cannot_change_current_pregame_state() -> None:
    games = _games()
    depth = canonicalize_depth_charts(_depth_history())
    baseline = enrich_with_qb_features(
        games,
        _qb_pbp(),
        depth,
        decision_hours_before_kickoff=24,
        max_depth_age_days=14,
        min_dropbacks=1,
    )
    changed_pbp = _qb_pbp()
    mask = changed_pbp["game_id"].eq("2022_02_B_A") & changed_pbp["posteam"].eq("A")
    changed_pbp.loc[mask, "epa"] = 1_000.0
    changed = enrich_with_qb_features(
        games,
        changed_pbp,
        depth,
        decision_hours_before_kickoff=24,
        max_depth_age_days=14,
        min_dropbacks=1,
    )
    column = "home_qb_epa_per_dropback"
    assert changed.loc[1, column] == pytest.approx(baseline.loc[1, column])
    assert changed.loc[2, column] != pytest.approx(baseline.loc[2, column])


def test_depth_rows_after_decision_are_never_visible() -> None:
    depth = canonicalize_depth_charts(_depth_history())
    before_second_update = latest_starting_qbs(depth, pd.Timestamp("2022-09-15T00:00Z"))
    assert before_second_update["observed_at_utc"].max() == pd.Timestamp("2022-09-09T12:00:00Z")


def test_depth_snapshot_discovery_and_guards(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No depth-chart"):
        latest_depth_snapshot(tmp_path)
    snapshot = write_depth_snapshot(_depth_history(), tmp_path, [2022], "fixed")
    assert latest_depth_snapshot(tmp_path) == snapshot
    assert depth_snapshot_from_root(snapshot.root) == snapshot
    assert len(load_depth_snapshot(snapshot)) == 5
    with pytest.raises(FileNotFoundError, match="manifest"):
        depth_snapshot_from_root(tmp_path / "missing")
    with pytest.raises(ValueError, match="max_age_days"):
        latest_starting_qbs(
            canonicalize_depth_charts(_depth_history()),
            pd.Timestamp("2022-09-17T00:00Z"),
            max_age_days=0,
        )
