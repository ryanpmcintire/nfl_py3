"""Tests for the point-in-time player-prop snapshot comparison tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "compare_player_prop_snapshots.py"
    spec = importlib.util.spec_from_file_location("compare_player_prop_snapshots", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load comparison script from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load_script()


def _load_ingest_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "ingest_player_props.py"
    spec = importlib.util.spec_from_file_location("ingest_player_props", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ingestion script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest_script()


def _rows(snapshot: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": [1, 1, 1, 1],
            "nflverse_game_id": ["g1", "g1", "g1", "g1"],
            "commence_time_utc": ["2024-09-08T17:00:00Z"] * 4,
            "snapshot_actual_at_utc": [snapshot] * 4,
            "bookmaker_key": ["draftkings", "draftkings", "betrivers", "betrivers"],
            "player_name": ["QB One", "QB One", "QB One", "QB One"],
            "line": [240.5, 240.5, 210.5, 270.5],
        }
    )


def test_compare_counts_presence_and_excludes_alt_line_ladder() -> None:
    earlier = _rows("2024-09-03T12:00:00Z")
    later = _rows("2024-09-04T12:00:00Z")
    later.loc[later["bookmaker_key"].eq("draftkings"), "line"] = 244.5
    extra = later.iloc[[0]].copy()
    extra["player_name"] = "QB Two"
    later = pd.concat([later, extra], ignore_index=True)

    result = compare.compare_snapshots(earlier, later)

    week = result["weeks"].iloc[0]
    assert week.to_dict() == {
        "week": 1,
        "earlier_player_games": 1,
        "later_player_games": 2,
        "both": 1,
        "earlier_only": 0,
        "later_only": 1,
    }
    assert result["common_book_player_lines"] == 1
    assert result["common_player_games"] == 1
    assert result["delta_summary"]["median"] == pytest.approx(4.0)


def test_compare_fails_closed_on_post_kickoff_snapshot() -> None:
    earlier = _rows("2024-09-03T12:00:00Z")
    leaked = _rows("2024-09-08T17:00:00Z")

    with pytest.raises(ValueError, match="not pregame"):
        compare.compare_snapshots(earlier, leaked)


def test_compare_reports_week_present_only_in_earlier_snapshot() -> None:
    earlier = _rows("2024-09-03T12:00:00Z")
    later = _rows("2024-09-04T12:00:00Z")
    earlier["week"] = 1
    later["week"] = 2

    result = compare.compare_snapshots(earlier, later)

    assert result["weeks"].to_dict("records") == [
        {
            "week": 1,
            "earlier_player_games": 1,
            "later_player_games": 0,
            "both": 0,
            "earlier_only": 1,
            "later_only": 0,
        },
        {
            "week": 2,
            "earlier_player_games": 0,
            "later_player_games": 1,
            "both": 0,
            "earlier_only": 0,
            "later_only": 1,
        },
    ]


def test_earliest_kickoff_selector_ignores_api_order_and_keeps_ties() -> None:
    matched = [
        ({"id": "late", "commence_time": "2024-09-08T20:00:00Z"}, "g3"),
        ({"id": "early-b", "commence_time": "2024-09-08T17:00:00Z"}, "g2"),
        ({"id": "early-a", "commence_time": "2024-09-08T17:00:00Z"}, "g1"),
    ]

    selected = ingest.select_earliest_kickoff_events(matched)

    assert [game_id for _event, game_id in selected] == ["g2", "g1"]


def test_earliest_kickoff_selector_fails_closed_on_invalid_time() -> None:
    matched = [({"id": "bad", "commence_time": None}, "g1")]

    with pytest.raises(ingest.ApiAbort, match="invalid commence_time"):
        ingest.select_earliest_kickoff_events(matched)
