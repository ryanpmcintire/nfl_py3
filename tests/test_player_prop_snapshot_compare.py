"""Tests for the point-in-time player-prop snapshot comparison tool."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
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


def _event_payload() -> dict[str, object]:
    return {
        "id": "event-1",
        "home_team": "Buffalo Bills",
        "away_team": "Miami Dolphins",
        "commence_time": "2024-09-08T17:00:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2024-09-03T11:54:00Z",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "last_update": "2024-09-03T11:54:00Z",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Josh Allen",
                                "point": 250.5,
                                "price": -110,
                            },
                            {
                                "name": "Under",
                                "description": "Josh Allen",
                                "point": 250.5,
                                "price": -110,
                            },
                        ],
                    }
                ],
            }
        ],
    }


def _raw_evidence() -> dict[str, str]:
    return {
        "response_sha256": "a" * 64,
        "response_path": "raw/event_odds/event-1.json",
        "captured_at_utc": "2026-09-02T20:00:00Z",
    }


def test_normalized_rows_have_deterministic_explicit_quote_identities() -> None:
    kwargs = {
        "season": 2024,
        "week": 1,
        "nflverse_game_id": "2024_01_MIA_BUF",
        "snapshot_requested": datetime(2024, 9, 3, 12, tzinfo=UTC),
        "snapshot_actual": datetime(2024, 9, 3, 11, 55, tzinfo=UTC),
        "expected_event_id": "event-1",
        "raw_evidence": _raw_evidence(),
    }

    first = ingest.normalize_event_odds(_event_payload(), **kwargs)
    second = ingest.normalize_event_odds(_event_payload(), **kwargs)

    assert first["event_id"].eq("event-1").all()
    assert first["bookmaker_key"].eq("draftkings").all()
    assert first["market"].eq("player_pass_yds").all()
    assert first["quote_identity"].is_unique
    assert first["quote_identity"].tolist() == second["quote_identity"].tolist()


def test_normalizer_fails_closed_on_event_identity_mismatch() -> None:
    with pytest.raises(ingest.ApiAbort, match="event identity mismatch"):
        ingest.normalize_event_odds(
            _event_payload(),
            season=2024,
            week=1,
            nflverse_game_id="2024_01_MIA_BUF",
            snapshot_requested=datetime(2024, 9, 3, 12, tzinfo=UTC),
            snapshot_actual=datetime(2024, 9, 3, 11, 55, tzinfo=UTC),
            expected_event_id="wrong-event",
            raw_evidence=_raw_evidence(),
        )


@pytest.mark.parametrize(
    "quota",
    [
        {},
        {"requests_remaining": "1000"},
        {"requests_remaining": "unknown", "requests_last": "10"},
        {"requests_remaining": "1000", "requests_last": "-1"},
    ],
)
def test_quota_headers_fail_closed_when_missing_or_malformed(quota: dict[str, str]) -> None:
    with pytest.raises(ingest.ApiAbort, match=r"quota headers|non-negative"):
        ingest.Ledger().record_call("event_odds", quota)


def test_archive_policy_rejects_tracked_public_destination_and_low_floor() -> None:
    root = Path(__file__).parents[1]
    with pytest.raises(ingest.SourcePolicyError, match="gitignored private data root"):
        ingest._validate_archive_request(root / "docs" / "raw-props", 1200)
    with pytest.raises(ingest.SourcePolicyError, match="below the policy minimum"):
        ingest._validate_archive_request(root / "data" / "raw" / "props", 599)


def test_existing_snapshot_is_rejected_before_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "_validate_archive_request", lambda *_args: None)
    destination = tmp_path / "snapshot"
    config = {"quota_floor": 1200, "schema": ingest.ARCHIVE_SCHEMA}
    ingest._prepare_archive(destination, config)

    with pytest.raises(ingest.ApiAbort, match="already exists"):
        ingest._prepare_archive(destination, config)


def test_raw_and_parquet_members_are_hash_coherent_and_write_once(tmp_path: Path) -> None:
    payload = b'{"timestamp":"2024-09-03T11:55:00Z","data":[]}'
    record = ingest._archive_raw_response(
        tmp_path,
        kind="events",
        identity="2024_wk01",
        payload=payload,
        requested_at=datetime(2024, 9, 3, 12, tzinfo=UTC),
        snapshot_actual_at=datetime(2024, 9, 3, 11, 55, tzinfo=UTC),
        quota={"requests_remaining": "1000", "requests_last": "1"},
    )
    raw_path = tmp_path / record["response_path"]
    metadata_path = tmp_path / record["metadata_path"]
    assert sha256(raw_path.read_bytes()).hexdigest() == record["response_sha256"]
    assert sha256(metadata_path.read_bytes()).hexdigest() == record["metadata_sha256"]
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["captured_at_utc"]

    parquet_path = tmp_path / "weekly" / "week.parquet"
    frame = pd.DataFrame({"event_id": ["event-1"], "market": ["player_pass_yds"]})
    digest = ingest._atomic_parquet_once(parquet_path, frame)
    assert sha256(parquet_path.read_bytes()).hexdigest() == digest
    with pytest.raises(ingest.ApiAbort, match="already exists"):
        ingest._atomic_parquet_once(parquet_path, frame)


def test_missing_credential_fails_before_capture() -> None:
    with pytest.raises(ingest.ApiAbort, match="not configured"):
        ingest._require_api_key(None)
