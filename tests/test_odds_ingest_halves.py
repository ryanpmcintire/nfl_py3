"""Tests for the LEAD-61 per-event half/quarter-game market capture.

No network calls anywhere in this file: ``capture_half_markets`` accepts a
``fetch`` callable exactly like ``nfl_ats.odds_backfill.execute_backfill``
does, and every test here supplies a stub.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats.market_data_halves import (
    HALF_MARKETS_DEFAULT,
    QuotaFloorRefusal,
    assemble_events_payload,
    capture_half_markets,
    current_week_kickoff_window,
    filter_events_to_week,
    newest_bulk_snapshot,
    plan_half_market_capture,
)
from nfl_ats.nfl_week import week_cycle_sunday


def _iso_z(instant: datetime) -> str:
    return instant.astimezone(UTC).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# current_week_kickoff_window / filter_events_to_week
# ---------------------------------------------------------------------------


def test_current_week_kickoff_window_spans_seven_days_from_tuesday() -> None:
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 9, 15, 0, tzinfo=UTC)  # arbitrary instant
    start, end = current_week_kickoff_window(now)

    cycle_sunday = week_cycle_sunday(now.astimezone(eastern).date())
    tuesday = cycle_sunday - timedelta(days=5)
    expected_start = datetime(tuesday.year, tuesday.month, tuesday.day, tzinfo=eastern).astimezone(
        UTC
    )
    assert start == expected_start
    assert end - start == timedelta(days=7)


def test_current_week_kickoff_window_requires_aware_datetime() -> None:
    with pytest.raises(ValueError, match="explicit timezone"):
        current_week_kickoff_window(datetime(2026, 9, 9, 15, 0))


def test_filter_events_to_week_keeps_only_the_current_cycle() -> None:
    now = datetime(2026, 9, 9, 15, 0, tzinfo=UTC)
    start, end = current_week_kickoff_window(now)

    events = [
        {"id": "in-early", "commence_time": _iso_z(start + timedelta(hours=1))},
        {"id": "in-late", "commence_time": _iso_z(end - timedelta(seconds=1))},
        {"id": "out-before", "commence_time": _iso_z(start - timedelta(hours=1))},
        {"id": "out-after", "commence_time": _iso_z(end)},
        {"id": "no-commence-time"},
        {"id": "bad-commence-time", "commence_time": "not-a-timestamp"},
        "not-a-dict",
    ]

    kept = filter_events_to_week(events, now)  # type: ignore[arg-type]
    assert {event["id"] for event in kept} == {"in-early", "in-late"}


def test_filter_events_to_week_never_selects_the_whole_272_event_board() -> None:
    """The regression this exists to prevent: iterating the whole remaining
    season instead of one week multiplies the measured per-event cost ~17x
    for nothing (docs/half_game_markets.md)."""

    now = datetime(2026, 9, 9, 15, 0, tzinfo=UTC)
    start, _end = current_week_kickoff_window(now)
    # 272 events spread one per day across the remaining season.
    events = [
        {"id": f"evt-{i}", "commence_time": _iso_z(start + timedelta(days=i))} for i in range(272)
    ]
    kept = filter_events_to_week(events, now)
    assert 0 < len(kept) < 20


# ---------------------------------------------------------------------------
# newest_bulk_snapshot
# ---------------------------------------------------------------------------


def _write_bulk_dir(
    root: Path, stamp: str, events: list[dict[str, Any]], *, remaining: str = "1000"
) -> None:
    directory = root / stamp
    directory.mkdir(parents=True)
    (directory / "response.json").write_text(json.dumps(events), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "snapshot_id": stamp,
        "quota": {"requests_remaining": remaining, "requests_used": "20", "requests_last": "3"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_newest_bulk_snapshot_ignores_halves_and_probe_directories(tmp_path: Path) -> None:
    root = tmp_path / "market" / "raw"
    _write_bulk_dir(root, "20260901T090000Z", [], remaining="500")
    _write_bulk_dir(root, "20260908T090000Z", [{"id": "evt-1"}], remaining="999")
    (root / "20260908T090000Z-halves").mkdir(parents=True)
    (root / "20260908T090000Z-halves" / "response.json").write_text("[]", encoding="utf-8")
    (root / "20260908T090000Z-halves" / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "20260908T100000Z-event-halves-probe").mkdir(parents=True)
    (root / "not-a-stamp").mkdir(parents=True)

    ref = newest_bulk_snapshot(root)
    assert ref is not None
    assert ref.snapshot_id == "20260908T090000Z"
    assert ref.manifest["quota"]["requests_remaining"] == "999"


def test_newest_bulk_snapshot_returns_none_when_absent(tmp_path: Path) -> None:
    assert newest_bulk_snapshot(tmp_path / "market" / "raw") is None


# ---------------------------------------------------------------------------
# plan_half_market_capture / quota refusal
# ---------------------------------------------------------------------------


def test_plan_never_refuses_with_no_known_remaining() -> None:
    plan = plan_half_market_capture(
        [f"evt-{i}" for i in range(16)], known_remaining=None, quota_floor=600
    )
    assert plan.refused is False
    assert plan.credits_per_event == 4  # 4 half markets x 1 region
    assert plan.planned_credits == 64


def test_plan_refuses_when_the_floor_would_be_breached() -> None:
    plan = plan_half_market_capture(["evt-1"], known_remaining=603, quota_floor=600)
    assert plan.credits_per_event == 4
    assert plan.planned_credits == 4
    assert plan.refused is True
    assert plan.refusal_reason is not None
    assert "600" in plan.refusal_reason


def test_plan_allows_when_comfortably_above_the_floor() -> None:
    plan = plan_half_market_capture(["evt-1"], known_remaining=1000, quota_floor=600)
    assert plan.refused is False


def test_plan_multiplies_credits_by_region_count() -> None:
    plan = plan_half_market_capture(["evt-1"], regions="us,us2", known_remaining=None)
    assert plan.credits_per_event == 8  # 4 markets x 2 regions


def test_plan_rejects_empty_markets_or_regions() -> None:
    with pytest.raises(ValueError, match="market"):
        plan_half_market_capture(["evt-1"], markets="", known_remaining=None)
    with pytest.raises(ValueError, match="region"):
        plan_half_market_capture(["evt-1"], regions="", known_remaining=None)


# ---------------------------------------------------------------------------
# assemble_events_payload
# ---------------------------------------------------------------------------


def test_assemble_events_payload_is_a_bulk_shaped_json_array() -> None:
    payload = assemble_events_payload([{"id": "a"}, {"id": "b"}])
    decoded = json.loads(payload)
    assert decoded == [{"id": "a"}, {"id": "b"}]


# ---------------------------------------------------------------------------
# capture_half_markets: end-to-end assembly, no network
# ---------------------------------------------------------------------------


def _half_market_event_payload(event_id: str, commence: datetime) -> dict[str, Any]:
    return {
        "id": event_id,
        "sport_key": "americanfootball_nfl",
        "commence_time": _iso_z(commence),
        "home_team": "Seattle Seahawks",
        "away_team": "New England Patriots",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": _iso_z(commence - timedelta(days=1)),
                "markets": [
                    {
                        "key": "spreads_h1",
                        "last_update": _iso_z(commence - timedelta(days=1)),
                        "outcomes": [
                            {"name": "Seattle Seahawks", "price": -110, "point": -1.5},
                            {"name": "New England Patriots", "price": -110, "point": 1.5},
                        ],
                    },
                    {
                        "key": "totals_h1",
                        "last_update": _iso_z(commence - timedelta(days=1)),
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 21.5},
                            {"name": "Under", "price": -110, "point": 21.5},
                        ],
                    },
                ],
            }
        ],
    }


def test_capture_half_markets_filters_fetches_and_writes_one_snapshot(tmp_path: Path) -> None:
    market_root = tmp_path / "market" / "raw"
    now = datetime(2026, 9, 9, 15, 0, tzinfo=UTC)
    start, end = current_week_kickoff_window(now)
    inside_commence = start + timedelta(hours=5)
    outside_commence = end + timedelta(hours=5)

    bulk_events = [
        {
            "id": "evt-in",
            "sport_key": "americanfootball_nfl",
            "commence_time": _iso_z(inside_commence),
            "home_team": "Seattle Seahawks",
            "away_team": "New England Patriots",
            "bookmakers": [],
        },
        {
            "id": "evt-out",
            "sport_key": "americanfootball_nfl",
            "commence_time": _iso_z(outside_commence),
            "home_team": "Kansas City Chiefs",
            "away_team": "Denver Broncos",
            "bookmakers": [],
        },
    ]
    _write_bulk_dir(market_root, "20260908T090000Z", bulk_events, remaining="1000")

    calls: list[str] = []

    def stub_fetch(
        *, api_key: str, event_id: str, markets: str, regions: str
    ) -> tuple[dict[str, Any], dict[str, str]]:
        calls.append(event_id)
        assert api_key == "test-key"
        assert markets == HALF_MARKETS_DEFAULT
        assert regions == "us"
        payload = _half_market_event_payload(event_id, inside_commence)
        quota = {
            "requests_remaining": str(996 - len(calls)),
            "requests_used": str(20 + len(calls)),
            "requests_last": "2",
        }
        return payload, quota

    features = pd.DataFrame(
        {
            "game_id": ["2026_01_NE_SEA"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "kickoff": [pd.Timestamp(inside_commence)],
        }
    )

    result = capture_half_markets(
        market_root=market_root,
        features=features,
        api_key="test-key",
        observed_at=now,
        fetch=stub_fetch,
        receipt_clock=lambda: now,
        sleep_seconds=0,
    )

    # Only the in-window event was ever fetched -- evt-out never called.
    assert calls == ["evt-in"]
    assert result.events_returned == 1
    assert result.quotes_written == 4  # 2 markets x 2 outcomes
    assert result.total_credits_spent == 2  # from the stub's requests_last header
    assert (
        result.snapshot.snapshot_id == "20260909T150000Z-halves"
    )  # from `now`, not the bulk stamp
    assert result.snapshot.root.name.endswith("-halves")

    quotes = pd.read_parquet(result.snapshot.quotes_path)
    assert set(quotes["market"]) == {"spreads_h1", "totals_h1"}
    assert quotes["nflverse_game_id"].eq("2026_01_NE_SEA").all()

    manifest = json.loads(result.snapshot.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_kind"] == "event_halves"
    assert manifest["events_requested"] == 1
    assert manifest["events_returned"] == 1
    assert manifest["credits_per_event"] == 4
    assert manifest["total_credits_this_run"] == 2
    assert manifest["source_bulk_snapshot_id"] == "20260908T090000Z"
    assert manifest["per_request"] == [
        {
            "event_id": "evt-in",
            "observed_at_utc": now.isoformat(),
            "requests_remaining": "995",
            "requests_used": "21",
            "requests_last": "2",
        }
    ]


def test_capture_half_markets_refuses_before_any_call_when_floor_would_breach(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "market" / "raw"
    now = datetime(2026, 9, 9, 15, 0, tzinfo=UTC)
    start, _end = current_week_kickoff_window(now)
    commence = start + timedelta(hours=5)
    bulk_events = [
        {
            "id": "evt-in",
            "sport_key": "americanfootball_nfl",
            "commence_time": _iso_z(commence),
            "home_team": "Seattle Seahawks",
            "away_team": "New England Patriots",
            "bookmakers": [],
        }
    ]
    # 1 event x 4 credits = 4; 603 - 4 = 599 < the 600 floor.
    _write_bulk_dir(market_root, "20260908T090000Z", bulk_events, remaining="603")

    calls: list[str] = []

    def stub_fetch(**kwargs: Any) -> tuple[dict[str, Any], dict[str, str]]:
        calls.append(kwargs["event_id"])
        raise AssertionError("must never be called once the plan is refused")

    features = pd.DataFrame(
        {"game_id": [], "home_team": [], "away_team": [], "kickoff": []},
    )

    with pytest.raises(QuotaFloorRefusal, match="600"):
        capture_half_markets(
            market_root=market_root,
            features=features,
            api_key="test-key",
            observed_at=now,
            fetch=stub_fetch,
            quota_floor=600,
        )
    assert calls == []


def test_capture_half_markets_requires_an_existing_bulk_snapshot(tmp_path: Path) -> None:
    market_root = tmp_path / "market" / "raw"
    features = pd.DataFrame({"game_id": [], "home_team": [], "away_team": [], "kickoff": []})
    with pytest.raises(ValueError, match="No bulk board snapshot"):
        capture_half_markets(
            market_root=market_root,
            features=features,
            api_key="test-key",
            observed_at=datetime(2026, 9, 9, 15, 0, tzinfo=UTC),
        )


def test_capture_half_markets_requires_at_least_one_in_week_event(tmp_path: Path) -> None:
    market_root = tmp_path / "market" / "raw"
    now = datetime(2026, 9, 9, 15, 0, tzinfo=UTC)
    _start, end = current_week_kickoff_window(now)
    bulk_events = [
        {
            "id": "evt-out",
            "sport_key": "americanfootball_nfl",
            "commence_time": _iso_z(end + timedelta(hours=1)),
            "home_team": "Seattle Seahawks",
            "away_team": "New England Patriots",
            "bookmakers": [],
        }
    ]
    _write_bulk_dir(market_root, "20260908T090000Z", bulk_events)
    features = pd.DataFrame({"game_id": [], "home_team": [], "away_team": [], "kickoff": []})
    with pytest.raises(ValueError, match="current week"):
        capture_half_markets(
            market_root=market_root,
            features=features,
            api_key="test-key",
            observed_at=now,
        )


def test_response_receipts_stay_distinct_and_in_play_is_not_pregame(tmp_path: Path) -> None:
    from nfl_ats.market_data import latest_book_quotes

    now = datetime(2026, 9, 9, 15, tzinfo=UTC)
    times = [now + timedelta(seconds=1), now + timedelta(seconds=3)]
    commence = now + timedelta(seconds=2)
    root = tmp_path / "raw"
    _write_bulk_dir(
        root,
        "20260909T140000Z",
        [{"id": name, "commence_time": _iso_z(commence)} for name in ("early", "late")],
    )
    receipts = iter(times)

    def fetch(**kwargs: Any) -> tuple[dict[str, Any], dict[str, str]]:
        return _half_market_event_payload(kwargs["event_id"], commence), {"requests_last": "2"}

    result = capture_half_markets(
        market_root=root,
        features=pd.DataFrame({"game_id": [], "home_team": [], "away_team": [], "kickoff": []}),
        api_key="test",
        observed_at=now,
        receipt_clock=lambda: next(receipts),
        fetch=fetch,
        sleep_seconds=0,
    )
    quotes = pd.read_parquet(result.snapshot.quotes_path)
    for event_id, instant in zip(("early", "late"), times, strict=True):
        subset = quotes.loc[quotes["provider_event_id"].eq(event_id)]
        assert len(subset) == 4
        assert pd.to_datetime(subset["observed_at_utc"], utc=True).eq(instant).all()
        assert subset["in_play"].eq(event_id == "late").all()
    assert set(latest_book_quotes(quotes)["provider_event_id"]) == {"early"}
    assert len(json.loads((result.snapshot.root / "response.json").read_text())) == 2
