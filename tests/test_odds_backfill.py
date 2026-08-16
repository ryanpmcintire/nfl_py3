from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.odds_backfill import (
    HISTORICAL_CAPTURE_KIND,
    BackfillTarget,
    completed_backfill_requests,
    execute_backfill,
    parse_historical_odds_response,
    plan_backfill,
    store_historical_snapshot,
    summarize_backfill_plan,
)


def _event(home_point: float = -3.5) -> dict[str, Any]:
    return {
        "id": "event-1",
        "sport_key": "americanfootball_nfl",
        "commence_time": "2024-09-15T17:00:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Cincinnati Bengals",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2024-09-10T12:54:00Z",
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": "2024-09-10T12:54:00Z",
                        "outcomes": [
                            {"name": "Kansas City Chiefs", "price": -110, "point": home_point},
                            {"name": "Cincinnati Bengals", "price": -110, "point": -home_point},
                        ],
                    },
                    {
                        "key": "totals",
                        "last_update": "2024-09-10T12:54:00Z",
                        "outcomes": [
                            {"name": "Over", "price": -108, "point": 47.5},
                            {"name": "Under", "price": -112, "point": 47.5},
                        ],
                    },
                ],
            }
        ],
    }


def _historical_payload(
    timestamp: str = "2024-09-10T12:55:00Z",
    previous_timestamp: str | None = "2024-09-10T12:50:00Z",
    next_timestamp: str | None = "2024-09-10T13:00:00Z",
    home_point: float = -3.5,
) -> bytes:
    wrapper = {
        "timestamp": timestamp,
        "previous_timestamp": previous_timestamp,
        "next_timestamp": next_timestamp,
        "data": [_event(home_point)],
    }
    return json.dumps(wrapper, separators=(",", ":")).encode()


def _schedule_frame(rows: list[tuple[int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [season for season, _, _ in rows],
            "week": [week for _, week, _ in rows],
            "gameday": [gameday for _, _, gameday in rows],
        }
    )


def _matching_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2024_02_CIN_KC"],
            "season": [2024],
            "week": [2],
            "gameday": ["2024-09-15"],
            "home_team": ["KC"],
            "away_team": ["CIN"],
            "kickoff": [pd.Timestamp("2024-09-15T17:00:00Z")],
        }
    )


def _target(
    requested_at: datetime,
    *,
    markets: str = "spreads,totals",
    credits: int = 20,
    label: str = "thu_pre_tnf",
) -> BackfillTarget:
    return BackfillTarget(
        season=2024,
        week=2,
        label=label,
        requested_at_utc=requested_at,
        markets=markets,
        regions="us",
        credits=credits,
    )


def test_planner_generates_dst_correct_decision_timestamps() -> None:
    schedule = _schedule_frame(
        [
            (2024, 2, "2024-09-12"),
            (2024, 2, "2024-09-15"),
            (2024, 2, "2024-09-15"),
            (2024, 2, "2024-09-16"),
            (2024, 10, "2024-11-07"),
            (2024, 10, "2024-11-10"),
            (2024, 10, "2024-11-10"),
            (2024, 10, "2024-11-11"),
        ]
    )
    targets = plan_backfill(schedule, 2024, 2024)
    assert len(targets) == 12
    by_key = {(target.week, target.label): target for target in targets}
    # September week: Eastern daylight time (UTC-4).
    assert by_key[(2, "tue_open")].requested_at_utc == datetime(2024, 9, 10, 13, 0, tzinfo=UTC)
    assert by_key[(2, "thu_pre_tnf")].requested_at_utc == datetime(2024, 9, 12, 22, 0, tzinfo=UTC)
    assert by_key[(2, "sat_midday")].requested_at_utc == datetime(2024, 9, 14, 16, 0, tzinfo=UTC)
    assert by_key[(2, "sun_early_close")].requested_at_utc == datetime(
        2024, 9, 15, 16, 30, tzinfo=UTC
    )
    assert by_key[(2, "sun_late_close")].requested_at_utc == datetime(
        2024, 9, 15, 20, 15, tzinfo=UTC
    )
    assert by_key[(2, "mon_pre_mnf")].requested_at_utc == datetime(2024, 9, 16, 23, 0, tzinfo=UTC)
    # November week: Eastern standard time (UTC-5) after the DST change.
    assert by_key[(10, "tue_open")].requested_at_utc == datetime(2024, 11, 5, 14, 0, tzinfo=UTC)
    assert by_key[(10, "sun_early_close")].requested_at_utc == datetime(
        2024, 11, 10, 17, 30, tzinfo=UTC
    )
    assert by_key[(10, "mon_pre_mnf")].requested_at_utc == datetime(2024, 11, 12, 0, 0, tzinfo=UTC)
    # h2h is planned only at the Tuesday open and Sunday early close.
    assert by_key[(2, "tue_open")].markets == "spreads,totals,h2h"
    assert by_key[(2, "sun_early_close")].markets == "spreads,totals,h2h"
    assert by_key[(2, "tue_open")].credits == 30
    assert by_key[(2, "thu_pre_tnf")].markets == "spreads,totals"
    assert by_key[(2, "thu_pre_tnf")].credits == 20
    summary = summarize_backfill_plan(targets)
    assert summary["total_calls"] == 12
    assert summary["total_credits"] == 2 * (30 + 20 + 20 + 30 + 20 + 20)
    assert summary["seasons"]["2024"] == {"weeks": 2, "calls": 12, "credits": 280}


def test_planner_spans_dst_transition_within_one_week() -> None:
    # DST ended 02:00 on Sunday 2021-11-07, in the middle of this week's cycle.
    schedule = _schedule_frame([(2021, 9, "2021-11-07")])
    targets = plan_backfill(schedule, 2021, 2021)
    by_label = {target.label: target for target in targets}
    assert by_label["sat_midday"].requested_at_utc == datetime(2021, 11, 6, 16, 0, tzinfo=UTC)
    assert by_label["sun_early_close"].requested_at_utc == datetime(2021, 11, 7, 17, 30, tzinfo=UTC)


def test_planner_anchor_ignores_rescheduled_tuesday_games() -> None:
    # 2020 week 5: BUF-TEN was moved to Tuesday 2020-10-13; the anchor Sunday
    # must remain 2020-10-11 (mode of the week's Tue..Mon cycle Sundays).
    schedule = _schedule_frame(
        [
            (2020, 5, "2020-10-08"),
            (2020, 5, "2020-10-11"),
            (2020, 5, "2020-10-11"),
            (2020, 5, "2020-10-12"),
            (2020, 5, "2020-10-13"),
        ]
    )
    targets = plan_backfill(schedule, 2020, 2020)
    by_label = {target.label: target for target in targets}
    assert by_label["tue_open"].requested_at_utc == datetime(2020, 10, 6, 13, 0, tzinfo=UTC)
    assert by_label["sun_early_close"].requested_at_utc == datetime(
        2020, 10, 11, 16, 30, tzinfo=UTC
    )


def test_planner_skips_timestamps_before_archive_start() -> None:
    schedule = _schedule_frame([(2020, 1, "2020-06-07")])
    targets = plan_backfill(schedule, 2020, 2020)
    assert [target.label for target in targets] == [
        "sat_midday",
        "sun_early_close",
        "sun_late_close",
        "mon_pre_mnf",
    ]
    assert min(target.requested_at_utc for target in targets) >= datetime(2020, 6, 6, tzinfo=UTC)


def test_planner_filters_and_guards() -> None:
    schedule = _schedule_frame([(2024, 1, "2024-09-08"), (2024, 2, "2024-09-15")])
    only_week_two = plan_backfill(schedule, 2024, 2024, weeks=[2])
    assert {target.week for target in only_week_two} == {2}
    only_tuesday = plan_backfill(schedule, 2024, 2024, labels=["tue_open"])
    assert [target.label for target in only_tuesday] == ["tue_open", "tue_open"]
    with pytest.raises(ValueError, match="Unknown decision labels"):
        plan_backfill(schedule, 2024, 2024, labels=["breakfast"])
    with pytest.raises(ValueError, match="end-season"):
        plan_backfill(schedule, 2024, 2023)
    with pytest.raises(ValueError, match="missing columns"):
        plan_backfill(pd.DataFrame({"season": []}), 2024, 2024)


def test_parse_historical_response_marks_backfill_rows() -> None:
    payload = _historical_payload()
    capture = parse_historical_odds_response(payload)
    assert capture.snapshot_at_utc == datetime(2024, 9, 10, 12, 55, tzinfo=UTC)
    assert capture.previous_snapshot_at_utc == datetime(2024, 9, 10, 12, 50, tzinfo=UTC)
    assert capture.next_snapshot_at_utc == datetime(2024, 9, 10, 13, 0, tzinfo=UTC)
    quotes = capture.quotes
    assert len(quotes) == 4
    assert set(quotes["market"]) == {"spreads", "totals"}
    assert quotes["capture_kind"].eq(HISTORICAL_CAPTURE_KIND).all()
    assert quotes["observed_at_utc"].eq(pd.Timestamp("2024-09-10T12:55:00Z")).all()
    assert quotes["raw_response_sha256"].eq(hashlib.sha256(payload).hexdigest()).all()
    boundary = parse_historical_odds_response(
        _historical_payload(previous_timestamp=None, next_timestamp=None)
    )
    assert boundary.previous_snapshot_at_utc is None
    assert boundary.next_snapshot_at_utc is None


def test_parse_historical_response_guards() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_historical_odds_response(b"[]")
    with pytest.raises(ValueError, match="snapshot timestamp"):
        parse_historical_odds_response(b'{"data": []}')
    with pytest.raises(ValueError, match="data array"):
        parse_historical_odds_response(b'{"timestamp": "2024-09-10T12:55:00Z"}')


def test_store_historical_snapshot_layout_and_immutability(tmp_path: Path) -> None:
    payload = _historical_payload()
    capture = parse_historical_odds_response(payload)
    target = _target(datetime(2024, 9, 10, 13, 0, tzinfo=UTC), label="tue_open", credits=30)
    snapshot = store_historical_snapshot(
        payload, capture, tmp_path, target=target, quota={"requests_remaining": "19970"}
    )
    assert snapshot.snapshot_id == "20240910T125500Z"
    assert snapshot.raw_path.read_bytes() == payload
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_kind"] == HISTORICAL_CAPTURE_KIND
    assert manifest["requested_at_utc"] == "2024-09-10T13:00:00+00:00"
    assert manifest["observed_at_utc"] == "2024-09-10T12:55:00+00:00"
    assert manifest["snapshot_timestamp_utc"] == "2024-09-10T12:55:00+00:00"
    assert manifest["previous_snapshot_timestamp_utc"] == "2024-09-10T12:50:00+00:00"
    assert manifest["next_snapshot_timestamp_utc"] == "2024-09-10T13:00:00+00:00"
    assert manifest["quota"] == {"requests_remaining": "19970"}
    assert manifest["request"]["endpoint"] == "historical"
    assert manifest["request"]["decision_label"] == "tue_open"
    stored = pd.read_parquet(snapshot.quotes_path)
    assert stored["capture_kind"].eq(HISTORICAL_CAPTURE_KIND).all()
    assert completed_backfill_requests(tmp_path) == {"2024-09-10T13:00:00+00:00"}
    with pytest.raises(ValueError, match="already exists"):
        store_historical_snapshot(payload, capture, tmp_path, target=target)


class _FakeFetcher:
    def __init__(self, remaining: list[str], timestamps: list[str] | None = None) -> None:
        self.remaining = remaining
        self.timestamps = timestamps
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, *, api_key: str, snapshot_at: datetime, regions: str, markets: str
    ) -> tuple[bytes, dict[str, str]]:
        index = len(self.calls)
        self.calls.append(
            {"api_key": api_key, "snapshot_at": snapshot_at, "regions": regions, "markets": markets}
        )
        if self.timestamps is None:
            timestamp = snapshot_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            timestamp = self.timestamps[index]
        return (
            _historical_payload(timestamp=timestamp),
            {
                "requests_remaining": self.remaining[index],
                "requests_used": str(20000 - int(self.remaining[index])),
                "requests_last": "20",
            },
        )


def test_execute_backfill_stops_at_quota_floor(tmp_path: Path) -> None:
    targets = [
        _target(datetime(2024, 9, 10, 13, 0, tzinfo=UTC)),
        _target(datetime(2024, 9, 12, 22, 0, tzinfo=UTC)),
        _target(datetime(2024, 9, 14, 16, 0, tzinfo=UTC)),
    ]
    fetcher = _FakeFetcher(remaining=["630", "610", "590"])
    sleeps: list[float] = []
    result = execute_backfill(
        targets,
        tmp_path,
        _matching_schedule(),
        api_key="secret",
        quota_floor=600,
        fetch=fetcher,
        sleeper=sleeps.append,
        log=lambda message: None,
    )
    assert len(fetcher.calls) == 2
    assert result["stopped_reason"] == "quota_floor"
    assert result["completed"] is False
    assert result["credits_spent"] == 40
    assert result["requests_remaining"] == 610.0
    assert len(result["stored_snapshots"]) == 2
    assert sleeps == [1.0]
    # The archived rows are matched to the schedule and marked as backfill.
    stored = pd.read_parquet(tmp_path / result["stored_snapshots"][0] / "quotes.parquet")
    assert stored["nflverse_game_id"].eq("2024_02_CIN_KC").all()
    assert stored["capture_kind"].eq(HISTORICAL_CAPTURE_KIND).all()


def test_execute_backfill_resume_skips_stored_targets(tmp_path: Path) -> None:
    target = _target(datetime(2024, 9, 10, 13, 0, tzinfo=UTC))
    first = _FakeFetcher(remaining=["19980"])
    execute_backfill(
        [target],
        tmp_path,
        _matching_schedule(),
        api_key="secret",
        fetch=first,
        log=lambda message: None,
    )
    with pytest.raises(ValueError, match="pass --resume"):
        execute_backfill(
            [target],
            tmp_path,
            _matching_schedule(),
            api_key="secret",
            fetch=first,
            log=lambda message: None,
        )
    second = _FakeFetcher(remaining=["19960"])
    result = execute_backfill(
        [target],
        tmp_path,
        _matching_schedule(),
        api_key="secret",
        resume=True,
        fetch=second,
        log=lambda message: None,
    )
    assert second.calls == []
    assert result["skipped_existing"] == 1
    assert result["attempted_calls"] == 0
    assert result["credits_spent"] == 0
    assert result["completed"] is True


def test_execute_backfill_budget_precheck_and_collisions(tmp_path: Path) -> None:
    targets = [
        _target(datetime(2024, 9, 10, 13, 0, tzinfo=UTC)),
        _target(datetime(2024, 9, 10, 13, 10, tzinfo=UTC)),
    ]
    with pytest.raises(ValueError, match="exceeds the budget"):
        execute_backfill(
            targets,
            tmp_path,
            _matching_schedule(),
            api_key="secret",
            budget=30,
            fetch=_FakeFetcher(remaining=["1", "1"]),
            log=lambda message: None,
        )
    # Two requested times resolving to one provider snapshot must not overwrite.
    fetcher = _FakeFetcher(
        remaining=["19980", "19960"],
        timestamps=["2024-09-10T12:55:00Z", "2024-09-10T12:55:00Z"],
    )
    result = execute_backfill(
        targets,
        tmp_path,
        _matching_schedule(),
        api_key="secret",
        budget=40,
        sleep_seconds=0.0,
        fetch=fetcher,
        log=lambda message: None,
    )
    assert len(fetcher.calls) == 2
    assert result["stored_snapshots"] == ["20240910T125500Z"]
    assert result["snapshot_collisions"] == ["2024-09-10T13:10:00+00:00"]
    assert result["completed"] is True


def test_cli_odds_backfill_dry_run_and_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    features_path = tmp_path / "game_features.parquet"
    _matching_schedule().to_parquet(features_path, index=False)

    assert (
        cli.main(
            [
                "odds-backfill",
                "--features",
                str(features_path),
                "--start-season",
                "2024",
                "--end-season",
                "2024",
                "--dry-run",
            ]
        )
        == 0
    )
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["dry_run"] is True
    assert dry_run["plan"]["total_calls"] == 6
    assert dry_run["plan"]["total_credits"] == 140
    assert dry_run["plan"]["seasons"]["2024"] == {"weeks": 1, "calls": 6, "credits": 140}
    assert not data_root.exists()

    fetcher = _FakeFetcher(remaining=["19970"])
    monkeypatch.setattr("nfl_ats.odds_backfill.fetch_historical_odds_api", fetcher)
    monkeypatch.setenv("THE_ODDS_API_KEY", "test-key")
    assert (
        cli.main(
            [
                "odds-backfill",
                "--features",
                str(features_path),
                "--start-season",
                "2024",
                "--end-season",
                "2024",
                "--labels",
                "tue_open",
                "--budget",
                "30",
                "--sleep-seconds",
                "0",
            ]
        )
        == 0
    )
    executed = json.loads(capsys.readouterr().out)
    assert executed["result"]["stored_snapshots"] == ["20240910T130000Z"]
    assert executed["result"]["credits_spent"] == 30
    assert fetcher.calls[0]["markets"] == "spreads,totals,h2h"
    assert fetcher.calls[0]["api_key"] == "test-key"
    manifest_path = data_root / "market" / "raw" / "20240910T130000Z" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_kind"] == HISTORICAL_CAPTURE_KIND
    assert manifest["requested_at_utc"] == "2024-09-10T13:00:00+00:00"
