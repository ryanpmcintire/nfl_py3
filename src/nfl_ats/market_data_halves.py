from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as _time
from pathlib import Path
from time import sleep as _sleep
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ats.market_data import (
    ODDS_API_SPORT,
    MarketSnapshot,
    attach_nflverse_game_ids,
    parse_odds_api_response,
    write_market_snapshot,
)
from nfl_ats.nfl_week import week_cycle_sunday
from nfl_ats.odds_backfill import DEFAULT_QUOTA_FLOOR

_EASTERN = ZoneInfo("America/New_York")

ODDS_API_EVENT_ODDS_URL_TEMPLATE = (
    f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/events/{{event_id}}/odds"
)

HALF_MARKETS: tuple[str, ...] = ("spreads_h1", "spreads_h2", "totals_h1", "totals_h2")
HALF_MARKETS_DEFAULT = ",".join(HALF_MARKETS)

CAPTURE_KIND = "event_halves"
SNAPSHOT_SUFFIX = "-halves"

_BULK_SNAPSHOT_NAME = re.compile(r"^(\d{8}T\d{6}Z)$")


class NoEventsToCapture(RuntimeError):
    pass


class QuotaFloorRefusal(RuntimeError):
    pass


def _utc(instant: datetime | None) -> datetime:
    value = instant or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _parse_iso_utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def current_week_kickoff_window(now: datetime) -> tuple[datetime, datetime]:

    if now.tzinfo is None:
        raise ValueError("now must carry an explicit timezone")
    eastern_date = now.astimezone(_EASTERN).date()
    cycle_sunday = week_cycle_sunday(eastern_date)
    tuesday = cycle_sunday - timedelta(days=5)
    next_tuesday = tuesday + timedelta(days=7)
    start = datetime.combine(tuesday, _time.min, tzinfo=_EASTERN).astimezone(UTC)
    end = datetime.combine(next_tuesday, _time.min, tzinfo=_EASTERN).astimezone(UTC)
    return start, end


def filter_events_to_week(events: Sequence[Any], now: datetime) -> list[dict[str, Any]]:

    start, end = current_week_kickoff_window(now)
    selected: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        commence = _parse_iso_utc(event.get("commence_time"))
        if commence is None:
            continue
        if start <= commence < end:
            selected.append(event)
    return selected


def filter_events_to_next_week(
    events: Sequence[Any], now: datetime, schedule: pd.DataFrame
) -> list[dict[str, Any]]:
    instant = _utc(now)
    horizon = instant + timedelta(days=8)
    upcoming = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("id")
        and (commence := _parse_iso_utc(event.get("commence_time"))) is not None
        and commence > instant
    ]
    if {"season", "week", "kickoff"}.issubset(schedule.columns):
        games = schedule.copy()
        games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True, errors="coerce")
        games = games.loc[games["kickoff"].gt(instant)].sort_values("kickoff")
        if games.empty or games.iloc[0]["kickoff"] > horizon:
            return []
        first = games.iloc[0]
        slate = games.loc[games["season"].eq(first["season"]) & games["week"].eq(first["week"])]
        start, _ = current_week_kickoff_window(first["kickoff"].to_pydatetime())
        _, end = current_week_kickoff_window(slate["kickoff"].max().to_pydatetime())
    else:
        times = [_parse_iso_utc(event["commence_time"]) for event in upcoming]
        valid = [value for value in times if value is not None and value <= horizon]
        if not valid:
            return []
        start, end = current_week_kickoff_window(min(valid))
    return [
        event
        for event in upcoming
        if (commence := _parse_iso_utc(event["commence_time"])) is not None
        and start <= commence < end
    ]


@dataclass(frozen=True)
class BulkSnapshotRef:
    snapshot_id: str
    root: Path
    raw_path: Path
    manifest: dict[str, Any]


def newest_bulk_snapshot(market_root: Path) -> BulkSnapshotRef | None:

    if not market_root.is_dir():
        return None
    candidates = [
        child
        for child in market_root.iterdir()
        if child.is_dir()
        and _BULK_SNAPSHOT_NAME.match(child.name)
        and (child / "response.json").is_file()
        and (child / "manifest.json").is_file()
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.name)
    manifest = json.loads((newest / "manifest.json").read_text(encoding="utf-8"))
    return BulkSnapshotRef(
        snapshot_id=newest.name,
        root=newest,
        raw_path=newest / "response.json",
        manifest=manifest,
    )


@dataclass(frozen=True)
class HalfMarketCapturePlan:
    event_ids: tuple[str, ...]
    markets: str
    regions: str
    credits_per_event: int
    planned_credits: int
    known_remaining: float | None
    quota_floor: int
    refused: bool
    refusal_reason: str | None


def plan_half_market_capture(
    event_ids: Sequence[str],
    *,
    markets: str = HALF_MARKETS_DEFAULT,
    regions: str = "us",
    known_remaining: float | None,
    quota_floor: int = DEFAULT_QUOTA_FLOOR,
) -> HalfMarketCapturePlan:

    market_count = len([m for m in markets.split(",") if m.strip()])
    if market_count == 0:
        raise ValueError("At least one market is required")
    region_count = len([r for r in regions.split(",") if r.strip()])
    if region_count == 0:
        raise ValueError("At least one region is required")
    credits_per_event = market_count * region_count
    planned_credits = credits_per_event * len(event_ids)
    refused = False
    reason: str | None = None
    if known_remaining is not None and (known_remaining - planned_credits) < quota_floor:
        refused = True
        reason = (
            f"{known_remaining:.0f} credits remaining; this capture would cost "
            f"{planned_credits} ({len(event_ids)} events x {credits_per_event} "
            f"credits/event), which would leave fewer than the {quota_floor}-credit floor"
        )
    return HalfMarketCapturePlan(
        event_ids=tuple(str(event_id) for event_id in event_ids),
        markets=markets,
        regions=regions,
        credits_per_event=credits_per_event,
        planned_credits=planned_credits,
        known_remaining=known_remaining,
        quota_floor=quota_floor,
        refused=refused,
        refusal_reason=reason,
    )


def fetch_event_odds(
    *,
    api_key: str,
    event_id: str,
    markets: str = HALF_MARKETS_DEFAULT,
    regions: str = "us",
    timeout: int = 30,
) -> tuple[dict[str, Any], dict[str, str]]:

    if not api_key.strip():
        raise ValueError("The Odds API key is empty")
    parameters = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = (
        ODDS_API_EVENT_ODDS_URL_TEMPLATE.format(event_id=urllib.parse.quote(event_id, safe=""))
        + "?"
        + urllib.parse.urlencode(parameters)
    )
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-ats/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            quota = {
                "requests_remaining": response.headers.get("x-requests-remaining", ""),
                "requests_used": response.headers.get("x-requests-used", ""),
                "requests_last": response.headers.get("x-requests-last", ""),
            }
    except urllib.error.HTTPError as error:
        raise ValueError(f"The Odds API returned HTTP {error.code} for event {event_id}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"Unable to reach The Odds API for event {event_id}") from error
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError(f"Per-event odds response for {event_id} must be a JSON object")
    return decoded, quota


def assemble_events_payload(events: Sequence[dict[str, Any]]) -> bytes:

    return json.dumps(list(events), separators=(",", ":")).encode()


@dataclass(frozen=True)
class HalfMarketCaptureResult:
    plan: HalfMarketCapturePlan
    snapshot: MarketSnapshot
    events_returned: int
    quotes_written: int
    total_credits_spent: int
    quota_after: dict[str, str]
    source_bulk_snapshot_id: str


def capture_half_markets(
    *,
    market_root: Path,
    features: pd.DataFrame,
    api_key: str,
    observed_at: datetime | None = None,
    week_reference: datetime | None = None,
    markets: str = HALF_MARKETS_DEFAULT,
    regions: str = "us",
    quota_floor: int = DEFAULT_QUOTA_FLOOR,
    fetch: Callable[..., tuple[dict[str, Any], dict[str, str]]] | None = None,
    sleep_seconds: float = 0.25,
    sleeper: Callable[[float], None] | None = None,
    receipt_clock: Callable[[], datetime] | None = None,
) -> HalfMarketCaptureResult:

    observed = _utc(observed_at)
    week_now = _utc(week_reference) if week_reference is not None else observed
    bulk = newest_bulk_snapshot(market_root)
    if bulk is None:
        raise ValueError(
            f"No bulk board snapshot found under {market_root}; run odds-ingest first "
            "(this job requires one to already exist -- see requires=('odds_tue_open',) / "
            "requires=('odds_sat',) in scripts/capture_scheduler.py)"
        )
    events = json.loads(bulk.raw_path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError(f"Bulk snapshot {bulk.snapshot_id} response.json is not a JSON array")
    week_events = filter_events_to_next_week(events, week_now, features)
    if not week_events:
        raise NoEventsToCapture(
            f"No events in bulk snapshot {bulk.snapshot_id} fall inside the upcoming week's "
            "kickoff window; nothing to capture"
        )
    event_ids = [str(event["id"]) for event in week_events if event.get("id")]
    known_remaining = _parse_float(bulk.manifest.get("quota", {}).get("requests_remaining"))
    plan = plan_half_market_capture(
        event_ids,
        markets=markets,
        regions=regions,
        known_remaining=known_remaining,
        quota_floor=quota_floor,
    )
    if plan.refused:
        raise QuotaFloorRefusal(plan.refusal_reason or "quota floor breached")

    fetch_event = fetch or fetch_event_odds
    pause = sleeper or _sleep
    collected: list[dict[str, Any]] = []
    quote_frames: list[pd.DataFrame] = []
    per_request: list[dict[str, Any]] = []
    quota_last: dict[str, str] = {}
    total_spent = 0
    for index, event_id in enumerate(event_ids):
        if index > 0 and sleep_seconds > 0:
            pause(sleep_seconds)
        payload, quota = fetch_event(
            api_key=api_key, event_id=event_id, markets=markets, regions=regions
        )
        received = _utc(receipt_clock() if receipt_clock is not None else None)
        collected.append(payload)
        event_quotes = parse_odds_api_response(
            assemble_events_payload([payload]), observed_at=received
        )
        event_quotes["in_play"] = pd.to_datetime(event_quotes["commence_time_utc"], utc=True).le(
            pd.Timestamp(received)
        )
        quote_frames.append(event_quotes)
        cost = _parse_float(quota.get("requests_last"))
        total_spent += int(cost) if cost is not None else plan.credits_per_event
        quota_last = quota
        per_request.append({"event_id": event_id, **quota, "observed_at_utc": received.isoformat()})

    payload_bytes = assemble_events_payload(collected)
    quotes = pd.concat(quote_frames, ignore_index=True)
    quotes = attach_nflverse_game_ids(quotes, features)
    request_metadata = {
        "sport": ODDS_API_SPORT,
        "endpoint": "per_event",
        "regions": regions,
        "markets": markets,
        "odds_format": "american",
        "source_bulk_snapshot_id": bulk.snapshot_id,
    }
    snapshot = write_market_snapshot(
        payload_bytes,
        quotes,
        market_root,
        observed_at=observed,
        request_metadata=request_metadata,
        quota=quota_last,
        snapshot_suffix=SNAPSHOT_SUFFIX,
        extra_manifest={
            "capture_kind": CAPTURE_KIND,
            "events_requested": len(event_ids),
            "events_returned": len(collected),
            "credits_per_event": plan.credits_per_event,
            "total_credits_this_run": total_spent,
            "quota_floor": quota_floor,
            "known_remaining_before_run": known_remaining,
            "per_request": per_request,
            "source_bulk_snapshot_id": bulk.snapshot_id,
            "week_reference_override_utc": (
                week_reference.astimezone(UTC).isoformat() if week_reference is not None else None
            ),
        },
    )
    return HalfMarketCaptureResult(
        plan=plan,
        snapshot=snapshot,
        events_returned=len(collected),
        quotes_written=len(quotes),
        total_credits_spent=total_spent,
        quota_after=quota_last,
        source_bulk_snapshot_id=bulk.snapshot_id,
    )


__all__ = [
    "CAPTURE_KIND",
    "HALF_MARKETS",
    "HALF_MARKETS_DEFAULT",
    "ODDS_API_EVENT_ODDS_URL_TEMPLATE",
    "SNAPSHOT_SUFFIX",
    "BulkSnapshotRef",
    "HalfMarketCapturePlan",
    "HalfMarketCaptureResult",
    "QuotaFloorRefusal",
    "assemble_events_payload",
    "capture_half_markets",
    "current_week_kickoff_window",
    "fetch_event_odds",
    "filter_events_to_week",
    "newest_bulk_snapshot",
    "plan_half_market_capture",
]
