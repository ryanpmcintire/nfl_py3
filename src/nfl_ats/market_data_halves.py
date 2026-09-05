"""Per-event half/quarter-game market capture (LEAD-61 step 2).

The bulk board endpoint (``/v4/sports/{sport}/odds/``, ``nfl_ats.market_data
.fetch_odds_api``) rejects ``spreads_h1``/``spreads_h2``/``totals_h1``/
``totals_h2`` with HTTP 422 ``INVALID_MARKET`` -- measured by lane AM,
``docs/half_game_markets.md`` -- because The Odds API serves period/alternate
markets only from the per-event endpoint,
``/v4/sports/{sport}/events/{eventId}/odds``. That endpoint is scoped to ONE
game, so capturing a week's slate costs one call per event rather than one
call for the whole board -- lane AO measured this at 1 credit per
market x region, confirmed live (``docs/half_game_markets.md``, "Build
plan").

This module implements that design exactly: read event ids for the CURRENT
week only (never the whole ~272-event remaining-season board a bulk snapshot
carries) from the newest bulk-board snapshot already on disk, fetch each
event's half markets, and assemble the per-event responses into one JSON
array shaped exactly like the bulk endpoint's own top-level array -- so the
existing ``nfl_ats.market_data.parse_odds_api_response`` /
``write_market_snapshot`` machinery needs no format-specific branch.

Cost accounting and the quota floor
------------------------------------
``DEFAULT_QUOTA_FLOOR`` is imported from ``nfl_ats.odds_backfill`` (never
redeclared, so the two can never drift) even though that constant's origin
is the historical-endpoint quota policy in ``config/source_policies.json``
-- lane AO's probe found no existing floor scoped to the live ``/odds`` or
per-event endpoints, and recommended adopting the same 600-credit
convention rather than inventing a new one. The refusal check
(:func:`plan_half_market_capture`) runs BEFORE any per-event call is made,
using the ``x-requests-remaining`` header already sitting in the paired
bulk-board snapshot's own manifest (that capture always runs first, via
``requires=("odds_tue_open",)`` / ``requires=("odds_sat",)`` in
``scripts/capture_scheduler.py``) -- never a value fetched fresh for this
purpose alone.
"""

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

#: The four markets LEAD-61 asks for -- no ``spreads_q1``/other quarter
#: markets: those are out of scope for this lead (see ``docs/half_game_markets
#: .md`` "Build plan" -> "Manifest shape", step 1) and would need their own
#: measured cost before a future lead adds them.
HALF_MARKETS: tuple[str, ...] = ("spreads_h1", "spreads_h2", "totals_h1", "totals_h2")
HALF_MARKETS_DEFAULT = ",".join(HALF_MARKETS)

CAPTURE_KIND = "event_halves"
SNAPSHOT_SUFFIX = "-halves"

#: Matches ONLY a bare bulk-board snapshot directory name -- excludes both
#: this module's own ``<stamp>-halves`` output and any ad-hoc probe directory
#: (e.g. lane AO's ``<stamp>-event-halves-probe``), the same full-match
#: pattern ``scripts/capture_scheduler.py``'s ``SNAPSHOT_NAME`` uses.
_BULK_SNAPSHOT_NAME = re.compile(r"^(\d{8}T\d{6}Z)$")


class QuotaFloorRefusal(RuntimeError):
    """The planned per-event capture would breach the provider quota floor."""


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
    """The half-open ``[Tuesday 00:00, next Tuesday 00:00)`` America/New_York
    window (as UTC bounds) for the NFL week cycle containing ``now``.

    Anchored on :func:`nfl_ats.nfl_week.week_cycle_sunday`, the same anchor
    ``nfl_ats.odds_backfill.plan_backfill`` uses for weekly decision
    timestamps -- a Tuesday-through-Monday cycle, matching the project's own
    "grade at the Tuesday opener" convention rather than a Sunday-anchored
    calendar week.
    """

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
    """Keep only events whose ``commence_time`` falls in the current week.

    A bulk-board snapshot carries the entire remaining-season schedule
    (measured 272 events, ``docs/half_game_markets.md``) -- calling the
    per-event endpoint for every one of them would multiply the measured
    per-event cost by ~17x for no benefit. Events with a missing or
    unparseable ``commence_time`` are dropped rather than guessed into the
    window. ``events`` is typed ``Sequence[Any]``, not
    ``Sequence[dict[str, Any]]``, because it is untrusted external JSON --
    a non-dict element is dropped by the ``isinstance`` guard below rather
    than raising.
    """

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


@dataclass(frozen=True)
class BulkSnapshotRef:
    """The newest plain-timestamped bulk-board snapshot on disk."""

    snapshot_id: str
    root: Path
    raw_path: Path
    manifest: dict[str, Any]


def newest_bulk_snapshot(market_root: Path) -> BulkSnapshotRef | None:
    """The newest ``data/market/raw/<stamp>/`` bulk snapshot, or ``None``.

    Only directories matching the bare ``YYYYMMDDTHHMMSSZ`` name are
    considered -- this module's own ``<stamp>-halves`` output, and any other
    suffixed probe directory, are never mistaken for a bulk snapshot to read
    event ids from.
    """

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
    """The exact call/credit plan for one capture run, decided before any
    per-event request is made."""

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
    """Decide, from the LAST known quota reading, whether this capture may run.

    ``known_remaining=None`` (no quota reading available yet) never refuses --
    matching ``nfl_ats.odds_backfill.execute_backfill``'s own convention that
    an absent quota reading is not evidence of a breach.
    """

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
    """One live call to the per-event odds endpoint for a single event."""

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
    """Serialize accumulated per-event objects into one bulk-shaped JSON array."""

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
    """Capture the current week's half markets and write ONE snapshot.

    Reads event ids from the newest bulk-board snapshot under
    ``market_root`` (never spends a network request to list events itself),
    filters to the current week, refuses before any call if the paired
    bulk capture's own last-known quota reading says the plan would breach
    ``quota_floor``, then fetches each event's half markets and writes them
    as one combined snapshot via the existing
    ``nfl_ats.market_data.write_market_snapshot`` machinery, suffixed
    ``"-halves"`` so it can never collide with the paired bulk snapshot.

    ``observed_at`` (defaults to the real current time) names the snapshot;
    each response is stamped when received using ``receipt_clock`` (UTC now
    by default). The observation must always be the
    REAL capture instant, never fabricated, so it is never derived from a
    week-selection override. ``week_reference`` (defaults to ``observed_at``)
    is ONLY the anchor :func:`current_week_kickoff_window` uses to pick which
    week's events to fetch; a caller running an ad-hoc verification ahead of
    the scheduled window (e.g. proving this job against next week's slate
    mid-week, as this module's own live-capture proof did) passes it
    separately so the resulting snapshot still carries an honest capture
    timestamp instead of a fabricated future one.
    """

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
    week_events = filter_events_to_week(events, week_now)
    if not week_events:
        raise ValueError(
            f"No events in bulk snapshot {bulk.snapshot_id} fall inside the current week's "
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
