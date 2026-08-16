"""Historical point-in-time snapshot backfill from The Odds API historical endpoint.

The Odds API archives full odds boards from 2020-06-06 (10-minute snapshot
intervals; 5-minute from September 2022). One historical call returns the whole
slate at the snapshot closest to, and never later than, the requested time, and
costs ``10 x markets x regions`` credits. This module plans weekly decision
timestamps from the canonical schedule, fetches those snapshots, and stores them
in the same append-only layout as live captures with a ``historical_backfill``
marker so backfilled rows are never confused with live ones.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from time import sleep
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ats.market_data import (
    ODDS_API_SPORT,
    MarketSnapshot,
    attach_nflverse_game_ids,
    parse_odds_api_response,
    write_market_snapshot,
)

ODDS_API_HISTORICAL_URL = f"https://api.the-odds-api.com/v4/historical/sports/{ODDS_API_SPORT}/odds"
HISTORICAL_ARCHIVE_START_UTC = datetime(2020, 6, 6, tzinfo=UTC)
HISTORICAL_CREDITS_PER_MARKET_REGION = 10
HISTORICAL_CAPTURE_KIND = "historical_backfill"
DECISION_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_QUOTA_FLOOR = 600
DEFAULT_SLEEP_SECONDS = 1.0

BASE_MARKETS = ("spreads", "totals")
H2H_MARKET = "h2h"

# (label, day offset from the week's anchor Sunday, local Eastern wall time, with h2h)
DECISION_TIMES: tuple[tuple[str, int, time, bool], ...] = (
    ("tue_open", -5, time(9, 0), True),
    ("thu_pre_tnf", -3, time(18, 0), False),
    ("sat_midday", -1, time(12, 0), False),
    ("sun_early_close", 0, time(12, 30), True),
    ("sun_late_close", 0, time(16, 15), False),
    ("mon_pre_mnf", 1, time(19, 0), False),
)
DECISION_LABELS = tuple(label for label, _, _, _ in DECISION_TIMES)


@dataclass(frozen=True)
class BackfillTarget:
    """One planned historical snapshot request."""

    season: int
    week: int
    label: str
    requested_at_utc: datetime
    markets: str
    regions: str
    credits: int

    @property
    def requested_at_iso(self) -> str:
        return self.requested_at_utc.astimezone(UTC).isoformat()


@dataclass(frozen=True, eq=False)
class HistoricalOddsCapture:
    """Parsed historical response: snapshot metadata plus normalized quotes."""

    snapshot_at_utc: datetime
    previous_snapshot_at_utc: datetime | None
    next_snapshot_at_utc: datetime | None
    quotes: pd.DataFrame


def _parse_iso_utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _week_cycle_sunday(game_day: date) -> date:
    """Sunday of the Tue..Mon NFL week cycle containing the game date."""

    weekday = game_day.weekday()  # Monday=0 .. Sunday=6
    if weekday == 0:
        return game_day - timedelta(days=1)
    return game_day + timedelta(days=6 - weekday)


def _region_count(regions: str) -> int:
    count = len([region for region in regions.split(",") if region.strip()])
    if count == 0:
        raise ValueError("At least one region is required")
    return count


def plan_backfill(
    schedule: pd.DataFrame,
    start_season: int,
    end_season: int,
    *,
    regions: str = "us",
    weeks: Iterable[int] | None = None,
    labels: Iterable[str] | None = None,
    archive_start: datetime = HISTORICAL_ARCHIVE_START_UTC,
) -> list[BackfillTarget]:
    """Derive weekly decision timestamps from schedule rows (season, week, gameday).

    Each scheduled week is anchored on the Sunday of its Tue..Mon cycle (the
    most common cycle Sunday among the week's game dates, so COVID-era Tuesday
    and Wednesday reschedules do not shift the anchor). Timestamps earlier than
    the provider archive start are dropped.
    """

    if end_season < start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    required = {"season", "week", "gameday"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise ValueError(f"Backfill planning is missing columns: {', '.join(missing)}")
    unknown = sorted(set(labels or ()).difference(DECISION_LABELS))
    if unknown:
        raise ValueError(f"Unknown decision labels: {', '.join(unknown)}")
    label_filter = set(labels) if labels is not None else None
    week_filter = {int(week) for week in weeks} if weeks is not None else None
    frame = schedule.loc[:, ["season", "week", "gameday"]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.loc[frame["season"].between(start_season, end_season)].dropna(subset=["gameday"])
    if week_filter is not None:
        frame = frame.loc[frame["week"].isin(week_filter)]
    region_count = _region_count(regions)
    targets: list[BackfillTarget] = []
    for (season, week), group in frame.groupby(["season", "week"], sort=True):
        cycle_sundays = Counter(
            _week_cycle_sunday(game_day) for game_day in group["gameday"].dt.date
        )
        anchor = min(cycle_sundays.items(), key=lambda item: (-item[1], item[0]))[0]
        for label, day_offset, local_time, with_h2h in DECISION_TIMES:
            if label_filter is not None and label not in label_filter:
                continue
            local = datetime.combine(
                anchor + timedelta(days=day_offset), local_time, tzinfo=DECISION_TIMEZONE
            )
            requested = local.astimezone(UTC)
            if requested < archive_start:
                continue
            markets = (*BASE_MARKETS, H2H_MARKET) if with_h2h else BASE_MARKETS
            targets.append(
                BackfillTarget(
                    season=int(cast(int, season)),
                    week=int(cast(int, week)),
                    label=label,
                    requested_at_utc=requested,
                    markets=",".join(markets),
                    regions=regions,
                    credits=HISTORICAL_CREDITS_PER_MARKET_REGION * len(markets) * region_count,
                )
            )
    return sorted(targets, key=lambda target: (target.requested_at_utc, target.label))


def summarize_backfill_plan(targets: Sequence[BackfillTarget]) -> dict[str, Any]:
    """Dry-run summary: exact call count and credit cost, broken down by season."""

    seasons: dict[str, dict[str, int]] = {}
    for target in targets:
        row = seasons.setdefault(str(target.season), {"weeks": 0, "calls": 0, "credits": 0})
        row["calls"] += 1
        row["credits"] += target.credits
    for season_key, row in seasons.items():
        row["weeks"] = len({target.week for target in targets if str(target.season) == season_key})
    return {
        "total_calls": len(targets),
        "total_credits": sum(target.credits for target in targets),
        "seasons": dict(sorted(seasons.items())),
        "first_requested_at_utc": targets[0].requested_at_iso if targets else None,
        "last_requested_at_utc": targets[-1].requested_at_iso if targets else None,
    }


def fetch_historical_odds_api(
    *,
    api_key: str,
    snapshot_at: datetime,
    regions: str = "us",
    markets: str = "spreads,totals",
    bookmakers: str | None = None,
    timeout: int = 30,
) -> tuple[bytes, dict[str, str]]:
    """Request the historical odds snapshot at or immediately before ``snapshot_at``."""

    if not api_key.strip():
        raise ValueError("The Odds API key is empty")
    parameters = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": snapshot_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if bookmakers:
        parameters["bookmakers"] = bookmakers
    url = ODDS_API_HISTORICAL_URL + "?" + urllib.parse.urlencode(parameters)
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
        raise ValueError(f"The Odds API returned HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise ValueError("Unable to reach The Odds API") from error
    return payload, quota


def parse_historical_odds_response(payload: bytes) -> HistoricalOddsCapture:
    """Unwrap the historical envelope and normalize its event data into quotes.

    The quotes carry the API's snapshot ``timestamp`` as ``observed_at_utc``,
    the SHA-256 of the full raw historical payload, and a ``capture_kind``
    column marking every row as ``historical_backfill``.
    """

    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("The Odds API historical response must be a JSON object")
    snapshot_at = _parse_iso_utc(decoded.get("timestamp"))
    if snapshot_at is None:
        raise ValueError("The Odds API historical response is missing its snapshot timestamp")
    events = decoded.get("data")
    if not isinstance(events, list):
        raise ValueError("The Odds API historical response must carry a data array")
    quotes = parse_odds_api_response(
        json.dumps(events, separators=(",", ":")).encode(), observed_at=snapshot_at
    )
    quotes["raw_response_sha256"] = hashlib.sha256(payload).hexdigest()
    quotes["capture_kind"] = HISTORICAL_CAPTURE_KIND
    return HistoricalOddsCapture(
        snapshot_at_utc=snapshot_at,
        previous_snapshot_at_utc=_parse_iso_utc(decoded.get("previous_timestamp")),
        next_snapshot_at_utc=_parse_iso_utc(decoded.get("next_timestamp")),
        quotes=quotes,
    )


def store_historical_snapshot(
    payload: bytes,
    capture: HistoricalOddsCapture,
    root: Path,
    *,
    target: BackfillTarget,
    quota: dict[str, str] | None = None,
) -> MarketSnapshot:
    """Write one immutable historical snapshot into the shared market store."""

    previous_at = capture.previous_snapshot_at_utc
    next_at = capture.next_snapshot_at_utc
    return write_market_snapshot(
        payload,
        capture.quotes,
        root,
        observed_at=capture.snapshot_at_utc,
        request_metadata={
            "sport": ODDS_API_SPORT,
            "endpoint": "historical",
            "regions": target.regions,
            "markets": target.markets,
            "odds_format": "american",
            "season": target.season,
            "week": target.week,
            "decision_label": target.label,
        },
        quota=quota,
        extra_manifest={
            "capture_kind": HISTORICAL_CAPTURE_KIND,
            "requested_at_utc": target.requested_at_iso,
            "snapshot_timestamp_utc": capture.snapshot_at_utc.isoformat(),
            "previous_snapshot_timestamp_utc": previous_at.isoformat() if previous_at else None,
            "next_snapshot_timestamp_utc": next_at.isoformat() if next_at else None,
        },
    )


def completed_backfill_requests(root: Path) -> set[str]:
    """Requested-at timestamps of historical snapshots already in the store."""

    completed: set[str] = set()
    if not root.is_dir():
        return completed
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("capture_kind") == HISTORICAL_CAPTURE_KIND:
            requested = manifest.get("requested_at_utc")
            if requested:
                completed.add(str(requested))
    return completed


def _parse_remaining(quota: dict[str, str]) -> float | None:
    raw = quota.get("requests_remaining", "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _stderr_log(message: str) -> None:
    print(message, file=sys.stderr)


def execute_backfill(
    targets: Sequence[BackfillTarget],
    root: Path,
    schedule: pd.DataFrame,
    *,
    api_key: str,
    budget: int | None = None,
    quota_floor: int = DEFAULT_QUOTA_FLOOR,
    resume: bool = False,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    fetch: Callable[..., tuple[bytes, dict[str, str]]] | None = None,
    sleeper: Callable[[float], None] | None = None,
    log: Callable[[str], None] = _stderr_log,
) -> dict[str, Any]:
    """Fetch and store planned snapshots with budget, quota-floor, and resume guards.

    Already-present targets are skipped with ``resume=True`` (without it any
    overlap aborts before spending credits). The run stops cleanly before any
    call that would drop the provider's remaining credits below ``quota_floor``.
    """

    fetch_snapshot = fetch or fetch_historical_odds_api
    pause = sleeper or sleep
    completed = completed_backfill_requests(root)
    overlap = [target for target in targets if target.requested_at_iso in completed]
    if overlap and not resume:
        raise ValueError(
            f"{len(overlap)} planned snapshots already exist in {root}; "
            "pass --resume to skip them and continue"
        )
    remaining_targets = [target for target in targets if target.requested_at_iso not in completed]
    planned_credits = sum(target.credits for target in remaining_targets)
    if budget is not None and planned_credits > budget:
        raise ValueError(
            f"Planned backfill cost of {planned_credits} credits exceeds the "
            f"budget of {budget}; narrow the plan or raise --budget"
        )
    stored: list[str] = []
    collisions: list[str] = []
    credits_spent = 0
    requests_remaining: float | None = None
    stopped_reason: str | None = None
    for index, target in enumerate(remaining_targets):
        if budget is not None and credits_spent + target.credits > budget:
            stopped_reason = "budget"
            break
        if requests_remaining is not None and requests_remaining - target.credits < quota_floor:
            stopped_reason = "quota_floor"
            log(
                f"[odds-backfill] stopping: {requests_remaining:.0f} credits remaining and the "
                f"next call costs {target.credits}, which would breach the floor of {quota_floor}"
            )
            break
        if index > 0 and sleep_seconds > 0:
            pause(sleep_seconds)
        payload, quota = fetch_snapshot(
            api_key=api_key,
            snapshot_at=target.requested_at_utc,
            regions=target.regions,
            markets=target.markets,
        )
        credits_spent += target.credits
        requests_remaining = _parse_remaining(quota)
        capture = parse_historical_odds_response(payload)
        matched = attach_nflverse_game_ids(capture.quotes, schedule)
        capture = HistoricalOddsCapture(
            snapshot_at_utc=capture.snapshot_at_utc,
            previous_snapshot_at_utc=capture.previous_snapshot_at_utc,
            next_snapshot_at_utc=capture.next_snapshot_at_utc,
            quotes=matched,
        )
        try:
            snapshot = store_historical_snapshot(payload, capture, root, target=target, quota=quota)
        except ValueError as error:
            if "already exists" not in str(error):
                raise
            collisions.append(target.requested_at_iso)
            log(
                f"[odds-backfill] snapshot for {target.requested_at_iso} already stored "
                f"({capture.snapshot_at_utc.isoformat()}); continuing"
            )
            continue
        stored.append(snapshot.snapshot_id)
        remaining_text = "unknown" if requests_remaining is None else f"{requests_remaining:.0f}"
        log(
            f"[odds-backfill] {index + 1}/{len(remaining_targets)} season {target.season} "
            f"week {target.week} {target.label} requested {target.requested_at_iso} -> "
            f"snapshot {capture.snapshot_at_utc.isoformat()} ({target.markets}; "
            f"{target.credits} credits; remaining {remaining_text})"
        )
    return {
        "planned_calls": len(targets),
        "skipped_existing": len(targets) - len(remaining_targets),
        "attempted_calls": len(stored) + len(collisions),
        "stored_snapshots": stored,
        "snapshot_collisions": collisions,
        "credits_spent": credits_spent,
        "requests_remaining": requests_remaining,
        "quota_floor": quota_floor,
        "stopped_reason": stopped_reason,
        "completed": stopped_reason is None,
    }
