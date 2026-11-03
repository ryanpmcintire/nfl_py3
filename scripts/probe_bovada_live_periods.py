from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capture_bovada_private import REPO
from capture_odds_tape import BLOCK_MARKER, SOURCE, URL, TapeStop, _fetch, _iso_ms

from nfl_ats.market_data import NFL_TEAM_NAMES
from nfl_ats.source_policy import require_acquisition, require_private_raw_destination

LIST_URL = URL.replace("&preMatchOnly=true", "")
EVENT_URL_BASE = "https://www.bovada.lv/services/sports/event/coupon/events/A/description"
WINDOW_START_MINUTES_AFTER_KICKOFF = 73.0
WINDOW_END_MINUTES_AFTER_KICKOFF = 136.0
HALFTIME_P05_MINUTES = 13.8
SAMPLES_PER_HALFTIME = 2
DEFAULT_INTERVAL_SECONDS = HALFTIME_P05_MINUTES * 60 / SAMPLES_PER_HALFTIME
SECOND_HALF_TOKENS = ("2nd half", "second half", "2h")


def _events(decoded: Any) -> list[dict[str, Any]]:
    coupons = decoded if isinstance(decoded, list) else [decoded]
    found: list[dict[str, Any]] = []
    for coupon in coupons:
        if isinstance(coupon, dict):
            found.extend(e for e in coupon.get("events", []) if isinstance(e, dict))
    return found


def _teams(event: dict[str, Any]) -> tuple[str, str] | None:
    names = {
        bool(t.get("home")): str(t.get("name", ""))
        for t in event.get("competitors", [])
        if isinstance(t, dict)
    }
    home, away = names.get(True, ""), names.get(False, "")
    if home in NFL_TEAM_NAMES and away in NFL_TEAM_NAMES:
        return NFL_TEAM_NAMES[home], NFL_TEAM_NAMES[away]
    return None


def _in_window(event: dict[str, Any], now: datetime, pregame_hours: float) -> bool:
    if event.get("live"):
        return True
    try:
        start = datetime.fromtimestamp(int(event["startTime"]) / 1000, UTC)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return False
    since = (now - start).total_seconds() / 60
    if WINDOW_START_MINUTES_AFTER_KICKOFF <= since <= WINDOW_END_MINUTES_AFTER_KICKOFF:
        return True
    return -pregame_hours * 60 <= since < 0


def _markets(detail: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        market
        for group in detail.get("displayGroups", [])
        for market in group.get("markets", [])
        if isinstance(market, dict)
    ]


def _periods(markets: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for market in markets:
        name = str((market.get("period") or {}).get("description"))
        counts[name] = counts.get(name, 0) + 1
    return counts


def _second_half_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for market in markets:
        period = str((market.get("period") or {}).get("description", "")).lower()
        text = f"{period} {str(market.get('description', '')).lower()}"
        if any(token in text for token in SECOND_HALF_TOKENS):
            found.append(
                {
                    "market_id": str(market.get("id")),
                    "description": market.get("description"),
                    "period": (market.get("period") or {}).get("description"),
                    "status": market.get("status"),
                    "outcomes": [
                        {
                            "type": o.get("type"),
                            "description": o.get("description"),
                            "handicap": (o.get("price") or {}).get("handicap"),
                            "american": (o.get("price") or {}).get("american"),
                        }
                        for o in market.get("outcomes", [])
                    ],
                }
            )
    return found


def _poll(poll: int, pregame_hours: float, handle: Any) -> tuple[int, int]:
    payload, sent, received = _fetch(LIST_URL)
    try:
        listing = _events(json.loads(payload))
    except json.JSONDecodeError as error:
        raise TapeStop("Bovada list response was not JSON") from error
    rows = hits = 0
    for event in listing:
        teams = _teams(event)
        if teams is None or not _in_window(event, received, pregame_hours):
            continue
        link = event.get("link")
        if not link:
            continue
        detail_payload, detail_sent, detail_received = _fetch(f"{EVENT_URL_BASE}{link}?lang=en")
        try:
            decoded = json.loads(detail_payload)
        except json.JSONDecodeError as error:
            raise TapeStop("Bovada event response was not JSON") from error
        detail = next(
            (e for e in _events(decoded) if str(e.get("id")) == str(event.get("id"))), None
        )
        markets = _markets(detail) if detail else []
        second_half = _second_half_markets(markets)
        row = {
            "poll": poll,
            "request_sent_utc": detail_sent.isoformat(),
            "response_received_utc": detail_received.isoformat(),
            "list_request_sent_utc": sent.isoformat(),
            "provider_event_id": str(event.get("id")),
            "provider_last_updated_utc": _iso_ms((detail or event).get("lastModified")),
            "start_time_utc": _iso_ms(event.get("startTime")),
            "live": bool(event.get("live")),
            "status": event.get("status"),
            "home_team": teams[0],
            "away_team": teams[1],
            "detail_found": detail is not None,
            "periods": _periods(markets),
            "second_half_markets": second_half,
        }
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        rows += 1
        hits += len(second_half)
    return rows, hits


def run(
    interval: float, repeat_minutes: float, pregame_hours: float, out_dir: Path
) -> dict[str, Any]:
    require_acquisition(SOURCE)
    require_private_raw_destination(SOURCE, REPO / "data" / "market" / "raw")
    started = datetime.now(UTC)
    run_dir = out_dir / started.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "source": SOURCE,
        "list_url": LIST_URL,
        "event_url_base": EVENT_URL_BASE,
        "started_utc": started.isoformat(),
        "interval_seconds": interval,
        "repeat_minutes": repeat_minutes,
        "pregame_hours": pregame_hours,
        "window_minutes_after_kickoff": [
            WINDOW_START_MINUTES_AFTER_KICKOFF,
            WINDOW_END_MINUTES_AFTER_KICKOFF,
        ],
        "status": "running",
        "polls": 0,
        "rows": 0,
        "second_half_markets_seen": 0,
    }
    deadline = time.monotonic() + repeat_minutes * 60
    next_poll = time.monotonic()
    error: str | None = None
    try:
        with (run_dir / "probe.jsonl").open("a", encoding="utf-8") as handle:
            while True:
                rows, hits = _poll(manifest["polls"], pregame_hours, handle)
                handle.flush()
                manifest["polls"] += 1
                manifest["rows"] += rows
                manifest["second_half_markets_seen"] += hits
                next_poll += interval
                if next_poll >= deadline:
                    break
                time.sleep(max(0.0, next_poll - time.monotonic()))
    except TapeStop as stop:
        error = str(stop)
    manifest["status"] = "stopped_on_error" if error else "complete"
    manifest["error"] = error
    manifest["ended_utc"] = datetime.now(UTC).isoformat()
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {**manifest, "probe": str(run_dir / "probe.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Bovada per-event periods for a live second-half market"
    )
    parser.add_argument("--repeat-minutes", type=float, required=True)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--pregame-hours", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "raw" / "odds_2h_probe")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.repeat_minutes <= 0 or args.pregame_hours < 0:
        parser.error("interval-seconds and repeat-minutes must be positive")
    if BLOCK_MARKER.is_file():
        print(json.dumps({"captured": False, "reason": "source_http_block"}))
        sys.exit(2)
    result = run(args.interval_seconds, args.repeat_minutes, args.pregame_hours, args.out_dir)
    print(json.dumps(result, sort_keys=True))
    sys.exit(1 if result["status"] != "complete" else 0)


if __name__ == "__main__":
    main()
