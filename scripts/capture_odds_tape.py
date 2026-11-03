from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capture_bovada_private import REPO, SOURCE, URL, _price

from nfl_ats.market_data import NFL_TEAM_NAMES
from nfl_ats.source_policy import require_acquisition, require_private_raw_destination

MEASURED_PROVIDER_LAG_SECONDS = 17.0
SAMPLES_PER_LAG = 2
WINDOW_MINUTES = 60.0
REQUEST_TIMEOUT_SECONDS = 30
BLOCK_MARKER = REPO / "data" / "market" / f"{SOURCE}_blocked.json"
DEFAULT_INTERVAL_SECONDS = MEASURED_PROVIDER_LAG_SECONDS / SAMPLES_PER_LAG


class TapeStop(Exception):
    pass


def _fetch(url: str = URL) -> tuple[bytes, datetime, datetime]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    sent = datetime.now(UTC)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read()
            received = datetime.now(UTC)
    except urllib.error.HTTPError as error:
        if error.code in (403, 429) or error.code >= 500:
            BLOCK_MARKER.parent.mkdir(parents=True, exist_ok=True)
            BLOCK_MARKER.write_text(
                json.dumps(
                    {
                        "source": SOURCE,
                        "blocked_at_utc": datetime.now(UTC).isoformat(),
                        "reason": f"Bovada public endpoint returned HTTP {error.code}",
                    }
                ),
                encoding="utf-8",
            )
        raise TapeStop(f"Bovada public endpoint returned HTTP {error.code}") from error
    except (urllib.error.URLError, OSError) as error:
        raise TapeStop(f"Bovada request failed: {error}") from error
    return payload, sent, received


def _iso_ms(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(value) / 1000, UTC).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _lines(payload: bytes, sent: datetime, received: datetime, poll: int) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise TapeStop("Bovada response was not JSON") from error
    if not isinstance(decoded, list):
        raise TapeStop(f"Bovada response must be an array, got {type(decoded).__name__}")
    rows: list[dict[str, Any]] = []
    for coupon in decoded:
        for event in coupon.get("events", []) if isinstance(coupon, dict) else []:
            if not isinstance(event, dict) or event.get("status") != "U" or event.get("live"):
                continue
            names = {
                bool(team.get("home")): str(team.get("name", ""))
                for team in event.get("competitors", [])
                if isinstance(team, dict)
            }
            home, away = names.get(True, ""), names.get(False, "")
            if home not in NFL_TEAM_NAMES or away not in NFL_TEAM_NAMES:
                continue
            for group in event.get("displayGroups", []):
                for market in group.get("markets", []):
                    period = market.get("period") or {}
                    if (
                        market.get("description") != "Point Spread"
                        or market.get("status") != "O"
                        or not period.get("main")
                        or period.get("description") != "Game"
                    ):
                        continue
                    sides: dict[str, Any] = {}
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price") or {}
                        try:
                            handicap = float(price["handicap"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        sides[str(outcome.get("type"))] = {
                            "line": handicap,
                            "price": _price(price.get("american")),
                        }
                    if set(sides) != {"H", "A"}:
                        continue
                    rows.append(
                        {
                            "poll": poll,
                            "request_sent_utc": sent.isoformat(),
                            "response_received_utc": received.isoformat(),
                            "provider_event_id": str(event["id"]),
                            "provider_last_updated_utc": _iso_ms(event.get("lastModified")),
                            "provider_last_updated_ms": event.get("lastModified"),
                            "start_time_utc": _iso_ms(event.get("startTime")),
                            "home_team": NFL_TEAM_NAMES[home],
                            "away_team": NFL_TEAM_NAMES[away],
                            "market_id": str(market.get("id")),
                            "home_spread_line": sides["H"]["line"],
                            "home_spread_price": sides["H"]["price"],
                            "away_spread_line": sides["A"]["line"],
                            "away_spread_price": sides["A"]["price"],
                        }
                    )
    return rows


def run(interval: float, duration_minutes: float, out_dir: Path) -> dict[str, Any]:
    require_acquisition(SOURCE)
    require_private_raw_destination(SOURCE, REPO / "data" / "market" / "raw")
    started = datetime.now(UTC)
    run_dir = out_dir / started.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    tape = run_dir / "tape.jsonl"
    manifest: dict[str, Any] = {
        "source": SOURCE,
        "url": URL,
        "started_utc": started.isoformat(),
        "interval_seconds": interval,
        "duration_minutes": duration_minutes,
        "provider_timestamp_field": "event.lastModified",
        "status": "running",
        "polls": 0,
        "rows": 0,
    }
    deadline = time.monotonic() + duration_minutes * 60
    next_poll = time.monotonic()
    error: str | None = None
    try:
        while True:
            payload, sent, received = _fetch()
            rows = _lines(payload, sent, received, manifest["polls"])
            with tape.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            manifest["polls"] += 1
            manifest["rows"] += len(rows)
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
    return {**manifest, "tape": str(tape)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a sub-minute Bovada NFL spread tape")
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--duration-minutes", type=float, default=WINDOW_MINUTES)
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "raw" / "odds_tape")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.duration_minutes <= 0:
        parser.error("interval-seconds and duration-minutes must be positive")
    if BLOCK_MARKER.is_file():
        print(json.dumps({"captured": False, "reason": "source_http_block"}))
        sys.exit(2)
    result = run(args.interval_seconds, args.duration_minutes, args.out_dir)
    print(json.dumps(result, sort_keys=True))
    sys.exit(1 if result["status"] != "complete" else 0)


if __name__ == "__main__":
    main()
