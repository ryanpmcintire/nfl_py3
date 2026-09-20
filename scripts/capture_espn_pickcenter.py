from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.market_data import (
    NFL_TEAM_NAMES,
    QUOTE_COLUMNS,
    attach_nflverse_game_ids,
    write_market_snapshot,
)
from nfl_ats.source_policy import require_acquisition, require_private_raw_destination

REPO = Path(__file__).resolve().parents[1]
SOURCE = "espn_scoreboard_pickcenter"
ROOT = REPO / "data" / "market" / "raw"
BLOCKED = REPO / "data" / "market" / "espn_pickcenter_blocked.json"
BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
USER_AGENT = "nfl-ats-research/0.2 (research; github.com/ryanpmcintire/nfl_py3)"


def _fetch(url: str, *, prior_at: float | None) -> tuple[dict[str, Any], float]:
    if prior_at is not None:
        time.sleep(max(0.0, 2.0 - (time.monotonic() - prior_at)))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            fetched_at = time.monotonic()
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code in (403, 429) or error.code >= 500:
            BLOCKED.parent.mkdir(parents=True, exist_ok=True)
            BLOCKED.write_text(
                json.dumps(
                    {
                        "source": SOURCE,
                        "blocked_at_utc": datetime.now(UTC).isoformat(),
                        "http_status": error.code,
                        "url": url,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise
    if not isinstance(payload, dict):
        raise ValueError(f"ESPN response is not a JSON object: {url}")
    return payload, fetched_at


def _price(value: Any) -> float | None:
    try:
        return float(str(value).replace("EVEN", "+100"))
    except (TypeError, ValueError):
        return None


def _rows(
    event: dict[str, Any], summary: dict[str, Any], observed: datetime, digest: str
) -> list[dict[str, Any]]:
    competition = event.get("competitions", [{}])[0]
    teams = {
        entry.get("homeAway"): entry.get("team", {}) for entry in competition.get("competitors", [])
    }
    home = teams.get("home", {})
    away = teams.get("away", {})
    home_name = str(home.get("displayName", ""))
    away_name = str(away.get("displayName", ""))
    if not home_name or not away_name:
        return []
    odds = summary.get("pickcenter") or []
    if not isinstance(odds, list):
        return []
    rows: list[dict[str, Any]] = []
    for offer in odds:
        provider = offer.get("provider") or {}
        provider_name = str(provider.get("name", ""))
        if "draftkings" not in provider_name.lower().replace(" ", ""):
            continue
        details = str(offer.get("details", ""))
        parts = details.split()
        if len(parts) != 2 or not parts[1].startswith("-"):
            continue
        try:
            favorite_margin = -float(parts[1])
            total = float(offer["overUnder"])
        except (TypeError, ValueError, KeyError):
            continue
        if favorite_margin <= 0 or total <= 0:
            continue
        favorite = parts[0].upper()
        home_abbr = str(home.get("abbreviation", "")).upper()
        away_abbr = str(away.get("abbreviation", "")).upper()
        if favorite == home_abbr:
            home_margin_line = favorite_margin
        elif favorite == away_abbr:
            home_margin_line = -favorite_margin
        else:
            continue
        home_odds = offer.get("homeTeamOdds") or {}
        away_odds = offer.get("awayTeamOdds") or {}
        shared = {
            "observed_at_utc": observed,
            "provider": SOURCE,
            "provider_event_id": str(event["id"]),
            "sport_key": "americanfootball_nfl",
            "commence_time_utc": event.get("date"),
            "home_team_name": home_name,
            "away_team_name": away_name,
            "home_team": NFL_TEAM_NAMES.get(home_name),
            "away_team": NFL_TEAM_NAMES.get(away_name),
            "nflverse_game_id": pd.NA,
            "bookmaker_key": "draftkings",
            "bookmaker_title": "DraftKings",
            "bookmaker_last_update_utc": pd.NaT,
            "market_last_update_utc": pd.NaT,
            "home_spread_line": home_margin_line,
            "raw_response_sha256": digest,
        }
        for side, name, line, price in (
            ("HOME", home_name, -home_margin_line, home_odds.get("spreadOdds")),
            ("AWAY", away_name, home_margin_line, away_odds.get("spreadOdds")),
            ("OVER", "Over", total, offer.get("overOdds")),
            ("UNDER", "Under", total, offer.get("underOdds")),
        ):
            rows.append(
                {
                    **shared,
                    "market": "spreads" if side in ("HOME", "AWAY") else "totals",
                    "outcome_name": name,
                    "outcome_side": side,
                    "line": line,
                    "price": _price(price),
                }
            )
    return rows


def capture(features_path: Path) -> dict[str, Any]:
    require_acquisition(SOURCE)
    require_private_raw_destination(SOURCE, ROOT)
    if BLOCKED.exists():
        return {
            "captured": False,
            "source": SOURCE,
            "blocked": json.loads(BLOCKED.read_text(encoding="utf-8")),
        }
    features = pd.read_parquet(features_path)
    observed = datetime.now(UTC)
    future = features.loc[pd.to_datetime(features["kickoff"], utc=True).gt(observed)]
    if future.empty:
        return {"captured": False, "source": SOURCE, "reason": "no_future_games"}
    week = future.sort_values("kickoff").iloc[0]
    season = int(week["season"])
    week_number = int(week["week"])
    scoreboard_url = f"{BASE}/scoreboard?dates={season}&seasontype=2&week={week_number}&limit=100"
    scoreboard, prior_at = _fetch(scoreboard_url, prior_at=None)
    events = [
        event
        for event in scoreboard.get("events", [])
        if pd.to_datetime(event.get("date"), utc=True, errors="coerce") > pd.Timestamp(observed)
    ]
    responses: dict[str, Any] = {"scoreboard": scoreboard, "summaries": {}}
    for event in events:
        event_id = str(event["id"])
        url = f"{BASE}/summary?event={event_id}"
        responses["summaries"][event_id], prior_at = _fetch(url, prior_at=prior_at)
    payload = json.dumps(responses, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    rows = [
        row
        for event in events
        for row in _rows(event, responses["summaries"][str(event["id"])], observed, digest)
    ]
    quotes = pd.DataFrame(rows, columns=QUOTE_COLUMNS)
    if quotes.empty:
        raise ValueError("No pregame DraftKings spreads and totals in ESPN pickcenter")
    for column in (
        "observed_at_utc",
        "commence_time_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
    ):
        quotes[column] = pd.to_datetime(quotes[column], utc=True)
    quotes = attach_nflverse_game_ids(quotes, features)
    snapshot = write_market_snapshot(
        payload,
        quotes,
        ROOT,
        observed_at=observed,
        request_metadata={
            "source": SOURCE,
            "scoreboard_url": scoreboard_url,
            "event_count": len(events),
        },
        extra_manifest={
            "quote_timestamp_semantics": "book_update_unknown_capture_time_observed",
            "book_scope": "single_book_draftkings",
        },
        provider=SOURCE,
    )
    return {
        "captured": True,
        "source": SOURCE,
        "bookmaker": "DraftKings",
        "observed_at_utc": observed.isoformat(),
        "snapshot": str(snapshot.root),
        "events_requested": len(events),
        "events_with_quotes": int(quotes["provider_event_id"].nunique()),
        "matched_events": int(quotes["nflverse_game_id"].nunique()),
        "quotes": len(quotes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture live single-book ESPN DraftKings NFL spreads and totals"
    )
    parser.add_argument(
        "--features", type=Path, default=REPO / "data" / "processed" / "game_features.parquet"
    )
    args = parser.parse_args()
    print(json.dumps(capture(args.features), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
