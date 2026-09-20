from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.market_data import NFL_TEAM_NAMES, QUOTE_COLUMNS, write_market_snapshot
from nfl_ats.source_policy import require_acquisition, require_private_raw_destination

REPO = Path(__file__).resolve().parents[1]
SOURCE = "the_odds_gap_lineshop_private"
ROOT = REPO / "data" / "market" / "raw"
URL = "https://theoddsgap.com/api/lineshop"
LEADERS = ("bovada", "williamhill_us", "mybookieag")


def _rows(
    payload: bytes, features: pd.DataFrame, retrieved: datetime
) -> tuple[pd.DataFrame, pd.Timestamp]:
    decoded = json.loads(payload)
    scanned = pd.to_datetime(decoded.get("last_updated"), utc=True, errors="coerce")
    if pd.isna(scanned) or scanned > pd.Timestamp(retrieved):
        raise ValueError("The Odds Gap source scan timestamp is missing or in the future")
    if pd.Timestamp(retrieved) - scanned > pd.Timedelta(minutes=70):
        raise ValueError("The Odds Gap source scan is more than 70 minutes old")
    digest = hashlib.sha256(payload).hexdigest()
    schedule = features.copy()
    schedule["kickoff"] = pd.to_datetime(schedule["kickoff"], utc=True)
    schedule = schedule.loc[schedule["kickoff"].gt(scanned)]
    current = schedule.loc[schedule["kickoff"].lt(scanned + pd.Timedelta(days=4))]
    if current.empty:
        raise ValueError("No upcoming current-week games in local schedule")
    week = current.sort_values("kickoff").iloc[0]
    current = current.loc[current["season"].eq(week["season"]) & current["week"].eq(week["week"])]
    ids = current.set_index(["home_team", "away_team"])
    rows: list[dict[str, Any]] = []
    for event in decoded.get("games", []):
        if event.get("sport") != "americanfootball_nfl":
            continue
        home_name = str(event.get("home", ""))
        away_name = str(event.get("away", ""))
        pair = (NFL_TEAM_NAMES.get(home_name), NFL_TEAM_NAMES.get(away_name))
        if pair not in ids.index:
            continue
        game = ids.loc[pair]
        if isinstance(game, pd.DataFrame):
            raise ValueError("Ambiguous scheduled matchup")
        books = (event.get("spread_data") or {}).get("all_books") or {}
        for book in LEADERS:
            quote = books.get(book)
            if not isinstance(quote, dict):
                continue
            try:
                home_line = float(quote["home_line"])
                away_line = float(quote["away_line"])
                home_price = float(quote["home_juice"])
                away_price = float(quote["away_juice"])
            except (KeyError, TypeError, ValueError):
                continue
            if home_line != -away_line:
                raise ValueError("Inconsistent The Odds Gap spread sides")
            shared = {
                "observed_at_utc": retrieved,
                "provider": SOURCE,
                "provider_event_id": str(event.get("id") or game["game_id"]),
                "sport_key": "americanfootball_nfl",
                "commence_time_utc": game["kickoff"],
                "home_team_name": home_name,
                "away_team_name": away_name,
                "home_team": pair[0],
                "away_team": pair[1],
                "nflverse_game_id": game["game_id"],
                "bookmaker_key": book,
                "bookmaker_title": str(quote.get("label") or book),
                "bookmaker_last_update_utc": pd.NaT,
                "quote_timestamp_basis": "capture_observed_utc",
                "source_scan_at_utc": scanned,
                "market": "spreads",
                "market_last_update_utc": pd.NaT,
                "home_spread_line": -home_line,
                "raw_response_sha256": digest,
            }
            rows.extend(
                [
                    {
                        **shared,
                        "outcome_name": home_name,
                        "outcome_side": "HOME",
                        "line": home_line,
                        "price": home_price,
                    },
                    {
                        **shared,
                        "outcome_name": away_name,
                        "outcome_side": "AWAY",
                        "line": away_line,
                        "price": away_price,
                    },
                ]
            )
    quotes = pd.DataFrame(rows, columns=QUOTE_COLUMNS)
    if quotes.empty:
        raise ValueError("No current-week leader spreads in The Odds Gap response")
    for column in (
        "observed_at_utc",
        "commence_time_utc",
        "bookmaker_last_update_utc",
        "source_scan_at_utc",
        "market_last_update_utc",
    ):
        quotes[column] = pd.to_datetime(quotes[column], utc=True)
    return quotes, scanned


def capture(features_path: Path) -> dict[str, Any]:
    require_acquisition(SOURCE)
    require_private_raw_destination(SOURCE, ROOT)
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            retrieved = datetime.now(UTC)
    except urllib.error.HTTPError as error:
        raise ValueError(f"The Odds Gap documented endpoint returned HTTP {error.code}") from error
    features = pd.read_parquet(features_path)
    quotes, scanned = _rows(payload, features, retrieved)
    coverage = (
        quotes.loc[quotes["outcome_side"].eq("HOME")]
        .groupby("bookmaker_key")["nflverse_game_id"]
        .nunique()
        .to_dict()
    )
    snapshot = write_market_snapshot(
        payload,
        quotes,
        ROOT,
        observed_at=retrieved,
        snapshot_suffix="-odds-gap-private",
        request_metadata={
            "source": SOURCE,
            "url": URL,
            "season": int(
                features.loc[features["game_id"].isin(quotes["nflverse_game_id"]), "season"].iloc[0]
            ),
            "week": int(
                features.loc[features["game_id"].isin(quotes["nflverse_game_id"]), "week"].iloc[0]
            ),
        },
        extra_manifest={
            "capture_kind": "live",
            "book_scope": "source_scanned_leader_books",
            "source_scan_at_utc": scanned.isoformat(),
            "retrieved_at_utc": retrieved.isoformat(),
            "quote_timestamp_semantics": (
                "retrieved_observed_source_scan_separate_book_update_unknown"
            ),
            "publication_scope": "private_research_only",
            "attribution": "https://theoddsgap.com",
        },
        provider=SOURCE,
    )
    return {
        "captured": True,
        "source": SOURCE,
        "source_scan_at_utc": scanned.isoformat(),
        "retrieved_at_utc": retrieved.isoformat(),
        "snapshot": str(snapshot.root),
        "games": int(quotes["nflverse_game_id"].nunique()),
        "book_games": coverage,
        "quotes": len(quotes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one private The Odds Gap leader spread snapshot"
    )
    parser.add_argument(
        "--features", type=Path, default=REPO / "data" / "processed" / "game_features.parquet"
    )
    args = parser.parse_args()
    print(json.dumps(capture(args.features), sort_keys=True))


if __name__ == "__main__":
    main()
