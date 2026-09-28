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

from nfl_ats.market_data import (
    NFL_TEAM_NAMES,
    QUOTE_COLUMNS,
    attach_nflverse_game_ids,
    write_market_snapshot,
)
from nfl_ats.source_policy import require_acquisition, require_private_raw_destination

REPO = Path(__file__).resolve().parents[1]
SOURCE = "bovada_public_nfl"
ROOT = REPO / "data" / "market" / "raw"
URL = "https://www.bovada.lv/services/sports/event/v2/events/A/description/football/nfl?marketFilterId=def&preMatchOnly=true&lang=en"


def _price(value: Any) -> float | None:
    if value == "EVEN":
        return 100.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snippet(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="replace")
    return text[:200]


def _rows(payload: bytes, observed: datetime) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"Bovada response was not JSON: {_snippet(payload)!r}") from error
    if isinstance(decoded, dict) and isinstance(decoded.get("events"), list):
        decoded = [decoded]
    if not isinstance(decoded, list):
        raise ValueError(
            f"Bovada response must be an array, got {type(decoded).__name__}: {_snippet(payload)!r}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    rows: list[dict[str, Any]] = []
    for coupon in decoded:
        if not isinstance(coupon, dict) or not isinstance(coupon.get("events", []), list):
            raise ValueError(f"Bovada response has an invalid coupon: {_snippet(payload)!r}")
        for event in coupon.get("events", []):
            if not isinstance(event, dict) or event.get("id") is None:
                raise ValueError(f"Bovada response has an invalid event: {_snippet(payload)!r}")
            if event.get("status") != "U" or event.get("live"):
                continue
            start = pd.to_datetime(event.get("startTime"), utc=True, unit="ms", errors="coerce")
            if pd.isna(start) or start <= pd.Timestamp(observed):
                continue
            competitors = event.get("competitors", [])
            if not isinstance(competitors, list) or not all(
                isinstance(team, dict) for team in competitors
            ):
                raise ValueError(
                    f"Bovada response has invalid event competitors: {_snippet(payload)!r}"
                )
            names = {bool(team.get("home")): str(team.get("name", "")) for team in competitors}
            home_name = names.get(True, "")
            away_name = names.get(False, "")
            if home_name not in NFL_TEAM_NAMES or away_name not in NFL_TEAM_NAMES:
                continue
            groups = event.get("displayGroups", [])
            if not isinstance(groups, list) or not all(isinstance(group, dict) for group in groups):
                raise ValueError(
                    f"Bovada response has invalid display groups: {_snippet(payload)!r}"
                )
            markets = []
            for group in groups:
                group_markets = group.get("markets", [])
                if not isinstance(group_markets, list) or not all(
                    isinstance(market, dict) for market in group_markets
                ):
                    raise ValueError(f"Bovada response has invalid markets: {_snippet(payload)!r}")
                markets.extend(group_markets)
            for market in markets:
                kind = {"Point Spread": "spreads", "Total": "totals"}.get(market.get("description"))
                period = market.get("period") or {}
                if not isinstance(period, dict):
                    raise ValueError(
                        f"Bovada response has an invalid period: {_snippet(payload)!r}"
                    )
                if (
                    kind is None
                    or not period.get("main")
                    or period.get("description") != "Game"
                    or market.get("status") != "O"
                ):
                    continue
                outcomes = market.get("outcomes", [])
                if not isinstance(outcomes, list) or not all(
                    isinstance(outcome, dict) for outcome in outcomes
                ):
                    raise ValueError(f"Bovada response has invalid outcomes: {_snippet(payload)!r}")
                if not all(isinstance(outcome.get("price") or {}, dict) for outcome in outcomes):
                    raise ValueError(
                        f"Bovada response has invalid outcome prices: {_snippet(payload)!r}"
                    )
                expected = {"H", "A"} if kind == "spreads" else {"O", "U"}
                if {outcome.get("type") for outcome in outcomes} != expected or len(outcomes) != 2:
                    continue
                home_outcome = next(
                    (outcome for outcome in outcomes if outcome.get("type") == "H"), None
                )
                if kind == "spreads" and home_outcome is None:
                    continue
                try:
                    home_margin = (
                        -float(home_outcome["price"]["handicap"]) if home_outcome else float("nan")
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                for outcome in outcomes:
                    side = {"H": "HOME", "A": "AWAY", "O": "OVER", "U": "UNDER"}[outcome["type"]]
                    price = outcome.get("price") or {}
                    try:
                        line = float(price["handicap"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    rows.append(
                        {
                            "observed_at_utc": observed,
                            "provider": SOURCE,
                            "provider_event_id": str(event["id"]),
                            "sport_key": "americanfootball_nfl",
                            "commence_time_utc": start,
                            "home_team_name": home_name,
                            "away_team_name": away_name,
                            "home_team": NFL_TEAM_NAMES[home_name],
                            "away_team": NFL_TEAM_NAMES[away_name],
                            "nflverse_game_id": pd.NA,
                            "bookmaker_key": "bovada",
                            "bookmaker_title": "Bovada",
                            "bookmaker_last_update_utc": pd.NaT,
                            "quote_timestamp_basis": "capture_observed_utc",
                            "market": kind,
                            "market_last_update_utc": pd.NaT,
                            "outcome_name": home_name
                            if side == "HOME"
                            else away_name
                            if side == "AWAY"
                            else side.title(),
                            "outcome_side": side,
                            "line": line,
                            "price": _price(price.get("american")),
                            "home_spread_line": home_margin,
                            "raw_response_sha256": digest,
                        }
                    )
    return rows


def capture(features_path: Path) -> dict[str, Any]:
    require_acquisition(SOURCE)
    require_private_raw_destination(SOURCE, ROOT)
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            observed = datetime.now(UTC)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Bovada public endpoint returned HTTP {error.code}") from error
    features = pd.read_parquet(features_path)
    quotes = pd.DataFrame(_rows(payload, observed), columns=QUOTE_COLUMNS)
    if quotes.empty:
        raise ValueError("No current pregame Bovada spreads or totals")
    for column in (
        "observed_at_utc",
        "commence_time_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
    ):
        quotes[column] = pd.to_datetime(quotes[column], utc=True)
    quotes = attach_nflverse_game_ids(quotes, features)
    spreads = quotes.loc[quotes["market"].eq("spreads") & quotes["outcome_side"].eq("HOME")]
    if spreads.empty:
        raise ValueError("No current pregame Bovada spreads")
    if spreads["nflverse_game_id"].isna().any() or spreads["nflverse_game_id"].duplicated().any():
        raise ValueError("Bovada spreads did not match unique scheduled NFL games")
    snapshot = write_market_snapshot(
        payload,
        quotes,
        ROOT,
        observed_at=observed,
        request_metadata={
            "source": SOURCE,
            "url": URL,
            "season": int(
                features.loc[features["game_id"].isin(spreads["nflverse_game_id"]), "season"].iloc[
                    0
                ]
            ),
            "week": int(
                features.loc[features["game_id"].isin(spreads["nflverse_game_id"]), "week"].iloc[0]
            ),
        },
        extra_manifest={
            "capture_kind": "live",
            "book_scope": "single_book_bovada",
            "quote_timestamp_semantics": "book_update_unknown_capture_time_observed",
        },
        provider=SOURCE,
    )
    return {
        "captured": True,
        "source": SOURCE,
        "bookmaker": "Bovada",
        "observed_at_utc": observed.isoformat(),
        "snapshot": str(snapshot.root),
        "games": len(spreads),
        "quotes": len(quotes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one private Bovada NFL odds snapshot")
    parser.add_argument(
        "--features", type=Path, default=REPO / "data" / "processed" / "game_features.parquet"
    )
    args = parser.parse_args()
    print(json.dumps(capture(args.features), sort_keys=True))


if __name__ == "__main__":
    main()
