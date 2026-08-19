"""Point-in-time market quote adapters, immutable storage, and CLV helpers."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.odds import implied_probability
from nfl_ats.provenance import sha256_file

ODDS_API_PROVIDER = "the-odds-api"
ODDS_API_SPORT = "americanfootball_nfl"
ODDS_API_URL = f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/odds/"

NFL_TEAM_NAMES = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

QUOTE_COLUMNS = (
    "observed_at_utc",
    "provider",
    "provider_event_id",
    "sport_key",
    "commence_time_utc",
    "home_team_name",
    "away_team_name",
    "home_team",
    "away_team",
    "nflverse_game_id",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update_utc",
    "market",
    "market_last_update_utc",
    "outcome_name",
    "outcome_side",
    "line",
    "price",
    "home_spread_line",
    "raw_response_sha256",
)


@dataclass(frozen=True)
class MarketSnapshot:
    snapshot_id: str
    root: Path
    raw_path: Path
    quotes_path: Path
    manifest_path: Path


def _utc(instant: datetime | None) -> datetime:
    value = instant or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _outcome_side(name: str, home_name: str, away_name: str) -> str:
    if name == home_name:
        return "HOME"
    if name == away_name:
        return "AWAY"
    lowered = name.lower()
    if lowered == "over":
        return "OVER"
    if lowered == "under":
        return "UNDER"
    if lowered == "draw":
        return "DRAW"
    return "OTHER"


def parse_odds_api_response(
    payload: bytes,
    *,
    observed_at: datetime | None = None,
) -> pd.DataFrame:
    """Normalize a V4 odds response without discarding the raw representation."""

    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        raise ValueError("The Odds API response must be a JSON array")
    observed = _utc(observed_at)
    digest = _sha256_bytes(payload)
    rows: list[dict[str, Any]] = []
    for event in decoded:
        if not isinstance(event, dict):
            raise ValueError("The Odds API event must be a JSON object")
        home_name = str(event.get("home_team", ""))
        away_name = str(event.get("away_team", ""))
        home_team = NFL_TEAM_NAMES.get(home_name)
        away_team = NFL_TEAM_NAMES.get(away_name)
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                market_key = str(market.get("key", ""))
                outcomes = market.get("outcomes", [])
                home_point = next(
                    (
                        outcome.get("point")
                        for outcome in outcomes
                        if outcome.get("name") == home_name
                    ),
                    None,
                )
                standardized_home_line = (
                    -float(home_point)
                    if market_key == "spreads" and home_point is not None
                    else np.nan
                )
                for outcome in outcomes:
                    rows.append(
                        {
                            "observed_at_utc": observed,
                            "provider": ODDS_API_PROVIDER,
                            "provider_event_id": event.get("id"),
                            "sport_key": event.get("sport_key"),
                            "commence_time_utc": event.get("commence_time"),
                            "home_team_name": home_name,
                            "away_team_name": away_name,
                            "home_team": home_team,
                            "away_team": away_team,
                            "nflverse_game_id": pd.NA,
                            "bookmaker_key": bookmaker.get("key"),
                            "bookmaker_title": bookmaker.get("title"),
                            "bookmaker_last_update_utc": bookmaker.get("last_update"),
                            "market": market_key,
                            "market_last_update_utc": market.get("last_update"),
                            "outcome_name": outcome.get("name"),
                            "outcome_side": _outcome_side(
                                str(outcome.get("name", "")), home_name, away_name
                            ),
                            "line": outcome.get("point"),
                            "price": outcome.get("price"),
                            "home_spread_line": standardized_home_line,
                            "raw_response_sha256": digest,
                        }
                    )
    quotes = pd.DataFrame(rows, columns=QUOTE_COLUMNS)
    for column in (
        "observed_at_utc",
        "commence_time_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
    ):
        quotes[column] = pd.to_datetime(quotes[column], errors="coerce", utc=True)
    for column in ("line", "price", "home_spread_line"):
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    return quotes


def attach_nflverse_game_ids(quotes: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Match quotes by normalized teams and nearest kickoff within twelve hours."""

    required = {"game_id", "home_team", "away_team", "kickoff"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise ValueError(f"Schedule matching is missing columns: {', '.join(missing)}")
    result = quotes.copy()
    schedule = schedules.loc[:, list(required)].copy()
    schedule["kickoff"] = pd.to_datetime(schedule["kickoff"], errors="coerce", utc=True)
    event_rows = result[
        ["provider_event_id", "home_team", "away_team", "commence_time_utc"]
    ].drop_duplicates()
    mapping: dict[str, str] = {}
    for _, event in event_rows.iterrows():
        candidates = schedule.loc[
            schedule["home_team"].eq(event["home_team"])
            & schedule["away_team"].eq(event["away_team"])
        ].copy()
        if candidates.empty or pd.isna(event["commence_time_utc"]):
            continue
        candidates["time_gap"] = (candidates["kickoff"] - event["commence_time_utc"]).abs()
        candidates = candidates.loc[candidates["time_gap"].le(pd.Timedelta(hours=12))]
        if len(candidates) == 1:
            mapping[str(event["provider_event_id"])] = str(candidates.iloc[0]["game_id"])
    result["nflverse_game_id"] = (
        result["provider_event_id"].astype(str).map(mapping).astype("string")
    )
    return result


def fetch_odds_api(
    *,
    api_key: str,
    regions: str = "us",
    markets: str = "spreads,h2h",
    bookmakers: str | None = None,
    timeout: int = 30,
) -> tuple[bytes, dict[str, str]]:
    if not api_key.strip():
        raise ValueError("The Odds API key is empty")
    parameters = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        parameters["bookmakers"] = bookmakers
    url = ODDS_API_URL + "?" + urllib.parse.urlencode(parameters)
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


def fetch_odds_api_from_environment(**kwargs: Any) -> tuple[bytes, dict[str, str]]:
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise ValueError("Set THE_ODDS_API_KEY before fetching live odds")
    return fetch_odds_api(api_key=api_key, **kwargs)


def write_market_snapshot(
    payload: bytes,
    quotes: pd.DataFrame,
    root: Path,
    *,
    observed_at: datetime,
    request_metadata: dict[str, Any],
    quota: dict[str, str] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> MarketSnapshot:
    identifier = run_id(observed_at)
    destination = root / identifier
    if destination.exists():
        raise ValueError(f"Market snapshot already exists: {destination}")
    destination.mkdir(parents=True)
    snapshot = MarketSnapshot(
        snapshot_id=identifier,
        root=destination,
        raw_path=destination / "response.json",
        quotes_path=destination / "quotes.parquet",
        manifest_path=destination / "manifest.json",
    )
    temporary_raw = snapshot.raw_path.with_suffix(".json.tmp")
    temporary_raw.write_bytes(payload)
    temporary_raw.replace(snapshot.raw_path)
    atomic_parquet(quotes, snapshot.quotes_path)
    manifest = {
        "schema_version": 1,
        "snapshot_id": identifier,
        "observed_at_utc": _utc(observed_at).isoformat(),
        "provider": ODDS_API_PROVIDER,
        "request": request_metadata,
        "quota": quota or {},
        "files": {
            "response.json": {
                "bytes": len(payload),
                "sha256": sha256_file(snapshot.raw_path),
            },
            "quotes.parquet": {
                "rows": len(quotes),
                "sha256": sha256_file(snapshot.quotes_path),
            },
        },
    }
    if extra_manifest:
        overlap = sorted(set(extra_manifest).intersection(manifest))
        if overlap:
            raise ValueError(f"extra_manifest may not override manifest keys: {', '.join(overlap)}")
        manifest.update(extra_manifest)
    atomic_json(manifest, snapshot.manifest_path)
    return snapshot


def load_quote_history(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/quotes.parquet")) if root.is_dir() else []
    if not paths:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    history = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    return history.sort_values(
        ["commence_time_utc", "provider_event_id", "bookmaker_key", "market", "observed_at_utc"]
    ).reset_index(drop=True)


def latest_book_quotes(quotes: pd.DataFrame, *, before_kickoff: bool = True) -> pd.DataFrame:
    history = quotes.copy()
    history["observed_at_utc"] = pd.to_datetime(history["observed_at_utc"], utc=True)
    history["commence_time_utc"] = pd.to_datetime(history["commence_time_utc"], utc=True)
    if before_kickoff:
        history = history.loc[history["observed_at_utc"].lt(history["commence_time_utc"])]
    keys = ["provider_event_id", "bookmaker_key", "market", "outcome_side"]
    return history.sort_values("observed_at_utc").groupby(keys, as_index=False).tail(1)


def spread_consensus(quotes: pd.DataFrame) -> pd.DataFrame:
    latest = latest_book_quotes(quotes)
    spreads = latest.loc[latest["market"].eq("spreads") & latest["outcome_side"].eq("HOME")]
    if spreads.empty:
        return pd.DataFrame(
            columns=[
                "nflverse_game_id",
                "commence_time_utc",
                "bookmakers",
                "consensus_home_spread",
                "spread_min",
                "spread_max",
                "spread_std",
            ]
        )
    return (
        spreads.groupby(["nflverse_game_id", "commence_time_utc"], as_index=False, dropna=False)
        .agg(
            bookmakers=("bookmaker_key", "nunique"),
            consensus_home_spread=("home_spread_line", "median"),
            spread_min=("home_spread_line", "min"),
            spread_max=("home_spread_line", "max"),
            spread_std=("home_spread_line", "std"),
        )
        .sort_values("commence_time_utc")
    )


def tuesday_opener_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """The earliest Tuesday-captured home spread per game (the "Tuesday opener").

    Bookmakers conventionally release opening lines for the coming week's
    slate on Tuesday. This selects, per game and bookmaker, the earliest
    quote observed on a Tuesday (UTC), then reports the cross-book median as
    the game's opener line -- distinct from ``spread_consensus``, which
    reports the *latest* pre-kickoff quote instead of the opening one.

    ``opener_std`` (cross-book standard deviation of each book's own earliest
    Tuesday line) is the same dispersion proxy ``nfl_ats.clv.build_pairing_table``
    computes for the historical decision-labeled archive (its ``spread_std``,
    ``line_std`` renamed) -- added here so a LIVE production caller has the
    identical measure available from the free-form ``odds-ingest`` capture
    this project's weekly pipeline actually writes to ``data/market/raw``,
    without needing a decision-labeled snapshot store
    (``nfl_ats.best_pick_nomination`` is the first consumer). ``std`` on a
    single-book game is ``NaN`` by construction (pandas' ddof=1 default), not
    zero -- callers that treat missing dispersion as "not measurable" get
    that for free rather than a false zero.
    """

    required = {
        "observed_at_utc",
        "commence_time_utc",
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "outcome_side",
        "home_spread_line",
    }
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise ValueError(f"Quote history is missing columns: {', '.join(missing)}")
    columns = [
        "nflverse_game_id",
        "commence_time_utc",
        "bookmakers",
        "opener_home_spread",
        "opener_min",
        "opener_max",
        "opener_std",
        "observed_at_utc",
    ]
    history = quotes.copy()
    history["observed_at_utc"] = pd.to_datetime(history["observed_at_utc"], utc=True)
    history["commence_time_utc"] = pd.to_datetime(history["commence_time_utc"], utc=True)
    spreads = history.loc[history["market"].eq("spreads") & history["outcome_side"].eq("HOME")]
    tuesday = spreads.loc[spreads["observed_at_utc"].dt.weekday.eq(1)]
    if tuesday.empty:
        return pd.DataFrame(columns=columns)
    earliest_per_book = (
        tuesday.sort_values("observed_at_utc")
        .groupby(["nflverse_game_id", "bookmaker_key"], as_index=False, dropna=False)
        .head(1)
    )
    opener: pd.DataFrame = (
        earliest_per_book.groupby(
            ["nflverse_game_id", "commence_time_utc"], as_index=False, dropna=False
        )
        .agg(
            bookmakers=("bookmaker_key", "nunique"),
            opener_home_spread=("home_spread_line", "median"),
            opener_min=("home_spread_line", "min"),
            opener_max=("home_spread_line", "max"),
            opener_std=("home_spread_line", "std"),
            observed_at_utc=("observed_at_utc", "min"),
        )
        .sort_values("commence_time_utc")
        .reset_index(drop=True)
    )
    return opener


def closing_line_value(decisions: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Compare stored decisions with the last same-book pre-kickoff spread quote."""

    required = {
        "game_id",
        "bookmaker_key",
        "bet_side",
        "spread_line",
        "bet_odds",
        "decision_time_utc",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"CLV decisions are missing columns: {', '.join(missing)}")
    closing = latest_book_quotes(quotes)
    closing = closing.loc[closing["market"].eq("spreads")].copy()
    closing["game_id"] = closing["nflverse_game_id"]
    home_close = closing.loc[closing["outcome_side"].eq("HOME")][
        ["game_id", "bookmaker_key", "home_spread_line", "price", "observed_at_utc"]
    ].rename(
        columns={
            "home_spread_line": "closing_home_spread",
            "price": "closing_home_odds",
            "observed_at_utc": "closing_observed_at_utc",
        }
    )
    away_close = closing.loc[closing["outcome_side"].eq("AWAY")][
        ["game_id", "bookmaker_key", "price"]
    ].rename(columns={"price": "closing_away_odds"})
    merged = decisions.merge(home_close, on=["game_id", "bookmaker_key"], how="left")
    merged = merged.merge(away_close, on=["game_id", "bookmaker_key"], how="left")
    direction = merged["bet_side"].map({"HOME": 1.0, "AWAY": -1.0})
    merged["clv_points"] = direction * (merged["closing_home_spread"] - merged["spread_line"])
    closing_odds = merged["closing_home_odds"].where(
        merged["bet_side"].eq("HOME"), merged["closing_away_odds"]
    )
    merged["decision_break_even"] = merged["bet_odds"].map(implied_probability)
    merged["closing_break_even"] = closing_odds.map(implied_probability)
    merged["clv_probability"] = merged["closing_break_even"] - merged["decision_break_even"]
    return merged
