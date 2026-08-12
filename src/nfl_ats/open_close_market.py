"""Free 2025 NFL opening/closing multi-book sample acquisition.

The source identifies quotes as an opener or a reported book close, but does
not provide quote timestamps.  The normalized contract preserves that useful
stage information without pretending the rows are timestamped observations.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.snapshots import utc_snapshot_id

DATASET_REF = "oliviersportsdata/nfl-multi-market-sample-2025"
DATASET_PAGE = f"https://www.kaggle.com/datasets/{DATASET_REF}"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
METADATA_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_REF}"

BOOK_PREFIXES = (
    "Opener",
    "BookMaker",
    "Heritage",
    "BetOnline",
    "Betcris",
    "Bovada",
    "BetAnything",
    "WilliamHill",
    "Ozoon",
    "Everygame",
)

TEAM_IDS = {
    "Arizona": "ARI",
    "Atlanta": "ATL",
    "Baltimore": "BAL",
    "Buffalo": "BUF",
    "Carolina": "CAR",
    "Chicago": "CHI",
    "Cincinnati": "CIN",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Green Bay": "GB",
    "Houston": "HOU",
    "Indianapolis": "IND",
    "Jacksonville": "JAX",
    "Kansas City": "KC",
    "L.A. Chargers": "LAC",
    "L.A. Rams": "LA",
    "Las Vegas": "LV",
    "Miami": "MIA",
    "Minnesota": "MIN",
    "N.Y. Giants": "NYG",
    "N.Y. Jets": "NYJ",
    "New England": "NE",
    "New Orleans": "NO",
    "Philadelphia": "PHI",
    "Pittsburgh": "PIT",
    "San Francisco": "SF",
    "Seattle": "SEA",
    "Tampa Bay": "TB",
    "Tennessee": "TEN",
    "Washington": "WAS",
}

BASE_REQUIRED_COLUMNS = (
    "Season",
    "Date",
    "Away_Team",
    "Home_Team",
    "Away_Score",
    "Home_Score",
    "ML_Away_Bet%",
    "ML_Home_Bet%",
    "Spread_Away_Bet%",
    "Spread_Home_Bet%",
    "Total_Over_Bet%",
    "Total_Under_Bet%",
)

QUOTE_COLUMNS = (
    "season",
    "game_date",
    "provider_event_id",
    "provider",
    "source_version",
    "home_team_name",
    "away_team_name",
    "home_team",
    "away_team",
    "nflverse_game_id",
    "bookmaker_key",
    "bookmaker_title",
    "quote_stage",
    "is_timestamped_quote",
    "observed_at_utc",
    "spread_moneyline_direction_consistent",
    "market",
    "outcome_name",
    "outcome_side",
    "line",
    "price",
    "home_spread_line",
    "public_bet_percent",
    "raw_archive_sha256",
)


@dataclass(frozen=True)
class OpenCloseSnapshot:
    snapshot_id: str
    root: Path
    archive_path: Path
    quotes_path: Path
    games_path: Path
    manifest_path: Path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-ats-research/0.2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return bytes(response.read())


def fetch_open_close_archive() -> tuple[bytes, dict[str, Any]]:
    payload = _http_bytes(DOWNLOAD_URL)
    metadata_payload = json.loads(_http_bytes(METADATA_URL))
    metadata = {
        "dataset_ref": metadata_payload.get("ref"),
        "title": metadata_payload.get("title"),
        "version": metadata_payload.get("currentVersionNumber"),
        "last_updated": metadata_payload.get("lastUpdated"),
        "license": metadata_payload.get("licenseName"),
        "download_url": DOWNLOAD_URL,
        "dataset_page": DATASET_PAGE,
    }
    return payload, metadata


def parse_open_close_archive(payload: bytes) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise DataContractError(
                    f"Open/close archive must contain one CSV; found {len(csv_names)}"
                )
            with archive.open(csv_names[0]) as source:
                table = pd.read_csv(source, sep=";", low_memory=False)
    except zipfile.BadZipFile as error:
        raise DataContractError("Open/close download is not a valid ZIP archive") from error
    require_columns(table, BASE_REQUIRED_COLUMNS, "open_close_sample")
    return table


def _bookmaker_key(prefix: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", prefix).lower()


def _number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _quote_row(
    base: dict[str, Any],
    *,
    market: str,
    side: str,
    name: str,
    line: float,
    price: float,
    home_spread_line: float,
    public_bet_percent: float,
) -> dict[str, Any]:
    return {
        **base,
        "market": market,
        "outcome_name": name,
        "outcome_side": side,
        "line": line,
        "price": price,
        "home_spread_line": home_spread_line,
        "public_bet_percent": public_bet_percent,
    }


def _implied_probability(price: float) -> float:
    if not np.isfinite(price) or price == 0:
        return np.nan
    return -price / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def _spread_moneyline_consistent(away_price: float, home_price: float, home_line: float) -> bool:
    """Check whether spread and moneyline identify the same favorite."""

    away_probability = _implied_probability(away_price)
    home_probability = _implied_probability(home_price)
    if not all(np.isfinite(value) for value in (away_probability, home_probability, home_line)):
        return False
    probability_direction = np.sign(home_probability - away_probability)
    spread_direction = np.sign(-home_line)
    if probability_direction == 0 or spread_direction == 0:
        return True
    return bool(probability_direction == spread_direction)


def normalize_open_close_sample(
    table: pd.DataFrame,
    *,
    source_version: int | str | None = None,
    raw_archive_sha256: str = "",
) -> pd.DataFrame:
    """Convert the wide source into one row per market outcome and book."""

    require_columns(table, BASE_REQUIRED_COLUMNS, "open_close_sample")
    required_market_columns: list[str] = []
    for prefix in BOOK_PREFIXES:
        required_market_columns.extend(
            (
                f"{prefix}_ML_Away_US",
                f"{prefix}_ML_Home_US",
                f"{prefix}_Spread_Away_Line",
                f"{prefix}_Spread_Away_US",
                f"{prefix}_Spread_Home_Line",
                f"{prefix}_Spread_Home_US",
                f"{prefix}_Total_Line",
                f"{prefix}_Total_Over_US",
                f"{prefix}_Total_Under_US",
            )
        )
    require_columns(table, required_market_columns, "open_close_sample markets")
    if table.duplicated(["Season", "Date", "Away_Team", "Home_Team"]).any():
        raise DataContractError("Open/close sample contains duplicate games")

    digest = raw_archive_sha256
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        home_name = str(row["Home_Team"])
        away_name = str(row["Away_Team"])
        if home_name not in TEAM_IDS or away_name not in TEAM_IDS:
            raise DataContractError(f"Unknown open/close team names: {away_name} at {home_name}")
        season = int(row["Season"])
        game_date = pd.Timestamp(row["Date"]).normalize()
        event_id = (
            f"{season}_{game_date.date().isoformat()}_{TEAM_IDS[away_name]}_{TEAM_IDS[home_name]}"
        )
        public = {
            "AWAY_ML": _number(row, "ML_Away_Bet%"),
            "HOME_ML": _number(row, "ML_Home_Bet%"),
            "AWAY_SPREAD": _number(row, "Spread_Away_Bet%"),
            "HOME_SPREAD": _number(row, "Spread_Home_Bet%"),
            "OVER": _number(row, "Total_Over_Bet%"),
            "UNDER": _number(row, "Total_Under_Bet%"),
        }
        for prefix in BOOK_PREFIXES:
            stage = "opening" if prefix == "Opener" else "reported_closing"
            home_raw = _number(row, f"{prefix}_Spread_Home_Line")
            home_standardized = -home_raw if np.isfinite(home_raw) else np.nan
            total_line = _number(row, f"{prefix}_Total_Line")
            away_moneyline = _number(row, f"{prefix}_ML_Away_US")
            home_moneyline = _number(row, f"{prefix}_ML_Home_US")
            base = {
                "season": season,
                "game_date": game_date,
                "provider_event_id": event_id,
                "provider": "olivier_kaggle_sample",
                "source_version": source_version,
                "home_team_name": home_name,
                "away_team_name": away_name,
                "home_team": TEAM_IDS[home_name],
                "away_team": TEAM_IDS[away_name],
                "nflverse_game_id": pd.NA,
                "bookmaker_key": _bookmaker_key(prefix),
                "bookmaker_title": prefix,
                "quote_stage": stage,
                "is_timestamped_quote": False,
                "observed_at_utc": pd.NaT,
                "spread_moneyline_direction_consistent": _spread_moneyline_consistent(
                    away_moneyline, home_moneyline, home_raw
                ),
                "raw_archive_sha256": digest,
            }
            rows.extend(
                (
                    _quote_row(
                        base,
                        market="h2h",
                        side="AWAY",
                        name=away_name,
                        line=np.nan,
                        price=away_moneyline,
                        home_spread_line=np.nan,
                        public_bet_percent=public["AWAY_ML"],
                    ),
                    _quote_row(
                        base,
                        market="h2h",
                        side="HOME",
                        name=home_name,
                        line=np.nan,
                        price=home_moneyline,
                        home_spread_line=np.nan,
                        public_bet_percent=public["HOME_ML"],
                    ),
                    _quote_row(
                        base,
                        market="spreads",
                        side="AWAY",
                        name=away_name,
                        line=_number(row, f"{prefix}_Spread_Away_Line"),
                        price=_number(row, f"{prefix}_Spread_Away_US"),
                        home_spread_line=home_standardized,
                        public_bet_percent=public["AWAY_SPREAD"],
                    ),
                    _quote_row(
                        base,
                        market="spreads",
                        side="HOME",
                        name=home_name,
                        line=home_raw,
                        price=_number(row, f"{prefix}_Spread_Home_US"),
                        home_spread_line=home_standardized,
                        public_bet_percent=public["HOME_SPREAD"],
                    ),
                    _quote_row(
                        base,
                        market="totals",
                        side="OVER",
                        name="Over",
                        line=total_line,
                        price=_number(row, f"{prefix}_Total_Over_US"),
                        home_spread_line=np.nan,
                        public_bet_percent=public["OVER"],
                    ),
                    _quote_row(
                        base,
                        market="totals",
                        side="UNDER",
                        name="Under",
                        line=total_line,
                        price=_number(row, f"{prefix}_Total_Under_US"),
                        home_spread_line=np.nan,
                        public_bet_percent=public["UNDER"],
                    ),
                )
            )
    quotes = pd.DataFrame(rows, columns=QUOTE_COLUMNS)
    for column in ("line", "price", "home_spread_line", "public_bet_percent"):
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    return quotes.sort_values(
        ["game_date", "provider_event_id", "bookmaker_key", "market", "outcome_side"],
        ignore_index=True,
    )


def summarize_open_close_games(
    quotes: pd.DataFrame,
    reference_games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one-game open/consensus-close movement and source-audit rows."""

    home_spreads = quotes.loc[
        quotes["market"].eq("spreads") & quotes["outcome_side"].eq("HOME")
    ].copy()
    keys = [
        "season",
        "game_date",
        "provider_event_id",
        "home_team",
        "away_team",
        "home_team_name",
        "away_team_name",
    ]
    opening = home_spreads.loc[home_spreads["quote_stage"].eq("opening")][
        [*keys, "home_spread_line", "spread_moneyline_direction_consistent"]
    ].rename(
        columns={
            "home_spread_line": "reported_opening_home_spread",
            "spread_moneyline_direction_consistent": "opening_direction_consistent",
        }
    )
    opening["opening_home_spread"] = opening["reported_opening_home_spread"].where(
        opening["opening_direction_consistent"]
    )
    closing_rows = home_spreads.loc[home_spreads["quote_stage"].eq("reported_closing")].copy()
    closing_quality = closing_rows.groupby(keys, as_index=False).agg(
        reported_closing_books=("bookmaker_key", "nunique"),
        inconsistent_closing_books=(
            "spread_moneyline_direction_consistent",
            lambda values: int((~values.astype(bool)).sum()),
        ),
    )
    closing = (
        closing_rows.loc[closing_rows["spread_moneyline_direction_consistent"]]
        .groupby(keys, as_index=False)
        .agg(
            closing_books=("bookmaker_key", "nunique"),
            consensus_closing_home_spread=("home_spread_line", "median"),
            closing_spread_min=("home_spread_line", "min"),
            closing_spread_max=("home_spread_line", "max"),
            closing_spread_std=("home_spread_line", "std"),
        )
    )
    closing = closing.merge(closing_quality, on=keys, how="left", validate="one_to_one")
    games = opening.merge(closing, on=keys, how="outer", validate="one_to_one")
    games["open_to_close_movement"] = (
        games["consensus_closing_home_spread"] - games["opening_home_spread"]
    )
    games["nflverse_game_id"] = pd.NA
    games["nflverse_spread_line"] = np.nan
    if reference_games is not None:
        require_columns(
            reference_games,
            ("game_id", "season", "gameday", "home_team", "away_team", "spread_line"),
            "nflverse_reference_games",
        )
        reference = reference_games[
            ["game_id", "season", "gameday", "home_team", "away_team", "spread_line"]
        ].copy()
        reference["game_date"] = pd.to_datetime(reference["gameday"]).dt.normalize()
        reference = reference.rename(
            columns={"game_id": "matched_game_id", "spread_line": "matched_spread_line"}
        )
        join_keys = ["season", "game_date", "home_team", "away_team"]
        if reference.duplicated(join_keys).any():
            raise DataContractError("Reference games are duplicated on open/close join keys")
        games = games.drop(columns=["nflverse_game_id", "nflverse_spread_line"]).merge(
            reference[[*join_keys, "matched_game_id", "matched_spread_line"]],
            on=join_keys,
            how="left",
            validate="one_to_one",
        )
        games = games.rename(
            columns={
                "matched_game_id": "nflverse_game_id",
                "matched_spread_line": "nflverse_spread_line",
            }
        )
    games["consensus_vs_nflverse"] = (
        games["consensus_closing_home_spread"] - games["nflverse_spread_line"]
    )
    return games.sort_values(["game_date", "provider_event_id"], ignore_index=True)


def _audit(games: pd.DataFrame, quotes: pd.DataFrame) -> dict[str, Any]:
    movement = games["open_to_close_movement"].dropna()
    comparison = games["consensus_vs_nflverse"].dropna().abs()
    return {
        "games": len(games),
        "quote_rows": len(quotes),
        "bookmakers_including_opener": int(quotes["bookmaker_key"].nunique()),
        "closing_books": int(
            quotes.loc[quotes["quote_stage"].eq("reported_closing"), "bookmaker_key"].nunique()
        ),
        "markets": sorted(quotes["market"].unique().tolist()),
        "matched_nflverse_games": int(games["nflverse_game_id"].notna().sum()),
        "games_with_movement": len(movement),
        "inconsistent_opener_games": int((~games["opening_direction_consistent"]).sum()),
        "inconsistent_closing_book_games": int(games["inconsistent_closing_books"].sum()),
        "spread_changed_rate": float(movement.ne(0).mean()) if len(movement) else None,
        "mean_absolute_open_to_close_movement": (
            float(movement.abs().mean()) if len(movement) else None
        ),
        "median_open_to_close_movement": float(movement.median()) if len(movement) else None,
        "consensus_close_exact_nflverse_rate": (
            float(comparison.eq(0).mean()) if len(comparison) else None
        ),
        "consensus_close_within_half_nflverse_rate": (
            float(comparison.le(0.5).mean()) if len(comparison) else None
        ),
    }


def write_open_close_snapshot(
    payload: bytes,
    table: pd.DataFrame,
    quotes: pd.DataFrame,
    games: pd.DataFrame,
    root: Path,
    *,
    source_metadata: dict[str, Any],
    now: datetime | None = None,
) -> OpenCloseSnapshot:
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    snapshot_id = utc_snapshot_id(instant)
    destination = root / snapshot_id
    if destination.exists():
        raise FileExistsError(f"Open/close snapshot already exists: {destination}")
    destination.mkdir(parents=True)
    archive_path = destination / "source.zip"
    archive_path.write_bytes(payload)
    quotes_path = destination / "quotes.parquet"
    games_path = destination / "games.parquet"
    atomic_parquet(quotes, quotes_path)
    atomic_parquet(games, games_path)
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "created_at_utc": instant.astimezone(UTC).isoformat(),
        "source": source_metadata,
        "semantics": {
            "stages": ["opening", "reported_closing"],
            "timestamped_quotes": False,
            "spread_convention": "home_spread_line positive means home favored",
            "warning": (
                "Source labels opener and close but supplies no quote timestamps; "
                "rows cannot represent arbitrary decision-time availability."
            ),
        },
        "rows": {"source_games": len(table), "normalized_quotes": len(quotes)},
        "sha256": {
            "source.zip": _sha256(payload),
            "quotes.parquet": _sha256(quotes_path.read_bytes()),
            "games.parquet": _sha256(games_path.read_bytes()),
        },
        "audit": _audit(games, quotes),
    }
    manifest_path = destination / "manifest.json"
    atomic_json(manifest, manifest_path)
    return OpenCloseSnapshot(
        snapshot_id,
        destination,
        archive_path,
        quotes_path,
        games_path,
        manifest_path,
    )


def fetch_open_close_snapshot(
    root: Path,
    *,
    reference_games: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> OpenCloseSnapshot:
    payload, metadata = fetch_open_close_archive()
    table = parse_open_close_archive(payload)
    quotes = normalize_open_close_sample(
        table,
        source_version=metadata.get("version"),
        raw_archive_sha256=_sha256(payload),
    )
    games = summarize_open_close_games(quotes, reference_games)
    if reference_games is not None:
        mapping = games.set_index("provider_event_id")["nflverse_game_id"]
        quotes["nflverse_game_id"] = quotes["provider_event_id"].map(mapping).astype("string")
    return write_open_close_snapshot(
        payload,
        table,
        quotes,
        games,
        root,
        source_metadata=metadata,
        now=now,
    )
