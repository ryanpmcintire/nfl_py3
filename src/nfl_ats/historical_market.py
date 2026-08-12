"""Free historical NFL market data acquisition and normalization.

The public Spreadspoke/Kaggle archive contains one reported closing spread per
game.  It is useful as an independent source and for opening-vs-closing work if
an opener is added later, but it is not a timestamped quote history.  This
module keeps that limitation explicit in every normalized row.
"""

from __future__ import annotations

import hashlib
import io
import json
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

SPREADSPOKE_DATASET_PAGE = (
    "https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data"
)
SPREADSPOKE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/tobycrabtree/nfl-scores-and-betting-data"
)
SPREADSPOKE_METADATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/view/tobycrabtree/nfl-scores-and-betting-data"
)

SCORES_REQUIRED_COLUMNS = (
    "schedule_date",
    "schedule_season",
    "schedule_week",
    "schedule_playoff",
    "team_home",
    "score_home",
    "score_away",
    "team_away",
    "team_favorite_id",
    "spread_favorite",
    "over_under_line",
)
TEAMS_REQUIRED_COLUMNS = ("team_name", "team_id")

# nflverse uses LA/LV while Spreadspoke's stable franchise IDs use LAR/LVR.
TEAM_ID_ALIASES = {"LAR": "LA", "LVR": "LV"}


@dataclass(frozen=True)
class HistoricalMarketSnapshot:
    snapshot_id: str
    root: Path
    archive_path: Path
    market_path: Path
    teams_path: Path
    audit_path: Path | None
    manifest_path: Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-ats-research/0.2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return bytes(response.read())


def fetch_spreadspoke_archive() -> tuple[bytes, dict[str, Any]]:
    """Download the public archive and its Kaggle version metadata."""

    archive = _http_bytes(SPREADSPOKE_DOWNLOAD_URL)
    metadata_payload = json.loads(_http_bytes(SPREADSPOKE_METADATA_URL))
    metadata = {
        "dataset_ref": metadata_payload.get("ref"),
        "title": metadata_payload.get("title"),
        "version": metadata_payload.get("currentVersionNumber"),
        "last_updated": metadata_payload.get("lastUpdated"),
        "license": metadata_payload.get("licenseName"),
        "download_url": SPREADSPOKE_DOWNLOAD_URL,
        "dataset_page": SPREADSPOKE_DATASET_PAGE,
    }
    return archive, metadata


def parse_spreadspoke_archive(payload: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate the scores and franchise map from an archive payload."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
            required_files = {"spreadspoke_scores.csv", "nfl_teams.csv"}
            missing_files = sorted(required_files.difference(names))
            if missing_files:
                raise DataContractError(
                    f"Spreadspoke archive is missing files: {', '.join(missing_files)}"
                )
            with archive.open("spreadspoke_scores.csv") as scores_file:
                scores = pd.read_csv(scores_file, low_memory=False)
            with archive.open("nfl_teams.csv") as teams_file:
                teams = pd.read_csv(teams_file, low_memory=False)
    except zipfile.BadZipFile as error:
        raise DataContractError("Spreadspoke download is not a valid ZIP archive") from error

    require_columns(scores, SCORES_REQUIRED_COLUMNS, "spreadspoke_scores")
    require_columns(teams, TEAMS_REQUIRED_COLUMNS, "spreadspoke_teams")
    if teams["team_name"].duplicated().any():
        raise DataContractError("Spreadspoke team map contains duplicate team names")
    return scores, teams


def _nflverse_team_id(value: Any) -> str:
    identifier = str(value).strip().upper()
    return TEAM_ID_ALIASES.get(identifier, identifier)


def normalize_spreadspoke(
    scores: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    source_version: int | str | None = None,
) -> pd.DataFrame:
    """Normalize favorite-oriented rows to the project's home-oriented convention."""

    require_columns(scores, SCORES_REQUIRED_COLUMNS, "spreadspoke_scores")
    require_columns(teams, TEAMS_REQUIRED_COLUMNS, "spreadspoke_teams")
    team_lookup = teams.set_index("team_name")["team_id"].astype(str).to_dict()

    result = pd.DataFrame(index=scores.index)
    result["source"] = "spreadspoke_kaggle"
    result["source_version"] = source_version
    result["source_line_type"] = "reported_closing"
    result["is_timestamped_quote"] = False
    result["observed_at_utc"] = pd.NaT
    result["game_date"] = pd.to_datetime(scores["schedule_date"], errors="raise").dt.normalize()
    result["season"] = pd.to_numeric(scores["schedule_season"], errors="raise").astype(int)
    result["week"] = scores["schedule_week"].astype("string")
    result["is_playoff"] = scores["schedule_playoff"].astype("string").str.upper().eq("TRUE")
    result["home_team_name"] = scores["team_home"].astype("string")
    result["away_team_name"] = scores["team_away"].astype("string")
    result["home_team_id"] = result["home_team_name"].map(team_lookup).map(_nflverse_team_id)
    result["away_team_id"] = result["away_team_name"].map(team_lookup).map(_nflverse_team_id)
    result["favorite_team_id"] = scores["team_favorite_id"].map(_nflverse_team_id)
    result["favorite_spread"] = pd.to_numeric(scores["spread_favorite"], errors="coerce")
    magnitude = result["favorite_spread"].abs()
    favorite = result["favorite_team_id"]
    result["spread_line"] = np.select(
        [
            favorite.eq(result["home_team_id"]),
            favorite.eq(result["away_team_id"]),
            favorite.isin(("PICK", "PK")),
        ],
        [magnitude, -magnitude, 0.0],
        default=np.nan,
    )
    result["total_line"] = pd.to_numeric(scores["over_under_line"], errors="coerce")
    result["home_score"] = pd.to_numeric(scores["score_home"], errors="coerce")
    result["away_score"] = pd.to_numeric(scores["score_away"], errors="coerce")
    result["nflverse_game_id"] = pd.NA

    unknown_teams = result["home_team_id"].isna() | result["away_team_id"].isna()
    if unknown_teams.any():
        examples = result.loc[unknown_teams, ["home_team_name", "away_team_name"]].head()
        raise DataContractError(
            f"Spreadspoke team mapping failed:\n{examples.to_string(index=False)}"
        )
    invalid_favorite = result["favorite_spread"].notna() & result["spread_line"].isna()
    if invalid_favorite.any():
        examples = result.loc[
            invalid_favorite,
            ["home_team_id", "away_team_id", "favorite_team_id", "favorite_spread"],
        ].head()
        raise DataContractError(
            f"Spreadspoke favorite does not match either team:\n{examples.to_string(index=False)}"
        )
    return result.sort_values(["game_date", "home_team_id", "away_team_id"], ignore_index=True)


def attach_nflverse_games(market: pd.DataFrame, reference_games: pd.DataFrame) -> pd.DataFrame:
    """Attach nflverse IDs by season/date and stable franchise identifiers."""

    require_columns(
        reference_games,
        ("game_id", "season", "gameday", "home_team", "away_team"),
        "nflverse_reference_games",
    )
    reference = reference_games[
        ["game_id", "season", "gameday", "home_team", "away_team", "spread_line"]
    ].copy()
    reference["game_date"] = pd.to_datetime(reference["gameday"], errors="raise").dt.normalize()
    reference["home_team_id"] = reference["home_team"].map(_nflverse_team_id)
    reference["away_team_id"] = reference["away_team"].map(_nflverse_team_id)
    reference = reference.rename(
        columns={"game_id": "matched_game_id", "spread_line": "nflverse_spread_line"}
    )
    keys = ["season", "game_date", "home_team_id", "away_team_id"]
    if reference.duplicated(keys).any():
        raise DataContractError("nflverse reference games are not unique on the market join key")
    joined = market.drop(columns="nflverse_game_id", errors="ignore").merge(
        reference[[*keys, "matched_game_id", "nflverse_spread_line"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    joined = joined.rename(columns={"matched_game_id": "nflverse_game_id"})
    return joined


def audit_spread_agreement(market: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare overlapping reported closes with nflverse's historical close."""

    require_columns(
        market,
        ("spread_line", "nflverse_game_id", "nflverse_spread_line"),
        "normalized_historical_market",
    )
    matched = market.loc[market["nflverse_game_id"].notna()].copy()
    compared = matched.loc[
        matched["spread_line"].notna() & matched["nflverse_spread_line"].notna()
    ].copy()
    compared["spread_difference"] = compared["spread_line"] - compared["nflverse_spread_line"]
    compared["absolute_spread_difference"] = compared["spread_difference"].abs()
    metrics: dict[str, Any] = {
        "historical_rows": len(market),
        "historical_rows_with_spread": int(market["spread_line"].notna().sum()),
        "matched_nflverse_games": len(matched),
        "compared_spreads": len(compared),
        "unmatched_rows": int(market["nflverse_game_id"].isna().sum()),
    }
    if not compared.empty:
        difference = compared["absolute_spread_difference"]
        metrics.update(
            {
                "exact_agreement_rate": float(difference.eq(0).mean()),
                "within_half_point_rate": float(difference.le(0.5).mean()),
                "within_one_point_rate": float(difference.le(1.0).mean()),
                "mean_absolute_difference": float(difference.mean()),
                "max_absolute_difference": float(difference.max()),
            }
        )
    discrepancies = compared.sort_values(
        "absolute_spread_difference", ascending=False, ignore_index=True
    )
    return metrics, discrepancies


def write_historical_market_snapshot(
    payload: bytes,
    scores: pd.DataFrame,
    teams: pd.DataFrame,
    market: pd.DataFrame,
    root: Path,
    *,
    source_metadata: dict[str, Any],
    audit_metrics: dict[str, Any] | None = None,
    discrepancies: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> HistoricalMarketSnapshot:
    """Write an immutable raw archive, normalized table, audit, and manifest."""

    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    snapshot_id = utc_snapshot_id(instant)
    destination = root / snapshot_id
    if destination.exists():
        raise FileExistsError(f"Historical market snapshot already exists: {destination}")
    destination.mkdir(parents=True)
    archive_path = destination / "source.zip"
    archive_path.write_bytes(payload)
    market_path = destination / "market.parquet"
    teams_path = destination / "teams.parquet"
    atomic_parquet(market, market_path)
    atomic_parquet(teams, teams_path)
    audit_path: Path | None = None
    if discrepancies is not None:
        audit_path = destination / "spread_discrepancies.parquet"
        atomic_parquet(discrepancies, audit_path)
    manifest = {
        "snapshot_id": snapshot_id,
        "created_at_utc": instant.astimezone(UTC).isoformat(),
        "source": source_metadata,
        "semantics": {
            "line_type": "reported_closing",
            "timestamped_quotes": False,
            "spread_convention": "positive means home favored",
            "warning": "One line per completed game; not a historical quote timeline.",
        },
        "rows": {
            "source_scores": len(scores),
            "source_teams": len(teams),
            "normalized_market": len(market),
        },
        "sha256": {
            "source.zip": _sha256_bytes(payload),
            "market.parquet": _sha256_bytes(market_path.read_bytes()),
            "teams.parquet": _sha256_bytes(teams_path.read_bytes()),
        },
        "audit": audit_metrics,
    }
    manifest_path = destination / "manifest.json"
    atomic_json(manifest, manifest_path)
    return HistoricalMarketSnapshot(
        snapshot_id=snapshot_id,
        root=destination,
        archive_path=archive_path,
        market_path=market_path,
        teams_path=teams_path,
        audit_path=audit_path,
        manifest_path=manifest_path,
    )


def fetch_historical_market_snapshot(
    root: Path,
    *,
    reference_games: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> HistoricalMarketSnapshot:
    """Fetch, normalize, cross-check, and preserve the public archive."""

    payload, metadata = fetch_spreadspoke_archive()
    scores, teams = parse_spreadspoke_archive(payload)
    market = normalize_spreadspoke(scores, teams, source_version=metadata.get("version"))
    audit_metrics: dict[str, Any] | None = None
    discrepancies: pd.DataFrame | None = None
    if reference_games is not None:
        market = attach_nflverse_games(market, reference_games)
        audit_metrics, discrepancies = audit_spread_agreement(market)
    return write_historical_market_snapshot(
        payload,
        scores,
        teams,
        market,
        root,
        source_metadata=metadata,
        audit_metrics=audit_metrics,
        discrepancies=discrepancies,
        now=now,
    )
