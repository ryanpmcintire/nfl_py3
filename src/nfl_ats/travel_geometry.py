"""Deterministic, decision-time-safe NFL travel geometry features.

This module turns the structural quantities first explored by ENV-03's
retrospective screen into a reusable feature contract. It never reads a game
result, betting line, observed weather value, or other postgame field.

The old screen intentionally recovered each team's home stadium from the
full-season modal schedule. This reusable builder uses a stricter rule: for
each decision row, a team's origin is its latest same-season true-home venue
at or before that row. Consequently, changing later schedule rows cannot
rewrite an earlier feature. Geometry is missing until an origin is available;
the builder does not backfill from the future.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES, TRAVEL_GEOMETRY_FEATURE_COLUMNS
from nfl_ats.data import DataContractError, require_columns

EARTH_RADIUS_MI = 3958.8
DEFAULT_STADIUM_COORDINATES_PATH = (
    Path(__file__).resolve().parents[2] / "registry" / "stadium_coordinates.json"
)

# Every US venue in the checked-in NFL registry uses one of these IANA zones.
# A venue in any other zone is international from the NFL's US perspective.
DOMESTIC_TIME_ZONES = frozenset(
    {
        "America/Chicago",
        "America/Denver",
        "America/Los_Angeles",
        "America/New_York",
        "America/Phoenix",
    }
)

REQUIRED_SCHEDULE_COLUMNS = (
    "game_id",
    "season",
    "gameday",
    "home_team",
    "away_team",
    "stadium",
    "location",
)


@dataclass(frozen=True)
class StadiumLocation:
    """Validated physical venue metadata."""

    latitude: float
    longitude: float
    timezone: str
    city: str

    @property
    def is_international(self) -> bool:
        return self.timezone not in DOMESTIC_TIME_ZONES


@dataclass(frozen=True)
class StadiumCoordinateRegistry:
    """Immutable validated registry plus its audit provenance."""

    venues: Mapping[str, StadiumLocation]
    source: str
    sha256: str


def _finite_number(raw: object, *, field: str, stadium: str) -> float:
    if isinstance(raw, bool):
        raise DataContractError(f"stadium {stadium!r} has non-numeric {field}")
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"stadium {stadium!r} has non-numeric {field}") from exc
    if not math.isfinite(value):
        raise DataContractError(f"stadium {stadium!r} has non-finite {field}")
    return value


def validate_stadium_coordinate_registry(
    raw: Mapping[str, object], *, source: str = "<memory>"
) -> StadiumCoordinateRegistry:
    """Validate raw registry JSON and return an immutable typed registry."""

    venues: dict[str, StadiumLocation] = {}
    for stadium, payload in raw.items():
        if stadium.startswith("_"):
            continue
        if not stadium.strip():
            raise DataContractError("stadium registry contains a blank stadium name")
        if not isinstance(payload, Mapping):
            raise DataContractError(f"stadium {stadium!r} metadata must be an object")
        missing = sorted({"lat", "lon", "tz", "city"}.difference(payload))
        if missing:
            raise DataContractError(
                f"stadium {stadium!r} metadata is missing: {', '.join(missing)}"
            )

        latitude = _finite_number(payload["lat"], field="lat", stadium=stadium)
        longitude = _finite_number(payload["lon"], field="lon", stadium=stadium)
        if not -90.0 <= latitude <= 90.0:
            raise DataContractError(f"stadium {stadium!r} lat must be between -90 and 90")
        if not -180.0 <= longitude <= 180.0:
            raise DataContractError(f"stadium {stadium!r} lon must be between -180 and 180")

        timezone = payload["tz"]
        city = payload["city"]
        if not isinstance(timezone, str) or not timezone.strip():
            raise DataContractError(f"stadium {stadium!r} tz must be a non-empty string")
        if not isinstance(city, str) or not city.strip():
            raise DataContractError(f"stadium {stadium!r} city must be a non-empty string")
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise DataContractError(
                f"stadium {stadium!r} has unknown IANA timezone {timezone!r}"
            ) from exc
        venues[stadium] = StadiumLocation(latitude, longitude, timezone, city)

    if not venues:
        raise DataContractError("stadium registry contains no venue entries")

    canonical = {
        name: {
            "lat": venue.latitude,
            "lon": venue.longitude,
            "tz": venue.timezone,
            "city": venue.city,
        }
        for name, venue in sorted(venues.items())
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return StadiumCoordinateRegistry(MappingProxyType(venues), source, digest)


def load_stadium_coordinate_registry(
    path: Path = DEFAULT_STADIUM_COORDINATES_PATH,
) -> StadiumCoordinateRegistry:
    """Load and validate the checked-in stadium coordinate registry."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"unable to load stadium registry {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise DataContractError(f"stadium registry {path} must contain a JSON object")
    return validate_stadium_coordinate_registry(raw, source=str(path.resolve()))


def haversine_mi(origin: StadiumLocation, destination: StadiumLocation) -> float:
    """Great-circle distance in miles between two validated venues."""

    phi1 = math.radians(origin.latitude)
    phi2 = math.radians(destination.latitude)
    dphi = math.radians(destination.latitude - origin.latitude)
    dlambda = math.radians(destination.longitude - origin.longitude)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def _offset_hours(venue: StadiumLocation, gameday: pd.Timestamp) -> float:
    # Noon avoids DST's skipped/repeated overnight wall-clock interval while
    # retaining the correct game-date-specific offset.
    local_noon = datetime.combine(gameday.date(), time(hour=12), tzinfo=ZoneInfo(venue.timezone))
    offset = local_noon.utcoffset()
    if offset is None:  # pragma: no cover - ZoneInfo always supplies one
        raise DataContractError(f"unable to resolve UTC offset for {venue.timezone}")
    return offset.total_seconds() / 3600.0


def _canonical_team(raw: object) -> str:
    team = str(raw).strip()
    return TEAM_ABBREVIATION_ALIASES.get(team, team)


def _validate_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    require_columns(schedules, REQUIRED_SCHEDULE_COLUMNS, "travel geometry schedules")
    frame = schedules.copy()
    if frame["game_id"].isna().any() or frame["game_id"].astype(str).str.strip().eq("").any():
        raise DataContractError("travel geometry schedules contains a blank game_id")
    frame["game_id"] = frame["game_id"].astype(str)
    duplicates = frame.loc[frame["game_id"].duplicated(), "game_id"].unique()
    if len(duplicates):
        raise DataContractError(
            f"travel geometry schedules contains duplicate game_id values: {duplicates[:5]}"
        )

    seasons = pd.to_numeric(frame["season"], errors="coerce")
    if seasons.isna().any() or (~np.isfinite(seasons)).any() or (seasons % 1.0 != 0.0).any():
        raise DataContractError("travel geometry schedules contains an invalid season")
    frame["season"] = seasons.astype(int)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    if frame["gameday"].isna().any():
        raise DataContractError("travel geometry schedules contains an invalid gameday")

    for side in ("home", "away"):
        if frame[f"{side}_team"].isna().any():
            raise DataContractError(f"travel geometry schedules contains a null {side}_team")
        frame[f"{side}_team"] = frame[f"{side}_team"].map(_canonical_team)
        if frame[f"{side}_team"].eq("").any():
            raise DataContractError(f"travel geometry schedules contains a blank {side}_team")
    if frame["home_team"].eq(frame["away_team"]).any():
        raise DataContractError("travel geometry schedules contains identical home and away teams")

    locations = frame["location"].astype("string").str.strip().str.lower()
    if locations.isna().any() or not locations.isin({"home", "neutral"}).all():
        invalid = sorted(
            frame.loc[~locations.isin({"home", "neutral"}), "location"].astype(str).unique()
        )
        raise DataContractError(
            f"travel geometry schedules contains invalid location values: {invalid[:5]}"
        )
    frame["location_norm"] = locations

    ambiguous = pd.concat(
        [
            frame[["season", "gameday", "home_team"]].rename(columns={"home_team": "team"}),
            frame[["season", "gameday", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).duplicated(["season", "gameday", "team"])
    if ambiguous.any():
        raise DataContractError(
            "travel geometry schedules contains multiple games for one team on one gameday"
        )
    return frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _latest_home_venue(
    histories: Mapping[tuple[str, int], tuple[list[tuple[pd.Timestamp, str]], list[str]]],
    *,
    team: str,
    season: int,
    gameday: pd.Timestamp,
    game_id: str,
) -> str | None:
    history = histories.get((team, season))
    if history is None:
        return None
    keys, stadiums = history
    position = bisect_right(keys, (gameday, game_id)) - 1
    return stadiums[position] if position >= 0 else None


def build_travel_geometry_features(
    schedules: pd.DataFrame,
    registry: StadiumCoordinateRegistry,
    *,
    strict_venues: bool = True,
) -> pd.DataFrame:
    """Build one deterministic, schedule-only travel row per game.

    Positive time-zone change means the team moves east / advances its body
    clock; negative means west / delays it. Body-clock direction is the sign
    of that value (``+1`` eastbound, ``0`` unchanged, ``-1`` westbound).

    A team's home origin is never learned from a later row. If no same-season
    true-home venue exists at or before the game, its current and prior travel
    quantities remain ``NaN``. With ``strict_venues=True`` (the default), any
    named game venue absent from the validated registry is a contract error.
    """

    frame = _validate_schedules(schedules)
    named_stadiums = frame["stadium"].dropna().astype(str).str.strip()
    unresolved = sorted(set(named_stadiums).difference(registry.venues))
    if strict_venues and unresolved:
        raise DataContractError(f"unresolved stadium names: {unresolved[:10]}")

    home_histories: dict[tuple[str, int], tuple[list[tuple[pd.Timestamp, str]], list[str]]] = {}
    true_home = frame.loc[frame["location_norm"].eq("home")]
    for key, group in true_home.groupby(["home_team", "season"], sort=False):
        team, season = cast("tuple[Any, Any]", key)
        valid = group.loc[group["stadium"].notna()].sort_values(["gameday", "game_id"])
        keys = list(zip(valid["gameday"], valid["game_id"], strict=True))
        stadiums = valid["stadium"].astype(str).str.strip().tolist()
        home_histories[(str(team), int(season))] = (keys, stadiums)

    rows: list[dict[str, Any]] = []
    previous_distance: dict[tuple[str, int], float] = {}
    for game in frame.itertuples(index=False):
        game_id = str(game.game_id)
        season = int(cast("Any", game.season))
        gameday = pd.Timestamp(cast("Any", game.gameday))
        venue_name = str(game.stadium).strip() if pd.notna(game.stadium) else ""
        venue = registry.venues.get(venue_name)

        row: dict[str, Any] = {
            "game_id": game_id,
            "travel_international_game": (
                float(venue.is_international) if venue is not None else math.nan
            ),
            "travel_neutral_site": float(game.location_norm == "neutral"),
        }
        current_distances: dict[str, float] = {}
        for side in ("home", "away"):
            team = str(getattr(game, f"{side}_team"))
            key = (team, season)
            origin_name = _latest_home_venue(
                home_histories,
                team=team,
                season=season,
                gameday=gameday,
                game_id=game_id,
            )
            origin = registry.venues.get(origin_name) if origin_name is not None else None
            if origin is None or venue is None:
                distance = math.nan
                tz_change = math.nan
                direction = math.nan
            else:
                distance = haversine_mi(origin, venue)
                tz_change = _offset_hours(venue, gameday) - _offset_hours(origin, gameday)
                direction = float(np.sign(tz_change))
            row[f"travel_{side}_distance_mi"] = distance
            row[f"travel_{side}_tz_change_hours"] = tz_change
            row[f"travel_{side}_body_clock_direction"] = direction
            row[f"travel_{side}_prior_game_distance_mi"] = previous_distance.get(key, math.nan)
            current_distances[side] = distance

        rows.append(row)
        for side in ("home", "away"):
            team = str(getattr(game, f"{side}_team"))
            previous_distance[(team, season)] = current_distances[side]

    result = pd.DataFrame(rows, columns=["game_id", *TRAVEL_GEOMETRY_FEATURE_COLUMNS])
    for column in TRAVEL_GEOMETRY_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    result.attrs["travel_geometry_provenance"] = {
        "registry_source": registry.source,
        "registry_sha256": registry.sha256,
        "schedule_columns_read": list(REQUIRED_SCHEDULE_COLUMNS),
        "outcome_columns_read": [],
        "home_origin_rule": "latest same-season true-home venue at or before decision row",
        "timezone_rule": "game-date IANA UTC offset; destination minus origin",
    }
    return result


def add_travel_geometry_features(
    games: pd.DataFrame,
    schedules: pd.DataFrame,
    registry: StadiumCoordinateRegistry,
    *,
    strict_venues: bool = True,
) -> pd.DataFrame:
    """Attach the family to caller-selected games without changing row order."""

    require_columns(games, ("game_id",), "travel geometry games")
    if games["game_id"].isna().any() or games["game_id"].astype(str).duplicated().any():
        raise DataContractError("travel geometry games requires unique non-null game_id values")
    features = build_travel_geometry_features(schedules, registry, strict_venues=strict_venues)
    result = games.copy()
    result["game_id"] = result["game_id"].astype(str)
    merged = result.merge(features, on="game_id", how="left", sort=False, validate="one_to_one")
    merged.attrs["travel_geometry_provenance"] = features.attrs["travel_geometry_provenance"]
    return merged
