"""Immutable player data and strictly pregame lineup-strength features."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from nfl_ats.availability import (
    availability_rate_lookup,
    canonicalize_availability_rates,
    fixed_unavailability,
    learned_unavailability,
)
from nfl_ats.constants import (
    PLAYER_ALL_STATE_METRICS,
    PLAYER_INJURY_STATE_METRICS,
    PLAYER_PARTICIPATION_STATE_METRICS,
    PLAYER_VALUE_STATE_METRICS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.participation import canonicalize_participation_ratings
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS, season_scope_mask
from nfl_ats.quarterbacks import build_qb_game_metrics, build_qb_states

PLAYER_DATA_VERSION = "v1"
PLAYER_FEATURE_VERSION = "v2"
PLAYER_PARTICIPATION_FEATURE_VERSION = "v3-participation-v1"
PLAYER_AVAILABILITY_FEATURE_VERSION = "v3-availability-v1"
PLAYER_VALUE_DATA_VERSION = "v1"

INJURY_REQUIRED_COLUMNS = (
    "season",
    "game_type",
    "team",
    "week",
    "gsis_id",
    "position",
    "report_status",
    "practice_status",
    "date_modified",
)
ROSTER_REQUIRED_COLUMNS = (
    "season",
    "team",
    "position",
    "status",
    "full_name",
    "gsis_id",
    "pfr_id",
    "years_exp",
    "week",
    "game_type",
)
SNAP_REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "game_type",
    "week",
    "player",
    "pfr_player_id",
    "position",
    "team",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
)
PLAYER_STATS_REQUIRED_COLUMNS = (
    "player_id",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "position",
    "rushing_epa",
    "receiving_epa",
    "def_tackles_for_loss",
    "def_fumbles_forced",
    "def_sacks",
    "def_qb_hits",
    "def_interceptions",
    "def_pass_defended",
)

_ACTIVE_ROSTER_STATUSES = frozenset(("ACT", "INA"))
_OFFENSIVE_LINE = frozenset(("C", "G", "OG", "OL", "OT", "T"))
_SKILL = frozenset(("FB", "HB", "QB", "RB", "TE", "WR"))
_FRONT = frozenset(("DE", "DL", "DT", "EDGE", "ILB", "LB", "NT", "OLB"))
_SECONDARY = frozenset(("CB", "DB", "FS", "S", "SAF", "SS"))
_REPLACEMENT_QB_EPA = -0.15


@dataclass(frozen=True)
class PlayerSnapshot:
    """An immutable injury, weekly-roster, and player-snap snapshot."""

    snapshot_id: str
    root: Path
    injury_seasons: tuple[int, ...]
    roster_seasons: tuple[int, ...]
    snap_seasons: tuple[int, ...]

    @property
    def injuries_path(self) -> Path:
        return self.root / "injuries.parquet"

    @property
    def rosters_path(self) -> Path:
        return self.root / "weekly_rosters.parquet"

    @property
    def snaps_path(self) -> Path:
        return self.root / "snap_counts.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"


@dataclass(frozen=True)
class PlayerValueSnapshot:
    """Immutable weekly player-stat source used for lagged player values."""

    snapshot_id: str
    root: Path
    seasons: tuple[int, ...]

    @property
    def stats_path(self) -> Path:
        return self.root / "player_stats.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"


@dataclass(frozen=True)
class _PlayerLineup:
    shares: dict[str, dict[str, float]]
    roles: tuple[tuple[str | None, float, float, float, str], ...]


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        converted = frame.to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted
    raise TypeError(f"Unsupported dataframe type: {type(frame)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_seasons(seasons: list[int], label: str) -> None:
    if not seasons or seasons != sorted(set(seasons)):
        raise ValueError(f"{label} seasons must be non-empty, unique, and sorted")


def canonicalize_injuries(frame: pd.DataFrame, *, include_postseason: bool = False) -> pd.DataFrame:
    """Normalize injury revisions while preserving their availability timestamp.

    nflverse injuries carry per-round ``game_type`` codes (REG/WC/DIV/CON/SB).
    ``include_postseason`` keeps the playoff rounds alongside the regular
    season; the default reproduces the historical regular-season-only frame
    exactly.
    """

    require_columns(frame, INJURY_REQUIRED_COLUMNS, "injuries")
    result = frame.loc[:, list(INJURY_REQUIRED_COLUMNS)].copy()
    result = result.loc[
        season_scope_mask(
            result["game_type"],
            include_postseason=include_postseason,
            dataset="injuries",
            column="game_type",
        )
    ].copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["week"] = pd.to_numeric(result["week"], errors="coerce")
    result["date_modified"] = pd.to_datetime(result["date_modified"], errors="coerce", utc=True)
    result["team"] = result["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    result["gsis_id"] = result["gsis_id"].astype("string")
    result["position"] = result["position"].astype("string").str.upper()
    result = result.loc[
        result["season"].notna()
        & result["week"].notna()
        & result["team"].notna()
        & result["gsis_id"].notna()
        & result["date_modified"].notna()
    ].copy()
    result["season"] = result["season"].astype(int)
    result["week"] = result["week"].astype(int)
    result = result.drop_duplicates().sort_values(
        ["season", "week", "team", "gsis_id", "date_modified"]
    )
    return result.reset_index(drop=True)


def canonicalize_rosters(frame: pd.DataFrame, *, include_postseason: bool = False) -> pd.DataFrame:
    """Normalize weekly rosters; their week is not treated as an observation timestamp.

    Playoff roster weeks continue past the regular-season maximum, so keeping
    them under ``include_postseason`` cannot collide with a regular-season
    season/week/team/player key.
    """

    require_columns(frame, ROSTER_REQUIRED_COLUMNS, "weekly_rosters")
    result = frame.loc[:, list(ROSTER_REQUIRED_COLUMNS)].copy()
    result = result.loc[
        season_scope_mask(
            result["game_type"],
            include_postseason=include_postseason,
            dataset="weekly_rosters",
            column="game_type",
        )
    ].copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["week"] = pd.to_numeric(result["week"], errors="coerce")
    result["years_exp"] = pd.to_numeric(result["years_exp"], errors="coerce")
    result["team"] = result["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    for column in ("position", "status"):
        result[column] = result[column].astype("string").str.upper()
    for column in ("gsis_id", "pfr_id"):
        result[column] = result[column].astype("string")
    result = result.loc[
        result["season"].notna()
        & result["week"].notna()
        & result["team"].notna()
        & result["gsis_id"].notna()
    ].copy()
    result["season"] = result["season"].astype(int)
    result["week"] = result["week"].astype(int)
    result = result.drop_duplicates().sort_values(["season", "week", "team", "gsis_id", "status"])
    return result.drop_duplicates(["season", "week", "team", "gsis_id"], keep="first").reset_index(
        drop=True
    )


def canonicalize_snaps(frame: pd.DataFrame, *, include_postseason: bool = False) -> pd.DataFrame:
    """Normalize realized player-game snaps, which may affect later games only."""

    require_columns(frame, SNAP_REQUIRED_COLUMNS, "snap_counts")
    result = frame.loc[:, list(SNAP_REQUIRED_COLUMNS)].copy()
    result = result.loc[
        season_scope_mask(
            result["game_type"],
            include_postseason=include_postseason,
            dataset="snap_counts",
            column="game_type",
        )
    ].copy()
    for column in ("season", "week", *SNAP_REQUIRED_COLUMNS[8:]):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["team"] = result["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    result["pfr_player_id"] = result["pfr_player_id"].astype("string")
    result["position"] = result["position"].astype("string").str.upper()
    result = result.loc[
        result["game_id"].notna()
        & result["season"].notna()
        & result["week"].notna()
        & result["team"].notna()
        & result["pfr_player_id"].notna()
    ].copy()
    result["season"] = result["season"].astype(int)
    result["week"] = result["week"].astype(int)
    for column in ("offense_pct", "defense_pct", "st_pct"):
        values = result[column].where(result[column].le(1.5), result[column] / 100.0)
        result[column] = values.clip(lower=0.0, upper=1.0)
    result = result.drop_duplicates().sort_values(
        ["season", "week", "game_id", "team", "pfr_player_id"]
    )
    duplicated = result.duplicated(["game_id", "team", "pfr_player_id"])
    if duplicated.any():
        raise DataContractError("snap_counts contains duplicate game/team/player rows")
    return result.reset_index(drop=True)


def canonicalize_player_stats(
    frame: pd.DataFrame, *, include_postseason: bool = False
) -> pd.DataFrame:
    """Normalize weekly player production; every value is a postgame outcome.

    Weekly player stats use a ``season_type`` column whose postseason value is
    the single code POST, not the per-round codes the injury/roster/snap feeds
    use.
    """

    require_columns(frame, PLAYER_STATS_REQUIRED_COLUMNS, "player_stats")
    result = frame.loc[:, list(PLAYER_STATS_REQUIRED_COLUMNS)].copy()
    result = result.loc[
        season_scope_mask(
            result["season_type"],
            include_postseason=include_postseason,
            dataset="player_stats",
            column="season_type",
        )
    ].copy()
    numeric = (
        "season",
        "week",
        "rushing_epa",
        "receiving_epa",
        "def_tackles_for_loss",
        "def_fumbles_forced",
        "def_sacks",
        "def_qb_hits",
        "def_interceptions",
        "def_pass_defended",
    )
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["team"] = result["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    result["player_id"] = result["player_id"].astype("string")
    result["position"] = result["position"].astype("string").str.upper()
    result = result.loc[
        result["season"].notna()
        & result["week"].notna()
        & result["game_id"].notna()
        & result["team"].notna()
        & result["player_id"].notna()
    ].copy()
    result["season"] = result["season"].astype(int)
    result["week"] = result["week"].astype(int)
    result[list(numeric[2:])] = result.loc[:, list(numeric[2:])].fillna(0.0)
    result = result.drop_duplicates().sort_values(
        ["season", "week", "game_id", "team", "player_id"]
    )
    duplicated = result.duplicated(["game_id", "team", "player_id"])
    if duplicated.any():
        raise DataContractError("player_stats contains duplicate game/team/player rows")
    return result.reset_index(drop=True)


def write_player_snapshot(
    injuries: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame,
    raw_root: Path,
    injury_seasons: list[int],
    roster_seasons: list[int],
    snap_seasons: list[int],
    snapshot_id: str | None = None,
    *,
    include_postseason: bool = False,
) -> PlayerSnapshot:
    """Write the three player sources and their hashes as one immutable snapshot."""

    _valid_seasons(injury_seasons, "Injury")
    _valid_seasons(roster_seasons, "Roster")
    _valid_seasons(snap_seasons, "Snap")
    identifier = snapshot_id or run_id()
    destination = raw_root / identifier
    if destination.exists():
        raise FileExistsError(f"Player snapshot already exists: {destination}")
    canonical_injuries = canonicalize_injuries(injuries, include_postseason=include_postseason)
    canonical_rosters = canonicalize_rosters(rosters, include_postseason=include_postseason)
    canonical_snaps = canonicalize_snaps(snaps, include_postseason=include_postseason)
    snapshot = PlayerSnapshot(
        identifier,
        destination,
        tuple(injury_seasons),
        tuple(roster_seasons),
        tuple(snap_seasons),
    )
    atomic_parquet(canonical_injuries, snapshot.injuries_path)
    atomic_parquet(canonical_rosters, snapshot.rosters_path)
    atomic_parquet(canonical_snaps, snapshot.snaps_path)
    files = {}
    for name, path, frame in (
        ("injuries", snapshot.injuries_path, canonical_injuries),
        ("weekly_rosters", snapshot.rosters_path, canonical_rosters),
        ("snap_counts", snapshot.snaps_path, canonical_snaps),
    ):
        files[name] = {
            "path": path.name,
            "rows": len(frame),
            "columns": frame.columns.tolist(),
            "sha256": _sha256(path),
        }
    manifest = {
        "snapshot_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "nflverse injuries, weekly rosters, and snap counts via nflreadpy",
        "contract_version": PLAYER_DATA_VERSION,
        "include_postseason": bool(include_postseason),
        "availability_contract": {
            "injuries": "latest revision with date_modified <= decision timestamp",
            "weekly_rosters": "strictly earlier season/week only",
            "snap_counts": "strictly earlier completed games only",
        },
        "injury_seasons": injury_seasons,
        "roster_seasons": roster_seasons,
        "snap_seasons": snap_seasons,
        "files": files,
    }
    atomic_json(manifest, snapshot.manifest_path)
    return snapshot


def fetch_player_snapshot(
    injury_seasons: list[int],
    roster_seasons: list[int],
    snap_seasons: list[int],
    raw_root: Path,
    *,
    include_postseason: bool = False,
) -> PlayerSnapshot:
    """Download historically feasible player sources into an immutable snapshot."""

    _valid_seasons(injury_seasons, "Injury")
    _valid_seasons(roster_seasons, "Roster")
    _valid_seasons(snap_seasons, "Snap")
    import nflreadpy as nfl

    injuries = _to_pandas(nfl.load_injuries(seasons=injury_seasons))
    rosters = _to_pandas(nfl.load_rosters_weekly(seasons=roster_seasons))
    snaps = _to_pandas(nfl.load_snap_counts(seasons=snap_seasons))
    return write_player_snapshot(
        injuries,
        rosters,
        snaps,
        raw_root,
        injury_seasons,
        roster_seasons,
        snap_seasons,
        include_postseason=include_postseason,
    )


def player_snapshot_from_root(root: Path) -> PlayerSnapshot:
    import json

    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Player manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return PlayerSnapshot(
        str(payload["snapshot_id"]),
        root,
        tuple(int(value) for value in payload["injury_seasons"]),
        tuple(int(value) for value in payload["roster_seasons"]),
        tuple(int(value) for value in payload["snap_seasons"]),
    )


def latest_player_snapshot(raw_root: Path) -> PlayerSnapshot:
    manifests = sorted(raw_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No player snapshots found in {raw_root}")
    return player_snapshot_from_root(manifests[-1].parent)


def _scoped(
    frame: pd.DataFrame, dataset: str, column: str, *, include_postseason: bool
) -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    keep = season_scope_mask(
        frame[column],
        include_postseason=include_postseason,
        dataset=dataset,
        column=column,
    )
    return frame.loc[keep].reset_index(drop=True)


def load_player_snapshot(
    snapshot: PlayerSnapshot,
    *,
    include_postseason: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three player sources, regular season only unless asked otherwise.

    ``enrich_with_player_features`` re-canonicalizes these frames with the same
    regular-season default, so this is belt-and-braces; it also protects the
    callers that use the raw frames directly (availability outcomes, research
    notebooks) from a postseason-inclusive snapshot on disk.
    """

    for path in (snapshot.injuries_path, snapshot.rosters_path, snapshot.snaps_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing player snapshot data: {path}")
    return (
        _scoped(
            pd.read_parquet(snapshot.injuries_path),
            "injuries snapshot",
            "game_type",
            include_postseason=include_postseason,
        ),
        _scoped(
            pd.read_parquet(snapshot.rosters_path),
            "weekly_rosters snapshot",
            "game_type",
            include_postseason=include_postseason,
        ),
        _scoped(
            pd.read_parquet(snapshot.snaps_path),
            "snap_counts snapshot",
            "game_type",
            include_postseason=include_postseason,
        ),
    )


def write_player_value_snapshot(
    stats: pd.DataFrame,
    raw_root: Path,
    seasons: list[int],
    snapshot_id: str | None = None,
    *,
    include_postseason: bool = False,
) -> PlayerValueSnapshot:
    """Persist weekly player statistics with provenance and a content hash."""

    _valid_seasons(seasons, "Player-stat")
    identifier = snapshot_id or run_id()
    destination = raw_root / identifier
    if destination.exists():
        raise FileExistsError(f"Player-value snapshot already exists: {destination}")
    canonical = canonicalize_player_stats(stats, include_postseason=include_postseason)
    snapshot = PlayerValueSnapshot(identifier, destination, tuple(seasons))
    atomic_parquet(canonical, snapshot.stats_path)
    manifest = {
        "snapshot_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "nflverse weekly player stats via nflreadpy",
        "contract_version": PLAYER_VALUE_DATA_VERSION,
        "include_postseason": bool(include_postseason),
        "availability_contract": "strictly earlier completed games only",
        "seasons": seasons,
        "file": {
            "path": snapshot.stats_path.name,
            "rows": len(canonical),
            "columns": canonical.columns.tolist(),
            "sha256": _sha256(snapshot.stats_path),
        },
    }
    atomic_json(manifest, snapshot.manifest_path)
    return snapshot


def fetch_player_value_snapshot(
    seasons: list[int],
    raw_root: Path,
    *,
    include_postseason: bool = False,
) -> PlayerValueSnapshot:
    """Download maintained weekly player production for lagged value estimates."""

    _valid_seasons(seasons, "Player-stat")
    import nflreadpy as nfl

    stats = _to_pandas(nfl.load_player_stats(seasons=seasons, summary_level="week"))
    return write_player_value_snapshot(
        stats, raw_root, seasons, include_postseason=include_postseason
    )


def player_value_snapshot_from_root(root: Path) -> PlayerValueSnapshot:
    import json

    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Player-value manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return PlayerValueSnapshot(
        str(payload["snapshot_id"]),
        root,
        tuple(int(value) for value in payload["seasons"]),
    )


def latest_player_value_snapshot(raw_root: Path) -> PlayerValueSnapshot:
    manifests = sorted(raw_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No player-value snapshots found in {raw_root}")
    return player_value_snapshot_from_root(manifests[-1].parent)


def load_player_value_snapshot(
    snapshot: PlayerValueSnapshot,
    *,
    include_postseason: bool = False,
) -> pd.DataFrame:
    if not snapshot.stats_path.is_file():
        raise FileNotFoundError(f"Missing player-value snapshot data: {snapshot.stats_path}")
    return _scoped(
        pd.read_parquet(snapshot.stats_path),
        "player_stats snapshot",
        "season_type",
        include_postseason=include_postseason,
    )


def _stable_crosswalk(rosters: pd.DataFrame) -> dict[str, str]:
    links = rosters.loc[
        rosters["gsis_id"].notna() & rosters["pfr_id"].notna(), ["pfr_id", "gsis_id"]
    ].copy()
    links["pfr_id"] = links["pfr_id"].astype(str)
    links["gsis_id"] = links["gsis_id"].astype(str)
    if links.empty:
        return {}
    counts = links.value_counts().rename("appearances").reset_index()
    selected = counts.sort_values(
        ["pfr_id", "appearances", "gsis_id"], ascending=[True, False, True]
    ).drop_duplicates("pfr_id")
    return dict(zip(selected["pfr_id"], selected["gsis_id"], strict=True))


def _normalized_player_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return "".join(character for character in text.lower() if character.isalnum())


def attach_snap_player_ids(snaps: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Link PFR snap identities to GSIS IDs without using player performance."""

    result = snaps.copy()
    result["gsis_id"] = result["pfr_player_id"].astype(str).map(_stable_crosswalk(rosters))
    result["normalized_name"] = result["player"].map(_normalized_player_name)
    names = rosters.loc[rosters["full_name"].notna() & rosters["gsis_id"].notna()].copy()
    names["normalized_name"] = names["full_name"].map(_normalized_player_name)
    counts = (
        names.groupby(
            ["season", "team", "normalized_name", "gsis_id"],
            sort=False,
            observed=True,
        )
        .size()
        .rename("appearances")
        .reset_index()
    )
    season_team_names = counts.sort_values(
        ["season", "team", "normalized_name", "appearances", "gsis_id"],
        ascending=[True, True, True, False, True],
    ).drop_duplicates(["season", "team", "normalized_name"])
    result = result.merge(
        season_team_names[["season", "team", "normalized_name", "gsis_id"]],
        on=["season", "team", "normalized_name"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_name_match"),
    )
    result["gsis_id"] = result["gsis_id"].fillna(result.pop("gsis_id_name_match"))

    # Trades can put the roster and the game team on opposite sides of a weekly
    # snapshot. A global name match is accepted only when that normalized name
    # has referred to exactly one GSIS identity in the complete identity table.
    unique_names = names.groupby("normalized_name", observed=True)["gsis_id"].agg(
        ["nunique", "first"]
    )
    global_map = unique_names.loc[unique_names["nunique"].eq(1), "first"].to_dict()
    result["gsis_id"] = result["gsis_id"].fillna(result["normalized_name"].map(global_map))
    return result.drop(columns="normalized_name")


def _injury_unavailability(injury: pd.Series) -> float:
    learned = injury.get("_unavailability")
    if learned is not None and pd.notna(learned):
        return float(learned)
    return fixed_unavailability(injury.get("report_status"), injury.get("practice_status"))


def _position_group(position: object) -> str:
    normalized = str(position).strip().upper()
    if normalized in _OFFENSIVE_LINE:
        return "offensive_line"
    if normalized in _SKILL:
        return "skill"
    if normalized in _FRONT:
        return "front"
    if normalized in _SECONDARY:
        return "secondary"
    return "other"


def _compile_lineup(frame: pd.DataFrame) -> _PlayerLineup:
    shares: dict[str, dict[str, float]] = {
        name: {}
        for name in (
            "offense",
            "offensive_line",
            "skill",
            "defense",
            "front",
            "secondary",
            "special_teams",
        )
    }
    roles: list[tuple[str | None, float, float, float, str]] = []
    for row in frame.itertuples(index=False):
        player_key = str(row.player_key)
        group = str(row.position_group)
        offense_value: Any = row.offense_pct
        defense_value: Any = row.defense_pct
        special_value: Any = row.st_pct
        offense = float(offense_value)
        defense = float(defense_value)
        special = float(special_value)
        shares["offense"][player_key] = offense
        shares["defense"][player_key] = defense
        shares["special_teams"][player_key] = special
        if group in ("offensive_line", "skill"):
            shares[group][player_key] = offense
        elif group in ("front", "secondary"):
            shares[group][player_key] = defense
        gsis_id = None if pd.isna(row.gsis_id) else str(row.gsis_id)
        roles.append((gsis_id, offense, defense, special, group))
    return _PlayerLineup(shares, tuple(roles))


def _lineup_overlap(latest: _PlayerLineup, previous: _PlayerLineup, category: str) -> float:
    latest_shares = latest.shares[category]
    previous_shares = previous.shares[category]
    denominator = sum(latest_shares.values())
    if denominator <= 0:
        return math.nan
    overlap = sum(
        min(share, previous_shares.get(player, 0.0)) for player, share in latest_shares.items()
    )
    return float(np.clip(overlap / denominator, 0.0, 1.0))


def _roster_history(rosters: pd.DataFrame) -> dict[str, list[tuple[tuple[int, int], pd.DataFrame]]]:
    output: dict[str, list[tuple[tuple[int, int], pd.DataFrame]]] = defaultdict(list)
    for (team, season, week), group in rosters.groupby(
        ["team", "season", "week"], sort=True, observed=True
    ):
        output[str(team)].append(((int(str(season)), int(str(week))), group.reset_index(drop=True)))
    return dict(output)


def _prior_rosters(
    history: list[tuple[tuple[int, int], pd.DataFrame]],
    game_key: tuple[int, int],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    eligible = [frame for key, frame in history if key < game_key]
    if not eligible:
        return None, None
    latest = eligible[-1]
    previous = eligible[-2] if len(eligible) >= 2 else None
    return latest, previous


def _active_roster_features(
    latest: pd.DataFrame | None, previous: pd.DataFrame | None
) -> tuple[float, float]:
    if latest is None:
        return math.nan, math.nan
    latest_active = latest.loc[latest["status"].isin(_ACTIVE_ROSTER_STATUSES)].copy()
    experience = float(latest_active["years_exp"].mean()) if not latest_active.empty else math.nan
    if previous is None or latest_active.empty:
        return math.nan, experience
    previous_active = previous.loc[previous["status"].isin(_ACTIVE_ROSTER_STATUSES)]
    prior_ids = set(previous_active["gsis_id"].dropna().astype(str))
    current_ids = set(latest_active["gsis_id"].dropna().astype(str))
    continuity = len(prior_ids & current_ids) / len(prior_ids) if prior_ids else math.nan
    return float(continuity), experience


def _latest_qb_state(
    histories: dict[str, pd.DataFrame], player_id: str, gameday: pd.Timestamp
) -> pd.Series | None:
    history = histories.get(player_id)
    if history is None or history.empty:
        return None
    dates = history["gameday"].to_numpy(dtype="datetime64[ns]")
    position = int(np.searchsorted(dates, np.datetime64(gameday, "ns"), side="left")) - 1
    return None if position < 0 else history.iloc[position]


def _injury_rows_asof(
    injuries_by_game: dict[tuple[int, int, str], pd.DataFrame],
    covered_seasons: set[int],
    season: int,
    week: int,
    team: str,
    decision_at: pd.Timestamp,
) -> pd.DataFrame | None:
    if season not in covered_seasons:
        return None
    rows = injuries_by_game.get((season, week, team))
    if rows is None:
        return pd.DataFrame(columns=INJURY_REQUIRED_COLUMNS)
    visible = rows.loc[rows["date_modified"].le(decision_at)].copy()
    if visible.empty:
        return None
    return visible.sort_values("date_modified").drop_duplicates("gsis_id", keep="last")


def _injury_features(
    visible: pd.DataFrame | None,
    roles: dict[str, dict[str, float | str]],
) -> dict[str, float]:
    if visible is None:
        return dict.fromkeys(PLAYER_INJURY_STATE_METRICS, math.nan)
    totals = dict.fromkeys(PLAYER_INJURY_STATE_METRICS, 0.0)
    for _, injury in visible.iterrows():
        severity = _injury_unavailability(injury)
        role = roles.get(str(injury["gsis_id"]), {})
        offense = float(role.get("offense_pct", 0.0))
        defense = float(role.get("defense_pct", 0.0))
        special = float(role.get("st_pct", 0.0))
        group = str(role.get("position_group", _position_group(injury.get("position"))))
        totals["injury_offense_unavailability"] += severity * offense / 11.0
        totals["injury_defense_unavailability"] += severity * defense / 11.0
        totals["injury_special_teams_unavailability"] += severity * special / 11.0
        if group == "offensive_line":
            totals["injury_offensive_line_unavailability"] += severity * offense / 5.0
        elif group == "skill":
            totals["injury_skill_unavailability"] += severity * offense / 6.0
        elif group == "front":
            totals["injury_front_unavailability"] += severity * defense / 7.0
        elif group == "secondary":
            totals["injury_secondary_unavailability"] += severity * defense / 5.0
    return totals


def _player_value_rate(
    state: dict[str, float], numerator: str, denominator: str, career: str, prior_snaps: float
) -> float:
    snaps = float(state.get(denominator, 0.0))
    if snaps <= 0.0:
        return 0.0
    reliability = float(state.get(career, 0.0)) / (float(state.get(career, 0.0)) + prior_snaps)
    return 100.0 * float(state.get(numerator, 0.0)) / snaps * reliability


def _channel_value_prior(
    player_values: dict[str, dict[str, float]],
    *,
    numerator: str,
    denominator: str,
    career: str,
    prior_snaps: float,
    pool_minimum: int,
) -> float:
    """Point-in-time-safe, data-derived shrinkage target for MOD-06's live arm.

    The mean per-100-unit rate across the currently "experienced" player pool
    for one value channel (``career >= prior_snaps``, i.e. the same threshold
    at which ``_player_value_rate``'s reliability weight already crosses 0.5,
    and a positive current EWMA denominator so a player who has simply never
    recorded a relevant snap does not enter the pool). No constant is
    hand-picked here: the prior itself is recomputed from ``player_values`` as
    it stands, which by construction (this is called before the current
    game's own snaps update player_values, mirroring how
    ``_injury_value_features`` is already called before that same update)
    only reflects strictly-earlier-or-same-day-already-processed games. Falls
    back to 0.0 -- bit-identical to the shrink-to-zero baseline -- when the
    pool is smaller than ``pool_minimum``, the same
    ``MIN_EXPERIENCED_POOL`` guard used by the reviewed CFB precedent
    (``scripts/cfb_james_stein_unit_screen.py``).
    """

    rates = [
        100.0 * state[numerator] / state[denominator]
        for state in player_values.values()
        if state.get(career, 0.0) >= prior_snaps and state.get(denominator, 0.0) > 0.0
    ]
    if len(rates) < pool_minimum:
        return 0.0
    return float(np.mean(rates))


def _player_value_rate_toward_prior(
    state: dict[str, float],
    numerator: str,
    denominator: str,
    career: str,
    prior_snaps: float,
    prior_mean: float,
) -> float:
    """MOD-06 candidate: same reliability weight as ``_player_value_rate``,
    shrunk toward a channel-level, data-derived prior instead of toward zero.

    ``_player_value_rate`` is the special case ``prior_mean == 0.0``:
    ``reliability * raw_rate + (1 - reliability) * 0 == reliability *
    raw_rate``. When the player has no observed snaps in this channel,
    ``career`` is (in every realistic case) also 0, so ``reliability`` is 0
    and the whole expression already converges to ``prior_mean`` -- an
    untested player is valued at the position-channel average, not at
    replacement level, which is the entire point of the hypothesis under
    test. This function is never called on the production default path
    (``value_shrinkage_target="zero"``); ``_player_value_rate`` above is left
    completely untouched so that path stays bit-identical by construction.
    """

    snaps = float(state.get(denominator, 0.0))
    career_value = float(state.get(career, 0.0))
    reliability = career_value / (career_value + prior_snaps)
    raw_rate = 100.0 * float(state.get(numerator, 0.0)) / snaps if snaps > 0.0 else 0.0
    return reliability * raw_rate + (1.0 - reliability) * prior_mean


def _injury_value_features(
    visible: pd.DataFrame | None,
    roles: dict[str, dict[str, float | str]],
    player_values: dict[str, dict[str, float]],
    prior_snaps: float,
    *,
    shrinkage_target: Literal["zero", "position_prior"] = "zero",
    channel_priors: dict[str, float] | None = None,
) -> dict[str, float]:
    if visible is None or not player_values:
        return dict.fromkeys(PLAYER_VALUE_STATE_METRICS, math.nan)
    totals = dict.fromkeys(PLAYER_VALUE_STATE_METRICS, 0.0)
    priors = channel_priors or {}
    for _, injury in visible.iterrows():
        player_id = str(injury["gsis_id"])
        state = player_values.get(player_id)
        if state is None:
            continue
        severity = _injury_unavailability(injury)
        role = roles.get(player_id, {})
        offense_share = float(role.get("offense_pct", 0.0))
        defense_share = float(role.get("defense_pct", 0.0))
        if shrinkage_target == "position_prior":
            skill_value = _player_value_rate_toward_prior(
                state,
                "skill_epa",
                "offense_snaps",
                "career_offense_snaps",
                prior_snaps,
                priors.get("skill", 0.0),
            )
            defense_value = _player_value_rate_toward_prior(
                state,
                "defense_disruption",
                "defense_snaps",
                "career_defense_snaps",
                prior_snaps,
                priors.get("defense", 0.0),
            )
        else:
            skill_value = _player_value_rate(
                state,
                "skill_epa",
                "offense_snaps",
                "career_offense_snaps",
                prior_snaps,
            )
            defense_value = _player_value_rate(
                state,
                "defense_disruption",
                "defense_snaps",
                "career_defense_snaps",
                prior_snaps,
            )
        totals["injury_skill_epa_value_lost"] += severity * offense_share * skill_value
        totals["injury_defense_disruption_value_lost"] += severity * defense_share * defense_value
    return totals


def _injury_participation_value_features(
    visible: pd.DataFrame | None,
    roles: dict[str, dict[str, float | str]],
    ratings: dict[str, tuple[float, float]] | None,
) -> dict[str, float]:
    if visible is None or not ratings:
        return dict.fromkeys(PLAYER_PARTICIPATION_STATE_METRICS, math.nan)
    totals = dict.fromkeys(PLAYER_PARTICIPATION_STATE_METRICS, 0.0)
    for _, injury in visible.iterrows():
        player_id = str(injury["gsis_id"])
        rating = ratings.get(player_id)
        if rating is None:
            continue
        severity = _injury_unavailability(injury)
        role = roles.get(player_id, {})
        totals["injury_offense_participation_value_lost"] += (
            severity * float(role.get("offense_pct", 0.0)) * rating[0]
        )
        totals["injury_defense_participation_value_lost"] += (
            severity * float(role.get("defense_pct", 0.0)) * rating[1]
        )
    return totals


def _update_player_value_state(
    state: dict[str, float],
    *,
    skill_epa: float,
    offense_snaps: float,
    defense_disruption: float,
    defense_snaps: float,
    alpha: float,
) -> None:
    for numerator, denominator, career, value, snaps in (
        (
            "skill_epa",
            "offense_snaps",
            "career_offense_snaps",
            skill_epa,
            offense_snaps,
        ),
        (
            "defense_disruption",
            "defense_snaps",
            "career_defense_snaps",
            defense_disruption,
            defense_snaps,
        ),
    ):
        previous_value = state.get(numerator)
        previous_snaps = state.get(denominator)
        state[numerator] = (
            value if previous_value is None else alpha * value + (1 - alpha) * previous_value
        )
        state[denominator] = (
            snaps if previous_snaps is None else alpha * snaps + (1 - alpha) * previous_snaps
        )
        state[career] = state.get(career, 0.0) + snaps


def enrich_with_player_features(
    games: pd.DataFrame,
    injuries: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame,
    pbp: pd.DataFrame,
    player_stats: pd.DataFrame | None = None,
    participation_ratings: pd.DataFrame | None = None,
    availability_rates: pd.DataFrame | None = None,
    *,
    decision_hours_before_kickoff: int = 24,
    role_span: int = 8,
    qb_span: int = 12,
    qb_min_dropbacks: int = 20,
    offseason_retention: float = 0.75,
    value_span: int = 16,
    value_prior_snaps: float = 200.0,
    value_shrinkage_target: Literal["zero", "position_prior"] = "zero",
    value_js_prior_pool_minimum: int = 20,
) -> pd.DataFrame:
    """Attach conservative expected-lineup features using strictly earlier outcomes.

    Weekly roster rows lack observation timestamps and are therefore delayed by
    one week. Injury revisions require ``date_modified <= decision_at``. Snap
    counts, player production, and quarterback performance from the game being
    predicted are added to state only after that game's features have been emitted.
    Optional participation ratings must be fitted entirely from seasons before
    each target season; their contract is revalidated here before use.

    ``value_shrinkage_target`` (MOD-06's one live arm, docs/mod06_position_prior_shrinkage.md):
    the two ``PLAYER_VALUE_STATE_METRICS`` (``injury_skill_epa_value_lost``,
    ``injury_defense_disruption_value_lost``) shrink a thin player's per-snap
    value rate by ``career / (career + value_prior_snaps)``. The default,
    ``"zero"``, is today's production behaviour, shrinking toward zero
    (worth nothing) -- this code path is untouched by the new argument and
    stays bit-identical. The opt-in candidate, ``"position_prior"``, shrinks
    toward a channel-level prior computed fresh, point-in-time-safe, from the
    league's currently experienced player pool (``_channel_value_prior``)
    instead of zero; ``value_js_prior_pool_minimum`` is the minimum pool size
    below which that prior falls back to 0.0 (bit-identical to the baseline).
    """

    required_games = {
        "game_id",
        "season",
        "week",
        "gameday",
        "kickoff",
        "home_team",
        "away_team",
    }
    missing = sorted(required_games.difference(games.columns))
    if missing:
        raise DataContractError(f"Game features are missing columns: {', '.join(missing)}")
    if decision_hours_before_kickoff < 0:
        raise ValueError("decision_hours_before_kickoff cannot be negative")
    if role_span < 2 or qb_span < 2 or value_span < 2 or qb_min_dropbacks < 1:
        raise ValueError("role/qb spans must be at least 2 and qb_min_dropbacks must be positive")
    if value_prior_snaps < 0:
        raise ValueError("value_prior_snaps cannot be negative")
    if value_shrinkage_target not in ("zero", "position_prior"):
        raise ValueError("value_shrinkage_target must be 'zero' or 'position_prior'")
    if value_js_prior_pool_minimum < 1:
        raise ValueError("value_js_prior_pool_minimum must be positive")
    if not 0.0 <= offseason_retention <= 1.0:
        raise ValueError("offseason_retention must be between zero and one")
    require_columns(pbp, PBP_SNAPSHOT_COLUMNS, "play_by_play snapshot")

    injuries = canonicalize_injuries(injuries)
    rosters = canonicalize_rosters(rosters)
    snaps = canonicalize_snaps(snaps)
    canonical_stats = (
        canonicalize_player_stats(player_stats)
        if player_stats is not None
        else pd.DataFrame(columns=PLAYER_STATS_REQUIRED_COLUMNS)
    )
    canonical_ratings = (
        canonicalize_participation_ratings(participation_ratings)
        if participation_ratings is not None
        else None
    )
    canonical_availability = (
        canonicalize_availability_rates(availability_rates)
        if availability_rates is not None
        else None
    )
    learned_availability_lookup = (
        availability_rate_lookup(canonical_availability)
        if canonical_availability is not None
        else None
    )
    feature_metrics = PLAYER_ALL_STATE_METRICS + (
        PLAYER_PARTICIPATION_STATE_METRICS if canonical_ratings is not None else ()
    )
    result = games.copy().sort_values(["gameday", "game_id"]).reset_index(drop=True)
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    result["kickoff"] = pd.to_datetime(result["kickoff"], errors="coerce", utc=True)
    for column in ("home_team", "away_team"):
        result[column] = result[column].replace(TEAM_ABBREVIATION_ALIASES)

    snaps = attach_snap_player_ids(snaps, rosters)
    snaps["player_key"] = snaps["gsis_id"].fillna("pfr:" + snaps["pfr_player_id"].astype(str))
    snaps["position_group"] = snaps["position"].map(_position_group)
    snap_games = {
        (str(game_id), str(team)): _compile_lineup(group)
        for (game_id, team), group in snaps.groupby(["game_id", "team"], sort=False)
    }
    snap_value_games = {
        (str(game_id), str(team)): group.reset_index(drop=True)
        for (game_id, team), group in snaps.groupby(["game_id", "team"], sort=False)
    }
    player_stats_rows: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in canonical_stats.iterrows():
        key = (str(row["game_id"]), str(row["team"]), str(row["player_id"]))
        player_stats_rows[key] = row
    participation_ratings_by_season: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    if canonical_ratings is not None:
        for rating in canonical_ratings.itertuples(index=False):
            participation_ratings_by_season[int(str(rating.target_season))][
                str(rating.player_id)
            ] = (
                float(str(rating.offense_rating)),
                float(str(rating.defense_rating)),
            )
    roster_groups = _roster_history(rosters)
    injuries_by_game = {
        (int(str(season)), int(str(week)), str(team)): group.sort_values(
            "date_modified"
        ).reset_index(drop=True)
        for (season, week, team), group in injuries.groupby(
            ["season", "week", "team"], sort=False, observed=True
        )
    }
    covered_injury_seasons = set(injuries["season"].astype(int).unique())

    qb_games = build_qb_game_metrics(pbp)
    qb_states = build_qb_states(
        qb_games,
        result,
        span=qb_span,
        min_dropbacks=qb_min_dropbacks,
        offseason_retention=offseason_retention,
    )
    qb_histories = {
        str(player): group.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        for player, group in qb_states.groupby("player_id", sort=False)
    }
    qb_appearances = {
        (str(game_id), str(team)): group.sort_values("qb_dropbacks", ascending=False).reset_index(
            drop=True
        )
        for (game_id, team), group in qb_games.groupby(["game_id", "team"], sort=False)
    }

    for side in ("home", "away"):
        for metric in feature_metrics:
            result[f"{side}_{metric}"] = np.nan
        result[f"{side}_projected_qb_id"] = pd.NA
        result[f"{side}_injury_observed_at"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )

    prior_lineups: dict[str, list[_PlayerLineup]] = defaultdict(list)
    latest_qb_appearance: dict[str, pd.Series] = {}
    role_states: dict[str, dict[str, dict[str, float | str]]] = defaultdict(dict)
    player_value_states: dict[str, dict[str, float]] = {}
    alpha = 2.0 / (role_span + 1.0)
    value_alpha = 2.0 / (value_span + 1.0)

    for index, game in result.iterrows():
        kickoff = game["kickoff"]
        decision_at = (
            pd.NaT
            if pd.isna(kickoff)
            else pd.Timestamp(kickoff) - pd.Timedelta(hours=decision_hours_before_kickoff)
        )
        channel_priors: dict[str, float] | None = None
        if value_shrinkage_target == "position_prior":
            # Computed once per game (both sides read the same snapshot) from
            # player_value_states as it stands BEFORE this game's own snaps
            # update it later in this same iteration -- the identical
            # point-in-time-safety property _injury_value_features already
            # relies on for the injured players themselves.
            channel_priors = {
                "skill": _channel_value_prior(
                    player_value_states,
                    numerator="skill_epa",
                    denominator="offense_snaps",
                    career="career_offense_snaps",
                    prior_snaps=value_prior_snaps,
                    pool_minimum=value_js_prior_pool_minimum,
                ),
                "defense": _channel_value_prior(
                    player_value_states,
                    numerator="defense_disruption",
                    denominator="defense_snaps",
                    career="career_defense_snaps",
                    prior_snaps=value_prior_snaps,
                    pool_minimum=value_js_prior_pool_minimum,
                ),
            }
        for side in ("home", "away"):
            team = str(game[f"{side}_team"])
            history = prior_lineups[team]
            if len(history) >= 2:
                latest, previous = history[-1], history[-2]
                continuity_values = {
                    "offense_lineup_continuity": _lineup_overlap(latest, previous, "offense"),
                    "offensive_line_continuity": _lineup_overlap(
                        latest, previous, "offensive_line"
                    ),
                    "skill_lineup_continuity": _lineup_overlap(latest, previous, "skill"),
                    "defense_lineup_continuity": _lineup_overlap(latest, previous, "defense"),
                    "front_lineup_continuity": _lineup_overlap(latest, previous, "front"),
                    "secondary_lineup_continuity": _lineup_overlap(latest, previous, "secondary"),
                    "special_teams_lineup_continuity": _lineup_overlap(
                        latest, previous, "special_teams"
                    ),
                }
                for metric, value in continuity_values.items():
                    result.at[index, f"{side}_{metric}"] = value

            latest_roster, previous_roster = _prior_rosters(
                roster_groups.get(team, []), (int(game["season"]), int(game["week"]))
            )
            roster_continuity, roster_experience = _active_roster_features(
                latest_roster, previous_roster
            )
            result.at[index, f"{side}_active_roster_continuity"] = roster_continuity
            result.at[index, f"{side}_active_roster_mean_experience"] = roster_experience

            visible_injuries = None
            if pd.notna(decision_at):
                visible_injuries = _injury_rows_asof(
                    injuries_by_game,
                    covered_injury_seasons,
                    int(game["season"]),
                    int(game["week"]),
                    team,
                    pd.Timestamp(decision_at),
                )
            if visible_injuries is not None and learned_availability_lookup is not None:
                visible_injuries = visible_injuries.copy()
                target_season = int(str(game["season"]))
                visible_injuries["_unavailability"] = [
                    learned_unavailability(
                        learned_availability_lookup,
                        target_season=target_season,
                        report_status=report,
                        practice_status=practice,
                        position=position,
                    )
                    for report, practice, position in zip(
                        visible_injuries["report_status"],
                        visible_injuries["practice_status"],
                        visible_injuries["position"],
                        strict=True,
                    )
                ]
            injury_values = _injury_features(visible_injuries, role_states[team])
            injury_values.update(
                _injury_value_features(
                    visible_injuries,
                    role_states[team],
                    player_value_states,
                    value_prior_snaps,
                    shrinkage_target=value_shrinkage_target,
                    channel_priors=channel_priors,
                )
            )
            if canonical_ratings is not None:
                injury_values.update(
                    _injury_participation_value_features(
                        visible_injuries,
                        role_states[team],
                        participation_ratings_by_season.get(int(game["season"])),
                    )
                )
            for metric, value in injury_values.items():
                result.at[index, f"{side}_{metric}"] = value
            if visible_injuries is not None and not visible_injuries.empty:
                result.at[index, f"{side}_injury_observed_at"] = visible_injuries[
                    "date_modified"
                ].max()

            starter = latest_qb_appearance.get(team)
            if starter is not None:
                player_id = str(starter["player_id"])
                result.at[index, f"{side}_projected_qb_id"] = player_id
                qb_state = _latest_qb_state(qb_histories, player_id, pd.Timestamp(game["gameday"]))
                if qb_state is not None:
                    starter_epa = float(qb_state["state_qb_epa_per_dropback"])
                    starter_cpoe = float(qb_state["state_qb_cpoe"])
                    experience = float(qb_state["career_dropbacks"])
                    unavailable = 0.0
                    if visible_injuries is not None and not visible_injuries.empty:
                        match = visible_injuries.loc[visible_injuries["gsis_id"].eq(player_id)]
                        if not match.empty:
                            row = match.iloc[-1]
                            unavailable = _injury_unavailability(row)
                    start_probability = 1.0 - unavailable if np.isfinite(unavailable) else math.nan
                    expected_epa = (
                        start_probability * starter_epa
                        + (1.0 - start_probability) * _REPLACEMENT_QB_EPA
                        if np.isfinite(start_probability)
                        else math.nan
                    )
                    result.at[index, f"{side}_qb_expected_epa_per_dropback"] = expected_epa
                    result.at[index, f"{side}_qb_starter_epa_per_dropback"] = starter_epa
                    result.at[index, f"{side}_qb_starter_cpoe"] = starter_cpoe
                    result.at[index, f"{side}_qb_start_probability"] = start_probability
                    result.at[index, f"{side}_qb_starter_experience_log"] = math.log1p(experience)

        for metric in feature_metrics:
            home_value: Any = result.at[index, f"home_{metric}"]
            away_value: Any = result.at[index, f"away_{metric}"]
            if metric in (
                *PLAYER_INJURY_STATE_METRICS,
                *PLAYER_VALUE_STATE_METRICS,
                *PLAYER_PARTICIPATION_STATE_METRICS,
            ) and (pd.isna(home_value) or pd.isna(away_value)):
                # Without both reports there is no directional injury
                # information. The neutral matchup contrast is zero; the
                # side-level values remain missing for auditing and are not
                # part of the research allowlist.
                result.at[index, f"diff_{metric}"] = 0.0
            else:
                result.at[index, f"diff_{metric}"] = float(home_value) - float(away_value)

        for side in ("home", "away"):
            team = str(game[f"{side}_team"])
            lineup = snap_games.get((str(game["game_id"]), team))
            if lineup is not None:
                prior_lineups[team].append(lineup)
                for gsis_id, offense, defense, special, position_group in lineup.roles:
                    if gsis_id is None:
                        continue
                    state = role_states[team].setdefault(gsis_id, {})
                    for share_column, value in (
                        ("offense_pct", offense),
                        ("defense_pct", defense),
                        ("st_pct", special),
                    ):
                        previous_value = state.get(share_column)
                        state[share_column] = (
                            value
                            if previous_value is None
                            else alpha * value + (1.0 - alpha) * float(previous_value)
                        )
                    state["position_group"] = position_group
            value_rows = snap_value_games.get((str(game["game_id"]), team))
            if value_rows is not None and not canonical_stats.empty:
                for snap in value_rows.itertuples(index=False):
                    raw_player_id: Any = snap.gsis_id
                    if pd.isna(raw_player_id):
                        continue
                    player_id = str(raw_player_id)
                    stats = player_stats_rows.get((str(game["game_id"]), team, player_id))
                    skill_epa = 0.0
                    defense_disruption = 0.0
                    if stats is not None:
                        if str(snap.position).upper() != "QB":
                            skill_epa = float(str(stats["rushing_epa"])) + float(
                                str(stats["receiving_epa"])
                            )
                        defense_disruption = (
                            0.5 * float(str(stats["def_tackles_for_loss"]))
                            + 2.0 * float(str(stats["def_fumbles_forced"]))
                            + 1.5 * float(str(stats["def_sacks"]))
                            + 0.25 * float(str(stats["def_qb_hits"]))
                            + 4.0 * float(str(stats["def_interceptions"]))
                            + 0.5 * float(str(stats["def_pass_defended"]))
                        )
                    value_state = player_value_states.setdefault(player_id, {})
                    _update_player_value_state(
                        value_state,
                        skill_epa=skill_epa,
                        offense_snaps=float(str(snap.offense_snaps)),
                        defense_disruption=defense_disruption,
                        defense_snaps=float(str(snap.defense_snaps)),
                        alpha=value_alpha,
                    )
            appearances = qb_appearances.get((str(game["game_id"]), team))
            if appearances is not None and not appearances.empty:
                latest_qb_appearance[team] = appearances.iloc[0]

    for metric in feature_metrics:
        for side in ("home", "away"):
            result[f"{side}_{metric}"] = pd.to_numeric(result[f"{side}_{metric}"], errors="coerce")
        result[f"diff_{metric}"] = pd.to_numeric(result[f"diff_{metric}"], errors="coerce")
    result["player_feature_version"] = (
        PLAYER_AVAILABILITY_FEATURE_VERSION
        if canonical_availability is not None
        else (
            PLAYER_PARTICIPATION_FEATURE_VERSION
            if canonical_ratings is not None
            else PLAYER_FEATURE_VERSION
        )
    )
    if value_shrinkage_target == "position_prior":
        # Provenance tag only; the "zero" branch above is never touched by
        # this line, so the default table's version string is unaffected.
        result["player_feature_version"] = result["player_feature_version"] + "-js-prior"
    return result.replace([np.inf, -np.inf], np.nan)
