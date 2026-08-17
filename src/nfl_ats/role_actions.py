"""Immutable nflverse weekly action-count snapshot for role-delivery replication.

This is a small, standalone sibling of ``nfl_ats.players``' snapshot machinery
(``PlayerValueSnapshot`` / ``fetch_player_value_snapshot`` / ``_to_pandas``):
same immutable-directory-per-fetch pattern, same "canonicalize before write"
contract, but scoped to exactly the weekly action counts the XLG-04
cross-league role-delivery replication (``nfl_ats.cfb_roles``) needs --
dropback/carry/reception raw counts, not the full player-value feature set.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.provenance import sha256_file

ROLE_ACTIONS_DATA_VERSION = "v1"

# The final, on-disk column contract. ``sacks_taken`` is the renamed
# passer-sacked count (see :func:`canonicalize_role_actions`).
ROLE_ACTIONS_REQUIRED_COLUMNS = (
    "player_id",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "position",
    "attempts",
    "carries",
    "receptions",
    "targets",
    "sacks_taken",
)

_IDENTITY_COLUMNS = (
    "player_id",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "position",
)
_COUNT_COLUMNS = ("attempts", "carries", "receptions", "targets", "sacks_taken")


@dataclass(frozen=True)
class RoleActionsSnapshot:
    """An immutable weekly player action-count snapshot."""

    snapshot_id: str
    root: Path
    seasons: tuple[int, ...]

    @property
    def actions_path(self) -> Path:
        return self.root / "role_actions.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        converted = frame.to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted
    raise TypeError(f"Unsupported dataframe type: {type(frame)!r}")


def _valid_seasons(seasons: list[int]) -> None:
    if not seasons or seasons != sorted(set(seasons)):
        raise ValueError("Seasons must be non-empty, unique, and sorted")


def canonicalize_role_actions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize weekly player action counts to the XLG-04 replication contract.

    Keeps the identity columns plus ``attempts``/``carries``/``receptions``/
    ``targets`` and a renamed ``sacks_taken`` column: nflverse has named the
    passer's times-sacked column either ``sacks_suffered`` (current) or
    ``sacks`` (older versions); whichever is present is kept and renamed
    (``sacks_suffered`` preferred when both exist), and a
    :class:`DataContractError` is raised if neither exists. Restricted to
    ``season_type == "REG"``; count columns are numeric-coerced and
    zero-filled; duplicate ``(game_id, team, player_id)`` rows raise.
    """

    sacks_column = (
        "sacks_suffered"
        if "sacks_suffered" in frame.columns
        else "sacks"
        if "sacks" in frame.columns
        else None
    )
    if sacks_column is None:
        raise DataContractError("player_stats has neither a 'sacks_suffered' nor a 'sacks' column")
    source_columns = (*_IDENTITY_COLUMNS, "attempts", "carries", "receptions", "targets")
    require_columns(frame, (*source_columns, sacks_column), "player_stats")

    result = frame.loc[:, [*source_columns, sacks_column]].rename(
        columns={sacks_column: "sacks_taken"}
    )
    result = result.loc[result["season_type"].eq("REG")].copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["week"] = pd.to_numeric(result["week"], errors="coerce")
    for column in _COUNT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result["team"] = result["team"].astype("string")
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
    result = result.drop_duplicates().sort_values(
        ["season", "week", "game_id", "team", "player_id"]
    )
    duplicated = result.duplicated(["game_id", "team", "player_id"])
    if duplicated.any():
        raise DataContractError("role_actions contains duplicate game/team/player rows")
    return result.loc[:, list(ROLE_ACTIONS_REQUIRED_COLUMNS)].reset_index(drop=True)


def write_role_actions_snapshot(
    frame: pd.DataFrame,
    root: Path,
    seasons: list[int],
    snapshot_id: str | None = None,
) -> RoleActionsSnapshot:
    """Canonicalize and persist one action-count source as an immutable snapshot."""

    _valid_seasons(seasons)
    identifier = snapshot_id or run_id()
    destination = root / identifier
    if destination.exists():
        raise FileExistsError(f"Role-actions snapshot already exists: {destination}")
    canonical = canonicalize_role_actions(frame)
    snapshot = RoleActionsSnapshot(identifier, destination, tuple(seasons))
    atomic_parquet(canonical, snapshot.actions_path)
    manifest = {
        "snapshot_id": identifier,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "seasons": seasons,
        "rows": len(canonical),
        "sha256": sha256_file(snapshot.actions_path),
        "source": "nflreadpy.load_player_stats(summary_level='week')",
    }
    atomic_json(manifest, snapshot.manifest_path)
    return snapshot


def fetch_role_actions_snapshot(root: Path, seasons: Iterable[int]) -> RoleActionsSnapshot:
    """Download nflverse weekly player action counts into an immutable snapshot."""

    season_list = sorted({int(season) for season in seasons})
    _valid_seasons(season_list)
    import nflreadpy as nfl

    frame = _to_pandas(nfl.load_player_stats(seasons=season_list, summary_level="week"))
    return write_role_actions_snapshot(frame, root, season_list)


def role_actions_snapshot_from_root(root: Path) -> RoleActionsSnapshot:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Role-actions manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RoleActionsSnapshot(
        str(payload["snapshot_id"]),
        root,
        tuple(int(value) for value in payload["seasons"]),
    )


def latest_role_actions_snapshot(root: Path) -> RoleActionsSnapshot:
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No role-actions snapshots found in {root}")
    return role_actions_snapshot_from_root(manifests[-1].parent)


def load_role_actions_snapshot(snapshot: RoleActionsSnapshot) -> pd.DataFrame:
    if not snapshot.actions_path.is_file():
        raise FileNotFoundError(f"Missing role-actions snapshot data: {snapshot.actions_path}")
    frame = pd.read_parquet(snapshot.actions_path)
    require_columns(frame, ROLE_ACTIONS_REQUIRED_COLUMNS, "role_actions snapshot")
    return frame
