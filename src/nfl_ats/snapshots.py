"""Immutable nflverse snapshot storage and provenance manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Snapshot:
    """Paths and provenance for one atomic source refresh."""

    snapshot_id: str
    root: Path
    schedules_path: Path
    team_stats_path: Path
    manifest_path: Path


def utc_snapshot_id(now: datetime | None = None) -> str:
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def write_snapshot(
    schedules: pd.DataFrame,
    team_stats: pd.DataFrame,
    seasons: list[int],
    raw_root: Path,
    snapshot_id: str | None = None,
    team_stat_seasons: list[int] | None = None,
) -> Snapshot:
    """Write data files first and the manifest last as the commit marker."""

    identifier = snapshot_id or utc_snapshot_id()
    root = raw_root / identifier
    if root.exists():
        raise FileExistsError(f"Snapshot already exists: {root}")
    root.mkdir(parents=True)

    snapshot = Snapshot(
        snapshot_id=identifier,
        root=root,
        schedules_path=root / "schedules.parquet",
        team_stats_path=root / "team_stats.parquet",
        manifest_path=root / "manifest.json",
    )
    _atomic_parquet(schedules, snapshot.schedules_path)
    _atomic_parquet(team_stats, snapshot.team_stats_path)
    resolved_team_stat_seasons = seasons if team_stat_seasons is None else team_stat_seasons
    manifest = {
        "snapshot_id": identifier,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "seasons": seasons,
        "schedule_seasons": seasons,
        "team_stat_seasons": resolved_team_stat_seasons,
        "sources": {
            "loader": "nflreadpy",
            "repository": "https://github.com/nflverse/nflverse-data",
            "schedules": "nflreadpy.load_schedules",
            "team_stats": "nflreadpy.load_team_stats(summary_level='week')",
        },
        "files": {
            "schedules.parquet": {
                "rows": len(schedules),
                "sha256": _sha256(snapshot.schedules_path),
            },
            "team_stats.parquet": {
                "rows": len(team_stats),
                "sha256": _sha256(snapshot.team_stats_path),
            },
        },
    }
    _atomic_json(manifest, snapshot.manifest_path)
    return snapshot


def snapshot_from_root(root: Path) -> Snapshot:
    return Snapshot(
        snapshot_id=root.name,
        root=root,
        schedules_path=root / "schedules.parquet",
        team_stats_path=root / "team_stats.parquet",
        manifest_path=root / "manifest.json",
    )


def latest_snapshot(raw_root: Path) -> Snapshot:
    # data/raw also hosts named source directories (injury_news, officials, ...) whose
    # own manifests must never be mistaken for a schedules snapshot, so a candidate
    # must carry the snapshot payload itself, not just a manifest.json.
    candidates = sorted(
        path
        for path in raw_root.glob("*")
        if (path / "manifest.json").is_file() and (path / "schedules.parquet").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No complete snapshots found under {raw_root}")
    return snapshot_from_root(candidates[-1])


def load_snapshot(snapshot: Snapshot) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not snapshot.manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete snapshot: {snapshot.root}")
    return pd.read_parquet(snapshot.schedules_path), pd.read_parquet(snapshot.team_stats_path)


def load_verified_snapshot(snapshot: Snapshot) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a snapshot only when its manifest still matches both payloads."""

    if not snapshot.manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete snapshot: {snapshot.root}")
    payload = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    if str(payload.get("snapshot_id")) != snapshot.snapshot_id:
        raise ValueError(f"Snapshot manifest identity mismatch: {snapshot.root}")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Snapshot manifest has no file provenance: {snapshot.root}")
    frames: list[pd.DataFrame] = []
    for name, path in (
        ("schedules.parquet", snapshot.schedules_path),
        ("team_stats.parquet", snapshot.team_stats_path),
    ):
        declared = files.get(name)
        if not isinstance(declared, dict):
            raise ValueError(f"Snapshot manifest omits {name}: {snapshot.root}")
        if not path.is_file() or _sha256(path) != str(declared.get("sha256")):
            raise ValueError(f"Snapshot payload hash mismatch for {name}: {snapshot.root}")
        frame = pd.read_parquet(path)
        if len(frame) != int(declared.get("rows", -1)):
            raise ValueError(f"Snapshot payload row-count mismatch for {name}: {snapshot.root}")
        frames.append(frame)
    return frames[0], frames[1]


def describe_snapshot(snapshot: Snapshot) -> dict[str, Any]:
    payload = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    paths = {key: str(value) for key, value in asdict(snapshot).items()}
    return {**paths, "manifest": payload}
