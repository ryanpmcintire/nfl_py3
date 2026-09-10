from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SNAPSHOT_KEYS: tuple[str, ...] = (
    "source_snapshot",
    "source_pbp_snapshot",
    "source_depth_snapshot",
    "source_player_snapshot",
    "source_player_value_snapshot",
    "source_participation_snapshot",
)

SOURCE_SNAPSHOTS_KEY = "source_snapshots"

DECISION_LINES_KEY = "decision_lines"

UPSTREAM_ABSENT_REASON = "upstream manifest absent"


def manifest_path_for(parquet_path: Path | str) -> Path:

    path = Path(parquet_path)
    return path.with_name(f"{path.stem}.manifest.json")


def _read_manifest(path: Path) -> Mapping[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _parse_capture(snapshot_id: str) -> str | None:

    try:
        instant = datetime.strptime(snapshot_id.strip(), "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    return instant.replace(tzinfo=UTC).isoformat()


def _label_for(path: Path) -> str:

    name = path.name
    if name.endswith(".manifest.json"):
        name = name[: -len(".manifest.json")]
    return name


def inherit_source_snapshots(parent_manifest_paths: Iterable[Path | str]) -> dict[str, Any]:

    merged: dict[str, Any] = {}
    for raw_path in parent_manifest_paths:
        path = Path(raw_path)
        manifest = _read_manifest(path)
        if manifest is None:
            merged[_label_for(path)] = {
                "snapshot_id": None,
                "reason": UPSTREAM_ABSENT_REASON,
            }
            continue

        parent_entries: dict[str, Any] = {}
        upstream = manifest.get(SOURCE_SNAPSHOTS_KEY)
        if isinstance(upstream, Mapping):
            for key, entry in upstream.items():
                if isinstance(entry, Mapping):
                    parent_entries[key] = dict(entry)
        for key in SNAPSHOT_KEYS:
            value = manifest.get(key)
            if value is None:
                continue
            snapshot_id = str(value)
            parent_entries[key] = {
                "snapshot_id": snapshot_id,
                "captured_at": _parse_capture(snapshot_id),
                "manifest_path": str(path),
            }
        own_decision_lines = manifest.get(DECISION_LINES_KEY)
        if isinstance(own_decision_lines, Mapping):
            parent_entries[DECISION_LINES_KEY] = dict(own_decision_lines)
        merged.update(parent_entries)
    return merged


def decision_lines_block(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:

    own = manifest.get(DECISION_LINES_KEY)
    if isinstance(own, Mapping):
        return own
    upstream = manifest.get(SOURCE_SNAPSHOTS_KEY)
    if isinstance(upstream, Mapping):
        inherited = upstream.get(DECISION_LINES_KEY)
        if isinstance(inherited, Mapping):
            return inherited
    return None


def decision_line_week(
    manifest: Mapping[str, Any], season: int | None, week: int | None
) -> Mapping[str, Any] | None:

    if season is None or week is None:
        return None
    block = decision_lines_block(manifest)
    if block is None:
        return None
    weeks = block.get("weeks")
    if not isinstance(weeks, Iterable) or isinstance(weeks, str | bytes | Mapping):
        return None
    shared = {
        key: block[key] for key in ("policy", "builder_module", "builder_version") if key in block
    }
    for entry in weeks:
        if not isinstance(entry, Mapping):
            continue
        try:
            entry_season = int(entry["season"])
            entry_week = int(entry["week"])
        except (KeyError, TypeError, ValueError):
            continue
        if entry_season == season and entry_week == week:
            return {**shared, **entry}
    return None


__all__ = [
    "DECISION_LINES_KEY",
    "SNAPSHOT_KEYS",
    "SOURCE_SNAPSHOTS_KEY",
    "UPSTREAM_ABSENT_REASON",
    "decision_line_week",
    "decision_lines_block",
    "inherit_source_snapshots",
    "manifest_path_for",
]
