"""Cross-manifest source-snapshot inheritance for derived feature tables (ENG-22).

ROADMAP Phase 13's definition of done: "Every enrichment step's manifest
carries the base nflverse ``source_snapshot`` id forward (today only
``game_features.manifest.json`` records it; derived tables record only a
``source_features`` path), so lineage records name a real snapshot instead of
a table digest." Measured 2026-09-04 (see ``docs/feature_lineage.md``): 9 of
15 decision-bearing fields on the live Week 1 card fall back to
``feature_table:sha256`` because every enrichment manifest under
``data/processed/`` records ``source_features`` -- a *path* to the parquet it
enriched -- but not the snapshot id that path's own manifest recorded.

:func:`inherit_source_snapshots` is the fix: given the manifest path(s) a
derived feature table was built directly from, it reads each one and returns
a merged ``source_snapshots`` block naming every snapshot-bearing key
(:data:`SNAPSHOT_KEYS`) that manifest -- or anything *it* already inherited
-- carries. Every enrichment writer in ``nfl_ats.cli_commands.features``
calls this with its own ``args.features`` manifest sibling and stores the
result under the ``source_snapshots`` key in its own metadata, so the chain
self-extends: a level-2 table (e.g. ``game_features_qb.parquet``, built from
``game_features_pbp.parquet``) picks up the base ``source_snapshot``
transitively through ``game_features_pbp.manifest.json``'s own
``source_snapshots`` block, without the level-2 writer needing to know the
base table exists at all.

Deliberately one-directional: this module is read by writers
(``nfl_ats.cli_commands.features``) and produces plain JSON that
``nfl_ats.lineage`` reads back. ``lineage.py`` imports only
:data:`SOURCE_SNAPSHOTS_KEY` from here (a string constant, so the two modules
cannot silently drift on what the block is called) -- it does not need the
rest of this module's API, and this module never imports ``nfl_ats.lineage``,
so the dependency only ever points one way. :data:`SNAPSHOT_KEYS` is a plain
tuple rather than an import from ``nfl_ats.lineage.FAMILY_BUILDERS`` for the
same reason: writers call this module long before a card is ever built.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Manifest keys, across every enrichment writer in
#: ``nfl_ats.cli_commands.features``, that name an immutable upstream
#: capture. Mirrors the ``manifest_snapshot_key`` values registered on
#: ``nfl_ats.lineage.FAMILY_BUILDERS`` -- kept as a plain tuple here (not an
#: import) so this module has no dependency on the lineage layer.
SNAPSHOT_KEYS: tuple[str, ...] = (
    "source_snapshot",
    "source_pbp_snapshot",
    "source_depth_snapshot",
    "source_player_snapshot",
    "source_player_value_snapshot",
    "source_participation_snapshot",
)

#: Key a derived manifest stores its merged upstream block under.
SOURCE_SNAPSHOTS_KEY = "source_snapshots"

#: Reason recorded when a caller names a parent manifest path that could not
#: be read (missing file, or not valid JSON).
UPSTREAM_ABSENT_REASON = "upstream manifest absent"


def manifest_path_for(parquet_path: Path | str) -> Path:
    """The ``*.manifest.json`` sibling of a feature-table parquet path.

    Matches the naming convention every writer in
    ``nfl_ats.cli_commands.features`` already uses for its own manifest:
    ``<stem>.manifest.json`` next to the parquet.
    """

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
    """Capture instant encoded in an ``nfl_ats.snapshots`` id, if any.

    A small, deliberate duplicate of ``nfl_ats.lineage.parse_snapshot_capture``
    -- see this module's docstring for why the two modules avoid importing
    each other. Returns ``None`` for ids that do not follow the
    ``%Y%m%dT%H%M%SZ`` convention rather than guessing.
    """

    try:
        instant = datetime.strptime(snapshot_id.strip(), "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    return instant.replace(tzinfo=UTC).isoformat()


def _label_for(path: Path) -> str:
    """A stable key for a parent manifest that could not be read at all."""

    name = path.name
    if name.endswith(".manifest.json"):
        name = name[: -len(".manifest.json")]
    return name


def inherit_source_snapshots(parent_manifest_paths: Iterable[Path | str]) -> dict[str, Any]:
    """Merge upstream manifests' source-snapshot ids into one lineage block.

    ``parent_manifest_paths`` are the ``*.manifest.json`` files a derived
    feature table was built directly from -- typically one: the manifest
    sibling of its own ``source_features`` parquet, via
    :func:`manifest_path_for`. Returns a dict with one entry per
    :data:`SNAPSHOT_KEYS` name any parent -- or anything a parent itself
    already inherited -- carries, e.g. ``{"source_snapshot": {"snapshot_id":
    ..., "captured_at": ..., "manifest_path": ...}, "source_pbp_snapshot":
    {...}}``. The base nflverse ``source_snapshot`` id is carried forward
    as-is: never reinterpreted, just copied string-for-string into its entry.

    A parent path that cannot be read (missing file, invalid JSON) degrades
    to one explicit ``{"snapshot_id": None, "reason": "upstream manifest
    absent"}`` entry, keyed by its own filename, rather than raising or
    silently dropping the dependency.

    Processing order matters on key collisions: within one parent, that
    parent's own directly-recorded key wins over anything it merely
    inherited; across parents, a later path in ``parent_manifest_paths``
    wins over an earlier one. Pass the closest/most-authoritative manifest
    last if a caller ever supplies more than one.
    """

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
        merged.update(parent_entries)
    return merged


__all__ = [
    "SNAPSHOT_KEYS",
    "SOURCE_SNAPSHOTS_KEY",
    "UPSTREAM_ABSENT_REASON",
    "inherit_source_snapshots",
    "manifest_path_for",
]
