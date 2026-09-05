"""ENG-22: source-snapshot inheritance across derived feature-table manifests.

``docs/feature_lineage.md`` measured (2026-09-04) that every enrichment
manifest under ``data/processed/`` records ``source_features`` -- a path to
the parquet it enriched -- but not the snapshot id that path's own manifest
recorded, so 9 of 15 decision-bearing fields on the live Week 1 card fall
back to a ``feature_table:sha256`` digest instead of naming a real snapshot.
:func:`nfl_ats.feature_manifest.inherit_source_snapshots` is the fix; these
tests prove the merge, the two-level transitive chain, and the
never-raises-on-a-missing-parent contract in isolation, on synthetic
manifests under ``tmp_path`` -- no production table under ``data/processed/``
is read or written.
"""

from __future__ import annotations

import json
from pathlib import Path

from nfl_ats.feature_manifest import (
    SOURCE_SNAPSHOTS_KEY,
    UPSTREAM_ABSENT_REASON,
    inherit_source_snapshots,
    manifest_path_for,
)


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_path_for_matches_the_writers_own_naming_convention(tmp_path: Path) -> None:
    parquet = tmp_path / "game_features_pbp.parquet"
    assert manifest_path_for(parquet) == tmp_path / "game_features_pbp.manifest.json"
    # Accepts a bare string too -- argparse gives writers a Path, but callers
    # should not have to care.
    assert manifest_path_for(str(parquet)) == tmp_path / "game_features_pbp.manifest.json"


def test_merges_a_single_parents_direct_snapshot_key(tmp_path: Path) -> None:
    base_manifest = _write_manifest(
        tmp_path / "game_features.manifest.json",
        {"source_snapshot": "20260824T115346Z", "built_at_utc": "2026-08-24T12:00:00Z"},
    )

    result = inherit_source_snapshots([base_manifest])

    assert result == {
        "source_snapshot": {
            "snapshot_id": "20260824T115346Z",
            "captured_at": "2026-08-24T11:53:46+00:00",
            "manifest_path": str(base_manifest),
        }
    }


def test_transitive_inheritance_across_two_levels(tmp_path: Path) -> None:
    """A level-2 table (e.g. game_features_qb) inherits the base
    source_snapshot through game_features_pbp's own inherited block, without
    ever reading the base manifest directly."""

    base_manifest = _write_manifest(
        tmp_path / "game_features.manifest.json",
        {"source_snapshot": "20260824T115346Z"},
    )

    # Level 1: exactly what _cmd_build_pbp_features now writes.
    level1_inherited = inherit_source_snapshots([base_manifest])
    assert "source_snapshot" in level1_inherited
    level1_manifest = _write_manifest(
        tmp_path / "game_features_pbp.manifest.json",
        {
            "source_pbp_snapshot": "20260817T184927Z",
            "source_features": str(tmp_path / "game_features.parquet"),
            SOURCE_SNAPSHOTS_KEY: level1_inherited,
        },
    )

    # Level 2: exactly what _cmd_build_qb_features now writes -- its only
    # input is the level-1 manifest, never the base one.
    level2 = inherit_source_snapshots([level1_manifest])

    assert level2["source_snapshot"]["snapshot_id"] == "20260824T115346Z"
    assert level2["source_snapshot"]["manifest_path"] == str(base_manifest)
    assert level2["source_pbp_snapshot"]["snapshot_id"] == "20260817T184927Z"
    assert level2["source_pbp_snapshot"]["manifest_path"] == str(level1_manifest)


def test_missing_upstream_manifest_degrades_to_an_explicit_null_entry(tmp_path: Path) -> None:
    missing = tmp_path / "game_features.manifest.json"
    assert not missing.exists()

    result = inherit_source_snapshots([missing])

    assert result == {"game_features": {"snapshot_id": None, "reason": UPSTREAM_ABSENT_REASON}}


def test_unreadable_json_also_degrades_rather_than_raising(tmp_path: Path) -> None:
    broken = tmp_path / "game_features_pbp.manifest.json"
    broken.write_text("{not valid json", encoding="utf-8")

    result = inherit_source_snapshots([broken])

    assert result["game_features_pbp"] == {"snapshot_id": None, "reason": UPSTREAM_ABSENT_REASON}


def test_a_missing_parent_never_blocks_a_readable_one(tmp_path: Path) -> None:
    """Never a KeyError, and never lets one absent parent hide another."""

    present = _write_manifest(
        tmp_path / "game_features.manifest.json",
        {"source_snapshot": "20260824T115346Z"},
    )
    missing = tmp_path / "game_features_player_value.manifest.json"

    result = inherit_source_snapshots([present, missing])

    assert result["source_snapshot"]["snapshot_id"] == "20260824T115346Z"
    assert result["game_features_player_value"] == {
        "snapshot_id": None,
        "reason": UPSTREAM_ABSENT_REASON,
    }


def test_later_parent_wins_key_collisions(tmp_path: Path) -> None:
    older = _write_manifest(tmp_path / "a.manifest.json", {"source_snapshot": "20260101T000000Z"})
    newer = _write_manifest(tmp_path / "b.manifest.json", {"source_snapshot": "20260201T000000Z"})

    result = inherit_source_snapshots([older, newer])

    assert result["source_snapshot"]["snapshot_id"] == "20260201T000000Z"
    assert result["source_snapshot"]["manifest_path"] == str(newer)


def test_empty_input_returns_an_empty_block() -> None:
    assert inherit_source_snapshots([]) == {}
