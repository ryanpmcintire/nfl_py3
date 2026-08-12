from __future__ import annotations

import hashlib

from nfl_ats.provenance import (
    artifact_provenance,
    configuration_hash,
    git_state,
    sha256_file,
)


def test_hashes_are_deterministic(tmp_path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()
    assert configuration_hash({"b": 2, "a": 1}) == configuration_hash({"a": 1, "b": 2})


def test_provenance_without_git_or_manifests(tmp_path) -> None:
    feature_path = tmp_path / "game_features.parquet"
    feature_path.write_bytes(b"features")
    payload = artifact_provenance({"model": "test"}, feature_path, project_root=tmp_path)
    assert payload["feature_table"]["manifest"] is None
    assert payload["code"] == {"revision": None, "dirty": None}
    assert payload["uv_lock_sha256"] is None
    assert git_state(tmp_path) == {"revision": None, "dirty": None}


def test_provenance_uses_matching_feature_manifest(tmp_path) -> None:
    feature_path = tmp_path / "game_features_pbp.parquet"
    feature_path.write_bytes(b"features")
    (tmp_path / "game_features.manifest.json").write_text('{"kind": "base"}', encoding="utf-8")
    (tmp_path / "game_features_pbp.manifest.json").write_text('{"kind": "pbp"}', encoding="utf-8")
    payload = artifact_provenance({}, feature_path, project_root=tmp_path)
    assert payload["feature_table"]["manifest"] == {"kind": "pbp"}
