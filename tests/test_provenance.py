from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from nfl_ats.provenance import (
    ExperimentRecordError,
    artifact_provenance,
    bounded_metrics,
    configuration_hash,
    experiment_command_slug,
    experiment_record_from_payload,
    experiment_record_to_payload,
    git_diff_sha256,
    git_state,
    load_experiment_record,
    sha256_file,
    write_experiment_artifact,
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


def test_git_diff_sha256_outside_git_repo(tmp_path) -> None:
    assert git_diff_sha256(tmp_path) is None


def _run_git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def _init_repo(root: Path) -> None:
    _run_git(["init", "-q"], root)
    _run_git(["config", "user.email", "test@example.com"], root)
    _run_git(["config", "user.name", "Test"], root)
    (root / "tracked.txt").write_text("original\n", encoding="utf-8")
    _run_git(["add", "tracked.txt"], root)
    _run_git(["commit", "-q", "-m", "initial"], root)


def test_git_diff_sha256_is_none_on_a_clean_tree(tmp_path) -> None:
    _init_repo(tmp_path)
    assert git_diff_sha256(tmp_path) == hashlib.sha256(b"").hexdigest()


def test_git_diff_sha256_changes_with_the_working_tree(tmp_path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")
    first = git_diff_sha256(tmp_path)
    assert first is not None
    assert first != hashlib.sha256(b"").hexdigest()

    (tmp_path / "tracked.txt").write_text("changed again\n", encoding="utf-8")
    second = git_diff_sha256(tmp_path)
    assert second is not None
    assert second != first


# ---------------------------------------------------------------------------
# The experiment-provenance registry (RWB-09)
# ---------------------------------------------------------------------------


def _write_experiment_record_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "experiment_id": "demo-command/20260101T000000Z",
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "command": "demo-command",
        "artifact_directory": "artifacts/demo_command/20260101T000000Z",
        "config_hash": "abc123",
        "code_revision": "deadbeef",
        "code_dirty": False,
        "code_diff_sha256": None,
        "feature_table_sha256": "feat123",
        "uv_lock_sha256": "lock123",
        "schema_version": 1,
        "metrics": {"accuracy_points": 1.1},
        "notes": "",
        "source": "nfl-ats demo-command",
        "weak_signal_name": None,
        "rotation_family": None,
        "provenance_backfilled": False,
        "backfill_note": None,
    }
    payload.update(overrides)
    return payload


def test_experiment_record_round_trips_through_payload() -> None:
    payload = _write_experiment_record_payload()
    record = experiment_record_from_payload(payload)
    assert experiment_record_to_payload(record) == payload


def test_experiment_record_rejects_unknown_fields() -> None:
    payload = _write_experiment_record_payload()
    payload["mystery_field"] = "nope"
    with pytest.raises(ExperimentRecordError, match="unknown fields"):
        experiment_record_from_payload(payload)


def test_experiment_record_rejects_missing_required_field() -> None:
    payload = _write_experiment_record_payload()
    del payload["config_hash"]
    with pytest.raises(ExperimentRecordError, match="missing 'config_hash'"):
        experiment_record_from_payload(payload)


def test_experiment_record_rejects_non_dict_metrics() -> None:
    payload = _write_experiment_record_payload(metrics="not a dict")
    with pytest.raises(ExperimentRecordError, match="non-dict metrics"):
        experiment_record_from_payload(payload)


def test_experiment_record_rejects_non_bool_code_dirty() -> None:
    payload = _write_experiment_record_payload(code_dirty="yes")
    with pytest.raises(ExperimentRecordError, match="non-bool code_dirty"):
        experiment_record_from_payload(payload)


def test_experiment_command_slug_is_filesystem_safe() -> None:
    assert experiment_command_slug("cfb-benchmark") == "cfb-benchmark"
    assert experiment_command_slug("scripts/foo.py") == "scripts_foo.py"
    assert experiment_command_slug("weird name (v2)") == "weird_name_v2_"
    assert experiment_command_slug("   ") == "unknown"


def test_bounded_metrics_passes_small_dicts_through_untouched() -> None:
    metrics = {"accuracy_points": 1.1, "note": "small"}
    assert bounded_metrics(metrics) == metrics


def test_bounded_metrics_truncates_and_names_oversized_entries() -> None:
    metrics = {
        "headline": 1.1,
        "huge_table": [{"row": i, "padding": "x" * 100} for i in range(200)],
    }
    bounded = bounded_metrics(metrics)
    assert bounded["headline"] == 1.1
    assert "huge_table" not in bounded
    assert bounded["_metrics_truncated_keys"] == ["huge_table"]
    # The bounded payload itself must actually be small -- that is the point.
    assert len(json.dumps(bounded).encode("utf-8")) < len(json.dumps(metrics).encode("utf-8"))


def _feature_setup(tmp_path: Path) -> Path:
    feature_path = tmp_path / "game_features.parquet"
    feature_path.write_bytes(b"features")
    return feature_path


def test_write_experiment_artifact_writes_metadata_and_registry_row(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    feature_path = _feature_setup(tmp_path)
    # Commit the feature file too, so this tree is genuinely clean -- an
    # untracked file would otherwise make git_state report dirty=True, which
    # is exactly correct but not what this particular test is exercising.
    _run_git(["add", "game_features.parquet"], tmp_path)
    _run_git(["commit", "-q", "-m", "add feature table"], tmp_path)
    configuration = {"command": "demo-command", "start_season": 2020}
    provenance = artifact_provenance(configuration, feature_path, project_root=tmp_path)
    metadata = {
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "command": "demo-command",
        "provenance": provenance,
    }
    output = tmp_path / "artifacts" / "demo_command" / "20260101T000000Z"
    output.mkdir(parents=True)

    payload = write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="demo-command",
        metrics={"accuracy_points": 1.1},
        project_root=tmp_path,
        registry_root=tmp_path / "registry",
    )

    assert (output / "metadata.json").is_file()
    assert json.loads((output / "metadata.json").read_text(encoding="utf-8")) == metadata

    registry_file = tmp_path / "registry" / "experiments" / "demo-command" / "20260101T000000Z.json"
    assert registry_file.is_file()
    on_disk = json.loads(registry_file.read_text(encoding="utf-8"))
    assert on_disk == payload

    record = load_experiment_record(registry_file)
    assert record.experiment_id == "demo-command/20260101T000000Z"
    assert record.command == "demo-command"
    assert record.config_hash == provenance["configuration_sha256"]
    assert record.code_revision == provenance["code"]["revision"]
    assert record.code_dirty is False
    assert record.code_diff_sha256 is None  # clean tree: no diff to hash
    assert record.artifact_directory == str(output)
    assert record.metrics["accuracy_points"] == 1.1
    assert record.provenance_backfilled is False


def test_write_experiment_artifact_records_a_dirty_tree_never_blocks(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("dirty now\n", encoding="utf-8")
    feature_path = _feature_setup(tmp_path)
    configuration = {"command": "demo-command"}
    provenance = artifact_provenance(configuration, feature_path, project_root=tmp_path)
    assert provenance["code"]["dirty"] is True  # sanity: the tree really is dirty

    metadata = {"provenance": provenance}
    output = tmp_path / "artifacts" / "demo_command" / "20260101T000000Z"
    output.mkdir(parents=True)

    payload = write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="demo-command",
        metrics={},
        project_root=tmp_path,
        registry_root=tmp_path / "registry",
    )

    # The whole point: a dirty run is RECORDED, never refused.
    assert payload["code_dirty"] is True
    assert payload["code_diff_sha256"] is not None
    assert payload["code_diff_sha256"] == git_diff_sha256(tmp_path)


def test_write_experiment_artifact_supports_bare_provenance_dict(tmp_path: Path) -> None:
    """The ``run.json`` convention: metadata IS the provenance dict, not wrapped."""

    _init_repo(tmp_path)
    feature_path = _feature_setup(tmp_path)
    provenance = artifact_provenance({"command": "backtest"}, feature_path, project_root=tmp_path)
    output = tmp_path / "artifacts" / "backtest" / "20260101T000000Z"
    output.mkdir(parents=True)

    payload = write_experiment_artifact(
        output,
        "run.json",
        provenance,
        command="backtest",
        metrics={"accuracy": 0.52},
        provenance_key=None,
        project_root=tmp_path,
        registry_root=tmp_path / "registry",
    )

    assert (output / "run.json").is_file()
    assert json.loads((output / "run.json").read_text(encoding="utf-8")) == provenance
    assert payload["config_hash"] == provenance["configuration_sha256"]


def test_write_experiment_artifact_tolerates_a_minimal_stubbed_provenance(tmp_path: Path) -> None:
    """Some CLI tests monkeypatch ``artifact_provenance`` to a bare stub -- must not crash."""

    metadata = {"provenance": {"test": True}}
    output = tmp_path / "artifacts" / "demo_command" / "20260101T000000Z"
    output.mkdir(parents=True)

    payload = write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="demo-command",
        metrics={},
        registry_root=tmp_path / "registry",
    )
    assert payload["config_hash"] == ""
    assert payload["code_revision"] is None
    assert payload["code_dirty"] is None


def test_write_experiment_artifact_reuses_the_artifact_directory_stamp(tmp_path: Path) -> None:
    feature_path = _feature_setup(tmp_path)
    metadata = {
        "provenance": artifact_provenance({"command": "demo"}, feature_path, project_root=tmp_path)
    }
    output = tmp_path / "artifacts" / "demo" / "20260215T093000Z"
    output.mkdir(parents=True)

    payload = write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="demo",
        metrics={},
        registry_root=tmp_path / "registry",
    )
    assert payload["experiment_id"] == "demo/20260215T093000Z"
    registry_file = tmp_path / "registry" / "experiments" / "demo" / "20260215T093000Z.json"
    assert registry_file.is_file()


def test_write_experiment_artifact_two_commands_do_not_clobber_each_other(tmp_path: Path) -> None:
    """Per-run files under registry/experiments/<command>/ -- no lost-update race."""

    feature_path = _feature_setup(tmp_path)
    registry_root = tmp_path / "registry"

    for command in ("command-a", "command-b"):
        output = tmp_path / "artifacts" / command.replace("-", "_") / "20260101T000000Z"
        output.mkdir(parents=True)
        metadata = {
            "provenance": artifact_provenance(
                {"command": command}, feature_path, project_root=tmp_path
            )
        }
        write_experiment_artifact(
            output,
            "metadata.json",
            metadata,
            command=command,
            metrics={"command": command},
            registry_root=registry_root,
        )

    row_a = load_experiment_record(
        registry_root / "experiments" / "command-a" / "20260101T000000Z.json"
    )
    row_b = load_experiment_record(
        registry_root / "experiments" / "command-b" / "20260101T000000Z.json"
    )
    assert row_a.command == "command-a"
    assert row_b.command == "command-b"
    assert row_a.metrics["command"] == "command-a"
    assert row_b.metrics["command"] == "command-b"
