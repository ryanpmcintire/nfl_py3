from __future__ import annotations

from pathlib import Path

from nfl_ats.provenance import (
    ExperimentRecordError,
    default_experiment_registry_root,
    experiment_command_slug,
    experiment_record_from_payload,
    load_experiment_record,
    save_experiment_record,
    verify_experiment_links,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


_JSON_WRITE_MARKERS = ("json.dump(", "json.dumps(", "atomic_json(")
_PROVENANCE_HELPER_NAMES = (
    "write_experiment_artifact",
    "write_stamped_artifact",
    "stamp_sidecar",
    "save_experiment_record",
)


def _writes_artifacts_json_without_helper(path: Path) -> bool:

    text = path.read_text(encoding="utf-8")
    if any(helper in text for helper in _PROVENANCE_HELPER_NAMES):
        return False
    if "artifacts" not in text:
        return False
    return any(marker in text for marker in _JSON_WRITE_MARKERS)


def _save_row(registry_root: Path, command: str, stamp: str, **overrides: object) -> None:
    payload: dict[str, object] = {
        "experiment_id": f"{command}/{stamp}",
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "command": command,
        "artifact_directory": f"artifacts/{command}/{stamp}",
        "config_hash": "abc123",
        "schema_version": 1,
        "metrics": {},
        "source": f"nfl-ats {command}",
        "provenance_backfilled": False,
    }
    payload.update(overrides)
    record = experiment_record_from_payload(payload)
    directory = default_experiment_registry_root(registry_root) / experiment_command_slug(command)
    directory.mkdir(parents=True, exist_ok=True)
    save_experiment_record(record, directory / f"{stamp}.json")


def test_verify_experiment_links_finds_existing_and_flags_missing(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    artifacts_root = tmp_path / "artifacts"
    existing = artifacts_root / "demo-command" / "20260101T000000Z"
    existing.mkdir(parents=True)
    _save_row(registry_root, "demo-command", "20260101T000000Z")
    _save_row(
        registry_root,
        "other-command",
        "20260102T000000Z",
        source="artifacts/other_command/20260102T000000Z/run.json",
    )

    results = verify_experiment_links(registry_root=registry_root, artifacts_roots=[artifacts_root])
    by_id = {r.experiment_id: r for r in results}
    assert by_id["demo-command/20260101T000000Z"].exists is True
    assert by_id["other-command/20260102T000000Z"].exists is False
    assert "source_not_a_path" in by_id["demo-command/20260101T000000Z"].flags
    assert "source_not_a_path" not in by_id["other-command/20260102T000000Z"].flags
    abs_row = tmp_path / "abs_command" / "20260103T000000Z"
    abs_row.mkdir(parents=True)
    _save_row(
        registry_root,
        "abs-command",
        "20260103T000000Z",
        artifact_directory=str(abs_row),
    )
    results = verify_experiment_links(registry_root=registry_root, artifacts_roots=[artifacts_root])
    abs_result = next(r for r in results if r.experiment_id == "abs-command/20260103T000000Z")
    assert "absolute_machine_path" in abs_result.flags
    assert abs_result.exists is True
    assert abs_result.resolved_path == str(abs_row)


def test_verify_experiment_links_flags_unsafe_id(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    command_with_spaces = "weird name (v2)"
    _save_row(registry_root, command_with_spaces, "20260101T000000Z")
    results = verify_experiment_links(registry_root=registry_root, artifacts_roots=[])
    assert len(results) == 1
    flags = results[0].flags
    assert "id_not_filesystem_safe" in flags
    assert "source_not_a_path" in flags


def test_committed_registry_experiment_rows_all_parse() -> None:
    experiments_root = REPO_ROOT / "registry" / "experiments"
    if not experiments_root.is_dir():
        return
    rows = sorted(experiments_root.glob("*/*.json"))
    assert rows, "expected at least one backfilled experiment row"
    for path in rows:
        try:
            record = load_experiment_record(path)
        except ExperimentRecordError as error:  # pragma: no cover - failure path
            raise AssertionError(f"{path} does not parse as an ExperimentRecord: {error}") from None
        assert record.experiment_id
        assert record.command


def test_every_scripts_import_in_cli_puts_the_repo_root_on_the_path_first() -> None:

    source = (REPO_ROOT / "src" / "nfl_ats" / "cli.py").read_text(encoding="utf-8")
    offenders: list[str] = []
    for block in source.split("\ndef ")[1:]:
        name = block.split("(", 1)[0]
        if "from scripts." not in block and "import scripts" not in block:
            continue
        guard = block.find("_repo_root_on_path()")
        first_import = min(
            (
                index
                for index in (block.find("from scripts."), block.find("import scripts"))
                if index >= 0
            ),
            default=-1,
        )
        if guard < 0 or guard > first_import:
            offenders.append(name)
    assert not offenders, (
        "CLI handler(s) import `scripts.*` without calling _repo_root_on_path() "
        f"first: {sorted(offenders)}. Under the `nfl-ats` console script that "
        "raises ModuleNotFoundError, and weekly-run dispatches in-process, so a "
        "fail-closed step importing this way aborts the whole lock-day run."
    )
