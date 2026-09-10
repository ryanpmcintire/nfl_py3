from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_ats.environment_report import environment_report
from nfl_ats.io import atomic_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def configuration_hash(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(command: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *command],
        cwd=workdir,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )


def git_state(workdir: Path) -> dict[str, Any]:
    revision = _git(["rev-parse", "HEAD"], workdir)
    if revision.returncode != 0:
        return {"revision": None, "dirty": None}
    status = _git(["status", "--porcelain", "--untracked-files=normal"], workdir)
    return {
        "revision": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def git_diff_sha256(workdir: Path) -> str | None:

    diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=workdir,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if diff.returncode != 0:
        return None
    return hashlib.sha256(diff.stdout).hexdigest()


def artifact_provenance(
    configuration: dict[str, Any],
    feature_path: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or Path.cwd()).resolve()
    feature_manifest_path = feature_path.with_name(f"{feature_path.stem}.manifest.json")
    feature_manifest = (
        json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if feature_manifest_path.is_file()
        else None
    )
    lockfile = root / "uv.lock"
    code = git_state(root)
    uv_lock_sha256 = sha256_file(lockfile) if lockfile.is_file() else None
    return {
        "configuration": configuration,
        "configuration_sha256": configuration_hash(configuration),
        "feature_table": {
            "path": str(feature_path.resolve()),
            "sha256": sha256_file(feature_path),
            "manifest": feature_manifest,
        },
        "code": code,
        "uv_lock_sha256": uv_lock_sha256,
        "environment": environment_report(
            project_root=root, git_info=code, uv_lock_sha256=uv_lock_sha256
        ),
    }


EXPERIMENT_REGISTRY_DIRNAME = "experiments"

_EXPERIMENT_RECORD_FIELDS = frozenset(
    {
        "experiment_id",
        "recorded_at",
        "command",
        "artifact_directory",
        "config_hash",
        "code_revision",
        "code_dirty",
        "code_diff_sha256",
        "feature_table_sha256",
        "uv_lock_sha256",
        "schema_version",
        "metrics",
        "notes",
        "source",
        "weak_signal_name",
        "rotation_family",
        "provenance_backfilled",
        "backfill_note",
    }
)

DEFAULT_METRICS_SCHEMA_VERSION = 1


class ExperimentRecordError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    recorded_at: str
    command: str
    artifact_directory: str
    config_hash: str
    schema_version: int
    metrics: dict[str, Any]
    source: str
    code_revision: str | None = None
    code_dirty: bool | None = None
    code_diff_sha256: str | None = None
    feature_table_sha256: str | None = None
    uv_lock_sha256: str | None = None
    notes: str = ""
    weak_signal_name: str | None = None
    rotation_family: str | None = None
    provenance_backfilled: bool = False
    backfill_note: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentRecordError(message)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def experiment_record_from_payload(payload: dict[str, Any]) -> ExperimentRecord:
    unknown = sorted(set(payload).difference(_EXPERIMENT_RECORD_FIELDS))
    _require(not unknown, f"Experiment record has unknown fields: {', '.join(unknown)}")
    required = (
        "experiment_id",
        "recorded_at",
        "command",
        "artifact_directory",
        "config_hash",
        "schema_version",
        "metrics",
        "source",
    )
    for field in required:
        _require(field in payload, f"Experiment record is missing {field!r}")

    experiment_id = str(payload["experiment_id"])
    metrics = payload["metrics"]
    _require(isinstance(metrics, dict), f"Experiment record {experiment_id!r} has non-dict metrics")
    schema_version = payload["schema_version"]
    _require(
        isinstance(schema_version, int) and not isinstance(schema_version, bool),
        f"Experiment record {experiment_id!r} has a non-integer schema_version",
    )
    code_dirty = payload.get("code_dirty")
    _require(
        code_dirty is None or isinstance(code_dirty, bool),
        f"Experiment record {experiment_id!r} has a non-bool code_dirty",
    )
    provenance_backfilled = payload.get("provenance_backfilled", False)
    _require(
        isinstance(provenance_backfilled, bool),
        f"Experiment record {experiment_id!r} has a non-bool provenance_backfilled",
    )

    return ExperimentRecord(
        experiment_id=experiment_id,
        recorded_at=str(payload["recorded_at"]),
        command=str(payload["command"]),
        artifact_directory=str(payload["artifact_directory"]),
        config_hash=str(payload["config_hash"]),
        schema_version=int(schema_version),
        metrics=dict(metrics),
        source=str(payload["source"]),
        code_revision=_optional_str(payload.get("code_revision")),
        code_dirty=code_dirty,
        code_diff_sha256=_optional_str(payload.get("code_diff_sha256")),
        feature_table_sha256=_optional_str(payload.get("feature_table_sha256")),
        uv_lock_sha256=_optional_str(payload.get("uv_lock_sha256")),
        notes=str(payload.get("notes", "")),
        weak_signal_name=_optional_str(payload.get("weak_signal_name")),
        rotation_family=_optional_str(payload.get("rotation_family")),
        provenance_backfilled=bool(provenance_backfilled),
        backfill_note=_optional_str(payload.get("backfill_note")),
    )


def experiment_record_to_payload(record: ExperimentRecord) -> dict[str, Any]:
    return {
        "experiment_id": record.experiment_id,
        "recorded_at": record.recorded_at,
        "command": record.command,
        "artifact_directory": record.artifact_directory,
        "config_hash": record.config_hash,
        "code_revision": record.code_revision,
        "code_dirty": record.code_dirty,
        "code_diff_sha256": record.code_diff_sha256,
        "feature_table_sha256": record.feature_table_sha256,
        "uv_lock_sha256": record.uv_lock_sha256,
        "schema_version": record.schema_version,
        "metrics": record.metrics,
        "notes": record.notes,
        "source": record.source,
        "weak_signal_name": record.weak_signal_name,
        "rotation_family": record.rotation_family,
        "provenance_backfilled": record.provenance_backfilled,
        "backfill_note": record.backfill_note,
    }


def load_experiment_record(path: Path) -> ExperimentRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return experiment_record_from_payload(payload)


def save_experiment_record(record: ExperimentRecord, path: Path) -> None:
    atomic_json(experiment_record_to_payload(record), path)


def default_experiment_registry_root(root: Path | None = None) -> Path:

    base = Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry")) if root is None else root
    return base / EXPERIMENT_REGISTRY_DIRNAME


def experiment_command_slug(command: str) -> str:

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", command.strip())
    return slug or "unknown"


@dataclass(frozen=True)
class ExperimentLinkVerification:
    experiment_id: str
    command: str
    artifact_directory: str | None
    source: str | None
    candidate_paths: tuple[str, ...]
    resolved_path: str | None
    exists: bool
    flags: tuple[str, ...]


def verify_experiment_links(
    registry_root: Path | None = None,
    *,
    artifacts_roots: list[Path] | None = None,
) -> list[ExperimentLinkVerification]:

    experiments_root = default_experiment_registry_root(registry_root)
    repo_root = experiments_root.parent.parent
    if artifacts_roots is None:
        env_root = os.environ.get("NFL_ATS_ARTIFACTS_DIR")
        roots: list[Path] = [Path(env_root)] if env_root else [Path("artifacts")]
    else:
        roots = list(artifacts_roots)

    results: list[ExperimentLinkVerification] = []
    if not experiments_root.is_dir():
        return results
    for path in sorted(experiments_root.glob("*/*.json")):
        record = load_experiment_record(path)
        flags: list[str] = []
        directory = record.artifact_directory
        candidates: list[Path] = []
        if directory:
            parsed = Path(directory)
            if parsed.is_absolute():
                flags.append("absolute_machine_path")
                candidates.append(parsed)
            else:
                stripped = (
                    parsed.relative_to("artifacts")
                    if parsed.parts[:1] == ("artifacts",)
                    else parsed
                )
                for base in roots:
                    candidates.append(base / stripped)
                candidates.append(repo_root / parsed)
        resolved = next((candidate for candidate in candidates if candidate.exists()), None)
        if record.source and (
            record.source.startswith("nfl-ats ")
            or record.source.endswith(".md")
            or record.source.startswith("docs/")
        ):
            flags.append("source_not_a_path")
        if experiment_command_slug(record.command) != record.command:
            flags.append("id_not_filesystem_safe")
        results.append(
            ExperimentLinkVerification(
                experiment_id=record.experiment_id,
                command=record.command,
                artifact_directory=record.artifact_directory,
                source=record.source,
                candidate_paths=tuple(str(candidate) for candidate in candidates),
                resolved_path=str(resolved) if resolved is not None else None,
                exists=resolved is not None,
                flags=tuple(flags),
            )
        )
    return results


_METRICS_MAX_BYTES = 4096
_METRICS_VALUE_MAX_BYTES = 512


def _json_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def bounded_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if _json_size(metrics) <= _METRICS_MAX_BYTES:
        return dict(metrics)
    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for key in sorted(metrics):
        if _json_size(metrics[key]) <= _METRICS_VALUE_MAX_BYTES:
            kept[key] = metrics[key]
        else:
            dropped.append(key)
    if dropped:
        kept["_metrics_truncated_keys"] = dropped
    return kept


def write_experiment_artifact(
    directory: Path,
    filename: str,
    metadata: dict[str, Any],
    *,
    command: str,
    metrics: dict[str, Any],
    schema_version: int = DEFAULT_METRICS_SCHEMA_VERSION,
    provenance_key: str | None = "provenance",
    notes: str = "",
    source: str | None = None,
    weak_signal_name: str | None = None,
    rotation_family: str | None = None,
    project_root: Path | None = None,
    registry_root: Path | None = None,
) -> dict[str, Any]:

    atomic_json(metadata, directory / filename)

    provenance: dict[str, Any] = metadata if provenance_key is None else metadata[provenance_key]
    code = provenance.get("code") or {}
    code_revision = code.get("revision")
    code_dirty = code.get("dirty")
    diff_hash = git_diff_sha256(project_root or Path.cwd()) if code_dirty else None
    feature_table = provenance.get("feature_table") or {}

    stamp = directory.name
    experiment_id = f"{command}/{stamp}"
    recorded_at = str(metadata.get("created_at_utc") or datetime.now(UTC).isoformat())

    record = ExperimentRecord(
        experiment_id=experiment_id,
        recorded_at=recorded_at,
        command=command,
        artifact_directory=str(directory),
        config_hash=str(provenance.get("configuration_sha256", "")),
        schema_version=schema_version,
        metrics=bounded_metrics(metrics),
        source=source or f"nfl-ats {command}",
        code_revision=code_revision,
        code_dirty=code_dirty,
        code_diff_sha256=diff_hash,
        feature_table_sha256=feature_table.get("sha256"),
        uv_lock_sha256=provenance.get("uv_lock_sha256"),
        notes=notes,
        weak_signal_name=weak_signal_name,
        rotation_family=rotation_family,
        provenance_backfilled=False,
        backfill_note=None,
    )
    payload = experiment_record_to_payload(record)
    registry_dir = default_experiment_registry_root(registry_root) / experiment_command_slug(
        command
    )
    atomic_json(payload, registry_dir / f"{stamp}.json")
    return payload


def write_stamped_artifact(
    payload: dict[str, Any],
    destination: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:

    root = (project_root or Path.cwd()).resolve()
    code = git_state(root)
    stamped = dict(payload)
    stamped["_provenance_stamp"] = {
        "recorded_at": utc_now(),
        "code_revision": code.get("revision"),
        "code_dirty": code.get("dirty"),
    }
    atomic_json(stamped, destination)
    return stamped


def stamp_sidecar(
    path: Path,
    extra: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
) -> Path:

    root = (project_root or Path.cwd()).resolve()
    code = git_state(root)
    payload: dict[str, Any] = {
        "path": str(path),
        "recorded_at": utc_now(),
        "code_revision": code.get("revision"),
        "code_dirty": code.get("dirty"),
    }
    if extra:
        payload.update(extra)
    sidecar = path.with_name(path.name + ".provenance.json")
    atomic_json(payload, sidecar)
    return sidecar
