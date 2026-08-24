"""Reproducibility metadata for generated research artifacts.

Also home to the RWB-09 experiment-provenance registry: a durable, git-
tracked record of which code/config/data produced a headline number, kept
independent of ``artifacts/`` (gitignored, local-disk-only, and proven twice
in one session to disappear -- see ``docs/closure_audit.md``). One small JSON
file per experiment run lives under ``registry/experiments/<command>/``,
written by :func:`write_experiment_artifact` as a side effect of the artifact
write every CLI command already does, so recording is not a discipline a
session has to remember.
"""

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

from nfl_ats.io import atomic_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_hash(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(command: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
    # Explicit utf-8: text=True alone decodes with the locale codepage (cp1252 on
    # Windows), which crashes on any non-cp1252 byte in the output.
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
    """SHA-256 of the working tree's diff against HEAD.

    ``git_state`` records ``dirty`` as a bare boolean, which only tells you
    that the revision hash alone will not reproduce the code that ran -- not
    *what* changed. Hashing ``git diff HEAD`` closes that gap without
    embedding the diff's own (potentially large) text into a git-tracked
    registry row. Returns ``None`` outside a git repository, matching
    ``git_state``'s own degenerate case.
    """

    # Hash the diff's raw bytes: decoding to text first is both unnecessary and
    # fragile (any single undecodable byte would change or crash the hash).
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
    return {
        "configuration": configuration,
        "configuration_sha256": configuration_hash(configuration),
        "feature_table": {
            "path": str(feature_path.resolve()),
            "sha256": sha256_file(feature_path),
            "manifest": feature_manifest,
        },
        "code": git_state(root),
        "uv_lock_sha256": sha256_file(lockfile) if lockfile.is_file() else None,
    }


# ---------------------------------------------------------------------------
# RWB-09: the experiment-provenance registry
#
# One JSON file per experiment RUN (not per curated finding -- that is what
# ``weak_signals.py``/``rotation_registry.py`` are for), under
# ``registry/experiments/<command>/<stamp>.json``. A directory of small files,
# not one growing ledger, because this project routinely runs several agents
# in parallel and a single read-modify-write JSON file is a lost-update race
# under concurrent writers; independently-named files are not.
# ---------------------------------------------------------------------------

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

# Metrics are deliberately schema-free (run shapes vary too much to enumerate
# up front, unlike weak_signals.json's narrow effect/interval/P+ triple), but
# every record still carries a required schema_version -- not a version of
# the registry file format (each file is a single record; the round-trip
# functions below are that contract), but a version of the *metrics* dict's
# own shape for this command, so a reader knows which fields to expect from a
# script whose output shape might change later.
DEFAULT_METRICS_SCHEMA_VERSION = 1


class ExperimentRecordError(ValueError):
    """Raised when an experiment-registry row is invalid.

    A ``ValueError`` subclass so callers see a user-facing error rather than
    a bare traceback, matching ``WeakSignalError``/``RegistryError``.
    """


@dataclass(frozen=True)
class ExperimentRecord:
    """One experiment run: enough to answer "does this number reproduce" and
    "what code/config/data produced it" months later, after ``artifacts/``
    (gitignored, local-disk-only) is gone.
    """

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
    # Optional pointers into the higher-level registries. Deliberately loose
    # -- a name string, not a foreign key -- so this module never has to
    # import weak_signals.py/rotation_registry.py or validate against them.
    weak_signal_name: str | None = None
    rotation_family: str | None = None
    # Honest about backfilled rows: a mechanical lift from an artifact that
    # already carried provenance, never an invented/approximated value.
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
    """Return the tracked experiment-registry root, honouring ``NFL_ATS_REGISTRY_DIR``
    when ``root`` is not given explicitly -- the same convention as
    ``rotation.default_registry_path``/``weak_signals.default_registry_path``, so
    a caller that forgets to thread an explicit root still lands in whatever
    isolated directory a test (or ``NFL_ATS_REGISTRY_DIR``-aware caller) has
    already set up, rather than silently falling back to the real tracked
    ``registry/`` tree.
    """

    base = Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry")) if root is None else root
    return base / EXPERIMENT_REGISTRY_DIRNAME


def experiment_command_slug(command: str) -> str:
    """Filesystem-safe directory name for a command or script path.

    ``registry/experiments/<slug>/<stamp>.json`` -- slugified so a future
    script-path command (e.g. ``scripts/foo.py``) cannot smuggle a path
    separator into the registry layout, even though this pass only wires
    ``cli.py``'s command names, which are already safe as-is.
    """

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", command.strip())
    return slug or "unknown"


@dataclass(frozen=True)
class ExperimentLinkVerification:
    """One row's artifact-link check.

    Records whether the stored ``artifact_directory`` resolves to something
    that actually exists on disk, the candidate paths tried, and any
    consistency flags the row carries.
    """

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
    """Check every committed experiment-registry row's linked artifact.

    Read-only: never writes. Walks ``registry/experiments/*/*.json`` and, for
    each row, resolves its ``artifact_directory`` against a set of candidate
    roots (an absolute stored path as-is; a relative one against each provided
    ``artifacts_roots`` entry and the registry's own repo root) and reports
    whether any candidate exists on disk.

    ``artifacts/`` is gitignored and local-disk-only, so a MISSING link is NOT
    necessarily a defect -- the reproducibility guarantee is the row's hashes
    (``config_hash``/``code_revision``/``feature_table_sha256``/``uv_lock_sha256``),
    not the path. This helper exists so a session can *measure* which links
    still resolve rather than guessing, and so path/identity inconsistencies
    surface instead of hiding.

    Flags:
      - ``absolute_machine_path``: the stored ``artifact_directory`` is an
        absolute, machine-specific path rather than a repo-relative one.
      - ``id_not_filesystem_safe``: the row's ``command`` required slugification
        to match its directory, so the raw ``experiment_id`` is not directly
        usable as a path.
      - ``source_not_a_path``: the stored ``source`` is not a path (e.g.
        ``nfl-ats <command>`` or a ``docs/...`` reference), so it cannot be
        checked for existence.
    """

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
                # A relative ``artifact_directory`` is repo-relative and already
                # begins with ``artifacts/`` (e.g. ``artifacts/backtests/<stamp>``).
                # Resolve it against each provided artifacts *directory* (stripping
                # the redundant prefix so we land in ``<artifacts>/backtests/<stamp>``)
                # and, as written, against the repo root.
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


# A git-tracked registry must not grow unbounded per row. Applied uniformly
# regardless of what a caller passes in: a hand-curated, small ``metrics``
# dict passes straight through untouched; a large, uncurated dict (as the
# mechanical cli.py retrofit passes, since hand-picking a "headline subset"
# per call site would not be mechanical) gets its oversized top-level entries
# dropped and NAMED, never silently truncated in place.
_METRICS_MAX_BYTES = 4096
_METRICS_VALUE_MAX_BYTES = 512


def _json_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def bounded_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Cap a metrics dict's serialized size, naming (not hiding) what was dropped.

    Public so the one-time backfill script (``scripts/backfill_experiment_registry.py``)
    can apply the identical rule to metrics lifted from an existing artifact,
    not just to metrics passed through :func:`write_experiment_artifact`.
    """
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
    """Write an artifact's own metadata file AND its experiment-registry row.

    ``metadata`` must already carry an ``artifact_provenance()``-shaped dict
    -- either under ``metadata[provenance_key]`` (the usual ``metadata.json``
    convention) or, when ``provenance_key`` is ``None``, ``metadata`` itself
    IS that dict (the bare ``run.json`` convention a few commands use). This
    function does not compute provenance; it stamps what the caller already
    computed, so it is a strict superset of the ``atomic_json(metadata,
    directory / filename)`` line it replaces -- not a second path a caller
    has to remember to keep in sync.

    A dirty tree is RECORDED, never blocked: research reality is dirty trees,
    and refusing to log a dirty run would just push it outside the registry
    entirely, defeating the point. ``code_diff_sha256`` is computed (only
    when dirty, to avoid a needless subprocess call on a clean run) so a
    dirty run's actual working tree is pinned too, not just its base
    revision.

    The registry stamp reuses ``directory.name`` -- the same ``run_id()``
    stamp already used for the artifact directory -- rather than generating
    a fresh one, so the two writes can never drift to different timestamps.

    Returns the registry row's payload (e.g. to fold into a command's own
    printed JSON summary).
    """

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
