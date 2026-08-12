"""Reproducibility metadata for generated research artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


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
    return subprocess.run(
        ["git", *command],
        cwd=workdir,
        capture_output=True,
        check=False,
        text=True,
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
