from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "pyarrow",
    "joblib",
    "nflreadpy",
    "pypdf",
    "tabulate",
)

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_SECRET_NAME_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)
_KNOWN_SECRET_ENV_VARS: tuple[str, ...] = ("THE_ODDS_API_KEY", "CFBD_API_KEY")

_ALLOWLISTED_ENV_EXACT = frozenset({"PYTHONHASHSEED", "TZ"})
_ALLOWLISTED_ENV_PREFIX = "NFL_ATS_"
_THREAD_SUFFIX = "_NUM_THREADS"


def _is_allowlisted_env_name(name: str) -> bool:

    return (
        name in _ALLOWLISTED_ENV_EXACT
        or name.endswith(_THREAD_SUFFIX)
        or name.startswith(_ALLOWLISTED_ENV_PREFIX)
    )


def _redact(value: Any) -> Any:

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and _SECRET_NAME_PATTERN.search(key):
                redacted[key] = _redact(val) if isinstance(val, dict) else bool(val)
            else:
                redacted[key] = _redact(val)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _allowlisted_env_vars() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, value in os.environ.items():
        if not _is_allowlisted_env_name(name):
            continue
        values[name] = True if _SECRET_NAME_PATTERN.search(name) else value
    return values


def _secrets_detected() -> dict[str, bool]:

    detected: dict[str, bool] = {name: name in os.environ for name in _KNOWN_SECRET_ENV_VARS}
    for name in os.environ:
        if _SECRET_NAME_PATTERN.search(name):
            detected[name] = True
    return detected


def _python_info() -> dict[str, Any]:
    info = sys.version_info
    return {
        "version": sys.version,
        "major": info.major,
        "minor": info.minor,
        "micro": info.micro,
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def _platform_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _git_info(repo_root: Path) -> dict[str, Any]:

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if revision.returncode != 0:
            return {"revision": None, "dirty": None}
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        return {
            "revision": revision.stdout.strip(),
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def _uv_lock_info(repo_root: Path) -> dict[str, Any]:
    lockfile = repo_root / "uv.lock"
    if not lockfile.is_file():
        return {"present": False, "sha256": None}
    digest = hashlib.sha256()
    with lockfile.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"present": True, "sha256": digest.hexdigest()}


def _uv_candidates(repo_root: Path) -> list[Path]:
    candidates = [repo_root / ".tools" / "uv.exe", repo_root / ".tools" / "uv"]
    found = shutil.which("uv")
    if found:
        candidates.append(Path(found))
    return candidates


@lru_cache(maxsize=8)
def _cached_uv_version(uv_executable: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [uv_executable, "--version"],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "version": None, "raw": None}
    if result.returncode != 0:
        return {"available": False, "version": None, "raw": None}
    raw = result.stdout.strip()
    match = re.match(r"uv\s+(\d+\.\d+\.\d+)", raw)
    return {"available": True, "version": match.group(1) if match else None, "raw": raw}


def _uv_info(repo_root: Path) -> dict[str, Any]:

    for candidate in _uv_candidates(repo_root):
        if candidate.is_file():
            info = _cached_uv_version(str(candidate))
            if info["available"]:
                return {**info, "executable": str(candidate)}
    return {"available": False, "version": None, "raw": None, "executable": None}


@lru_cache(maxsize=1)
def _package_versions() -> dict[str, str | None]:

    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@lru_cache(maxsize=1)
def _blas_summary() -> dict[str, Any]:

    try:
        import numpy as np

        config: Any = np.show_config(mode="dicts")
    except Exception:
        return {"available": False}
    if not isinstance(config, dict):
        return {"available": True, "detail": None}
    build = config.get("Build Dependencies")
    if not isinstance(build, dict):
        return {"available": True, "detail": None}
    blas_raw = build.get("blas")
    blas = blas_raw if isinstance(blas_raw, dict) else {}
    lapack_raw = build.get("lapack")
    lapack = lapack_raw if isinstance(lapack_raw, dict) else {}
    return {
        "available": True,
        "blas_name": blas.get("name"),
        "blas_version": blas.get("version"),
        "lapack_name": lapack.get("name"),
        "lapack_version": lapack.get("version"),
    }


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_environment_report(
    project_root: Path | None,
    git_info: dict[str, Any] | None,
    uv_lock_sha256: str | None,
) -> dict[str, Any]:
    repo_root = (project_root or _default_project_root()).resolve()
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": _python_info(),
        "uv": _uv_info(repo_root),
        "platform": _platform_info(),
        "packages": _package_versions(),
        "blas": _blas_summary(),
        "thread_counts": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "git": git_info if git_info is not None else _git_info(repo_root),
        "uv_lock": (
            {"present": uv_lock_sha256 is not None, "sha256": uv_lock_sha256}
            if uv_lock_sha256 is not None
            else _uv_lock_info(repo_root)
        ),
        "environment_variables": _allowlisted_env_vars(),
        "secrets_detected": _secrets_detected(),
    }
    redacted: dict[str, Any] = _redact(report)
    return redacted


def environment_report(
    project_root: Path | None = None,
    *,
    git_info: dict[str, Any] | None = None,
    uv_lock_sha256: str | None = None,
) -> dict[str, Any]:

    try:
        return _build_environment_report(project_root, git_info, uv_lock_sha256)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


_COSMETIC_EXACT_PATHS = frozenset(
    {
        "python.version",
        "python.micro",
        "python.executable",
        "uv.version",
        "uv.raw",
        "uv.available",
        "uv.executable",
        "platform.release",
        "platform.version",
        "platform.processor",
        "git.dirty",
        "generated_at_utc",
    }
)
_COSMETIC_PREFIXES = ("secrets_detected.",)


def classify_field(path: str) -> str:

    if path in _COSMETIC_EXACT_PATHS:
        return "cosmetic"
    if any(path.startswith(prefix) for prefix in _COSMETIC_PREFIXES):
        return "cosmetic"
    return "reproducibility_affecting"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            flat[prefix] = value
            return flat
        for key, val in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(val, path))
        return flat
    flat[prefix] = value
    return flat


def compare_environment(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:

    flat_a = _flatten(a)
    flat_b = _flatten(b)
    fields: dict[str, Any] = {}
    reproducibility_affecting: list[str] = []
    cosmetic: list[str] = []
    for path in sorted(set(flat_a) | set(flat_b)):
        value_a = flat_a.get(path)
        value_b = flat_b.get(path)
        if value_a == value_b:
            continue
        classification = classify_field(path)
        fields[path] = {"a": value_a, "b": value_b, "classification": classification}
        bucket = (
            reproducibility_affecting if classification == "reproducibility_affecting" else cosmetic
        )
        bucket.append(path)
    return {
        "differs": bool(fields),
        "reproducibility_affecting": bool(reproducibility_affecting),
        "fields": fields,
        "reproducibility_affecting_fields": reproducibility_affecting,
        "cosmetic_fields": cosmetic,
    }
