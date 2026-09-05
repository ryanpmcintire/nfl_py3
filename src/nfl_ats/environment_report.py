"""ENG-21: a deterministic environment lock report.

Motivation: a headline number this project records is only as reproducible
as the environment that produced it. ``nfl_ats.provenance.artifact_provenance``
already pins the CODE (git revision/dirty flag) and the DATA (feature-table
sha256, ``uv.lock`` sha256) behind every artifact; this module is the third
leg -- the INTERPRETER and its resolved dependency graph. Without it, "does
this number reproduce" silently assumed numpy/pandas/scikit-learn never
changed underneath a re-run.

Public surface (deliberately small, so ``ENG-01``'s ``lockday_package.py``
and ``ENG-02``'s ``preflight.py`` can each call it without pulling in the rest
of this project's provenance machinery):

- :func:`environment_report` -- one flat-ish dict: Python version/implementation,
  ``uv`` version, OS/platform, resolved versions of every numerically relevant
  dependency, BLAS/LAPACK backend, thread-count env vars, git revision/dirty
  flag, ``uv.lock`` sha256, an allow-listed env-var dump, and a secrets-
  presence map. **Never raises** -- any failure anywhere in the assembly
  becomes ``{"error": "<Type>: <message>"}`` rather than propagating, so a
  broken environment probe can never abort the run it is describing.
- :func:`compare_environment` -- diffs two report dicts and classifies every
  differing field as ``reproducibility_affecting`` or ``cosmetic``.

Secret handling (binding, per the ENG-21 spec this module implements):
project secrets (``THE_ODDS_API_KEY``, ``CFBD_API_KEY``) and anything whose
NAME matches ``KEY``/``TOKEN``/``SECRET``/``PASSWORD`` (case-insensitive) may
be recorded as a boolean presence flag. Their VALUE is never read into the
report. This is enforced twice: the allow-listed env-var dump
(:data:`_ALLOWLISTED_ENV_EXACT` / the ``*_NUM_THREADS`` family / ``NFL_ATS_*``)
never includes a secret-shaped name in the first place, and a final recursive
redaction pass (:func:`_redact`) walks the WHOLE assembled dict and collapses
any surviving secret-shaped key to a boolean, so a secret cannot leak by
entering through an unexpected path either.

Reproducibility-affecting vs. cosmetic (the split :func:`compare_environment`
implements): a field is reproducibility-affecting when a difference in it
CAN change which code path runs or what numbers a run produces --
interpreter minor version, resolved package versions, the ``uv.lock`` hash
(a different lock can resolve different transitive versions even with the
same direct pins), BLAS/LAPACK backend identity (different backends can shift
floating point results in the low bits, which matters for bootstrap CIs even
though it never matters for a forced ATS pick), thread-count env vars (BLAS/
sklearn parallelism can change floating-point summation order), and
``PYTHONHASHSEED``/``platform.machine`` (hash-order and instruction-set
dependent code paths). A field is cosmetic when it varies run-to-run or
box-to-box without touching numeric behavior: ``uv``'s own patch version (the
resolver, not a runtime dependency -- what matters is ``uv.lock``'s hash, a
separate field), the OS build/release string, the git *dirty* boolean (it
only says "something is uncommitted", not what -- ``git.revision`` itself, by
contrast, is reproducibility-affecting, since a different revision is
different code), timestamps, and hostnames/executable paths. Unmatched or
newly introduced fields default to reproducibility-affecting rather than
cosmetic: a false "this might matter" costs an extra look at a comparison; a
false "this is definitely fine" can hide a real non-reproduction, which is
the more expensive mistake for a project whose whole premise is trusting a
small measured number.
"""

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

#: Distributions whose resolved version can change a computed number:
#: pyproject.toml's direct dependencies plus scipy (a transitive
#: scikit-learn/numpy dependency this backlog item calls out by name as
#: "numerically relevant").
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

#: Common BLAS/OpenMP thread-count env vars. Wider than the generic
#: "*_NUM_THREADS" allow-list rule below because two real backends
#: (Accelerate/veclib) do not follow that suffix convention.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_SECRET_NAME_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)
#: Named explicitly so their presence/absence is always reported, even when
#: not set (an absent credential is a fact worth recording too).
_KNOWN_SECRET_ENV_VARS: tuple[str, ...] = ("THE_ODDS_API_KEY", "CFBD_API_KEY")

_ALLOWLISTED_ENV_EXACT = frozenset({"PYTHONHASHSEED", "TZ"})
_ALLOWLISTED_ENV_PREFIX = "NFL_ATS_"
_THREAD_SUFFIX = "_NUM_THREADS"


def _is_allowlisted_env_name(name: str) -> bool:
    """PYTHONHASHSEED, TZ, the ``*_NUM_THREADS`` family, and ``NFL_ATS_*``.

    Deliberately narrow: this is the allow-list the ENG-21 spec names
    verbatim, not a general env dump. Everything else in ``os.environ`` --
    including any secret -- is excluded before redaction is even considered.
    """

    return (
        name in _ALLOWLISTED_ENV_EXACT
        or name.endswith(_THREAD_SUFFIX)
        or name.startswith(_ALLOWLISTED_ENV_PREFIX)
    )


def _redact(value: Any) -> Any:
    """Recursively collapse any secret-shaped dict key to a boolean.

    Applied as the LAST step before :func:`environment_report` returns, over
    the whole assembled dict -- not just the env-var section -- so a secret
    cannot survive by entering through an unexpected path (e.g. a stray
    ``NFL_ATS_API_KEY`` override name).
    """

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
        # Defensive: an allow-listed prefix could still be named like a
        # secret (e.g. a hypothetical NFL_ATS_API_KEY). _redact() would catch
        # this too, but resolving it here keeps the allow-list itself honest.
        values[name] = True if _SECRET_NAME_PATTERN.search(name) else value
    return values


def _secrets_detected() -> dict[str, bool]:
    """Boolean-only presence map. VALUES are never read into this dict.

    Always includes the two named project secrets (present or not); also
    reports every OTHER env var whose name matches the generic secret
    pattern, so an unexpected credential (e.g. a test's ``FAKE_API_KEY``)
    still shows up as "something secret-shaped is set" without its value
    ever reaching the report.
    """

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
    """Revision + dirty flag. Mirrors ``provenance.git_state`` but kept local
    and self-contained so this module never imports ``nfl_ats.provenance``
    (which imports ``nfl_ats.io`` and would risk a future circular import,
    since ``provenance.artifact_provenance`` is the one call site that pulls
    :func:`environment_report` in).
    """

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
    """``uv --version``, tolerating absence entirely (returns available=False).

    Cached per resolved executable path: ``uv``'s own version cannot change
    mid-process, and this function sits behind ``artifact_provenance()``,
    which is called on every CLI command's artifact write -- an uncached
    subprocess spawn there would tax the whole test suite for no benefit.
    """

    for candidate in _uv_candidates(repo_root):
        if candidate.is_file():
            info = _cached_uv_version(str(candidate))
            if info["available"]:
                return {**info, "executable": str(candidate)}
    return {"available": False, "version": None, "raw": None, "executable": None}


@lru_cache(maxsize=1)
def _package_versions() -> dict[str, str | None]:
    """Resolved versions of :data:`TRACKED_PACKAGES`. Cached: installed
    package versions cannot change within one running process.
    """

    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@lru_cache(maxsize=1)
def _blas_summary() -> dict[str, Any]:
    """A cheap summary of numpy's BLAS/LAPACK backend. Cached (process-invariant)."""

    try:
        import numpy as np

        # Typed ``Any``: numpy's own stub declares a strict TypedDict for
        # mode="dicts", but this is best-effort telemetry against whatever
        # numpy version is actually installed, so the isinstance guards below
        # must stay real checks rather than statically-unreachable ones.
        config: Any = np.show_config(mode="dicts")
    except Exception:  # deliberately broad: this is best-effort telemetry
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
    # src/nfl_ats/environment_report.py -> parents[2] is the repo root, the
    # same pattern nfl_ats.cli._repo_root_on_path uses -- robust regardless
    # of the process's working directory.
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
    """Assemble the deterministic environment lock report.

    ``project_root`` defaults to this repository's root (resolved from this
    file's own location, not the working directory). ``git_info`` and
    ``uv_lock_sha256`` are optional pre-computed overrides so a caller that
    already paid for a git subprocess call and a ``uv.lock`` hash (as
    ``nfl_ats.provenance.artifact_provenance`` does) does not pay for them
    twice; omit both to have this function compute them itself, which is
    what a standalone caller (``scripts/env_report.py``,
    ``nfl_ats.lockday_package``) should do.

    **Fail-safe, always**: any exception anywhere in the assembly is caught
    here and turned into ``{"error": "<Type>: <message>"}`` rather than
    propagating, so a broken environment probe can never abort the run that
    called it. Most individual fields already degrade on their own (a
    missing ``uv`` executable, an uninstalled package, an unavailable numpy
    BLAS summary, a non-git working tree all produce ``None``/``False``
    sub-fields rather than raising) -- this is the last-resort net for
    anything that does not.
    """

    try:
        return _build_environment_report(project_root, git_info, uv_lock_sha256)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


# ---------------------------------------------------------------------------
# compare_environment
# ---------------------------------------------------------------------------

#: Explicit cosmetic allow-list (dotted leaf paths into a flattened
#: environment_report() dict). See the module docstring for the rationale
#: behind each entry. Every leaf NOT on this list, or matched by
#: :data:`_COSMETIC_PREFIXES`, defaults to reproducibility_affecting.
_COSMETIC_EXACT_PATHS = frozenset(
    {
        "python.version",  # full string incl. patch; python.major/.minor cover what matters
        "python.micro",
        "python.executable",
        "uv.version",
        "uv.raw",
        "uv.available",
        "uv.executable",
        "platform.release",
        "platform.version",  # OS build number
        "platform.processor",
        "git.dirty",  # a bare boolean; git.revision (reproducibility-affecting) is the pin
        "generated_at_utc",
    }
)
#: Path prefixes treated as cosmetic wholesale (dynamic keys under them).
_COSMETIC_PREFIXES = ("secrets_detected.",)


def classify_field(path: str) -> str:
    """Classify one dotted field path as ``reproducibility_affecting`` or
    ``cosmetic``. See the module docstring for the justification; unmatched
    paths default to ``reproducibility_affecting`` (the safe-by-default
    direction for a project whose premise is trusting small numbers).
    """

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
    """Diff two :func:`environment_report` dicts field by field.

    Every leaf that differs between ``a`` and ``b`` is classified via
    :func:`classify_field` as ``reproducibility_affecting`` or ``cosmetic``
    (see the module docstring for the split's rationale). Returns a summary
    dict rather than raising or asserting: this is a reporting tool, not a
    gate, and per this repository's binding research invariant an
    environment difference alone is never grounds to reject a result on its
    own -- it is context for interpreting one.
    """

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
