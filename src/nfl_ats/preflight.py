"""Read-only environment and configuration preflight (ENG-02).

This module answers one question before any research command runs: is the
local machine set up correctly, independent of whether research data has
been rebuilt yet? It never writes configuration, never fetches from a
network source, and never mutates ``data/``, ``artifacts/``, or
``registry/``. Writability checks create a uniquely named temporary probe
file and remove it immediately; nothing it touches is left behind.

Every check is tagged with one of three categories, matching the ENG-02
definition of done:

- ``environment`` -- local tooling: Python interpreter version, the ``uv``
  executable, the ``uv`` cache directory, Git, the ``core.hooksPath`` Git
  setting, and whether the data/artifacts/registry destinations are
  writable. A ``fail`` here blocks real work regardless of what research
  data exists.
- ``configuration`` -- source-policy inputs: whether ``THE_ODDS_API_KEY``
  and ``CFBD_API_KEY`` are present (never their values) and whether the
  ``NFL_ATS_DATA_DIR`` / ``NFL_ATS_ARTIFACTS_DIR`` / ``NFL_ATS_REGISTRY_DIR``
  overrides are set. Missing API keys are reported as ``warn``, not
  ``fail`` -- most commands do not need live network sources.
- ``research_data`` -- the same local-artifact inventory rendered in
  ``HANDOFF.md``'s "Local reproducibility inventory" section
  (:func:`nfl_ats.handoff._local_inventory`, reused here rather than
  duplicated so the two views cannot drift). A fresh clone legitimately
  lacks every one of these files, so this category only ever reports
  ``ok``/``warn``, never ``fail``.

``run_preflight`` returns a :class:`PreflightReport`; :func:`preflight_exit_code`
implements the CLI's exit-code rule: nonzero only for an ``environment`` or
``configuration`` row with status ``fail``, unless ``strict=True`` is passed,
in which case any non-``ok`` row (including missing research data) is also
fatal.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Reused verbatim rather than duplicated: this is the same list rendered in
# HANDOFF.md's "Local reproducibility inventory" section, and duplicating it
# here would let the two views silently drift.
from nfl_ats.handoff import _display_path, _local_inventory

PREFLIGHT_VERSION = 1

Category = Literal["environment", "research_data", "configuration"]
Status = Literal["ok", "warn", "fail"]

_FALLBACK_REQUIRES_PYTHON = ">=3.12,<3.14"
_EXPECTED_HOOKS_PATH = ".githooks"

#: (env var, human description of what breaks without it). Values are never
#: read into a check's ``detail`` -- presence/absence only.
_SOURCE_POLICY_KEYS: tuple[tuple[str, str], ...] = (
    ("THE_ODDS_API_KEY", "live odds fetches and the historical market backfill"),
    ("CFBD_API_KEY", "CFB Data API ingestion"),
)

#: (env var, default relative path) for the three overridable roots.
_DIRECTORY_ENV_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("NFL_ATS_DATA_DIR", "data"),
    ("NFL_ATS_ARTIFACTS_DIR", "artifacts"),
    ("NFL_ATS_REGISTRY_DIR", "registry"),
)

_COMPARISON_OPERATORS: tuple[str, ...] = (">=", "<=", "==", "!=", ">", "<")


@dataclass(frozen=True)
class PreflightCheck:
    """One structured, independently reportable preflight result."""

    name: str
    category: Category
    status: Status
    detail: str
    remedy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
        }


@dataclass(frozen=True)
class PreflightReport:
    """The full ordered set of checks from one preflight run."""

    generated_at_utc: str
    checks: tuple[PreflightCheck, ...]

    def counts(self) -> dict[Status, int]:
        counts: dict[Status, int] = {"ok": 0, "warn": 0, "fail": 0}
        for check in self.checks:
            counts[check.status] += 1
        return counts

    def has_environment_or_configuration_failure(self) -> bool:
        return any(
            check.status == "fail" and check.category in ("environment", "configuration")
            for check in self.checks
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PREFLIGHT_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "checks": [check.to_dict() for check in self.checks],
            "summary": self.counts(),
        }


def preflight_exit_code(report: PreflightReport, *, strict: bool = False) -> int:
    """Exit-code rule: environment/configuration ``fail`` is always fatal.

    Missing research data (``research_data`` rows, which are only ever
    ``ok``/``warn``) is reported but not fatal by default. ``strict=True``
    additionally fails on any non-``ok`` row in any category, which makes
    missing research data fatal too.
    """

    if report.has_environment_or_configuration_failure():
        return 1
    if strict and any(check.status != "ok" for check in report.checks):
        return 1
    return 0


def _parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in text.strip().split("."):
        digits = "".join(character for character in chunk if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _clause_satisfied(running: tuple[int, ...], clause: str) -> bool | None:
    """Evaluate one comparison clause; ``None`` means it could not be parsed."""

    for operator_text in sorted(_COMPARISON_OPERATORS, key=len, reverse=True):
        if not clause.startswith(operator_text):
            continue
        target = _parse_version(clause[len(operator_text) :])
        width = max(len(running), len(target))
        lhs = running + (0,) * (width - len(running))
        rhs = target + (0,) * (width - len(target))
        if operator_text == ">=":
            return lhs >= rhs
        if operator_text == "<=":
            return lhs <= rhs
        if operator_text == "==":
            return lhs == rhs
        if operator_text == "!=":
            return lhs != rhs
        if operator_text == ">":
            return lhs > rhs
        return lhs < rhs
    return None


def _python_satisfies(running: tuple[int, ...], spec: str) -> tuple[bool, list[str]]:
    """Return (satisfied, unparsed_clauses) for a ``requires-python`` spec string."""

    satisfied = True
    unparsed: list[str] = []
    for raw_clause in spec.split(","):
        clause = raw_clause.strip()
        if not clause:
            continue
        result = _clause_satisfied(running, clause)
        if result is None:
            unparsed.append(clause)
        elif not result:
            satisfied = False
    return satisfied, unparsed


def _read_requires_python(repo_root: Path) -> tuple[str, str]:
    """Return (spec, source_description); falls back when unreadable."""

    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            with pyproject_path.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        declared = data.get("project", {}).get("requires-python") if data else None
        if isinstance(declared, str) and declared.strip():
            return declared.strip(), "pyproject.toml [project.requires-python]"
    return _FALLBACK_REQUIRES_PYTHON, "fallback constant (pyproject.toml unreadable)"


def check_python_version(
    repo_root: Path, *, running_version: tuple[int, int, int] | None = None
) -> PreflightCheck:
    """Compare the running interpreter against the repo's declared requirement."""

    running = running_version or sys.version_info[:3]
    spec, source = _read_requires_python(repo_root)
    satisfied, unparsed = _python_satisfies(running, spec)
    running_text = ".".join(str(part) for part in running)
    if unparsed:
        return PreflightCheck(
            name="python interpreter version",
            category="environment",
            status="warn",
            detail=(
                f"running Python {running_text}; could not fully parse requires-python "
                f"{spec!r} from {source} (unparsed clauses: {unparsed})"
            ),
            remedy="Verify the interpreter manually against pyproject.toml's requires-python.",
        )
    if satisfied:
        return PreflightCheck(
            name="python interpreter version",
            category="environment",
            status="ok",
            detail=f"running Python {running_text} satisfies {spec!r} ({source})",
        )
    return PreflightCheck(
        name="python interpreter version",
        category="environment",
        status="fail",
        detail=f"running Python {running_text} does not satisfy {spec!r} ({source})",
        remedy="Use the locked uv environment's Python 3.12 interpreter (`uv run ...`).",
    )


def _uv_candidates(repo_root: Path) -> list[Path]:
    candidates = [repo_root / ".tools" / "uv.exe", repo_root / ".tools" / "uv"]
    found = shutil.which("uv")
    if found:
        candidates.append(Path(found))
    return candidates


def check_uv_available(repo_root: Path) -> tuple[PreflightCheck, Path | None]:
    """Locate a working ``uv`` executable: ``.tools/uv(.exe)`` first, then PATH."""

    for candidate in _uv_candidates(repo_root):
        if not candidate.is_file():
            continue
        try:
            result = subprocess.run(
                [str(candidate), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except OSError as error:
            return (
                PreflightCheck(
                    name="uv executable available",
                    category="environment",
                    status="fail",
                    detail=f"found {candidate} but could not execute it: {error}",
                    remedy="Reinstall uv or repair .tools/uv.exe.",
                ),
                None,
            )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return (
                PreflightCheck(
                    name="uv executable available",
                    category="environment",
                    status="ok",
                    detail=f"{candidate} -> {version}",
                ),
                candidate,
            )
        return (
            PreflightCheck(
                name="uv executable available",
                category="environment",
                status="fail",
                detail=(
                    f"{candidate} --version exited {result.returncode}: {result.stderr.strip()}"
                ),
                remedy="Reinstall uv or repair .tools/uv.exe.",
            ),
            None,
        )
    return (
        PreflightCheck(
            name="uv executable available",
            category="environment",
            status="fail",
            detail="no uv executable found at .tools/uv(.exe) or on PATH",
            remedy="Install uv (https://docs.astral.sh/uv/) or restore .tools/uv.exe.",
        ),
        None,
    )


def _nearest_existing_ancestor(path: Path) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_dir():
            return ancestor
    return Path(absolute.anchor or ".")


def _probe_writable(directory: Path) -> tuple[bool, str]:
    """Create and immediately remove a uniquely named temp file. Read-only in effect."""

    probe_path = directory / f".nfl_ats_preflight_{uuid.uuid4().hex}.tmp"
    try:
        probe_path.write_text("preflight probe", encoding="utf-8")
    except OSError as error:
        return False, str(error)
    # Best-effort cleanup; the write already proved writability either way.
    with contextlib.suppress(OSError):
        probe_path.unlink()
    return True, "ok"


def check_uv_cache(uv_path: Path | None) -> PreflightCheck:
    """Check that ``uv cache dir`` resolves to a directory this user can write to."""

    if uv_path is None:
        return PreflightCheck(
            name="uv cache accessible",
            category="environment",
            status="fail",
            detail="cannot determine the uv cache location: no working uv executable",
            remedy="Fix 'uv executable available' first.",
        )
    try:
        result = subprocess.run(
            [str(uv_path), "cache", "dir"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError as error:
        return PreflightCheck(
            name="uv cache accessible",
            category="environment",
            status="fail",
            detail=f"`uv cache dir` could not run: {error}",
            remedy="Reinstall uv or check its permissions.",
        )
    if result.returncode != 0:
        return PreflightCheck(
            name="uv cache accessible",
            category="environment",
            status="fail",
            detail=f"`uv cache dir` exited {result.returncode}: {result.stderr.strip()}",
            remedy="Reinstall uv or check its permissions.",
        )
    cache_dir = Path(result.stdout.strip())
    exists = cache_dir.is_dir()
    probe_dir = cache_dir if exists else _nearest_existing_ancestor(cache_dir)
    writable, reason = _probe_writable(probe_dir)
    if writable:
        return PreflightCheck(
            name="uv cache accessible",
            category="environment",
            status="ok" if exists else "warn",
            detail=(
                f"uv cache directory {cache_dir} is writable"
                if exists
                else (
                    f"uv cache directory {cache_dir} does not exist yet; nearest existing "
                    f"ancestor {probe_dir} is writable (uv creates the cache lazily)"
                )
            ),
        )
    return PreflightCheck(
        name="uv cache accessible",
        category="environment",
        status="fail",
        detail=f"uv cache directory {cache_dir} is not writable (probed {probe_dir}): {reason}",
        remedy=f"Grant write access to {cache_dir}, or set UV_CACHE_DIR to a writable location.",
    )


def check_git_available() -> tuple[PreflightCheck, str | None]:
    found = shutil.which("git")
    if found is None:
        return (
            PreflightCheck(
                name="git executable available",
                category="environment",
                status="fail",
                detail="git was not found on PATH",
                remedy="Install Git; it is required for the hooksPath check and session handoff.",
            ),
            None,
        )
    return (
        PreflightCheck(
            name="git executable available",
            category="environment",
            status="ok",
            detail=f"git found at {found}",
        ),
        found,
    )


def check_hooks_path(repo_root: Path, git_path: str | None) -> PreflightCheck:
    """Read (never set) ``core.hooksPath``; AGENTS.md requires it to be ``.githooks``."""

    if git_path is None:
        return PreflightCheck(
            name="git hooksPath configuration",
            category="environment",
            status="fail",
            detail="cannot check core.hooksPath: git executable unavailable",
            remedy="Install Git, then run `git config --local core.hooksPath .githooks`.",
        )
    try:
        result = subprocess.run(
            [git_path, "config", "--get", "core.hooksPath"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError as error:
        return PreflightCheck(
            name="git hooksPath configuration",
            category="environment",
            status="fail",
            detail=f"`git config --get core.hooksPath` could not run: {error}",
            remedy="Run `git config --local core.hooksPath .githooks`.",
        )
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        return PreflightCheck(
            name="git hooksPath configuration",
            category="environment",
            status="fail",
            detail="core.hooksPath is not configured",
            remedy="Run `git config --local core.hooksPath .githooks`.",
        )
    if value != _EXPECTED_HOOKS_PATH:
        return PreflightCheck(
            name="git hooksPath configuration",
            category="environment",
            status="fail",
            detail=f"core.hooksPath is {value!r}, expected {_EXPECTED_HOOKS_PATH!r}",
            remedy="Run `git config --local core.hooksPath .githooks`.",
        )
    return PreflightCheck(
        name="git hooksPath configuration",
        category="environment",
        status="ok",
        detail=f"core.hooksPath is {value!r}",
    )


def check_writable_directory(name: str, directory: Path) -> PreflightCheck:
    """Probe whether ``directory`` (or its nearest existing ancestor) is writable."""

    exists = directory.is_dir()
    probe_dir = directory if exists else _nearest_existing_ancestor(directory)
    writable, reason = _probe_writable(probe_dir)
    if writable:
        return PreflightCheck(
            name=name,
            category="environment",
            status="ok" if exists else "warn",
            detail=(
                f"{directory} is writable"
                if exists
                else (
                    f"{directory} does not exist yet; nearest existing ancestor {probe_dir} "
                    "is writable and the directory will be created on first write"
                )
            ),
        )
    return PreflightCheck(
        name=name,
        category="environment",
        status="fail",
        detail=f"{directory} is not writable (probed {probe_dir}): {reason}",
        remedy=f"Grant write access to {probe_dir}, or point the matching NFL_ATS_*_DIR "
        "override elsewhere.",
    )


def check_source_policy(env: Mapping[str, str]) -> list[PreflightCheck]:
    """Report presence/absence of source-policy API keys. Values are never read out."""

    checks: list[PreflightCheck] = []
    for key, purpose in _SOURCE_POLICY_KEYS:
        present = bool(env.get(key, "").strip())
        checks.append(
            PreflightCheck(
                name=f"source policy: {key}",
                category="configuration",
                status="ok" if present else "warn",
                detail=(
                    f"{key} is set (value not read or logged by this check)"
                    if present
                    else f"{key} is not set; {purpose} will fail closed until it is set"
                ),
                remedy=(
                    None
                    if present
                    else f"Set {key} in the environment before running commands that need it "
                    "(never store it in the repo)."
                ),
            )
        )
    return checks


def check_directory_overrides(env: Mapping[str, str]) -> list[PreflightCheck]:
    """Report the three NFL_ATS_*_DIR overrides. Paths are not secrets."""

    checks: list[PreflightCheck] = []
    for key, default in _DIRECTORY_ENV_OVERRIDES:
        value = env.get(key)
        checks.append(
            PreflightCheck(
                name=f"directory override: {key}",
                category="configuration",
                status="ok",
                detail=(
                    f"{key} is not set; using default {default!r}"
                    if not value
                    else f"{key} is set to {value!r} (default {default!r})"
                ),
            )
        )
    return checks


def check_research_artifacts(repo_root: Path, artifacts_root: Path) -> list[PreflightCheck]:
    """Presence-only check of the same inventory HANDOFF.md renders.

    Never ``fail`` -- an absent parquet table or artifact directory is
    expected and legitimate on a fresh clone, which is exactly the
    distinction ENG-02 asks this module to preserve.
    """

    checks: list[PreflightCheck] = []
    for label, path in _local_inventory(repo_root, artifacts_root):
        present = path.is_file()
        checks.append(
            PreflightCheck(
                name=f"local artifact: {label}",
                category="research_data",
                status="ok" if present else "warn",
                detail=(
                    f"present at {_display_path(path, repo_root)}"
                    if present
                    else (
                        "absent (expected on a fresh clone until rebuilt): "
                        f"{_display_path(path, repo_root)}"
                    )
                ),
                remedy=(
                    None
                    if present
                    else "Rebuild it per HANDOFF.md's 'Local reproducibility inventory' "
                    "section, or the relevant `nfl-ats` build/experiment command."
                ),
            )
        )
    return checks


def run_preflight(
    repo_root: Path,
    *,
    data_root: Path | None = None,
    artifacts_root: Path | None = None,
    registry_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    generated_at: datetime | None = None,
) -> PreflightReport:
    """Run every read-only check and return the combined report.

    ``data_root``/``artifacts_root``/``registry_root`` default to the same
    ``NFL_ATS_DATA_DIR`` / ``NFL_ATS_ARTIFACTS_DIR`` / ``NFL_ATS_REGISTRY_DIR``
    environment overrides (falling back to ``data``/``artifacts``/``registry``)
    that ``nfl_ats.cli`` uses, so this reads the same locations the CLI would.
    """

    resolved_env: Mapping[str, str] = os.environ if env is None else env
    data_root = (
        data_root if data_root is not None else Path(resolved_env.get("NFL_ATS_DATA_DIR", "data"))
    )
    artifacts_root = (
        artifacts_root
        if artifacts_root is not None
        else Path(resolved_env.get("NFL_ATS_ARTIFACTS_DIR", "artifacts"))
    )
    registry_root = (
        registry_root
        if registry_root is not None
        else Path(resolved_env.get("NFL_ATS_REGISTRY_DIR", "registry"))
    )
    instant = generated_at if generated_at is not None else datetime.now(UTC)

    checks: list[PreflightCheck] = [check_python_version(repo_root)]

    uv_check, uv_path = check_uv_available(repo_root)
    checks.append(uv_check)
    checks.append(check_uv_cache(uv_path))

    git_check, git_path = check_git_available()
    checks.append(git_check)
    checks.append(check_hooks_path(repo_root, git_path))

    checks.append(check_writable_directory("artifacts directory writable", artifacts_root))
    checks.append(check_writable_directory("data directory writable", data_root))
    checks.append(check_writable_directory("registry directory writable", registry_root))

    checks.extend(check_source_policy(resolved_env))
    checks.extend(check_directory_overrides(resolved_env))
    checks.extend(check_research_artifacts(repo_root, artifacts_root))

    return PreflightReport(
        generated_at_utc=instant.astimezone(UTC).isoformat(), checks=tuple(checks)
    )
