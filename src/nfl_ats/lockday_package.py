"""ENG-01: the immutable lock-day decision package.

Why this exists
---------------
Week 1 2026 locks on Tuesday 2026-09-08 as one command --
``nfl-ats weekly-run --record-decisions``. That single run downloads a
snapshot, rebuilds four feature tables, re-fits the walk-forward evaluation,
scores the week, publishes the tracked card and the public site, and appends
rows to up to seven append-only ledgers. Afterwards the evidence for "what
exactly did we decide, from what inputs, with which model" is scattered across
``data/processed/`` manifests, ``artifacts/active_ats_model.json``, a
timestamped ``artifacts/margin_predictions/`` directory, a tracked Markdown
card, twenty nested JSON recorder keys inside one stdout blob, and the parquet
ledgers themselves. ``artifacts/`` is gitignored and local-disk-only, and has
been observed to disappear.

So the run writes ONE folder that links all of it by SHA-256:

    artifacts/lockday_packages/<season>_wk<week>_<UTC stamp>/
        manifest.json      the package (written read-only)
        manifest.sha256    the manifest's own digest, so tampering shows
        README.md          how to read and verify it without this code

Design contracts
----------------
**Fail-safe, always.** By the time this runs, the ledger rows are already
appended and the card is already published. An exception in here must never
abort or roll back a lock that already happened, so every component is
collected behind :func:`_collect`, which records the failure in the
manifest's ``errors`` list and keeps going. A package with an ``errors`` list
is the designed output of a partially-broken run, not a failure.

**Read-only, not tamper-proof.** ``manifest.json`` gets the read-only
attribute (best effort; Windows/POSIX both), and ``manifest.sha256`` pins its
content. Neither stops a determined edit -- they stop an accidental one, and
they make a deliberate one detectable.

**Independently readable.** Nothing in the manifest requires this module to
interpret: it is plain JSON, every hash names the algorithm and the exact
bytes hashed, and ``scripts/lockday_package_verify.py`` recomputes the lot.
:func:`load_package` and :func:`summarise_package` are conveniences, not the
contract.

**Ledgers are append-only and are never touched here.** This module only ever
reads them, and it reads them twice: :func:`capture_ledger_state` before the
run, the package build after. The difference is the week's write.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import paper_decision_ledger_path
from nfl_ats.crew_tilt_refresh_overlay import crew_tilt_refresh_ledger_path
from nfl_ats.environment_report import environment_report
from nfl_ats.inactives_refresh_overlay import inactives_refresh_overlay_ledger_path
from nfl_ats.injury_signal_refresh_tilt import injury_signal_ledger_path
from nfl_ats.io import atomic_json, atomic_text, run_id
from nfl_ats.nflcom_refresh_overlay import nflcom_refresh_overlay_ledger_path
from nfl_ats.pick_refresh import pick_revision_ledger_path
from nfl_ats.prospective_scoring import challenger_ledger_path
from nfl_ats.provenance import git_diff_sha256, git_state, sha256_bytes, sha256_file

#: Bump when the manifest's shape changes in a way a reader must notice.
PACKAGE_SCHEMA_VERSION = 1
PACKAGE_KIND = "lockday_decision_package"
PACKAGES_DIRNAME = "lockday_packages"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_DIGEST_FILENAME = "manifest.sha256"
PACKAGE_README_FILENAME = "README.md"

#: Files above this size are LISTED but not hashed, so one stray multi-GB
#: artifact cannot turn the lock-day package write into a long stall.
MAX_HASHED_BYTES = 512 * 1024 * 1024

#: Every append-only ledger a lock-day run can write, by the same names the
#: rehearsal empties in ``scripts/lockday_rehearsal.build_isolated_root``.
#: Resolved through each module's own path function rather than restated as
#: literals, so a relocation cannot silently drop a ledger from the package.
LEDGER_PATH_FUNCTIONS: dict[str, Callable[[Path], Path]] = {
    "paper_decisions": paper_decision_ledger_path,
    "challenger_decisions": challenger_ledger_path,
    "pick_revisions": pick_revision_ledger_path,
    "injury_signal_refresh_decisions": injury_signal_ledger_path,
    "nflcom_friday_refresh_decisions": nflcom_refresh_overlay_ledger_path,
    "inactives_refresh_decisions": inactives_refresh_overlay_ledger_path,
    "crew_tilt_refresh_decisions": crew_tilt_refresh_ledger_path,
}

#: How an appended-row digest is computed, stated in the manifest so a reader
#: with pandas and no access to this file can reproduce it exactly.
APPENDED_ROWS_DIGEST_METHOD = (
    "sha256 of pandas.read_parquet(path).iloc[rows_before:].to_csv(index=False) "
    "encoded utf-8; the ledgers are append-only, so the tail beyond rows_before "
    "is exactly this run's write"
)

#: Step-command flags whose value is a snapshot id worth pinning by name.
_SNAPSHOT_FLAGS = (
    "--snapshot",
    "--player-snapshot",
    "--player-value-snapshot",
    "--pbp-snapshot",
)

#: Roles whose bytes are EXPECTED to change after the package is written --
#: the ledgers keep being appended to by later refresh passes in the same
#: week. The verifier reports them but never fails on them.
MUTABLE_ROLES = frozenset({"ledger_after"})


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _collect(
    errors: list[dict[str, Any]],
    component: str,
    builder: Callable[[], Any],
    default: Any = None,
) -> Any:
    """Run one manifest component, recording rather than raising its failure.

    The whole point of the package is that it survives a partially-broken
    run: the ledger rows are already on disk by the time anything here
    executes, so an exception must degrade one section, never the package
    and never the lock.
    """

    try:
        return builder()
    # Deliberately broad: see the module docstring's fail-safe contract.
    except Exception as error:
        errors.append({"component": component, "error": _describe(error)})
        print(f"lockday package: component {component!r} failed: {error}", file=sys.stderr)
        return default


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def _repo_relative(path: Path, repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def hash_entry(
    role: str,
    path: Path,
    *,
    repo_root: Path | None = None,
    mutable: bool | None = None,
    note: str = "",
) -> dict[str, Any]:
    """One hashed file, in the flat shape the verifier recomputes.

    Never raises: a missing or unreadable file is a recorded fact, because a
    lock that produced no card is exactly the run whose package matters most.
    """

    entry: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "repo_relative": _repo_relative(path, repo_root),
        "exists": False,
        "bytes": None,
        "algorithm": "sha256",
        "sha256": None,
        "mutable": bool(role in MUTABLE_ROLES if mutable is None else mutable),
        "note": note,
        "error": None,
    }
    try:
        if not path.is_file():
            return entry
        entry["exists"] = True
        size = path.stat().st_size
        entry["bytes"] = int(size)
        if size > MAX_HASHED_BYTES:
            entry["note"] = (note + " " if note else "") + (
                f"not hashed: larger than MAX_HASHED_BYTES ({MAX_HASHED_BYTES} bytes)"
            )
            return entry
        entry["sha256"] = sha256_file(path)
    except Exception as error:  # deliberately broad; see module docstring
        entry["error"] = _describe(error)
    return entry


def _directory_entries(
    role: str, directory: Path, *, repo_root: Path | None
) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return [
            {
                "role": role,
                "path": str(directory),
                "repo_relative": _repo_relative(directory, repo_root),
                "exists": False,
                "bytes": None,
                "algorithm": "sha256",
                "sha256": None,
                "mutable": False,
                "note": "directory not found",
                "error": None,
            }
        ]
    return [
        hash_entry(role, child, repo_root=repo_root)
        for child in sorted(directory.rglob("*"))
        if child.is_file()
    ]


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ledgers
# ---------------------------------------------------------------------------


def ledger_paths(artifacts_root: Path) -> dict[str, Path]:
    """Every append-only ledger a lock-day run can write, by name."""

    return {name: resolve(artifacts_root) for name, resolve in LEDGER_PATH_FUNCTIONS.items()}


def _ledger_snapshot(name: str, path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "ledger": name,
        "path": str(path),
        "exists": path.is_file(),
        "rows": 0,
        "sha256": None,
        "error": None,
    }
    if not entry["exists"]:
        return entry
    try:
        entry["sha256"] = sha256_file(path)
        entry["rows"] = len(pd.read_parquet(path))
    except Exception as error:  # deliberately broad; see module docstring
        entry["error"] = _describe(error)
    return entry


def capture_ledger_state(artifacts_root: Path) -> dict[str, dict[str, Any]]:
    """Row counts and file digests for every ledger, BEFORE the run writes.

    Call this immediately before ``run_weekly``; pass the result to
    :func:`build_manifest`. Read-only: it opens the parquet files and nothing
    else. A ledger that does not exist yet is recorded as zero rows, which is
    the correct "before" for Week 1's first ever write.
    """

    return {
        name: _ledger_snapshot(name, path) for name, path in ledger_paths(artifacts_root).items()
    }


def _appended_rows(path: Path, rows_before: int) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    appended = frame.iloc[int(rows_before) :]
    payload = appended.to_csv(index=False).encode("utf-8")
    return {
        "appended_rows": len(appended),
        "appended_rows_sha256": sha256_bytes(payload),
        "appended_rows_digest_method": APPENDED_ROWS_DIGEST_METHOD,
    }


def ledger_diff(
    artifacts_root: Path,
    before: Mapping[str, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Per-ledger before/after row counts plus a digest of this run's rows."""

    rows: list[dict[str, Any]] = []
    for name, path in ledger_paths(artifacts_root).items():
        prior = dict(before.get(name, {})) if before else {}
        after = _ledger_snapshot(name, path)
        rows_before = int(prior.get("rows", 0) or 0)
        entry: dict[str, Any] = {
            "ledger": name,
            "path": str(path),
            "captured_before": bool(prior),
            "exists_before": bool(prior.get("exists", False)),
            "exists_after": bool(after["exists"]),
            "rows_before": rows_before,
            "rows_after": int(after["rows"]),
            "sha256_before": prior.get("sha256"),
            "sha256_after": after["sha256"],
            "appended_rows": max(int(after["rows"]) - rows_before, 0),
            "appended_rows_sha256": None,
            "appended_rows_digest_method": APPENDED_ROWS_DIGEST_METHOD,
            "unchanged": bool(prior.get("sha256")) and prior.get("sha256") == after["sha256"],
            "error": after["error"],
        }
        if after["exists"] and int(after["rows"]) > rows_before:
            try:
                entry.update(_appended_rows(path, rows_before))
            except Exception as error:  # deliberately broad; see module docstring
                entry["error"] = _describe(error)
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# run-summary readers
# ---------------------------------------------------------------------------


def _steps(run_summary: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not run_summary:
        return []
    steps = run_summary.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _step_commands(run_summary: Mapping[str, Any] | None) -> list[list[str]]:
    commands: list[list[str]] = []
    for step in _steps(run_summary):
        command = step.get("command")
        if isinstance(command, list | tuple):
            commands.append([str(token) for token in command])
    return commands


def _referenced_tables(run_summary: Mapping[str, Any] | None) -> list[str]:
    """Every ``.parquet`` path any executed step named on its command line.

    Mechanical on purpose: ``weekly-run``'s plan passes the card's feature
    table as ``--features`` and the learned-availability build's outputs as
    ``--destination``/``--rates-destination``, and the set of flags has
    already changed once (the 2026-08-18 promotion moved the card path from
    ``player`` to ``weak_stack``). Reading every parquet-shaped token off the
    commands that ACTUALLY ran cannot go stale the way an enumerated flag
    list would.
    """

    seen: list[str] = []
    for command in _step_commands(run_summary):
        for token in command:
            if token.endswith(".parquet") and token not in seen:
                seen.append(token)
    return seen


def _snapshot_ids(run_summary: Mapping[str, Any] | None) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {}
    for command in _step_commands(run_summary):
        for index, token in enumerate(command[:-1]):
            if token in _SNAPSHOT_FLAGS:
                value = command[index + 1]
                bucket = ids.setdefault(token.lstrip("-"), [])
                if value not in bucket:
                    bucket.append(value)
    return ids


def _recorder_results(run_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    """Each step's output JSON verbatim, plus a flat challenger_id index.

    Verbatim matters: the recorders are deliberately fail-open
    (``{"recorded": 0, "error": ...}``), so the ONLY durable evidence that a
    challenger skipped for a documented reason rather than silently breaking
    is the JSON it returned at the time.
    """

    steps: dict[str, Any] = {}
    for step in _steps(run_summary):
        name = str(step.get("name", ""))
        if not name:
            continue
        steps[name] = {
            "number": step.get("number"),
            "status": step.get("status"),
            "seconds": step.get("seconds"),
            "error": step.get("error"),
            "output": step.get("output"),
        }

    by_challenger: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        challenger_id = node.get("challenger_id")
        if isinstance(challenger_id, str):
            by_challenger.setdefault(challenger_id, node)
        for value in node.values():
            walk(value)

    walk(dict(run_summary) if run_summary else {})
    return {"steps": steps, "by_challenger_id": by_challenger}


# ---------------------------------------------------------------------------
# lockday_verify
# ---------------------------------------------------------------------------

VerifyRunner = Callable[[Path, int, int, Mapping[str, Any] | None], dict[str, Any]]


def _load_lockday_verify(repo_root: Path) -> Any:
    """Import ``scripts/lockday_verify.py`` by path.

    ``scripts/`` is not part of the installed package, so this is a file-
    location import rather than a module import -- the same pattern
    ``nfl_ats.cli`` already uses for script reuse, and it keeps the verifier
    out of ``mypy src``'s import graph.
    """

    path = repo_root / "scripts" / "lockday_verify.py"
    spec = importlib.util.spec_from_file_location("nfl_ats_lockday_verify", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load lock-day verifier from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_lockday_verify(
    artifacts_root: Path,
    season: int,
    week: int,
    run_summary: Mapping[str, Any] | None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """The lock-day verifier's report plus its rendered human text."""

    module = _load_lockday_verify(repo_root or Path.cwd())
    report = module.verify(
        artifacts_root,
        season=season,
        week=week,
        run_summary=dict(run_summary) if run_summary else None,
    )
    payload = dict(report)
    payload["rendered"] = module.render(report)
    payload["exit_code"] = 1 if (report.get("missing") or report.get("pending_wiring")) else 0
    return payload


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _model_identity(artifacts_root: Path, repo_root: Path | None) -> dict[str, Any]:
    manifest_path = artifacts_root / "active_ats_model.json"
    manifest = load_active_ats_model(artifacts_root)
    if manifest is None:
        return {
            "available": False,
            "manifest_path": str(manifest_path),
            "manifest_sha256": None,
        }
    fields = (
        "model_id",
        "method",
        "target",
        "feature_profile",
        "regressor",
        "ridge_alpha",
        "calibration_method",
        "probability_method",
        "status",
        "version",
        "activated_at_utc",
        "feature_table_sha256",
        "evaluation_configuration_sha256",
    )
    identity: dict[str, Any] = {
        "available": True,
        "manifest_path": str(manifest_path),
        "manifest_repo_relative": _repo_relative(manifest_path, repo_root),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
    }
    for field in fields:
        identity[field] = manifest.get(field)
    identity["historical_evaluation"] = manifest.get("historical_evaluation")
    identity["weekly_forecast"] = manifest.get("weekly_forecast")
    return identity


def _input_section(
    data_root: Path,
    repo_root: Path | None,
    run_summary: Mapping[str, Any] | None,
    hashed: list[dict[str, Any]],
) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for raw in _referenced_tables(run_summary):
        path = Path(raw)
        entry = hash_entry("feature_table", path, repo_root=repo_root)
        hashed.append(entry)
        tables.append({"path": entry["path"], "sha256": entry["sha256"], "bytes": entry["bytes"]})
        manifest_path = path.with_name(f"{path.stem}.manifest.json")
        manifest_entry = hash_entry("feature_manifest", manifest_path, repo_root=repo_root)
        hashed.append(manifest_entry)
        content: Any = None
        if manifest_path.is_file():
            try:
                content = _read_json_file(manifest_path)
            except Exception as error:  # deliberately broad; see module docstring
                content = {"error": _describe(error)}
        manifests.append(
            {
                "path": manifest_entry["path"],
                "sha256": manifest_entry["sha256"],
                "manifest": content,
            }
        )

    processed = data_root / "processed"
    known = {entry["path"] for entry in manifests}
    for name in sorted(processed.glob("*.manifest.json")) if processed.is_dir() else []:
        if str(name) in known:
            continue
        entry = hash_entry("source_snapshot_manifest", name, repo_root=repo_root)
        hashed.append(entry)
        manifests.append({"path": entry["path"], "sha256": entry["sha256"], "manifest": None})

    return {
        "data_root": str(data_root),
        "feature_tables": tables,
        "snapshot_manifests": manifests,
        "snapshot_ids": _snapshot_ids(run_summary),
        "note": (
            "feature_tables are every .parquet path an executed weekly-run step named on "
            "its command line; snapshot_manifests hash each table's sibling "
            "<stem>.manifest.json plus every other manifest under data/processed/"
        ),
    }


def _output_section(
    artifacts_root: Path,
    repo_root: Path | None,
    card_paths: Sequence[Path],
    hashed: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = load_active_ats_model(artifacts_root)
    forecast_directory: Path | None = None
    evaluation_directory: Path | None = None
    if manifest is not None:
        forecast_directory = active_artifact_path(artifacts_root, manifest, "weekly_forecast")
        evaluation_directory = active_artifact_path(
            artifacts_root, manifest, "historical_evaluation"
        )

    forecast_files: list[dict[str, Any]] = []
    if forecast_directory is not None:
        forecast_files = _directory_entries("forecast", forecast_directory, repo_root=repo_root)
        hashed.extend(forecast_files)
    evaluation_files: list[dict[str, Any]] = []
    if evaluation_directory is not None:
        evaluation_files = _directory_entries(
            "historical_evaluation", evaluation_directory, repo_root=repo_root
        )
        hashed.extend(evaluation_files)

    cards: list[dict[str, Any]] = []
    for path in card_paths:
        entry = hash_entry("published_card", path, repo_root=repo_root)
        hashed.append(entry)
        cards.append(
            {
                "path": entry["path"],
                "repo_relative": entry["repo_relative"],
                "exists": entry["exists"],
                "sha256": entry["sha256"],
            }
        )

    return {
        "forecast": {
            "directory": str(forecast_directory) if forecast_directory else None,
            "files": [{"path": item["path"], "sha256": item["sha256"]} for item in forecast_files],
        },
        "historical_evaluation": {
            "directory": str(evaluation_directory) if evaluation_directory else None,
            "files": [
                {"path": item["path"], "sha256": item["sha256"]} for item in evaluation_files
            ],
        },
        "cards": cards,
    }


def default_card_paths(repo_root: Path) -> list[Path]:
    """The published artefacts a lock-day run rewrites, in publish order."""

    return [
        repo_root / "CURRENT_PREDICTIONS.md",
        repo_root / "docs" / "index.html",
    ]


def build_manifest(
    *,
    season: int,
    week: int,
    artifacts_root: Path,
    data_root: Path,
    repo_root: Path,
    run_summary: Mapping[str, Any] | None = None,
    ledger_state_before: Mapping[str, Mapping[str, Any]] | None = None,
    card_paths: Sequence[Path] | None = None,
    rehearsal: bool = False,
    now: datetime | None = None,
    verify_runner: VerifyRunner | None = None,
    command: str = "weekly-run --record-decisions",
) -> dict[str, Any]:
    """Assemble the manifest dict. Never raises; failures land in ``errors``."""

    errors: list[dict[str, Any]] = []
    hashed: list[dict[str, Any]] = []
    instant = (now or datetime.now(UTC)).astimezone(UTC)

    manifest: dict[str, Any] = {
        "kind": PACKAGE_KIND,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "season": int(season),
        "week": int(week),
        "rehearsal": bool(rehearsal),
        "command": command,
        "created_at_utc": instant.isoformat(),
        "repo_root": str(repo_root),
        "artifacts_root": str(artifacts_root),
        "data_root": str(data_root),
    }

    manifest["code"] = _collect(
        errors,
        "code",
        lambda: {
            **git_state(repo_root),
            "diff_sha256": git_diff_sha256(repo_root),
            "uv_lock_sha256": (
                sha256_file(repo_root / "uv.lock") if (repo_root / "uv.lock").is_file() else None
            ),
        },
        default={"revision": None, "dirty": None, "diff_sha256": None, "uv_lock_sha256": None},
    )
    # ENG-21: the deterministic environment lock report (Python/uv/package/
    # platform/env-var details, secrets redacted to booleans). Additive to
    # "code" above, not a replacement -- environment_report() never raises,
    # so this can only ever add a section, never break the lock-day package.
    manifest["environment"] = _collect(
        errors,
        "environment",
        lambda: environment_report(project_root=repo_root),
        default={"error": "unavailable"},
    )
    manifest["model_identity"] = _collect(
        errors,
        "model_identity",
        lambda: _model_identity(artifacts_root, repo_root),
        default={"available": False},
    )
    manifest["inputs"] = _collect(
        errors,
        "inputs",
        lambda: _input_section(data_root, repo_root, run_summary, hashed),
        default={"feature_tables": [], "snapshot_manifests": []},
    )
    manifest["outputs"] = _collect(
        errors,
        "outputs",
        lambda: _output_section(
            artifacts_root,
            repo_root,
            list(card_paths) if card_paths is not None else default_card_paths(repo_root),
            hashed,
        ),
        default={"forecast": None, "historical_evaluation": None, "cards": []},
    )
    manifest["recorders"] = _collect(
        errors,
        "recorders",
        lambda: _recorder_results(run_summary),
        default={"steps": {}, "by_challenger_id": {}},
    )
    ledgers = _collect(
        errors,
        "ledgers",
        lambda: ledger_diff(artifacts_root, ledger_state_before),
        default=[],
    )
    manifest["ledgers"] = ledgers
    for ledger in ledgers or []:
        path = Path(str(ledger.get("path", "")))
        entry = hash_entry(
            "ledger_after",
            path,
            repo_root=repo_root,
            note="append-only: later refresh passes legitimately change these bytes",
        )
        hashed.append(entry)
    manifest["lockday_verify"] = _collect(
        errors,
        "lockday_verify",
        lambda: (
            verify_runner(artifacts_root, int(season), int(week), run_summary)
            if verify_runner is not None
            else run_lockday_verify(
                artifacts_root, int(season), int(week), run_summary, repo_root=repo_root
            )
        ),
        default=None,
    )
    manifest["run_summary"] = dict(run_summary) if run_summary else None
    manifest["hashed_files"] = hashed
    manifest["errors"] = errors
    manifest["ok"] = not errors
    return manifest


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def packages_root(artifacts_root: Path) -> Path:
    return artifacts_root / PACKAGES_DIRNAME


def package_directory(
    root: Path,
    season: int,
    week: int,
    *,
    now: datetime | None = None,
) -> Path:
    """A fresh, never-reused directory: ``<season>_wk<week>_<UTC stamp>``.

    The stamp has one-second resolution, so a same-second second write gets a
    ``-2`` suffix rather than landing on top of an existing, read-only
    package.
    """

    base = f"{int(season)}_wk{int(week):02d}_{run_id(now)}"
    candidate = root / base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base}-{suffix}"
    return candidate


def _set_read_only(path: Path) -> bool:
    """Best-effort immutability flag. Never fatal: a package that exists and
    is writable beats no package at all."""

    try:
        mode = path.stat().st_mode
        os.chmod(path, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        return not os.access(path, os.W_OK)
    except OSError:
        return False


def package_readme(manifest: Mapping[str, Any]) -> str:
    season = manifest.get("season")
    week = manifest.get("week")
    rehearsal = " (REHEARSAL -- not a real lock)" if manifest.get("rehearsal") else ""
    return f"""# Lock-day decision package -- {season} week {week}{rehearsal}

Written by `{manifest.get("command")}` at {manifest.get("created_at_utc")}.

This folder is the durable answer to "what did we decide, from what, with
which model" for one lock. `artifacts/` is gitignored and local-disk-only, so
this package -- not the surrounding tree -- is what has to survive.

## Files

| file | what it is |
|---|---|
| `{MANIFEST_FILENAME}` | the package. Plain JSON, written read-only. |
| `{MANIFEST_DIGEST_FILENAME}` | SHA-256 of `{MANIFEST_FILENAME}`, so an edit to it is detectable. |
| `{PACKAGE_README_FILENAME}` | this file. |

## What the manifest links

* `code` -- git revision, dirty flag, working-tree diff digest, `uv.lock` digest.
* `model_identity` -- the active model manifest's id, method, feature profile,
  regressor, ridge alpha, calibration and probability method, plus the digest
  of `active_ats_model.json` itself.
* `inputs` -- every feature table an executed step named, each with its SHA-256
  and its sibling build manifest (which carries the source snapshot ids), plus
  every other `data/processed/*.manifest.json`.
* `outputs` -- the weekly forecast directory and the historical evaluation
  directory file-by-file, and the published card(s).
* `recorders` -- each `weekly-run` step's output JSON **verbatim**, plus a flat
  index by `challenger_id`. The recorders are fail-open by design, so this is
  the only durable evidence that a challenger skipped for a documented reason
  rather than breaking silently.
* `ledgers` -- every append-only ledger's path, row count before and after this
  run, file digest before and after, and a SHA-256 of the rows THIS run
  appended.
* `lockday_verify` -- `scripts/lockday_verify.py`'s report for this season/week,
  including its rendered text and exit code.
* `run_summary` -- the whole `weekly-run` JSON summary, verbatim.
* `errors` -- components that failed while the package was assembled. A
  non-empty list is a degraded package, not an aborted lock: by the time this
  runs, the ledger rows are already written and the card is already published,
  so nothing here is ever allowed to roll a lock back.

## Verifying it independently

```powershell
.\\.tools\\uv.exe run --no-sync python scripts/lockday_package_verify.py "<this folder>"
```

That recomputes `{MANIFEST_DIGEST_FILENAME}` against `{MANIFEST_FILENAME}` and re-hashes
every entry in `hashed_files`. Ledger entries are flagged `mutable` because
later in-week refresh passes legitimately append to them; everything else must
match byte for byte.

Without this repository at all, the manifest is still readable: every digest
names its algorithm (`sha256`) and the exact bytes hashed, and the appended-row
digest states its own recipe in `appended_rows_digest_method`.

## What this package is NOT

It is not tamper-proof. The read-only attribute and the sibling digest stop an
accidental edit and make a deliberate one visible; they do not prevent one.
"""


def write_package(
    manifest: Mapping[str, Any],
    directory: Path,
) -> dict[str, Any]:
    """Write manifest.json, its digest, and the README; flag them read-only."""

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_FILENAME
    atomic_json(dict(manifest), manifest_path)
    digest = sha256_file(manifest_path)
    digest_path = directory / MANIFEST_DIGEST_FILENAME
    atomic_text(f"{digest}  {MANIFEST_FILENAME}\n", digest_path)
    readme_path = directory / PACKAGE_README_FILENAME
    atomic_text(package_readme(manifest), readme_path)
    read_only = _set_read_only(manifest_path)
    _set_read_only(digest_path)
    return {
        "package_directory": str(directory),
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "manifest_sha256_path": str(digest_path),
        "readme_path": str(readme_path),
        "read_only": read_only,
        "rehearsal": bool(manifest.get("rehearsal", False)),
        "errors": list(manifest.get("errors", [])),
    }


def write_decision_package(
    *,
    season: int,
    week: int,
    artifacts_root: Path,
    data_root: Path,
    repo_root: Path,
    run_summary: Mapping[str, Any] | None = None,
    ledger_state_before: Mapping[str, Mapping[str, Any]] | None = None,
    card_paths: Sequence[Path] | None = None,
    rehearsal: bool = False,
    now: datetime | None = None,
    destination: Path | None = None,
    verify_runner: VerifyRunner | None = None,
    command: str = "weekly-run --record-decisions",
) -> dict[str, Any]:
    """Build and write the package. **Never raises.**

    This is the entry point ``weekly-run`` calls as its last step. By then the
    ledger rows are appended and the card is published, so the contract is
    absolute: any failure here is reported (in the returned payload and on
    stderr) and the lock stands.
    """

    try:
        manifest = build_manifest(
            season=season,
            week=week,
            artifacts_root=artifacts_root,
            data_root=data_root,
            repo_root=repo_root,
            run_summary=run_summary,
            ledger_state_before=ledger_state_before,
            card_paths=card_paths,
            rehearsal=rehearsal,
            now=now,
            verify_runner=verify_runner,
            command=command,
        )
        directory = destination or package_directory(
            packages_root(artifacts_root), season, week, now=now
        )
        written = write_package(manifest, directory)
        written["written"] = True
        written["ok"] = bool(manifest.get("ok", False))
        return written
    except Exception as error:  # deliberately broad; see module docstring
        print(
            "lockday package: FAILED to write the decision package "
            f"({_describe(error)}). The lock itself is unaffected: the ledger rows "
            "and the published card were written before this step.",
            file=sys.stderr,
        )
        return {
            "written": False,
            "ok": False,
            "package_directory": None,
            "manifest_path": None,
            "errors": [{"component": "write_decision_package", "error": _describe(error)}],
        }


# ---------------------------------------------------------------------------
# reading and verifying
# ---------------------------------------------------------------------------


def resolve_manifest_path(path: Path) -> Path:
    """Accept either the package folder or the manifest file itself."""

    return path / MANIFEST_FILENAME if path.is_dir() else path


def load_package(path: Path) -> dict[str, Any]:
    """Read a written package back. Accepts the folder or the manifest file."""

    manifest_path = resolve_manifest_path(path)
    payload = _read_json_file(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Not a lock-day decision package manifest: {manifest_path}")
    if payload.get("kind") != PACKAGE_KIND:
        raise ValueError(
            f"{manifest_path} is not a {PACKAGE_KIND} manifest (kind={payload.get('kind')!r})"
        )
    return payload


def _resolve_hashed_path(entry: Mapping[str, Any], repo_root: Path | None) -> Path | None:
    recorded = Path(str(entry.get("path", "")))
    if recorded.is_file():
        return recorded
    relative = entry.get("repo_relative")
    if repo_root is not None and isinstance(relative, str) and relative:
        candidate = repo_root / relative
        if candidate.is_file():
            return candidate
    return recorded if str(recorded) else None


def verify_package(
    path: Path,
    *,
    repo_root: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Recompute every digest the package claims.

    ``ok`` is the package's own integrity: the manifest matches its recorded
    digest, and every immutable file that is still on disk still hashes to
    what the manifest says. Ledgers are ``mutable`` -- later in-week refresh
    passes append to them by design -- so a changed ledger is reported, never
    fatal. ``strict`` additionally requires every file that WAS hashed at
    write time to still exist, which ``artifacts/`` being gitignored and
    local-disk-only makes a deliberate opt-in rather than the default. Entries
    carrying no digest at all (a ledger this lock never wrote, an over-size
    file) are reported under ``unhashed`` and are fatal in neither mode --
    there is no claim to check.
    """

    manifest_path = resolve_manifest_path(path)
    directory = manifest_path.parent
    manifest = load_package(manifest_path)

    digest_path = directory / MANIFEST_DIGEST_FILENAME
    recorded_digest = None
    if digest_path.is_file():
        recorded_digest = digest_path.read_text(encoding="utf-8").split()[0].strip()
    actual_digest = sha256_file(manifest_path)
    manifest_ok = recorded_digest is not None and recorded_digest == actual_digest

    verified: list[str] = []
    changed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    unhashed: list[dict[str, Any]] = []
    mutable_changed: list[dict[str, Any]] = []
    for entry in manifest.get("hashed_files", []):
        if not isinstance(entry, Mapping):
            continue
        expected = entry.get("sha256")
        role = str(entry.get("role", ""))
        described = {"role": role, "path": entry.get("path")}
        if not expected:
            unhashed.append({**described, "note": entry.get("note") or entry.get("error")})
            continue
        resolved = _resolve_hashed_path(entry, repo_root)
        if resolved is None or not resolved.is_file():
            missing.append(described)
            continue
        actual = sha256_file(resolved)
        if actual == expected:
            verified.append(str(resolved))
        elif entry.get("mutable"):
            mutable_changed.append({**described, "expected": expected, "actual": actual})
        else:
            changed.append({**described, "expected": expected, "actual": actual})

    ok = manifest_ok and not changed and (not missing if strict else True)
    return {
        "package_directory": str(directory),
        "manifest_path": str(manifest_path),
        "kind": manifest.get("kind"),
        "schema_version": manifest.get("schema_version"),
        "season": manifest.get("season"),
        "week": manifest.get("week"),
        "rehearsal": bool(manifest.get("rehearsal", False)),
        "strict": bool(strict),
        "manifest_sha256_recorded": recorded_digest,
        "manifest_sha256_actual": actual_digest,
        "manifest_sha256_ok": manifest_ok,
        "files_checked": len(verified) + len(changed) + len(missing) + len(mutable_changed),
        "files_verified": len(verified),
        "changed": changed,
        "mutable_changed": mutable_changed,
        "missing": missing,
        "unhashed": unhashed,
        "build_errors": list(manifest.get("errors", [])),
        "ok": bool(ok),
    }


def _ledger_lines(manifest: Mapping[str, Any]) -> Iterable[str]:
    ledgers = manifest.get("ledgers")
    if not isinstance(ledgers, list) or not ledgers:
        yield "  (no ledger section)"
        return
    width = max(len(str(row.get("ledger", ""))) for row in ledgers if isinstance(row, Mapping))
    for row in ledgers:
        if not isinstance(row, Mapping):
            continue
        appended = int(row.get("appended_rows", 0) or 0)
        marker = "+" if appended else " "
        digest = str(row.get("appended_rows_sha256") or "")[:12]
        line = (
            f"  {marker} {row.get('ledger', '')!s:<{width}}  "
            f"{row.get('rows_before', 0):>4} -> {row.get('rows_after', 0):<4} rows"
        )
        if appended:
            line += f"   appended sha256 {digest}..."
        if row.get("error"):
            line += f"   ERROR {row['error']}"
        yield line


def summarise_package(manifest: Mapping[str, Any]) -> str:
    """A human-readable read of one package, for a session report."""

    identity = manifest.get("model_identity") or {}
    code = manifest.get("code") or {}
    verify = manifest.get("lockday_verify") or {}
    inputs = manifest.get("inputs") or {}
    outputs = manifest.get("outputs") or {}
    errors = manifest.get("errors") or []
    rehearsal = "  [REHEARSAL]" if manifest.get("rehearsal") else ""

    lines = [
        f"lock-day decision package  {manifest.get('season')} week {manifest.get('week')}"
        f"{rehearsal}",
        f"  written        : {manifest.get('created_at_utc')}  by {manifest.get('command')}",
        f"  code           : {str(code.get('revision') or '(unknown)')[:12]}"
        f"{'  DIRTY' if code.get('dirty') else ''}",
        f"  model          : {identity.get('model_id')}  "
        f"{identity.get('method')}/{identity.get('feature_profile')}  "
        f"{identity.get('regressor')} alpha={identity.get('ridge_alpha')}  "
        f"calibration={identity.get('calibration_method')} "
        f"probability={identity.get('probability_method')}",
        f"  inputs         : {len(inputs.get('feature_tables') or [])} feature tables, "
        f"{len(inputs.get('snapshot_manifests') or [])} source manifests",
        f"  outputs        : forecast {(outputs.get('forecast') or {}).get('directory')}",
    ]
    for card in outputs.get("cards") or []:
        lines.append(
            f"  card           : {card.get('path')}  {str(card.get('sha256') or '(missing)')[:12]}"
        )
    lines.append(f"  hashed files   : {len(manifest.get('hashed_files') or [])}")
    lines.append("  ledgers        :")
    lines.extend(_ledger_lines(manifest))
    if verify:
        lines.append(
            f"  lockday_verify : {verify.get('recorded', 0)} recorded, "
            f"{verify.get('skipped', 0)} skipped, {len(verify.get('missing') or [])} MISSING, "
            f"{len(verify.get('pending_wiring') or [])} pending wiring"
        )
    if errors:
        lines.append(f"  BUILD ERRORS   : {len(errors)}")
        lines.extend(
            f"    {item.get('component')}: {item.get('error')}"
            for item in errors
            if isinstance(item, Mapping)
        )
    else:
        lines.append("  build errors   : none")
    return "\n".join(lines)
