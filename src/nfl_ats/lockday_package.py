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

PACKAGE_SCHEMA_VERSION = 1
PACKAGE_KIND = "lockday_decision_package"
PACKAGES_DIRNAME = "lockday_packages"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_DIGEST_FILENAME = "manifest.sha256"
PACKAGE_README_FILENAME = "README.md"

MAX_HASHED_BYTES = 512 * 1024 * 1024

LEDGER_PATH_FUNCTIONS: dict[str, Callable[[Path], Path]] = {
    "paper_decisions": paper_decision_ledger_path,
    "challenger_decisions": challenger_ledger_path,
    "pick_revisions": pick_revision_ledger_path,
    "injury_signal_refresh_decisions": injury_signal_ledger_path,
    "nflcom_friday_refresh_decisions": nflcom_refresh_overlay_ledger_path,
    "inactives_refresh_decisions": inactives_refresh_overlay_ledger_path,
    "crew_tilt_refresh_decisions": crew_tilt_refresh_ledger_path,
}

APPENDED_ROWS_DIGEST_METHOD = (
    "sha256 of pandas.read_parquet(path).iloc[rows_before:].to_csv(index=False) "
    "encoded utf-8; the ledgers are append-only, so the tail beyond rows_before "
    "is exactly this run's write"
)

_SNAPSHOT_FLAGS = (
    "--snapshot",
    "--player-snapshot",
    "--player-value-snapshot",
    "--pbp-snapshot",
)

MUTABLE_ROLES = frozenset({"ledger_after"})


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _collect(
    errors: list[dict[str, Any]],
    component: str,
    builder: Callable[[], Any],
    default: Any = None,
) -> Any:

    try:
        return builder()
    except Exception as error:
        errors.append({"component": component, "error": _describe(error)})
        print(f"lockday package: component {component!r} failed: {error}", file=sys.stderr)
        return default


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
    except Exception as error:
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


def ledger_paths(artifacts_root: Path) -> dict[str, Path]:

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
    except Exception as error:
        entry["error"] = _describe(error)
    return entry


def capture_ledger_state(artifacts_root: Path) -> dict[str, dict[str, Any]]:

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
            except Exception as error:
                entry["error"] = _describe(error)
        rows.append(entry)
    return rows


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


VerifyRunner = Callable[[Path, int, int, Mapping[str, Any] | None], dict[str, Any]]


def _load_lockday_verify(repo_root: Path) -> Any:

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
            except Exception as error:
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


def packages_root(artifacts_root: Path) -> Path:
    return artifacts_root / PACKAGES_DIRNAME


def package_directory(
    root: Path,
    season: int,
    week: int,
    *,
    now: datetime | None = None,
) -> Path:

    base = f"{int(season)}_wk{int(week):02d}_{run_id(now)}"
    candidate = root / base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base}-{suffix}"
    return candidate


def _set_read_only(path: Path) -> bool:

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
    except Exception as error:
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


def resolve_manifest_path(path: Path) -> Path:

    return path / MANIFEST_FILENAME if path.is_dir() else path


def load_package(path: Path) -> dict[str, Any]:

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
