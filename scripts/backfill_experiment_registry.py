"""One-time mechanical backfill of the RWB-09 experiment-provenance registry.

``registry/experiments/`` (git-tracked) is new; ``artifacts/`` (gitignored,
local-disk-only) already holds ~100 run directories from before it existed.
This script lifts a registry row for every run directory whose
``metadata.json``/``run.json`` already carries an ``artifact_provenance()``
-shaped block -- no invented data, purely a mechanical copy of what is
already on disk into the durable, tracked location.

Anything that carries no such block gets NO invented row. Per this project's
"label how you know it" rule, an approximated ``code_revision`` (say, from
``git log --before=<file mtime>``) would not actually PIN the code that ran --
it would just be a guess wearing a fact's clothes. Those run directories are
instead listed, with a reason, in ``registry/experiments/UNBACKFILLABLE.md``:
the honest record that ``docs/closure_audit.md``'s PageRank/HITS closure
never had (nothing to backfill FROM -- the artifact directory itself never
existed).

    ./.tools/uv.exe run --no-sync python scripts/backfill_experiment_registry.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_ats.provenance import (
    ExperimentRecord,
    bounded_metrics,
    default_experiment_registry_root,
    experiment_command_slug,
    save_experiment_record,
)

_METADATA_FILENAMES: tuple[str, ...] = ("metadata.json", "run.json")
# What an artifact_provenance() dict looks like -- present either as the
# metadata file's own top level (the bare "run.json" convention) or under
# some key inside it ("provenance", "baseline_provenance",
# "provenance_candidate", ...). Any dict carrying both of these is treated as
# one, regardless of what key it sits under.
_PROVENANCE_SHAPE_KEYS: tuple[str, ...] = ("code", "configuration_sha256")


def _provenance_blocks(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every key (including the payload itself) shaped like ``artifact_provenance()``."""

    blocks: list[tuple[str, dict[str, Any]]] = []
    if all(key in payload for key in _PROVENANCE_SHAPE_KEYS):
        blocks.append(("<root>", payload))
    for key, value in sorted(payload.items()):
        if isinstance(value, dict) and all(k in value for k in _PROVENANCE_SHAPE_KEYS):
            blocks.append((key, value))
    return blocks


def _stamp_to_iso(stamp: str) -> str | None:
    try:
        parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return parsed.isoformat()


def _command_from(
    payload: dict[str, Any], chosen_provenance: dict[str, Any], directory: Path
) -> tuple[str, bool]:
    """The command string, and whether it was recorded (True) vs inferred (False).

    Checked in order: the metadata file's own top-level ``command`` (the
    ``cli.py`` ``metadata.json`` convention); the chosen provenance block's
    ``configuration.command`` (the bare ``run.json`` convention, where
    ``command`` lives one level deeper, inside the config dict that was
    hashed); and only then the artifact directory's parent folder name, which
    is a real but less precise source (e.g. "backtests" vs the command's own
    "backtest").
    """

    for candidate in (
        payload.get("command"),
        (chosen_provenance.get("configuration") or {}).get("command"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate, True
    return directory.parent.name, False


def backfill_run_directory(directory: Path) -> tuple[ExperimentRecord | None, str | None]:
    """One backfill row for an ``artifacts/<command>/<stamp>/`` directory.

    Returns ``(record, None)`` on success or ``(None, reason)`` when nothing
    in the directory can be mechanically lifted.
    """

    found_file: Path | None = None
    payload: dict[str, Any] | None = None
    for filename in _METADATA_FILENAMES:
        candidate = directory / filename
        if candidate.is_file():
            found_file = candidate
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            break
    if found_file is None or payload is None:
        return None, "no metadata.json or run.json in this run directory"

    blocks = _provenance_blocks(payload)
    if not blocks:
        return None, (
            f"{found_file.name} exists but carries no artifact_provenance()-shaped "
            "block (no code revision or configuration hash to lift)"
        )

    chosen_key, chosen = blocks[0]
    alternate_keys = [key for key, _ in blocks[1:]]

    command, command_recorded = _command_from(payload, chosen, directory)
    stamp = directory.name
    if payload.get("created_at_utc"):
        recorded_at = str(payload["created_at_utc"])
    else:
        recorded_at = (
            _stamp_to_iso(stamp)
            or datetime.fromtimestamp(found_file.stat().st_mtime, tz=UTC).isoformat()
        )

    code = chosen.get("code") or {}
    feature_table = chosen.get("feature_table") or {}

    note_parts = [f"Mechanically backfilled from {found_file} (provenance key {chosen_key!r})."]
    if alternate_keys:
        note_parts.append(
            f"File also carries provenance under: {', '.join(alternate_keys)}; "
            f"only {chosen_key!r} was used for this row."
        )
    if not command_recorded:
        note_parts.append(
            f"No top-level 'command' field in the artifact; {command!r} is the artifact "
            "directory's parent folder name, not a recorded value."
        )

    provenance_keys = {key for key, _ in blocks}
    metrics_source = {k: v for k, v in payload.items() if k not in provenance_keys}

    record = ExperimentRecord(
        experiment_id=f"{command}/{stamp}",
        recorded_at=recorded_at,
        command=command,
        artifact_directory=str(directory).replace("\\", "/"),
        config_hash=str(chosen.get("configuration_sha256", "")),
        schema_version=1,
        metrics=bounded_metrics(metrics_source),
        source=str(found_file).replace("\\", "/"),
        code_revision=code.get("revision"),
        code_dirty=code.get("dirty"),
        # Backfill cannot recover a historical working-tree diff -- only a
        # LIVE write_experiment_artifact() call can hash `git diff HEAD` at
        # the moment the dirty run actually happened.
        code_diff_sha256=None,
        feature_table_sha256=feature_table.get("sha256"),
        uv_lock_sha256=chosen.get("uv_lock_sha256"),
        notes="",
        provenance_backfilled=True,
        backfill_note=" ".join(note_parts),
    )
    return record, None


def find_run_directories(artifacts_root: Path) -> list[Path]:
    if not artifacts_root.is_dir():
        return []
    return sorted(path for path in artifacts_root.glob("*/*") if path.is_dir())


def _write_unbackfillable_doc(path: Path, entries: list[tuple[str, str]]) -> None:
    lines = [
        "# Unbackfillable artifacts",
        "",
        "RWB-09 experiment-provenance registry: `scripts/backfill_experiment_registry.py`",
        "could not produce a registry row for these `artifacts/` run directories, because",
        "nothing in them carries an `artifact_provenance()`-shaped block (a git revision",
        "or a configuration hash) to lift.",
        "",
        'Per this project\'s "label how you know it" rule, an approximated',
        "`code_revision` (e.g. from `git log --before=<file mtime>`) would not actually",
        "PIN the code that ran -- it would be a guess wearing a fact's clothes. So these",
        "are recorded here, with a reason, rather than backfilled with an invented value.",
        "This is the honest record `docs/closure_audit.md` S3's PageRank/HITS closure",
        "never had: there, no artifact directory existed at all, so there was nothing",
        "even to list. Here, the directory exists but never captured provenance.",
        "",
        f"{len(entries)} run directories, as of this backfill pass:",
        "",
    ]
    for directory, reason in sorted(entries):
        lines.append(f"- `{directory}` -- {reason}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_backfill(artifacts_root: Path, registry_root: Path) -> dict[str, Any]:
    backfilled: list[str] = []
    unbackfillable: list[tuple[str, str]] = []
    for directory in find_run_directories(artifacts_root):
        record, reason = backfill_run_directory(directory)
        if record is None:
            unbackfillable.append((str(directory).replace("\\", "/"), reason or "unknown reason"))
            continue
        destination = (
            default_experiment_registry_root(registry_root)
            / experiment_command_slug(record.command)
            / f"{directory.name}.json"
        )
        save_experiment_record(record, destination)
        backfilled.append(record.experiment_id)

    unbackfillable_path = default_experiment_registry_root(registry_root) / "UNBACKFILLABLE.md"
    _write_unbackfillable_doc(unbackfillable_path, unbackfillable)

    return {
        "run_directories": len(backfilled) + len(unbackfillable),
        "backfilled": len(backfilled),
        "unbackfillable": len(unbackfillable),
        "unbackfillable_doc": str(unbackfillable_path).replace("\\", "/"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--registry-root", type=Path, default=Path("registry"))
    args = parser.parse_args()
    summary = run_backfill(args.artifacts_root, args.registry_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
