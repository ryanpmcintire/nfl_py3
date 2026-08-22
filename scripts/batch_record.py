"""Serializing batch recorder for ``nfl-ats weak-signals record``.

During mass recording waves many worker agents need to append to
``registry/weak_signals.json``. Concurrent direct invocations race on the
registry write. The fleet convention is therefore: workers NEVER call the
recorder CLI directly; they enqueue validated task files into a shared queue
directory, and ONE recorder process drains the queue oldest-first while
holding an OS-level lock, so two concurrent drains serialize instead of
interleaving registry writes.

Queue convention: ``<queue-dir>/<priority>-<timestamp>-<nonce>.json`` where
the file body is a single JSON object of snake_case ``weak-signals record``
flag values. Filenames sort priority-major then chronological (FIFO) because
the timestamp carries microsecond precision. The default queue directory is
``<temp>/nfl_ats_record_queue`` (override per command with ``--queue-dir`` or
the ``NFL_ATS_RECORD_QUEUE`` environment variable).

Commands:
    enqueue  validate a task file (object or list of objects) and drop it in
             the queue; rejects unknown fields and bad enums up front with
             actionable errors; prints the queue id per accepted task
    drain    process every queued file oldest-first under the lock. WITHOUT
             ``--execute`` this is a dry-run: it prints the exact command
             lines and touches nothing. WITH ``--execute`` it invokes the real
             CLI per task; success moves the file to ``<queue>/done/``, any
             failure moves it to ``<queue>/failed/`` beside a captured-stderr
             ``.stderr.txt`` file. Done and failed files leave the queue, so
             draining twice never double-records.
    status   queue/done/failed counts and listings.

The lock is a byte-range lock (``msvcrt.locking`` on Windows, ``fcntl.flock``
elsewhere) on ``<queue-dir>/drain.lock`` held for the whole drain, including
listing the queue, so a second drain either waits or fails loudly rather than
processing the same files.

If a record command fails, the file lands in ``failed/`` with its stderr;
per repository policy a record error means the verdict or payload is wrong,
not the validator -- fix the payload and re-enqueue, never bypass.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import msvcrt
except ImportError:
    msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None

REPO = Path(__file__).resolve().parents[1]

QUEUE_DIR_NAME = "nfl_ats_record_queue"
LOCK_FILE_NAME = "drain.lock"
DEFAULT_PRIORITY = 5
FAILED_DIR_NAME = "failed"
DONE_DIR_NAME = "done"

EFFECT_UNITS = frozenset({"ats_points", "accuracy_points", "brier", "log_loss", "mae"})
CLASSIFICATIONS = frozenset({"unresolved_below_power", "refuted_mechanism", "bounded_by_control"})
LEAGUES = frozenset({"nfl", "cfb"})
CLOSING_GROUNDS = frozenset(
    {"wrong_sign_resolved", "no_split_half_reliability", "positive_control_bound"}
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "source",
    "effect",
    "effect_units",
    "classification",
    "league",
    "season_start",
    "season_end",
    "interval_low",
    "interval_high",
    "probability_positive",
    "sample_games",
    "sample_blocks",
)
OPTIONAL_FIELDS: tuple[str, ...] = (
    "standard_error",
    "reliability",
    "classification_evidence",
    "closing_ground",
    "recorded_at",
    "notes",
    "replace",
)
FLAG_FIELD_ORDER: tuple[str, ...] = (
    "name",
    "description",
    "source",
    "effect",
    "effect_units",
    "classification",
    "league",
    "season_start",
    "season_end",
    "standard_error",
    "interval_low",
    "interval_high",
    "probability_positive",
    "sample_games",
    "sample_blocks",
    "reliability",
    "classification_evidence",
    "closing_ground",
    "notes",
    "recorded_at",
)
STRING_FIELDS = frozenset(
    {"name", "description", "source", "classification_evidence", "recorded_at", "notes"}
)
NUMBER_FIELDS = frozenset(
    {
        "effect",
        "interval_low",
        "interval_high",
        "probability_positive",
        "standard_error",
        "reliability",
    }
)
INTEGER_FIELDS = frozenset({"season_start", "season_end", "sample_games", "sample_blocks"})
ENUM_FIELDS: dict[str, frozenset[str]] = {
    "effect_units": EFFECT_UNITS,
    "classification": CLASSIFICATIONS,
    "league": LEAGUES,
}
TERMINAL_CLASSIFICATIONS = frozenset({"refuted_mechanism", "bounded_by_control"})


class ValidationError(ValueError):
    """A task payload does not match the weak-signals record contract."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_task(task: Any) -> None:
    """Validate one task object against the actual record CLI flags."""
    if not isinstance(task, dict):
        raise ValidationError(
            "task must be a single JSON object mapping flag names to values, "
            f"got {type(task).__name__}"
        )
    errors: list[str] = []
    allowed = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)
    unknown = sorted(set(task) - allowed)
    if unknown:
        errors.append(
            "unknown field(s) "
            + ", ".join(repr(field) for field in unknown)
            + "; allowed fields: "
            + ", ".join(sorted(allowed))
        )
    missing = [field for field in REQUIRED_FIELDS if field not in task]
    if missing:
        errors.append("missing required field(s): " + ", ".join(repr(field) for field in missing))
    for field, value in task.items():
        if field in STRING_FIELDS and not isinstance(value, str):
            errors.append(f"{field!r} must be a string, got {type(value).__name__}")
        elif field in NUMBER_FIELDS and not _is_number(value):
            errors.append(f"{field!r} must be a number, got {type(value).__name__}")
        elif field in INTEGER_FIELDS and not _is_integer(value):
            errors.append(f"{field!r} must be an integer, got {type(value).__name__}")
        elif field in ENUM_FIELDS and value not in ENUM_FIELDS[field]:
            errors.append(f"{field!r} must be one of {sorted(ENUM_FIELDS[field])}, got {value!r}")
        elif field == "closing_ground" and value not in CLOSING_GROUNDS:
            errors.append(
                f"'closing_ground' must be one of {sorted(CLOSING_GROUNDS)}, got {value!r}"
            )
        elif field == "replace" and not isinstance(value, bool):
            errors.append(f"'replace' must be a boolean, got {type(value).__name__}")
        elif field == "name" and isinstance(value, str) and not value.strip():
            errors.append("'name' must be a non-empty string")
    low = task.get("interval_low")
    high = task.get("interval_high")
    if _is_number(low) and _is_number(high) and low > high:
        errors.append(f"'interval_low' ({low}) must be <= 'interval_high' ({high})")
    prob = task.get("probability_positive")
    if _is_number(prob) and not 0.0 <= float(prob) <= 1.0:
        errors.append(f"'probability_positive' must be within [0, 1], got {prob!r}")
    start = task.get("season_start")
    end = task.get("season_end")
    if _is_integer(start) and _is_integer(end) and start > end:
        errors.append(f"'season_start' ({start}) must be <= 'season_end' ({end})")
    classification = task.get("classification")
    if classification in TERMINAL_CLASSIFICATIONS and "closing_ground" not in task:
        errors.append(
            f"classification {classification!r} is terminal and requires an admissible "
            f"'closing_ground' from {sorted(CLOSING_GROUNDS)}; an interval crossing "
            "zero is NOT a closing ground (repository policy)"
        )
    ground = task.get("closing_ground")
    if (
        ground == "wrong_sign_resolved"
        and _is_number(low)
        and _is_number(high)
        and max(float(low), float(high)) >= 0.0
    ):
        errors.append(
            f"'wrong_sign_resolved' requires the whole interval below zero; got [{low}, {high}]"
        )
    if errors:
        name = task.get("name") if isinstance(task, dict) else None
        prefix = f"invalid task {name!r}: " if name else "invalid task: "
        raise ValidationError(prefix + "; ".join(errors))


def default_queue_dir() -> Path:
    env = os.environ.get("NFL_ATS_RECORD_QUEUE")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / QUEUE_DIR_NAME


def queue_file_name(priority: int, nonce: str | None = None) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    token = nonce if nonce is not None else secrets.token_hex(4)
    return f"{priority:02d}-{stamp}-{token}.json"


def queued_files(queue_dir: Path) -> list[Path]:
    return sorted(queue_dir.glob("*.json"))


def _format_flag_value(value: Any) -> str:
    if isinstance(value, bool):
        raise ValidationError("boolean flags never take a value here")
    if isinstance(value, float):
        return repr(value)
    return str(value)


def build_command(task: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nfl_ats.cli",
        "weak-signals",
        "record",
    ]
    for field in FLAG_FIELD_ORDER:
        if task.get(field) is not None:
            command += ["--" + field.replace("_", "-"), _format_flag_value(task[field])]
    if task.get("replace"):
        command.append("--replace")
    return command


@contextmanager
def drain_lock(queue_dir: Path, timeout: float = 60.0) -> Iterator[None]:
    if msvcrt is None and fcntl is None:
        raise RuntimeError("no OS file-locking primitive available on this platform")
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock_path = queue_dir / LOCK_FILE_NAME
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                if msvcrt is not None:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"another drain holds {lock_path}; refusing to double-record"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                if msvcrt is not None:
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)


def load_payload(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and payload:
        return payload
    raise ValidationError("payload must be one JSON object or a non-empty list of objects")


def enqueue_file(payload_path: Path, queue_dir: Path, priority: int) -> list[Path]:
    if not 0 <= priority <= 99:
        raise ValidationError(f"--priority must be within [0, 99], got {priority}")
    tasks = load_payload(payload_path)
    for task in tasks:
        validate_task(task)
    queue_dir.mkdir(parents=True, exist_ok=True)
    dropped: list[Path] = []
    for task in tasks:
        target = queue_dir / queue_file_name(priority)
        target.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
        dropped.append(target)
    return dropped


def run_drain(queue_dir: Path, execute: bool, lock_timeout: float) -> dict[str, int]:
    counts = {"queued": 0, "executed": 0, "recorded": 0, "failed": 0}
    if not execute:
        pending = queued_files(queue_dir)
        counts["queued"] = len(pending)
        for path in pending:
            print(f"[dry-run] {path.stem}")
            print(f"  {subprocess.list2cmdline(build_command_for(path))}")
        print(f"dry-run: {len(pending)} task(s) would execute; pass --execute to record")
        return counts

    with drain_lock(queue_dir, timeout=lock_timeout):
        pending = queued_files(queue_dir)
        counts["queued"] = len(pending)
        done_dir = queue_dir / DONE_DIR_NAME
        failed_dir = queue_dir / FAILED_DIR_NAME
        for path in pending:
            print(f"processing {path.stem}")
            counts["executed"] += 1
            try:
                task = json.loads(path.read_text(encoding="utf-8-sig"))
                validate_task(task)
            except (json.JSONDecodeError, ValidationError) as exc:
                failed_dir.mkdir(parents=True, exist_ok=True)
                _move(path, failed_dir / path.name)
                (failed_dir / (path.stem + ".stderr.txt")).write_text(
                    f"validation failed: {exc}\n", encoding="utf-8"
                )
                print(f"  FAILED validation: {exc}")
                counts["failed"] += 1
                continue
            result = subprocess.run(
                build_command(task),
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout.rstrip())
            if result.returncode == 0:
                done_dir.mkdir(parents=True, exist_ok=True)
                _move(path, done_dir / path.name)
                print("  recorded")
                counts["recorded"] += 1
            else:
                failed_dir.mkdir(parents=True, exist_ok=True)
                _move(path, failed_dir / path.name)
                (failed_dir / (path.stem + ".stderr.txt")).write_text(
                    result.stderr, encoding="utf-8"
                )
                print(
                    f"  FAILED (exit {result.returncode}); moved to {FAILED_DIR_NAME}/ "
                    "with captured stderr"
                )
                counts["failed"] += 1
    print("drain complete: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return counts


def build_command_for(path: Path) -> list[str]:
    try:
        task = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return ["<unreadable task file>"]
    if not isinstance(task, dict):
        return ["<malformed task file>"]
    return build_command(task)


def _move(source: Path, target: Path) -> None:
    if target.exists():
        target = target.with_name(target.stem + "-" + secrets.token_hex(4) + target.suffix)
    source.replace(target)


def _list_dir_json(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json")) if directory.exists() else []


def print_status(queue_dir: Path) -> None:
    queued = queued_files(queue_dir)
    done = _list_dir_json(queue_dir / DONE_DIR_NAME)
    failed = _list_dir_json(queue_dir / FAILED_DIR_NAME)
    print(f"queue dir: {queue_dir}")
    print(f"queued={len(queued)} done={len(done)} failed={len(failed)}")
    for label, entries in (("QUEUED", queued), ("DONE", done), ("FAILED", failed)):
        for entry in entries:
            print(f"  [{label}] {entry.stem}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="batch_record",
        description="Serialize weak-signals record calls through a file queue",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue_parser = subparsers.add_parser(
        "enqueue", help="validate a task file and drop it into the queue"
    )
    enqueue_parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="JSON file holding one task object or a list of task objects",
    )
    enqueue_parser.add_argument(
        "--priority",
        type=int,
        default=DEFAULT_PRIORITY,
        help="lower drains first (default %(default)s)",
    )
    enqueue_parser.add_argument("--queue-dir", type=Path, default=None)

    drain_parser = subparsers.add_parser(
        "drain", help="process every queued task oldest-first under the lock"
    )
    drain_parser.add_argument(
        "--execute",
        action="store_true",
        help="actually run the recorder CLI; default is a dry-run",
    )
    drain_parser.add_argument(
        "--lock-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for the drain lock (default %(default)s)",
    )
    drain_parser.add_argument("--queue-dir", type=Path, default=None)

    status_parser = subparsers.add_parser("status", help="queue counts and listings")
    status_parser.add_argument("--queue-dir", type=Path, default=None)

    args = parser.parse_args(argv)
    queue_dir = args.queue_dir if args.queue_dir is not None else default_queue_dir()

    if args.command == "enqueue":
        payload_path: Path = args.file
        try:
            dropped = enqueue_file(payload_path, queue_dir, args.priority)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            print(f"enqueue rejected: {exc}", file=sys.stderr)
            return 2
        for path in dropped:
            print(f"enqueued {path.stem}")
        return 0

    if args.command == "drain":
        try:
            run_drain(queue_dir, execute=args.execute, lock_timeout=args.lock_timeout)
        except TimeoutError as exc:
            print(f"drain aborted: {exc}", file=sys.stderr)
            return 3
        return 0

    print_status(queue_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
