"""Small, atomic output helpers used by command-line workflows."""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def run_id(now: datetime | None = None) -> str:
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def json_default(obj: Any) -> Any:
    """``json.dumps(default=...)`` hook for the value types our summaries carry.

    Added 2026-09-04 (ENG-35) after the first ``lockday_rehearsal.py
    --full-replay`` crashed writing its report: the weekly-run summary now
    carries ``pandas.Timestamp`` instants, and the decision package embeds that
    summary verbatim. Serialising them is strictly widening -- every input that
    used to succeed is unchanged -- and it keeps the lock-day package writer
    from failing on the real lock.
    """

    if isinstance(obj, pd.Timestamp | datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set | frozenset):
        return sorted(obj, key=str)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item") and not isinstance(obj, str | bytes):
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _temporary_path(destination: Path) -> Path:
    """A temp name unique to THIS writer, in the destination's directory.

    2026-09-08: two research lanes recording into ``registry/weak_signals.json``
    at the same moment both wrote ``weak_signals.json.tmp``; one handle's
    bytes landed over the other's and the surviving file was one valid
    document followed by the tail of a longer one. A per-process, per-call
    name keeps every writer's bytes its own; ``replace`` stays atomic.
    """

    token = f"{os.getpid()}-{secrets.token_hex(4)}"
    return destination.with_name(f"{destination.name}.{token}.tmp")


def _replace(temporary: Path, destination: Path, *, attempts: int = 40) -> None:
    """``os.replace`` with a short retry: on Windows a concurrent writer's own
    replace of the same destination surfaces as a transient PermissionError."""

    for attempt in range(attempts):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


#: A lock file older than this is treated as abandoned (a crashed writer).
STALE_LOCK_SECONDS = 180.0


@contextmanager
def file_lock(path: Path, *, timeout: float = 60.0, poll: float = 0.05) -> Iterator[None]:
    """Serialise read-modify-write of one file across processes.

    Exclusive creation of ``<path>.lock`` (``O_CREAT | O_EXCL``, atomic on
    POSIX and Windows); waits up to ``timeout`` seconds, breaking a lock older
    than ``STALE_LOCK_SECONDS``. The registry CLI commands hold this around
    load -> modify -> save so concurrent lanes cannot lose each other's rows
    (an atomic write alone only prevents corruption, not lost updates).
    """

    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > STALE_LOCK_SECONDS:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Could not acquire {lock_path} within {timeout:.0f}s; another writer holds it"
                ) from None
            time.sleep(poll)
            continue
        os.close(fd)
        break
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    _replace(temporary, destination)


def atomic_text(text: str, destination: Path) -> None:
    """Replace a UTF-8 text file only after its complete content is written."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    temporary.write_text(text, encoding="utf-8")
    _replace(temporary, destination)


def atomic_bytes(payload: bytes, destination: Path) -> None:
    """Replace a binary file only after its complete payload is written."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    temporary.write_bytes(payload)
    _replace(temporary, destination)


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    _replace(temporary, destination)


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    _replace(temporary, destination)
