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

    token = f"{os.getpid()}-{secrets.token_hex(4)}"
    return destination.with_name(f"{destination.name}.{token}.tmp")


def _replace(temporary: Path, destination: Path, *, attempts: int = 40) -> None:

    for attempt in range(attempts):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


STALE_LOCK_SECONDS = 180.0


@contextmanager
def file_lock(path: Path, *, timeout: float = 60.0, poll: float = 0.05) -> Iterator[None]:

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

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    temporary.write_text(text, encoding="utf-8")
    _replace(temporary, destination)


def atomic_bytes(payload: bytes, destination: Path) -> None:

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
