"""Small, atomic output helpers used by command-line workflows."""

from __future__ import annotations

import json
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


def atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def atomic_text(text: str, destination: Path) -> None:
    """Replace a UTF-8 text file only after its complete content is written."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


def atomic_bytes(payload: bytes, destination: Path) -> None:
    """Replace a binary file only after its complete payload is written."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)
