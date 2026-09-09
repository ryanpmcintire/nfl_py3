from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.io import (
    atomic_bytes,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    atomic_text,
    json_default,
    run_id,
)


def test_atomic_output_helpers(tmp_path) -> None:
    frame = pd.DataFrame({"value": [1, 2]})
    json_path = tmp_path / "nested" / "value.json"
    parquet_path = tmp_path / "nested" / "value.parquet"
    csv_path = tmp_path / "nested" / "value.csv"
    text_path = tmp_path / "nested" / "value.md"
    binary_path = tmp_path / "nested" / "value.bin"

    atomic_json({"ok": True}, json_path)
    atomic_parquet(frame, parquet_path)
    atomic_csv(frame, csv_path)
    atomic_text("complete\n", text_path)
    binary_path.write_bytes(b"old")
    atomic_bytes(b"\x00\xff", binary_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), frame)
    assert text_path.read_text(encoding="utf-8") == "complete\n"
    assert binary_path.read_bytes() == b"\x00\xff"
    assert not list(tmp_path.rglob("*.tmp"))


def test_run_id_is_utc() -> None:
    timestamp = datetime(2022, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert run_id(timestamp) == "20220102T030405Z"


def test_json_default_serialises_summary_value_types(tmp_path) -> None:
    """ENG-35 (2026-09-04): the first --full-replay crashed on a pandas Timestamp."""

    import json
    from datetime import UTC, datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd

    payload = {
        "ts": pd.Timestamp("2026-09-08T16:00:00Z"),
        "dt": datetime(2026, 9, 8, 16, tzinfo=UTC),
        "n": np.int64(3),
        "f": np.float64(0.5),
        "p": Path("a") / "b",
        "s": {"y", "x"},
    }
    text = json.dumps(payload, default=json_default, sort_keys=True)
    decoded = json.loads(text)
    assert decoded["ts"].startswith("2026-09-08T16:00:00")
    assert decoded["n"] == 3 and decoded["f"] == 0.5
    assert decoded["s"] == ["x", "y"]
    destination = tmp_path / "out.json"
    atomic_json(payload, destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["dt"].startswith("2026-09-08")
    with pytest.raises(TypeError):
        json.dumps({"o": object()}, default=json_default)


def test_atomic_json_survives_concurrent_writers(tmp_path: Path) -> None:
    import threading

    from nfl_ats.io import atomic_json

    destination = tmp_path / "registry.json"
    errors: list[BaseException] = []

    def writer(index: int) -> None:
        try:
            for _ in range(25):
                atomic_json({"writer": index, "rows": list(range(4000 + index * 500))}, destination)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["writer"] in range(4)
    assert len(payload["rows"]) == 4000 + payload["writer"] * 500
    assert not list(tmp_path.glob("*.tmp"))


def test_file_lock_is_exclusive_and_breaks_stale_locks(tmp_path: Path) -> None:
    import os
    import time

    from nfl_ats import io as io_module
    from nfl_ats.io import file_lock

    target = tmp_path / "registry.json"
    lock_path = tmp_path / "registry.json.lock"
    with file_lock(target):
        assert lock_path.exists()
        with pytest.raises(TimeoutError), file_lock(target, timeout=0.2, poll=0.02):
            pass
    assert not lock_path.exists()

    lock_path.write_text("", encoding="utf-8")
    stale = time.time() - io_module.STALE_LOCK_SECONDS - 5
    os.utime(lock_path, (stale, stale))
    with file_lock(target, timeout=1.0, poll=0.02):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_weak_signal_record_command_serialises_under_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two record commands started together both land: no lost update."""

    import threading
    from types import SimpleNamespace

    from nfl_ats.cli_commands import registry as commands
    from nfl_ats.weak_signals import load_registry

    path = tmp_path / "weak_signals.json"
    monkeypatch.setattr(commands, "weak_signal_registry_path", lambda: path)
    monkeypatch.setattr(commands, "_print_json", lambda payload: None)

    def args(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            recorded_at="2026-09-08",
            description="lock test",
            source="tests",
            effect=0.1,
            effect_units="accuracy_points",
            classification="unresolved_below_power",
            league="nfl",
            season_start=2020,
            season_end=2025,
            standard_error=None,
            interval_low=-0.5,
            interval_high=0.7,
            probability_positive=0.6,
            sample_games=100,
            sample_blocks=10,
            reliability=None,
            family="lock_test_family",
            classification_evidence="test",
            closing_ground=None,
            notes="",
            plain_summary="Lock test row.",
            category="modeling",
            replace=False,
        )

    errors: list[BaseException] = []

    def run(index: int) -> None:
        try:
            commands._cmd_weak_signals_record(args(f"lock_test_{index}"))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    registry = load_registry(path)
    assert sorted(registry.signals) == [f"lock_test_{i}" for i in range(6)]
    assert not (tmp_path / "weak_signals.json.lock").exists()
