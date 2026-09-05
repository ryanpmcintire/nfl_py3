from __future__ import annotations

import json
from datetime import UTC, datetime

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
