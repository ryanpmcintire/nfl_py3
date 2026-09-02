from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from nfl_ats.io import atomic_bytes, atomic_csv, atomic_json, atomic_parquet, atomic_text, run_id


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
