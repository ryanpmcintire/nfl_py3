from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id


def test_atomic_output_helpers(tmp_path) -> None:
    frame = pd.DataFrame({"value": [1, 2]})
    json_path = tmp_path / "nested" / "value.json"
    parquet_path = tmp_path / "nested" / "value.parquet"
    csv_path = tmp_path / "nested" / "value.csv"

    atomic_json({"ok": True}, json_path)
    atomic_parquet(frame, parquet_path)
    atomic_csv(frame, csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), frame)
    assert not list(tmp_path.rglob("*.tmp"))


def test_run_id_is_utc() -> None:
    timestamp = datetime(2022, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert run_id(timestamp) == "20220102T030405Z"
