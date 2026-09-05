"""Raw depth snapshots retain upstream columns and refuse overwrite."""

import json
import runpy
from pathlib import Path

import nflreadpy
import pandas as pd
import pytest

from nfl_ats.provenance import sha256_file


def test_depth_ingest_preserves_raw_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/nflverse_depth_charts_ingest.py")
    )["run_ingest"]
    frame = pd.DataFrame(
        {"dt": ["2025-09-01", "invalid"], "team": ["KC", "DEN"], "upstream_extra": [17, 23]}
    )
    calls = []

    def fetch(*, seasons: list[int]) -> pd.DataFrame:
        calls.append(seasons)
        return frame

    monkeypatch.setattr(nflreadpy, "load_depth_charts", fetch)
    destination = tmp_path / "snapshot"
    result = ingest(destination)
    pd.testing.assert_frame_equal(pd.read_parquet(destination / "depth_charts.parquet"), frame)
    assert result == json.loads((destination / "manifest.json").read_text())
    assert result["rows"] == 2 and result["usable_dt_rows"] == 1
    assert result["requested_seasons"] == [2025]
    assert result["output_parquet_sha256"] == sha256_file(destination / "depth_charts.parquet")
    assert result["source_url"] == (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        "depth_charts/depth_charts_2025.parquet"
    )
    with pytest.raises(FileExistsError):
        ingest(destination)
    assert calls == [[2025]]


def test_depth_ingest_does_not_publish_unusable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/nflverse_depth_charts_ingest.py")
    )["run_ingest"]
    monkeypatch.setattr(
        nflreadpy,
        "load_depth_charts",
        lambda **kwargs: pd.DataFrame({"dt": [None], "team": ["KC"]}),
    )
    destination = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="no usable"):
        ingest(destination)
    assert not destination.exists()
