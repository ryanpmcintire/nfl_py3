from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sagarin_divergence_battery import (
    load_sagarin_ratings,
    sagarin_source_provenance,
)


def test_sagarin_loader_physically_projects_only_frozen_pregame_columns(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}
    safe = pd.DataFrame(
        {
            "season": [2020],
            "week": [1],
            "team_code": ["BUF"],
            "rating": [25.0],
            "home_edge_rating": [2.5],
            "has_tuesday_snapshot": [True],
        }
    )

    def fake_read_parquet(path: Path, *, columns: list[str]) -> pd.DataFrame:
        observed["path"] = path
        observed["columns"] = columns
        return safe.loc[:, columns].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    load_sagarin_ratings(Path("snapshot"))

    assert observed["columns"] == [
        "season",
        "week",
        "team_code",
        "rating",
        "home_edge_rating",
        "has_tuesday_snapshot",
    ]
    assert "result" not in observed["columns"]
    assert "home_cover" not in observed["columns"]


def test_sagarin_source_provenance_hashes_every_consolidated_input(tmp_path: Path) -> None:
    manifest = {
        "fetched_at_utc": "2026-08-20T16:49:26Z",
        "captures_attempted": 592,
        "captures_fetch_ok": 585,
        "captures_fetch_failed": 7,
        "captures_parse_ok": 585,
        "index_rows": 18473,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("asof_tuesday_view.parquet", "index.parquet", "captures_log.parquet"):
        (tmp_path / name).write_bytes(name.encode())

    provenance = sagarin_source_provenance(tmp_path)

    assert provenance["snapshot_id"] == tmp_path.name
    assert provenance["captures_fetch_ok"] == 585
    assert provenance["captures_fetch_failed"] == 7
    assert set(provenance["sha256"]) == {
        "manifest",
        "asof_tuesday_view",
        "index",
        "captures_log",
    }
    assert all(len(value) == 64 for value in provenance["sha256"].values())
