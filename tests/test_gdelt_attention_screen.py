from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_script("gdelt_attention_screen_test", "gdelt_attention_screen.py")
builder = _load_script("build_gdelt_weekly_features_test", "build_gdelt_weekly_features.py")


def _write_weekly_fixture(path: Path) -> pd.DataFrame:
    weekly = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "season": [2024] * 4,
            "week": [1, 1, 2, 2],
            "team": ["ARI", "BUF", "ARI", "BUF"],
            "is_home": [True, False, True, False],
            "tuesday_z": [-0.75, -0.60, 3.0, -0.80],
            "tuesday_has_baseline": [True, True, False, True],
            # Extreme Saturday values are a canary: the Tuesday loader must ignore them.
            "saturday_z": [99.0, 99.0, 99.0, 99.0],
            "saturday_has_baseline": [True, True, True, True],
        }
    )
    weekly.to_parquet(path, index=False)
    path.with_suffix(".manifest.json").write_text(
        json.dumps({"n_teams_with_volume": 2, "n_teams_total": 32}), encoding="utf-8"
    )
    return weekly


def test_processed_replication_loader_uses_only_tuesday_columns(tmp_path: Path) -> None:
    path = tmp_path / "gdelt_weekly.parquet"
    _write_weekly_fixture(path)
    games = pd.DataFrame({"game_id": ["g1", "g2"]})

    long_df, meta = screen.load_gdelt_weekly_long(path, games)

    assert long_df["attention_z"].tolist()[:2] == [-0.75, -0.60]
    assert pd.isna(long_df.loc[long_df["game_id"] == "g2", "attention_z"]).any()
    assert 99.0 not in long_df["attention_z"].dropna().tolist()
    assert meta["n_teams_covered"] == 2
    assert meta["n_teams_total"] == 32


def test_processed_replication_loader_rejects_duplicate_game_sides(tmp_path: Path) -> None:
    path = tmp_path / "gdelt_weekly.parquet"
    weekly = _write_weekly_fixture(path)
    pd.concat([weekly, weekly.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="duplicate game_id/is_home"):
        screen.load_gdelt_weekly_long(path, pd.DataFrame({"game_id": ["g1", "g2"]}))


def test_tuesday_features_ignore_news_after_tuesday_cutoff() -> None:
    gamedays = pd.to_datetime(["2024-09-08", "2024-09-15", "2024-09-22", "2024-09-29"])
    tuesday_ends = gamedays - pd.to_timedelta((gamedays.weekday - 1) % 7, unit="D")
    grid = pd.DataFrame(
        {
            "season": 2024,
            "week": [1, 2, 3, 4],
            "team": "ARI",
            "game_id": ["g1", "g2", "g3", "g4"],
            "is_home": True,
            "gameday": gamedays,
            "gameday_weekday_name": "Sunday",
            "saturday_cutoff_safe": True,
            "tuesday_window_end": tuesday_ends,
            "tuesday_window_start": tuesday_ends - pd.Timedelta(days=6),
            "saturday_window_end": tuesday_ends + pd.Timedelta(days=4),
            "saturday_window_start": tuesday_ends - pd.Timedelta(days=2),
        }
    )
    dates = pd.date_range("2024-08-28", "2024-09-28", freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "raw_count": np.arange(1, len(dates) + 1, dtype=float),
            "monitored_total": 1000.0,
            "avg_tone": np.nan,
        }
    )
    changed = daily.copy()
    changed.loc[changed["date"] == pd.Timestamp("2024-09-25"), "raw_count"] += 1000.0

    before = builder.build_weekly_table(grid, {"ARI": daily})
    after = builder.build_weekly_table(grid, {"ARI": changed})
    final_before = before.loc[before["game_id"] == "g4"].iloc[0]
    final_after = after.loc[after["game_id"] == "g4"].iloc[0]

    assert final_before["tuesday_raw_count"] == final_after["tuesday_raw_count"]
    assert final_before["tuesday_z"] == final_after["tuesday_z"]
    assert final_before["saturday_raw_count"] != final_after["saturday_raw_count"]
