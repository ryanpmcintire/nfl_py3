from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_script("bye_overvaluation_screen_test", "bye_overvaluation_screen.py")


def _fixture_frame() -> pd.DataFrame:
    season_2023 = pd.date_range("2023-09-10", periods=17, freq="7D")
    game_ids = [f"p23-{i}" for i in range(len(season_2023))]
    seasons = [2023] * len(season_2023)
    home_teams = ["ARI"] * len(season_2023)
    away_teams = ["BUF"] * len(season_2023)
    game_ids += ["g24-1", "g24-2", "g24-3", "g24-4"]
    seasons += [2024] * 4
    home_teams += ["ARI", "BUF", "BUF", "ARI"]
    away_teams += ["BUF", "ARI", "ARI", "BUF"]
    gamedays = (
        list(season_2023)
        + pd.to_datetime(
            [
                "2024-09-08",
                "2024-09-15",
                "2024-10-06",
                "2024-10-13",
            ]
        ).tolist()
    )
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "season": seasons,
            "home_team": home_teams,
            "away_team": away_teams,
            "gameday_dt": gamedays,
        }
    )


def test_season_opener_is_not_flagged_off_bye() -> None:
    df = _fixture_frame()

    home_pb, away_pb = screen.build_bye_maps(df)

    flags = dict(zip(df["game_id"], zip(home_pb, away_pb, strict=True), strict=True))
    assert not any(flags["g24-1"]), f"season opener misflagged off-bye: {flags['g24-1']}"
    assert not any(flags["p23-0"]), f"first-ever opener misflagged off-bye: {flags['p23-0']}"


def test_only_true_strict_bye_gap_is_flagged() -> None:
    df = _fixture_frame()

    home_pb, away_pb = screen.build_bye_maps(df)

    flags = dict(zip(df["game_id"], zip(home_pb, away_pb, strict=True), strict=True))
    for game_id in [f"p23-{i}" for i in range(17)] + ["g24-1", "g24-2"]:
        assert not any(flags[game_id]), f"{game_id} wrongly flagged: {flags[game_id]}"
    assert flags["g24-3"] == (True, True), f"21-day bye gap not flagged: {flags['g24-3']}"
    assert flags["g24-4"] == (False, False), f"weekly gap flagged: {flags['g24-4']}"


def test_build_bye_maps_requires_season_column() -> None:
    df = _fixture_frame().drop(columns=["season"])

    with pytest.raises(KeyError):
        screen.build_bye_maps(df)
