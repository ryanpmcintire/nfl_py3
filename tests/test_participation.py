from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.participation import (
    PARTICIPATION_SOURCE_COLUMNS,
    build_participation_play_table,
    build_season_lagged_player_ratings,
    canonicalize_participation,
    canonicalize_participation_ratings,
    fetch_participation_snapshot,
    latest_participation_snapshot,
    load_participation_snapshot,
    participation_snapshot_from_root,
    write_participation_snapshot,
)
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS


def _lineup(prefix: str) -> str:
    return ";".join(f"{prefix}{index:02d}" for index in range(11))


def _participation(seasons: tuple[int, ...] = (2020, 2021, 2022)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in seasons:
        game_id = f"{season}_01_B_A"
        for play_id in range(1, 9):
            offense = "A" if play_id <= 4 else "B"
            defense = "B" if offense == "A" else "A"
            row = dict.fromkeys(PARTICIPATION_SOURCE_COLUMNS, pd.NA)
            row.update(
                {
                    "nflverse_game_id": game_id,
                    "old_game_id": f"old-{season}",
                    "play_id": play_id,
                    "possession_team": offense,
                    "offense_formation": "SHOTGUN",
                    "offense_personnel": "1 RB, 1 TE, 3 WR",
                    "defense_personnel": "4 DL, 2 LB, 5 DB",
                    "players_on_play": f"{_lineup(offense + 'O')};{_lineup(defense + 'D')}",
                    "offense_players": _lineup(offense + "O"),
                    "defense_players": _lineup(defense + "D"),
                    "n_offense": 11,
                    "n_defense": 11,
                    "was_pressure": 0,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _pbp(seasons: tuple[int, ...] = (2020, 2021, 2022)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in seasons:
        game_id = f"{season}_01_B_A"
        for play_id in range(1, 9):
            offense = "A" if play_id <= 4 else "B"
            defense = "B" if offense == "A" else "A"
            row = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
            row.update(
                {
                    "game_id": game_id,
                    "play_id": play_id,
                    "season": season,
                    "season_type": "REG",
                    "week": 1,
                    "home_team": "A",
                    "away_team": "B",
                    "posteam": offense,
                    "defteam": defense,
                    "fixed_drive": 1 if offense == "A" else 2,
                    "down": 1,
                    "play_type": "pass",
                    "yards_gained": 8,
                    "pass_attempt": 1,
                    "rush_attempt": 0,
                    "sack": 0,
                    "qb_hit": 0,
                    "epa": (0.4 if offense == "A" else -0.2) + play_id / 100,
                    "success": int(offense == "A"),
                    "wp": 0.5,
                    "qb_kneel": 0,
                    "qb_spike": 0,
                    "aborted_play": 0,
                    "play": 1,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_participation_snapshot_round_trip_and_contract(tmp_path: Path) -> None:
    source = _participation((2020, 2021))
    snapshot = write_participation_snapshot(
        {
            season: source.loc[source["nflverse_game_id"].str.startswith(str(season))]
            for season in (2020, 2021)
        },
        tmp_path,
        "fixed",
    )
    loaded = load_participation_snapshot(snapshot)
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert len(loaded) == 16
    assert manifest["rows"] == 16
    assert all(partition["sha256"] for partition in manifest["partitions"])
    assert latest_participation_snapshot(tmp_path) == snapshot
    assert participation_snapshot_from_root(snapshot.root) == snapshot

    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate"):
        canonicalize_participation(duplicate)
    with pytest.raises(FileExistsError):
        write_participation_snapshot({2020: source.iloc[:8]}, tmp_path, "fixed")


@pytest.mark.full  # ENG-11: dominates --durations (multi-season partitioned fetch)
def test_participation_fetch_is_season_partitioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nflreadpy

    monkeypatch.setattr(
        nflreadpy,
        "load_participation",
        lambda seasons: _participation(tuple(seasons)),
    )
    snapshot = fetch_participation_snapshot([2020, 2021], tmp_path)
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert snapshot.seasons == (2020, 2021)
    assert manifest["rows"] == 16
    assert snapshot.season_path(2020).is_file()
    with pytest.raises(ValueError, match="non-empty"):
        fetch_participation_snapshot([], tmp_path)


def test_participation_play_table_requires_competitive_valid_lineups() -> None:
    participation = _participation((2020,))
    participation.loc[0, "n_offense"] = 10
    pbp = _pbp((2020,))
    pbp.loc[pbp["play_id"].eq(2), "wp"] = 0.99
    table = build_participation_play_table(participation, pbp)
    assert len(table) == 6
    assert table["offense_players"].str.split(";").map(len).eq(11).all()
    assert table["defense_players"].str.split(";").map(len).eq(11).all()


def test_target_season_ratings_ignore_target_season_outcomes() -> None:
    participation = _participation()
    pbp = _pbp()
    baseline = build_season_lagged_player_ratings(
        participation,
        pbp,
        target_seasons=(2022, 2023),
        lookback_seasons=3,
        ridge_alpha=1.0,
        team_feature_scale=1.0,
        reliability_prior_plays=0.0,
    )

    changed_participation = participation.copy()
    target_rows = changed_participation["nflverse_game_id"].str.startswith("2022_")
    changed_participation.loc[target_rows, "offense_players"] = changed_participation.loc[
        target_rows, "offense_players"
    ].str.replace("AO00", "NEW", regex=False)
    changed_pbp = pbp.copy()
    changed_pbp.loc[changed_pbp["season"].eq(2022), "epa"] = 4.5
    changed = build_season_lagged_player_ratings(
        changed_participation,
        changed_pbp,
        target_seasons=(2022, 2023),
        lookback_seasons=3,
        ridge_alpha=1.0,
        team_feature_scale=1.0,
        reliability_prior_plays=0.0,
    )

    baseline_target = baseline.loc[baseline["target_season"].eq(2022)].reset_index(drop=True)
    changed_target = changed.loc[changed["target_season"].eq(2022)].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_target, changed_target)
    assert "NEW" not in set(baseline["player_id"])
    assert "NEW" in set(changed.loc[changed["target_season"].eq(2023), "player_id"])
    assert baseline.groupby("target_season")["source_end_season"].first().to_dict() == {
        2022: 2021,
        2023: 2022,
    }


def test_participation_rating_contract_rejects_leakage_and_bad_parameters(
    tmp_path: Path,
) -> None:
    ratings = build_season_lagged_player_ratings(
        _participation((2020,)),
        _pbp((2020,)),
        target_seasons=(2021,),
        ridge_alpha=1.0,
        reliability_prior_plays=0.0,
    )
    leaked = ratings.copy()
    leaked["source_end_season"] = leaked["target_season"]
    with pytest.raises(DataContractError, match="end before"):
        canonicalize_participation_ratings(leaked)
    with pytest.raises(ValueError, match="ridge_alpha"):
        build_season_lagged_player_ratings(_participation((2020,)), _pbp((2020,)), ridge_alpha=0)
    with pytest.raises(FileNotFoundError, match="No participation snapshots"):
        latest_participation_snapshot(tmp_path)
