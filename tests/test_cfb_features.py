from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import (
    CFB_GAME_NO_LINES,
    CFB_GAME_NO_PBP,
    CFB_GAME_REPAIRED,
    CFB_GAME_UNRESOLVED,
    cfb_true_spread,
)

from nfl_ats.cfb_features import (
    CFB_MODEL_FEATURE_COLUMNS,
    CFB_STATE_METRICS,
    CFB_TEAM_STATE_FEATURES,
    attach_cfb_team_states,
    build_cfb_game_features,
    build_cfb_team_states,
    cfb_season_partitions,
    load_cfb_seasons,
)


def test_canonical_table_contracts_and_logged_exclusions(
    cfb_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, lines, pbp = cfb_inputs
    features, audit = build_cfb_game_features(
        schedules, lines, pbp, start_season=2013, end_season=2014
    )

    assert audit["excluded_postseason_rows"] == 1
    assert audit["excluded_incomplete_rows"] == 1
    assert audit["excluded_non_fbs_rows"] == 1
    assert audit["spread_games_excluded_unresolved"] == 1
    assert audit["excluded_games_without_spread"] == 2
    assert audit["excluded_games_without_pbp"] == 1
    assert audit["canonical_games"] == 109
    assert audit["canonical_games"] == len(features)
    assert set(CFB_MODEL_FEATURE_COLUMNS).issubset(features.columns)

    excluded = {CFB_GAME_UNRESOLVED, CFB_GAME_NO_LINES, CFB_GAME_NO_PBP}
    assert excluded.isdisjoint(set(features["game_id"]))
    assert CFB_GAME_REPAIRED in set(features["game_id"])

    game = features.loc[features["game_id"].eq(20130101)].iloc[0]
    expected_spread = cfb_true_spread(int(game["home_id"]), int(game["away_id"]))
    assert game["spread_line"] == pytest.approx(expected_spread)
    assert game["spread_book_count"] == 2
    assert game["spread_dispersion"] == pytest.approx(0.5)
    assert game["spread_open"] == pytest.approx(expected_spread + 1.5)
    assert game["spread_open_book_count"] == 1
    assert game["home_spread_odds"] == pytest.approx(-105.0)
    assert game["away_spread_odds"] == pytest.approx(-115.0)
    assert game["total_line"] == pytest.approx(50.0)
    assert game["total_book_count"] == 2
    assert game["source_regime"] == "sbr_multibook"

    repaired = features.loc[features["game_id"].eq(CFB_GAME_REPAIRED)].iloc[0]
    assert repaired["spread_line"] == pytest.approx(
        cfb_true_spread(int(repaired["home_id"]), int(repaired["away_id"]))
    )


def test_ats_semantics_match_the_nfl_convention(cfb_features_frame: pd.DataFrame) -> None:
    frame = cfb_features_frame
    np.testing.assert_allclose(
        frame["result"], frame["home_points"] - frame["away_points"], atol=1e-12
    )
    np.testing.assert_allclose(
        frame["ats_margin"], frame["result"] - frame["spread_line"], atol=1e-12
    )
    covered = frame["ats_margin"].gt(0)
    lost = frame["ats_margin"].lt(0)
    assert frame.loc[covered, "home_cover"].eq(1.0).all()
    assert frame.loc[lost, "home_cover"].eq(0.0).all()
    assert frame.loc[~covered & ~lost, "home_cover"].isna().all()
    # Home-oriented sign convention: positive spread_line means home favored,
    # and stronger fixture homes carry positive spreads.
    strong_home = frame.loc[frame["home_id"].lt(frame["away_id"])]
    assert strong_home["spread_line"].gt(0).all()


def test_state_maturity_rule_and_ewm_recursion() -> None:
    rows = []
    for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
        row = {
            "game_id": 100 + index,
            "season": 2013,
            "gameday": pd.Timestamp("2013-09-01") + pd.Timedelta(days=7 * index),
            "team_id": 10,
            **dict.fromkeys(CFB_STATE_METRICS, value),
        }
        rows.append(row)
    states = build_cfb_team_states(pd.DataFrame(rows))
    column = "state_off_epa_per_play"
    alpha = 2.0 / 9.0
    second = alpha * 2.0 + (1 - alpha) * 1.0
    third = alpha * 3.0 + (1 - alpha) * second
    fourth = alpha * 4.0 + (1 - alpha) * third
    assert states[column].iloc[0:2].isna().all()
    assert states[column].iloc[2] == pytest.approx(third)
    assert states[column].iloc[3] == pytest.approx(fourth)
    assert states["team_games"].tolist() == [1, 2, 3, 4]

    with pytest.raises(ValueError, match="span"):
        build_cfb_team_states(pd.DataFrame(rows), span=1)
    with pytest.raises(ValueError, match="min_periods"):
        build_cfb_team_states(pd.DataFrame(rows), min_periods=0)
    with pytest.raises(ValueError, match="offseason_retention"):
        build_cfb_team_states(pd.DataFrame(rows), offseason_retention=1.5)


def test_offseason_regression_toward_league_mean() -> None:
    rows = []
    for team_id, level in ((10, 4.0), (11, 0.0)):
        for index in range(4):
            rows.append(
                {
                    "game_id": team_id * 100 + index,
                    "season": 2013,
                    "gameday": pd.Timestamp("2013-09-01") + pd.Timedelta(days=7 * index),
                    "team_id": team_id,
                    **dict.fromkeys(CFB_STATE_METRICS, level),
                }
            )
    states = build_cfb_team_states(pd.DataFrame(rows))
    games = pd.DataFrame(
        {
            "game_id": [900],
            "season": [2014],
            "gameday": [pd.Timestamp("2014-09-01")],
            "home_id": [10],
            "away_id": [11],
        }
    )
    attached = attach_cfb_team_states(games, states)
    league_mean = 2.0
    expected = league_mean + 0.67 * (4.0 - league_mean)
    assert attached.loc[0, "home_off_epa_per_play"] == pytest.approx(expected)
    assert attached.loc[0, "away_off_epa_per_play"] == pytest.approx(
        league_mean + 0.67 * (0.0 - league_mean)
    )
    assert attached.loc[0, "home_team_games"] == 0
    assert attached.loc[0, "diff_off_epa_per_play"] == pytest.approx(
        attached.loc[0, "home_off_epa_per_play"] - attached.loc[0, "away_off_epa_per_play"]
    )


def test_current_game_cannot_change_current_pregame_features(
    cfb_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, lines, pbp = cfb_inputs
    baseline, _ = build_cfb_game_features(schedules, lines, pbp, start_season=2013, end_season=2014)

    target_game = 20140701  # mid-season 2014, both teams have later games
    changed_pbp = pbp.copy()
    changed_pbp.loc[changed_pbp["game_id"].eq(target_game), "EPA"] = 1_000.0
    changed_schedules = schedules.copy()
    changed_schedules.loc[changed_schedules["game_id"].eq(target_game), "home_points"] = 99
    changed, _ = build_cfb_game_features(
        changed_schedules, lines, changed_pbp, start_season=2013, end_season=2014
    )

    baseline_row = baseline.loc[baseline["game_id"].eq(target_game)].iloc[0]
    changed_row = changed.loc[changed["game_id"].eq(target_game)].iloc[0]
    for column in CFB_MODEL_FEATURE_COLUMNS:
        base_value = baseline_row[column]
        new_value = changed_row[column]
        assert (pd.isna(base_value) and pd.isna(new_value)) or base_value == pytest.approx(
            new_value
        ), f"pregame feature {column} changed with the current game's data"

    home_team = int(baseline_row["home_id"])
    later = baseline.loc[
        (baseline["gameday"] > baseline_row["gameday"])
        & (baseline["home_id"].eq(home_team) | baseline["away_id"].eq(home_team))
    ]
    assert not later.empty
    next_game = int(later.iloc[0]["game_id"])
    later_baseline = baseline.loc[baseline["game_id"].eq(next_game)].iloc[0]
    later_changed = changed.loc[changed["game_id"].eq(next_game)].iloc[0]
    differences = [
        column
        for column in CFB_TEAM_STATE_FEATURES
        if pd.notna(later_baseline[column])
        and later_baseline[column] != pytest.approx(later_changed[column])
    ]
    assert differences, "earlier-game information should flow into later pregame states"


def test_rest_and_context_controls(cfb_features_frame: pd.DataFrame) -> None:
    frame = cfb_features_frame
    early = frame.loc[frame["game_id"].eq(20140801)].iloc[0]
    affected_teams = {int(early["home_id"]), int(early["away_id"])}
    week9 = frame.loc[frame["season"].eq(2014) & frame["week"].eq(9)]
    follow_ups = week9.loc[
        week9["home_id"].isin(affected_teams) ^ week9["away_id"].isin(affected_teams)
    ]
    assert not follow_ups.empty
    assert follow_ups["rest_diff"].abs().eq(1.0).all()
    normal = frame.loc[frame["season"].eq(2014) & frame["week"].eq(6)]
    assert normal["rest_diff"].eq(0.0).all()

    sec_home = frame.loc[frame["home_id"].le(4)]
    assert sec_home["home_power5"].eq(1).all()
    mac_home = frame.loc[frame["home_id"].ge(5)]
    assert mac_home["home_power5"].eq(0).all()
    assert frame["week_sin"].between(-1, 1).all()
    assert frame["conference_game"].isin([0, 1]).all()


def test_cfb_snapshot_loader_prefers_newest_snapshot_per_season(tmp_path: Path) -> None:
    root = tmp_path / "cfb"
    older = root / "pbp" / "raw" / "20260101T000000Z"
    newer = root / "pbp" / "raw" / "20260201T000000Z"
    for directory, seasons, marker in ((older, [2013, 2014], "old"), (newer, [2014], "new")):
        for season in seasons:
            partition = directory / f"season={season}" / "plays.parquet"
            partition.parent.mkdir(parents=True)
            pd.DataFrame({"game_id": [1], "source": [marker]}).to_parquet(partition, index=False)
        (directory / "manifest.json").write_text(json.dumps({"seasons": seasons}), encoding="utf-8")

    partitions = cfb_season_partitions(root, "pbp")
    assert "20260101T000000Z" in str(partitions[2013])
    assert "20260201T000000Z" in str(partitions[2014])
    loaded = load_cfb_seasons(root, "pbp", [2013, 2014])
    assert sorted(loaded["source"]) == ["new", "old"]
    with pytest.raises(FileNotFoundError, match="2015"):
        load_cfb_seasons(root, "pbp", [2015])
    with pytest.raises(FileNotFoundError, match="No CFB"):
        cfb_season_partitions(tmp_path / "empty", "pbp")
