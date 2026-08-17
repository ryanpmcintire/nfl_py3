from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import (
    DRIVE_STATE_METRICS,
    PBP_OPPONENT_ADJUSTED_FEATURE_COLUMNS,
    PBP_STATE_METRICS,
)
from nfl_ats.data import DataContractError
from nfl_ats.pbp import (
    PBP_FILTER_VERSION,
    PBP_SNAPSHOT_COLUMNS,
    analysis_plays,
    build_drive_table,
    build_pbp_team_game_metrics,
    canonicalize_pbp,
    enrich_with_pbp_features,
    latest_pbp_snapshot,
    load_pbp_snapshot,
    snapshot_from_root,
    write_pbp_snapshot,
)


def _pbp_frame(games: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_number in range(1, games + 1):
        game_id = f"2022_{game_number:02d}_B_A"
        for team, opponent, direction in (("A", "B", 1.0), ("B", "A", -1.0)):
            for play_number in range(1, 4):
                row = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
                row.update(
                    {
                        "play_id": float(play_number + (0 if team == "A" else 100)),
                        "game_id": game_id,
                        "season": 2022,
                        "season_type": "REG",
                        "week": game_number,
                        "home_team": "A",
                        "away_team": "B",
                        "posteam": team,
                        "defteam": opponent,
                        "fixed_drive": 1 if team == "A" else 2,
                        "qtr": 1,
                        "down": 1 if play_number < 3 else 2,
                        "ydstogo": 10,
                        "yardline_100": 75 - play_number * 5,
                        "game_seconds_remaining": 3600 - play_number * 30,
                        "play_type": "pass" if play_number < 3 else "run",
                        "yards_gained": 8 + game_number * direction,
                        "qb_dropback": 1 if play_number < 3 else 0,
                        "qb_kneel": 0,
                        "qb_spike": 0,
                        "aborted_play": 0,
                        "pass_attempt": 1 if play_number < 3 else 0,
                        "rush_attempt": 0 if play_number < 3 else 1,
                        "complete_pass": 1 if play_number < 3 else 0,
                        "interception": 0,
                        "fumble_lost": 0,
                        "sack": 0,
                        "qb_hit": int(team == "B" and play_number == 2),
                        "touchdown": 0,
                        "first_down": int(play_number == 2),
                        "epa": direction * game_number + play_number / 10,
                        "success": int(direction > 0),
                        "wp": 0.5,
                        "passer_player_id": f"QB-{team}",
                        "passer_player_name": f"QB {team}",
                        "cpoe": direction * 2,
                        "xpass": 0.6,
                        "pass_oe": 0.1 * direction,
                        "score_differential": 0,
                        "penalty": 0,
                        "penalty_yards": 0,
                        "fixed_drive_result": "Punt",
                        "posteam_score": 0,
                        "posteam_score_post": 0,
                        "play": 1,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"2022_{week:02d}_B_A" for week in range(1, 4)],
            "season": 2022,
            "week": range(1, 4),
            "gameday": pd.date_range("2022-09-11", periods=3, freq="7D"),
            "away_team": "B",
            "home_team": "A",
        }
    )


def _recorded_scope(manifest_path: Path) -> bool:
    return bool(json.loads(manifest_path.read_text(encoding="utf-8"))["include_postseason"])


def _postseason_plays() -> pd.DataFrame:
    """A wild-card game in the same season, spelled POST as nflverse PBP does."""

    frame = _pbp_frame(1)
    frame["season_type"] = "POST"
    frame["week"] = 19
    frame["game_id"] = "2022_19_B_A"
    return frame


def test_canonicalize_pbp_keeps_regular_season_by_default() -> None:
    regular = _pbp_frame(2)
    mixed = pd.concat([regular, _postseason_plays()], ignore_index=True)
    expected = canonicalize_pbp(regular, season=2022)
    pd.testing.assert_frame_equal(canonicalize_pbp(mixed, season=2022), expected)

    widened = canonicalize_pbp(mixed, season=2022, include_postseason=True)
    assert set(widened["season_type"]) == {"REG", "POST"}
    assert len(widened) == len(mixed)
    pd.testing.assert_frame_equal(
        widened.loc[widened["season_type"].eq("REG")].reset_index(drop=True), expected
    )


def test_canonicalize_pbp_rejects_unknown_season_codes_when_widened() -> None:
    frame = _pbp_frame(1)
    frame.loc[0, "season_type"] = "REGULAR"
    with pytest.raises(DataContractError, match="unrecognized season codes"):
        canonicalize_pbp(frame, season=2022, include_postseason=True)
    # The default path keeps its historical, silent "== REG" comparison so the
    # frozen regular-season contract cannot start failing on new source values.
    assert len(canonicalize_pbp(frame, season=2022)) == len(frame) - 1


def test_postseason_snapshot_reads_back_as_regular_season_only(tmp_path: Path) -> None:
    regular = _pbp_frame(3)
    mixed = pd.concat([regular, _postseason_plays()], ignore_index=True)
    reg_only = write_pbp_snapshot({2022: regular}, tmp_path / "reg", "fixed")
    with_post = write_pbp_snapshot(
        {2022: mixed}, tmp_path / "post", "fixed", include_postseason=True
    )
    assert _recorded_scope(reg_only.manifest_path) is False
    assert _recorded_scope(with_post.manifest_path) is True

    # The invariant: what the feature builds see is identical either way.
    pd.testing.assert_frame_equal(load_pbp_snapshot(with_post), load_pbp_snapshot(reg_only))
    widened = load_pbp_snapshot(with_post, include_postseason=True)
    assert set(widened["season_type"]) == {"REG", "POST"}

    options = {"span": 3, "min_periods": 1, "opponent_min_team_games": 2}
    pd.testing.assert_frame_equal(
        enrich_with_pbp_features(_games(), load_pbp_snapshot(with_post), **options),
        enrich_with_pbp_features(_games(), load_pbp_snapshot(reg_only), **options),
    )


def test_snapshot_without_the_scope_key_still_loads(tmp_path: Path) -> None:
    snapshot = write_pbp_snapshot({2022: _pbp_frame(1)}, tmp_path, "legacy")
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    del manifest["include_postseason"]
    snapshot.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert snapshot_from_root(snapshot.root) == snapshot
    assert len(load_pbp_snapshot(snapshot_from_root(snapshot.root))) == 6


def test_snapshot_is_partitioned_hashed_and_loadable(tmp_path: Path) -> None:
    frame = _pbp_frame(1)
    snapshot = write_pbp_snapshot({2022: frame}, tmp_path, "fixed")
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert manifest["filter_version"] == PBP_FILTER_VERSION
    assert manifest["partitions"][0]["sha256"]
    assert len(load_pbp_snapshot(snapshot)) == len(frame)


def test_analysis_filter_drives_and_team_metrics() -> None:
    frame = _pbp_frame(1)
    frame.loc[0, "qb_kneel"] = 1
    plays = analysis_plays(frame)
    assert len(plays) == len(frame) - 1
    assert len(build_drive_table(frame)) == 2
    metrics = build_pbp_team_game_metrics(frame)
    assert len(metrics) == 2
    assert set(PBP_STATE_METRICS).issubset(metrics.columns)
    assert set(DRIVE_STATE_METRICS).issubset(metrics.columns)
    assert metrics["opponent"].notna().all()
    assert metrics["pbp_off_epa_per_play"].notna().all()
    team_a = metrics.loc[metrics["team"].eq("A")].iloc[0]
    assert team_a["drive_yards_per_drive"] == pytest.approx(18.0)
    assert team_a["drive_plays_per_drive"] == pytest.approx(2.0)
    assert team_a["drive_seconds_per_drive"] == pytest.approx(30.0)


def test_current_game_plays_cannot_change_current_pregame_features() -> None:
    games = _games()
    pbp = _pbp_frame()
    baseline = enrich_with_pbp_features(games, pbp, span=3, min_periods=1)
    changed_pbp = pbp.copy()
    mask = changed_pbp["game_id"].eq("2022_02_B_A") & changed_pbp["posteam"].eq("A")
    changed_pbp.loc[mask, "epa"] = 1_000.0
    changed = enrich_with_pbp_features(games, changed_pbp, span=3, min_periods=1)
    column = "home_pbp_off_epa_per_play"
    assert changed.loc[1, column] == pytest.approx(baseline.loc[1, column])
    assert changed.loc[2, column] != pytest.approx(baseline.loc[2, column])


def test_opponent_adjusted_matchups_are_weekly_and_point_in_time() -> None:
    games = _games()
    pbp = _pbp_frame()
    baseline = enrich_with_pbp_features(
        games,
        pbp,
        span=3,
        min_periods=1,
        opponent_min_team_games=2,
    )
    assert set(PBP_OPPONENT_ADJUSTED_FEATURE_COLUMNS).issubset(baseline.columns)
    assert baseline.loc[1, "diff_pbp_matchup_epa_per_play"] > 0

    changed_pbp = pbp.copy()
    current = changed_pbp["game_id"].eq("2022_02_B_A")
    changed_pbp.loc[current & changed_pbp["posteam"].eq("A"), "epa"] = -1_000.0
    changed = enrich_with_pbp_features(
        games,
        changed_pbp,
        span=3,
        min_periods=1,
        opponent_min_team_games=2,
    )
    column = "diff_pbp_matchup_epa_per_play"
    assert changed.loc[1, column] == pytest.approx(baseline.loc[1, column])
    assert changed.loc[2, column] != pytest.approx(baseline.loc[2, column])


def test_opponent_adjustment_is_invariant_to_input_order_and_future_plays() -> None:
    games = _games()
    pbp = _pbp_frame()
    expected = enrich_with_pbp_features(
        games,
        pbp,
        min_periods=1,
        opponent_min_team_games=2,
    )
    shuffled = enrich_with_pbp_features(
        games.sample(frac=1.0, random_state=7),
        pbp.sample(frac=1.0, random_state=11),
        min_periods=1,
        opponent_min_team_games=2,
    )
    columns = ["game_id", *PBP_OPPONENT_ADJUSTED_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(
        expected[columns].sort_values("game_id").reset_index(drop=True),
        shuffled[columns].sort_values("game_id").reset_index(drop=True),
    )

    changed_future = pbp.copy()
    future = changed_future["game_id"].eq("2022_03_B_A")
    changed_future.loc[future, "epa"] = 1_000.0
    rescored = enrich_with_pbp_features(
        games,
        changed_future,
        min_periods=1,
        opponent_min_team_games=2,
    )
    pd.testing.assert_frame_equal(expected[columns], rescored[columns])


def test_pbp_contract_rejects_duplicate_plays(tmp_path: Path) -> None:
    frame = pd.concat([_pbp_frame(1), _pbp_frame(1).iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate"):
        write_pbp_snapshot({2022: frame}, tmp_path, "duplicate")


def test_pbp_snapshot_discovery_and_guards(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No play-by-play"):
        latest_pbp_snapshot(tmp_path)
    snapshot = write_pbp_snapshot({2022: _pbp_frame(1)}, tmp_path, "fixed")
    assert latest_pbp_snapshot(tmp_path) == snapshot
    assert snapshot_from_root(snapshot.root) == snapshot
    with pytest.raises(FileNotFoundError, match="manifest"):
        snapshot_from_root(tmp_path / "missing")
    with pytest.raises(ValueError, match="span"):
        enrich_with_pbp_features(_games(), _pbp_frame(), span=1)
    with pytest.raises(ValueError, match="min_periods"):
        enrich_with_pbp_features(_games(), _pbp_frame(), min_periods=0)
    with pytest.raises(ValueError, match="offseason"):
        enrich_with_pbp_features(_games(), _pbp_frame(), offseason_retention=2.0)
    with pytest.raises(ValueError, match="half_life"):
        enrich_with_pbp_features(_games(), _pbp_frame(), opponent_half_life_weeks=0)
    with pytest.raises(ValueError, match="ridge_alpha"):
        enrich_with_pbp_features(_games(), _pbp_frame(), opponent_ridge_alpha=0)
    with pytest.raises(ValueError, match="min_team_games"):
        enrich_with_pbp_features(_games(), _pbp_frame(), opponent_min_team_games=1)
