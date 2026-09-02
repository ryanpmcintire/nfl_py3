from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.availability import resolve_unavailability
from nfl_ats.constants import FEATURE_FAMILIES
from nfl_ats.data import DataContractError
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS
from nfl_ats.quarterbacks import (
    canonicalize_depth_charts,
    canonicalize_historical_depth_charts,
    depth_snapshot_from_root,
    enrich_with_qb_features,
    latest_depth_snapshot,
    latest_starting_qbs,
    load_depth_snapshot,
    write_depth_snapshot,
    write_historical_depth_snapshot,
)


def _depth_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dt": [
                "2022-09-09T12:00:00Z",
                "2022-09-09T12:00:00Z",
                "2022-09-09T12:00:00Z",
                "2022-09-16T12:00:00Z",
                "2022-09-16T12:00:00Z",
                None,
            ],
            "team": ["A", "A", "B", "A", "B", "A"],
            "player_name": ["QB A", "Backup A", "QB B", "QB A", "QB B", "Old A"],
            "gsis_id": ["QB-A", "QB-A2", "QB-B", "QB-A", "QB-B", "OLD-A"],
            "pos_abb": ["QB", "QB", "QB", "QB", "QB", "QB"],
            "pos_rank": [1, 2, 1, 1, 1, 1],
            "espn_id": [1, 2, 3, 1, 3, 4],
            "pos_slot": 9,
        }
    )


def _qb_pbp() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 4):
        game_id = f"2022_{week:02d}_B_A"
        for team, opponent, player, direction in (
            ("A", "B", "QB-A", 1.0),
            ("B", "A", "QB-B", -1.0),
        ):
            for play_number in range(1, 7):
                row = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
                row.update(
                    {
                        "play_id": play_number + (100 if team == "B" else 0),
                        "game_id": game_id,
                        "season": 2022,
                        "season_type": "REG",
                        "week": week,
                        "home_team": "A",
                        "away_team": "B",
                        "posteam": team,
                        "defteam": opponent,
                        "fixed_drive": 1 if team == "A" else 2,
                        "down": 1,
                        "play_type": "pass",
                        "yards_gained": 8,
                        "pass_attempt": 1,
                        "rush_attempt": 0,
                        "qb_dropback": 1,
                        "qb_kneel": 0,
                        "qb_spike": 0,
                        "aborted_play": 0,
                        "sack": 0,
                        "qb_hit": 0,
                        "interception": int(play_number == 6 and team == "B"),
                        "epa": direction * week + play_number / 100,
                        "success": int(direction > 0),
                        "wp": 0.5,
                        "passer_player_id": player,
                        "passer_player_name": f"QB {team}",
                        "cpoe": direction * 2,
                        "pass_oe": 0.1,
                        "yardline_100": 60,
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
            "kickoff": pd.date_range("2022-09-11 17:00Z", periods=3, freq="7D"),
            "away_team": "B",
            "home_team": "A",
        }
    )


def _named_depth_history() -> pd.DataFrame:
    rows = _depth_history()
    backups = pd.DataFrame(
        {
            "dt": [
                "2022-09-09T12:00:00Z",
                "2022-09-16T12:00:00Z",
                "2022-09-16T12:00:00Z",
            ],
            "team": ["B", "A", "B"],
            "player_name": ["Backup B", "Backup A", "Backup B"],
            "gsis_id": ["QB-B2", "QB-A2", "QB-B2"],
            "pos_abb": "QB",
            "pos_rank": 2,
            "espn_id": [4, 2, 4],
            "pos_slot": 9,
        }
    )
    return canonicalize_depth_charts(pd.concat([rows, backups], ignore_index=True))


def _pbp_with_backup_history() -> pd.DataFrame:
    pbp = _qb_pbp()
    prior = pbp.loc[pbp["game_id"].eq("2022_01_B_A")].copy()
    backup_rows = []
    for team, player_id, epa in (("A", "QB-A2", -2.0), ("B", "QB-B2", -1.0)):
        rows = prior.loc[prior["posteam"].eq(team)].copy()
        rows["play_id"] = pd.to_numeric(rows["play_id"]) + 1_000
        rows["passer_player_id"] = player_id
        rows["passer_player_name"] = f"Backup {team}"
        rows["epa"] = epa
        backup_rows.append(rows)
    return pd.concat([pbp, *backup_rows], ignore_index=True)


def _qb_injuries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "week": [2, 2, 2],
            "team": ["A", "A", "B"],
            "gsis_id": ["QB-A", "QB-A", "WR-B"],
            "position": ["QB", "QB", "WR"],
            "report_status": ["Questionable", "Out", "Questionable"],
            "practice_status": [
                "Limited Participation in Practice",
                "Did Not Participate In Practice",
                "Limited Participation in Practice",
            ],
            "date_modified": [
                "2022-09-16T12:00:00Z",
                "2022-09-18T16:30:00Z",
                "2022-09-16T12:00:00Z",
            ],
        }
    )


def _legacy_depth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2022, 2022],
            "club_code": ["A", "A", "B", "B"],
            "week": [1, 1, 1, 1],
            "game_type": ["REG", "REG", "REG", "REG"],
            "depth_team": [1, 2, 1, 2],
            "gsis_id": ["QB-A", "QB-A2", "QB-B", "QB-B2"],
            "position": ["QB", "QB", "QB", "QB"],
            "formation": ["Offense", "Offense", "Offense", "Offense"],
            "depth_position": ["QB", "QB", "QB", "QB"],
            "full_name": ["QB A", "Backup A", "QB B", "Backup B"],
        }
    )


def test_depth_snapshot_keeps_only_timestamped_qbs(tmp_path: Path) -> None:
    snapshot = write_depth_snapshot(_depth_history(), tmp_path, [2022], "fixed")
    stored = pd.read_parquet(snapshot.data_path)
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert len(stored) == 5
    assert manifest["rows"] == 5
    assert manifest["sha256"]


def test_asof_depth_uses_latest_rank_one_and_rejects_stale() -> None:
    depth = canonicalize_depth_charts(_depth_history())
    starters = latest_starting_qbs(depth, pd.Timestamp("2022-09-17T00:00Z"))
    assert starters.set_index("team").loc["A", "gsis_id"] == "QB-A"
    assert starters.set_index("team").loc["B", "gsis_id"] == "QB-B"
    stale = latest_starting_qbs(depth, pd.Timestamp("2022-10-20T00:00Z"), max_age_days=14)
    assert stale.empty


def test_current_qb_game_cannot_change_current_pregame_state() -> None:
    games = _games()
    depth = canonicalize_depth_charts(_depth_history())
    baseline = enrich_with_qb_features(
        games,
        _qb_pbp(),
        depth,
        decision_hours_before_kickoff=24,
        max_depth_age_days=14,
        min_dropbacks=1,
    )
    changed_pbp = _qb_pbp()
    mask = changed_pbp["game_id"].eq("2022_02_B_A") & changed_pbp["posteam"].eq("A")
    changed_pbp.loc[mask, "epa"] = 1_000.0
    changed = enrich_with_qb_features(
        games,
        changed_pbp,
        depth,
        decision_hours_before_kickoff=24,
        max_depth_age_days=14,
        min_dropbacks=1,
    )
    column = "home_qb_epa_per_dropback"
    assert changed.loc[1, column] == pytest.approx(baseline.loc[1, column])
    assert changed.loc[2, column] != pytest.approx(baseline.loc[2, column])


def test_depth_rows_after_decision_are_never_visible() -> None:
    depth = canonicalize_depth_charts(_depth_history())
    before_second_update = latest_starting_qbs(depth, pd.Timestamp("2022-09-15T00:00Z"))
    assert before_second_update["observed_at_utc"].max() == pd.Timestamp("2022-09-09T12:00:00Z")


def test_named_backup_value_and_visible_starter_probability_are_mixed() -> None:
    enriched = enrich_with_qb_features(
        _games(),
        _pbp_with_backup_history(),
        _named_depth_history(),
        _qb_injuries(),
        decision_hours_before_kickoff=24,
        min_dropbacks=1,
    )
    row = enriched.iloc[1]
    assert row["home_qb_id"] == "QB-A"
    assert row["home_depth_qb_backup_id"] == "QB-A2"
    assert row["home_depth_qb_start_probability"] == pytest.approx(0.65)
    assert row["home_depth_qb_availability_source"] == "fixed_status_prior"
    assert row["home_depth_qb_availability_observed_at"] == pd.Timestamp("2022-09-16T12:00:00Z")
    expected_epa = (
        0.65 * row["home_depth_qb_starter_epa_per_dropback"]
        + 0.35 * row["home_depth_qb_backup_epa_per_dropback"]
    )
    assert row["home_depth_qb_expected_epa_per_dropback"] == pytest.approx(expected_epa)
    assert row["home_depth_qb_backup_adjustment_epa_per_dropback"] == pytest.approx(
        row["home_depth_qb_backup_epa_per_dropback"] - row["home_depth_qb_starter_epa_per_dropback"]
    )
    assert set(FEATURE_FAMILIES["quarterback_depth"]).issubset(enriched.columns)


def test_depth_and_player_qb_families_are_disjoint_and_share_availability_policy() -> None:
    assert set(FEATURE_FAMILIES["quarterback_depth"]).isdisjoint(FEATURE_FAMILIES["player_qb"])
    unavailable, source = resolve_unavailability(
        None,
        target_season=None,
        report_status="Questionable",
        practice_status="Limited Participation in Practice",
        position="QB",
    )
    enriched = enrich_with_qb_features(
        _games(),
        _pbp_with_backup_history(),
        _named_depth_history(),
        _qb_injuries(),
        decision_hours_before_kickoff=24,
        min_dropbacks=1,
    )
    assert source == "fixed_status_prior"
    assert enriched.loc[1, "home_depth_qb_start_probability"] == pytest.approx(1.0 - unavailable)


def test_season_lagged_availability_replaces_fixed_starter_probability() -> None:
    rates = pd.DataFrame(
        {
            "target_season": [2022],
            "report_category": ["questionable"],
            "practice_category": ["limited"],
            "position_group": ["skill"],
            "unavailability_probability": [0.8],
            "observations": [100],
            "unavailable": [80],
            "source_start_season": [2013],
            "source_end_season": [2021],
            "combination_prior": [20.0],
            "position_prior": [100.0],
            "rate_version": ["v1"],
        }
    )
    enriched = enrich_with_qb_features(
        _games(),
        _pbp_with_backup_history(),
        _named_depth_history(),
        _qb_injuries(),
        rates,
        decision_hours_before_kickoff=24,
        min_dropbacks=1,
    )
    assert enriched.loc[1, "home_depth_qb_start_probability"] == pytest.approx(0.2)
    assert enriched.loc[1, "home_depth_qb_availability_source"] == "season_lagged_rate"


def test_future_depth_and_injury_revisions_cannot_change_pregame_qb_state() -> None:
    games = _games()
    baseline = enrich_with_qb_features(
        games,
        _pbp_with_backup_history(),
        _named_depth_history(),
        _qb_injuries(),
        decision_hours_before_kickoff=24,
        min_dropbacks=1,
    )
    depth = _named_depth_history()
    future_depth = depth.loc[
        depth["team"].eq("A") & depth["observed_at_utc"].eq("2022-09-16T12:00:00Z")
    ].copy()
    future_depth["observed_at_utc"] = pd.Timestamp("2022-09-18T16:30:00Z")
    future_depth["effective_at_utc"] = pd.Timestamp("2022-09-18T16:30:00Z")
    future_depth["pos_rank"] = future_depth["pos_rank"].map({1: 2, 2: 1})
    changed_depth = pd.concat([depth, future_depth], ignore_index=True)
    changed_injuries = _qb_injuries()
    changed_injuries.loc[1, "report_status"] = "Probable"
    changed = enrich_with_qb_features(
        games,
        _pbp_with_backup_history(),
        changed_depth,
        changed_injuries,
        decision_hours_before_kickoff=24,
        min_dropbacks=1,
    )
    columns = [
        "home_qb_id",
        "home_depth_qb_backup_id",
        "home_depth_qb_start_probability",
        "home_depth_qb_expected_epa_per_dropback",
        "home_depth_qb_expected_cpoe",
    ]
    pd.testing.assert_series_equal(
        changed.loc[1, columns], baseline.loc[1, columns], check_names=False
    )


def test_uncovered_injury_season_does_not_assume_starter_is_healthy() -> None:
    enriched = enrich_with_qb_features(
        _games(),
        _pbp_with_backup_history(),
        _named_depth_history(),
        decision_hours_before_kickoff=24,
        min_dropbacks=1,
    )
    assert pd.isna(enriched.loc[1, "home_depth_qb_start_probability"])
    assert enriched.loc[1, "home_depth_qb_availability_source"] == "injury_season_uncovered"
    assert pd.isna(enriched.loc[1, "home_depth_qb_expected_epa_per_dropback"])


def test_legacy_depth_becomes_visible_only_after_its_source_week_finishes() -> None:
    games = _games().copy()
    monday = games.iloc[[0]].copy()
    monday["game_id"] = "2022_01_D_C"
    monday["kickoff"] = pd.Timestamp("2022-09-13T00:15:00Z")
    monday["home_team"] = "C"
    monday["away_team"] = "D"
    chronology = pd.concat([games, monday], ignore_index=True)
    depth = canonicalize_historical_depth_charts(_legacy_depth(), chronology)
    expected_effective = pd.Timestamp("2022-09-13T00:15:00.000001Z")
    assert depth["observed_at_utc"].isna().all()
    assert depth["effective_at_utc"].eq(expected_effective).all()
    assert depth["provenance_mode"].eq("legacy_prior_week").all()
    assert latest_starting_qbs(depth, pd.Timestamp("2022-09-12T12:00:00Z")).empty
    visible = latest_starting_qbs(depth, pd.Timestamp("2022-09-17T00:00:00Z"))
    assert set(visible["gsis_id"]) == {"QB-A", "QB-B"}


def test_legacy_duplicate_player_roles_do_not_manufacture_a_backup() -> None:
    legacy = _legacy_depth()
    conflict = legacy.iloc[[0]].copy()
    conflict["depth_team"] = 2
    canonical = canonicalize_historical_depth_charts(
        pd.concat([legacy, conflict], ignore_index=True), _games()
    )
    player = canonical.loc[canonical["gsis_id"].eq("QB-A")]
    assert len(player) == 1
    assert player.iloc[0]["pos_rank"] == 1
    assert bool(player.iloc[0]["source_role_conflict"])


def test_historical_depth_fails_closed_without_schedule_chronology(tmp_path: Path) -> None:
    incomplete_games = _games().loc[_games()["week"].ne(1)]
    with pytest.raises(DataContractError, match="lack completed-week chronology"):
        canonicalize_historical_depth_charts(_legacy_depth(), incomplete_games)

    snapshot = write_historical_depth_snapshot(
        _legacy_depth(), _games(), tmp_path, [2022], snapshot_id="legacy"
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "v1-prior-week-effective"
    assert manifest["sha256"]
    assert load_depth_snapshot(snapshot)["observed_at_utc"].isna().all()
    with snapshot.data_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(DataContractError, match="hash"):
        load_depth_snapshot(snapshot)


def test_depth_snapshot_discovery_and_guards(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No depth-chart"):
        latest_depth_snapshot(tmp_path)
    snapshot = write_depth_snapshot(_depth_history(), tmp_path, [2022], "fixed")
    assert latest_depth_snapshot(tmp_path) == snapshot
    assert depth_snapshot_from_root(snapshot.root) == snapshot
    assert len(load_depth_snapshot(snapshot)) == 5
    with pytest.raises(FileNotFoundError, match="manifest"):
        depth_snapshot_from_root(tmp_path / "missing")
    with pytest.raises(ValueError, match="max_age_days"):
        latest_starting_qbs(
            canonicalize_depth_charts(_depth_history()),
            pd.Timestamp("2022-09-17T00:00Z"),
            max_age_days=0,
        )
