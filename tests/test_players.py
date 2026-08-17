from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import PLAYER_PARTICIPATION_STATE_METRICS, PLAYER_STATE_METRICS
from nfl_ats.data import DataContractError
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS
from nfl_ats.players import (
    canonicalize_injuries,
    canonicalize_player_stats,
    canonicalize_rosters,
    canonicalize_snaps,
    enrich_with_player_features,
    latest_player_snapshot,
    latest_player_value_snapshot,
    load_player_snapshot,
    load_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
    write_player_snapshot,
    write_player_value_snapshot,
)


def _games() -> pd.DataFrame:
    dates = pd.date_range("2022-09-11", periods=4, freq="7D")
    return pd.DataFrame(
        {
            "game_id": [f"2022_{week:02d}_B_A" for week in range(1, 5)],
            "season": 2022,
            "week": range(1, 5),
            "gameday": dates,
            "kickoff": pd.date_range("2022-09-11 17:00Z", periods=4, freq="7D"),
            "away_team": "B",
            "home_team": "A",
        }
    )


def _rosters() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, player_id, pfr_id, position, experience in (
            ("A", "QB-A", "PFR-A", "QB", 5),
            ("A", "WR-A", "PFR-WR-A", "WR", 4),
            ("B", "QB-B", "PFR-B", "QB", 3),
            ("B", "WR-B", "PFR-WR-B", "WR", 2),
        ):
            rows.append(
                {
                    "season": 2022,
                    "team": team,
                    "position": position,
                    "status": "ACT",
                    "full_name": f"{position} {team}",
                    "gsis_id": player_id,
                    "pfr_id": pfr_id,
                    "years_exp": experience,
                    "week": week,
                    "game_type": "REG",
                }
            )
    return pd.DataFrame(rows)


def _injuries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2022],
            "game_type": ["REG", "REG"],
            "team": ["A", "A"],
            "week": [2, 2],
            "gsis_id": ["QB-A", "QB-A"],
            "position": ["QB", "QB"],
            "report_status": ["Questionable", "Out"],
            "practice_status": ["Limited Participation in Practice", "Did Not Participate"],
            "date_modified": ["2022-09-16T12:00:00Z", "2022-09-18T16:30:00Z"],
        }
    )


def _snaps() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, opponent, pfr_id, position, offense_pct in (
            ("A", "B", "PFR-A", "QB", 1.0),
            ("A", "B", "PFR-WR-A", "WR", 0.8),
            ("B", "A", "PFR-B", "QB", 1.0),
            ("B", "A", "PFR-WR-B", "WR", 0.8),
        ):
            rows.append(
                {
                    "game_id": f"2022_{week:02d}_B_A",
                    "season": 2022,
                    "game_type": "REG",
                    "week": week,
                    "player": f"{position} {team}",
                    "pfr_player_id": pfr_id,
                    "position": position,
                    "team": team,
                    "opponent": opponent,
                    "offense_snaps": 60 * offense_pct,
                    "offense_pct": offense_pct,
                    "defense_snaps": 0,
                    "defense_pct": 0.0,
                    "st_snaps": 0,
                    "st_pct": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _pbp() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, opponent, quarterback, direction in (
            ("A", "B", "QB-A", 1.0),
            ("B", "A", "QB-B", -1.0),
        ):
            for play_id in range(1, 7):
                row = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
                row.update(
                    {
                        "play_id": play_id + (100 if team == "B" else 0),
                        "game_id": f"2022_{week:02d}_B_A",
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
                        "interception": 0,
                        "epa": direction * week + play_id / 100,
                        "success": int(direction > 0),
                        "wp": 0.5,
                        "passer_player_id": quarterback,
                        "passer_player_name": f"QB {team}",
                        "cpoe": direction * 2,
                        "pass_oe": 0.1,
                        "yardline_100": 60,
                        "play": 1,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _player_stats() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, player_id, position, direction in (
            ("A", "QB-A", "QB", 1.0),
            ("A", "WR-A", "WR", 1.0),
            ("B", "QB-B", "QB", -1.0),
            ("B", "WR-B", "WR", -1.0),
        ):
            rows.append(
                {
                    "player_id": player_id,
                    "season": 2022,
                    "week": week,
                    "season_type": "REG",
                    "game_id": f"2022_{week:02d}_B_A",
                    "team": team,
                    "position": position,
                    "rushing_epa": 0.0,
                    "receiving_epa": direction * week if position == "WR" else 0.0,
                    "def_tackles_for_loss": 0.0,
                    "def_fumbles_forced": 0.0,
                    "def_sacks": 0.0,
                    "def_qb_hits": 0.0,
                    "def_interceptions": 0.0,
                    "def_pass_defended": 0.0,
                }
            )
    return pd.DataFrame(rows)


# nflverse injuries, weekly rosters, and snap counts spell the postseason with
# the per-round game_type codes (WC/DIV/CON/SB); weekly player stats use a
# season_type column whose only postseason value is POST.
def _postseason_injuries() -> pd.DataFrame:
    frame = _injuries()
    frame["game_type"] = "WC"
    frame["week"] = 19
    return frame


def _postseason_rosters() -> pd.DataFrame:
    frame = _rosters()
    frame = frame.loc[frame["week"].eq(4)].reset_index(drop=True)
    frame["game_type"] = "WC"
    frame["week"] = 19
    return frame


def _postseason_snaps() -> pd.DataFrame:
    frame = _snaps()
    frame = frame.loc[frame["week"].eq(4)].reset_index(drop=True)
    frame["game_type"] = "WC"
    frame["week"] = 19
    frame["game_id"] = "2022_19_B_A"
    return frame


def _postseason_player_stats() -> pd.DataFrame:
    frame = _player_stats()
    frame = frame.loc[frame["week"].eq(4)].reset_index(drop=True)
    frame["season_type"] = "POST"
    frame["week"] = 19
    frame["game_id"] = "2022_19_B_A"
    return frame


def test_canonicalize_player_sources_keep_regular_season_by_default() -> None:
    cases = (
        (canonicalize_injuries, "game_type", _injuries(), _postseason_injuries()),
        (canonicalize_rosters, "game_type", _rosters(), _postseason_rosters()),
        (canonicalize_snaps, "game_type", _snaps(), _postseason_snaps()),
        (
            canonicalize_player_stats,
            "season_type",
            _player_stats(),
            _postseason_player_stats(),
        ),
    )
    for canonicalize, scope_column, regular, postseason in cases:
        mixed = pd.concat([regular, postseason], ignore_index=True)
        expected = canonicalize(regular)
        pd.testing.assert_frame_equal(canonicalize(mixed), expected)

        widened = canonicalize(mixed, include_postseason=True)
        assert len(widened) == len(mixed)
        assert set(widened[scope_column]) == {"REG", set(postseason[scope_column]).pop()}
        pd.testing.assert_frame_equal(
            widened.loc[widened[scope_column].eq("REG")].reset_index(drop=True), expected
        )


def test_canonicalize_player_sources_reject_unknown_codes_when_widened() -> None:
    injuries = _injuries()
    injuries.loc[0, "game_type"] = "PLAYOFF"
    with pytest.raises(DataContractError, match="unrecognized season codes"):
        canonicalize_injuries(injuries, include_postseason=True)
    # The default path keeps its historical, silent "== REG" comparison.
    assert len(canonicalize_injuries(injuries)) == 1

    stats = _player_stats()
    stats.loc[0, "season_type"] = "POSTSEASON"
    with pytest.raises(DataContractError, match="unrecognized season codes"):
        canonicalize_player_stats(stats, include_postseason=True)


def test_postseason_player_snapshot_reads_back_as_regular_season_only(tmp_path: Path) -> None:
    mixed = {
        "injuries": pd.concat([_injuries(), _postseason_injuries()], ignore_index=True),
        "rosters": pd.concat([_rosters(), _postseason_rosters()], ignore_index=True),
        "snaps": pd.concat([_snaps(), _postseason_snaps()], ignore_index=True),
        "stats": pd.concat([_player_stats(), _postseason_player_stats()], ignore_index=True),
    }
    seasons = [2022]
    regular_snapshot = write_player_snapshot(
        _injuries(), _rosters(), _snaps(), tmp_path / "reg", seasons, seasons, seasons, "fixed"
    )
    postseason_snapshot = write_player_snapshot(
        mixed["injuries"],
        mixed["rosters"],
        mixed["snaps"],
        tmp_path / "post",
        seasons,
        seasons,
        seasons,
        "fixed",
        include_postseason=True,
    )
    regular_values = write_player_value_snapshot(
        _player_stats(), tmp_path / "reg_values", seasons, "fixed"
    )
    postseason_values = write_player_value_snapshot(
        mixed["stats"], tmp_path / "post_values", seasons, "fixed", include_postseason=True
    )
    for snapshot, expected in (
        (regular_snapshot, False),
        (postseason_snapshot, True),
        (regular_values, False),
        (postseason_values, True),
    ):
        manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
        assert manifest["include_postseason"] is expected

    for expected_frame, actual_frame in zip(
        load_player_snapshot(regular_snapshot),
        load_player_snapshot(postseason_snapshot),
        strict=True,
    ):
        pd.testing.assert_frame_equal(actual_frame, expected_frame)
    pd.testing.assert_frame_equal(
        load_player_value_snapshot(postseason_values),
        load_player_value_snapshot(regular_values),
    )
    widened_injuries, _, _ = load_player_snapshot(postseason_snapshot, include_postseason=True)
    assert set(widened_injuries["game_type"]) == {"REG", "WC"}
    assert set(
        load_player_value_snapshot(postseason_values, include_postseason=True)["season_type"]
    ) == {"REG", "POST"}

    # The invariant: identical feature-build inputs produce identical features,
    # and enrich_with_player_features re-canonicalizes to REG even when handed
    # postseason-inclusive frames directly.
    baseline = enrich_with_player_features(
        _games(),
        *load_player_snapshot(regular_snapshot),
        _pbp(),
        load_player_value_snapshot(regular_values),
        qb_min_dropbacks=1,
    )
    from_postseason_snapshot = enrich_with_player_features(
        _games(),
        *load_player_snapshot(postseason_snapshot),
        _pbp(),
        load_player_value_snapshot(postseason_values),
        qb_min_dropbacks=1,
    )
    from_raw_postseason_rows = enrich_with_player_features(
        _games(),
        mixed["injuries"],
        mixed["rosters"],
        mixed["snaps"],
        _pbp(),
        mixed["stats"],
        qb_min_dropbacks=1,
    )
    pd.testing.assert_frame_equal(from_postseason_snapshot, baseline)
    pd.testing.assert_frame_equal(from_raw_postseason_rows, baseline)


def test_player_snapshot_round_trip_and_contract(tmp_path: Path) -> None:
    snapshot = write_player_snapshot(
        _injuries(),
        _rosters(),
        _snaps(),
        tmp_path,
        [2022],
        [2022],
        [2022],
        "fixed",
    )
    injuries, rosters, snaps = load_player_snapshot(snapshot)
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert len(injuries) == 2
    assert len(rosters) == 16
    assert len(snaps) == 16
    assert manifest["files"]["snap_counts"]["sha256"]
    assert latest_player_snapshot(tmp_path) == snapshot
    assert player_snapshot_from_root(snapshot.root) == snapshot

    value_snapshot = write_player_value_snapshot(
        _player_stats(), tmp_path / "values", [2022], "fixed"
    )
    assert len(load_player_value_snapshot(value_snapshot)) == 16
    assert latest_player_value_snapshot(tmp_path / "values") == value_snapshot
    assert player_value_snapshot_from_root(value_snapshot.root) == value_snapshot


def test_current_game_outcomes_cannot_change_current_player_state() -> None:
    baseline = enrich_with_player_features(
        _games(), _injuries(), _rosters(), _snaps(), _pbp(), qb_min_dropbacks=1
    )
    changed_snaps = _snaps()
    changed_snaps.loc[
        changed_snaps["game_id"].eq("2022_02_B_A")
        & changed_snaps["team"].eq("A")
        & changed_snaps["position"].eq("QB"),
        "pfr_player_id",
    ] = "PFR-NEW"
    changed_snaps.loc[
        changed_snaps["game_id"].eq("2022_02_B_A")
        & changed_snaps["team"].eq("A")
        & changed_snaps["position"].eq("QB"),
        "player",
    ] = "Different Player"
    changed_pbp = _pbp()
    changed_pbp.loc[
        changed_pbp["game_id"].eq("2022_02_B_A") & changed_pbp["posteam"].eq("A"),
        "epa",
    ] = 1_000.0
    changed = enrich_with_player_features(
        _games(),
        _injuries(),
        _rosters(),
        changed_snaps,
        changed_pbp,
        qb_min_dropbacks=1,
    )
    assert changed.loc[1, "home_qb_starter_epa_per_dropback"] == pytest.approx(
        baseline.loc[1, "home_qb_starter_epa_per_dropback"]
    )
    assert changed.loc[2, "home_qb_starter_epa_per_dropback"] != pytest.approx(
        baseline.loc[2, "home_qb_starter_epa_per_dropback"]
    )
    assert pd.isna(changed.loc[1, "home_offense_lineup_continuity"])
    assert changed.loc[2, "home_offense_lineup_continuity"] != pytest.approx(
        baseline.loc[2, "home_offense_lineup_continuity"]
    )


def test_injury_cutoff_uses_latest_visible_revision_and_delays_rosters() -> None:
    enriched = enrich_with_player_features(
        _games(), _injuries(), _rosters(), _snaps(), _pbp(), qb_min_dropbacks=1
    )
    # The questionable Friday report is visible 24 hours before Sunday kickoff;
    # the out designation posted 30 minutes before kickoff is not.
    assert enriched.loc[1, "home_qb_start_probability"] == pytest.approx(0.65)
    assert enriched.loc[1, "home_injury_skill_unavailability"] > 0
    assert enriched.loc[1, "home_injury_observed_at"] == pd.Timestamp("2022-09-16T12:00:00Z")

    changed_rosters = _rosters()
    changed_rosters.loc[
        changed_rosters["week"].eq(2) & changed_rosters["team"].eq("A"), "years_exp"
    ] = 99
    changed = enrich_with_player_features(
        _games(), _injuries(), changed_rosters, _snaps(), _pbp(), qb_min_dropbacks=1
    )
    assert changed.loc[1, "home_active_roster_mean_experience"] == pytest.approx(
        enriched.loc[1, "home_active_roster_mean_experience"]
    )
    assert changed.loc[2, "home_active_roster_mean_experience"] == pytest.approx(99)
    assert set(PLAYER_STATE_METRICS).issubset(
        column.removeprefix("home_") for column in enriched.columns if column.startswith("home_")
    )


def test_player_value_uses_only_prior_game_stats() -> None:
    injuries = pd.concat(
        [
            _injuries(),
            pd.DataFrame(
                {
                    "season": [2022, 2022],
                    "game_type": ["REG", "REG"],
                    "team": ["A", "A"],
                    "week": [2, 3],
                    "gsis_id": ["WR-A", "WR-A"],
                    "position": ["WR", "WR"],
                    "report_status": ["Questionable", "Questionable"],
                    "practice_status": [
                        "Limited Participation in Practice",
                        "Limited Participation in Practice",
                    ],
                    "date_modified": [
                        "2022-09-16T12:00:00Z",
                        "2022-09-23T12:00:00Z",
                    ],
                }
            ),
        ],
        ignore_index=True,
    )
    baseline = enrich_with_player_features(
        _games(),
        injuries,
        _rosters(),
        _snaps(),
        _pbp(),
        _player_stats(),
        qb_min_dropbacks=1,
        value_prior_snaps=0,
    )
    assert baseline.loc[1, "home_injury_skill_epa_value_lost"] > 0
    changed_stats = _player_stats()
    changed_stats.loc[
        changed_stats["game_id"].eq("2022_02_B_A") & changed_stats["player_id"].eq("WR-A"),
        "receiving_epa",
    ] = 1_000.0
    changed = enrich_with_player_features(
        _games(),
        injuries,
        _rosters(),
        _snaps(),
        _pbp(),
        changed_stats,
        qb_min_dropbacks=1,
        value_prior_snaps=0,
    )
    assert changed.loc[1, "home_injury_skill_epa_value_lost"] == pytest.approx(
        baseline.loc[1, "home_injury_skill_epa_value_lost"]
    )
    assert changed.loc[2, "home_injury_skill_epa_value_lost"] != pytest.approx(
        baseline.loc[2, "home_injury_skill_epa_value_lost"]
    )


def test_participation_ratings_weight_visible_injuries_by_prior_role() -> None:
    injuries = pd.concat(
        [
            _injuries(),
            pd.DataFrame(
                {
                    "season": [2022],
                    "game_type": ["REG"],
                    "team": ["A"],
                    "week": [2],
                    "gsis_id": ["WR-A"],
                    "position": ["WR"],
                    "report_status": ["Questionable"],
                    "practice_status": ["Limited Participation in Practice"],
                    "date_modified": ["2022-09-16T12:00:00Z"],
                }
            ),
        ],
        ignore_index=True,
    )
    ratings = pd.DataFrame(
        {
            "target_season": [2022],
            "player_id": ["WR-A"],
            "offense_rating": [0.2],
            "defense_rating": [0.0],
            "offense_plays": [500],
            "defense_plays": [0],
            "source_start_season": [2019],
            "source_end_season": [2021],
            "source_plays": [100_000],
            "lookback_seasons": [3],
            "ridge_alpha": [1000.0],
            "team_feature_scale": [11.0],
            "reliability_prior_plays": [500.0],
            "epa_clip": [5.0],
            "rating_version": ["v1"],
        }
    )
    enriched = enrich_with_player_features(
        _games(),
        injuries,
        _rosters(),
        _snaps(),
        _pbp(),
        _player_stats(),
        ratings,
        qb_min_dropbacks=1,
    )
    expected = 0.35 * 0.8 * 0.2
    assert enriched.loc[1, "home_injury_offense_participation_value_lost"] == pytest.approx(
        expected
    )
    assert enriched.loc[1, "diff_injury_offense_participation_value_lost"] == pytest.approx(
        expected
    )
    assert set(PLAYER_PARTICIPATION_STATE_METRICS).issubset(
        column.removeprefix("home_") for column in enriched.columns if column.startswith("home_")
    )
    assert enriched["player_feature_version"].eq("v3-participation-v1").all()


def test_learned_availability_replaces_fixed_injury_weight() -> None:
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
    fixed = enrich_with_player_features(
        _games(), _injuries(), _rosters(), _snaps(), _pbp(), qb_min_dropbacks=1
    )
    learned = enrich_with_player_features(
        _games(),
        _injuries(),
        _rosters(),
        _snaps(),
        _pbp(),
        availability_rates=rates,
        qb_min_dropbacks=1,
    )
    assert fixed.loc[1, "home_qb_start_probability"] == pytest.approx(0.65)
    assert learned.loc[1, "home_qb_start_probability"] == pytest.approx(0.2)
    assert (
        learned.loc[1, "home_injury_skill_unavailability"]
        > fixed.loc[1, "home_injury_skill_unavailability"]
    )
    assert learned["player_feature_version"].eq("v3-availability-v1").all()


def test_player_contract_guards(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No player snapshots"):
        latest_player_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError, match="manifest"):
        player_snapshot_from_root(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="No player-value snapshots"):
        latest_player_value_snapshot(tmp_path / "values")
    with pytest.raises(ValueError, match="decision_hours"):
        enrich_with_player_features(
            _games(),
            _injuries(),
            _rosters(),
            _snaps(),
            _pbp(),
            decision_hours_before_kickoff=-1,
        )
