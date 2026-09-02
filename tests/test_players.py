from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import (
    FEATURE_FAMILIES,
    FEATURE_SETS,
    PLAYER_PARTICIPATION_STATE_METRICS,
    PLAYER_STATE_METRICS,
    ROSTER_RETURNING_SNAP_STATE_METRICS,
)
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


def _offseason_continuity_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a prior season with known retained/departed snap mass."""

    rosters = _rosters()
    prior_rosters = []
    prior_snaps = []
    for team, returning, departed, snap_triplets in (
        ("A", ("QB-A", "PFR-A"), ("OLD-A", "PFR-OLD-A"), ((80, 40, 10), (20, 60, 90))),
        ("B", ("QB-B", "PFR-B"), ("OLD-B", "PFR-OLD-B"), ((50, 20, 0), (50, 80, 100))),
    ):
        for (player_id, pfr_id), snaps in zip((returning, departed), snap_triplets, strict=True):
            prior_rosters.append(
                {
                    "season": 2021,
                    "team": team,
                    "position": "QB",
                    "status": "ACT",
                    "full_name": player_id,
                    "gsis_id": player_id,
                    "pfr_id": pfr_id,
                    "years_exp": 1,
                    "week": 18,
                    "game_type": "REG",
                }
            )
            offense, defense, special = snaps
            prior_snaps.append(
                {
                    "game_id": f"2021_18_{team}",
                    "season": 2021,
                    "game_type": "REG",
                    "week": 18,
                    "player": player_id,
                    "pfr_player_id": pfr_id,
                    "position": "QB",
                    "team": team,
                    "offense_snaps": offense,
                    "offense_pct": offense / 100,
                    "defense_snaps": defense,
                    "defense_pct": defense / 100,
                    "st_snaps": special,
                    "st_pct": special / 100,
                }
            )
    return (
        pd.concat([rosters, pd.DataFrame(prior_rosters)], ignore_index=True),
        pd.concat([_snaps(), pd.DataFrame(prior_snaps)], ignore_index=True),
    )


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


def test_roster_source_aliases_use_schedule_team_identity() -> None:
    rosters = _rosters().iloc[:6].copy()
    rosters["team"] = ["ARZ", "BLT", "CLV", "HST", "SL", "OAK"]
    assert canonicalize_rosters(rosters)["team"].to_list() == [
        "ARI",
        "BAL",
        "CLE",
        "HOU",
        "LA",
        "LV",
    ]


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


def test_returning_snap_prior_is_isolated_and_point_in_time_safe() -> None:
    rosters, snaps = _offseason_continuity_sources()
    baseline = enrich_with_player_features(
        _games(), _injuries(), rosters, snaps, _pbp(), qb_min_dropbacks=1
    )
    home_columns = [f"home_{metric}" for metric in ROSTER_RETURNING_SNAP_STATE_METRICS]
    away_columns = [f"away_{metric}" for metric in ROSTER_RETURNING_SNAP_STATE_METRICS]
    diff_columns = [f"diff_{metric}" for metric in ROSTER_RETURNING_SNAP_STATE_METRICS]

    # Week 1 cannot use its undated Week-1 roster; Week 2 may use that row.
    assert baseline.loc[0, home_columns + away_columns + diff_columns].isna().all()
    assert baseline.loc[1, home_columns].to_list() == pytest.approx([0.8, 0.4, 0.1])
    assert baseline.loc[1, away_columns].to_list() == pytest.approx([0.5, 0.2, 0.0])
    assert baseline.loc[1, diff_columns].to_list() == pytest.approx([0.3, 0.2, 0.1])

    # A target-game roster revision is not visible until the following week.
    changed_rosters = rosters.copy()
    changed_rosters.loc[
        changed_rosters["season"].eq(2022)
        & changed_rosters["week"].eq(2)
        & changed_rosters["team"].eq("A")
        & changed_rosters["gsis_id"].eq("QB-A"),
        "gsis_id",
    ] = "NEW-QB-A"
    changed = enrich_with_player_features(
        _games(), _injuries(), changed_rosters, snaps, _pbp(), qb_min_dropbacks=1
    )
    pd.testing.assert_series_equal(changed.loc[1, home_columns], baseline.loc[1, home_columns])
    assert changed.loc[2, "home_returning_offense_snap_share"] != pytest.approx(
        baseline.loc[2, "home_returning_offense_snap_share"]
    )

    # Future rosters and every target-season snap outcome are structurally
    # excluded from the prior-season numerator and denominator.
    future_rosters = rosters.copy()
    future_rosters.loc[future_rosters["week"].eq(4), "gsis_id"] = "FUTURE"
    target_snaps = snaps.copy()
    target_snaps.loc[target_snaps["season"].eq(2022), "offense_snaps"] = 1_000_000
    future_changed = enrich_with_player_features(
        _games(), _injuries(), future_rosters, target_snaps, _pbp(), qb_min_dropbacks=1
    )
    pd.testing.assert_frame_equal(
        future_changed.loc[:, home_columns + away_columns + diff_columns],
        baseline.loc[:, home_columns + away_columns + diff_columns],
    )
    assert set(FEATURE_FAMILIES["roster_returning_snaps"]).isdisjoint(
        FEATURE_FAMILIES["player_continuity"]
    )
    assert all(
        set(FEATURE_FAMILIES["roster_returning_snaps"]).isdisjoint(columns)
        for columns in FEATURE_SETS.values()
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


def test_value_shrinkage_target_zero_is_bit_identical_to_default() -> None:
    """MOD-06's opt-in path must leave default production behaviour untouched."""

    default_call = enrich_with_player_features(
        _games(), _injuries(), _rosters(), _snaps(), _pbp(), _player_stats(), qb_min_dropbacks=1
    )
    explicit_zero = enrich_with_player_features(
        _games(),
        _injuries(),
        _rosters(),
        _snaps(),
        _pbp(),
        _player_stats(),
        qb_min_dropbacks=1,
        value_shrinkage_target="zero",
    )
    pd.testing.assert_frame_equal(default_call, explicit_zero)


def _thin_player_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rosters/snaps/stats/injuries for MOD-06's position-prior candidate.

    WR-A and WR-B are given inflated (unrealistic, deliberately so -- this is
    synthetic fixture data, not a realistic snap count) offense_snaps so that,
    with ``value_prior_snaps=200``, both clear the "experienced" pool
    threshold after two games and form a POSITIVE (receiving_epa=2.0 flat
    every week, both teams -- no cancellation) league-wide skill-channel
    prior by week 3. WR2-A is a thin bench player on team A: one week (week 1)
    of a handful of snaps and zero recorded production, then injured "Out"
    from week 2 onward. Their own career_offense_snaps stays tiny (3), so
    under the "zero" target their value-lost contribution is exactly 0
    (raw_rate is exactly 0.0, since their only recorded receiving_epa is
    0.0); under "position_prior" it should shrink toward the positive pool
    prior instead, once the pool is populated (week 3 injury, not week 2 --
    the pool snapshot for a game reflects state only through the PRIOR
    completed week, so WR-A/WR-B are not yet "experienced" for the week-2
    game's own snapshot).
    """

    rosters = pd.concat(
        [
            _rosters(),
            pd.DataFrame(
                [
                    {
                        "season": 2022,
                        "team": "A",
                        "position": "WR",
                        "status": "ACT",
                        "full_name": "WR2 A",
                        "gsis_id": "WR2-A",
                        "pfr_id": "PFR-WR2-A",
                        "years_exp": 1,
                        "week": 1,
                        "game_type": "REG",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    snap_rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, opponent, pfr_id, position, offense_snaps, offense_pct in (
            ("A", "B", "PFR-A", "QB", 60, 1.0),
            ("A", "B", "PFR-WR-A", "WR", 150, 1.0),
            ("B", "A", "PFR-B", "QB", 60, 1.0),
            ("B", "A", "PFR-WR-B", "WR", 150, 1.0),
        ):
            snap_rows.append(
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
                    "offense_snaps": offense_snaps,
                    "offense_pct": offense_pct,
                    "defense_snaps": 0,
                    "defense_pct": 0.0,
                    "st_snaps": 0,
                    "st_pct": 0.0,
                }
            )
    snap_rows.append(
        {
            "game_id": "2022_01_B_A",
            "season": 2022,
            "game_type": "REG",
            "week": 1,
            "player": "WR2 A",
            "pfr_player_id": "PFR-WR2-A",
            "position": "WR",
            "team": "A",
            "opponent": "B",
            "offense_snaps": 3,
            "offense_pct": 0.05,
            "defense_snaps": 0,
            "defense_pct": 0.0,
            "st_snaps": 0,
            "st_pct": 0.0,
        }
    )
    snaps = pd.DataFrame(snap_rows)

    stats_rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, player_id in (("A", "WR-A"), ("B", "WR-B")):
            stats_rows.append(
                {
                    "player_id": player_id,
                    "season": 2022,
                    "week": week,
                    "season_type": "REG",
                    "game_id": f"2022_{week:02d}_B_A",
                    "team": team,
                    "position": "WR",
                    "rushing_epa": 0.0,
                    "receiving_epa": 2.0,
                    "def_tackles_for_loss": 0.0,
                    "def_fumbles_forced": 0.0,
                    "def_sacks": 0.0,
                    "def_qb_hits": 0.0,
                    "def_interceptions": 0.0,
                    "def_pass_defended": 0.0,
                }
            )
    stats_rows.append(
        {
            "player_id": "WR2-A",
            "season": 2022,
            "week": 1,
            "season_type": "REG",
            "game_id": "2022_01_B_A",
            "team": "A",
            "position": "WR",
            "rushing_epa": 0.0,
            "receiving_epa": 0.0,
            "def_tackles_for_loss": 0.0,
            "def_fumbles_forced": 0.0,
            "def_sacks": 0.0,
            "def_qb_hits": 0.0,
            "def_interceptions": 0.0,
            "def_pass_defended": 0.0,
        }
    )
    player_stats = pd.DataFrame(stats_rows)

    injuries = pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "game_type": ["REG", "REG", "REG"],
            "team": ["A", "A", "A"],
            "week": [2, 3, 4],
            "gsis_id": ["WR2-A", "WR2-A", "WR2-A"],
            "position": ["WR", "WR", "WR"],
            "report_status": ["Out", "Out", "Out"],
            "practice_status": [
                "Did Not Participate",
                "Did Not Participate",
                "Did Not Participate",
            ],
            "date_modified": [
                "2022-09-16T12:00:00Z",
                "2022-09-23T12:00:00Z",
                "2022-09-30T12:00:00Z",
            ],
        }
    )
    return rosters, snaps, player_stats, injuries


def test_position_prior_shrinkage_falls_back_to_zero_below_pool_minimum_and_differs_above_it() -> (
    None
):
    rosters, snaps, player_stats, injuries = _thin_player_fixture()

    zero_target = enrich_with_player_features(
        _games(),
        injuries,
        rosters,
        snaps,
        _pbp(),
        player_stats,
        qb_min_dropbacks=1,
        value_prior_snaps=200.0,
    )
    # WR2-A's own career_offense_snaps (3) is thin and their only recorded
    # receiving_epa is 0.0, so the shrink-to-zero target's contribution is
    # exactly zero at week 3 (index 2).
    assert zero_target.loc[2, "home_injury_skill_epa_value_lost"] == pytest.approx(0.0)

    # With the pool minimum set above the two-player experienced pool that
    # exists by week 3, position_prior must fall back to the same 0.0 target
    # -- bit-identical to shrink-to-zero on this row.
    prior_below_minimum = enrich_with_player_features(
        _games(),
        injuries,
        rosters,
        snaps,
        _pbp(),
        player_stats,
        qb_min_dropbacks=1,
        value_prior_snaps=200.0,
        value_shrinkage_target="position_prior",
        value_js_prior_pool_minimum=5,
    )
    assert prior_below_minimum.loc[2, "home_injury_skill_epa_value_lost"] == pytest.approx(
        zero_target.loc[2, "home_injury_skill_epa_value_lost"]
    )

    # With the pool minimum small enough to admit WR-A and WR-B (both
    # "experienced" -- career_offense_snaps=300 >= 200 -- by the week-3
    # snapshot, built from state through week 2), the thin, never-productive
    # WR2-A should be shrunk toward that positive pool prior instead of zero:
    # strictly positive, and different from the shrink-to-zero reading.
    prior_above_minimum = enrich_with_player_features(
        _games(),
        injuries,
        rosters,
        snaps,
        _pbp(),
        player_stats,
        qb_min_dropbacks=1,
        value_prior_snaps=200.0,
        value_shrinkage_target="position_prior",
        value_js_prior_pool_minimum=2,
    )
    assert prior_above_minimum.loc[2, "home_injury_skill_epa_value_lost"] > 0.0
    assert prior_above_minimum.loc[2, "home_injury_skill_epa_value_lost"] != pytest.approx(
        zero_target.loc[2, "home_injury_skill_epa_value_lost"]
    )
    # The week-2 injury (index 1) sees an EMPTY pool -- WR-A/WR-B only clear
    # career_offense_snaps=150 by their own week-1 snapshot, still below the
    # prior_snaps=200 experienced threshold -- so it must fall back to 0.0
    # exactly like the shrink-to-zero target, even with a lenient pool
    # minimum.
    assert prior_above_minimum.loc[1, "home_injury_skill_epa_value_lost"] == pytest.approx(
        zero_target.loc[1, "home_injury_skill_epa_value_lost"]
    )


def test_position_prior_shrinkage_uses_only_prior_game_stats() -> None:
    """Leakage regression test for MOD-06's new opt-in path (AGENTS.md).

    The channel prior is recomputed fresh at every game from
    ``player_value_states`` -- this proves it only ever reflects state
    strictly before the game being predicted, mirroring
    ``test_player_value_uses_only_prior_game_stats`` above: modify a week's
    own production, confirm that SAME week's already-computed feature is
    unaffected (the prior snapshot for a game is taken before that game's
    own updates are folded in), then confirm the NEXT week's feature does
    move (proving the modification was actually visible to the pipeline
    and this isn't a vacuous no-op check).
    """

    rosters, snaps, player_stats, injuries = _thin_player_fixture()
    kwargs = {
        "qb_min_dropbacks": 1,
        "value_prior_snaps": 200.0,
        "value_shrinkage_target": "position_prior",
        "value_js_prior_pool_minimum": 2,
    }

    baseline = enrich_with_player_features(
        _games(), injuries, rosters, snaps, _pbp(), player_stats, **kwargs
    )
    assert baseline.loc[2, "home_injury_skill_epa_value_lost"] > 0.0

    changed_stats = player_stats.copy()
    # Week 3 is the SAME week as the checked injury row (index 2): the
    # channel prior consumed there was already snapshotted from state
    # through week 2, strictly before this update is applied.
    changed_stats.loc[
        changed_stats["game_id"].eq("2022_03_B_A") & changed_stats["player_id"].eq("WR-A"),
        "receiving_epa",
    ] = 1_000.0
    changed = enrich_with_player_features(
        _games(), injuries, rosters, snaps, _pbp(), changed_stats, **kwargs
    )
    assert changed.loc[2, "home_injury_skill_epa_value_lost"] == pytest.approx(
        baseline.loc[2, "home_injury_skill_epa_value_lost"]
    )
    # Week 4's (index 3) channel prior IS built from state through week 3,
    # so WR2-A's week-4 "Out" injury feature -- which reads that prior --
    # moves. This confirms the modification was actually visible to the
    # pipeline, not silently no-op'd.
    assert changed.loc[3, "home_injury_skill_epa_value_lost"] != pytest.approx(
        baseline.loc[3, "home_injury_skill_epa_value_lost"]
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
