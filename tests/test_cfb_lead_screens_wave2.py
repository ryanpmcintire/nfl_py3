"""Construction, sign-convention, restriction and leakage tests for the CFB
free-screen wave 2 (LEAD-47, LEAD-49; ``docs/cfb_lead_screens_wave2.md``).

Per AGENTS.md's "add a leakage regression test for every new feature family"
rule: every candidate column here is proved to be a pure function of pregame
schedule/identity/roster/portal facts by shuffling the outcome columns
(``result``, ``ats_margin``, ``home_cover``, ``home_points``,
``away_points``) and asserting the flag is bit-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cfb_lead_screens_wave2 import (
    CANDIDATE_COLUMNS,
    attach_portal_qb_early_flag,
    attach_previous_game_starter,
    attach_true_freshman_road_qb_flag,
    build_season_game_index,
    leading_passer_per_game_team,
    match_portal_qbs_to_athletes,
    resolve_team_name_map,
    starter_agreement_rate,
)

FRESHMAN_COLUMN = CANDIDATE_COLUMNS["true_freshman_road_qb"]
PORTAL_COLUMN = CANDIDATE_COLUMNS["portal_qb_early"]

_OUTCOME_COLUMNS = ("result", "ats_margin", "home_cover", "home_points", "away_points")


def _with_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    n = len(frame)
    frame["result"] = np.arange(n, dtype=float) - 3.0
    frame["ats_margin"] = np.arange(n, dtype=float) + 1.5
    frame["home_cover"] = np.where(np.arange(n) % 2 == 0, 1.0, 0.0)
    frame["home_points"] = np.arange(n, dtype=float) + 20.0
    frame["away_points"] = np.arange(n, dtype=float) + 17.0
    return frame


def _shuffle_outcomes(frame: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    frame = frame.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    for column in _OUTCOME_COLUMNS:
        frame[column] = frame[column].to_numpy()[order]
    return frame


def _pbp_rows(
    *,
    game_id: int,
    season: int,
    week: int,
    team_id: int,
    passer_id: str,
    home_team_id: int,
    away_team_id: int,
    is_home: bool,
    n: int = 6,
) -> pd.DataFrame:
    """``n`` competitive pass plays for one (game, team, passer)."""

    return pd.DataFrame(
        {
            "game_id": [game_id] * n,
            "season": [season] * n,
            "week": [week] * n,
            "seasonType": [2] * n,
            "pos_team_id": [team_id] * n,
            "homeTeamId": [home_team_id] * n,
            "awayTeamId": [away_team_id] * n,
            "is_home": [is_home] * n,
            "EPA": [0.1] * n,
            "EPA_success": [1] * n,
            "rush": [False] * n,
            "pass": [True] * n,
            "kneel_down": [False] * n,
            "statYardage": [5.0] * n,
            "home_wp_before": [0.5] * n,
            "away_wp_before": [0.5] * n,
            "passer_player_id": [passer_id] * n,
        }
    )


def _schedule_row(
    *,
    game_id: int,
    season: int,
    home_id: int,
    away_id: int,
    home_team: str,
    away_team: str,
    start_date: str,
    season_type: str = "regular",
    completed: bool = True,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": season,
        "home_id": home_id,
        "away_id": away_id,
        "home_team": home_team,
        "away_team": away_team,
        "start_date": start_date,
        "season_type": season_type,
        "completed": completed,
    }


# ---------------------------------------------------------------------------
# Shared starter-identification helpers
# ---------------------------------------------------------------------------


def test_leading_passer_per_game_team_picks_max_dropbacks_and_drops_thin_rows() -> None:
    pbp = pd.concat(
        [
            _pbp_rows(
                game_id=1,
                season=2024,
                week=1,
                team_id=10,
                passer_id="501",
                home_team_id=10,
                away_team_id=20,
                is_home=True,
                n=6,
            ),
            _pbp_rows(
                game_id=1,
                season=2024,
                week=1,
                team_id=10,
                passer_id="502",
                home_team_id=10,
                away_team_id=20,
                is_home=True,
                n=3,  # below the 5-dropback floor: dropped entirely
            ),
        ],
        ignore_index=True,
    )
    leading = leading_passer_per_game_team(pbp)
    assert len(leading) == 1
    assert leading.iloc[0]["passer_player_id"] == "501"


def test_leading_passer_tie_break_is_deterministic_lowest_id() -> None:
    pbp = pd.concat(
        [
            _pbp_rows(
                game_id=2,
                season=2024,
                week=1,
                team_id=10,
                passer_id="999",
                home_team_id=10,
                away_team_id=20,
                is_home=True,
                n=6,
            ),
            _pbp_rows(
                game_id=2,
                season=2024,
                week=1,
                team_id=10,
                passer_id="100",
                home_team_id=10,
                away_team_id=20,
                is_home=True,
                n=6,
            ),
        ],
        ignore_index=True,
    )
    leading = leading_passer_per_game_team(pbp)
    assert len(leading) == 1
    assert leading.iloc[0]["passer_player_id"] == "100"


def test_attach_previous_game_starter_is_strictly_earlier_and_carries_forward() -> None:
    pbp = pd.concat(
        [
            _pbp_rows(
                game_id=101,
                season=2024,
                week=1,
                team_id=10,
                passer_id="701",
                home_team_id=10,
                away_team_id=20,
                is_home=True,
            ),
            _pbp_rows(
                game_id=102,
                season=2024,
                week=2,
                team_id=10,
                passer_id="701",
                home_team_id=30,
                away_team_id=10,
                is_home=False,
            ),
            _pbp_rows(
                game_id=103,
                season=2024,
                week=3,
                team_id=10,
                passer_id="702",
                home_team_id=10,
                away_team_id=40,
                is_home=True,
            ),
        ],
        ignore_index=True,
    )
    schedules = pd.DataFrame(
        [
            _schedule_row(
                game_id=101,
                season=2024,
                home_id=10,
                away_id=20,
                home_team="T10",
                away_team="T20",
                start_date="2024-09-07",
            ),
            _schedule_row(
                game_id=102,
                season=2024,
                home_id=30,
                away_id=10,
                home_team="T30",
                away_team="T10",
                start_date="2024-09-14",
            ),
            _schedule_row(
                game_id=103,
                season=2024,
                home_id=10,
                away_id=40,
                home_team="T10",
                away_team="T40",
                start_date="2024-09-21",
            ),
        ]
    )
    leading = leading_passer_per_game_team(pbp)
    walked = attach_previous_game_starter(leading, schedules).set_index("game_id")
    assert pd.isna(walked.loc["101", "prev_game_starter_id"])  # no earlier known game
    assert walked.loc["102", "prev_game_starter_id"] == "701"  # continuity
    assert walked.loc["103", "prev_game_starter_id"] == "701"  # still P1, before P2 started

    agreement = starter_agreement_rate(walked.reset_index())
    # game 102 (prev P1, actual P1) agrees; game 103 (prev P1, actual P2) disagrees;
    # game 101 has no comparable previous starter and is excluded.
    assert agreement["n_comparable"] == 2
    assert agreement["agreement_rate"] == 0.5


# ---------------------------------------------------------------------------
# LEAD-47: true_freshman_road_qb
# ---------------------------------------------------------------------------


def _lead47_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {
            "game_id": [9001, 9002, 9003, 9004],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "gameday": pd.to_datetime(["2024-09-07", "2024-09-14", "2024-09-21", "2024-09-28"]),
            "home_id": [10, 10, 40, 10],
            "away_id": [20, 30, 20, 60],
        }
    )
    features = _with_outcome_columns(features)

    pbp = pd.concat(
        [
            # G1 9001: away team 20 starter F1 -> true freshman (roster: 2024 only)
            _pbp_rows(
                game_id=9001,
                season=2024,
                week=1,
                team_id=20,
                passer_id="F1",
                home_team_id=10,
                away_team_id=20,
                is_home=False,
            ),
            # G2 9002: away team 30 starter F2 -> NOT true freshman (roster: 2023 & 2024)
            _pbp_rows(
                game_id=9002,
                season=2024,
                week=2,
                team_id=30,
                passer_id="F2",
                home_team_id=10,
                away_team_id=30,
                is_home=False,
            ),
            # G3 9003: home team 40 starter F3 -> true freshman but HOME side (diagnostic only)
            _pbp_rows(
                game_id=9003,
                season=2024,
                week=3,
                team_id=40,
                passer_id="F3",
                home_team_id=40,
                away_team_id=20,
                is_home=True,
            ),
            # G3 9003: away team 20 starter V1 -> NOT true freshman (roster: 2022 & 2024)
            _pbp_rows(
                game_id=9003,
                season=2024,
                week=3,
                team_id=20,
                passer_id="V1",
                home_team_id=40,
                away_team_id=20,
                is_home=False,
            ),
            # G4 9004: away team 60 has NO pbp coverage at all (unidentified starter)
        ],
        ignore_index=True,
    )

    rosters = pd.DataFrame(
        {
            "athlete_id": [1, 1, 2, 2, 3, 4, 4],
            "season": [2024, 2024, 2023, 2024, 2024, 2022, 2024],
        }
    )
    # Map passer string ids to athlete_ids used above: F1->1, F2->2, F3->3, V1->4
    id_map = {"F1": "1", "F2": "2", "F3": "3", "V1": "4"}
    pbp["passer_player_id"] = pbp["passer_player_id"].map(id_map).fillna(pbp["passer_player_id"])
    return features, pbp, rosters


def test_true_freshman_road_qb_flag_hand_computed_cases() -> None:
    features, pbp, rosters = _lead47_fixture()
    attached, diagnostics = attach_true_freshman_road_qb_flag(features, pbp=pbp, rosters=rosters)
    attached = attached.set_index("game_id")

    assert attached.loc[9001, FRESHMAN_COLUMN] == 1.0  # away true freshman F1
    assert attached.loc[9002, FRESHMAN_COLUMN] == 0.0  # away starter F2 not a true freshman
    assert attached.loc[9003, FRESHMAN_COLUMN] == 0.0  # away starter V1 not true freshman
    assert attached.loc[9003, "_lead47_home_true_freshman_starter"]  # home F3 counted, not pooled
    assert attached.loc[9004, FRESHMAN_COLUMN] == 0.0  # unidentified away starter -> 0, not error

    assert diagnostics["away_true_freshman_starter_count"] == 1
    assert diagnostics["home_true_freshman_starter_count"] == 1
    assert diagnostics["away_starter_identified"] == 3  # 9001, 9002, 9003 (not 9004)


def test_true_freshman_road_qb_flag_never_takes_a_third_value() -> None:
    features, pbp, rosters = _lead47_fixture()
    attached, _ = attach_true_freshman_road_qb_flag(features, pbp=pbp, rosters=rosters)
    assert set(attached[FRESHMAN_COLUMN].unique()).issubset({0.0, 1.0})
    assert attached[FRESHMAN_COLUMN].notna().all()


def test_true_freshman_road_qb_flag_is_pregame_safe_under_outcome_permutation() -> None:
    features, pbp, rosters = _lead47_fixture()
    before, _ = attach_true_freshman_road_qb_flag(features, pbp=pbp, rosters=rosters)
    shuffled_features = _shuffle_outcomes(features)
    after, _ = attach_true_freshman_road_qb_flag(shuffled_features, pbp=pbp, rosters=rosters)
    np.testing.assert_array_equal(
        before[FRESHMAN_COLUMN].to_numpy(), after[FRESHMAN_COLUMN].to_numpy()
    )


# ---------------------------------------------------------------------------
# LEAD-49: portal_qb_early
# ---------------------------------------------------------------------------


def test_resolve_team_name_map_matches_home_and_away_columns() -> None:
    schedules = pd.DataFrame(
        [
            _schedule_row(
                game_id=1,
                season=2024,
                home_id=10,
                away_id=20,
                home_team="Alpha",
                away_team="Beta",
                start_date="2024-09-07",
            )
        ]
    )
    names = resolve_team_name_map(schedules)
    assert names["Alpha"] == 10
    assert names["Beta"] == 20


def test_match_portal_qbs_to_athletes_disclosed_join_stages() -> None:
    schedules = pd.DataFrame(
        [
            _schedule_row(
                game_id=1,
                season=2022,
                home_id=700,
                away_id=999,
                home_team="PortalHome",
                away_team="Opp999",
                start_date="2022-09-01",
            )
        ]
    )
    rosters = pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "team_id": [700, 700, 700],
            "athlete_id": [8001, 8002, 8003],
            "first_name": ["Sam", "Sam", "Dup"],
            "last_name": ["Rivers", "Other", "Licate"],
        }
    )
    # Add a genuine ambiguous key: two athletes sharing (team, season, name).
    rosters = pd.concat(
        [
            rosters,
            pd.DataFrame(
                {
                    "season": [2022],
                    "team_id": [700],
                    "athlete_id": [8004],
                    "first_name": ["Dup"],
                    "last_name": ["Licate"],
                }
            ),
        ],
        ignore_index=True,
    )
    portal = pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "firstName": ["Sam", "Nobody", "Dup"],
            "lastName": ["Rivers", "Home", "Licate"],
            "position": ["QB", "QB", "QB"],
            "origin": ["Old", "Old", "Old"],
            # row 0: resolves cleanly; row 1: destination doesn't exist in schedules;
            # row 2: name is ambiguous on the roster (ties to two athlete_ids)
            "destination": ["PortalHome", "NowhereState", "PortalHome"],
        }
    )
    portal_qb_ids, diagnostics = match_portal_qbs_to_athletes(portal, rosters, schedules)
    assert portal_qb_ids == {(700, 2022, "8001")}
    assert diagnostics["portal_qb_rows"] == 3
    assert diagnostics["unresolved_destination_rows"] == 1
    assert diagnostics["matched_portal_qb_rows"] == 1
    assert diagnostics["ambiguous_roster_name_keys"] == 1
    assert diagnostics["unmatched_name_rows"] == 1  # the ambiguous "Dup Licate" row


def test_build_season_game_index_orders_chronologically_regular_completed_only() -> None:
    schedules = pd.DataFrame(
        [
            _schedule_row(
                game_id=1,
                season=2022,
                home_id=200,
                away_id=999,
                home_team="T200",
                away_team="Opp999",
                start_date="2022-09-01",
            ),
            _schedule_row(
                game_id=2,
                season=2022,
                home_id=10,
                away_id=200,
                home_team="T10",
                away_team="T200",
                start_date="2022-09-08",
            ),
            # postseason row for the same team: must NOT count toward the index
            _schedule_row(
                game_id=3,
                season=2022,
                home_id=200,
                away_id=998,
                home_team="T200",
                away_team="Opp998",
                start_date="2022-09-15",
                season_type="postseason",
            ),
            _schedule_row(
                game_id=4,
                season=2022,
                home_id=200,
                away_id=997,
                home_team="T200",
                away_team="Opp997",
                start_date="2022-09-22",
            ),
        ]
    )
    index = build_season_game_index(schedules)
    assert index[(200, 2022, "1")] == 1
    assert index[(200, 2022, "2")] == 2
    assert (200, 2022, "3") not in index  # postseason excluded
    assert index[(200, 2022, "4")] == 3  # third REGULAR game, not fourth


def _lead49_fixture() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Team 200 ("TransferTeam") starts portal QB 9001 across its first four
    2022 games; games 9102 (away, index 2) and 9103 (home, index 3) are the
    scored rows, plus 9101 (index 1, structurally can't flag) and 9104
    (index 4, outside "first three")."""

    schedules = pd.DataFrame(
        [
            _schedule_row(
                game_id=9101,
                season=2022,
                home_id=200,
                away_id=500,
                home_team="TransferTeam",
                away_team="Opp500",
                start_date="2022-09-01",
            ),
            _schedule_row(
                game_id=9102,
                season=2022,
                home_id=10,
                away_id=200,
                home_team="Home10",
                away_team="TransferTeam",
                start_date="2022-09-08",
            ),
            _schedule_row(
                game_id=9103,
                season=2022,
                home_id=200,
                away_id=20,
                home_team="TransferTeam",
                away_team="Away20",
                start_date="2022-09-15",
            ),
            _schedule_row(
                game_id=9104,
                season=2022,
                home_id=30,
                away_id=200,
                home_team="Home30",
                away_team="TransferTeam",
                start_date="2022-09-22",
            ),
        ]
    )
    features = pd.DataFrame(
        {
            "game_id": [9101, 9102, 9103, 9104],
            "season": [2022, 2022, 2022, 2022],
            "week": [1, 2, 3, 4],
            "gameday": pd.to_datetime(["2022-09-01", "2022-09-08", "2022-09-15", "2022-09-22"]),
            "home_id": [200, 10, 200, 30],
            "away_id": [500, 200, 20, 200],
        }
    )
    features = _with_outcome_columns(features)

    pbp = pd.concat(
        [
            _pbp_rows(
                game_id=9101,
                season=2022,
                week=1,
                team_id=200,
                passer_id="9001",
                home_team_id=200,
                away_team_id=500,
                is_home=True,
            ),
            _pbp_rows(
                game_id=9102,
                season=2022,
                week=2,
                team_id=200,
                passer_id="9001",
                home_team_id=10,
                away_team_id=200,
                is_home=False,
            ),
            _pbp_rows(
                game_id=9103,
                season=2022,
                week=3,
                team_id=200,
                passer_id="9001",
                home_team_id=200,
                away_team_id=20,
                is_home=True,
            ),
            _pbp_rows(
                game_id=9104,
                season=2022,
                week=4,
                team_id=200,
                passer_id="9001",
                home_team_id=30,
                away_team_id=200,
                is_home=False,
            ),
        ],
        ignore_index=True,
    )

    rosters = pd.DataFrame(
        {
            "season": [2022],
            "team_id": [200],
            "athlete_id": [9001],
            "first_name": ["Jordan"],
            "last_name": ["Quinn"],
        }
    )
    portal = pd.DataFrame(
        {
            "season": [2022],
            "firstName": ["Jordan"],
            "lastName": ["Quinn"],
            "position": ["QB"],
            "origin": ["OldSchool"],
            "destination": ["TransferTeam"],
        }
    )
    return features, pbp, portal, rosters, schedules


def test_portal_qb_early_signed_hand_computed_cases() -> None:
    features, pbp, portal, rosters, schedules = _lead49_fixture()
    attached, diagnostics = attach_portal_qb_early_flag(
        features, pbp=pbp, portal=portal, rosters=rosters, schedules=schedules
    )
    attached = attached.set_index("game_id")

    # Game index 1: the portal QB cannot yet be "the previous game's starter"
    # for his own new team -- structurally can never flag, even though he IS
    # this game's own (post-hoc) starter. Disclosed in the predeclaration doc.
    assert attached.loc[9101, PORTAL_COLUMN] == 0.0
    # Game index 2, team 200 AWAY: prior game's starter (9001) is the portal
    # QB -> away fires -> signed +1 (favours home).
    assert attached.loc[9102, PORTAL_COLUMN] == 1.0
    # Game index 3, team 200 HOME: mirror -> signed -1.
    assert attached.loc[9103, PORTAL_COLUMN] == -1.0
    # Game index 4: outside the first-three-games window -> 0.
    assert attached.loc[9104, PORTAL_COLUMN] == 0.0

    assert diagnostics["matched_portal_qb_rows"] == 1
    assert diagnostics["away_flagged_games"] == 1
    assert diagnostics["home_flagged_games"] == 1
    assert diagnostics["both_fire_games"] == 0


def test_portal_qb_early_signed_zero_when_both_sides_fire() -> None:
    schedules = pd.DataFrame(
        [
            _schedule_row(
                game_id=1,
                season=2022,
                home_id=700,
                away_id=999,
                home_team="PortalHome",
                away_team="Opp999",
                start_date="2022-09-01",
            ),
            _schedule_row(
                game_id=2,
                season=2022,
                home_id=998,
                away_id=800,
                home_team="Opp998",
                away_team="PortalAway",
                start_date="2022-09-03",
            ),
            _schedule_row(
                game_id=3,
                season=2022,
                home_id=700,
                away_id=800,
                home_team="PortalHome",
                away_team="PortalAway",
                start_date="2022-09-10",
            ),
        ]
    )
    features = pd.DataFrame(
        {
            "game_id": [3],
            "season": [2022],
            "week": [2],
            "gameday": pd.to_datetime(["2022-09-10"]),
            "home_id": [700],
            "away_id": [800],
        }
    )
    features = _with_outcome_columns(features)
    pbp = pd.concat(
        [
            _pbp_rows(
                game_id=1,
                season=2022,
                week=1,
                team_id=700,
                passer_id="8001",
                home_team_id=700,
                away_team_id=999,
                is_home=True,
            ),
            _pbp_rows(
                game_id=2,
                season=2022,
                week=1,
                team_id=800,
                passer_id="8002",
                home_team_id=998,
                away_team_id=800,
                is_home=False,
            ),
        ],
        ignore_index=True,
    )
    rosters = pd.DataFrame(
        {
            "season": [2022, 2022],
            "team_id": [700, 800],
            "athlete_id": [8001, 8002],
            "first_name": ["Sam", "Casey"],
            "last_name": ["Rivers", "Bloom"],
        }
    )
    portal = pd.DataFrame(
        {
            "season": [2022, 2022],
            "firstName": ["Sam", "Casey"],
            "lastName": ["Rivers", "Bloom"],
            "position": ["QB", "QB"],
            "origin": ["Old1", "Old2"],
            "destination": ["PortalHome", "PortalAway"],
        }
    )
    attached, diagnostics = attach_portal_qb_early_flag(
        features, pbp=pbp, portal=portal, rosters=rosters, schedules=schedules
    )
    assert attached.set_index("game_id").loc[3, PORTAL_COLUMN] == 0.0
    assert diagnostics["both_fire_games"] == 1


def test_portal_qb_early_signed_never_outside_expected_set() -> None:
    features, pbp, portal, rosters, schedules = _lead49_fixture()
    attached, _ = attach_portal_qb_early_flag(
        features, pbp=pbp, portal=portal, rosters=rosters, schedules=schedules
    )
    assert set(attached[PORTAL_COLUMN].unique()).issubset({-1.0, 0.0, 1.0})
    assert attached[PORTAL_COLUMN].notna().all()


def test_portal_qb_early_signed_is_pregame_safe_under_outcome_permutation() -> None:
    features, pbp, portal, rosters, schedules = _lead49_fixture()
    before, _ = attach_portal_qb_early_flag(
        features, pbp=pbp, portal=portal, rosters=rosters, schedules=schedules
    )
    shuffled_features = _shuffle_outcomes(features)
    after, _ = attach_portal_qb_early_flag(
        shuffled_features, pbp=pbp, portal=portal, rosters=rosters, schedules=schedules
    )
    np.testing.assert_array_equal(before[PORTAL_COLUMN].to_numpy(), after[PORTAL_COLUMN].to_numpy())
