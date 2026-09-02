from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.coordinator_changes import build_coordinator_change_features
from nfl_ats.data import DataContractError


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_B_A",
                "season": 2026,
                "week": 1,
                "decision_at": "2026-09-01T16:00:00Z",
                "kickoff": "2026-09-05T17:00:00Z",
                "away_team": "B",
                "home_team": "A",
            },
            {
                "game_id": "2026_02_B_A",
                "season": 2026,
                "week": 2,
                "decision_at": "2026-09-08T16:00:00Z",
                "kickoff": "2026-09-12T17:00:00Z",
                "away_team": "B",
                "home_team": "A",
            },
        ]
    )


def _assignments() -> pd.DataFrame:
    rows = [
        ("A", "OC", "Old OC", "2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z"),
        ("A", "OC", "New OC", "2026-09-07T12:00:00Z", "2026-09-06T12:00:00Z"),
        ("A", "DC", "Stable A DC", "2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z"),
        ("B", "OC", "Stable B OC", "2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z"),
        ("B", "DC", "Stable B DC", "2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z"),
    ]
    return pd.DataFrame(
        [
            {
                "team": team,
                "role": role,
                "coordinator_name": name,
                "effective_at": effective,
                "observed_at": observed,
                "source_url": f"https://example.test/{team.lower()}/{role.lower()}/{name}",
            }
            for team, role, name, effective, observed in rows
        ]
    )


def test_builds_nullable_change_state_from_assignments_known_at_each_decision() -> None:
    result = build_coordinator_change_features(_games(), _assignments()).set_index("game_id")

    assert pd.isna(result.loc["2026_01_B_A", "home_oc_changed"])
    assert result.loc["2026_02_B_A", "home_oc_name"] == "New OC"
    assert result.loc["2026_02_B_A", "home_oc_changed"] == 1
    assert result.loc["2026_02_B_A", "home_dc_changed"] == 0
    assert result.loc["2026_02_B_A", "away_coordinator_change_count"] == 0
    assert result.loc["2026_02_B_A", "diff_coordinator_change_count"] == 1
    assert result.loc["2026_02_B_A", "home_oc_observed_at"] == pd.Timestamp("2026-09-06T12:00:00Z")


def test_postdecision_assignment_fails_closed_as_unknown() -> None:
    assignments = _assignments()
    assignments = assignments.loc[
        ~(
            assignments["team"].eq("B")
            & assignments["role"].eq("DC")
            & assignments["coordinator_name"].eq("Stable B DC")
        )
    ]
    assignments = pd.concat(
        [
            assignments,
            pd.DataFrame(
                [
                    {
                        "team": "B",
                        "role": "DC",
                        "coordinator_name": "Late B DC",
                        "effective_at": "2026-01-01T00:00:00Z",
                        "observed_at": "2026-09-09T00:00:00Z",
                        "source_url": "https://example.test/late",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = build_coordinator_change_features(_games(), assignments).set_index("game_id")

    assert pd.isna(result.loc["2026_02_B_A", "away_dc_changed"])
    assert pd.isna(result.loc["2026_02_B_A", "away_coordinator_change_count"])
    assert pd.isna(result.loc["2026_02_B_A", "diff_coordinator_change_count"])


def test_future_revision_results_and_games_cannot_change_prior_rows() -> None:
    baseline = build_coordinator_change_features(_games(), _assignments())
    mutated_games = _games().assign(home_score=[7, 30], away_score=[10, 0], result=[3, -30])
    mutated_games = pd.concat(
        [
            mutated_games,
            pd.DataFrame(
                [
                    {
                        "game_id": "2026_03_C_A",
                        "season": 2026,
                        "week": 3,
                        "decision_at": "2026-09-15T16:00:00Z",
                        "kickoff": "2026-09-19T17:00:00Z",
                        "away_team": "C",
                        "home_team": "A",
                        "home_score": 99,
                        "away_score": 0,
                        "result": -99,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    future_revision = pd.DataFrame(
        [
            {
                "team": "A",
                "role": "OC",
                "coordinator_name": "Corrected New OC",
                "effective_at": "2026-09-07T12:00:00Z",
                "observed_at": "2026-09-14T12:00:00Z",
                "source_url": "https://example.test/future-correction",
            }
        ]
    )
    mutated_assignments = pd.concat([_assignments(), future_revision], ignore_index=True)

    changed = build_coordinator_change_features(mutated_games, mutated_assignments)

    pd.testing.assert_frame_equal(
        changed.loc[changed["game_id"].isin(baseline["game_id"])].reset_index(drop=True),
        baseline.reset_index(drop=True),
        check_exact=True,
    )


def test_ambiguous_role_fails_closed() -> None:
    assignments = _assignments()
    assignments.loc[0, "role"] = "OC/DC"

    with pytest.raises(DataContractError, match="ambiguous or unsupported coordinator roles"):
        build_coordinator_change_features(_games(), assignments)


def test_conflicting_revision_identity_fails_closed() -> None:
    assignments = _assignments()
    conflict = assignments.iloc[[0]].assign(coordinator_name="Different OC")

    with pytest.raises(DataContractError, match="conflicting coordinator names"):
        build_coordinator_change_features(
            _games(), pd.concat([assignments, conflict], ignore_index=True)
        )


def test_decision_at_or_after_kickoff_is_rejected() -> None:
    games = _games()
    games.loc[0, "decision_at"] = games.loc[0, "kickoff"]

    with pytest.raises(DataContractError, match="strictly before kickoff"):
        build_coordinator_change_features(games, _assignments())
