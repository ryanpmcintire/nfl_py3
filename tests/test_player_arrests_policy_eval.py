from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.player_arrests_policy_eval import (
    apply_frozen_policy,
    broad_incident_game_flags,
    paired_policy_bootstrap,
)


def test_broad_flags_are_strictly_pregame_and_ignore_retrospective_fields() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "gameday": ["2024-09-22", "2024-09-22"],
            "home_team": ["JAX", "IND"],
            "away_team": ["BUF", "CHI"],
        }
    )
    incidents = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4, 5],
            "incident_date": [
                "2024-09-03",  # JAC: 14 days before Tuesday, included
                "2024-09-02",  # BUF: 15 days before Tuesday, excluded
                "2024-09-17",  # IN: same Tuesday, excluded
                "2024-09-18",  # CHI: after Tuesday, excluded
                "2024-09-10",  # IN: seven days before Tuesday, included
            ],
            "team": ["JAC", "BUF", "IN", "CHI", "IN"],
            "outcome_archive_only": ["a", "b", "c", "d", "e"],
        }
    )

    flags, coverage = broad_incident_game_flags(games, incidents)

    assert flags.to_dict("records") == [
        {"game_id": "g1", "home_incident_flag": True, "away_incident_flag": False},
        {"game_id": "g2", "home_incident_flag": True, "away_incident_flag": False},
    ]
    assert coverage == {"source_incidents": 5, "schedule_mapped_incidents": 5}

    incidents["outcome_archive_only"] = "retrospective mutation"
    mutated, _ = broad_incident_game_flags(games, incidents)
    pd.testing.assert_frame_equal(flags, mutated)


def test_frozen_policy_flips_only_when_sole_flag_opposes_production_and_preserves_pushes() -> None:
    opener = pd.DataFrame(
        {
            "game_id": ["home_flip", "away_flip", "none", "both", "already", "push"],
            "season": [2024] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "margin_vs_open": [3.0, -2.0, 1.0, -1.0, 4.0, 0.0],
            "pick_home_at_open_probability_rule": [False, True, True, False, True, False],
            "correct_at_open_probability_rule": [0.0, 0.0, 1.0, 1.0, 1.0, np.nan],
        }
    )
    flags = pd.DataFrame(
        {
            "game_id": opener["game_id"],
            "home_incident_flag": [True, False, False, True, True, True],
            "away_incident_flag": [False, True, False, True, False, False],
        }
    )

    scored = apply_frozen_policy(opener, flags)

    assert scored["policy_flip"].tolist() == [True, True, False, False, False, True]
    assert scored["candidate_pick_home"].tolist() == [True, False, True, False, True, True]
    assert scored["candidate_correct_at_open"].iloc[:5].tolist() == [1.0] * 5
    assert np.isnan(scored.loc[scored["game_id"].eq("push"), "candidate_correct_at_open"].iloc[0])


def test_paired_policy_bootstrap_is_accuracy_points_and_deterministic() -> None:
    scored = pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "week": [1, 2, 1, 2],
            "correct_at_open_probability_rule": [0.0, 1.0, 0.0, 1.0],
            "candidate_correct_at_open": [1.0, 1.0, 0.0, 1.0],
        }
    )

    first = paired_policy_bootstrap(scored, block="week", samples=100, seed=7)
    second = paired_policy_bootstrap(scored, block="week", samples=100, seed=7)

    assert first == second
    assert first["estimate"] == 25.0
    assert first["paired_games"] == 4
    assert first["blocks"] == 4
    assert 0.0 <= first["probability_positive"] <= 1.0
