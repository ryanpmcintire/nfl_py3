from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.lineup_view import load_lineups, team_lineup, validate_lineup_model_sync


def test_lineup_loader_fails_open_when_artifact_is_absent(tmp_path: Path) -> None:
    assert load_lineups(tmp_path) == {}


def test_team_lineup_keeps_qb_mismatch_visible_and_does_not_invent_probability() -> None:
    lineup = team_lineup(
        {
            "team": "LV",
            "as_of": "2026-09-03T11:53:47Z",
            "source": "nflverse depth charts",
            "injury_status": "unavailable",
            "note": "Current depth chart QB differs from forecast input",
            "players": [
                {
                    "name": "Kirk Cousins",
                    "position": "QB",
                    "slot": "QB",
                    "gsis_id": "00-0029604",
                    "model_role": "context_only",
                }
            ],
        }
    )
    updated = lineup.with_model_impact(family_points=1.2, model_qb_id="00-0038579")
    assert updated.players[0].name == "Kirk Cousins"
    assert updated.players[0].play_probability is None
    assert updated.players[0].model_impact_points is None
    assert updated.note is not None


def test_model_sync_guard_refuses_a_different_current_qb() -> None:
    lineup = team_lineup(
        {
            "team": "LV",
            "players": [{"name": "Kirk Cousins", "position": "QB", "gsis_id": "cousins"}],
        }
    )
    with pytest.raises(ValueError, match="Lineup/model mismatch"):
        validate_lineup_model_sync(
            {"G": (lineup, lineup)},
            pd.DataFrame(
                [
                    {
                        "game_id": "G",
                        "home_projected_qb_id": "old-qb",
                        "away_projected_qb_id": "cousins",
                    }
                ]
            ),
        )


def test_model_sync_guard_skips_historical_rows_without_a_lineup_entry() -> None:
    lineup = team_lineup(
        {
            "team": "LV",
            "players": [{"name": "Kirk Cousins", "position": "QB", "gsis_id": "cousins"}],
        }
    )
    # The predictions artifact also carries historical rows that the current
    # lineup artifact cannot cover; they must not trip the fail-closed guard.
    validate_lineup_model_sync(
        {"G": (lineup, lineup)},
        pd.DataFrame(
            [
                {
                    "game_id": "G",
                    "home_projected_qb_id": "cousins",
                    "away_projected_qb_id": "cousins",
                },
                {
                    "game_id": "HISTORICAL",
                    "home_projected_qb_id": "old-qb",
                    "away_projected_qb_id": "old-qb",
                },
            ]
        ),
    )


def test_lineup_player_defaults_to_offense_unit() -> None:
    lineup = team_lineup(
        {
            "team": "LV",
            "players": [{"name": "Kirk Cousins", "position": "QB", "gsis_id": "cousins"}],
        }
    )
    assert lineup.players[0].unit == "offense"
