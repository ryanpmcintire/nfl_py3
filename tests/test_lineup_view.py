from pathlib import Path

from nfl_ats.lineup_view import load_lineups, team_lineup


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
