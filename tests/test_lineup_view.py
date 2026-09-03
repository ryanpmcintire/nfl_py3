import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _lineup_payload(team: str) -> dict:
    return {
        "team": team,
        "players": [{"name": "Q B", "position": "QB", "gsis_id": "qb-1"}],
        "games": {},
    }


def _write_artifact(root: Path, team: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    games = {"G": {"home": _lineup_payload(team), "away": _lineup_payload(team)}}
    (root / "lineups.json").write_text(json.dumps({"games": games}), encoding="utf-8")


def test_loader_prefers_stable_path_over_legacy_stamped_runs(tmp_path: Path) -> None:
    from nfl_ats.lineup_view import STABLE_LINEUP_PATH

    _write_artifact(tmp_path / "lineups" / "2026-week-1-20200101T000000Z", "STALE")
    _write_artifact(tmp_path / STABLE_LINEUP_PATH.parent, "CURRENT")
    lineups = load_lineups(tmp_path)
    assert lineups["G"][0].team == "CURRENT"


def test_loader_falls_back_to_legacy_stamped_run_without_stable_path(
    tmp_path: Path,
) -> None:
    _write_artifact(tmp_path / "lineups" / "2026-week-1-20200101T000000Z", "STALE")
    lineups = load_lineups(tmp_path)
    assert lineups["G"][0].team == "STALE"


def test_oversized_lineup_artifact_refuses_to_publish(tmp_path: Path) -> None:
    from scripts.build_week_lineups import MAX_LINEUP_BYTES, _check_artifact_size

    small = tmp_path / "lineups.json"
    small.write_bytes(b"x" * 100)
    _check_artifact_size(small)
    big = tmp_path / "big.json"
    big.write_bytes(b"x" * (MAX_LINEUP_BYTES + 1))
    with pytest.raises(SystemExit, match="Refusing to publish"):
        _check_artifact_size(big)


def test_legacy_stamped_runs_are_removed_but_live_and_foreign_dirs_survive(
    tmp_path: Path,
) -> None:
    from scripts.build_week_lineups import _remove_legacy_stamped_runs

    root = tmp_path / "lineups"
    _write_artifact(root / "2026-week-1-20200101T000000Z", "STALE")
    _write_artifact(root / "2026-week-1-20200202T000000Z", "STALE")
    _write_artifact(root / "current", "CURRENT")
    foreign = root / "notes"
    foreign.mkdir(parents=True)
    (foreign / "readme.txt").write_text("not an artifact", encoding="utf-8")
    _remove_legacy_stamped_runs(root, keep=root / "current" / "lineups.json")
    assert not (root / "2026-week-1-20200101T000000Z").exists()
    assert not (root / "2026-week-1-20200202T000000Z").exists()
    assert (root / "current" / "lineups.json").is_file()
    assert (foreign / "readme.txt").is_file()
