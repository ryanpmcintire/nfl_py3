from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from _board_content_fixtures import build_fixture_content

from nfl_ats import board_assistant_lineups as lineups_module
from nfl_ats.board_assistant import answer, build_knowledge_for_board
from nfl_ats.lineup_view import STABLE_LINEUP_PATH, load_lineups

_ANCHOR_RE = re.compile(r"as of [^ ]+ from [\w .'-]+")


def _write_lineups_artifact(tmp_path: Path) -> None:
    payload = {
        "season": 2026,
        "week": 1,
        "generated_at": "20260831T110000Z",
        "games": {
            "2026_01_MIA_LV": {
                "home": {
                    "team": "LV",
                    "as_of": "2026-08-31T11:00:00Z",
                    "source": "nflverse depth charts",
                    "injury_status": "unavailable — current injury feed not attached",
                    "note": (
                        "Current depth chart QB differs from forecast input; rerun forecast "
                        "before treating this as a model update."
                    ),
                    "players": [
                        {
                            "name": "Aidan O'Connell",
                            "position": "QB",
                            "slot": "QB1",
                            "depth": 1,
                            "unit": "offense",
                            "gsis_id": "lv-oconnell",
                            "model_role": "context_only",
                        },
                        {
                            "name": "Geno Smith",
                            "position": "QB",
                            "slot": "QB2",
                            "depth": 2,
                            "unit": "offense",
                            "gsis_id": "lv-smith",
                            "model_role": "base_model",
                        },
                    ],
                },
                "away": {
                    "team": "MIA",
                    "as_of": "2026-08-31T11:00:00Z",
                    "source": "nflverse depth charts",
                    "injury_status": "unavailable — current injury feed not attached",
                    "note": None,
                    "players": [
                        {
                            "name": "Tua Tagovailoa",
                            "position": "QB",
                            "slot": "QB1",
                            "depth": 1,
                            "unit": "offense",
                            "gsis_id": "mia-tua",
                            "model_role": "base_model",
                            "play_probability": 0.92,
                            "injury_status": "questionable",
                            "has_injury_designation": True,
                        },
                        {
                            "name": "Tyreek Hill",
                            "position": "WR",
                            "slot": "WR1",
                            "depth": 1,
                            "unit": "offense",
                            "gsis_id": "mia-hill",
                            "model_role": "context_only",
                            "play_probability": 0.97,
                            "injury_status": "questionable",
                            "has_injury_designation": True,
                        },
                        {
                            "name": "Malik Washington",
                            "position": "WR",
                            "slot": "WR3",
                            "depth": 3,
                            "unit": "offense",
                            "gsis_id": "mia-washington",
                            "model_role": "context_only",
                            "play_probability": 0.15,
                        },
                    ],
                },
            },
            "2026_01_DEN_KC": {
                "home": {
                    "team": "KC",
                    "as_of": "2020-01-01T00:00:00Z",
                    "source": "nflverse depth charts",
                    "injury_status": "unavailable — current injury feed not attached",
                    "note": None,
                    "players": [
                        {
                            "name": "Patrick Mahomes",
                            "position": "QB",
                            "slot": "QB1",
                            "depth": 1,
                            "unit": "offense",
                            "gsis_id": "kc-mahomes",
                            "model_role": "base_model",
                        },
                    ],
                },
                "away": {
                    "team": "DEN",
                    "as_of": "2020-01-01T00:00:00Z",
                    "source": "nflverse depth charts",
                    "injury_status": "unavailable — current injury feed not attached",
                    "note": None,
                    "players": [
                        {
                            "name": "Bo Nix",
                            "position": "QB",
                            "slot": "QB1",
                            "depth": 1,
                            "unit": "offense",
                            "gsis_id": "den-nix",
                            "model_role": "base_model",
                        },
                    ],
                },
            },
        },
    }
    target = tmp_path / STABLE_LINEUP_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _content_with_lineups(tmp_path: Path):
    _write_lineups_artifact(tmp_path)
    loaded = load_lineups(tmp_path)
    content = build_fixture_content()
    dives = tuple(
        replace(dive, home_lineup=loaded[dive.game_id][0], away_lineup=loaded[dive.game_id][1])
        if dive.game_id in loaded
        else dive
        for dive in content.dives
    )
    return replace(content, dives=dives)


def _knowledge(tmp_path: Path):
    return build_knowledge_for_board(_content_with_lineups(tmp_path))


def test_lineup_block_is_merged_and_corpus_stays_sorted(tmp_path: Path) -> None:
    knowledge = _knowledge(tmp_path)
    assert "2026_01_MIA_LV" in knowledge["lineups"]["games"]
    assert "2026_01_DEN_KC" in knowledge["lineups"]["games"]
    assert "2026_01_NE_SEA" not in knowledge["lineups"]["games"]
    blob = json.dumps(knowledge)
    assert blob == json.dumps(knowledge, sort_keys=True)


def test_no_lineup_artifact_leaves_an_empty_but_present_block() -> None:
    knowledge = build_knowledge_for_board(build_fixture_content())
    assert knowledge["lineups"]["games"] == {}
    assert knowledge["lineups"]["players"] == []


def test_qb_starter_answers_cleanly_when_forecast_and_lineup_agree(tmp_path: Path) -> None:
    resolved = answer("Who is starting at QB for the Dolphins?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:qb"
    assert "Tua Tagovailoa" in resolved.text
    assert _ANCHOR_RE.search(resolved.text), resolved.text
    assert "as of 2026-08-31T11:00:00Z from nflverse depth charts" in resolved.text
    assert resolved.anchors


def test_qb_starter_refuses_a_single_name_on_a_forecast_lineup_mismatch(tmp_path: Path) -> None:
    resolved = answer("Who's starting at QB for the Raiders?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:qb"
    assert "forecast assumed Geno Smith" in resolved.text
    assert "lists Aidan O'Connell" in resolved.text
    assert "can't state a single starter" in resolved.text
    assert "as of 2026-08-31T11:00:00Z from nflverse depth charts" in resolved.text


def test_qb_starter_degrades_to_stale_fallback_never_guessing(tmp_path: Path) -> None:
    resolved = answer("Who is starting at QB for the Chiefs?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:qb"
    assert "won't guess" in resolved.text
    assert "freshness budget" in resolved.text
    assert "as of 2020-01-01T00:00:00Z from nflverse depth charts" in resolved.text
    assert "Mahomes" not in resolved.text


def test_qb_starter_degrades_to_absent_fallback_for_an_unpublished_team(tmp_path: Path) -> None:
    resolved = answer("Who is starting at QB for the Patriots?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:qb"
    assert "No projected-lineup artifact is published for NE this week" in resolved.text
    assert "won't guess" in resolved.text


def test_team_injuries_reports_a_flagged_player(tmp_path: Path) -> None:
    resolved = answer("Any injuries for the Dolphins?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:injuries"
    assert "Tua Tagovailoa (questionable)" in resolved.text
    assert "as of 2026-08-31T11:00:00Z from nflverse depth charts" in resolved.text


def test_team_injuries_falls_back_to_team_level_status_with_no_flagged_players(
    tmp_path: Path,
) -> None:
    resolved = answer("Any injuries for the Raiders?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:injuries"
    assert "no per-player injury designation" in resolved.text
    assert "team-level injury feed status" in resolved.text


def test_team_injuries_degrades_to_stale_fallback(tmp_path: Path) -> None:
    resolved = answer("Any injuries for the Broncos?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:injuries"
    assert "won't guess" in resolved.text
    assert "freshness budget" in resolved.text


def test_player_availability_reports_probability_injury_and_role(tmp_path: Path) -> None:
    resolved = answer("Is Tua Tagovailoa playing?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:availability"
    assert "92% chance of taking the field" in resolved.text
    assert "injury status questionable" in resolved.text
    assert "the model's starter" in resolved.text
    assert "as of 2026-08-31T11:00:00Z from nflverse depth charts" in resolved.text


def test_player_availability_marks_a_non_scored_player_context_only(tmp_path: Path) -> None:
    resolved = answer("Is Tyreek Hill available?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:availability"
    assert "97% chance of taking the field" in resolved.text
    assert "context only" in resolved.text


def test_player_availability_shows_a_percentage_even_without_a_designation(
    tmp_path: Path,
) -> None:

    resolved = answer("Is Malik Washington playing?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:availability"
    assert "15% chance of taking the field" in resolved.text
    assert "no injury designation this week" not in resolved.text


def test_player_availability_degrades_to_stale_fallback(tmp_path: Path) -> None:
    resolved = answer("Is Patrick Mahomes playing?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:availability"
    assert "won't guess" in resolved.text
    assert "freshness budget" in resolved.text


def test_player_availability_falls_through_when_no_player_resolves(tmp_path: Path) -> None:
    resolved = answer("Is Bilbo Baggins playing?", _knowledge(tmp_path))
    assert resolved.topic != "lineup:availability"


def test_unresolved_availability_question_never_hijacks_unrelated_routing(
    tmp_path: Path,
) -> None:
    resolved = answer("how good is the model?", _knowledge(tmp_path))
    assert resolved.topic == "record"


def test_backup_qb_games_lists_the_mismatch_and_notes_excluded_stale_teams(
    tmp_path: Path,
) -> None:
    resolved = answer("Which games have a backup QB?", _knowledge(tmp_path))
    assert resolved.topic == "lineup:backup_qb"
    assert "LV" in resolved.text
    assert "forecast assumed Geno Smith" in resolved.text
    assert "current snapshot lists Aidan O'Connell" in resolved.text
    assert "stale" in resolved.text


def test_backup_qb_games_reports_none_when_nothing_disagrees() -> None:
    knowledge = lineups_module.build_lineup_knowledge(
        {},
        reference=datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC),
    )
    resolved = lineups_module.backup_qb_games_answer(knowledge)
    assert resolved.topic == "lineup:backup_qb"
    assert "No projected-lineup artifact is published" in resolved.text


def test_backup_qb_games_absent_artifact_via_answer() -> None:
    knowledge = build_knowledge_for_board(build_fixture_content())
    resolved = answer("which games have a backup QB?", knowledge)
    assert resolved.topic == "lineup:backup_qb"
    assert "No projected-lineup artifact is published this week" in resolved.text


def test_build_lineup_knowledge_flags_stale_precisely(tmp_path: Path) -> None:
    _write_lineups_artifact(tmp_path)
    loaded = load_lineups(tmp_path)
    knowledge = lineups_module.build_lineup_knowledge(
        loaded, reference=datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC)
    )
    mia_lv = knowledge["games"]["2026_01_MIA_LV"]
    den_kc = knowledge["games"]["2026_01_DEN_KC"]
    assert mia_lv["home"]["stale"] is False
    assert mia_lv["away"]["stale"] is False
    assert den_kc["home"]["stale"] is True
    assert den_kc["away"]["stale"] is True


def test_build_lineup_knowledge_carries_the_consistency_note_verbatim(tmp_path: Path) -> None:
    _write_lineups_artifact(tmp_path)
    loaded = load_lineups(tmp_path)
    knowledge = lineups_module.build_lineup_knowledge(
        loaded, reference=datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC)
    )
    lv_entry = knowledge["games"]["2026_01_MIA_LV"]["home"]
    assert lv_entry["note"] is not None
    assert lv_entry["current_qb_name"] == "Aidan O'Connell"
    assert lv_entry["model_qb_name"] == "Geno Smith"
    mia_entry = knowledge["games"]["2026_01_MIA_LV"]["away"]
    assert mia_entry["note"] is None


def test_missing_as_of_is_treated_as_unverifiable_not_fresh() -> None:
    from nfl_ats.lineup_view import team_lineup

    lineup = team_lineup(
        {
            "team": "LV",
            "source": "nflverse depth charts",
            "players": [{"name": "Q B", "position": "QB", "gsis_id": "q"}],
        }
    )
    knowledge = lineups_module.build_lineup_knowledge(
        {"G": (lineup, lineup)}, reference=datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC)
    )
    assert knowledge["games"]["G"]["home"]["stale"] is True
