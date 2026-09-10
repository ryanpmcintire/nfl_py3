from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _board_content_fixtures import build_fixture_content, build_fixture_history_content
from test_assistant_golden import (
    GOLDEN_QUESTIONS,
    HOME_PUSH_QUESTIONS,
    SEASON_RECORD_QUESTIONS,
    WEEK_TIMELINE_QUESTIONS,
    _build_home_push_weak_spots,
    _write_lineups_artifact,
)

from nfl_ats import board_assistant
from nfl_ats.board_assistant import answer, assistant_script, build_knowledge_for_board
from nfl_ats.board_site_content import headline_with_season_record
from nfl_ats.lineup_view import load_lineups

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_PATH = _REPO_ROOT / "tests" / "parity" / "assistant_parity.mjs"
_SCRIPT_RE = re.compile(r"<script>(.*)</script>", re.S)

LINEUP_REGRESSION_QUESTIONS: tuple[str, ...] = (
    "Who is starting at QB for the Dolphins?",
    "Who's starting at QB for the Raiders?",
    "Who is starting at QB for the Chiefs?",
    "Who is starting at QB for the Patriots?",
    "Any injuries for the Dolphins?",
    "Any injuries for the Raiders?",
    "Any injuries for the Broncos?",
    "Is Tua Tagovailoa playing?",
    "Is Tyreek Hill available?",
    "Is Patrick Mahomes playing?",
    "Is Bilbo Baggins playing?",
    "how good is the model?",
    "Which games have a backup QB?",
    "which games have a backup QB?",
)


def _node_executable() -> str | None:

    return shutil.which("node")


def _extract_inline_script(rendered: str) -> str:

    match = _SCRIPT_RE.search(rendered)
    assert match is not None, "assistant_script() did not return a <script>...</script> block"
    return match.group(1)


@pytest.fixture(scope="module")
def parity_knowledge(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:

    tmp_path = tmp_path_factory.mktemp("assistant_js_parity")
    _write_lineups_artifact(tmp_path)
    loaded = load_lineups(tmp_path)
    content = build_fixture_content()
    dives = tuple(
        replace(dive, home_lineup=loaded[dive.game_id][0], away_lineup=loaded[dive.game_id][1])
        if dive.game_id in loaded
        else dive
        for dive in content.dives
    )
    content = replace(content, dives=dives)
    knowledge = build_knowledge_for_board(content, weak_spots=_build_home_push_weak_spots())
    teams = [
        {"code": code, "aliases": list(aliases)}
        for code, aliases in sorted(board_assistant._TEAM_SYNONYMS.items())
    ]
    knowledge_for_js = dict(knowledge)
    knowledge_for_js["teams"] = teams
    return knowledge_for_js


@pytest.fixture(scope="module")
def all_questions() -> tuple[str, ...]:

    seen: list[str] = []
    for case in GOLDEN_QUESTIONS:
        if case.question not in seen:
            seen.append(case.question)
    for question in (
        *LINEUP_REGRESSION_QUESTIONS,
        *HOME_PUSH_QUESTIONS,
        *SEASON_RECORD_QUESTIONS,
        *WEEK_TIMELINE_QUESTIONS,
    ):
        if question not in seen:
            seen.append(question)
    return tuple(seen)


def test_harness_file_is_present() -> None:
    assert _HARNESS_PATH.exists(), f"missing Node parity harness: {_HARNESS_PATH}"


@pytest.mark.parametrize("settled", [False, True])
def test_python_and_js_engines_agree_on_every_question(
    tmp_path: Path,
    parity_knowledge: dict[str, Any],
    all_questions: tuple[str, ...],
    settled: bool,
) -> None:
    history = build_fixture_history_content(settled=settled)
    board = build_fixture_content()
    board = replace(board, headline=headline_with_season_record(board.headline, history))
    season_knowledge = build_knowledge_for_board(board, weak_spots=_build_home_push_weak_spots())
    season_entry = next(row for row in season_knowledge["entries"] if row["id"] == "season_record")
    parity_knowledge = dict(parity_knowledge)
    parity_knowledge["entries"] = [
        season_entry if row["id"] == "season_record" else row for row in parity_knowledge["entries"]
    ]
    node = _node_executable()
    if node is None:
        pytest.skip("node is not installed on this machine; JS parity check skipped")

    script_path = tmp_path / "assistant_engine.cjs"
    script_path.write_text(_extract_inline_script(assistant_script()), encoding="utf-8")
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(json.dumps(parity_knowledge), encoding="utf-8")
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(list(all_questions)), encoding="utf-8")

    result = subprocess.run(
        [node, str(_HARNESS_PATH), str(script_path), str(knowledge_path), str(questions_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, (
        f"Node parity harness exited {result.returncode}\nstderr:\n{result.stderr}"
    )
    js_answers = json.loads(result.stdout)
    assert len(js_answers) == len(all_questions)

    mismatches: list[dict[str, Any]] = []
    for question, js_answer in zip(all_questions, js_answers, strict=True):
        py_answer = answer(question, parity_knowledge)
        py_repr = {
            "topic": py_answer.topic,
            "text": py_answer.text,
            "anchors": list(py_answer.anchors),
        }
        js_repr = {
            "topic": js_answer["topic"],
            "text": js_answer["text"],
            "anchors": list(js_answer["anchors"]),
        }
        if py_repr != js_repr:
            mismatches.append({"question": question, "python": py_repr, "js": js_repr})

    assert not mismatches, (
        f"{len(mismatches)} of {len(all_questions)} question(s) diverge between Python and JS:\n"
        + json.dumps(mismatches, indent=2)
    )
