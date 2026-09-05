"""ENG-25: Python/JS parity for the board assistant, including the four
ENG-04 lineup intents ported into ``board_assistant.assistant_script`` by
this same backlog item.

Builds ONE synthetic knowledge corpus from the shared fixtures ENG-05 already
uses (``_board_content_fixtures.build_fixture_content`` plus the mixed
lineups artifact ``test_assistant_golden.py`` writes -- reused here, not
duplicated, so both suites exercise the identical MIA/LV mismatch and
NE/SEA clean game), then runs every golden-corpus question
(``tests/fixtures/assistant_golden/questions.json``) plus the lineup
regression phrasings ``test_board_assistant_lineups.py`` pins in Python
through BOTH engines: the Python reference
(:func:`nfl_ats.board_assistant.answer`) and the inline-JS port, evaluated
under Node by ``tests/parity/assistant_parity.mjs`` (the script text is
extracted from :func:`nfl_ats.board_assistant.assistant_script`'s own
output, never hand-copied, so the two can never silently diverge from what
actually ships).

Skips (never fails) when ``node`` is not on this machine's PATH -- a missing
local dev dependency is not a code defect, and the project rule is to skip
loudly, not report a false pass or a false fail. When Node IS available,
every question must match exactly (topic, text, anchors) or the test fails
with the full list of mismatches.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _board_content_fixtures import build_fixture_content
from test_assistant_golden import GOLDEN_QUESTIONS, _write_lineups_artifact

from nfl_ats import board_assistant
from nfl_ats.board_assistant import answer, assistant_script, build_knowledge_for_board
from nfl_ats.lineup_view import load_lineups

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_PATH = _REPO_ROOT / "tests" / "parity" / "assistant_parity.mjs"
_SCRIPT_RE = re.compile(r"<script>(.*)</script>", re.S)

#: Lineup-intent phrasings pinned in ``test_board_assistant_lineups.py``'s
#: own Python-only unit tests, harvested verbatim (not re-derived) so the JS
#: port is proven against the SAME wording that file already exercises --
#: run here against the shared golden knowledge below (not that file's own,
#: differently-staled lineups fixture), so every phrasing still gets a real
#: answer to compare, just not necessarily via the same fallback branch that
#: file's assertions target.
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
    """The ``node`` binary on PATH, or ``None`` when this machine doesn't
    have one. Callers must skip, never fail, when this returns ``None`` --
    see the ENG-25 spec: the harness must be authored and skip cleanly
    without Node, not be treated as a hard CI dependency."""

    return shutil.which("node")


def _extract_inline_script(rendered: str) -> str:
    """Strip the ``<script>``/``</script>`` wrapper :func:`assistant_script`
    returns, leaving bare JS the Node harness can ``require()`` as
    CommonJS (written by the caller to a ``.cjs`` path so Node treats it as
    CommonJS regardless of any nearby ``package.json`` ``"type"`` field)."""

    match = _SCRIPT_RE.search(rendered)
    assert match is not None, "assistant_script() did not return a <script>...</script> block"
    return match.group(1)


@pytest.fixture(scope="module")
def parity_knowledge(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The exact corpus a page embeds: ``build_knowledge_for_board``'s
    output plus the ``"teams"`` list :func:`board_assistant.assistant_section`
    merges in before inlining it -- the JS ``parse()`` reads team aliases
    from ``corpus.teams`` (Python reads its own private ``_TEAM_SYNONYMS``
    instead), so the JS side needs that key present or every team-scoped
    question would silently fail to resolve a team on the JS side only."""

    tmp_path = tmp_path_factory.mktemp("assistant_js_parity")
    _write_lineups_artifact(tmp_path)
    loaded = load_lineups(tmp_path)  # real nfl_ats.lineup_view parser
    content = build_fixture_content()
    dives = tuple(
        replace(dive, home_lineup=loaded[dive.game_id][0], away_lineup=loaded[dive.game_id][1])
        if dive.game_id in loaded
        else dive
        for dive in content.dives
    )
    content = replace(content, dives=dives)
    knowledge = build_knowledge_for_board(content)
    teams = [
        {"code": code, "aliases": list(aliases)}
        for code, aliases in sorted(board_assistant._TEAM_SYNONYMS.items())
    ]
    knowledge_for_js = dict(knowledge)
    knowledge_for_js["teams"] = teams
    return knowledge_for_js


@pytest.fixture(scope="module")
def all_questions() -> tuple[str, ...]:
    """Every golden-corpus question (deduped, in fixture order) plus the
    lineup regression phrasings above (skipping any exact duplicate)."""

    seen: list[str] = []
    for case in GOLDEN_QUESTIONS:
        if case.question not in seen:
            seen.append(case.question)
    for question in LINEUP_REGRESSION_QUESTIONS:
        if question not in seen:
            seen.append(question)
    return tuple(seen)


def test_harness_file_is_present() -> None:
    assert _HARNESS_PATH.exists(), f"missing Node parity harness: {_HARNESS_PATH}"


def test_python_and_js_engines_agree_on_every_question(
    tmp_path: Path,
    parity_knowledge: dict[str, Any],
    all_questions: tuple[str, ...],
) -> None:
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
        # Node's JSON.stringify writes raw UTF-8 (unlike Python's ensure_ascii
        # json.dumps default); without an explicit encoding, subprocess falls
        # back to the OS locale codepage, which mis-decodes non-ASCII text
        # (e.g. an em dash) on a non-UTF-8 Windows console.
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
