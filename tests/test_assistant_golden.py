"""ENG-05: golden-question evaluation for the board assistant.

Runs the fixed corpus in ``tests/fixtures/assistant_golden/questions.json``
(60-110 rows spanning routing, unsupported-question fallback, numeric
provenance, stale-data behaviour, the ENG-04 lineup intents, and
accessibility-text safety -- widened from 100 by ENG-36's six added
multi-word-glossary routing rows) through
:func:`nfl_ats.assistant_eval.evaluate_golden`,
plus a direct check of the rendered chat panel's keyboard/no-JS
accessibility contract on the real ``board_terminal.render`` output.

Reuses the SAME ``BoardContent`` fixture (``_board_content_fixtures.
build_fixture_content``) and the same synthetic-lineups-artifact technique
``tests/test_board_assistant_lineups.py`` already exercises (a tmp_path
``lineups.json`` loaded through the real ``nfl_ats.lineup_view`` parser) --
no new content-building machinery, only a new question corpus and grader.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from _board_content_fixtures import build_fixture_content

from nfl_ats import board_assistant, board_terminal
from nfl_ats.assistant_eval import (
    CATEGORIES,
    EvalReport,
    evaluate_golden,
    has_provenance_anchor,
    load_questions,
    make_stale_lineup_knowledge,
    render_report,
)
from nfl_ats.board_assistant import build_knowledge_for_board
from nfl_ats.lineup_view import STABLE_LINEUP_PATH, load_lineups

_QUESTIONS_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "assistant_golden" / "questions.json"
)

#: Same refresh-diff text ``test_board_assistant.py``/``test_assistant_battery.py``
#: use for their "refresh" intent coverage -- fixture DATA, not code, so
#: reusing it verbatim keeps this file's "what changed since Tuesday"
#: expectations matching the SAME MIA/LV refresh those files already proved,
#: rather than inventing a second, divergent refresh line.
_REFRESH_LINES = (
    "MIA at LV refresh (refresh_sat): pick now MIA (Tuesday card: LV); "
    "frozen Tuesday line (home +3.5); line moved +1.5 points.",
)

# Read once at import time -- a pure, cheap JSON parse (no artifact I/O),
# the same way ``test_assistant_battery.py`` defines its ``BATTERY`` tuple
# at module scope, so it's usable both directly and as parametrize input.
GOLDEN_QUESTIONS = load_questions(_QUESTIONS_PATH)


def _write_lineups_artifact(tmp_path: Path) -> None:
    """A small, deliberately mixed ``lineups.json``: MIA (clean) / LV
    (fail-closed forecast/lineup mismatch) share the fixture's Best Pick
    game; NE/SEA is a second, entirely clean game. Every other fixture game
    (including DEN/KC) is left unpublished on purpose, exercising the "no
    artifact for this team" fallback for free -- mirrors the payload shape
    ``tests/test_board_assistant_lineups.py`` already proved against the
    real ``nfl_ats.lineup_view`` parser, sized down to just the two games
    this golden corpus needs."""

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
                        },
                    ],
                },
            },
            "2026_01_NE_SEA": {
                "home": {
                    "team": "SEA",
                    "as_of": "2026-08-31T11:00:00Z",
                    "source": "nflverse depth charts",
                    "injury_status": "unavailable — current injury feed not attached",
                    "note": None,
                    "players": [
                        {
                            "name": "Sam Darnold",
                            "position": "QB",
                            "slot": "QB1",
                            "depth": 1,
                            "unit": "offense",
                            "gsis_id": "sea-darnold",
                            "model_role": "base_model",
                            "play_probability": 0.95,
                        },
                    ],
                },
                "away": {
                    "team": "NE",
                    "as_of": "2026-08-31T11:00:00Z",
                    "source": "nflverse depth charts",
                    "injury_status": "unavailable — current injury feed not attached",
                    "note": None,
                    "players": [
                        {
                            "name": "Drake Maye",
                            "position": "QB",
                            "slot": "QB1",
                            "depth": 1,
                            "unit": "offense",
                            "gsis_id": "ne-maye",
                            "model_role": "base_model",
                            "play_probability": 0.9,
                        },
                    ],
                },
            },
        },
    }
    target = tmp_path / STABLE_LINEUP_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(scope="module")
def golden_environment(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Built once per module: the shared 16-game fixture plus a refresh
    line and a mixed lineups block, its built knowledge corpus, a forced
    all-stale variant of that SAME corpus (:func:`make_stale_lineup_knowledge`
    -- the "second knowledge object whose lineups/source timestamps are
    older than the documented budget" the ENG-05 spec calls for), and the
    loaded golden-question rows."""

    tmp_path = tmp_path_factory.mktemp("assistant_golden")
    _write_lineups_artifact(tmp_path)
    loaded = load_lineups(tmp_path)  # real nfl_ats.lineup_view parser
    content = replace(build_fixture_content(), refresh_lines=_REFRESH_LINES)
    dives = tuple(
        replace(dive, home_lineup=loaded[dive.game_id][0], away_lineup=loaded[dive.game_id][1])
        if dive.game_id in loaded
        else dive
        for dive in content.dives
    )
    content = replace(content, dives=dives)
    knowledge = build_knowledge_for_board(content)
    stale_knowledge = make_stale_lineup_knowledge(knowledge)
    return SimpleNamespace(
        content=content,
        knowledge=knowledge,
        stale_knowledge=stale_knowledge,
    )


@pytest.fixture(scope="module")
def golden_report(golden_environment: SimpleNamespace) -> EvalReport:
    return evaluate_golden(
        golden_environment.knowledge,
        GOLDEN_QUESTIONS,
        stale_knowledge=golden_environment.stale_knowledge,
    )


# ---------------------------------------------------------------------------
# Corpus shape.
# ---------------------------------------------------------------------------


def test_golden_fixture_has_60_to_110_rows_covering_every_category() -> None:
    assert 60 <= len(GOLDEN_QUESTIONS) <= 110
    assert {case.category for case in GOLDEN_QUESTIONS} == set(CATEGORIES)


def test_golden_fixture_covers_every_router_intent() -> None:
    """Enumerated from the code (not memory): every deflect body, every
    glossary term (single- and multi-word), and every other topic
    :func:`answer` can return. ENG-36 fixed ``board_assistant._parse`` to
    do longest-match-first phrase matching over normalised n-grams, so
    multi-word terms ("cover probability", "closing line", "Best Pick")
    are now reachable the same way single-word terms always were -- this
    test previously excluded them on purpose to document the gap (see the
    ENG-05 session report); now that the router is fixed, excluding them
    would paper back over a regression instead of catching one."""

    deflect_ids = {entry.entry_id for entry in board_assistant._deflect_entries(2026, 1)}
    reachable_glossary_ids = {f"glossary:{item.term}" for item in board_assistant.GLOSSARY}
    fixed_topics = {
        "team_pick",
        "team_confidence",
        "team_schedule",
        "refresh",
        "rankings",
        "dogs",
        "favorites",
        "slots",
        "best_pick",
        "record",
        "policy",
        "findings",
        "timing",
        "scope:winners",
        "scope:injury",
        "scope:weather",
        "fallback",
        "lineup:qb",
        "lineup:injuries",
        "lineup:availability",
        "lineup:backup_qb",
    }
    expected = deflect_ids | reachable_glossary_ids | fixed_topics
    observed = {case.expected_intent for case in GOLDEN_QUESTIONS}
    missing = expected - observed
    assert not missing, f"golden corpus never exercises: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Pass rate: 100% overall and per category (aggregate report, plus one
# parametrized test per question for isolated pass/fail reporting).
# ---------------------------------------------------------------------------


def test_golden_report_is_100_percent_overall_and_per_category(golden_report: EvalReport) -> None:
    if not golden_report.overall_pass:
        pytest.fail(render_report(golden_report))
    for cat_report in golden_report.category_reports:
        assert cat_report.all_passed, render_report(golden_report)


@pytest.mark.parametrize(
    "case",
    GOLDEN_QUESTIONS,
    ids=[f"{c.category}-{i:03d}-{c.question[:24]!r}" for i, c in enumerate(GOLDEN_QUESTIONS)],
)
def test_each_golden_question_passes(golden_environment: SimpleNamespace, case) -> None:  # type: ignore[no-untyped-def]
    report = evaluate_golden(
        golden_environment.knowledge, [case], stale_knowledge=golden_environment.stale_knowledge
    )
    assert report.overall_pass, render_report(report)


# ---------------------------------------------------------------------------
# Numeric provenance: never a bare number.
# ---------------------------------------------------------------------------


def test_numeric_provenance_answers_never_print_a_bare_number(
    golden_environment: SimpleNamespace,
) -> None:
    cases = [c for c in GOLDEN_QUESTIONS if c.category == "numeric_provenance"]
    assert cases, "no numeric_provenance rows in the golden fixture"
    for case in cases:
        resolved = board_assistant.answer(case.question, golden_environment.knowledge)
        assert re.search(r"\d", resolved.text), f"{case.question!r} has no number to anchor"
        assert has_provenance_anchor(resolved.text), (
            f"{case.question!r} prints a number with no provenance anchor: {resolved.text!r}"
        )


# ---------------------------------------------------------------------------
# Stale-data behaviour: a second knowledge object, aged past the documented
# 48h budget (nfl_ats.board_assistant_lineups.LINEUP_STALE_BUDGET_HOURS),
# must never name a starter and must always carry the stale-fallback text.
# ---------------------------------------------------------------------------


def test_stale_knowledge_never_names_a_starter(golden_environment: SimpleNamespace) -> None:
    stale = golden_environment.stale_knowledge
    for question, forbidden_names in (
        ("Who is starting at QB for the Dolphins?", ("Tua Tagovailoa",)),
        ("Who's starting at QB for the Raiders?", ("Geno Smith", "Aidan O'Connell")),
        ("Who is starting at QB for New England?", ("Drake Maye",)),
        ("Is Sam Darnold playing for Seattle?", ("95%",)),
    ):
        resolved = board_assistant.answer(question, stale)
        assert "won't guess" in resolved.text
        assert "freshness budget" in resolved.text
        for name in forbidden_names:
            assert name not in resolved.text, f"{question!r} named a starter from stale data"


def test_stale_data_rows_are_graded_against_a_genuinely_different_knowledge(
    golden_environment: SimpleNamespace,
) -> None:
    """Proves the fresh/stale comparison is not an accidental no-op: same
    question, same topic (staleness changes the answer body, never the
    intent it routes to), but a different rendered answer."""

    cases = [c for c in GOLDEN_QUESTIONS if c.category == "stale_data"]
    assert cases
    for case in cases:
        fresh = board_assistant.answer(case.question, golden_environment.knowledge)
        stale = board_assistant.answer(case.question, golden_environment.stale_knowledge)
        assert fresh.topic == stale.topic == case.expected_intent
        assert fresh.text != stale.text


# ---------------------------------------------------------------------------
# Accessibility contract: labelled input, keyboard-reachable submit, a live
# region for answers, and a <noscript> fallback that keeps the picks table
# visible and explains that the ASSISTANT needs JavaScript (the rest of the
# page does not). Runs against the real ``board_terminal.render`` output,
# not just ``assistant_section`` in isolation.
# ---------------------------------------------------------------------------


def _assistant_section_html(full_page_html: str) -> str:
    match = re.search(r'<section class="assistant"[^>]*>.*?</section>', full_page_html, re.S)
    assert match is not None, 'no <section class="assistant"> in the rendered page'
    return match.group(0)


def test_chat_panel_has_a_labelled_input(golden_environment: SimpleNamespace) -> None:
    section = _assistant_section_html(board_terminal.render(golden_environment.content))
    assert "<label" in section
    assert 'for="assistant-q"' in section
    assert 'id="assistant-q"' in section


def test_chat_panel_submit_is_keyboard_reachable(golden_environment: SimpleNamespace) -> None:
    section = _assistant_section_html(board_terminal.render(golden_environment.content))
    assert '<form class="assistant-form">' in section
    assert 'type="submit"' in section
    assert 'tabindex="-1"' not in section
    assert "onclick" not in section  # answered via the form's submit event, never a bare click


def test_chat_panel_has_a_live_region_for_answers(golden_environment: SimpleNamespace) -> None:
    section = _assistant_section_html(board_terminal.render(golden_environment.content))
    assert 'aria-live="polite"' in section


def test_noscript_fallback_explains_javascript_is_needed(
    golden_environment: SimpleNamespace,
) -> None:
    section = _assistant_section_html(board_terminal.render(golden_environment.content))
    match = re.search(r"<noscript>(.*?)</noscript>", section, re.S)
    assert match is not None
    noscript_body = match.group(1)
    assert "JavaScript" in noscript_body, (
        "the assistant's <noscript> fallback must say it needs JavaScript, not just list "
        f"links: {noscript_body!r}"
    )


def test_picks_table_renders_unconditionally_outside_any_noscript_gate(
    golden_environment: SimpleNamespace,
) -> None:
    """The picks table itself must never depend on JavaScript: strip every
    ``<noscript>...</noscript>`` block from the page and confirm the table
    markup survives -- proves a no-JS reader still sees the real picks, not
    just the assistant's own topic links."""

    html = board_terminal.render(golden_environment.content)
    without_noscript = re.sub(r"<noscript>.*?</noscript>", "", html, flags=re.S)
    assert '<table class="board' in without_noscript
