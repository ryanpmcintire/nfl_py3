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
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from _board_content_fixtures import (
    build_fixture_content,
    build_fixture_history_content,
    build_fixture_weak_spots,
)

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
from nfl_ats.board_site_content import _load_model_page_content, headline_with_season_record
from nfl_ats.lineup_view import STABLE_LINEUP_PATH, load_lineups
from nfl_ats.model_weak_spots import HomeCorrection, HomeCorrectionRow, WeakSpots
from nfl_ats.public_board import OpenerEvaluationArtifacts

SEASON_RECORD_QUESTIONS = (
    "how is the model doing this season",
    "what is the record this year",
    "how many picks has it got right in 2026",
)


WEEK_TIMELINE_QUESTIONS = (
    "when do the lines lock",
    "when does the card update",
    "when are picks due",
    "when is the deadline for the Monday game",
)


HOME_PUSH_QUESTIONS = (
    "what is the home-side push",
    "why does the model lean home on big spreads",
    "what changed in the forecast this season",
    "how big is the home adjustment",
)


def _build_home_push_weak_spots() -> WeakSpots:
    return replace(
        build_fixture_weak_spots(),
        home_correction=HomeCorrection(
            rows=(HomeCorrectionRow("7.5-10", 0.75, 40, 20, 3, 0.6, 0.55),),
            archive_games=20,
            picks_changed=3,
            accuracy_with=0.6,
            accuracy_without=0.55,
            this_week_available=True,
        ),
    )


_QUESTIONS_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "assistant_golden" / "questions.json"
)

_REFRESH_LINES = (
    "MIA at LV refresh (refresh_sat): pick now MIA (Tuesday card: LV); "
    "frozen Tuesday line (home +3.5); line moved +1.5 points.",
)

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
                            "has_injury_designation": True,
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
                            "has_injury_designation": True,
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
    loaded = load_lineups(tmp_path)
    content = replace(build_fixture_content(), refresh_lines=_REFRESH_LINES)
    dives = tuple(
        replace(dive, home_lineup=loaded[dive.game_id][0], away_lineup=loaded[dive.game_id][1])
        if dive.game_id in loaded
        else dive
        for dive in content.dives
    )
    content = replace(content, dives=dives)
    knowledge = build_knowledge_for_board(content, weak_spots=_build_home_push_weak_spots())
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


def test_golden_fixture_has_60_to_120_rows_covering_every_category() -> None:
    assert 60 <= len(GOLDEN_QUESTIONS) <= 120
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
        "season_record",
        "policy",
        "findings",
        "timing",
        "week_timeline",
        "scope:winners",
        "scope:injury",
        "scope:weather",
        "fallback",
        "lineup:qb",
        "lineup:injuries",
        "lineup:availability",
        "lineup:backup_qb",
        "weak_spots",
        "weak_spots_home_split",
        "weak_spots_home_push",
    }
    expected = deflect_ids | reachable_glossary_ids | fixed_topics
    observed = {case.expected_intent for case in GOLDEN_QUESTIONS}
    missing = expected - observed
    assert not missing, f"golden corpus never exercises: {sorted(missing)}"


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
    assert "onclick" not in section


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


@pytest.mark.parametrize("question", HOME_PUSH_QUESTIONS)
@pytest.mark.parametrize("state", ("available", "archive_only", "empty", "missing"))
def test_home_push_matches_model_reader_text(tmp_path: Path, question: str, state: str) -> None:
    weak_spots = _build_home_push_weak_spots()
    correction = weak_spots.home_correction
    assert correction is not None
    if state == "archive_only":
        correction = replace(
            correction,
            rows=tuple(
                replace(row, this_week_points=None, learned_from_games=None)
                for row in correction.rows
            ),
            this_week_available=False,
        )
    elif state == "empty":
        correction = HomeCorrection()
    elif state == "missing":
        correction = None
    weak_spots = replace(weak_spots, home_correction=correction)
    model = _load_model_page_content(
        tmp_path,
        registry_root=tmp_path,
        board=build_fixture_content(),
        opener=OpenerEvaluationArtifacts({}, pd.DataFrame()),
        active={},
        generated_at=datetime(2026, 9, 8, tzinfo=UTC),
    )
    model = replace(model, weak_spots=weak_spots)
    knowledge = board_assistant.build_knowledge_for_model(model)
    response = board_assistant.answer(question, knowledge)
    assert response.topic == "weak_spots_home_push"
    assert response.text == model.weak_spots.home_correction_text
    assert response.anchors == ("model.html#weak-spots-push-h",)
    expected = correction or HomeCorrection()
    assert knowledge["weak_spots_home_push"] == [asdict(row) for row in expected.rows]
    assert knowledge["weak_spots_home_push_summary"] == {
        key: value for key, value in asdict(expected).items() if key != "rows"
    }
    for forbidden in ("raw model", "p+", "policy"):
        assert forbidden not in response.text.lower()


@pytest.mark.parametrize("settled", [False, True])
@pytest.mark.parametrize("question", SEASON_RECORD_QUESTIONS)
def test_season_record_across_pages(tmp_path: Path, settled: bool, question: str) -> None:
    history = build_fixture_history_content(settled=settled)
    board = build_fixture_content()
    headline = headline_with_season_record(board.headline, history)
    board = replace(board, headline=headline)
    model = _load_model_page_content(
        tmp_path,
        registry_root=tmp_path,
        board=board,
        opener=OpenerEvaluationArtifacts({}, pd.DataFrame()),
        active={},
        generated_at=datetime(2026, 9, 8, tzinfo=UTC),
    )
    answers = [
        board_assistant.answer(question, knowledge)
        for knowledge in (
            build_knowledge_for_board(board, weak_spots=WeakSpots()),
            board_assistant.build_knowledge_for_model(model),
            board_assistant.build_knowledge_for_history(history),
        )
    ]
    assert answers[0] == answers[1] == answers[2]
    response = answers[0]
    assert response.topic == "season_record"
    assert response.anchors == ("history.html#history-picks-h", "model.html")
    live = (
        "In 2026, the played card is 2-1-1 (wins-losses-pushes). "
        "1 week fully settled. The played card's rate is 66.7%, excluding pushes. "
        "Unfinished picks are not counted."
        if settled
        else "No picks have settled in 2026 yet."
    )
    assert response.text == (
        f"{live} The played card's archive score is 55.4%; that is the archive, not this season."
    )
    for forbidden in ("raw model", "p+", "policy", "nan", "none"):
        assert forbidden not in response.text.lower()


@pytest.mark.parametrize("state", ["pushes", "partial", "unavailable", "no_archive"])
def test_season_record_edge_cases(state: str) -> None:
    history = build_fixture_history_content(settled=True)
    if state == "pushes":
        history = replace(
            history,
            picks=tuple(replace(row, status="push", correct=None) for row in history.picks),
        )
    elif state == "partial":
        history = replace(
            history,
            picks=(
                *history.picks,
                replace(history.picks[-2], game_id="pending"),
            ),
        )
        history = replace(
            history,
            picks=tuple(
                replace(row, status="settled", correct=True)
                if row.game_id == "2026_02_NE_SEA"
                else row
                for row in history.picks
            ),
        )
    elif state == "unavailable":
        history = replace(history, primary_error="private path and error details")
    else:
        history = replace(
            history, headline=replace(build_fixture_content().headline, played_card_pct=None)
        )
    response = board_assistant.answer(
        SEASON_RECORD_QUESTIONS[0], board_assistant.build_knowledge_for_history(history)
    )
    if state == "pushes":
        assert "0-0-5 (wins-losses-pushes)" in response.text
        assert "2 weeks fully settled" in response.text
        assert "no win rate yet" in response.text
    elif state == "partial":
        assert "3-1-1 (wins-losses-pushes)" in response.text
        assert "1 week fully settled" in response.text
        assert "75.0%" in response.text
    elif state == "unavailable":
        assert "record is unavailable" in response.text
        assert "private" not in response.text
        assert "No picks have settled" not in response.text
    else:
        assert "archive score is unavailable" in response.text
        assert "55.4%" not in response.text


def test_home_push_golden_questions_are_pinned() -> None:
    cases = {
        case.question: case
        for case in GOLDEN_QUESTIONS
        if case.expected_intent == "weak_spots_home_push"
    }
    assert set(cases) == {
        "what is the home-side push",
        "why does the model lean home on big spreads",
    }


@pytest.mark.parametrize("question", HOME_PUSH_QUESTIONS)
def test_home_push_board_answer_uses_shared_reader_text(
    golden_environment: SimpleNamespace, question: str
) -> None:
    response = board_assistant.answer(question, golden_environment.knowledge)
    assert response.topic == "weak_spots_home_push"
    assert response.text == _build_home_push_weak_spots().home_correction_text
    assert response.anchors == ("model.html#weak-spots-push-h",)


def test_week_timeline_golden_questions_are_pinned() -> None:
    pinned = {case.question: case.expected_intent for case in GOLDEN_QUESTIONS}
    for question in WEEK_TIMELINE_QUESTIONS[:2]:
        assert pinned[question] == "week_timeline"


@pytest.mark.parametrize("question", WEEK_TIMELINE_QUESTIONS)
def test_week_timeline_answer_uses_page_content(question: str) -> None:
    content = build_fixture_content()
    result = board_assistant.answer(question, build_knowledge_for_board(content))
    assert result.topic == "week_timeline"
    assert result.anchors == ("index.html#week-timeline-h",)
    if "Monday game" in question:
        assert result.text == dict(content.week_timeline.deadlines)["2026_01_DEN_KC"]
        assert "Sunday 4:00 PM ET, before kickoff" in result.text
        assert "NE at SEA" not in result.text
    else:
        assert result.text == content.week_timeline.text
