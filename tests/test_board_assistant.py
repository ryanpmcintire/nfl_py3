"""Board assistant (UI-16): retrieval-only answers over a publish-time
corpus -- the Python reference matcher, the corpus builder, and the
rendered panel contract (inline JSON, guardrails, no-JS fallback)."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest
from _board_content_fixtures import build_fixture_content

from nfl_ats import board_assistant, board_terminal
from nfl_ats.board_assistant import (
    ASSISTANT_VERSION,
    GLOSSARY,
    SYNONYMS,
    answer,
    assistant_section,
    build_knowledge,
    build_knowledge_for_board,
)


def _knowledge(**overrides):
    content = build_fixture_content()
    knowledge = build_knowledge_for_board(content)
    if overrides:
        knowledge = {**knowledge, **overrides}
    return knowledge


def _entry_ids(knowledge) -> list[str]:
    return [entry["id"] for entry in knowledge["entries"]]


# ---------------------------------------------------------------------------
# Corpus shape and determinism (A1).
# ---------------------------------------------------------------------------


def test_corpus_is_deterministic_with_sorted_keys_and_provenance() -> None:
    first = _knowledge()
    second = _knowledge()
    assert first == second
    blob = json.dumps(first)
    assert blob == json.dumps(first, sort_keys=True)
    assert first["assistant_version"] == ASSISTANT_VERSION == 1
    assert first["provenance"]["builder"] == "nfl_ats.board_assistant.build_knowledge"
    assert first["provenance"]["model_id"] == "d1f07d773475dc58"
    assert first["provenance"]["game_count"] == 16


def test_corpus_covers_games_record_policy_findings_glossary() -> None:
    knowledge = _knowledge()
    entry_ids = _entry_ids(knowledge)
    assert entry_ids.count("best_pick") == 1
    assert entry_ids.count("record") == 1
    assert entry_ids.count("policy") == 1
    assert entry_ids.count("timing") == 1
    assert len(knowledge["games"]) == 16
    assert len(knowledge["ranked"]) == 16
    assert len(knowledge["dogs"]) + len(knowledge["favorites"]) + len(knowledge["flat_picks"]) == 16
    assert sum(item.startswith("finding:") for item in entry_ids) == 2
    assert sum(item.startswith("glossary:") for item in entry_ids) == len(GLOSSARY)
    assert sum(item.startswith("deflect:") for item in entry_ids) == len(
        board_assistant._deflect_rule_sets()
    )
    assert knowledge["fallback"]["anchors"] == ["index.html", "model.html", "findings.html"]
    assert knowledge["counts"]["games"] == 16
    assert knowledge["counts"]["dogs"] == len(knowledge["dogs"])
    assert knowledge["counts"]["favorites"] == len(knowledge["favorites"])


def test_game_body_carries_pick_line_probability_and_flip() -> None:
    knowledge = _knowledge()
    bodies = {game["game_id"]: game["why"] for game in knowledge["games"]}
    mia = bodies["2026_01_MIA_LV"]
    assert "MIA +3.5" in mia
    assert "54.4%" in mia
    assert "Best Pick of the week." in mia
    bal = bodies["2026_01_BAL_IND"]
    assert "Policy flip: coach fade." in bal


# ---------------------------------------------------------------------------
# Core routing (the wide battery lives in test_assistant_battery.py).
# ---------------------------------------------------------------------------

_CORE_ROUTING: tuple[tuple[str, str, str], ...] = (
    ("Why MIA?", "team_pick", "MIA +3.5"),
    ("dolphins", "team_pick", "MIA +3.5"),
    ("Seahawks Patriots", "team_pick", "SEA -3.5"),
    ("best pick", "best_pick", "MIA +3.5"),
    ("should I bet on the Chiefs?", "deflect:wager", "never advises wagers"),
    ("who wins week 2?", "deflect:future", "Week 1 only"),
    ("how good is the model?", "record", "55.4%"),
    ("why did the pick flip?", "policy", "Flipped 2 picks"),
    ("what does push mean?", "glossary:push", "tie against the spread"),
    ("when do picks lock?", "timing", "freeze Tuesday"),
    ("asdkjfh qzx", "fallback", "not in this week's published card"),
    ("", "fallback", "not in this week's published card"),
)


@pytest.mark.parametrize(("question", "topic", "substring"), _CORE_ROUTING)
def test_core_routing(question: str, topic: str, substring: str) -> None:
    resolved = answer(question, _knowledge())
    assert resolved.topic == topic, f"{question!r} -> {resolved.topic!r}"
    assert substring in resolved.text, f"{question!r} missing {substring!r}"
    assert resolved.anchors, f"{question!r} carries no anchor"


_REFRESH_LINES = (
    "MIA at LV refresh (refresh_sat): pick now MIA (Tuesday card: LV); "
    "frozen Tuesday line (home +3.5); line moved +1.5 points.",
)


def _refreshed_knowledge():
    content = replace(build_fixture_content(), refresh_lines=_REFRESH_LINES)
    return build_knowledge_for_board(content)


def test_refresh_entry_appears_only_when_a_refresh_ran() -> None:
    plain_ids = [entry["id"] for entry in _knowledge()["entries"]]
    assert "refresh" not in plain_ids
    lined_ids = [entry["id"] for entry in _refreshed_knowledge()["entries"]]
    assert "refresh" in lined_ids


def test_refresh_questions_route_to_the_diff() -> None:
    knowledge = _refreshed_knowledge()
    for question in (
        "what changed since Tuesday?",
        "has the MIA pick been refreshed?",
        "did the Sunday update change anything?",
    ):
        resolved = answer(question, knowledge)
        assert resolved.topic == "refresh", f"{question!r} -> {resolved.topic!r}"
        assert "pick now MIA (Tuesday card: LV)" in resolved.text


def test_clock_questions_stay_with_timing_after_a_refresh() -> None:
    resolved = answer("when do picks lock?", _refreshed_knowledge())
    assert resolved.topic == "timing"


def test_refresh_answer_numbers_occur_in_the_corpus() -> None:
    knowledge = _refreshed_knowledge()
    allowed = set(re.findall(r"\d+(?:\.\d+)?%?", json.dumps(knowledge)))
    for question in ("what changed?", "MIA refresh?", "Sunday update?"):
        text = answer(question, knowledge).text
        for number in re.findall(r"\d+(?:\.\d+)?%?", text):
            assert number in allowed, f"{number!r} from {question!r} not in corpus"


def test_multi_team_question_routes_to_the_shared_game() -> None:
    resolved = answer("Chiefs Broncos?", _knowledge())
    assert resolved.topic == "team_pick"
    assert "DEN at KC" in resolved.text


def test_nickname_prefers_the_exact_team_over_a_fragment() -> None:
    resolved = answer("Chiefs?", _knowledge())
    assert resolved.topic == "team_pick"
    assert "DEN at KC" in resolved.text


# ---------------------------------------------------------------------------
# Guardrails (A4): never emit a number absent from the corpus.
# ---------------------------------------------------------------------------


def _corpus_numbers(knowledge) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", json.dumps(knowledge)))


def test_deflect_bodies_carry_no_invented_numbers() -> None:
    knowledge = _knowledge()
    allowed = _corpus_numbers(knowledge)
    for entry in knowledge["entries"]:
        if str(entry["id"]).startswith("deflect:"):
            for number in re.findall(r"\d+(?:\.\d+)?%?", str(entry["body"])):
                assert number in allowed


# ---------------------------------------------------------------------------
# Rendered panel contract (A3): inline JSON, escape, no-JS fallback.
# ---------------------------------------------------------------------------


def _embedded_corpus(section: str) -> dict:
    match = re.search(
        r'<script type="application/json" class="assistant-data">(.*?)</script>',
        section,
        re.S,
    )
    assert match is not None
    blob = match.group(1)
    assert "<" not in blob and ">" not in blob
    return json.loads(blob)


def test_embedded_corpus_matches_python_source_of_truth() -> None:
    section = assistant_section(_knowledge())
    embedded = _embedded_corpus(section)
    assert embedded["synonyms"] == {key: list(value) for key, value in SYNONYMS.items()}
    assert {team["code"] for team in embedded["teams"]} >= {"ne", "mia", "kc", "sea"}


def test_panel_is_keyboard_operable_with_live_region_and_fallback() -> None:
    section = assistant_section(_knowledge())
    assert "<details" in section and "<summary>" in section
    assert 'aria-live="polite"' in section
    assert 'for="assistant-q"' in section and 'id="assistant-q"' in section
    assert 'type="submit"' in section
    assert "<noscript>" in section
    assert "index.html" in section and "model.html" in section


def test_panel_never_embeds_raw_markup() -> None:
    content = replace(build_fixture_content(), best_pick_note='Quote "x" & <tag>')
    knowledge = build_knowledge_for_board(content)
    section = assistant_section(knowledge)
    assert "<tag>" not in section
    assert section.count("</script>") == 1
    # The payload still decodes back verbatim for the reader.
    embedded = _embedded_corpus(section)
    bodies = [entry["body"] for entry in embedded["entries"]]
    assert any('Quote "x" & <tag>' in body for body in bodies)


def test_js_deflect_aliases_cover_every_python_rule_alias() -> None:
    script = board_assistant.assistant_script()
    for rule in board_assistant._deflect_rule_sets():
        for group in rule:
            for alias in group:
                assert f'"{alias}"' in script, f"JS port drops deflect alias {alias!r}"


def test_js_port_shares_the_stopword_list() -> None:
    script = board_assistant.assistant_script()
    for word in board_assistant.STOPWORDS:
        assert f'"{word}"' in script, f"JS port drops stopword {word!r}"


def test_assistant_script_is_deterministic() -> None:
    assert board_assistant.assistant_script() == board_assistant.assistant_script()


def test_assistant_css_classes_are_defined_in_the_shipped_sheet() -> None:
    body_match = re.search(
        r"</head>\s*<body>(.*)</body>", board_terminal.render(build_fixture_content()), re.S
    )
    assert body_match is not None
    used = {
        cls
        for value in re.findall(r'class="([^"]*)"', body_match.group(1))
        for cls in value.split()
        if cls.startswith("assistant-")
    }
    assert used, "panel classes missing from the rendered page"
    defined = set(re.findall(r"\.(assistant-[\w-]+)", board_terminal.TERMINAL_STYLE_CSS))
    assert used <= defined, f"panel classes without CSS: {used - defined}"


def test_index_page_carries_panel_and_script() -> None:
    html = board_terminal.render(build_fixture_content())
    assert "assistant-box" in html
    assert "assistant-data" in html
    assert "Board assistant" in html
    assert board_assistant.assistant_script().strip() in html


def test_glossary_terms_are_evergreen_definitions() -> None:
    assert len(GLOSSARY) >= 8
    for item in GLOSSARY:
        assert item.term and item.definition
        assert not re.search(r"\d", item.term + item.definition), item.term


def test_build_knowledge_invents_no_numbers_for_empty_weeks() -> None:
    knowledge = build_knowledge(
        page="index.html",
        season=None,
        week=None,
        generated_at_text="2026-08-31 12:00:00 UTC",
        model_id=None,
        method_label="weak_stack",
        games=(),
        best_pick_game_id=None,
        best_pick_note=None,
        policy_text=None,
        record_lines=(),
        finding_items=(),
        watching_items=(),
    )
    assert knowledge["entries"]
    resolved = answer("hello?", knowledge)
    assert resolved.topic == "fallback"
    assert "published card" in resolved.text
