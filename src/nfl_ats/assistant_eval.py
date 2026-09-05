"""ENG-05: golden-question evaluation for the board assistant.

A broader fixture/evaluation suite for intent routing, unsupported-question
fallback, numeric provenance, stale-data behaviour, and (together with the
accessibility markup assertions in ``tests/test_assistant_golden.py``) the
keyboard/no-JS accessibility contract -- see ROADMAP.md Phase 13, ENG-05.

This module never opens an artifact or a network resource: everything here
is a pure function over an already-built knowledge mapping (the same
``dict`` :func:`nfl_ats.board_assistant.build_knowledge_for_board` and its
siblings return) and a fixed corpus of golden-question rows loaded from
``tests/fixtures/assistant_golden/questions.json``.

``scripts/assistant_eval.py`` is the thin CLI that builds the real corpus
from local artifacts and calls :func:`evaluate_golden`;
``tests/test_assistant_golden.py`` calls it against synthetic,
``build_fixture_content``-derived knowledge. Both share this one engine so
the golden corpus is graded identically in CI and by hand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nfl_ats.board_assistant import answer

#: The only categories a golden-question row may carry -- keeps the JSON
#: fixture and this module's per-category reporting in lockstep; an
#: unrecognized category is a fixture bug (caught by :func:`load_questions`),
#: not a silently-ignored row.
CATEGORIES: tuple[str, ...] = (
    "routing",
    "unsupported_fallback",
    "numeric_provenance",
    "stale_data",
    "lineup",
    "accessibility_text",
)

#: Substrings that mark a number in an answer as source-anchored -- a named
#: probability/record/timestamp origin, not a bare digit. Matched
#: case-insensitively against the FULL answer text (not positionally next to
#: the number): every numeric answer template in
#: ``nfl_ats.board_assistant``/``nfl_ats.board_assistant_lineups`` already
#: puts its anchor phrase in the same sentence as its number, so a
#: text-level check is precise enough without brittle proximity parsing, and
#: stays honest because this marker list is closed and reviewed here, not
#: inferred at check time.
PROVENANCE_MARKERS: tuple[str, ...] = (
    "cover probability",
    "as of ",
    "opener-graded",
    "season-blocked",
    "week-blocked",
    "paired games",
    "play probability",
    "probability positive",
    "95% ci",
    "most confident",
    "least confident",
)

_NUMBER_RE = re.compile(r"\d")


def has_provenance_anchor(text: str) -> bool:
    """Whether ``text`` carries at least one recognized provenance marker
    (see :data:`PROVENANCE_MARKERS`), case-insensitively."""

    lowered = text.lower()
    return any(marker in lowered for marker in PROVENANCE_MARKERS)


@dataclass(frozen=True)
class QuestionCase:
    """One golden-question row."""

    question: str
    expected_intent: str
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class CaseResult:
    """One evaluated row: the case, the actual answer, and why it failed
    (empty when it passed)."""

    case: QuestionCase
    actual_topic: str
    actual_text: str
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CategoryReport:
    category: str
    total: int
    passed: int

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total


@dataclass(frozen=True)
class EvalReport:
    """The full run: totals, one :class:`CategoryReport` per category
    present in the corpus, and every individual result (pass and fail) so
    callers can inspect specific questions without re-running anything."""

    total: int
    passed: int
    category_reports: tuple[CategoryReport, ...]
    results: tuple[CaseResult, ...]

    @property
    def overall_pass(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(result for result in self.results if not result.passed)


def load_questions(path: Path) -> tuple[QuestionCase, ...]:
    """Load and validate the golden-question fixture at ``path`` (a JSON
    array of objects; see ``tests/fixtures/assistant_golden/questions.json``
    for the shape). Raises ``ValueError`` on a malformed row or an
    unrecognized ``category`` -- a fixture bug should fail loud, not be
    silently dropped from the report."""

    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: golden question fixture must be a JSON array")
    cases: list[QuestionCase] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}[{index}]: row must be a JSON object")
        category = str(row["category"])
        if category not in CATEGORIES:
            raise ValueError(
                f"{path}[{index}]: unknown category {category!r}, expected one of {CATEGORIES}"
            )
        cases.append(
            QuestionCase(
                question=str(row["question"]),
                expected_intent=str(row["expected_intent"]),
                must_contain=tuple(str(item) for item in row.get("must_contain", ())),
                must_not_contain=tuple(str(item) for item in row.get("must_not_contain", ())),
                category=category,
            )
        )
    return tuple(cases)


def make_stale_lineup_knowledge(
    knowledge: Mapping[str, Any], *, as_of: str = "2000-01-01T00:00:00Z"
) -> dict[str, Any]:
    """A deep copy of ``knowledge`` with every published lineup entry aged
    past the documented staleness budget
    (``nfl_ats.board_assistant_lineups.LINEUP_STALE_BUDGET_HOURS``).

    Forces ``stale: True`` directly on every game-side and player entry --
    the exact boolean
    :func:`nfl_ats.board_assistant_lineups.build_lineup_knowledge` would
    compute from an ``as_of`` this old, so every lineup-derived answer
    function (which gates purely on that precomputed boolean, per that
    module's "never re-derive at query time" contract) degrades to its
    stale fallback exactly as it would from a real stale artifact.
    ``as_of`` is overwritten too so the rendered anchor text
    ("as of ... from ...") reads consistently old, not just the gating
    flag. A knowledge mapping with no ``"lineups"`` block (or an empty one)
    is returned unchanged -- there is nothing to age.
    """

    clone: dict[str, Any] = json.loads(json.dumps(knowledge))
    lineups = clone.get("lineups")
    if not lineups:
        return clone
    for sides in lineups.get("games", {}).values():
        for side in ("home", "away"):
            entry = sides.get(side)
            if entry is None:
                continue
            entry["stale"] = True
            entry["as_of"] = as_of
    for player in lineups.get("players", ()):
        player["stale"] = True
        player["as_of"] = as_of
    return clone


def _check_case(
    case: QuestionCase,
    knowledge: Mapping[str, Any],
    stale_knowledge: Mapping[str, Any] | None,
) -> CaseResult:
    source = knowledge
    if case.category == "stale_data" and stale_knowledge is not None:
        source = stale_knowledge
    resolved = answer(case.question, source)
    reasons: list[str] = []
    if resolved.topic != case.expected_intent:
        reasons.append(f"expected intent {case.expected_intent!r}, got {resolved.topic!r}")
    for needle in case.must_contain:
        if needle not in resolved.text:
            reasons.append(f"missing required text {needle!r}")
    for needle in case.must_not_contain:
        if needle in resolved.text:
            reasons.append(f"forbidden text {needle!r} present")
    if case.category == "numeric_provenance":
        if not _NUMBER_RE.search(resolved.text):
            reasons.append("numeric_provenance case has no number in its answer")
        elif not has_provenance_anchor(resolved.text):
            reasons.append("number present without a recognized provenance anchor")
    return CaseResult(
        case=case,
        actual_topic=resolved.topic,
        actual_text=resolved.text,
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def evaluate_golden(
    knowledge: Mapping[str, Any],
    questions: Sequence[QuestionCase],
    *,
    stale_knowledge: Mapping[str, Any] | None = None,
) -> EvalReport:
    """Run every row in ``questions`` against ``knowledge`` and report
    pass/fail per category plus every result with its actual answer.

    Rows tagged ``category="stale_data"`` are graded against
    ``stale_knowledge`` instead, when supplied (see
    :func:`make_stale_lineup_knowledge`) -- staleness is a property of a
    DIFFERENT snapshot in time, not of the primary corpus, so those rows
    need their own knowledge object exactly as the ENG-05 spec describes.
    When ``stale_knowledge`` is omitted, ``stale_data`` rows fall back to
    ``knowledge`` itself (useful only if the caller already knows that
    corpus is stale; :mod:`scripts.assistant_eval` always builds and passes
    a forced-stale variant so this path never silently under-tests staleness
    against a real, unpublished, or already-fresh corpus).

    Rows tagged ``category="numeric_provenance"`` get one extra, automatic
    check beyond ``must_contain``/``must_not_contain``: the answer must
    contain a digit AND a recognized provenance marker (see
    :func:`has_provenance_anchor`) -- the report can never call a
    numeric_provenance case a pass if its answer prints a number with no
    source anchor.
    """

    results = tuple(_check_case(case, knowledge, stale_knowledge) for case in questions)
    by_category: dict[str, list[CaseResult]] = {}
    for result in results:
        by_category.setdefault(result.case.category, []).append(result)
    category_reports = tuple(
        CategoryReport(
            category=category,
            total=len(rows),
            passed=sum(1 for row in rows if row.passed),
        )
        for category, rows in sorted(by_category.items())
    )
    return EvalReport(
        total=len(results),
        passed=sum(1 for result in results if result.passed),
        category_reports=category_reports,
        results=results,
    )


def render_report(report: EvalReport) -> str:
    """A short, deterministic text summary: overall pass/fail, one line per
    category, then every failing question with its expectation and the
    actual answer -- meant to be read directly off a CI log or a terminal,
    never parsed."""

    lines = [
        f"Golden assistant eval: {report.passed}/{report.total} passed "
        f"({'PASS' if report.overall_pass else 'FAIL'})",
    ]
    for cat_report in report.category_reports:
        status = "ok" if cat_report.all_passed else "FAIL"
        lines.append(f"  {cat_report.category}: {cat_report.passed}/{cat_report.total} [{status}]")
    failures = report.failures
    if failures:
        lines.append("")
        lines.append(f"{len(failures)} failing question(s):")
        for result in failures:
            lines.append(f"  [{result.case.category}] {result.case.question!r}")
            lines.append(
                f"    expected intent {result.case.expected_intent!r}, got {result.actual_topic!r}"
            )
            lines.append(f"    answer: {result.actual_text!r}")
            for reason in result.failure_reasons:
                lines.append(f"    - {reason}")
    return "\n".join(lines)


__all__ = [
    "CATEGORIES",
    "PROVENANCE_MARKERS",
    "CaseResult",
    "CategoryReport",
    "EvalReport",
    "QuestionCase",
    "evaluate_golden",
    "has_provenance_anchor",
    "load_questions",
    "make_stale_lineup_knowledge",
    "render_report",
]
