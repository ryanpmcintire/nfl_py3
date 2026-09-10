from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nfl_ats.board_assistant import answer

CATEGORIES: tuple[str, ...] = (
    "routing",
    "unsupported_fallback",
    "numeric_provenance",
    "stale_data",
    "lineup",
    "accessibility_text",
)

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

    lowered = text.lower()
    return any(marker in lowered for marker in PROVENANCE_MARKERS)


@dataclass(frozen=True)
class QuestionCase:
    question: str
    expected_intent: str
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class CaseResult:
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
