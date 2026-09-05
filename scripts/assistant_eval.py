"""ENG-05: run the golden-question corpus against the REAL, currently
published assistant corpus.

Builds the This Week page's knowledge from local artifacts exactly the way
``scripts/build_full_site.py`` does (via
:func:`nfl_ats.board_site_content.load_site_content`), but keeps it in
memory only -- this script never writes a site, a knowledge file, or
anything else to disk. A forced-all-stale variant of that SAME corpus
(:func:`nfl_ats.assistant_eval.make_stale_lineup_knowledge`) is built
alongside it so the corpus's ``stale_data`` rows are graded honestly
regardless of whether the real ``lineups.json`` happens to be fresh,
stale, or absent this week.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts\\assistant_eval.py [--json]

Exits non-zero when any golden question fails, so this doubles as a
CI-style gate a release checklist can run directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.assistant_eval import (  # noqa: E402
    EvalReport,
    evaluate_golden,
    load_questions,
    make_stale_lineup_knowledge,
    render_report,
)
from nfl_ats.board_assistant import build_knowledge_for_board  # noqa: E402
from nfl_ats.board_site_content import load_site_content  # noqa: E402

_QUESTIONS_PATH = REPO_ROOT / "tests" / "fixtures" / "assistant_golden" / "questions.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="artifacts root to read the synchronized weekly forecast from (default: ./artifacts)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="data root for schedule/market snapshots (default: NFL_ATS_DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help="registry root for weak_signals.json/rotation_registry.json (default: each "
        "module's own NFL_ATS_REGISTRY_DIR-aware default)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report as JSON instead of the human-readable summary",
    )
    return parser


def _report_to_json(report: EvalReport) -> dict[str, object]:
    return {
        "total": report.total,
        "passed": report.passed,
        "overall_pass": report.overall_pass,
        "categories": [
            {
                "category": cat_report.category,
                "total": cat_report.total,
                "passed": cat_report.passed,
                "all_passed": cat_report.all_passed,
            }
            for cat_report in report.category_reports
        ],
        "failures": [
            {
                "question": result.case.question,
                "category": result.case.category,
                "expected_intent": result.case.expected_intent,
                "actual_topic": result.actual_topic,
                "actual_text": result.actual_text,
                "reasons": list(result.failure_reasons),
            }
            for result in report.failures
        ],
    }


READ_ONLY_SCRIPT = True
# ENG-29: read-only; the ENG-29 scanner confirms zero write sites -- it builds its knowledge base
# from artifacts/ in memory and prints the report (--json prints to stdout), never writing under
# artifacts/ or registry/.


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # require_fresh_arrest_overlay=False: a read-only rehearsal read, same
    # default as scripts/build_full_site.py -- this script builds knowledge
    # in memory only and writes nothing, so a stale arrest snapshot has
    # nothing real to corrupt.
    content = load_site_content(
        args.artifacts_root,
        data_root=args.data_root,
        registry_root=args.registry_root,
        require_fresh_arrest_overlay=False,
    )
    knowledge = build_knowledge_for_board(content.board, findings_page=content.findings)
    stale_knowledge = make_stale_lineup_knowledge(knowledge)
    questions = load_questions(_QUESTIONS_PATH)
    report = evaluate_golden(knowledge, questions, stale_knowledge=stale_knowledge)

    if args.json:
        print(json.dumps(_report_to_json(report), indent=2, sort_keys=True))
    else:
        print(render_report(report))

    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
