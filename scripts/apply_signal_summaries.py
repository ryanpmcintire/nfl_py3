"""Apply plain-English summaries + categories onto ``registry/weak_signals.json``.

A separate process (not this repo's automated tooling) maintains
``registry/reference/signal_summaries.json``: a hand/agent-curated mapping of
``signal_name -> {"category": ..., "plain_summary": ...}`` written in the
words a football fan with no statistics background can read. This script is
the one-way bridge from that reference file onto the two OPTIONAL fields
``nfl_ats.weak_signals.WeakSignal.plain_summary``/``.category`` actually
carry, so the public Signal Ledger page (``docs/ledger.html``) can render
plain language instead of a raw technical description without either file
owning the other's job: the reference file owns the WORDS, this script (and
the registry) owns the SCHEMA.

Idempotent: re-running applies nothing new once every matching name is
already up to date, and never touches any field this script does not own
(``plain_summary``/``category``) -- every other field on every signal is
round-tripped byte-for-byte through ``registry_to_payload``.

Usage::

    .tools\\uv.exe run --no-sync python scripts\\apply_signal_summaries.py
    .tools\\uv.exe run --no-sync python scripts\\apply_signal_summaries.py --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from nfl_ats.weak_signals import (
    CATEGORIES,
    WeakSignalError,
    default_registry_path,
    load_registry,
    save_registry,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARIES_PATH = REPO / "registry" / "reference" / "signal_summaries.json"


def _load_summaries(path: Path) -> dict[str, Any]:
    """Return the ``signal_name -> {"category": ..., "plain_summary": ...}``
    mapping from ``path``.

    Accepts either that mapping directly at the top level, or the produced
    shape (an envelope carrying ``generated_at_utc``/``source_registry_sha256``
    provenance fields beside a ``"summaries"`` key holding the actual
    mapping) -- both are "a JSON object of signal_name -> summary" as far as
    this script's contract goes; only the second happens to carry its own
    provenance alongside it.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WeakSignalError(f"{path} must contain a JSON object of signal_name -> summary")
    summaries = payload.get("summaries")
    if isinstance(summaries, dict):
        return summaries
    return payload


def apply_signal_summaries(
    summaries_path: Path = DEFAULT_SUMMARIES_PATH,
    registry_path: Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply ``summaries_path`` onto the weak-signal registry.

    Returns a report dict: counts of applied / already-up-to-date / unmatched
    names / rejected (invalid category) entries, plus how many registry
    records still lack a ``plain_summary`` or a ``category`` afterward. Never
    raises on a missing summaries file -- that is the "not produced yet"
    state the caller (``AGENTS.md``-governed sessions) must be able to detect
    and report rather than crash on.
    """

    path = registry_path if registry_path is not None else default_registry_path()
    registry = load_registry(path)

    if not summaries_path.is_file():
        remaining = sum(
            1
            for signal in registry.signals.values()
            if signal.plain_summary is None or signal.category is None
        )
        return {
            "summaries_path": str(summaries_path),
            "summaries_found": False,
            "applied": 0,
            "already_up_to_date": 0,
            "unmatched_names": [],
            "rejected_invalid_category": [],
            "remaining_without_plain_summary_or_category": remaining,
            "total_signals": len(registry.signals),
        }

    summaries = _load_summaries(summaries_path)

    signals = dict(registry.signals)
    applied: list[str] = []
    unchanged: list[str] = []
    unmatched: list[str] = []
    rejected: list[dict[str, str]] = []

    for name, entry in sorted(summaries.items()):
        existing = signals.get(name)
        if existing is None:
            unmatched.append(name)
            continue
        if not isinstance(entry, dict):
            rejected.append({"name": name, "reason": f"summary entry is not an object: {entry!r}"})
            continue

        new_category = existing.category
        if "category" in entry and entry["category"] is not None:
            candidate_category = str(entry["category"])
            if candidate_category not in CATEGORIES:
                rejected.append(
                    {
                        "name": name,
                        "reason": (
                            f"category {candidate_category!r} is not one of {', '.join(CATEGORIES)}"
                        ),
                    }
                )
            else:
                new_category = candidate_category

        new_plain_summary = existing.plain_summary
        if "plain_summary" in entry and entry["plain_summary"] is not None:
            candidate_summary = str(entry["plain_summary"]).strip()
            if candidate_summary:
                new_plain_summary = candidate_summary

        if new_category == existing.category and new_plain_summary == existing.plain_summary:
            unchanged.append(name)
            continue

        signals[name] = dataclasses.replace(
            existing, category=new_category, plain_summary=new_plain_summary
        )
        applied.append(name)

    updated_registry = dataclasses.replace(registry, signals=signals)
    if applied and not dry_run:
        save_registry(updated_registry, path)

    remaining = sum(
        1
        for signal in updated_registry.signals.values()
        if signal.plain_summary is None or signal.category is None
    )

    return {
        "registry": str(path),
        "summaries_path": str(summaries_path),
        "summaries_found": True,
        "dry_run": dry_run,
        "applied": len(applied),
        "applied_names": applied,
        "already_up_to_date": len(unchanged),
        "unmatched_names": unmatched,
        "rejected_invalid_category": rejected,
        "remaining_without_plain_summary_or_category": remaining,
        "total_signals": len(updated_registry.signals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries",
        type=Path,
        default=DEFAULT_SUMMARIES_PATH,
        help=f"path to signal_summaries.json (default: {DEFAULT_SUMMARIES_PATH})",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="path to weak_signals.json (default: the tracked registry)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing the registry",
    )
    args = parser.parse_args()
    report = apply_signal_summaries(args.summaries, args.registry, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
