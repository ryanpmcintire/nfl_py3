"""ENG-08 timing-policy instrumentation: read-only refresh-trigger scan.

Reconstructs every refresh trigger this project can currently observe for one
(season, week) -- the fixed clock checkpoints (from the scheduler's own state
file) plus the real non-clock events (an inactives snapshot posting, a new
injury-report snapshot, a projected-lineup change, an opener-vs-current line
move) -- and appends them to the append-only, idempotent JSONL evidence log
under ``artifacts/refresh_triggers/<season>/week_<n>.jsonl`` (gitignored,
matching every other generated artifact in this repo).

The actual detection logic lives in the importable, testable package module
``src/nfl_ats/refresh_triggers.py``; this script only resolves paths and
season/week and prints a summary, mirroring
``scripts/capture_referee_assignments.py``'s / ``scripts/capture_inactives.py``'s
own thin-wrapper precedent.

**This script never runs ``refresh-picks``, ``publish-predictions``, or any
``weak-signals``/``rotation`` recorder, and never writes to ``registry/``.**
It is read-only against every capture directory it scans and only ever
appends to its own evidence log.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/refresh_trigger_log.py \\
        --scan --current
    .\\.tools\\uv.exe run --no-sync python scripts/refresh_trigger_log.py \\
        --scan --season 2026 --week 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from nfl_ats.refresh_triggers import (  # noqa: E402
    TriggerScanRoots,
    append_triggers_to_evidence_log,
    archive_lineup_snapshot,
    detect_all_triggers,
    evidence_log_path,
)
from scripts.ingest_nflcom_injuries import resolve_current_reg_week  # noqa: E402


def _load_scheduler_state(repo: Path) -> dict[str, object]:
    path = repo / "data" / "scheduler_state.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true", required=True, help="run the read-only scan")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument(
        "--current",
        action="store_true",
        help="resolve the live (season, REG week) from the newest schedule snapshot",
    )
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--artifacts-root", type=Path, default=REPO / "artifacts")
    args = parser.parse_args(argv)

    if not args.current and (args.season is None or args.week is None):
        raise SystemExit("pass --current, or both --season and --week")

    if args.current:
        season, week = resolve_current_reg_week(REPO, pd.Timestamp.now(tz="UTC"))
    else:
        season, week = int(args.season), int(args.week)

    scheduler_state = _load_scheduler_state(REPO)

    lineup_archive_dir = args.artifacts_root / "refresh_triggers" / "_lineup_archive"
    stable_lineup = args.artifacts_root / "lineups" / "current" / "lineups.json"
    archive_lineup_snapshot(stable_lineup, lineup_archive_dir)

    roots = TriggerScanRoots(
        repo_root=REPO,
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        lineup_archive_dir=lineup_archive_dir,
        scheduler_state=scheduler_state,
    )
    triggers = detect_all_triggers(roots, season=season, week=week)

    log_path = evidence_log_path(args.artifacts_root, season=season, week=week)
    written, skipped = append_triggers_to_evidence_log(log_path, triggers)

    by_source: dict[str, int] = {}
    for trigger in triggers:
        by_source[trigger.trigger_source] = by_source.get(trigger.trigger_source, 0) + 1
    invalid = sum(1 for trigger in triggers if not trigger.deadline_valid)

    print(f"season={season} week={week}")
    print(f"evidence log: {log_path}")
    print(
        f"reconstructed {len(triggers)} trigger(s); appended {written} new, "
        f"skipped {skipped} already-logged duplicate(s)"
    )
    print(f"deadline_valid=False (excluded from any future comparison): {invalid}")
    for source in sorted(by_source):
        print(f"  {source}: {by_source[source]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
