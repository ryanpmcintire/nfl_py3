"""Standalone weekly recorder for the ``bye_edge_fade_overlay`` challenger.

A WEEKLY LEDGER RECORDER, not an experiment: it has no hypothesis, no cell,
and writes nothing to ``registry/weak_signals.json``. It exists only because
this challenger's ``weekly_recording_command`` must not be
``nfl-ats publish-predictions --record-decisions`` -- that command's
challenger-result key map is asserted exactly equal to the live
``ACTIVE_PROSPECTIVE`` registry in
``tests/test_cli.py::test_publish_challenger_result_map_covers_live_active_registry``,
and ``src/nfl_ats/cli.py`` is off-limits to this build (a separate
orchestrator integration pass owns that wiring). Until that wiring lands,
this script is the standalone entry point that calls
``nfl_ats.bye_edge_fade_overlay.record_bye_edge_fade_challenger_decisions``
directly and prints its JSON result -- exactly the same shape the CLI's own
per-overlay ``try``/``except`` block would produce.

Usage (from the repo root, per AGENTS.md environment conventions)::

    .\\.tools\\uv.exe run --no-sync python scripts/record_bye_edge_fade_challenger.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from nfl_ats.bye_edge_fade_overlay import record_bye_edge_fade_challenger_decisions

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="ISO 8601 UTC instant to record at (defaults to the current time).",
    )
    args = parser.parse_args(argv)

    now = datetime.fromisoformat(args.now).astimezone(UTC) if args.now else None
    result = record_bye_edge_fade_challenger_decisions(args.artifacts_root, args.data_root, now=now)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
