"""Record the tank-zone fade tilt overlay's weekly arm to the challenger ledger.

Standalone entry point for
``nfl_ats.tank_zone_fade_tilt_overlay.record_tank_zone_fade_tilt_challenger_decisions``.

**Why a standalone script and not ``nfl-ats publish-predictions --record-decisions``**:
``tests/test_cli.py::test_publish_challenger_result_map_covers_live_active_registry``
requires exact equality between ``cli.PUBLISH_CHALLENGER_RESULT_KEYS`` and the
set of ACTIVE_PROSPECTIVE challengers whose ``weekly_recording_command`` names
that publish command. Until this challenger is wired into ``cli.py`` (the
pending follow-up recorded in its ``known_gap``), naming the publish command in
its registration would break that test. This script is the interim recording
path, and it carries every write-refusal guarantee the CLI path has, because
they all live inside the recorder function itself: ACTIVE_PROSPECTIVE-only,
active-model fingerprint match, pre-kickoff only, append-only (never rewrites an
existing row), and ``nfl_ats.clv.refuse_if_outside_recording_lock_window``.

This WRITES to ``artifacts/prospective/challenger_decisions.parquet`` when it
succeeds. The first real write is the 2026-09-08 Week 1 lock; before then, note
that the overlay is structurally inert in Week 1 (its window is weeks 14-18), so
a Week 1 run records the active model's own un-flipped picks as this
challenger's arm.

Usage (from the repo root, per AGENTS.md environment conventions)::

    .\\.tools\\uv.exe run --no-sync python scripts/record_tank_zone_fade_challenger.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.tank_zone_fade_tilt_overlay import (  # noqa: E402
    record_tank_zone_fade_tilt_challenger_decisions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args(argv)

    result = record_tank_zone_fade_tilt_challenger_decisions(args.artifacts_root, args.data_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
