"""Thin CLI wrapper for ``nfl_ats.inactives_capture`` (WP17).

Mirrors how the existing NFL.com injury-report capture is wired into the
scheduler: ``scripts/capture_scheduler.py`` invokes
``scripts/ingest_nflcom_injuries.py --current`` directly as a subprocess
rather than through a ``nfl-ats`` subcommand (see that file's
``INJURY_CAPTURE`` constant). The inactives capture follows the same pattern
-- a plain script the scheduler can call with ``--current --slot <name>`` --
so this file exists ONLY to keep the actual fetch/parse/write logic
(``run_capture`` and friends) in the importable, testable package module
``src/nfl_ats/inactives_capture.py``, per WP17's scope: ``src/nfl_ats/cli.py``
is off-limits for this work package, so wiring through ``nfl-ats`` is not an
option even though it might otherwise be the more discoverable path.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/capture_inactives.py \\
        --current --slot sun_early
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.inactives_capture import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
