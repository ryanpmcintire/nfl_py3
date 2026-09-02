"""Thin CLI wrapper for ``nfl_ats.referee_assignments_capture`` (WP22).

Mirrors ``scripts/capture_inactives.py``'s own precedent (2026-09-01,
WP17): the scheduler invokes a plain script directly as a subprocess rather
than through a ``nfl-ats`` subcommand, so the actual fetch/parse/write logic
(``run_capture`` and friends) lives in the importable, testable package
module ``src/nfl_ats/referee_assignments_capture.py`` and this file exists
only to give the scheduler a stable entry point -- ``src/nfl_ats/cli.py`` is
out of scope for this work package.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/capture_referee_assignments.py \\
        --current
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.referee_assignments_capture import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
