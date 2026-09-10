from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.referee_assignments_capture import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
