"""Run the guarded, idempotent Tuesday paper-forecast lock job.

This entry point is owned by ``capture_scheduler.py``. It deliberately has no
season/week flags: the verified schedule must identify exactly one game week
whose line-lock Tuesday is today, preventing manual backdating.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from nfl_ats.scheduled_lock import execute_scheduled_lock  # noqa: E402
from nfl_ats.snapshots import latest_snapshot, load_verified_snapshot  # noqa: E402
from scripts.lockday_verify import verify  # noqa: E402

ET = ZoneInfo("America/New_York")
UV = REPO / ".tools" / "uv.exe"


def _run_weekly(season: int, week: int) -> dict[str, Any]:
    proc = subprocess.run(
        [
            str(UV),
            "run",
            "--no-sync",
            "nfl-ats",
            "weekly-run",
            "--season",
            str(season),
            "--week",
            str(week),
            "--record-decisions",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"weekly-run failed ({proc.returncode}): {proc.stderr[-500:]}")
    return json.loads(proc.stdout)


def main() -> int:
    try:
        schedules, _ = load_verified_snapshot(latest_snapshot(REPO / "data" / "raw"))
        result = execute_scheduled_lock(
            schedules,
            artifacts_root=REPO / "artifacts",
            now=datetime.now(tz=ET),
            weekly_runner=_run_weekly,
            verifier=lambda season, week, summary: verify(
                REPO / "artifacts", season=season, week=week, run_summary=summary
            ),
        )
    except Exception as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
