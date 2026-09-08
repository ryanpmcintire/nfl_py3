"""Run the guarded, idempotent Tuesday paper-forecast lock job.

This entry point is owned by ``capture_scheduler.py``. It deliberately has no
season/week flags: the verified schedule must identify exactly one game week
whose line-lock Tuesday is today, preventing manual backdating.

A failure here used to be diagnosable only from a doubly-truncated string:
this script kept the last 500 characters of the child's stderr, and
``capture_scheduler`` then logged the first 200 characters of the JSON line
that carried it. On 2026-09-08 that left a lock-day abort recorded as the
tail of a ``BootstrapDegeneracyWarning`` from step 4 plus the step 5 banner,
with the actual error past the cut and gone. Every failure now persists the
child's COMPLETE stdout, stderr and traceback under ``FAILURE_LOG_DIR`` and
leads the reported error with that path, so the 200-character survivor still
says where the whole story lives.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import traceback
from datetime import UTC, datetime
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

#: Where the complete output of a failed lock lands. Under ``data/`` (never
#: committed) beside the scheduler's own log, so the scheduler's one-line
#: record and the full story sit in the same tree.
FAILURE_LOG_DIR = REPO / "data" / "lock_failures"

#: A ``file.py:123: SomeWarning: ...`` header line. Its continuation line (the
#: echoed source) is indented, which ``_error_summary`` skips on its own.
_WARNING_HEADER = re.compile(r":\d+: \w*Warning: ")

#: Progress chatter ``run_weekly`` and the decision-package writer print to
#: stderr. Never the reason a run failed.
_PROGRESS_PREFIXES = ("weekly-run step ", "lock-day decision package: ")


class WeeklyRunFailure(RuntimeError):
    """``weekly-run`` exited non-zero; the complete child output is on disk."""

    def __init__(self, message: str, *, log_path: Path) -> None:
        super().__init__(message)
        self.log_path = log_path


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def write_failure_log(kind: str, sections: dict[str, str]) -> Path:
    """Persist every section verbatim and return the file written.

    Deliberately dumb: no truncation, no filtering, no parsing. The whole
    point is that the next person reading a failed lock gets the same bytes
    the child produced.
    """

    FAILURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    path = FAILURE_LOG_DIR / f"{stamp}-{kind}.log"
    body = "".join(
        f"===== {name} =====\n{(text or '(empty)').rstrip()}\n\n" for name, text in sections.items()
    )
    path.write_text(body, encoding="utf-8")
    return path


def _error_summary(stderr: str | None, *, limit: int = 240) -> str:
    """The one line that says WHY, pulled out of a stderr full of chatter.

    ``nfl_ats.cli.main`` ends a failed run with ``error: <message>``, and an
    unhandled exception ends with ``ExceptionType: message`` at column 0, so
    both survive; step banners, the decision-package line, warning headers and
    every indented continuation (warning source echoes, traceback frames) are
    skipped.
    """

    lines = [line.rstrip() for line in (stderr or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("error: "):
            return line[:limit]
    for line in reversed(lines):
        if line.startswith(_PROGRESS_PREFIXES) or line.startswith((" ", "\t")):
            continue
        if _WARNING_HEADER.search(line):
            continue
        return line[:limit]
    return "(no stderr captured)"


def _run_weekly(season: int, week: int) -> dict[str, Any]:
    command = [
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
    ]
    proc = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if proc.returncode:
        log_path = write_failure_log(
            "weekly-run",
            {
                "command": " ".join(command),
                "returncode": str(proc.returncode),
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
            },
        )
        raise WeeklyRunFailure(
            f"weekly-run failed ({proc.returncode}): {_error_summary(proc.stderr)} "
            f"[full output: {_repo_relative(log_path)}]",
            log_path=log_path,
        )
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
        log_path = getattr(error, "log_path", None)
        if not isinstance(log_path, Path):
            log_path = write_failure_log("scheduled-lock", {"traceback": traceback.format_exc()})
        # ``error_log`` before ``error`` on purpose: capture_scheduler records
        # only the first 200 characters of this line, and the path is the one
        # field that must survive that cut.
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "error_log": _repo_relative(log_path),
                    "error": str(error),
                }
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
