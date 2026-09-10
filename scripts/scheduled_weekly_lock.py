from __future__ import annotations

import argparse
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

from nfl_ats.scheduled_lock import execute_scheduled_lock, resolve_lock_target  # noqa: E402
from nfl_ats.snapshots import latest_snapshot, load_verified_snapshot  # noqa: E402
from scripts.check_splash_board import PoolBoardMissing, require_captured_board  # noqa: E402
from scripts.lockday_verify import verify  # noqa: E402

ET = ZoneInfo("America/New_York")
UV = REPO / ".tools" / "uv.exe"

FAILURE_LOG_DIR = REPO / "data" / "lock_failures"

_WARNING_HEADER = re.compile(r":\d+: \w*Warning: ")

_PROGRESS_PREFIXES = ("weekly-run step ", "lock-day decision package: ")


class WeeklyRunFailure(RuntimeError):
    def __init__(self, message: str, *, log_path: Path) -> None:
        super().__init__(message)
        self.log_path = log_path


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def write_failure_log(kind: str, sections: dict[str, str]) -> Path:

    FAILURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    path = FAILURE_LOG_DIR / f"{stamp}-{kind}.log"
    body = "".join(
        f"===== {name} =====\n{(text or '(empty)').rstrip()}\n\n" for name, text in sections.items()
    )
    path.write_text(body, encoding="utf-8")
    return path


def _error_summary(stderr: str | None, *, limit: int = 240) -> str:

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


def _run_weekly(season: int, week: int, *, replace: bool = False) -> dict[str, Any]:
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
        *(["--replace-week"] if replace else []),
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the guarded Tuesday paper-forecast lock.")
    parser.add_argument("--season", type=int, default=None, help="lock this week on any day")
    parser.add_argument("--week", type=int, default=None, help="lock this week on any day")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="re-record a week that already has rows (previous ledger kept as a .bak)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    now = datetime.now(tz=ET)
    try:
        schedules, _ = load_verified_snapshot(latest_snapshot(REPO / "data" / "raw"))
        target = resolve_lock_target(schedules, now=now, season=args.season, week=args.week)
        require_captured_board(REPO / "data", target.season, target.week)
        result = execute_scheduled_lock(
            schedules,
            artifacts_root=REPO / "artifacts",
            now=now,
            weekly_runner=_run_weekly,
            season=args.season,
            week=args.week,
            replace=args.replace,
            verifier=lambda season, week, summary: verify(
                REPO / "artifacts", season=season, week=week, run_summary=summary
            ),
        )
    except PoolBoardMissing as error:
        print(str(error), file=sys.stderr)
        print(json.dumps({"status": "missing_pool_board", "error": str(error)}))
        return 1
    except Exception as error:
        log_path = getattr(error, "log_path", None)
        if not isinstance(log_path, Path):
            log_path = write_failure_log("scheduled-lock", {"traceback": traceback.format_exc()})
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
