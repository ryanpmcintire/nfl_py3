from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nfl_ats.market_data import POOL_SPREAD_LOCK_ET  # noqa: E402
from scripts.capture_scheduler import ET, LINEUP_CAPTURE, _cli, _ps  # noqa: E402

READ_ONLY_SCRIPT = True

MANUAL_NOTE = "manual_refresh"


@dataclass(frozen=True)
class Step:
    name: str
    what: str
    argv: tuple[str, ...]
    skip_reason: str | None = None


def plan(now: datetime) -> tuple[Step, ...]:

    local = now.astimezone(ET)
    et_tuesday = local.weekday() == 1
    pool_locked = local.time() >= POOL_SPREAD_LOCK_ET
    odds_skip = None
    if et_tuesday and not pool_locked:
        lock = POOL_SPREAD_LOCK_ET.strftime("%H:%M")
        odds_skip = (
            f"it is Tuesday (ET) before the pool's {lock} ET spread lock: the week's "
            "opener is the earliest Tuesday capture at or after the lock, falling back "
            "to the earliest pre-lock capture when none exists, so a capture now could "
            "become the opener line; the scheduler captures the locked line at 12:05"
        )
    return (
        Step("spreads", "capture the current spreads", tuple(_ps("odds_capture.ps1")), odds_skip),
        Step(
            "lineups and injuries",
            "refresh depth charts and the player snapshot (injury reports), rebuild the forecast",
            tuple(LINEUP_CAPTURE),
        ),
        Step(
            "picks",
            "apply the late-week rule against the frozen Tuesday line and label the card",
            tuple(
                _cli(
                    "refresh-picks",
                    "--record-decisions",
                    "--publish-card",
                    "--note",
                    MANUAL_NOTE,
                )
            ),
        ),
        Step("board", "regenerate the site pages", tuple(_cli("publish-board"))),
    )


def run(steps: tuple[Step, ...], *, dry: bool = False) -> int:
    failures = 0
    for step in steps:
        if step.skip_reason:
            print(f"[skip] {step.name}: {step.skip_reason}")
            continue
        print(f"[run ] {step.name}: {step.what}")
        if dry:
            print("       " + " ".join(step.argv))
            continue
        started = perf_counter()
        completed = subprocess.run(list(step.argv), cwd=REPO, check=False)
        elapsed = perf_counter() - started
        if completed.returncode:
            failures += 1
            print(f"[FAIL] {step.name} (exit {completed.returncode}, {elapsed:.0f}s)")
        else:
            print(f"[ ok ] {step.name} ({elapsed:.0f}s)")
    if failures:
        print(f"{failures} step(s) failed; see the output above.")
    else:
        print("Everything refreshed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the card now")
    parser.add_argument("--dry", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args(argv)
    return run(plan(datetime.now(tz=ET)), dry=args.dry)


if __name__ == "__main__":
    raise SystemExit(main())
