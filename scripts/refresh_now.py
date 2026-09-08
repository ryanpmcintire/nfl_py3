"""One-click refresh: spreads, lineups and injury reports, then the picks and the board.

Owner, 2026-09-08: "it would be awesome if i had a button to refresh the
lineups/injuryreports/spreads all at once". This is that button's engine;
``scripts/refresh_now.cmd`` is the double-clickable face of it.

It runs, in order, the SAME commands the scheduler already runs on its own
clock -- nothing here is a new code path:

1. ``odds_capture.ps1`` -- a fresh point-in-time spread capture (The Odds
   API, 3 requests). Skipped on a Tuesday before the 12:05 opener window has
   passed (the pool locks its spreads at noon), because the card's opener is
   the EARLIEST Tuesday quote per book
   (``nfl_ats.market_data.tuesday_opener_quotes``) and a button press at
   10:30 would silently become the week's opener line.
2. ``refresh_lineup_forecast.py`` -- current depth charts, then a
   ``weekly-run`` with ``--refresh-player-data`` (the nflverse player
   snapshot, which is where the injury reports the model reads come from),
   regenerating the forecast and the board. About fifteen minutes. The
   NFL.com injury-page capture is not run: it is paused by the MKT-09 source
   policy (``injuries_*`` jobs in ``capture_scheduler.py``).
3. ``refresh-picks --record-decisions --publish-card --note manual_refresh``
   -- the served late-week rule against the frozen Tuesday line, appending
   to the pick-revision ledger and labelling changed picks on the card. It
   refuses to write outside the recording window; that refusal is reported,
   not hidden.
4. ``publish-board`` -- the site pages, so the dashboard shows the result.

Each step's outcome is printed in plain words and the exit code is non-zero
if any step failed. Steps run even when an earlier one failed: a dead odds
key must not stop the lineups from refreshing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.capture_scheduler import ET, LINEUP_CAPTURE, _cli, _ps  # noqa: E402

READ_ONLY_SCRIPT = True

#: Tuesday-opener guard: the capture scheduler's ``odds_tue_open`` window
#: opens at 12:05 ET, just after the pool locks its spreads at noon; until
#: then a manual capture would BE the opener (the earliest Tuesday quote).
OPENER_HOUR, OPENER_MINUTE = 12, 5
MANUAL_NOTE = "manual_refresh"


@dataclass(frozen=True)
class Step:
    name: str
    what: str
    argv: tuple[str, ...]
    skip_reason: str | None = None


def plan(now: datetime) -> tuple[Step, ...]:
    """The four steps, with the Tuesday-opener guard applied for ``now`` (ET)."""

    local = now.astimezone(ET)
    # ``tuesday_opener_quotes`` keys "Tuesday" on the UTC day, which starts at
    # 20:00 ET (EDT): a Monday-evening press is already Tuesday to it.
    utc_tuesday = now.astimezone(UTC).weekday() == 1
    opener_landed = local.weekday() == 1 and (local.hour, local.minute) >= (
        OPENER_HOUR,
        OPENER_MINUTE,
    )
    odds_skip = None
    if utc_tuesday and not opener_landed:
        odds_skip = (
            "it is Tuesday (UTC) before the 12:05 ET opener capture: the week's opener "
            "is the earliest Tuesday capture, so a capture now would become the opener "
            "line; the scheduler captures the pool's locked line at 12:05"
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args(argv)
    return run(plan(datetime.now(tz=ET)), dry=args.dry)


if __name__ == "__main__":
    raise SystemExit(main())
