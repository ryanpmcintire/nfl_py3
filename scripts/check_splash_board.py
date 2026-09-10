from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.market_data import POOL_TIMEZONE  # noqa: E402
from nfl_ats.scheduled_lock import resolve_lock_target  # noqa: E402
from nfl_ats.snapshots import latest_snapshot, load_verified_snapshot  # noqa: E402
from nfl_ats.splash_lines import (  # noqa: E402
    SPLASH_SUBDIRECTORY,
    SplashCapture,
    load_splash_capture,
)

READ_ONLY_SCRIPT = True

BOARD_CAPTURE_WINDOW_ET = "12:00-12:05 ET Tue"


class PoolBoardMissing(RuntimeError):
    pass


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def lock_tuesday(now: datetime) -> datetime:

    return now + timedelta(days=(1 - now.weekday()) % 7)


def resolve_board_week(
    schedules: pd.DataFrame,
    *,
    now: datetime,
    season: int | None = None,
    week: int | None = None,
) -> tuple[int, int]:

    named = season is not None or week is not None
    target = resolve_lock_target(
        schedules, now=now if named else lock_tuesday(now), season=season, week=week
    )
    return target.season, target.week


def capture_command(season: int, week: int) -> str:

    return (
        ".tools\\uv.exe run --no-sync python scripts\\capture_splash_lines.py "
        f"--season {season} --week {week} --text-file board.txt"
    )


def require_captured_board(data_root: Path, season: int, week: int) -> SplashCapture:

    where = f"{season} week {week}"
    try:
        capture = load_splash_capture(Path(data_root), season, week)
    except DataContractError as error:
        raise PoolBoardMissing(
            f"pool board for {where} does not validate: {_one_line(error)} -- read the Splash "
            f"board ({BOARD_CAPTURE_WINDOW_ET}) and rewrite the capture: "
            f"{capture_command(season, week)} --replace"
        ) from error
    if capture is None:
        raise PoolBoardMissing(
            f"pool board never captured for {where}: data/{SPLASH_SUBDIRECTORY}/ has no "
            f"{season}_week{week:02d}_*.json, so the lock would grade on lines the pool never "
            f"posted. Read the board {BOARD_CAPTURE_WINDOW_ET}, then run: "
            f"{capture_command(season, week)}"
        )
    return capture


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the pool board against the card")
    parser.add_argument("--season", type=int, default=None, help="check this week on any day")
    parser.add_argument("--week", type=int, default=None, help="check this week on any day")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "data",
        help=f"repository data root; captures live in <data-root>/{SPLASH_SUBDIRECTORY}/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        schedules, _ = load_verified_snapshot(latest_snapshot(Path(args.data_root) / "raw"))
        season, week = resolve_board_week(
            schedules,
            now=datetime.now(tz=POOL_TIMEZONE),
            season=args.season,
            week=args.week,
        )
        capture = require_captured_board(args.data_root, season, week)
    except PoolBoardMissing as error:
        print(_one_line(error), file=sys.stderr)
        return 1
    except (DataContractError, FileNotFoundError, ValueError) as error:
        print(
            f"cannot tell whether the pool board is captured: {_one_line(error)}",
            file=sys.stderr,
        )
        return 1
    name = capture.path.name if capture.path is not None else "(unnamed capture)"
    print(
        f"pool board captured for {capture.season} week {capture.week}: {name}, "
        f"{len(capture.games)} games, read {capture.captured_at_et.isoformat()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
