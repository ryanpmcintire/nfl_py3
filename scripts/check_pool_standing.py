from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.market_data import POOL_TIMEZONE  # noqa: E402
from nfl_ats.snapshots import latest_snapshot, load_verified_snapshot  # noqa: E402

READ_ONLY_SCRIPT = True

POINTS_NAME = re.compile(r"^(?P<season>\d{4})_weekly_points_w01-(?P<week>\d{2})\.parquet$")
REGULAR_SEASON = "REG"


def newest_points_file(field_root: Path, season: int | None) -> tuple[int, int, Path] | None:
    found: list[tuple[int, int, Path]] = []
    for path in field_root.glob("*_weekly_points_w01-*.parquet"):
        match = POINTS_NAME.match(path.name)
        if match is None:
            continue
        found.append((int(match["season"]), int(match["week"]), path))
    if season is not None:
        found = [item for item in found if item[0] == season]
    return max(found, key=lambda item: (item[0], item[1])) if found else None


def last_settled_week(schedules: pd.DataFrame, season: int, as_of: date) -> int:
    games = schedules.loc[
        schedules["season"].astype(int).eq(season)
        & schedules["game_type"].astype(str).eq(REGULAR_SEASON)
    ]
    last_day = pd.to_datetime(games["gameday"], errors="coerce").dt.date.groupby(games["week"])
    finished = last_day.max()
    settled = finished.loc[finished.lt(as_of)]
    return int(settled.index.max()) if not settled.empty else 0


def check(season: int | None, as_of: date, data_root: Path) -> dict[str, object]:
    newest = newest_points_file(data_root / "splash" / "field", season)
    if newest is None:
        return {
            "season": season,
            "as_of": as_of.isoformat(),
            "problems": ["No weekly points file exists under data/splash/field."],
        }
    file_season, file_week, path = newest
    schedules, _ = load_verified_snapshot(latest_snapshot(data_root / "raw"))
    settled = last_settled_week(schedules, file_season, as_of)
    problems: list[str] = []
    if file_week < settled:
        problems.append(
            f"The pool standing panel stops at Week {file_week} but Week {settled} of "
            f"{file_season} is settled as of {as_of.isoformat()}. Read Weekly results and Pick "
            "Distribution from Splash and rerun the pool refresh in "
            "docs/lanes/pool-standing-refresh.md."
        )
    return {
        "season": file_season,
        "points_file": path.name,
        "points_last_week": file_week,
        "last_settled_week": settled,
        "as_of": as_of.isoformat(),
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when the pool standing files are behind the last settled week."
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--as-of", type=date.fromisoformat, default=None, help="YYYY-MM-DD")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    args = parser.parse_args(argv)
    as_of = args.as_of or datetime.now(tz=POOL_TIMEZONE).date()
    result = check(args.season, as_of, args.data_root)
    print(json.dumps(result))
    if result["problems"]:
        print("; ".join(result["problems"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
