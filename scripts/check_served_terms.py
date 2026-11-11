from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nfl_ats import pool_hook  # noqa: E402
from nfl_ats.active_model import load_active_ats_model  # noqa: E402
from nfl_ats.market_data import POOL_TIMEZONE  # noqa: E402
from nfl_ats.pick_probability import (  # noqa: E402
    MOVE_AVAILABLE_COLUMN,
    load_pick_probability_model,
    market_move_toward_home,
)
from nfl_ats.splash_lines import load_splash_capture  # noqa: E402

TUESDAY_LOCK_CHECK_MINUTES = 12 * 60 + 30


def _newest_capture_week(data_root: Path, now: pd.Timestamp) -> tuple[int, int] | None:
    newest: tuple[pd.Timestamp, int, int] | None = None
    live: tuple[pd.Timestamp, int, int] | None = None
    for path in sorted((data_root / "splash").glob("*_week*_*.json")):
        stem = path.stem.split("_")
        season = int(stem[0])
        week = int(stem[1].removeprefix("week"))
        capture = load_splash_capture(data_root, season, week)
        if capture is None:
            continue
        stamp = pd.Timestamp(capture.captured_at_et)
        if newest is None or stamp > newest[0]:
            newest = (stamp, season, week)
        pending = any(pd.Timestamp(game.kickoff_et) > now for game in capture.games)
        if pending and (live is None or stamp > live[0]):
            live = (stamp, season, week)
    chosen = live or newest
    return None if chosen is None else (chosen[1], chosen[2])


def _forecast_rows(artifacts_root: Path) -> pd.DataFrame:
    manifest = load_active_ats_model(artifacts_root)
    if manifest is None:
        raise SystemExit("no active ATS model manifest")
    directory = artifacts_root / str(manifest["weekly_forecast"]["artifact"])
    rows = pd.read_csv(directory / "recommendations.csv")
    if "method" in rows.columns:
        rows = rows.loc[rows["method"].eq(manifest["method"])]
    return rows.drop_duplicates("game_id").reset_index(drop=True)


def check(season: int | None, week: int | None, now: datetime) -> dict[str, object]:
    data_root = REPO / "data"
    artifacts_root = REPO / "artifacts"
    now_ts = pd.Timestamp(now).tz_convert("UTC")
    if season is None or week is None:
        found = _newest_capture_week(data_root, now_ts)
        if found is None:
            raise SystemExit("no pool board capture under data/splash")
        season, week = found
    model = load_pick_probability_model(artifacts_root)
    rows = _forecast_rows(artifacts_root)
    rows = rows.loc[rows["season"].eq(season) & rows["week"].eq(week)].reset_index(drop=True)
    kickoff = pd.to_datetime(rows["kickoff"], utc=True, errors="coerce")
    first_kickoff = kickoff.min()
    pending = rows.loc[kickoff.gt(now_ts)].reset_index(drop=True)
    problems: list[str] = []
    anchors: dict[str, float] = {}
    differs = 0
    nonzero = 0
    move_games = 0
    if not pending.empty:
        push_table = dict(model.hook_push_mass)
        game_ids = pending["game_id"].astype(str).tolist()
        anchors = pool_hook.tuesday_market_lines(data_root, season, week, game_ids)
        pool = pd.to_numeric(pending["spread_line"], errors="coerce")
        for game, line in zip(game_ids, pool, strict=True):
            anchor = anchors.get(game)
            if anchor is not None and float(anchor) != float(line):
                differs += 1
        hook = pool_hook.served_hook_frame(pending, data_root, push_table)
        nonzero = int(
            (pd.to_numeric(hook[pool_hook.HOOK_PUSH_MASS_COLUMN], errors="coerce") > 0).sum()
        )
        move = market_move_toward_home(
            pending,
            data_root,
            now=now_ts.to_pydatetime(),
            feature_version=model.market_move_feature_version,
        )
        move_games = int((pd.to_numeric(move[MOVE_AVAILABLE_COLUMN], errors="coerce") > 0).sum())
    capture = load_splash_capture(data_root, season, week)
    if model.has_pool_hook and differs > 0 and nonzero == 0:
        problems.append(
            f"The pool-line offset term is zero on every game even though {differs} games have "
            "a pool line different from the market anchor."
        )
    if pd.notna(first_kickoff) and not pending.empty:
        first_et = first_kickoff.tz_convert(POOL_TIMEZONE)
        tuesday = (first_et - pd.Timedelta(days=(first_et.weekday() - 1) % 7)).normalize()
        now_et = now_ts.tz_convert(POOL_TIMEZONE)
        tuesday_check = tuesday + pd.Timedelta(minutes=TUESDAY_LOCK_CHECK_MINUTES)
        if capture is not None and now_et >= tuesday_check and not anchors:
            problems.append(
                "The pool board is captured and it is past Tuesday 12:30 Eastern, "
                "but no game has a market anchor for the offset term."
            )
        if now_et >= tuesday + timedelta(days=2) and move_games == 0:
            problems.append(
                "It is Thursday or later of the game week and no game has a line-move input."
            )
    return {
        "season": season,
        "week": week,
        "games": len(pending),
        "anchors_available": len(anchors),
        "pool_differs_from_anchor": differs,
        "offset_term_nonzero": nonzero,
        "move_available_games": move_games,
        "checked_at_utc": now_ts.isoformat(),
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when a served probability term is silent for the live week."
    )
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    args = parser.parse_args()
    result = check(args.season, args.week, datetime.now(UTC))
    print(json.dumps(result))
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
