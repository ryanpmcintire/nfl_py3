"""How far the line moved before the pool's noon lock on one Tuesday (OPS-05).

Owner, 2026-09-08: "Spreads lock: Tue, Sep 8, 2026, 12:00 PM". The pool's
spreads are fixed at noon ET and captured by the ``odds_tue_open`` job at
12:05; the live opener (``nfl_ats.market_data.tuesday_opener_quotes``) is,
per book, the earliest Tuesday quote at or after that lock
(``nfl_ats.market_data.POOL_SPREAD_LOCK_ET``), with the earliest pre-lock
quote only as the fallback. This read-only report answers the OPS-05
question for one Tuesday: per game, the EARLIEST pre-lock capture that day
(a 09:00 legacy capture, a manual press) against the FIRST post-lock capture,
and the move between them -- how far the line moved before the pool locked
it. The served opener and its ``opener_basis`` sit beside them so the reader
can see which of the two the card's consumers actually used.

A day with no pre-lock capture (the normal case now that no scheduled job
captures before the lock) has nothing to compare against and says so; a day
whose post-lock capture has not landed yet says that instead. Neither is
padded with a zero.

Days are Eastern calendar days (``nfl_ats.market_data.pool_calendar_day``),
the same convention the opener rule uses, so a Monday-night game (00:15Z
Tuesday kickoff) is reported on the Tuesday BEFORE it like every other game
of its week.

Usage::

    uv run --no-sync python scripts/tuesday_line_gap.py             # today's Tuesday
    uv run --no-sync python scripts/tuesday_line_gap.py --date 2026-09-01
    uv run --no-sync python scripts/tuesday_line_gap.py --lock 12:00  # override the ET lock

Prints a table and a JSON summary; writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes  # noqa: E402
from nfl_ats.market_data import (  # noqa: E402
    POOL_SPREAD_LOCK_ET,
    POOL_TIMEZONE,
    QUOTE_COLUMNS,
    own_week_tuesday,
    pool_calendar_day,
    tuesday_opener_quotes,
)

READ_ONLY_SCRIPT = True
ET = POOL_TIMEZONE
DEFAULT_LOCK = POOL_SPREAD_LOCK_ET.strftime("%H:%M")

COLUMNS = [
    "game_id",
    "pre_lock",
    "pre_lock_at",
    "post_lock",
    "post_lock_at",
    "move",
    "opener",
    "opener_basis",
]

NO_PRE_LOCK_MESSAGE = (
    "no pre-lock capture that day: nothing to compare the locked line against "
    "(the first capture landed at or after the lock)"
)
NO_POST_LOCK_MESSAGE = "no post-lock capture that day yet: the locked line has not been captured"


def home_spreads(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    spreads = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["outcome_side"].eq("HOME")
        & quotes["nflverse_game_id"].notna()
    ].copy()
    spreads["observed_at_utc"] = pd.to_datetime(spreads["observed_at_utc"], utc=True)
    spreads["commence_time_utc"] = pd.to_datetime(spreads["commence_time_utc"], utc=True)
    return spreads


def _capture_line(rows: pd.DataFrame, *, first: bool) -> pd.DataFrame:
    """Per game, the cross-book median at that game's earliest (``first``) or
    latest capture instant among ``rows``, with the instant."""

    if rows.empty:
        return pd.DataFrame(columns=["game_id", "line", "at"])
    instant = rows.groupby("nflverse_game_id")["observed_at_utc"].transform(
        "min" if first else "max"
    )
    at_instant = rows.loc[rows["observed_at_utc"].eq(instant)]
    return (
        at_instant.groupby("nflverse_game_id")
        .agg(line=("home_spread_line", "median"), at=("observed_at_utc", "min"))
        .reset_index()
        .rename(columns={"nflverse_game_id": "game_id"})
    )


def tuesday_gap(quotes: pd.DataFrame, tuesday: date, lock: time) -> pd.DataFrame:
    """Per game of the week ``tuesday`` opens: the earliest pre-lock capture
    that day, the first post-lock capture, the move between them, and the
    served opener with its basis. ``pre_lock``/``post_lock``/``move`` are
    null (never zero) when the corresponding capture does not exist."""

    spreads = home_spreads(quotes)
    tuesday_day = pd.Timestamp(tuesday)
    on_day = spreads.loc[
        pool_calendar_day(spreads["observed_at_utc"]).eq(tuesday_day)
        & own_week_tuesday(spreads["commence_time_utc"]).eq(tuesday_day)
        & spreads["observed_at_utc"].lt(spreads["commence_time_utc"])
    ].copy()
    if on_day.empty:
        return pd.DataFrame(columns=COLUMNS)
    lock_ts = pd.Timestamp(datetime.combine(tuesday, lock, tzinfo=ET))
    pre = _capture_line(on_day.loc[on_day["observed_at_utc"].lt(lock_ts)], first=True).rename(
        columns={"line": "pre_lock", "at": "pre_lock_at"}
    )
    post = _capture_line(on_day.loc[on_day["observed_at_utc"].ge(lock_ts)], first=True).rename(
        columns={"line": "post_lock", "at": "post_lock_at"}
    )
    opener = tuesday_opener_quotes(on_day).rename(
        columns={"nflverse_game_id": "game_id", "opener_home_spread": "opener"}
    )[["game_id", "opener", "opener_basis"]]
    table = opener.merge(pre, on="game_id", how="left").merge(post, on="game_id", how="left")
    table["move"] = table["post_lock"] - table["pre_lock"]
    return table[COLUMNS].sort_values("game_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="Tuesday (ET)")
    parser.add_argument(
        "--lock",
        default=DEFAULT_LOCK,
        help=f"pool lock time, ET (HH:MM); default {DEFAULT_LOCK}, the declared pool lock",
    )
    args = parser.parse_args(argv)
    tuesday = args.date or datetime.now(tz=ET).date()
    hour, minute = (int(part) for part in args.lock.split(":"))
    quotes = load_decision_quotes(REPO / "data" / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    table = tuesday_gap(quotes, tuesday, time(hour, minute))
    if table.empty:
        print(json.dumps({"tuesday": tuesday.isoformat(), "games": 0}))
        return 0
    print(table.to_string(index=False))
    if table["pre_lock"].isna().all():
        print(NO_PRE_LOCK_MESSAGE)
    if table["post_lock"].isna().all():
        print(NO_POST_LOCK_MESSAGE)
    moved = table["move"].dropna()
    summary: dict[str, object] = {
        "tuesday": tuesday.isoformat(),
        "lock_et": args.lock,
        "games": len(table),
        "with_pre_lock_capture": int(table["pre_lock"].notna().sum()),
        "with_post_lock_capture": int(table["post_lock"].notna().sum()),
        "opener_basis": table["opener_basis"].value_counts().to_dict(),
        "comparable": len(moved),
    }
    if not moved.empty:
        summary.update(
            {
                "moved": int(moved.ne(0.0).sum()),
                "moved_half_point_or_more": int(moved.abs().ge(0.5).sum()),
                "mean_abs_move": float(moved.abs().mean()),
                "max_abs_move": float(moved.abs().max()),
            }
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
