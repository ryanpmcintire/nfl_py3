"""How far the line moved between the 09:00 opener and the pool's noon lock (OPS-05).

Owner, 2026-09-08: "Spreads lock: Tue, Sep 8, 2026, 12:00 PM". The card is
formed at the Tuesday opener (the earliest Tuesday quote per book,
``nfl_ats.market_data.tuesday_opener_quotes``); the pool's spreads are fixed
at noon and captured by the ``odds_tue_noon`` job. This read-only report
puts the two side by side for one Tuesday: per game, the opener consensus,
the latest pre-noon-lock consensus, and the move, so the question "should
the card be formed at the noon line?" is answered from recorded lines.

Usage::

    uv run --no-sync python scripts/tuesday_line_gap.py             # today's Tuesday
    uv run --no-sync python scripts/tuesday_line_gap.py --date 2026-09-01
    uv run --no-sync python scripts/tuesday_line_gap.py --lock 12:00  # ET lock time

Prints a table and a JSON summary; writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes  # noqa: E402
from nfl_ats.market_data import tuesday_opener_quotes  # noqa: E402

READ_ONLY_SCRIPT = True
ET = ZoneInfo("America/New_York")


def home_spreads(quotes: pd.DataFrame) -> pd.DataFrame:
    spreads = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["outcome_side"].eq("HOME")
        & quotes["nflverse_game_id"].notna()
    ].copy()
    spreads["observed_at_utc"] = pd.to_datetime(spreads["observed_at_utc"], utc=True)
    spreads["commence_time_utc"] = pd.to_datetime(spreads["commence_time_utc"], utc=True)
    return spreads


def tuesday_gap(quotes: pd.DataFrame, tuesday: date, lock: time) -> pd.DataFrame:
    """Per game: opener consensus vs the latest quote at or before the lock."""

    spreads = home_spreads(quotes)
    observed_et = spreads["observed_at_utc"].dt.tz_convert(ET)
    # This Tuesday's captures, for the games of the week it opens (kickoffs
    # within the next eight days); the live store quotes the whole season.
    week_end = pd.Timestamp(datetime.combine(tuesday, time(0, 0), tzinfo=ET)) + pd.Timedelta(days=8)
    on_day = spreads.loc[
        observed_et.dt.date.eq(tuesday) & spreads["commence_time_utc"].le(week_end)
    ].copy()
    if on_day.empty:
        return pd.DataFrame(
            columns=["game_id", "opener", "at_lock", "move", "opener_at", "lock_quote_at"]
        )
    lock_at = datetime.combine(tuesday, lock, tzinfo=ET)
    opener = tuesday_opener_quotes(on_day).rename(
        columns={
            "nflverse_game_id": "game_id",
            "opener_home_spread": "opener",
            "observed_at_utc": "opener_at",
        }
    )[["game_id", "opener", "opener_at"]]
    # The lock-moment line is the FIRST capture at or after the lock (the
    # ``odds_tue_noon`` job, 12:05 + grace); when none exists yet, the latest
    # capture before the lock stands in and is labelled as such.
    lock_ts = pd.Timestamp(lock_at)
    after_lock = on_day.loc[
        on_day["observed_at_utc"].between(lock_ts, lock_ts + pd.Timedelta(minutes=65))
    ]
    if not after_lock.empty:
        lock_capture = after_lock["observed_at_utc"].min()
        lock_source = "first capture after the lock"
    else:
        before_lock = on_day.loc[on_day["observed_at_utc"].lt(lock_ts)]
        lock_capture = before_lock["observed_at_utc"].max()
        lock_source = "latest capture BEFORE the lock (no post-lock capture yet)"
    print(f"lock line: {lock_source}")
    at_lock = (
        on_day.loc[on_day["observed_at_utc"].eq(lock_capture)]
        .groupby("nflverse_game_id")["home_spread_line"]
        .median()
        .rename("at_lock")
        .reset_index()
        .rename(columns={"nflverse_game_id": "game_id"})
    )
    at_lock["lock_quote_at"] = lock_capture
    table = opener.merge(at_lock, on="game_id", how="left")
    table["move"] = table["at_lock"] - table["opener"]
    return table.sort_values("game_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="Tuesday (ET)")
    parser.add_argument("--lock", default="12:00", help="pool lock time, ET (HH:MM)")
    args = parser.parse_args(argv)
    tuesday = args.date or datetime.now(tz=ET).date()
    hour, minute = (int(part) for part in args.lock.split(":"))
    quotes = load_decision_quotes(REPO / "data" / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    table = tuesday_gap(quotes, tuesday, time(hour, minute))
    if table.empty:
        print(json.dumps({"tuesday": tuesday.isoformat(), "games": 0}))
        return 0
    print(table.to_string(index=False))
    moved = table["move"].dropna()
    print(
        json.dumps(
            {
                "tuesday": tuesday.isoformat(),
                "lock_et": args.lock,
                "games": len(table),
                "with_lock_quote": int(table["at_lock"].notna().sum()),
                "moved": int(moved.ne(0.0).sum()),
                "moved_half_point_or_more": int(moved.abs().ge(0.5).sum()),
                "mean_abs_move": float(moved.abs().mean()),
                "max_abs_move": float(moved.abs().max()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
