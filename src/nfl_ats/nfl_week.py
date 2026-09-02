"""Shared NFL calendar anchors."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")


def week_cycle_sunday(game_day: date) -> date:
    """Sunday of the Tuesday-through-Monday NFL week containing ``game_day``."""

    weekday = game_day.weekday()  # Monday=0 .. Sunday=6
    if weekday == 0:
        return game_day - timedelta(days=1)
    return game_day + timedelta(days=6 - weekday)


def pool_decision_cutoff(kickoff: datetime) -> datetime:
    """Return the pool's real pick deadline as an aware UTC datetime.

    A game locks at its kickoff or at 16:00 America/New_York on the Sunday
    in its Tuesday-through-Monday NFL week, whichever comes first.  Building
    the Sunday wall-clock time in the named zone keeps the UTC conversion
    correct on both sides of daylight-saving transitions.
    """

    if kickoff.tzinfo is None or kickoff.utcoffset() is None:
        raise ValueError("kickoff must carry an explicit timezone")
    kickoff_utc = kickoff.astimezone(UTC)
    kickoff_eastern = kickoff_utc.astimezone(_EASTERN)
    sunday = week_cycle_sunday(kickoff_eastern.date())
    sunday_lock = datetime.combine(sunday, time(16), tzinfo=_EASTERN).astimezone(UTC)
    return min(kickoff_utc, sunday_lock)
