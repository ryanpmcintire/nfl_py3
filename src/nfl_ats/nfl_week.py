"""Shared NFL calendar anchors."""

from __future__ import annotations

from datetime import date, timedelta


def week_cycle_sunday(game_day: date) -> date:
    """Sunday of the Tuesday-through-Monday NFL week containing ``game_day``."""

    weekday = game_day.weekday()  # Monday=0 .. Sunday=6
    if weekday == 0:
        return game_day - timedelta(days=1)
    return game_day + timedelta(days=6 - weekday)
