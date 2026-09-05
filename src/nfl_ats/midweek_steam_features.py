"""LEAD-04: observable midweek spread steam, independent of game outcomes."""

from __future__ import annotations

from typing import Any

import pandas as pd

EASTERN = "America/New_York"
EVENT_COLUMNS = ["game_id", "observed_at_utc", "steam_side", "books"]


def decision_cutoff(kickoff: Any) -> pd.Timestamp:
    """Earlier of kickoff and the game week's Sunday at 16:00 Eastern."""
    local = pd.Timestamp(kickoff).tz_convert(EASTERN)
    # Monday night belongs to the preceding Sunday; all other days to the next.
    days = -1 if local.weekday() == 0 else 6 - local.weekday()
    sunday = (local.normalize().tz_localize(None) + pd.Timedelta(days=days, hours=16)).tz_localize(
        EASTERN
    )
    return min(local, sunday).tz_convert("UTC")


def spread_move_events(quotes: pd.DataFrame) -> pd.DataFrame:
    """One steam event per game, observed timestamp and direction.

    Require three distinct changed books in a trailing 60-minute observation
    window, also within 60 minutes by provider update. Each update must be
    later than its book's previous observation and no later than capture.
    The archive stores required home winning margin, so HOME means an increase
    in home_spread_line; AWAY means a decrease (opposite the home handicap).
    """
    q = quotes.loc[quotes["market"].eq("spreads")].copy()
    for name in ("observed_at_utc", "bookmaker_last_update_utc"):
        q[name] = pd.to_datetime(q[name], utc=True)
    q["home_spread_line"] = pd.to_numeric(q["home_spread_line"], errors="coerce")
    keys = ["nflverse_game_id", "bookmaker_key", "observed_at_utc"]
    q = q.dropna(subset=[*keys, "home_spread_line"])
    conflicting = q.groupby(keys)["home_spread_line"].transform("nunique").gt(1)
    if conflicting.any():
        raise ValueError("Conflicting home spreads within a game/book/snapshot")
    q = q.sort_values([*keys, "bookmaker_last_update_utc"]).drop_duplicates(keys, keep="last")
    grouped = q.groupby(keys[:2])
    previous_line = grouped["home_spread_line"].shift()
    previous_time = grouped["observed_at_utc"].shift()
    change = q["home_spread_line"] - previous_line
    local = q["observed_at_utc"].dt.tz_convert(EASTERN)
    valid = (
        change.notna()
        & change.ne(0)
        & local.dt.weekday.between(2, 5)
        & q["bookmaker_last_update_utc"].gt(previous_time)
        & q["bookmaker_last_update_utc"].le(q["observed_at_utc"])
        & (q["observed_at_utc"] - q["bookmaker_last_update_utc"]).le(pd.Timedelta(hours=1))
    )
    moves = q.loc[valid].copy()
    moves["steam_side"] = change.loc[valid].gt(0).map({True: "HOME", False: "AWAY"})
    events: list[dict[str, Any]] = []
    hour = pd.Timedelta(hours=1)
    for (game, side), group in moves.groupby(["nflverse_game_id", "steam_side"]):
        group = group.sort_values("observed_at_utc")
        for observed, fresh in group.groupby("observed_at_utc"):
            window = group.loc[
                group["observed_at_utc"].between(observed - hour, observed)
                & group["bookmaker_last_update_utc"].ge(observed - hour)
            ]
            if len(window) < 3:
                continue
            # A book counts once, irrespective of duplicate outcome rows or moves.
            books = sorted(window["bookmaker_key"].unique())
            if len(books) >= 3 and not fresh.empty:
                events.append(
                    {
                        "game_id": str(game),
                        "observed_at_utc": observed,
                        "steam_side": str(side),
                        "books": ",".join(books),
                    }
                )
    return (
        pd.DataFrame(events, columns=EVENT_COLUMNS)
        .sort_values(["game_id", "observed_at_utc", "steam_side"])
        .reset_index(drop=True)
    )


def attach_midweek_steam(
    games: pd.DataFrame, events: pd.DataFrame, *, as_of: Any | None = None
) -> pd.DataFrame:
    """Latest observable steam wins; conflicting simultaneous sides are a no-op.

    ``games`` supplies game_id and kickoff. Strictly exclude events at or after
    the deadline/kickoff. A refresh as_of includes already observed events only.
    """
    result = games.copy()
    result["decision_cutoff_utc"] = result["kickoff"].map(decision_cutoff)
    sides: list[str] = []
    starts: list[pd.Timestamp] = []
    for game in result.itertuples(index=False):
        cutoff = game.decision_cutoff_utc
        # The Tuesday preceding the cutoff's Sunday defines this NFL week.
        local = pd.Timestamp(str(game.kickoff)).tz_convert(EASTERN)
        days = -1 if local.weekday() == 0 else 6 - local.weekday()
        sunday = local.normalize().tz_localize(None) + pd.Timedelta(days=days)
        start = (sunday - pd.Timedelta(days=4)).tz_localize(EASTERN).tz_convert("UTC")
        starts.append(start)
        times = pd.to_datetime(events["observed_at_utc"], utc=True)
        valid = events["game_id"].eq(game.game_id) & times.ge(start) & times.lt(cutoff)
        valid &= times.dt.tz_convert(EASTERN).dt.weekday.between(2, 5)
        if as_of is not None:
            valid &= times.le(pd.Timestamp(as_of))
        available = events.loc[valid]
        if available.empty:
            sides.append("")
        else:
            latest = available.loc[times.loc[available.index].eq(times.loc[available.index].max())]
            unique = latest["steam_side"].unique()
            sides.append(str(unique[0]) if len(unique) == 1 else "")
    result["midweek_steam_side"] = sides
    result["midweek_start_utc"] = starts
    result["midweek_steam_exposure"] = result["midweek_steam_side"].ne("")
    return result
