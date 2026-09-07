"""Frozen MKT-15 refresh exposure; no outcomes or model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError

LEADERSHIP_WEIGHTS = {
    "bovada": 0.6135,
    "williamhill_us": 0.6405,
    "mybookieag": 0.6276,
    "draftkings": 0.4477,
    "betus": 0.4035,
    "betrivers": 0.3563,
    "pointsbetus": 0.5397,
    "fanatics": 0.5569,
    "lowvig": 0.2885,
    "fanduel": 0.3045,
    "betmgm": 0.3026,
    "betonlineag": 0.2200,
}
LEADER_BOOKS = ("bovada", "williamhill_us", "mybookieag")
THRESHOLD = 0.5


def sharp_book_movement_features(quotes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row/game, at the earlier of supplied cutoff and pool deadline.

    Games require game_id, commence_time_utc, week_first_commence_utc.
    Optional cutoff_utc allows an earlier prospective refresh timestamp.
    Quotes use the existing archive's standardized positive-home margin line.
    """
    required = {"game_id", "commence_time_utc", "week_first_commence_utc"}
    if not required.issubset(games.columns) or games.game_id.duplicated().any():
        raise DataContractError("Games require unique IDs and kickoff/week anchors")
    result = games.copy()
    kickoff = pd.to_datetime(result.commence_time_utc, utc=True)
    anchor = pd.to_datetime(result.week_first_commence_utc, utc=True).dt.tz_convert(
        "America/New_York"
    )
    local = anchor.dt.tz_localize(None).dt.normalize()
    sunday = local + pd.to_timedelta((6 - anchor.dt.weekday) % 7, unit="D")
    deadline = (
        (sunday + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    cutoffs = [kickoff, deadline]
    if "cutoff_utc" in result:
        cutoffs.append(pd.to_datetime(result.cutoff_utc, utc=True))
    result["cutoff_utc"] = pd.concat(cutoffs, axis=1).min(axis=1)
    result["_monday"] = (
        (sunday - pd.Timedelta(days=6)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    result["_wednesday"] = (
        (sunday - pd.Timedelta(days=4)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    result["_sunday"] = sunday.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    for name in ("leader", "equal"):
        result[f"{name}_net_move"] = 0.0
    result["eligible_books"] = 0
    result["leader_move_observed"] = False
    if not quotes.empty:
        needed = {
            "nflverse_game_id",
            "bookmaker_key",
            "market",
            "home_spread_line",
            "observed_at_utc",
            "bookmaker_last_update_utc",
        }
        if not needed.issubset(quotes.columns):
            raise DataContractError("Missing spread quote columns")
        q = quotes.loc[
            quotes.market.eq("spreads") & quotes.bookmaker_key.isin(LEADERSHIP_WEIGHTS),
            sorted(needed),
        ].rename(columns={"nflverse_game_id": "game_id"})
        q = q.merge(
            result[["game_id", "cutoff_utc", "_monday", "_wednesday", "_sunday"]], on="game_id"
        )
        for column in ("observed_at_utc", "bookmaker_last_update_utc"):
            q[column] = pd.to_datetime(q[column], utc=True, errors="coerce")
        q["home_spread_line"] = pd.to_numeric(q.home_spread_line, errors="coerce")
        q = q.loc[
            q.observed_at_utc.lt(q.cutoff_utc)
            & q.observed_at_utc.ge(q._monday)
            & q.observed_at_utc.lt(q._sunday)
            & q.bookmaker_last_update_utc.le(q.observed_at_utc)
            & np.isfinite(q.home_spread_line)
        ].copy()
        keys = ["game_id", "bookmaker_key", "observed_at_utc"]
        if q.groupby(keys).home_spread_line.nunique().gt(1).any():
            raise DataContractError("Conflicting book lines at one observed timestamp")
        q = q.sort_values(keys).drop_duplicates(keys)
        q["move"] = q.groupby(["game_id", "bookmaker_key"]).home_spread_line.diff()
        q = q.loc[q.observed_at_utc.ge(q._wednesday) & q.move.notna()].copy()
        moved = q.loc[q.bookmaker_key.isin(LEADER_BOOKS) & q.move.ne(0), "game_id"]
        result["leader_move_observed"] = result.game_id.isin(moved)
        books = q.groupby(["game_id", "bookmaker_key"], as_index=False).move.sum()
        books["weight"] = books.bookmaker_key.map(LEADERSHIP_WEIGHTS)
        books["weighted_move"] = books.move * books.weight
        summary = books.groupby("game_id").agg(
            weighted_move=("weighted_move", "sum"),
            weight=("weight", "sum"),
            equal_net_move=("move", "mean"),
            eligible_books=("move", "size"),
        )
        summary["leader_net_move"] = summary.weighted_move / summary.weight
        for column in ("leader_net_move", "equal_net_move", "eligible_books"):
            result[column] = result.game_id.map(summary[column]).fillna(0)
    for name in ("leader", "equal"):
        result[f"{name}_flag"] = result[f"{name}_net_move"].abs().ge(THRESHOLD)
    return result.drop(columns=["_monday", "_wednesday", "_sunday"])


def refresh_pick(production_home: pd.Series, net_move: pd.Series) -> pd.Series:
    """Follow qualifying movement; absent/subthreshold movement keeps production."""
    return production_home.astype(bool).mask(net_move.abs().ge(THRESHOLD), net_move.gt(0))


#: Games columns the shared late-week follow computation requires. ``cutoff_utc``
#: is the refresh instant; the effective evidence cutoff per game is
#: ``min(kickoff, Sunday pool deadline, cutoff_utc)`` (see
#: :func:`sharp_book_movement_features`).
LATE_WEEK_GAMES_COLUMNS = (
    "game_id",
    "commence_time_utc",
    "week_first_commence_utc",
    "cutoff_utc",
)


def late_week_follow_frame(
    quotes: pd.DataFrame,
    games: pd.DataFrame,
    *,
    now: pd.Timestamp,
    tuesday_pick_side: pd.Series,
) -> tuple[pd.DataFrame, int]:
    """The frozen MKT-15/CX18 follow rule, shared by the paired challenger and
    the served refresh pick so the two can never drift apart.

    ``quotes`` are live intraday spread rows (never historical backfills);
    ``games`` carries :data:`LATE_WEEK_GAMES_COLUMNS`; ``tuesday_pick_side``
    maps game_id to the frozen Tuesday played side (``HOME``/``AWAY``).
    A later snapshot must not backdate an earlier observation into this pass:
    rows with ``observed_at_utc`` or ``snapshot_timestamp_utc`` at/after
    ``now``, or a provider update later than the observation, are refused and
    counted, never scored. Returns ``(exposure, refused_quote_rows)`` where
    exposure carries ``equal_net_move``, ``eligible_books``,
    ``tuesday_pick_side``, ``movement_would_be_pick_side`` (the
    :func:`refresh_pick` decision: market side when ``|equal_net_move|`` is at
    least 0.5, else the Tuesday side) and ``movement_flip`` (market side
    differs from the Tuesday side), one row per input game.
    """

    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None or now_ts > pd.Timestamp.now(tz="UTC"):
        raise DataContractError("Refresh time must be timezone-aware and cannot be in the future")
    missing_games = sorted(set(LATE_WEEK_GAMES_COLUMNS).difference(games.columns))
    if missing_games or games.game_id.duplicated().any():
        raise DataContractError(
            "Late-week follow games require unique IDs and columns: "
            + ", ".join(LATE_WEEK_GAMES_COLUMNS)
        )
    needed = {
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "observed_at_utc",
        "snapshot_timestamp_utc",
        "bookmaker_last_update_utc",
    }
    if not needed.issubset(quotes.columns):
        raise DataContractError("Missing spread quote columns")
    q = quotes.copy()
    observed = pd.to_datetime(q.observed_at_utc, utc=True, errors="coerce")
    snapshot = pd.to_datetime(q.snapshot_timestamp_utc, utc=True, errors="coerce")
    updated = pd.to_datetime(q.bookmaker_last_update_utc, utc=True, errors="coerce")
    safe = observed.lt(now_ts) & snapshot.lt(now_ts) & updated.le(observed)
    refused = int((~safe).sum())
    q = q.loc[safe].copy()
    exposure = sharp_book_movement_features(q, games.copy())
    if not exposure.eligible_books.gt(0).any():
        return exposure.assign(
            tuesday_pick_side=pd.Series(dtype=str),
            movement_would_be_pick_side=pd.Series(dtype=str),
            movement_flip=pd.Series(dtype=bool),
        ), refused
    sides = tuesday_pick_side.astype(str)
    if sides.isna().any() or not sides.isin(["HOME", "AWAY"]).all():
        raise DataContractError("Tuesday card has an invalid pick side")
    unmapped = sorted(set(exposure.game_id.astype(str)) - set(sides.index.astype(str)))
    if unmapped:
        raise DataContractError("Tuesday card is missing games in the refresh plan")
    exposure["tuesday_pick_side"] = exposure.game_id.astype(str).map(sides)
    home = refresh_pick(exposure.tuesday_pick_side.eq("HOME"), exposure.equal_net_move)
    exposure["movement_would_be_pick_side"] = home.map({True: "HOME", False: "AWAY"})
    exposure["movement_flip"] = exposure.movement_would_be_pick_side.ne(exposure.tuesday_pick_side)
    return exposure, refused
