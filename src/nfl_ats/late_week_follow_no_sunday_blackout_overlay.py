from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.clv import (
    LIVE_CAPTURE_KIND,
    load_decision_quotes,
    refuse_if_outside_recording_lock_window,
)
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import PICK_LOCK_TIMEZONE, RefreshResult, original_card, sunday_pick_lock
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS, leader_follow_threshold

CHALLENGER_ID = "late_week_follow_no_sunday_blackout"
SERVED_CHALLENGER_ID = "late_week_leader_median_follow_v1"
LEDGER_NAME = "late_week_follow_no_sunday_blackout_decisions.parquet"

LEDGER_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "challenger_id",
    "served_challenger_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "deadline",
    "decision_home_spread",
    "tuesday_pick_side",
    "served_pick_side",
    "no_blackout_net_move",
    "no_blackout_eligible_books",
    "no_blackout_threshold_applied",
    "no_blackout_pick_side",
    "no_blackout_flip",
    "explanation",
    "model_id",
    "feature_table_sha256",
)


def leader_net_move_no_sunday_blackout(
    quotes: pd.DataFrame, original: pd.DataFrame, *, now: pd.Timestamp
) -> pd.DataFrame:

    empty = pd.DataFrame(columns=["game_id", "net_move", "eligible_books"])
    if quotes.empty or original.empty:
        return empty
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
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        raise DataContractError("Refresh time must be timezone-aware")
    sunday_lock = sunday_pick_lock(original["kickoff"])
    anchor_date = sunday_lock.tz_convert(PICK_LOCK_TIMEZONE).date()
    wednesday = pd.Timestamp(
        datetime.combine(anchor_date - timedelta(days=4), time(0, 0), tzinfo=PICK_LOCK_TIMEZONE)
    ).tz_convert("UTC")
    monday = pd.Timestamp(
        datetime.combine(anchor_date - timedelta(days=6), time(0, 0), tzinfo=PICK_LOCK_TIMEZONE)
    ).tz_convert("UTC")
    game_ids = set(original["game_id"].astype(str))
    q = quotes.loc[
        quotes["market"].eq("spreads") & quotes["bookmaker_key"].isin(LEADER_BOOKS),
        sorted(needed),
    ].rename(columns={"nflverse_game_id": "game_id"})
    q = q.loc[q["game_id"].astype(str).isin(game_ids)].copy()
    if q.empty:
        return empty
    for column in ("observed_at_utc", "bookmaker_last_update_utc"):
        q[column] = pd.to_datetime(q[column], utc=True, errors="coerce")
    q["home_spread_line"] = pd.to_numeric(q["home_spread_line"], errors="coerce")
    keep = (
        q["observed_at_utc"].lt(now_ts)
        & q["observed_at_utc"].ge(monday)
        & q["bookmaker_last_update_utc"].le(q["observed_at_utc"])
        & np.isfinite(q["home_spread_line"])
    )
    q = q.loc[keep].copy()
    if q.empty:
        return empty
    keys = ["game_id", "bookmaker_key", "observed_at_utc"]
    q = q.sort_values(keys).drop_duplicates(keys)
    q["move"] = q.groupby(["game_id", "bookmaker_key"])["home_spread_line"].diff()
    q = q.loc[q["observed_at_utc"].ge(wednesday) & q["move"].notna()]
    if q.empty:
        return empty
    books = q.groupby(["game_id", "bookmaker_key"], as_index=False)["move"].sum()
    summary = books.groupby("game_id").agg(
        net_move=("move", "median"), eligible_books=("move", "size")
    )
    return cast(pd.DataFrame, summary.reset_index())


def build_no_sunday_blackout_rows(
    plan: RefreshResult, *, original: pd.DataFrame, quotes: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:

    empty = pd.DataFrame(columns=list(LEDGER_COLUMNS))
    if original.empty:
        return empty, {"skipped": True, "reason": "Tuesday card is absent."}
    eligible_games = [game for game in plan.games if game.eligible]
    if not eligible_games:
        return empty, {"skipped": True, "reason": "No games remain before their pick deadline."}
    if quotes.empty:
        return empty, {"skipped": True, "reason": "live intraday odds archive is absent"}
    tuesday = original.set_index("game_id")["pick_side"].astype(str)
    exposure = leader_net_move_no_sunday_blackout(
        quotes, original, now=plan.computed_at_utc
    ).set_index("game_id")
    rows: list[dict[str, Any]] = []
    fires_count = 0
    for game in eligible_games:
        if game.game_id not in tuesday.index:
            continue
        base_side = str(tuesday[game.game_id])
        if game.game_id in exposure.index:
            net_move = float(cast(Any, exposure.loc[game.game_id, "net_move"]))
            eligible_books = int(cast(Any, exposure.loc[game.game_id, "eligible_books"]))
        else:
            net_move = 0.0
            eligible_books = 0
        threshold = leader_follow_threshold(game.decision_home_spread)
        fires = eligible_books > 0 and abs(net_move) >= threshold
        fires_count += int(fires)
        pick_side = ("HOME" if net_move > 0.0 else "AWAY") if fires else base_side
        flip = pick_side != game.new_pick_side
        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "challenger_id": CHALLENGER_ID,
                "served_challenger_id": SERVED_CHALLENGER_ID,
                "season": plan.season,
                "week": plan.week,
                "game_id": game.game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "deadline": game.deadline,
                "decision_home_spread": game.decision_home_spread,
                "tuesday_pick_side": base_side,
                "served_pick_side": game.new_pick_side,
                "no_blackout_net_move": net_move if eligible_books else None,
                "no_blackout_eligible_books": eligible_books,
                "no_blackout_threshold_applied": threshold,
                "no_blackout_pick_side": pick_side,
                "no_blackout_flip": flip,
                "explanation": (
                    "Counting Sunday-morning movement would have switched this pick."
                    if flip
                    else "Counting Sunday-morning movement leaves this pick where it is."
                ),
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )
    if not rows:
        return empty, {"skipped": True, "reason": "No eligible game matched the Tuesday card."}
    frame = pd.DataFrame(rows)[list(LEDGER_COLUMNS)]
    return frame, {
        "skipped": False,
        "games_considered": len(frame),
        "games_with_exposure": int(frame["no_blackout_eligible_books"].gt(0).sum()),
        "rule_fires": fires_count,
        "flips_vs_served": int(frame["no_blackout_flip"].sum()),
    }


def record_late_week_follow_no_sunday_blackout_overlay(
    artifacts_root: Path, data_root: Path, plan: RefreshResult, *, record_decisions: bool = False
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "challenger_id": CHALLENGER_ID,
        "served_challenger_id": SERVED_CHALLENGER_ID,
        "recorded": 0,
    }
    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    if original.empty:
        return {**result, "skipped": True, "reason": "Tuesday card is absent."}
    try:
        quotes = load_decision_quotes(data_root / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    except (OSError, ValueError, DataContractError) as error:
        return {
            **result,
            "skipped": True,
            "reason": f"live intraday odds archive is unreadable: {error}",
        }
    rows, diagnostics = build_no_sunday_blackout_rows(plan, original=original, quotes=quotes)
    if not record_decisions:
        return {**result, **diagnostics, "recording_skipped": "Recording was not requested."}
    if rows.empty:
        return {**result, **diagnostics}
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="late-week-follow-no-sunday-blackout"
    )
    path = artifacts_root / "prospective" / LEDGER_NAME
    existing = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    combined = combined.drop_duplicates(["refresh_run_id", "game_id"], keep="first")
    added = len(combined) - len(existing)
    if added:
        atomic_parquet(combined, path)
    return {**result, **diagnostics, "recorded": added, "ledger_rows": len(combined)}
