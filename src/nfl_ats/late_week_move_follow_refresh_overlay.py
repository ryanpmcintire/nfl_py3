"""MKT-15: paired Tuesday/late-week movement picks at the frozen Tuesday line.

Reuse the CX18 Wednesday-Saturday increments and twelve-book universe exactly.
Sunday refreshes consume Saturday evidence; Sunday moves are outside this rule.
Only live captures are prospective inputs, never historical backfills.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import (
    LIVE_CAPTURE_KIND,
    load_decision_quotes,
    refuse_if_outside_recording_lock_window,
)
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card, sunday_pick_lock
from nfl_ats.sharp_book_movement_features import refresh_pick, sharp_book_movement_features

CHALLENGER_ID = "late_week_move_follow_refresh_v1"
LEDGER_NAME = "late_week_move_follow_refresh_decisions.parquet"


def build_late_week_move_follow_refresh_rows(
    plan: RefreshResult, *, original: pd.DataFrame, quotes: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute both arms without mutating the refresh plan or original card."""
    empty = pd.DataFrame()
    if original.empty or quotes.empty:
        return empty, {
            "skipped": True,
            "reason": "Tuesday card or intraday odds archive is absent.",
        }
    now = pd.Timestamp(plan.computed_at_utc)
    if now.tzinfo is None or now > pd.Timestamp.now(tz="UTC"):
        raise DataContractError("Refresh time must be timezone-aware and cannot be in the future")
    recorded = pd.to_datetime(original.recorded_at_utc, utc=True, errors="coerce")
    if recorded.isna().any() or recorded.gt(now).any() or original.game_id.duplicated().any():
        raise DataContractError(
            "Tuesday card has invalid or future recording times or duplicate games"
        )
    if not original.pick_side.isin(["HOME", "AWAY"]).all():
        raise DataContractError("Tuesday card has an invalid pick side")
    lock = sunday_pick_lock(original.kickoff)
    originals = original.set_index("game_id")
    games = []
    for game in plan.games:
        cutoff = min(pd.Timestamp(game.kickoff), pd.Timestamp(game.deadline), lock)
        if not game.eligible or now >= cutoff or game.game_id not in originals.index:
            continue
        games.append(
            {
                "game_id": game.game_id,
                "commence_time_utc": game.kickoff,
                "week_first_commence_utc": lock,
                "cutoff_utc": now,
            }
        )
    if not games:
        return empty, {"skipped": True, "reason": "No games remain before their pick deadline."}
    q = quotes.loc[quotes.nflverse_game_id.isin([g["game_id"] for g in games])].copy()
    # A later snapshot must not backdate an earlier observation into this pass.
    observed = pd.to_datetime(q.observed_at_utc, utc=True, errors="coerce")
    snapshot = pd.to_datetime(q.snapshot_timestamp_utc, utc=True, errors="coerce")
    updated = pd.to_datetime(q.bookmaker_last_update_utc, utc=True, errors="coerce")
    safe = observed.lt(now) & snapshot.lt(now) & updated.le(observed)
    refused = int((~safe).sum())
    q = q.loc[safe].copy()
    exposure = sharp_book_movement_features(q, pd.DataFrame(games))
    if not exposure.eligible_books.gt(0).any():
        return empty, {
            "skipped": True,
            "reason": "No pre-deadline late-week book changes are available.",
            "refused_quote_rows": refused,
        }
    exposure["tuesday_pick_side"] = exposure.game_id.map(originals.pick_side)
    home = refresh_pick(exposure.tuesday_pick_side.eq("HOME"), exposure.equal_net_move)
    exposure["movement_would_be_pick_side"] = home.map({True: "HOME", False: "AWAY"})
    exposure["movement_flip"] = exposure.movement_would_be_pick_side.ne(exposure.tuesday_pick_side)
    exposure["decision_home_spread"] = exposure.game_id.map(originals.decision_home_spread)
    exposure["tuesday_recorded_at_utc"] = exposure.game_id.map(originals.recorded_at_utc)
    exposure["kickoff"] = exposure["commence_time_utc"]
    exposure["deadline"] = exposure.kickoff.map(lambda kickoff: min(kickoff, lock))
    exposure["home_team"] = exposure.game_id.map(
        {game.game_id: game.home_team for game in plan.games}
    )
    exposure["away_team"] = exposure.game_id.map(
        {game.game_id: game.away_team for game in plan.games}
    )
    exposure["explanation"] = [
        "Keep Tuesday's pick because no late-week book changes are available."
        if books == 0
        else "Follow the late-week line move toward the other team."
        if flip
        else "Keep Tuesday's pick; the line move does not call for a switch."
        for books, flip in zip(exposure.eligible_books, exposure.movement_flip, strict=True)
    ]
    exposure["revision_recorded_at_utc"] = now
    exposure["refresh_run_id"] = plan.refresh_run_id
    exposure["challenger_id"] = CHALLENGER_ID
    exposure["season"] = plan.season
    exposure["week"] = plan.week
    exposure["model_id"] = plan.model_id
    exposure["feature_table_sha256"] = plan.feature_table_sha256
    return exposure, {
        "skipped": False,
        "games_considered": len(exposure),
        "flips": int(exposure.movement_flip.sum()),
        "refused_quote_rows": refused,
    }


def record_late_week_move_follow_refresh_overlay(
    artifacts_root: Path, data_root: Path, plan: RefreshResult, *, record_decisions: bool = False
) -> dict[str, Any]:
    """Append paired arms in a separate ledger, once per game and refresh run."""
    result: dict[str, Any] = {"challenger_id": CHALLENGER_ID, "recorded": 0}
    if not record_decisions:
        return {**result, "skipped": True, "reason": "Recording was not requested."}
    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    if original.empty:
        return {**result, "skipped": True, "reason": "Tuesday card is absent."}
    refuse_if_outside_recording_lock_window(
        original.kickoff, plan.computed_at_utc, ledger="late-week-move-follow-refresh"
    )
    quotes = load_decision_quotes(data_root / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    rows, diagnostics = build_late_week_move_follow_refresh_rows(
        plan, original=original, quotes=quotes
    )
    if rows.empty:
        return {**result, **diagnostics}
    path = artifacts_root / "prospective" / LEDGER_NAME
    existing = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    combined = combined.drop_duplicates(["refresh_run_id", "game_id"], keep="first")
    added = len(combined) - len(existing)
    if added:
        atomic_parquet(combined, path)
    return {**result, **diagnostics, "recorded": added, "ledger_rows": len(combined)}
