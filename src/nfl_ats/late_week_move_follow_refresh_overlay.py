"""MKT-15: paired Tuesday/late-week movement picks at the frozen Tuesday line.

Reuse the CX18 Wednesday-Saturday increments and twelve-book universe exactly.
Sunday refreshes consume Saturday evidence; Sunday moves are outside this rule.
Only live captures are prospective inputs, never historical backfills.
Since 2026-09-09 the SERVED arm is the leading books' median move
(``late_week_leader_median_follow_v1``) and this ledger's own challenger id
records the equal-book arm it replaced; both are on every row. Since
2026-09-10 the served gate is a FULL point below a 10.5-point line and half a
point at or above it, so three more OFF arms record beside it on every row:
the flat full-point gate
(``late_week_leader_median_follow_flat_1_0_off_incumbent``), the retired
half-point gate (``late_week_leader_median_follow_0_5_off_incumbent``) and
following every move without the injury-news veto
(``late_week_follow_no_news_veto_off_incumbent``).
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
from nfl_ats.sharp_book_movement_features import (
    LEADER_FOLLOW_THRESHOLD,
    late_week_follow_frame,
)

CHALLENGER_ID = "late_week_move_follow_refresh_v1"
SERVED_CHALLENGER_ID = "late_week_leader_median_follow_v1"
OFF_THRESHOLD_CHALLENGER_ID = "late_week_leader_median_follow_0_5_off_incumbent"
FLAT_THRESHOLD_CHALLENGER_ID = "late_week_leader_median_follow_flat_1_0_off_incumbent"
NEWS_VETO_OFF_CHALLENGER_ID = "late_week_follow_no_news_veto_off_incumbent"
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
                "decision_home_spread": originals.decision_home_spread.get(game.game_id),
            }
        )
    if not games:
        return empty, {"skipped": True, "reason": "No games remain before their pick deadline."}
    q = quotes.loc[quotes.nflverse_game_id.isin([g["game_id"] for g in games])].copy()
    exposure, refused = late_week_follow_frame(
        q,
        pd.DataFrame(games),
        now=now,
        tuesday_pick_side=originals.pick_side.astype(str),
    )
    if not exposure.eligible_books.gt(0).any():
        return empty, {
            "skipped": True,
            "reason": "No pre-deadline late-week book changes are available.",
            "refused_quote_rows": refused,
        }
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
        "Keep Tuesday's pick because the leading books did not move the line."
        if books == 0
        else "The leading books moved the line late; follow it to the other team."
        if flip
        else "Keep Tuesday's pick; the leading books' move does not call for a switch."
        for books, flip in zip(exposure.leader_books, exposure.movement_flip, strict=True)
    ]
    vetoes = {game.game_id: game for game in plan.games}
    exposure["follow_news_veto"] = [
        bool(vetoes[game_id].follow_news_veto) if game_id in vetoes else False
        for game_id in exposure.game_id
    ]
    exposure["follow_news_source"] = [
        str(vetoes[game_id].follow_news_source) if game_id in vetoes else ""
        for game_id in exposure.game_id
    ]
    exposure["follow_news_team"] = [
        str(vetoes[game_id].follow_news_team) if game_id in vetoes else ""
        for game_id in exposure.game_id
    ]
    exposure["news_veto_would_be_pick_side"] = [
        tuesday if veto else served
        for veto, tuesday, served in zip(
            exposure.follow_news_veto,
            exposure.tuesday_pick_side,
            exposure.movement_would_be_pick_side,
            strict=True,
        )
    ]
    exposure["news_veto_movement_flip"] = exposure.news_veto_would_be_pick_side.ne(
        exposure.tuesday_pick_side
    )
    exposure["revision_recorded_at_utc"] = now
    exposure["refresh_run_id"] = plan.refresh_run_id
    exposure["challenger_id"] = CHALLENGER_ID
    exposure["served_challenger_id"] = SERVED_CHALLENGER_ID
    exposure["off_threshold_challenger_id"] = OFF_THRESHOLD_CHALLENGER_ID
    exposure["flat_threshold_challenger_id"] = FLAT_THRESHOLD_CHALLENGER_ID
    exposure["news_veto_off_challenger_id"] = NEWS_VETO_OFF_CHALLENGER_ID
    exposure["season"] = plan.season
    exposure["week"] = plan.week
    exposure["model_id"] = plan.model_id
    exposure["feature_table_sha256"] = plan.feature_table_sha256
    return exposure, {
        "skipped": False,
        "games_considered": len(exposure),
        "flips": int(exposure.movement_flip.sum()),
        "off_threshold_flips": int(exposure.leader_median_half_movement_flip.sum()),
        "flat_threshold_flips": int(exposure.leader_median_flat_movement_flip.sum()),
        "big_spread_gate_games": int(
            exposure.late_week_threshold_applied.lt(LEADER_FOLLOW_THRESHOLD).sum()
        ),
        "equal_book_flips": int(exposure.equal_movement_flip.sum()),
        "news_vetoes": int(exposure.follow_news_veto.sum()),
        "news_veto_flips": int(exposure.news_veto_movement_flip.sum()),
        "refused_quote_rows": refused,
    }


def record_late_week_move_follow_refresh_overlay(
    artifacts_root: Path, data_root: Path, plan: RefreshResult, *, record_decisions: bool = False
) -> dict[str, Any]:
    """Append paired arms in a separate ledger, once per game and refresh run."""
    result: dict[str, Any] = {
        "challenger_id": CHALLENGER_ID,
        "served_challenger_id": SERVED_CHALLENGER_ID,
        "off_threshold_challenger_id": OFF_THRESHOLD_CHALLENGER_ID,
        "flat_threshold_challenger_id": FLAT_THRESHOLD_CHALLENGER_ID,
        "news_veto_off_challenger_id": NEWS_VETO_OFF_CHALLENGER_ID,
        "recorded": 0,
    }
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
