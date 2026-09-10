from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import HANDLE_FOLLOW_POLICY, RefreshResult, original_card

CHALLENGER_ID = "handle_follow_refresh_off_incumbent"
LEDGER_NAME = "handle_follow_refresh_decisions.parquet"

HANDLE_FOLLOW_LEDGER_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "challenger_id",
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
    "off_arm_pick_side",
    "handle_pick_side",
    "handle_money_pct",
    "handle_ticket_pct",
    "handle_follow_flip",
    "explanation",
    "model_id",
    "feature_table_sha256",
)


def build_handle_follow_refresh_rows(
    plan: RefreshResult, *, original: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:

    empty = pd.DataFrame(columns=list(HANDLE_FOLLOW_LEDGER_COLUMNS))
    if original.empty:
        return empty, {"skipped": True, "reason": "Tuesday card is absent."}
    tuesday = original.set_index("game_id")["pick_side"].astype(str)
    rows: list[dict[str, Any]] = []
    for game in plan.games:
        if not game.eligible or game.handle_money_pct is None:
            continue
        if game.game_id not in tuesday.index:
            continue
        flip = game.movement_policy == HANDLE_FOLLOW_POLICY
        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "challenger_id": CHALLENGER_ID,
                "season": plan.season,
                "week": plan.week,
                "game_id": game.game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "deadline": game.deadline,
                "decision_home_spread": game.decision_home_spread,
                "tuesday_pick_side": str(tuesday[game.game_id]),
                "served_pick_side": game.new_pick_side,
                "off_arm_pick_side": game.handle_pre_rule_pick_side,
                "handle_pick_side": game.handle_pick_side,
                "handle_money_pct": game.handle_money_pct,
                "handle_ticket_pct": game.handle_ticket_pct,
                "handle_follow_flip": flip,
                "explanation": (
                    "Switched to the side holding most of the money."
                    if flip
                    else "Kept the pick; the money does not call for a switch."
                ),
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )
    if not rows:
        return empty, {
            "skipped": True,
            "reason": "No eligible game carried a money reading on this pass.",
        }
    frame = pd.DataFrame(rows)[list(HANDLE_FOLLOW_LEDGER_COLUMNS)]
    return frame, {
        "skipped": False,
        "games_considered": len(frame),
        "flips": int(frame["handle_follow_flip"].sum()),
    }


def record_handle_follow_refresh_overlay(
    artifacts_root: Path, plan: RefreshResult, *, record_decisions: bool = False
) -> dict[str, Any]:

    result: dict[str, Any] = {"challenger_id": CHALLENGER_ID, "recorded": 0}
    if not record_decisions:
        return {**result, "skipped": True, "reason": "Recording was not requested."}
    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    if original.empty:
        return {**result, "skipped": True, "reason": "Tuesday card is absent."}
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="handle-follow-refresh"
    )
    rows, diagnostics = build_handle_follow_refresh_rows(plan, original=original)
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
