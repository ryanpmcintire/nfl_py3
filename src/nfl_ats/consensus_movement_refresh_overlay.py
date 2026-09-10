"""The retired consensus rule's paired arms: the served refresh pick with and without it.

The 1.0-point whole-market consensus rule (``movement_ge_1.0``) stopped
governing served picks on 2026-09-10, measured at -1.627 accuracy points
through the served chain over 2023-2025 (``docs/served_refresh_card.md``).
This ledger records, on every pass that carries a captured line, the pick that
pass WOULD have served with the rule still applied beside the one it did
serve, at the same frozen Tuesday line, so both arms accrue game for game
rather than being reconstructed later from a rule description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import (
    CONSENSUS_MOVEMENT_OFF_CHALLENGER_ID,
    MOVEMENT_POLICY_THRESHOLD,
    RefreshResult,
    original_card,
)

CHALLENGER_ID = CONSENSUS_MOVEMENT_OFF_CHALLENGER_ID
LEDGER_NAME = "consensus_movement_refresh_decisions.parquet"

CONSENSUS_MOVEMENT_LEDGER_COLUMNS: tuple[str, ...] = (
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
    "consensus_arm_pick_side",
    "consensus_pick_side",
    "consensus_delta",
    "consensus_fires",
    "consensus_arm_flip",
    "explanation",
    "model_id",
    "feature_table_sha256",
)


def build_consensus_movement_refresh_rows(
    plan: RefreshResult, *, original: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Both arms for every eligible game the pass held a captured line on."""

    empty = pd.DataFrame(columns=list(CONSENSUS_MOVEMENT_LEDGER_COLUMNS))
    if original.empty:
        return empty, {"skipped": True, "reason": "Tuesday card is absent."}
    tuesday = original.set_index("game_id")["pick_side"].astype(str)
    rows: list[dict[str, Any]] = []
    for game in plan.games:
        if not game.eligible or game.consensus_delta is None:
            continue
        if game.game_id not in tuesday.index:
            continue
        fires = abs(game.consensus_delta) >= MOVEMENT_POLICY_THRESHOLD
        arm_side = game.consensus_arm_pick_side or game.new_pick_side
        flip = arm_side != game.new_pick_side
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
                "consensus_arm_pick_side": arm_side,
                "consensus_pick_side": game.consensus_pick_side,
                "consensus_delta": game.consensus_delta,
                "consensus_fires": fires,
                "consensus_arm_flip": flip,
                "explanation": (
                    "The retired rule would have switched this pick to the side the whole "
                    "market drifted toward."
                    if flip
                    else "The retired rule would have left this pick where it is."
                ),
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )
    if not rows:
        return empty, {
            "skipped": True,
            "reason": "No eligible game carried a captured line on this pass.",
        }
    frame = pd.DataFrame(rows)[list(CONSENSUS_MOVEMENT_LEDGER_COLUMNS)]
    return frame, {
        "skipped": False,
        "games_considered": len(frame),
        "rule_fires": int(frame["consensus_fires"].sum()),
        "flips": int(frame["consensus_arm_flip"].sum()),
    }


def record_consensus_movement_refresh_overlay(
    artifacts_root: Path, plan: RefreshResult, *, record_decisions: bool = False
) -> dict[str, Any]:
    """Append both arms in a separate ledger, once per game and refresh run."""

    result: dict[str, Any] = {"challenger_id": CHALLENGER_ID, "recorded": 0}
    if not record_decisions:
        return {**result, "skipped": True, "reason": "Recording was not requested."}
    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    if original.empty:
        return {**result, "skipped": True, "reason": "Tuesday card is absent."}
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="consensus-movement-refresh"
    )
    rows, diagnostics = build_consensus_movement_refresh_rows(plan, original=original)
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
