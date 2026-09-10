from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card, sunday_pick_lock
from nfl_ats.prospective import (
    NFLCOM_STARTER_OUT_THRESHOLD,
    latest_nflcom_injuries_snapshot,
    nflcom_out2_starters_flip,
    nflcom_team_starter_out_counts,
)

CHALLENGER_ID = "nflcom_friday_refresh_out2_starters_v1"

NFLCOM_REFRESH_OVERLAY_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "decision_home_spread",
    "played_pick_side",
    "nflcom_would_be_pick_side",
    "nflcom_flip",
    "picked_team",
    "opponent_team",
    "picked_starter_out",
    "opponent_starter_out",
    "picked_flag_ge_threshold",
    "opponent_flag_ge_threshold",
    "injury_page_snapshot",
    "injury_page_fetched_at_utc",
    "model_id",
    "feature_table_sha256",
)


def nflcom_refresh_overlay_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "nflcom_friday_refresh_decisions.parquet"


def load_nflcom_refresh_overlay_decisions(artifacts_root: Path) -> pd.DataFrame:

    path = nflcom_refresh_overlay_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(NFLCOM_REFRESH_OVERLAY_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(NFLCOM_REFRESH_OVERLAY_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"NFL.com refresh-overlay ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(NFLCOM_REFRESH_OVERLAY_COLUMNS)]


def _friday_gate(kickoffs: pd.Series) -> pd.Timestamp:

    return sunday_pick_lock(kickoffs) - pd.Timedelta(days=2)


def build_nflcom_refresh_overlay_rows(
    plan: RefreshResult, *, data_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:

    empty = pd.DataFrame(columns=list(NFLCOM_REFRESH_OVERLAY_COLUMNS))

    eligible_games = [game for game in plan.games if game.eligible]
    kickoffs = pd.Series(pd.to_datetime([game.kickoff for game in plan.games], utc=True))
    snapshot = latest_nflcom_injuries_snapshot(data_root, week_key=(plan.season, plan.week))
    if snapshot is None:
        return empty, {
            "skipped": True,
            "reason": "no_nflcom_injuries_snapshot",
            "detail": "no data/raw/nflcom_injuries/*/manifest.json snapshot exists yet",
        }
    snaps_candidates = sorted((data_root / "players" / "raw").glob("*/snap_counts.parquet"))
    if not snaps_candidates:
        return empty, {
            "skipped": True,
            "reason": "no_snap_counts_snapshot",
            "detail": "no data/players/raw/*/snap_counts.parquet snapshot exists yet",
        }
    if not eligible_games:
        return empty, {"skipped": True, "reason": "no eligible games in this refresh pass"}
    snapshot_dir, fetched_by_week = snapshot

    fetched_raw = fetched_by_week.get((plan.season, plan.week))
    friday_gate = _friday_gate(kickoffs)
    gate_reason = ""
    if fetched_raw is None:
        gate_reason = f"page ({plan.season}, week {plan.week}) absent from snapshot manifest"
    else:
        fetched = pd.Timestamp(fetched_raw)
        fetched = (
            fetched.tz_localize("UTC") if fetched.tzinfo is None else fetched.tz_convert("UTC")
        )
        if fetched < friday_gate:
            gate_reason = (
                f"page fetched {fetched.isoformat()} is before Friday 16:00 ET of the "
                f"game week ({friday_gate.isoformat()})"
            )
    if gate_reason:
        return empty, {
            "skipped": True,
            "reason": f"freshness gate failed: {gate_reason}",
        }

    starter_out = nflcom_team_starter_out_counts(snapshot_dir, snaps_candidates[-1])
    fetched_at = pd.Timestamp(fetched_by_week[(plan.season, plan.week)])
    fetched_at = fetched_at.tz_localize("UTC") if fetched_at.tzinfo is None else fetched_at

    rows: list[dict[str, Any]] = []
    skipped_page_after_deadline: list[str] = []
    for game in eligible_games:
        if fetched_at >= pd.Timestamp(game.deadline):
            skipped_page_after_deadline.append(str(game.game_id))
            continue
        played_side = game.new_pick_side
        picked_is_home = played_side == "HOME"
        picked_raw = game.home_team if picked_is_home else game.away_team
        opponent_raw = game.away_team if picked_is_home else game.home_team
        picked_team = TEAM_ABBREVIATION_ALIASES.get(str(picked_raw), str(picked_raw))
        opponent_team = TEAM_ABBREVIATION_ALIASES.get(str(opponent_raw), str(opponent_raw))
        picked_count = starter_out.get((plan.season, plan.week, picked_team), 0)
        opponent_count = starter_out.get((plan.season, plan.week, opponent_team), 0)
        would_be_home = nflcom_out2_starters_flip(
            picked_is_home,
            float(picked_count),
            float(opponent_count),
        )
        would_be_side = "HOME" if would_be_home else "AWAY"
        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "season": plan.season,
                "week": plan.week,
                "game_id": game.game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "decision_home_spread": game.decision_home_spread,
                "played_pick_side": played_side,
                "nflcom_would_be_pick_side": would_be_side,
                "nflcom_flip": would_be_side != played_side,
                "picked_team": picked_team,
                "opponent_team": opponent_team,
                "picked_starter_out": int(picked_count),
                "opponent_starter_out": int(opponent_count),
                "picked_flag_ge_threshold": bool(picked_count >= NFLCOM_STARTER_OUT_THRESHOLD),
                "opponent_flag_ge_threshold": bool(opponent_count >= NFLCOM_STARTER_OUT_THRESHOLD),
                "injury_page_snapshot": snapshot_dir.name,
                "injury_page_fetched_at_utc": fetched_at,
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )

    frame = pd.DataFrame(rows, columns=list(NFLCOM_REFRESH_OVERLAY_COLUMNS))
    if frame.empty:
        return empty, {
            "skipped": True,
            "reason": (
                "freshness gate failed: page fetched "
                f"{fetched_at.isoformat()} is at or after the pick deadline of every "
                "eligible game in this pass"
            ),
            "page_after_deadline_skipped_game_ids": skipped_page_after_deadline,
        }
    diagnostics = {
        "skipped": False,
        "snapshot_dir": snapshot_dir.name,
        "page_fetched_at_utc": cast(pd.Timestamp, frame["injury_page_fetched_at_utc"].iloc[0]),
        "games_considered": len(frame),
        "page_after_deadline_skipped_game_ids": skipped_page_after_deadline,
        "would_flip_game_ids": frame.loc[frame["nflcom_flip"], "game_id"].astype(str).tolist(),
        "both_flagged_kept_game_ids": frame.loc[
            frame["picked_flag_ge_threshold"] & frame["opponent_flag_ge_threshold"],
            "game_id",
        ]
        .astype(str)
        .tolist(),
    }
    return frame, diagnostics


def record_nflcom_refresh_overlay(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:

    if not record_decisions:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": (
                "pass --record-decisions to append this pass's would-be picks to the "
                "NFL.com refresh-overlay ledger"
            ),
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="nflcom-refresh-overlay"
    )

    rows, diagnostics = build_nflcom_refresh_overlay_rows(plan, data_root=data_root)
    existing = load_nflcom_refresh_overlay_decisions(artifacts_root)
    if rows.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "ledger_rows": len(existing),
            **diagnostics,
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(NFLCOM_REFRESH_OVERLAY_COLUMNS)],
        nflcom_refresh_overlay_ledger_path(artifacts_root),
    )
    return {
        "challenger_id": CHALLENGER_ID,
        "recorded": len(rows),
        "ledger_rows": len(combined),
        **diagnostics,
    }
