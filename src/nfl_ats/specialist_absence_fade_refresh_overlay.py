from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card
from nfl_ats.roster_availability_flag_features import (
    SPECIALIST_POSITIONS,
    specialist_player_slugs,
)
from nfl_ats.transaction_wire_features import canonical_team

CHALLENGER_ID = "specialist_absence_fade_refresh_v1"

OVERLAY_STATUS_APPLIED = "specialist_resolved"
OVERLAY_STATUS_NO_SNAPSHOT = "no_injury_snapshot_available"
OVERLAY_STATUS_NO_REPORT_FOR_WEEK = "no_lsp_out_report_for_week"
OVERLAY_STATUS_BOTH_OUT = "both_teams_missing_a_specialist"
OVERLAY_STATUS_NEITHER_OUT = "neither_team_missing_a_specialist"

SPECIALIST_ABSENCE_REFRESH_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "deadline",
    "decision_home_spread",
    "played_pick_side",
    "production_home_cover_probability",
    "injury_snapshot_id",
    "home_specialist_out",
    "away_specialist_out",
    "specialist_would_be_pick_side",
    "specialist_fade_flip",
    "overlay_status",
    "model_id",
    "feature_table_sha256",
)


def specialist_absence_fade_refresh_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "specialist_absence_fade_refresh_decisions.parquet"


def load_specialist_absence_fade_refresh_decisions(artifacts_root: Path) -> pd.DataFrame:

    path = specialist_absence_fade_refresh_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(SPECIALIST_ABSENCE_REFRESH_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(SPECIALIST_ABSENCE_REFRESH_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(
            f"Specialist-absence-fade refresh ledger is missing columns: {', '.join(missing)}"
        )
    return ledger[list(SPECIALIST_ABSENCE_REFRESH_COLUMNS)]


def latest_nflverse_injuries_snapshot(
    data_root: Path, *, as_of: pd.Timestamp | None = None
) -> Path | None:

    root = data_root / "raw" / "nflverse_injuries"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/injuries.parquet"))
    if as_of is not None:
        candidates = [
            path
            for path in candidates
            if pd.to_datetime(path.parent.name, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
            <= as_of
        ]
    return candidates[-1] if candidates else None


_REQUIRED_INJURY_COLUMNS = {"season", "week", "team", "position", "report_status", "game_type"}


def live_specialist_out_qualifying(injuries: pd.DataFrame) -> pd.DataFrame:

    missing = sorted(_REQUIRED_INJURY_COLUMNS.difference(injuries.columns))
    if missing:
        raise DataContractError(f"injuries is missing columns: {', '.join(missing)}")
    frame = injuries.copy()
    frame["team"] = frame["team"].astype(str).map(canonical_team)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    mask = (
        frame["position"].isin(SPECIALIST_POSITIONS)
        & frame["report_status"].astype(str).eq("Out")
        & frame["game_type"].astype(str).eq("REG")
        & frame["season"].notna()
        & frame["week"].notna()
    )
    rows = frame.loc[mask, ["season", "week", "team"]].copy()
    rows["season"] = rows["season"].astype(int)
    rows["week"] = rows["week"].astype(int)
    return rows.drop_duplicates().reset_index(drop=True)


def _opposite(side: str) -> str:
    return "AWAY" if side == "HOME" else "HOME"


@dataclass(frozen=True)
class _InjurySnapshot:
    snapshot_id: str
    out_teams_by_week: pd.DataFrame


def build_specialist_absence_fade_refresh_rows(
    plan: RefreshResult, *, data_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:

    empty = pd.DataFrame(columns=list(SPECIALIST_ABSENCE_REFRESH_COLUMNS))
    eligible_games = [game for game in plan.games if game.eligible]
    if not eligible_games:
        return empty, {"skipped": True, "reason": "no eligible games in this refresh pass"}

    snapshot_path = latest_nflverse_injuries_snapshot(
        data_root, as_of=pd.Timestamp(plan.computed_at_utc)
    )
    if snapshot_path is None:
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_NO_SNAPSHOT,
            "detail": (
                f"no {data_root / 'raw' / 'nflverse_injuries'}/*/injuries.parquet snapshot found"
            ),
        }

    try:
        injuries = pd.read_parquet(snapshot_path)
        if "date_modified" in injuries.columns:
            modified = pd.to_datetime(injuries["date_modified"], utc=True, errors="coerce")
            injuries = injuries.loc[modified.le(pd.Timestamp(plan.computed_at_utc))].copy()
        specialist_player_slugs(injuries)
        qualifying = live_specialist_out_qualifying(injuries)
    except (DataContractError, KeyError, ValueError) as error:
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_NO_SNAPSHOT,
            "detail": f"{type(error).__name__}: {error}",
            "injury_snapshot_id": snapshot_path.parent.name,
        }

    week_qualifying = qualifying.loc[
        qualifying["season"].eq(int(plan.season)) & qualifying["week"].eq(int(plan.week))
    ]
    report_for_week = (
        pd.to_numeric(injuries["season"], errors="coerce").eq(int(plan.season))
        & pd.to_numeric(injuries["week"], errors="coerce").eq(int(plan.week))
        & injuries["game_type"].astype(str).eq("REG")
    )
    if not report_for_week.any():
        return empty, {
            "skipped": True,
            "reason": OVERLAY_STATUS_NO_REPORT_FOR_WEEK,
            "detail": (
                f"no injury report resolves for season {plan.season} week {plan.week} in "
                f"{snapshot_path}"
            ),
            "injury_snapshot_id": snapshot_path.parent.name,
        }

    out_teams = set(week_qualifying["team"].astype(str))
    rows: list[dict[str, Any]] = []
    for game in eligible_games:
        home_out = canonical_team(str(game.home_team)) in out_teams
        away_out = canonical_team(str(game.away_team)) in out_teams
        if away_out and not home_out:
            would_be_side = "HOME"
            status = OVERLAY_STATUS_APPLIED
        elif home_out and not away_out:
            would_be_side = "AWAY"
            status = OVERLAY_STATUS_APPLIED
        elif home_out and away_out:
            would_be_side = game.new_pick_side
            status = OVERLAY_STATUS_BOTH_OUT
        else:
            would_be_side = game.new_pick_side
            status = OVERLAY_STATUS_NEITHER_OUT
        rows.append(
            {
                "revision_recorded_at_utc": plan.computed_at_utc,
                "refresh_run_id": plan.refresh_run_id,
                "season": plan.season,
                "week": plan.week,
                "game_id": str(game.game_id),
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.kickoff,
                "deadline": game.deadline,
                "decision_home_spread": game.decision_home_spread,
                "played_pick_side": game.new_pick_side,
                "production_home_cover_probability": float(game.new_home_cover_probability),
                "injury_snapshot_id": snapshot_path.parent.name,
                "home_specialist_out": bool(home_out),
                "away_specialist_out": bool(away_out),
                "specialist_would_be_pick_side": would_be_side,
                "specialist_fade_flip": bool(would_be_side != game.new_pick_side),
                "overlay_status": status,
                "model_id": plan.model_id,
                "feature_table_sha256": plan.feature_table_sha256,
            }
        )

    frame = pd.DataFrame(rows, columns=list(SPECIALIST_ABSENCE_REFRESH_COLUMNS))
    diagnostics = {
        "skipped": False,
        "injury_snapshot_id": snapshot_path.parent.name,
        "games_considered": len(frame),
        "home_specialist_out_game_ids": frame.loc[frame["home_specialist_out"], "game_id"].tolist(),
        "away_specialist_out_game_ids": frame.loc[frame["away_specialist_out"], "game_id"].tolist(),
        "would_flip_game_ids": frame.loc[frame["specialist_fade_flip"], "game_id"].tolist(),
        "status_counts": frame["overlay_status"].value_counts().to_dict(),
    }
    return frame, diagnostics


def record_specialist_absence_fade_refresh_overlay(
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
                "specialist-absence-fade refresh ledger"
            ),
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="specialist-absence-fade-refresh"
    )

    rows, diagnostics = build_specialist_absence_fade_refresh_rows(plan, data_root=data_root)
    existing = load_specialist_absence_fade_refresh_decisions(artifacts_root)
    if rows.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "ledger_rows": len(existing),
            **diagnostics,
        }

    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(
        combined[list(SPECIALIST_ABSENCE_REFRESH_COLUMNS)],
        specialist_absence_fade_refresh_ledger_path(artifacts_root),
    )
    return {
        "challenger_id": CHALLENGER_ID,
        "recorded": len(rows),
        "ledger_rows": len(combined),
        **diagnostics,
    }


__all__ = [
    "CHALLENGER_ID",
    "OVERLAY_STATUS_APPLIED",
    "OVERLAY_STATUS_BOTH_OUT",
    "OVERLAY_STATUS_NEITHER_OUT",
    "OVERLAY_STATUS_NO_REPORT_FOR_WEEK",
    "OVERLAY_STATUS_NO_SNAPSHOT",
    "SPECIALIST_ABSENCE_REFRESH_COLUMNS",
    "build_specialist_absence_fade_refresh_rows",
    "latest_nflverse_injuries_snapshot",
    "live_specialist_out_qualifying",
    "load_specialist_absence_fade_refresh_decisions",
    "record_specialist_absence_fade_refresh_overlay",
    "specialist_absence_fade_refresh_ledger_path",
]
