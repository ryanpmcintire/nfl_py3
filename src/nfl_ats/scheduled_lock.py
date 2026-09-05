"""Fail-closed orchestration for the scheduled Tuesday paper forecast."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import load_paper_decisions
from nfl_ats.data import DataContractError
from nfl_ats.provenance import write_stamped_artifact

IN_CONTRACT_GAME_TYPES = frozenset({"REG", "WC", "DIV", "CON", "SB"})


@dataclass(frozen=True)
class LockTarget:
    season: int
    week: int
    game_ids: frozenset[str]


def resolve_lock_target(schedules: pd.DataFrame, *, now: datetime) -> LockTarget:
    """Resolve the one game week whose declared line-lock Tuesday is today."""

    required = {"game_id", "season", "week", "game_type", "gameday"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(f"Schedule is missing lock-day columns: {', '.join(missing)}")
    games = schedules.loc[
        schedules["game_type"].astype(str).isin(IN_CONTRACT_GAME_TYPES), list(required)
    ].copy()
    games["_gameday"] = pd.to_datetime(games["gameday"], errors="coerce")
    if games["_gameday"].isna().any():
        raise DataContractError("Schedule contains an invalid in-contract gameday")
    games["_lock_date"] = games["_gameday"].map(
        lambda value: value.date() - timedelta(days=(value.weekday() - 1) % 7)
    )
    target = games.loc[games["_lock_date"].eq(now.date())].copy()
    keys = target[["season", "week"]].drop_duplicates()
    if len(keys) != 1:
        raise DataContractError(
            f"Expected exactly one scheduled game week for lock date {now.date()}, "
            f"found {len(keys)}"
        )
    if (target["_gameday"].dt.date <= now.date()).any():
        raise DataContractError("Refusing scheduled paper forecast after a target game date began")
    game_ids = target["game_id"].astype(str)
    if game_ids.eq("").any() or game_ids.duplicated().any():
        raise DataContractError("Target schedule has blank or duplicate game_id values")
    key = keys.iloc[0]
    return LockTarget(int(key["season"]), int(key["week"]), frozenset(game_ids))


def execute_scheduled_lock(
    schedules: pd.DataFrame,
    *,
    artifacts_root: Path,
    now: datetime,
    weekly_runner: Callable[[int, int], dict[str, Any]],
    verifier: Callable[[int, int, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Record one paper forecast, or prove that the complete week already exists."""

    target = resolve_lock_target(schedules, now=now)
    ledger = load_paper_decisions(artifacts_root)
    existing = ledger.loc[ledger["season"].eq(target.season) & ledger["week"].eq(target.week)]
    if not existing.empty:
        existing_ids = frozenset(existing["game_id"].astype(str))
        if existing_ids != target.game_ids:
            raise DataContractError(
                "Refusing to append or repair a partially recorded lock week; "
                "the paper ledger is first-write-wins"
            )
        return {
            "status": "already_recorded",
            "season": target.season,
            "week": target.week,
            "games": len(target.game_ids),
        }

    summary = weekly_runner(target.season, target.week)
    expected = {
        "command": "weekly-run",
        "season": target.season,
        "week": target.week,
        "record_decisions": True,
        "dry_run": False,
        "published": True,
    }
    mismatches = [key for key, value in expected.items() if summary.get(key) != value]
    if mismatches or summary.get("failed_step"):
        raise DataContractError(
            "Scheduled weekly-run returned an unsafe summary; mismatched fields: "
            + ", ".join(sorted(mismatches or ["failed_step"]))
        )

    summary_path = (
        artifacts_root
        / "scheduled_locks"
        / f"{target.season}-week-{target.week:02d}"
        / "weekly_summary.json"
    )
    # ENG-38: write_stamped_artifact() stamps code_revision/code_dirty onto
    # the summary and writes it atomically -- a strict superset of the manual
    # atomic write this replaced, not a second path to keep in sync.
    write_stamped_artifact(summary, summary_path)

    report = verifier(target.season, target.week, summary)
    if report.get("missing") or report.get("pending_wiring"):
        raise DataContractError("Lock-day verification found missing or pending paper recorders")
    if int(report.get("paper_ledger_rows", 0)) != len(target.game_ids):
        raise DataContractError("Lock-day verification paper row count does not match the schedule")
    return {
        "status": "recorded_and_verified",
        "season": target.season,
        "week": target.week,
        "games": len(target.game_ids),
        "summary_path": str(summary_path),
    }
