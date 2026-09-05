"""LEAD-54 paired totals; frozen protocol in docs/prospective_bestpick_tiebreaker.md."""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock
from nfl_ats.score_lattice import pick_consistent_top_score, score_lattice
from nfl_ats.served_total_challenger import _schedule_kickoff_utc
from nfl_ats.tiebreaker import last_game_of_week, lined_finals, newest_schedules_path

CHALLENGER_ID = "tiebreaker_low_side_shade"
LOGGER = logging.getLogger(__name__)


def skip(reason: str) -> dict[str, Any]:
    LOGGER.warning("Prospective paired record skipped: %s", reason)
    return {"recorded": 0, "skipped": True, "reason": reason}


def ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "tiebreaker_shade_decisions.parquet"


def load_decisions(artifacts_root: Path) -> pd.DataFrame:
    path = ledger_path(artifacts_root)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def settle_decisions(decisions: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Backfill totals and paired absolute errors without replacing frozen arms."""
    if decisions.empty:
        return decisions.copy()
    result = decisions.copy()
    scores = schedules.drop_duplicates("game_id").set_index("game_id")
    totals = pd.to_numeric(scores["home_score"], errors="coerce") + pd.to_numeric(
        scores["away_score"], errors="coerce"
    )
    result["actual_total"] = result["actual_total"].fillna(result["game_id"].map(totals))
    for arm in ("served", "shaded"):
        result[f"{arm}_absolute_error"] = (result[f"{arm}_total"] - result["actual_total"]).abs()
    delta = result["served_absolute_error"] - result["shaded_absolute_error"]
    result["closer_arm"] = delta.map(
        lambda value: (
            "pending"
            if pd.isna(value)
            else "shaded"
            if value > 0
            else "served"
            if value < 0
            else "tie"
        )
    )
    return result


def shaded_score(payload: dict[str, Any], schedules: pd.DataFrame) -> tuple[int, int]:
    """Move only the total centre; preserve production margin and side constraint."""
    target = float(payload["market_total"]) - 1.0
    margin = float(payload["lattice_centre_margin"])
    spread = float(payload["pick_spread_line"])
    if not all(math.isfinite(value) for value in (target, margin, spread)):
        raise ValueError("nonfinite lattice input")
    season, week = int(payload["season"]), int(payload["week"])
    previous = schedules["season"].lt(season) | (
        schedules["season"].eq(season) & schedules["week"].lt(week)
    )
    before_day = pd.to_datetime(schedules["gameday"], utc=True).lt(
        pd.Timestamp(payload["generated_at_utc"]).normalize()
    )
    history = lined_finals(schedules.loc[previous & before_day])
    if history.empty:
        raise ValueError("no prior-week lattice history")
    lattice = score_lattice(history, margin, target)
    chosen = pick_consistent_top_score(
        lattice,
        pick_side=str(payload["pick_side"]),
        spread_line=spread,
        served_total=target,
        centre_margin=margin,
    )
    if chosen is None:
        raise ValueError("no pick-consistent shaded lattice cell")
    return chosen[0], chosen[1]


def record_tiebreaker_shade_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    published_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Publication recorder; every pass settles prior rows before checking new inputs."""
    try:
        instant = pd.Timestamp(now or datetime.now(UTC))
        if instant.tzinfo is None:
            raise ValueError("recording time must include timezone")
        schedules = pd.read_parquet(newest_schedules_path(data_root))
        existing = load_decisions(artifacts_root)
        settled = settle_decisions(existing, schedules)
        if not settled.equals(existing):
            atomic_parquet(settled, ledger_path(artifacts_root))
        if published_path is None:
            return skip("no published tiebreaker artifact")
        payload = json.loads(published_path.read_text(encoding="utf-8"))
        season, week = int(payload["season"]), int(payload["week"])
        if season < 2026:
            return skip("prospective seasons start in 2026")
        if not settled.empty and (settled["season"].eq(season) & settled["week"].eq(week)).any():
            return {"recorded": 0, "already_recorded": 1, "ledger_rows": len(settled)}
        generated = pd.Timestamp(payload["generated_at_utc"])
        if generated.tzinfo is None or generated != instant:
            return skip("tiebreaker artifact is not from this publication")
        games = schedules.loc[
            schedules["season"].eq(season)
            & schedules["week"].eq(week)
            & schedules["game_type"].eq("REG")
        ]
        last = last_game_of_week(schedules, season, week)
        if str(last["game_id"]) != payload["game_id"]:
            return skip("published tiebreaker is not the week's last game")
        kickoffs = _schedule_kickoff_utc(games)
        kickoff = _schedule_kickoff_utc(last.to_frame().T).iloc[0]
        deadline = pick_deadline(kickoff, sunday_pick_lock(kickoffs))
        refuse_if_outside_recording_lock_window(kickoffs, instant, ledger=CHALLENGER_ID)
        if (
            pd.isna(kickoff)
            or instant >= deadline
            or pd.notna(last["home_score"])
            or pd.notna(last["away_score"])
        ):
            return skip("tiebreaker is past its playable deadline or already has a score")
        home, away = shaded_score(payload, schedules)
        served = float(payload["guess_home"]) + float(payload["guess_away"])
        if not math.isfinite(served):
            return skip("nonfinite served score")
        row = pd.DataFrame(
            [
                {
                    "season": season,
                    "week": week,
                    "game_id": payload["game_id"],
                    "recorded_at_utc": instant,
                    "kickoff": kickoff,
                    "deadline": deadline,
                    "served_home": payload["guess_home"],
                    "served_away": payload["guess_away"],
                    "served_total": served,
                    "shaded_home": home,
                    "shaded_away": away,
                    "shaded_total": home + away,
                    "market_total": float(payload["market_total"]),
                    "shade_target": float(payload["market_total"]) - 1.0,
                    "lattice_rounding_delta": home + away - (float(payload["market_total"]) - 1.0),
                    "pick_side": payload["pick_side"],
                    "pick_spread_line": payload["pick_spread_line"],
                    "actual_total": float("nan"),
                    "served_absolute_error": float("nan"),
                    "shaded_absolute_error": float("nan"),
                    "closer_arm": "pending",
                }
            ]
        )
        combined = pd.concat([settled, row], ignore_index=True) if not settled.empty else row
        atomic_parquet(combined, ledger_path(artifacts_root))
        return {"recorded": 1, "ledger_rows": len(combined)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return skip(f"{CHALLENGER_ID}: {error}")
