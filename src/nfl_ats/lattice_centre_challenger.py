from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
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

CHALLENGER_ID = "tiebreaker_lattice_centre"

CENTRE_POLICY = "pick_deciding_margin_with_minimum_consistent_step_v1"

RULE_PICK_DECIDING_POINT = "pick_deciding_point"

RULE_MINIMUM_CONSISTENT_STEP = "minimum_consistent_step"

LOGGER = logging.getLogger(__name__)

LEDGER_COLUMNS: tuple[str, ...] = (
    "recorded_at_utc",
    "challenger_id",
    "centre_policy",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "deadline",
    "pick_side",
    "pick_spread_line",
    "served_total",
    "served_centre_margin",
    "pick_deciding_point",
    "residual_location",
    "challenger_centre_margin",
    "challenger_centre_rule",
    "served_home",
    "served_away",
    "served_score_total",
    "challenger_home",
    "challenger_away",
    "challenger_score_total",
    "actual_home",
    "actual_away",
    "actual_total",
    "served_total_absolute_error",
    "challenger_total_absolute_error",
    "served_score_absolute_error",
    "challenger_score_absolute_error",
    "closer_arm_total",
    "closer_arm_score",
)


@dataclass(frozen=True)
class CentreChoice:
    centre: float
    rule: str


def skip(reason: str) -> dict[str, Any]:
    LOGGER.warning("Prospective paired record skipped: %s", reason)
    return {"recorded": 0, "skipped": True, "reason": reason}


def strictly_on_pick_side(margin: float, spread_line: float, pick_side: str) -> bool:
    if pick_side == "HOME":
        return float(margin) > float(spread_line)
    if pick_side == "AWAY":
        return float(margin) < float(spread_line)
    raise ValueError(f"pick_side must be 'HOME' or 'AWAY', got {pick_side!r}")


def minimum_consistent_centre(spread_line: float, pick_side: str) -> float:
    if pick_side == "HOME":
        return float(math.floor(float(spread_line)) + 1)
    if pick_side == "AWAY":
        return float(math.ceil(float(spread_line)) - 1)
    raise ValueError(f"pick_side must be 'HOME' or 'AWAY', got {pick_side!r}")


def challenger_centre(point: float, spread_line: float, pick_side: str) -> CentreChoice:
    if not math.isfinite(float(point)):
        raise ValueError("the pick-deciding point must be finite")
    if strictly_on_pick_side(point, spread_line, pick_side):
        return CentreChoice(centre=float(point), rule=RULE_PICK_DECIDING_POINT)
    return CentreChoice(
        centre=minimum_consistent_centre(spread_line, pick_side),
        rule=RULE_MINIMUM_CONSISTENT_STEP,
    )


def pick_consistent_cell(
    history: pd.DataFrame,
    *,
    centre_margin: float,
    served_total: float,
    pick_side: str,
    spread_line: float,
) -> tuple[int, int, float] | None:
    lattice = score_lattice(history, centre_margin, served_total)
    chosen = pick_consistent_top_score(
        lattice,
        pick_side=pick_side,
        spread_line=spread_line,
        served_total=served_total,
        centre_margin=centre_margin,
    )
    if chosen is None:
        return None
    return int(chosen[0]), int(chosen[1]), float(chosen[3])


def walk_forward_history(
    schedules: pd.DataFrame, *, season: int, week: int, before: pd.Timestamp
) -> pd.DataFrame:
    previous = schedules["season"].lt(season) | (
        schedules["season"].eq(season) & schedules["week"].lt(week)
    )
    before_day = pd.to_datetime(schedules["gameday"], utc=True).lt(before)
    return lined_finals(schedules.loc[previous & before_day])


def forecast_pick_deciding_points(forecast_dir: Path) -> dict[str, float]:
    path = forecast_dir / "discrete_push_read.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    games = payload.get("games")
    if not isinstance(games, list):
        return {}
    points: dict[str, float] = {}
    for entry in games:
        if not isinstance(entry, dict):
            continue
        game_id = str(entry.get("game_id", ""))
        value = entry.get("point")
        if not game_id or value is None:
            continue
        try:
            points[game_id] = float(value)
        except (TypeError, ValueError):
            continue
    return points


def ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "lattice_centre_decisions.parquet"


def load_decisions(artifacts_root: Path) -> pd.DataFrame:
    path = ledger_path(artifacts_root)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def settle_decisions(decisions: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return decisions.copy()
    result = decisions.copy()
    scores = schedules.drop_duplicates("game_id").set_index("game_id")
    home = pd.to_numeric(scores["home_score"], errors="coerce")
    away = pd.to_numeric(scores["away_score"], errors="coerce")
    result["actual_home"] = result["actual_home"].fillna(result["game_id"].map(home))
    result["actual_away"] = result["actual_away"].fillna(result["game_id"].map(away))
    result["actual_total"] = result["actual_home"] + result["actual_away"]
    for arm in ("served", "challenger"):
        result[f"{arm}_total_absolute_error"] = (
            result[f"{arm}_score_total"] - result["actual_total"]
        ).abs()
        result[f"{arm}_score_absolute_error"] = (
            result[f"{arm}_home"] - result["actual_home"]
        ).abs() + (result[f"{arm}_away"] - result["actual_away"]).abs()
    for metric in ("total", "score"):
        delta = (
            result[f"served_{metric}_absolute_error"]
            - result[f"challenger_{metric}_absolute_error"]
        )
        result[f"closer_arm_{metric}"] = delta.map(
            lambda value: (
                "pending"
                if pd.isna(value)
                else "challenger"
                if value > 0
                else "served"
                if value < 0
                else "tie"
            )
        )
    return result


def record_lattice_centre_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    published_path: Path | None = None,
    now: datetime | None = None,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:
    try:
        instant = pd.Timestamp(now or datetime.now(UTC))
        if instant.tzinfo is None:
            raise ValueError("recording time must include timezone")
        schedules = pd.read_parquet(newest_schedules_path(data_root))
        existing = load_decisions(artifacts_root)
        settled = settle_decisions(existing, schedules)
        if not settled.equals(existing):
            atomic_parquet(settled, ledger_path(artifacts_root))
        if forecast_artifact is not None:
            named = (artifacts_root / forecast_artifact).resolve()
            if not named.is_dir() or artifacts_root.resolve() not in named.parents:
                return skip("named forecast artifact is not a directory under artifacts")
            published_path = named / "tiebreaker.json"
        if published_path is None or not published_path.is_file():
            return skip("no published tiebreaker artifact")
        payload = json.loads(published_path.read_text(encoding="utf-8"))
        season, week = int(payload["season"]), int(payload["week"])
        if season < 2026:
            return skip("prospective seasons start in 2026")
        recorded_week = (
            settled["season"].eq(season) & settled["week"].eq(week)
            if not settled.empty
            else pd.Series(dtype=bool)
        )
        if bool(recorded_week.any()) and not replace_week:
            return {"recorded": 0, "already_recorded": 1, "ledger_rows": len(settled)}
        generated = pd.Timestamp(payload["generated_at_utc"])
        if generated.tzinfo is None or (forecast_artifact is None and generated != instant):
            return skip("tiebreaker artifact is not from this publication")
        last = last_game_of_week(schedules, season, week)
        if str(last["game_id"]) != payload["game_id"]:
            return skip("published tiebreaker is not the week's last game")
        games = schedules.loc[
            schedules["season"].eq(season)
            & schedules["week"].eq(week)
            & schedules["game_type"].eq("REG")
        ]
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
        forecast_name = str(payload.get("forecast_artifact") or forecast_artifact or "")
        if not forecast_name:
            return skip("published tiebreaker names no forecast artifact")
        points = forecast_pick_deciding_points(artifacts_root / forecast_name)
        game_id = str(payload["game_id"])
        if game_id not in points:
            return skip(
                "the served discrete push read publishes no pick-deciding point for "
                f"{game_id}, so the challenger centre cannot be read from the served read"
            )
        pick_side = str(payload["pick_side"])
        spread_line = float(payload["pick_spread_line"])
        served_centre = float(payload["lattice_centre_margin"])
        served_total_value = float(payload["served_total"])
        point = points[game_id]
        choice = challenger_centre(point, spread_line, pick_side)
        history = walk_forward_history(
            schedules, season=season, week=week, before=pd.Timestamp(generated).normalize()
        )
        if history.empty:
            return skip("no prior-week lattice history")
        cell = pick_consistent_cell(
            history,
            centre_margin=choice.centre,
            served_total=served_total_value,
            pick_side=pick_side,
            spread_line=spread_line,
        )
        if cell is None:
            return skip("no pick-consistent lattice cell at the challenger centre")
        challenger_home, challenger_away, _tolerance = cell
        served_home = int(payload["guess_home"])
        served_away = int(payload["guess_away"])
        row = pd.DataFrame(
            [
                {
                    "recorded_at_utc": instant,
                    "challenger_id": CHALLENGER_ID,
                    "centre_policy": CENTRE_POLICY,
                    "season": season,
                    "week": week,
                    "game_id": game_id,
                    "home_team": str(payload["home"]),
                    "away_team": str(payload["away"]),
                    "kickoff": kickoff,
                    "deadline": deadline,
                    "pick_side": pick_side,
                    "pick_spread_line": spread_line,
                    "served_total": served_total_value,
                    "served_centre_margin": served_centre,
                    "pick_deciding_point": point,
                    "residual_location": point - served_centre,
                    "challenger_centre_margin": choice.centre,
                    "challenger_centre_rule": choice.rule,
                    "served_home": served_home,
                    "served_away": served_away,
                    "served_score_total": served_home + served_away,
                    "challenger_home": challenger_home,
                    "challenger_away": challenger_away,
                    "challenger_score_total": challenger_home + challenger_away,
                    "actual_home": float("nan"),
                    "actual_away": float("nan"),
                    "actual_total": float("nan"),
                    "served_total_absolute_error": float("nan"),
                    "challenger_total_absolute_error": float("nan"),
                    "served_score_absolute_error": float("nan"),
                    "challenger_score_absolute_error": float("nan"),
                    "closer_arm_total": "pending",
                    "closer_arm_score": "pending",
                }
            ]
        )
        replaced_rows = 0
        left_post_kickoff = 0
        if replace_week and bool(recorded_week.any()):
            from nfl_ats.recorder_override import replace_week_rows

            settled, replaced_rows, left_post_kickoff = replace_week_rows(
                settled,
                ledger_path(artifacts_root),
                season=season,
                week=week,
                recorded_at=instant,
                columns=tuple(settled.columns),
            )
            if left_post_kickoff:
                return skip("the recorded tiebreaker game has already kicked off")
        combined = pd.concat([settled, row], ignore_index=True) if not settled.empty else row
        atomic_parquet(combined[list(LEDGER_COLUMNS)], ledger_path(artifacts_root))
        return {
            "recorded": 1,
            "game_id": game_id,
            "served_score": f"{served_home}-{served_away}",
            "challenger_score": f"{challenger_home}-{challenger_away}",
            "challenger_centre_rule": choice.rule,
            "ledger_rows": len(combined),
            "replaced_rows": replaced_rows,
            "left_post_kickoff": left_post_kickoff,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return skip(f"{CHALLENGER_ID}: {error}")


__all__ = [
    "CENTRE_POLICY",
    "CHALLENGER_ID",
    "LEDGER_COLUMNS",
    "RULE_MINIMUM_CONSISTENT_STEP",
    "RULE_PICK_DECIDING_POINT",
    "CentreChoice",
    "challenger_centre",
    "forecast_pick_deciding_points",
    "ledger_path",
    "load_decisions",
    "minimum_consistent_centre",
    "pick_consistent_cell",
    "record_lattice_centre_decisions",
    "settle_decisions",
    "strictly_on_pick_side",
    "walk_forward_history",
]
