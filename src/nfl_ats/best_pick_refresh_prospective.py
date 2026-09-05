"""LEAD-53: one frozen Tuesday/Sunday nomination pair per regular-season week."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.best_pick_nomination import select_nominee
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card, pick_deadline, sunday_pick_lock
from nfl_ats.prospective_scoring import settle_prospective_picks
from nfl_ats.tiebreaker import newest_schedules_path
from nfl_ats.tiebreaker_shade_prospective import skip

CHALLENGER_ID = "best_pick_sunday_renomination"
ARM_FIELDS = (
    "game_id",
    "kickoff",
    "recorded_at_utc",
    "pick_side",
    "decision_home_spread",
    "probability",
)


def ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "best_pick_refresh_decisions.parquet"


def load_decisions(artifacts_root: Path) -> pd.DataFrame:
    path = ledger_path(artifacts_root)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def settle_decisions(decisions: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Use the existing recorded-line ATS grader separately for both frozen arms."""
    result = decisions.copy()
    if result.empty:
        return result
    outcomes = schedules[["game_id"]].copy()
    outcomes["result"] = pd.to_numeric(schedules["home_score"], errors="coerce") - pd.to_numeric(
        schedules["away_score"], errors="coerce"
    )
    for arm in ("tuesday", "sunday"):
        ready = result[f"{arm}_game_id"].notna()
        rows = result.loc[
            ready, ["season", "week", *[f"{arm}_{name}" for name in ARM_FIELDS]]
        ].rename(columns={f"{arm}_{name}": name for name in ARM_FIELDS})
        if rows.empty:
            continue
        scored = settle_prospective_picks(rows, outcomes)
        for column, source in (
            ("cover", "correct_at_decision_line"),
            ("status", "status_at_decision_line"),
        ):
            values = pd.Series(scored[source].to_numpy(), index=result.index[ready])
            pending = ready & result[f"{arm}_status"].eq("pending")
            result.loc[pending, f"{arm}_{column}"] = values.loc[pending.loc[values.index]]
    result["paired_cover_delta"] = result["sunday_cover"] - result["tuesday_cover"]
    return result


def settle_ledger(artifacts_root: Path, data_root: Path) -> pd.DataFrame:
    existing = load_decisions(artifacts_root)
    if existing.empty:
        return existing
    schedules = pd.read_parquet(newest_schedules_path(data_root))
    settled = settle_decisions(existing, schedules)
    if not settled.equals(existing):
        atomic_parquet(settled, ledger_path(artifacts_root))
    return settled


def record_best_pick_tuesday(
    artifacts_root: Path,
    data_root: Path,
    publication: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Freeze publication probabilities and eligibility only after paper recording."""
    try:
        existing = settle_ledger(artifacts_root, data_root)
        season, week = int(publication["season"]), int(publication["week"])
        if season < 2026:
            return skip("prospective seasons start in 2026")
        if not existing.empty and (existing["season"].eq(season) & existing["week"].eq(week)).any():
            return {"recorded": 0, "already_recorded": 1}
        instant = pd.Timestamp(now or datetime.now(UTC))
        original = original_card(artifacts_root, season=season, week=week)
        if original.empty:
            return skip("no recorded Tuesday card")
        nominee = original.loc[original["is_best_pick"].eq(True)]
        if len(nominee) != 1:
            return skip("Tuesday card must have exactly one Best Pick")
        selected = nominee.iloc[0]
        if str(selected["game_id"]) != publication["best_pick_game_id"]:
            return skip("publication nominee differs from frozen Tuesday nominee")
        times = pd.to_datetime(original["recorded_at_utc"], utc=True)
        if instant.tzinfo is None or not times.eq(instant).all():
            return skip("probabilities are not from the original Tuesday recording")
        kickoffs = pd.to_datetime(original["kickoff"], utc=True)
        refuse_if_outside_recording_lock_window(kickoffs, instant, ledger=CHALLENGER_ID)
        if instant >= kickoffs.min():
            return skip("original nomination must precede the week's first kickoff")
        inputs = publication["best_pick_prospective_input"]
        predictions = pd.DataFrame(inputs["predictions"]).set_index("game_id")
        pool = pd.DataFrame(inputs["pool"])
        if pool["game_id"].duplicated().any() or set(pool["game_id"]) != set(original["game_id"]):
            return skip("nomination pool does not match Tuesday card")
        home_probability = float(predictions.loc[selected["game_id"], "home_cover_probability"])
        if not math.isfinite(home_probability) or not 0 <= home_probability <= 1:
            return skip("invalid Tuesday probability")
        row: dict[str, Any] = {
            "season": season,
            "week": week,
            "pool_json": pool.to_json(orient="records"),
            "paired_at_utc": pd.NaT,
            "nominees_differ": None,
            "paired_cover_delta": float("nan"),
        }
        for arm in ("tuesday", "sunday"):
            for name in ARM_FIELDS:
                row[f"{arm}_{name}"] = (
                    selected[name] if arm == "tuesday" and name != "probability" else None
                )
            row[f"{arm}_cover"] = float("nan")
            row[f"{arm}_status"] = "pending"
        row["tuesday_probability"] = (
            home_probability if selected["pick_side"] == "HOME" else 1 - home_probability
        )
        rows = pd.DataFrame([row])
        atomic_parquet(
            pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows,
            ledger_path(artifacts_root),
        )
        return {"recorded": 1, "paired": False}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return skip(f"{CHALLENGER_ID}: {error}")


def record_best_pick_refresh(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:
    """Record the first Sunday-morning pair, including no-change refreshes."""
    if not record_decisions:
        return skip("pass --record-decisions for the Best Pick pair")
    try:
        existing = settle_ledger(artifacts_root, data_root)
        instant = plan.computed_at_utc
        local = instant.tz_convert("America/New_York")
        if local.weekday() != 6 or local.hour >= 12:
            return skip("not the Sunday-morning nomination window")
        if existing.empty:
            return skip("Tuesday nomination was not recorded")
        match = existing.index[existing["season"].eq(plan.season) & existing["week"].eq(plan.week)]
        if len(match) != 1:
            return skip("Tuesday nomination is missing or duplicated")
        index = match[0]
        frozen = existing.loc[index]
        if pd.notna(frozen["sunday_game_id"]):
            return {"recorded": 0, "already_recorded": 1}
        original = original_card(artifacts_root, season=plan.season, week=plan.week)
        kickoffs = pd.to_datetime(original["kickoff"], utc=True)
        lock = sunday_pick_lock(kickoffs)
        if local.date() != lock.tz_convert("America/New_York").date() or instant >= lock:
            return skip("refresh is outside this week's playable window")
        if pd.Timestamp(frozen["tuesday_recorded_at_utc"]) >= instant:
            return skip("Tuesday nomination is not before refresh")
        tuesday_deadline = pick_deadline(pd.Timestamp(frozen["tuesday_kickoff"]), lock)
        if instant >= tuesday_deadline:
            # Both arms retain an already-locked nominee, independent of its result.
            arm = {name: frozen[f"tuesday_{name}"] for name in ARM_FIELDS}
        else:
            playable = original.loc[
                kickoffs.map(lambda kickoff: instant < pick_deadline(kickoff, lock))
            ]
            refreshed = {
                game.game_id: game
                for game in plan.games
                if game.eligible
                and instant < min(game.kickoff, game.deadline, lock)
                and game.original_recorded_at_utc < instant
            }
            if not set(playable["game_id"]).issubset(refreshed):
                return skip("refreshed probabilities missing for playable games")
            pool = pd.DataFrame(json.loads(frozen["pool_json"]))
            pool = pool.loc[
                pool["game_id"].isin(playable["game_id"]) & pool["pool_pass"].eq(True)
            ].copy()
            if pool.empty:
                return skip("no playable member of Tuesday's nomination pool")
            probabilities = {
                game_id: refreshed[game_id].new_home_cover_probability
                for game_id in playable["game_id"]
            }
            if not all(
                math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()
            ):
                return skip("invalid refreshed probability")
            pool["candidate_dist"] = pool["game_id"].map(probabilities).sub(0.5).abs()
            game_id, _, _ = select_nominee(pool)
            game = refreshed[game_id]
            anchor = playable.set_index("game_id").loc[game_id]
            if game.new_pick_side not in {"HOME", "AWAY"} or game.decision_home_spread != float(
                cast(Any, anchor["decision_home_spread"])
            ):
                return skip("refreshed side or recorded line is invalid")
            arm = {
                "game_id": game_id,
                "kickoff": anchor["kickoff"],
                "recorded_at_utc": instant,
                "pick_side": game.new_pick_side,
                "decision_home_spread": anchor["decision_home_spread"],
                "probability": game.new_home_cover_probability
                if game.new_pick_side == "HOME"
                else 1 - game.new_home_cover_probability,
            }
        for name, value in arm.items():
            column = f"sunday_{name}"
            existing[column] = existing[column].astype(object)
            existing.at[index, column] = value
        existing["paired_at_utc"] = pd.to_datetime(existing["paired_at_utc"], utc=True)
        existing.at[index, "paired_at_utc"] = instant
        existing["nominees_differ"] = existing["nominees_differ"].astype(object)
        existing.at[index, "nominees_differ"] = arm["game_id"] != frozen["tuesday_game_id"]
        atomic_parquet(existing, ledger_path(artifacts_root))
        return {
            "recorded": 1,
            "paired": True,
            "nominees_differ": bool(existing.at[index, "nominees_differ"]),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return skip(f"{CHALLENGER_ID}: {error}")
