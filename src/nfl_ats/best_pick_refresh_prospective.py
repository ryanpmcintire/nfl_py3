from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.best_pick_renomination import renomination_pool, select_renominee
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.pick_refresh import RefreshResult, original_card, pick_deadline, sunday_pick_lock
from nfl_ats.prospective_scoring import settle_prospective_picks
from nfl_ats.recorder_override import replace_week_rows
from nfl_ats.tiebreaker import newest_schedules_path
from nfl_ats.tiebreaker_shade_prospective import skip

CHALLENGER_ID = "best_pick_sunday_renomination"
CONTENDER_LEDGER = "prospective/best_pick_refresh_contenders.parquet"
PROBABILITY_SEMANTICS = (
    "home_cover_probability is the fitted home-side probability used by the "
    "Best Pick recorder; pick_probability equals it for HOME and one minus it for AWAY"
)
CONTENDER_FIELDS = (
    "game_id",
    "kickoff",
    "selection_deadline_utc",
    "decision_home_spread",
    "pick_side",
    "home_cover_probability",
    "pick_probability",
    "pool_pass",
    "spread_std",
    "playable_at_capture",
    "eligible_at_capture",
    "eligibility_reason",
    "selected",
)
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


def diagnostic_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "best_pick_refresh_latest_attempt.json"


def contender_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / CONTENDER_LEDGER


def load_contender_snapshots(artifacts_root: Path) -> pd.DataFrame:
    path = contender_ledger_path(artifacts_root)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _recovery_snapshot(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.Series, str] | None:
    existing = load_contender_snapshots(artifacts_root)
    if existing.empty:
        return None
    match = existing.loc[
        existing["season"].astype(int).eq(season)
        & existing["week"].astype(int).eq(week)
        & existing["phase"].astype(str).eq(phase)
    ].copy()
    if match.empty:
        return None
    hashes = set(match["snapshot_sha256"].astype(str))
    captures = pd.to_datetime(match["captured_at_utc"], utc=True)
    selected = match.loc[match["selected"].fillna(False).astype(bool)]
    if len(hashes) != 1 or captures.nunique() != 1 or len(selected) != 1:
        raise ValueError(f"ambiguous contender recovery snapshot for {season} week {week} {phase}")
    stored_hash = next(iter(hashes))
    for column in (
        "refresh_run_id",
        "model_id",
        "feature_table_sha256",
        "source_forecast_artifact",
        "probability_semantics",
    ):
        if match[column].astype(str).nunique() != 1:
            raise ValueError(f"inconsistent contender recovery {column}")
    metadata = {
        "season": int(match["season"].iloc[0]),
        "week": int(match["week"].iloc[0]),
        "phase": str(match["phase"].iloc[0]),
        "captured_at_utc": captures.iloc[0].isoformat(),
        "refresh_run_id": str(match["refresh_run_id"].iloc[0]),
        "model_id": str(match["model_id"].iloc[0]),
        "feature_table_sha256": str(match["feature_table_sha256"].iloc[0]),
        "source_forecast_artifact": str(match["source_forecast_artifact"].iloc[0]),
        "probability_semantics": str(match["probability_semantics"].iloc[0]),
    }
    if _contender_snapshot_sha256(match, metadata) != stored_hash:
        raise ValueError(f"invalid contender recovery hash for {season} week {week} {phase}")
    return match, selected.iloc[0], stored_hash


def _canonical(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _contender_snapshot_sha256(frame: pd.DataFrame, metadata: dict[str, Any]) -> str:
    ordered = frame.loc[:, list(CONTENDER_FIELDS)].copy().sort_values("game_id")
    payload = {
        "metadata": metadata,
        "contenders": [
            {column: _canonical(value) for column, value in zip(CONTENDER_FIELDS, row, strict=True)}
            for row in ordered.itertuples(index=False, name=None)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_contender_snapshot(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    phase: str,
    captured_at_utc: pd.Timestamp,
    refresh_run_id: str,
    model_id: str,
    feature_table_sha256: str,
    source_forecast_artifact: str,
    contenders: pd.DataFrame,
) -> str:
    missing = sorted(set(CONTENDER_FIELDS).difference(contenders.columns))
    if missing:
        raise ValueError("contender snapshot is missing columns: " + ", ".join(missing))
    frame = (
        contenders.loc[:, list(CONTENDER_FIELDS)]
        .copy()
        .sort_values("game_id")
        .reset_index(drop=True)
    )
    if phase not in {"tuesday", "sunday"}:
        raise ValueError("contender snapshot phase must be tuesday or sunday")
    if frame.empty or frame["game_id"].astype(str).duplicated().any():
        raise ValueError("contender snapshot requires unique game rows")
    if not set(frame["pick_side"].astype(str)).issubset({"HOME", "AWAY"}):
        raise ValueError("contender snapshot has an invalid pick side")
    selected = frame["selected"].fillna(False).astype(bool)
    eligible = frame["eligible_at_capture"].fillna(False).astype(bool)
    if int(selected.sum()) != 1 or not bool(eligible.loc[selected].all()):
        raise ValueError("contender snapshot requires one eligible selected game")
    for column in ("kickoff", "selection_deadline_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in ("home_cover_probability", "pick_probability"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame[column].map(lambda value: math.isfinite(value) and 0 <= value <= 1).all():
            raise ValueError(f"contender snapshot has invalid {column}")
    captured = pd.Timestamp(captured_at_utc)
    captured = (
        captured.tz_localize("UTC") if captured.tzinfo is None else captured.tz_convert("UTC")
    )
    metadata = {
        "season": int(season),
        "week": int(week),
        "phase": phase,
        "captured_at_utc": captured.isoformat(),
        "refresh_run_id": refresh_run_id,
        "model_id": model_id,
        "feature_table_sha256": feature_table_sha256,
        "source_forecast_artifact": source_forecast_artifact,
        "probability_semantics": PROBABILITY_SEMANTICS,
    }
    snapshot_sha256 = _contender_snapshot_sha256(frame, metadata)
    frame["season"] = int(season)
    frame["week"] = int(week)
    frame["phase"] = phase
    frame["captured_at_utc"] = captured
    frame["refresh_run_id"] = refresh_run_id
    frame["model_id"] = model_id
    frame["feature_table_sha256"] = feature_table_sha256
    frame["source_forecast_artifact"] = source_forecast_artifact
    frame["probability_semantics"] = PROBABILITY_SEMANTICS
    frame["snapshot_sha256"] = snapshot_sha256
    existing = load_contender_snapshots(artifacts_root)
    if not existing.empty:
        match = existing.loc[
            existing["season"].astype(int).eq(season)
            & existing["week"].astype(int).eq(week)
            & existing["phase"].astype(str).eq(phase)
        ]
        if not match.empty:
            exact = match.loc[match["snapshot_sha256"].astype(str).eq(snapshot_sha256)]
            same_games = set(exact["game_id"].astype(str)) == set(frame["game_id"].astype(str))
            if len(exact) == len(frame) and same_games:
                return snapshot_sha256
            same_instant = pd.to_datetime(match["captured_at_utc"], utc=True).eq(captured)
            if bool(same_instant.any()):
                raise ValueError(
                    f"immutable contender snapshot conflicts at {season} week {week} {phase}"
                )
    combined = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame
    atomic_parquet(combined, contender_ledger_path(artifacts_root))
    return snapshot_sha256


def load_decisions(artifacts_root: Path) -> pd.DataFrame:
    path = ledger_path(artifacts_root)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def settle_decisions(decisions: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
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
    replace_week: bool = False,
) -> dict[str, Any]:
    try:
        existing = settle_ledger(artifacts_root, data_root)
        season, week = int(publication["season"]), int(publication["week"])
        if season < 2026:
            return skip("prospective seasons start in 2026")
        recorded_week = (
            existing["season"].eq(season) & existing["week"].eq(week)
            if not existing.empty
            else pd.Series(dtype=bool)
        )
        if bool(recorded_week.any()) and not replace_week:
            return {"recorded": 0, "already_recorded": 1}
        recovery = (
            None
            if replace_week
            else _recovery_snapshot(
                artifacts_root,
                season=season,
                week=week,
                phase="tuesday",
            )
        )
        if recovery is not None:
            recovered, selected, contender_snapshot_sha256 = recovery
            if str(selected["game_id"]) != publication["best_pick_game_id"]:
                return skip("publication nominee differs from pending Tuesday contender snapshot")
            recovered_row: dict[str, Any] = {
                "season": season,
                "week": week,
                "pool_json": recovered[["game_id", "pool_pass", "spread_std"]].to_json(
                    orient="records"
                ),
                "paired_at_utc": pd.NaT,
                "nominees_differ": None,
                "paired_cover_delta": float("nan"),
            }
            for arm in ("tuesday", "sunday"):
                values = {
                    "game_id": selected["game_id"],
                    "kickoff": selected["kickoff"],
                    "recorded_at_utc": selected["captured_at_utc"],
                    "pick_side": selected["pick_side"],
                    "decision_home_spread": selected["decision_home_spread"],
                    "probability": selected["pick_probability"],
                }
                for name in ARM_FIELDS:
                    recovered_row[f"{arm}_{name}"] = values[name] if arm == "tuesday" else None
                recovered_row[f"{arm}_cover"] = float("nan")
                recovered_row[f"{arm}_status"] = "pending"
            rows = pd.DataFrame([recovered_row])
            atomic_parquet(
                pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows,
                ledger_path(artifacts_root),
            )
            return {
                "recorded": 1,
                "paired": False,
                "recovered_contender_snapshot": True,
                "replaced_rows": 0,
                "left_post_kickoff": 0,
                "contender_snapshot_sha256": contender_snapshot_sha256,
            }
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
        lock = sunday_pick_lock(kickoffs)
        contenders = original[["game_id", "kickoff", "decision_home_spread", "pick_side"]].merge(
            predictions.reset_index(), on="game_id", how="left"
        )
        contenders = contenders.merge(pool, on="game_id", how="left")
        contenders["selection_deadline_utc"] = contenders["kickoff"].map(
            lambda kickoff: pick_deadline(pd.Timestamp(kickoff), lock)
        )
        contenders["pick_probability"] = contenders["home_cover_probability"].where(
            contenders["pick_side"].eq("HOME"),
            1.0 - contenders["home_cover_probability"],
        )
        contenders["pool_pass"] = contenders["pool_pass"].fillna(False).astype(bool)
        contenders["playable_at_capture"] = contenders["selection_deadline_utc"].map(
            lambda deadline: instant < deadline
        )
        contenders["eligible_at_capture"] = (
            contenders["playable_at_capture"] & contenders["pool_pass"]
        )
        contenders["eligibility_reason"] = contenders["eligible_at_capture"].map(
            lambda eligible: "pool_pass" if eligible else "not_playable_or_pool_fail"
        )
        contenders["selected"] = contenders["game_id"].astype(str).eq(str(selected["game_id"]))
        forecast_sources = (
            sorted(original["forecast_artifact"].dropna().astype(str).unique())
            if "forecast_artifact" in original.columns
            else []
        )
        contender_snapshot_sha256 = _record_contender_snapshot(
            artifacts_root,
            season=season,
            week=week,
            phase="tuesday",
            captured_at_utc=instant,
            refresh_run_id="",
            model_id=str(publication.get("model_id") or ""),
            feature_table_sha256=str(publication.get("feature_table_sha256") or ""),
            source_forecast_artifact=",".join(forecast_sources),
            contenders=contenders,
        )
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
        replaced_rows = 0
        left_post_kickoff = 0
        if replace_week and bool(recorded_week.any()):
            existing, replaced_rows, left_post_kickoff = replace_week_rows(
                existing,
                ledger_path(artifacts_root),
                season=season,
                week=week,
                recorded_at=instant,
                columns=tuple(existing.columns),
                kickoff_column="tuesday_kickoff",
            )
            if left_post_kickoff:
                return skip("the recorded Tuesday nomination's game has already kicked off")
        atomic_parquet(
            pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows,
            ledger_path(artifacts_root),
        )
        return {
            "recorded": 1,
            "paired": False,
            "replaced_rows": replaced_rows,
            "left_post_kickoff": left_post_kickoff,
            "contender_snapshot_sha256": contender_snapshot_sha256,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return skip(f"{CHALLENGER_ID}: {error}")


def _record_best_pick_refresh(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:
    if not record_decisions:
        return skip("pass --record-decisions for the Best Pick pair")
    try:
        existing = settle_ledger(artifacts_root, data_root)
        instant = plan.computed_at_utc
        if existing.empty:
            return skip("Tuesday nomination was not recorded")
        match = existing.index[existing["season"].eq(plan.season) & existing["week"].eq(plan.week)]
        if len(match) != 1:
            return skip("Tuesday nomination is missing or duplicated")
        index = match[0]
        frozen = existing.loc[index]
        if pd.notna(frozen["sunday_game_id"]):
            return {"recorded": 0, "already_recorded": 1}
        recovery = _recovery_snapshot(
            artifacts_root,
            season=int(plan.season),
            week=int(plan.week),
            phase="sunday",
        )
        if recovery is not None:
            _, selected, contender_snapshot_sha256 = recovery
            arm = {
                "game_id": selected["game_id"],
                "kickoff": selected["kickoff"],
                "recorded_at_utc": selected["captured_at_utc"],
                "pick_side": selected["pick_side"],
                "decision_home_spread": selected["decision_home_spread"],
                "probability": selected["pick_probability"],
            }
            for name, value in arm.items():
                column = f"sunday_{name}"
                existing[column] = existing[column].astype(object)
                existing.at[index, column] = value
            existing["paired_at_utc"] = pd.to_datetime(existing["paired_at_utc"], utc=True).astype(
                "datetime64[ns, UTC]"
            )
            existing.at[index, "paired_at_utc"] = selected["captured_at_utc"]
            existing["nominees_differ"] = existing["nominees_differ"].astype(object)
            existing.at[index, "nominees_differ"] = str(selected["game_id"]) != str(
                frozen["tuesday_game_id"]
            )
            atomic_parquet(existing, ledger_path(artifacts_root))
            return {
                "recorded": 1,
                "paired": True,
                "recovered_contender_snapshot": True,
                "nominees_differ": bool(existing.at[index, "nominees_differ"]),
                "contender_snapshot_sha256": contender_snapshot_sha256,
            }
        local = instant.tz_convert("America/New_York")
        if local.weekday() != 6 or local.hour >= 12:
            return skip("not the Sunday-morning nomination window")
        original = original_card(artifacts_root, season=plan.season, week=plan.week)
        kickoffs = pd.to_datetime(original["kickoff"], utc=True)
        lock = sunday_pick_lock(kickoffs)
        if local.date() != lock.tz_convert("America/New_York").date() or instant >= lock:
            return skip("refresh is outside this week's playable window")
        if pd.Timestamp(frozen["tuesday_recorded_at_utc"]) >= instant:
            return skip("Tuesday nomination is not before refresh")
        tuesday_deadline = pick_deadline(pd.Timestamp(frozen["tuesday_kickoff"]), lock)
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
        probabilities = {
            game_id: refreshed[game_id].new_home_cover_probability
            for game_id in playable["game_id"]
        }
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()):
            return skip("invalid refreshed probability")
        playable_ids = playable["game_id"].astype(str).tolist()
        pool_frame, _, _ = renomination_pool(data_root, playable_ids, instant=instant)
        contenders = playable[["game_id", "kickoff", "decision_home_spread", "pick_side"]].merge(
            pool_frame[["game_id", "spread_std", "pool_pass"]],
            on="game_id",
            how="left",
        )
        contenders["home_cover_probability"] = contenders["game_id"].map(probabilities)
        contenders["pick_side"] = contenders["game_id"].map(
            {game_id: str(refreshed[game_id].new_pick_side) for game_id in playable_ids}
        )
        contenders["pick_probability"] = contenders["home_cover_probability"].where(
            contenders["pick_side"].eq("HOME"),
            1.0 - contenders["home_cover_probability"],
        )
        contenders["pool_pass"] = contenders["pool_pass"].fillna(False).astype(bool)
        contenders["selection_deadline_utc"] = contenders["game_id"].map(
            {game_id: min(refreshed[game_id].deadline, lock) for game_id in playable_ids}
        )
        contenders["playable_at_capture"] = True
        contenders["eligible_at_capture"] = contenders["pool_pass"]
        contenders["eligibility_reason"] = contenders["pool_pass"].map(
            lambda passed: "pool_pass" if passed else "pool_fail"
        )
        contenders["selected"] = False
        if instant >= tuesday_deadline:
            arm = {name: frozen[f"tuesday_{name}"] for name in ARM_FIELDS}
            held_side = str(frozen["tuesday_pick_side"])
            held_pick_probability = float(frozen["tuesday_probability"])
            held_home_probability = (
                held_pick_probability if held_side == "HOME" else 1.0 - held_pick_probability
            )
            held = pd.DataFrame(
                [
                    {
                        "game_id": str(frozen["tuesday_game_id"]),
                        "kickoff": frozen["tuesday_kickoff"],
                        "selection_deadline_utc": tuesday_deadline,
                        "decision_home_spread": frozen["tuesday_decision_home_spread"],
                        "pick_side": held_side,
                        "home_cover_probability": held_home_probability,
                        "pick_probability": held_pick_probability,
                        "pool_pass": True,
                        "spread_std": None,
                        "playable_at_capture": False,
                        "eligible_at_capture": True,
                        "eligibility_reason": "held_locked_tuesday_nominee",
                        "selected": True,
                    }
                ]
            )
            contenders = pd.concat(
                [
                    contenders.loc[
                        ~contenders["game_id"].astype(str).eq(str(frozen["tuesday_game_id"]))
                    ],
                    held,
                ],
                ignore_index=True,
            )
        else:
            selection_pool = contenders.loc[contenders["eligible_at_capture"]].copy()
            if selection_pool.empty:
                return skip("no playable member of the Sunday nomination pool")
            selection_pool["statistic"] = selection_pool["pick_probability"]
            game_id, _, _ = select_renominee(selection_pool)
            contenders["selected"] = contenders["game_id"].astype(str).eq(str(game_id))
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
        forecast_sources = (
            sorted(original["forecast_artifact"].dropna().astype(str).unique())
            if "forecast_artifact" in original.columns
            else []
        )
        contender_snapshot_sha256 = _record_contender_snapshot(
            artifacts_root,
            season=int(plan.season),
            week=int(plan.week),
            phase="sunday",
            captured_at_utc=instant,
            refresh_run_id=str(plan.refresh_run_id),
            model_id=str(getattr(plan, "model_id", "") or ""),
            feature_table_sha256=str(getattr(plan, "feature_table_sha256", "") or ""),
            source_forecast_artifact=",".join(forecast_sources),
            contenders=contenders,
        )
        for name, value in arm.items():
            column = f"sunday_{name}"
            existing[column] = existing[column].astype(object)
            existing.at[index, column] = value
        existing["paired_at_utc"] = pd.to_datetime(existing["paired_at_utc"], utc=True).astype(
            "datetime64[ns, UTC]"
        )
        existing.at[index, "paired_at_utc"] = instant
        existing["nominees_differ"] = existing["nominees_differ"].astype(object)
        existing.at[index, "nominees_differ"] = arm["game_id"] != frozen["tuesday_game_id"]
        atomic_parquet(existing, ledger_path(artifacts_root))
        return {
            "recorded": 1,
            "paired": True,
            "nominees_differ": bool(existing.at[index, "nominees_differ"]),
            "contender_snapshot_sha256": contender_snapshot_sha256,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return skip(f"{CHALLENGER_ID}: {error}")


def record_best_pick_refresh(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    record_decisions: bool = False,
) -> dict[str, Any]:
    result = _record_best_pick_refresh(
        artifacts_root,
        data_root,
        plan,
        record_decisions=record_decisions,
    )
    if record_decisions:
        atomic_json(
            {
                "attempted_at_utc": plan.computed_at_utc,
                "season": plan.season,
                "week": plan.week,
                "refresh_run_id": plan.refresh_run_id,
                "result": result,
            },
            diagnostic_path(artifacts_root),
        )
    return result
