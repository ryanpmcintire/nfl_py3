from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, time
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError, require_columns
from nfl_ats.nfl_week import week_cycle_sunday

TUESDAY_ENTRANT_ID = "best_pick_sunday_renomination:tuesday_nominee"
SUNDAY_ENTRANT_ID = "best_pick_sunday_renomination:sunday_nominee"
SUPPORTED_ENTRANT_IDS = (TUESDAY_ENTRANT_ID, SUNDAY_ENTRANT_ID)
ARM_FIELDS = (
    "game_id",
    "kickoff",
    "recorded_at_utc",
    "pick_side",
    "decision_home_spread",
    "probability",
)
LEDGER_COLUMNS = (
    "season",
    "week",
    "pool_json",
    "paired_at_utc",
    "nominees_differ",
    *(f"tuesday_{field}" for field in ARM_FIELDS),
    *(f"sunday_{field}" for field in ARM_FIELDS),
)
FEATURE_COLUMNS = ("game_id", "season", "week", "kickoff")
OUTPUT_COLUMNS = (
    "game_id",
    "season",
    "week",
    "kickoff",
    "recorded_at_utc",
    "pick_side",
    "decision_home_spread",
    "probability",
    "protocol_enrollment_at_utc",
    "enrollment_kind",
)
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class InvalidProspectiveNomineeArmError(DataContractError):
    def __init__(self, entrant_id: str, diagnostics: dict[str, Any]) -> None:
        self.entrant_id = entrant_id
        self.diagnostics = diagnostics
        invalid_rows = diagnostics.get("invalid_rows", [])
        super().__init__(
            f"{entrant_id} has {len(invalid_rows)} invalid weekly rows: {invalid_rows}"
        )


def _missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _text(value: Any, field: str) -> str:
    if _missing(value):
        raise ValueError(f"{field} is missing")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} is empty")
    return result


def _finite(value: Any, field: str) -> float:
    if _missing(value):
        raise ValueError(f"{field} is missing")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} is boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is nonfinite")
    return result


def _integral(value: Any, field: str) -> int:
    result = _finite(value, field)
    if not result.is_integer():
        raise ValueError(f"{field} is not integral")
    return int(result)


def _utc(value: Any, field: str) -> pd.Timestamp:
    if _missing(value):
        raise ValueError(f"{field} is missing")
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} is invalid") from exc
    if pd.isna(result):
        raise ValueError(f"{field} is invalid")
    if result.tzinfo is None:
        raise ValueError(f"{field} is timezone-naive")
    return result.tz_convert(UTC)


def _pool_ids(value: Any) -> list[str]:
    raw = _text(value, "pool_json")
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("pool_json is invalid JSON") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("pool_json is not a nonempty list")
    game_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("pool_json contains a non-object row")
        game_ids.append(_text(row.get("game_id"), "pool_json.game_id"))
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("pool_json contains duplicate game_id values")
    return game_ids


def _weekly_lock(kickoffs: list[pd.Timestamp]) -> pd.Timestamp:
    sundays = [week_cycle_sunday(value.tz_convert(EASTERN).date()) for value in kickoffs]
    counts = Counter(sundays)
    highest = max(counts.values())
    modes = [value for value, count in counts.items() if count == highest]
    if len(modes) != 1:
        raise ValueError("pool kickoff dates do not identify one protocol Sunday")
    return pd.Timestamp(datetime.combine(modes[0], time(16, 0), tzinfo=EASTERN)).tz_convert(UTC)


def _arm_presence(row: pd.Series, prefix: str) -> tuple[bool, bool]:
    values = [row[f"{prefix}_{field}"] for field in ARM_FIELDS]
    present = [not _missing(value) for value in values]
    return all(present), any(present)


def _arm_payload(row: pd.Series, prefix: str) -> dict[str, Any]:
    side = _text(row[f"{prefix}_pick_side"], f"{prefix}_pick_side").upper()
    if side not in {"HOME", "AWAY"}:
        raise ValueError(f"{prefix}_pick_side is unknown")
    probability = _finite(row[f"{prefix}_probability"], f"{prefix}_probability")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{prefix}_probability is outside [0, 1]")
    return {
        "game_id": _text(row[f"{prefix}_game_id"], f"{prefix}_game_id"),
        "kickoff": _utc(row[f"{prefix}_kickoff"], f"{prefix}_kickoff"),
        "recorded_at_utc": _utc(row[f"{prefix}_recorded_at_utc"], f"{prefix}_recorded_at_utc"),
        "pick_side": side,
        "decision_home_spread": _finite(
            row[f"{prefix}_decision_home_spread"],
            f"{prefix}_decision_home_spread",
        ),
        "probability": probability,
    }


def _same_payload(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left[field] == right[field] for field in ARM_FIELDS)


def _feature_identity(
    payload: dict[str, Any],
    season: int,
    week: int,
    pool_ids: set[str],
    feature_by_id: pd.DataFrame,
) -> None:
    game_id = payload["game_id"]
    if game_id not in pool_ids:
        raise ValueError(f"selected game_id {game_id} is outside pool_json")
    if game_id not in feature_by_id.index:
        raise ValueError(f"selected game_id {game_id} is absent from features")
    feature = cast(pd.Series, feature_by_id.loc[game_id])
    if _integral(feature["season"], "feature.season") != season:
        raise ValueError(f"selected game_id {game_id} has a season mismatch")
    if _integral(feature["week"], "feature.week") != week:
        raise ValueError(f"selected game_id {game_id} has a week mismatch")
    if _utc(feature["kickoff"], "feature.kickoff") != payload["kickoff"]:
        raise ValueError(f"selected game_id {game_id} has a kickoff mismatch")


def _output_row(
    payload: dict[str, Any],
    season: int,
    week: int,
    enrollment_at: pd.Timestamp,
    enrollment_kind: str,
) -> dict[str, Any]:
    return {
        **payload,
        "season": season,
        "week": week,
        "protocol_enrollment_at_utc": enrollment_at,
        "enrollment_kind": enrollment_kind,
    }


def adapt_best_pick_nominee_arm(
    ledger: pd.DataFrame,
    features: pd.DataFrame,
    *,
    entrant_id: str,
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if entrant_id not in SUPPORTED_ENTRANT_IDS:
        raise DataContractError(f"unsupported nominee entrant_id: {entrant_id}")
    require_columns(ledger, LEDGER_COLUMNS, "best-pick refresh nominee ledger")
    require_columns(features, FEATURE_COLUMNS, "game features")
    if features["game_id"].isna().any():
        raise DataContractError("game features contain missing game_id values")
    feature_ids = features["game_id"].astype(str)
    if feature_ids.duplicated().any():
        duplicates = sorted(feature_ids.loc[feature_ids.duplicated(keep=False)].unique())
        raise DataContractError(f"game features contain duplicate game_id values: {duplicates}")
    feature_by_id = features.assign(game_id=feature_ids).set_index("game_id", drop=False)
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    seen_weeks: set[tuple[int, int]] = set()
    absent_rows = 0
    fresh_rows = 0
    carry_rows = 0
    for row_number, (_, row) in enumerate(ledger.iterrows(), start=1):
        row_errors: list[str] = []
        season: int | None = None
        week: int | None = None
        try:
            season = _integral(row["season"], "season")
            week = _integral(row["week"], "week")
            week_key = (season, week)
            if week_key in seen_weeks:
                raise ValueError(f"duplicate season/week row {season}/{week}")
            seen_weeks.add(week_key)
            pool_ids = _pool_ids(row["pool_json"])
            pool_features: list[pd.Series] = []
            for game_id in pool_ids:
                if game_id not in feature_by_id.index:
                    raise ValueError(f"pool game_id {game_id} is absent from features")
                feature = cast(pd.Series, feature_by_id.loc[game_id])
                if _integral(feature["season"], "feature.season") != season:
                    raise ValueError(f"pool game_id {game_id} has a season mismatch")
                if _integral(feature["week"], "feature.week") != week:
                    raise ValueError(f"pool game_id {game_id} has a week mismatch")
                pool_features.append(feature)
            pool_kickoffs = [
                _utc(feature["kickoff"], "feature.kickoff") for feature in pool_features
            ]
            lock = _weekly_lock(pool_kickoffs)
            tuesday_complete, tuesday_any = _arm_presence(row, "tuesday")
            if not tuesday_complete:
                if tuesday_any:
                    raise ValueError("Tuesday nominee payload is partially populated")
                raise ValueError("Tuesday nominee payload is absent")
            tuesday = _arm_payload(row, "tuesday")
            _feature_identity(tuesday, season, week, set(pool_ids), feature_by_id)
            tuesday_deadline = min(tuesday["kickoff"], lock)
            if tuesday["recorded_at_utc"] >= tuesday_deadline:
                raise ValueError("Tuesday nominee was recorded at or after its deadline")
            if entrant_id == TUESDAY_ENTRANT_ID:
                rows.append(
                    _output_row(
                        tuesday,
                        season,
                        week,
                        tuesday["recorded_at_utc"],
                        "tuesday_nominee",
                    )
                )
                continue
            sunday_complete, sunday_any = _arm_presence(row, "sunday")
            if not sunday_any:
                if not _missing(row["paired_at_utc"]) or not _missing(row["nominees_differ"]):
                    raise ValueError("absent Sunday nominee has populated pairing metadata")
                absent_rows += 1
                continue
            if not sunday_complete:
                raise ValueError("Sunday nominee payload is partially populated")
            paired_at = _utc(row["paired_at_utc"], "paired_at_utc")
            differs = row["nominees_differ"]
            if not isinstance(differs, (bool, np.bool_)):
                raise ValueError("nominees_differ is not boolean")
            local_paired = paired_at.tz_convert(EASTERN)
            local_lock = lock.tz_convert(EASTERN)
            if local_paired.date() != local_lock.date() or local_paired.hour >= 12:
                raise ValueError("paired_at_utc is outside Sunday morning")
            if paired_at >= lock:
                raise ValueError("paired_at_utc is at or after the Sunday lock")
            if paired_at <= tuesday["recorded_at_utc"]:
                raise ValueError("paired_at_utc is not after the Tuesday commitment")
            sunday = _arm_payload(row, "sunday")
            _feature_identity(sunday, season, week, set(pool_ids), feature_by_id)
            if bool(differs) != (sunday["game_id"] != tuesday["game_id"]):
                raise ValueError("nominees_differ conflicts with selected game identities")
            if _same_payload(sunday, tuesday):
                if paired_at < tuesday_deadline:
                    raise ValueError("carry-forward was paired before the Tuesday deadline")
                rows.append(
                    _output_row(
                        sunday,
                        season,
                        week,
                        tuesday["recorded_at_utc"],
                        "carried_forward_tuesday_commitment",
                    )
                )
                carry_rows += 1
                continue
            sunday_deadline = min(sunday["kickoff"], lock)
            if paired_at >= tuesday_deadline:
                raise ValueError(
                    "fresh Sunday nominee appears after the Tuesday carry-forward boundary"
                )
            if sunday["recorded_at_utc"] != paired_at:
                raise ValueError("fresh Sunday nominee recorded_at_utc differs from paired_at_utc")
            if paired_at >= sunday_deadline:
                raise ValueError("fresh Sunday nominee was paired at or after its deadline")
            rows.append(
                _output_row(
                    sunday,
                    season,
                    week,
                    paired_at,
                    "fresh_sunday_nominee",
                )
            )
            fresh_rows += 1
        except (TypeError, ValueError) as exc:
            row_errors.append(str(exc))
        if row_errors:
            invalid_rows.append(
                {
                    "row": row_number,
                    "season": season,
                    "week": week,
                    "reasons": row_errors,
                }
            )
    selected_ids = [row["game_id"] for row in rows]
    duplicate_ids = sorted(game_id for game_id, count in Counter(selected_ids).items() if count > 1)
    if duplicate_ids:
        invalid_rows.append(
            {
                "row": None,
                "season": None,
                "week": None,
                "reasons": [f"selected game_id values are nonunique: {duplicate_ids}"],
            }
        )
    status = "invalid_arm" if invalid_rows else "ready" if rows else "no_recorded_rows"
    result_diagnostics = {
        "entrant_id": entrant_id,
        "status": status,
        "source_rows": len(ledger),
        "selected_rows": len(rows),
        "absent_rows": absent_rows,
        "fresh_sunday_rows": fresh_rows,
        "carried_forward_rows": carry_rows,
        "invalid_rows": invalid_rows,
    }
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(result_diagnostics)
    if invalid_rows:
        raise InvalidProspectiveNomineeArmError(entrant_id, result_diagnostics)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
