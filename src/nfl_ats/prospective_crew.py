from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_ats.io import atomic_parquet

SCHEMA_VERSION = "rookie_crew_prospective_v1"
ENTRANT_ID = "rookie_crew_underdog_off_incumbent"
ON_POLICY_ID = "rookie_crew_underdog_v1"
OFF_POLICY_ID = "model_only_off_arm"
LEDGER_RELATIVE_PATH = Path("prospective") / "rookie_crew_decisions.parquet"
_SIDES = {"HOME", "AWAY"}
_COLUMNS = (
    "schema_version",
    "entrant_id",
    "enrollment_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff_utc",
    "deadline_utc",
    "decision_spread_home",
    "decision_recorded_at_utc",
    "original_pick_recorded_at_utc",
    "crew_snapshot_id",
    "crew_snapshot_captured_at_utc",
    "rookie_crew_referee",
    "rookie_crew_flag",
    "on_policy_id",
    "on_pick_side",
    "on_home_cover_probability",
    "off_policy_id",
    "off_pick_side",
    "off_home_cover_probability",
    "arms_differ",
    "model_id",
    "features_sha256",
    "candidate_method",
    "probability_method",
    "calibration_method",
    "served_pick_side_at_capture",
    "served_policy_id_at_capture",
    "trigger_type",
    "trigger_source",
    "trigger_observed_at_utc",
    "payload_sha256",
)


class ProspectiveCrewValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RookieCrewProspectiveGame:
    game_id: str
    home_team: str
    away_team: str
    kickoff_utc: str | datetime | pd.Timestamp
    deadline_utc: str | datetime | pd.Timestamp
    decision_spread_home: float
    original_pick_recorded_at_utc: str | datetime | pd.Timestamp
    eligible: bool
    rookie_crew_referee: str | None
    rookie_crew_flag: int | None
    on_pick_side: str | None
    on_home_cover_probability: float | None
    off_pick_side: str | None
    off_home_cover_probability: float | None
    served_pick_side_at_capture: str
    served_policy_id_at_capture: str


@dataclass(frozen=True, slots=True)
class RookieCrewProspectivePlan:
    season: int
    week: int
    enrollment_id: str
    decision_recorded_at_utc: str | datetime | pd.Timestamp
    model_id: str
    features_sha256: str
    crew_snapshot_id: str
    crew_snapshot_captured_at_utc: str | datetime | pd.Timestamp
    candidate_method: str
    probability_method: str
    calibration_method: str
    games: Sequence[RookieCrewProspectiveGame]


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectiveCrewValidationError(f"{field} must be a nonempty string")
    return value.strip()


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProspectiveCrewValidationError(f"{field} must be bool")
    return value


def _integral(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProspectiveCrewValidationError(f"{field} must be an integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ProspectiveCrewValidationError(f"{field} must be an integer")
    return int(number)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProspectiveCrewValidationError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ProspectiveCrewValidationError(f"{field} must be finite")
    return number


def _probability(value: object, field: str) -> float:
    number = _finite(value, field)
    if number < 0.0 or number > 1.0:
        raise ProspectiveCrewValidationError(f"{field} must be between zero and one")
    return number


def _side(value: object, field: str) -> str:
    side = _text(value, field).upper()
    if side not in _SIDES:
        raise ProspectiveCrewValidationError(f"{field} must be HOME or AWAY")
    return side


def _utc(value: object, field: str) -> pd.Timestamp:
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        raise ProspectiveCrewValidationError(f"{field} must be a timestamp")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProspectiveCrewValidationError(f"{field} must be a timestamp") from exc
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise ProspectiveCrewValidationError(f"{field} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _utc_text(value: object, field: str) -> str:
    return _utc(value, field).isoformat().replace("+00:00", "Z")


def _payload_hash(row: Mapping[str, object]) -> str:
    payload = {column: row[column] for column in _COLUMNS if column != "payload_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_row(values: Mapping[str, object], *, verify_payload_hash: bool) -> dict[str, object]:
    if set(values) != set(_COLUMNS):
        missing = sorted(set(_COLUMNS) - set(values))
        extra = sorted(set(values) - set(_COLUMNS))
        raise ProspectiveCrewValidationError(
            f"ledger schema mismatch: missing={missing}, extra={extra}"
        )
    row: dict[str, object] = {
        "schema_version": _text(values["schema_version"], "schema_version"),
        "entrant_id": _text(values["entrant_id"], "entrant_id"),
        "enrollment_id": _text(values["enrollment_id"], "enrollment_id"),
        "season": _integral(values["season"], "season"),
        "week": _integral(values["week"], "week"),
        "game_id": _text(values["game_id"], "game_id"),
        "home_team": _text(values["home_team"], "home_team"),
        "away_team": _text(values["away_team"], "away_team"),
        "kickoff_utc": _utc_text(values["kickoff_utc"], "kickoff_utc"),
        "deadline_utc": _utc_text(values["deadline_utc"], "deadline_utc"),
        "decision_spread_home": _finite(values["decision_spread_home"], "decision_spread_home"),
        "decision_recorded_at_utc": _utc_text(
            values["decision_recorded_at_utc"], "decision_recorded_at_utc"
        ),
        "original_pick_recorded_at_utc": _utc_text(
            values["original_pick_recorded_at_utc"], "original_pick_recorded_at_utc"
        ),
        "crew_snapshot_id": _text(values["crew_snapshot_id"], "crew_snapshot_id"),
        "crew_snapshot_captured_at_utc": _utc_text(
            values["crew_snapshot_captured_at_utc"], "crew_snapshot_captured_at_utc"
        ),
        "rookie_crew_referee": _text(values["rookie_crew_referee"], "rookie_crew_referee"),
        "rookie_crew_flag": _integral(values["rookie_crew_flag"], "rookie_crew_flag"),
        "on_policy_id": _text(values["on_policy_id"], "on_policy_id"),
        "on_pick_side": _side(values["on_pick_side"], "on_pick_side"),
        "on_home_cover_probability": _probability(
            values["on_home_cover_probability"], "on_home_cover_probability"
        ),
        "off_policy_id": _text(values["off_policy_id"], "off_policy_id"),
        "off_pick_side": _side(values["off_pick_side"], "off_pick_side"),
        "off_home_cover_probability": _probability(
            values["off_home_cover_probability"], "off_home_cover_probability"
        ),
        "arms_differ": _boolean(values["arms_differ"], "arms_differ"),
        "model_id": _text(values["model_id"], "model_id"),
        "features_sha256": _text(values["features_sha256"], "features_sha256"),
        "candidate_method": _text(values["candidate_method"], "candidate_method"),
        "probability_method": _text(values["probability_method"], "probability_method"),
        "calibration_method": _text(values["calibration_method"], "calibration_method"),
        "served_pick_side_at_capture": _side(
            values["served_pick_side_at_capture"], "served_pick_side_at_capture"
        ),
        "served_policy_id_at_capture": _text(
            values["served_policy_id_at_capture"], "served_policy_id_at_capture"
        ),
        "trigger_type": _text(values["trigger_type"], "trigger_type"),
        "trigger_source": _text(values["trigger_source"], "trigger_source"),
        "trigger_observed_at_utc": _utc_text(
            values["trigger_observed_at_utc"], "trigger_observed_at_utc"
        ),
        "payload_sha256": _text(values["payload_sha256"], "payload_sha256"),
    }
    if row["schema_version"] != SCHEMA_VERSION or row["entrant_id"] != ENTRANT_ID:
        raise ProspectiveCrewValidationError("ledger identity does not match recorder contract")
    if row["on_policy_id"] != ON_POLICY_ID or row["off_policy_id"] != OFF_POLICY_ID:
        raise ProspectiveCrewValidationError("ledger arm identity does not match recorder contract")
    season = _integral(row["season"], "season")
    week = _integral(row["week"], "week")
    if season <= 0 or week <= 0:
        raise ProspectiveCrewValidationError("season and week must be positive")
    if row["home_team"] == row["away_team"]:
        raise ProspectiveCrewValidationError("home_team and away_team must differ")
    if row["rookie_crew_flag"] not in {-1, 1}:
        raise ProspectiveCrewValidationError("rookie_crew_flag must be -1 or 1 for recorded rows")
    on_probability = _probability(row["on_home_cover_probability"], "on_home_cover_probability")
    off_probability = _probability(row["off_home_cover_probability"], "off_home_cover_probability")
    on_expected = "HOME" if on_probability >= 0.5 else "AWAY"
    off_expected = "HOME" if off_probability >= 0.5 else "AWAY"
    if row["on_pick_side"] != on_expected or row["off_pick_side"] != off_expected:
        raise ProspectiveCrewValidationError(
            "pick side must agree with its explicit home-cover probability"
        )
    if row["arms_differ"] != (row["on_pick_side"] != row["off_pick_side"]):
        raise ProspectiveCrewValidationError("arms_differ does not match recorded arm sides")
    kickoff = _utc(row["kickoff_utc"], "kickoff_utc")
    deadline = _utc(row["deadline_utc"], "deadline_utc")
    if deadline > kickoff:
        raise ProspectiveCrewValidationError("deadline_utc must not be after kickoff_utc")
    decision_at = _utc(row["decision_recorded_at_utc"], "decision_recorded_at_utc")
    original_at = _utc(row["original_pick_recorded_at_utc"], "original_pick_recorded_at_utc")
    snapshot_at = _utc(row["crew_snapshot_captured_at_utc"], "crew_snapshot_captured_at_utc")
    trigger_at = _utc(row["trigger_observed_at_utc"], "trigger_observed_at_utc")
    for field, timestamp in (
        ("decision_recorded_at_utc", decision_at),
        ("original_pick_recorded_at_utc", original_at),
        ("crew_snapshot_captured_at_utc", snapshot_at),
        ("trigger_observed_at_utc", trigger_at),
    ):
        if timestamp >= deadline or timestamp >= kickoff:
            raise ProspectiveCrewValidationError(
                f"{field} must be strictly pre-deadline and pre-kickoff"
            )
    if original_at > decision_at:
        raise ProspectiveCrewValidationError(
            "original_pick_recorded_at_utc must not follow capture decision"
        )
    if snapshot_at > decision_at:
        raise ProspectiveCrewValidationError("crew snapshot must not follow capture decision")
    if trigger_at > decision_at:
        raise ProspectiveCrewValidationError("trigger observation must not follow capture decision")
    expected_hash = _payload_hash(row)
    if verify_payload_hash and row["payload_sha256"] != expected_hash:
        raise ProspectiveCrewValidationError(f"payload hash mismatch for {row['game_id']}")
    row["payload_sha256"] = expected_hash
    return row


def _identity(row: Mapping[str, object]) -> tuple[str, int, int, str]:
    return (
        _text(row["entrant_id"], "entrant_id"),
        _integral(row["season"], "season"),
        _integral(row["week"], "week"),
        _text(row["game_id"], "game_id"),
    )


def _plan_context(
    plan: RookieCrewProspectivePlan,
    *,
    trigger_type: str,
    trigger_source: str,
    trigger_observed_at_utc: str | datetime | pd.Timestamp,
) -> dict[str, object]:
    season = _integral(plan.season, "season")
    week = _integral(plan.week, "week")
    if season <= 0 or week <= 0:
        raise ProspectiveCrewValidationError("season and week must be positive")
    return {
        "schema_version": SCHEMA_VERSION,
        "entrant_id": ENTRANT_ID,
        "enrollment_id": _text(plan.enrollment_id, "enrollment_id"),
        "season": season,
        "week": week,
        "decision_recorded_at_utc": _utc_text(
            plan.decision_recorded_at_utc, "decision_recorded_at_utc"
        ),
        "crew_snapshot_id": _text(plan.crew_snapshot_id, "crew_snapshot_id"),
        "crew_snapshot_captured_at_utc": _utc_text(
            plan.crew_snapshot_captured_at_utc, "crew_snapshot_captured_at_utc"
        ),
        "on_policy_id": ON_POLICY_ID,
        "off_policy_id": OFF_POLICY_ID,
        "model_id": _text(plan.model_id, "model_id"),
        "features_sha256": _text(plan.features_sha256, "features_sha256"),
        "candidate_method": _text(plan.candidate_method, "candidate_method"),
        "probability_method": _text(plan.probability_method, "probability_method"),
        "calibration_method": _text(plan.calibration_method, "calibration_method"),
        "trigger_type": _text(trigger_type, "trigger_type"),
        "trigger_source": _text(trigger_source, "trigger_source"),
        "trigger_observed_at_utc": _utc_text(trigger_observed_at_utc, "trigger_observed_at_utc"),
    }


def _game_row(context: Mapping[str, object], game: RookieCrewProspectiveGame) -> dict[str, object]:
    on_side = _side(game.on_pick_side, "on_pick_side")
    off_side = _side(game.off_pick_side, "off_pick_side")
    row = {
        **context,
        "game_id": _text(game.game_id, "game_id"),
        "home_team": _text(game.home_team, "home_team"),
        "away_team": _text(game.away_team, "away_team"),
        "kickoff_utc": _utc_text(game.kickoff_utc, "kickoff_utc"),
        "deadline_utc": _utc_text(game.deadline_utc, "deadline_utc"),
        "decision_spread_home": _finite(game.decision_spread_home, "decision_spread_home"),
        "original_pick_recorded_at_utc": _utc_text(
            game.original_pick_recorded_at_utc, "original_pick_recorded_at_utc"
        ),
        "rookie_crew_referee": _text(game.rookie_crew_referee, "rookie_crew_referee"),
        "rookie_crew_flag": _integral(game.rookie_crew_flag, "rookie_crew_flag"),
        "on_pick_side": on_side,
        "on_home_cover_probability": _probability(
            game.on_home_cover_probability, "on_home_cover_probability"
        ),
        "off_pick_side": off_side,
        "off_home_cover_probability": _probability(
            game.off_home_cover_probability, "off_home_cover_probability"
        ),
        "arms_differ": on_side != off_side,
        "served_pick_side_at_capture": _side(
            game.served_pick_side_at_capture, "served_pick_side_at_capture"
        ),
        "served_policy_id_at_capture": _text(
            game.served_policy_id_at_capture, "served_policy_id_at_capture"
        ),
        "payload_sha256": "pending",
    }
    return _canonical_row(row, verify_payload_hash=False)


def _validated_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    canonical = [_canonical_row(row, verify_payload_hash=True) for row in rows]
    identities = [_identity(row) for row in canonical]
    if len(identities) != len(set(identities)):
        raise ProspectiveCrewValidationError("duplicate entrant season week game identity")
    return canonical


def _read_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise ProspectiveCrewValidationError(f"unable to read existing ledger: {exc}") from exc
    if tuple(frame.columns) != _COLUMNS:
        raise ProspectiveCrewValidationError(
            "existing ledger schema does not match recorder schema"
        )
    records = [
        {str(key): value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
    return _validated_rows(records)


def load_rookie_crew_prospective_decisions(artifacts_root: Path) -> pd.DataFrame:
    rows = _read_existing(artifacts_root / LEDGER_RELATIVE_PATH)
    return pd.DataFrame(rows, columns=_COLUMNS)


def record_rookie_crew_prospective(
    artifacts_root: Path,
    plan: RookieCrewProspectivePlan,
    *,
    enrolled: bool = False,
    trigger_type: str,
    trigger_source: str,
    trigger_observed_at_utc: str | datetime | pd.Timestamp,
    _write_observed_at_utc: str | datetime | pd.Timestamp | None = None,
) -> dict[str, object]:
    ledger_path = artifacts_root / LEDGER_RELATIVE_PATH
    if not _boolean(enrolled, "enrolled"):
        return {
            "status": "not_enrolled",
            "ledger_path": str(ledger_path),
            "eligible_rows": 0,
            "written_rows": 0,
            "unchanged_rows": 0,
        }
    context = _plan_context(
        plan,
        trigger_type=trigger_type,
        trigger_source=trigger_source,
        trigger_observed_at_utc=trigger_observed_at_utc,
    )
    game_ids = [_text(game.game_id, "game_id") for game in plan.games]
    if len(game_ids) != len(set(game_ids)):
        raise ProspectiveCrewValidationError("plan contains duplicate game_id values")
    rows = [_game_row(context, game) for game in plan.games if _boolean(game.eligible, "eligible")]
    if not rows:
        return {
            "status": "no_eligible_rows",
            "ledger_path": str(ledger_path),
            "eligible_rows": 0,
            "written_rows": 0,
            "unchanged_rows": 0,
        }
    rows = _validated_rows(rows)
    existing = _read_existing(ledger_path)
    existing_by_identity = {_identity(row): row for row in existing}
    additions: list[dict[str, object]] = []
    unchanged = 0
    for row in rows:
        prior = existing_by_identity.get(_identity(row))
        if prior is None:
            additions.append(row)
        elif prior["payload_sha256"] == row["payload_sha256"]:
            unchanged += 1
        else:
            raise ProspectiveCrewValidationError(f"immutable ledger conflict for {row['game_id']}")
    if additions:
        write_at = (
            pd.Timestamp.now(tz="UTC")
            if _write_observed_at_utc is None
            else _utc(_write_observed_at_utc, "_write_observed_at_utc")
        )
        for row in additions:
            deadline = _utc(row["deadline_utc"], "deadline_utc")
            kickoff = _utc(row["kickoff_utc"], "kickoff_utc")
            decision_at = _utc(row["decision_recorded_at_utc"], "decision_recorded_at_utc")
            if write_at < decision_at:
                raise ProspectiveCrewValidationError(
                    f"new row {row['game_id']} cannot be written before its capture decision"
                )
            if write_at >= deadline or write_at >= kickoff:
                raise ProspectiveCrewValidationError(
                    f"new row {row['game_id']} must be written strictly "
                    "pre-deadline and pre-kickoff"
                )
        combined = _validated_rows([*existing, *additions])
        combined.sort(key=_identity)
        atomic_parquet(pd.DataFrame(combined, columns=_COLUMNS), ledger_path)
    return {
        "status": "recorded" if additions else "unchanged",
        "ledger_path": str(ledger_path),
        "eligible_rows": len(rows),
        "written_rows": len(additions),
        "unchanged_rows": unchanged,
    }
