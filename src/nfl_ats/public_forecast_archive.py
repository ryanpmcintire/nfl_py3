"""Canonical, hash-chained public forecast records.

The chain is tamper-evident when a verifier retains or separately publishes a
known head hash. It is not a cryptographic signature and does not authenticate
the publisher; SKY-06 still needs an owner-held signing key or provider.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.data import DataContractError, require_columns
from nfl_ats.provenance import sha256_bytes

PUBLIC_FORECAST_SCHEMA_VERSION = 1
PUBLIC_FORECAST_RECORD_TYPE = "nfl_ats_public_forecast"
PUBLIC_FORECAST_COLUMNS = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "kickoff_utc",
    "spread_line",
    "predicted_margin",
    "home_win_probability",
    "home_cover_probability",
    "home_cover_probability_excluding_push",
    "push_probability",
    "home_loss_probability",
)
PUBLIC_FORECAST_PROVENANCE_FIELDS = (
    "model_id",
    "feature_profile",
    "probability_method",
    "model_configuration_sha256",
    "feature_table_sha256",
    "prediction_artifact_sha256",
)
_TOP_LEVEL_FIELDS = frozenset(
    (
        "schema_version",
        "record_type",
        "publication_id",
        "published_at_utc",
        "decision",
        "provenance",
        "forecasts",
        "previous_record_sha256",
        "content_sha256",
    )
)
_DECISION_FIELDS = frozenset(("label", "decision_at_utc", "inputs_observed_through_utc"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PublicForecastArchiveVerification:
    """Integrity and chronology summary for one verified archive."""

    records: int
    forecasts: int
    head_sha256: str | None
    first_published_at_utc: str | None
    last_published_at_utc: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise DataContractError("Public forecast record is not canonical JSON") from error
    return text.encode("utf-8")


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise DataContractError(f"Public forecast record has invalid {label}")
    return pd.Timestamp(parsed)


def _timestamp_text(value: Any, label: str) -> str:
    return _timestamp(value, label).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _nonempty(value: Any, label: str) -> str:
    if value is None or bool(pd.isna(value)):
        raise DataContractError(f"Public forecast {label} cannot be missing")
    text = str(value).strip()
    if not text:
        raise DataContractError(f"Public forecast {label} cannot be blank")
    return text


def _digest(value: Any, label: str) -> str:
    text = _nonempty(value, label).lower()
    if _SHA256.fullmatch(text) is None:
        raise DataContractError(f"Public forecast {label} must be a 64-character SHA-256 hash")
    return text


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DataContractError(f"Public forecast {label} must be numeric") from error
    if not math.isfinite(result):
        raise DataContractError(f"Public forecast {label} must be finite")
    return 0.0 if result == 0.0 else result


def _probability(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if not 0.0 <= result <= 1.0:
        raise DataContractError(f"Public forecast {label} must be in [0, 1]")
    return result


def _normalise_provenance(provenance: Mapping[str, Any]) -> dict[str, str]:
    keys = set(provenance)
    required = set(PUBLIC_FORECAST_PROVENANCE_FIELDS)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise DataContractError(
            "Public forecast provenance fields are invalid: " + "; ".join(details)
        )
    return {
        "model_id": _nonempty(provenance["model_id"], "model_id"),
        "feature_profile": _nonempty(provenance["feature_profile"], "feature_profile"),
        "probability_method": _nonempty(provenance["probability_method"], "probability_method"),
        "model_configuration_sha256": _digest(
            provenance["model_configuration_sha256"], "model_configuration_sha256"
        ),
        "feature_table_sha256": _digest(provenance["feature_table_sha256"], "feature_table_sha256"),
        "prediction_artifact_sha256": _digest(
            provenance["prediction_artifact_sha256"], "prediction_artifact_sha256"
        ),
    }


def _normalise_forecasts(
    forecasts: pd.DataFrame, *, published_at: pd.Timestamp
) -> list[dict[str, Any]]:
    require_columns(forecasts, PUBLIC_FORECAST_COLUMNS, "public forecasts")
    if forecasts.empty:
        raise DataContractError("A public forecast record cannot be empty")
    if forecasts["game_id"].astype(str).duplicated().any():
        raise DataContractError("A public forecast record contains duplicate game_id values")

    rows: list[dict[str, Any]] = []
    for raw in forecasts.loc[:, list(PUBLIC_FORECAST_COLUMNS)].to_dict(orient="records"):
        kickoff = _timestamp(raw["kickoff_utc"], "kickoff_utc")
        if published_at >= kickoff:
            raise DataContractError("Every public forecast must be published strictly pre-kickoff")
        season_value = _finite_float(raw["season"], "season")
        week_value = _finite_float(raw["week"], "week")
        if not season_value.is_integer() or not week_value.is_integer():
            raise DataContractError("Public forecast season/week must be integers")
        season = int(season_value)
        week = int(week_value)
        if season < 2000 or week < 1:
            raise DataContractError("Public forecast season/week are outside the supported range")
        home_team = _nonempty(raw["home_team"], "home_team")
        away_team = _nonempty(raw["away_team"], "away_team")
        if home_team == away_team:
            raise DataContractError("Public forecast home_team and away_team must differ")

        cover_excluding_push = _probability(
            raw["home_cover_probability_excluding_push"],
            "home_cover_probability_excluding_push",
        )
        push = _probability(raw["push_probability"], "push_probability")
        loss = _probability(raw["home_loss_probability"], "home_loss_probability")
        settlement_total = math.fsum((cover_excluding_push, push, loss))
        if not math.isclose(settlement_total, 1.0, abs_tol=_PROBABILITY_TOLERANCE):
            raise DataContractError("Public forecast cover/push/loss probabilities must sum to one")
        rows.append(
            {
                "game_id": _nonempty(raw["game_id"], "game_id"),
                "season": season,
                "week": week,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_utc": _timestamp_text(kickoff, "kickoff_utc"),
                "spread_line": _finite_float(raw["spread_line"], "spread_line"),
                "predicted_margin": _finite_float(raw["predicted_margin"], "predicted_margin"),
                "home_win_probability": _probability(
                    raw["home_win_probability"], "home_win_probability"
                ),
                "home_cover_probability": _probability(
                    raw["home_cover_probability"], "home_cover_probability"
                ),
                "home_cover_probability_excluding_push": cover_excluding_push,
                "push_probability": push,
                "home_loss_probability": loss,
            }
        )
    return sorted(
        rows, key=lambda row: (row["season"], row["week"], row["kickoff_utc"], row["game_id"])
    )


def build_public_forecast_record(
    forecasts: pd.DataFrame,
    *,
    publication_id: str,
    published_at_utc: Any,
    decision_label: str,
    decision_at_utc: Any,
    inputs_observed_through_utc: Any,
    provenance: Mapping[str, Any],
    previous_record_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic canonical record and its non-authenticating hash."""

    published_at = _timestamp(published_at_utc, "published_at_utc")
    decision_at = _timestamp(decision_at_utc, "decision_at_utc")
    inputs_through = _timestamp(inputs_observed_through_utc, "inputs_observed_through_utc")
    if inputs_through > decision_at:
        raise DataContractError("Forecast inputs cannot be observed after the decision timestamp")
    if decision_at > published_at:
        raise DataContractError("Forecast decision timestamp cannot be after publication")
    previous = (
        None
        if previous_record_sha256 is None
        else _digest(previous_record_sha256, "previous_record_sha256")
    )
    content: dict[str, Any] = {
        "schema_version": PUBLIC_FORECAST_SCHEMA_VERSION,
        "record_type": PUBLIC_FORECAST_RECORD_TYPE,
        "publication_id": _nonempty(publication_id, "publication_id"),
        "published_at_utc": _timestamp_text(published_at, "published_at_utc"),
        "decision": {
            "label": _nonempty(decision_label, "decision_label"),
            "decision_at_utc": _timestamp_text(decision_at, "decision_at_utc"),
            "inputs_observed_through_utc": _timestamp_text(
                inputs_through, "inputs_observed_through_utc"
            ),
        },
        "provenance": _normalise_provenance(provenance),
        "forecasts": _normalise_forecasts(forecasts, published_at=published_at),
        "previous_record_sha256": previous,
    }
    return {**content, "content_sha256": sha256_bytes(_canonical_bytes(content))}


def _verified_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    previous: str | None = None
    previous_published: pd.Timestamp | None = None
    publication_ids: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n") or raw_line == b"\n":
                raise DataContractError(
                    f"Public forecast archive line {line_number} is blank or unterminated"
                )
            encoded = raw_line[:-1]
            try:
                record = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DataContractError(
                    f"Public forecast archive line {line_number} is invalid JSON"
                ) from error
            if not isinstance(record, dict) or set(record) != _TOP_LEVEL_FIELDS:
                raise DataContractError(
                    f"Public forecast archive line {line_number} has an invalid envelope"
                )
            if encoded != _canonical_bytes(record):
                raise DataContractError(
                    f"Public forecast archive line {line_number} is not canonical JSON"
                )
            claimed_hash = _digest(record["content_sha256"], "content_sha256")
            content = {key: value for key, value in record.items() if key != "content_sha256"}
            actual_hash = sha256_bytes(_canonical_bytes(content))
            if claimed_hash != actual_hash:
                raise DataContractError(
                    f"Public forecast archive line {line_number} content hash mismatch"
                )
            if record["previous_record_sha256"] != previous:
                raise DataContractError(
                    f"Public forecast archive line {line_number} breaks the previous-record chain"
                )
            if record["schema_version"] != PUBLIC_FORECAST_SCHEMA_VERSION:
                raise DataContractError(
                    f"Public forecast archive line {line_number} has an unsupported schema"
                )
            if record["record_type"] != PUBLIC_FORECAST_RECORD_TYPE:
                raise DataContractError(
                    f"Public forecast archive line {line_number} has an invalid record type"
                )
            if (
                not isinstance(record["decision"], dict)
                or set(record["decision"]) != _DECISION_FIELDS
            ):
                raise DataContractError(
                    f"Public forecast archive line {line_number} has invalid decision provenance"
                )
            rebuilt = build_public_forecast_record(
                pd.DataFrame(record["forecasts"]),
                publication_id=record["publication_id"],
                published_at_utc=record["published_at_utc"],
                decision_label=record["decision"]["label"],
                decision_at_utc=record["decision"]["decision_at_utc"],
                inputs_observed_through_utc=record["decision"]["inputs_observed_through_utc"],
                provenance=record["provenance"],
                previous_record_sha256=previous,
            )
            if rebuilt != record:
                raise DataContractError(
                    f"Public forecast archive line {line_number} is not schema-canonical"
                )
            publication_id = str(record["publication_id"])
            if publication_id in publication_ids:
                raise DataContractError("Public forecast archive repeats a publication_id")
            published = _timestamp(record["published_at_utc"], "published_at_utc")
            if previous_published is not None and published <= previous_published:
                raise DataContractError("Public forecast publication timestamps must increase")
            publication_ids.add(publication_id)
            previous_published = published
            previous = claimed_hash
            records.append(record)
    return records


def verify_public_forecast_archive(
    path: Path, *, expected_head_sha256: str | None = None
) -> PublicForecastArchiveVerification:
    """Verify canonical encoding, chronology, content hashes, and hash links."""

    records = _verified_records(path)
    head = None if not records else str(records[-1]["content_sha256"])
    if expected_head_sha256 is not None and head != _digest(
        expected_head_sha256, "expected_head_sha256"
    ):
        raise DataContractError("Public forecast archive head does not match the pinned hash")
    return PublicForecastArchiveVerification(
        records=len(records),
        forecasts=sum(len(record["forecasts"]) for record in records),
        head_sha256=head,
        first_published_at_utc=(None if not records else str(records[0]["published_at_utc"])),
        last_published_at_utc=(None if not records else str(records[-1]["published_at_utc"])),
    )


@contextmanager
def _exclusive_archive_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise DataContractError(f"Public forecast archive is locked: {lock_path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        with suppress(OSError):
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def append_public_forecast_record(
    path: Path,
    forecasts: pd.DataFrame,
    *,
    publication_id: str,
    published_at_utc: Any,
    decision_label: str,
    decision_at_utc: Any,
    inputs_observed_through_utc: Any,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the existing chain and append one canonical line without rewriting it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_archive_lock(path):
        records = _verified_records(path) if path.exists() else []
        if any(record["publication_id"] == publication_id for record in records):
            raise DataContractError(
                f"Public forecast publication_id already exists: {publication_id}"
            )
        previous = None if not records else str(records[-1]["content_sha256"])
        record = build_public_forecast_record(
            forecasts,
            publication_id=publication_id,
            published_at_utc=published_at_utc,
            decision_label=decision_label,
            decision_at_utc=decision_at_utc,
            inputs_observed_through_utc=inputs_observed_through_utc,
            provenance=provenance,
            previous_record_sha256=previous,
        )
        if records and _timestamp(record["published_at_utc"], "published_at_utc") <= _timestamp(
            records[-1]["published_at_utc"], "published_at_utc"
        ):
            raise DataContractError("Public forecast publication timestamps must increase")
        encoded = _canonical_bytes(record) + b"\n"
        with path.open("ab", buffering=0) as handle:
            handle.write(encoded)
            os.fsync(handle.fileno())
        return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify integrity of a canonical public forecast hash chain (not a signature)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify one public forecast JSONL archive")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--expected-head-sha256")
    args = parser.parse_args(argv)
    try:
        result = verify_public_forecast_archive(
            args.archive, expected_head_sha256=args.expected_head_sha256
        )
    except (DataContractError, FileNotFoundError, OSError) as error:
        print(
            json.dumps({"status": "invalid", "error": str(error)}, sort_keys=True), file=sys.stderr
        )
        return 1
    print(json.dumps({"status": "valid", **result.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
