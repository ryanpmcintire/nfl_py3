"""Deterministic, point-in-time-safe public benchmark exchange format.

This module defines the publication-neutral foundation for SKY-07.  It writes
small, portable CSV/JSON bundles and validates their integrity and chronology.
It deliberately does not score submissions: public labels, licensing, hosting,
and leaderboard governance must be resolved before a benchmark is published.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_bytes
from nfl_ats.provenance import sha256_bytes, sha256_file

OPEN_BENCHMARK_SCHEMA_VERSION = 1
OBSERVATIONS_FILENAME = "observations.csv"
MANIFEST_FILENAME = "manifest.json"
SUBMISSION_FILENAME = "submission.csv"
SUBMISSION_MANIFEST_FILENAME = "submission.json"

SPLITS = ("train", "validation", "test")
LABELED_SPLITS = frozenset(("train", "validation"))
CORE_COLUMNS = (
    "game_id",
    "season",
    "week",
    "kickoff_utc",
    "decision_time_utc",
    "inputs_observed_through_utc",
    "home_team",
    "away_team",
    "spread_line",
    "split",
)
LABEL_COLUMNS = ("ats_margin", "cover_side")
SUBMISSION_COLUMNS = (
    "game_id",
    "home_cover_probability",
    "predicted_cover_side",
    "prediction_created_at_utc",
    "inputs_observed_through_utc",
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_ORDER = {name: index for index, name in enumerate(SPLITS)}


@dataclass(frozen=True)
class OpenBenchmarkDefinition:
    """Declared identity and distribution policy for one benchmark release."""

    benchmark_id: str
    dataset_version: str
    title: str
    feature_columns: tuple[str, ...] = ()
    license_spdx: str = "NOASSERTION"
    source_urls: tuple[str, ...] = ()
    public_url: str | None = None
    decision_policy: str = "Predictions use only inputs observed by decision_time_utc."


@dataclass(frozen=True)
class OpenBenchmarkValidation:
    """Integrity and publication-readiness summary for a benchmark bundle."""

    benchmark_id: str
    dataset_version: str
    rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    dataset_content_sha256: str
    publication_ready: bool
    publication_blockers: tuple[str, ...]


@dataclass(frozen=True)
class OpenBenchmarkSubmissionValidation:
    """Integrity summary for an unscored benchmark submission."""

    system_id: str
    rows: int
    dataset_content_sha256: str
    submission_content_sha256: str


def _canonical_json(payload: dict[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise DataContractError("Open benchmark metadata is not canonical JSON") from error
    return (text + "\n").encode("utf-8")


def _identifier(value: Any, label: str) -> str:
    text = str(value).strip().lower()
    if _IDENTIFIER.fullmatch(text) is None:
        raise DataContractError(f"{label} must match {_IDENTIFIER.pattern!r}; received {value!r}")
    return text


def _nonempty(value: Any, label: str) -> str:
    if value is None or bool(pd.isna(value)):
        raise DataContractError(f"{label} cannot be missing")
    text = str(value).strip()
    if not text:
        raise DataContractError(f"{label} cannot be blank")
    return text


def _utc_text(value: Any, label: str) -> str:
    try:
        unconverted = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise DataContractError(f"{label} must be a timezone-aware timestamp") from error
    if unconverted.tzinfo is None:
        raise DataContractError(f"{label} must be a timezone-aware timestamp")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise DataContractError(f"{label} must be a timezone-aware timestamp")
    return pd.Timestamp(parsed).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DataContractError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise DataContractError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _float_text(value: float) -> str:
    return format(value, ".17g")


def _definition_payload(definition: OpenBenchmarkDefinition) -> dict[str, Any]:
    benchmark_id = _identifier(definition.benchmark_id, "benchmark_id")
    dataset_version = _identifier(definition.dataset_version, "dataset_version")
    title = _nonempty(definition.title, "title")
    decision_policy = _nonempty(definition.decision_policy, "decision_policy")
    features = tuple(_identifier(name, "feature column") for name in definition.feature_columns)
    if len(features) != len(set(features)):
        raise DataContractError("Open benchmark feature columns must be unique")
    reserved = set(CORE_COLUMNS) | set(LABEL_COLUMNS)
    collisions = sorted(reserved.intersection(features))
    if collisions:
        raise DataContractError(
            "Open benchmark feature columns collide with reserved columns: " + ", ".join(collisions)
        )
    license_spdx = _nonempty(definition.license_spdx, "license_spdx")
    source_urls = tuple(sorted({_nonempty(url, "source URL") for url in definition.source_urls}))
    public_url = (
        None if definition.public_url is None else _nonempty(definition.public_url, "public_url")
    )
    return {
        "benchmark_id": benchmark_id,
        "dataset_version": dataset_version,
        "title": title,
        "feature_columns": sorted(features),
        "license_spdx": license_spdx,
        "source_urls": list(source_urls),
        "public_url": public_url,
        "decision_policy": decision_policy,
    }


def _cover_side(ats_margin: float) -> Literal["HOME", "AWAY", "PUSH"]:
    if ats_margin > 0.0:
        return "HOME"
    if ats_margin < 0.0:
        return "AWAY"
    return "PUSH"


def _normalise_observations(
    observations: pd.DataFrame, definition: OpenBenchmarkDefinition
) -> tuple[list[str], list[dict[str, str]]]:
    definition_payload = _definition_payload(definition)
    features = tuple(definition_payload["feature_columns"])
    columns = [*CORE_COLUMNS, *features, *LABEL_COLUMNS]
    missing = sorted(set(columns).difference(observations.columns))
    extra = sorted(set(observations.columns).difference(columns))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise DataContractError(
            "Open benchmark observation columns are invalid: " + "; ".join(details)
        )
    if observations.empty:
        raise DataContractError("Open benchmark observations cannot be empty")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for raw in observations.loc[:, columns].to_dict(orient="records"):
        game_id = _nonempty(raw["game_id"], "game_id")
        if game_id in seen_ids:
            raise DataContractError(f"Open benchmark repeats game_id {game_id!r}")
        seen_ids.add(game_id)
        try:
            season = int(raw["season"])
            week = int(raw["week"])
        except (TypeError, ValueError) as error:
            raise DataContractError("Open benchmark season/week must be integers") from error
        if season < 2000 or not 1 <= week <= 25:
            raise DataContractError("Open benchmark season/week are outside the supported range")
        kickoff = _utc_text(raw["kickoff_utc"], "kickoff_utc")
        decision = _utc_text(raw["decision_time_utc"], "decision_time_utc")
        observed = _utc_text(raw["inputs_observed_through_utc"], "inputs_observed_through_utc")
        if pd.Timestamp(observed) > pd.Timestamp(decision):
            raise DataContractError("Benchmark inputs cannot be observed after decision_time_utc")
        if pd.Timestamp(decision) >= pd.Timestamp(kickoff):
            raise DataContractError("Benchmark decision_time_utc must be strictly pre-kickoff")
        home_team = _nonempty(raw["home_team"], "home_team")
        away_team = _nonempty(raw["away_team"], "away_team")
        if home_team == away_team:
            raise DataContractError("Open benchmark home_team and away_team must differ")
        split = str(raw["split"]).strip().lower()
        if split not in _SPLIT_ORDER:
            raise DataContractError(f"Open benchmark split must be one of {SPLITS}")

        row = {
            "game_id": game_id,
            "season": str(season),
            "week": str(week),
            "kickoff_utc": kickoff,
            "decision_time_utc": decision,
            "inputs_observed_through_utc": observed,
            "home_team": home_team,
            "away_team": away_team,
            "spread_line": _float_text(_finite_float(raw["spread_line"], "spread_line")),
            "split": split,
        }
        for feature in features:
            row[feature] = _float_text(_finite_float(raw[feature], feature))
        label_missing = pd.isna(raw["ats_margin"]) and pd.isna(raw["cover_side"])
        if split in LABELED_SPLITS:
            if label_missing:
                raise DataContractError(f"{split} row {game_id!r} must include its label")
            margin = _finite_float(raw["ats_margin"], "ats_margin")
            side = _nonempty(raw["cover_side"], "cover_side").upper()
            if side != _cover_side(margin):
                raise DataContractError(f"Row {game_id!r} cover_side disagrees with ats_margin")
            row["ats_margin"] = _float_text(margin)
            row["cover_side"] = side
        else:
            if not label_missing:
                raise DataContractError("Public test rows must withhold ats_margin and cover_side")
            row["ats_margin"] = ""
            row["cover_side"] = ""
        rows.append(row)

    rows.sort(
        key=lambda row: (
            _SPLIT_ORDER[row["split"]],
            int(row["season"]),
            int(row["week"]),
            row["kickoff_utc"],
            row["game_id"],
        )
    )
    split_times = {
        split: [pd.Timestamp(row["kickoff_utc"]) for row in rows if row["split"] == split]
        for split in SPLITS
    }
    empty_splits = [split for split, values in split_times.items() if not values]
    if empty_splits:
        raise DataContractError("Open benchmark requires all splits: " + ", ".join(empty_splits))
    if max(split_times["train"]) >= min(split_times["validation"]):
        raise DataContractError("Training games must precede every validation game")
    if max(split_times["validation"]) >= min(split_times["test"]):
        raise DataContractError("Validation games must precede every test game")
    return columns, rows


def _csv_bytes(columns: list[str] | tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _publication_blockers(definition_payload: dict[str, Any]) -> tuple[str, ...]:
    blockers = []
    if definition_payload["license_spdx"] == "NOASSERTION":
        blockers.append("source licensing is not declared")
    if not definition_payload["source_urls"]:
        blockers.append("source provenance URLs are not declared")
    if definition_payload["public_url"] is None:
        blockers.append("external hosting location is not configured")
    return tuple(blockers)


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise DataContractError(f"Refusing to replace a different benchmark file: {path}")
        return
    atomic_bytes(payload, path)


def export_open_benchmark(
    observations: pd.DataFrame,
    destination: Path,
    *,
    definition: OpenBenchmarkDefinition,
) -> OpenBenchmarkValidation:
    """Write a deterministic public dataset bundle without exposing test labels."""

    definition_payload = _definition_payload(definition)
    columns, rows = _normalise_observations(observations, definition)
    observations_payload = _csv_bytes(columns, rows)
    observations_sha256 = sha256_bytes(observations_payload)
    blockers = _publication_blockers(definition_payload)
    split_counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    content_identity = {
        "definition": definition_payload,
        "observations_sha256": observations_sha256,
        "schema_version": OPEN_BENCHMARK_SCHEMA_VERSION,
    }
    dataset_content_sha256 = sha256_bytes(_canonical_json(content_identity))
    manifest = {
        "schema_version": OPEN_BENCHMARK_SCHEMA_VERSION,
        "record_type": "nfl_ats_open_benchmark",
        **definition_payload,
        "task": {
            "prediction": "forced_pick_cover_side",
            "probability": "home_cover_probability_excluding_push",
            "push_policy": "excluded_from_accuracy",
            "spread_sign": "positive_spread_line_means_home_favorite",
            "ats_margin": "home_score_minus_away_score_minus_spread_line",
        },
        "splits": {
            "order": list(SPLITS),
            "labels_public": ["train", "validation"],
            "labels_withheld": ["test"],
            "rows": split_counts,
        },
        "columns": columns,
        "files": [
            {
                "path": OBSERVATIONS_FILENAME,
                "bytes": len(observations_payload),
                "rows": len(rows),
                "sha256": observations_sha256,
            }
        ],
        "dataset_content_sha256": dataset_content_sha256,
        "publication": {"ready": not blockers, "blockers": list(blockers)},
    }
    manifest_payload = _canonical_json(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in destination.iterdir()
        if path.name not in {OBSERVATIONS_FILENAME, MANIFEST_FILENAME}
    )
    if unexpected:
        raise DataContractError(
            "Benchmark destination contains unexpected entries: " + ", ".join(unexpected)
        )
    _write_new_or_identical(destination / OBSERVATIONS_FILENAME, observations_payload)
    _write_new_or_identical(destination / MANIFEST_FILENAME, manifest_payload)
    return validate_open_benchmark(destination)


def _load_manifest(path: Path, *, record_type: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read benchmark metadata: {path}") from error
    if not isinstance(payload, dict) or payload.get("record_type") != record_type:
        raise DataContractError(f"Benchmark metadata has invalid record_type: {path}")
    if raw != _canonical_json(payload):
        raise DataContractError(f"Benchmark metadata is not canonical JSON: {path}")
    if payload.get("schema_version") != OPEN_BENCHMARK_SCHEMA_VERSION:
        raise DataContractError("Open benchmark metadata has an unsupported schema version")
    return payload


def _read_csv(path: Path, expected_columns: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DataContractError(f"Cannot read benchmark CSV: {path}") from error
    reader = csv.DictReader(io.StringIO(payload, newline=""))
    if reader.fieldnames != list(expected_columns):
        raise DataContractError(f"Benchmark CSV has an invalid header: {path}")
    return [dict(row) for row in reader]


def _definition_from_manifest(manifest: dict[str, Any]) -> OpenBenchmarkDefinition:
    try:
        return OpenBenchmarkDefinition(
            benchmark_id=manifest["benchmark_id"],
            dataset_version=manifest["dataset_version"],
            title=manifest["title"],
            feature_columns=tuple(manifest["feature_columns"]),
            license_spdx=manifest["license_spdx"],
            source_urls=tuple(manifest["source_urls"]),
            public_url=manifest["public_url"],
            decision_policy=manifest["decision_policy"],
        )
    except (KeyError, TypeError) as error:
        raise DataContractError("Open benchmark manifest has an invalid definition") from error


def validate_open_benchmark(destination: Path) -> OpenBenchmarkValidation:
    """Verify hashes, canonical bytes, chronology, split isolation, and label withholding."""

    manifest_path = destination / MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path, record_type="nfl_ats_open_benchmark")
    definition = _definition_from_manifest(manifest)
    definition_payload = _definition_payload(definition)
    expected_columns = [*CORE_COLUMNS, *definition_payload["feature_columns"], *LABEL_COLUMNS]
    if manifest.get("columns") != expected_columns:
        raise DataContractError("Open benchmark manifest columns do not match its definition")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise DataContractError(
            "Open benchmark manifest must describe exactly one observation file"
        )
    file_entry = files[0]
    if file_entry.get("path") != OBSERVATIONS_FILENAME:
        raise DataContractError("Open benchmark observation filename is invalid")
    observations_path = destination / OBSERVATIONS_FILENAME
    if not observations_path.is_file():
        raise DataContractError("Open benchmark observations.csv is missing")
    actual_sha256 = sha256_file(observations_path)
    if file_entry.get("sha256") != actual_sha256:
        raise DataContractError("Open benchmark observations.csv hash mismatch")
    if file_entry.get("bytes") != observations_path.stat().st_size:
        raise DataContractError("Open benchmark observations.csv byte count mismatch")

    raw_rows = _read_csv(observations_path, expected_columns)
    frame = pd.DataFrame(raw_rows, columns=expected_columns).replace({"": None})
    columns, rows = _normalise_observations(frame, definition)
    if observations_path.read_bytes() != _csv_bytes(columns, rows):
        raise DataContractError("Open benchmark observations.csv is not canonical")
    if file_entry.get("rows") != len(rows):
        raise DataContractError("Open benchmark observations.csv row count mismatch")
    split_counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    if manifest.get("splits", {}).get("rows") != split_counts:
        raise DataContractError("Open benchmark manifest split counts are stale")
    identity = {
        "definition": definition_payload,
        "observations_sha256": actual_sha256,
        "schema_version": OPEN_BENCHMARK_SCHEMA_VERSION,
    }
    content_sha256 = sha256_bytes(_canonical_json(identity))
    if manifest.get("dataset_content_sha256") != content_sha256:
        raise DataContractError("Open benchmark dataset content hash mismatch")
    blockers = _publication_blockers(definition_payload)
    publication = manifest.get("publication")
    if publication != {"ready": not blockers, "blockers": list(blockers)}:
        raise DataContractError("Open benchmark publication readiness is stale")
    return OpenBenchmarkValidation(
        benchmark_id=definition_payload["benchmark_id"],
        dataset_version=definition_payload["dataset_version"],
        rows=len(rows),
        train_rows=split_counts["train"],
        validation_rows=split_counts["validation"],
        test_rows=split_counts["test"],
        dataset_content_sha256=content_sha256,
        publication_ready=not blockers,
        publication_blockers=blockers,
    )


def _normalise_submission(
    submission: pd.DataFrame, test_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    missing = sorted(set(SUBMISSION_COLUMNS).difference(submission.columns))
    extra = sorted(set(submission.columns).difference(SUBMISSION_COLUMNS))
    if missing or extra:
        raise DataContractError(
            "Open benchmark submission columns are invalid: "
            + "; ".join(
                part
                for part in (
                    "missing " + ", ".join(missing) if missing else "",
                    "unexpected " + ", ".join(extra) if extra else "",
                )
                if part
            )
        )
    expected = {row["game_id"]: row for row in test_rows}
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in submission.loc[:, list(SUBMISSION_COLUMNS)].to_dict(orient="records"):
        game_id = _nonempty(raw["game_id"], "submission game_id")
        if game_id in seen:
            raise DataContractError(f"Submission repeats game_id {game_id!r}")
        if game_id not in expected:
            raise DataContractError(f"Submission contains unknown test game_id {game_id!r}")
        seen.add(game_id)
        probability = _finite_float(raw["home_cover_probability"], "home_cover_probability")
        if not 0.0 <= probability <= 1.0:
            raise DataContractError("Submission home_cover_probability must be in [0, 1]")
        side = _nonempty(raw["predicted_cover_side"], "predicted_cover_side").upper()
        if side not in {"HOME", "AWAY"}:
            raise DataContractError("A forced-pick submission must choose HOME or AWAY")
        if (probability > 0.5 and side != "HOME") or (probability < 0.5 and side != "AWAY"):
            raise DataContractError("Submission pick disagrees with home_cover_probability")
        created = _utc_text(raw["prediction_created_at_utc"], "prediction_created_at_utc")
        observed = _utc_text(raw["inputs_observed_through_utc"], "inputs_observed_through_utc")
        test_row = expected[game_id]
        if pd.Timestamp(observed) > pd.Timestamp(created):
            raise DataContractError(
                "Submission inputs cannot be observed after prediction creation"
            )
        if pd.Timestamp(created) > pd.Timestamp(test_row["decision_time_utc"]):
            raise DataContractError(
                "Submission prediction was created after the benchmark deadline"
            )
        rows.append(
            {
                "game_id": game_id,
                "home_cover_probability": _float_text(probability),
                "predicted_cover_side": side,
                "prediction_created_at_utc": created,
                "inputs_observed_through_utc": observed,
            }
        )
    missing_ids = sorted(set(expected).difference(seen))
    if missing_ids:
        raise DataContractError("Submission omits test game IDs: " + ", ".join(missing_ids))
    return sorted(rows, key=lambda row: row["game_id"])


def export_open_benchmark_submission(
    submission: pd.DataFrame,
    destination: Path,
    *,
    benchmark_directory: Path,
    system_id: str,
) -> OpenBenchmarkSubmissionValidation:
    """Write a deterministic, complete, unscored submission for the test split."""

    benchmark = validate_open_benchmark(benchmark_directory)
    benchmark_manifest = _load_manifest(
        benchmark_directory / MANIFEST_FILENAME, record_type="nfl_ats_open_benchmark"
    )
    expected_columns = list(benchmark_manifest["columns"])
    test_rows = [
        row
        for row in _read_csv(benchmark_directory / OBSERVATIONS_FILENAME, expected_columns)
        if row["split"] == "test"
    ]
    system = _identifier(system_id, "system_id")
    rows = _normalise_submission(submission, test_rows)
    csv_payload = _csv_bytes(SUBMISSION_COLUMNS, rows)
    content_sha256 = sha256_bytes(csv_payload)
    metadata = {
        "schema_version": OPEN_BENCHMARK_SCHEMA_VERSION,
        "record_type": "nfl_ats_open_benchmark_submission",
        "system_id": system,
        "benchmark_id": benchmark.benchmark_id,
        "dataset_version": benchmark.dataset_version,
        "dataset_content_sha256": benchmark.dataset_content_sha256,
        "benchmark_manifest_sha256": sha256_file(benchmark_directory / MANIFEST_FILENAME),
        "submission": {
            "path": SUBMISSION_FILENAME,
            "bytes": len(csv_payload),
            "rows": len(rows),
            "sha256": content_sha256,
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in destination.iterdir()
        if path.name not in {SUBMISSION_FILENAME, SUBMISSION_MANIFEST_FILENAME}
    )
    if unexpected:
        raise DataContractError(
            "Submission destination contains unexpected entries: " + ", ".join(unexpected)
        )
    _write_new_or_identical(destination / SUBMISSION_FILENAME, csv_payload)
    _write_new_or_identical(destination / SUBMISSION_MANIFEST_FILENAME, _canonical_json(metadata))
    return validate_open_benchmark_submission(destination, benchmark_directory=benchmark_directory)


def validate_open_benchmark_submission(
    destination: Path, *, benchmark_directory: Path
) -> OpenBenchmarkSubmissionValidation:
    """Verify an unscored submission against one exact benchmark test split."""

    benchmark = validate_open_benchmark(benchmark_directory)
    metadata = _load_manifest(
        destination / SUBMISSION_MANIFEST_FILENAME,
        record_type="nfl_ats_open_benchmark_submission",
    )
    if metadata.get("dataset_content_sha256") != benchmark.dataset_content_sha256:
        raise DataContractError("Submission targets a different benchmark dataset")
    if metadata.get("benchmark_manifest_sha256") != sha256_file(
        benchmark_directory / MANIFEST_FILENAME
    ):
        raise DataContractError("Submission benchmark manifest hash mismatch")
    entry = metadata.get("submission")
    if not isinstance(entry, dict) or entry.get("path") != SUBMISSION_FILENAME:
        raise DataContractError("Submission metadata has an invalid file entry")
    submission_path = destination / SUBMISSION_FILENAME
    if not submission_path.is_file() or entry.get("sha256") != sha256_file(submission_path):
        raise DataContractError("Submission CSV is missing or has a hash mismatch")
    if entry.get("bytes") != submission_path.stat().st_size:
        raise DataContractError("Submission CSV byte count mismatch")
    benchmark_manifest = _load_manifest(
        benchmark_directory / MANIFEST_FILENAME, record_type="nfl_ats_open_benchmark"
    )
    test_rows = [
        row
        for row in _read_csv(
            benchmark_directory / OBSERVATIONS_FILENAME,
            list(benchmark_manifest["columns"]),
        )
        if row["split"] == "test"
    ]
    raw_rows = _read_csv(submission_path, SUBMISSION_COLUMNS)
    rows = _normalise_submission(pd.DataFrame(raw_rows), test_rows)
    if submission_path.read_bytes() != _csv_bytes(SUBMISSION_COLUMNS, rows):
        raise DataContractError("Submission CSV is not canonical")
    if entry.get("rows") != len(rows):
        raise DataContractError("Submission CSV row count mismatch")
    system_id = _identifier(metadata.get("system_id"), "system_id")
    return OpenBenchmarkSubmissionValidation(
        system_id=system_id,
        rows=len(rows),
        dataset_content_sha256=benchmark.dataset_content_sha256,
        submission_content_sha256=str(entry["sha256"]),
    )


__all__ = [
    "CORE_COLUMNS",
    "LABEL_COLUMNS",
    "MANIFEST_FILENAME",
    "OBSERVATIONS_FILENAME",
    "OPEN_BENCHMARK_SCHEMA_VERSION",
    "SUBMISSION_COLUMNS",
    "SUBMISSION_FILENAME",
    "SUBMISSION_MANIFEST_FILENAME",
    "OpenBenchmarkDefinition",
    "OpenBenchmarkSubmissionValidation",
    "OpenBenchmarkValidation",
    "export_open_benchmark",
    "export_open_benchmark_submission",
    "validate_open_benchmark",
    "validate_open_benchmark_submission",
]
