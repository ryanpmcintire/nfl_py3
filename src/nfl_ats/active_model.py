"""Shared active-model manifest linking evaluation and weekly forecast artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.io import atomic_json
from nfl_ats.reporting import artifact_directories, read_json

ACTIVE_ATS_MODEL_FILENAME = "active_ats_model.json"
ACTIVE_ATS_MODEL_VERSION = 1


def _feature_table_sha256(metadata: dict[str, Any]) -> str | None:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        return None
    feature_table = provenance.get("feature_table")
    if not isinstance(feature_table, dict):
        return None
    value = feature_table.get("sha256")
    return str(value) if value is not None else None


def _matching_evaluation(
    artifacts_root: Path,
    forecast_metadata: dict[str, Any],
) -> Path | None:
    profile = forecast_metadata.get("feature_profile")
    regressor = forecast_metadata.get("regressor")
    feature_sha256 = _feature_table_sha256(forecast_metadata)
    for directory in artifact_directories(artifacts_root / "margins", "summary.csv"):
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = read_json(metadata_path)
        if metadata.get("feature_profile") != profile:
            continue
        if metadata.get("regressor") != regressor:
            continue
        if _feature_table_sha256(metadata) != feature_sha256:
            continue
        return directory
    return None


def _accuracy_intervals(evaluation: Path, method: str) -> dict[str, dict[str, float]]:
    uncertainty_path = evaluation / "uncertainty.csv"
    if not uncertainty_path.is_file():
        return {}
    uncertainty = pd.read_csv(uncertainty_path)
    rows = uncertainty.loc[
        uncertainty["method"].eq(method) & uncertainty["metric"].eq("cover_accuracy")
    ]
    intervals: dict[str, dict[str, float]] = {}
    for _, row in rows.iterrows():
        block = str(row["block"])
        intervals[block] = {
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
        }
    return intervals


def activate_matching_ats_model(
    artifacts_root: Path,
    forecast_directory: Path,
    forecast_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Atomically activate a forecast only when an exact evaluation match exists."""

    method = str(forecast_metadata.get("ats_method", "market_residual"))
    evaluation = _matching_evaluation(artifacts_root, forecast_metadata)
    if evaluation is None:
        return None
    summary = pd.read_csv(evaluation / "summary.csv")
    rows = summary.loc[summary["method"].eq(method)]
    if rows.empty or pd.isna(rows.iloc[0].get("cover_accuracy")):
        return None
    row = rows.iloc[0]
    evaluation_metadata = read_json(evaluation / "metadata.json")
    model_identity = {
        "method": method,
        "feature_profile": forecast_metadata.get("feature_profile"),
        "regressor": forecast_metadata.get("regressor"),
        "feature_table_sha256": _feature_table_sha256(forecast_metadata),
        "evaluation_configuration_sha256": evaluation_metadata.get("provenance", {}).get(
            "configuration_sha256"
        ),
    }
    model_id = hashlib.sha256(
        json.dumps(model_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    evaluation_relative = evaluation.resolve().relative_to(artifacts_root.resolve())
    forecast_relative = forecast_directory.resolve().relative_to(artifacts_root.resolve())
    accuracy = float(row["cover_accuracy"])
    games = int(row["cover_games"])
    manifest: dict[str, Any] = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "target": "ats_classification",
        "model_id": model_id,
        "activated_at_utc": forecast_metadata.get("created_at_utc"),
        **model_identity,
        "historical_evaluation": {
            "artifact": evaluation_relative.as_posix(),
            "accuracy": accuracy,
            "correct": round(accuracy * games),
            "games": games,
            "intervals": _accuracy_intervals(evaluation, method),
        },
        "weekly_forecast": {
            "artifact": forecast_relative.as_posix(),
            "season": forecast_metadata.get("season"),
            "week": forecast_metadata.get("week"),
            "created_at_utc": forecast_metadata.get("created_at_utc"),
        },
    }
    atomic_json(manifest, artifacts_root / ACTIVE_ATS_MODEL_FILENAME)
    return manifest


def load_active_ats_model(artifacts_root: Path) -> dict[str, Any] | None:
    path = artifacts_root / ACTIVE_ATS_MODEL_FILENAME
    if not path.is_file():
        return None
    manifest = read_json(path)
    if manifest.get("version") != ACTIVE_ATS_MODEL_VERSION:
        raise ValueError(f"Unsupported active ATS model manifest version in {path}")
    if manifest.get("status") != "SYNCHRONIZED":
        raise ValueError(f"Active ATS model is not synchronized: {path}")
    return manifest


def active_artifact_path(
    artifacts_root: Path, manifest: dict[str, Any], section: str
) -> Path | None:
    value = manifest.get(section)
    if not isinstance(value, dict):
        return None
    artifact = value.get("artifact")
    if not isinstance(artifact, str):
        return None
    candidate = (artifacts_root / artifact).resolve()
    try:
        candidate.relative_to(artifacts_root.resolve())
    except ValueError as error:
        raise ValueError(f"Active model artifact escapes artifacts root: {candidate}") from error
    return candidate
