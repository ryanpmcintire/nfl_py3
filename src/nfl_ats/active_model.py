"""Shared active-model manifest linking evaluation and weekly forecast artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.artifact_contracts import read_contract
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


def _feature_table_contract_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """ENG-09: the feature table's own stamped contract, if the manifest has one.

    Additive: read-only, never raises. Returns an empty dict for a feature
    table built before ``artifact_contracts.stamp()`` existed, so
    ``check_compatible`` sees the same ``legacy_unversioned`` shape it
    reports for any other pre-ENG-09 artifact rather than a KeyError.
    """

    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    feature_table = provenance.get("feature_table")
    if not isinstance(feature_table, dict):
        return {}
    manifest = feature_table.get("manifest")
    if not isinstance(manifest, dict):
        return {}
    contract = read_contract(manifest)
    if contract.legacy:
        return {}
    return {
        "feature_table_schema_version": contract.schema_version,
        "feature_table_builder_version": contract.builder_version,
    }


def _ridge_alpha(metadata: dict[str, Any]) -> float | None:
    if metadata.get("regressor") != "ridge":
        return None
    return float(metadata.get("ridge_alpha", 10.0))


def _calibration_method(metadata: dict[str, Any]) -> str:
    return str(metadata.get("calibration_method", "none"))


def _probability_method(metadata: dict[str, Any]) -> str:
    """The residual-distribution probability read (MOD-08, 2026-08-19).

    Defaults to ``"ecdf"`` when absent -- true both for every historical
    ``margins/`` evaluation directory recorded before this field existed and
    for the raw empirical CDF those evaluations actually used, so old
    artifacts keep matching correctly. Part of the model identity so a
    ``margin-predict`` run's OWN probability method must match the
    evaluation it activates against: without this, a
    ``--probability-method ecdf`` forecast (e.g. an incumbent-tracking
    challenger built the naive way) could silently re-synchronize the active
    manifest against the pre-promotion evaluation and revert the promotion
    -- see docs/smooth_cdf_mapping.md and HANDOFF.md item 6 / the "Known
    divergence" incident this guards against.
    """

    return str(metadata.get("probability_method", "ecdf"))


def _matching_evaluation(
    artifacts_root: Path,
    forecast_metadata: dict[str, Any],
) -> Path | None:
    profile = forecast_metadata.get("feature_profile")
    regressor = forecast_metadata.get("regressor")
    ridge_alpha = _ridge_alpha(forecast_metadata)
    calibration_method = _calibration_method(forecast_metadata)
    probability_method = _probability_method(forecast_metadata)
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
        if _ridge_alpha(metadata) != ridge_alpha:
            continue
        if _calibration_method(metadata) != calibration_method:
            continue
        if _probability_method(metadata) != probability_method:
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
        "ridge_alpha": _ridge_alpha(forecast_metadata),
        "calibration_method": _calibration_method(forecast_metadata),
        "probability_method": _probability_method(forecast_metadata),
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
        # ENG-09: additive record of the feature table's own contract version
        # at fit time, if the table was stamped -- NOT folded into
        # model_identity/model_id above, so this never changes the hash an
        # existing model_id was already computed from. A later
        # check_compatible() call reads these two keys to detect a feature
        # table whose builder/schema version has since moved on.
        **_feature_table_contract_fields(forecast_metadata),
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
            "game_type": forecast_metadata.get("game_type"),
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


def matching_opener_evaluation(
    artifacts_root: Path, manifest: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    """Return the newest ``opener_evaluation/`` run matching ``manifest``'s recipe.

    ``active_ats_model.json`` only links a close-graded ``historical_evaluation``
    (see above); the pool-relevant opener-graded probability-rule accuracy lives
    in a separate ``opener_evaluation/`` artifact that is not part of the atomic
    activation manifest and must be located by matching feature profile,
    regressor, alpha, and target. Shared by ``nfl_ats.handoff`` (session
    handoff) and ``nfl_ats.readme_state`` (the README's generated active-model
    block) so both surfaces report the same number from the same lookup.
    """

    root = artifacts_root / "opener_evaluation"
    runs = (
        sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
        if root.is_dir()
        else []
    )
    for run in runs:
        metadata_path = run / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        config = metadata.get("active_model_config", {})
        expected = {
            "feature_profile": manifest.get("feature_profile"),
            "regressor": manifest.get("regressor"),
            "ridge_alpha": manifest.get("ridge_alpha", 10.0),
            "target": manifest.get("method"),
        }
        if config != expected:
            continue
        metrics = metadata.get("metrics", {})
        if not isinstance(metrics.get("opener_accuracy_probability_rule"), (int, float)):
            continue
        if not isinstance(metadata.get("games"), int):
            continue
        return run, metadata
    return None


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
