"""ENG-13: reproducible run replay.

Every test builds its own synthetic manifest under ``tmp_path`` -- never
reads or writes the real ``artifacts/`` tree -- except for the git-revision
checks, which deliberately point at this repository's own working tree
(read-only: ``git rev-parse HEAD`` / ``git status --porcelain``, the same
calls ``nfl_ats.provenance.git_state`` already makes elsewhere in the suite).
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats.environment_report import environment_report
from nfl_ats.provenance import sha256_file
from nfl_ats.run_replay import (
    KIND_FORECAST_METADATA,
    KIND_UNKNOWN,
    replay_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _current_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _write_feature_table(path: Path) -> None:
    pd.DataFrame({"game_id": ["2026_01_AAA_BBB"], "season": [2026]}).to_parquet(path, index=False)


def _base_manifest(
    *,
    feature_table_path: Path,
    feature_table_sha256: str,
    season: int = 2026,
    week: int = 1,
    code: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    environment_key: str = "provenance",
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A minimal forecast-``metadata.json``-shaped manifest: a ``provenance``
    block is what makes ``run_replay`` classify it as ``forecast_metadata``.
    """

    provenance: dict[str, Any] = {
        "code": code or {"revision": _current_head(), "dirty": False},
        "feature_table": {"path": str(feature_table_path), "sha256": feature_table_sha256},
        "configuration": configuration
        or {
            "season": season,
            "week": week,
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "min_edge": 0.02,
            "min_train_games": 500,
            "feature_profile": "base",
            "probability_method": "gaussian",
        },
    }
    if environment is not None and environment_key == "provenance":
        provenance["environment"] = environment
    manifest: dict[str, Any] = {
        "season": season,
        "week": week,
        "games": 0,
        "methods": [],
        "game_type": None,
        "provenance": provenance,
    }
    if environment is not None and environment_key == "top_level":
        manifest["environment"] = environment
    return manifest


# ---------------------------------------------------------------------------
# manifest-kind detection
# ---------------------------------------------------------------------------


def test_unrecognized_manifest_reports_unknown_kind_and_fails(tmp_path: Path) -> None:
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps({"nothing": "recognizable"}), encoding="utf-8")

    report = replay_manifest(manifest_path, output_root=tmp_path / "out", recompute=False)

    assert report.manifest_kind == KIND_UNKNOWN
    assert report.ok is False
    assert any("Unrecognized manifest shape" in note for note in report.notes)


# ---------------------------------------------------------------------------
# digest verification
# ---------------------------------------------------------------------------


def test_feature_table_digest_match(tmp_path: Path) -> None:
    feature_table_path = tmp_path / "game_features.parquet"
    _write_feature_table(feature_table_path)
    digest = sha256_file(feature_table_path)
    manifest = _base_manifest(feature_table_path=feature_table_path, feature_table_sha256=digest)
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = replay_manifest(manifest_path, output_root=tmp_path / "out", recompute=False)

    assert report.manifest_kind == KIND_FORECAST_METADATA
    assert report.digest_verification["ok"] is True
    assert report.digest_verification["files_verified"] == 1
    assert report.digest_verification["changed"] == []
    assert report.digest_verification["missing"] == []


def test_feature_table_digest_mismatch_is_reported_and_fails(tmp_path: Path) -> None:
    feature_table_path = tmp_path / "game_features.parquet"
    _write_feature_table(feature_table_path)
    recorded_digest = sha256_file(feature_table_path)
    # Mutate the file after the digest was recorded, exactly the tamper case
    # this whole command exists to catch.
    pd.DataFrame({"game_id": ["DIFFERENT"], "season": [2099]}).to_parquet(
        feature_table_path, index=False
    )
    manifest = _base_manifest(
        feature_table_path=feature_table_path, feature_table_sha256=recorded_digest
    )
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = replay_manifest(manifest_path, output_root=tmp_path / "out", recompute=False)

    assert report.digest_verification["ok"] is False
    assert len(report.digest_verification["changed"]) == 1
    assert report.digest_verification["changed"][0]["role"] == "feature_table"
    assert report.ok is False


def test_feature_table_missing_from_disk_is_reported(tmp_path: Path) -> None:
    feature_table_path = tmp_path / "does_not_exist.parquet"
    manifest = _base_manifest(feature_table_path=feature_table_path, feature_table_sha256="0" * 64)
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = replay_manifest(manifest_path, output_root=tmp_path / "out", recompute=False)

    assert report.digest_verification["ok"] is False
    assert len(report.digest_verification["missing"]) == 1
    # Recompute must refuse to run from unverified inputs.
    assert report.recompute is None  # recompute=False in this test
    report2 = replay_manifest(manifest_path, output_root=tmp_path / "out2", recompute=True)
    assert report2.recompute is not None
    assert report2.recompute["attempted"] is False
    assert "digest verification failed" in str(report2.recompute["reason"])


# ---------------------------------------------------------------------------
# environment comparison: cosmetic vs. reproducibility-affecting
# ---------------------------------------------------------------------------


def test_cosmetic_environment_difference_does_not_fail_replay(tmp_path: Path) -> None:
    feature_table_path = tmp_path / "game_features.parquet"
    _write_feature_table(feature_table_path)
    digest = sha256_file(feature_table_path)

    recorded_env = copy.deepcopy(environment_report())
    # platform.release is on environment_report's own cosmetic allow-list.
    recorded_env["platform"]["release"] = str(recorded_env["platform"].get("release")) + "-modified"

    manifest = _base_manifest(
        feature_table_path=feature_table_path,
        feature_table_sha256=digest,
        environment=recorded_env,
    )
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = replay_manifest(manifest_path, output_root=tmp_path / "out", recompute=False)

    assert report.environment_comparison["available"] is True
    assert "platform.release" in report.environment_comparison["cosmetic_fields"]
    assert report.environment_comparison["reproducibility_affecting"] is False
    assert report.ok is True


def test_reproducibility_affecting_environment_difference_fails_replay(tmp_path: Path) -> None:
    feature_table_path = tmp_path / "game_features.parquet"
    _write_feature_table(feature_table_path)
    digest = sha256_file(feature_table_path)

    recorded_env = copy.deepcopy(environment_report())
    # python.minor is NOT on the cosmetic allow-list, so it defaults to
    # reproducibility_affecting -- a different interpreter minor version can
    # change which code path runs.
    recorded_env["python"]["minor"] = int(recorded_env["python"]["minor"]) + 1

    manifest = _base_manifest(
        feature_table_path=feature_table_path,
        feature_table_sha256=digest,
        environment=recorded_env,
        environment_key="top_level",
    )
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = replay_manifest(manifest_path, output_root=tmp_path / "out", recompute=False)

    assert report.environment_comparison["available"] is True
    assert "python.minor" in report.environment_comparison["reproducibility_affecting_fields"]
    assert report.environment_comparison["reproducibility_affecting"] is True
    assert report.ok is False


def test_manifest_with_no_environment_block_skips_comparison(tmp_path: Path) -> None:
    feature_table_path = tmp_path / "game_features.parquet"
    _write_feature_table(feature_table_path)
    digest = sha256_file(feature_table_path)
    manifest = _base_manifest(feature_table_path=feature_table_path, feature_table_sha256=digest)
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = replay_manifest(manifest_path, output_root=tmp_path / "out", recompute=False)

    assert report.environment_comparison["available"] is False
    assert any("environment comparison skipped" in note for note in report.notes)
    # Vacuously true: nothing to compare, so this alone never fails replay.
    assert report.environment_comparison["reproducibility_affecting"] is False


# ---------------------------------------------------------------------------
# git revision
# ---------------------------------------------------------------------------


def test_git_revision_match_is_reported(tmp_path: Path) -> None:
    feature_table_path = tmp_path / "game_features.parquet"
    _write_feature_table(feature_table_path)
    digest = sha256_file(feature_table_path)
    manifest = _base_manifest(
        feature_table_path=feature_table_path,
        feature_table_sha256=digest,
        code={"revision": _current_head(), "dirty": False},
    )
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = replay_manifest(
        manifest_path, output_root=tmp_path / "out", recompute=False, repo_root=REPO_ROOT
    )

    assert report.git_revision["revision_match"] is True
    assert report.git_revision["current_revision"] == _current_head()


def test_git_revision_mismatch_is_reported_but_does_not_gate_ok(tmp_path: Path) -> None:
    feature_table_path = tmp_path / "game_features.parquet"
    _write_feature_table(feature_table_path)
    digest = sha256_file(feature_table_path)
    manifest = _base_manifest(
        feature_table_path=feature_table_path,
        feature_table_sha256=digest,
        code={"revision": "0" * 40, "dirty": False},
    )
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = replay_manifest(
        manifest_path, output_root=tmp_path / "out", recompute=False, repo_root=REPO_ROOT
    )

    assert report.git_revision["revision_match"] is False
    assert any("git revision mismatch" in note for note in report.notes)
    # Per the CLI exit-code contract: digests verify, no environment block to
    # compare, recompute not requested -- a revision mismatch alone must not
    # flip ok to False.
    assert report.ok is True


# ---------------------------------------------------------------------------
# recompute round-trip
# ---------------------------------------------------------------------------


def _recorded_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_AAA_BBB", "2026_01_AAA_BBB", "2026_01_CCC_DDD"],
            "method": ["market", "market_residual", "market_residual"],
            "model_probability": [0.52, 0.55, 0.60],
            "game_type": ["REG", "REG", "REG"],
        }
    )


def _recompute_manifest(tmp_path: Path, predictions: pd.DataFrame) -> Path:
    feature_table_path = tmp_path / "game_features.parquet"
    _write_feature_table(feature_table_path)
    digest = sha256_file(feature_table_path)

    manifest = _base_manifest(
        feature_table_path=feature_table_path,
        feature_table_sha256=digest,
        season=2026,
        week=1,
    )
    manifest["games"] = int(predictions["game_id"].nunique())
    manifest["methods"] = sorted(predictions["method"].unique().tolist())
    manifest["game_type"] = str(predictions["game_type"].iloc[0])

    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    return manifest_path


def test_recompute_matches_when_regeneration_is_identical(tmp_path: Path) -> None:
    recorded = _recorded_predictions()
    manifest_path = _recompute_manifest(tmp_path, recorded)

    def mock_generate(feature_table: pd.DataFrame, configuration: dict[str, Any]) -> pd.DataFrame:
        # Ignores its inputs on purpose: this is a mock of the fitted model,
        # not a re-implementation of score_outcome_week.
        return recorded.copy()

    report = replay_manifest(
        manifest_path,
        output_root=tmp_path / "out",
        recompute=True,
        generate_forecast=mock_generate,
    )

    assert report.recompute is not None
    assert report.recompute["attempted"] is True
    assert report.recompute["predictions_comparison"]["match"] is True
    assert report.recompute["metadata_comparison"]["match"] is True
    assert report.recompute["match"] is True
    assert report.ok is True
    # Never written outside output_root.
    assert (tmp_path / "out" / "regenerated_predictions.csv").is_file()
    assert not (tmp_path / "regenerated_predictions.csv").is_file()


def test_recompute_reports_injected_drift(tmp_path: Path) -> None:
    recorded = _recorded_predictions()
    manifest_path = _recompute_manifest(tmp_path, recorded)

    drifted = recorded.copy()
    drifted.loc[drifted["method"].eq("market_residual"), "model_probability"] += 0.07

    def mock_generate(feature_table: pd.DataFrame, configuration: dict[str, Any]) -> pd.DataFrame:
        return drifted.copy()

    report = replay_manifest(
        manifest_path,
        output_root=tmp_path / "out",
        recompute=True,
        generate_forecast=mock_generate,
    )

    assert report.recompute is not None
    predictions_comparison = report.recompute["predictions_comparison"]
    assert predictions_comparison["match"] is False
    column_report = predictions_comparison["columns"]["model_probability"]
    assert column_report["equal"] is False
    assert column_report["mismatched_rows"] == 2
    assert column_report["max_abs_diff"] == pytest.approx(0.07, abs=1e-6)
    assert report.recompute["match"] is False
    assert report.ok is False


def test_recompute_skipped_when_no_recompute_requested(tmp_path: Path) -> None:
    recorded = _recorded_predictions()
    manifest_path = _recompute_manifest(tmp_path, recorded)

    report = replay_manifest(manifest_path, output_root=tmp_path / "out", recompute=False)

    assert report.recompute is None
    assert not (tmp_path / "out").exists()
