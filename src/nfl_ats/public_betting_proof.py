from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.provenance import sha256_file

_Parser = Callable[[str, pd.Timestamp], tuple[str, list[dict[str, Any]], str | None]]


def _path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def _source_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {_path_text(path): sha256_file(path) for path in paths}


def _issue(
    artifact: Path, reason: str, source_hashes: dict[str, str] | None = None
) -> dict[str, object]:
    return {
        "public_artifact": _path_text(artifact),
        "public_verification_reason": reason,
        "public_inspected_source_hashes": json.dumps(
            dict(sorted((source_hashes or {}).items())), separators=(",", ":")
        ),
    }


def load_verified_public_snapshots(
    data_root: Path, *, repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parser_path = repo_root / "scripts" / "ingest_public_betting.py"
    if not parser_path.is_file():
        raise DataContractError(f"Public-betting parser is unavailable: {parser_path}")
    namespace = runpy.run_path(str(parser_path))
    parser_value = namespace.get("parse_actionnetwork_snapshot")
    if not callable(parser_value):
        raise DataContractError("Public-betting parser has no snapshot parser")
    parser = cast(_Parser, parser_value)
    parser_hash = sha256_file(parser_path)
    frames: list[pd.DataFrame] = []
    issues: list[dict[str, object]] = []
    root = data_root / "raw" / "public_betting_live"
    for index_path in sorted(root.glob("*/index.parquet")):
        manifest_path = index_path.parent / "manifest.json"
        html_paths = sorted((index_path.parent / "raw_html").glob("*.html"))
        existing = tuple(
            path for path in (index_path, manifest_path, *html_paths, parser_path) if path.is_file()
        )
        before = _source_hashes(existing)
        if not manifest_path.is_file() or len(html_paths) != 1:
            issues.append(_issue(index_path, "raw_capture_files_missing", before))
            continue
        html_path = html_paths[0]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original = pd.read_parquet(index_path)
        except (OSError, ValueError, TypeError):
            issues.append(_issue(index_path, "raw_capture_invalid", before))
            continue
        if not isinstance(manifest, dict):
            issues.append(_issue(index_path, "raw_capture_manifest_invalid", before))
            continue
        declared_index_hash = manifest.get("index_sha256")
        declared_html_hash = manifest.get("raw_html_sha256")
        declared_hashes_valid = (
            declared_index_hash is None or declared_index_hash == before[_path_text(index_path)]
        ) and (declared_html_hash is None or declared_html_hash == before[_path_text(html_path)])
        if not declared_hashes_valid:
            issues.append(_issue(index_path, "raw_capture_declared_hash_mismatch", before))
            continue
        fetched_at = manifest.get("fetched_at")
        row_count = manifest.get("n_game_rows")
        public_row_count = manifest.get("n_game_rows_with_public_data")
        observation_time_basis = manifest.get("observation_time_basis")
        parsed_capture_ts = (
            pd.to_datetime(fetched_at, utc=True, errors="coerce")
            if isinstance(fetched_at, str)
            else pd.NaT
        )
        capture_ts = None if pd.isna(parsed_capture_ts) else pd.Timestamp(parsed_capture_ts)
        valid_manifest = (
            manifest.get("source") == "actionnetwork_live"
            and manifest.get("http_status") == 200
            and manifest.get("parse_error") is None
            and capture_ts is not None
            and isinstance(row_count, int)
            and not isinstance(row_count, bool)
            and row_count > 0
            and isinstance(public_row_count, int)
            and not isinstance(public_row_count, bool)
            and 0 <= public_row_count <= row_count
        )
        if not valid_manifest or capture_ts is None:
            issues.append(_issue(index_path, "raw_capture_manifest_invalid", before))
            continue
        if "capture_ts" not in original:
            issues.append(_issue(index_path, "raw_capture_time_missing", before))
            continue
        original_capture = pd.to_datetime(original["capture_ts"], utc=True, errors="coerce")
        if (
            original.empty
            or original_capture.isna().any()
            or not original_capture.eq(capture_ts).all()
        ):
            issues.append(_issue(index_path, "raw_capture_time_mismatch", before))
            continue
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        try:
            era, parsed_rows, parse_error = parser(html, capture_ts)
        except (KeyError, TypeError, ValueError):
            issues.append(_issue(index_path, "raw_identity_reparse_failed", before))
            continue
        repaired = pd.DataFrame(parsed_rows)
        required = {
            "capture_ts",
            "source",
            "era",
            "site_game_id",
            "site_home_team_id",
            "site_away_team_id",
            "team_side_basis",
            "season",
            "week",
            "away_team",
            "home_team",
            "start_time_utc",
            "spread_home_bet_pct",
            "spread_away_bet_pct",
            "has_any_public_data",
        }
        if (
            parse_error is not None
            or repaired.empty
            or not required.issubset(repaired.columns)
            or era != manifest.get("era")
            or len(repaired) != row_count
            or int(repaired["has_any_public_data"].fillna(False).astype(bool).sum())
            != public_row_count
        ):
            issues.append(_issue(index_path, "raw_identity_reparse_invalid", before))
            continue
        valid_identity = repaired["team_side_basis"].eq("game_team_ids")
        valid_identity &= (
            repaired[["site_game_id", "site_home_team_id", "site_away_team_id"]].notna().all(axis=1)
        )
        valid_identity &= (
            repaired["site_home_team_id"].astype(str).ne(repaired["site_away_team_id"].astype(str))
        )
        valid_identity &= ~repaired["site_game_id"].astype(str).duplicated()
        if not valid_identity.all():
            issues.append(_issue(index_path, "raw_identity_mapping_invalid", before))
            continue
        if "site_game_id" not in original or set(original["site_game_id"].astype(str)) != set(
            repaired["site_game_id"].astype(str)
        ):
            issues.append(_issue(index_path, "raw_game_identity_set_mismatch", before))
            continue
        percentages = [
            column
            for column in (
                "spread_home_bet_pct",
                "spread_away_bet_pct",
                "spread_home_money_pct",
                "spread_away_money_pct",
                "ml_home_bet_pct",
                "ml_away_bet_pct",
                "ml_home_money_pct",
                "ml_away_money_pct",
                "total_over_bet_pct",
                "total_under_bet_pct",
                "total_over_money_pct",
                "total_under_money_pct",
            )
            if column in original and column in repaired
        ]
        original_values = original[["site_game_id", *percentages]].copy()
        repaired_values = repaired[["site_game_id", *percentages]].copy()
        compared = original_values.merge(
            repaired_values,
            on="site_game_id",
            suffixes=("_saved", "_reparsed"),
            validate="one_to_one",
        )
        numeric_unchanged = all(
            np.isclose(
                pd.to_numeric(compared[f"{column}_saved"], errors="coerce"),
                pd.to_numeric(compared[f"{column}_reparsed"], errors="coerce"),
                rtol=0,
                atol=1e-10,
                equal_nan=True,
            ).all()
            for column in percentages
        )
        if not numeric_unchanged:
            issues.append(_issue(index_path, "raw_public_values_changed_on_reparse", before))
            continue
        after = _source_hashes(existing)
        if before != after or before.get(_path_text(parser_path)) != parser_hash:
            raise DataContractError(
                f"Public-betting source changed during read: {index_path.parent}"
            )
        repaired["capture_ts"] = pd.to_datetime(repaired["capture_ts"], utc=True, errors="coerce")
        repaired["start_time_utc"] = pd.to_datetime(
            repaired["start_time_utc"], utc=True, errors="coerce"
        )
        repaired["public_artifact"] = _path_text(index_path)
        repaired["public_artifact_sha256"] = before[_path_text(index_path)]
        repaired["public_manifest_artifact"] = _path_text(manifest_path)
        repaired["public_manifest_sha256"] = before[_path_text(manifest_path)]
        repaired["public_raw_html_artifact"] = _path_text(html_path)
        repaired["public_raw_html_sha256"] = before[_path_text(html_path)]
        repaired["public_parser_artifact"] = _path_text(parser_path)
        repaired["public_parser_sha256"] = parser_hash
        repaired["public_identity_reparsed"] = True
        repaired["public_verification_reason"] = "verified_raw_game_team_ids"
        repaired["public_observation_time_basis"] = (
            "response_received"
            if observation_time_basis == "response_received"
            else "request_started_legacy"
        )
        repaired["public_chronology_verified"] = observation_time_basis == "response_received"
        repaired["public_inspected_source_hashes"] = json.dumps(
            dict(sorted(before.items())), separators=(",", ":")
        )
        frames.append(repaired)
    verified = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return verified, pd.DataFrame(issues)


def verify_public_snapshot_sources(frame: pd.DataFrame) -> None:
    if "public_inspected_source_hashes" not in frame:
        raise DataContractError("Verified public snapshots have no source hashes")
    expected: dict[Path, str] = {}
    for payload in frame["public_inspected_source_hashes"].dropna().unique():
        try:
            hashes = json.loads(str(payload))
        except (TypeError, ValueError) as error:
            raise DataContractError("Verified public snapshot hashes are invalid") from error
        if not isinstance(hashes, dict):
            raise DataContractError("Verified public snapshot hashes are invalid")
        for path_text, digest in hashes.items():
            path = Path(str(path_text))
            previous = expected.get(path)
            if previous is not None and previous != str(digest):
                raise DataContractError(f"Conflicting verified public hashes: {path}")
            expected[path] = str(digest)
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise DataContractError(f"Public snapshot source changed during measurement: {path}")


__all__ = ["load_verified_public_snapshots", "verify_public_snapshot_sources"]
