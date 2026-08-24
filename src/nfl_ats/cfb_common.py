"""Shared mechanical plumbing for the CFB cluster.

This module holds only the helpers that ``nfl_ats.cfb``,
``nfl_ats.cfb_features``, and ``nfl_ats.cross_league_transfer`` had each
re-implemented inline. It is a behavior-preserving consolidation: every
function here is moved verbatim from one of those modules, and no market-
aggregation logic (spread orientation, book aggregation, side repair)
lives here -- that code is contract-bearing and stays in
``cfb_features.py`` untouched.

Three families:

- **Snapshot loading**: manifest discovery under a source's ``raw/``
  directory, manifest parsing, season-partition pathing, and partition
  parquet loading.
- **Column normalization**: the cast loops every canonicalizer repeats
  (int64 / nullable Int64 / float / pandas ``string``), plus the
  fill-missing-columns and single-season checks.
- **Bootstrap plumbing**: week-block index construction and the blocked
  resample-position generator shared by the transfer shrinkage bootstrap.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.data import DataContractError

# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------


def snapshot_manifest_paths(raw_root: Path, label: str) -> list[Path]:
    """Every snapshot manifest under one source's ``raw`` directory, sorted."""

    manifests = sorted(raw_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No {label} snapshots found in {raw_root}")
    return manifests


def latest_manifest_path(raw_root: Path, label: str) -> Path:
    """The newest snapshot manifest under one source's ``raw`` directory."""

    return snapshot_manifest_paths(raw_root, label)[-1]


def manifest_payload(manifest_path: Path) -> dict[str, Any]:
    """Parse one snapshot manifest.json."""

    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload


def require_manifest_payload(root: Path, missing_message: str) -> dict[str, Any]:
    """Parse ``root/manifest.json``, failing with ``missing_message`` if absent."""

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(missing_message)
    return manifest_payload(manifest_path)


def season_partition_path(snapshot_root: Path, season: int, partition_filename: str) -> Path:
    """One season's canonical partition inside a snapshot directory."""

    return snapshot_root / f"season={season}" / partition_filename


def load_parquet_partitions(
    paths: Sequence[Path], columns: list[str] | None = None
) -> pd.DataFrame:
    """Read season partitions in the given order and stack them row-wise."""

    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------


def fill_missing_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Add any missing column as all-NA, then project exactly onto ``columns``."""

    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    return result.loc[:, list(columns)].copy()


def require_single_season(
    frame: pd.DataFrame, season: int, dataset: str, column: str = "season"
) -> None:
    """Fail unless a season partition holds exactly the requested season."""

    if frame.empty:
        raise DataContractError(f"{dataset} season {season} contains no rows")
    observed = set(pd.to_numeric(frame[column], errors="coerce").dropna().astype(int))
    if observed != {season}:
        raise DataContractError(
            f"{dataset} season partition {season} contains seasons {sorted(observed)}"
        )


def cast_int_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    errors: Literal["raise", "coerce"] = "raise",
) -> pd.DataFrame:
    """Coerce columns to numeric and store as plain int64, in place."""

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors=errors).astype("int64")
    return frame


def cast_nullable_int_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    errors: Literal["raise", "coerce"] = "coerce",
) -> pd.DataFrame:
    """Coerce columns to numeric and store as nullable Int64, in place."""

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors=errors).astype("Int64")
    return frame


def cast_float_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    errors: Literal["raise", "coerce"] = "coerce",
) -> pd.DataFrame:
    """Coerce columns to numeric floats, in place."""

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors=errors)
    return frame


def cast_string_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Store columns as pandas string dtype, in place."""

    for column in columns:
        frame[column] = frame[column].astype("string")
    return frame


# ---------------------------------------------------------------------------
# Bootstrap plumbing
# ---------------------------------------------------------------------------


def week_block_indices(frame: pd.DataFrame) -> list[npt.NDArray[np.intp]]:
    """Row positions of each (season, week) block, preserving frame order."""

    grouped = frame.groupby(["season", "week"], sort=False).indices
    return [np.asarray(positions, dtype=np.intp) for positions in grouped.values()]


def blocked_bootstrap_positions(
    blocks: Sequence[npt.NDArray[np.intp]], *, samples: int, seed: int
) -> Iterator[npt.NDArray[np.intp]]:
    """Yield ``samples`` block-resampled row-position arrays.

    Whole blocks are drawn with replacement until each resample matches the
    original block count, preserving whatever within-block dependence the
    caller's blocks encode. Draws come from one ``default_rng(seed)`` in
    sample order, so identical inputs yield identical resamples.
    """

    generator = np.random.default_rng(seed)
    for _ in range(samples):
        chosen = generator.integers(0, len(blocks), size=len(blocks))
        yield np.concatenate([blocks[index] for index in chosen])
