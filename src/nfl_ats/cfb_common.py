from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.data import DataContractError


def snapshot_manifest_paths(raw_root: Path, label: str) -> list[Path]:

    manifests = sorted(raw_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No {label} snapshots found in {raw_root}")
    return manifests


def latest_manifest_path(raw_root: Path, label: str) -> Path:

    return snapshot_manifest_paths(raw_root, label)[-1]


def manifest_payload(manifest_path: Path) -> dict[str, Any]:

    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload


def require_manifest_payload(root: Path, missing_message: str) -> dict[str, Any]:

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(missing_message)
    return manifest_payload(manifest_path)


def season_partition_path(snapshot_root: Path, season: int, partition_filename: str) -> Path:

    return snapshot_root / f"season={season}" / partition_filename


def load_parquet_partitions(
    paths: Sequence[Path], columns: list[str] | None = None
) -> pd.DataFrame:

    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)


def fill_missing_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:

    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    return result.loc[:, list(columns)].copy()


def require_single_season(
    frame: pd.DataFrame, season: int, dataset: str, column: str = "season"
) -> None:

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

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors=errors).astype("int64")
    return frame


def cast_nullable_int_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    errors: Literal["raise", "coerce"] = "coerce",
) -> pd.DataFrame:

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors=errors).astype("Int64")
    return frame


def cast_float_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    errors: Literal["raise", "coerce"] = "coerce",
) -> pd.DataFrame:

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors=errors)
    return frame


def cast_string_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:

    for column in columns:
        frame[column] = frame[column].astype("string")
    return frame


def week_block_indices(frame: pd.DataFrame) -> list[npt.NDArray[np.intp]]:

    grouped = frame.groupby(["season", "week"], sort=False).indices
    return [np.asarray(positions, dtype=np.intp) for positions in grouped.values()]


def blocked_bootstrap_positions(
    blocks: Sequence[npt.NDArray[np.intp]], *, samples: int, seed: int
) -> Iterator[npt.NDArray[np.intp]]:

    generator = np.random.default_rng(seed)
    for _ in range(samples):
        chosen = generator.integers(0, len(blocks), size=len(blocks))
        yield np.concatenate([blocks[index] for index in chosen])
