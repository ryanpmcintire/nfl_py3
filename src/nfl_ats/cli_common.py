"""Shared CLI helpers used by more than one command module.

These are the pieces ``nfl_ats.cli`` and every ``nfl_ats.cli_commands`` module
need in common: repository roots, JSON printing, feature-table loading, and the
reusable ``add_argument`` groups. They live here rather than in ``cli`` so the
command modules can import them without importing the parser that imports
them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.constants import DEFAULT_OFFSEASON_RETENTION
from nfl_ats.margin import MARGIN_FEATURE_PROFILES
from nfl_ats.pbp import PbpSnapshot, latest_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.players import (
    PlayerSnapshot,
    PlayerValueSnapshot,
    latest_player_snapshot,
    latest_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
)
from nfl_ats.snapshots import Snapshot, latest_snapshot, snapshot_from_root


def _data_root() -> Path:
    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


def _artifacts_root() -> Path:
    return Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts"))


def _registry_root() -> Path:
    return Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry"))


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_features(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run `nfl-ats build-features` first."
        )
    return pd.read_parquet(path)


def _season_range(start_season: int, end_season: int) -> list[int]:
    if end_season < start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    return list(range(start_season, end_season + 1))


def _repo_root_on_path() -> None:
    """Make ``scripts.*`` importable however this process was launched.

    ``scripts`` is not part of the installed package, so it resolves only when
    the repository root happens to be on ``sys.path``. ``python -m nfl_ats``
    puts the working directory there and the console script does NOT, so
    ``nfl-ats ingest-player-arrests`` raised ``ModuleNotFoundError: No module
    named 'scripts'`` while ``python -m nfl_ats ingest-player-arrests``
    succeeded from the same directory.

    That is a lock-day abort, not a cosmetic difference.
    ``nfl_ats.weekly._cli_runner`` dispatches every step IN-PROCESS, so
    ``weekly-run`` step 7 (``ingest-player-arrests``, fail-closed) inherits
    whatever ``sys.path`` launched it -- and the documented Tuesday command in
    ``docs/week1_readiness.md`` is the console script. Left alone, the real
    2026-09-08 run would have aborted before publishing anything.

    Resolved from this file's own location rather than the working directory,
    so it holds no matter where the command is invoked from.
    """

    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_INCLUDE_POSTSEASON_HELP = (
    "also store postseason rows (WC/DIV/CON/SB, spelled POST in the play-by-play "
    "and player-stat feeds). Off by default: every feature build re-filters to the "
    "regular season on read, so this only widens what a future snapshot can serve. "
    "Pass it when building snapshots intended for playoff-game predictions."
)


def _resolve_snapshot(identifier: str | None) -> Snapshot:
    raw_root = _data_root() / "raw"
    return snapshot_from_root(raw_root / identifier) if identifier else latest_snapshot(raw_root)


def _resolve_pbp_snapshot(identifier: str | None) -> PbpSnapshot:
    root = _data_root() / "pbp" / "raw"
    return pbp_snapshot_from_root(root / identifier) if identifier else latest_pbp_snapshot(root)


def _resolve_player_snapshot(identifier: str | None) -> PlayerSnapshot:
    root = _data_root() / "players" / "raw"
    return (
        player_snapshot_from_root(root / identifier) if identifier else latest_player_snapshot(root)
    )


def _resolve_player_value_snapshot(identifier: str | None) -> PlayerValueSnapshot:
    root = _data_root() / "players" / "values" / "raw"
    return (
        player_value_snapshot_from_root(root / identifier)
        if identifier
        else latest_player_value_snapshot(root)
    )


def _add_features_arg(
    parser: argparse.ArgumentParser,
    filename: str = "game_features.parquet",
    *,
    help_text: str | None = None,
) -> None:
    """Register the shared --features feature-table flag under data/processed."""
    parser.add_argument(
        "--features",
        type=Path,
        default=_data_root() / "processed" / filename,
        help=help_text,
    )


def _add_bootstrap_args(
    parser: argparse.ArgumentParser,
    samples: int = 2_000,
    seed: int = 20260812,
) -> None:
    """Register the shared bootstrap-uncertainty pair."""
    parser.add_argument("--bootstrap-samples", type=int, default=samples)
    parser.add_argument("--bootstrap-seed", type=int, default=seed)


def _add_season_range_args(
    parser: argparse.ArgumentParser,
    start_default: int | None,
    end_default: int | None,
) -> None:
    """Register the shared --start-season/--end-season pair."""
    parser.add_argument("--start-season", type=int, default=start_default)
    parser.add_argument("--end-season", type=int, default=end_default)


def _add_season_week_args(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    """Register the shared --season/--week pair (required, or prospective defaults)."""
    if required:
        parser.add_argument("--season", type=int, required=True)
        parser.add_argument("--week", type=int, required=True)
    else:
        parser.add_argument("--season", type=int, default=2026)
        parser.add_argument("--week", type=int, default=1)


def _add_snapshot_args(parser: argparse.ArgumentParser, *specs: tuple[str, str]) -> None:
    """Register "(label) snapshot ID; defaults to latest" flags as (flag, label) pairs."""
    for flag, label in specs:
        head = f"{label} snapshot ID" if label else "snapshot ID"
        parser.add_argument(flag, help=f"{head}; defaults to latest")


def _add_include_postseason_arg(parser: argparse.ArgumentParser) -> None:
    """Register the shared --include-postseason flag."""
    parser.add_argument(
        "--include-postseason",
        action="store_true",
        help=_INCLUDE_POSTSEASON_HELP,
    )


def _add_ewm_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared EWM smoothing trio."""
    parser.add_argument("--ewm-span", type=int, default=8)
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--offseason-retention", type=float, default=DEFAULT_OFFSEASON_RETENTION)


def _add_regressor_args(parser: argparse.ArgumentParser, *, choices: bool = True) -> None:
    """Register the shared --regressor/--ridge-alpha pair."""
    if choices:
        parser.add_argument("--regressor", choices=("ridge", "hgb"), default="ridge")
    else:
        parser.add_argument("--regressor", default="ridge")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)


def _add_feature_profile_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    help_text: str | None = None,
) -> None:
    """Register the shared --feature-profile choice over MARGIN_FEATURE_PROFILES."""
    parser.add_argument(
        "--feature-profile",
        choices=MARGIN_FEATURE_PROFILES,
        default=default,
        help=help_text,
    )


def _add_board_destination_args(
    parser: argparse.ArgumentParser,
    *,
    legacy_flag: str,
) -> None:
    """Register the duplicated board/site destination pair for the publish commands."""
    parser.add_argument(
        legacy_flag,
        type=Path,
        default=Path("docs/index.html"),
        help="deprecated alias for --site-destination; a file path is reduced to its directory",
    )
    parser.add_argument(
        "--site-destination",
        type=Path,
        default=None,
        help="directory to write the three public pages into (default: docs/)",
    )


def _add_player_feature_tuning_args(parser: argparse.ArgumentParser) -> None:
    """Register the seven tuning flags shared by the three player-feature builders."""
    parser.add_argument("--decision-hours", type=int, default=24)
    parser.add_argument("--role-span", type=int, default=8)
    parser.add_argument("--qb-span", type=int, default=12)
    parser.add_argument("--qb-min-dropbacks", type=int, default=20)
    parser.add_argument("--offseason-retention", type=float, default=0.75)
    parser.add_argument("--value-span", type=int, default=16)
    parser.add_argument("--value-prior-snaps", type=float, default=200.0)
