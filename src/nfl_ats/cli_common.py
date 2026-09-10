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
    injury_first_seen_index,
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
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _load_features(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run `nfl-ats build-features` first."
        )
    return pd.read_parquet(path)


def _injury_first_seen_index(seasons: list[int] | None = None) -> pd.DataFrame:

    root = _data_root()
    return injury_first_seen_index(
        [root / "raw" / "nflverse_injuries", root / "players" / "raw"], seasons=seasons
    )


def _season_range(start_season: int, end_season: int) -> list[int]:
    if end_season < start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    return list(range(start_season, end_season + 1))


def _repo_root_on_path() -> None:

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
    parser.add_argument("--bootstrap-samples", type=int, default=samples)
    parser.add_argument("--bootstrap-seed", type=int, default=seed)


def _add_season_range_args(
    parser: argparse.ArgumentParser,
    start_default: int | None,
    end_default: int | None,
) -> None:
    parser.add_argument("--start-season", type=int, default=start_default)
    parser.add_argument("--end-season", type=int, default=end_default)


def _add_season_week_args(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    if required:
        parser.add_argument("--season", type=int, required=True)
        parser.add_argument("--week", type=int, required=True)
    else:
        parser.add_argument("--season", type=int, default=2026)
        parser.add_argument("--week", type=int, default=1)


def _add_active_forecast_season_week_args(parser: argparse.ArgumentParser) -> None:

    help_suffix = (
        "; defaults to the active model's linked weekly forecast "
        "(artifacts/active_ats_model.json), which is the week publish-predictions "
        "locked -- pass both --season and --week or neither"
    )
    parser.add_argument("--season", type=int, default=None, help="season" + help_suffix)
    parser.add_argument("--week", type=int, default=None, help="week" + help_suffix)


def _resolve_active_forecast_season_week(
    args: argparse.Namespace, artifacts_root: Path
) -> tuple[int, int]:

    from nfl_ats.active_model import active_forecast_season_week

    season = getattr(args, "season", None)
    week = getattr(args, "week", None)
    if season is not None and week is not None:
        return int(season), int(week)
    if (season is None) != (week is None):
        raise ValueError("pass both --season and --week, or neither")
    resolved = active_forecast_season_week(artifacts_root)
    if resolved is None:
        raise ValueError(
            "no --season/--week given and the active model manifest "
            f"({artifacts_root / 'active_ats_model.json'}) has no synchronized linked "
            "weekly forecast to default to; run publish-predictions first or pass both flags"
        )
    return resolved


def _add_snapshot_args(parser: argparse.ArgumentParser, *specs: tuple[str, str]) -> None:
    for flag, label in specs:
        head = f"{label} snapshot ID" if label else "snapshot ID"
        parser.add_argument(flag, help=f"{head}; defaults to latest")


def _add_include_postseason_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-postseason",
        action="store_true",
        help=_INCLUDE_POSTSEASON_HELP,
    )


def _add_ewm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ewm-span", type=int, default=8)
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--offseason-retention", type=float, default=DEFAULT_OFFSEASON_RETENTION)


def _add_regressor_args(parser: argparse.ArgumentParser, *, choices: bool = True) -> None:
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
    parser.add_argument("--decision-hours", type=int, default=24)
    parser.add_argument("--role-span", type=int, default=8)
    parser.add_argument("--qb-span", type=int, default=12)
    parser.add_argument("--qb-min-dropbacks", type=int, default=20)
    parser.add_argument("--offseason-retention", type=float, default=0.75)
    parser.add_argument("--value-span", type=int, default=16)
    parser.add_argument("--value-prior-snaps", type=float, default=200.0)
