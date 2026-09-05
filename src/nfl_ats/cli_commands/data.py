"""Snapshot ingest commands for every raw data source."""

from __future__ import annotations

import argparse
import json

from nfl_ats.cli_common import (
    _add_features_arg,
    _add_include_postseason_arg,
    _add_season_range_args,
    _data_root,
    _load_features,
    _print_json,
    _repo_root_on_path,
    _season_range,
)
from nfl_ats.data import check_nflverse_contract, fetch_nflverse
from nfl_ats.participation import fetch_participation_snapshot
from nfl_ats.pbp import fetch_pbp_snapshot
from nfl_ats.players import fetch_player_snapshot, fetch_player_value_snapshot
from nfl_ats.quarterbacks import fetch_depth_snapshot, fetch_historical_depth_snapshot
from nfl_ats.role_actions import fetch_role_actions_snapshot
from nfl_ats.snapshots import describe_snapshot


def _cmd_ingest_player_arrests(args: argparse.Namespace) -> None:
    """Refresh the production arrest snapshot through the audited ingester."""

    _repo_root_on_path()
    from scripts.ingest_player_arrests import (
        DEFAULT_DELAY_SECONDS,
        PlayerArrestsIngestError,
        ingest,
        new_snapshot_dir,
    )

    delay = args.delay_seconds if args.delay_seconds is not None else DEFAULT_DELAY_SECONDS
    if args.max_pages is not None and args.max_pages < 1:
        raise ValueError("--max-pages must be >= 1")
    if delay < 0:
        raise ValueError("--delay-seconds must be >= 0")
    snapshot_dir = new_snapshot_dir(_data_root() / "raw" / "player_arrests", args.snapshot)
    try:
        manifest = ingest(
            snapshot_dir,
            max_pages=args.max_pages,
            delay_seconds=delay,
        )
    except PlayerArrestsIngestError as error:
        raise ValueError(str(error)) from error
    _print_json(manifest)


def _cmd_ingest(args: argparse.Namespace) -> None:
    seasons = _season_range(args.start_season, args.end_season)
    stats_end_season = args.stats_end_season or args.end_season
    if stats_end_season < args.start_season or stats_end_season > args.end_season:
        raise ValueError("stats-end-season must be within the requested schedule seasons")
    team_stat_seasons = list(range(args.start_season, stats_end_season + 1))
    snapshot = fetch_nflverse(
        seasons,
        _data_root() / "raw",
        team_stat_seasons=team_stat_seasons,
    )
    _print_json(describe_snapshot(snapshot))


def _cmd_smoke_source(args: argparse.Namespace) -> None:
    _print_json(check_nflverse_contract(args.schedule_season, args.stats_season))


def _cmd_pbp_ingest(args: argparse.Namespace) -> None:
    snapshot = fetch_pbp_snapshot(
        _season_range(args.start_season, args.end_season),
        _data_root() / "pbp" / "raw",
        include_postseason=args.include_postseason,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "seasons": list(snapshot.seasons),
            "rows": manifest["rows"],
            "include_postseason": manifest["include_postseason"],
            "filter_version": manifest["filter_version"],
        }
    )


def _cmd_depth_ingest(args: argparse.Namespace) -> None:
    snapshot = fetch_depth_snapshot(
        _season_range(args.start_season, args.end_season),
        _data_root() / "quarterbacks" / "depth" / "raw",
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "rows": manifest["rows"],
            "teams": manifest["teams"],
            "first_observation": manifest["first_observation"],
            "last_observation": manifest["last_observation"],
            "contract_version": manifest["contract_version"],
        }
    )


def _cmd_depth_history_ingest(args: argparse.Namespace) -> None:
    snapshot = fetch_historical_depth_snapshot(
        _season_range(args.start_season, args.end_season),
        _load_features(args.features),
        _data_root() / "quarterbacks" / "depth" / "historical",
        games_source=args.features,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "rows": manifest["rows"],
            "teams": manifest["teams"],
            "first_effective": manifest["first_effective"],
            "last_effective": manifest["last_effective"],
            "contract_version": manifest["contract_version"],
            "coverage_by_season": manifest["coverage_by_season"],
        }
    )


def _cmd_player_ingest(args: argparse.Namespace) -> None:
    ranges = {
        "injury": (args.injury_start_season, args.injury_end_season),
        "roster": (args.roster_start_season, args.roster_end_season),
        "snap": (args.snap_start_season, args.snap_end_season),
    }
    for label, (start, end) in ranges.items():
        if end < start:
            raise ValueError(f"{label}-end-season cannot be earlier than {label}-start-season")
    snapshot = fetch_player_snapshot(
        list(range(args.injury_start_season, args.injury_end_season + 1)),
        list(range(args.roster_start_season, args.roster_end_season + 1)),
        list(range(args.snap_start_season, args.snap_end_season + 1)),
        _data_root() / "players" / "raw",
        include_postseason=args.include_postseason,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "contract_version": manifest["contract_version"],
            "include_postseason": manifest["include_postseason"],
            "files": manifest["files"],
            "availability_contract": manifest["availability_contract"],
        }
    )


def _cmd_player_value_ingest(args: argparse.Namespace) -> None:
    snapshot = fetch_player_value_snapshot(
        _season_range(args.start_season, args.end_season),
        _data_root() / "players" / "values" / "raw",
        include_postseason=args.include_postseason,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "contract_version": manifest["contract_version"],
            "include_postseason": manifest["include_postseason"],
            "file": manifest["file"],
            "availability_contract": manifest["availability_contract"],
        }
    )


def _cmd_role_actions_fetch(args: argparse.Namespace) -> None:
    snapshot = fetch_role_actions_snapshot(
        _data_root() / "players" / "role_actions" / "raw",
        args.seasons,
        include_postseason=args.include_postseason,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "seasons": manifest["seasons"],
            "include_postseason": manifest["include_postseason"],
            "rows": manifest["rows"],
            "sha256": manifest["sha256"],
            "source": manifest["source"],
        }
    )


def _cmd_participation_ingest(args: argparse.Namespace) -> None:
    snapshot = fetch_participation_snapshot(
        _season_range(args.start_season, args.end_season),
        _data_root() / "players" / "participation" / "raw",
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "contract_version": manifest["contract_version"],
            "seasons": manifest["seasons"],
            "rows": manifest["rows"],
            "partitions": manifest["partitions"],
            "availability_contract": manifest["availability_contract"],
        }
    )


def register_player_arrests(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the player-arrests ingest command."""

    arrests_ingest = subparsers.add_parser(
        "ingest-player-arrests",
        help="build a fresh, complete point-in-time player-arrests snapshot",
    )
    arrests_ingest.add_argument("--snapshot", type=str, default=None)
    arrests_ingest.add_argument("--max-pages", type=int, default=None)
    arrests_ingest.add_argument("--delay-seconds", type=float, default=None)
    arrests_ingest.set_defaults(handler=_cmd_ingest_player_arrests)


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the nflverse and role-action ingest commands."""

    ingest = subparsers.add_parser("ingest", help="download an immutable nflverse snapshot")
    _add_season_range_args(ingest, 2009, current_year)
    ingest.add_argument(
        "--stats-end-season",
        type=int,
        help="last team-stat season; use the prior year before current-season stats exist",
    )
    ingest.set_defaults(handler=_cmd_ingest)

    smoke = subparsers.add_parser(
        "smoke-source", help="check current nflverse source availability and schemas"
    )
    smoke.add_argument("--schedule-season", type=int, default=current_year)
    smoke.add_argument("--stats-season", type=int, default=current_year - 1)
    smoke.set_defaults(handler=_cmd_smoke_source)

    pbp_ingest = subparsers.add_parser(
        "pbp-ingest", help="download a versioned, season-partitioned nflverse PBP snapshot"
    )
    _add_season_range_args(pbp_ingest, 2009, current_year - 1)
    _add_include_postseason_arg(pbp_ingest)
    pbp_ingest.set_defaults(handler=_cmd_pbp_ingest)

    depth_ingest = subparsers.add_parser(
        "depth-ingest", help="archive timestamped nflverse quarterback depth charts"
    )
    _add_season_range_args(depth_ingest, current_year - 1, current_year - 1)
    depth_ingest.set_defaults(handler=_cmd_depth_ingest)

    depth_history_ingest = subparsers.add_parser(
        "depth-history-ingest",
        help="archive legacy weekly depth identities with conservative effective times",
    )
    _add_season_range_args(depth_history_ingest, 2009, 2024)
    _add_features_arg(depth_history_ingest)
    depth_history_ingest.set_defaults(handler=_cmd_depth_history_ingest)

    player_ingest = subparsers.add_parser(
        "player-ingest",
        help="archive injuries, earlier-week rosters, and lagged player snaps",
    )
    player_ingest.add_argument("--injury-start-season", type=int, default=2009)
    player_ingest.add_argument("--injury-end-season", type=int, default=2024)
    player_ingest.add_argument("--roster-start-season", type=int, default=2009)
    player_ingest.add_argument("--roster-end-season", type=int, default=current_year - 1)
    player_ingest.add_argument("--snap-start-season", type=int, default=2013)
    player_ingest.add_argument("--snap-end-season", type=int, default=current_year - 1)
    _add_include_postseason_arg(player_ingest)
    player_ingest.set_defaults(handler=_cmd_player_ingest)

    player_value_ingest = subparsers.add_parser(
        "player-value-ingest",
        help="archive weekly nflverse player production for lagged value estimates",
    )
    _add_season_range_args(player_value_ingest, 2009, current_year - 1)
    _add_include_postseason_arg(player_value_ingest)
    player_value_ingest.set_defaults(handler=_cmd_player_value_ingest)

    participation_ingest = subparsers.add_parser(
        "participation-ingest",
        help="archive season-partitioned nflverse player participation",
    )
    _add_season_range_args(participation_ingest, 2016, current_year - 1)
    participation_ingest.set_defaults(handler=_cmd_participation_ingest)

    role_actions_fetch = subparsers.add_parser(
        "role-actions-fetch",
        help="archive nflverse weekly player action counts for the XLG-04 replication",
    )
    role_actions_fetch.add_argument(
        "--seasons", type=int, nargs="+", default=list(range(2013, 2026))
    )
    _add_include_postseason_arg(role_actions_fetch)
    role_actions_fetch.set_defaults(handler=_cmd_role_actions_fetch)
