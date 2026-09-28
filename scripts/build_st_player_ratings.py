from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.io import atomic_bytes, atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.participation import (  # noqa: E402
    PARTICIPATION_RATING_EPA_CLIP,
    PARTICIPATION_RATING_LOOKBACK_SEASONS,
    PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
    PARTICIPATION_RATING_RIDGE_ALPHA,
    PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
)
from nfl_ats.pbp import season_scope_mask  # noqa: E402
from nfl_ats.provenance import sha256_file  # noqa: E402
from nfl_ats.special_teams import (  # noqa: E402
    SPECIAL_TEAMS_RATING_COLUMNS,
    SPECIAL_TEAMS_RATING_VERSION,
    SPECIAL_TEAMS_SOURCE_SEASONS,
    SPECIAL_TEAMS_TARGET_SEASONS,
    build_season_lagged_special_teams_ratings,
)

DEFAULT_PARTICIPATION_SNAPSHOT = (
    REPO / "data" / "players" / "participation" / "raw" / "20260813T131635Z"
)
DEFAULT_PBP_SNAPSHOT = REPO / "data" / "pbp" / "raw" / "20260817T184927Z"
DEFAULT_DECLARATION = REPO / "docs" / "st_player_ratings.md"
PBP_SOURCE_COLUMNS = ("game_id", "play_id", "season", "season_type", "posteam", "epa")


def _required_source_seasons(target_seasons: list[int]) -> tuple[int, ...]:
    seasons = {
        season
        for target in target_seasons
        for season in range(target - PARTICIPATION_RATING_LOOKBACK_SEASONS, target)
    }
    undeclared = seasons.difference(SPECIAL_TEAMS_SOURCE_SEASONS)
    if undeclared:
        raise DataContractError(
            f"Target seasons require undeclared source seasons: {sorted(undeclared)}"
        )
    return tuple(sorted(seasons))


def _load_source_partitions(
    snapshot_root: Path,
    required_seasons: tuple[int, ...],
    filename: str,
    source_name: str,
    columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], bytes]:
    manifest_path = snapshot_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {source_name} manifest: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if str(manifest.get("snapshot_id")) != snapshot_root.name:
        raise DataContractError(f"{source_name} snapshot ID does not match its directory name")
    manifest_seasons = [int(season) for season in manifest.get("seasons", [])]
    partitions = manifest.get("partitions", [])
    if columns is not None:
        declared_columns = manifest.get("columns")
        if not isinstance(declared_columns, list):
            raise DataContractError(f"{source_name} manifest is missing column metadata")
        missing_columns = set(columns).difference(map(str, declared_columns))
        if missing_columns:
            raise DataContractError(
                f"{source_name} manifest is missing required columns: {sorted(missing_columns)}"
            )
    if len(manifest_seasons) != len(set(manifest_seasons)):
        raise DataContractError(f"{source_name} manifest contains duplicate seasons")
    by_season: dict[int, dict[str, Any]] = {}
    for partition in partitions:
        season = int(partition["season"])
        if season in by_season:
            raise DataContractError(
                f"{source_name} manifest contains duplicate partition season {season}"
            )
        by_season[season] = partition
    if set(manifest_seasons) != set(by_season):
        raise DataContractError(
            f"{source_name} manifest season metadata does not match its partitions"
        )
    missing = set(required_seasons).difference(by_season)
    if missing:
        raise DataContractError(
            f"{source_name} snapshot is missing required seasons: {sorted(missing)}"
        )
    frames: list[pd.DataFrame] = []
    used_partitions: list[dict[str, Any]] = []
    for season in required_seasons:
        partition = by_season[season]
        expected_relative = f"season={season}/{filename}"
        declared_relative = str(partition.get("path", "")).replace("\\", "/")
        if declared_relative != expected_relative:
            raise DataContractError(
                f"{source_name} season {season} has unexpected path {declared_relative!r}"
            )
        path = snapshot_root / f"season={season}" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {source_name} partition: {path}")
        actual_sha256 = sha256_file(path)
        declared_sha256 = str(partition.get("sha256", ""))
        if actual_sha256 != declared_sha256:
            raise DataContractError(
                f"{source_name} season {season} partition hash does not match its manifest"
            )
        frame = (
            pd.read_parquet(path, columns=list(columns))
            if columns is not None
            else pd.read_parquet(path)
        )
        if sha256_file(path) != actual_sha256:
            raise DataContractError(
                f"{source_name} season {season} partition changed while it was read"
            )
        observed_seasons = set(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int))
        if observed_seasons != {season}:
            raise DataContractError(
                f"{source_name} season {season} partition contains seasons "
                f"{sorted(observed_seasons)}"
            )
        frames.append(frame)
        used_partitions.append(
            {
                "path": expected_relative,
                "rows": len(frame),
                "season": season,
                "sha256": actual_sha256,
            }
        )
    return pd.concat(frames, ignore_index=True), manifest, used_partitions, manifest_bytes


def _target_summary(ratings: pd.DataFrame) -> list[dict[str, int]]:
    grouped = (
        ratings.groupby(
            ["target_season", "source_start_season", "source_end_season", "source_plays"],
            sort=True,
        )
        .size()
        .rename("rated_players")
        .reset_index()
    )
    return [
        {str(key): int(value) for key, value in row.items()}
        for row in grouped.to_dict(orient="records")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen, unserved season-lagged special-teams player ratings."
    )
    parser.add_argument(
        "--participation-snapshot", type=Path, default=DEFAULT_PARTICIPATION_SNAPSHOT
    )
    parser.add_argument("--pbp-snapshot", type=Path, default=DEFAULT_PBP_SNAPSHOT)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument(
        "--target-seasons",
        type=int,
        nargs="+",
        default=list(SPECIAL_TEAMS_TARGET_SEASONS),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = sorted({int(season) for season in args.target_seasons})
    if tuple(targets) != SPECIAL_TEAMS_TARGET_SEASONS:
        raise DataContractError(
            f"Target seasons must equal the declared {list(SPECIAL_TEAMS_TARGET_SEASONS)}"
        )
    required_seasons = _required_source_seasons(targets)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not args.declaration.is_file():
        raise FileNotFoundError(f"Missing declaration: {args.declaration}")
    declaration_bytes = args.declaration.read_bytes()
    implementation_sha256 = {
        "scripts/build_st_player_ratings.py": sha256_file(
            REPO / "scripts" / "build_st_player_ratings.py"
        ),
        "src/nfl_ats/participation.py": sha256_file(REPO / "src" / "nfl_ats" / "participation.py"),
        "src/nfl_ats/special_teams.py": sha256_file(REPO / "src" / "nfl_ats" / "special_teams.py"),
    }

    (
        participation,
        participation_manifest,
        participation_partitions,
        participation_manifest_bytes,
    ) = _load_source_partitions(
        args.participation_snapshot,
        required_seasons,
        "participation.parquet",
        "participation",
    )
    pbp, pbp_manifest, pbp_partitions, pbp_manifest_bytes = _load_source_partitions(
        args.pbp_snapshot,
        required_seasons,
        "plays.parquet",
        "play-by-play",
        columns=PBP_SOURCE_COLUMNS,
    )
    keep = season_scope_mask(
        pbp["season_type"],
        include_postseason=False,
        dataset="special-teams play-by-play source",
        column="season_type",
    )
    pbp = pbp.loc[keep].reset_index(drop=True)

    ratings = build_season_lagged_special_teams_ratings(
        participation,
        pbp,
        target_seasons=targets,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    ratings_path = args.output / "ratings.parquet"
    declaration_path = args.output / "declaration.md"
    participation_manifest_path = args.output / "source_participation_manifest.json"
    pbp_manifest_path = args.output / "source_pbp_manifest.json"
    atomic_parquet(ratings, ratings_path)
    atomic_bytes(declaration_bytes, declaration_path)
    atomic_bytes(participation_manifest_bytes, participation_manifest_path)
    atomic_bytes(pbp_manifest_bytes, pbp_manifest_path)

    manifest = {
        "artifact_version": SPECIAL_TEAMS_RATING_VERSION,
        "availability": "each target uses exactly its prior three complete seasons",
        "declaration": {
            "path": declaration_path.name,
            "sha256": sha256_file(declaration_path),
        },
        "implementation_sha256": implementation_sha256,
        "fit_configuration": {
            "epa_clip": PARTICIPATION_RATING_EPA_CLIP,
            "lookback_seasons": PARTICIPATION_RATING_LOOKBACK_SEASONS,
            "personnel_population": "either personnel string contains 1 K, 1 P, or 1 LS",
            "player_effect": "one pooled special-teams coefficient",
            "reliability_prior_plays": PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
            "ridge_alpha": PARTICIPATION_RATING_RIDGE_ALPHA,
            "season_scope": "regular season only; postseason excluded",
            "solver": "lsqr",
            "team_feature": "posteam at positive scale",
            "team_feature_scale": PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
            "valid_ids": "11 unique offense_players and 11 unique defense_players",
        },
        "look_accounting": {
            "ats_outcome_comparisons": 0,
            "builder_family": "st_player_rating_season_lagged_builder",
            "fitted_looks": len(targets),
            "promotion_decisions": 0,
            "target_seasons": targets,
        },
        "output": {
            "columns": list(SPECIAL_TEAMS_RATING_COLUMNS),
            "path": ratings_path.name,
            "rows": len(ratings),
            "sha256": sha256_file(ratings_path),
        },
        "source_artifacts": {
            "participation": {
                "manifest_path": participation_manifest_path.name,
                "manifest_sha256": sha256_file(participation_manifest_path),
                "partitions": participation_partitions,
                "snapshot_id": str(participation_manifest["snapshot_id"]),
            },
            "play_by_play": {
                "manifest_path": pbp_manifest_path.name,
                "manifest_sha256": sha256_file(pbp_manifest_path),
                "partitions": pbp_partitions,
                "snapshot_id": str(pbp_manifest["snapshot_id"]),
            },
        },
        "source_seasons": list(required_seasons),
        "target_seasons": _target_summary(ratings),
        "usage": "offline and unserved; no ATS grading or promotion",
    }
    atomic_json(manifest, args.output / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
