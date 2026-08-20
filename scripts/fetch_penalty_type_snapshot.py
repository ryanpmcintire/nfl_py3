"""Widen the project's referee-battery penalty data from totals to penalty-TYPE rates.

2026-08-20 session, `docs/data_source_scout_v4.md` lead #1 ("Penalty-type crew
tendencies"): the repo's existing referee battery
(`data/raw/officials/*/game_penalties.parquet`, built 2026-08-19) already
aggregates per-game penalty COUNTS (`penalties_total`/`penalties_on_home`/
`penalties_on_away`), but never persisted `penalty_type` -- confirming the
scout doc's finding that this is a re-pull, not a new source: nflverse PBP's
own `load_pbp()` already carries `penalty_type`/`penalty_team`/
`penalty_player_id`/`penalty_player_name` (MEASURED this session,
`nflreadpy.load_pbp(seasons=[2023])` columns), the project's local snapshots
(`nfl_ats.pbp.PBP_SNAPSHOT_COLUMNS`, `data/pbp/team_style/raw_pbp_narrow.parquet`)
just never retained them.

This script re-fetches nflverse PBP one season at a time (bounding memory,
matching `nfl_ats.pbp.fetch_pbp_snapshot`'s own pattern) for the SAME season
window the existing officials/game_penalties snapshot covers (2015-2025 --
`nflreadr.load_officials()`'s own documented floor), and persists ONLY a
small derived long-format aggregate -- one row per (game_id, penalty_type)
with `penalties_total`/`penalties_on_home`/`penalties_on_away` -- alongside a
NEW timestamped snapshot directory under `data/raw/officials/`. The raw PBP
itself is not persisted, mirroring the existing `game_penalties.parquet`
convention exactly (see its own manifest: "raw PBP itself is NOT persisted
here").

Home/away attribution matches the existing `game_penalties.parquet` exactly:
`penalty_team == home_team` -> `penalties_on_home`, `penalty_team ==
away_team` -> `penalties_on_away` (MEASURED-verified against
`game_penalties.parquet`'s own recorded 2015_01_BAL_DEN row: 8 home / 3 away
-- this script reproduces that split bit-for-bit as part of its own
alignment check, see `verify_against_existing_totals`).

Nothing about the EXISTING `officials.parquet`/`game_penalties.parquet`
snapshot is mutated -- this writes a NEW, separate snapshot directory only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_ats.io import atomic_json, atomic_parquet, run_id

DEFAULT_SEASONS = tuple(range(2015, 2026))


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        converted = frame.to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted
    raise TypeError(f"Unsupported dataframe type: {type(frame)!r}")


def fetch_season_penalty_types(season: int) -> pd.DataFrame:
    """One row per (game_id, penalty_type) for a single season's full PBP."""

    import nflreadpy as nfl

    pbp = _to_pandas(nfl.load_pbp(seasons=[season]))
    required = {
        "game_id",
        "season",
        "week",
        "season_type",
        "home_team",
        "away_team",
        "penalty",
        "penalty_team",
        "penalty_type",
        "penalty_yards",
    }
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"season {season}: nflverse PBP is missing columns {missing}")

    pbp["penalty"] = pd.to_numeric(pbp["penalty"], errors="coerce").fillna(0.0)
    flagged = pbp.loc[pbp["penalty"] == 1].copy()
    if flagged["penalty_type"].isna().any():
        # MEASURED 2015/2023: every penalty==1 row carries a non-null
        # penalty_type. Guard rather than silently drop if a future season
        # ever violates that.
        flagged["penalty_type"] = flagged["penalty_type"].fillna("Unknown")
    flagged["penalty_yards"] = pd.to_numeric(flagged["penalty_yards"], errors="coerce").fillna(0.0)
    flagged["_on_home"] = (flagged["penalty_team"] == flagged["home_team"]).astype(int)
    flagged["_on_away"] = (flagged["penalty_team"] == flagged["away_team"]).astype(int)

    grouped = (
        flagged.groupby(["game_id", "season", "week", "season_type", "penalty_type"], sort=False)
        .agg(
            penalties_total=("penalty", "size"),
            penalties_on_home=("_on_home", "sum"),
            penalties_on_away=("_on_away", "sum"),
            penalty_yards_total=("penalty_yards", "sum"),
        )
        .reset_index()
    )
    return grouped


def verify_against_existing_totals(
    new_long: pd.DataFrame, existing_game_penalties: pd.DataFrame
) -> dict[str, Any]:
    """Row-alignment check: per-game_id summed type counts vs. the existing snapshot."""

    resummed = (
        new_long.groupby("game_id")
        .agg(
            penalties_total=("penalties_total", "sum"),
            penalties_on_home=("penalties_on_home", "sum"),
            penalties_on_away=("penalties_on_away", "sum"),
        )
        .reset_index()
    )
    merged = resummed.merge(
        existing_game_penalties[
            ["game_id", "penalties_total", "penalties_on_home", "penalties_on_away"]
        ],
        on="game_id",
        how="outer",
        suffixes=("_new", "_existing"),
        indicator=True,
    )
    only_new = int((merged["_merge"] == "left_only").sum())
    only_existing = int((merged["_merge"] == "right_only").sum())
    both = merged.loc[merged["_merge"] == "both"].copy()
    mismatched = both.loc[
        (both["penalties_total_new"] != both["penalties_total_existing"])
        | (both["penalties_on_home_new"] != both["penalties_on_home_existing"])
        | (both["penalties_on_away_new"] != both["penalties_on_away_existing"])
    ]
    existing_games_per_season = (
        existing_game_penalties.groupby("season")["game_id"].nunique().to_dict()
    )
    new_games_per_season = new_long.groupby("season")["game_id"].nunique().to_dict()
    return {
        "games_only_in_new": only_new,
        "games_only_in_existing": only_existing,
        "games_matched": len(both),
        "games_with_count_mismatch": len(mismatched),
        "mismatch_examples": mismatched["game_id"].head(5).tolist(),
        "existing_games_per_season": {int(k): int(v) for k, v in existing_games_per_season.items()},
        "new_games_per_season": {int(k): int(v) for k, v in new_games_per_season.items()},
        "season_game_counts_match": {
            int(season): int(existing_games_per_season.get(season, -1)) == int(count)
            for season, count in new_games_per_season.items()
        },
    }


def build_snapshot(
    seasons: tuple[int, ...],
    raw_root: Path,
    existing_officials_snapshot_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    frames = [fetch_season_penalty_types(season) for season in seasons]
    long_table = pd.concat(frames, ignore_index=True)

    existing_game_penalties_path = existing_officials_snapshot_dir / "game_penalties.parquet"
    existing_game_penalties = pd.read_parquet(existing_game_penalties_path)
    verification = verify_against_existing_totals(long_table, existing_game_penalties)

    identifier = run_id()
    destination = raw_root / "officials" / identifier
    output_path = destination / "game_penalty_types.parquet"
    atomic_parquet(long_table, output_path)

    manifest = {
        "snapshot_id": identifier,
        "source": (
            "nflreadpy.load_pbp(seasons=...), one season at a time, full-column pull -- "
            "widened to retain penalty_type/penalty_team (absent from "
            "nfl_ats.pbp.PBP_SNAPSHOT_COLUMNS and data/pbp/team_style/raw_pbp_narrow.parquet). "
            "Raw PBP is NOT persisted, only this derived (game_id, penalty_type) aggregate -- "
            "same convention as the existing game_penalties.parquet."
        ),
        "seasons": list(seasons),
        "rows": len(long_table),
        "distinct_games": int(long_table["game_id"].nunique()),
        "distinct_penalty_types": int(long_table["penalty_type"].nunique()),
        "compared_against": str(existing_game_penalties_path),
        "verification": verification,
        "columns": list(long_table.columns),
    }
    atomic_json(manifest, destination / "manifest_penalty_types.json")
    return output_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seasons-start", type=int, default=DEFAULT_SEASONS[0])
    parser.add_argument("--seasons-end", type=int, default=DEFAULT_SEASONS[-1])
    parser.add_argument(
        "--existing-officials-snapshot",
        type=Path,
        default=None,
        help="directory containing the existing officials.parquet/game_penalties.parquet",
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    raw_root = repo_root / "data" / "raw"
    existing_dir = args.existing_officials_snapshot
    if existing_dir is None:
        candidates = sorted((raw_root / "officials").glob("*/officials.parquet"))
        if not candidates:
            raise SystemExit("No existing data/raw/officials/*/officials.parquet snapshot found")
        existing_dir = candidates[-1].parent

    seasons = tuple(range(args.seasons_start, args.seasons_end + 1))
    output_path, manifest = build_snapshot(seasons, raw_root, existing_dir)
    print(f"Wrote {output_path}")
    print(f"Rows: {manifest['rows']}, distinct games: {manifest['distinct_games']}")
    print(f"Distinct penalty types: {manifest['distinct_penalty_types']}")
    verification = manifest["verification"]
    print(
        "Verification: games_only_in_new="
        f"{verification['games_only_in_new']}, games_only_in_existing="
        f"{verification['games_only_in_existing']}, "
        f"count_mismatches={verification['games_with_count_mismatch']}"
    )
    print(f"Per-season game count match: {verification['season_game_counts_match']}")


if __name__ == "__main__":
    main()
