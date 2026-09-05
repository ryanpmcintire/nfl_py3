"""Rebuild the play-probability training panel from local archives only.

Reads the all-position depth history, player snapshot, injury revisions and
latest local schedules.parquet. Injury visibility uses the game's pool
cutoff. Daily depth rows without a provable pre-decision observation time
are excluded; their count is recorded in the output sidecar. Legacy weekly
rows retain the archive's week-labelled pregame assumption.

Writes data/processed/play_probability_panel.parquet and its provenance
sidecar (or --output). Source ingestion is a separate operation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nfl_ats.io import atomic_parquet
from nfl_ats.play_probability import (
    build_player_week_panel,
    canonicalize_depth_chart_history,
    latest_depth_chart_history_snapshot,
    load_depth_chart_history_snapshot,
)
from nfl_ats.players import (
    canonicalize_injuries,
    latest_player_snapshot,
    load_player_snapshot,
)
from nfl_ats.provenance import stamp_sidecar

DEPTH_CHART_HISTORY_ROOT = Path("data") / "players" / "raw" / "depth_charts"
PLAYER_SNAPSHOT_ROOT = Path("data") / "players" / "raw"
RAW_INJURIES_ROOT = Path("data") / "raw" / "nflverse_injuries"
RAW_DEPTH_ROOT = Path("data/raw/depth_charts")
PANEL_OUTPUT_PATH = Path("data") / "processed" / "play_probability_panel.parquet"


def resolve_injuries_path(path: Path | None = None) -> Path:
    """Use an explicit archive or the newest local timestamped injury snapshot."""
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"No local injury archive at {path}")
        return path
    paths = sorted(RAW_INJURIES_ROOT.glob("*/injuries.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No local injuries.parquet in {RAW_INJURIES_ROOT}; ingest separately"
        )
    return paths[-1]


def _load_or_fetch_depth_history(start_season: int, end_season: int) -> pd.DataFrame:
    try:
        snapshot = latest_depth_chart_history_snapshot(DEPTH_CHART_HISTORY_ROOT)
        covered = set(range(start_season, end_season + 1)).issubset(set(snapshot.requested_seasons))
        if covered:
            return load_depth_chart_history_snapshot(snapshot)
    except FileNotFoundError:
        pass
    raise FileNotFoundError(
        "No local depth history covers the requested seasons; ingest separately"
    )


def load_panel_depth_history(
    start_season: int, end_season: int, schedule: pd.DataFrame
) -> pd.DataFrame:
    """Replace only 2025 with the newest complete immutable daily snapshot."""
    history = _load_or_fetch_depth_history(start_season, end_season)
    history = history.loc[history["season"].between(start_season, end_season)].copy()
    if not start_season <= 2025 <= end_season:
        return history
    paths = sorted(
        path
        for path in RAW_DEPTH_ROOT.glob("*/depth_charts.parquet")
        if (path.parent / "manifest.json").is_file()
    )
    if not paths:
        return history
    path = paths[-1]
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("requested_seasons") != [2025]:
        raise ValueError(f"Expected a 2025 daily depth snapshot: {path}")
    daily = canonicalize_depth_chart_history(
        pd.read_parquet(path), schedule.loc[schedule["season"].eq(2025)]
    )
    result = pd.concat([history.loc[history["season"].ne(2025)], daily], ignore_index=True)
    result.attrs["raw_2025_depth_source"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2013)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=PANEL_OUTPUT_PATH)
    parser.add_argument(
        "--injuries-path", type=Path, help="Pinned injury parquet; default: newest local snapshot"
    )
    args = parser.parse_args()

    player_snapshot = latest_player_snapshot(PLAYER_SNAPSHOT_ROOT)
    _, rosters, snaps = load_player_snapshot(player_snapshot, include_postseason=False)

    injuries_path = resolve_injuries_path(args.injuries_path)
    raw_injuries = pd.read_parquet(injuries_path)

    from nfl_ats.players import _schedule_kickoff_utc

    schedule_paths = sorted(Path("data/raw").glob("*/schedules.parquet"))
    if not schedule_paths:
        raise FileNotFoundError("No local schedules.parquet; ingest separately")
    schedule = pd.read_parquet(schedule_paths[-1])
    schedule["kickoff"] = _schedule_kickoff_utc(schedule)
    depth_history = load_panel_depth_history(args.start_season, args.end_season, schedule)
    injuries = canonicalize_injuries(
        raw_injuries, include_postseason=False, timestamp_fallback="week_proxy", schedule=schedule
    )

    panel = build_player_week_panel(depth_history, rosters, snaps, injuries, schedule=schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(panel, args.output)
    stamp_sidecar(
        args.output,
        extra={
            "rows": len(panel),
            "schedule_source": str(schedule_paths[-1]),
            "excluded_unverifiable_daily_rows": int(
                depth_history["source_schema"].eq("daily_dt").sum()
                - panel["source_schema"].eq("daily_dt").sum()
            ),
            "seasons_covered": sorted(int(value) for value in panel["season"].unique()),
            "depth_chart_history_seasons": list(depth_history["season"].unique().tolist()),
            "raw_injuries_source": str(injuries_path),
            "raw_2025_depth_source": depth_history.attrs.get("raw_2025_depth_source"),
            "player_snapshot_id": player_snapshot.snapshot_id,
        },
    )
    print(args.output)
    print(f"rows={len(panel)} seasons={sorted(panel['season'].unique().tolist())}")


if __name__ == "__main__":
    main()
