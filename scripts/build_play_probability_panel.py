"""Build the cached training panel `nfl_ats.play_probability` fits on.

Kept as a SEPARATE, occasionally-rerun step from `scripts/build_week_lineups.py`
on purpose: building the panel needs a full nflverse schedule fetch (only to
resolve each historical injury revision's ENG-39 week_proxy visibility
timestamp -- see `nfl_ats.players.canonicalize_injuries`) and roughly a
minute of joins/merges over ~390k depth-chart-row-weeks. Refitting the
walk-forward model on every noon-Eastern lineup refresh would repeat both
costs for no benefit, since the training panel itself does not change
within a season; `build_week_lineups.py` instead reads this script's cached
output (`data/processed/play_probability_panel.parquet`) offline and only
FITS (a few seconds) on every refresh.

Re-run this script whenever a new season's snap_counts/injuries/rosters/
depth-chart data becomes available (new-season data ingest, not a per-refresh
task). Writes:

- `data/players/raw/depth_charts/<stamp>/depth_charts.parquet` (only if no
  local archive covering `--start-season`..`--end-season` already exists) --
  the all-position depth-chart history `nfl_ats.quarterbacks`'s QB-only
  archives never captured. Network fetch via nflreadpy (nflverse is GREEN in
  `config/source_policies.json`). Nested one level under `depth_charts/`
  rather than directly in `data/players/raw/<stamp>/` (as first tried, then
  reverted -- measured this session): `nfl_ats.players.latest_player_snapshot`
  globs `data/players/raw/*/manifest.json` and blindly picks the
  lexicographically-last match assuming it is always a `PlayerSnapshot`
  manifest; a depth-chart-history manifest living at that same depth broke
  it (`KeyError: 'injury_seasons'`) for every OTHER caller of
  `latest_player_snapshot` sharing this tree, including
  `scripts/build_week_lineups.py`'s already-shipped `_no_designation_lookup`.
  `latest_player_snapshot` is a shared module this lane may not edit, so the
  archive moved one directory deeper instead, which the one-level `*/` glob
  cannot reach.
- `data/processed/play_probability_panel.parquet` -- the built training
  panel, plus a `nfl_ats.provenance.stamp_sidecar`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_ats.io import atomic_parquet
from nfl_ats.play_probability import (
    build_player_week_panel,
    fetch_depth_chart_history_snapshot,
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
RAW_INJURIES_PATH = (
    Path("data") / "raw" / "nflverse_injuries" / "20260826T122850Z" / "injuries.parquet"
)
PANEL_OUTPUT_PATH = Path("data") / "processed" / "play_probability_panel.parquet"


def _load_or_fetch_depth_history(start_season: int, end_season: int) -> pd.DataFrame:
    try:
        snapshot = latest_depth_chart_history_snapshot(DEPTH_CHART_HISTORY_ROOT)
        covered = set(range(start_season, end_season + 1)).issubset(set(snapshot.requested_seasons))
        if covered:
            return load_depth_chart_history_snapshot(snapshot)
    except FileNotFoundError:
        pass
    snapshot = fetch_depth_chart_history_snapshot(
        list(range(start_season, end_season + 1)), DEPTH_CHART_HISTORY_ROOT
    )
    return load_depth_chart_history_snapshot(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2013)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=PANEL_OUTPUT_PATH)
    args = parser.parse_args()

    depth_history = _load_or_fetch_depth_history(args.start_season, args.end_season)

    player_snapshot = latest_player_snapshot(PLAYER_SNAPSHOT_ROOT)
    _, rosters, snaps = load_player_snapshot(player_snapshot, include_postseason=False)

    raw_injuries = pd.read_parquet(RAW_INJURIES_PATH)
    import nflreadpy as nfl

    from nfl_ats.players import _schedule_kickoff_utc

    schedule = nfl.load_schedules(
        seasons=list(range(args.start_season, args.end_season + 1))
    ).to_pandas()
    schedule["kickoff"] = _schedule_kickoff_utc(schedule)
    injuries = canonicalize_injuries(
        raw_injuries, include_postseason=False, timestamp_fallback="week_proxy", schedule=schedule
    )

    panel = build_player_week_panel(depth_history, rosters, snaps, injuries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(panel, args.output)
    stamp_sidecar(
        args.output,
        extra={
            "rows": len(panel),
            "seasons_covered": sorted(int(value) for value in panel["season"].unique()),
            "depth_chart_history_seasons": list(depth_history["season"].unique().tolist()),
            "raw_injuries_source": str(RAW_INJURIES_PATH),
            "player_snapshot_id": player_snapshot.snapshot_id,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
