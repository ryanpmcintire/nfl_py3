"""Compare two point-in-time player-prop snapshots without grading outcomes.

This is an ingestion-diagnostic tool for ``docs/player_props_sourcing.md``.
It measures whether the same player/game markets exist at two pre-kickoff
timestamps and how common-book main lines moved. It deliberately does not
score ATS outcomes or write an experiment artifact.

BetRivers is excluded from line comparisons by default because the source
returns an alternate-line ladder under the same market key. Presence counts
still include every bookmaker because they collapse to one player/game row.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "week",
    "nflverse_game_id",
    "commence_time_utc",
    "snapshot_actual_at_utc",
    "bookmaker_key",
    "player_name",
    "line",
}
PLAYER_GAME_KEYS = ["week", "nflverse_game_id", "player_key"]
LINE_KEYS = [*PLAYER_GAME_KEYS, "bookmaker_key"]


def _prepare(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Validate one snapshot and add a stable normalized player key."""

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} snapshot is missing columns: {sorted(missing)}")

    prepared = frame.copy()
    prepared["snapshot_actual_at_utc"] = pd.to_datetime(
        prepared["snapshot_actual_at_utc"], utc=True, errors="raise"
    )
    prepared["commence_time_utc"] = pd.to_datetime(
        prepared["commence_time_utc"], utc=True, errors="raise"
    )
    leaked = prepared["snapshot_actual_at_utc"] >= prepared["commence_time_utc"]
    if leaked.any():
        examples = prepared.loc[leaked, "nflverse_game_id"].astype(str).unique()[:3]
        raise ValueError(
            f"{label} snapshot is not pregame for {int(leaked.sum())} rows; "
            f"examples={examples.tolist()}"
        )

    prepared["player_key"] = prepared["player_name"].astype("string").str.strip().str.lower()
    if prepared["player_key"].isna().any() or prepared["player_key"].eq("").any():
        raise ValueError(f"{label} snapshot contains a blank player name")
    return prepared


def compare_snapshots(
    earlier: pd.DataFrame,
    later: pd.DataFrame,
    *,
    excluded_bookmakers: tuple[str, ...] = ("betrivers",),
) -> dict[str, Any]:
    """Return presence and common-book line comparisons for two snapshots."""

    left = _prepare(earlier, label="earlier")
    right = _prepare(later, label="later")

    left_players = left[PLAYER_GAME_KEYS].drop_duplicates()
    right_players = right[PLAYER_GAME_KEYS].drop_duplicates()
    week_rows: list[dict[str, int]] = []
    weeks = sorted(set(left_players["week"]).union(right_players["week"]))
    for week in weeks:
        left_week = left_players[left_players["week"].eq(week)]
        right_week = right_players[right_players["week"].eq(week)]
        both = left_week.merge(right_week, on=PLAYER_GAME_KEYS)
        left_only = left_week.merge(
            right_week, on=PLAYER_GAME_KEYS, how="left", indicator=True
        ).query("_merge == 'left_only'")
        right_only = right_week.merge(
            left_week, on=PLAYER_GAME_KEYS, how="left", indicator=True
        ).query("_merge == 'left_only'")
        week_rows.append(
            {
                "week": int(week),
                "earlier_player_games": len(left_week),
                "later_player_games": len(right_week),
                "both": len(both),
                "earlier_only": len(left_only),
                "later_only": len(right_only),
            }
        )

    def collapse_lines(frame: pd.DataFrame) -> pd.DataFrame:
        eligible = frame[~frame["bookmaker_key"].isin(excluded_bookmakers)]
        return eligible.groupby(LINE_KEYS, as_index=False).agg(line=("line", "median"))

    common_lines = collapse_lines(left).merge(
        collapse_lines(right), on=LINE_KEYS, suffixes=("_earlier", "_later")
    )
    common_lines["delta"] = common_lines["line_later"] - common_lines["line_earlier"]
    common_games = {
        int(week): int(count)
        for week, count in common_lines.groupby("week")["nflverse_game_id"].nunique().items()
    }
    delta = common_lines["delta"]
    delta_summary = {
        "count": int(delta.count()),
        "mean": float(delta.mean()) if not delta.empty else None,
        "median": float(delta.median()) if not delta.empty else None,
        "min": float(delta.min()) if not delta.empty else None,
        "max": float(delta.max()) if not delta.empty else None,
    }
    return {
        "weeks": pd.DataFrame(week_rows),
        "common_book_player_lines": len(common_lines),
        "common_player_games": len(common_lines[PLAYER_GAME_KEYS].drop_duplicates()),
        "common_games_by_week": common_games,
        "delta_summary": delta_summary,
        "largest_moves": common_lines.assign(abs_delta=common_lines["delta"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(20),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--later", type=Path, required=True)
    parser.add_argument(
        "--exclude-bookmaker",
        action="append",
        default=["betrivers"],
        help="Bookmaker key to exclude from line movement (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = compare_snapshots(
        pd.read_parquet(args.earlier),
        pd.read_parquet(args.later),
        excluded_bookmakers=tuple(args.exclude_bookmaker),
    )
    print("Player/game pairability by week")
    print(summary["weeks"].to_string(index=False))
    print("\nCommon-line summary")
    print(f"common_book_player_lines={summary['common_book_player_lines']}")
    print(f"common_player_games={summary['common_player_games']}")
    print(f"common_games_by_week={summary['common_games_by_week']}")
    print(f"delta_summary={summary['delta_summary']}")
    print("\nLargest absolute common-book moves")
    print(summary["largest_moves"].to_string(index=False))


if __name__ == "__main__":
    main()
