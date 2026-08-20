"""PBP-06 special-teams trait builder: kicking, punting, return, and block-rate
tendencies, aggregated to team-game and team-season tables from fresh nflverse
play-by-play.

**Data source and pregame-safety argument.** ``nfl_ats.pbp`` intentionally
narrows the stored play-by-play snapshot to ``PBP_SNAPSHOT_COLUMNS`` (read,
``src/nfl_ats/pbp.py``), which omits every special-teams column this task
needs (``kick_distance``, ``field_goal_result``, ``return_yards``,
``return_team``, ``touchback``, ``punt_blocked``, etc.) -- confirmed by
reading the constant. This script therefore fetches nflreadpy's full-column
PBP directly, ONE SEASON AT A TIME, immediately aggregates that season's raw
rows down to small per-team-game and per-team-season tables, and DISCARDS the
season's raw frame before fetching the next -- the ~740k-row raw PBP is never
concatenated in memory across seasons and never written to disk, per the
task's explicit instruction (stricter than the ``team_style`` precedent,
which does cache a narrowed raw parquet; this module follows the
``referee_battery`` precedent instead -- "the raw PBP itself is NOT
persisted, only this small derived aggregate"). Every trait here is a
TRAILING prior-season value at screening time (see
``special_teams_screen.py``), so nothing here reads a game's own outcome.

**Traits built** (one row per (season, team) in ``team_season`` and one row
per (season, week, game_id, team) in ``team_game``):

- ``fg_oe``: field-goal makes minus a season-and-distance-bucket expected
  make rate, per attempt, averaged. Distance buckets (<30, 30-39, 40-49,
  50+) computed from THAT SEASON's own league-wide attempts (self-
  referential in the same sense nflverse's own ``pass_oe`` is -- a team's
  own attempts contribute a small amount to the season baseline it is
  compared against; with ~32 teams this dilution is minor and is the same
  convention ``proe`` already ships with).
- ``punt_net_yards``: standard net-punting formula, ``kick_distance -
  return_yards`` for a normal punt, capped at ``yardline_100 - 20`` for a
  touchback (the ball is only ever "worth" up to the receiving team's 20;
  verified against a real 2024 row: yardline_100=56, kick_distance=56,
  touchback=1 -> net=36, matching the standard NFL net-punting convention
  by hand -- MEASURED this session).
- ``punt_return_yards``: mean return_yards on punts with a genuine return
  (excludes fair catch, touchback, blocked, downed, out-of-bounds).
- ``kickoff_return_yards``: mirror of the above for kickoffs (excludes
  touchback, fair catch).
- ``block_rate``: share of this team's own FG+punt attempts that were
  blocked (protection-unit trait, not kicker leg trait).

Every dimension is pooled directly from the underlying attempts at each
level (team-game, team-season), never averaged from a lower granularity
(Simpson's-paradox precaution, PER-07/PBP-08 convention), then centered
against its OWN SEASON's unweighted team mean in a final pass across all
17 seasons (era-drift removal -- kickoff return average in particular is
expected to show a large level shift after the 2024 "dynamic kickoff" rule
change; centering absorbs the level, not the persistence).

``TEAM_ABBREVIATION_ALIASES`` (OAK->LV, SD->LAC, STL->LA) is applied to every
team-identifying column so specialist continuity survives relocations.

Output (both gitignored under ``data/raw/**``):
  - ``data/raw/special_teams/<UTC timestamp>/team_game.parquet``
  - ``data/raw/special_teams/<UTC timestamp>/team_season.parquet``
  - ``data/raw/special_teams/<UTC timestamp>/manifest.json``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402

SEASON_START = 2009
SEASON_END = 2025

RAW_COLUMNS = (
    "season",
    "week",
    "season_type",
    "game_id",
    "posteam",
    "defteam",
    "punt_attempt",
    "field_goal_attempt",
    "kickoff_attempt",
    "kick_distance",
    "return_yards",
    "return_team",
    "field_goal_result",
    "touchback",
    "punt_blocked",
    "punt_fair_catch",
    "punt_downed",
    "punt_out_of_bounds",
    "yardline_100",
    "kickoff_fair_catch",
    "kickoff_out_of_bounds",
    "kickoff_downed",
)

STYLE_DIMENSIONS = (
    "fg_oe",
    "punt_net_yards",
    "punt_return_yards",
    "kickoff_return_yards",
    "block_rate",
)

FG_DISTANCE_BINS = (0, 29, 39, 49, 1000)
FG_DISTANCE_LABELS = ("u30", "d30_39", "d40_49", "d50p")


def _alias(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_ABBREVIATION_ALIASES)


def _fetch_season(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    frame = nfl.load_pbp(seasons=[season])
    pdf = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    if not isinstance(pdf, pd.DataFrame):
        raise TypeError(f"Unexpected nflreadpy return type: {type(pdf)!r}")
    pdf = pdf.loc[pdf["season_type"] == "REG"].copy()
    for column in RAW_COLUMNS:
        if column not in pdf.columns:
            pdf[column] = np.nan
    return pdf.loc[:, list(RAW_COLUMNS)].reset_index(drop=True)


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _fg_residuals(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per FG attempt: season, week, game_id, team, fg_residual."""

    fg = raw.loc[raw["field_goal_attempt"].fillna(0).eq(1)].copy()
    if fg.empty:
        return pd.DataFrame(
            columns=["season", "week", "game_id", "team", "fg_residual", "fg_blocked"]
        )
    fg["kick_distance"] = pd.to_numeric(fg["kick_distance"], errors="coerce")
    fg["made"] = (fg["field_goal_result"] == "made").astype(float)
    fg["fg_blocked"] = (fg["field_goal_result"] == "blocked").astype(float)
    fg["distance_bucket"] = pd.cut(
        fg["kick_distance"], bins=FG_DISTANCE_BINS, labels=FG_DISTANCE_LABELS
    )
    # Season-and-distance-bucket league make rate, computed from THIS season's
    # own attempts only (self-referential in the same documented sense as
    # nflverse's own pass_oe -- see module docstring).
    league_rate = fg.groupby(["season", "distance_bucket"], observed=True)["made"].transform("mean")
    fg["fg_residual"] = fg["made"] - league_rate
    fg["team"] = _alias(fg["posteam"])
    return fg[["season", "week", "game_id", "team", "fg_residual", "fg_blocked"]]


def _punt_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per punt attempt: season, week, game_id, team (kicking),
    net_yards, is_blocked, plus return-side rows keyed by the returning team.
    """

    punt = raw.loc[raw["punt_attempt"].fillna(0).eq(1)].copy()
    if punt.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_id",
                "team",
                "net_yards",
                "is_blocked",
                "return_team",
                "return_yards",
                "is_real_return",
            ]
        )
    punt = _numeric(punt, ["kick_distance", "return_yards", "yardline_100", "touchback"])
    punt["is_blocked"] = punt["punt_blocked"].fillna(0).eq(1).astype(float)
    # Standard net-punting formula: gross kick distance minus the return,
    # capped at (yardline_100 - 20) on a touchback -- the ball is only ever
    # "worth" up to the receiving team's 20 (verified against a real row,
    # see module docstring).
    normal_net = punt["kick_distance"] - punt["return_yards"].fillna(0.0)
    touchback_net = punt["yardline_100"] - 20.0
    punt["net_yards"] = np.where(punt["touchback"].fillna(0).eq(1), touchback_net, normal_net)
    punt["team"] = _alias(punt["posteam"])
    no_real_return = (
        punt["punt_fair_catch"].fillna(0).eq(1)
        | punt["touchback"].fillna(0).eq(1)
        | punt["punt_blocked"].fillna(0).eq(1)
        | punt["punt_downed"].fillna(0).eq(1)
        | punt["punt_out_of_bounds"].fillna(0).eq(1)
        | punt["return_team"].isna()
    )
    punt["is_real_return"] = ~no_real_return
    punt["return_team_aliased"] = _alias(punt["return_team"])
    return punt[
        [
            "season",
            "week",
            "game_id",
            "team",
            "net_yards",
            "is_blocked",
            "return_team_aliased",
            "return_yards",
            "is_real_return",
        ]
    ].rename(columns={"return_team_aliased": "return_team"})


def _kickoff_return_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per kickoff attempt with return-side info."""

    ko = raw.loc[raw["kickoff_attempt"].fillna(0).eq(1)].copy()
    if ko.empty:
        return pd.DataFrame(
            columns=["season", "week", "game_id", "return_team", "return_yards", "is_real_return"]
        )
    ko = _numeric(ko, ["return_yards", "touchback"])
    no_real_return = (
        ko["kickoff_fair_catch"].fillna(0).eq(1)
        | ko["touchback"].fillna(0).eq(1)
        | ko["kickoff_out_of_bounds"].fillna(0).eq(1)
        | ko["kickoff_downed"].fillna(0).eq(1)
        | ko["return_team"].isna()
    )
    ko["is_real_return"] = ~no_real_return
    ko["return_team_aliased"] = _alias(ko["return_team"])
    return ko[
        ["season", "week", "game_id", "return_team_aliased", "return_yards", "is_real_return"]
    ].rename(columns={"return_team_aliased": "return_team"})


def aggregate_season(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build this season's team_game/team_season fragments from raw PBP.

    Returns small per-season fragments -- the caller concatenates fragments
    across seasons, never the raw frames.
    """

    fg = _fg_residuals(raw)
    punt = _punt_rows(raw)
    ko = _kickoff_return_rows(raw)

    game_keys = ["season", "week", "game_id", "team"]
    season_keys = ["season", "team"]

    fg_game = (
        fg.groupby(game_keys, sort=False)
        .agg(n_fg_attempts=("fg_residual", "size"), fg_oe=("fg_residual", "mean"))
        .reset_index()
    )
    fg_season = (
        fg.groupby(season_keys, sort=False)
        .agg(n_fg_attempts=("fg_residual", "size"), fg_oe=("fg_residual", "mean"))
        .reset_index()
    )

    kick_game = (
        punt.groupby(game_keys, sort=False)
        .agg(
            n_punts=("net_yards", "size"),
            punt_net_yards=("net_yards", "mean"),
            n_kicks_blocked=("is_blocked", "sum"),
        )
        .reset_index()
    )
    kick_season = (
        punt.groupby(season_keys, sort=False)
        .agg(
            n_punts=("net_yards", "size"),
            punt_net_yards=("net_yards", "mean"),
            n_kicks_blocked=("is_blocked", "sum"),
        )
        .reset_index()
    )

    preturn = punt.loc[punt["is_real_return"]].dropna(subset=["return_team"]).copy()
    preturn_game = (
        preturn.groupby(["season", "week", "game_id", "return_team"], sort=False)
        .agg(n_punt_returns=("return_yards", "size"), punt_return_yards=("return_yards", "mean"))
        .reset_index()
    )
    preturn_game = preturn_game.rename(columns={"return_team": "team"})
    preturn_season = (
        preturn.groupby(["season", "return_team"], sort=False)
        .agg(n_punt_returns=("return_yards", "size"), punt_return_yards=("return_yards", "mean"))
        .reset_index()
        .rename(columns={"return_team": "team"})
    )

    kreturn = ko.loc[ko["is_real_return"]].dropna(subset=["return_team"]).copy()
    kreturn_game = (
        kreturn.groupby(["season", "week", "game_id", "return_team"], sort=False)
        .agg(
            n_kickoff_returns=("return_yards", "size"),
            kickoff_return_yards=("return_yards", "mean"),
        )
        .reset_index()
        .rename(columns={"return_team": "team"})
    )
    kreturn_season = (
        kreturn.groupby(["season", "return_team"], sort=False)
        .agg(
            n_kickoff_returns=("return_yards", "size"),
            kickoff_return_yards=("return_yards", "mean"),
        )
        .reset_index()
        .rename(columns={"return_team": "team"})
    )

    return {
        "fg_game": fg_game,
        "fg_season": fg_season,
        "kick_game": kick_game,
        "kick_season": kick_season,
        "preturn_game": preturn_game,
        "preturn_season": preturn_season,
        "kreturn_game": kreturn_game,
        "kreturn_season": kreturn_season,
    }


def _merge_all(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=keys, how="outer")
    return result


def add_block_rate(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    n_total_kicks = table["n_fg_attempts"].fillna(0) + table["n_punts"].fillna(0)
    table["n_kicks_total"] = n_total_kicks
    table["block_rate"] = np.where(
        n_total_kicks > 0, table["n_kicks_blocked"].fillna(0) / n_total_kicks, np.nan
    )
    return table


def add_league_centered(table: pd.DataFrame, *, season_col: str = "season") -> pd.DataFrame:
    result = table.copy()
    for dim in STYLE_DIMENSIONS:
        if dim not in result.columns:
            continue
        league_mean = result.groupby(season_col)[dim].transform("mean")
        result[f"{dim}_centered"] = result[dim] - league_mean
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season-start", type=int, default=SEASON_START)
    parser.add_argument("--season-end", type=int, default=SEASON_END)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    started = time.time()
    seasons = range(args.season_start, args.season_end + 1)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "data" / "raw" / "special_teams" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    fg_game_parts, fg_season_parts = [], []
    kick_game_parts, kick_season_parts = [], []
    preturn_game_parts, preturn_season_parts = [], []
    kreturn_game_parts, kreturn_season_parts = [], []
    total_raw_rows = 0

    print(f"=== fetching nflverse PBP {seasons.start}-{seasons.stop - 1} season-by-season ===")
    print("(raw PBP is aggregated immediately and NEVER persisted or held across seasons)")
    for season in seasons:
        t0 = time.time()
        raw = _fetch_season(season)
        total_raw_rows += len(raw)
        frags = aggregate_season(raw)
        del raw  # discard the season's raw frame before the next fetch
        fg_game_parts.append(frags["fg_game"])
        fg_season_parts.append(frags["fg_season"])
        kick_game_parts.append(frags["kick_game"])
        kick_season_parts.append(frags["kick_season"])
        preturn_game_parts.append(frags["preturn_game"])
        preturn_season_parts.append(frags["preturn_season"])
        kreturn_game_parts.append(frags["kreturn_game"])
        kreturn_season_parts.append(frags["kreturn_season"])
        print(f"  season {season}: aggregated ({time.time() - t0:.1f}s)")

    fg_game = pd.concat(fg_game_parts, ignore_index=True)
    fg_season = pd.concat(fg_season_parts, ignore_index=True)
    kick_game = pd.concat(kick_game_parts, ignore_index=True)
    kick_season = pd.concat(kick_season_parts, ignore_index=True)
    preturn_game = pd.concat(preturn_game_parts, ignore_index=True)
    preturn_season = pd.concat(preturn_season_parts, ignore_index=True)
    kreturn_game = pd.concat(kreturn_game_parts, ignore_index=True)
    kreturn_season = pd.concat(kreturn_season_parts, ignore_index=True)

    team_game = _merge_all(
        [fg_game, kick_game, preturn_game, kreturn_game], ["season", "week", "game_id", "team"]
    )
    team_game = add_block_rate(team_game)
    team_game = add_league_centered(team_game)

    team_season = _merge_all(
        [fg_season, kick_season, preturn_season, kreturn_season], ["season", "team"]
    )
    team_season = add_block_rate(team_season)
    team_season = add_league_centered(team_season)

    team_game = team_game.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    team_season = team_season.sort_values(["season", "team"]).reset_index(drop=True)

    atomic_parquet(team_game, output_dir / "team_game.parquet")
    atomic_parquet(team_season, output_dir / "team_season.parquet")
    atomic_json(
        {
            "seasons": [seasons.start, seasons.stop - 1],
            "total_raw_pbp_rows_processed": total_raw_rows,
            "raw_pbp_persisted": False,
            "team_game_rows": len(team_game),
            "team_season_rows": len(team_season),
            "dimensions": list(STYLE_DIMENSIONS),
            "source": (
                "nflreadpy.load_pbp, full columns, fetched season-by-season and aggregated "
                "immediately (never concatenated across seasons, never written to disk)"
            ),
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.time() - started,
        },
        output_dir / "manifest.json",
    )
    print(
        f"\nwrote team_game.parquet ({len(team_game)} rows), "
        f"team_season.parquet ({len(team_season)} rows) -> {output_dir}"
    )
    print(f"total raw PBP rows processed (never persisted): {total_raw_rows}")


if __name__ == "__main__":
    main()
