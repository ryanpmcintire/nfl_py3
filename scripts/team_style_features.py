"""Trailing, pregame-safe team "style"/personality feature builder (PBP-08).

Builds team-season and team-game play-calling TENDENCY vectors -- short-pass
share, deep-pass share, average air yards, a screen-play proxy, pass-rate-
over-expected (PROE, using nflverse's own down/distance/score/time-adjusted
``pass_oe`` -- adjusted and cheap, since it ships with the play-by-play),
shotgun rate, no-huddle rate, run-direction concentration (Herfindahl index
over left/middle/right), and seconds-per-play pace (drive time-of-possession
divided by drive play count). These are deliberately QUALITY-ORTHOGONAL:
none of them measure how good a team is, only how it chooses to play, so
they sit outside the measured team-quality ceiling (ROADMAP.md PBP-05) that
already bounds "measure team quality better" features near zero.

Every dimension is centered against ITS OWN SEASON'S league mean (the exact
PER-07 convention: "expressed relative to each season's league norm ... strips
the well-documented league-wide drift"), so a raw era-wide drift (e.g. the
leaguewide shift to more shotgun/no-huddle over 2009-2025) does not read as a
team identity.

**Full nflverse PBP, not the trimmed in-repo snapshot.** ``nfl_ats.pbp``
intentionally narrows the stored play-by-play snapshot to
``PBP_SNAPSHOT_COLUMNS`` (no ``air_yards``, ``shotgun``, ``no_huddle``,
``run_location`` -- confirmed by reading ``src/nfl_ats/pbp.py``), and no
untrimmed snapshot exists in this local clone (``data/raw/*/manifest.json``
has no ``season=*`` PBP partitions locally -- read, this session). This
script therefore fetches nflreadpy's full-column PBP directly (network
call, one-time per season, cached locally in ``data/pbp/team_style/`` which
is already gitignored) rather than routing through the narrowed contract.
It reuses ``nfl_ats.pbp.analysis_plays``/``build_drive_table`` (the same
play-eligibility filter and drive aggregation already used by the production
PBP-05 pipeline) and ``nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`` for
franchise continuity (OAK->LV, SD->LAC, STL->LA) -- applied by hand here
because ``build_pbp_team_game_metrics`` applies the alias to its own
``plays`` frame but calls ``build_drive_table`` on the UNALIASED raw pbp,
so drive-level columns silently fail to match aliased team codes for
OAK/SD/STL-era games (read in ``src/nfl_ats/pbp.py``, not used directly
here for that reason -- this script re-derives drive pace itself with the
alias applied consistently on both sides).

Output (both gitignored, ``data/pbp/**``):
  - ``data/pbp/team_style/team_game_style.parquet``: one row per
    (season REG, week, game_id, team) with the 9 raw dimensions plus
    league-season-centered versions.
  - ``data/pbp/team_style/team_season_style.parquet``: one row per
    (season, team), pooled directly from plays/drives (not an average of
    game rates, to avoid Simpson's-paradox bias from uneven per-game play
    counts), raw + centered, plus n_pass_plays/n_off_plays/n_rush_plays.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS, analysis_plays, build_drive_table  # noqa: E402

SEASON_START = 2009
SEASON_END = 2025

CACHE_DIR = REPO / "data" / "pbp" / "team_style"
RAW_CACHE_PATH = CACHE_DIR / "raw_pbp_narrow.parquet"
RAW_MANIFEST_PATH = CACHE_DIR / "raw_pbp_narrow.manifest.json"
TEAM_GAME_PATH = CACHE_DIR / "team_game_style.parquet"
TEAM_SEASON_PATH = CACHE_DIR / "team_season_style.parquet"
TEAM_SEASON_FACED_PATH = CACHE_DIR / "team_season_style_faced.parquet"

# nflverse full-column fields this script needs beyond the in-repo
# PBP_SNAPSHOT_COLUMNS contract (air-yards/formation/direction fields the
# narrowed snapshot never kept).
EXTRA_STYLE_COLUMNS = (
    "air_yards",
    "pass_length",
    "pass_location",
    "shotgun",
    "no_huddle",
    "qb_scramble",
    "run_location",
    "run_gap",
)
RAW_COLUMNS = tuple(dict.fromkeys((*PBP_SNAPSHOT_COLUMNS, *EXTRA_STYLE_COLUMNS)))

STYLE_DIMENSIONS = (
    "short_pass_share",
    "deep_share",
    "avg_air_yards",
    "screen_rate",
    "proe",
    "shotgun_rate",
    "no_huddle_rate",
    "run_direction_hhi",
    "seconds_per_play_pace",
)


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


def build_raw_cache(*, refresh: bool, seasons: range) -> pd.DataFrame:
    if RAW_CACHE_PATH.is_file() and not refresh:
        print(f"=== using cached raw PBP: {RAW_CACHE_PATH} ===")
        return pd.read_parquet(RAW_CACHE_PATH)

    print(f"=== fetching nflverse PBP {seasons.start}-{seasons.stop - 1} via nflreadpy ===")
    frames: list[pd.DataFrame] = []
    for season in seasons:
        started = time.time()
        season_frame = _fetch_season(season)
        frames.append(season_frame)
        print(f"  season {season}: {len(season_frame)} REG plays ({time.time() - started:.1f}s)")
    raw = pd.concat(frames, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_parquet(raw, RAW_CACHE_PATH)
    atomic_json(
        {
            "seasons": [seasons.start, seasons.stop - 1],
            "rows": len(raw),
            "columns": list(RAW_COLUMNS),
            "source": "nflreadpy.load_pbp, full columns, fetched directly (not via nfl_ats.pbp's "
            "narrowed snapshot contract)",
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        RAW_MANIFEST_PATH,
    )
    print(f"=== cached {len(raw)} rows -> {RAW_CACHE_PATH} ===")
    return raw


def _alias_team(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_ABBREVIATION_ALIASES)


def _herfindahl(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    shares = counts.to_numpy(dtype=float) / total
    return float(np.sum(shares**2))


def _drive_pace_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Season/game/team seconds-per-play pace from drive time-of-possession.

    Reuses ``nfl_ats.pbp.build_drive_table`` verbatim (same eligibility
    filter as the production PBP-05 pipeline: real scrimmage plays, kneels/
    spikes/aborted plays already excluded before drives are aggregated).
    Applies ``TEAM_ABBREVIATION_ALIASES`` to the drive table's own team
    column -- ``build_pbp_team_game_metrics`` does NOT do this consistently
    (see module docstring), so this recomputes it directly rather than
    importing that function.
    """

    drives = build_drive_table(raw)
    if drives.empty:
        return pd.DataFrame(
            columns=["game_id", "season", "week", "team", "drive_seconds", "drive_plays"]
        )
    drives = drives.copy()
    drives["posteam"] = _alias_team(drives["posteam"])
    drives = drives.rename(columns={"posteam": "team"})
    drives["drive_plays"] = pd.to_numeric(drives["plays"], errors="coerce")
    drives["drive_seconds"] = pd.to_numeric(drives["drive_seconds"], errors="coerce")
    return drives[["game_id", "season", "week", "team", "drive_seconds", "drive_plays"]]


def build_team_game_style(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, game_id, team): the 9 raw style dimensions."""

    plays = analysis_plays(raw)
    plays = plays.loc[plays["competitive_play"]].copy()
    for column in (
        "down",
        "pass_attempt",
        "rush_attempt",
        "air_yards",
        "complete_pass",
        "shotgun",
        "no_huddle",
        "qb_scramble",
        "pass_oe",
    ):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    plays["posteam"] = _alias_team(plays["posteam"])

    pass_plays = plays.loc[plays["pass_attempt"].eq(1) & plays["air_yards"].notna()].copy()
    pass_plays["is_short"] = pass_plays["air_yards"].le(5)
    pass_plays["is_deep"] = pass_plays["air_yards"].ge(20)
    pass_plays["is_screen"] = pass_plays["complete_pass"].eq(1) & pass_plays["air_yards"].le(0)

    off_plays = plays.loc[plays["pass_attempt"].eq(1) | plays["rush_attempt"].eq(1)].copy()

    run_plays = plays.loc[
        plays["rush_attempt"].eq(1)
        & plays["qb_scramble"].fillna(0).ne(1)
        & plays["run_location"].notna()
    ].copy()

    game_keys = ["game_id", "season", "week", "posteam"]
    pass_agg = (
        pass_plays.groupby(game_keys, sort=False)
        .agg(
            n_pass_plays=("air_yards", "size"),
            short_pass_share=("is_short", "mean"),
            deep_share=("is_deep", "mean"),
            avg_air_yards=("air_yards", "mean"),
            screen_rate=("is_screen", "mean"),
        )
        .reset_index()
    )
    off_agg = (
        off_plays.groupby(game_keys, sort=False)
        .agg(
            n_off_plays=("pass_attempt", "size"),
            proe=("pass_oe", "mean"),
            shotgun_rate=("shotgun", "mean"),
            no_huddle_rate=("no_huddle", "mean"),
        )
        .reset_index()
    )
    run_counts = (
        run_plays.groupby([*game_keys, "run_location"], sort=False).size().rename("n").reset_index()
    )
    hhi_rows: list[dict[str, Any]] = []
    for (game_id, season, week, team), group in run_counts.groupby(game_keys, sort=False):
        hhi_rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "posteam": team,
                "n_run_plays": int(group["n"].sum()),
                "run_direction_hhi": _herfindahl(group.set_index("run_location")["n"]),
            }
        )
    hhi_table = pd.DataFrame(
        hhi_rows,
        columns=["game_id", "season", "week", "posteam", "n_run_plays", "run_direction_hhi"],
    )

    game_table = pass_agg.merge(off_agg, on=game_keys, how="outer")
    game_table = game_table.merge(hhi_table, on=game_keys, how="outer")
    game_table = game_table.rename(columns={"posteam": "team"})
    game_table["season"] = game_table["season"].astype(int)
    game_table["week"] = game_table["week"].astype(int)

    pace = _drive_pace_table(raw)
    pace_game = (
        pace.groupby(["game_id", "team"], sort=False)
        .agg(drive_seconds=("drive_seconds", "sum"), drive_plays=("drive_plays", "sum"))
        .reset_index()
    )
    pace_game["seconds_per_play_pace"] = pace_game["drive_seconds"] / pace_game[
        "drive_plays"
    ].replace(0, np.nan)

    game_table = game_table.merge(
        pace_game[["game_id", "team", "seconds_per_play_pace"]],
        on=["game_id", "team"],
        how="left",
    )
    return game_table.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def build_team_season_style(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team), pooled directly from plays/drives."""

    plays = analysis_plays(raw)
    plays = plays.loc[plays["competitive_play"]].copy()
    for column in (
        "pass_attempt",
        "rush_attempt",
        "air_yards",
        "complete_pass",
        "shotgun",
        "no_huddle",
        "qb_scramble",
        "pass_oe",
    ):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    plays["posteam"] = _alias_team(plays["posteam"])

    pass_plays = plays.loc[plays["pass_attempt"].eq(1) & plays["air_yards"].notna()].copy()
    pass_plays["is_short"] = pass_plays["air_yards"].le(5)
    pass_plays["is_deep"] = pass_plays["air_yards"].ge(20)
    pass_plays["is_screen"] = pass_plays["complete_pass"].eq(1) & pass_plays["air_yards"].le(0)

    off_plays = plays.loc[plays["pass_attempt"].eq(1) | plays["rush_attempt"].eq(1)].copy()

    run_plays = plays.loc[
        plays["rush_attempt"].eq(1)
        & plays["qb_scramble"].fillna(0).ne(1)
        & plays["run_location"].notna()
    ].copy()

    pass_agg = (
        pass_plays.groupby(["season", "posteam"], sort=False)
        .agg(
            n_pass_plays=("air_yards", "size"),
            short_pass_share=("is_short", "mean"),
            deep_share=("is_deep", "mean"),
            avg_air_yards=("air_yards", "mean"),
            screen_rate=("is_screen", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off_agg = (
        off_plays.groupby(["season", "posteam"], sort=False)
        .agg(
            n_off_plays=("pass_attempt", "size"),
            proe=("pass_oe", "mean"),
            shotgun_rate=("shotgun", "mean"),
            no_huddle_rate=("no_huddle", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    run_counts = (
        run_plays.groupby(["season", "posteam", "run_location"], sort=False)
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    hhi_rows = []
    for (season, team), group in run_counts.groupby(["season", "team"], sort=False):
        hhi_rows.append(
            {
                "season": season,
                "team": team,
                "run_direction_hhi": _herfindahl(group.set_index("run_location")["n"]),
                "n_run_plays": int(group["n"].sum()),
            }
        )
    hhi_table = pd.DataFrame(hhi_rows)

    pace = _drive_pace_table(raw)
    pace_season = (
        pace.groupby(["season", "team"], sort=False)
        .agg(drive_seconds=("drive_seconds", "sum"), drive_plays=("drive_plays", "sum"))
        .reset_index()
    )
    pace_season["seconds_per_play_pace"] = pace_season["drive_seconds"] / pace_season[
        "drive_plays"
    ].replace(0, np.nan)

    table = pass_agg.merge(off_agg, on=["season", "team"], how="outer")
    table = table.merge(hhi_table, on=["season", "team"], how="outer")
    table = table.merge(
        pace_season[["season", "team", "seconds_per_play_pace"]],
        on=["season", "team"],
        how="outer",
    )
    table["season"] = table["season"].astype(int)
    return table.sort_values(["season", "team"]).reset_index(drop=True)


def build_defense_faced_style(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (season, defteam): the offensive-formation response this defense
    DRAWS from opposing offenses -- a pressure-NEUTRAL proxy for defensive
    "aggression" identity, since sack/pressure rate is quality-laden (a
    genuinely better pass rush produces both more sacks AND more shotgun
    response, confounding style with quality). ``shotgun_rate_faced`` is the
    exact proxy the task predeclaration names as acceptable ("opponent
    shotgun-forced rate"): the rate at which opposing offenses line up in
    shotgun against this defense, centered vs league mean. Same eligibility
    filter (``analysis_plays`` + ``competitive_play``) and same
    ``TEAM_ABBREVIATION_ALIASES`` handling as the offensive-side tables,
    just grouped by ``defteam`` instead of ``posteam``.
    """

    plays = analysis_plays(raw)
    plays = plays.loc[plays["competitive_play"]].copy()
    plays["pass_attempt"] = pd.to_numeric(plays["pass_attempt"], errors="coerce")
    plays["rush_attempt"] = pd.to_numeric(plays["rush_attempt"], errors="coerce")
    plays["shotgun"] = pd.to_numeric(plays["shotgun"], errors="coerce")
    plays["defteam"] = _alias_team(plays["defteam"])

    off_faced = plays.loc[plays["pass_attempt"].eq(1) | plays["rush_attempt"].eq(1)].copy()
    table = (
        off_faced.groupby(["season", "defteam"], sort=False)
        .agg(n_off_plays_faced=("shotgun", "size"), shotgun_rate_faced=("shotgun", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    table["season"] = table["season"].astype(int)
    league_mean = table.groupby("season")["shotgun_rate_faced"].transform("mean")
    table["shotgun_rate_faced_centered"] = table["shotgun_rate_faced"] - league_mean
    return table.sort_values(["season", "team"]).reset_index(drop=True)


def add_league_centered(table: pd.DataFrame, *, season_col: str = "season") -> pd.DataFrame:
    """Center every style dimension against ITS OWN SEASON's unweighted team mean."""

    result = table.copy()
    for dim in STYLE_DIMENSIONS:
        if dim not in result.columns:
            continue
        league_mean = result.groupby(season_col)[dim].transform("mean")
        result[f"{dim}_centered"] = result[dim] - league_mean
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-raw", action="store_true", help="force re-fetch nflverse PBP")
    parser.add_argument("--season-start", type=int, default=SEASON_START)
    parser.add_argument("--season-end", type=int, default=SEASON_END)
    args = parser.parse_args()

    seasons = range(args.season_start, args.season_end + 1)
    raw = build_raw_cache(refresh=args.refresh_raw, seasons=seasons)
    raw = raw.loc[raw["season"].astype(int).between(seasons.start, seasons.stop - 1)].copy()

    print("=== building team-game style table ===")
    team_game = build_team_game_style(raw)
    team_game = add_league_centered(team_game)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_parquet(team_game, TEAM_GAME_PATH)
    print(f"  {len(team_game)} team-games -> {TEAM_GAME_PATH}")

    print("=== building team-season style table ===")
    team_season = build_team_season_style(raw)
    team_season = add_league_centered(team_season)
    atomic_parquet(team_season, TEAM_SEASON_PATH)
    print(f"  {len(team_season)} team-seasons -> {TEAM_SEASON_PATH}")

    print("=== building defense-faced style table (pressure-neutral proxy) ===")
    faced = build_defense_faced_style(raw)
    atomic_parquet(faced, TEAM_SEASON_FACED_PATH)
    print(f"  {len(faced)} team-seasons -> {TEAM_SEASON_FACED_PATH}")

    print("\n=== league-season mean by dimension (2009 vs 2025, era-drift check) ===")
    for dim in STYLE_DIMENSIONS:
        early = team_season.loc[team_season["season"] == seasons.start, dim].mean()
        late = team_season.loc[team_season["season"] == seasons.stop - 1, dim].mean()
        print(f"  {dim:<24} {seasons.start}={early:.4f}  {seasons.stop - 1}={late:.4f}")


if __name__ == "__main__":
    main()
