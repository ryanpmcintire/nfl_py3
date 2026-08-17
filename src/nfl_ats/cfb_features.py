"""Leak-safe college-football game table and pregame team state (XLG-03).

This module derives the CFB-only market-residual benchmark inputs from the
immutable XLG-02 snapshots. It deliberately mirrors the NFL machinery instead
of inventing new conventions:

- **ATS semantics** match ``features.add_ats_outcomes``: ``result`` is the
  home margin, ``spread_line`` is the home team's expected margin (positive =
  home favored), ``ats_margin = result - spread_line``, and ``home_cover`` is
  1/0/NaN with pushes excluded from classification.
- **Market spread aggregation**: the line archive stores one resolved quote
  per game/market/book/side (a close proxy; no observation timestamps). Each
  book's home-oriented spread is the mean of its oriented side rows, the
  per-game ``spread_line`` is the **median across books**, and the table
  records the contributing ``spread_book_count``, the population standard
  deviation ``spread_dispersion``, the median opener where present, and the
  season's ``source_regime``. Team sides are identified without name joins by
  intersecting each season's line abbreviation against the schedule's ESPN
  home/away team ids, with a per-game partner-repair pass; games whose sides
  cannot be identified are excluded and counted, never guessed.
- **Team state** mirrors ``features.build_team_states``: a span-8
  exponentially weighted mean per team over strictly earlier completed games,
  a three-game maturity rule, and explicit offseason regression toward the
  season league mean with retention 0.67 (all NFL values taken verbatim).
  Metrics are EPA/play offense and defense, success rate, explosive rate, and
  pace, built from competitive scrimmage plays (rush/pass, kneels removed,
  win probability between 5% and 95%) of regular-season FBS-vs-FBS games.

Exclusions are logged in the build audit, never silent: non-FBS games,
postseason rows (present in the schedule source only from 2024), incomplete
games, games without an orientable spread, and games without play-by-play.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb import cfb_line_source_regime, cfb_source_spec
from nfl_ats.constants import DEFAULT_OFFSEASON_RETENTION
from nfl_ats.data import DataContractError, require_columns

CFB_FEATURE_VERSION = "v1"

# The regular season spans weeks 1-16 in the schedule source.
CFB_WEEK_PERIOD = 16.0

# BCS automatic-qualifier / power conferences as named by the schedule source.
# Declared once; used only as a coarse conference control, never tuned.
CFB_POWER_CONFERENCES = frozenset(
    {"ACC", "Big 12", "Big East", "Big Ten", "Pac-10", "Pac-12", "SEC"}
)

# Pregame team-state metrics measured per completed game and turned into a
# strictly-lagged exponentially weighted state (NFL span/maturity/offseason
# parameters taken verbatim).
CFB_STATE_METRICS = (
    "off_epa_per_play",
    "off_success_rate",
    "off_explosive_rate",
    "off_plays_per_game",
    "def_epa_per_play",
    "def_success_rate_allowed",
    "def_explosive_rate_allowed",
    "def_plays_per_game_allowed",
)

CFB_MARKET_FEATURES = ("spread_line", "total_line")
CFB_CONTEXT_FEATURES = (
    "neutral_site",
    "conference_game",
    "rest_diff",
    "week_sin",
    "week_cos",
    "home_power5",
    "away_power5",
)
CFB_EXPERIENCE_FEATURES = ("home_team_games", "away_team_games")
CFB_TEAM_STATE_FEATURES = tuple(
    column
    for metric in CFB_STATE_METRICS
    for column in (f"home_{metric}", f"away_{metric}", f"diff_{metric}")
)
# The frozen model feature contract for the CFB market-residual benchmark.
CFB_MODEL_FEATURE_COLUMNS = (
    *CFB_MARKET_FEATURES,
    *CFB_CONTEXT_FEATURES,
    *CFB_EXPERIENCE_FEATURES,
    *CFB_TEAM_STATE_FEATURES,
)

CFB_IDENTIFIER_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "kickoff",
    "home_id",
    "away_id",
    "home_team",
    "away_team",
    "home_conference",
    "away_conference",
    "source_regime",
)
CFB_MARKET_DETAIL_COLUMNS = (
    "spread_line",
    "spread_book_count",
    "spread_dispersion",
    "spread_open",
    "spread_open_book_count",
    "total_line",
    "total_book_count",
    "home_spread_odds",
    "away_spread_odds",
)
CFB_OUTCOME_COLUMNS = (
    "home_points",
    "away_points",
    "result",
    "ats_margin",
    "home_cover",
)

_PBP_LOAD_COLUMNS = (
    "game_id",
    "season",
    "week",
    "seasonType",
    "pos_team_id",
    "homeTeamId",
    "awayTeamId",
    "is_home",
    "EPA",
    "EPA_success",
    "rush",
    "pass",
    "kneel_down",
    "statYardage",
    "home_wp_before",
    "away_wp_before",
)

_SCHEDULE_LOAD_COLUMNS = (
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "completed",
    "neutral_site",
    "conference_game",
    "home_id",
    "home_team",
    "home_division",
    "home_conference",
    "home_points",
    "away_id",
    "away_team",
    "away_division",
    "away_conference",
    "away_points",
)

_LINE_LOAD_COLUMNS = (
    "game_id",
    "season",
    "market_type",
    "abbr",
    "book",
    "lines",
    "odds",
    "opening_lines",
)


# ---------------------------------------------------------------------------
# Snapshot loading (play-by-play spans multiple snapshots)
# ---------------------------------------------------------------------------


def cfb_season_partitions(cfb_root: Path, source: str) -> dict[int, Path]:
    """Map each ingested season to its newest snapshot partition."""

    spec = cfb_source_spec(source)
    partitions: dict[int, Path] = {}
    for manifest_path in sorted((cfb_root / spec.key / "raw").glob("*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for season in payload["seasons"]:
            partitions[int(season)] = (
                manifest_path.parent / f"season={int(season)}" / spec.partition_filename
            )
    if not partitions:
        raise FileNotFoundError(f"No CFB {spec.key} snapshots found in {cfb_root}")
    return partitions


def load_cfb_seasons(
    cfb_root: Path,
    source: str,
    seasons: list[int],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load season partitions across snapshots, newest snapshot per season."""

    partitions = cfb_season_partitions(cfb_root, source)
    missing = sorted(season for season in seasons if season not in partitions)
    if missing:
        raise FileNotFoundError(f"CFB {source} has no ingested partitions for seasons: {missing}")
    frames = [
        pd.read_parquet(partitions[season], columns=columns) for season in sorted(set(seasons))
    ]
    return pd.concat(frames, ignore_index=True)


def load_cfb_benchmark_inputs(
    cfb_root: Path, start_season: int, end_season: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the schedules, lines, and play-by-play slices the benchmark needs."""

    seasons = list(range(start_season, end_season + 1))
    schedules = load_cfb_seasons(
        cfb_root, "schedules", seasons, columns=list(_SCHEDULE_LOAD_COLUMNS)
    )
    lines = load_cfb_seasons(cfb_root, "lines", seasons, columns=list(_LINE_LOAD_COLUMNS))
    pbp = load_cfb_seasons(cfb_root, "pbp", seasons, columns=list(_PBP_LOAD_COLUMNS))
    return schedules, lines, pbp


# ---------------------------------------------------------------------------
# Schedule normalization
# ---------------------------------------------------------------------------


def _filtered_schedule(
    schedules: pd.DataFrame, start_season: int, end_season: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    require_columns(schedules, _SCHEDULE_LOAD_COLUMNS, "cfb_schedules")
    frame = schedules.copy()
    for column in ("game_id", "season", "week"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame = frame.loc[frame["season"].between(start_season, end_season)].copy()
    in_window = len(frame)

    regular = frame["season_type"].astype(str).eq("regular")
    completed = frame["completed"].astype(bool)
    fbs = frame["home_division"].astype("string").str.lower().eq("fbs") & frame[
        "away_division"
    ].astype("string").str.lower().eq("fbs")
    keep = regular & completed & fbs
    audit = {
        "schedule_rows_in_window": in_window,
        "excluded_postseason_rows": int((~regular).sum()),
        "excluded_incomplete_rows": int((regular & ~completed).sum()),
        "excluded_non_fbs_rows": int((regular & completed & ~fbs).sum()),
        "fbs_regular_completed_games": int(keep.sum()),
    }
    result = frame.loc[keep].copy()
    if result["game_id"].duplicated().any():
        raise DataContractError("cfb_schedules contains duplicate game_id rows in the window")
    kickoff = pd.to_datetime(result["start_date"], errors="raise", utc=True, format="ISO8601")
    result["kickoff"] = kickoff
    result["gameday"] = kickoff.dt.tz_localize(None).dt.normalize()
    for column in ("home_id", "away_id", "home_points", "away_points"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    result["neutral_site"] = result["neutral_site"].astype(bool).astype(int)
    result["conference_game"] = result["conference_game"].astype(bool).astype(int)
    for column in ("home_conference", "away_conference"):
        result[column] = result[column].astype("string")
    return result.sort_values(["gameday", "game_id"]).reset_index(drop=True), audit


# ---------------------------------------------------------------------------
# Market spread orientation and aggregation
# ---------------------------------------------------------------------------


def _season_abbr_map(pairs: pd.DataFrame) -> dict[str, int]:
    """Resolve line abbreviations to ESPN team ids for one season.

    Each spread row names a team only by the archive's abbreviation. Within a
    season an abbreviation's team id must appear in every game it quotes, so
    intersecting the home/away id pairs across its games identifies it without
    any name join. Abbreviations whose intersection is empty (two teams share
    the label) or not unique (too few games) stay unresolved here and are
    handled by the per-game partner repair.
    """

    resolved: dict[str, int] = {}
    for abbr, group in pairs.groupby("abbr", sort=False):
        candidates: set[int] | None = None
        for home_id, away_id in zip(group["home_id"], group["away_id"], strict=True):
            game_teams = {int(home_id), int(away_id)}
            candidates = game_teams if candidates is None else candidates & game_teams
            if not candidates:
                break
        if candidates and len(candidates) == 1:
            resolved[str(abbr)] = next(iter(candidates))
    return resolved


def _repair_game_sides(
    game_rows: pd.DataFrame,
) -> tuple[dict[str, str] | None, str]:
    """Assign home/away to a game's abbreviations, or name the exclusion."""

    per_abbr = game_rows.drop_duplicates("abbr")
    abbrs = per_abbr["abbr"].astype(str).tolist()
    if len(abbrs) > 2:
        return None, "spread_games_excluded_gt2_abbrs"
    known = {
        str(row.abbr): str(row.side) for row in per_abbr.itertuples() if isinstance(row.side, str)
    }
    if len(abbrs) == 2:
        first, second = abbrs
        if first in known and second in known:
            if known[first] == known[second]:
                return None, "spread_games_excluded_conflicting_sides"
            return known, ""
        if first in known:
            return {first: known[first], second: "away" if known[first] == "home" else "home"}, ""
        if second in known:
            return {second: known[second], first: "away" if known[second] == "home" else "home"}, ""
        return None, "spread_games_excluded_unresolved"
    if abbrs and abbrs[0] in known:
        return known, ""
    return None, "spread_games_excluded_unresolved"


def _oriented_spread_rows(
    spread: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach a home/away side to every spread row, excluding unresolvable games."""

    working = spread.copy()
    sides: dict[str, str] = {}
    for _, season_pairs in (
        working[["season", "game_id", "abbr", "home_id", "away_id"]]
        .drop_duplicates()
        .groupby("season", sort=False)
    ):
        resolved = _season_abbr_map(season_pairs)
        for game_value, abbr_value, home_value, away_value in zip(
            season_pairs["game_id"],
            season_pairs["abbr"],
            season_pairs["home_id"],
            season_pairs["away_id"],
            strict=True,
        ):
            team_id = resolved.get(str(abbr_value))
            if team_id == int(home_value):
                sides[f"{game_value}|{abbr_value}"] = "home"
            elif team_id == int(away_value):
                sides[f"{game_value}|{abbr_value}"] = "away"

    keys = working["game_id"].astype(str) + "|" + working["abbr"].astype(str)
    working["side"] = keys.map(sides)

    audit = {
        "spread_games_excluded_gt2_abbrs": 0,
        "spread_games_excluded_conflicting_sides": 0,
        "spread_games_excluded_unresolved": 0,
    }
    needs_repair = working.loc[working["side"].isna(), "game_id"].unique()
    ambiguous = working.groupby("game_id")["abbr"].nunique()
    multi_abbr = set(ambiguous.loc[ambiguous > 2].index)
    excluded_games: set[int] = set()
    for game_id in set(needs_repair) | multi_abbr:
        game_rows = working.loc[working["game_id"].eq(game_id)]
        repaired, reason = _repair_game_sides(game_rows)
        if repaired is None:
            audit[reason] += 1
            excluded_games.add(int(game_id))
            continue
        mask = working["game_id"].eq(game_id)
        working.loc[mask, "side"] = working.loc[mask, "abbr"].astype(str).map(repaired)
    result = working.loc[~working["game_id"].isin(excluded_games) & working["side"].notna()].copy()
    return result, audit


def build_cfb_market_table(
    lines: pd.DataFrame, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate the multi-book line archive to one market row per game.

    ``spread_line`` is the median across books of each book's home-oriented
    close-proxy spread; ``spread_open`` aggregates the opener column the same
    way where present. ``home_spread_odds``/``away_spread_odds`` are the
    median American prices across the oriented side rows and feed only the
    market baseline's no-vig probability, never the residual model.
    """

    require_columns(lines, _LINE_LOAD_COLUMNS, "cfb_line_odds")
    require_columns(schedule, ("game_id", "season", "home_id", "away_id"), "cfb schedule table")
    working = lines.copy()
    working["game_id"] = pd.to_numeric(working["game_id"], errors="raise").astype("int64")
    working["season"] = pd.to_numeric(working["season"], errors="raise").astype("int64")
    for column in ("lines", "odds", "opening_lines"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["abbr"] = working["abbr"].astype("string")
    working = working.merge(
        schedule[["game_id", "home_id", "away_id"]],
        on="game_id",
        how="inner",
        validate="many_to_one",
    )

    spread = working.loc[
        working["market_type"].astype(str).eq("spread")
        & (working["lines"].notna() | working["opening_lines"].notna())
    ].copy()
    spread_rows_in = len(spread)
    oriented, orientation_audit = _oriented_spread_rows(spread)
    oriented["home_oriented_close"] = np.where(
        oriented["side"].eq("home"), -oriented["lines"], oriented["lines"]
    )
    oriented["home_oriented_open"] = np.where(
        oriented["side"].eq("home"), -oriented["opening_lines"], oriented["opening_lines"]
    )

    book_level = (
        oriented.groupby(["game_id", "season", "book"], sort=False)
        .agg(
            close=("home_oriented_close", "mean"),
            open=("home_oriented_open", "mean"),
        )
        .reset_index()
    )
    close_level = book_level.loc[book_level["close"].notna()]
    spread_summary = (
        close_level.groupby(["game_id", "season"], sort=False)["close"]
        .agg(
            spread_line="median",
            spread_book_count="size",
            spread_dispersion=lambda values: float(np.std(values, ddof=0)),
        )
        .reset_index()
    )
    open_level = book_level.loc[book_level["open"].notna()]
    open_summary = (
        open_level.groupby(["game_id", "season"], sort=False)["open"]
        .agg(spread_open="median", spread_open_book_count="size")
        .reset_index()
    )

    odds_rows = oriented.loc[oriented["odds"].notna()]
    if odds_rows.empty:
        odds_summary = pd.DataFrame(
            {
                "game_id": pd.Series(dtype="int64"),
                "season": pd.Series(dtype="int64"),
                "home_spread_odds": pd.Series(dtype="float64"),
                "away_spread_odds": pd.Series(dtype="float64"),
            }
        )
    else:
        odds_summary = (
            odds_rows.pivot_table(
                index=["game_id", "season"],
                columns="side",
                values="odds",
                aggfunc="median",
                observed=True,
            )
            .rename(columns={"home": "home_spread_odds", "away": "away_spread_odds"})
            .reset_index()
        )
        for column in ("home_spread_odds", "away_spread_odds"):
            if column not in odds_summary:
                odds_summary[column] = np.nan
        odds_summary = odds_summary[["game_id", "season", "home_spread_odds", "away_spread_odds"]]

    totals = working.loc[working["market_type"].astype(str).eq("total") & working["lines"].notna()]
    total_books = (
        totals.groupby(["game_id", "season", "book"], sort=False)["lines"].mean().reset_index()
    )
    total_summary = (
        total_books.groupby(["game_id", "season"], sort=False)["lines"]
        .agg(total_line="median", total_book_count="size")
        .reset_index()
    )

    market = spread_summary.merge(open_summary, on=["game_id", "season"], how="left")
    market = market.merge(odds_summary, on=["game_id", "season"], how="left")
    market = market.merge(total_summary, on=["game_id", "season"], how="left")
    market["spread_open_book_count"] = market["spread_open_book_count"].fillna(0).astype("int64")
    market["total_book_count"] = market["total_book_count"].fillna(0).astype("int64")
    market["source_regime"] = market["season"].map(
        lambda season: cfb_line_source_regime(int(season))
    )

    two_sided = (
        oriented.loc[oriented["lines"].notna()]
        .groupby(["game_id", "book"], sort=False)["home_oriented_close"]
        .agg(["min", "max", "size"])
    )
    two_sided = two_sided.loc[two_sided["size"].ge(2)]
    audit: dict[str, Any] = {
        "spread_rows_in": spread_rows_in,
        "spread_rows_oriented": len(oriented),
        **orientation_audit,
        "games_with_close_proxy_spread": len(market),
        "two_sided_book_quotes": len(two_sided),
        "mean_two_sided_asymmetry": (
            float((two_sided["max"] - two_sided["min"]).abs().mean()) if len(two_sided) else 0.0
        ),
    }
    return market, audit


# ---------------------------------------------------------------------------
# Team-game metrics from play-by-play
# ---------------------------------------------------------------------------


def cfb_competitive_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented CFB play filter, mirroring the NFL v1 filter.

    Keeps regular-season scrimmage plays (rush or pass) with a finite EPA,
    removes kneels, and marks the 5%-95% possession-team win-probability
    subset used for state aggregation.
    """

    require_columns(pbp, _PBP_LOAD_COLUMNS, "cfb play_by_play")
    result = pbp.copy()
    for column in ("EPA", "home_wp_before", "away_wp_before", "statYardage"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("rush", "pass", "kneel_down", "is_home", "EPA_success"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(bool)
    regular = pd.to_numeric(result["seasonType"], errors="coerce").eq(2)
    eligible = (
        regular
        & (result["rush"] | result["pass"])
        & result["EPA"].notna()
        & ~result["kneel_down"]
        & result["pos_team_id"].notna()
    )
    result = result.loc[eligible].copy()
    pos_wp = np.where(result["is_home"], result["home_wp_before"], result["away_wp_before"])
    result["competitive_play"] = pd.Series(pos_wp, index=result.index).between(
        0.05, 0.95, inclusive="both"
    )
    return result.reset_index(drop=True)


def build_cfb_team_game_metrics(pbp: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build offense and opponent-derived defense metrics per completed game."""

    plays = cfb_competitive_plays(pbp)
    plays = plays.loc[plays["competitive_play"]].copy()
    empty_columns = [
        "game_id",
        "season",
        "week",
        "team_id",
        "opponent_id",
        *CFB_STATE_METRICS,
    ]
    if plays.empty:
        return pd.DataFrame(columns=empty_columns), {
            "team_games": 0,
            "unpaired_games_dropped": 0,
        }
    plays["explosive"] = (
        (plays["rush"] & plays["statYardage"].ge(10))
        | (plays["pass"] & plays["statYardage"].ge(20))
    ).astype(float)
    plays["team_id"] = pd.to_numeric(plays["pos_team_id"], errors="raise").astype("int64")

    offense = (
        plays.groupby(["game_id", "team_id"], sort=False)
        .agg(
            season=("season", "first"),
            week=("week", "first"),
            off_epa_per_play=("EPA", "mean"),
            off_success_rate=("EPA_success", "mean"),
            off_explosive_rate=("explosive", "mean"),
            off_plays_per_game=("EPA", "size"),
        )
        .reset_index()
    )
    opponent = offense[
        [
            "game_id",
            "team_id",
            "off_epa_per_play",
            "off_success_rate",
            "off_explosive_rate",
            "off_plays_per_game",
        ]
    ].rename(
        columns={
            "team_id": "opponent_id",
            "off_epa_per_play": "def_epa_per_play",
            "off_success_rate": "def_success_rate_allowed",
            "off_explosive_rate": "def_explosive_rate_allowed",
            "off_plays_per_game": "def_plays_per_game_allowed",
        }
    )
    paired = offense.merge(opponent, on="game_id", how="inner", validate="many_to_many")
    paired = paired.loc[paired["team_id"].ne(paired["opponent_id"])].copy()
    counts = paired.groupby("game_id")["team_id"].size()
    well_formed = set(counts.loc[counts.eq(2)].index)
    unpaired = int(offense["game_id"].nunique() - len(well_formed))
    paired = paired.loc[paired["game_id"].isin(well_formed)].copy()
    paired["off_plays_per_game"] = paired["off_plays_per_game"].astype(float)
    paired["def_plays_per_game_allowed"] = paired["def_plays_per_game_allowed"].astype(float)
    result = paired[empty_columns].sort_values(["season", "week", "game_id", "team_id"])
    return result.reset_index(drop=True), {
        "team_games": len(result),
        "unpaired_games_dropped": unpaired,
    }


# ---------------------------------------------------------------------------
# Exponentially weighted pregame state (NFL parameters verbatim)
# ---------------------------------------------------------------------------


def build_cfb_team_states(
    team_games: pd.DataFrame,
    span: int = 8,
    min_periods: int = 3,
    offseason_retention: float = DEFAULT_OFFSEASON_RETENTION,
) -> pd.DataFrame:
    """Calculate the state after each completed game, exactly as the NFL does.

    Rows deliberately include the game on the row; ``attach_cfb_team_states``
    uses a strictly-earlier lookup so a state is available only to the team's
    next game.
    """

    if span < 2:
        raise ValueError("span must be at least 2")
    if min_periods < 1:
        raise ValueError("min_periods must be positive")
    if not 0.0 <= offseason_retention <= 1.0:
        raise ValueError("offseason_retention must be between 0 and 1")
    required = {"game_id", "season", "gameday", "team_id", *CFB_STATE_METRICS}
    missing = sorted(required.difference(team_games.columns))
    if missing:
        raise DataContractError(f"cfb team games missing columns: {', '.join(missing)}")

    states = team_games.copy().sort_values(["team_id", "gameday", "game_id"])
    states["season"] = pd.to_numeric(states["season"], errors="raise").astype(int)
    alpha = 2.0 / (span + 1.0)
    for metric in CFB_STATE_METRICS:
        states[metric] = pd.to_numeric(states[metric], errors="coerce")
        league_means = states.groupby("season", sort=False)[metric].mean().to_dict()
        states[f"league_mean_{metric}"] = states["season"].map(league_means)
        output = pd.Series(np.nan, index=states.index, dtype="float64")
        for _, group in states.groupby("team_id", sort=False):
            current = math.nan
            observations = 0
            previous_season: int | None = None
            for index, row in group.iterrows():
                season = int(row["season"])
                if (
                    previous_season is not None
                    and season != previous_season
                    and math.isfinite(current)
                ):
                    league_mean = float(league_means.get(previous_season, 0.0))
                    retention = offseason_retention ** max(1, season - previous_season)
                    current = league_mean + retention * (current - league_mean)
                value = float(row[metric])
                if math.isfinite(value):
                    current = (
                        value
                        if not math.isfinite(current)
                        else alpha * value + (1.0 - alpha) * current
                    )
                    observations += 1
                if observations >= min_periods:
                    output.at[index] = current
                previous_season = season
        states[f"state_{metric}"] = output
    states["team_games"] = states.groupby(["team_id", "season"], sort=False).cumcount() + 1
    return states[
        [
            "game_id",
            "season",
            "gameday",
            "team_id",
            "team_games",
            *[f"state_{metric}" for metric in CFB_STATE_METRICS],
            *[f"league_mean_{metric}" for metric in CFB_STATE_METRICS],
        ]
    ]


def attach_cfb_team_states(
    games: pd.DataFrame,
    states: pd.DataFrame,
    offseason_retention: float = DEFAULT_OFFSEASON_RETENTION,
) -> pd.DataFrame:
    """Attach the most recent state strictly before each game's date."""

    result = games.copy()
    state_columns = [f"state_{metric}" for metric in CFB_STATE_METRICS]
    groups: dict[int, pd.DataFrame] = {
        int(group["team_id"].iloc[0]): group.sort_values(["gameday", "game_id"]).reset_index(
            drop=True
        )
        for _, group in states.groupby("team_id", sort=False)
    }
    for side in ("home", "away"):
        for metric in CFB_STATE_METRICS:
            result[f"{side}_{metric}"] = np.nan
        result[f"{side}_team_games"] = np.nan
        for index, game in result.iterrows():
            history = groups.get(int(game[f"{side}_id"]))
            if history is None or history.empty:
                continue
            dates = history["gameday"].to_numpy(dtype="datetime64[ns]")
            game_date = np.datetime64(pd.Timestamp(game["gameday"]), "ns")
            position = int(np.searchsorted(dates, game_date, side="left")) - 1
            if position < 0:
                continue
            state = history.iloc[position]
            season_gap = int(game["season"]) - int(state["season"])
            result.at[index, f"{side}_team_games"] = state["team_games"] if season_gap == 0 else 0
            for metric, state_column in zip(CFB_STATE_METRICS, state_columns, strict=True):
                value = state[state_column]
                if season_gap > 0 and pd.notna(value):
                    league_mean = state[f"league_mean_{metric}"]
                    retention = offseason_retention ** max(1, season_gap)
                    value = league_mean + retention * (value - league_mean)
                result.at[index, f"{side}_{metric}"] = value
    for metric in CFB_STATE_METRICS:
        result[f"diff_{metric}"] = result[f"home_{metric}"] - result[f"away_{metric}"]
    return result


# ---------------------------------------------------------------------------
# Rest and assembly
# ---------------------------------------------------------------------------


def _rest_base_schedule(
    schedules: pd.DataFrame, start_season: int, end_season: int
) -> pd.DataFrame:
    """All completed regular-season appearances (any division) with a date."""

    frame = schedules.copy()
    for column in ("season", "home_id", "away_id"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    keep = (
        frame["season"].between(start_season, end_season)
        & frame["season_type"].astype(str).eq("regular")
        & frame["completed"].astype(bool)
        & frame["home_id"].notna()
        & frame["away_id"].notna()
    )
    frame = frame.loc[keep].copy()
    kickoff = pd.to_datetime(frame["start_date"], errors="raise", utc=True, format="ISO8601")
    frame["gameday"] = kickoff.dt.tz_localize(None).dt.normalize()
    frame["season"] = frame["season"].astype("int64")
    for column in ("home_id", "away_id"):
        frame[column] = frame[column].astype("int64")
    return frame[["season", "gameday", "home_id", "away_id"]]


def _add_rest_features(games: pd.DataFrame, full_schedule: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game this season, from the full schedule.

    Rest uses every completed regular-season game the team played (including
    against non-FBS opponents) because only dates are consumed - no outcomes.
    """

    appearances = pd.concat(
        [
            full_schedule[["season", "gameday", "home_id"]].rename(columns={"home_id": "team_id"}),
            full_schedule[["season", "gameday", "away_id"]].rename(columns={"away_id": "team_id"}),
        ],
        ignore_index=True,
    ).sort_values(["team_id", "gameday"])
    appearances["previous_gameday"] = appearances.groupby(["team_id", "season"], sort=False)[
        "gameday"
    ].shift(1)
    rest_lookup = {
        (int(team_value), gameday_value): previous_value
        for team_value, gameday_value, previous_value in zip(
            appearances["team_id"],
            appearances["gameday"],
            appearances["previous_gameday"],
            strict=True,
        )
    }

    result = games.copy()
    for side in ("home", "away"):
        previous = [
            rest_lookup.get((int(row[f"{side}_id"]), row["gameday"]), pd.NaT)
            for _, row in result.iterrows()
        ]
        previous_series = pd.Series(previous, index=result.index, dtype="datetime64[ns]")
        delta = result["gameday"] - previous_series
        result[f"{side}_rest"] = delta.dt.days.astype("float64")
    result["rest_diff"] = result["home_rest"] - result["away_rest"]
    return result


def build_cfb_game_features(
    schedules: pd.DataFrame,
    lines: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    start_season: int = 2006,
    end_season: int = 2025,
    span: int = 8,
    min_periods: int = 3,
    offseason_retention: float = DEFAULT_OFFSEASON_RETENTION,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the canonical CFB benchmark table with one row per eligible game.

    Eligible games are completed regular-season FBS-vs-FBS games in the season
    window that carry both an orientable close-proxy spread and play-by-play.
    Every other exclusion is counted in the returned audit.
    """

    all_schedule, schedule_audit = _filtered_schedule(schedules, start_season, end_season)
    rest_base = _rest_base_schedule(schedules, start_season, end_season)
    market, market_audit = build_cfb_market_table(lines, all_schedule)
    team_games, metric_audit = build_cfb_team_game_metrics(pbp)

    pbp_game_ids = set(
        pd.to_numeric(pbp.loc[pd.to_numeric(pbp["seasonType"], errors="coerce").eq(2), "game_id"])
        .dropna()
        .astype("int64")
        .unique()
    )
    games = all_schedule.merge(
        market.drop(columns="season"), on="game_id", how="left", validate="one_to_one"
    )
    has_spread = games["spread_line"].notna()
    has_pbp = games["game_id"].isin(pbp_game_ids)
    audit: dict[str, Any] = {
        **schedule_audit,
        **market_audit,
        **metric_audit,
        "excluded_games_without_spread": int((~has_spread & has_pbp).sum()),
        "excluded_games_without_pbp": int((has_spread & ~has_pbp).sum()),
        "excluded_games_without_spread_or_pbp": int((~has_spread & ~has_pbp).sum()),
    }
    games = games.loc[has_spread & has_pbp].copy()
    audit["canonical_games"] = len(games)
    audit["games_per_season"] = {
        str(season): int(count) for season, count in games.groupby("season").size().items()
    }

    games["result"] = (games["home_points"] - games["away_points"]).astype(float)
    games["ats_margin"] = games["result"] - games["spread_line"]
    games["home_cover"] = np.select(
        [games["ats_margin"] > 0, games["ats_margin"] < 0], [1.0, 0.0], default=np.nan
    )

    games["week_sin"] = np.sin(2.0 * np.pi * games["week"] / CFB_WEEK_PERIOD)
    games["week_cos"] = np.cos(2.0 * np.pi * games["week"] / CFB_WEEK_PERIOD)
    games["home_power5"] = games["home_conference"].isin(sorted(CFB_POWER_CONFERENCES)).astype(int)
    games["away_power5"] = games["away_conference"].isin(sorted(CFB_POWER_CONFERENCES)).astype(int)
    games = _add_rest_features(games, rest_base)

    state_input = team_games.loc[team_games["game_id"].isin(all_schedule["game_id"])].copy()
    game_dates = all_schedule[["game_id", "gameday"]]
    state_input = state_input.merge(game_dates, on="game_id", how="left", validate="many_to_one")
    if state_input["gameday"].isna().any():
        raise DataContractError("CFB team games contain games absent from the schedule window")
    states = build_cfb_team_states(
        state_input,
        span=span,
        min_periods=min_periods,
        offseason_retention=offseason_retention,
    )
    games = attach_cfb_team_states(games, states, offseason_retention=offseason_retention)
    audit["state_input_team_games"] = len(state_input)

    for column in CFB_MODEL_FEATURE_COLUMNS:
        if column not in games:
            games[column] = np.nan
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games = games.replace([np.inf, -np.inf], np.nan)
    games["cfb_feature_version"] = CFB_FEATURE_VERSION

    ordered = list(
        dict.fromkeys(
            [
                *CFB_IDENTIFIER_COLUMNS,
                *CFB_MARKET_DETAIL_COLUMNS,
                *CFB_OUTCOME_COLUMNS,
                *CFB_MODEL_FEATURE_COLUMNS,
                "cfb_feature_version",
            ]
        )
    )
    result = games[ordered].sort_values(["gameday", "game_id"]).reset_index(drop=True)
    audit["feature_columns"] = list(CFB_MODEL_FEATURE_COLUMNS)
    return result, audit
