"""Immutable college-football source ingestion for cross-league research.

XLG-02 phase 1 ingests the no-key bulk sources selected by the XLG-01
feasibility audit: sportsdataverse-data GitHub release assets (ESPN CFB
play-by-play, game rosters, play participants, and the per-season betting
cross-check) plus cfbfastR-data raw files (CFBD-flavor schedules and the
multi-book line-odds archive). Upstream assets are rebuilt in place, so every
refresh writes an immutable snapshot holding the verbatim downloaded bytes,
SHA-256 hashes of raw and canonical files, and the upstream release-asset
timestamps or pinned git commit that produced them.

XLG-02 phase 3 adds the CollegeFootballData API gap-fillers that no bulk
archive carries: NFL draft picks with the collegeAthleteId/nflAthleteId
crosswalk, returning production, team and player recruiting, player usage,
and the transfer portal. Every CFBD call is authenticated with the
CFBD_API_KEY environment variable (free tier: 1,000 calls per month), the
verbatim JSON response bytes are snapshotted next to the canonical parquet
partitions, and the manifest records the endpoint, params, and API version
behind each partition. The key itself is never written anywhere.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet, run_id

CFB_CONTRACT_VERSION = "v1"

GITHUB_API_ROOT = "https://api.github.com"
RAW_CONTENT_ROOT = "https://raw.githubusercontent.com"
CFBFASTR_DATA_REPOSITORY = "sportsdataverse/cfbfastR-data"
CFBFASTR_DATA_BRANCH = "main"
SPORTSDATAVERSE_DATA_REPOSITORY = "sportsdataverse/sportsdataverse-data"

# Licensing from the XLG-01 audit (2026-08-16): cfbfastR-data declares
# CC BY 4.0 and sportsdataverse-data is MIT, but both archives reshape
# ESPN/CFBD data whose upstream rights the maintainers cannot enlarge.
CFB_REDISTRIBUTION_RULE = (
    "Private research retention only. Raw CFB source tables are never "
    "republished from this repository; publish derived aggregates with "
    "attribution instead."
)
CFBFASTR_LICENSE = "CC BY 4.0 (repository DESCRIPTION); ESPN/CFBD-derived rows"
SPORTSDATAVERSE_LICENSE = "MIT (repository); ESPN-derived rows"

# CollegeFootballData API (XLG-01 audit, terms fetched 2026-08-16): commercial
# use and private caching/retention are permitted, including after access
# ends; republishing raw API data as a dataset, mirror, or proxy is prohibited.
CFBD_API_ROOT = "https://api.collegefootballdata.com"
CFBD_API_DOCS_URL = "https://apinext.collegefootballdata.com/api-docs.json"
CFBD_API_KEY_ENV = "CFBD_API_KEY"
CFBD_LICENSE = (
    "CollegeFootballData.com terms: commercial use and private "
    "caching/retention permitted; republishing raw data prohibited"
)
# Candidate response headers that could carry the remaining monthly quota;
# whichever are present on the last response are recorded in the manifest.
# Observed live 2026-08-16: X-CallLimit-Remaining counts down from 1,000 and
# HTTP 429 burst rejections do NOT consume it.
CFBD_QUOTA_HEADERS = (
    "X-CallLimit-Remaining",
    "X-RateLimit-Remaining",
    "RateLimit-Remaining",
    "X-Api-Calls-Remaining",
)
# CFBD also enforces an undocumented burst rate limit (HTTP 429 observed after
# ~39 back-to-back calls on 2026-08-16). Calls are spaced politely and a 429
# is retried after increasing waits; 429 responses do not spend monthly quota.
CFBD_THROTTLE_SECONDS = 1.2
CFBD_RETRY_WAITS = (15.0, 45.0)

_sleep = time.sleep  # indirection so tests can observe or suppress waits


@dataclass(frozen=True)
class CfbSourceSpec:
    """Where one CFB dataset lives upstream and which seasons are ingestable."""

    key: str
    dataset: str
    partition_filename: str
    origin: str  # "cfbfastr_raw", "sportsdataverse_release", or "cfbd_api"
    first_season: int
    default_start_season: int
    release_tag: str | None = None
    asset_prefix: str | None = None
    raw_directory: str | None = None
    raw_name_pattern: str | None = None
    refused_seasons: frozenset[int] = frozenset()
    refusal_reason: str | None = None
    api_path: str | None = None  # cfbd_api origin only
    season_param: str = "year"  # query parameter naming the season
    season_column: str = "season"  # response field carrying the season
    single_call: bool = False  # one unfiltered call, partitioned locally


CFB_SOURCES: dict[str, CfbSourceSpec] = {
    "schedules": CfbSourceSpec(
        key="schedules",
        dataset="CFBD-flavor CFB schedules (cfbfastR-data raw files)",
        partition_filename="schedules.parquet",
        origin="cfbfastr_raw",
        first_season=2001,
        default_start_season=2001,
        raw_directory="schedules/parquet",
        raw_name_pattern="cfb_schedules_{season}.parquet",
    ),
    "lines": CfbSourceSpec(
        key="lines",
        dataset="multi-book CFB betting line archive (cfbfastR-data raw files)",
        partition_filename="lines.parquet",
        origin="cfbfastr_raw",
        first_season=2006,
        default_start_season=2006,
        raw_directory="betting/parquet",
        raw_name_pattern="cfb_line_odds.parquet",
    ),
    "pbp": CfbSourceSpec(
        key="pbp",
        dataset="ESPN CFB play-by-play (sportsdataverse-data release assets)",
        partition_filename="plays.parquet",
        origin="sportsdataverse_release",
        first_season=2004,
        default_start_season=2004,
        release_tag="espn_cfb_pbp",
        asset_prefix="play_by_play",
    ),
    "rosters": CfbSourceSpec(
        key="rosters",
        dataset="ESPN CFB game rosters (sportsdataverse-data release assets)",
        partition_filename="rosters.parquet",
        origin="sportsdataverse_release",
        first_season=2004,
        default_start_season=2004,
        release_tag="espn_cfb_game_rosters",
        asset_prefix="game_rosters",
    ),
    "participants": CfbSourceSpec(
        key="participants",
        dataset="ESPN CFB play participants (sportsdataverse-data release assets)",
        partition_filename="participants.parquet",
        origin="sportsdataverse_release",
        first_season=2014,
        default_start_season=2014,
        release_tag="espn_cfb_play_participants",
        asset_prefix="play_participants",
    ),
    "espn_betting": CfbSourceSpec(
        key="espn_betting",
        dataset="ESPN CFB single-line betting cross-check (sportsdataverse-data)",
        partition_filename="betting.parquet",
        origin="sportsdataverse_release",
        first_season=2004,
        default_start_season=2012,
        release_tag="espn_cfb_betting",
        asset_prefix="betting",
        refused_seasons=frozenset(range(2004, 2012)),
        refusal_reason=(
            "the 2004-2011 espn_cfb_betting assets contain placeholder defaults "
            "(constant spread 2.5/total 55.5, odds_source='default', "
            "game_spread_available=False on every row) and were rejected by the "
            "XLG-01 audit"
        ),
    ),
    "draft_picks": CfbSourceSpec(
        key="draft_picks",
        dataset="NFL draft picks with college/NFL ESPN athlete-id crosswalk (CFBD API)",
        partition_filename="draft_picks.parquet",
        origin="cfbd_api",
        first_season=1967,
        default_start_season=2004,
        api_path="/draft/picks",
        season_column="year",
        single_call=True,
    ),
    "returning_production": CfbSourceSpec(
        key="returning_production",
        dataset="team-season returning PPA/usage production shares (CFBD API)",
        partition_filename="returning_production.parquet",
        origin="cfbd_api",
        first_season=2014,
        default_start_season=2014,
        api_path="/player/returning",
    ),
    "recruiting_teams": CfbSourceSpec(
        key="recruiting_teams",
        dataset="team recruiting class rankings (CFBD API)",
        partition_filename="recruiting_teams.parquet",
        origin="cfbd_api",
        first_season=2002,
        default_start_season=2002,
        api_path="/recruiting/teams",
        season_column="year",
    ),
    "recruiting_players": CfbSourceSpec(
        key="recruiting_players",
        dataset="player recruiting rankings by class year (CFBD API)",
        partition_filename="recruiting_players.parquet",
        origin="cfbd_api",
        first_season=2000,
        default_start_season=2000,
        api_path="/recruiting/players",
        season_column="year",
    ),
    "usage": CfbSourceSpec(
        key="usage",
        dataset="player-season usage shares by down and play type (CFBD API)",
        partition_filename="usage.parquet",
        origin="cfbd_api",
        first_season=2013,
        default_start_season=2013,
        api_path="/player/usage",
    ),
    "portal": CfbSourceSpec(
        key="portal",
        dataset="transfer portal entries by season (CFBD API)",
        partition_filename="portal.parquet",
        origin="cfbd_api",
        first_season=2021,
        default_start_season=2021,
        api_path="/player/portal",
    ),
}


def cfb_source_spec(source: str) -> CfbSourceSpec:
    key = source.replace("-", "_")
    if key not in CFB_SOURCES:
        raise ValueError(f"Unknown CFB source {source!r}; expected one of {sorted(CFB_SOURCES)}")
    return CFB_SOURCES[key]


# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------

CFB_SCHEDULE_REQUIRED_COLUMNS = (
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
CFB_SCHEDULE_SNAPSHOT_COLUMNS = (
    *CFB_SCHEDULE_REQUIRED_COLUMNS,
    "start_time_tbd",
    "attendance",
    "venue_id",
    "venue",
    "home_pregame_elo",
    "home_postgame_elo",
    "home_post_win_prob",
    "away_pregame_elo",
    "away_postgame_elo",
    "away_post_win_prob",
    "excitement_index",
    "notes",
)

CFB_LINE_REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "date_time",
    "market_type",
    "abbr",
    "lines",
    "odds",
    "opening_lines",
    "opening_odds",
    "book",
    "season_type",
    "week",
    "home_team_id",
    "away_team_id",
)
CFB_LINE_SNAPSHOT_COLUMNS = (
    "game_id",
    "season",
    "source_regime",
    "week",
    "season_type",
    "date_time",
    "market_type",
    "abbr",
    "book",
    "lines",
    "odds",
    "opening_lines",
    "opening_odds",
    "home_team_id",
    "away_team_id",
)
CFB_LINE_MARKET_TYPES = frozenset({"spread", "total", "money_line"})
# Openers are essentially absent before 2012, carried by a single SBR book
# through 2019, absent again in 2020, and near-universal from 2021.
CFB_LINE_OPENERS_EXPECTED_FROM = 2012
CFB_LINE_ZERO_OPENER_SEASONS = frozenset({2020})
CFB_LINE_SOURCE_REGIMES = {
    "sbr_multibook": (
        "2006-2019 SBR odds-archive era: ~15-20 offshore books per game; "
        "openers only from 2012, from the single '5Dimes & sportbet' column"
    ),
    "cfbd_provider_sparse": (
        "2020-2022 CFBD provider era: sparse books; 2020 has zero openers and zero moneylines"
    ),
    "espn_book_era": ("2023+ ESPN Bet/DraftKings/Bovada era: openers for ~99-100% of games"),
}
CFB_LINE_SEMANTICS = {
    "line_stage": "opener plus an unidentified-time resolved quote (close proxy)",
    "timestamped_quotes": False,
    "date_time_meaning": "scheduled kickoff, never a quote observation time",
    "grain": "one row per game x market x book x side",
}

CFB_PBP_REQUIRED_COLUMNS = (
    "game_id",
    "game_play_number",
    "season",
    "week",
    "seasonType",
    "pos_team",
    "def_pos_team",
    "homeTeamId",
    "awayTeamId",
    "period",
    "down",
    "distance",
    "text",
    "EPA",
    "wpa",
    "rush",
    "pass",
)
CFB_PBP_SNAPSHOT_COLUMNS = (
    *CFB_PBP_REQUIRED_COLUMNS,
    "pos_team_id",
    "def_pos_team_id",
    "homeTeamAbbrev",
    "awayTeamAbbrev",
    "is_home",
    "statYardage",
    "start.yardsToEndzone",
    "clock.minutes",
    "clock.seconds",
    "pos_team_score",
    "def_pos_team_score",
    "pos_score_diff_start",
    "type.text",
    "pass_attempt",
    "touchdown",
    "rush_td",
    "pass_td",
    "kneel_down",
    "penalty_in_text",
    "yds_rushed",
    "air_yards",
    "yards_after_catch",
    "EPA_success",
    "EPA_rush",
    "EPA_pass",
    "home_wp_before",
    "away_wp_before",
    "xpass",
    "pass_oe",
    "rusher_player_id",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_name",
    "passer_player_name",
    "receiver_player_name",
    "wallclock",
)
CFB_PBP_SEASON_TYPE_CODES = {"1": "preseason", "2": "regular", "3": "postseason"}
# The 2026-08-03 upstream rebuild labels a handful of ESPN off-season/all-star
# exhibitions (2-4 games per season, 2008+) with codes outside the competitive
# calendar. They are dropped and counted rather than failing the whole season;
# any other unknown code still fails closed.
CFB_PBP_EXCLUDED_SEASON_TYPE_CODES = {
    "4": "espn_offseason_allstar",
    "5": "espn_offseason_allstar",
}
CFB_PBP_SOURCE_REGIMES = {
    "2004-2013": "early ESPN coverage: thinner play detail, stat-credited rosters only",
    "2014-present": "full ESPN coverage: ~98% of completed FBS games (verified on 2024)",
}

# The XLG-01 audit proved these ESPN game-roster flags are scrape-time athlete
# attributes, not game-day designations: zero of 27,471 players changed
# Active/Inactive across all 2024 games, and did_not_play/starter/valid are
# False on every audited row. They must never feed an availability feature.
CFB_ROSTER_QUARANTINED_COLUMNS: dict[str, str] = {
    "active": "static athlete attribute stamped at scrape time; never varies within a season",
    "is_active": "near-constant scrape-time attribute, not a game-day designation",
    "status_id": "mirror of the static Active/Inactive attribute",
    "status_name": "mirror of the static Active/Inactive attribute",
    "status_type": "mirror of the static Active/Inactive attribute",
    "status_abbreviation": "mirror of the static Active/Inactive attribute",
    "did_not_play": "False on every audited row; carries no participation signal",
    "starter": "False on every audited row; carries no lineup signal",
    "valid": "False on every audited row; upstream default only",
}
CFB_ROSTER_REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "week",
    "athlete_id",
    "team_id",
    "home_away",
    "full_name",
    "jersey",
    "team_abbreviation",
)
CFB_ROSTER_SNAPSHOT_COLUMNS = (
    "game_id",
    "season",
    "week",
    "athlete_id",
    "first_name",
    "last_name",
    "full_name",
    "jersey",
    "weight",
    "height",
    "date_of_birth",
    "experience_years",
    "team_id",
    "team_abbreviation",
    "team_display_name",
    "home_away",
    "order",
)
CFB_ROSTER_AVAILABILITY_CONTRACT = (
    "scrape-time roster listing, not game-day availability: listed does not "
    "mean played, Active/Inactive is a static athlete attribute, and "
    "2004-2013 files contain only stat-credited players (~32 per team-game)"
)
CFB_ROSTER_SOURCE_REGIMES = {
    "2004-2013": "stat-credited players only (~32 per team-game); flags defaulted",
    "2014-present": "full ~120-player scrape-time rosters; flags static, quarantined",
}

CFB_PARTICIPANT_ROLES = (
    "kicker",
    "passer",
    "rusher",
    "assisted_by",
    "punter",
    "receiver",
    "tackler",
    "returner",
    "scorer",
    "pat_scorer",
    "pass_defender",
    "penalized",
    "sacked_by",
    "pat_passer",
    "recoverer",
    "forced_by",
)
CFB_PARTICIPANT_KEY_COLUMNS = ("game_id", "play_id", "season", "week")
CFB_PARTICIPANT_REQUIRED_COLUMNS = (
    *CFB_PARTICIPANT_KEY_COLUMNS,
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
    "tackler_player_id",
)
CFB_PARTICIPANT_SNAPSHOT_COLUMNS = (
    *CFB_PARTICIPANT_KEY_COLUMNS,
    *(
        f"{role}_player_{suffix}"
        for role in CFB_PARTICIPANT_ROLES
        for suffix in ("id", "name", "ids", "names")
    ),
)
CFB_PARTICIPANT_AVAILABILITY_CONTRACT = (
    "credited actors from play text only: positive evidence of participation. "
    "Absence of credit is never evidence of absence; no lineups or snap "
    "counts exist for CFB"
)

ESPN_CFB_BETTING_REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "week",
    "game_spread",
    "over_under",
    "home_favorite",
    "home_team_spread",
    "game_spread_available",
    "odds_source",
)

# CFBD gap-filler contracts. Column names are kept verbatim from the CFBD v5
# response schemas (camelCase; nested objects flattened with a dot separator)
# so the canonical parquet stays traceable to the snapshotted raw JSON.
CFBD_DRAFT_PICK_REQUIRED_COLUMNS = (
    "year",
    "round",
    "pick",
    "overall",
    "collegeAthleteId",
    "nflAthleteId",
    "collegeId",
    "collegeTeam",
    "collegeConference",
    "nflTeamId",
    "nflTeam",
    "name",
    "position",
)
CFBD_DRAFT_PICK_SNAPSHOT_COLUMNS = (
    *CFBD_DRAFT_PICK_REQUIRED_COLUMNS,
    "height",
    "weight",
    "preDraftRanking",
    "preDraftPositionRanking",
    "preDraftGrade",
    "hometownInfo.city",
    "hometownInfo.state",
    "hometownInfo.country",
)
CFBD_DRAFT_CROSSWALK_CONTRACT = (
    "Both id columns must exist; nulls are allowed and per-season non-null "
    "rates are reported in the audit. collegeAthleteId is the CFBD/ESPN "
    "college athlete id and is the redundant no-name-join path into NFL "
    "identity (collegeAthleteId -> nflverse espn_id -> gsis_id; verified "
    "live 2026-08-16 at ~99% per class for 2019+ drafts, degrading to ~11% "
    "for 2015 where an older id space appears). nflAthleteId is a "
    "CFBD-internal NFL id space that does NOT match nflverse espn_id and "
    "must never be joined against it."
)

CFBD_RETURNING_REQUIRED_COLUMNS = (
    "season",
    "team",
    "conference",
    "totalPPA",
    "totalPassingPPA",
    "totalReceivingPPA",
    "totalRushingPPA",
    "percentPPA",
    "percentPassingPPA",
    "percentReceivingPPA",
    "percentRushingPPA",
    "usage",
    "passingUsage",
    "receivingUsage",
    "rushingUsage",
)

CFBD_RECRUITING_TEAM_REQUIRED_COLUMNS = ("year", "rank", "team", "points")

CFBD_RECRUIT_CLASSIFICATIONS = frozenset({"HighSchool", "JUCO", "PrepSchool"})
CFBD_RECRUIT_REQUIRED_COLUMNS = (
    "id",
    "athleteId",
    "recruitType",
    "year",
    "ranking",
    "name",
    "school",
    "committedTo",
    "position",
    "stars",
    "rating",
)
CFBD_RECRUIT_SNAPSHOT_COLUMNS = (
    *CFBD_RECRUIT_REQUIRED_COLUMNS,
    "height",
    "weight",
    "city",
    "stateProvince",
    "country",
)

CFBD_USAGE_SHARE_COLUMNS = (
    "usage.overall",
    "usage.pass",
    "usage.rush",
    "usage.firstDown",
    "usage.secondDown",
    "usage.thirdDown",
    "usage.standardDowns",
    "usage.passingDowns",
)
CFBD_USAGE_REQUIRED_COLUMNS = (
    "season",
    "id",
    "name",
    "position",
    "team",
    "conference",
    *CFBD_USAGE_SHARE_COLUMNS,
)

CFBD_PORTAL_ELIGIBILITY_VALUES = frozenset(
    {"Immediate", "PendingAppeal", "SittingOne", "TBD", "Withdrawn"}
)
CFBD_PORTAL_REQUIRED_COLUMNS = (
    "season",
    "firstName",
    "lastName",
    "position",
    "origin",
    "destination",
    "transferDate",
    "rating",
    "stars",
    "eligibility",
)
CFBD_PORTAL_IDENTITY_CONTRACT = (
    "CFBD /player/portal publishes no athlete ids: rows are name-keyed only. "
    "Linking portal entries into any id space requires a reviewed "
    "name+school+date match and must never silently join on names."
)


def assert_no_quarantined_roster_columns(
    frame: pd.DataFrame, dataset: str = "cfb_game_rosters"
) -> None:
    """Refuse any roster table that still carries the unreliable flags."""

    present = sorted(set(CFB_ROSTER_QUARANTINED_COLUMNS).intersection(frame.columns))
    if present:
        raise DataContractError(
            f"{dataset} contains quarantined availability columns: {', '.join(present)}. "
            "The XLG-01 audit showed these flags never vary; they must not feed features."
        )


def cfb_line_source_regime(season: int) -> str:
    if 2006 <= season <= 2019:
        return "sbr_multibook"
    if 2020 <= season <= 2022:
        return "cfbd_provider_sparse"
    if season >= 2023:
        return "espn_book_era"
    raise DataContractError(f"No CFB line source regime is defined for season {season}")


def _require_single_season(
    frame: pd.DataFrame, season: int, dataset: str, column: str = "season"
) -> None:
    if frame.empty:
        raise DataContractError(f"{dataset} season {season} contains no rows")
    observed = set(pd.to_numeric(frame[column], errors="coerce").dropna().astype(int))
    if observed != {season}:
        raise DataContractError(
            f"{dataset} season partition {season} contains seasons {sorted(observed)}"
        )


def _fill_missing_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    return result.loc[:, list(columns)].copy()


def canonicalize_cfb_schedules(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one CFBD-flavor schedule season and audit its composition."""

    require_columns(frame, CFB_SCHEDULE_REQUIRED_COLUMNS, "cfb_schedules")
    result = _fill_missing_columns(frame, CFB_SCHEDULE_SNAPSHOT_COLUMNS)
    for column in ("game_id", "season", "week"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    _require_single_season(result, season, "cfb_schedules")
    if result["game_id"].duplicated().any():
        raise DataContractError(f"cfb_schedules season {season} contains duplicate game_id values")
    if result["home_team"].eq(result["away_team"]).any():
        raise DataContractError(
            f"cfb_schedules season {season} contains a game with identical teams"
        )
    result = result.sort_values(["week", "start_date", "game_id"]).reset_index(drop=True)
    divisions = result["home_division"].astype("string").str.lower()
    audit = {
        "rows": len(result),
        "fbs_home_games": int(divisions.eq("fbs").sum()),
        "season_types": {
            str(key): int(value)
            for key, value in result["season_type"].value_counts(dropna=False).items()
        },
    }
    return result, audit


def canonicalize_cfb_lines(
    frame: pd.DataFrame, seasons: list[int]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize the multi-book line archive with regime and opener contracts."""

    require_columns(frame, CFB_LINE_REQUIRED_COLUMNS, "cfb_line_odds")
    result = frame.loc[:, list(CFB_LINE_REQUIRED_COLUMNS)].copy()
    rows_in = len(result)
    result = result.drop_duplicates()
    full_row_duplicates = rows_in - len(result)
    result["game_id"] = pd.to_numeric(result["game_id"], errors="coerce")
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    unjoinable = result["game_id"].isna() | result["season"].isna()
    unjoinable_rows = int(unjoinable.sum())
    result = result.loc[~unjoinable].copy()
    result["game_id"] = result["game_id"].astype("int64")
    result["season"] = result["season"].astype("int64")
    result["market_type"] = result["market_type"].astype("string")
    unknown_markets = sorted(set(result["market_type"].dropna()) - CFB_LINE_MARKET_TYPES)
    if unknown_markets:
        raise DataContractError(
            f"cfb_line_odds contains unknown market types: {', '.join(unknown_markets)}"
        )
    for column in ("lines", "odds", "opening_lines", "opening_odds"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["week"] = pd.to_numeric(result["week"], errors="coerce").astype("Int64")
    for column in ("date_time", "abbr", "book", "season_type"):
        result[column] = result[column].astype("string")

    result = result.loc[result["season"].isin(seasons)].copy()
    result["source_regime"] = result["season"].map(lambda value: cfb_line_source_regime(int(value)))
    per_season: dict[str, dict[str, Any]] = {}
    for season in seasons:
        group = result.loc[result["season"].eq(season)]
        if group.empty:
            raise DataContractError(f"cfb_line_odds contains no rows for season {season}")
        spread_rows = group.loc[group["market_type"].eq("spread")]
        opener_rows = int(group["opening_lines"].notna().sum())
        if (
            season >= CFB_LINE_OPENERS_EXPECTED_FROM
            and season not in CFB_LINE_ZERO_OPENER_SEASONS
            and opener_rows == 0
        ):
            raise DataContractError(
                f"cfb_line_odds season {season} has zero openers although the "
                f"{cfb_line_source_regime(season)} regime carries them; upstream regression"
            )
        per_season[str(season)] = {
            "rows": len(group),
            "spread_rows": len(spread_rows),
            "moneyline_rows": int(group["market_type"].eq("money_line").sum()),
            "opener_rows": opener_rows,
            "books": int(group["book"].nunique()),
            "games": int(group["game_id"].nunique()),
            "source_regime": cfb_line_source_regime(season),
            "unexpected_openers": bool(season in CFB_LINE_ZERO_OPENER_SEASONS and opener_rows > 0),
        }
    result = (
        result.loc[:, list(CFB_LINE_SNAPSHOT_COLUMNS)]
        .sort_values(["season", "week", "game_id", "market_type", "book", "abbr"])
        .reset_index(drop=True)
    )
    audit = {
        "rows_in_source": rows_in,
        "full_row_duplicates_dropped": full_row_duplicates,
        "unjoinable_rows_dropped": unjoinable_rows,
        "per_season": per_season,
    }
    return result, audit


def canonicalize_cfb_pbp(frame: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one ESPN CFB play-by-play season to the storage contract."""

    require_columns(frame, CFB_PBP_REQUIRED_COLUMNS, "espn_cfb_pbp")
    result = _fill_missing_columns(frame, CFB_PBP_SNAPSHOT_COLUMNS)
    key_columns = ("game_id", "game_play_number", "season", "week", "seasonType")
    numeric_keys = {
        column: pd.to_numeric(result[column], errors="coerce") for column in key_columns
    }
    null_key_mask = pd.concat(list(numeric_keys.values()), axis=1).isna().any(axis=1)
    null_key_rows_dropped = int(null_key_mask.sum())
    result = result.loc[~null_key_mask].copy()
    for column in key_columns:
        result[column] = numeric_keys[column].loc[result.index].astype("int64")
    observed_codes = set(result["seasonType"].astype(str))
    unknown_types = sorted(
        observed_codes - set(CFB_PBP_SEASON_TYPE_CODES) - set(CFB_PBP_EXCLUDED_SEASON_TYPE_CODES)
    )
    if unknown_types:
        raise DataContractError(
            f"espn_cfb_pbp season {season} has unknown seasonType codes: {', '.join(unknown_types)}"
        )
    excluded_mask = result["seasonType"].astype(str).isin(CFB_PBP_EXCLUDED_SEASON_TYPE_CODES)
    excluded = result.loc[excluded_mask]
    excluded_offseason_rows = {
        "rows": len(excluded),
        "games": int(excluded["game_id"].nunique()),
        "codes": {
            str(key): int(value) for key, value in excluded["seasonType"].value_counts().items()
        },
    }
    result = result.loc[~excluded_mask]
    _require_single_season(result, season, "espn_cfb_pbp")
    if result.duplicated(["game_id", "game_play_number"]).any():
        raise DataContractError(
            f"espn_cfb_pbp season {season} contains duplicate game_id/game_play_number rows"
        )
    result = result.sort_values(["season", "week", "game_id", "game_play_number"]).reset_index(
        drop=True
    )
    audit = {
        "rows": len(result),
        "games": int(result["game_id"].nunique()),
        "season_types": {
            CFB_PBP_SEASON_TYPE_CODES[str(key)]: int(value)
            for key, value in result["seasonType"].value_counts().items()
        },
        "null_epa_rows": int(pd.to_numeric(result["EPA"], errors="coerce").isna().sum()),
        "null_key_rows_dropped": null_key_rows_dropped,
        "excluded_offseason_rows": excluded_offseason_rows,
    }
    return result, audit


def canonicalize_cfb_game_rosters(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one game-roster season, excluding quarantined availability flags."""

    require_columns(frame, CFB_ROSTER_REQUIRED_COLUMNS, "cfb_game_rosters")
    rows_in = len(frame)
    result = _fill_missing_columns(frame, CFB_ROSTER_SNAPSHOT_COLUMNS)
    assert_no_quarantined_roster_columns(result)
    for column in ("game_id", "season", "week", "athlete_id", "team_id"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    result = result.sort_values(["game_id", "team_id", "athlete_id"]).drop_duplicates(
        ["game_id", "team_id", "athlete_id"], keep="first"
    )
    duplicate_rows_dropped = rows_in - len(result)
    _require_single_season(result, season, "cfb_game_rosters")
    result = result.reset_index(drop=True)
    team_game_sizes = result.groupby(["game_id", "team_id"], sort=False)["athlete_id"].size()
    audit = {
        "rows": len(result),
        "games": int(result["game_id"].nunique()),
        "athletes": int(result["athlete_id"].nunique()),
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "mean_players_per_team_game": float(round(team_game_sizes.mean(), 2)),
        "quarantined_columns_excluded": sorted(CFB_ROSTER_QUARANTINED_COLUMNS),
    }
    return result, audit


def canonicalize_cfb_play_participants(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one play-participants season (credited actors only)."""

    require_columns(frame, CFB_PARTICIPANT_REQUIRED_COLUMNS, "cfb_play_participants")
    result = _fill_missing_columns(frame, CFB_PARTICIPANT_SNAPSHOT_COLUMNS)
    for column in CFB_PARTICIPANT_KEY_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    _require_single_season(result, season, "cfb_play_participants")
    if result.duplicated(["game_id", "play_id"]).any():
        raise DataContractError(
            f"cfb_play_participants season {season} contains duplicate game_id/play_id rows"
        )
    result = result.sort_values(["season", "week", "game_id", "play_id"]).reset_index(drop=True)
    audit = {"rows": len(result), "games": int(result["game_id"].nunique())}
    return result, audit


def canonicalize_espn_cfb_betting(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize the single-line ESPN betting cross-check, refusing placeholders."""

    require_columns(frame, ESPN_CFB_BETTING_REQUIRED_COLUMNS, "espn_cfb_betting")
    result = frame.loc[:, list(ESPN_CFB_BETTING_REQUIRED_COLUMNS)].copy()
    for column in ("game_id", "season", "week"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    result["odds_source"] = result["odds_source"].astype("string")
    placeholder = result["odds_source"].eq("default") | ~result["game_spread_available"].astype(
        bool
    )
    placeholder_rows = int(placeholder.sum())
    result = result.loc[~placeholder].copy()
    if result.empty:
        raise DataContractError(
            f"espn_cfb_betting season {season} contains only placeholder default rows "
            "(odds_source='default', game_spread_available=False); refusing to ingest"
        )
    _require_single_season(result, season, "espn_cfb_betting")
    if result["game_id"].duplicated().any():
        raise DataContractError(f"espn_cfb_betting season {season} contains duplicate game_id rows")
    result = result.sort_values(["week", "game_id"]).reset_index(drop=True)
    audit = {
        "rows": len(result),
        "placeholder_rows_dropped": placeholder_rows,
        "odds_sources": {
            str(key): int(value) for key, value in result["odds_source"].value_counts().items()
        },
    }
    return result, audit


def _nonnull_rate(series: pd.Series) -> float:
    return float(round(series.notna().mean(), 4))


def canonicalize_cfbd_draft_picks(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one draft year, enforcing the athlete-id crosswalk contract."""

    require_columns(frame, CFBD_DRAFT_PICK_REQUIRED_COLUMNS, "cfbd_draft_picks")
    result = _fill_missing_columns(frame, CFBD_DRAFT_PICK_SNAPSHOT_COLUMNS)
    for column in ("year", "round", "pick", "overall", "nflTeamId"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    for column in ("collegeAthleteId", "nflAthleteId", "collegeId"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    _require_single_season(result, season, "cfbd_draft_picks", column="year")
    if result.duplicated(["year", "overall"]).any():
        raise DataContractError(
            f"cfbd_draft_picks year {season} contains duplicate year/overall picks"
        )
    result = result.sort_values(["round", "pick", "overall"]).reset_index(drop=True)
    both_ids = result["collegeAthleteId"].notna() & result["nflAthleteId"].notna()
    audit = {
        "rows": len(result),
        "rounds": int(result["round"].nunique()),
        "college_athlete_id_nonnull_rate": _nonnull_rate(result["collegeAthleteId"]),
        "nfl_athlete_id_nonnull_rate": _nonnull_rate(result["nflAthleteId"]),
        "both_ids_nonnull_rate": float(round(both_ids.mean(), 4)),
        "zero_value_ids": {
            "collegeAthleteId": int(result["collegeAthleteId"].eq(0).sum()),
            "nflAthleteId": int(result["nflAthleteId"].eq(0).sum()),
        },
    }
    return result, audit


def canonicalize_cfbd_returning_production(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one returning-production season (team grain, no athlete ids)."""

    require_columns(frame, CFBD_RETURNING_REQUIRED_COLUMNS, "cfbd_returning_production")
    result = _fill_missing_columns(frame, CFBD_RETURNING_REQUIRED_COLUMNS)
    result["season"] = pd.to_numeric(result["season"], errors="raise").astype("int64")
    for column in CFBD_RETURNING_REQUIRED_COLUMNS[3:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("team", "conference"):
        result[column] = result[column].astype("string")
    _require_single_season(result, season, "cfbd_returning_production")
    if result["team"].isna().any():
        raise DataContractError(f"cfbd_returning_production season {season} has null team values")
    if result["team"].duplicated().any():
        raise DataContractError(
            f"cfbd_returning_production season {season} contains duplicate teams"
        )
    result = result.sort_values("team").reset_index(drop=True)
    audit = {
        "rows": len(result),
        "conferences": int(result["conference"].nunique()),
        "null_percent_ppa_rows": int(result["percentPPA"].isna().sum()),
    }
    return result, audit


def canonicalize_cfbd_recruiting_teams(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one team recruiting-class ranking year (team grain, no ids)."""

    require_columns(frame, CFBD_RECRUITING_TEAM_REQUIRED_COLUMNS, "cfbd_recruiting_teams")
    result = _fill_missing_columns(frame, CFBD_RECRUITING_TEAM_REQUIRED_COLUMNS)
    for column in ("year", "rank"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    result["points"] = pd.to_numeric(result["points"], errors="coerce")
    result["team"] = result["team"].astype("string")
    _require_single_season(result, season, "cfbd_recruiting_teams", column="year")
    if result["team"].duplicated().any():
        raise DataContractError(f"cfbd_recruiting_teams year {season} contains duplicate teams")
    result = result.sort_values(["rank", "team"]).reset_index(drop=True)
    audit = {"rows": len(result), "max_rank": int(result["rank"].max())}
    return result, audit


def canonicalize_cfbd_recruiting_players(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one player recruiting class, requiring stable recruit ids."""

    require_columns(frame, CFBD_RECRUIT_REQUIRED_COLUMNS, "cfbd_recruiting_players")
    result = _fill_missing_columns(frame, CFBD_RECRUIT_SNAPSHOT_COLUMNS)
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype("int64")
    for column in ("ranking", "stars"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    for column in ("rating", "height", "weight"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("id", "athleteId", "recruitType", "name"):
        result[column] = result[column].astype("string")
    _require_single_season(result, season, "cfbd_recruiting_players", column="year")
    if result["id"].isna().any():
        raise DataContractError(f"cfbd_recruiting_players year {season} has null recruit ids")
    if result["id"].duplicated().any():
        raise DataContractError(
            f"cfbd_recruiting_players year {season} contains duplicate recruit ids"
        )
    unknown_types = sorted(set(result["recruitType"].dropna()) - CFBD_RECRUIT_CLASSIFICATIONS)
    if unknown_types:
        raise DataContractError(
            f"cfbd_recruiting_players year {season} has unknown recruit types: "
            f"{', '.join(unknown_types)}"
        )
    result = result.sort_values(["ranking", "id"]).reset_index(drop=True)
    audit = {
        "rows": len(result),
        "athlete_id_nonnull_rate": _nonnull_rate(result["athleteId"]),
        "recruit_types": {
            str(key): int(value) for key, value in result["recruitType"].value_counts().items()
        },
    }
    return result, audit


def canonicalize_cfbd_usage(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one player-usage season, requiring stable athlete ids."""

    require_columns(frame, CFBD_USAGE_REQUIRED_COLUMNS, "cfbd_usage")
    result = _fill_missing_columns(frame, CFBD_USAGE_REQUIRED_COLUMNS)
    result["season"] = pd.to_numeric(result["season"], errors="raise").astype("int64")
    for column in CFBD_USAGE_SHARE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("id", "name", "position", "team", "conference"):
        result[column] = result[column].astype("string")
    result["conference"] = result["conference"].replace("", pd.NA)
    _require_single_season(result, season, "cfbd_usage")
    if result["id"].isna().any():
        raise DataContractError(f"cfbd_usage season {season} has null athlete ids")
    # Upstream artifact (observed live on 2023): some player rows appear twice,
    # identical except for a blank conference. Shadow copies are dropped, but
    # duplicates that disagree on any usage share stay a contract violation.
    duplicated = result.duplicated(["id", "team"], keep=False)
    if duplicated.any():
        share_variants = (
            result.loc[duplicated]
            .groupby(["id", "team"])[list(CFBD_USAGE_SHARE_COLUMNS)]
            .nunique(dropna=False)
        )
        if bool(share_variants.gt(1).any().any()):
            raise DataContractError(
                f"cfbd_usage season {season} contains duplicate athlete/team rows "
                "with conflicting usage shares"
            )
    rows_in = len(result)
    result = result.sort_values(["team", "id", "conference"], na_position="last")
    result = result.drop_duplicates(["id", "team"], keep="first").reset_index(drop=True)
    audit = {
        "rows": len(result),
        "players": int(result["id"].nunique()),
        "teams": int(result["team"].nunique()),
        "shadow_duplicate_rows_dropped": rows_in - len(result),
    }
    return result, audit


def canonicalize_cfbd_portal(
    frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one transfer-portal season (name-keyed source, no athlete ids)."""

    require_columns(frame, CFBD_PORTAL_REQUIRED_COLUMNS, "cfbd_portal")
    result = _fill_missing_columns(frame, CFBD_PORTAL_REQUIRED_COLUMNS)
    result["season"] = pd.to_numeric(result["season"], errors="raise").astype("int64")
    result["stars"] = pd.to_numeric(result["stars"], errors="coerce").astype("Int64")
    result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    for column in ("firstName", "lastName", "position", "origin", "destination"):
        result[column] = result[column].astype("string")
    for column in ("transferDate", "eligibility"):
        result[column] = result[column].astype("string")
    _require_single_season(result, season, "cfbd_portal")
    unknown = sorted(set(result["eligibility"].dropna()) - CFBD_PORTAL_ELIGIBILITY_VALUES)
    if unknown:
        raise DataContractError(
            f"cfbd_portal season {season} has unknown eligibility values: {', '.join(unknown)}"
        )
    rows_in = len(result)
    result = result.drop_duplicates().reset_index(drop=True)
    result = result.sort_values(["transferDate", "lastName", "firstName"]).reset_index(drop=True)
    audit = {
        "rows": len(result),
        "duplicate_rows_dropped": rows_in - len(result),
        "eligibility": {
            str(key): int(value) for key, value in result["eligibility"].value_counts().items()
        },
        "identity_contract": CFBD_PORTAL_IDENTITY_CONTRACT,
    }
    return result, audit


# ---------------------------------------------------------------------------
# Download plumbing
# ---------------------------------------------------------------------------


def _http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-ats-research/0.2"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return bytes(response.read())


def _download(url: str, description: str) -> bytes:
    try:
        return _http_bytes(url)
    except urllib.error.HTTPError as error:
        raise DataContractError(f"{description} is unavailable ({error.code}) at {url}") from error


def _github_json(url: str) -> Any:
    return json.loads(_http_bytes(url).decode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(payload: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def _read_parquet_bytes(payload: bytes, description: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(payload))
    except Exception as error:
        raise DataContractError(f"{description} is not a readable parquet file") from error


def cfbd_api_key() -> str:
    """Read the CFBD key from the environment, failing closed when unset."""

    key = os.environ.get(CFBD_API_KEY_ENV, "").strip()
    if not key:
        raise DataContractError(
            f"CFBD ingestion requires the {CFBD_API_KEY_ENV} environment variable "
            "(free key from collegefootballdata.com/key, 1,000 calls/month); "
            "refusing to run without it"
        )
    return key


def _cfbd_http(url: str, api_key: str) -> tuple[bytes, dict[str, str]]:
    """One authenticated CFBD request; the key travels only in the header."""

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nfl-ats-research/0.2",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = bytes(response.read())
        quota = {
            name: str(response.headers.get(name))
            for name in CFBD_QUOTA_HEADERS
            if response.headers.get(name)
        }
    return payload, quota


def _cfbd_get(path: str, params: dict[str, Any], api_key: str) -> tuple[bytes, dict[str, str], str]:
    query = urllib.parse.urlencode(params)
    url = f"{CFBD_API_ROOT}{path}" + (f"?{query}" if query else "")
    waits = iter(CFBD_RETRY_WAITS)
    while True:
        try:
            payload, quota = _cfbd_http(url, api_key)
        except urllib.error.HTTPError as error:
            wait = next(waits, None) if error.code == 429 else None
            if wait is not None:
                _sleep(wait)
                continue
            raise DataContractError(
                f"CFBD endpoint {path} returned HTTP {error.code} for params {params}"
            ) from error
        return payload, quota, url


def _cfbd_records(payload: bytes, description: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataContractError(f"{description} response is not valid JSON") from error
    if not isinstance(parsed, list) or not all(isinstance(record, dict) for record in parsed):
        raise DataContractError(f"{description} response is not a JSON array of records")
    return cast(list[dict[str, Any]], parsed)


def resolve_cfbd_api() -> dict[str, str]:
    """Record the CFBD host and OpenAPI version without spending a quota call."""

    payload = _download(CFBD_API_DOCS_URL, "CFBD OpenAPI document")
    try:
        version = str(json.loads(payload.decode("utf-8"))["info"]["version"])
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataContractError("CFBD OpenAPI document does not expose info.version") from error
    return {
        "api_root": CFBD_API_ROOT,
        "api_docs_url": CFBD_API_DOCS_URL,
        "api_version": version,
        "auth": (
            f"Authorization: Bearer key from the {CFBD_API_KEY_ENV} environment "
            "variable; the key is never written to disk"
        ),
        "resolved_at_utc": datetime.now(UTC).isoformat(),
    }


def resolve_cfbfastr_commit() -> dict[str, str]:
    """Pin the cfbfastR-data head commit so raw-file downloads are reproducible."""

    payload = _github_json(
        f"{GITHUB_API_ROOT}/repos/{CFBFASTR_DATA_REPOSITORY}/branches/{CFBFASTR_DATA_BRANCH}"
    )
    return {
        "repository": f"https://github.com/{CFBFASTR_DATA_REPOSITORY}",
        "branch": CFBFASTR_DATA_BRANCH,
        "commit_sha": str(payload["commit"]["sha"]),
        "resolved_at_utc": datetime.now(UTC).isoformat(),
    }


def _cfbfastr_raw_url(commit_sha: str, relative_path: str) -> str:
    return f"{RAW_CONTENT_ROOT}/{CFBFASTR_DATA_REPOSITORY}/{commit_sha}/{relative_path}"


def resolve_release_assets(release_tag: str) -> dict[str, Any]:
    """List one sportsdataverse-data release's assets with sizes and timestamps."""

    payload = _github_json(
        f"{GITHUB_API_ROOT}/repos/{SPORTSDATAVERSE_DATA_REPOSITORY}/releases/tags/{release_tag}"
    )
    assets = {
        str(asset["name"]): {
            "bytes": int(asset["size"]),
            "updated_at": str(asset["updated_at"]),
            "download_url": str(asset["browser_download_url"]),
        }
        for asset in payload.get("assets", [])
    }
    if not assets:
        raise DataContractError(f"Release {release_tag} has no downloadable assets")
    return {
        "repository": f"https://github.com/{SPORTSDATAVERSE_DATA_REPOSITORY}",
        "release_tag": release_tag,
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "assets": assets,
    }


def _release_asset_for_season(
    release: dict[str, Any], spec: CfbSourceSpec, season: int
) -> tuple[str, dict[str, Any]]:
    name = f"{spec.asset_prefix}_{season}.parquet"
    assets: dict[str, Any] = release["assets"]
    if name not in assets:
        available = sorted(asset for asset in assets if str(asset).endswith(".parquet"))
        raise DataContractError(
            f"Release {spec.release_tag} has no asset {name}; parquet assets: "
            f"{', '.join(available[:8])}..."
        )
    asset: dict[str, Any] = assets[name]
    return name, asset


def _validate_requested_seasons(spec: CfbSourceSpec, seasons: list[int]) -> list[int]:
    values = [int(season) for season in seasons]
    if not values or values != sorted(set(values)):
        raise ValueError("Seasons must be non-empty, unique, and sorted")
    early = [season for season in values if season < spec.first_season]
    if early:
        raise DataContractError(
            f"CFB source {spec.key} has no data before {spec.first_season}; requested {early}"
        )
    refused = sorted(set(values).intersection(spec.refused_seasons))
    if refused:
        raise DataContractError(
            f"CFB source {spec.key} refuses seasons {refused}: {spec.refusal_reason}"
        )
    return values


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfbSnapshot:
    """An immutable, season-partitioned CFB source snapshot."""

    source: str
    snapshot_id: str
    root: Path
    seasons: tuple[int, ...]

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def season_path(self, season: int) -> Path:
        filename = CFB_SOURCES[self.source].partition_filename
        return self.root / f"season={season}" / filename


def _source_manifest_extras(spec: CfbSourceSpec) -> dict[str, Any]:
    if spec.key == "schedules":
        return {"grain": "one row per game, all divisions (filter home/away_division for FBS)"}
    if spec.key == "lines":
        return {"semantics": CFB_LINE_SEMANTICS, "source_regimes": CFB_LINE_SOURCE_REGIMES}
    if spec.key == "pbp":
        return {
            "season_type_codes": CFB_PBP_SEASON_TYPE_CODES,
            "excluded_season_type_codes": CFB_PBP_EXCLUDED_SEASON_TYPE_CODES,
            "source_regimes": CFB_PBP_SOURCE_REGIMES,
        }
    if spec.key == "rosters":
        return {
            "availability_contract": CFB_ROSTER_AVAILABILITY_CONTRACT,
            "quarantined_columns": CFB_ROSTER_QUARANTINED_COLUMNS,
            "source_regimes": CFB_ROSTER_SOURCE_REGIMES,
        }
    if spec.key == "participants":
        return {"availability_contract": CFB_PARTICIPANT_AVAILABILITY_CONTRACT}
    if spec.key == "draft_picks":
        return {
            "grain": "one row per draft pick, partitioned by draft year",
            "crosswalk_contract": CFBD_DRAFT_CROSSWALK_CONTRACT,
        }
    if spec.key == "returning_production":
        return {"grain": "one row per team-season; no athlete ids in this source"}
    if spec.key == "recruiting_teams":
        return {"grain": "one row per team per class year; no ids in this source"}
    if spec.key == "recruiting_players":
        return {
            "grain": "one row per recruit per class year",
            "identity": "id is the CFBD recruit id; athleteId (nullable) is the ESPN athlete id",
        }
    if spec.key == "usage":
        return {
            "grain": "one row per player-season-team",
            "identity": "id is the CFBD/ESPN athlete id",
            "known_upstream_artifacts": (
                "some seasons repeat a player row with a blank conference; "
                "identical shadow copies are dropped and counted in the audit"
            ),
        }
    if spec.key == "portal":
        return {
            "grain": "one row per portal entry",
            "identity_contract": CFBD_PORTAL_IDENTITY_CONTRACT,
        }
    return {
        "semantics": {
            "line_stage": "one unidentified-time resolved line per game (close proxy)",
            "role": "cross-check only; the primary line source is the cfb line-odds archive",
        },
        "refused_seasons": sorted(spec.refused_seasons),
    }


def _snapshot_columns(spec: CfbSourceSpec) -> list[str]:
    columns = {
        "schedules": CFB_SCHEDULE_SNAPSHOT_COLUMNS,
        "lines": CFB_LINE_SNAPSHOT_COLUMNS,
        "pbp": CFB_PBP_SNAPSHOT_COLUMNS,
        "rosters": CFB_ROSTER_SNAPSHOT_COLUMNS,
        "participants": CFB_PARTICIPANT_SNAPSHOT_COLUMNS,
        "espn_betting": ESPN_CFB_BETTING_REQUIRED_COLUMNS,
        "draft_picks": CFBD_DRAFT_PICK_SNAPSHOT_COLUMNS,
        "returning_production": CFBD_RETURNING_REQUIRED_COLUMNS,
        "recruiting_teams": CFBD_RECRUITING_TEAM_REQUIRED_COLUMNS,
        "recruiting_players": CFBD_RECRUIT_SNAPSHOT_COLUMNS,
        "usage": CFBD_USAGE_REQUIRED_COLUMNS,
        "portal": CFBD_PORTAL_REQUIRED_COLUMNS,
    }
    return list(columns[spec.key])


def _canonicalize_season(
    spec: CfbSourceSpec, frame: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonicalizers: dict[
        str, Callable[[pd.DataFrame, int], tuple[pd.DataFrame, dict[str, Any]]]
    ] = {
        "schedules": canonicalize_cfb_schedules,
        "pbp": canonicalize_cfb_pbp,
        "rosters": canonicalize_cfb_game_rosters,
        "participants": canonicalize_cfb_play_participants,
        "espn_betting": canonicalize_espn_cfb_betting,
        "draft_picks": canonicalize_cfbd_draft_picks,
        "returning_production": canonicalize_cfbd_returning_production,
        "recruiting_teams": canonicalize_cfbd_recruiting_teams,
        "recruiting_players": canonicalize_cfbd_recruiting_players,
        "usage": canonicalize_cfbd_usage,
        "portal": canonicalize_cfbd_portal,
    }
    return canonicalizers[spec.key](frame, season)


def _season_source_urls(
    spec: CfbSourceSpec, seasons: list[int]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Resolve provenance and one source file per season without downloading."""

    files: dict[int, dict[str, Any]]
    if spec.origin == "cfbfastr_raw":
        provenance: dict[str, Any] = resolve_cfbfastr_commit()
        files = {
            season: {
                "name": str(spec.raw_name_pattern).format(season=season),
                "url": _cfbfastr_raw_url(
                    provenance["commit_sha"],
                    f"{spec.raw_directory}/{str(spec.raw_name_pattern).format(season=season)}",
                ),
            }
            for season in seasons
        }
        return provenance, files
    release = resolve_release_assets(str(spec.release_tag))
    files = {}
    for season in seasons:
        name, asset = _release_asset_for_season(release, spec, season)
        files[season] = {
            "name": name,
            "url": str(asset["download_url"]),
            "bytes": int(asset["bytes"]),
            "updated_at": str(asset["updated_at"]),
        }
    provenance = {key: value for key, value in release.items() if key != "assets"}
    return provenance, files


def _write_manifest(
    spec: CfbSourceSpec,
    snapshot: CfbSnapshot,
    provenance: dict[str, Any],
    partitions: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    license_notes = {
        "cfbfastr_raw": CFBFASTR_LICENSE,
        "sportsdataverse_release": SPORTSDATAVERSE_LICENSE,
        "cfbd_api": CFBD_LICENSE,
    }
    license_note = license_notes[spec.origin]
    manifest: dict[str, Any] = {
        "snapshot_id": snapshot.snapshot_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "contract_version": CFB_CONTRACT_VERSION,
        "cfb_source": spec.key,
        "source": {
            "dataset": spec.dataset,
            "origin": spec.origin,
            "license": license_note,
            "redistribution": CFB_REDISTRIBUTION_RULE,
            **provenance,
        },
        "seasons": list(snapshot.seasons),
        "columns": _snapshot_columns(spec),
        "partitions": partitions,
        "rows": sum(int(partition["rows"]) for partition in partitions),
        "audit": audit,
        **_source_manifest_extras(spec),
    }
    atomic_json(manifest, snapshot.manifest_path)
    return manifest


def _fetch_lines_snapshot(
    spec: CfbSourceSpec, seasons: list[int], destination: Path, identifier: str
) -> CfbSnapshot:
    provenance = resolve_cfbfastr_commit()
    source_name = str(spec.raw_name_pattern)
    url = _cfbfastr_raw_url(provenance["commit_sha"], f"{spec.raw_directory}/{source_name}")
    payload = _download(url, "cfb_line_odds archive")
    source_path = destination / "source" / source_name
    _atomic_bytes(payload, source_path)
    canonical, audit = canonicalize_cfb_lines(
        _read_parquet_bytes(payload, "cfb_line_odds archive"), seasons
    )
    source_file = {
        "path": str(source_path.relative_to(destination)),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "url": url,
    }
    snapshot = CfbSnapshot(spec.key, identifier, destination, tuple(seasons))
    partitions: list[dict[str, Any]] = []
    for season in seasons:
        partition_frame = canonical.loc[canonical["season"].eq(season)].reset_index(drop=True)
        path = snapshot.season_path(season)
        atomic_parquet(partition_frame, path)
        partitions.append(
            {
                "season": season,
                "rows": len(partition_frame),
                "path": str(path.relative_to(destination)),
                "sha256": _sha256_file(path),
                "source_file": source_file,
            }
        )
    _write_manifest(spec, snapshot, provenance, partitions, audit)
    return snapshot


def _cfbd_partition(
    snapshot: CfbSnapshot,
    spec: CfbSourceSpec,
    frame: pd.DataFrame,
    season: int,
    source_file: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonicalize and write one CFBD season partition, returning its manifest row."""

    canonical, season_audit = _canonicalize_season(spec, frame, season)
    path = snapshot.season_path(season)
    atomic_parquet(canonical, path)
    partition = {
        "season": season,
        "rows": len(canonical),
        "path": str(path.relative_to(snapshot.root)),
        "sha256": _sha256_file(path),
        "source_file": source_file,
    }
    return partition, season_audit


def _fetch_cfbd_snapshot(
    spec: CfbSourceSpec, seasons: list[int], destination: Path, identifier: str
) -> CfbSnapshot:
    """Snapshot one CFBD gap-filler source with raw JSON bytes and call audit."""

    api_key = cfbd_api_key()
    provenance: dict[str, Any] = dict(resolve_cfbd_api())
    snapshot = CfbSnapshot(spec.key, identifier, destination, tuple(seasons))
    partitions: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"per_season": {}}
    api_calls = 0
    quota: dict[str, str] = {}
    if spec.single_call:
        payload, quota, url = _cfbd_get(str(spec.api_path), {}, api_key)
        api_calls += 1
        source_path = destination / "source" / f"{spec.key}.json"
        _atomic_bytes(payload, source_path)
        frame = pd.json_normalize(_cfbd_records(payload, f"CFBD {spec.key}"))
        if spec.season_column not in frame.columns:
            raise DataContractError(
                f"CFBD {spec.key} response carries no {spec.season_column!r} field"
            )
        source_file = {
            "path": str(source_path.relative_to(destination)),
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
            "url": url,
            "endpoint": spec.api_path,
            "params": {},
        }
        season_values = pd.to_numeric(frame[spec.season_column], errors="coerce")
        for season in seasons:
            subset = frame.loc[season_values.eq(season)].reset_index(drop=True)
            partition, season_audit = _cfbd_partition(snapshot, spec, subset, season, source_file)
            partitions.append(partition)
            audit["per_season"][str(season)] = season_audit
        audit["seasons_in_response"] = sorted(
            int(value) for value in season_values.dropna().unique()
        )
    else:
        for index, season in enumerate(seasons):
            if index:
                _sleep(CFBD_THROTTLE_SECONDS)
            params = {spec.season_param: season}
            payload, season_quota, url = _cfbd_get(str(spec.api_path), params, api_key)
            api_calls += 1
            quota = season_quota or quota
            source_path = destination / "source" / f"{spec.key}_{season}.json"
            _atomic_bytes(payload, source_path)
            frame = pd.json_normalize(_cfbd_records(payload, f"CFBD {spec.key} season {season}"))
            source_file = {
                "path": str(source_path.relative_to(destination)),
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
                "url": url,
                "endpoint": spec.api_path,
                "params": params,
            }
            partition, season_audit = _cfbd_partition(snapshot, spec, frame, season, source_file)
            partitions.append(partition)
            audit["per_season"][str(season)] = season_audit
    audit["api_calls"] = api_calls
    if quota:
        audit["quota_headers_last"] = quota
    _write_manifest(spec, snapshot, provenance, partitions, audit)
    return snapshot


def fetch_cfb_snapshot(
    source: str,
    seasons: list[int],
    cfb_root: Path,
    snapshot_id: str | None = None,
) -> CfbSnapshot:
    """Download one CFB source into an immutable season-partitioned snapshot."""

    spec = cfb_source_spec(source)
    requested = _validate_requested_seasons(spec, seasons)
    identifier = snapshot_id or run_id()
    destination = cfb_root / spec.key / "raw" / identifier
    if destination.exists():
        raise FileExistsError(f"CFB {spec.key} snapshot already exists: {destination}")
    if spec.origin == "cfbd_api":
        return _fetch_cfbd_snapshot(spec, requested, destination, identifier)
    if spec.key == "lines":
        return _fetch_lines_snapshot(spec, requested, destination, identifier)

    provenance, files = _season_source_urls(spec, requested)
    snapshot = CfbSnapshot(spec.key, identifier, destination, tuple(requested))
    partitions: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"per_season": {}}
    for season in requested:
        entry = files[season]
        payload = _download(str(entry["url"]), f"CFB {spec.key} season {season}")
        source_path = destination / "source" / str(entry["name"])
        _atomic_bytes(payload, source_path)
        frame = _read_parquet_bytes(payload, f"CFB {spec.key} season {season}")
        canonical, season_audit = _canonicalize_season(spec, frame, season)
        path = snapshot.season_path(season)
        atomic_parquet(canonical, path)
        source_file = {
            "path": str(source_path.relative_to(destination)),
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
            "url": str(entry["url"]),
        }
        if "updated_at" in entry:
            source_file["updated_at"] = str(entry["updated_at"])
        partitions.append(
            {
                "season": season,
                "rows": len(canonical),
                "path": str(path.relative_to(destination)),
                "sha256": _sha256_file(path),
                "source_file": source_file,
            }
        )
        audit["per_season"][str(season)] = season_audit
    _write_manifest(spec, snapshot, provenance, partitions, audit)
    return snapshot


def plan_cfb_ingest(source: str, seasons: list[int]) -> dict[str, Any]:
    """Resolve upstream files and sizes for an ingest without downloading data."""

    spec = cfb_source_spec(source)
    requested = _validate_requested_seasons(spec, seasons)
    files: list[dict[str, Any]]
    provenance: dict[str, Any]
    if spec.origin == "cfbd_api":
        provenance = dict(resolve_cfbd_api())
        if spec.single_call:
            files = [
                {
                    "season": None,
                    "endpoint": spec.api_path,
                    "params": {},
                    "url": f"{CFBD_API_ROOT}{spec.api_path}",
                }
            ]
        else:
            files = [
                {
                    "season": season,
                    "endpoint": spec.api_path,
                    "params": {spec.season_param: season},
                    "url": (
                        f"{CFBD_API_ROOT}{spec.api_path}"
                        f"?{urllib.parse.urlencode({spec.season_param: season})}"
                    ),
                }
                for season in requested
            ]
        return {
            "source": spec.key,
            "seasons": requested,
            "dry_run": True,
            "provenance": provenance,
            "files": files,
            "api_calls_required": len(files),
            "api_key_configured": bool(os.environ.get(CFBD_API_KEY_ENV, "").strip()),
        }
    if spec.origin == "cfbfastr_raw":
        provenance = resolve_cfbfastr_commit()
        listing = _github_json(
            f"{GITHUB_API_ROOT}/repos/{CFBFASTR_DATA_REPOSITORY}/contents/"
            f"{spec.raw_directory}?ref={provenance['commit_sha']}"
        )
        sizes = {str(entry["name"]): int(entry["size"]) for entry in listing}
        needed = (
            {requested[0]: str(spec.raw_name_pattern)}
            if spec.key == "lines"
            else {season: str(spec.raw_name_pattern).format(season=season) for season in requested}
        )
        missing = sorted(name for name in needed.values() if name not in sizes)
        if missing:
            raise DataContractError(
                f"cfbfastR-data is missing files at the pinned commit: {', '.join(missing)}"
            )
        files = [
            {
                "season": None if spec.key == "lines" else season,
                "name": name,
                "bytes": sizes[name],
                "url": _cfbfastr_raw_url(provenance["commit_sha"], f"{spec.raw_directory}/{name}"),
            }
            for season, name in needed.items()
        ]
    else:
        release = resolve_release_assets(str(spec.release_tag))
        provenance = {key: value for key, value in release.items() if key != "assets"}
        files = []
        for season in requested:
            name, asset = _release_asset_for_season(release, spec, season)
            files.append(
                {
                    "season": season,
                    "name": name,
                    "bytes": int(asset["bytes"]),
                    "updated_at": str(asset["updated_at"]),
                    "url": str(asset["download_url"]),
                }
            )
    return {
        "source": spec.key,
        "seasons": requested,
        "dry_run": True,
        "provenance": provenance,
        "files": files,
        "total_bytes": sum(int(entry["bytes"]) for entry in files),
    }


def cfb_snapshot_from_root(root: Path) -> CfbSnapshot:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CFB manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return CfbSnapshot(
        str(payload["cfb_source"]),
        str(payload["snapshot_id"]),
        root,
        tuple(int(season) for season in payload["seasons"]),
    )


def latest_cfb_snapshot(cfb_root: Path, source: str) -> CfbSnapshot:
    spec = cfb_source_spec(source)
    manifests = sorted((cfb_root / spec.key / "raw").glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No CFB {spec.key} snapshots found in {cfb_root}")
    return cfb_snapshot_from_root(manifests[-1].parent)


def load_cfb_snapshot(snapshot: CfbSnapshot) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in snapshot.seasons:
        path = snapshot.season_path(season)
        if not path.is_file():
            raise FileNotFoundError(f"Missing CFB {snapshot.source} partition: {path}")
        frames.append(pd.read_parquet(path))
    result = pd.concat(frames, ignore_index=True)
    if snapshot.source == "rosters":
        assert_no_quarantined_roster_columns(result)
    return result


def summarize_cfb_snapshots(cfb_root: Path) -> dict[str, Any]:
    """Report the latest snapshot per CFB source for the cfb-summary command."""

    summary: dict[str, Any] = {}
    for key in sorted(CFB_SOURCES):
        try:
            snapshot = latest_cfb_snapshot(cfb_root, key)
        except FileNotFoundError:
            summary[key] = None
            continue
        manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
        summary[key] = {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "created_at_utc": manifest["created_at_utc"],
            "seasons": manifest["seasons"],
            "rows": manifest["rows"],
            "contract_version": manifest["contract_version"],
        }
    return summary
