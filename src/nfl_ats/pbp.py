"""Versioned nflverse play-by-play ingestion and leak-safe feature building."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    DRIVE_STATE_METRICS,
    PBP_ENRICHMENT_STATE_METRICS,
    PBP_STATE_METRICS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.opponent_adjustment import add_opponent_adjusted_pbp_features

PBP_FILTER_VERSION = "v1"
PBP_FEATURE_VERSION = "v3"

# nflverse spells "postseason" differently across the feeds this package
# ingests. Verified against nflreadpy for 2015/2019/2021/2023/2024:
#
#   load_injuries / load_rosters_weekly / load_snap_counts -> ``game_type``
#       carries the per-round codes REG, WC, DIV, CON, SB
#   load_pbp / load_player_stats -> ``season_type`` carries REG and POST
#
# Both spellings are accepted wherever a season scope is applied so a feed that
# switches spelling keeps working. PRE is a recognized code that is never kept:
# preseason has always been out of contract.
REGULAR_SEASON_CODE = "REG"
PRESEASON_CODE = "PRE"
POSTSEASON_CODES = ("WC", "DIV", "CON", "SB", "POST")
IN_CONTRACT_SEASON_CODES = frozenset((REGULAR_SEASON_CODE, *POSTSEASON_CODES))
KNOWN_SEASON_CODES = frozenset((PRESEASON_CODE, *IN_CONTRACT_SEASON_CODES))


def season_scope_mask(
    values: pd.Series,
    *,
    include_postseason: bool,
    dataset: str,
    column: str,
) -> pd.Series:
    """Select the rows a snapshot keeps for the requested season scope.

    With ``include_postseason`` unset this is exactly the historical
    ``values == "REG"`` comparison, down to its silent treatment of every other
    value, so regular-season snapshots stay bit-identical. The opt-in path is
    strict instead: an unrecognized code raises rather than being kept as
    garbage or dropped without comment.
    """

    if not include_postseason:
        return values.eq(REGULAR_SEASON_CODE)
    codes = values.astype("string")
    unknown = sorted(set(codes.dropna().unique()).difference(KNOWN_SEASON_CODES))
    if unknown:
        raise DataContractError(
            f"{dataset} {column} contains unrecognized season codes: {', '.join(unknown)}"
        )
    return codes.isin(sorted(IN_CONTRACT_SEASON_CODES)).fillna(False).astype(bool)


PBP_REQUIRED_COLUMNS = (
    "play_id",
    "game_id",
    "season",
    "season_type",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "fixed_drive",
    "down",
    "play_type",
    "yards_gained",
    "pass_attempt",
    "rush_attempt",
    "sack",
    "qb_hit",
    "epa",
    "success",
    "wp",
)

# The raw nflverse table is deliberately narrowed at ingestion. These fields
# are sufficient to reproduce the v1 play filter, drives, team efficiencies,
# and a future quarterback layer without persisting hundreds of unused fields.
PBP_SNAPSHOT_COLUMNS = (
    *PBP_REQUIRED_COLUMNS,
    "qtr",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
    "aborted_play",
    "complete_pass",
    "interception",
    "fumble_lost",
    "touchdown",
    "first_down",
    "passer_player_id",
    "passer_player_name",
    "cpoe",
    "xpass",
    "pass_oe",
    "score_differential",
    "penalty",
    "penalty_yards",
    "fixed_drive_result",
    "posteam_score",
    "posteam_score_post",
    "play",
)


@dataclass(frozen=True)
class PbpSnapshot:
    """An immutable, season-partitioned play-by-play snapshot."""

    snapshot_id: str
    root: Path
    seasons: tuple[int, ...]

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def season_path(self, season: int) -> Path:
        return self.root / f"season={season}" / "plays.parquet"


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        converted = frame.to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted
    raise TypeError(f"Unsupported dataframe type: {type(frame)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pbp(frame: pd.DataFrame, season: int | None = None) -> None:
    """Fail loudly when nflverse changes fields needed by the v1 contract."""

    require_columns(frame, PBP_REQUIRED_COLUMNS, "play_by_play")
    if frame[["game_id", "play_id"]].isna().any(axis=None):
        raise DataContractError("play_by_play contains null game_id or play_id values")
    if season is not None:
        observed = set(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int))
        if observed and observed != {season}:
            raise DataContractError(
                f"play_by_play season partition {season} contains seasons {sorted(observed)}"
            )
    duplicated = frame.duplicated(["game_id", "play_id"])
    if duplicated.any():
        raise DataContractError("play_by_play contains duplicate game_id/play_id rows")


def canonicalize_pbp(
    frame: pd.DataFrame,
    season: int | None = None,
    *,
    include_postseason: bool = False,
) -> pd.DataFrame:
    """Select and normalize the stable v1 play-by-play storage contract.

    ``include_postseason`` widens the stored scope to REG plus POST plays so a
    snapshot can serve playoff predictions. It changes what is written, never
    what is read: every read path re-applies the regular-season filter by
    default, so the frozen model's feature values are unaffected.
    """

    validate_pbp(frame, season=season)
    result = frame.copy()
    for column in PBP_SNAPSHOT_COLUMNS:
        if column not in result:
            result[column] = np.nan
    keep = season_scope_mask(
        result["season_type"],
        include_postseason=include_postseason,
        dataset="play_by_play",
        column="season_type",
    )
    result = result.loc[keep, PBP_SNAPSHOT_COLUMNS].copy()
    for column in ("season", "week"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    return result.sort_values(["season", "week", "game_id", "play_id"]).reset_index(drop=True)


def write_pbp_snapshot(
    season_frames: dict[int, pd.DataFrame],
    raw_root: Path,
    snapshot_id: str | None = None,
    *,
    include_postseason: bool = False,
) -> PbpSnapshot:
    """Write season partitions and a manifest, publishing the manifest last."""

    seasons = tuple(sorted(season_frames))
    if not seasons:
        raise ValueError("At least one play-by-play season is required")
    identifier = snapshot_id or run_id()
    destination = raw_root / identifier
    if destination.exists():
        raise FileExistsError(f"Play-by-play snapshot already exists: {destination}")

    partitions: list[dict[str, Any]] = []
    for season in seasons:
        canonical = canonicalize_pbp(
            season_frames[season], season=season, include_postseason=include_postseason
        )
        path = destination / f"season={season}" / "plays.parquet"
        atomic_parquet(canonical, path)
        partitions.append(
            {
                "season": season,
                "rows": len(canonical),
                "path": str(path.relative_to(destination)),
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "snapshot_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "nflverse play-by-play via nflreadpy",
        "seasons": list(seasons),
        "include_postseason": bool(include_postseason),
        "filter_version": PBP_FILTER_VERSION,
        "feature_version": PBP_FEATURE_VERSION,
        "columns": list(PBP_SNAPSHOT_COLUMNS),
        "partitions": partitions,
        "rows": sum(int(partition["rows"]) for partition in partitions),
    }
    atomic_json(manifest, destination / "manifest.json")
    return PbpSnapshot(identifier, destination, seasons)


def fetch_pbp_snapshot(
    seasons: list[int],
    raw_root: Path,
    *,
    include_postseason: bool = False,
) -> PbpSnapshot:
    """Download nflverse PBP one season at a time to bound memory use."""

    if not seasons or seasons != sorted(set(seasons)):
        raise ValueError("Seasons must be non-empty, unique, and sorted")
    import nflreadpy as nfl

    identifier = run_id()
    destination = raw_root / identifier
    partitions: list[dict[str, Any]] = []
    for season in seasons:
        canonical = canonicalize_pbp(
            _to_pandas(nfl.load_pbp(seasons=[season])),
            season=season,
            include_postseason=include_postseason,
        )
        path = destination / f"season={season}" / "plays.parquet"
        atomic_parquet(canonical, path)
        partitions.append(
            {
                "season": season,
                "rows": len(canonical),
                "path": str(path.relative_to(destination)),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "snapshot_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "nflverse play-by-play via nflreadpy",
        "seasons": seasons,
        "include_postseason": bool(include_postseason),
        "filter_version": PBP_FILTER_VERSION,
        "feature_version": PBP_FEATURE_VERSION,
        "columns": list(PBP_SNAPSHOT_COLUMNS),
        "partitions": partitions,
        "rows": sum(int(partition["rows"]) for partition in partitions),
    }
    atomic_json(manifest, destination / "manifest.json")
    return PbpSnapshot(identifier, destination, tuple(seasons))


def snapshot_from_root(root: Path) -> PbpSnapshot:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Play-by-play manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    seasons = tuple(int(season) for season in payload["seasons"])
    return PbpSnapshot(str(payload["snapshot_id"]), root, seasons)


def latest_pbp_snapshot(raw_root: Path) -> PbpSnapshot:
    manifests = sorted(raw_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No play-by-play snapshots found in {raw_root}")
    return snapshot_from_root(manifests[-1].parent)


def load_pbp_snapshot(snapshot: PbpSnapshot, *, include_postseason: bool = False) -> pd.DataFrame:
    """Read a snapshot's plays, regular season only unless asked otherwise.

    ``canonicalize_pbp`` runs at write time, so a postseason-inclusive snapshot
    on disk would otherwise feed POST plays straight into
    ``enrich_with_pbp_features``, ``build_qb_game_metrics``, and the
    participation ratings that join against these plays. Re-applying the scope
    here keeps every existing feature build regular-season only regardless of
    what the snapshot contains; callers that want playoff plays opt in.
    """

    frames: list[pd.DataFrame] = []
    for season in snapshot.seasons:
        path = snapshot.season_path(season)
        if not path.is_file():
            raise FileNotFoundError(f"Missing play-by-play partition: {path}")
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=PBP_SNAPSHOT_COLUMNS)
    plays = pd.concat(frames, ignore_index=True)
    if "season_type" not in plays.columns:
        return plays
    keep = season_scope_mask(
        plays["season_type"],
        include_postseason=include_postseason,
        dataset="play_by_play snapshot",
        column="season_type",
    )
    return plays.loc[keep].reset_index(drop=True)


def analysis_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented v1 efficiency filter.

    The filter keeps real scrimmage plays with an offense, EPA, and win
    probability; removes kneels, spikes, aborted plays, and no-plays; and marks
    the 5%-95% win-probability subset used for efficiency aggregation.
    """

    require_columns(pbp, PBP_SNAPSHOT_COLUMNS, "play_by_play snapshot")
    result = pbp.copy()
    numeric = (
        "play",
        "epa",
        "wp",
        "qb_kneel",
        "qb_spike",
        "aborted_play",
        "pass_attempt",
        "rush_attempt",
    )
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    eligible = (
        result["posteam"].notna()
        & result["defteam"].notna()
        & result["epa"].notna()
        & result["wp"].notna()
        & result["play"].fillna(1).eq(1)
        & result["qb_kneel"].fillna(0).eq(0)
        & result["qb_spike"].fillna(0).eq(0)
        & result["aborted_play"].fillna(0).eq(0)
        & (result["pass_attempt"].fillna(0).eq(1) | result["rush_attempt"].fillna(0).eq(1))
    )
    result = result.loc[eligible].copy()
    result["competitive_play"] = result["wp"].between(0.05, 0.95, inclusive="both")
    return result.reset_index(drop=True)


def build_drive_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per offensive drive from eligible v1 plays."""

    plays = analysis_plays(pbp)
    if plays.empty:
        return pd.DataFrame()
    plays["yardline_100"] = pd.to_numeric(plays["yardline_100"], errors="coerce")
    plays["yards_gained"] = pd.to_numeric(plays["yards_gained"], errors="coerce")
    plays["game_seconds_remaining"] = pd.to_numeric(
        plays["game_seconds_remaining"], errors="coerce"
    )
    plays["posteam_score"] = pd.to_numeric(plays["posteam_score"], errors="coerce")
    plays["posteam_score_post"] = pd.to_numeric(plays["posteam_score_post"], errors="coerce")
    plays["drive_turnover_play"] = (
        pd.to_numeric(plays["interception"], errors="coerce").fillna(0).eq(1)
        | pd.to_numeric(plays["fumble_lost"], errors="coerce").fillna(0).eq(1)
    ).astype(float)
    drives = (
        plays.sort_values(["game_id", "fixed_drive", "play_id"])
        .groupby(
            ["game_id", "season", "week", "posteam", "defteam", "fixed_drive"],
            dropna=False,
            sort=False,
        )
        .agg(
            plays=("play_id", "size"),
            drive_epa=("epa", "sum"),
            drive_success_rate=("success", "mean"),
            drive_yards=("yards_gained", "sum"),
            start_yardline_100=("yardline_100", "first"),
            result=("fixed_drive_result", "last"),
            start_game_seconds=("game_seconds_remaining", "first"),
            end_game_seconds=("game_seconds_remaining", "last"),
            start_score=("posteam_score", "first"),
            end_score=("posteam_score_post", "last"),
            turnover_play=("drive_turnover_play", "max"),
        )
        .reset_index()
    )
    drives["drive_seconds"] = (drives["start_game_seconds"] - drives["end_game_seconds"]).clip(
        lower=0
    )
    drives["drive_points"] = (drives["end_score"] - drives["start_score"]).clip(lower=0)
    result_text = drives["result"].astype("string").str.lower()
    result_turnover = result_text.str.contains("interception|fumble|turnover", regex=True, na=False)
    drives["drive_turnover"] = drives["turnover_play"].eq(1) | result_turnover
    drives["drive_scoring"] = drives["drive_points"].gt(0)
    return drives


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> float:
    denominator_sum = float(denominator.fillna(0).sum())
    if denominator_sum <= 0:
        return math.nan
    return float(numerator.fillna(0).sum() / denominator_sum)


def build_pbp_team_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Build offense and opponent-derived defense metrics per completed game."""

    plays = analysis_plays(pbp)
    plays = plays.loc[plays["competitive_play"]].copy()
    if plays.empty:
        return pd.DataFrame(
            columns=(
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
                *PBP_STATE_METRICS,
                *DRIVE_STATE_METRICS,
            )
        )
    plays["posteam"] = plays["posteam"].replace(TEAM_ABBREVIATION_ALIASES)
    plays["defteam"] = plays["defteam"].replace(TEAM_ABBREVIATION_ALIASES)
    for column in (
        "down",
        "yards_gained",
        "pass_attempt",
        "rush_attempt",
        "qb_dropback",
        "sack",
        "qb_hit",
        "epa",
        "success",
        "pass_oe",
        "yardline_100",
    ):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    plays["early_down"] = plays["down"].isin([1.0, 2.0])
    plays["explosive"] = (
        (plays["pass_attempt"].eq(1) & plays["yards_gained"].ge(20))
        | (plays["rush_attempt"].eq(1) & plays["yards_gained"].ge(10))
    ).astype(float)
    plays["pressure"] = (plays["qb_hit"].fillna(0).eq(1) | plays["sack"].fillna(0).eq(1)).astype(
        float
    )

    drives = build_drive_table(pbp)
    drive_summary = (
        drives.groupby(["game_id", "posteam"], sort=False)
        .agg(
            pbp_drives=("fixed_drive", "nunique"),
            pbp_start_yardline_100=("start_yardline_100", "mean"),
            drive_points_per_drive=("drive_points", "mean"),
            drive_yards_per_drive=("drive_yards", "mean"),
            drive_plays_per_drive=("plays", "mean"),
            drive_seconds_per_drive=("drive_seconds", "mean"),
            drive_scoring_rate=("drive_scoring", "mean"),
            drive_turnover_rate=("drive_turnover", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    rows: list[dict[str, Any]] = []
    for (game_id, team), group in plays.groupby(["game_id", "posteam"], sort=False):
        early = group.loc[group["early_down"]]
        pass_rush = group["pass_attempt"].fillna(0) + group["rush_attempt"].fillna(0)
        dropbacks = group["qb_dropback"].fillna(0)
        rows.append(
            {
                "game_id": game_id,
                "season": int(group["season"].iloc[0]),
                "week": int(group["week"].iloc[0]),
                "team": team,
                "opponent": group["defteam"].mode().iloc[0],
                "pbp_off_epa_per_play": group["epa"].mean(),
                "pbp_off_early_down_epa": early["epa"].mean(),
                "pbp_off_success_rate": group["success"].mean(),
                "pbp_off_explosive_rate": group["explosive"].mean(),
                "pbp_off_pass_rate": _safe_rate(group["pass_attempt"], pass_rush),
                "pbp_off_proe": group["pass_oe"].mean(),
                "pbp_pressure_allowed_rate": _safe_rate(group["pressure"], dropbacks),
                "pbp_sack_allowed_rate": _safe_rate(group["sack"], dropbacks),
            }
        )
    offense = pd.DataFrame(rows).merge(
        drive_summary, on=["game_id", "team"], how="left", validate="one_to_one"
    )
    pair_counts = offense.groupby("game_id")["team"].nunique()
    malformed = pair_counts.loc[pair_counts != 2]
    if not malformed.empty:
        examples = ", ".join(malformed.index.astype(str).tolist()[:5])
        raise DataContractError(f"Expected two PBP team rows per game; malformed: {examples}")

    opponent = offense[
        [
            "game_id",
            "team",
            "pbp_off_epa_per_play",
            "pbp_off_early_down_epa",
            "pbp_off_success_rate",
            "pbp_off_explosive_rate",
            "pbp_pressure_allowed_rate",
            "pbp_sack_allowed_rate",
            "drive_points_per_drive",
            "drive_yards_per_drive",
            "drive_plays_per_drive",
            "drive_seconds_per_drive",
            "drive_scoring_rate",
            "drive_turnover_rate",
        ]
    ].rename(
        columns={
            "team": "paired_opponent",
            "pbp_off_epa_per_play": "pbp_def_epa_per_play",
            "pbp_off_early_down_epa": "pbp_def_early_down_epa",
            "pbp_off_success_rate": "pbp_def_success_rate_allowed",
            "pbp_off_explosive_rate": "pbp_def_explosive_rate_allowed",
            "pbp_pressure_allowed_rate": "pbp_pressure_rate",
            "pbp_sack_allowed_rate": "pbp_sack_rate",
            "drive_points_per_drive": "drive_points_per_drive_allowed",
            "drive_yards_per_drive": "drive_yards_per_drive_allowed",
            "drive_plays_per_drive": "drive_plays_per_drive_allowed",
            "drive_seconds_per_drive": "drive_seconds_per_drive_allowed",
            "drive_scoring_rate": "drive_scoring_rate_allowed",
            "drive_turnover_rate": "drive_takeaway_rate",
        }
    )
    paired = offense.merge(opponent, on="game_id", how="inner", validate="many_to_many")
    paired = paired.loc[paired["team"].ne(paired["paired_opponent"])].copy()
    if len(paired) != len(offense):
        raise DataContractError("Unable to pair each PBP team row with exactly one opponent")
    return paired[
        [
            "game_id",
            "season",
            "week",
            "team",
            "opponent",
            *PBP_STATE_METRICS,
            *DRIVE_STATE_METRICS,
        ]
    ].sort_values(["season", "week", "game_id", "team"])


def _build_pbp_states(
    team_games: pd.DataFrame,
    games: pd.DataFrame,
    span: int,
    min_periods: int,
    offseason_retention: float,
) -> pd.DataFrame:
    game_dates = games[["game_id", "gameday"]].drop_duplicates("game_id")
    states = team_games.merge(game_dates, on="game_id", how="left", validate="many_to_one")
    if states["gameday"].isna().any():
        raise DataContractError("PBP contains games absent from the canonical schedule table")
    states["gameday"] = pd.to_datetime(states["gameday"], errors="raise")
    states = states.sort_values(["team", "gameday", "game_id"])
    alpha = 2.0 / (span + 1.0)
    for metric in PBP_ENRICHMENT_STATE_METRICS:
        states[metric] = pd.to_numeric(states[metric], errors="coerce")
        league_means = states.groupby("season", sort=False)[metric].mean().to_dict()
        states[f"league_mean_{metric}"] = states["season"].map(league_means)
        output = pd.Series(np.nan, index=states.index, dtype="float64")
        for _, group in states.groupby("team", sort=False):
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
                        else alpha * value + (1 - alpha) * current
                    )
                    observations += 1
                if observations >= min_periods:
                    output.at[index] = current
                previous_season = season
        states[f"state_{metric}"] = output
    return states


def enrich_with_pbp_features(
    games: pd.DataFrame,
    pbp: pd.DataFrame,
    span: int = 8,
    min_periods: int = 3,
    offseason_retention: float = 0.67,
    opponent_half_life_weeks: float = 16.0,
    opponent_ridge_alpha: float = 10.0,
    opponent_min_team_games: int = 64,
) -> pd.DataFrame:
    """Attach PBP states from games strictly earlier than the game being scored."""

    if span < 2:
        raise ValueError("span must be at least 2")
    if min_periods < 1:
        raise ValueError("min_periods must be positive")
    if not 0 <= offseason_retention <= 1:
        raise ValueError("offseason_retention must be between 0 and 1")
    required_games = {"game_id", "season", "week", "gameday", "home_team", "away_team"}
    missing = sorted(required_games.difference(games.columns))
    if missing:
        raise DataContractError(f"game features missing columns: {', '.join(missing)}")

    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    metrics = build_pbp_team_game_metrics(pbp)
    states = _build_pbp_states(metrics, result, span, min_periods, offseason_retention)
    groups = {
        str(team): group.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        for team, group in states.groupby("team", sort=False)
    }
    for side in ("home", "away"):
        for metric in PBP_ENRICHMENT_STATE_METRICS:
            result[f"{side}_{metric}"] = np.nan
        for index, game in result.iterrows():
            history = groups.get(str(game[f"{side}_team"]))
            if history is None or history.empty:
                continue
            dates = history["gameday"].to_numpy(dtype="datetime64[ns]")
            position = (
                int(
                    np.searchsorted(
                        dates,
                        np.datetime64(pd.Timestamp(game["gameday"]), "ns"),
                        side="left",
                    )
                )
                - 1
            )
            if position < 0:
                continue
            state = history.iloc[position]
            season_gap = int(game["season"]) - int(state["season"])
            for metric in PBP_ENRICHMENT_STATE_METRICS:
                value = state[f"state_{metric}"]
                if season_gap > 0 and pd.notna(value):
                    league_mean = state[f"league_mean_{metric}"]
                    retention = offseason_retention ** max(1, season_gap)
                    value = league_mean + retention * (value - league_mean)
                result.at[index, f"{side}_{metric}"] = value
    for metric in PBP_ENRICHMENT_STATE_METRICS:
        result[f"diff_{metric}"] = result[f"home_{metric}"] - result[f"away_{metric}"]
    result = add_opponent_adjusted_pbp_features(
        result,
        metrics,
        half_life_weeks=opponent_half_life_weeks,
        ridge_alpha=opponent_ridge_alpha,
        min_team_games=opponent_min_team_games,
    )
    result["pbp_feature_version"] = PBP_FEATURE_VERSION
    return result.sort_values(["gameday", "game_id"]).reset_index(drop=True)
