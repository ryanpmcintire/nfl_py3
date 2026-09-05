"""Point-in-time quarterback depth charts and prior-performance states."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.availability import (
    availability_rate_lookup,
    canonicalize_availability_rates,
    resolve_unavailability,
)
from nfl_ats.constants import (
    QB_DEPTH_STATE_METRICS,
    QB_STATE_METRICS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.nfl_week import week_cycle_sunday
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS, analysis_plays

DEPTH_CHART_VERSION = "v1"
HISTORICAL_DEPTH_CHART_VERSION = "v1-prior-week-effective"
QB_FEATURE_VERSION = "v2-named-depth-availability"
DEPTH_REQUIRED_COLUMNS = (
    "dt",
    "team",
    "player_name",
    "gsis_id",
    "pos_abb",
    "pos_rank",
)
DEPTH_SNAPSHOT_COLUMNS = (
    *DEPTH_REQUIRED_COLUMNS,
    "espn_id",
    "pos_slot",
)

QB_AVAILABILITY_COLUMNS = (
    "season",
    "week",
    "team",
    "gsis_id",
    "position",
    "report_status",
    "practice_status",
    "date_modified",
)

#: ENG-39: duplicated (not imported) from ``nfl_ats.players.
#: INJURY_PROXY_HOURS_BEFORE_KICKOFF`` -- importing it back would be a
#: circular import (``players`` already imports from this module), and this
#: module's module docstring convention (see ``TEAM_ABBREVIATION_ALIASES``
#: usage patterns elsewhere in the codebase, e.g.
#: ``transaction_wire_features.kickoff_utc``) is to duplicate a small,
#: cross-module-shared constant/helper rather than restructure imports.
QB_INJURY_PROXY_HOURS_BEFORE_KICKOFF = 24
_QB_EASTERN = ZoneInfo("America/New_York")

LEGACY_DEPTH_REQUIRED_COLUMNS = (
    "season",
    "club_code",
    "week",
    "game_type",
    "depth_team",
    "gsis_id",
    "position",
    "formation",
    "depth_position",
    "full_name",
)


@dataclass(frozen=True)
class DepthSnapshot:
    snapshot_id: str
    root: Path
    requested_seasons: tuple[int, ...]

    @property
    def data_path(self) -> Path:
        return self.root / "quarterbacks.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"


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


def canonicalize_depth_charts(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep timestamped quarterback depth rows and normalize the as-of contract."""

    require_columns(frame, DEPTH_REQUIRED_COLUMNS, "depth_charts")
    result = frame.copy()
    for column in DEPTH_SNAPSHOT_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    result = result.loc[result["pos_abb"].astype("string").eq("QB"), DEPTH_SNAPSHOT_COLUMNS].copy()
    result["observed_at_utc"] = pd.to_datetime(result.pop("dt"), errors="coerce", utc=True)
    result["effective_at_utc"] = result["observed_at_utc"]
    result["provenance_mode"] = "timestamped_revision"
    result["pos_rank"] = pd.to_numeric(result["pos_rank"], errors="coerce")
    result = result.loc[
        result["observed_at_utc"].notna()
        & result["team"].notna()
        & result["gsis_id"].notna()
        & result["pos_rank"].notna()
    ].copy()
    result["team"] = result["team"].astype(str)
    result["gsis_id"] = result["gsis_id"].astype(str)
    result["pos_rank"] = result["pos_rank"].astype(int)
    duplicated = result.duplicated(["observed_at_utc", "team", "gsis_id"])
    if duplicated.any():
        result = result.loc[~duplicated].copy()
    return result.sort_values(["observed_at_utc", "team", "pos_rank", "gsis_id"]).reset_index(
        drop=True
    )


def canonicalize_historical_depth_charts(frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Convert legacy week-level rows to a conservative prior-week timeline.

    The source has no publication timestamp. A source-week row therefore
    becomes effective only one microsecond after the final kickoff in that NFL
    week. It can describe later games, never a game in its own labeled week.
    ``observed_at_utc`` remains explicitly null rather than inventing a time.
    """

    require_columns(frame, LEGACY_DEPTH_REQUIRED_COLUMNS, "historical depth charts")
    require_columns(
        games,
        ("season", "week", "kickoff", "home_team", "away_team"),
        "games for historical depth chronology",
    )
    result = frame.loc[:, list(LEGACY_DEPTH_REQUIRED_COLUMNS)].copy()
    result = result.loc[
        result["game_type"].astype("string").eq("REG")
        & result["position"].astype("string").str.upper().eq("QB")
        & result["formation"].astype("string").eq("Offense")
        & result["depth_position"].astype("string").eq("QB")
    ].copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["week"] = pd.to_numeric(result["week"], errors="coerce")
    result["pos_rank"] = pd.to_numeric(result["depth_team"], errors="coerce")
    result["team"] = result["club_code"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    result["gsis_id"] = result["gsis_id"].astype("string")
    result["player_name"] = result["full_name"].astype("string")
    result = result.loc[
        result["season"].notna()
        & result["week"].notna()
        & result["pos_rank"].notna()
        & result["team"].notna()
        & result["gsis_id"].notna()
    ].copy()
    result[["season", "week", "pos_rank"]] = result[["season", "week", "pos_rank"]].astype(int)

    schedule = games.loc[:, ["season", "week", "kickoff"]].copy()
    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
    schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce")
    schedule["kickoff"] = pd.to_datetime(schedule["kickoff"], errors="coerce", utc=True)
    schedule = schedule.dropna(subset=["season", "week", "kickoff"])
    schedule[["season", "week"]] = schedule[["season", "week"]].astype(int)
    completed = schedule.groupby(["season", "week"], as_index=False, observed=True).agg(
        effective_at_utc=("kickoff", "max")
    )
    completed["effective_at_utc"] += pd.Timedelta(microseconds=1)
    result = result.merge(completed, on=["season", "week"], how="left", validate="many_to_one")
    if result["effective_at_utc"].isna().any():
        gaps = (
            result.loc[result["effective_at_utc"].isna(), ["season", "week"]]
            .drop_duplicates()
            .sort_values(["season", "week"])
            .to_dict(orient="records")
        )
        raise DataContractError(f"Historical depth rows lack completed-week chronology: {gaps}")
    result["observed_at_utc"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["provenance_mode"] = "legacy_prior_week"
    result = result.rename(
        columns={
            "season": "source_season",
            "week": "source_week",
            "game_type": "source_game_type",
        }
    )
    columns = (
        "team",
        "player_name",
        "gsis_id",
        "pos_rank",
        "observed_at_utc",
        "effective_at_utc",
        "source_season",
        "source_week",
        "source_game_type",
        "provenance_mode",
        "source_role_conflict",
    )
    identity_key = ["source_season", "source_week", "team", "gsis_id"]
    result["source_role_conflict"] = result.duplicated(identity_key, keep=False)
    # Nineteen official team-week/player keys in 2009-2024 are duplicated,
    # including ten that assign the same player multiple ranks. A player
    # cannot fill QB1 and QB2 simultaneously, so collapse to the best listed
    # rank and retain an explicit conflict flag instead of manufacturing a
    # second identity.
    result = (
        result.loc[:, list(columns)]
        .sort_values([*identity_key, "pos_rank"])
        .drop_duplicates(identity_key, keep="first")
    )
    return result.sort_values(["effective_at_utc", "team", "pos_rank", "gsis_id"]).reset_index(
        drop=True
    )


def write_depth_snapshot(
    frame: pd.DataFrame,
    raw_root: Path,
    requested_seasons: list[int],
    snapshot_id: str | None = None,
) -> DepthSnapshot:
    if not requested_seasons or requested_seasons != sorted(set(requested_seasons)):
        raise ValueError("Requested seasons must be non-empty, unique, and sorted")
    identifier = snapshot_id or run_id()
    destination = raw_root / identifier
    if destination.exists():
        raise FileExistsError(f"Depth-chart snapshot already exists: {destination}")
    canonical = canonicalize_depth_charts(frame)
    if canonical.empty:
        raise DataContractError(
            "Depth charts contain no timestamped quarterback rows; historical weekly rows "
            "without observation timestamps cannot be used as point-in-time inputs"
        )
    path = destination / "quarterbacks.parquet"
    atomic_parquet(canonical, path)
    manifest = {
        "snapshot_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "nflverse depth charts via nflreadpy",
        "requested_seasons": requested_seasons,
        "contract_version": DEPTH_CHART_VERSION,
        "rows": len(canonical),
        "teams": int(canonical["team"].nunique()),
        "first_observation": canonical["observed_at_utc"].min().isoformat(),
        "last_observation": canonical["observed_at_utc"].max().isoformat(),
        "sha256": _sha256(path),
        "columns": canonical.columns.tolist(),
    }
    atomic_json(manifest, destination / "manifest.json")
    return DepthSnapshot(identifier, destination, tuple(requested_seasons))


def fetch_depth_snapshot(seasons: list[int], raw_root: Path) -> DepthSnapshot:
    if not seasons or seasons != sorted(set(seasons)):
        raise ValueError("Seasons must be non-empty, unique, and sorted")
    import nflreadpy as nfl

    frame = _to_pandas(nfl.load_depth_charts(seasons=seasons))
    return write_depth_snapshot(frame, raw_root, seasons)


def write_historical_depth_snapshot(
    frame: pd.DataFrame,
    games: pd.DataFrame,
    raw_root: Path,
    requested_seasons: list[int],
    *,
    games_source: Path | None = None,
    snapshot_id: str | None = None,
) -> DepthSnapshot:
    """Write an immutable legacy archive with explicit conservative visibility."""

    if not requested_seasons or requested_seasons != sorted(set(requested_seasons)):
        raise ValueError("Requested seasons must be non-empty, unique, and sorted")
    canonical = canonicalize_historical_depth_charts(frame, games)
    actual_seasons = sorted(canonical["source_season"].astype(int).unique().tolist())
    if actual_seasons != requested_seasons:
        raise DataContractError(
            f"Historical depth season coverage {actual_seasons} does not match requested "
            f"{requested_seasons}"
        )
    identifier = snapshot_id or run_id()
    destination = raw_root / identifier
    if destination.exists():
        raise FileExistsError(f"Depth-chart snapshot already exists: {destination}")
    path = destination / "quarterbacks.parquet"
    atomic_parquet(canonical, path)
    coverage = (
        canonical.groupby("source_season", observed=True)
        .agg(
            rows=("gsis_id", "size"),
            weeks=("source_week", "nunique"),
            teams=("team", "nunique"),
            quarterbacks=("gsis_id", "nunique"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    manifest = {
        "snapshot_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "nflverse-data annual depth_charts releases via nflreadpy",
        "requested_seasons": requested_seasons,
        "contract_version": HISTORICAL_DEPTH_CHART_VERSION,
        "chronology": {
            "observed_at_utc": "unavailable in legacy source; stored null",
            "effective_at_utc": "one microsecond after final kickoff in source season/week",
            "eligible_targets": "strictly later games only",
        },
        "rows": len(canonical),
        "teams": int(canonical["team"].nunique()),
        "first_effective": canonical["effective_at_utc"].min().isoformat(),
        "last_effective": canonical["effective_at_utc"].max().isoformat(),
        "sha256": _sha256(path),
        "columns": canonical.columns.tolist(),
        "coverage_by_season": coverage,
        "source_role_conflict_rows": int(canonical["source_role_conflict"].sum()),
        "games_source": (
            {"path": str(games_source), "sha256": _sha256(games_source)}
            if games_source is not None
            else None
        ),
    }
    atomic_json(manifest, destination / "manifest.json")
    return DepthSnapshot(identifier, destination, tuple(requested_seasons))


def fetch_historical_depth_snapshot(
    seasons: list[int],
    games: pd.DataFrame,
    raw_root: Path,
    *,
    games_source: Path | None = None,
) -> DepthSnapshot:
    if not seasons or seasons != sorted(set(seasons)):
        raise ValueError("Seasons must be non-empty, unique, and sorted")
    import nflreadpy as nfl

    frame = _to_pandas(nfl.load_depth_charts(seasons=seasons))
    return write_historical_depth_snapshot(
        frame,
        games,
        raw_root,
        seasons,
        games_source=games_source,
    )


def depth_snapshot_from_root(root: Path) -> DepthSnapshot:
    import json

    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Depth-chart manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return DepthSnapshot(
        str(payload["snapshot_id"]),
        root,
        tuple(int(season) for season in payload["requested_seasons"]),
    )


def latest_depth_snapshot(raw_root: Path) -> DepthSnapshot:
    manifests = sorted(raw_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No depth-chart snapshots found in {raw_root}")
    return depth_snapshot_from_root(manifests[-1].parent)


def load_depth_snapshot(snapshot: DepthSnapshot) -> pd.DataFrame:
    if not snapshot.data_path.is_file():
        raise FileNotFoundError(f"Missing depth-chart data: {snapshot.data_path}")
    import json

    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != _sha256(snapshot.data_path):
        raise DataContractError("Depth-chart snapshot hash does not match its manifest")
    return pd.read_parquet(snapshot.data_path)


def _prepare_depth_timeline(depth_history: pd.DataFrame) -> pd.DataFrame:
    required = {"observed_at_utc", "team", "gsis_id", "player_name", "pos_rank"}
    missing = sorted(required.difference(depth_history.columns))
    if missing:
        raise DataContractError(f"Depth history is missing columns: {', '.join(missing)}")
    history = depth_history.copy()
    history["observed_at_utc"] = pd.to_datetime(
        history["observed_at_utc"], errors="coerce", utc=True
    )
    if "effective_at_utc" not in history:
        history["effective_at_utc"] = history["observed_at_utc"]
    history["effective_at_utc"] = pd.to_datetime(
        history["effective_at_utc"], errors="coerce", utc=True
    )
    if history["effective_at_utc"].isna().any():
        raise DataContractError("Depth history contains rows without an effective timestamp")
    observed_later = history["observed_at_utc"].notna() & history["observed_at_utc"].gt(
        history["effective_at_utc"]
    )
    if observed_later.any():
        raise DataContractError("Depth history becomes effective before it was observed")
    return history


def latest_starting_qbs(
    depth_history: pd.DataFrame,
    decision_at: pd.Timestamp,
    *,
    max_age_days: int = 14,
) -> pd.DataFrame:
    """Return rank-one QBs from the latest team observation before a decision."""

    if max_age_days < 1:
        raise ValueError("max_age_days must be positive")
    as_of = pd.Timestamp(decision_at)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    history = _prepare_depth_timeline(depth_history)
    history = history.loc[history["effective_at_utc"].le(as_of)].copy()
    if history.empty:
        return history
    latest_times = history.groupby("team")["effective_at_utc"].transform("max")
    latest = history.loc[history["effective_at_utc"].eq(latest_times)].copy()
    latest = latest.loc[latest["pos_rank"].eq(latest.groupby("team")["pos_rank"].transform("min"))]
    latest["depth_age_days"] = (as_of - latest["effective_at_utc"]).dt.total_seconds() / 86_400
    latest = latest.loc[latest["depth_age_days"].le(max_age_days)].copy()
    return latest.sort_values(["team", "gsis_id"]).drop_duplicates("team")


def build_qb_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate meaningful quarterback appearances from PBP."""

    require_columns(pbp, PBP_SNAPSHOT_COLUMNS, "play_by_play snapshot")
    plays = analysis_plays(pbp)
    plays = plays.loc[plays["passer_player_id"].notna() & plays["qb_dropback"].eq(1)].copy()
    if plays.empty:
        return pd.DataFrame(
            columns=("game_id", "season", "week", "team", "player_id", *QB_STATE_METRICS)
        )
    for column in (
        "qb_dropback",
        "pass_attempt",
        "yards_gained",
        "epa",
        "cpoe",
        "sack",
        "interception",
    ):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    plays["explosive_pass"] = (plays["pass_attempt"].eq(1) & plays["yards_gained"].ge(20)).astype(
        float
    )
    grouped = (
        plays.groupby(
            [
                "game_id",
                "season",
                "week",
                "posteam",
                "passer_player_id",
                "passer_player_name",
            ],
            dropna=False,
            sort=False,
        )
        .agg(
            qb_dropbacks=("qb_dropback", "sum"),
            pass_attempts=("pass_attempt", "sum"),
            total_epa=("epa", "sum"),
            qb_cpoe=("cpoe", "mean"),
            sacks=("sack", "sum"),
            interceptions=("interception", "sum"),
            explosive_passes=("explosive_pass", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "posteam": "team",
                "passer_player_id": "player_id",
                "passer_player_name": "player_name",
            }
        )
    )
    grouped = grouped.loc[grouped["qb_dropbacks"].ge(5)].copy()
    grouped["qb_epa_per_dropback"] = grouped["total_epa"] / grouped["qb_dropbacks"]
    grouped["qb_sack_rate"] = grouped["sacks"] / grouped["qb_dropbacks"]
    grouped["qb_interception_rate"] = grouped["interceptions"] / grouped["pass_attempts"]
    grouped["qb_explosive_pass_rate"] = grouped["explosive_passes"] / grouped["pass_attempts"]
    return grouped[
        [
            "game_id",
            "season",
            "week",
            "team",
            "player_id",
            "player_name",
            "qb_dropbacks",
            *QB_STATE_METRICS,
        ]
    ].sort_values(["season", "week", "game_id", "team", "player_id"])


def build_qb_states(
    qb_games: pd.DataFrame,
    games: pd.DataFrame,
    *,
    span: int = 12,
    min_dropbacks: int = 50,
    offseason_retention: float = 0.75,
) -> pd.DataFrame:
    """Build player states after each appearance, regressing across offseasons."""

    if span < 2 or min_dropbacks < 1:
        raise ValueError("span must be at least 2 and min_dropbacks must be positive")
    dates = games[["game_id", "gameday"]].drop_duplicates("game_id")
    states = qb_games.merge(dates, on="game_id", how="left", validate="many_to_one")
    states = states.loc[states["gameday"].notna()].copy()
    states["gameday"] = pd.to_datetime(states["gameday"], errors="raise")
    states = states.sort_values(["player_id", "gameday", "game_id"])
    alpha = 2.0 / (span + 1.0)
    for metric in QB_STATE_METRICS:
        league = states.groupby("season")[metric].mean().to_dict()
        states[f"league_mean_{metric}"] = states["season"].map(league)
        output = pd.Series(np.nan, index=states.index, dtype=float)
        for _, group in states.groupby("player_id", sort=False):
            current = np.nan
            career_dropbacks = 0.0
            prior_season: int | None = None
            for index, row in group.iterrows():
                season = int(row["season"])
                if prior_season is not None and season != prior_season and np.isfinite(current):
                    mean = float(league.get(prior_season, 0.0))
                    retention = offseason_retention ** max(1, season - prior_season)
                    current = mean + retention * (current - mean)
                value = float(row[metric])
                if np.isfinite(value):
                    current = (
                        value if not np.isfinite(current) else alpha * value + (1 - alpha) * current
                    )
                career_dropbacks += float(row["qb_dropbacks"])
                if career_dropbacks >= min_dropbacks:
                    output.at[index] = current
                prior_season = season
        states[f"state_{metric}"] = output
    states["career_dropbacks"] = states.groupby("player_id")["qb_dropbacks"].cumsum()
    return states


def _qb_injury_week_tuesday_floor_utc(kickoff_utc: pd.Timestamp) -> pd.Timestamp:
    """00:00 America/New_York on the Tuesday that starts ``kickoff_utc``'s NFL week.

    Duplicated (not imported) from
    ``nfl_ats.players._injury_week_tuesday_floor_utc`` -- see
    ``QB_INJURY_PROXY_HOURS_BEFORE_KICKOFF`` for why.
    """

    kickoff_eastern = kickoff_utc.tz_convert(_QB_EASTERN)
    sunday = week_cycle_sunday(kickoff_eastern.date())
    tuesday = sunday - timedelta(days=5)
    tuesday_midnight = datetime.combine(tuesday, time(0), tzinfo=_QB_EASTERN)
    return pd.Timestamp(tuesday_midnight).tz_convert("UTC")


def _qb_injury_proxy_times(schedule: pd.DataFrame) -> pd.DataFrame:
    """Kickoff-derived per-(season, week, team) injury visibility proxy time.

    Duplicated (not imported) from ``nfl_ats.players._injury_proxy_times``
    -- see ``QB_INJURY_PROXY_HOURS_BEFORE_KICKOFF`` for why. Requires
    ``season``, ``week``, ``home_team``, ``away_team``, ``kickoff``.
    """

    required = {"season", "week", "home_team", "away_team", "kickoff"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Injury proxy schedule is missing columns: {', '.join(missing)}")
    frame = schedule.loc[:, ["season", "week", "home_team", "away_team", "kickoff"]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], errors="coerce", utc=True)
    frame = frame.loc[
        frame["season"].notna() & frame["week"].notna() & frame["kickoff"].notna()
    ].copy()
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    long = pd.concat(
        [
            frame.rename(columns={"home_team": "team"})[["season", "week", "team", "kickoff"]],
            frame.rename(columns={"away_team": "team"})[["season", "week", "team", "kickoff"]],
        ],
        ignore_index=True,
    )
    long = long.loc[long["team"].notna()].copy()
    long["team"] = long["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    long = long.drop_duplicates(["season", "week", "team"]).reset_index(drop=True)
    proxy_at: list[pd.Timestamp] = []
    for kickoff in long["kickoff"]:
        kickoff_ts = pd.Timestamp(kickoff)
        floor_utc = _qb_injury_week_tuesday_floor_utc(kickoff_ts)
        candidate = kickoff_ts - pd.Timedelta(hours=QB_INJURY_PROXY_HOURS_BEFORE_KICKOFF)
        candidate = max(candidate, floor_utc)
        candidate = min(candidate, kickoff_ts - pd.Timedelta(minutes=1))
        proxy_at.append(candidate)
    long["injury_proxy_at"] = proxy_at
    return long[["season", "week", "team", "injury_proxy_at"]]


def _canonicalize_qb_availability(
    injuries: pd.DataFrame,
    *,
    timestamp_fallback: Literal["drop", "week_proxy"] = "drop",
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize only the injury fields needed by the named-QB state builder.

    ``timestamp_fallback`` (ENG-39, default ``"drop"``) mirrors
    ``nfl_ats.players.canonicalize_injuries``: ``"drop"`` is byte-identical
    to the pre-ENG-39 behaviour -- no new columns, no schedule dependency.
    ``"week_proxy"`` tolerates a missing/unparsable ``date_modified`` by
    substituting a kickoff-derived, leakage-safe proxy (that function's
    docstring has the exact visibility rule) and requires ``schedule``
    (``season``, ``week``, ``home_team``, ``away_team``, ``kickoff``).
    Output then also carries ``effective_observed_at`` and
    ``observed_at_basis``; a real ``date_modified`` is never overwritten.
    """

    if timestamp_fallback not in ("drop", "week_proxy"):
        raise ValueError("timestamp_fallback must be 'drop' or 'week_proxy'")
    if timestamp_fallback == "week_proxy" and schedule is None:
        raise ValueError("timestamp_fallback='week_proxy' requires a schedule frame")

    working = injuries
    if timestamp_fallback == "week_proxy" and "date_modified" not in injuries.columns:
        working = injuries.copy()
        working["date_modified"] = pd.NaT

    require_columns(working, QB_AVAILABILITY_COLUMNS, "quarterback availability")
    result = working.loc[:, list(QB_AVAILABILITY_COLUMNS)].copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["week"] = pd.to_numeric(result["week"], errors="coerce")
    result["date_modified"] = pd.to_datetime(result["date_modified"], errors="coerce", utc=True)
    result["team"] = result["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    result["gsis_id"] = result["gsis_id"].astype("string")
    result["position"] = result["position"].astype("string").str.upper()

    if timestamp_fallback == "drop":
        result = result.loc[
            result["season"].notna()
            & result["week"].notna()
            & result["team"].notna()
            & result["gsis_id"].notna()
            & result["date_modified"].notna()
        ].copy()
        result["season"] = result["season"].astype(int)
        result["week"] = result["week"].astype(int)
        return result.sort_values(
            ["season", "week", "team", "gsis_id", "date_modified"]
        ).reset_index(drop=True)

    assert schedule is not None  # narrowed by the ValueError check above
    result = result.loc[
        result["season"].notna()
        & result["week"].notna()
        & result["team"].notna()
        & result["gsis_id"].notna()
    ].copy()
    result["season"] = result["season"].astype(int)
    result["week"] = result["week"].astype(int)
    proxy_lookup = _qb_injury_proxy_times(schedule)
    result = result.merge(proxy_lookup, on=["season", "week", "team"], how="left")
    result["effective_observed_at"] = result["date_modified"].where(
        result["date_modified"].notna(), result["injury_proxy_at"]
    )
    result["observed_at_basis"] = np.where(
        result["date_modified"].notna(), "date_modified", "week_proxy"
    )
    result = result.loc[result["effective_observed_at"].notna()].copy()
    result = result.drop(columns=["injury_proxy_at"])
    return result.sort_values(
        ["season", "week", "team", "gsis_id", "effective_observed_at"]
    ).reset_index(drop=True)


def _expected_value(starter: float, backup: float, start_probability: float) -> float:
    """Mix named-player states without requiring a zero-weight missing state."""

    if not np.isfinite(start_probability):
        return math.nan
    if start_probability == 1.0 and np.isfinite(starter):
        return starter
    if start_probability == 0.0 and np.isfinite(backup):
        return backup
    if not np.isfinite(starter) or not np.isfinite(backup):
        return math.nan
    return start_probability * starter + (1.0 - start_probability) * backup


def _prior_qb_state(
    histories: dict[str, pd.DataFrame], player_id: str, game_date: pd.Timestamp
) -> pd.Series | None:
    history = histories.get(player_id)
    if history is None or history.empty:
        return None
    dates = history["gameday"].to_numpy(dtype="datetime64[ns]")
    position = int(np.searchsorted(dates, np.datetime64(game_date, "ns"), side="left")) - 1
    return None if position < 0 else history.iloc[position]


def enrich_with_qb_features(
    games: pd.DataFrame,
    pbp: pd.DataFrame,
    depth_history: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    availability_rates: pd.DataFrame | None = None,
    *,
    decision_hours_before_kickoff: int = 24,
    max_depth_age_days: int = 14,
    span: int = 12,
    min_dropbacks: int = 50,
    offseason_retention: float = 0.75,
    injury_timestamp_fallback: Literal["drop", "week_proxy"] = "drop",
) -> pd.DataFrame:
    """Attach named starter/backup states from information visible at decision time.

    Depth identity is selected from the latest observation no later than the
    configured decision timestamp.  Player performance comes only from games
    before the target game's date.  When an injury source covers the target
    season, the latest visible starter report supplies the existing fixed or
    season-lagged start probability; the expected state mixes that named
    starter with the named QB2 instead of a generic replacement constant.
    Uncovered injury seasons and missing player histories remain null rather
    than being silently treated as healthy or replacement-level.

    ``injury_timestamp_fallback`` (ENG-39, default ``"drop"``): forwarded to
    ``_canonicalize_qb_availability``. ``"drop"`` needs no schedule and is
    byte-identical to the pre-ENG-39 behaviour. ``"week_proxy"`` derives
    each team-game's own kickoff from ``games`` itself (already required
    above) to resolve a leakage-safe proxy for a row with no real
    ``date_modified`` -- see ``nfl_ats.players.canonicalize_injuries`` for
    the exact rule this mirrors.
    """

    if injury_timestamp_fallback not in ("drop", "week_proxy"):
        raise ValueError("injury_timestamp_fallback must be 'drop' or 'week_proxy'")
    required = {"game_id", "season", "gameday", "kickoff", "home_team", "away_team"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise DataContractError(f"Game features are missing columns: {', '.join(missing)}")
    if decision_hours_before_kickoff < 0:
        raise ValueError("decision_hours_before_kickoff cannot be negative")
    if max_depth_age_days < 1:
        raise ValueError("max_depth_age_days must be positive")
    if not 0.0 <= offseason_retention <= 1.0:
        raise ValueError("offseason_retention must be between zero and one")
    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    result["kickoff"] = pd.to_datetime(result["kickoff"], errors="coerce", utc=True)
    for column in ("home_team", "away_team"):
        result[column] = result[column].replace(TEAM_ABBREVIATION_ALIASES)
    states = build_qb_states(
        build_qb_game_metrics(pbp),
        result,
        span=span,
        min_dropbacks=min_dropbacks,
        offseason_retention=offseason_retention,
    )
    player_groups = {
        str(player): group.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        for player, group in states.groupby("player_id", sort=False)
    }
    depth = _prepare_depth_timeline(depth_history)
    depth["pos_rank"] = pd.to_numeric(depth["pos_rank"], errors="raise").astype(int)
    depth["team"] = depth["team"].replace(TEAM_ABBREVIATION_ALIASES).astype(str)
    depth["gsis_id"] = depth["gsis_id"].astype(str)
    depth_groups = {
        str(team): group.sort_values(["effective_at_utc", "pos_rank", "gsis_id"]).reset_index(
            drop=True
        )
        for team, group in depth.groupby("team", sort=False)
    }
    qb_injury_schedule = None
    if injury_timestamp_fallback == "week_proxy":
        qb_injury_schedule = result.loc[
            :, ["season", "week", "home_team", "away_team", "kickoff"]
        ].copy()
    availability = (
        _canonicalize_qb_availability(
            injuries,
            timestamp_fallback=injury_timestamp_fallback,
            schedule=qb_injury_schedule,
        )
        if injuries is not None
        else pd.DataFrame(columns=QB_AVAILABILITY_COLUMNS)
    )
    covered_injury_seasons = set(availability["season"].astype(int).unique())
    availability_groups = {
        (int(str(season)), int(str(week)), str(team)): group.reset_index(drop=True)
        for (season, week, team), group in availability.groupby(
            ["season", "week", "team"], sort=False, observed=True
        )
    }
    learned_lookup = (
        availability_rate_lookup(canonicalize_availability_rates(availability_rates))
        if availability_rates is not None
        else None
    )
    for side in ("home", "away"):
        result[f"{side}_qb_id"] = pd.NA
        result[f"{side}_qb_name"] = pd.NA
        result[f"{side}_depth_qb_backup_id"] = pd.NA
        result[f"{side}_depth_qb_backup_name"] = pd.NA
        result[f"{side}_qb_depth_observed_at"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        result[f"{side}_qb_depth_effective_at"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        result[f"{side}_qb_depth_age_days"] = np.nan
        result[f"{side}_qb_career_dropbacks"] = np.nan
        result[f"{side}_depth_qb_availability_observed_at"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        result[f"{side}_depth_qb_availability_source"] = pd.NA
        for metric in QB_STATE_METRICS:
            result[f"{side}_{metric}"] = np.nan
        for metric in QB_DEPTH_STATE_METRICS:
            result[f"{side}_{metric}"] = np.nan

    for index, game in result.iterrows():
        kickoff = game["kickoff"]
        if pd.isna(kickoff):
            continue
        decision_at = pd.Timestamp(kickoff) - pd.Timedelta(hours=decision_hours_before_kickoff)
        for side in ("home", "away"):
            team = str(game[f"{side}_team"])
            team_depth = depth_groups.get(team)
            if team_depth is None or team_depth.empty:
                continue
            position = (
                int(team_depth["effective_at_utc"].searchsorted(decision_at, side="right")) - 1
            )
            if position < 0:
                continue
            effective_at = team_depth.iloc[position]["effective_at_utc"]
            age_days = (decision_at - effective_at).total_seconds() / 86_400
            if age_days > max_depth_age_days:
                continue
            candidates = team_depth.loc[team_depth["effective_at_utc"].eq(effective_at)]
            candidates = candidates.sort_values(["pos_rank", "gsis_id"]).drop_duplicates("gsis_id")
            starter = candidates.iloc[0]
            backup = candidates.iloc[1] if len(candidates) > 1 else None
            player_id = str(starter["gsis_id"])
            result.at[index, f"{side}_qb_id"] = player_id
            result.at[index, f"{side}_qb_name"] = starter["player_name"]
            result.at[index, f"{side}_qb_depth_observed_at"] = starter["observed_at_utc"]
            result.at[index, f"{side}_qb_depth_effective_at"] = starter["effective_at_utc"]
            result.at[index, f"{side}_qb_depth_age_days"] = age_days
            backup_id: str | None = None
            if backup is not None:
                backup_id = str(backup["gsis_id"])
                result.at[index, f"{side}_depth_qb_backup_id"] = backup_id
                result.at[index, f"{side}_depth_qb_backup_name"] = backup["player_name"]

            season = int(game["season"])
            week = int(game["week"])
            start_probability = math.nan
            availability_source = "injury_season_uncovered"
            if season in covered_injury_seasons:
                start_probability = 1.0
                availability_source = "not_reported"
                reports = availability_groups.get((season, week, team))
                if reports is not None:
                    # ENG-39: mirrors nfl_ats.players._injury_rows_asof --
                    # "effective_observed_at" (real date_modified, else the
                    # leakage-safe week_proxy fallback) when present, else
                    # the historical "date_modified" filter unchanged.
                    availability_timestamp_column = (
                        "effective_observed_at"
                        if "effective_observed_at" in reports.columns
                        else "date_modified"
                    )
                    visible = reports.loc[reports[availability_timestamp_column].le(decision_at)]
                    visible = visible.loc[visible["gsis_id"].eq(player_id)]
                    if not visible.empty:
                        report = visible.iloc[-1]
                        result.at[index, f"{side}_depth_qb_availability_observed_at"] = report[
                            availability_timestamp_column
                        ]
                        unavailable, availability_source = resolve_unavailability(
                            learned_lookup,
                            target_season=season,
                            report_status=report["report_status"],
                            practice_status=report["practice_status"],
                            position=report["position"],
                        )
                        start_probability = 1.0 - unavailable
            result.at[index, f"{side}_depth_qb_start_probability"] = start_probability
            result.at[index, f"{side}_depth_qb_availability_source"] = availability_source

            game_date = pd.Timestamp(game["gameday"])
            starter_state = _prior_qb_state(player_groups, player_id, game_date)
            backup_state = (
                _prior_qb_state(player_groups, backup_id, game_date)
                if backup_id is not None
                else None
            )
            starter_experience = (
                float(starter_state["career_dropbacks"]) if starter_state is not None else math.nan
            )
            backup_experience = (
                float(backup_state["career_dropbacks"]) if backup_state is not None else math.nan
            )
            result.at[index, f"{side}_depth_qb_starter_experience_log"] = (
                math.log1p(starter_experience) if np.isfinite(starter_experience) else math.nan
            )
            result.at[index, f"{side}_qb_career_dropbacks"] = starter_experience
            result.at[index, f"{side}_depth_qb_backup_experience_log"] = (
                math.log1p(backup_experience) if np.isfinite(backup_experience) else math.nan
            )
            for metric in QB_STATE_METRICS:
                starter_value = (
                    float(starter_state[f"state_{metric}"])
                    if starter_state is not None
                    else math.nan
                )
                backup_value = (
                    float(backup_state[f"state_{metric}"]) if backup_state is not None else math.nan
                )
                result.at[index, f"{side}_{metric}"] = starter_value
                if metric in ("qb_epa_per_dropback", "qb_cpoe"):
                    suffix = metric.removeprefix("qb_")
                    result.at[index, f"{side}_depth_qb_starter_{suffix}"] = starter_value
                    result.at[index, f"{side}_depth_qb_backup_{suffix}"] = backup_value
                    result.at[index, f"{side}_depth_qb_expected_{suffix}"] = _expected_value(
                        starter_value, backup_value, start_probability
                    )
                    result.at[index, f"{side}_depth_qb_backup_adjustment_{suffix}"] = (
                        backup_value - starter_value
                        if np.isfinite(starter_value) and np.isfinite(backup_value)
                        else math.nan
                    )
    for metric in (*QB_STATE_METRICS, *QB_DEPTH_STATE_METRICS):
        result[f"diff_{metric}"] = result[f"home_{metric}"] - result[f"away_{metric}"]
    result["qb_feature_version"] = QB_FEATURE_VERSION
    return result
