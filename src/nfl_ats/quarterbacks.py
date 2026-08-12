"""Point-in-time quarterback depth charts and prior-performance states."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import QB_STATE_METRICS
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS, analysis_plays

DEPTH_CHART_VERSION = "v1"
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
    return pd.read_parquet(snapshot.data_path)


def latest_starting_qbs(
    depth_history: pd.DataFrame,
    decision_at: pd.Timestamp,
    *,
    max_age_days: int = 14,
) -> pd.DataFrame:
    """Return rank-one QBs from the latest team observation before a decision."""

    required = {"observed_at_utc", "team", "gsis_id", "player_name", "pos_rank"}
    missing = sorted(required.difference(depth_history.columns))
    if missing:
        raise DataContractError(f"Depth history is missing columns: {', '.join(missing)}")
    if max_age_days < 1:
        raise ValueError("max_age_days must be positive")
    as_of = pd.Timestamp(decision_at)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    history = depth_history.copy()
    history["observed_at_utc"] = pd.to_datetime(
        history["observed_at_utc"], errors="raise", utc=True
    )
    history = history.loc[history["observed_at_utc"].le(as_of)].copy()
    if history.empty:
        return history
    latest_times = history.groupby("team")["observed_at_utc"].transform("max")
    latest = history.loc[history["observed_at_utc"].eq(latest_times)].copy()
    latest = latest.loc[latest["pos_rank"].eq(latest.groupby("team")["pos_rank"].transform("min"))]
    latest["depth_age_days"] = (as_of - latest["observed_at_utc"]).dt.total_seconds() / 86_400
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


def enrich_with_qb_features(
    games: pd.DataFrame,
    pbp: pd.DataFrame,
    depth_history: pd.DataFrame,
    *,
    decision_hours_before_kickoff: int = 24,
    max_depth_age_days: int = 14,
    span: int = 12,
    min_dropbacks: int = 50,
    offseason_retention: float = 0.75,
) -> pd.DataFrame:
    """Attach expected-starter identity and that player's strictly prior state."""

    required = {"game_id", "season", "gameday", "kickoff", "home_team", "away_team"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise DataContractError(f"Game features are missing columns: {', '.join(missing)}")
    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    result["kickoff"] = pd.to_datetime(result["kickoff"], errors="coerce", utc=True)
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
    depth = depth_history.copy()
    depth["observed_at_utc"] = pd.to_datetime(depth["observed_at_utc"], errors="raise", utc=True)
    depth_groups = {
        str(team): group.sort_values(["observed_at_utc", "pos_rank", "gsis_id"]).reset_index(
            drop=True
        )
        for team, group in depth.groupby("team", sort=False)
    }
    for side in ("home", "away"):
        result[f"{side}_qb_id"] = pd.NA
        result[f"{side}_qb_name"] = pd.NA
        result[f"{side}_qb_depth_observed_at"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        result[f"{side}_qb_depth_age_days"] = np.nan
        result[f"{side}_qb_career_dropbacks"] = np.nan
        for metric in QB_STATE_METRICS:
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
                int(team_depth["observed_at_utc"].searchsorted(decision_at, side="right")) - 1
            )
            if position < 0:
                continue
            observed_at = team_depth.iloc[position]["observed_at_utc"]
            age_days = (decision_at - observed_at).total_seconds() / 86_400
            if age_days > max_depth_age_days:
                continue
            candidates = team_depth.loc[team_depth["observed_at_utc"].eq(observed_at)]
            starter = candidates.sort_values(["pos_rank", "gsis_id"]).iloc[0]
            player_id = str(starter["gsis_id"])
            result.at[index, f"{side}_qb_id"] = player_id
            result.at[index, f"{side}_qb_name"] = starter["player_name"]
            result.at[index, f"{side}_qb_depth_observed_at"] = starter["observed_at_utc"]
            result.at[index, f"{side}_qb_depth_age_days"] = age_days
            history = player_groups.get(player_id)
            if history is None or history.empty:
                continue
            dates = history["gameday"].to_numpy(dtype="datetime64[ns]")
            game_date = np.datetime64(pd.Timestamp(game["gameday"]), "ns")
            position = int(np.searchsorted(dates, game_date, side="left")) - 1
            if position < 0:
                continue
            state = history.iloc[position]
            result.at[index, f"{side}_qb_career_dropbacks"] = state["career_dropbacks"]
            for metric in QB_STATE_METRICS:
                result.at[index, f"{side}_{metric}"] = state[f"state_{metric}"]
    for metric in QB_STATE_METRICS:
        result[f"diff_{metric}"] = result[f"home_{metric}"] - result[f"away_{metric}"]
    result["qb_feature_version"] = DEPTH_CHART_VERSION
    return result
