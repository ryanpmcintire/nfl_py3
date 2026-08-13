"""Immutable play-participation data and season-lagged player ratings."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.pbp import analysis_plays

PARTICIPATION_DATA_VERSION = "v1"
PARTICIPATION_RATING_VERSION = "v1"
PARTICIPATION_RATING_LOOKBACK_SEASONS = 3
PARTICIPATION_RATING_RIDGE_ALPHA = 1000.0
PARTICIPATION_RATING_TEAM_FEATURE_SCALE = 11.0
PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS = 500.0
PARTICIPATION_RATING_EPA_CLIP = 5.0

PARTICIPATION_SOURCE_COLUMNS = (
    "nflverse_game_id",
    "old_game_id",
    "play_id",
    "possession_team",
    "offense_formation",
    "offense_personnel",
    "defenders_in_box",
    "defense_personnel",
    "number_of_pass_rushers",
    "players_on_play",
    "offense_players",
    "defense_players",
    "n_offense",
    "n_defense",
    "ngs_air_yards",
    "time_to_throw",
    "was_pressure",
    "route",
    "defense_man_zone_type",
    "defense_coverage_type",
)
PARTICIPATION_COLUMNS = (
    "game_id",
    "season",
    *PARTICIPATION_SOURCE_COLUMNS[1:],
)
PARTICIPATION_RATING_COLUMNS = (
    "target_season",
    "player_id",
    "offense_rating",
    "defense_rating",
    "offense_plays",
    "defense_plays",
    "source_start_season",
    "source_end_season",
    "source_plays",
    "lookback_seasons",
    "ridge_alpha",
    "team_feature_scale",
    "reliability_prior_plays",
    "epa_clip",
    "rating_version",
)


@dataclass(frozen=True)
class ParticipationSnapshot:
    """An immutable, season-partitioned nflverse participation snapshot."""

    snapshot_id: str
    root: Path
    seasons: tuple[int, ...]

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def season_path(self, season: int) -> Path:
        return self.root / f"season={season}" / "participation.parquet"


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


def canonicalize_participation(
    frame: pd.DataFrame,
    *,
    season: int | None = None,
) -> pd.DataFrame:
    """Normalize the stable play-participation storage contract."""

    result = frame.copy()
    if "nflverse_game_id" in result.columns:
        require_columns(result, PARTICIPATION_SOURCE_COLUMNS, "play_participation")
        result = result.rename(columns={"nflverse_game_id": "game_id"})
    else:
        require_columns(result, PARTICIPATION_COLUMNS, "play_participation")

    result["game_id"] = result["game_id"].astype("string")
    parsed_season = pd.to_numeric(
        result["game_id"].str.extract(r"^(\d{4})_", expand=False), errors="coerce"
    )
    if parsed_season.isna().any():
        raise DataContractError("play_participation contains an invalid nflverse game ID")
    result["season"] = parsed_season.astype(int)
    result["play_id"] = pd.to_numeric(result["play_id"], errors="coerce")
    if result[["game_id", "play_id"]].isna().any(axis=None):
        raise DataContractError("play_participation contains null game_id or play_id values")
    if not np.equal(result["play_id"], np.floor(result["play_id"])).all():
        raise DataContractError("play_participation play_id values must be integers")
    result["play_id"] = result["play_id"].astype(int)

    for column in (
        "defenders_in_box",
        "number_of_pass_rushers",
        "n_offense",
        "n_defense",
        "ngs_air_yards",
        "time_to_throw",
        "was_pressure",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["possession_team"] = (
        result["possession_team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string").str.upper()
    )
    for column in (
        "old_game_id",
        "offense_formation",
        "offense_personnel",
        "defense_personnel",
        "players_on_play",
        "offense_players",
        "defense_players",
        "route",
        "defense_man_zone_type",
        "defense_coverage_type",
    ):
        result[column] = result[column].astype("string")

    observed_seasons = set(result["season"].astype(int).unique())
    if season is not None and observed_seasons and observed_seasons != {season}:
        raise DataContractError(
            f"play_participation season partition {season} contains seasons "
            f"{sorted(observed_seasons)}"
        )
    if result.duplicated(["game_id", "play_id"]).any():
        raise DataContractError("play_participation contains duplicate game_id/play_id rows")
    return (
        result.loc[:, list(PARTICIPATION_COLUMNS)]
        .sort_values(["season", "game_id", "play_id"])
        .reset_index(drop=True)
    )


def write_participation_snapshot(
    season_frames: dict[int, pd.DataFrame],
    raw_root: Path,
    snapshot_id: str | None = None,
) -> ParticipationSnapshot:
    """Write season partitions and publish their manifest last."""

    seasons = tuple(sorted(season_frames))
    if not seasons:
        raise ValueError("At least one participation season is required")
    identifier = snapshot_id or run_id()
    destination = raw_root / identifier
    if destination.exists():
        raise FileExistsError(f"Participation snapshot already exists: {destination}")

    partitions: list[dict[str, Any]] = []
    for source_season in seasons:
        canonical = canonicalize_participation(season_frames[source_season], season=source_season)
        path = destination / f"season={source_season}" / "participation.parquet"
        atomic_parquet(canonical, path)
        partitions.append(
            {
                "season": source_season,
                "rows": len(canonical),
                "path": str(path.relative_to(destination)),
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "snapshot_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "loader": "nflreadpy.load_participation",
            "repository": "https://github.com/nflverse/nflverse-data",
            "license": "Review the upstream dataset-specific terms before redistribution",
        },
        "contract_version": PARTICIPATION_DATA_VERSION,
        "availability_contract": (
            "realized postgame participation; player ratings for a target season use only "
            "earlier completed seasons"
        ),
        "source_regimes": {
            "2016-2022": "Next Gen Stats participation",
            "2023-present": "FTN participation",
        },
        "seasons": list(seasons),
        "columns": list(PARTICIPATION_COLUMNS),
        "partitions": partitions,
        "rows": sum(int(partition["rows"]) for partition in partitions),
    }
    atomic_json(manifest, destination / "manifest.json")
    return ParticipationSnapshot(identifier, destination, seasons)


def fetch_participation_snapshot(
    seasons: list[int],
    raw_root: Path,
) -> ParticipationSnapshot:
    """Download participation one season at a time to bound peak memory."""

    if not seasons or seasons != sorted(set(seasons)):
        raise ValueError("Participation seasons must be non-empty, unique, and sorted")
    import nflreadpy as nfl

    identifier = run_id()
    destination = raw_root / identifier
    if destination.exists():
        raise FileExistsError(f"Participation snapshot already exists: {destination}")
    partitions: list[dict[str, Any]] = []
    for season in seasons:
        canonical = canonicalize_participation(
            _to_pandas(nfl.load_participation(seasons=[season])), season=season
        )
        path = destination / f"season={season}" / "participation.parquet"
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
        "source": {
            "loader": "nflreadpy.load_participation",
            "repository": "https://github.com/nflverse/nflverse-data",
            "license": "Review the upstream dataset-specific terms before redistribution",
        },
        "contract_version": PARTICIPATION_DATA_VERSION,
        "availability_contract": (
            "realized postgame participation; player ratings for a target season use only "
            "earlier completed seasons"
        ),
        "source_regimes": {
            "2016-2022": "Next Gen Stats participation",
            "2023-present": "FTN participation",
        },
        "seasons": seasons,
        "columns": list(PARTICIPATION_COLUMNS),
        "partitions": partitions,
        "rows": sum(int(partition["rows"]) for partition in partitions),
    }
    atomic_json(manifest, destination / "manifest.json")
    return ParticipationSnapshot(identifier, destination, tuple(seasons))


def participation_snapshot_from_root(root: Path) -> ParticipationSnapshot:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Participation manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ParticipationSnapshot(
        str(payload["snapshot_id"]),
        root,
        tuple(int(value) for value in payload["seasons"]),
    )


def latest_participation_snapshot(raw_root: Path) -> ParticipationSnapshot:
    manifests = sorted(raw_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No participation snapshots found in {raw_root}")
    return participation_snapshot_from_root(manifests[-1].parent)


def load_participation_snapshot(snapshot: ParticipationSnapshot) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in snapshot.seasons:
        path = snapshot.season_path(season)
        if not path.is_file():
            raise FileNotFoundError(f"Missing participation partition: {path}")
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=PARTICIPATION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _player_ids(raw: Any) -> tuple[str, ...]:
    if pd.isna(raw):
        return ()
    return tuple(value.strip() for value in str(raw).split(";") if value.strip())


def build_participation_play_table(
    participation: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Join valid competitive 11-on-11 participation to canonical PBP EPA."""

    canonical = canonicalize_participation(participation)
    plays = analysis_plays(pbp)
    plays = plays.loc[
        plays["competitive_play"],
        ["game_id", "play_id", "season", "posteam", "defteam", "epa"],
    ].copy()
    plays["play_id"] = pd.to_numeric(plays["play_id"], errors="raise").astype(int)
    for column in ("posteam", "defteam"):
        plays[column] = plays[column].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    joined = canonical.merge(
        plays,
        on=["game_id", "play_id", "season"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise DataContractError("No competitive participation plays match the PBP snapshot")
    joined["offense_player_ids"] = joined["offense_players"].map(_player_ids)
    joined["defense_player_ids"] = joined["defense_players"].map(_player_ids)
    valid_lineups = (
        joined["n_offense"].eq(11)
        & joined["n_defense"].eq(11)
        & joined["offense_player_ids"].map(len).eq(11)
        & joined["defense_player_ids"].map(len).eq(11)
        & joined["offense_player_ids"].map(lambda values: len(set(values))).eq(11)
        & joined["defense_player_ids"].map(lambda values: len(set(values))).eq(11)
        & joined["possession_team"].eq(joined["posteam"])
        & joined["defteam"].notna()
    )
    result = joined.loc[
        valid_lineups,
        [
            "season",
            "game_id",
            "play_id",
            "posteam",
            "defteam",
            "epa",
            "offense_players",
            "defense_players",
        ],
    ].copy()
    if result.empty:
        raise DataContractError("No valid competitive 11-on-11 participation plays remain")
    return result.reset_index(drop=True)


def _rating_feature_rows(
    rows: Iterable[Any],
    *,
    team_feature_scale: float,
    offense_counts: dict[int, Counter[str]],
    defense_counts: dict[int, Counter[str]],
) -> Iterator[dict[str, float]]:
    for row in rows:
        offense_players = _player_ids(row.offense_players)
        defense_players = _player_ids(row.defense_players)
        offense_counts[int(row.season)].update(offense_players)
        defense_counts[int(row.season)].update(defense_players)
        values = {f"offense_player::{player_id}": 1.0 for player_id in offense_players}
        values.update({f"defense_player::{player_id}": -1.0 for player_id in defense_players})
        values[f"offense_team::{row.posteam}"] = team_feature_scale
        values[f"defense_team::{row.defteam}"] = -team_feature_scale
        yield values


def canonicalize_participation_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate ratings as season-lagged inputs rather than current-game outcomes."""

    require_columns(frame, PARTICIPATION_RATING_COLUMNS, "participation_ratings")
    result = frame.loc[:, list(PARTICIPATION_RATING_COLUMNS)].copy()
    integer_columns = (
        "target_season",
        "offense_plays",
        "defense_plays",
        "source_start_season",
        "source_end_season",
        "source_plays",
        "lookback_seasons",
    )
    numeric_columns = (
        "offense_rating",
        "defense_rating",
        "ridge_alpha",
        "team_feature_scale",
        "reliability_prior_plays",
        "epa_clip",
    )
    for column in (*integer_columns, *numeric_columns):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["player_id"] = result["player_id"].astype("string")
    if result[["target_season", "player_id"]].isna().any(axis=None):
        raise DataContractError("participation_ratings contains a null season or player ID")
    if result[[*integer_columns, *numeric_columns]].isna().any(axis=None):
        raise DataContractError("participation_ratings contains nonnumeric rating metadata")
    if not np.isfinite(result.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise DataContractError("participation_ratings contains a non-finite value")
    for column in integer_columns:
        result[column] = result[column].astype(int)
    if result.duplicated(["target_season", "player_id"]).any():
        raise DataContractError("participation_ratings contains duplicate season/player rows")
    if result["source_end_season"].ge(result["target_season"]).any():
        raise DataContractError(
            "participation_ratings source seasons must end before every target season"
        )
    if result[["offense_plays", "defense_plays", "source_plays"]].lt(0).any(axis=None):
        raise DataContractError("participation_ratings play counts cannot be negative")
    if (
        result["ridge_alpha"].le(0).any()
        or result["team_feature_scale"].le(0).any()
        or result["reliability_prior_plays"].lt(0).any()
        or result["epa_clip"].le(0).any()
    ):
        raise DataContractError("participation_ratings shrinkage parameters are invalid")
    return result.sort_values(["target_season", "player_id"]).reset_index(drop=True)


def build_season_lagged_player_ratings(
    participation: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    target_seasons: Iterable[int] | None = None,
    lookback_seasons: int = PARTICIPATION_RATING_LOOKBACK_SEASONS,
    ridge_alpha: float = PARTICIPATION_RATING_RIDGE_ALPHA,
    team_feature_scale: float = PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    reliability_prior_plays: float = PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
    epa_clip: float = PARTICIPATION_RATING_EPA_CLIP,
) -> pd.DataFrame:
    """Estimate regularized player effects using only seasons before each target."""

    if lookback_seasons < 1:
        raise ValueError("lookback_seasons must be positive")
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be finite and positive")
    if not np.isfinite(team_feature_scale) or team_feature_scale <= 0:
        raise ValueError("team_feature_scale must be finite and positive")
    if not np.isfinite(reliability_prior_plays) or reliability_prior_plays < 0:
        raise ValueError("reliability_prior_plays must be finite and nonnegative")
    if not np.isfinite(epa_clip) or epa_clip <= 0:
        raise ValueError("epa_clip must be finite and positive")

    play_table = build_participation_play_table(participation, pbp)
    available_seasons = sorted(play_table["season"].astype(int).unique())
    if target_seasons is None:
        targets = list(range(available_seasons[0] + 1, available_seasons[-1] + 2))
    else:
        targets = sorted({int(value) for value in target_seasons})
    if not targets:
        raise ValueError("At least one target season is required")

    # Build one sparse design matrix. Its global vocabulary contains only
    # participant identifiers, never outcomes. A future-only player therefore
    # contributes an all-zero column to an earlier fit and cannot change it.
    offense_counts_by_season: dict[int, Counter[str]] = {
        season: Counter() for season in available_seasons
    }
    defense_counts_by_season: dict[int, Counter[str]] = {
        season: Counter() for season in available_seasons
    }
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(
        _rating_feature_rows(
            play_table.itertuples(index=False),
            team_feature_scale=team_feature_scale,
            offense_counts=offense_counts_by_season,
            defense_counts=defense_counts_by_season,
        )
    )
    feature_names = vectorizer.get_feature_names_out()
    play_seasons = play_table["season"].to_numpy(dtype=int)
    target_values = (
        pd.to_numeric(play_table["epa"], errors="raise")
        .clip(-epa_clip, epa_clip)
        .to_numpy(dtype=float)
    )
    output: list[dict[str, Any]] = []
    for target_season in targets:
        source_end = target_season - 1
        source_start = target_season - lookback_seasons
        training_mask = (play_seasons >= source_start) & (play_seasons <= source_end)
        if not training_mask.any():
            continue
        estimator = Ridge(alpha=ridge_alpha, solver="lsqr", fit_intercept=True, tol=1e-6)
        estimator.fit(matrix[training_mask], target_values[training_mask])
        coefficients = dict(zip(feature_names, np.asarray(estimator.coef_), strict=True))
        training_seasons = sorted(set(play_seasons[training_mask]))
        offense_counts: Counter[str] = Counter()
        defense_counts: Counter[str] = Counter()
        for training_season in training_seasons:
            offense_counts.update(offense_counts_by_season[int(training_season)])
            defense_counts.update(defense_counts_by_season[int(training_season)])
        source_plays = int(training_mask.sum())
        observed_source_start = int(training_seasons[0])
        observed_source_end = int(training_seasons[-1])
        for player_id in sorted(set(offense_counts).union(defense_counts)):
            offense_plays = int(offense_counts[player_id])
            defense_plays = int(defense_counts[player_id])
            offense_reliability = (
                offense_plays / (offense_plays + reliability_prior_plays) if offense_plays else 0.0
            )
            defense_reliability = (
                defense_plays / (defense_plays + reliability_prior_plays) if defense_plays else 0.0
            )
            output.append(
                {
                    "target_season": target_season,
                    "player_id": player_id,
                    "offense_rating": float(
                        coefficients.get(f"offense_player::{player_id}", 0.0) * offense_reliability
                    ),
                    "defense_rating": float(
                        coefficients.get(f"defense_player::{player_id}", 0.0) * defense_reliability
                    ),
                    "offense_plays": offense_plays,
                    "defense_plays": defense_plays,
                    "source_start_season": observed_source_start,
                    "source_end_season": observed_source_end,
                    "source_plays": source_plays,
                    "lookback_seasons": lookback_seasons,
                    "ridge_alpha": ridge_alpha,
                    "team_feature_scale": team_feature_scale,
                    "reliability_prior_plays": reliability_prior_plays,
                    "epa_clip": epa_clip,
                    "rating_version": PARTICIPATION_RATING_VERSION,
                }
            )
    if not output:
        raise DataContractError("No season-lagged player ratings could be estimated")
    return canonicalize_participation_ratings(pd.DataFrame(output))
