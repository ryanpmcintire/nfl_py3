from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from nfl_ats.data import DataContractError, require_columns
from nfl_ats.participation import (
    PARTICIPATION_RATING_EPA_CLIP,
    PARTICIPATION_RATING_LOOKBACK_SEASONS,
    PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
    PARTICIPATION_RATING_RIDGE_ALPHA,
    PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    _player_ids,
    canonicalize_participation,
)

SPECIAL_TEAMS_RATING_VERSION = "v1"
SPECIAL_TEAMS_SOURCE_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
SPECIAL_TEAMS_TARGET_SEASONS = (2022, 2023, 2024, 2025)
SPECIAL_TEAMS_RATING_COLUMNS = (
    "target_season",
    "player_id",
    "special_teams_rating",
    "special_teams_plays",
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


def _personnel_tokens(personnel: Any) -> set[str]:
    if pd.isna(personnel):
        return set()
    tokens: set[str] = set()
    for piece in str(personnel).split(","):
        parts = piece.strip().split()
        if len(parts) == 2 and parts[0] == "1":
            tokens.add(parts[1])
    return tokens


def classify_special_teams_unit(offense_personnel: Any, defense_personnel: Any) -> str | None:
    tokens = _personnel_tokens(offense_personnel) | _personnel_tokens(defense_personnel)
    if not ({"K", "P", "LS"} & tokens):
        return None
    if "K" in tokens and "LS" in tokens:
        return "FG_XP"
    if "P" in tokens and "LS" in tokens:
        return "PUNT"
    if "K" in tokens:
        return "KICKOFF"
    return "OTHER_ST"


def build_special_teams_play_table(participation: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    canonical = canonicalize_participation(participation)
    required_pbp = ("game_id", "play_id", "season", "posteam", "epa")
    require_columns(pbp, required_pbp, "special_teams_play_by_play")
    plays = pbp.loc[:, list(required_pbp)].copy()
    plays["game_id"] = plays["game_id"].astype("string")
    for column in ("play_id", "season", "epa"):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    if plays[["game_id", "play_id", "season"]].isna().any(axis=None):
        raise DataContractError("special-teams PBP contains an invalid game, play, or season ID")
    if not np.equal(plays["play_id"], np.floor(plays["play_id"])).all():
        raise DataContractError("special-teams PBP play IDs must be integers")
    if not np.equal(plays["season"], np.floor(plays["season"])).all():
        raise DataContractError("special-teams PBP seasons must be integers")
    plays["play_id"] = plays["play_id"].astype(int)
    plays["season"] = plays["season"].astype(int)
    plays = plays.loc[plays["epa"].notna()].copy()
    if plays.duplicated(["game_id", "play_id", "season"]).any():
        raise DataContractError("special-teams PBP contains duplicate game/play/season rows")
    joined = canonical.merge(
        plays,
        on=["game_id", "play_id", "season"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise DataContractError("No participation plays match the PBP snapshot")
    joined["special_teams_unit"] = [
        classify_special_teams_unit(offense, defense)
        for offense, defense in zip(
            joined["offense_personnel"], joined["defense_personnel"], strict=True
        )
    ]
    joined = joined.loc[joined["special_teams_unit"].notna()].copy()
    joined["side_a_ids"] = joined["offense_players"].map(_player_ids)
    joined["side_b_ids"] = joined["defense_players"].map(_player_ids)
    valid_ids = (
        joined["side_a_ids"].map(len).eq(11)
        & joined["side_b_ids"].map(len).eq(11)
        & joined["side_a_ids"].map(lambda values: len(set(values))).eq(11)
        & joined["side_b_ids"].map(lambda values: len(set(values))).eq(11)
    )
    result = joined.loc[
        valid_ids,
        [
            "season",
            "game_id",
            "play_id",
            "posteam",
            "possession_team",
            "epa",
            "special_teams_unit",
            "side_a_ids",
            "side_b_ids",
        ],
    ].copy()
    if result.empty:
        raise DataContractError("No valid special-teams plays remain")
    return result.sort_values(["season", "game_id", "play_id"]).reset_index(drop=True)


def canonicalize_special_teams_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, SPECIAL_TEAMS_RATING_COLUMNS, "special_teams_ratings")
    result = frame.loc[:, list(SPECIAL_TEAMS_RATING_COLUMNS)].copy()
    integer_columns = (
        "target_season",
        "special_teams_plays",
        "source_start_season",
        "source_end_season",
        "source_plays",
        "lookback_seasons",
    )
    numeric_columns = (
        "special_teams_rating",
        "ridge_alpha",
        "team_feature_scale",
        "reliability_prior_plays",
        "epa_clip",
    )
    for column in (*integer_columns, *numeric_columns):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["player_id"] = result["player_id"].astype("string").str.strip()
    result["rating_version"] = result["rating_version"].astype("string")
    if result[["target_season", "player_id", "rating_version"]].isna().any(axis=None):
        raise DataContractError(
            "special_teams_ratings contains a null season, player ID, or rating version"
        )
    if result["player_id"].eq("").any():
        raise DataContractError("special_teams_ratings contains an empty player ID")
    if result[[*integer_columns, *numeric_columns]].isna().any(axis=None):
        raise DataContractError("special_teams_ratings contains nonnumeric rating metadata")
    if not np.isfinite(result.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise DataContractError("special_teams_ratings contains a non-finite value")
    for column in integer_columns:
        if not np.equal(result[column], np.floor(result[column])).all():
            raise DataContractError(f"special_teams_ratings {column} values must be integers")
        result[column] = result[column].astype(int)
    if result.duplicated(["target_season", "player_id"]).any():
        raise DataContractError("special_teams_ratings contains duplicate season/player rows")
    if result["source_end_season"].ge(result["target_season"]).any():
        raise DataContractError(
            "special_teams_ratings source seasons must end before every target season"
        )
    if not result["lookback_seasons"].eq(PARTICIPATION_RATING_LOOKBACK_SEASONS).all():
        raise DataContractError("special_teams_ratings must use the frozen three-season lookback")
    if (
        not result["source_start_season"]
        .eq(result["target_season"] - PARTICIPATION_RATING_LOOKBACK_SEASONS)
        .all()
    ):
        raise DataContractError("special_teams_ratings source start does not match its target")
    if not result["source_end_season"].eq(result["target_season"] - 1).all():
        raise DataContractError("special_teams_ratings source end does not match its target")
    if result[["special_teams_plays", "source_plays"]].le(0).any(axis=None):
        raise DataContractError("special_teams_ratings play counts must be positive")
    expected_parameters = {
        "ridge_alpha": PARTICIPATION_RATING_RIDGE_ALPHA,
        "team_feature_scale": PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
        "reliability_prior_plays": PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
        "epa_clip": PARTICIPATION_RATING_EPA_CLIP,
    }
    for column, expected in expected_parameters.items():
        if not result[column].eq(expected).all():
            raise DataContractError(
                f"special_teams_ratings {column} differs from the frozen recipe"
            )
    if not result["rating_version"].eq(SPECIAL_TEAMS_RATING_VERSION).all():
        raise DataContractError("special_teams_ratings has an unsupported rating version")
    return result.sort_values(["target_season", "player_id"]).reset_index(drop=True)


def _fit_special_teams_window(
    table: pd.DataFrame,
) -> tuple[dict[str, float], Counter[str]]:
    counts: Counter[str] = Counter()
    design_rows: list[dict[str, float]] = []
    for row in table.itertuples(index=False):
        side_a = cast(tuple[str, ...], row.side_a_ids)
        side_b = cast(tuple[str, ...], row.side_b_ids)
        counts.update(side_a)
        counts.update(side_b)
        values = {f"st_player::{player_id}": 1.0 for player_id in side_a}
        values.update({f"st_player::{player_id}": -1.0 for player_id in side_b})
        values[f"st_team::{row.posteam}"] = PARTICIPATION_RATING_TEAM_FEATURE_SCALE
        design_rows.append(values)
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(design_rows)
    target = (
        pd.to_numeric(table["epa"], errors="raise")
        .clip(-PARTICIPATION_RATING_EPA_CLIP, PARTICIPATION_RATING_EPA_CLIP)
        .to_numpy(dtype=float)
    )
    estimator = Ridge(
        alpha=PARTICIPATION_RATING_RIDGE_ALPHA,
        solver="lsqr",
        fit_intercept=True,
        tol=1e-6,
    )
    estimator.fit(matrix, target)
    coefficients = dict(
        zip(vectorizer.get_feature_names_out(), np.asarray(estimator.coef_), strict=True)
    )
    return coefficients, counts


def build_season_lagged_special_teams_ratings(
    participation: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    target_seasons: Iterable[int] = SPECIAL_TEAMS_TARGET_SEASONS,
) -> pd.DataFrame:
    targets = sorted({int(season) for season in target_seasons})
    if not targets:
        raise ValueError("At least one target season is required")
    required_by_target = {
        target: tuple(range(target - PARTICIPATION_RATING_LOOKBACK_SEASONS, target))
        for target in targets
    }
    required_seasons = set().union(*map(set, required_by_target.values()))
    undeclared = required_seasons.difference(SPECIAL_TEAMS_SOURCE_SEASONS)
    if undeclared:
        raise DataContractError(
            f"Target seasons require undeclared source seasons: {sorted(undeclared)}"
        )
    table = build_special_teams_play_table(participation, pbp)
    available_seasons = set(table["season"].astype(int).unique())
    missing_required = required_seasons.difference(available_seasons)
    if missing_required:
        raise DataContractError(
            f"Complete source coverage is missing seasons: {sorted(missing_required)}"
        )
    output: list[dict[str, Any]] = []
    for target_season in targets:
        source_seasons = required_by_target[target_season]
        source_end = source_seasons[-1]
        if source_end >= target_season:
            raise DataContractError(
                f"Target season {target_season} has a non-lagged source end {source_end}"
            )
        training = table.loc[table["season"].astype(int).isin(source_seasons)].copy()
        coefficients, counts = _fit_special_teams_window(training)
        source_plays = len(training)
        for player_id in sorted(counts):
            plays = int(counts[player_id])
            reliability = plays / (plays + PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS)
            output.append(
                {
                    "target_season": target_season,
                    "player_id": player_id,
                    "special_teams_rating": float(
                        coefficients[f"st_player::{player_id}"] * reliability
                    ),
                    "special_teams_plays": plays,
                    "source_start_season": source_seasons[0],
                    "source_end_season": source_end,
                    "source_plays": source_plays,
                    "lookback_seasons": PARTICIPATION_RATING_LOOKBACK_SEASONS,
                    "ridge_alpha": PARTICIPATION_RATING_RIDGE_ALPHA,
                    "team_feature_scale": PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
                    "reliability_prior_plays": PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
                    "epa_clip": PARTICIPATION_RATING_EPA_CLIP,
                    "rating_version": SPECIAL_TEAMS_RATING_VERSION,
                }
            )
    if not output:
        raise DataContractError("No season-lagged special-teams ratings could be estimated")
    return canonicalize_special_teams_ratings(pd.DataFrame(output))
