from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.experiment_runner import (
    _REFEREE_POSITION,
    _REFEREE_SEASON_TYPE,
    _build_referee_trait_data,
    _latest_officials_snapshot,
)
from nfl_ats.officials_archive import load_officials
from nfl_ats.pbp_coaching_traits import (
    PBP_TRAIT_N_BOOT,
    PBP_TRAIT_N_NULL,
    PBP_TRAIT_RELIABILITY_SEED,
    build_odd_even_halves,
    build_season_to_season_pairs,
    paired_split_half_reliability,
)
from nfl_ats.schedule_flag_features import _attach, default_opener_lines
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]

CREW_HOME_BIAS_COLUMN = "crew_home_bias_flag"
SECOND_MEETING_FAVORITE_COLUMN = "crew_second_meeting_favorite_flag"
ROOKIE_CREW_UNDERDOG_COLUMN = "rookie_crew_underdog_flag"

TRAILING_HOME_BIAS_MIN_GAMES = 3

ROOKIE_ELIGIBLE_SEASON_FLOOR = 2016
ROOKIE_PRIOR_EXPERIENCE_MAX = 1

_PENALTY_TABLE_REQUIRED_COLUMNS = {
    "game_id",
    "official_name",
    "season",
    "week",
    "home_team",
    "away_team",
    "penalties_total",
    "home_minus_away",
}


def _require_penalty_table_columns(table: pd.DataFrame) -> None:
    missing = sorted(_PENALTY_TABLE_REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise DataContractError(f"officials penalty table is missing columns: {', '.join(missing)}")


def _rookie_eligible_season_floor(seasons: pd.Series) -> int:
    if seasons.empty:
        return ROOKIE_ELIGIBLE_SEASON_FLOOR
    return int(cast(Any, seasons.min())) + 1


def home_away_penalty_game_table(repo_root: Path | None = None) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    officials_path, game_penalties_path, _snapshot_id = _latest_officials_snapshot(root)
    officials = load_officials(root, officials_path=officials_path)
    refs = officials.loc[
        (officials["position"] == _REFEREE_POSITION)
        & (officials["season_type"] == _REFEREE_SEASON_TYPE)
    ].copy()

    schedules = pd.read_parquet(latest_schedules_snapshot(root)).loc[
        :, ["game_id", "old_game_id", "home_team", "away_team"]
    ]
    refs = refs.merge(
        schedules, left_on="game_id", right_on="old_game_id", how="inner", suffixes=("_legacy", "")
    )
    refs = refs.loc[
        :,
        [
            "game_id",
            "official_name",
            "season",
            "week",
            "season_type",
            "home_team",
            "away_team",
        ],
    ]

    game_penalties = pd.read_parquet(game_penalties_path)
    required_gp = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "penalties_total",
        "penalties_on_home",
        "penalties_on_away",
    }
    missing_gp = sorted(required_gp.difference(game_penalties.columns))
    if missing_gp:
        raise DataContractError(
            f"{game_penalties_path} is missing columns: {', '.join(missing_gp)}"
        )

    merged = refs.merge(game_penalties, on="game_id", how="left", suffixes=("_crew", ""))
    for column in sorted(
        set(refs.columns).intersection(set(game_penalties.columns)).difference({"game_id"})
    ):
        merged[column] = merged[column].fillna(merged[f"{column}_crew"])
        merged = merged.drop(columns=[f"{column}_crew"])
    merged["game_id"] = merged["game_id"].astype(str)
    merged["season"] = pd.to_numeric(merged["season"], errors="raise").astype(int)
    merged["week"] = pd.to_numeric(merged["week"], errors="raise").astype(int)
    merged["home_minus_away"] = merged["penalties_on_home"] - merged["penalties_on_away"]
    return merged


def officials_home_bias_reliability(
    repo_root: Path | None = None,
    *,
    table: pd.DataFrame | None = None,
    seed: int = PBP_TRAIT_RELIABILITY_SEED,
    n_boot: int = PBP_TRAIT_N_BOOT,
    n_null: int = PBP_TRAIT_N_NULL,
) -> dict[str, Any]:

    table = table if table is not None else home_away_penalty_game_table(repo_root or REPO_ROOT)

    long = table.rename(columns={"official_name": "team"})[
        ["team", "season", "week", "home_minus_away"]
    ]
    within_pairs = build_odd_even_halves(long, "home_minus_away", min_per_half=1)
    within = paired_split_half_reliability(
        within_pairs,
        metric="referee_home_minus_away_penalty_count",
        method="within_season_odd_even_week",
        seed=seed,
        n_boot=n_boot,
        n_null=n_null,
        spearman_brown=True,
    )

    team_season = (
        table.groupby(["official_name", "season"])["home_minus_away"]
        .mean()
        .reset_index()
        .rename(columns={"official_name": "team"})
    )
    across_pairs = build_season_to_season_pairs(team_season, "home_minus_away")
    across = paired_split_half_reliability(
        across_pairs,
        metric="referee_home_minus_away_penalty_count",
        method="season_to_season_same_referee",
        seed=seed + 1,
        n_boot=n_boot,
        n_null=n_null,
        spearman_brown=False,
    )

    return {
        "n_game_rows": len(table),
        "n_distinct_officials": int(table["official_name"].nunique()),
        "n_seasons": int(table["season"].nunique()),
        "within_season_odd_even_week": within,
        "season_to_season_same_referee": across,
    }


def trailing_home_bias_table(
    repo_root: Path | None = None, *, table: pd.DataFrame | None = None
) -> pd.DataFrame:

    table = table if table is not None else home_away_penalty_game_table(repo_root or REPO_ROOT)
    _require_penalty_table_columns(table)
    table = table.sort_values(["official_name", "season", "week", "game_id"]).reset_index(drop=True)

    def _trailing(group: pd.DataFrame) -> pd.Series:
        trailing_mean = group["home_minus_away"].expanding().mean().shift(1)
        prior_count = np.arange(len(group))
        eligible = prior_count >= TRAILING_HOME_BIAS_MIN_GAMES
        return pd.Series(np.where(eligible, trailing_mean, np.nan), index=group.index)

    table["trailing_home_bias"] = table.groupby(
        ["official_name", "season"], group_keys=False
    ).apply(_trailing)
    return table


def derive_crew_home_bias_features(
    repo_root: Path | None = None, *, table: pd.DataFrame | None = None
) -> pd.DataFrame:

    table = trailing_home_bias_table(repo_root, table=table)
    valid_mask = table["trailing_home_bias"].notna()
    quartile = pd.Series(np.nan, index=table.index)
    if int(valid_mask.sum()) >= 4:
        quartile.loc[valid_mask] = (
            pd.qcut(table.loc[valid_mask, "trailing_home_bias"], 4, labels=[1, 2, 3, 4])
            .astype(int)
            .astype(float)
        )
    flag = quartile.eq(4.0).fillna(False).astype(float)
    out = pd.DataFrame({"game_id": table["game_id"].astype(str), CREW_HOME_BIAS_COLUMN: flag})
    return out.drop_duplicates("game_id").reset_index(drop=True)


def attach_crew_home_bias_features(
    features: pd.DataFrame, *, repo_root: Path | None = None, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:

    root = repo_root or REPO_ROOT

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        del sched
        return derive_crew_home_bias_features(root)

    merged = _attach(features, schedule, _derive, (CREW_HOME_BIAS_COLUMN,))
    merged[CREW_HOME_BIAS_COLUMN] = merged[CREW_HOME_BIAS_COLUMN].fillna(0.0)
    return merged


def crew_familiarity_table(
    repo_root: Path | None = None, *, table: pd.DataFrame | None = None
) -> pd.DataFrame:

    table = table if table is not None else home_away_penalty_game_table(repo_root or REPO_ROOT)
    _require_penalty_table_columns(table)
    table = table.sort_values(["official_name", "season", "week", "game_id"]).reset_index(drop=True)

    seen: dict[tuple[str, int], set[str]] = {}
    flags: list[bool] = []
    for row in table.itertuples(index=False):
        key = (str(row.official_name), int(cast(Any, row.season)))
        teams_seen = seen.setdefault(key, set())
        flags.append(bool(row.home_team in teams_seen or row.away_team in teams_seen))
        teams_seen.add(str(row.home_team))
        teams_seen.add(str(row.away_team))
    table = table.assign(second_meeting=flags)
    return table


def describe_crew_familiarity(
    repo_root: Path | None = None, *, table: pd.DataFrame | None = None
) -> dict[str, Any]:

    table = crew_familiarity_table(repo_root, table=table)
    flagged = table.loc[table["second_meeting"]]
    unflagged = table.loc[~table["second_meeting"]]
    return {
        "n_games_with_referee": len(table),
        "n_second_meeting": len(flagged),
        "pct_second_meeting": float(len(flagged) / len(table)) if len(table) else float("nan"),
        "mean_penalties_total_second_meeting": float(flagged["penalties_total"].mean())
        if len(flagged)
        else float("nan"),
        "mean_penalties_total_first_meeting": float(unflagged["penalties_total"].mean())
        if len(unflagged)
        else float("nan"),
        "penalties_total_diff_second_minus_first": (
            float(flagged["penalties_total"].mean() - unflagged["penalties_total"].mean())
            if len(flagged) and len(unflagged)
            else float("nan")
        ),
    }


def derive_second_meeting_favorite_features(
    repo_root: Path | None,
    opener_lines: pd.DataFrame,
    *,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:

    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    familiarity = crew_familiarity_table(repo_root, table=table)[
        ["game_id", "second_meeting"]
    ].drop_duplicates("game_id")
    merged = familiarity.merge(
        opener_lines[["game_id", "tue_open_home_spread"]], on="game_id", how="left"
    )
    spread = merged["tue_open_home_spread"]
    home_favorite = merged["second_meeting"] & spread.notna() & spread.gt(0.0)
    away_favorite = merged["second_meeting"] & spread.notna() & spread.lt(0.0)
    flag = np.where(home_favorite, 1.0, np.where(away_favorite, -1.0, 0.0))
    return pd.DataFrame(
        {"game_id": merged["game_id"].astype(str), SECOND_MEETING_FAVORITE_COLUMN: flag}
    )


def attach_second_meeting_favorite_features(
    features: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:

    root = repo_root or REPO_ROOT

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_second_meeting_favorite_features(root, lines)

    merged = _attach(features, schedule, _derive, (SECOND_MEETING_FAVORITE_COLUMN,))
    merged[SECOND_MEETING_FAVORITE_COLUMN] = merged[SECOND_MEETING_FAVORITE_COLUMN].fillna(0.0)
    return merged


def describe_referee_left_censoring(repo_root: Path | None = None) -> dict[str, Any]:

    root = repo_root or REPO_ROOT
    officials_path, _game_penalties_path, _snapshot_id = _latest_officials_snapshot(root)
    officials = load_officials(root, officials_path=officials_path)
    refs = officials.loc[
        (officials["position"] == _REFEREE_POSITION)
        & (officials["season_type"] == _REFEREE_SEASON_TYPE)
    ]
    first_season = refs.groupby("official_name")["season"].min()
    floor = _rookie_eligible_season_floor(refs["season"])
    n_censored = int((first_season < floor).sum())
    n_genuine = int((first_season >= floor).sum())
    return {
        "n_officials_total": len(first_season),
        "n_censored_2015_debut": n_censored,
        "n_genuine_debut_after_floor": n_genuine,
    }


def rookie_crew_table(
    repo_root: Path | None = None, *, trait: pd.DataFrame | None = None
) -> pd.DataFrame:

    if trait is not None:
        return trait[["game_id", "official_name", "season", "prior_seasons_experience"]].copy()
    root = repo_root or REPO_ROOT
    built = _build_referee_trait_data(root).game_trait
    return built[["game_id", "official_name", "season", "prior_seasons_experience"]].copy()


def derive_rookie_crew_underdog_features(
    repo_root: Path | None,
    opener_lines: pd.DataFrame,
    *,
    trait: pd.DataFrame | None = None,
) -> pd.DataFrame:

    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    trait = rookie_crew_table(repo_root, trait=trait)
    floor = _rookie_eligible_season_floor(trait["season"])
    eligible_season = trait["season"] >= floor
    is_rookie = trait["prior_seasons_experience"].le(ROOKIE_PRIOR_EXPERIENCE_MAX)
    rookie_crew = eligible_season & is_rookie

    merged = trait.assign(rookie_crew=rookie_crew).merge(
        opener_lines[["game_id", "tue_open_home_spread"]], on="game_id", how="left"
    )
    spread = merged["tue_open_home_spread"]
    home_dog = merged["rookie_crew"] & spread.notna() & spread.lt(0.0)
    away_dog = merged["rookie_crew"] & spread.notna() & spread.gt(0.0)
    flag = np.where(home_dog, 1.0, np.where(away_dog, -1.0, 0.0))
    return pd.DataFrame(
        {"game_id": merged["game_id"].astype(str), ROOKIE_CREW_UNDERDOG_COLUMN: flag}
    )


def attach_rookie_crew_underdog_features(
    features: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:

    root = repo_root or REPO_ROOT

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_rookie_crew_underdog_features(root, lines)

    merged = _attach(features, schedule, _derive, (ROOKIE_CREW_UNDERDOG_COLUMN,))
    merged[ROOKIE_CREW_UNDERDOG_COLUMN] = merged[ROOKIE_CREW_UNDERDOG_COLUMN].fillna(0.0)
    return merged


__all__ = [
    "CREW_HOME_BIAS_COLUMN",
    "ROOKIE_CREW_UNDERDOG_COLUMN",
    "ROOKIE_ELIGIBLE_SEASON_FLOOR",
    "ROOKIE_PRIOR_EXPERIENCE_MAX",
    "SECOND_MEETING_FAVORITE_COLUMN",
    "TRAILING_HOME_BIAS_MIN_GAMES",
    "attach_crew_home_bias_features",
    "attach_rookie_crew_underdog_features",
    "attach_second_meeting_favorite_features",
    "crew_familiarity_table",
    "derive_crew_home_bias_features",
    "derive_rookie_crew_underdog_features",
    "derive_second_meeting_favorite_features",
    "describe_crew_familiarity",
    "describe_referee_left_censoring",
    "home_away_penalty_game_table",
    "officials_home_bias_reliability",
    "rookie_crew_table",
    "trailing_home_bias_table",
]
