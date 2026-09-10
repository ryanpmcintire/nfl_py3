from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import require_columns
from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.pbp import build_drive_table

PBP_TRAIT_RELIABILITY_SEED = 20260905
PBP_TRAIT_N_BOOT = 2000
PBP_TRAIT_N_NULL = 2000
PBP_TRAIT_MIN_PER_HALF = 1

FOURTH_DOWN_MAX_YDSTOGO = 3
FOURTH_DOWN_YARDLINE_LOW = 30.0
FOURTH_DOWN_YARDLINE_HIGH = 70.0
FOURTH_DOWN_ELIGIBLE_PLAY_TYPES = ("run", "pass", "punt", "field_goal")
FOURTH_DOWN_GO_PLAY_TYPES = ("run", "pass")

REQUIRED_PBP_COLUMNS = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "fixed_drive",
    "fixed_drive_result",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "play_type",
    "play_id",
    "epa",
    "wp",
    "play",
    "pass_attempt",
    "rush_attempt",
    "qb_kneel",
    "qb_spike",
    "aborted_play",
    "posteam_score",
    "score_differential",
)


def _require_pbp_columns(pbp: pd.DataFrame) -> None:
    require_columns(pbp, REQUIRED_PBP_COLUMNS, "play_by_play")


def _normalize_teams(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].replace(TEAM_ABBREVIATION_ALIASES)
    return result


def build_opening_drive_team_games(pbp: pd.DataFrame) -> pd.DataFrame:

    _require_pbp_columns(pbp)
    drives = build_drive_table(pbp)
    columns = [
        "game_id",
        "season",
        "week",
        "team",
        "opening_drive_td",
        "opening_drive_epa",
        "opening_drive_plays",
    ]
    if drives.empty:
        return pd.DataFrame(columns=columns)
    drives = _normalize_teams(drives, ("posteam", "defteam"))
    idx = drives.groupby(["game_id", "posteam"], sort=False)["fixed_drive"].idxmin()
    opening = drives.loc[idx].copy()
    opening["opening_drive_td"] = (
        opening["result"].astype("string").str.strip().str.lower().eq("touchdown").astype(float)
    )
    opening = opening.rename(
        columns={
            "posteam": "team",
            "drive_epa": "opening_drive_epa",
            "plays": "opening_drive_plays",
        }
    )
    return (
        opening[columns].sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    )


def build_opening_drive_team_seasons(pbp: pd.DataFrame) -> pd.DataFrame:

    team_games = build_opening_drive_team_games(pbp)
    columns = [
        "team",
        "season",
        "n_games",
        "opening_drive_td_count",
        "opening_drive_epa_sum",
        "opening_drive_plays_sum",
        "opening_drive_td_rate",
        "opening_drive_epa_per_play",
    ]
    if team_games.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        team_games.groupby(["team", "season"])
        .agg(
            n_games=("game_id", "nunique"),
            opening_drive_td_count=("opening_drive_td", "sum"),
            opening_drive_epa_sum=("opening_drive_epa", "sum"),
            opening_drive_plays_sum=("opening_drive_plays", "sum"),
        )
        .reset_index()
    )
    grouped["opening_drive_td_rate"] = grouped["opening_drive_td_count"] / grouped["n_games"]
    grouped["opening_drive_epa_per_play"] = grouped["opening_drive_epa_sum"] / grouped[
        "opening_drive_plays_sum"
    ].replace(0, np.nan)
    return grouped[columns]


def build_opening_drive_rolling(pbp: pd.DataFrame) -> pd.DataFrame:

    team_games = (
        build_opening_drive_team_games(pbp)
        .sort_values(["team", "season", "week", "game_id"])
        .reset_index(drop=True)
    )
    columns = [
        "game_id",
        "season",
        "week",
        "team",
        "opening_drive_td",
        "opening_drive_epa",
        "opening_drive_plays",
        "rolling_opening_drive_td_rate",
        "rolling_opening_drive_epa_per_play",
        "rolling_opening_drive_games",
    ]
    if team_games.empty:
        return pd.DataFrame(columns=columns)
    grp = team_games.groupby("team", sort=False)
    team_games["cum_td"] = grp["opening_drive_td"].cumsum()
    team_games["cum_epa"] = grp["opening_drive_epa"].cumsum()
    team_games["cum_plays"] = grp["opening_drive_plays"].cumsum()
    team_games["cum_games"] = team_games.groupby("team", sort=False).cumcount() + 1
    team_games["prior_cum_td"] = team_games.groupby("team", sort=False)["cum_td"].shift(1)
    team_games["prior_cum_epa"] = team_games.groupby("team", sort=False)["cum_epa"].shift(1)
    team_games["prior_cum_plays"] = team_games.groupby("team", sort=False)["cum_plays"].shift(1)
    team_games["prior_cum_games"] = team_games.groupby("team", sort=False)["cum_games"].shift(1)
    team_games["rolling_opening_drive_td_rate"] = (
        team_games["prior_cum_td"] / team_games["prior_cum_games"]
    )
    team_games["rolling_opening_drive_epa_per_play"] = team_games["prior_cum_epa"] / team_games[
        "prior_cum_plays"
    ].replace(0, np.nan)
    team_games["rolling_opening_drive_games"] = team_games["prior_cum_games"]
    return team_games[columns]


def build_third_quarter_point_diff_team_games(pbp: pd.DataFrame) -> pd.DataFrame:

    _require_pbp_columns(pbp)
    columns = ["game_id", "season", "week", "team", "q3_point_diff"]
    frame = pbp.loc[
        :,
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "posteam",
            "qtr",
            "score_differential",
            "play_id",
        ],
    ].copy()
    frame = _normalize_teams(frame, ("home_team", "away_team", "posteam"))
    for column in ("qtr", "score_differential", "play_id"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["posteam"].notna() & frame["qtr"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["home_lead_pre"] = np.where(
        frame["posteam"].to_numpy() == frame["home_team"].to_numpy(),
        frame["score_differential"].to_numpy(),
        -frame["score_differential"].to_numpy(),
    )
    frame = frame.sort_values(["game_id", "play_id"])
    q3_first = frame.loc[frame["qtr"].eq(3)].groupby("game_id", sort=False).first()
    q4_first = frame.loc[frame["qtr"].eq(4)].groupby("game_id", sort=False).first()
    if q3_first.empty or q4_first.empty:
        return pd.DataFrame(columns=columns)
    merged = q3_first[["season", "week", "home_team", "away_team", "home_lead_pre"]].rename(
        columns={"home_lead_pre": "lead_start_q3"}
    )
    merged = merged.join(
        q4_first[["home_lead_pre"]].rename(columns={"home_lead_pre": "lead_start_q4"}), how="inner"
    )
    merged["home_q3_point_diff"] = merged["lead_start_q4"] - merged["lead_start_q3"]
    merged = merged.reset_index()

    home_rows = merged[["game_id", "season", "week", "home_team", "home_q3_point_diff"]].rename(
        columns={"home_team": "team", "home_q3_point_diff": "q3_point_diff"}
    )
    away_rows = merged[["game_id", "season", "week", "away_team", "home_q3_point_diff"]].rename(
        columns={"away_team": "team"}
    )
    away_rows["q3_point_diff"] = -away_rows["home_q3_point_diff"]
    away_rows = away_rows.drop(columns=["home_q3_point_diff"])
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    return (
        team_games[columns]
        .sort_values(["season", "week", "game_id", "team"])
        .reset_index(drop=True)
    )


def build_third_quarter_team_seasons(pbp: pd.DataFrame) -> pd.DataFrame:

    team_games = build_third_quarter_point_diff_team_games(pbp)
    columns = ["team", "season", "n_games", "q3_point_diff"]
    if team_games.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        team_games.groupby(["team", "season"])
        .agg(n_games=("game_id", "nunique"), q3_point_diff=("q3_point_diff", "mean"))
        .reset_index()
    )
    return grouped[columns]


def build_third_quarter_rolling(pbp: pd.DataFrame) -> pd.DataFrame:

    team_games = (
        build_third_quarter_point_diff_team_games(pbp)
        .sort_values(["team", "season", "week", "game_id"])
        .reset_index(drop=True)
    )
    columns = [
        "game_id",
        "season",
        "week",
        "team",
        "q3_point_diff",
        "rolling_q3_point_diff",
        "rolling_q3_games",
    ]
    if team_games.empty:
        return pd.DataFrame(columns=columns)
    team_games["cum_sum"] = team_games.groupby("team", sort=False)["q3_point_diff"].cumsum()
    team_games["cum_games"] = team_games.groupby("team", sort=False).cumcount() + 1
    team_games["prior_cum_sum"] = team_games.groupby("team", sort=False)["cum_sum"].shift(1)
    team_games["prior_cum_games"] = team_games.groupby("team", sort=False)["cum_games"].shift(1)
    team_games["rolling_q3_point_diff"] = (
        team_games["prior_cum_sum"] / team_games["prior_cum_games"]
    )
    team_games["rolling_q3_games"] = team_games["prior_cum_games"]
    return team_games[columns]


def build_fourth_down_opportunities(pbp: pd.DataFrame) -> pd.DataFrame:

    _require_pbp_columns(pbp)
    columns = ["game_id", "season", "week", "team", "go_for_it"]
    frame = pbp.loc[
        :,
        [
            "game_id",
            "season",
            "week",
            "posteam",
            "down",
            "ydstogo",
            "yardline_100",
            "play_type",
            "qb_kneel",
            "qb_spike",
            "aborted_play",
        ],
    ].copy()
    frame = _normalize_teams(frame, ("posteam",))
    for column in (
        "down",
        "ydstogo",
        "yardline_100",
        "qb_kneel",
        "qb_spike",
        "aborted_play",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    eligible = (
        frame["posteam"].notna()
        & frame["down"].eq(4)
        & frame["ydstogo"].between(1, FOURTH_DOWN_MAX_YDSTOGO, inclusive="both")
        & frame["yardline_100"].between(
            FOURTH_DOWN_YARDLINE_LOW, FOURTH_DOWN_YARDLINE_HIGH, inclusive="both"
        )
        & frame["qb_kneel"].fillna(0).eq(0)
        & frame["qb_spike"].fillna(0).eq(0)
        & frame["aborted_play"].fillna(0).eq(0)
        & frame["play_type"].isin(FOURTH_DOWN_ELIGIBLE_PLAY_TYPES)
    )
    opportunities = frame.loc[
        eligible, ["game_id", "season", "week", "posteam", "play_type"]
    ].copy()
    if opportunities.empty:
        return pd.DataFrame(columns=columns)
    opportunities["go_for_it"] = (
        opportunities["play_type"].isin(FOURTH_DOWN_GO_PLAY_TYPES).astype(float)
    )
    opportunities = opportunities.rename(columns={"posteam": "team"})
    return (
        opportunities[columns]
        .sort_values(["season", "week", "game_id", "team"])
        .reset_index(drop=True)
    )


def build_fourth_down_team_games(pbp: pd.DataFrame) -> pd.DataFrame:

    opportunities = build_fourth_down_opportunities(pbp)
    columns = ["game_id", "season", "week", "team", "go_count", "eligible_count"]
    if opportunities.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        opportunities.groupby(["game_id", "season", "week", "team"])["go_for_it"]
        .agg(go_count="sum", eligible_count="count")
        .reset_index()
    )
    return grouped[columns]


def build_fourth_down_team_seasons(pbp: pd.DataFrame) -> pd.DataFrame:

    team_games = build_fourth_down_team_games(pbp)
    columns = ["team", "season", "n_games", "go_count", "eligible_count", "fourth_down_go_rate"]
    if team_games.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        team_games.groupby(["team", "season"])
        .agg(
            n_games=("game_id", "nunique"),
            go_count=("go_count", "sum"),
            eligible_count=("eligible_count", "sum"),
        )
        .reset_index()
    )
    grouped["fourth_down_go_rate"] = grouped["go_count"] / grouped["eligible_count"].replace(
        0, np.nan
    )
    return grouped[columns]


def build_fourth_down_rolling(pbp: pd.DataFrame) -> pd.DataFrame:

    team_games = (
        build_fourth_down_team_games(pbp)
        .sort_values(["team", "season", "week", "game_id"])
        .reset_index(drop=True)
    )
    columns = [
        "game_id",
        "season",
        "week",
        "team",
        "go_count",
        "eligible_count",
        "rolling_fourth_down_go_rate",
        "rolling_fourth_down_eligible",
    ]
    if team_games.empty:
        return pd.DataFrame(columns=columns)
    team_games["cum_go"] = team_games.groupby("team", sort=False)["go_count"].cumsum()
    team_games["cum_eligible"] = team_games.groupby("team", sort=False)["eligible_count"].cumsum()
    team_games["prior_cum_go"] = team_games.groupby("team", sort=False)["cum_go"].shift(1)
    team_games["prior_cum_eligible"] = team_games.groupby("team", sort=False)["cum_eligible"].shift(
        1
    )
    team_games["rolling_fourth_down_go_rate"] = team_games["prior_cum_go"] / team_games[
        "prior_cum_eligible"
    ].replace(0, np.nan)
    team_games["rolling_fourth_down_eligible"] = team_games["prior_cum_eligible"]
    return team_games[columns]


def build_odd_even_halves(long: pd.DataFrame, value_col: str, *, min_per_half: int) -> pd.DataFrame:

    columns = ["team", "season", "block_season", "value_a", "value_b"]
    frame = long.loc[:, ["team", "season", "week", value_col]].dropna(subset=[value_col]).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["half"] = np.where(frame["week"] % 2 == 0, "even", "odd")
    grouped = frame.groupby(["team", "season", "half"])[value_col]
    means = grouped.mean().unstack("half")
    if not {"odd", "even"}.issubset(means.columns):
        return pd.DataFrame(columns=columns)
    counts = grouped.size().unstack("half")
    odd_n = counts["odd"].reindex(means.index).fillna(0)
    even_n = counts["even"].reindex(means.index).fillna(0)
    keep = (odd_n >= min_per_half) & (even_n >= min_per_half)
    means = means.loc[keep].dropna(subset=["odd", "even"])
    result = means.reset_index().rename(columns={"odd": "value_a", "even": "value_b"})
    result["block_season"] = result["season"]
    return result[columns]


def build_season_to_season_pairs(team_season: pd.DataFrame, value_col: str) -> pd.DataFrame:

    columns = ["team", "season", "block_season", "value_a", "value_b"]
    frame = team_season.loc[:, ["team", "season", value_col]].dropna(subset=[value_col]).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["season"] = frame["season"].astype(int)
    shifted = frame.copy()
    shifted["season"] = shifted["season"] - 1
    shifted = shifted.rename(columns={value_col: "value_b"})[["team", "season", "value_b"]]
    merged = frame.rename(columns={value_col: "value_a"}).merge(
        shifted, on=["team", "season"], how="inner"
    )
    merged["block_season"] = merged["season"]
    return merged[columns]


def _season_blocked_bootstrap(
    value_a: np.ndarray, value_b: np.ndarray, block_season: np.ndarray, *, n_boot: int, seed: int
) -> dict[str, np.ndarray]:

    seasons = np.unique(block_season)
    season_to_idx = {season: np.where(block_season == season)[0] for season in seasons}
    n_seasons = len(seasons)
    rng = np.random.default_rng(seed)
    pearson_draws = np.full(n_boot, np.nan)
    spearman_draws = np.full(n_boot, np.nan)
    for draw in range(n_boot):
        drawn = rng.choice(seasons, size=n_seasons, replace=True)
        idx = np.concatenate([season_to_idx[season] for season in drawn])
        a, b = value_a[idx], value_b[idx]
        if len(a) >= 3 and a.std() > 0 and b.std() > 0:
            pearson_draws[draw] = np.corrcoef(a, b)[0, 1]
            spearman_draws[draw] = spearmanr(a, b).correlation
    return {"pearson": pearson_draws, "spearman": spearman_draws}


def _within_season_label_shuffle_null(
    value_a: np.ndarray, value_b: np.ndarray, block_season: np.ndarray, *, n_shuffle: int, seed: int
) -> np.ndarray:

    seasons = np.unique(block_season)
    rng = np.random.default_rng(seed)
    draws = np.full(n_shuffle, np.nan)
    for i in range(n_shuffle):
        shuffled_b = value_b.copy()
        for season in seasons:
            idx = np.where(block_season == season)[0]
            if len(idx) > 1:
                shuffled_b[idx] = rng.permutation(shuffled_b[idx])
        if value_a.std() > 0 and shuffled_b.std() > 0:
            draws[i] = np.corrcoef(value_a, shuffled_b)[0, 1]
    return draws


def paired_split_half_reliability(
    pairs: pd.DataFrame,
    *,
    metric: str,
    method: str,
    seed: int = PBP_TRAIT_RELIABILITY_SEED,
    n_boot: int = PBP_TRAIT_N_BOOT,
    n_null: int = PBP_TRAIT_N_NULL,
    spearman_brown: bool,
) -> dict[str, Any]:

    n = len(pairs)
    n_seasons = int(pairs["block_season"].nunique()) if n else 0
    base: dict[str, Any] = {
        "metric": metric,
        "method": method,
        "n_units": n,
        "n_seasons": n_seasons,
        "seed": seed,
        "n_boot": n_boot,
        "n_null": n_null,
    }
    if n < 3:
        return {
            **base,
            "status": "insufficient_units",
            "pearson_r": math.nan,
            "pearson_r_ci95": [math.nan, math.nan],
            "pearson_probability_positive": math.nan,
            "spearman_rho": math.nan,
            "spearman_rho_ci95": [math.nan, math.nan],
            "spearman_probability_positive": math.nan,
            "spearman_brown_full_length_reliability": None,
            "null_mean_r": math.nan,
            "null_sd_r": math.nan,
        }

    value_a = pairs["value_a"].to_numpy(dtype=float)
    value_b = pairs["value_b"].to_numpy(dtype=float)
    block_season = pairs["block_season"].to_numpy()

    pearson_r = (
        float(np.corrcoef(value_a, value_b)[0, 1])
        if value_a.std() > 0 and value_b.std() > 0
        else math.nan
    )
    spearman_rho = float(spearmanr(value_a, value_b).correlation)

    boots = _season_blocked_bootstrap(value_a, value_b, block_season, n_boot=n_boot, seed=seed)
    pearson_ci = [
        float(np.nanquantile(boots["pearson"], 0.025)),
        float(np.nanquantile(boots["pearson"], 0.975)),
    ]
    pearson_pp = float(probability_positive_from_draws(boots["pearson"], ignore_nan=True))
    spearman_ci = [
        float(np.nanquantile(boots["spearman"], 0.025)),
        float(np.nanquantile(boots["spearman"], 0.975)),
    ]
    spearman_pp = float(probability_positive_from_draws(boots["spearman"], ignore_nan=True))

    sb: float | None = None
    if spearman_brown and math.isfinite(pearson_r) and pearson_r > -1.0:
        candidate = (2.0 * pearson_r) / (1.0 + pearson_r)
        if math.isfinite(candidate) and -1.0 <= candidate <= 1.0:
            sb = float(candidate)

    null_draws = _within_season_label_shuffle_null(
        value_a, value_b, block_season, n_shuffle=n_null, seed=seed + 500_000
    )

    return {
        **base,
        "status": "measured",
        "pearson_r": pearson_r,
        "pearson_r_ci95": pearson_ci,
        "pearson_probability_positive": pearson_pp,
        "spearman_rho": spearman_rho,
        "spearman_rho_ci95": spearman_ci,
        "spearman_probability_positive": spearman_pp,
        "spearman_brown_full_length_reliability": sb,
        "null_mean_r": float(np.nanmean(null_draws)),
        "null_sd_r": float(np.nanstd(null_draws)),
    }


def compute_trait_reliability(
    long: pd.DataFrame,
    team_season: pd.DataFrame,
    *,
    metric: str,
    min_per_half: int = PBP_TRAIT_MIN_PER_HALF,
    seed: int = PBP_TRAIT_RELIABILITY_SEED,
    n_boot: int = PBP_TRAIT_N_BOOT,
    n_null: int = PBP_TRAIT_N_NULL,
) -> dict[str, Any]:

    within = paired_split_half_reliability(
        build_odd_even_halves(long, "value", min_per_half=min_per_half),
        metric=metric,
        method="within_season_odd_even_week",
        seed=seed,
        n_boot=n_boot,
        n_null=n_null,
        spearman_brown=True,
    )
    across = paired_split_half_reliability(
        build_season_to_season_pairs(team_season, "value"),
        metric=metric,
        method="season_to_season_same_franchise",
        seed=seed + 1,
        n_boot=n_boot,
        n_null=n_null,
        spearman_brown=False,
    )
    return {
        "metric": metric,
        "within_season_odd_even_week": within,
        "season_to_season_same_franchise": across,
    }


def run_all_trait_reliabilities(
    pbp: pd.DataFrame,
    *,
    seed: int = PBP_TRAIT_RELIABILITY_SEED,
    n_boot: int = PBP_TRAIT_N_BOOT,
    n_null: int = PBP_TRAIT_N_NULL,
    min_per_half: int = PBP_TRAIT_MIN_PER_HALF,
) -> dict[str, Any]:

    _require_pbp_columns(pbp)

    opening_games = build_opening_drive_team_games(pbp)
    opening_seasons = build_opening_drive_team_seasons(pbp)
    q3_games = build_third_quarter_point_diff_team_games(pbp)
    q3_seasons = build_third_quarter_team_seasons(pbp)
    fourth_opportunities = build_fourth_down_opportunities(pbp)
    fourth_seasons = build_fourth_down_team_seasons(pbp)

    results: dict[str, Any] = {}

    results["opening_drive_td_rate"] = {
        "n_team_seasons": len(opening_seasons),
        "n_team_games": len(opening_games),
        **compute_trait_reliability(
            opening_games.rename(columns={"opening_drive_td": "value"}),
            opening_seasons.rename(columns={"opening_drive_td_rate": "value"}),
            metric="opening_drive_td_rate",
            min_per_half=min_per_half,
            seed=seed,
            n_boot=n_boot,
            n_null=n_null,
        ),
    }

    results["opening_drive_epa_per_play"] = {
        "n_team_seasons": len(opening_seasons),
        "n_team_games": len(opening_games),
        **compute_trait_reliability(
            opening_games.assign(
                value=opening_games["opening_drive_epa"]
                / opening_games["opening_drive_plays"].replace(0, np.nan)
            ),
            opening_seasons.rename(columns={"opening_drive_epa_per_play": "value"}),
            metric="opening_drive_epa_per_play",
            min_per_half=min_per_half,
            seed=seed + 10,
            n_boot=n_boot,
            n_null=n_null,
        ),
    }

    results["q3_point_diff"] = {
        "n_team_seasons": len(q3_seasons),
        "n_team_games": len(q3_games),
        **compute_trait_reliability(
            q3_games.rename(columns={"q3_point_diff": "value"}),
            q3_seasons.rename(columns={"q3_point_diff": "value"}),
            metric="q3_point_diff",
            min_per_half=min_per_half,
            seed=seed + 20,
            n_boot=n_boot,
            n_null=n_null,
        ),
    }

    results["fourth_down_go_rate"] = {
        "n_team_seasons": len(fourth_seasons),
        "n_opportunities": len(fourth_opportunities),
        **compute_trait_reliability(
            fourth_opportunities.rename(columns={"go_for_it": "value"}),
            fourth_seasons.rename(columns={"fourth_down_go_rate": "value"}),
            metric="fourth_down_go_rate",
            min_per_half=min_per_half,
            seed=seed + 30,
            n_boot=n_boot,
            n_null=n_null,
        ),
    }

    return results


__all__ = [
    "FOURTH_DOWN_ELIGIBLE_PLAY_TYPES",
    "FOURTH_DOWN_GO_PLAY_TYPES",
    "FOURTH_DOWN_MAX_YDSTOGO",
    "FOURTH_DOWN_YARDLINE_HIGH",
    "FOURTH_DOWN_YARDLINE_LOW",
    "PBP_TRAIT_MIN_PER_HALF",
    "PBP_TRAIT_N_BOOT",
    "PBP_TRAIT_N_NULL",
    "PBP_TRAIT_RELIABILITY_SEED",
    "REQUIRED_PBP_COLUMNS",
    "build_fourth_down_opportunities",
    "build_fourth_down_rolling",
    "build_fourth_down_team_games",
    "build_fourth_down_team_seasons",
    "build_odd_even_halves",
    "build_opening_drive_rolling",
    "build_opening_drive_team_games",
    "build_opening_drive_team_seasons",
    "build_season_to_season_pairs",
    "build_third_quarter_point_diff_team_games",
    "build_third_quarter_rolling",
    "build_third_quarter_team_seasons",
    "compute_trait_reliability",
    "paired_split_half_reliability",
    "run_all_trait_reliabilities",
]
