from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    GAP_V3_BIAS_METRICS,
    GAP_V3_TRAVEL_FEATURE_COLUMNS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.features import add_ats_outcomes
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot

BLOWOUT_MARGIN_POINTS = 17.0
LONG_DISTANCE_MI = 1500.0
RETURN_TRIP_MAX_HOME_REST_DAYS = 8
EARTH_RADIUS_MI = 3958.8

POSTSEASON_GAME_TYPES = ("WC", "DIV", "CON", "SB")


def _canonical(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _column_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:

    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return frame[column]


def latest_schedules_snapshot(repo_root: Path) -> Path:

    candidates = sorted((repo_root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {repo_root}")
    return candidates[-1]


def _team_long_table(schedules: pd.DataFrame) -> pd.DataFrame:

    df = schedules.copy()
    df = df.loc[df["game_type"].astype(str) == "REG"].copy()
    df = add_ats_outcomes(df)
    df["home_team"] = _canonical(df["home_team"])
    df["away_team"] = _canonical(df["away_team"])
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    df["div_game"] = (
        pd.to_numeric(_column_or_default(df, "div_game", 0), errors="coerce").fillna(0).astype(int)
    )
    df["weekday"] = _column_or_default(df, "weekday", "").astype(str)

    sides = []
    for is_home in (True, False):
        team_col, opp_col = ("home_team", "away_team") if is_home else ("away_team", "home_team")
        sign = 1.0 if is_home else -1.0
        sides.append(
            pd.DataFrame(
                {
                    "game_id": df["game_id"].astype(str),
                    "season": df["season"],
                    "week": df["week"],
                    "gameday": df["gameday"],
                    "team": df[team_col],
                    "opponent": df[opp_col],
                    "is_home": is_home,
                    "div_game": df["div_game"],
                    "weekday": df["weekday"],
                    "team_score_margin": sign * pd.to_numeric(df["result"], errors="coerce"),
                }
            )
        )
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.loc[long_df["team_score_margin"].notna()].copy()
    return long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)


def _add_gap_bias_flags(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df = long_df.copy()

    ordered = long_df.sort_values(["team", "opponent", "season", "gameday"]).copy()
    grouped = ordered.groupby(["team", "opponent", "season"], sort=False)
    meeting_rank = grouped.cumcount()
    first_margin = grouped["team_score_margin"].transform("first")
    ordered["gap_division_revenge"] = (meeting_rank >= 1) & (first_margin < 0)
    long_df = long_df.merge(
        ordered[["game_id", "team", "gap_division_revenge"]], on=["game_id", "team"], how="left"
    )

    grouped = long_df.groupby(["team", "season"], sort=False)
    prior_div = grouped["div_game"].shift(1)
    next_div = grouped["div_game"].shift(-1)
    long_df["gap_sandwich_spot"] = (long_df["div_game"] == 0) & (prior_div == 1) & (next_div == 1)

    grouped = long_df.groupby(["team", "season"], sort=False)
    prior_margin = grouped["team_score_margin"].shift(1)
    long_df["gap_post_blowout_win_letdown"] = prior_margin >= BLOWOUT_MARGIN_POINTS
    long_df["gap_post_blowout_loss_bounce"] = prior_margin <= -BLOWOUT_MARGIN_POINTS

    for metric in GAP_V3_BIAS_METRICS:
        long_df[metric] = long_df[metric].fillna(False).astype(bool)
    return long_df


def build_gap_bias_features(schedules: pd.DataFrame) -> pd.DataFrame:

    long_df = _add_gap_bias_flags(_team_long_table(schedules))
    wide_frames = []
    for is_home, side in ((True, "home"), (False, "away")):
        subset = long_df.loc[long_df["is_home"] == is_home, ["game_id", *GAP_V3_BIAS_METRICS]]
        subset = subset.rename(columns={m: f"{m}_{side}" for m in GAP_V3_BIAS_METRICS}).astype(
            {f"{m}_{side}": "float64" for m in GAP_V3_BIAS_METRICS}
        )
        wide_frames.append(subset.set_index("game_id"))
    wide = wide_frames[0].join(wide_frames[1], how="outer")
    for metric in GAP_V3_BIAS_METRICS:
        wide[f"{metric}_diff"] = wide[f"{metric}_home"] - wide[f"{metric}_away"]
    return wide.reset_index()


def team_season_penalty_rate(pbp: pd.DataFrame) -> pd.DataFrame:

    plays = pbp.loc[pbp["posteam"].notna()].copy()
    plays["penalty"] = pd.to_numeric(plays["penalty"], errors="coerce").fillna(0.0)
    plays["team"] = _canonical(plays["posteam"])
    grouped = plays.groupby(["season", "team"]).agg(
        plays=("penalty", "size"), penalties=("penalty", "sum")
    )
    grouped["rate"] = grouped["penalties"] / grouped["plays"]
    return grouped.reset_index()


def build_gap_penalty_feature(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:

    rate = team_season_penalty_rate(pbp)
    lag = rate.copy()
    lag["prev_season"] = lag["season"] + 1
    lag = lag.rename(columns={"rate": "prior_rate"})[["team", "prev_season", "prior_rate"]]

    games = schedules[["game_id", "season", "home_team", "away_team"]].copy()
    games["game_id"] = games["game_id"].astype(str)
    games["season"] = pd.to_numeric(games["season"], errors="raise").astype(int)
    games["home_team"] = _canonical(games["home_team"])
    games["away_team"] = _canonical(games["away_team"])

    result = games[["game_id"]].copy()
    for side in ("home", "away"):
        joined = games[["game_id", "season", f"{side}_team"]].merge(
            lag,
            left_on=["season", f"{side}_team"],
            right_on=["prev_season", "team"],
            how="left",
        )
        result[f"{side}_penalty_rate_prior"] = joined["prior_rate"].to_numpy()
    result["diff_penalty_rate_prior"] = (
        result["home_penalty_rate_prior"] - result["away_penalty_rate_prior"]
    )
    return result[["game_id", "diff_penalty_rate_prior"]]


def load_stadium_coordinates(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def build_gap_travel_rest_features(
    schedules: pd.DataFrame, coords: dict[str, dict[str, Any]]
) -> pd.DataFrame:

    df = schedules.copy()
    df = df.loc[df["game_type"].astype(str) == "REG"].copy()
    df["game_id"] = df["game_id"].astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["home_team"] = _canonical(df["home_team"])
    df["away_team"] = _canonical(df["away_team"])
    df["home_rest"] = pd.to_numeric(_column_or_default(df, "home_rest", np.nan), errors="coerce")
    df["gameday_dt"] = pd.to_datetime(_column_or_default(df, "gameday", pd.NaT), errors="coerce")
    df["weekday"] = _column_or_default(df, "weekday", "").astype(str)
    df["stadium"] = _column_or_default(df, "stadium", None)
    df["location"] = _column_or_default(df, "location", "Home")

    thursday_flag = (df["weekday"] == "Thursday").astype(float)

    home_rows = df.loc[df["location"] == "Home"]
    modal_stadium = home_rows.groupby(["home_team", "season"])["stadium"].agg(
        lambda s: s.mode().iat[0] if not s.mode(dropna=True).empty else None  # type: ignore[type-var]
    )

    def coord(name: object) -> dict[str, Any] | None:
        return coords.get(name) if isinstance(name, str) else None

    def team_home_coord(team: str, season: int) -> dict[str, Any] | None:
        name = modal_stadium.get((team, season))
        return coord(name)

    long_rows: list[dict[str, Any]] = []
    for _, g in df.iterrows():
        venue = coord(g["stadium"])
        for team in (g["home_team"], g["away_team"]):
            home_coord = team_home_coord(team, g["season"])
            if venue is None or home_coord is None:
                distance = np.nan
            else:
                distance = haversine_mi(
                    home_coord["lat"], home_coord["lon"], venue["lat"], venue["lon"]
                )
            long_rows.append(
                {
                    "game_id": g["game_id"],
                    "team": team,
                    "season": g["season"],
                    "gameday_dt": g["gameday_dt"],
                    "own_travel_mi": distance,
                }
            )
    long_df = pd.DataFrame(long_rows).sort_values(["team", "season", "gameday_dt"])
    long_df["prev_own_travel_mi"] = long_df.groupby(["team", "season"])["own_travel_mi"].shift(1)

    home_prev = df[["game_id", "home_team"]].merge(
        long_df[["game_id", "team", "prev_own_travel_mi"]],
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )[["game_id", "prev_own_travel_mi"]]

    result = df[["game_id"]].copy()
    result["gap_thursday_pure_flag"] = thursday_flag.to_numpy()
    result = result.merge(home_prev, on="game_id", how="left")
    hangover = (result["prev_own_travel_mi"] >= LONG_DISTANCE_MI) & (
        df["home_rest"].to_numpy() <= RETURN_TRIP_MAX_HOME_REST_DAYS
    )
    result["gap_return_trip_hangover_flag"] = hangover.fillna(False).astype(float)
    return result[["game_id", "gap_thursday_pure_flag", "gap_return_trip_hangover_flag"]]


def attach_weak_stack_v3_gap_features(base: pd.DataFrame, *, repo_root: Path) -> pd.DataFrame:

    schedules_path = latest_schedules_snapshot(repo_root)
    schedules = pd.read_parquet(schedules_path)

    bias = build_gap_bias_features(schedules)
    coords = load_stadium_coordinates(repo_root / "registry" / "stadium_coordinates.json")
    travel = build_gap_travel_rest_features(schedules, coords)

    snapshot = latest_pbp_snapshot(repo_root / "data" / "pbp" / "raw")
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    penalty = build_gap_penalty_feature(pbp, schedules)

    result = base.copy()
    result["game_id"] = result["game_id"].astype(str)
    for frame in (bias, travel, penalty):
        frame = frame.copy()
        frame["game_id"] = frame["game_id"].astype(str)
        result = result.merge(frame, on="game_id", how="left")

    for metric in GAP_V3_BIAS_METRICS:
        for side in ("home", "away", "diff"):
            result[f"{metric}_{side}"] = result[f"{metric}_{side}"].fillna(0.0)
    for column in GAP_V3_TRAVEL_FEATURE_COLUMNS:
        result[column] = result[column].fillna(0.0)
    return result
