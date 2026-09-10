from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.pbp import analysis_plays, load_pbp_snapshot, snapshot_from_root

WINDOW_GAMES = 4
MIN_WINDOW_OBS = 3
MIN_QUANTILE_POOL = 200

QUARTILE_UNASSIGNED = -1
QUARTILE_BOTTOM = 0
QUARTILE_MIDDLE = 1
QUARTILE_TOP = 2

FLAG_TABLE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_press_allow_w",
    "away_press_allow_w",
    "home_press_gen_w",
    "away_press_gen_w",
    "home_offense_flagged",
    "away_offense_flagged",
    "back_side",
)


def build_game_pressure_traits(pbp_snapshot_path: Path) -> pd.DataFrame:

    snapshot = snapshot_from_root(pbp_snapshot_path)
    plays = analysis_plays(load_pbp_snapshot(snapshot))
    plays = plays.loc[plays["competitive_play"]].copy()
    plays["posteam"] = plays["posteam"].replace(TEAM_ABBREVIATION_ALIASES)
    plays["defteam"] = plays["defteam"].replace(TEAM_ABBREVIATION_ALIASES)
    for column in ("qb_dropback", "sack", "qb_hit"):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")

    dropbacks = plays.loc[plays["qb_dropback"].fillna(0).eq(1)].copy()
    dropbacks["pressure"] = (
        dropbacks["sack"].fillna(0).eq(1) | dropbacks["qb_hit"].fillna(0).eq(1)
    ).astype(float)

    allowed = (
        dropbacks.groupby(["game_id", "posteam"], sort=False)["pressure"]
        .mean()
        .rename("press_allow_g")
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    generated = (
        dropbacks.groupby(["game_id", "defteam"], sort=False)["pressure"]
        .mean()
        .rename("press_gen_g")
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    traits = allowed.merge(generated, on=["game_id", "team"], how="outer")
    traits["game_id"] = traits["game_id"].astype(str)
    traits["team"] = traits["team"].astype(str)
    return traits


def expanding_quartile_flags(values: pd.Series, blocks: pd.Series) -> np.ndarray:

    raw = values.to_numpy(dtype=np.float64)
    block_values = blocks.to_numpy()
    sort_order = np.argsort(block_values, kind="stable")
    sorted_values = raw[sort_order]
    sorted_blocks = block_values[sort_order]
    sorted_flags = np.full(len(raw), np.int8(QUARTILE_UNASSIGNED))

    pool: list[np.ndarray] = []
    start = 0
    total = len(raw)
    while start < total:
        end = start
        while end < total and sorted_blocks[end] == sorted_blocks[start]:
            end += 1
        if pool:
            pooled = np.concatenate(pool)
            if len(pooled) >= MIN_QUANTILE_POOL:
                q25, q75 = np.quantile(pooled, [0.25, 0.75])
                segment = sorted_values[start:end]
                assigned = ~np.isnan(segment)
                codes = np.where(
                    segment <= q25,
                    np.int8(QUARTILE_BOTTOM),
                    np.where(segment >= q75, np.int8(QUARTILE_TOP), np.int8(QUARTILE_MIDDLE)),
                )
                sorted_flags[start:end] = np.where(assigned, codes, np.int8(QUARTILE_UNASSIGNED))
        present = sorted_values[start:end]
        present = present[~np.isnan(present)]
        if len(present):
            pool.append(present)
        start = end

    result = np.empty(total, dtype=np.int8)
    result[sort_order] = sorted_flags
    return result


def _team_game_windows(schedule: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:

    sides = []
    for is_home in (True, False):
        sides.append(
            pd.DataFrame(
                {
                    "game_id": schedule["game_id"].astype(str),
                    "season": schedule["season"].astype(int),
                    "week": schedule["week"].astype(int),
                    "gameday": schedule["gameday"],
                    "team": (schedule["home_team"] if is_home else schedule["away_team"]).astype(
                        str
                    ),
                    "opponent": (
                        schedule["away_team"] if is_home else schedule["home_team"]
                    ).astype(str),
                    "is_home": is_home,
                }
            )
        )
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.merge(traits, on=["game_id", "team"], how="left", validate="many_to_one")
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    long_df = long_df.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    for source, target in (("press_allow_g", "press_allow_w"), ("press_gen_g", "press_gen_w")):
        values = pd.to_numeric(long_df[source], errors="coerce")
        long_df[target] = values.groupby(long_df["team"]).transform(
            lambda series: series.shift(1).rolling(WINDOW_GAMES, min_periods=MIN_WINDOW_OBS).mean()
        )

    long_df["press_allow_q"] = expanding_quartile_flags(
        long_df["press_allow_w"], long_df["week_block"]
    )
    long_df["press_gen_q"] = expanding_quartile_flags(long_df["press_gen_w"], long_df["week_block"])
    return long_df


def build_flag_table(schedule: pd.DataFrame, pbp_snapshot_path: Path) -> pd.DataFrame:

    required = {"game_id", "season", "week", "gameday", "home_team", "away_team"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")

    frame = schedule.copy()
    for column in ("home_team", "away_team"):
        frame[column] = frame[column].astype(str).replace(TEAM_ABBREVIATION_ALIASES)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    if frame["gameday"].isna().any():
        raise DataContractError("schedule has games without a gameday")

    traits = build_game_pressure_traits(pbp_snapshot_path)
    long_df = _team_game_windows(frame, traits)

    home = long_df.loc[long_df["is_home"]].set_index("game_id")
    away = long_df.loc[~long_df["is_home"]].set_index("game_id")

    table = pd.DataFrame(index=frame["game_id"].astype(str))
    table["season"] = frame.set_index(frame["game_id"].astype(str))["season"].astype(int)
    table["week"] = frame.set_index(frame["game_id"].astype(str))["week"].astype(int)
    table["home_team"] = frame.set_index(frame["game_id"].astype(str))["home_team"]
    table["away_team"] = frame.set_index(frame["game_id"].astype(str))["away_team"]
    table["home_press_allow_w"] = home["press_allow_w"]
    table["away_press_allow_w"] = away["press_allow_w"]
    table["home_press_gen_w"] = home["press_gen_w"]
    table["away_press_gen_w"] = away["press_gen_w"]

    home_flagged = home["press_allow_q"].eq(QUARTILE_TOP) & away["press_gen_q"].eq(QUARTILE_TOP)
    away_flagged = away["press_allow_q"].eq(QUARTILE_TOP) & home["press_gen_q"].eq(QUARTILE_TOP)
    table["home_offense_flagged"] = home_flagged.reindex(table.index).fillna(False).astype(bool)
    table["away_offense_flagged"] = away_flagged.reindex(table.index).fillna(False).astype(bool)

    only_home = table["home_offense_flagged"] & ~table["away_offense_flagged"]
    only_away = table["away_offense_flagged"] & ~table["home_offense_flagged"]
    table["back_side"] = np.where(only_home, "AWAY", np.where(only_away, "HOME", ""))

    return table.reset_index()[list(FLAG_TABLE_COLUMNS)]


def flag_summary(table: pd.DataFrame) -> dict[str, Any]:

    return {
        "games": len(table),
        "backs_home": int((table["back_side"] == "HOME").sum()),
        "backs_away": int((table["back_side"] == "AWAY").sum()),
        "no_lean": int((table["back_side"] == "").sum()),
        "both_sides_flagged": int(
            (table["home_offense_flagged"] & table["away_offense_flagged"]).sum()
        ),
    }


__all__ = [
    "FLAG_TABLE_COLUMNS",
    "MIN_QUANTILE_POOL",
    "MIN_WINDOW_OBS",
    "WINDOW_GAMES",
    "build_flag_table",
    "build_game_pressure_traits",
    "expanding_quartile_flags",
    "flag_summary",
]
