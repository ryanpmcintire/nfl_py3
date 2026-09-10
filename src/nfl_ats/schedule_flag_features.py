from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import HISTORICAL_CAPTURE_KIND, build_pairing_table
from nfl_ats.constants import (
    ATS_STREAK_REGRESS_ON_PRODUCTION_FEATURE_COLUMNS,
    DIVISION_DOG_ON_PRODUCTION_FEATURE_COLUMNS,
    DOME_SHOOTOUT_FAVORITE_ON_PRODUCTION_FEATURE_COLUMNS,
    HOME_THURSDAY_ON_PRODUCTION_FEATURE_COLUMNS,
    LOW_TOTAL_DIV_HOME_DOG_ON_PRODUCTION_FEATURE_COLUMNS,
    MNF_ROAD_SHORT_WEEK_ON_PRODUCTION_FEATURE_COLUMNS,
    NEW_STADIUM_HOME_ON_PRODUCTION_FEATURE_COLUMNS,
    POST_OT_FATIGUE_ON_PRODUCTION_FEATURE_COLUMNS,
    ROAD_FAV_BIG_FADE_ON_PRODUCTION_FEATURE_COLUMNS,
    SEPT_HEAT_HOME_ON_PRODUCTION_FEATURE_COLUMNS,
    WEEK1_DOG_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.features import add_ats_outcomes
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot

POST_OT_FATIGUE_COLUMN = POST_OT_FATIGUE_ON_PRODUCTION_FEATURE_COLUMNS[0]
MNF_ROAD_SHORT_WEEK_COLUMN = MNF_ROAD_SHORT_WEEK_ON_PRODUCTION_FEATURE_COLUMNS[0]
HOME_THURSDAY_COLUMN = HOME_THURSDAY_ON_PRODUCTION_FEATURE_COLUMNS[0]
NEW_STADIUM_COLUMN = NEW_STADIUM_HOME_ON_PRODUCTION_FEATURE_COLUMNS[0]
DOME_SHOOTOUT_COLUMN = DOME_SHOOTOUT_FAVORITE_ON_PRODUCTION_FEATURE_COLUMNS[0]
LOW_TOTAL_DIV_DOG_COLUMN = LOW_TOTAL_DIV_HOME_DOG_ON_PRODUCTION_FEATURE_COLUMNS[0]
SEPT_HEAT_COLUMN = SEPT_HEAT_HOME_ON_PRODUCTION_FEATURE_COLUMNS[0]
ROAD_FAV_BIG_FADE_COLUMN = ROAD_FAV_BIG_FADE_ON_PRODUCTION_FEATURE_COLUMNS[0]
DIVISION_DOG_COLUMN = DIVISION_DOG_ON_PRODUCTION_FEATURE_COLUMNS[0]
WEEK1_DOG_COLUMN = WEEK1_DOG_ON_PRODUCTION_FEATURE_COLUMNS[0]
ATS_STREAK_REGRESS_COLUMN = ATS_STREAK_REGRESS_ON_PRODUCTION_FEATURE_COLUMNS[0]

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market/raw"

_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "gameday",
    "weekday",
    "home_team",
    "away_team",
    "overtime",
}

MNF_ROAD_SHORT_WEEK_REST_DAYS = 6


def default_schedule(repo_root: Path | None = None) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    return pd.read_parquet(latest_schedules_snapshot(root))


def _require_schedule_columns(schedule: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


def _team_long_table(schedule: pd.DataFrame) -> pd.DataFrame:

    _require_schedule_columns(schedule)

    frame = schedule.loc[:, sorted(_REQUIRED_SCHEDULE_COLUMNS)].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["gameday_dt"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame["overtime"] = pd.to_numeric(frame["overtime"], errors="coerce")

    sides = []
    for is_home, team_col, opponent_col in (
        (True, "home_team", "away_team"),
        (False, "away_team", "home_team"),
    ):
        side = frame.loc[
            :, ["game_id", "season", "gameday_dt", "weekday", "overtime", team_col, opponent_col]
        ].rename(columns={team_col: "team", opponent_col: "opponent"})
        side["is_home"] = is_home
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.sort_values(["team", "season", "gameday_dt", "game_id"]).reset_index(
        drop=True
    )

    grouped = long_df.groupby(["team", "season"], sort=False)
    long_df["prev_gameday_dt"] = grouped["gameday_dt"].shift(1)
    long_df["prev_weekday"] = grouped["weekday"].shift(1)
    long_df["prev_overtime"] = grouped["overtime"].shift(1)
    long_df["prev_is_home"] = grouped["is_home"].shift(1)
    return long_df


def _pivot_home_away(long_df: pd.DataFrame, value_column: str) -> pd.DataFrame:

    home = long_df.loc[long_df["is_home"], ["game_id", value_column]].rename(
        columns={value_column: f"home_{value_column}"}
    )
    away = long_df.loc[~long_df["is_home"], ["game_id", value_column]].rename(
        columns={value_column: f"away_{value_column}"}
    )
    return home.merge(away, on="game_id", how="inner", validate="one_to_one")


def derive_post_ot_fatigue_features(schedule: pd.DataFrame) -> pd.DataFrame:

    long_df = _team_long_table(schedule)
    qualifies = long_df["prev_overtime"].eq(1.0)
    long_df = long_df.assign(post_ot_qualifies=qualifies)

    pivoted = _pivot_home_away(long_df, "post_ot_qualifies")
    away_only = pivoted["away_post_ot_qualifies"] & ~pivoted["home_post_ot_qualifies"]
    home_only = pivoted["home_post_ot_qualifies"] & ~pivoted["away_post_ot_qualifies"]
    flag = np.where(away_only, 1.0, np.where(home_only, -1.0, 0.0))
    return pd.DataFrame({"game_id": pivoted["game_id"], POST_OT_FATIGUE_COLUMN: flag})


def attach_post_ot_fatigue_features(
    features: pd.DataFrame, *, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:

    return _attach(features, schedule, derive_post_ot_fatigue_features, (POST_OT_FATIGUE_COLUMN,))


def derive_mnf_road_short_week_features(schedule: pd.DataFrame) -> pd.DataFrame:

    long_df = _team_long_table(schedule)
    gap_days = (long_df["gameday_dt"] - long_df["prev_gameday_dt"]).dt.days
    qualifies = (
        long_df["prev_weekday"].eq("Monday")
        & long_df["prev_is_home"].eq(False)
        & long_df["weekday"].eq("Sunday")
        & gap_days.eq(MNF_ROAD_SHORT_WEEK_REST_DAYS)
    )
    long_df = long_df.assign(mnf_road_qualifies=qualifies)

    pivoted = _pivot_home_away(long_df, "mnf_road_qualifies")
    away_only = pivoted["away_mnf_road_qualifies"] & ~pivoted["home_mnf_road_qualifies"]
    home_only = pivoted["home_mnf_road_qualifies"] & ~pivoted["away_mnf_road_qualifies"]
    flag = np.where(away_only, 1.0, np.where(home_only, -1.0, 0.0))
    return pd.DataFrame({"game_id": pivoted["game_id"], MNF_ROAD_SHORT_WEEK_COLUMN: flag})


def attach_mnf_road_short_week_features(
    features: pd.DataFrame, *, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:

    return _attach(
        features, schedule, derive_mnf_road_short_week_features, (MNF_ROAD_SHORT_WEEK_COLUMN,)
    )


def derive_home_thursday_features(schedule: pd.DataFrame) -> pd.DataFrame:

    _require_schedule_columns(schedule)
    weekday = schedule["weekday"]
    flag = np.where(weekday.isna(), np.nan, weekday.eq("Thursday").astype(float))
    return pd.DataFrame({"game_id": schedule["game_id"].astype(str), HOME_THURSDAY_COLUMN: flag})


def attach_home_thursday_features(
    features: pd.DataFrame, *, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:

    return _attach(features, schedule, derive_home_thursday_features, (HOME_THURSDAY_COLUMN,))


NEW_STADIUM_HONEYMOON_SEASONS: dict[str, tuple[int, int]] = {
    "NYC01": (2010, 2011),
    "SFO01": (2014, 2015),
    "MIN01": (2016, 2017),
    "ATL97": (2017, 2018),
    "LAX01": (2020, 2021),
    "VEG00": (2020, 2021),
}

_NEW_STADIUM_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "season", "stadium_id"}


def derive_new_stadium_home_features(schedule: pd.DataFrame) -> pd.DataFrame:

    missing = sorted(_NEW_STADIUM_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")

    season = pd.to_numeric(schedule["season"], errors="raise").astype(int)
    qualifies = pd.Series(False, index=schedule.index)
    for stadium_id, (season_one, season_two) in NEW_STADIUM_HONEYMOON_SEASONS.items():
        qualifies |= schedule["stadium_id"].eq(stadium_id) & season.isin((season_one, season_two))
    flag = qualifies.astype(float)
    return pd.DataFrame({"game_id": schedule["game_id"].astype(str), NEW_STADIUM_COLUMN: flag})


def attach_new_stadium_home_features(
    features: pd.DataFrame, *, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:

    return _attach(features, schedule, derive_new_stadium_home_features, (NEW_STADIUM_COLUMN,))


def default_opener_lines(
    schedule: pd.DataFrame, *, market_root: Path | None = None
) -> pd.DataFrame:

    if "game_id" not in schedule.columns:
        raise DataContractError("schedule is missing the game_id column")
    root = market_root or DEFAULT_MARKET_ROOT
    pairing = build_pairing_table(
        root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open",),
        schedule=schedule[["game_id", "season", "week"]],
    )
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")]
    lines = tue_open[["game_id", "home_spread", "total_line"]].rename(
        columns={"home_spread": "tue_open_home_spread", "total_line": "tue_open_total_line"}
    )
    return lines.drop_duplicates("game_id").reset_index(drop=True)


DOME_SHOOTOUT_ROOFS = frozenset({"dome", "closed"})
DOME_SHOOTOUT_TOTAL_MIN = 49.0
DOME_SHOOTOUT_SPREAD_MAX_ABS = 3.0

_DOME_SHOOTOUT_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "roof"}


def oracle_derive_dome_shootout_favorite_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:

    missing = sorted(_DOME_SHOOTOUT_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    merged = schedule[["game_id", "roof"]].merge(
        opener_lines[["game_id", "tue_open_home_spread", "tue_open_total_line"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    dome_or_closed = merged["roof"].isin(DOME_SHOOTOUT_ROOFS)
    high_total = merged["tue_open_total_line"].notna() & merged["tue_open_total_line"].ge(
        DOME_SHOOTOUT_TOTAL_MIN
    )
    close_spread = merged["tue_open_home_spread"].notna() & merged["tue_open_home_spread"].abs().le(
        DOME_SHOOTOUT_SPREAD_MAX_ABS
    )
    archetype = dome_or_closed & high_total & close_spread
    home_favorite = archetype & merged["tue_open_home_spread"].gt(0.0)
    away_favorite = archetype & merged["tue_open_home_spread"].lt(0.0)
    flag = np.where(home_favorite, 1.0, np.where(away_favorite, -1.0, 0.0))
    return pd.DataFrame(
        {"game_id": merged["game_id"].astype(str), "oracle_" + DOME_SHOOTOUT_COLUMN: flag}
    )


def attach_dome_shootout_favorite_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
    announcements: pd.DataFrame | None = None,
) -> pd.DataFrame:

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_dome_shootout_favorite_features(sched, lines, announcements=announcements)

    return _attach(features, schedule, _derive, (DOME_SHOOTOUT_COLUMN,))


LOW_TOTAL_DIV_DOG_TOTAL_MAX = 42.0

_LOW_TOTAL_DIV_DOG_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "div_game"}


def derive_low_total_div_home_dog_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:

    missing = sorted(_LOW_TOTAL_DIV_DOG_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    merged = schedule[["game_id", "div_game"]].merge(
        opener_lines[["game_id", "tue_open_home_spread", "tue_open_total_line"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    divisional = pd.to_numeric(merged["div_game"], errors="coerce").eq(1.0)
    low_total = merged["tue_open_total_line"].notna() & merged["tue_open_total_line"].le(
        LOW_TOTAL_DIV_DOG_TOTAL_MAX
    )
    home_dog = merged["tue_open_home_spread"].notna() & merged["tue_open_home_spread"].lt(0.0)
    flag = (divisional & low_total & home_dog).astype(float)
    return pd.DataFrame({"game_id": merged["game_id"].astype(str), LOW_TOTAL_DIV_DOG_COLUMN: flag})


def attach_low_total_div_home_dog_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_low_total_div_home_dog_features(sched, lines)

    return _attach(features, schedule, _derive, (LOW_TOTAL_DIV_DOG_COLUMN,))


SEPT_HEAT_UNCONDITIONAL_HOME_TEAMS = frozenset({"MIA", "TB", "JAX"})
SEPT_HEAT_ROOF_CONDITIONAL_HOME_TEAMS = frozenset({"HOU", "NO", "ATL"})
SEPT_HEAT_OPEN_AIR_ROOFS = frozenset({"outdoors", "open"})
SEPT_HEAT_COLD_VISITOR_TEAMS = frozenset(
    {
        "BUF",
        "NE",
        "NYJ",
        "NYG",
        "GB",
        "CHI",
        "MIN",
        "DET",
        "CLE",
        "PIT",
        "CIN",
        "DEN",
        "SEA",
        "KC",
        "PHI",
        "BAL",
        "WAS",
    }
)
SEPT_HEAT_MAX_WEEK = 3
SEPT_HEAT_HOME_TEAM_ET_OFFSET_HOURS: dict[str, int] = {
    "MIA": 0,
    "TB": 0,
    "JAX": 0,
    "ATL": 0,
    "HOU": 1,
    "NO": 1,
}
_SEPT_HEAT_LOCAL_ONE_PM_START_MIN = 13 * 60
_SEPT_HEAT_LOCAL_ONE_PM_END_MIN = 14 * 60

_SEPT_HEAT_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_type",
    "home_team",
    "away_team",
    "roof",
    "gametime",
}


def oracle_derive_sept_heat_home_features(schedule: pd.DataFrame) -> pd.DataFrame:

    missing = sorted(_SEPT_HEAT_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")

    reg_early_week = schedule["game_type"].eq("REG") & pd.to_numeric(
        schedule["week"], errors="raise"
    ).le(SEPT_HEAT_MAX_WEEK)
    home_team = schedule["home_team"]
    unconditional_home = home_team.isin(SEPT_HEAT_UNCONDITIONAL_HOME_TEAMS)
    conditional_home = home_team.isin(SEPT_HEAT_ROOF_CONDITIONAL_HOME_TEAMS) & schedule[
        "roof"
    ].isin(SEPT_HEAT_OPEN_AIR_ROOFS)
    heat_home = unconditional_home | conditional_home
    cold_visitor = schedule["away_team"].isin(SEPT_HEAT_COLD_VISITOR_TEAMS)

    kickoff_et = pd.to_datetime(schedule["gametime"], format="%H:%M", errors="coerce")
    et_minutes = kickoff_et.dt.hour * 60.0 + kickoff_et.dt.minute
    offset_hours = home_team.map(SEPT_HEAT_HOME_TEAM_ET_OFFSET_HOURS).astype(float)
    local_minutes = et_minutes - offset_hours * 60.0
    one_pm_local = local_minutes.between(
        _SEPT_HEAT_LOCAL_ONE_PM_START_MIN, _SEPT_HEAT_LOCAL_ONE_PM_END_MIN, inclusive="left"
    )

    qualifies = reg_early_week & heat_home & cold_visitor & one_pm_local.fillna(False)
    flag = qualifies.astype(float)
    return pd.DataFrame(
        {"game_id": schedule["game_id"].astype(str), "oracle_" + SEPT_HEAT_COLUMN: flag}
    )


def attach_sept_heat_home_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    announcements: pd.DataFrame | None = None,
) -> pd.DataFrame:

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        return derive_sept_heat_home_features(sched, announcements=announcements)

    return _attach(features, schedule, _derive, (SEPT_HEAT_COLUMN,))


ROAD_FAV_BIG_FADE_SPREAD_MIN_ABS = 7.0

_ROAD_FAV_BIG_FADE_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "game_type"}


def derive_road_fav_big_fade_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:

    missing = sorted(_ROAD_FAV_BIG_FADE_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    merged = schedule[["game_id", "game_type"]].merge(
        opener_lines[["game_id", "tue_open_home_spread"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    reg = merged["game_type"].eq("REG")
    spread = merged["tue_open_home_spread"]
    away_big_favorite = reg & spread.notna() & spread.le(-ROAD_FAV_BIG_FADE_SPREAD_MIN_ABS)
    home_big_favorite = reg & spread.notna() & spread.ge(ROAD_FAV_BIG_FADE_SPREAD_MIN_ABS)
    flag = np.where(away_big_favorite, 1.0, np.where(home_big_favorite, -1.0, 0.0))
    return pd.DataFrame({"game_id": merged["game_id"].astype(str), ROAD_FAV_BIG_FADE_COLUMN: flag})


def attach_road_fav_big_fade_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_road_fav_big_fade_features(sched, lines)

    return _attach(features, schedule, _derive, (ROAD_FAV_BIG_FADE_COLUMN,))


def _dog_flag_from_opener_spread(eligible: pd.Series, spread: pd.Series) -> np.ndarray:
    home_dog = eligible & spread.notna() & spread.lt(0.0)
    away_dog = eligible & spread.notna() & spread.gt(0.0)
    return np.where(home_dog, 1.0, np.where(away_dog, -1.0, 0.0))


_DIVISION_DOG_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "game_type", "div_game"}


def derive_division_dog_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:

    missing = sorted(_DIVISION_DOG_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    merged = schedule[["game_id", "game_type", "div_game"]].merge(
        opener_lines[["game_id", "tue_open_home_spread"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    eligible = merged["game_type"].eq("REG") & pd.to_numeric(
        merged["div_game"], errors="coerce"
    ).eq(1.0)
    flag = _dog_flag_from_opener_spread(eligible, merged["tue_open_home_spread"])
    return pd.DataFrame({"game_id": merged["game_id"].astype(str), DIVISION_DOG_COLUMN: flag})


def attach_division_dog_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_division_dog_features(sched, lines)

    return _attach(features, schedule, _derive, (DIVISION_DOG_COLUMN,))


_WEEK1_DOG_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "game_type", "week"}


def derive_week1_dog_features(schedule: pd.DataFrame, opener_lines: pd.DataFrame) -> pd.DataFrame:

    missing = sorted(_WEEK1_DOG_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    merged = schedule[["game_id", "game_type", "week"]].merge(
        opener_lines[["game_id", "tue_open_home_spread"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    eligible = merged["game_type"].eq("REG") & pd.to_numeric(merged["week"], errors="coerce").eq(
        1.0
    )
    flag = _dog_flag_from_opener_spread(eligible, merged["tue_open_home_spread"])
    return pd.DataFrame({"game_id": merged["game_id"].astype(str), WEEK1_DOG_COLUMN: flag})


def attach_week1_dog_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_week1_dog_features(sched, lines)

    return _attach(features, schedule, _derive, (WEEK1_DOG_COLUMN,))


ATS_STREAK_REGRESS_MIN_STREAK = 3.0
_ATS_STREAK_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "gameday",
    "game_type",
    "home_team",
    "away_team",
    "result",
    "spread_line",
}


def _team_ats_streak_entering_each_game(schedule: pd.DataFrame) -> pd.DataFrame:

    missing = sorted(_ATS_STREAK_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")

    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg = add_ats_outcomes(reg)
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    sides = []
    for side_column, is_home in (("home_team", True), ("away_team", False)):
        covered = reg["home_cover"] if is_home else 1.0 - reg["home_cover"]
        covered = covered.where(reg["home_cover"].notna())
        sides.append(
            pd.DataFrame(
                {
                    "game_id": reg["game_id"].astype(str),
                    "team": reg[side_column].astype(str),
                    "season": reg["season"],
                    "gameday_dt": reg["gameday_dt"],
                    "covered": covered,
                }
            )
        )
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.sort_values(["team", "season", "gameday_dt", "game_id"]).reset_index(
        drop=True
    )

    covered_array = long_df["covered"].to_numpy(dtype=float)
    streaks_entering = np.zeros(len(long_df), dtype=float)
    for _, group in long_df.groupby(["team", "season"], sort=False):
        current = 0.0
        for position in group.index:
            streaks_entering[position] = current
            covered_value = covered_array[position]
            if np.isnan(covered_value):
                continue
            current = 0.0 if covered_value >= 1.0 else current + 1.0
    long_df["streak_entering"] = streaks_entering
    return long_df[["game_id", "team", "streak_entering"]]


def derive_ats_streak_regress_features(schedule: pd.DataFrame) -> pd.DataFrame:

    per_team = _team_ats_streak_entering_each_game(schedule)
    home_side = per_team.rename(columns={"team": "home_team", "streak_entering": "home_streak"})
    away_side = per_team.rename(columns={"team": "away_team", "streak_entering": "away_streak"})

    reg_ids = schedule.loc[schedule["game_type"].eq("REG"), ["game_id", "home_team", "away_team"]]
    reg_ids = reg_ids.astype({"game_id": str, "home_team": str, "away_team": str})
    reg_ids = reg_ids.merge(
        home_side[["game_id", "home_team", "home_streak"]], on=["game_id", "home_team"], how="left"
    )
    reg_ids = reg_ids.merge(
        away_side[["game_id", "away_team", "away_streak"]], on=["game_id", "away_team"], how="left"
    )

    home_qualifies = reg_ids["home_streak"] >= ATS_STREAK_REGRESS_MIN_STREAK
    away_qualifies = reg_ids["away_streak"] >= ATS_STREAK_REGRESS_MIN_STREAK
    reg_flag = np.where(
        home_qualifies & ~away_qualifies,
        1.0,
        np.where(away_qualifies & ~home_qualifies, -1.0, 0.0),
    )
    reg_out = pd.DataFrame({"game_id": reg_ids["game_id"], ATS_STREAK_REGRESS_COLUMN: reg_flag})

    all_ids = schedule[["game_id"]].astype({"game_id": str})
    result = all_ids.merge(reg_out, on="game_id", how="left")
    result[ATS_STREAK_REGRESS_COLUMN] = result[ATS_STREAK_REGRESS_COLUMN].fillna(0.0)
    return result


def attach_ats_streak_regress_features(
    features: pd.DataFrame, *, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:

    return _attach(
        features, schedule, derive_ats_streak_regress_features, (ATS_STREAK_REGRESS_COLUMN,)
    )


def _attach(
    features: pd.DataFrame,
    schedule: pd.DataFrame | None,
    derive: Callable[[pd.DataFrame], pd.DataFrame],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(columns).intersection(features.columns))
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    if schedule is None:
        schedule = default_schedule()
    derived = derive(schedule)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_schedule_flag"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_schedule_flag") if c in merged.columns]
    )
    merged.index = features.index
    return merged


__all__ = [
    "ATS_STREAK_REGRESS_COLUMN",
    "ATS_STREAK_REGRESS_MIN_STREAK",
    "DEFAULT_MARKET_ROOT",
    "DIVISION_DOG_COLUMN",
    "DOME_SHOOTOUT_COLUMN",
    "DOME_SHOOTOUT_ROOFS",
    "DOME_SHOOTOUT_SPREAD_MAX_ABS",
    "DOME_SHOOTOUT_TOTAL_MIN",
    "HOME_THURSDAY_COLUMN",
    "LOW_TOTAL_DIV_DOG_COLUMN",
    "LOW_TOTAL_DIV_DOG_TOTAL_MAX",
    "MNF_ROAD_SHORT_WEEK_COLUMN",
    "MNF_ROAD_SHORT_WEEK_REST_DAYS",
    "NEW_STADIUM_COLUMN",
    "NEW_STADIUM_HONEYMOON_SEASONS",
    "POST_OT_FATIGUE_COLUMN",
    "ROAD_FAV_BIG_FADE_COLUMN",
    "ROAD_FAV_BIG_FADE_SPREAD_MIN_ABS",
    "SEPT_HEAT_COLD_VISITOR_TEAMS",
    "SEPT_HEAT_COLUMN",
    "SEPT_HEAT_HOME_TEAM_ET_OFFSET_HOURS",
    "SEPT_HEAT_MAX_WEEK",
    "SEPT_HEAT_OPEN_AIR_ROOFS",
    "SEPT_HEAT_ROOF_CONDITIONAL_HOME_TEAMS",
    "SEPT_HEAT_UNCONDITIONAL_HOME_TEAMS",
    "WEEK1_DOG_COLUMN",
    "attach_ats_streak_regress_features",
    "attach_division_dog_features",
    "attach_dome_shootout_favorite_features",
    "attach_home_thursday_features",
    "attach_low_total_div_home_dog_features",
    "attach_mnf_road_short_week_features",
    "attach_new_stadium_home_features",
    "attach_post_ot_fatigue_features",
    "attach_road_fav_big_fade_features",
    "attach_sept_heat_home_features",
    "attach_week1_dog_features",
    "decision_time_roof_schedule",
    "default_opener_lines",
    "default_schedule",
    "derive_ats_streak_regress_features",
    "derive_division_dog_features",
    "derive_dome_shootout_favorite_features",
    "derive_home_thursday_features",
    "derive_low_total_div_home_dog_features",
    "derive_mnf_road_short_week_features",
    "derive_new_stadium_home_features",
    "derive_post_ot_fatigue_features",
    "derive_road_fav_big_fade_features",
    "derive_sept_heat_home_features",
    "derive_week1_dog_features",
    "oracle_derive_dome_shootout_favorite_features",
    "oracle_derive_sept_heat_home_features",
]


VENUE_INDOOR_DEFAULTS = {
    **dict.fromkeys(
        (
            "University of Phoenix Stadium",
            "State Farm Stadium",
            "Reliant Stadium",
            "NRG Stadium",
            "Lucas Oil Stadium",
            "Cowboys Stadium",
            "AT&T Stadium",
            "Mercedes-Benz Stadium",
            "Rogers Centre",
        ),
        "closed",
    ),
    **dict.fromkeys(
        (
            "Georgia Dome",
            "Louisiana Superdome",
            "Mercedes-Benz Superdome",
            "Caesars Superdome",
            "Ford Field",
            "Hubert H. Humphrey Metrodome",
            "Mall of America Field",
            "Edward Jones Dome",
            "U.S. Bank Stadium",
            "SoFi Stadium",
            "Allegiant Stadium",
        ),
        "dome",
    ),
}


def decision_time_roof_schedule(
    schedule: pd.DataFrame, announcements: pd.DataFrame | None = None
) -> pd.DataFrame:
    import json

    from nfl_ats.nfl_week import pool_decision_cutoff
    from nfl_ats.players import _schedule_kickoff_utc

    result = schedule.copy()
    result["oracle_roof"] = result.get("roof", pd.Series(pd.NA, index=result.index))
    if "venue_default_roof" in result:
        result["roof"] = result["venue_default_roof"]
    elif "stadium" in result:
        venues = json.loads((REPO_ROOT / "registry/stadium_coordinates.json").read_text())
        defaults = {
            name: VENUE_INDOOR_DEFAULTS.get(name, "outdoors")
            for name in venues
            if not name.startswith("_")
        }
        result["roof"] = result["stadium"].map(defaults)
    else:
        raise DataContractError("decision-time roof requires stadium or venue_default_roof")
    if announcements is not None and not announcements.empty:
        required = {"game_id", "roof", "observed_at_utc"}
        if not required.issubset(announcements):
            raise DataContractError("roof announcements require game_id, roof, observed_at_utc")
        rows = announcements.copy()
        rows["observed_at_utc"] = pd.to_datetime(rows["observed_at_utc"], utc=True, errors="coerce")
        kickoff = (
            pd.to_datetime(result["kickoff"], utc=True)
            if "kickoff" in result
            else _schedule_kickoff_utc(result)
        )
        cutoffs = kickoff.map(
            lambda value: pool_decision_cutoff(value) if pd.notna(value) else pd.NaT
        )
        for position, (index, game) in enumerate(result.iterrows()):
            visible = rows.loc[
                rows["game_id"].eq(game["game_id"])
                & rows["observed_at_utc"].lt(cutoffs.iloc[position])
            ]
            if not visible.empty:
                latest = visible.sort_values("observed_at_utc").iloc[-1]
                result.at[index, "roof"] = latest["roof"]
    return result


def derive_dome_shootout_favorite_features(
    schedule: pd.DataFrame,
    opener_lines: pd.DataFrame,
    *,
    announcements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    projected = decision_time_roof_schedule(schedule, announcements)
    result = oracle_derive_dome_shootout_favorite_features(projected, opener_lines).rename(
        columns={"oracle_" + DOME_SHOOTOUT_COLUMN: DOME_SHOOTOUT_COLUMN}
    )
    result.loc[projected["roof"].isna().to_numpy(), DOME_SHOOTOUT_COLUMN] = np.nan
    return result


def derive_sept_heat_home_features(
    schedule: pd.DataFrame,
    *,
    announcements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    projected = decision_time_roof_schedule(schedule, announcements)
    result = oracle_derive_sept_heat_home_features(projected).rename(
        columns={"oracle_" + SEPT_HEAT_COLUMN: SEPT_HEAT_COLUMN}
    )
    result.loc[projected["roof"].isna().to_numpy(), SEPT_HEAT_COLUMN] = np.nan
    return result
