"""Three pure-schedule pregame flags, each stacked on PRODUCTION.

``docs/schedule_flag_battery.md`` predeclares LEAD-21 (post-overtime
fatigue), LEAD-22 (Monday-night-road short week) and LEAD-40 (home-Thursday
rest compound) before any of them was scored. Every flag here is a
deterministic function of ``data/raw/*/schedules.parquet`` alone -- no PBP,
no injuries, no market data -- and every input a flag reads (``gameday``,
``weekday``, ``home_team``/``away_team``, ``overtime``) is a pregame-known
schedule fact for the game that PRODUCED it. Section "leakage" in the
predeclaration doc states the binding claim this module exists to satisfy:
shuffling or altering a game's own outcome (its score, its margin, its own
``result``) never changes any flag, because no flag reads any column that
depends on a game's own outcome other than ``overtime`` (whether OT was
PLAYED, not who won it) -- and even that column is read only from a game's
*own preceding* game, never from the game the flag is attached to.

Mirrors ``nfl_ats.team_style_pace_production_feature`` /
``nfl_ats.redzone_reversion_production_feature``'s additive-merge discipline:
every pre-existing column comes back bit-identical, only the one new column
is added.

**Within-season lookback only.** Each flag's "previous game" is the
immediately preceding row for that team, sorted by (season, gameday), and the
shift never crosses a season boundary -- a team's first game of a season has
no in-season predecessor, so LEAD-21/LEAD-22 evaluate to their "does not
qualify" state (**0.0**, not NaN) for it, deliberately: physical fatigue from
an offseason-old overtime game, or a short week off an offseason-old Monday
road trip, is not a real mechanism either predeclaration claims, so "no
prior game this season" is a genuine fact (definitely not fatigued/short-
week), not missing information. This differs from
``nfl_ats.team_style_pace_production_feature``'s NaN-on-missing convention,
where "no prior-season data" really is an unknown team-quality state; the
predeclaration doc states this distinction explicitly for each construct.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    HOME_THURSDAY_ON_PRODUCTION_FEATURE_COLUMNS,
    MNF_ROAD_SHORT_WEEK_ON_PRODUCTION_FEATURE_COLUMNS,
    POST_OT_FATIGUE_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot

#: The one new column each candidate profile adds. Frozen names.
POST_OT_FATIGUE_COLUMN = POST_OT_FATIGUE_ON_PRODUCTION_FEATURE_COLUMNS[0]
MNF_ROAD_SHORT_WEEK_COLUMN = MNF_ROAD_SHORT_WEEK_ON_PRODUCTION_FEATURE_COLUMNS[0]
HOME_THURSDAY_COLUMN = HOME_THURSDAY_ON_PRODUCTION_FEATURE_COLUMNS[0]

REPO_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "gameday",
    "weekday",
    "home_team",
    "away_team",
    "overtime",
}

#: Six days after a Monday game is a Sunday -- the "following Sunday" LEAD-22
#: predeclares. Checked directly against the calendar gap, not inferred from
#: it, so a data inconsistency (a mislabeled weekday) cannot silently pass.
MNF_ROAD_SHORT_WEEK_REST_DAYS = 6


def default_schedule(repo_root: Path | None = None) -> pd.DataFrame:
    """Load the newest ``data/raw/*/schedules.parquet`` snapshot.

    Reuses ``nfl_ats.weak_stack_v3_features.latest_schedules_snapshot`` --
    the same "newest snapshot, sorted lexicographically" convention every
    schedule-only battery in this repo already uses -- rather than
    re-implementing snapshot discovery.
    """

    root = repo_root or REPO_ROOT
    return pd.read_parquet(latest_schedules_snapshot(root))


def _require_schedule_columns(schedule: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


def _team_long_table(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side): team, this game's own weekday/overtime, plus
    that team's PRECEDING in-season game's weekday/overtime/site/gameday.

    Sorted by (team, season, gameday) before the ``shift(1)``, so byes are
    skipped automatically (they are not rows in the schedule) and the shift
    never reaches across a season boundary (grouped by ``(team, season)``).
    """

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
    """(game_id, home_<value>, away_<value>) from the per-side long table."""

    home = long_df.loc[long_df["is_home"], ["game_id", value_column]].rename(
        columns={value_column: f"home_{value_column}"}
    )
    away = long_df.loc[~long_df["is_home"], ["game_id", value_column]].rename(
        columns={value_column: f"away_{value_column}"}
    )
    return home.merge(away, on="game_id", how="inner", validate="one_to_one")


# ---------------------------------------------------------------------------
# LEAD-21: post-overtime fatigue
# ---------------------------------------------------------------------------


def derive_post_ot_fatigue_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return ``(game_id, post_ot_fatigue_flag)`` for every game in ``schedule``.

    ``+1`` if the AWAY team's immediately preceding in-season game went to
    overtime and the HOME team's did not; ``-1`` if the reverse; ``0`` if
    both did, neither did, or either side has no in-season preceding game.
    Sign chosen so a positive fitted coefficient means "fading the post-OT
    side helped" (docs/schedule_flag_battery.md section 1).
    """

    long_df = _team_long_table(schedule)
    # A missing overtime value NEVER occurs for an actual completed
    # preceding game in this dataset (verified: overtime is NaN exactly for
    # not-yet-played games, never for a resolved one) -- so treating a
    # missing/absent preceding-game value as "not post-OT" is a safe,
    # measured simplification, not a silent assumption.
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
    """Additively join ``post_ot_fatigue_flag`` onto ``features`` by ``game_id``."""

    return _attach(features, schedule, derive_post_ot_fatigue_features, (POST_OT_FATIGUE_COLUMN,))


# ---------------------------------------------------------------------------
# LEAD-22: Monday-night-road short week
# ---------------------------------------------------------------------------


def derive_mnf_road_short_week_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return ``(game_id, mnf_road_short_week_flag)`` for every game in ``schedule``.

    A side "qualifies" when its immediately preceding in-season game was on
    a Monday, it was the away (road) team in that game, AND this game is
    exactly ``MNF_ROAD_SHORT_WEEK_REST_DAYS`` (6) calendar days later on a
    Sunday. ``+1`` if the AWAY team qualifies and the HOME team does not;
    ``-1`` if the reverse; ``0`` if both qualify, neither does, or a side has
    no in-season preceding game. Sign chosen so a positive fitted
    coefficient means "fading the short-week road side helped"
    (docs/schedule_flag_battery.md section 2), the same convention LEAD-21
    uses.
    """

    long_df = _team_long_table(schedule)
    gap_days = (long_df["gameday_dt"] - long_df["prev_gameday_dt"]).dt.days
    qualifies = (
        long_df["prev_weekday"].eq("Monday")
        & long_df["prev_is_home"].eq(False)  # NaN-safe: NaN == False evaluates to False
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
    """Additively join ``mnf_road_short_week_flag`` onto ``features`` by ``game_id``."""

    return _attach(
        features, schedule, derive_mnf_road_short_week_features, (MNF_ROAD_SHORT_WEEK_COLUMN,)
    )


# ---------------------------------------------------------------------------
# LEAD-40: home-Thursday rest compound
# ---------------------------------------------------------------------------


def derive_home_thursday_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return ``(game_id, home_thursday_flag)`` for every game in ``schedule``.

    ``1.0`` when this game's own ``weekday`` is Thursday, ``0.0`` otherwise
    -- a plain calendar fact about the CURRENT game, unlike LEAD-21/LEAD-22,
    which needs no in-season lookback. Unsigned (not home-minus-away) because
    the construct is not a comparison between the two teams' conditions: on a
    Thursday game the home side never travels while the away side does, so
    "Thursday" already IS the home-favouring condition (matching the parent
    ``travel_rest_thursday_pure`` cell's own plain boolean shape). A missing
    ``weekday`` value returns NaN, never a silent 0.
    """

    _require_schedule_columns(schedule)
    weekday = schedule["weekday"]
    flag = np.where(weekday.isna(), np.nan, weekday.eq("Thursday").astype(float))
    return pd.DataFrame({"game_id": schedule["game_id"].astype(str), HOME_THURSDAY_COLUMN: flag})


def attach_home_thursday_features(
    features: pd.DataFrame, *, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Additively join ``home_thursday_flag`` onto ``features`` by ``game_id``."""

    return _attach(features, schedule, derive_home_thursday_features, (HOME_THURSDAY_COLUMN,))


# ---------------------------------------------------------------------------
# Shared additive-merge helper
# ---------------------------------------------------------------------------


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
    "HOME_THURSDAY_COLUMN",
    "MNF_ROAD_SHORT_WEEK_COLUMN",
    "MNF_ROAD_SHORT_WEEK_REST_DAYS",
    "POST_OT_FATIGUE_COLUMN",
    "attach_home_thursday_features",
    "attach_mnf_road_short_week_features",
    "attach_post_ot_fatigue_features",
    "default_schedule",
    "derive_home_thursday_features",
    "derive_mnf_road_short_week_features",
    "derive_post_ot_fatigue_features",
]
