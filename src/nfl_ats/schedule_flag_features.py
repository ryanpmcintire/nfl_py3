"""Seven pure-schedule (plus, for two, the Tuesday-opener market consensus)
pregame flags, each stacked on PRODUCTION.

``docs/schedule_flag_battery.md`` predeclares LEAD-21 (post-overtime
fatigue), LEAD-22 (Monday-night-road short week) and LEAD-40 (home-Thursday
rest compound) -- Wave 1 -- before any of them was scored. Every flag here is
a deterministic function of ``data/raw/*/schedules.parquet`` alone -- no PBP,
no injuries, no market data -- and every input a flag reads (``gameday``,
``weekday``, ``home_team``/``away_team``, ``overtime``) is a pregame-known
schedule fact for the game that PRODUCED it. Section "leakage" in the
predeclaration doc states the binding claim this module exists to satisfy:
shuffling or altering a game's own outcome (its score, its margin, its own
``result``) never changes any flag, because no flag reads any column that
depends on a game's own outcome other than ``overtime`` (whether OT was
PLAYED, not who won it) -- and even that column is read only from a game's
*own preceding* game, never from the game the flag is attached to.

The doc's "Wave 2" section predeclares four more: LEAD-39 (new-stadium
honeymoon), LEAD-41 (dome-shootout favorite archetype), LEAD-42 (low-total
divisional home dog), and LEAD-35 (September heat-humidity home edge). The
first and last are pure schedule facts, same discipline as Wave 1;
LEAD-41/LEAD-42 additionally read the Tuesday-OPENER consensus spread/total
from :func:`default_opener_lines` (``nfl_ats.clv.build_pairing_table``'s
historical decision-labeled archive) -- never the nflverse schedule's own
(closing) ``spread_line``/``total_line`` -- which is still pregame-known
information (a Tuesday market quote), not an outcome of any game.

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

#: The one new column each candidate profile adds. Frozen names.
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

#: Wave 2 (LEAD-41/LEAD-42) needs the Tuesday-OPENER consensus spread/total,
#: never the nflverse schedule's own (closing) ``spread_line``/``total_line``.
#: Same store and decision label ``scripts/on_production_opener_confirmation.py``
#: already grades every on-production candidate against.
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
# Wave 2 (docs/schedule_flag_battery.md "Wave 2"): LEAD-39 new-stadium
# honeymoon, LEAD-41 dome-shootout favorite, LEAD-42 low-total divisional
# home dog, LEAD-35 September heat-humidity home edge.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LEAD-39: new-stadium honeymoon (seasons 1-2)
# ---------------------------------------------------------------------------

#: (stadium_id -> its first two REG-season-of-use calendar years), frozen
#: 2026-09-05 by measuring data/raw/20260824T115346Z/schedules.parquet:
#: every stadium_id whose first REG-season use is >= 2010, EXCLUDING (a)
#: neutral/international one-off sites -- LON00/LON01/LON02, MEX00, GER00,
#: FRA00, SAO00 (task-given) plus the same-class 2026 one-off sites
#: MAD01/MEL00/MUN01/PAR00/RIO00 (single-game international friendlies, not
#: a team's repeat home base) -- and (b) temporary construction-displacement
#: homes superseded by a team's own LATER permanent venue already in this
#: same table: MIN98 (TCF Bank Stadium; Vikings' 2010 storm-displacement
#: game and 2014-2015 home while U.S. Bank Stadium/MIN01 was built), LAX99
#: (LA Memorial Coliseum; Rams' 2016-2019 home while SoFi/LAX01 was built),
#: LAX97 (StubHub Center; Chargers' 2017-2019 home while SoFi/LAX01 was
#: built). What remains is exactly the six permanent-build stadium_ids
#: docs/schedule_flag_battery.md Wave 2 section 4 states, matching the fleet
#: task's own worked examples verbatim.
NEW_STADIUM_HONEYMOON_SEASONS: dict[str, tuple[int, int]] = {
    "NYC01": (2010, 2011),  # MetLife Stadium (NYG, NYJ)
    "SFO01": (2014, 2015),  # Levi's Stadium (SF)
    "MIN01": (2016, 2017),  # U.S. Bank Stadium (MIN)
    "ATL97": (2017, 2018),  # Mercedes-Benz Stadium (ATL)
    "LAX01": (2020, 2021),  # SoFi Stadium (LA, LAC)
    "VEG00": (2020, 2021),  # Allegiant Stadium (LV)
}

_NEW_STADIUM_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "season", "stadium_id"}


def derive_new_stadium_home_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return ``(game_id, new_stadium_home_flag)`` for every game in ``schedule``.

    ``1.0`` when the game's own ``stadium_id`` is one of the six frozen
    permanent-build venues in :data:`NEW_STADIUM_HONEYMOON_SEASONS` AND
    ``season`` is one of that venue's own first two REG seasons of use;
    ``0.0`` otherwise. No in-season lookback, no team-level state, and no
    outcome of any game (this game's or any other) is read -- a plain
    venue-assignment/calendar fact known long before kickoff. Unsigned,
    matching LEAD-40's shape: this is a single-side effect (BACK the home
    team), not a differential between the two teams' conditions.
    """

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
    """Additively join ``new_stadium_home_flag`` onto ``features`` by ``game_id``."""

    return _attach(features, schedule, derive_new_stadium_home_features, (NEW_STADIUM_COLUMN,))


# ---------------------------------------------------------------------------
# Opener (Tuesday-consensus) line loader -- LEAD-41/LEAD-42 need the OPENER
# total/spread, never the nflverse schedule's own (closing) total_line/
# spread_line. Reuses nfl_ats.clv.build_pairing_table's HISTORICAL
# decision-labeled archive, the SAME "opener store"
# scripts/on_production_opener_confirmation.py already grades every
# on-production candidate against (its "tue_open" decision label) -- not a
# new market pipeline, no new network fetch.
# ---------------------------------------------------------------------------


def default_opener_lines(
    schedule: pd.DataFrame, *, market_root: Path | None = None
) -> pd.DataFrame:
    """Tuesday-opener consensus home spread + total line, keyed by ``game_id``.

    ``tue_open_home_spread`` follows this repo's uniform sign convention
    (positive = HOME favored by that many points; see
    ``nfl_ats.open_benchmark``'s ``"positive_spread_line_means_home_favorite"``
    and ``nfl_ats.market_data.parse_odds_api_response``'s
    ``standardized_home_line = -home_point``) -- identical to the nflverse
    schedule's own ``spread_line``, just measured at the Tuesday opener
    instead of the close. A game absent from the historical decision-labeled
    archive, or present without a resolved total, gets NaN in the
    corresponding column here; callers must treat NaN as "unknown," never as
    zero, before applying a threshold.
    """

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


# ---------------------------------------------------------------------------
# LEAD-41: dome-shootout favorite archetype
# ---------------------------------------------------------------------------

DOME_SHOOTOUT_ROOFS = frozenset({"dome", "closed"})
DOME_SHOOTOUT_TOTAL_MIN = 49.0
DOME_SHOOTOUT_SPREAD_MAX_ABS = 3.0

_DOME_SHOOTOUT_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "roof"}


def oracle_derive_dome_shootout_favorite_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, dome_shootout_favorite_flag)`` for every game.

    Archetype: ``roof`` is a fixed dome or a retractable roof recorded
    CLOSED for this game, AND the Tuesday-opener total is >= 49, AND the
    Tuesday-opener home spread's absolute value is <= 3. ``+1`` when the
    HOME team is the favorite in an archetype game (opener home spread >
    0), ``-1`` when the AWAY team is (opener home spread < 0), ``0``
    otherwise -- including a non-archetype game, an exact pick'em (spread
    == 0, no favorite to back), or a game missing an opener total/spread in
    the store (never silently treated as satisfying either threshold).

    Declared approximation, the same kind ``nfl_ats.clv.opener_pick_evaluation``
    already accepts for every sibling on-production candidate: ``roof`` is
    read from the schedule's own (post-decision) recorded value, which for a
    retractable-roof venue may not be finalized until close to kickoff --
    later than the Tuesday opener this flag is otherwise scored against.
    """

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
    """Additively join ``dome_shootout_favorite_flag`` onto ``features``.

    ``opener_lines`` may be supplied directly (fixtures, tests) to avoid
    touching the real market store; otherwise it is loaded via
    :func:`default_opener_lines` from the resolved schedule.
    """

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_dome_shootout_favorite_features(sched, lines, announcements=announcements)

    return _attach(features, schedule, _derive, (DOME_SHOOTOUT_COLUMN,))


# ---------------------------------------------------------------------------
# LEAD-42: low-total divisional home dog
# ---------------------------------------------------------------------------

LOW_TOTAL_DIV_DOG_TOTAL_MAX = 42.0

_LOW_TOTAL_DIV_DOG_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "div_game"}


def derive_low_total_div_home_dog_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, low_total_div_home_dog_flag)`` for every game.

    ``1.0`` when the game is divisional (``div_game == 1``), the
    Tuesday-opener total is <= 42, AND the home team is the underdog at the
    Tuesday opener (opener home spread < 0); ``0.0`` otherwise -- including
    a game missing an opener total/spread in the store. A missing opener
    total is NEVER encoded as 0 for the threshold comparison itself (that
    would wrongly satisfy "<= 42"); only the FINAL flag defaults to 0 when
    any required opener input is unresolved.
    """

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
    """Additively join ``low_total_div_home_dog_flag`` onto ``features``.

    ``opener_lines`` may be supplied directly (fixtures, tests) to avoid
    touching the real market store; otherwise it is loaded via
    :func:`default_opener_lines` from the resolved schedule.
    """

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_low_total_div_home_dog_features(sched, lines)

    return _attach(features, schedule, _derive, (LOW_TOTAL_DIV_DOG_COLUMN,))


# ---------------------------------------------------------------------------
# LEAD-35: September heat-humidity home edge
# ---------------------------------------------------------------------------

SEPT_HEAT_UNCONDITIONAL_HOME_TEAMS = frozenset({"MIA", "TB", "JAX"})
#: Only when this game's own roof is outdoors/open -- these three venues can
#: also play under a closed roof/dome, which removes the heat/humidity
#: mechanism entirely (measured 2026-09-05: HOU roof is closed for 124/139
#: recorded games, "open" for 15; NO is a fixed dome for 145/147, "outdoors"
#: for 2; ATL is closed 55, dome 63 [Georgia Dome era], open 18, outdoors 2,
#: of 138).
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
#: Eastern Time is this repo's established schedule.parquet ``gametime``
#: convention (scripts/body_clock_screen.py compares raw "%H:%M" ``gametime``
#: minutes directly against ET-labeled thresholds); "1 PM local" needs each
#: home team's OWN clock, so its ET kickoff is shifted back by its zone's
#: constant offset from Eastern. All six heat-candidate teams sit in either
#: America/New_York (0h behind ET) or America/Chicago (always exactly 1h
#: behind ET -- both zones observe US daylight saving on the same calendar
#: dates, so the gap never varies by season or date); verified against
#: registry/stadium_coordinates.json's own tz entries for Hard Rock Stadium
#: (MIA), Raymond James Stadium (TB), TIAA Bank/EverBank Stadium (JAX),
#: Mercedes-Benz Stadium (ATL) = America/New_York, and NRG/Reliant Stadium
#: (HOU), Caesars/Mercedes-Benz/Louisiana Superdome (NO) = America/Chicago.
SEPT_HEAT_HOME_TEAM_ET_OFFSET_HOURS: dict[str, int] = {
    "MIA": 0,
    "TB": 0,
    "JAX": 0,
    "ATL": 0,
    "HOU": 1,
    "NO": 1,
}
_SEPT_HEAT_LOCAL_ONE_PM_START_MIN = 13 * 60
_SEPT_HEAT_LOCAL_ONE_PM_END_MIN = 14 * 60  # exclusive

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
    """Return ``(game_id, sept_heat_home_flag)`` for every game in ``schedule``.

    ``1.0`` when ALL hold: the game is REG season, week <= 3; the HOME team
    is heat-acclimated (MIA/TB/JAX unconditionally, or HOU/NO/ATL only when
    this game's own roof is outdoors/open); the AWAY team is on the frozen
    cold-climate list; and the home team's own LOCAL kickoff hour is 13 (1
    PM local, converted from the schedule's Eastern-Time ``gametime`` by the
    home team's fixed ET offset). ``0.0`` otherwise. A plain pregame
    schedule/roster-assignment fact; no outcome of any game is read.
    """

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
    """Additively join ``sept_heat_home_flag`` onto ``features`` by ``game_id``."""

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        return derive_sept_heat_home_features(sched, announcements=announcements)

    return _attach(features, schedule, _derive, (SEPT_HEAT_COLUMN,))


# ---------------------------------------------------------------------------
# Wave 3 (docs/schedule_flag_battery.md "Wave 3"): LEAD-57 public-claim leads
# on production. road_fav_big_fade, division_dog, and week1_dog read the
# Tuesday-OPENER consensus spread via default_opener_lines (never the
# nflverse schedule's own closing spread_line); ats_streak_regress reads only
# the schedule's own CLOSE result/spread_line, matching
# docs/public_claim_battery.md's own close-graded convention for the streak
# history itself (a frozen, predeclared design choice, not an oversight).
# ---------------------------------------------------------------------------

#: docs/public_claim_battery.md's own team_spread >= 7 threshold.
ROAD_FAV_BIG_FADE_SPREAD_MIN_ABS = 7.0

_ROAD_FAV_BIG_FADE_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "game_type"}


def derive_road_fav_big_fade_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, road_fav_big_fade_flag)`` for every game.

    docs/public_claim_battery.md's ``public_claim_road_fav_big_fade`` tested
    only fading a ROAD favorite of 7+ (sign -1: back the home team) and said
    nothing about a home favorite of 7+. docs/schedule_flag_battery.md
    "Wave 3" instructs a symmetric on-production extension: ``+1`` when the
    AWAY team is favored by >= 7 points at the Tuesday opener (the tested
    claim -- fade the road favorite, back home), ``-1`` when the HOME team is
    favored by >= 7 points at the opener (the mirror case, NOT separately
    tested by lane G's battery, disclosed rather than silently assumed),
    ``0`` otherwise -- including a non-REG game, an opener spread inside
    +/-7, or a game missing a resolved opener spread (never silently treated
    as satisfying either threshold).
    """

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
    """Additively join ``road_fav_big_fade_flag`` onto ``features``.

    ``opener_lines`` may be supplied directly (fixtures, tests) to avoid
    touching the real market store; otherwise it is loaded via
    :func:`default_opener_lines` from the resolved schedule.
    """

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_road_fav_big_fade_features(sched, lines)

    return _attach(features, schedule, _derive, (ROAD_FAV_BIG_FADE_COLUMN,))


# ---------------------------------------------------------------------------
# division_dog / week1_dog share exactly one shape (docs/public_claim_battery.md
# claims 4 and 9, "week1 dog likewise" per the fleet task): BACK whichever
# side is the underdog at the Tuesday opener, within an eligible REG-season
# population (divisional game / Week 1). ``+1`` if the HOME team is the
# underdog, ``-1`` if the AWAY team is, ``0`` if the game is not eligible, is
# an exact opener pick'em, or the opener store lacks a resolved spread.
# Eligibility is restricted to game_type == "REG": measured 2026-09-05,
# postseason rows can carry div_game == 1 (26 of 371 postseason games) and
# can share week numbers with REG season (postseason week ranges 18-22,
# overlapping REG's own week 18), so an unrestricted mask would silently
# admit games lane G's REG-only population never tested.
# ---------------------------------------------------------------------------


def _dog_flag_from_opener_spread(eligible: pd.Series, spread: pd.Series) -> np.ndarray:
    home_dog = eligible & spread.notna() & spread.lt(0.0)
    away_dog = eligible & spread.notna() & spread.gt(0.0)
    return np.where(home_dog, 1.0, np.where(away_dog, -1.0, 0.0))


_DIVISION_DOG_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "game_type", "div_game"}


def derive_division_dog_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, division_dog_flag)`` for every game.

    ``+1`` when the game is a REG-season divisional game AND the home team
    is the underdog at the Tuesday opener; ``-1`` when it is divisional AND
    the away team is the underdog; ``0`` otherwise (non-divisional, not REG
    season, an exact opener pick'em, or a missing opener spread).
    """

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
    """Additively join ``division_dog_flag`` onto ``features``.

    ``opener_lines`` may be supplied directly (fixtures, tests) to avoid
    touching the real market store; otherwise it is loaded via
    :func:`default_opener_lines` from the resolved schedule.
    """

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
    """Return ``(game_id, week1_dog_flag)`` for every game.

    Same shape as :func:`derive_division_dog_features`: ``+1`` when it is a
    REG-season Week 1 game AND the home team is the underdog at the Tuesday
    opener, ``-1`` when it is Week 1 AND the away team is, ``0`` otherwise.
    """

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
    """Additively join ``week1_dog_flag`` onto ``features``.

    ``opener_lines`` may be supplied directly (fixtures, tests) to avoid
    touching the real market store; otherwise it is loaded via
    :func:`default_opener_lines` from the resolved schedule.
    """

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_week1_dog_features(sched, lines)

    return _attach(features, schedule, _derive, (WEEK1_DOG_COLUMN,))


# ---------------------------------------------------------------------------
# ats_streak_regress: BACK a team on a 3+ game ATS losing streak entering
# this game (docs/public_claim_battery.md's ``public_claim_ats_streak_regress``).
# Streak history is graded at the CLOSE (schedule's own result/spread_line),
# matching the archive's own convention exactly (docs/schedule_flag_battery.md
# "Wave 3" states this is a frozen, predeclared choice); only the streak
# LENGTH counts toward this game's flag -- nothing here ever reads this
# game's own result or spread_line.
# ---------------------------------------------------------------------------

#: docs/public_claim_battery.md's own ats_streak_len >= 3 threshold.
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
    """One row per (team, REG game), that team's own cover-loss streak
    length STRICTLY ENTERING that game (0.0 for a team's first REG game of a
    season).

    A push (``home_cover`` NaN) neither extends nor resets the streak -- it
    is skipped from the team's own ordered sequence entirely, exactly the
    convention docs/public_claim_battery.md's own ``ats_streak_len`` column
    freezes. The streak resets to 0 at every season boundary and on any
    cover. Nothing here reads a row's own outcome to compute that SAME row's
    entering streak -- only strictly earlier rows in the team's own ordered
    sequence feed each entering value.
    """

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
        covered = covered.where(reg["home_cover"].notna())  # push stays NaN either side
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
                continue  # push: neither extends nor resets
            current = 0.0 if covered_value >= 1.0 else current + 1.0
    long_df["streak_entering"] = streaks_entering
    return long_df[["game_id", "team", "streak_entering"]]


def derive_ats_streak_regress_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return ``(game_id, ats_streak_regress_flag)`` for every game in ``schedule``.

    ``+1`` if the HOME team enters this game on a losing ATS streak of
    :data:`ATS_STREAK_REGRESS_MIN_STREAK` (3) or more AND the AWAY team does
    not; ``-1`` if the reverse; ``0`` if both do, neither does, or the game
    is not REG season (streak history and the flag itself are both built
    from REG-season games only, matching docs/public_claim_battery.md's own
    population -- a stated design choice, not an oversight).
    """

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
    """Additively join ``ats_streak_regress_flag`` onto ``features`` by ``game_id``."""

    return _attach(
        features, schedule, derive_ats_streak_regress_features, (ATS_STREAK_REGRESS_COLUMN,)
    )


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


# Frozen venue policy: retractable roofs default closed, fixed roofs dome.
# Names, not home-team stadium ids, preserve neutral-site assignments.
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
    """Use venue metadata unless a roof announcement was observed before cutoff."""
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
