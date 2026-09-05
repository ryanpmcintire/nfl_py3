"""LEAD-15 backup tenure-gap valuation flag, stacked on PRODUCTION
(``docs/schedule_flag_battery.md`` "Wave 9").

**Mechanism (ROADMAP.md LEAD-15).** The market applies a roughly uniform
"backup QB" haircut whenever a team's listed starter is not its normal
starter, but a backup who has spent two or more seasons learning the SAME
team's system is hypothesised to outperform a backup who just joined that
team -- the haircut the market applies does not distinguish the two.
Predeclared direction: BACK the team starting a system-tenured backup, FADE
the team starting a new-system (fresh) backup.

**Data sources, both already-captured local snapshots, no network fetch:**
the newest ``data/raw/*/schedules.parquet`` snapshot's listed starters
(``home_qb_id``/``away_qb_id``, keyed by ``gsis_id``) plus the PINNED
``data/players/raw/20260817T184901Z/weekly_rosters.parquet`` (season/team/
gsis_id membership, the same pinned convention
``nfl_ats.roster_availability_flag_features`` already uses for its own
player-level inputs -- pinned, not "newest", so a later snapshot landing
mid-session from a concurrent lane cannot silently change this family's
population after predeclaration).

**Measurement caveat, stated up front (matches ``nfl_ats.qb_identity_features``'s
own disclosure verbatim):** the schedule's own ``home_qb_id``/``away_qb_id``
are the POST-HOC recorded starter for a played game, not a pregame
depth-chart projection. A listed starting quarterback is announced well
before Sunday in the real world, so this is knowable before kickoff; the
project's live weekly card would source the same starter identity from the
injury/depth-chart pipeline (``lineups.json``) instead of the schedule's own
post-hoc column. This module's population is therefore a measurement of
history, not a claim that the schedule parquet itself is a legitimate LIVE
input.

**"Depth-chart QB1" declared approximation.** No pregame depth-chart archive
is used to identify a team's presumptive starter (the fleet task explicitly
places lane AB's new all-position depth-chart archive under
``data/players/raw/depth_charts/`` out of scope for this family: even though
that archive now exists locally, per instruction it is not read here). A
team's "depth-chart QB1 entering this game" is instead approximated as the
starter of that SAME team's most recent PRIOR game in the full schedule
archive, with **no season-boundary reset** -- unlike the schedule-only flags
in ``nfl_ats.schedule_flag_features`` (LEAD-21/LEAD-22), which deliberately
reset their lookback every season because their mechanism (physical fatigue)
genuinely does not survive an offseason. Here the opposite is true: the best
pregame-knowable guess for who a team's Week 1 starter will be, absent a
depth-chart feed, is whoever started that team's last game the previous
season -- exactly the "preseason" anchor the task names. A bye week is
automatically skipped (it is not a row in the schedule). A team's very first
archived game (2009 Week 1, for every team) has no prior game at all and is
therefore never counted as a backup start -- moot for this family's declared
population (below), which starts in 2013.

**Backup start.** A side "starts a backup" in a game when its actual listed
starter differs from that depth-chart-QB1 proxy AND a proxy exists (i.e. the
team has played at least one prior archived game, home or away, anywhere).

**Tenure.** For an identified backup starter, franchise tenure is the count
of DISTINCT SEASONS STRICTLY BEFORE this game's season in which that
``gsis_id`` appears on ``weekly_rosters`` for the SAME (alias-canonicalized)
franchise the game credits him to. This reads only strictly-prior-season
roster membership, so it can never leak this game's own season. A backup
whose ``gsis_id`` never appears anywhere in ``weekly_rosters`` (an
unresolved identity, e.g. a crosswalk gap) contributes to neither bucket
below -- never guessed. ``BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS`` (2)
sets "system-tenured" (>= 2 prior seasons with the SAME franchise, i.e. this
is at least his third season there); a resolved backup with 0 or 1 prior
seasons with that franchise is "new-system" (fresh).

**Franchise continuity across relocation.** Both the schedule's own
``home_team``/``away_team`` and the roster's own ``team`` column are
canonicalized through ``nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`` (the
same alias table every other franchise-continuity feature in this repo uses:
``nfl_ats.qb_identity_features``, ``nfl_ats.transaction_wire_features``,
``nfl_ats.pbp_coaching_traits``) -- a backup who has been with the Rams since
their St. Louis seasons, or the Raiders since Oakland, or the Chargers since
San Diego, counts that tenure as continuous with the SAME franchise, matching
the mechanism's own "same system" language, not a fresh 0-tenure reset on
relocation.

**Declared population restriction: seasons 2013-2025**, per the fleet task's
explicit instruction (not separately re-derived here) -- the flag evaluates
to 0.0 for every game outside this range regardless of what the backup/tenure
computation above would otherwise say, and is reported as a stated design
choice, not a measured optimum.

**Signed ``backup_tenure_gap_flag``**, following the fleet task's own
worked truth table: ``+1`` when the HOME team starts a system-tenured backup
OR the AWAY team starts a new-system backup (both favour home); ``-1`` the
mirror (both favour away); ``0`` otherwise -- including no backup start on
either side, an unresolved backup identity contributing to neither bucket,
or a "both sides favour the same direction independently" case cancelling
out (e.g. both teams start system-tenured backups, or both start new-system
backups) exactly as the task's own "0 ... or both" clause specifies.

Mirrors ``nfl_ats.schedule_flag_features``'s additive-merge discipline: every
pre-existing column comes back bit-identical, only the one new column is
added. This module is intentionally self-contained (its own
``default_schedule`` loader is duplicated, not imported, from
``nfl_ats.schedule_flag_features``/``nfl_ats.qb_identity_features``) so it has
no dependency on either concurrently-edited module.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    BACKUP_TENURE_GAP_ON_PRODUCTION_FEATURE_COLUMNS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError
from nfl_ats.players import canonicalize_rosters

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The one new column this candidate profile adds. Frozen name.
BACKUP_TENURE_GAP_COLUMN = BACKUP_TENURE_GAP_ON_PRODUCTION_FEATURE_COLUMNS[0]

#: Pinned, not "newest" -- see module docstring.
DEFAULT_WEEKLY_ROSTERS_PATH = REPO_ROOT / "data/players/raw/20260817T184901Z/weekly_rosters.parquet"

#: Frozen population restriction, per the fleet task's explicit instruction.
BACKUP_TENURE_POPULATION_SEASON_START = 2013
BACKUP_TENURE_POPULATION_SEASON_END = 2025

#: "system-tenured" threshold: prior seasons with the SAME franchise.
BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS = 2

_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "gameday",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
}
_REQUIRED_ROSTER_COLUMNS = {"season", "team", "gsis_id"}


def default_schedule(repo_root: Path | None = None) -> pd.DataFrame:
    """Load the newest ``data/raw/*/schedules.parquet`` snapshot.

    Duplicated (not imported) from ``nfl_ats.schedule_flag_features`` /
    ``nfl_ats.qb_identity_features``'s identical "newest snapshot, sorted
    lexicographically" convention -- see module docstring for why this
    module avoids a dependency on either concurrently-edited module.
    """

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {root}")
    return pd.read_parquet(candidates[-1])


def default_weekly_rosters(path: Path | None = None) -> pd.DataFrame:
    """Load and canonicalize the PINNED weekly-rosters snapshot (see module
    docstring). Reuses ``nfl_ats.players.canonicalize_rosters`` (read-only
    import; ``players.py`` itself is never edited by this family)."""

    raw = pd.read_parquet(path or DEFAULT_WEEKLY_ROSTERS_PATH)
    return canonicalize_rosters(raw)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"{label} is missing columns: {', '.join(missing)}")


def _canonical_team(codes: pd.Series) -> pd.Series:
    """Canonicalize a team-code column through the shared
    ``TEAM_ABBREVIATION_ALIASES`` table (OAK->LV, SD->LAC, STL/SL->LA, ...)."""

    return codes.astype(str).replace(TEAM_ABBREVIATION_ALIASES)


# ---------------------------------------------------------------------------
# Starter identity + backup-start detection
# ---------------------------------------------------------------------------


def _qb_start_long_table(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side) with a resolved starter: ``team``, ``qb_id``,
    that team's own PRECEDING archived-game starter (``prev_qb_id``, no
    season-boundary reset -- see module docstring), and ``is_backup_start``.
    """

    _require_columns(schedule, _REQUIRED_SCHEDULE_COLUMNS, "schedule")

    frame = schedule.loc[:, sorted(_REQUIRED_SCHEDULE_COLUMNS)].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["gameday_dt"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame["game_id"] = frame["game_id"].astype(str)

    sides = []
    for is_home, team_col, qb_col in (
        (True, "home_team", "home_qb_id"),
        (False, "away_team", "away_qb_id"),
    ):
        side = frame.loc[
            frame[qb_col].notna(), ["game_id", "season", "gameday_dt", team_col, qb_col]
        ].rename(columns={team_col: "team", qb_col: "qb_id"})
        side["team"] = _canonical_team(side["team"])
        side["qb_id"] = side["qb_id"].astype(str)
        side["is_home"] = is_home
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.sort_values(["team", "gameday_dt", "game_id"]).reset_index(drop=True)

    # Deliberately grouped by team ONLY (no season) -- the presumed starter
    # carries across the offseason, the declared proxy for a "preseason"
    # depth-chart QB1 the archive has no direct feed for. See module
    # docstring.
    grouped = long_df.groupby("team", sort=False)
    long_df["prev_qb_id"] = grouped["qb_id"].shift(1)
    long_df["is_backup_start"] = long_df["prev_qb_id"].notna() & long_df["qb_id"].ne(
        long_df["prev_qb_id"]
    )
    return long_df


# ---------------------------------------------------------------------------
# Tenure lookup
# ---------------------------------------------------------------------------


def _prior_season_tenure(long_df: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Attach ``tenure_prior_seasons`` (count of distinct SAME-team roster
    seasons strictly before this game's season) and ``tenure_resolved``
    (whether the starting ``qb_id`` appears anywhere in ``rosters`` at all)
    to every row of ``long_df``. Reads only ``season < this game's season``
    -- never this game's own season -- so tenure can never leak.
    """

    _require_columns(rosters, _REQUIRED_ROSTER_COLUMNS, "rosters")

    roster_pairs = rosters.loc[:, ["season", "team", "gsis_id"]].dropna(subset=["gsis_id"]).copy()
    roster_pairs["season"] = pd.to_numeric(roster_pairs["season"], errors="raise").astype(int)
    roster_pairs["team"] = _canonical_team(roster_pairs["team"])
    roster_pairs["gsis_id"] = roster_pairs["gsis_id"].astype(str)
    roster_pairs = roster_pairs.drop_duplicates(["gsis_id", "team", "season"]).sort_values(
        ["gsis_id", "team", "season"]
    )
    # Cumulative count of this player's OWN distinct (gsis_id, team) seasons
    # up to and including each row -- at the row for season S this equals
    # the number of distinct seasons <= S for that (gsis_id, team) pair.
    roster_pairs["seasons_through_row"] = (
        roster_pairs.groupby(["gsis_id", "team"], sort=False).cumcount() + 1
    )

    known_ids = set(roster_pairs["gsis_id"].unique())

    # ``merge_asof`` requires BOTH frames sorted globally by the ``on`` key
    # (season) -- sorting only within each (gsis_id, team) group is not
    # sufficient and raises "left/right keys must be sorted" (verified
    # 2026-09-05 against a synthetic two-group case where group order and
    # global season order disagree).
    query = long_df.loc[:, ["qb_id", "team", "season"]].rename(columns={"qb_id": "gsis_id"})
    query = query.reset_index().sort_values("season")
    matched = pd.merge_asof(
        query,
        roster_pairs.sort_values("season"),
        on="season",
        by=["gsis_id", "team"],
        direction="backward",
        allow_exact_matches=False,
    )
    matched = matched.sort_values("index").set_index("index")
    tenure_prior_seasons = matched["seasons_through_row"].fillna(0.0).astype(int)

    result = long_df.copy()
    result["tenure_prior_seasons"] = (
        tenure_prior_seasons.reindex(result.index).fillna(0).astype(int)
    )
    result["tenure_resolved"] = result["qb_id"].isin(known_ids)
    return result


def describe_backup_tenure_population(schedule: pd.DataFrame, rosters: pd.DataFrame) -> dict:
    """Diagnostic counts for the backup-tenure population (never used to
    build the flag itself, only reported alongside it, per this repo's
    standing convention -- see ``nfl_ats.qb_identity_features.
    describe_rookie_qb_debut_population``): total backup-start sides, how
    many resolve a tenure at all, how many are system-tenured vs new-system,
    and flagged-GAME counts by season within the declared 2013-2025
    population.
    """

    long_df = _qb_start_long_table(schedule)
    long_df = _prior_season_tenure(long_df, rosters)

    in_population = long_df["season"].between(
        BACKUP_TENURE_POPULATION_SEASON_START, BACKUP_TENURE_POPULATION_SEASON_END
    )
    backups = long_df.loc[long_df["is_backup_start"] & in_population]
    tenured = backups.loc[
        backups["tenure_resolved"]
        & backups["tenure_prior_seasons"].ge(BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS)
    ]
    fresh = backups.loc[
        backups["tenure_resolved"]
        & backups["tenure_prior_seasons"].lt(BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS)
    ]
    unresolved = backups.loc[~backups["tenure_resolved"]]

    by_season = backups.groupby("season")["game_id"].nunique().sort_index().to_dict()
    return {
        "n_backup_start_sides_2013_2025": len(backups),
        "n_backup_start_games_2013_2025": int(backups["game_id"].nunique()),
        "n_system_tenured_backup_sides": len(tenured),
        "n_new_system_backup_sides": len(fresh),
        "n_unresolved_tenure_backup_sides": len(unresolved),
        "flagged_games_by_season": {int(cast(int, k)): int(v) for k, v in by_season.items()},
    }


# ---------------------------------------------------------------------------
# Signed flag
# ---------------------------------------------------------------------------


def derive_backup_tenure_gap_features(
    schedule: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, backup_tenure_gap_flag)`` for EVERY game in
    ``schedule`` (including a game missing one or both starters, e.g. a not-
    yet-played future game, which gets ``0.0`` -- matching
    ``nfl_ats.qb_identity_features.derive_rookie_qb_debut_fade_features``'s
    own "start from every game_id, left-merge, fillna(False)" convention, so
    this candidate profile can be evaluated over the full production table
    without introducing new NaNs). See module docstring for the sign
    convention.
    """

    long_df = _qb_start_long_table(schedule)
    long_df = _prior_season_tenure(long_df, rosters)

    long_df["is_system_tenured"] = (
        long_df["is_backup_start"]
        & long_df["tenure_resolved"]
        & long_df["tenure_prior_seasons"].ge(BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS)
    )
    long_df["is_new_system"] = (
        long_df["is_backup_start"]
        & long_df["tenure_resolved"]
        & long_df["tenure_prior_seasons"].lt(BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS)
    )

    all_ids = schedule[["game_id"]].astype({"game_id": str})
    home_flags = long_df.loc[
        long_df["is_home"], ["game_id", "is_system_tenured", "is_new_system"]
    ].rename(
        columns={
            "is_system_tenured": "home_is_system_tenured",
            "is_new_system": "home_is_new_system",
        }
    )
    away_flags = long_df.loc[
        ~long_df["is_home"], ["game_id", "is_system_tenured", "is_new_system"]
    ].rename(
        columns={
            "is_system_tenured": "away_is_system_tenured",
            "is_new_system": "away_is_new_system",
        }
    )
    result = all_ids.merge(home_flags, on="game_id", how="left")
    result = result.merge(away_flags, on="game_id", how="left")
    for column in (
        "home_is_system_tenured",
        "home_is_new_system",
        "away_is_system_tenured",
        "away_is_new_system",
    ):
        result[column] = result[column].fillna(False)

    favor_home = result["home_is_system_tenured"] | result["away_is_new_system"]
    favor_away = result["away_is_system_tenured"] | result["home_is_new_system"]
    flag = np.where(favor_home & ~favor_away, 1.0, np.where(favor_away & ~favor_home, -1.0, 0.0))
    result[BACKUP_TENURE_GAP_COLUMN] = flag

    # Frozen population restriction (see module docstring): 0.0 outside
    # 2013-2025 regardless of what the computation above would say.
    season_by_game = (
        schedule.assign(game_id=schedule["game_id"].astype(str))
        .loc[:, ["game_id", "season"]]
        .drop_duplicates("game_id")
    )
    result = result.merge(season_by_game, on="game_id", how="left", validate="one_to_one")
    in_population = result["season"].between(
        BACKUP_TENURE_POPULATION_SEASON_START, BACKUP_TENURE_POPULATION_SEASON_END
    )
    result.loc[~in_population, BACKUP_TENURE_GAP_COLUMN] = 0.0
    return result.loc[:, ["game_id", BACKUP_TENURE_GAP_COLUMN]]


def attach_backup_tenure_gap_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join ``backup_tenure_gap_flag`` onto ``features`` by
    ``game_id``. Mirrors ``nfl_ats.schedule_flag_features``'s additive-merge
    discipline (duplicated, not imported -- see module docstring)."""

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if BACKUP_TENURE_GAP_COLUMN in features.columns:
        raise DataContractError(f"features already carries {BACKUP_TENURE_GAP_COLUMN}")

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_rosters = rosters if rosters is not None else default_weekly_rosters()
    derived = derive_backup_tenure_gap_features(resolved_schedule, resolved_rosters)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_backup_tenure"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_backup_tenure") if c in merged.columns]
    )
    merged.index = features.index
    return merged


__all__ = [
    "BACKUP_TENURE_GAP_COLUMN",
    "BACKUP_TENURE_POPULATION_SEASON_END",
    "BACKUP_TENURE_POPULATION_SEASON_START",
    "BACKUP_TENURE_SYSTEM_TENURED_MIN_SEASONS",
    "DEFAULT_WEEKLY_ROSTERS_PATH",
    "attach_backup_tenure_gap_features",
    "default_schedule",
    "default_weekly_rosters",
    "derive_backup_tenure_gap_features",
    "describe_backup_tenure_population",
]
