"""Two quarterback-identity pregame flags, each stacked on PRODUCTION
(``docs/schedule_flag_battery.md`` "Wave 5"): LEAD-20 rookie-QB debut fade,
LEAD-25 quarterback revenge game.

Both flags read only the newest ``data/raw/*/schedules.parquet`` snapshot's
listed starters (``home_qb_id``/``away_qb_id``, keyed by ``gsis_id``) plus
already-captured local rosters/combine data -- no network fetch, no PBP, no
market data. **Measurement caveat, stated up front and repeated in the
predeclaration doc:** the schedule's own ``home_qb_id``/``away_qb_id`` are the
POST-HOC recorded starter for a played game, not a pregame depth-chart
projection. Both quantities used here (who started, and that starter's own
draft history/career-experience) are knowable before kickoff in the real
world -- a listed starting quarterback is announced well before Sunday, and a
player's draft team and years of NFL experience are historical facts fixed
long before this season began -- so neither flag reads any information that
postdates the prediction timestamp. The project's live weekly card would
source the same starter identity from the injury/depth-chart pipeline
(``lineups.json``) instead of the schedule's own post-hoc column; this
module's population is therefore a measurement of history, not a claim that
the schedule parquet itself is a legitimate LIVE input.

LEAD-20: rookie-QB debut fade
------------------------------
A debut is the quarterback's first **REG-season** start anywhere in the
2009-2025 archive AND that player is a rookie THAT SEASON per
``weekly_rosters.years_exp == 0`` -- the rookie gate exists because the
archive itself begins in 2009, so an established veteran whose first
*archived* start happens to be a 2009 game (a genuine, real NFL veteran, not
a debut) would otherwise be mislabelled as debuting. ``describe_rookie_qb_debut_population``
reports the count of first-archived-starts that are NOT rookies as a
diagnostic, exactly as predeclared.

Signed ``rookie_qb_debut_fade_flag``: ``+1`` when the AWAY starter is a debut
rookie (fade the road debut -> favour home), ``-1`` when the HOME starter is,
``0`` otherwise (including both sides debuting, which cannot happen in
practice since two rookies cannot both have zero prior starts against each
other on debut day without one of them being credited with a start already --
kept as an explicit branch for completeness and symmetry with every sibling
flag in this repo).

LEAD-25: quarterback revenge game
-----------------------------------
BACK the quarterback facing the franchise that drafted him. Draft team comes
from ``data/raw/combine/*/combine.parquet``'s ``draft_team`` (a full team
name, e.g. "Oakland Raiders") plus ``draft_year``/``pfr_id``; ``pfr_id`` is
joined to ``gsis_id`` through ``weekly_rosters``' own pfr/gsis crosswalk,
reusing ``nfl_ats.players._stable_crosswalk`` (the identical helper
``nfl_ats.players.attach_snap_player_ids`` already uses to link PFR player
identities to GSIS IDs, imported rather than re-derived so both call sites
share one crosswalk-selection rule: for a ``pfr_id`` with more than one
observed ``gsis_id`` across roster rows, take the most frequently co-occurring
one, GSIS-id ascending as a deterministic tiebreak).

Franchise relocations are normalised through a FROZEN
``DRAFT_TEAM_NAME_TO_CODE`` mapping (every historical AND current full team
name -> current canonical abbreviation: "Oakland Raiders"/"Las Vegas Raiders"
-> ``LV``; "San Diego Chargers"/"Los Angeles Chargers" -> ``LAC``;
"St. Louis Rams"/"Los Angeles Rams" -> ``LA``; "Washington
Redskins"/"Washington Football Team"/"Washington Commanders" -> ``WAS``) and
the schedule's own ``home_team``/``away_team`` (which still carry the
historical ``OAK``/``SD``/``STL`` codes for old games) are canonicalised
through the SAME ``nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`` every other
franchise-continuity feature in this repo already uses
(``nfl_ats.transaction_wire_features.canonical_team``,
``nfl_ats.pbp_coaching_traits``, ``nfl_ats.pbp_trait_on_production_features``),
so both sides of the revenge comparison share one canonical code space
regardless of which season's schedule row is being read.

Signed ``qb_revenge_flag``: ``+1`` when the HOME QB faces the franchise that
drafted him, ``-1`` when the AWAY QB does, ``0`` otherwise (including both
sides being a revenge game simultaneously, or a QB whose draft team could not
be resolved -- an unjoined QB is treated as ``0`` for that side, never
guessed).

**Distinct from the deployed division-revenge TEAM overlay.** ``gap_division_revenge``
(``nfl_ats.weak_stack_v3_features._add_gap_bias_flags``, already in
PRODUCTION's own ``weak_stack_v3`` feature set) fires when a TEAM plays a
divisional opponent it already lost to earlier the same season -- a
team-level rematch-after-a-loss construct with no reference to any individual
player. ``qb_revenge_flag`` is a PLAYER-level construct (a specific
quarterback facing the specific franchise that drafted him, regardless of
division and regardless of any earlier result this season) and is never
pooled with, or read as confirming/contradicting, the division-revenge cell.

Mirrors ``nfl_ats.schedule_flag_features``'s additive-merge discipline: every
pre-existing column comes back bit-identical, only the one new column is
added.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    QB_REVENGE_ON_PRODUCTION_FEATURE_COLUMNS,
    ROOKIE_QB_DEBUT_FADE_ON_PRODUCTION_FEATURE_COLUMNS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError
from nfl_ats.players import _stable_crosswalk, canonicalize_rosters, latest_player_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The one new column each candidate profile adds. Frozen names.
ROOKIE_QB_DEBUT_FADE_COLUMN = ROOKIE_QB_DEBUT_FADE_ON_PRODUCTION_FEATURE_COLUMNS[0]
QB_REVENGE_COLUMN = QB_REVENGE_ON_PRODUCTION_FEATURE_COLUMNS[0]

DEFAULT_PLAYERS_RAW_ROOT = REPO_ROOT / "data/players/raw"
DEFAULT_COMBINE_RAW_ROOT = REPO_ROOT / "data/raw/combine"

_ROOKIE_QB_DEBUT_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "gameday",
    "game_type",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
}
_QB_REVENGE_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
}

#: Frozen 2026-09-05 against ``data/raw/combine/20260822T143152Z/combine.parquet``
#: -- exhaustively covers every one of the 36 unique ``draft_team`` values
#: observed in that snapshot. Historical franchise names map to the CURRENT
#: canonical abbreviation, matching ``nfl_ats.constants.TEAM_ABBREVIATION_ALIASES``'s
#: own OAK->LV / SD->LAC / STL,SL->LA canonicalization of the schedule's own
#: historical codes, so both sides of the revenge join share one code space.
DRAFT_TEAM_NAME_TO_CODE: dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Oakland Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "St. Louis Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------


def _require_schedule_columns(schedule: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


def default_schedule(repo_root: Path | None = None) -> pd.DataFrame:
    """Load the newest ``data/raw/*/schedules.parquet`` snapshot.

    Same "newest snapshot, sorted lexicographically" convention every
    schedule-only battery in this repo uses
    (``nfl_ats.schedule_flag_features.default_schedule``), duplicated here
    rather than imported so this module has no dependency on the concurrently
    edited ``schedule_flag_features`` module.
    """

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {root}")
    return pd.read_parquet(candidates[-1])


def default_weekly_rosters(repo_root: Path | None = None) -> pd.DataFrame:
    """Load and canonicalize the newest ``data/players/raw/<snapshot>/weekly_rosters.parquet``."""

    root = repo_root or REPO_ROOT
    snapshot = latest_player_snapshot(root / "data" / "players" / "raw")
    raw = pd.read_parquet(snapshot.rosters_path)
    return canonicalize_rosters(raw)


def latest_combine_snapshot(repo_root: Path | None = None) -> Path:
    """Newest ``data/raw/combine/<snapshot>/combine.parquet``."""

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw" / "combine").glob("*/combine.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no data/raw/combine/*/combine.parquet snapshot found under {root}"
        )
    return candidates[-1]


def default_combine(repo_root: Path | None = None) -> pd.DataFrame:
    """Load the newest local combine snapshot."""

    return pd.read_parquet(latest_combine_snapshot(repo_root))


# ---------------------------------------------------------------------------
# LEAD-20: rookie-QB debut fade
# ---------------------------------------------------------------------------


def _first_reg_start_table(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (qb_id, REG game they started), flagging whether THIS game
    is that quarterback's first-ever archived REG start, chronologically,
    regardless of which team he started for."""

    _require_schedule_columns(schedule, _ROOKIE_QB_DEBUT_REQUIRED_SCHEDULE_COLUMNS)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    sides = []
    for qb_id_column, is_home in (("home_qb_id", True), ("away_qb_id", False)):
        side = reg.loc[reg[qb_id_column].notna(), ["game_id", "season", "gameday_dt", qb_id_column]]
        side = side.rename(columns={qb_id_column: "qb_id"})
        side["is_home"] = is_home
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)
    long_df["qb_id"] = long_df["qb_id"].astype(str)
    long_df["game_id"] = long_df["game_id"].astype(str)
    long_df = long_df.sort_values(["qb_id", "gameday_dt", "game_id"]).reset_index(drop=True)
    long_df["is_first_archived_start"] = long_df.groupby("qb_id", sort=False).cumcount().eq(0)
    return long_df[["game_id", "season", "qb_id", "is_home", "is_first_archived_start"]]


def _season_years_exp(rosters: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, gsis_id) -> years_exp entering that season."""

    required = {"season", "gsis_id", "years_exp"}
    missing = sorted(required.difference(rosters.columns))
    if missing:
        raise DataContractError(f"rosters is missing columns: {', '.join(missing)}")
    cols = rosters.loc[:, ["season", "gsis_id", "years_exp"]].dropna(subset=["gsis_id"]).copy()
    cols["season"] = pd.to_numeric(cols["season"], errors="raise").astype(int)
    cols["gsis_id"] = cols["gsis_id"].astype(str)
    return cols.drop_duplicates(["season", "gsis_id"], keep="first")


def describe_rookie_qb_debut_population(schedule: pd.DataFrame, rosters: pd.DataFrame) -> dict:
    """Diagnostic counts for the rookie-QB debut population (never used to
    build the flag itself, only reported alongside it): the number of
    first-archived-REG-starts, how many are confirmed rookies
    (``years_exp == 0``), how many are confirmed NOT rookies (an established
    veteran whose first *archived* start happens not to be a real debut --
    the exact population the rookie gate exists to exclude), and how many
    could not be resolved against ``weekly_rosters`` at all.
    """

    starts = _first_reg_start_table(schedule)
    debut = starts.loc[starts["is_first_archived_start"]].copy()
    years_exp = _season_years_exp(rosters)
    debut = debut.merge(
        years_exp, left_on=["season", "qb_id"], right_on=["season", "gsis_id"], how="left"
    )
    resolved = debut["years_exp"].notna()
    is_rookie = resolved & debut["years_exp"].eq(0.0)
    is_established = resolved & debut["years_exp"].gt(0.0)
    return {
        "n_first_archived_reg_starts": len(debut),
        "n_confirmed_rookie_debuts": int(is_rookie.sum()),
        "n_confirmed_non_rookie_first_starts": int(is_established.sum()),
        "n_unresolved_years_exp": int((~resolved).sum()),
    }


def derive_rookie_qb_debut_fade_features(
    schedule: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, rookie_qb_debut_fade_flag)`` for every game in ``schedule``.

    ``+1`` if the AWAY starter is making his first-ever archived REG start
    AND is a rookie that season (``years_exp == 0``); ``-1`` if the HOME
    starter is; ``0`` otherwise -- including a non-REG game (a debut is only
    ever defined against a REG start), a first-archived start whose
    ``years_exp`` resolves to something other than 0 (an established veteran
    whose true NFL debut predates the 2009 archive -- see
    :func:`describe_rookie_qb_debut_population`), or a first-archived start
    that could not be joined to ``weekly_rosters`` at all (never guessed).
    """

    starts = _first_reg_start_table(schedule)
    debut = starts.loc[starts["is_first_archived_start"]].copy()
    years_exp = _season_years_exp(rosters)
    debut = debut.merge(
        years_exp, left_on=["season", "qb_id"], right_on=["season", "gsis_id"], how="left"
    )
    debut["is_debut_rookie"] = debut["years_exp"].eq(0.0)  # NaN-safe: NaN == 0.0 is False

    home_flags = debut.loc[debut["is_home"], ["game_id", "is_debut_rookie"]].rename(
        columns={"is_debut_rookie": "home_debut_rookie"}
    )
    away_flags = debut.loc[~debut["is_home"], ["game_id", "is_debut_rookie"]].rename(
        columns={"is_debut_rookie": "away_debut_rookie"}
    )

    all_ids = schedule[["game_id"]].astype({"game_id": str})
    result = all_ids.merge(home_flags, on="game_id", how="left")
    result = result.merge(away_flags, on="game_id", how="left")
    result["home_debut_rookie"] = result["home_debut_rookie"].fillna(False)
    result["away_debut_rookie"] = result["away_debut_rookie"].fillna(False)

    flag = np.where(
        result["away_debut_rookie"] & ~result["home_debut_rookie"],
        1.0,
        np.where(result["home_debut_rookie"] & ~result["away_debut_rookie"], -1.0, 0.0),
    )
    return pd.DataFrame({"game_id": result["game_id"], ROOKIE_QB_DEBUT_FADE_COLUMN: flag})


def attach_rookie_qb_debut_fade_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join ``rookie_qb_debut_fade_flag`` onto ``features`` by ``game_id``."""

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if ROOKIE_QB_DEBUT_FADE_COLUMN in features.columns:
        raise DataContractError(f"features already carries {ROOKIE_QB_DEBUT_FADE_COLUMN}")

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_rosters = rosters if rosters is not None else default_weekly_rosters()
    derived = derive_rookie_qb_debut_fade_features(resolved_schedule, resolved_rosters)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_qb_identity"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_qb_identity") if c in merged.columns]
    )
    merged.index = features.index
    return merged


# ---------------------------------------------------------------------------
# LEAD-25: quarterback revenge game
# ---------------------------------------------------------------------------


def _canonical_schedule_team(codes: pd.Series) -> pd.Series:
    """Canonicalize a schedule ``home_team``/``away_team`` column through the
    same ``TEAM_ABBREVIATION_ALIASES`` every other franchise-continuity
    feature in this repo uses (OAK->LV, SD->LAC, STL/SL->LA)."""

    return codes.astype(str).replace(TEAM_ABBREVIATION_ALIASES)


def draft_team_by_gsis_id(combine: pd.DataFrame, rosters: pd.DataFrame) -> dict[str, str]:
    """``gsis_id`` -> the CANONICAL current code of the franchise that drafted
    that player, via ``pfr_id`` -> ``gsis_id`` (``nfl_ats.players._stable_crosswalk``)
    then ``draft_team`` -> code (:data:`DRAFT_TEAM_NAME_TO_CODE`).

    Not restricted to combine rows whose own ``pos`` says "QB": the
    population that matters is whichever ``gsis_id`` later appears as a
    schedule QB starter, and a player's real draft team does not depend on
    how combine.parquet happens to have labelled his position. A player
    combine-invited more than once, or drafted more than once (a rare
    supplemental-draft edge case), keeps only his EARLIEST ``draft_year`` row
    -- his actual original draft.
    """

    required = {"pfr_id", "draft_team", "draft_year"}
    missing = sorted(required.difference(combine.columns))
    if missing:
        raise DataContractError(f"combine is missing columns: {', '.join(missing)}")

    rows = combine.loc[combine["pfr_id"].notna() & combine["draft_team"].notna()].copy()
    unrecognized = sorted(set(rows["draft_team"].unique()) - set(DRAFT_TEAM_NAME_TO_CODE))
    if unrecognized:
        raise DataContractError(f"unrecognized combine draft_team values: {unrecognized}")

    rows["pfr_id"] = rows["pfr_id"].astype(str)
    crosswalk = _stable_crosswalk(rosters)
    rows["gsis_id"] = rows["pfr_id"].map(crosswalk)
    rows = rows.loc[rows["gsis_id"].notna()].copy()
    rows["draft_team_code"] = rows["draft_team"].map(DRAFT_TEAM_NAME_TO_CODE)
    rows["draft_year"] = pd.to_numeric(rows["draft_year"], errors="coerce")
    rows = rows.sort_values(["gsis_id", "draft_year"]).drop_duplicates("gsis_id", keep="first")
    return dict(zip(rows["gsis_id"], rows["draft_team_code"], strict=True))


def qb_revenge_join_diagnostics(schedule: pd.DataFrame, draft_team_lookup: dict[str, str]) -> dict:
    """Measured join-rate diagnostic: of every non-null
    ``home_qb_id``/``away_qb_id`` occurrence in ``schedule`` (one row per
    side per game, i.e. weighted by how many games each quarterback
    started), what fraction resolve to a known draft-team code. Reported
    alongside the flag, never used to build it."""

    home = schedule["home_qb_id"].dropna().astype(str)
    away = schedule["away_qb_id"].dropna().astype(str)
    all_starts = pd.concat([home, away], ignore_index=True)
    resolved = all_starts.isin(draft_team_lookup)
    return {
        "n_qb_side_starts": len(all_starts),
        "n_resolved_draft_team": int(resolved.sum()),
        "join_rate": float(resolved.mean()) if len(all_starts) else float("nan"),
    }


def derive_qb_revenge_features(
    schedule: pd.DataFrame, draft_team_lookup: dict[str, str]
) -> pd.DataFrame:
    """Return ``(game_id, qb_revenge_flag)`` for every game in ``schedule``.

    ``+1`` when the HOME starter's draft-team code equals the (canonicalized)
    AWAY team; ``-1`` when the AWAY starter's draft-team code equals the
    (canonicalized) HOME team; ``0`` otherwise -- including both sides
    qualifying simultaneously, or a starter whose draft team could not be
    resolved (treated as ``0`` for that side, never guessed).
    """

    _require_schedule_columns(schedule, _QB_REVENGE_REQUIRED_SCHEDULE_COLUMNS)
    home_team = _canonical_schedule_team(schedule["home_team"])
    away_team = _canonical_schedule_team(schedule["away_team"])

    home_qb_known = schedule["home_qb_id"].notna()
    away_qb_known = schedule["away_qb_id"].notna()
    home_draft_team = schedule["home_qb_id"].astype(str).map(draft_team_lookup)
    away_draft_team = schedule["away_qb_id"].astype(str).map(draft_team_lookup)

    home_revenge = home_qb_known & home_draft_team.notna() & home_draft_team.eq(away_team)
    away_revenge = away_qb_known & away_draft_team.notna() & away_draft_team.eq(home_team)

    flag = np.where(
        home_revenge & ~away_revenge, 1.0, np.where(away_revenge & ~home_revenge, -1.0, 0.0)
    )
    return pd.DataFrame({"game_id": schedule["game_id"].astype(str), QB_REVENGE_COLUMN: flag})


def attach_qb_revenge_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    combine: pd.DataFrame | None = None,
    rosters: pd.DataFrame | None = None,
    draft_team_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Additively join ``qb_revenge_flag`` onto ``features`` by ``game_id``.

    ``draft_team_lookup`` may be supplied directly (fixtures, tests) to avoid
    touching the real combine/roster stores; otherwise it is built from
    ``combine``/``rosters`` (each loaded from the newest local snapshot if
    not supplied either).
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if QB_REVENGE_COLUMN in features.columns:
        raise DataContractError(f"features already carries {QB_REVENGE_COLUMN}")

    resolved_schedule = schedule if schedule is not None else default_schedule()
    if draft_team_lookup is not None:
        lookup = draft_team_lookup
    else:
        resolved_combine = combine if combine is not None else default_combine()
        resolved_rosters = rosters if rosters is not None else default_weekly_rosters()
        lookup = draft_team_by_gsis_id(resolved_combine, resolved_rosters)
    derived = derive_qb_revenge_features(resolved_schedule, lookup)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_qb_identity"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_qb_identity") if c in merged.columns]
    )
    merged.index = features.index
    return merged


__all__ = [
    "DEFAULT_COMBINE_RAW_ROOT",
    "DEFAULT_PLAYERS_RAW_ROOT",
    "DRAFT_TEAM_NAME_TO_CODE",
    "QB_REVENGE_COLUMN",
    "ROOKIE_QB_DEBUT_FADE_COLUMN",
    "attach_qb_revenge_features",
    "attach_rookie_qb_debut_fade_features",
    "default_combine",
    "default_schedule",
    "default_weekly_rosters",
    "derive_qb_revenge_features",
    "derive_rookie_qb_debut_fade_features",
    "describe_rookie_qb_debut_population",
    "draft_team_by_gsis_id",
    "latest_combine_snapshot",
    "qb_revenge_join_diagnostics",
]
