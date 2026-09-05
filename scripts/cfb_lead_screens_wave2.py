"""CFB free-screen wave 2 (LEAD-47, LEAD-49; ``docs/cfb_lead_screens_wave2.md``).

Predeclared before any outcome is scored: two signed/flag columns, each its
own weak-signal family, on top of the frozen XLG-03 benchmark arm. Spends no
NFL evaluation window and no rotation window -- CFB is this project's
sanctioned free replication ground, the same one
``scripts/cfb_lead_screens_wave1.py`` used. All cells are recorded regardless
of sign; an interval crossing zero is never a rejection.

Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md): an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control -- the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is ``unresolved_below_power``:
record it with ``nfl-ats weak-signals record --league cfb``, report
``probability_positive``, never the binary "contains zero". If a record
command errors, the verdict is wrong, not the validator.

Leads:

* ``true_freshman_road_qb`` (LEAD-47): flag
  ``cfb_lead47_true_freshman_road_qb_flag`` -- 1 when the AWAY team's
  post-hoc starter (most dropbacks that game, via
  ``nfl_ats.cfb_qb_dependence.build_cfb_qb_game_metrics``, reused not
  rebuilt) is a true freshman by the "first local roster appearance season"
  proxy (the local ``experience_years`` field is measured to be a frozen,
  scrape-time-static artifact and is NOT used -- see the predeclaration doc),
  else 0. Population restricted to 2014-2019 + 2021-2025 (2012-2013 excluded:
  the stat-credited-only roster regime cannot distinguish true from redshirt
  freshmen). Home-side true freshmen are counted as a diagnostic but never
  folded into the flag or pooled with it.
* ``portal_qb_early`` (LEAD-49): signed
  ``cfb_lead49_portal_qb_early_signed`` -- +1 when the AWAY team's
  pregame-safe presumed starter (the leading passer of that team's most
  recently completed EARLIER game) is one of that team's transfer-portal QB
  arrivals this season AND this is one of that team's first three games of
  the season, -1 for the mirror, 0 otherwise/both. Population restricted to
  2021-2025 (CFBD portal data starts 2021). Portal entries are name-keyed
  only (no athlete ids); this script performs a disclosed, measured
  name+school+season match against the local roster archive, per CFBD's own
  identity contract (``nfl_ats.cfb.CFBD_PORTAL_IDENTITY_CONTRACT``).

See ``docs/cfb_lead_screens_wave2.md`` for the full predeclaration,
including the structural reason the pregame-safe LEAD-49 variant can never
flag a team's first game of a portal QB's tenure, and the measured
``experience_years`` data-quality finding behind LEAD-47's proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS, load_cfb_seasons
from nfl_ats.cfb_qb_dependence import build_cfb_qb_game_metrics
from nfl_ats.cfb_rest_bye_feature import default_cfb_schedules
from nfl_ats.clv import pick_correct, week_blocked_bootstrap
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
DEFAULT_CFB_ROOT = REPO_ROOT / "data" / "cfb"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "cfb_lead_screens_wave2"
BOOTSTRAP_SAMPLES = 1_000
SEED = 20260905
PERMUTATIONS = 200

# LEAD-47 population: excludes 2012-2013 (stat-credited-only roster regime;
# see docs/cfb_lead_screens_wave2.md for the measured finding).
LEAD47_SEASONS: tuple[int, ...] = tuple(range(2014, 2020)) + tuple(range(2021, 2026))
LEAD47_ERAS: tuple[tuple[str, int, int], ...] = (
    ("2014_2019", 2014, 2019),
    ("2021_2025", 2021, 2025),
)
LEAD47_ROSTER_ARCHIVE_START = 2004  # full local roster archive floor

# LEAD-49 population: CFBD portal data starts 2021.
LEAD49_SEASONS: tuple[int, ...] = tuple(range(2021, 2026))
LEAD49_ERAS: tuple[tuple[str, int, int], ...] = ()  # single regime; no split

CANDIDATE_COLUMNS: dict[str, str] = {
    "true_freshman_road_qb": "cfb_lead47_true_freshman_road_qb_flag",
    "portal_qb_early": "cfb_lead49_portal_qb_early_signed",
}

PREDICTED_DIRECTION: dict[str, str] = {
    "true_freshman_road_qb": (
        "fade the away team when its starter is a true freshman (unsigned flag; favours home)"
    ),
    "portal_qb_early": (
        "fade the team starting an early-tenure transfer-portal QB "
        "(signed column; +1 favours home when away qualifies, -1 mirror)"
    ),
}

LEAD_CONFIG: dict[str, dict[str, Any]] = {
    "true_freshman_road_qb": {"scored_seasons": LEAD47_SEASONS, "eras": LEAD47_ERAS},
    "portal_qb_early": {"scored_seasons": LEAD49_SEASONS, "eras": LEAD49_ERAS},
}

# Columns build_cfb_qb_game_metrics' internal cfb_competitive_plays() needs
# (nfl_ats.cfb_features._PBP_LOAD_COLUMNS) plus the passer identity column
# that function itself requires. Declared locally rather than importing a
# private module symbol, matching wave 1's self-contained style.
PBP_QB_LOAD_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "seasonType",
    "pos_team_id",
    "homeTeamId",
    "awayTeamId",
    "is_home",
    "EPA",
    "EPA_success",
    "rush",
    "pass",
    "kneel_down",
    "statYardage",
    "home_wp_before",
    "away_wp_before",
    "passer_player_id",
)


def _game_id_key(values: pd.Series) -> pd.Series:
    """Canonical string game-id key, regardless of the source column's dtype."""

    return pd.to_numeric(values, errors="raise").astype("int64").astype(str)


def _normalize_name(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


# ---------------------------------------------------------------------------
# Shared starter-identification (reused by both leads; see module docstring)
# ---------------------------------------------------------------------------


def leading_passer_per_game_team(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team_id): the post-hoc leading passer that game.

    Reuses ``nfl_ats.cfb_qb_dependence.build_cfb_qb_game_metrics`` (the
    project's already-built, already-tested CFB QB-identity work) unchanged,
    then takes the max-dropback row per (game_id, team_id), ties broken
    deterministically (highest dropbacks, then lowest passer_player_id).

    ``build_cfb_qb_game_metrics`` casts ``passer_player_id`` from pbp's own
    float64 column straight to ``str`` (e.g. ``"4775196.0"``), which is
    internally consistent for that module's own downstream joins but
    **measured** (this session) to break a cross-source join against
    roster/portal ``athlete_id`` values (clean int64, e.g. ``"4775196"``) --
    a real 0-of-294 overlap on a 2024 sample before this normalization was
    added. Re-cast through ``int64`` here, once, so every caller of this
    function gets a join-safe identity key.
    """

    qb_games = build_cfb_qb_game_metrics(pbp)
    if qb_games.empty:
        return qb_games
    qb_games = qb_games.copy()
    qb_games["passer_player_id"] = (
        pd.to_numeric(qb_games["passer_player_id"], errors="raise").astype("int64").astype(str)
    )
    ordered = qb_games.sort_values(
        ["game_id", "team_id", "qb_dropbacks", "passer_player_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    leading = ordered.groupby(["game_id", "team_id"], sort=False, as_index=False).first()
    return leading.reset_index(drop=True)


def attach_previous_game_starter(leading: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Add ``prev_game_starter_id``: the STRICTLY earlier previous identified
    starter for this (game's) team, chronological, leak-safe by construction.

    Mirrors ``cfb_qb_dependence.attach_cfb_qb_dependence``'s own
    ``latest_passer`` lookback loop, applied to raw starter identity.
    """

    if leading.empty:
        return leading.assign(gameday=pd.Series(dtype="datetime64[ns]"), prev_game_starter_id=None)

    sched = schedules.loc[:, ["game_id", "start_date"]].copy()
    sched["game_id"] = _game_id_key(sched["game_id"])
    sched["gameday"] = pd.to_datetime(sched["start_date"], errors="coerce")
    sched = sched.dropna(subset=["gameday"]).drop_duplicates("game_id")

    frame = leading.copy()
    frame["game_id"] = _game_id_key(frame["game_id"])
    frame = frame.merge(sched[["game_id", "gameday"]], on="game_id", how="left")
    frame = frame.loc[frame["gameday"].notna()].copy()
    frame = frame.sort_values(["gameday", "game_id", "team_id"], kind="mergesort").reset_index(
        drop=True
    )

    previous: dict[int, str] = {}
    prev_ids: list[Any] = []
    for team_id, passer_id in zip(frame["team_id"], frame["passer_player_id"], strict=True):
        key = int(team_id)
        prev_ids.append(previous.get(key))
        previous[key] = passer_id
    frame["prev_game_starter_id"] = prev_ids
    return frame


def build_team_starter_timeline(
    leading: pd.DataFrame, schedules: pd.DataFrame
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per team, a strictly-chronological ``(dates, passer_ids)`` timeline for
    point-in-time "who started most recently before date D" lookups.

    Unlike :func:`attach_previous_game_starter` (which only ever answers "the
    previous starter" for a game that ITSELF has an identified starter), this
    can be queried against ANY game's own kickoff date -- including a scored
    game with no pbp-identified starter of its own -- via
    :func:`lookup_previous_starter`, the same ``np.searchsorted`` point-in-time
    pattern ``cfb_qb_dependence.attach_cfb_qb_dependence`` uses for its state
    lookups.
    """

    if leading.empty:
        return {}
    sched = schedules.loc[:, ["game_id", "start_date"]].copy()
    sched["game_id"] = _game_id_key(sched["game_id"])
    sched["gameday"] = pd.to_datetime(sched["start_date"], errors="coerce")
    sched = sched.dropna(subset=["gameday"]).drop_duplicates("game_id")

    frame = leading.copy()
    frame["game_id"] = _game_id_key(frame["game_id"])
    frame = frame.merge(sched[["game_id", "gameday"]], on="game_id", how="left")
    frame = frame.loc[frame["gameday"].notna()].copy()
    frame = frame.sort_values(["team_id", "gameday", "game_id"], kind="mergesort")

    timelines: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for team_id, group in frame.groupby("team_id", sort=False):
        dates = group["gameday"].to_numpy(dtype="datetime64[ns]")
        passers = group["passer_player_id"].to_numpy()
        timelines[int(team_id)] = (dates, passers)
    return timelines


def lookup_previous_starter(
    timeline: dict[int, tuple[np.ndarray, np.ndarray]], team_id: int, gameday: Any
) -> Any:
    """The most recent identified starter for ``team_id`` STRICTLY before
    ``gameday``, or ``None`` if no earlier identified game exists."""

    entry = timeline.get(int(team_id))
    if entry is None:
        return None
    dates, passers = entry
    position = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(gameday), "ns"), side="left"))
    position -= 1
    if position < 0:
        return None
    return passers[position]


def starter_agreement_rate(walked: pd.DataFrame) -> dict[str, Any]:
    """Diagnostic: how often the pregame proxy matches the same-game post-hoc
    identity, over rows where both are defined."""

    known = walked.loc[walked["prev_game_starter_id"].notna()]
    if known.empty:
        return {"n_comparable": 0, "agreement_rate": float("nan")}
    agree = (known["passer_player_id"] == known["prev_game_starter_id"]).mean()
    return {"n_comparable": len(known), "agreement_rate": float(agree)}


# ---------------------------------------------------------------------------
# LEAD-47: true-freshman road QB
# ---------------------------------------------------------------------------


def attach_true_freshman_road_qb_flag(
    features: pd.DataFrame, *, pbp: pd.DataFrame, rosters: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = features.copy()
    frame["_game_id_key"] = _game_id_key(frame["game_id"])
    frame["_season_int"] = pd.to_numeric(frame["season"], errors="raise").astype("int64")

    leading = leading_passer_per_game_team(pbp)
    leading_key = leading.copy()
    if not leading_key.empty:
        leading_key["game_id"] = _game_id_key(leading_key["game_id"])
        leading_key["team_id"] = pd.to_numeric(leading_key["team_id"], errors="raise").astype(
            "int64"
        )

    roster_seasons = rosters.copy()
    roster_seasons["athlete_id"] = (
        pd.to_numeric(roster_seasons["athlete_id"], errors="raise").astype("int64").astype(str)
    )
    roster_seasons["season"] = pd.to_numeric(roster_seasons["season"], errors="raise").astype(
        "int64"
    )
    first_season_by_athlete = roster_seasons.groupby("athlete_id")["season"].min()

    diagnostics: dict[str, Any] = {}
    for side in ("home", "away"):
        side_id_col = f"{side}_id"
        side_ids = pd.to_numeric(frame[side_id_col], errors="raise").astype("int64")
        if leading_key.empty:
            starter_ids = pd.Series([None] * len(frame), index=frame.index)
        else:
            key_frame = pd.DataFrame(
                {"game_id": frame["_game_id_key"].to_numpy(), "team_id": side_ids.to_numpy()}
            )
            joined = key_frame.merge(
                leading_key[["game_id", "team_id", "passer_player_id"]],
                on=["game_id", "team_id"],
                how="left",
            )
            starter_ids = joined["passer_player_id"]
            starter_ids.index = frame.index
        first_season = starter_ids.map(first_season_by_athlete)
        identified = starter_ids.notna()
        is_true_fr = identified & (first_season == frame["_season_int"])
        frame[f"_lead47_{side}_true_freshman_starter"] = is_true_fr.fillna(False).to_numpy()
        diagnostics[f"{side}_starter_identified"] = int(identified.sum())
        diagnostics[f"{side}_true_freshman_starter_count"] = int(is_true_fr.fillna(False).sum())

    frame[CANDIDATE_COLUMNS["true_freshman_road_qb"]] = np.where(
        frame["_lead47_away_true_freshman_starter"], 1.0, 0.0
    )
    frame = frame.drop(columns=["_game_id_key", "_season_int"])
    diagnostics["n_games"] = len(frame)
    return frame, diagnostics


# ---------------------------------------------------------------------------
# LEAD-49: portal-QB early starts
# ---------------------------------------------------------------------------


def resolve_team_name_map(schedules: pd.DataFrame) -> pd.Series:
    """``team_name -> team_id`` using the same naming convention the
    benchmark table's own ``home_team``/``away_team`` columns carry."""

    home = schedules[["home_id", "home_team"]].rename(
        columns={"home_id": "team_id", "home_team": "team_name"}
    )
    away = schedules[["away_id", "away_team"]].rename(
        columns={"away_id": "team_id", "away_team": "team_name"}
    )
    combined = pd.concat([home, away], ignore_index=True).dropna()
    combined["team_id"] = pd.to_numeric(combined["team_id"], errors="raise").astype("int64")
    combined["team_name"] = combined["team_name"].astype("string")
    combined = combined.drop_duplicates(subset=["team_name"], keep="first")
    return combined.set_index("team_name")["team_id"]


def match_portal_qbs_to_athletes(
    portal: pd.DataFrame, rosters: pd.DataFrame, schedules: pd.DataFrame
) -> tuple[set[tuple[int, int, str]], dict[str, Any]]:
    """Disclosed name+school+season match, per CFBD_PORTAL_IDENTITY_CONTRACT.

    Returns the matched (team_id, season, athlete_id) set plus a diagnostics
    dict counting every stage's exclusions.
    """

    diagnostics: dict[str, Any] = {}
    name_to_id = resolve_team_name_map(schedules)

    portal_qb = portal.loc[
        portal["position"].astype("string").eq("QB") & portal["destination"].notna()
    ].copy()
    diagnostics["portal_qb_rows"] = len(portal_qb)

    portal_qb["destination_team_id"] = portal_qb["destination"].astype("string").map(name_to_id)
    unresolved_destination = int(portal_qb["destination_team_id"].isna().sum())
    diagnostics["unresolved_destination_rows"] = unresolved_destination
    resolved = portal_qb.dropna(subset=["destination_team_id"]).copy()
    resolved["destination_team_id"] = resolved["destination_team_id"].astype("int64")
    resolved["season"] = pd.to_numeric(resolved["season"], errors="raise").astype("int64")
    resolved["_first_norm"] = _normalize_name(resolved["firstName"])
    resolved["_last_norm"] = _normalize_name(resolved["lastName"])

    roster_names = rosters.copy()
    roster_names["athlete_id"] = (
        pd.to_numeric(roster_names["athlete_id"], errors="raise").astype("int64").astype(str)
    )
    roster_names["team_id"] = pd.to_numeric(roster_names["team_id"], errors="raise").astype("int64")
    roster_names["season"] = pd.to_numeric(roster_names["season"], errors="raise").astype("int64")
    roster_names["_first_norm"] = _normalize_name(roster_names["first_name"])
    roster_names["_last_norm"] = _normalize_name(roster_names["last_name"])
    roster_names = roster_names.dropna(subset=["_first_norm", "_last_norm"])

    key_cols = ["team_id", "season", "_first_norm", "_last_norm"]
    distinct_athletes = roster_names.groupby(key_cols)["athlete_id"].nunique()
    ambiguous_keys = int((distinct_athletes > 1).sum())
    diagnostics["ambiguous_roster_name_keys"] = ambiguous_keys
    unambiguous_keys = distinct_athletes.loc[distinct_athletes.eq(1)].index
    roster_lookup = (
        roster_names.sort_values("athlete_id")
        .drop_duplicates(subset=key_cols, keep="first")
        .set_index(key_cols)["athlete_id"]
    )
    # Ambiguous (team, season, name) keys are counted and EXCLUDED, never
    # guessed -- per CFBD_PORTAL_IDENTITY_CONTRACT's "must never silently
    # join on names".
    roster_lookup = roster_lookup.loc[roster_lookup.index.isin(unambiguous_keys)]

    matched_ids: list[Any] = []
    for team_id, season, first_n, last_n in zip(
        resolved["destination_team_id"],
        resolved["season"],
        resolved["_first_norm"],
        resolved["_last_norm"],
        strict=True,
    ):
        matched_ids.append(roster_lookup.get((int(team_id), int(season), first_n, last_n)))
    resolved["athlete_id"] = matched_ids
    matched = resolved.dropna(subset=["athlete_id"]).copy()
    diagnostics["unmatched_name_rows"] = int(len(resolved) - len(matched))
    diagnostics["matched_portal_qb_rows"] = len(matched)

    portal_qb_ids = {
        (int(team_id), int(season), str(athlete_id))
        for team_id, season, athlete_id in zip(
            matched["destination_team_id"], matched["season"], matched["athlete_id"], strict=True
        )
    }
    diagnostics["distinct_portal_qb_team_seasons"] = len({(t, s) for t, s, _ in portal_qb_ids})
    return portal_qb_ids, diagnostics


def build_season_game_index(schedules: pd.DataFrame) -> pd.Series:
    """``(team_id, season, game_id_key) -> 1-based game index that season``."""

    reg_complete = schedules.loc[
        schedules["season_type"].eq("regular") & schedules["completed"].eq(True)
    ]
    home = reg_complete[["home_id", "season", "game_id", "start_date"]].rename(
        columns={"home_id": "team_id"}
    )
    away = reg_complete[["away_id", "season", "game_id", "start_date"]].rename(
        columns={"away_id": "team_id"}
    )
    long_games = pd.concat([home, away], ignore_index=True).dropna(subset=["team_id"])
    long_games["team_id"] = pd.to_numeric(long_games["team_id"], errors="raise").astype("int64")
    long_games["season"] = pd.to_numeric(long_games["season"], errors="raise").astype("int64")
    long_games["start_date"] = pd.to_datetime(long_games["start_date"], errors="raise")
    long_games["game_id_key"] = _game_id_key(long_games["game_id"])
    long_games = long_games.sort_values(
        ["team_id", "season", "start_date", "game_id_key"], kind="mergesort"
    )
    long_games["season_game_index"] = (
        long_games.groupby(["team_id", "season"], sort=False).cumcount() + 1
    )
    return long_games.set_index(["team_id", "season", "game_id_key"])["season_game_index"]


def attach_portal_qb_early_flag(
    features: pd.DataFrame,
    *,
    pbp: pd.DataFrame,
    portal: pd.DataFrame,
    rosters: pd.DataFrame,
    schedules: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = features.copy()
    frame["_game_id_key"] = _game_id_key(frame["game_id"])
    frame["_season_int"] = pd.to_numeric(frame["season"], errors="raise").astype("int64")

    portal_qb_ids, match_diagnostics = match_portal_qbs_to_athletes(portal, rosters, schedules)
    index_lookup = build_season_game_index(schedules)

    leading = leading_passer_per_game_team(pbp)
    walked = attach_previous_game_starter(leading, schedules)
    agreement = starter_agreement_rate(walked)
    timeline = build_team_starter_timeline(leading, schedules)

    fires: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {**match_diagnostics, "starter_agreement": agreement}
    for side in ("home", "away"):
        side_id_col = f"{side}_id"
        side_ids = pd.to_numeric(frame[side_id_col], errors="raise").astype("int64")
        game_index = pd.Series(
            [
                index_lookup.get((int(team), int(season), key), np.nan)
                for team, season, key in zip(
                    side_ids, frame["_season_int"], frame["_game_id_key"], strict=True
                )
            ],
            index=frame.index,
        )
        # Point-in-time lookup keyed on this game's OWN kickoff date, not on
        # whether this exact game happens to have an identified starter of
        # its own -- see build_team_starter_timeline's docstring.
        prev_starter = pd.Series(
            [
                lookup_previous_starter(timeline, int(team), gameday)
                for team, gameday in zip(side_ids, frame["gameday"], strict=True)
            ],
            index=frame.index,
        )
        is_portal_qb = pd.Series(
            [
                pd.notna(starter) and (int(team), int(season), str(starter)) in portal_qb_ids
                for team, season, starter in zip(
                    side_ids, frame["_season_int"], prev_starter, strict=True
                )
            ],
            index=frame.index,
        )
        early_tenure = game_index.le(3)
        side_fires = (early_tenure & is_portal_qb).to_numpy()
        fires[side] = side_fires
        frame[f"_lead49_{side}_portal_early_fires"] = side_fires
        frame[f"_lead49_{side}_season_game_index"] = game_index.to_numpy()
        diagnostics[f"{side}_flagged_games"] = int(side_fires.sum())
        diagnostics[f"{side}_starter_unidentified"] = int(prev_starter.isna().sum())

    home_fires, away_fires = fires["home"], fires["away"]
    signed = np.where(away_fires & ~home_fires, 1.0, np.where(home_fires & ~away_fires, -1.0, 0.0))
    frame[CANDIDATE_COLUMNS["portal_qb_early"]] = signed
    diagnostics["both_fire_games"] = int((home_fires & away_fires).sum())
    diagnostics["n_games"] = len(frame)
    frame = frame.drop(columns=["_game_id_key", "_season_int"])
    return frame, diagnostics


def attach_candidate(
    lead: str,
    features: pd.DataFrame,
    *,
    pbp: pd.DataFrame,
    rosters: pd.DataFrame,
    portal: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if lead == "true_freshman_road_qb":
        return attach_true_freshman_road_qb_flag(features, pbp=pbp, rosters=rosters)
    if lead == "portal_qb_early":
        assert portal is not None and schedules is not None
        return attach_portal_qb_early_flag(
            features, pbp=pbp, portal=portal, rosters=rosters, schedules=schedules
        )
    raise ValueError(f"no scoring attacher for lead {lead!r}")


# ---------------------------------------------------------------------------
# Shared walk-forward harness (mirrors scripts/cfb_lead_screens_wave1.py)
# ---------------------------------------------------------------------------


def run_walk_forward(
    attached: pd.DataFrame,
    scored_seasons: tuple[int, ...],
    candidate_column: str,
    *,
    leak_treatment: bool,
) -> pd.DataFrame:
    completed = attached.loc[
        pd.to_numeric(attached["result"], errors="coerce").notna()
        & pd.to_numeric(attached["ats_margin"], errors="coerce").notna()
    ].copy()
    candidate_source = completed
    if leak_treatment:
        candidate_source = completed.copy()
        candidate_source[candidate_column] = pd.to_numeric(
            candidate_source["ats_margin"], errors="coerce"
        )
    candidate_columns = (*CFB_MODEL_FEATURE_COLUMNS, candidate_column)
    scored = completed.loc[completed["season"].astype(int).isin(scored_seasons)]
    rows: list[dict[str, Any]] = []
    for (season_value, week_value), group in scored.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        baseline_training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(baseline_training) < CFB_BENCHMARK_MIN_TRAIN_GAMES:
            continue
        candidate_training = candidate_source.loc[candidate_source["gameday"].lt(cutoff)]
        baseline_model = fit_cfb_residual_model(
            baseline_training,
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            feature_columns=CFB_MODEL_FEATURE_COLUMNS,
        )
        candidate_model = fit_cfb_residual_model(
            candidate_training,
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            feature_columns=candidate_columns,
        )
        candidate_scoring = (
            group
            if not leak_treatment
            else group.assign(
                **{candidate_column: pd.to_numeric(group["ats_margin"], errors="coerce")}
            )
        )
        settle_margin = pd.to_numeric(group["result"], errors="coerce") - pd.to_numeric(
            group["spread_line"], errors="coerce"
        )
        baseline_probability = baseline_model.predict(group)["home_cover_probability"]
        candidate_probability = candidate_model.predict(candidate_scoring)["home_cover_probability"]
        for game_id, margin, base, cand, feature_value in zip(
            group["game_id"],
            settle_margin,
            baseline_probability,
            candidate_probability,
            group[candidate_column],
            strict=True,
        ):
            rows.append(
                {
                    "game_id": game_id,
                    "season": int(str(season_value)),
                    "week": int(str(week_value)),
                    "settle_margin": margin,
                    "baseline_probability": base,
                    "candidate_probability": cand,
                    "feature_value": feature_value,
                }
            )
    return pd.DataFrame(rows)


def grade(frame: pd.DataFrame, margins: pd.Series | None = None) -> pd.DataFrame:
    settle = frame["settle_margin"] if margins is None else margins
    graded = frame.copy()
    for arm, column in (
        ("baseline", "baseline_probability"),
        ("candidate", "candidate_probability"),
    ):
        graded[f"{arm}_correct"] = pick_correct(graded[column].ge(0.5), settle)
    return graded


def _paired_metric(reference: str, candidate: str) -> Any:
    def metric(df: pd.DataFrame) -> dict[str, float]:
        valid = df.dropna(subset=[reference, candidate])
        if valid.empty:
            return {
                "delta_accuracy": float("nan"),
                "candidate_accuracy": float("nan"),
                "reference_accuracy": float("nan"),
            }
        return {
            "delta_accuracy": float((valid[candidate] - valid[reference]).mean()),
            "candidate_accuracy": float(valid[candidate].mean()),
            "reference_accuracy": float(valid[reference].mean()),
        }

    return metric


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def null_distribution(frame: pd.DataFrame, *, permutations: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metric = _paired_metric("baseline_correct", "candidate_correct")
    groups = week_positions(frame)
    deltas = []
    for _ in range(permutations):
        values = frame["settle_margin"].to_numpy(dtype=float, copy=True)
        for positions in groups:
            values[positions] = rng.permutation(values[positions])
        permuted = pd.Series(values, index=frame.index)
        deltas.append(metric(grade(frame, permuted))["delta_accuracy"])
    values_arr = np.asarray(deltas, dtype=float)
    finite = values_arr[np.isfinite(values_arr)]
    observed = metric(grade(frame))["delta_accuracy"]
    return {
        "permutations": len(finite),
        "null_mean_delta": float(finite.mean()),
        "null_sd_delta": float(finite.std(ddof=1)),
        "null_q025": float(np.quantile(finite, 0.025)),
        "null_q975": float(np.quantile(finite, 0.975)),
        "observed_delta": float(observed),
        "fraction_of_null_below_observed": float((finite < observed).mean()),
    }


def summarize_pair(paired: pd.DataFrame, samples: int, seed: int) -> dict[str, Any] | None:
    if paired.empty or paired.dropna(subset=["baseline_correct", "candidate_correct"]).empty:
        return None
    metric = _paired_metric("baseline_correct", "candidate_correct")
    point = metric(paired)
    week = week_blocked_bootstrap(paired, metric, block="week", samples=samples, seed=seed)
    week_row = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    feature_value = pd.to_numeric(paired["feature_value"], errors="coerce")
    summary: dict[str, Any] = {
        "delta_accuracy": point["delta_accuracy"],
        "candidate_accuracy": point["candidate_accuracy"],
        "baseline_accuracy": point["reference_accuracy"],
        "week_blocked_ci95": [float(week_row["lower"]), float(week_row["upper"])],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "n_games": len(paired.dropna(subset=["baseline_correct", "candidate_correct"])),
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
        "n_flagged_nonzero": int((feature_value.notna() & feature_value.ne(0.0)).sum()),
    }
    if paired["season"].nunique() >= 2:
        season = week_blocked_bootstrap(paired, metric, block="season", samples=samples, seed=seed)
        season_row = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
        summary["season_blocked_ci95"] = [float(season_row["lower"]), float(season_row["upper"])]
        summary["season_blocked_probability_positive"] = float(season_row["probability_positive"])
    else:
        summary["season_blocked_ci95"] = None
        summary["season_blocked_probability_positive"] = None
    return summary


def _print_screen_summary(result: dict[str, Any], eras: tuple[tuple[str, int, int], ...]) -> None:
    pooled = result["candidate_vs_baseline_pooled"]
    low, high = pooled["week_blocked_ci95"]
    print(
        f"pooled: delta {pooled['delta_accuracy'] * 100:+.3f} pts  P+ "
        f"{pooled['week_blocked_probability_positive']:.3f}  week 95% "
        f"[{low * 100:+.3f}, {high * 100:+.3f}]  n={pooled['n_games']} games, "
        f"{pooled['n_weeks']} weeks, flagged={pooled['n_flagged_nonzero']}"
    )
    null = result["permutation_null"]
    print(
        f"null: mean {null['null_mean_delta'] * 100:+.3f}, observed at the "
        f"{null['fraction_of_null_below_observed'] * 100:.1f}th percentile"
    )
    for label, _start, _end in eras:
        era = result["era_results"][label]
        if era is None:
            print(f"era {label}: no scored games")
        else:
            elow, ehigh = era["week_blocked_ci95"]
            print(
                f"era {label}: delta {era['delta_accuracy'] * 100:+.3f} pts  P+ "
                f"{era['week_blocked_probability_positive']:.3f}  "
                f"[{elow * 100:+.3f}, {ehigh * 100:+.3f}]"
            )


def _load_lead_inputs(
    lead: str, cfb_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Returns (pbp, rosters, portal_or_None, schedules_or_None)."""

    if lead == "true_freshman_road_qb":
        print("=== loading pbp (LEAD-47 seasons) ===", flush=True)
        pbp = load_cfb_seasons(
            cfb_root, "pbp", list(LEAD47_SEASONS), columns=list(PBP_QB_LOAD_COLUMNS)
        )
        print("=== loading full roster archive (first-appearance lookup) ===", flush=True)
        rosters = load_cfb_seasons(
            cfb_root,
            "rosters",
            list(range(LEAD47_ROSTER_ARCHIVE_START, 2026)),
            columns=["athlete_id", "season"],
        )
        return pbp, rosters, None, None

    print("=== loading pbp (LEAD-49 seasons) ===", flush=True)
    pbp = load_cfb_seasons(cfb_root, "pbp", list(LEAD49_SEASONS), columns=list(PBP_QB_LOAD_COLUMNS))
    print("=== loading rosters (LEAD-49 seasons, name match) ===", flush=True)
    rosters = load_cfb_seasons(
        cfb_root,
        "rosters",
        list(LEAD49_SEASONS),
        columns=["season", "team_id", "athlete_id", "first_name", "last_name"],
    )
    print("=== loading portal (LEAD-49 seasons) ===", flush=True)
    portal = load_cfb_seasons(
        cfb_root,
        "portal",
        list(LEAD49_SEASONS),
        columns=["season", "firstName", "lastName", "position", "origin", "destination"],
    )
    print("=== loading full local schedules snapshot ===", flush=True)
    schedules = default_cfb_schedules(cfb_root)
    return pbp, rosters, portal, schedules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lead", choices=("portal_qb_early", "true_freshman_road_qb"), required=True
    )
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--cfb-root", type=Path, default=DEFAULT_CFB_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    config = LEAD_CONFIG[args.lead]
    candidate_column = CANDIDATE_COLUMNS[args.lead]

    print(f"=== loading features (lead={args.lead}) ===", flush=True)
    features = pd.read_parquet(args.features)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")

    pbp, rosters, portal, schedules = _load_lead_inputs(args.lead, args.cfb_root)
    attached, diagnostics = attach_candidate(
        args.lead, features, pbp=pbp, rosters=rosters, portal=portal, schedules=schedules
    )
    attached = attached.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    print(f"diagnostics: {json.dumps(diagnostics, default=str)}", flush=True)

    fitted = run_walk_forward(
        attached,
        tuple(config["scored_seasons"]),
        candidate_column,
        leak_treatment=args.mode == "positive-control",
    )
    if fitted.empty:
        print("no scored games")
        return 1
    if args.mode == "null":
        null = null_distribution(fitted, permutations=args.permutations, seed=args.seed)
        print(json.dumps(null, indent=2))
        return 0

    graded = grade(fitted)
    eras: tuple[tuple[str, int, int], ...] = tuple(config["eras"])
    result: dict[str, Any] = {
        "status": "scored",
        "lead": args.lead,
        "candidate_column": candidate_column,
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "seed": args.seed,
        "candidate_vs_baseline_pooled": summarize_pair(
            graded, samples=args.bootstrap_samples, seed=args.seed
        ),
        "permutation_null": null_distribution(
            fitted, permutations=args.permutations, seed=args.seed
        ),
        "era_results": {
            label: (
                summarize_pair(
                    graded.loc[graded["season"].between(start, end)],
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            )
            for label, start, end in eras
        },
        "home_pick_rate": {
            "baseline": float(graded["baseline_probability"].ge(0.5).mean()),
            "candidate": float(graded["candidate_probability"].ge(0.5).mean()),
        },
        "diagnostics": diagnostics,
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = ARTIFACT_ROOT / args.lead / stamp
    configuration = {
        "cell": args.lead,
        "predicted_direction": PREDICTED_DIRECTION[args.lead],
        "mode": args.mode,
        "scored_seasons": list(config["scored_seasons"]),
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "baseline_feature_columns": list(CFB_MODEL_FEATURE_COLUMNS),
        "candidate_column": candidate_column,
        "regressor": "ridge",
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
        "min_train_games": CFB_BENCHMARK_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "predeclaration": "docs/cfb_lead_screens_wave2.md",
        "features_path": str(args.features),
    }
    payload = {
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    write_experiment_artifact(
        out_dir,
        "results.json",
        payload,
        command=f"cfb-lead-screens-wave2-{args.lead}",
        metrics={"cell": args.lead, "mode": args.mode, "status": result.get("status")},
        notes=(
            f"Free CFB screen wave 2, lead={args.lead}, on the frozen XLG-03 "
            "benchmark arm; no NFL window spent. See "
            "docs/cfb_lead_screens_wave2.md."
        ),
    )
    _print_screen_summary(result, eras)
    print(f"wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
