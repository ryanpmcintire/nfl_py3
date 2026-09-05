"""Per-player, per-game play/start probability -- a real forecast, not a base rate.

Owner directive (2026-09-05, verbatim): "the percentages should obviously make
sense my dude... it needs to be a forecast about the game and it needs to
consider depth chart." The lineup panel's per-player number was a
no-designation BASE RATE keyed only on (position_group, recent_role) --
``nfl_ats.lineup_availability``'s ``returning_contributor``/``no_recent_role``/
``unknown_no_history`` buckets (0.952 / 0.109 / 0.465). That base rate never
looks at depth chart at all, so a rookie QB2 with no injury designation read
47% (the ``unknown_no_history`` bucket) and a veteran healthy QB3 read 95%
(``returning_contributor``) -- exactly backwards from what "makes sense."

This module replaces that per-player number with a walk-forward, isotonic-
calibrated gradient-boosting model of two probabilities:

* ``played``  -- P(this player takes at least one snap this game).
* ``started`` -- P(this player fills a starting slot by playing time): the
  highest-snap players in each specific-position group, up to that group's
  slot count, ties broken by pregame depth rank and then player id. This is
  a playing-time proxy, not an official first-snap starter designation.

Features exclude gameday roster status. Timestamped injury revisions are
visible at or before the pool decision cutoff; daily depth observations must
be strictly earlier. Outcomes require team-game snap coverage.

For scored season S, train on seasons before S-1, calibrate on S-1, and
predict S. If either historical fold is absent, fit all seasons before S
without calibration and expose that fallback explicitly.

Depth-chart history (all positions, not only QB) is not archived anywhere
in this repository before this module: ``nfl_ats.quarterbacks``'s
``depth-ingest``/``depth-history-ingest`` filter to QB rows only
(``canonicalize_depth_charts``/``canonicalize_historical_depth_charts``,
both ``pos_abb``/``depth_position`` == "QB"). ``config/source_policies.json``
marks nflverse GREEN, so this module fetches and archives a full,
all-position depth-chart history via nflreadpy's own
``load_depth_charts`` -- the ONLY new network dependency this lane adds, run
once to build the training archive under
``data/players/raw/depth_charts/<stamp>/depth_charts.parquet`` (one directory
deeper than a bare ``data/players/raw/<stamp>/`` -- measured this session:
``nfl_ats.players.latest_player_snapshot`` globs ``data/players/raw/*/
manifest.json`` and assumes every match is a ``PlayerSnapshot`` manifest; a
depth-chart-history manifest living at that same depth broke it for every
other caller sharing this tree, including ``scripts/build_week_lineups.py``'s
``_no_designation_lookup``. Nesting one level further keeps the archive
under ``data/players/raw`` while staying outside that glob's reach, without
editing the shared, unowned ``players.py``).

**A schema seam, measured this session:** ``nfl.load_depth_charts`` returns
TWO different schemas depending on season. Seasons <= 2024 return legacy,
week-labelled rows (``season``, ``week``, ``club_code``, ``depth_team``,
``position``, ``depth_position``, ``formation``, one row per team per
week). Seasons >= 2025 return daily snapshot rows instead (``dt``, ``team``,
``pos_abb``, ``pos_rank``, no week label at all -- nflverse switched to
continuous point-in-time capture). ``canonicalize_depth_chart_history``
unifies both into one (season, week, team, gsis_id, position, position_group,
depth_rank) table: legacy rows keep their own week label directly ("the
depth chart's week used only for the game it describes" -- unlike
``nfl_ats.quarterbacks.canonicalize_historical_depth_charts``'s conservative
"strictly later games only" rule for the QB-only archive, which exists
because that module could not otherwise rule out looking at a
not-yet-finalized depth chart; a full weekly depth chart IS the team's own
pregame lineup announcement for that week's game, so using week W's chart
for week W's game is the correct, not the conservative, choice here); daily
rows have no week label, so each (season, week, team) is assigned the most
recent depth-chart snapshot observed strictly before that team's own
decision cutoff via ``pandas.merge_asof`` -- never a later one. The archive's own
``captured_at_utc`` (when THIS SESSION fetched it) is recorded in the
manifest, separately from the per-row week label that determines which
game a row may inform; the archive is a 2026 retrospective pull, and only
the week label -- never the fetch date -- gates what a training row may see.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.availability import practice_category, report_category
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.lineup_availability import depth_chart_position_group
from nfl_ats.nfl_week import pool_decision_cutoff
from nfl_ats.pbp import season_scope_mask
from nfl_ats.players import attach_snap_player_ids

PLAY_PROBABILITY_MODEL_VERSION = "v2-decision-safe-disjoint-calibration"
DEPTH_CHART_HISTORY_VERSION = "v1-legacy-week-and-daily-dt"

# ---------------------------------------------------------------------------
# Fixed category vocabularies. HistGradientBoostingClassifier's
# ``categorical_features="from_dtype"`` encodes a pandas "category" column by
# its ``.cat.categories`` array; a training frame and a one-row serving frame
# built independently could otherwise assign the SAME category string a
# DIFFERENT integer code (e.g. if a rare category is simply absent from one
# frame), silently corrupting predictions with no error raised. Every
# categorical feature is built through ``_categorical`` below using one of
# these FIXED, ordered tuples so training and serving always agree.
# ---------------------------------------------------------------------------
DEPTH_RANK_BUCKET_CATEGORIES: tuple[str, ...] = ("1", "2", "3+", "unknown")
POSITION_GROUP_CATEGORIES: tuple[str, ...] = (
    "offensive_line",
    "skill",
    "front",
    "secondary",
    "other",
)
REPORT_CATEGORY_CATEGORIES: tuple[str, ...] = (
    "out",
    "doubtful",
    "questionable",
    "probable",
    "none",
    "other",
)
PRACTICE_CATEGORY_CATEGORIES: tuple[str, ...] = ("out", "dnp", "limited", "full", "none", "other")
ROSTER_STATUS_CATEGORIES: tuple[str, ...] = ("ACT", "INA", "other", "unknown")
QB1_REPORT_CATEGORY_CATEGORIES: tuple[str, ...] = (*REPORT_CATEGORY_CATEGORIES, "not_applicable")
QB1_PRACTICE_CATEGORY_CATEGORIES: tuple[str, ...] = (
    *PRACTICE_CATEGORY_CATEGORIES,
    "not_applicable",
)
QB1_NOT_APPLICABLE = "not_applicable"

FEATURE_COLUMNS: tuple[str, ...] = (
    "depth_rank_bucket",
    "position_group",
    "report_category",
    "practice_category",
    "weeks_since_last_snap",
    "trailing4_snap_share",
    "season_week",
    "qb1_report_category",
    "qb1_practice_category",
)
CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "depth_rank_bucket",
    "position_group",
    "report_category",
    "practice_category",
    "qb1_report_category",
    "qb1_practice_category",
)
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "weeks_since_last_snap",
    "trailing4_snap_share",
    "season_week",
)
_CATEGORY_VOCABULARY: dict[str, tuple[str, ...]] = {
    "depth_rank_bucket": DEPTH_RANK_BUCKET_CATEGORIES,
    "position_group": POSITION_GROUP_CATEGORIES,
    "report_category": REPORT_CATEGORY_CATEGORIES,
    "practice_category": PRACTICE_CATEGORY_CATEGORIES,
    "roster_status": ROSTER_STATUS_CATEGORIES,
    "qb1_report_category": QB1_REPORT_CATEGORY_CATEGORIES,
    "qb1_practice_category": QB1_PRACTICE_CATEGORY_CATEGORIES,
}

LABEL_PLAYED = "played"
LABEL_STARTED = "started"

#: Specific-position buckets used ONLY for the "started" proxy label's own
#: competition for starting slots, ranked by recorded snap share.
#: Deliberately finer than ``POSITION_GROUP_CATEGORIES``: grouping every
#: offensive lineman together (5 simultaneous roles) or every "skill" player
#: together (QB pooled with RB/WR/TE/FB) makes a 50%-of-group threshold
#: unreachable for almost anyone. These buckets instead group only players
#: who genuinely compete for the SAME handful of simultaneous slots.
_START_SHARE_POSITION_BUCKETS: dict[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "FB",
    "WR": "WR",
    "TE": "TE",
    "T": "T",
    "OT": "T",
    "G": "G",
    "OG": "G",
    "C": "C",
    "OL": "OL",
    "DE": "DE",
    "EDGE": "DE",
    "DT": "DT",
    "DL": "DL",
    "NT": "NT",
    "LB": "LB",
    "ILB": "LB",
    "OLB": "LB",
    "MLB": "LB",
    "CB": "CB",
    "DB": "CB",
    "S": "S",
    "FS": "S",
    "SS": "S",
    "SAF": "S",
    "K": "K",
    "P": "P",
    "LS": "LS",
}


def _started_share_bucket(position: object) -> str:
    normalized = str(position).strip().upper()
    return _START_SHARE_POSITION_BUCKETS.get(normalized, normalized)


def depth_rank_bucket(rank: object) -> str:
    """Bucket a numeric depth-chart rank into ``"1"`` / ``"2"`` / ``"3+"``."""

    numeric = pd.to_numeric(pd.Series([rank]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "unknown"
    value = int(numeric)
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def _categorical(values: Iterable[object], column: str) -> pd.Categorical[str]:
    vocabulary = _CATEGORY_VOCABULARY[column]
    text = pd.Series(list(values), dtype="object").astype("string").fillna(vocabulary[-1])
    text = text.where(
        text.isin(vocabulary), vocabulary[-1] if column != "roster_status" else "other"
    )
    return pd.Categorical(text, categories=list(vocabulary))


def _ordinal(season: pd.Series, week: pd.Series) -> pd.Series:
    """A monotonic within/across-season week key: ``season * 100 + week``.

    Only used for chronological ORDERING and gap arithmetic ("weeks since
    last snap"), never as a real calendar distance -- the gap between week
    17/18 of one season and week 1 of the next overstates the true calendar
    gap (it counts the offseason as if it were consecutive weeks). Documented
    simplification: a real calendar-week distance would need each season's
    actual week count, which varies (16 vs 17 vs 18) across the training
    window, for no benefit to a tree model that only needs monotonic,
    consistent spacing to split on.
    """

    return pd.to_numeric(season, errors="coerce") * 100 + pd.to_numeric(week, errors="coerce")


# ---------------------------------------------------------------------------
# Depth-chart history archive (all positions; QB-only archives already
# exist under nfl_ats.quarterbacks and are untouched by this module).
# ---------------------------------------------------------------------------

LEGACY_DEPTH_HISTORY_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "club_code",
    "game_type",
    "depth_team",
    "position",
    "formation",
    "depth_position",
    "gsis_id",
    "full_name",
)
DAILY_DEPTH_HISTORY_COLUMNS: tuple[str, ...] = (
    "dt",
    "team",
    "player_name",
    "gsis_id",
    "pos_abb",
    "pos_rank",
)
DEPTH_CHART_HISTORY_OUTPUT_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "team",
    "gsis_id",
    "player_name",
    "position",
    "position_group",
    "depth_rank",
    "source_schema",
)
#: Specialist Special-Teams positions with no Offense/Defense counterpart --
#: kept even though their only rows are ``formation == "Special Teams"``
#: (legacy) / a specialist ``pos_abb`` (daily). Every other Special Teams row
#: (KR/PR/H for a player whose primary role is offense or defense) is a
#: SECOND listing for a player already captured via his primary-role row and
#: is dropped -- keeping it would let a WR's return-only rank stand in for
#: his real receiver depth rank. See ``_dedupe_primary_role``.
_SPECIALIST_POSITIONS = frozenset(("K", "P", "LS", "PK"))


@dataclass(frozen=True)
class DepthChartHistorySnapshot:
    """An immutable, all-position, point-in-time depth-chart archive."""

    snapshot_id: str
    root: Path
    requested_seasons: tuple[int, ...]

    @property
    def data_path(self) -> Path:
        return self.root / "depth_charts.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        converted = frame.to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted
    raise TypeError(f"Unsupported dataframe type: {type(frame)!r}")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schedule_kickoff_utc(schedules: pd.DataFrame) -> pd.Series:
    """Combine nflverse ``gameday`` + Eastern ``gametime`` into UTC.

    Duplicated (not imported) from ``nfl_ats.players._schedule_kickoff_utc`` --
    the same cross-module duplication convention every copy of this helper
    already follows in this repository.
    """

    if "gametime" not in schedules:
        return pd.Series(pd.NaT, index=schedules.index, dtype="datetime64[ns, UTC]")
    date_text = pd.to_datetime(schedules["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = schedules["gametime"].astype("string")
    local = pd.to_datetime(date_text + " " + time_text, errors="coerce")
    return local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def _team_week_kickoffs(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with that team's own REG kickoff."""

    require_columns(schedule, ("season", "week", "home_team", "away_team", "game_type"), "schedule")
    frame = schedule.copy()
    frame = frame.loc[
        season_scope_mask(
            frame["game_type"], include_postseason=False, dataset="schedule", column="game_type"
        )
    ].copy()
    frame["kickoff"] = _schedule_kickoff_utc(frame)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    long = pd.concat(
        [
            frame.rename(columns={"home_team": "team"})[["season", "week", "team", "kickoff"]],
            frame.rename(columns={"away_team": "team"})[["season", "week", "team", "kickoff"]],
        ],
        ignore_index=True,
    )
    long = long.loc[long["season"].notna() & long["week"].notna() & long["kickoff"].notna()].copy()
    long["season"] = long["season"].astype(int)
    long["week"] = long["week"].astype(int)
    long["team"] = long["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    return long.drop_duplicates(["season", "week", "team"]).reset_index(drop=True)


def _dedupe_primary_role(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, gsis_id): the player's PRIMARY role.

    A return specialist can carry a second Special-Teams-only row (KR/PR/H)
    alongside his real offensive or defensive depth-chart row; keeping both
    would let the return-specialist rank stand in for his real position
    rank. Ties are broken toward a non-"other" ``position_group`` first,
    then the best (lowest) ``depth_rank``.
    """

    priority = frame["position_group"].ne("other").astype(int)
    ordered = frame.assign(_priority=priority).sort_values(
        ["season", "week", "team", "gsis_id", "_priority", "depth_rank"],
        ascending=[True, True, True, True, False, True],
    )
    return (
        ordered.drop_duplicates(["season", "week", "team", "gsis_id"], keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def _canonicalize_legacy_depth_history(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, LEGACY_DEPTH_HISTORY_COLUMNS, "legacy depth charts")
    result = frame.loc[:, list(LEGACY_DEPTH_HISTORY_COLUMNS)].copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["week"] = pd.to_numeric(result["week"], errors="coerce")
    result = result.loc[result["season"].notna() & result["week"].notna()].copy()
    result = result.loc[
        season_scope_mask(
            result["game_type"],
            include_postseason=False,
            dataset="depth_charts",
            column="game_type",
        )
    ].copy()
    keep_specialist = result["formation"].astype("string").eq("Special Teams") & result[
        "position"
    ].astype("string").str.upper().isin(_SPECIALIST_POSITIONS)
    keep_primary = result["formation"].astype("string").isin(("Offense", "Defense"))
    result = result.loc[keep_primary | keep_specialist].copy()
    result["team"] = result["club_code"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    result["gsis_id"] = result["gsis_id"].astype("string")
    result["player_name"] = result["full_name"].astype("string")
    result["depth_rank"] = pd.to_numeric(result["depth_team"], errors="coerce")
    result["position"] = result["position"].astype("string").str.upper()
    result["position_group"] = result["position"].map(depth_chart_position_group)
    result = result.loc[
        result["team"].notna()
        & result["gsis_id"].notna()
        & result["depth_rank"].notna()
        & result["position"].notna()
    ].copy()
    result["season"] = result["season"].astype(int)
    result["week"] = result["week"].astype(int)
    result["depth_rank"] = result["depth_rank"].astype(int)
    result["source_schema"] = "legacy_week"
    return result[list(DEPTH_CHART_HISTORY_OUTPUT_COLUMNS)]


def _canonicalize_daily_depth_history(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, DAILY_DEPTH_HISTORY_COLUMNS, "daily depth charts")
    result = frame.loc[:, list(DAILY_DEPTH_HISTORY_COLUMNS)].copy()
    result["dt"] = pd.to_datetime(result["dt"], errors="coerce", utc=True)
    result["team"] = result["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    result = result.loc[result["dt"].notna() & result["team"].notna()].copy()
    if result.empty:
        return pd.DataFrame(columns=list(DEPTH_CHART_HISTORY_OUTPUT_COLUMNS))

    kickoffs = _team_week_kickoffs(schedule)
    snapshots = result[["team", "dt"]].drop_duplicates().sort_values("dt")
    kickoffs["decision_at"] = kickoffs["kickoff"].map(pool_decision_cutoff)
    kickoffs = kickoffs.sort_values("decision_at")
    matched = pd.merge_asof(
        kickoffs,
        snapshots.rename(columns={"dt": "effective_dt"}),
        left_on="decision_at",
        right_on="effective_dt",
        by="team",
        direction="backward",
        allow_exact_matches=False,
    )
    matched = matched.loc[
        matched["effective_dt"].notna(), ["season", "week", "team", "effective_dt", "decision_at"]
    ]
    if matched.empty:
        return pd.DataFrame(columns=list(DEPTH_CHART_HISTORY_OUTPUT_COLUMNS))

    merged = result.merge(
        matched, left_on=["team", "dt"], right_on=["team", "effective_dt"], how="inner"
    )
    merged["gsis_id"] = merged["gsis_id"].astype("string")
    merged["player_name"] = merged["player_name"].astype("string")
    merged["position"] = merged["pos_abb"].astype("string").str.upper()
    merged["position_group"] = merged["position"].map(depth_chart_position_group)
    merged["depth_rank"] = pd.to_numeric(merged["pos_rank"], errors="coerce")
    merged = merged.loc[
        merged["gsis_id"].notna() & merged["depth_rank"].notna() & merged["position"].notna()
    ].copy()
    merged["depth_rank"] = merged["depth_rank"].astype(int)
    merged["source_schema"] = "daily_dt"
    merged["depth_observed_at"] = merged["effective_dt"]
    return merged[[*DEPTH_CHART_HISTORY_OUTPUT_COLUMNS, "depth_observed_at", "decision_at"]]


def canonicalize_depth_chart_history(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Unify legacy week-labelled and daily dt-timestamped depth-chart rows.

    ``frame`` is the raw union nflreadpy's ``load_depth_charts`` returns when
    fetched across seasons spanning the 2025 schema change (all columns from
    both schemas present, half of them null on any given row depending on
    which schema produced it -- measured this session). ``schedule`` is the
    raw nflverse schedule (``season``, ``week``, ``home_team``, ``away_team``,
    ``game_type``, ``gameday``, ``gametime``), used only to assign a week to
    the schema-less daily rows.
    """

    pieces: list[pd.DataFrame] = []
    if "week" in frame.columns and "depth_team" in frame.columns:
        legacy_rows = frame.loc[frame["week"].notna() & frame["depth_team"].notna()]
        if not legacy_rows.empty:
            pieces.append(_canonicalize_legacy_depth_history(legacy_rows))
    if "dt" in frame.columns and "pos_rank" in frame.columns:
        daily_rows = frame.loc[frame["dt"].notna() & frame["pos_rank"].notna()]
        if not daily_rows.empty:
            pieces.append(_canonicalize_daily_depth_history(daily_rows, schedule))
    if not pieces:
        raise DataContractError("depth_charts contains neither legacy week rows nor daily dt rows")
    combined = pd.concat(pieces, ignore_index=True)
    combined = _dedupe_primary_role(combined)
    return combined.sort_values(["season", "week", "team", "depth_rank", "gsis_id"]).reset_index(
        drop=True
    )


def write_depth_chart_history_snapshot(
    frame: pd.DataFrame,
    schedule: pd.DataFrame,
    raw_root: Path,
    requested_seasons: list[int],
    snapshot_id: str | None = None,
) -> DepthChartHistorySnapshot:
    if not requested_seasons or requested_seasons != sorted(set(requested_seasons)):
        raise ValueError("Requested seasons must be non-empty, unique, and sorted")
    identifier = snapshot_id or run_id()
    destination = raw_root / identifier
    if destination.exists():
        raise FileExistsError(f"Depth-chart history snapshot already exists: {destination}")
    canonical = canonicalize_depth_chart_history(frame, schedule)
    if canonical.empty:
        raise DataContractError("Depth-chart history produced no rows for the requested seasons")
    snapshot = DepthChartHistorySnapshot(identifier, destination, tuple(requested_seasons))
    atomic_parquet(canonical, snapshot.data_path)
    manifest = {
        "snapshot_id": identifier,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source": "nflverse depth charts (all positions) via nflreadpy",
        "contract_version": DEPTH_CHART_HISTORY_VERSION,
        "requested_seasons": requested_seasons,
        "rows": len(canonical),
        "rows_by_source_schema": {
            str(schema): int(count)
            for schema, count in canonical["source_schema"].value_counts().items()
        },
        "seasons_covered": sorted(int(value) for value in canonical["season"].unique()),
        "sha256": _sha256(snapshot.data_path),
        "columns": canonical.columns.tolist(),
        "note": (
            "captured_at_utc is when THIS ARCHIVE was fetched, not a per-row "
            "observation time; each row's own season/week determines which "
            "game it may inform (see the module docstring)."
        ),
    }
    atomic_json(manifest, snapshot.manifest_path)
    return snapshot


def fetch_depth_chart_history_snapshot(
    seasons: list[int], raw_root: Path
) -> DepthChartHistorySnapshot:
    """Fetch and archive the FULL (all-position) depth-chart history.

    nflverse is GREEN in ``config/source_policies.json``. This is the ONLY
    network fetch this module performs; every training/evaluation function
    below reads only from the local archive it writes.
    """

    if not seasons or seasons != sorted(set(seasons)):
        raise ValueError("Seasons must be non-empty, unique, and sorted")
    import nflreadpy as nfl

    depth = _to_pandas(nfl.load_depth_charts(seasons=seasons))
    schedule = _to_pandas(nfl.load_schedules(seasons=seasons))
    return write_depth_chart_history_snapshot(depth, schedule, raw_root, seasons)


def depth_chart_history_snapshot_from_root(root: Path) -> DepthChartHistorySnapshot:
    import json

    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Depth-chart history manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return DepthChartHistorySnapshot(
        str(payload["snapshot_id"]),
        root,
        tuple(int(value) for value in payload["requested_seasons"]),
    )


def latest_depth_chart_history_snapshot(raw_root: Path) -> DepthChartHistorySnapshot:
    manifests = sorted(raw_root.glob("*/manifest.json"))
    matching = [
        manifest for manifest in manifests if (manifest.parent / "depth_charts.parquet").is_file()
    ]
    if not matching:
        raise FileNotFoundError(f"No depth-chart history snapshots found in {raw_root}")
    return depth_chart_history_snapshot_from_root(matching[-1].parent)


def load_depth_chart_history_snapshot(snapshot: DepthChartHistorySnapshot) -> pd.DataFrame:
    return pd.read_parquet(snapshot.data_path)


def _player_snap_history(rosters: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Each ``gsis_id``'s own chronological log of games it recorded a snap.

    One row per (gsis_id, season, week) the player actually played (any
    team), with ``trailing4_inclusive``: the mean "own side" snap share
    (``max(offense_pct, defense_pct, st_pct)``) over that game and up to the
    3 before it. Games the player did not dress for have no row at all
    (nflverse's snap_counts only lists players who recorded >=1 snap), so a
    gap between two consecutive rows for the same player already IS "weeks
    absent" -- byes and healthy scratches need no separate handling.
    """

    linked = attach_snap_player_ids(snaps, rosters)
    linked = linked.loc[linked["gsis_id"].notna()].copy()
    linked["total_snaps"] = sum(
        pd.to_numeric(linked[column], errors="coerce").fillna(0.0)
        for column in ("offense_snaps", "defense_snaps", "st_snaps")
    )
    linked = linked.loc[linked["total_snaps"].gt(0)].copy()
    for column in ("offense_pct", "defense_pct", "st_pct"):
        linked[column] = pd.to_numeric(linked[column], errors="coerce").fillna(0.0)
    linked["own_side_share"] = linked[["offense_pct", "defense_pct", "st_pct"]].max(axis=1)
    linked["ordinal"] = _ordinal(linked["season"], linked["week"])
    history = (
        linked.groupby(["gsis_id", "ordinal"], observed=True, as_index=False)
        .agg(own_side_share=("own_side_share", "max"))
        .sort_values(["gsis_id", "ordinal"])
        .reset_index(drop=True)
    )
    history["trailing4_inclusive"] = history.groupby("gsis_id", sort=False)[
        "own_side_share"
    ].transform(lambda series: series.rolling(4, min_periods=1).mean())
    return history[["gsis_id", "ordinal", "trailing4_inclusive"]]


def attach_history_features(
    population: pd.DataFrame, rosters: pd.DataFrame, snaps: pd.DataFrame
) -> pd.DataFrame:
    """Attach ``weeks_since_last_snap`` and ``trailing4_snap_share``.

    Both use ``pandas.merge_asof`` with ``allow_exact_matches=False``: only a
    STRICTLY EARLIER game (any team) this ``gsis_id`` played may inform
    either value -- the leakage-safety property the module docstring's test
    plan checks directly.
    """

    history = _player_snap_history(rosters, snaps)
    result = population.reset_index(drop=True).copy()
    result["_row"] = np.arange(len(result))
    result["ordinal"] = _ordinal(result["season"], result["week"])
    left = result[["_row", "gsis_id", "ordinal"]].copy()
    # merge_asof's `by=` requires an EXACT dtype match on both sides; the
    # population's gsis_id (parquet-loaded, pyarrow-backed "string") and
    # history's (produced by attach_snap_player_ids's dict .map(), numpy-
    # backed "string") are both "string" dtype but not the SAME string
    # dtype, and pandas refuses to merge on them -- cast both to plain
    # Python str first.
    left["gsis_id"] = left["gsis_id"].astype(str)
    left = left.sort_values("ordinal")
    right = history.rename(columns={"ordinal": "last_play_ordinal"}).copy()
    right["gsis_id"] = right["gsis_id"].astype(str)
    right = right.sort_values("last_play_ordinal")
    merged = pd.merge_asof(
        left,
        right,
        left_on="ordinal",
        right_on="last_play_ordinal",
        by="gsis_id",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("_row")
    merged = merged.reindex(result["_row"])
    result["weeks_since_last_snap"] = (
        result["ordinal"].to_numpy() - merged["last_play_ordinal"].to_numpy()
    )
    result["trailing4_snap_share"] = merged["trailing4_inclusive"].to_numpy()
    return result.drop(columns=["_row", "ordinal"])


def _injury_status_lookup(injuries: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "team"]
    columns = [*keys, "gsis_id", "report_status", "practice_status"]
    require_columns(injuries, columns, "injuries")
    working = injuries.merge(decisions[[*keys, "decision_at"]].drop_duplicates(), on=keys)
    sort_column = "effective_observed_at" if "effective_observed_at" in working else "date_modified"
    if sort_column not in working:
        return working.iloc[:0][columns]
    working[sort_column] = pd.to_datetime(working[sort_column], errors="coerce", utc=True)
    working = working.loc[working[sort_column].le(working["decision_at"])].sort_values(sort_column)
    return working.drop_duplicates([*keys, "gsis_id"], keep="last")[columns]


def _attach_injury_categories(
    population: pd.DataFrame, injury_lookup: pd.DataFrame
) -> pd.DataFrame:
    result = population.reset_index(drop=True).copy()
    result["_row"] = np.arange(len(result))
    merged = result.merge(injury_lookup, on=["season", "week", "team", "gsis_id"], how="left")
    merged = merged.drop_duplicates("_row").set_index("_row").reindex(result["_row"])
    result["has_injury_designation"] = (
        merged["report_status"].notna().to_numpy() | merged["practice_status"].notna().to_numpy()
    )
    result["report_status_raw"] = merged["report_status"].to_numpy()
    result["practice_status_raw"] = merged["practice_status"].to_numpy()
    result["report_category"] = [report_category(value) for value in result["report_status_raw"]]
    result["practice_category"] = [
        practice_category(value) for value in result["practice_status_raw"]
    ]
    return result.drop(columns="_row")


def _qb1_status_table(population: pd.DataFrame, injury_lookup: pd.DataFrame) -> pd.DataFrame:
    qb_rows = population.loc[population["position"].astype("string").str.upper().eq("QB")].copy()
    if qb_rows.empty:
        return pd.DataFrame(
            columns=["season", "week", "team", "qb1_report_category", "qb1_practice_category"]
        )
    qb_rows["_min_rank"] = qb_rows.groupby(["season", "week", "team"], observed=True)[
        "depth_rank"
    ].transform("min")
    qb1 = qb_rows.loc[qb_rows["depth_rank"].eq(qb_rows["_min_rank"])].sort_values(
        ["season", "week", "team", "gsis_id"]
    )
    qb1 = qb1.drop_duplicates(["season", "week", "team"], keep="first")[
        ["season", "week", "team", "gsis_id"]
    ].rename(columns={"gsis_id": "qb1_gsis_id"})
    qb1 = qb1.merge(
        injury_lookup,
        left_on=["season", "week", "team", "qb1_gsis_id"],
        right_on=["season", "week", "team", "gsis_id"],
        how="left",
    )
    qb1["qb1_report_category"] = [report_category(value) for value in qb1["report_status"]]
    qb1["qb1_practice_category"] = [practice_category(value) for value in qb1["practice_status"]]
    return qb1[["season", "week", "team", "qb1_report_category", "qb1_practice_category"]]


def _attach_qb1_status(population: pd.DataFrame, injury_lookup: pd.DataFrame) -> pd.DataFrame:
    qb1_table = _qb1_status_table(population, injury_lookup)
    result = population.reset_index(drop=True).copy()
    result["_row"] = np.arange(len(result))
    merged = result.merge(qb1_table, on=["season", "week", "team"], how="left")
    merged = merged.drop_duplicates("_row").set_index("_row").reindex(result["_row"])
    is_qb = result["position"].astype("string").str.upper().eq("QB")
    result["qb1_report_category"] = np.where(
        is_qb, merged["qb1_report_category"].fillna("none").to_numpy(), QB1_NOT_APPLICABLE
    )
    result["qb1_practice_category"] = np.where(
        is_qb, merged["qb1_practice_category"].fillna("none").to_numpy(), QB1_NOT_APPLICABLE
    )
    return result.drop(columns="_row")


def _played_label(
    population: pd.DataFrame, rosters: pd.DataFrame, snaps: pd.DataFrame
) -> np.ndarray:
    linked = attach_snap_player_ids(snaps, rosters)
    linked = linked.loc[linked["gsis_id"].notna()].copy()
    linked["total_snaps"] = sum(
        pd.to_numeric(linked[column], errors="coerce").fillna(0.0)
        for column in ("offense_snaps", "defense_snaps", "st_snaps")
    )
    played = (
        linked.groupby(["season", "week", "team", "gsis_id"], observed=True)["total_snaps"]
        .max()
        .gt(0)
        .rename("_played_outcome")
        .reset_index()
    )
    # ``population`` may already carry its own "played" column (this is
    # called a second time, from inside ``_started_label``, on a population
    # that already has LABEL_PLAYED assigned) -- select only the join keys
    # so that pre-existing column can never collide with the computed one.
    result = population[["season", "week", "team", "gsis_id"]].reset_index(drop=True).copy()
    result["_row"] = np.arange(len(result))
    merged = result.merge(played, on=["season", "week", "team", "gsis_id"], how="left")
    merged = merged.drop_duplicates("_row").set_index("_row").reindex(result["_row"])
    return merged["_played_outcome"].fillna(False).to_numpy(dtype=bool)


def _started_label(
    population: pd.DataFrame, rosters: pd.DataFrame, snaps: pd.DataFrame
) -> np.ndarray:
    linked = attach_snap_player_ids(snaps, rosters)
    linked = linked.loc[linked["gsis_id"].notna()].copy()
    linked["total_snaps"] = sum(
        pd.to_numeric(linked[column], errors="coerce").fillna(0.0)
        for column in ("offense_snaps", "defense_snaps", "st_snaps")
    )
    linked["bucket"] = linked["position"].map(_started_share_bucket)
    keys = ["season", "week", "team", "gsis_id"]
    groups = ["season", "week", "team", "bucket"]
    per_player = (
        linked.groupby([*groups, "gsis_id"], observed=True)["total_snaps"].max().reset_index()
    )
    ranks = population[[*keys, "depth_rank"]].drop_duplicates(keys)
    ranked = per_player.merge(ranks, on=keys, how="left").sort_values(
        ["total_snaps", "depth_rank", "gsis_id"], ascending=[False, True, True], na_position="last"
    )
    slots = {
        "QB": 1,
        "RB": 1,
        "FB": 1,
        "WR": 3,
        "TE": 1,
        "T": 2,
        "G": 2,
        "C": 1,
        "OL": 5,
        "DE": 2,
        "DT": 2,
        "DL": 4,
        "NT": 1,
        "LB": 3,
        "CB": 2,
        "S": 2,
        "K": 1,
        "P": 1,
        "LS": 1,
    }
    ranked["_started"] = ranked["total_snaps"].gt(0) & ranked.groupby(
        groups, observed=True
    ).cumcount().lt(ranked["bucket"].map(slots).fillna(1))
    outcomes = ranked.groupby(keys, observed=True)["_started"].any().reset_index()
    merged = population[keys].merge(outcomes, on=keys, how="left", sort=False)
    return merged["_started"].fillna(False).to_numpy(dtype=bool)


def build_player_week_panel(
    depth_history: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (season, week, team, gsis_id) depth-chart appearance with
    every strictly-pregame feature and the two postgame labels
    (``played``/``started``). The population is depth-chart rows -- the SAME
    population ``scripts/build_week_lineups.py`` scores every player from --
    not the broader weekly-roster population, so training and serving see
    the same distribution of players.
    """

    require_columns(depth_history, DEPTH_CHART_HISTORY_OUTPUT_COLUMNS, "depth_chart_history")
    population = depth_history.drop_duplicates(
        ["season", "week", "team", "gsis_id"], keep="first"
    ).reset_index(drop=True)
    if schedule is not None:
        decisions = _team_week_kickoffs(schedule)
        decisions["decision_at"] = decisions["kickoff"].map(pool_decision_cutoff)
        population = population.drop(columns="decision_at", errors="ignore").merge(
            decisions[["season", "week", "team", "decision_at"]],
            on=["season", "week", "team"],
            how="inner",
        )
    require_columns(population, ("decision_at",), "depth history or schedule")
    population["decision_at"] = pd.to_datetime(population["decision_at"], utc=True, errors="coerce")
    daily = population["source_schema"].eq("daily_dt")
    observed = pd.to_datetime(
        population.get("depth_observed_at", pd.Series(pd.NaT, index=population.index)), utc=True
    )
    population = population.loc[
        population["decision_at"].notna() & (~daily | observed.lt(population["decision_at"]))
    ].copy()
    coverage = snaps.loc[
        snaps[["offense_snaps", "defense_snaps", "st_snaps"]]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
        .gt(0),
        ["season", "week", "team"],
    ].drop_duplicates()
    population = population.merge(coverage, on=["season", "week", "team"], how="inner")
    population["depth_rank_bucket"] = population["depth_rank"].map(depth_rank_bucket)
    population["season_week"] = pd.to_numeric(population["week"], errors="coerce").astype(float)

    population = attach_history_features(population, rosters, snaps)

    injury_lookup = _injury_status_lookup(injuries, population)
    population = _attach_injury_categories(population, injury_lookup)
    population = _attach_qb1_status(population, injury_lookup)

    population[LABEL_PLAYED] = _played_label(population, rosters, snaps)
    population[LABEL_STARTED] = _started_label(population, rosters, snaps)
    return population


# ---------------------------------------------------------------------------
# Model: walk-forward, isotonic-calibrated HistGradientBoostingClassifier.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayProbabilityModel:
    """A fitted, calibrated pair of boosters (played, started) for one
    scored season, plus the provenance needed to describe how it was fit."""

    version: str
    scored_season: int
    train_seasons: tuple[int, ...]
    calibration_season: int | None
    calibration_status: str
    played_booster: Any
    played_calibrator: Any
    started_booster: Any
    started_calibrator: Any


def _prepare_matrix(features: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, Any] = {}
    for column in CATEGORICAL_FEATURE_COLUMNS:
        columns[column] = _categorical(features[column], column)
    for column in NUMERIC_FEATURE_COLUMNS:
        columns[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    return pd.DataFrame(columns, index=features.index)[list(FEATURE_COLUMNS)]


def _fit_one_label(
    train: pd.DataFrame, *, label: str, calibration: pd.DataFrame
) -> tuple[Any, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    design = _prepare_matrix(train)
    target = train[label].astype(int).to_numpy()
    booster = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        random_state=0,
        max_depth=6,
        learning_rate=0.08,
        max_iter=150,
    )
    booster.fit(design, target)
    if calibration.empty:
        return booster, None
    raw_calibration_predictions = booster.predict_proba(_prepare_matrix(calibration))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_calibration_predictions, calibration[label].astype(int).to_numpy())
    return booster, calibrator


def fit_play_probability_model(panel: pd.DataFrame, *, scored_season: int) -> PlayProbabilityModel:
    """Train before S-1, calibrate on S-1, score S; flag uncalibrated fallback."""

    require_columns(
        panel, (*FEATURE_COLUMNS, "season", LABEL_PLAYED, LABEL_STARTED), "player_week_panel"
    )
    train = panel.loc[panel["season"].lt(scored_season)].copy()
    if train.empty:
        raise DataContractError(f"No training seasons available strictly before {scored_season}")
    calibration = train.loc[train["season"].eq(scored_season - 1)]
    booster_train = train.loc[train["season"].lt(scored_season - 1)]
    calibration_season: int | None = scored_season - 1
    calibration_status = "held_out_previous_season"
    if calibration.empty or booster_train.empty:
        calibration = train.iloc[:0]
        calibration_season = None
        calibration_status = "uncalibrated_insufficient_history"
    else:
        train = booster_train
    played_booster, played_calibrator = _fit_one_label(
        train, label=LABEL_PLAYED, calibration=calibration
    )
    started_booster, started_calibrator = _fit_one_label(
        train, label=LABEL_STARTED, calibration=calibration
    )
    return PlayProbabilityModel(
        version=PLAY_PROBABILITY_MODEL_VERSION,
        scored_season=scored_season,
        train_seasons=tuple(sorted(int(value) for value in train["season"].unique())),
        calibration_season=calibration_season,
        calibration_status=calibration_status,
        played_booster=played_booster,
        played_calibrator=played_calibrator,
        started_booster=started_booster,
        started_calibrator=started_calibrator,
    )


def predict_play_probabilities(model: PlayProbabilityModel, features: pd.DataFrame) -> pd.DataFrame:
    """``play_probability`` (``played``) and ``start_probability`` (``started``),
    both isotonic-calibrated, for every row in ``features``."""

    design = _prepare_matrix(features)
    played_raw = model.played_booster.predict_proba(design)[:, 1]
    played = np.clip(
        (
            model.played_calibrator.predict(played_raw)
            if model.played_calibrator is not None
            else played_raw
        ),
        0.0,
        1.0,
    )
    started_raw = model.started_booster.predict_proba(design)[:, 1]
    started = np.clip(
        (
            model.started_calibrator.predict(started_raw)
            if model.started_calibrator is not None
            else started_raw
        ),
        0.0,
        1.0,
    )
    return pd.DataFrame(
        {"play_probability": played, "start_probability": np.minimum(started, played)},
        index=features.index,
    )


def _log_loss(actual: np.ndarray, predicted: np.ndarray, *, eps: float = 1e-9) -> float:
    clipped = np.clip(predicted, eps, 1.0 - eps)
    return float(-np.mean(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped)))


def walk_forward_evaluate(panel: pd.DataFrame, *, scored_seasons: Iterable[int]) -> pd.DataFrame:
    """Per-season Brier/log-loss for the model, one row per (season, label)."""

    rows: list[dict[str, Any]] = []
    for season in sorted({int(value) for value in scored_seasons}):
        try:
            model = fit_play_probability_model(panel, scored_season=season)
        except DataContractError:
            continue
        test = panel.loc[panel["season"].eq(season)]
        if test.empty:
            continue
        predictions = predict_play_probabilities(model, test)
        for label, column in (
            (LABEL_PLAYED, "play_probability"),
            (LABEL_STARTED, "start_probability"),
        ):
            actual = test[label].astype(float).to_numpy()
            predicted = predictions[column].to_numpy()
            rows.append(
                {
                    "season": season,
                    "label": label,
                    "n": len(test),
                    "calibration_status": model.calibration_status,
                    "train_seasons": list(model.train_seasons),
                    "calibration_season": model.calibration_season,
                    "base_rate_brier": float(
                        np.mean(
                            (actual - float(panel.loc[panel["season"].lt(season), label].mean()))
                            ** 2
                        )
                    ),
                    "brier": float(np.mean((predicted - actual) ** 2)),
                    "log_loss": _log_loss(actual, predicted),
                    "mean_predicted": float(predicted.mean()),
                    "mean_observed": float(actual.mean()),
                }
            )
    return pd.DataFrame(rows)


def depth_rank_only_baseline(
    panel: pd.DataFrame, *, scored_seasons: Iterable[int], prior: float = 50.0
) -> pd.DataFrame:
    """Season-lagged, shrunk (position_group, depth_rank_bucket) rate --
    everything this model's own depth-rank feature knows, and nothing else
    (no injury report, no history, no roster status)."""

    frames: list[pd.DataFrame] = []
    for season in sorted({int(value) for value in scored_seasons}):
        train = panel.loc[panel["season"].lt(season)]
        if train.empty:
            continue
        global_rate = float(train[LABEL_PLAYED].mean())
        grouped = train.groupby(["position_group", "depth_rank_bucket"], observed=True)[
            LABEL_PLAYED
        ].agg(["mean", "count"])
        grouped["rate"] = (grouped["mean"] * grouped["count"] + prior * global_rate) / (
            grouped["count"] + prior
        )
        lookup = grouped["rate"].to_dict()
        test = panel.loc[
            panel["season"].eq(season), ["position_group", "depth_rank_bucket", LABEL_PLAYED]
        ].copy()
        test["season"] = season
        test["baseline_prediction"] = [
            lookup.get((group, bucket), global_rate)
            for group, bucket in zip(test["position_group"], test["depth_rank_bucket"], strict=True)
        ]
        frames.append(test)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def current_approach_baseline_predictions(
    panel: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    scored_seasons: Iterable[int],
) -> pd.DataFrame:
    """Walk-forward predictions from the approach this lane replaces:
    ``nfl_ats.lineup_availability``'s no-designation base rate (by position
    group and recent role) for a player with no visible injury designation
    that week, and ``nfl_ats.availability.fixed_unavailability`` for a
    player who IS listed -- the same two rules ``build_week_lineups.py``
    used for every non-QB player before this lane's change. Documented
    simplification: production's LEARNED availability-rate table (an
    alternative to the fixed prior for listed players) is not reproduced
    here -- building it needs a completed-game schedule this module
    otherwise has no reason to fetch, and the base-rate branch (which IS
    reproduced exactly) governs the large majority of rows, including the
    QB2/QB3 sanity-check case the owner's complaint named.
    """

    from nfl_ats.availability import fixed_unavailability
    from nfl_ats.lineup_availability import (
        build_no_designation_outcomes,
        build_no_designation_rates,
        latest_recent_roles,
        no_designation_rate_lookup,
        no_designation_unavailability,
    )

    roster_season = pd.to_numeric(rosters["season"], errors="coerce")
    snap_season = pd.to_numeric(snaps["season"], errors="coerce")
    injury_season = pd.to_numeric(injuries["season"], errors="coerce")
    frames: list[pd.DataFrame] = []
    for season in sorted({int(value) for value in scored_seasons}):
        train_rosters = rosters.loc[roster_season.lt(season)]
        train_snaps = snaps.loc[snap_season.lt(season)]
        train_injuries = injuries.loc[injury_season.lt(season)]
        test = panel.loc[panel["season"].eq(season)].copy()
        if test.empty or train_rosters.empty or train_snaps.empty:
            continue
        outcomes = build_no_designation_outcomes(train_injuries, train_rosters, train_snaps)
        rates = build_no_designation_rates(outcomes, target_seasons=[season])
        lookup = no_designation_rate_lookup(rates)
        recent_roles = latest_recent_roles(train_rosters, train_snaps, before_season=season)
        fallback_rate = float(outcomes["unavailable"].mean())
        predictions = []
        for row in test.itertuples(index=False):
            if bool(row.has_injury_designation):
                unavailable = fixed_unavailability(row.report_status_raw, row.practice_status_raw)
            else:
                role = recent_roles.get(str(row.gsis_id), "unknown_no_history")
                rate = no_designation_unavailability(
                    lookup, target_season=season, position=row.position, recent_role=role
                )
                unavailable = rate if rate is not None else fallback_rate
            predictions.append(1.0 - unavailable)
        scored = test[["position_group", "depth_rank_bucket", LABEL_PLAYED]].copy()
        scored["season"] = season
        scored["baseline_prediction"] = predictions
        frames.append(scored)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def season_blocked_bootstrap(
    per_season_improvement: pd.Series,
    *,
    n_bootstrap: int = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    """Resample SEASONS (never rows or weeks) with replacement -- the
    AGENTS.md/FLEET_BRIEF convention of blocking a bootstrap at the unit that
    is genuinely independent, generalized here to season blocks because the
    walk-forward evaluation itself only ever produces one independent draw
    per scored season (each season's model is fit once, on data the OTHER
    scored seasons never see)."""

    values = per_season_improvement.to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("per_season_improvement must have at least one season")
    rng = np.random.default_rng(random_state)
    point_estimate = float(values.mean())
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draw = rng.integers(0, n, size=n)
        samples[index] = values[draw].mean()
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "point_estimate": point_estimate,
        "interval_low": float(low),
        "interval_high": float(high),
        "probability_positive": float(np.mean(samples > 0.0)),
        "n_seasons": int(n),
    }


def calibration_slot(position: object, depth_rank: object) -> str:
    """One of QB1/QB2/QB3+, RB1/RB2+, WR1-3/WR4+, OL, DL, LB, CB, S, K/P,
    other -- the reader-facing depth-slot grouping for the calibration
    table (coarser than ``_START_SHARE_POSITION_BUCKETS``, which is used
    only for the "started" label's own snap-share math)."""

    pos = str(position).strip().upper()
    rank_value = pd.to_numeric(pd.Series([depth_rank]), errors="coerce").iloc[0]
    rank = int(rank_value) if pd.notna(rank_value) else None
    if pos == "QB":
        if rank in (1, 2):
            return f"QB{rank}"
        return "QB3+"
    if pos in ("RB", "HB", "FB"):
        return "RB1" if rank == 1 else "RB2+"
    if pos == "WR":
        if rank in (1, 2, 3):
            return f"WR{rank}"
        return "WR4+"
    # Side-specific depth-chart slot tags (nflverse's daily-schema `pos_abb`,
    # seasons >= 2025 -- see the module docstring's schema-seam note) never
    # match the bare generic checks below on their own; normalized here so
    # e.g. a starting "LCB" lands in the same "CB" slot as a generic "CB".
    if pos in ("LT", "RT"):
        pos = "T"
    elif pos in ("LG", "RG"):
        pos = "G"
    elif pos in ("LDE", "RDE"):
        pos = "DE"
    elif pos in ("LDT", "RDT"):
        pos = "DT"
    elif pos in ("WLB", "SLB", "LILB", "RILB"):
        pos = "LB"
    elif pos in ("LCB", "RCB", "NB"):
        pos = "CB"
    elif pos == "PK":
        pos = "K"
    if pos in ("T", "G", "C", "OT", "OG", "OL"):
        return "OL"
    if pos in ("DE", "DT", "NT", "DL", "EDGE"):
        return "DL"
    if pos in ("LB", "ILB", "OLB", "MLB", "TE"):
        return "TE/LB" if pos == "TE" else "LB"
    if pos in ("CB", "DB"):
        return "CB"
    if pos in ("S", "FS", "SS", "SAF"):
        return "S"
    if pos in ("K", "P", "PK", "LS"):
        return "K/P"
    return "other"


def calibration_table(
    frame: pd.DataFrame, *, prediction_column: str, actual_column: str
) -> pd.DataFrame:
    """Mean predicted vs. observed by calibration slot -- the table the task
    asks for by name: "QB1/QB2/QB3, RB1/RB2, WR1-3, OL, DL, LB, CB, S, K/P"."""

    working = frame.copy()
    working["slot"] = [
        calibration_slot(position, rank)
        for position, rank in zip(working["position"], working["depth_rank"], strict=True)
    ]
    grouped = (
        working.groupby("slot", observed=True)
        .agg(
            n=("slot", "size"),
            mean_predicted=(prediction_column, "mean"),
            mean_observed=(actual_column, "mean"),
        )
        .reset_index()
    )
    grouped["gap"] = grouped["mean_predicted"] - grouped["mean_observed"]
    return grouped.sort_values("slot").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Serving adapters -- used by scripts/build_week_lineups.py, no training data
# fetched or re-derived here beyond what the caller already loaded.
# ---------------------------------------------------------------------------


def serving_player_history(
    rosters: pd.DataFrame, snaps: pd.DataFrame, *, as_of_season: int, as_of_week: int
) -> dict[str, dict[str, float]]:
    """``{gsis_id: {"weeks_since_last_snap":..., "trailing4_snap_share":...}}``
    as of strictly before (``as_of_season``, ``as_of_week``), from whatever
    local snap/roster history is on file -- the SAME history construction
    ``attach_history_features`` uses for training, queried once for every
    player instead of joined against a training panel."""

    history = _player_snap_history(rosters, snaps)
    if history.empty:
        return {}
    as_of_ordinal = float(_ordinal(pd.Series([as_of_season]), pd.Series([as_of_week])).iloc[0])
    eligible = history.loc[history["ordinal"].lt(as_of_ordinal)]
    if eligible.empty:
        return {}
    latest = eligible.sort_values(["gsis_id", "ordinal"]).groupby("gsis_id", sort=False).tail(1)
    gsis_ids = latest["gsis_id"].astype(str).to_numpy()
    ordinals = latest["ordinal"].to_numpy(dtype=float)
    trailing4 = latest["trailing4_inclusive"].to_numpy(dtype=float)
    return {
        gsis_id: {
            "weeks_since_last_snap": float(as_of_ordinal - ordinal),
            "trailing4_snap_share": float(value),
        }
        for gsis_id, ordinal, value in zip(gsis_ids, ordinals, trailing4, strict=True)
    }


def serving_feature_frame(
    depth_rows: pd.DataFrame,
    *,
    week: int,
    current_injuries: pd.DataFrame | None,
    player_history: dict[str, dict[str, float]],
    default_roster_status: str = "ACT",
) -> pd.DataFrame:
    """Build ``FEATURE_COLUMNS`` for one team's current depth-chart rows.

    ``depth_rows`` needs ``gsis_id``, ``position`` (nflverse's ``pos_abb``,
    possibly side-specific), and ``depth_rank`` (nflverse's ``pos_rank``).
    ``current_injuries`` is that team's visible injury rows, indexed by
    ``gsis_id``. ``default_roster_status`` is retained for caller compatibility
    but roster status is deliberately excluded from the model features.
    """

    working = depth_rows.copy()
    working["position"] = working["position"].astype("string").str.upper()
    working["position_group"] = working["position"].map(depth_chart_position_group)
    working["depth_rank_bucket"] = working["depth_rank"].map(depth_rank_bucket)
    working["season_week"] = float(week)
    working["roster_status"] = default_roster_status

    def _lookup_injury(gsis_id: Any) -> Any:
        if (
            current_injuries is None
            or current_injuries.empty
            or gsis_id not in current_injuries.index
        ):
            return None
        row = current_injuries.loc[gsis_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row

    report_values = []
    practice_values = []
    weeks_since_values = []
    trailing4_values = []
    for gsis_id in working["gsis_id"]:
        injury_row = _lookup_injury(gsis_id)
        report_values.append(injury_row.get("report_status") if injury_row is not None else None)
        practice_values.append(
            injury_row.get("practice_status") if injury_row is not None else None
        )
        history = player_history.get(str(gsis_id), {})
        weeks_since_values.append(history.get("weeks_since_last_snap"))
        trailing4_values.append(history.get("trailing4_snap_share"))
    working["report_category"] = [report_category(value) for value in report_values]
    working["practice_category"] = [practice_category(value) for value in practice_values]
    working["weeks_since_last_snap"] = weeks_since_values
    working["trailing4_snap_share"] = trailing4_values

    is_qb = working["position"].eq("QB")
    qb1_report = QB1_NOT_APPLICABLE
    qb1_practice = QB1_NOT_APPLICABLE
    qb_rows = working.loc[is_qb]
    if not qb_rows.empty:
        qb1_rank = pd.to_numeric(qb_rows["depth_rank"], errors="coerce").min()
        qb1_candidates = qb_rows.loc[
            pd.to_numeric(qb_rows["depth_rank"], errors="coerce").eq(qb1_rank)
        ]
        qb1_gsis_id = qb1_candidates.iloc[0]["gsis_id"]
        qb1_injury = _lookup_injury(qb1_gsis_id)
        qb1_report = report_category(
            qb1_injury.get("report_status") if qb1_injury is not None else None
        )
        qb1_practice = practice_category(
            qb1_injury.get("practice_status") if qb1_injury is not None else None
        )
    working["qb1_report_category"] = np.where(is_qb, qb1_report, QB1_NOT_APPLICABLE)
    working["qb1_practice_category"] = np.where(is_qb, qb1_practice, QB1_NOT_APPLICABLE)
    return working


PlayProbabilityPredictor = Callable[[pd.DataFrame], pd.DataFrame]


def make_predictor(model: PlayProbabilityModel) -> PlayProbabilityPredictor:
    """A small closure ``scripts/build_week_lineups.py`` injects into
    ``_team_payload`` -- keeps that function's tests able to stub prediction
    without needing a real trained model."""

    def _predict(features: pd.DataFrame) -> pd.DataFrame:
        return predict_play_probabilities(model, features)

    return _predict


__all__ = [
    "CATEGORICAL_FEATURE_COLUMNS",
    "DEPTH_CHART_HISTORY_VERSION",
    "DEPTH_RANK_BUCKET_CATEGORIES",
    "FEATURE_COLUMNS",
    "LABEL_PLAYED",
    "LABEL_STARTED",
    "NUMERIC_FEATURE_COLUMNS",
    "PLAY_PROBABILITY_MODEL_VERSION",
    "POSITION_GROUP_CATEGORIES",
    "DepthChartHistorySnapshot",
    "PlayProbabilityModel",
    "PlayProbabilityPredictor",
    "attach_history_features",
    "build_player_week_panel",
    "calibration_slot",
    "calibration_table",
    "canonicalize_depth_chart_history",
    "current_approach_baseline_predictions",
    "depth_chart_history_snapshot_from_root",
    "depth_rank_bucket",
    "depth_rank_only_baseline",
    "fetch_depth_chart_history_snapshot",
    "fit_play_probability_model",
    "latest_depth_chart_history_snapshot",
    "load_depth_chart_history_snapshot",
    "make_predictor",
    "predict_play_probabilities",
    "season_blocked_bootstrap",
    "serving_feature_frame",
    "serving_player_history",
    "walk_forward_evaluate",
    "write_depth_chart_history_snapshot",
]
