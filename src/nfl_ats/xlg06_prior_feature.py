"""Rookie-prior skill features, stacked on PRODUCTION (XLG-06 Stage 4 wiring).

``docs/xlg06_stage4_wiring_eval.md`` predeclared and froze this construction:
per team-game, the snap-weighted mean Stage-3 rookie-prior expectation over
the side's active drafted skill players (recruiting WR/RB/TE with a usable
rating), plus the differential -- 3 columns, N0 fixed at 300, everything
strictly pregame.

**Point-in-time discipline.** Every player input Row must come from a
completed REG week strictly before the target game: career snaps, career
EPA/game, trailing-4-week activity, and latest team are all computed from
rows with ``(season, week)`` below the game's own ``(season, week)``. A
leakage test pins that post-cutoff rows never move a prior. Players without
a usable rating, without trailing activity, or without a linked identity
contribute weight zero through team NaN (missing, never imputed here --
imputation belongs to the model's own training-fold median).

Mirrors ``nfl_ats.fluview_production_feature``'s additive-merge discipline:
every pre-existing column comes back bit-identical, only the three new
columns are added. Frozen Stage-3 parameters are read from the recorded
prior-spec artifact, never re-fit here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.xlg06_prior import blend_prior

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Frozen Stage-3 prior parameters (docs/xlg06_stage3_prior_spec.md §10).
PRIOR_SPEC_DEFAULT = (
    REPO_ROOT / "artifacts" / "xlg06_stage3_prior" / "20260903T191431Z" / "prior_spec.json"
)
#: Frozen crosswalk population (docs/xlg06_stage2_crosswalk.md).
CROSSWALK_DEFAULT = (
    REPO_ROOT
    / "artifacts"
    / "xlg06_crosswalk"
    / "20260903T104848Z"
    / "recruit_to_nfl_crosswalk.parquet"
)

HOME_PRIOR_COLUMN = "home_rookie_prior_skill"
AWAY_PRIOR_COLUMN = "away_rookie_prior_skill"
DIFF_PRIOR_COLUMN = "diff_rookie_prior_skill"
PRIOR_ON_PRODUCTION_FEATURE_COLUMNS = (
    HOME_PRIOR_COLUMN,
    AWAY_PRIOR_COLUMN,
    DIFF_PRIOR_COLUMN,
)

SKILL_POSITIONS = ("WR", "RB", "TE")
TRAILING_WEEKS = 4
N0_SNAPS = 300.0

_REQUIRED_GAME_COLUMNS = {"game_id", "season", "week", "home_team", "away_team"}


def load_prior_spec(path: Path | None = None) -> dict[str, float]:
    """Frozen Stage-3 intercept/slope; read, never re-fit."""

    resolved = path or PRIOR_SPEC_DEFAULT
    payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
    return {"intercept": float(payload["intercept"]), "slope": float(payload["slope"])}


def load_linked_skill(
    crosswalk: pd.DataFrame | None = None, path: Path | None = None
) -> pd.DataFrame:
    """Linked drafted skill population: gsis_id, rating (usable only)."""

    frame = crosswalk if crosswalk is not None else pd.read_parquet(path or CROSSWALK_DEFAULT)
    linked = frame.loc[frame["gsis_id"].notna() & frame["position"].isin(SKILL_POSITIONS)].copy()
    linked["gsis_id"] = linked["gsis_id"].astype(str)
    linked["rating_num"] = pd.to_numeric(linked["rating"], errors="coerce")
    linked = linked.loc[linked["rating_num"].notna()].copy()
    return linked.loc[:, ["gsis_id", "rating_num"]].drop_duplicates(subset="gsis_id")


def _season_weeks(season: int) -> int:
    """REG weeks per season: 17 through 2020, 18 since 2021."""

    return 18 if season >= 2021 else 17


def _trailing_slots(season: int, week: int, n: int = TRAILING_WEEKS) -> list[tuple[int, int]]:
    """The ``n`` completed REG week slots strictly before ``(season, week)``."""

    slots: list[tuple[int, int]] = []
    cursor = (int(season), int(week))
    for _ in range(n):
        year, number = cursor
        number -= 1
        if number < 1:
            year -= 1
            number = _season_weeks(year)
        cursor = (year, number)
        slots.append(cursor)
    return slots


def build_player_week_panel(
    snap_counts: pd.DataFrame,
    player_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Per-GSIS per-week REG panel: offensive snaps and rushing+receiving EPA.

    ``snap_counts`` must already carry ``gsis_id`` (see
    ``nfl_ats.players.attach_snap_player_ids``); ``player_stats`` uses its
    native GSIS ``player_id``. Only REG rows enter either side.
    """

    for label, frame, columns in (
        ("snap_counts", snap_counts, ("gsis_id", "season", "week", "team", "offense_snaps")),
        (
            "player_stats",
            player_stats,
            ("player_id", "season", "week", "rushing_epa", "receiving_epa", "season_type"),
        ),
    ):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise DataContractError(f"{label} is missing columns: {', '.join(missing)}")
    snaps = snap_counts.copy()
    if "game_type" in snaps.columns:
        snaps = snaps.loc[snaps["game_type"].astype(str).eq("REG")].copy()
    snaps = snaps.loc[snaps["gsis_id"].notna()].copy()
    snaps["gsis_id"] = snaps["gsis_id"].astype(str)
    snaps["season"] = pd.to_numeric(snaps["season"], errors="coerce")
    snaps["week"] = pd.to_numeric(snaps["week"], errors="coerce")
    snaps["offense_snaps"] = pd.to_numeric(snaps["offense_snaps"], errors="coerce").fillna(0.0)
    stats = player_stats.loc[player_stats["season_type"].eq("REG")].copy()
    stats["gsis_id"] = stats["player_id"].astype(str)
    stats["season"] = pd.to_numeric(stats["season"], errors="coerce")
    stats["week"] = pd.to_numeric(stats["week"], errors="coerce")
    stats["rushing_epa"] = pd.to_numeric(stats["rushing_epa"], errors="coerce").fillna(0.0)
    stats["receiving_epa"] = pd.to_numeric(stats["receiving_epa"], errors="coerce").fillna(0.0)
    stats["weekly_epa"] = stats["rushing_epa"] + stats["receiving_epa"]
    snap_weekly = (
        snaps.groupby(["gsis_id", "season", "week"], sort=False)
        .agg(team=("team", "last"), off_snaps=("offense_snaps", "sum"))
        .reset_index()
    )
    epa_weekly = (
        stats.groupby(["gsis_id", "season", "week"], sort=False)["weekly_epa"].sum().reset_index()
    )
    panel = snap_weekly.merge(epa_weekly, on=["gsis_id", "season", "week"], how="left")
    panel["weekly_epa"] = panel["weekly_epa"].fillna(0.0)
    panel = panel.loc[panel["season"].notna() & panel["week"].notna()].copy()
    panel["season"] = panel["season"].astype(int)
    panel["week"] = panel["week"].astype(int)
    return panel.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)


def _career_totals(visible: pd.DataFrame) -> pd.DataFrame:
    """Career sums over every strictly-pregame row (indexed by player).

    Aggregated directly over the visible frame -- never read off one row's
    exclusive cumulative state, which would drop that row's own week from
    the career total.
    """

    return (
        visible.groupby("gsis_id", sort=False)
        .agg(
            career_snaps=("off_snaps", "sum"),
            career_epa=("weekly_epa", "sum"),
            career_games=("weekly_epa", "size"),
        )
        .reset_index()
    )


def _visible_before(panel: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Rows from completed REG weeks strictly before ``(season, week)``."""

    return panel.loc[
        (panel["season"] < season) | ((panel["season"] == season) & (panel["week"] < week))
    ].copy()


def derive_prior_expectations(
    games: pd.DataFrame,
    panel: pd.DataFrame,
    linked: pd.DataFrame,
    params: dict[str, float],
) -> pd.DataFrame:
    """Per-game snap-weighted rookie-prior expectations, strictly pregame.

    For each game, eligible skill players are linked recruits whose trailing
    4 completed REG weeks (evaluated AT the game week, not at their latest
    row) carry offensive snaps above zero. Returns
    ``(game_id, home/away/diff_rookie_prior_skill)``; teams with no eligible
    player come back NaN. Career snaps/EPA average over all strictly-prior
    rows; team identity is the latest pregame row's team.
    """

    missing = sorted(_REQUIRED_GAME_COLUMNS.difference(games.columns))
    if missing:
        raise DataContractError(f"games is missing columns: {', '.join(missing)}")
    rated = panel.merge(linked.loc[:, ["gsis_id", "rating_num"]], on="gsis_id", how="inner")
    game_keys = games.loc[:, ["game_id", "season", "week", "home_team", "away_team"]].copy()
    game_keys["game_id"] = game_keys["game_id"].astype(str)
    game_keys["season"] = pd.to_numeric(game_keys["season"], errors="raise").astype(int)
    game_keys["week"] = pd.to_numeric(game_keys["week"], errors="raise").astype(int)
    rows: list[dict[str, Any]] = []
    for game in game_keys.itertuples():
        season = int(str(game.season))
        week = int(str(game.week))
        visible = _visible_before(rated, season, week)
        slots = set(_trailing_slots(season, week))
        slot_mask = [
            (int(row_season), int(row_week)) in slots
            for row_season, row_week in zip(visible["season"], visible["week"], strict=True)
        ]
        trailing = visible.loc[slot_mask].copy()
        trailing_sums = trailing.groupby("gsis_id", sort=False)["off_snaps"].sum()
        latest = (
            visible.sort_values(["season", "week"])
            .groupby("gsis_id", sort=False)
            .tail(1)
            .set_index("gsis_id")
        )
        career = _career_totals(visible).set_index("gsis_id")
        row: dict[str, Any] = {"game_id": game.game_id}
        for side, column in (("home_team", HOME_PRIOR_COLUMN), ("away_team", AWAY_PRIOR_COLUMN)):
            team = str(getattr(game, side))
            active_ids = trailing_sums.loc[trailing_sums.gt(0)].index
            candidates = [
                gsis
                for gsis in active_ids
                if gsis in latest.index and str(latest.loc[gsis, "team"]) == team
            ]
            if not candidates:
                row[column] = np.nan
                continue
            weights = np.array([float(trailing_sums.loc[gsis]) for gsis in candidates], dtype=float)
            values = np.array(
                [
                    blend_prior(
                        float(latest.loc[gsis, "rating_num"]),
                        float(career.loc[gsis, "career_epa"])
                        / max(int(career.loc[gsis, "career_games"]), 1)
                        if int(career.loc[gsis, "career_games"]) > 0
                        else 0.0,
                        float(career.loc[gsis, "career_snaps"]),
                        intercept=params["intercept"],
                        slope=params["slope"],
                        n0=N0_SNAPS,
                    )
                    for gsis in candidates
                ],
                dtype=float,
            )
            row[column] = float(np.average(values, weights=weights))
        home_value = row[HOME_PRIOR_COLUMN]
        away_value = row[AWAY_PRIOR_COLUMN]
        row[DIFF_PRIOR_COLUMN] = (
            float(home_value - away_value)
            if pd.notna(home_value) and pd.notna(away_value)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def attach_rookie_prior_features(
    features: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
    linked: pd.DataFrame | None = None,
    params: dict[str, float] | None = None,
    snap_counts: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join the three prior columns onto ``features``.

    Every pre-existing column is returned bit-identical; only the new
    columns are added. Pass ``panel``/``linked``/``params`` in tests;
    production callers leave them unset to read the frozen snapshots.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(PRIOR_ON_PRODUCTION_FEATURE_COLUMNS).intersection(features.columns))
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")
    if panel is None:
        if snap_counts is None or player_stats is None:
            raise DataContractError("attach needs panel or (snap_counts, player_stats)")
        panel = build_player_week_panel(snap_counts, player_stats)
    resolved_linked = linked if linked is not None else load_linked_skill()
    resolved_params = params if params is not None else load_prior_spec()
    derived = derive_prior_expectations(features, panel, resolved_linked, resolved_params)
    merged = features.merge(derived, on="game_id", how="left", validate="one_to_one")
    merged.index = features.index
    return merged


__all__ = [
    "AWAY_PRIOR_COLUMN",
    "DIFF_PRIOR_COLUMN",
    "HOME_PRIOR_COLUMN",
    "PRIOR_ON_PRODUCTION_FEATURE_COLUMNS",
    "attach_rookie_prior_features",
    "build_player_week_panel",
    "derive_prior_expectations",
    "load_linked_skill",
    "load_prior_spec",
]
