"""PBP-08 protection-mismatch flags, including for games not yet played.

The screen (``scripts/pbp08_matchup_screen.py``, predeclaration and results in
``docs/pbp08_matchup_screen.md``) measured four scheme/matchup interaction
cells on REG 2009-2025. One resolved positive-shaped on both blockings:

    ``pbp08_protection_mismatch`` -- the offense's 4-game pressure-allowed
    rate is top-quartile AND its opponent defense's pressure-generated rate
    is top-quartile. Flagged offenses cover 45.19% (2009-2017) against a
    50.45% complement; full-slate effect +0.336 accuracy points, week-blocked
    95% [+0.014, +0.658], ``probability_positive`` 0.9785, season-blocked
    0.9797, era-consistent (+0.445 / +0.225), and BOTH bottom-vs-bottom
    mirror controls land where a null should (+0.019/+0.033, P+ ~0.56).

This module exists because the screen cannot answer the production question.
``scripts/pbp08_matchup_screen.py:load_population`` drops every game whose
``home_cover`` is null, which is exactly the set an upcoming week consists of.
The traits themselves are strictly-prior by construction, so a game with no
outcome still has a perfectly well-defined window -- it is only the screen's
scoring population that excludes it.

Every frozen parameter below is copied from the screen, not re-chosen:
``WINDOW_GAMES``, ``MIN_WINDOW_OBS``, ``MIN_QUANTILE_POOL``, the
pressure definition (sack OR qb_hit), the v1 ``analysis_plays`` competitive
filter, and the expanding strictly-prior quartile assignment. Re-deriving any
of them would make this a different hypothesis wearing the screen's numbers.

**Game-level rule, frozen here BEFORE any 2026 game is scored.** The screen
measured team-games; a card needs one answer per GAME:

* exactly one side flagged -> back that side's OPPONENT (the defense);
* both sides flagged -> no lean (the mismatch is mutual and cancels);
* neither flagged -> no lean.

The both-flagged case is stated rather than silently folded into "back
somebody", because a mutual mismatch is not the construct the screen measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.pbp import analysis_plays, load_pbp_snapshot, snapshot_from_root

#: Frozen by the screen's predeclaration. Not tunable here.
WINDOW_GAMES = 4
MIN_WINDOW_OBS = 3
MIN_QUANTILE_POOL = 200

#: Quartile codes emitted by :func:`expanding_quartile_flags`.
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
    """Per team-game pressure-allowed and pressure-generated rates.

    Identical construction to the screen's ``build_game_trait_tables`` for the
    two protection legs; the two passing legs are not built because this
    module serves the protection cell only.
    """

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
    """Quartile code per row from STRICTLY EARLIER week-blocks only.

    Byte-for-byte the screen's own routine. Thresholds never see the block
    they are applied to, so no future information reaches a flag, and a row
    is left ``QUARTILE_UNASSIGNED`` until at least ``MIN_QUANTILE_POOL``
    strictly-prior observations exist.
    """

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
    """One row per team-game with its strictly-prior 4-game trait window.

    ``schedule`` must carry every REG game the windows may draw on, INCLUDING
    games with no outcome yet -- an upcoming game contributes nothing to its
    own window but must be present to receive one.
    """

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
    """One row per game: which side (if any) the protection mismatch backs.

    ``schedule`` is the REG schedule frame -- it must include the upcoming
    week, and it must include enough completed history for a 4-game window
    (in practice the prior season, since a Week 1 window reaches back into it).
    """

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

    # The cell fires on the OFFENSE row: that team's pressure-allowed window is
    # top-quartile AND the team it faces generates pressure at a top-quartile
    # rate. A row whose window never reached the minimum pool is UNASSIGNED and
    # is never folded into the complement -- same posture as the screen.
    home_flagged = home["press_allow_q"].eq(QUARTILE_TOP) & away["press_gen_q"].eq(QUARTILE_TOP)
    away_flagged = away["press_allow_q"].eq(QUARTILE_TOP) & home["press_gen_q"].eq(QUARTILE_TOP)
    table["home_offense_flagged"] = home_flagged.reindex(table.index).fillna(False).astype(bool)
    table["away_offense_flagged"] = away_flagged.reindex(table.index).fillna(False).astype(bool)

    only_home = table["home_offense_flagged"] & ~table["away_offense_flagged"]
    only_away = table["away_offense_flagged"] & ~table["home_offense_flagged"]
    table["back_side"] = np.where(only_home, "AWAY", np.where(only_away, "HOME", ""))

    return table.reset_index()[list(FLAG_TABLE_COLUMNS)]


def flag_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Counts a caller can log without re-deriving them."""

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
