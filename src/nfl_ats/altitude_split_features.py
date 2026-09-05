"""Schedule-only Denver flag and separate, retrospective quarter diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def denver_home_flag(games: pd.DataFrame) -> pd.Series:
    """No game outcomes, PBP, or future observations enter this pregame flag."""
    if games["home_team"].isna().any():
        raise ValueError("Missing home team")
    return games["home_team"].eq("DEN").rename("denver_home")


def quarter_margins(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Home margins at regulation boundaries; never use this as a pregame input.

    Snapshot score_differential is preplay and oriented to posteam. Use the
    first observed Q4 state only if its clock is 900; never substitute a later
    state. Overtime games end regulation at their first OT preplay state.
    """
    plays = pbp.loc[pbp["season_type"].eq("REG")].copy()
    plays = plays.sort_values(["game_id", "play_id"])
    plays = plays.loc[
        (plays.posteam.eq(plays.home_team) | plays.posteam.eq(plays.away_team))
        & plays["score_differential"].notna()
    ].copy()
    plays["home_margin"] = np.where(
        plays.posteam.eq(plays.home_team),
        plays.score_differential,
        -plays.score_differential,
    )
    q4 = plays.loc[plays.qtr.eq(4)].drop_duplicates("game_id")
    q4 = q4.loc[q4.game_seconds_remaining.eq(900), ["game_id", "home_margin"]]
    q4 = q4.rename(columns={"home_margin": "first_three_margin"})
    overtime = plays.loc[plays.qtr.ge(5)].drop_duplicates("game_id")
    overtime = overtime[["game_id", "home_margin"]].rename(
        columns={"home_margin": "regulation_margin"}
    )
    result = games.loc[games.season.between(2013, 2025) & games.game_type.eq("REG")].merge(
        q4, on="game_id", validate="one_to_one"
    )
    result = result.merge(overtime, on="game_id", how="left", validate="one_to_one")
    # An OT marker without a usable score state must not fall back to OT final.
    ot_ids = set(pbp.loc[pbp.qtr.ge(5), "game_id"])
    result = result.loc[~result.game_id.isin(ot_ids) | result.regulation_margin.notna()].copy()
    result["regulation_margin"] = result.regulation_margin.fillna(result["result"])
    result["fourth_margin"] = result.regulation_margin - result.first_three_margin
    result["late_minus_early"] = result.fourth_margin - result.first_three_margin
    result["late_minus_early_rate"] = result.fourth_margin - result.first_three_margin / 3
    return result.dropna(subset=["fourth_margin"]).sort_values(["season", "week", "game_id"])
