"""Strict decision-time hierarchical APM units; frozen PER-09 recipe."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from nfl_ats.data import DataContractError
from nfl_ats.participation import _player_ids

APM_UNIT_COLUMNS = (
    "home_apm_off_rating",
    "home_apm_def_rating",
    "away_apm_off_rating",
    "away_apm_def_rating",
    "apm_off_rating_diff",
    "apm_def_rating_diff",
)
UNIT_BY_POSITION = {
    **dict.fromkeys(["C", "G", "T", "OT", "OG", "OC", "OL"], "OFF_OL"),
    **dict.fromkeys(["QB", "RB", "FB", "HB", "WR", "TE"], "OFF_SKILL"),
    **dict.fromkeys(
        ["DE", "DT", "NT", "LB", "OLB", "ILB", "MLB", "DL", "EDGE", "LDE", "RDE", "LDT", "RDT"],
        "DEF_FRONT",
    ),
    **dict.fromkeys(["CB", "S", "SS", "FS", "DB", "NB", "LCB", "RCB", "SAF"], "DEF_SECONDARY"),
}


def fit_unit_ratings(
    plays: pd.DataFrame, rosters: pd.DataFrame
) -> dict[tuple[str, str], tuple[float, int]]:
    """Fit one already time-filtered season; return team-side EPA and play count."""
    if plays.empty:
        return {}
    roster = rosters.merge(
        plays[["season", "week"]].drop_duplicates(), on=["season", "week"], how="inner"
    ).dropna(subset=["gsis_id", "position"])
    roster["unit"] = roster.position.astype(str).str.upper().str.strip().map(UNIT_BY_POSITION)
    lookup = (
        roster.dropna(subset=["unit"])
        .groupby("gsis_id")["unit"]
        .agg(lambda values: values.mode().iloc[0])
        .to_dict()
    )
    rows: list[dict[str, float]] = []
    counts: Counter[str] = Counter()
    for row in plays.itertuples(index=False):
        offense, defense = _player_ids(row.offense_players), _player_ids(row.defense_players)
        counts.update(offense)
        counts.update(defense)
        values = {f"offense_player::{p}": 1.0 for p in offense}
        values.update({f"defense_player::{p}": -1.0 for p in defense})
        values[f"offense_team::{row.posteam}"] = 11.0
        values[f"defense_team::{row.defteam}"] = -11.0
        rows.append(values)
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(rows)
    fit = Ridge(alpha=1000.0, solver="lsqr", fit_intercept=True, tol=1e-6)
    fit.fit(matrix, pd.to_numeric(plays.epa, errors="raise").clip(-5.0, 5.0))
    flat = dict(zip(vectorizer.get_feature_names_out(), fit.coef_, strict=True))
    shrunk = dict(flat)
    for side in ("offense", "defense"):
        for unit in sorted(set(lookup.values())):
            keys = [f"{side}_player::{p}" for p, u in lookup.items() if u == unit]
            keys = [key for key in keys if key in flat]
            if not keys:
                continue
            mean = float(np.mean([flat[key] for key in keys]))
            for key in keys:
                n = counts[key.split("::", 1)[1]]
                shrunk[key] = (n * flat[key] + 500.0 * mean) / (n + 500.0)
    result: dict[tuple[str, str], tuple[float, int]] = {}
    for side, short, team_column in (
        ("offense", "off", "posteam"),
        ("defense", "def", "defteam"),
    ):
        for team, group in plays.groupby(team_column):
            lineup_sums = [
                sum(shrunk[f"{side}_player::{p}"] for p in _player_ids(raw))
                for raw in group[f"{side}_players"]
            ]
            rating = float(np.mean(lineup_sums)) + 11.0 * shrunk[f"{side}_team::{team}"]
            result[str(team), short] = (rating, len(group))
    return result


def attach_apm_unit_features(
    games: pd.DataFrame, plays: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    """Seed from prior season and update only after source games complete.

    ``plays`` is the valid competitive participation table plus week and
    completed_at. Missing completion timestamps never enter a fit. Roster
    weeks are restricted to completed plays before deriving modal units.
    """
    for frame, required in (
        (games, {"game_id", "season", "home_team", "away_team", "decision_timestamp"}),
        (
            plays,
            {
                "game_id",
                "season",
                "week",
                "completed_at",
                "posteam",
                "defteam",
                "offense_players",
                "defense_players",
                "epa",
            },
        ),
        (rosters, {"gsis_id", "season", "week", "position"}),
    ):
        if missing := required.difference(frame.columns):
            raise DataContractError(f"APM input missing {sorted(missing)}")
    if games.game_id.duplicated().any():
        raise DataContractError("APM prediction game_id must be unique")
    if set(APM_UNIT_COLUMNS).intersection(games.columns):
        raise DataContractError("APM columns already present")
    decisions = pd.to_datetime(games.decision_timestamp, utc=True, errors="raise")
    if decisions.isna().any():
        raise DataContractError("APM decision timestamp must be present")
    source = plays.copy()
    source["completed_at"] = pd.to_datetime(source.completed_at, utc=True, errors="raise")
    if source.groupby("game_id").completed_at.nunique(dropna=False).gt(1).any():
        raise DataContractError("APM game has inconsistent completion times")
    source = source.sort_values(["completed_at", "game_id"], kind="stable")
    values = np.full((len(games), len(APM_UNIT_COLUMNS)), np.nan)
    cache: dict[tuple[int, tuple[str, ...]], dict[tuple[str, str], tuple[float, int]]] = {}
    for index, row in enumerate(games.itertuples(index=False)):
        available = source.loc[source.completed_at.lt(decisions.iloc[index])]
        ratings = []
        for year in (int(str(row.season)) - 1, int(str(row.season))):
            season_plays = available.loc[available.season.eq(year)]
            key = year, tuple(sorted(season_plays.game_id.unique()))
            if key not in cache:
                cache[key] = fit_unit_ratings(season_plays, rosters)
            ratings.append(cache[key])
        seed, current = ratings
        for offset, team in ((0, str(row.home_team)), (2, str(row.away_team))):
            for side_index, side in enumerate(("off", "def")):
                old, new = seed.get((team, side)), current.get((team, side))
                if new is None:
                    value = np.nan if old is None else old[0]
                elif old is None:
                    value = new[0]
                else:
                    value = (new[1] * new[0] + 500.0 * old[0]) / (new[1] + 500.0)
                values[index, offset + side_index] = value
        values[index, 4:] = values[index, :2] - values[index, 2:4]
    result = games.copy()
    result[list(APM_UNIT_COLUMNS)] = values
    return result
