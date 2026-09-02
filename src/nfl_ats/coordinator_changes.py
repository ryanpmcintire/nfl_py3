"""Point-in-time coordinator-change state.

This module deliberately does not fetch or infer coaching staff history.  It
turns caller-supplied OC/DC assignment observations into auditable pregame
state and fails closed when an assignment or its preceding-game comparison is
not known by the decision timestamp.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError

ROLES = ("OC", "DC")
COORDINATOR_CHANGE_FEATURES = (
    "home_oc_changed",
    "away_oc_changed",
    "home_dc_changed",
    "away_dc_changed",
    "home_coordinator_change_count",
    "away_coordinator_change_count",
    "diff_coordinator_change_count",
)

_GAME_COLUMNS = {
    "game_id",
    "season",
    "week",
    "decision_at",
    "kickoff",
    "away_team",
    "home_team",
}
_ASSIGNMENT_COLUMNS = {
    "team",
    "role",
    "coordinator_name",
    "effective_at",
    "observed_at",
    "source_url",
}


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f"{label} missing required columns: {sorted(missing)}")


def _utc_series(values: pd.Series, *, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise DataContractError(f"{label} contains a missing or invalid timestamp")
    return parsed


def _team(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        raise DataContractError("team identity must be non-empty")
    raw = str(value).strip().upper()
    return TEAM_ABBREVIATION_ALIASES.get(raw, raw)


def _prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(games, _GAME_COLUMNS, label="games")
    result = games.copy()
    if result.empty:
        return result
    if result["game_id"].isna().any() or result["game_id"].astype(str).duplicated().any():
        raise DataContractError("games.game_id must be non-null and unique")
    result["decision_at"] = _utc_series(result["decision_at"], label="games.decision_at")
    result["kickoff"] = _utc_series(result["kickoff"], label="games.kickoff")
    if result["decision_at"].ge(result["kickoff"]).any():
        raise DataContractError("each decision_at must be strictly before kickoff")
    result["away_team"] = result["away_team"].map(_team)
    result["home_team"] = result["home_team"].map(_team)
    result["season"] = pd.to_numeric(result["season"], errors="raise").astype(int)
    result["week"] = pd.to_numeric(result["week"], errors="raise").astype(int)
    return result.sort_values(["decision_at", "game_id"], kind="stable").reset_index(drop=True)


def _prepare_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    _require_columns(assignments, _ASSIGNMENT_COLUMNS, label="assignments")
    result = assignments.copy()
    if result.empty:
        return result
    if result[list(_ASSIGNMENT_COLUMNS)].isna().any().any():
        raise DataContractError("coordinator assignment fields must be non-null")
    result["team"] = result["team"].map(_team)
    result["role"] = result["role"].astype(str).str.strip().str.upper()
    invalid_roles = sorted(set(result["role"]).difference(ROLES))
    if invalid_roles:
        raise DataContractError(f"ambiguous or unsupported coordinator roles: {invalid_roles}")
    result["coordinator_name"] = result["coordinator_name"].astype(str).str.strip()
    if (
        result["coordinator_name"].eq("").any()
        or result["source_url"].astype(str).str.strip().eq("").any()
    ):
        raise DataContractError("coordinator_name and source_url must be non-empty")
    result["effective_at"] = _utc_series(result["effective_at"], label="assignments.effective_at")
    result["observed_at"] = _utc_series(result["observed_at"], label="assignments.observed_at")

    identity = ["team", "role", "effective_at", "observed_at"]
    conflicts = result.groupby(identity, dropna=False)["coordinator_name"].nunique()
    if conflicts.gt(1).any():
        raise DataContractError("conflicting coordinator names share one observation identity")
    return result.drop_duplicates([*identity, "coordinator_name", "source_url"]).reset_index(
        drop=True
    )


def _assignment_at(
    assignments: pd.DataFrame,
    *,
    team: str,
    role: str,
    decision_at: pd.Timestamp,
) -> pd.Series | None:
    eligible = assignments.loc[
        assignments["team"].eq(team)
        & assignments["role"].eq(role)
        & assignments["effective_at"].le(decision_at)
        & assignments["observed_at"].le(decision_at)
    ]
    if eligible.empty:
        return None
    latest_effective = eligible["effective_at"].max()
    eligible = eligible.loc[eligible["effective_at"].eq(latest_effective)]
    latest_observed = eligible["observed_at"].max()
    eligible = eligible.loc[eligible["observed_at"].eq(latest_observed)]
    if eligible["coordinator_name"].nunique() != 1:
        raise DataContractError(
            f"ambiguous {role} assignment for {team} at {decision_at.isoformat()}"
        )
    return eligible.sort_values("source_url", kind="stable").iloc[-1]


def build_coordinator_change_features(
    games: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Build pregame OC/DC change flags relative to each team's prior game.

    A flag is nullable unless both the current assignment and the assignment
    known at the preceding game's own decision timestamp are available.  This
    prevents a late historical correction from rewriting an already-decided
    game or fabricating continuity from an incomplete source.
    """

    prepared_games = _prepare_games(games)
    prepared_assignments = _prepare_assignments(assignments)
    if prepared_games.empty:
        return pd.DataFrame(columns=["game_id", *COORDINATOR_CHANGE_FEATURES])

    prior_decisions: dict[str, pd.Timestamp] = {}
    records: list[dict[str, Any]] = []
    for game in prepared_games.itertuples(index=False):
        decision_at = pd.Timestamp(str(game.decision_at))
        record: dict[str, Any] = {"game_id": str(game.game_id)}
        for side in ("home", "away"):
            team = str(getattr(game, f"{side}_team"))
            prior_decision = prior_decisions.get(team)
            for role in ROLES:
                current = _assignment_at(
                    prepared_assignments,
                    team=team,
                    role=role,
                    decision_at=decision_at,
                )
                previous = (
                    _assignment_at(
                        prepared_assignments,
                        team=team,
                        role=role,
                        decision_at=prior_decision,
                    )
                    if prior_decision is not None
                    else None
                )
                prefix = f"{side}_{role.lower()}"
                record[f"{prefix}_name"] = (
                    str(current["coordinator_name"]) if current is not None else None
                )
                record[f"{prefix}_effective_at"] = (
                    current["effective_at"] if current is not None else pd.NaT
                )
                record[f"{prefix}_observed_at"] = (
                    current["observed_at"] if current is not None else pd.NaT
                )
                record[f"{prefix}_source_url"] = (
                    str(current["source_url"]) if current is not None else None
                )
                record[f"{prefix}_changed"] = (
                    pd.NA
                    if current is None or previous is None
                    else str(current["coordinator_name"]) != str(previous["coordinator_name"])
                )
            flags = [record[f"{side}_{role.lower()}_changed"] for role in ROLES]
            record[f"{side}_coordinator_change_count"] = (
                pd.NA
                if any(pd.isna(flag) for flag in flags)
                else int(sum(bool(flag) for flag in flags))
            )
        home_count = record["home_coordinator_change_count"]
        away_count = record["away_coordinator_change_count"]
        record["diff_coordinator_change_count"] = (
            pd.NA
            if pd.isna(home_count) or pd.isna(away_count)
            else int(home_count) - int(away_count)
        )
        records.append(record)
        prior_decisions[str(game.home_team)] = decision_at
        prior_decisions[str(game.away_team)] = decision_at

    result = pd.DataFrame(records)
    for column in COORDINATOR_CHANGE_FEATURES:
        result[column] = result[column].astype("Int8")
    return result
