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


def build_coordinator_history_features(games: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Adapt dated canonical snapshots without carrying assignments across seasons.

    Revision time is a conservative observation/effective boundary, not the
    appointment date. Undated season summaries and HC rows are never eligible.
    The original builder still owns all ambiguity and timestamp checks.
    """
    _require_columns(
        history,
        {
            "season",
            "team",
            "role",
            "person",
            "effective_observed_at",
            "observed_at_basis",
            "source_url",
        },
        label="coordinator history",
    )
    games = _prepare_games(games)
    eligible = history.loc[
        history["role"].isin(ROLES)
        & history["observed_at_basis"].isin(["wikipedia_revision", "dated_announcement"])
    ].copy()
    eligible["coordinator_name"] = eligible["person"]
    eligible["effective_at"] = eligible["effective_observed_at"]
    eligible["observed_at"] = eligible["effective_observed_at"]
    if games.empty:
        return build_coordinator_change_features(games, eligible)
    results = [
        build_coordinator_change_features(group, eligible.loc[eligible["season"].eq(season)])
        for season, group in games.groupby("season", sort=True)
    ]
    return pd.concat(results, ignore_index=True)


COORDINATOR_SEASON_COLUMNS = (
    "coord_new_oc_diff",
    "coord_new_dc_diff",
    "coord_new_hc_diff",
)


def build_coordinator_season_features(games: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """September-to-September turnover, strictly as observed before each decision.

    A later revision, including an in-season correction, cannot rewrite a
    preseason flag. Missing/ambiguous roles are unknown, never continuity.
    """
    prepared = _prepare_games(games)
    _require_columns(
        history,
        {
            "season",
            "team",
            "role",
            "person",
            "effective_observed_at",
            "sampled_as_of",
            "sample_mode",
            "observed_at_basis",
        },
        label="history",
    )
    history = history.loc[
        history.sample_mode.eq("preseason") & history.observed_at_basis.eq("wikipedia_revision")
    ].copy()
    history["team"] = history.team.map(_team)
    history["observed"] = _utc_series(history.effective_observed_at, label="history observation")
    history["cutoff"] = _utc_series(history.sampled_as_of, label="history cutoff")
    if history.observed.gt(history.cutoff).any():
        raise DataContractError("history observation after requested cutoff")
    groups = {
        (int(str(season)), str(team), str(role)): frame
        for (season, team, role), frame in history.groupby(["season", "team", "role"])
    }

    def person_at(season: int, team: str, role: str, decision: pd.Timestamp) -> str | None:
        group = groups.get((season, team, role))
        if group is None:
            return None
        eligible = group.loc[group.observed.lt(decision) & group.cutoff.le(decision)]
        names = eligible.person.dropna().astype(str).unique()
        return str(names[0]) if len(names) == 1 and names[0] else None

    records = []
    for game in prepared.itertuples(index=False):
        row: dict[str, Any] = {"game_id": game.game_id}
        for role in ("OC", "DC", "HC"):
            flags = []
            for side in ("home", "away"):
                team = str(getattr(game, f"{side}_team"))
                current = person_at(
                    int(str(game.season)), team, role, pd.Timestamp(str(game.decision_at))
                )
                prior = person_at(
                    int(str(game.season)) - 1, team, role, pd.Timestamp(str(game.decision_at))
                )
                flags.append(None if current is None or prior is None else int(current != prior))
            home, away = flags
            row[f"coord_new_{role.lower()}_diff"] = (
                float("nan") if home is None or away is None else float(home - away)
            )
        records.append(row)
    return pd.DataFrame(records, columns=["game_id", *COORDINATOR_SEASON_COLUMNS])


# ---------------------------------------------------------------------------
# Screen helpers (docs/playcaller_change_leads.md): games after an in-season
# change, and OC tenure at the season-start observation. Additive; neither
# is wired into any feature table.
# ---------------------------------------------------------------------------

PLAYCALLER_EVENT_COLUMNS = {"event_id", "season", "team", "role", "revision_at"}
_SCREEN_GAME_COLUMNS = {"game_id", "season", "kickoff", "home_team", "away_team"}
GAMES_AFTER_CHANGE_COLUMNS = (
    "game_id",
    "team",
    "season",
    "event_id",
    "role",
    "revision_at",
    "kickoff",
    "game_number_after_change",
)
OC_TENURE_COLUMNS = ("season", "team", "oc_name", "oc_tenure_years", "observed_at", "cutoff")


def _base_person_name(value: Any) -> str:
    """Strip Wikipedia's parenthetical disambiguator and case-fold."""

    text = str(value).strip()
    if text.endswith(")") and "(" in text:
        text = text[: text.rfind("(")].strip()
    return text.casefold()


def games_after_coordinator_change(games: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Number each team's games after an in-season coordinator change.

    ``events`` rows carry ``event_id``, ``season``, ``team``, ``role`` and the
    revision instant ``revision_at`` (the information boundary).  A game is
    "after" an event only when its ``kickoff`` is STRICTLY later than the
    revision instant; ``game_number_after_change`` counts that team's games
    (kickoff order, same season) since the MOST RECENT event before the
    game's kickoff, restarting at every event.  Games before any event, or
    whose kickoff is at or before the revision instant, are not returned, so
    a revision recorded after a game can never flag that game, and adding a
    later event never changes an earlier game's number.  An event with no
    later game in its season fails closed.
    """

    _require_columns(games, _SCREEN_GAME_COLUMNS, label="games")
    _require_columns(events, PLAYCALLER_EVENT_COLUMNS, label="events")
    if events.empty:
        return pd.DataFrame(columns=GAMES_AFTER_CHANGE_COLUMNS)
    if games["game_id"].isna().any() or games["game_id"].astype(str).duplicated().any():
        raise DataContractError("games.game_id must be non-null and unique")
    if events["event_id"].isna().any() or events["event_id"].astype(str).duplicated().any():
        raise DataContractError("events.event_id must be non-null and unique")

    prepared_events = events.copy()
    prepared_events["team"] = prepared_events["team"].map(_team)
    prepared_events["role"] = prepared_events["role"].astype(str).str.strip().str.upper()
    invalid_roles = sorted(set(prepared_events["role"]).difference(ROLES))
    if invalid_roles:
        raise DataContractError(f"ambiguous or unsupported coordinator roles: {invalid_roles}")
    prepared_events["season"] = pd.to_numeric(prepared_events["season"], errors="raise").astype(int)
    prepared_events["revision_at"] = _utc_series(
        prepared_events["revision_at"], label="events.revision_at"
    )

    sides = []
    for side in ("home", "away"):
        sides.append(
            pd.DataFrame(
                {
                    "game_id": games["game_id"].astype(str),
                    "season": pd.to_numeric(games["season"], errors="raise").astype(int),
                    "team": games[f"{side}_team"].map(_team),
                    "kickoff": _utc_series(games["kickoff"], label="games.kickoff"),
                }
            )
        )
    team_games = pd.concat(sides, ignore_index=True)

    records: list[dict[str, Any]] = []
    for (season, team), team_events in prepared_events.groupby(["season", "team"], sort=True):
        team_events = team_events.sort_values(["revision_at", "event_id"], kind="stable")
        schedule = team_games.loc[
            team_games["season"].eq(int(str(season))) & team_games["team"].eq(str(team))
        ].sort_values(["kickoff", "game_id"], kind="stable")
        for event in team_events.itertuples(index=False):
            if not schedule["kickoff"].gt(event.revision_at).any():
                raise DataContractError(
                    f"event {event.event_id} ({team} {season} {event.role}) has no game "
                    "with a kickoff after its revision instant"
                )
        for game in schedule.itertuples(index=False):
            before = team_events.loc[team_events["revision_at"].lt(game.kickoff)]
            if before.empty:
                continue
            latest = before.iloc[-1]
            number = int(
                (
                    schedule["kickoff"].gt(latest["revision_at"])
                    & schedule["kickoff"].le(game.kickoff)
                ).sum()
            )
            records.append(
                {
                    "game_id": str(game.game_id),
                    "team": str(team),
                    "season": int(str(season)),
                    "event_id": latest["event_id"],
                    "role": str(latest["role"]),
                    "revision_at": latest["revision_at"],
                    "kickoff": game.kickoff,
                    "game_number_after_change": number,
                }
            )
    if not records:
        return pd.DataFrame(columns=GAMES_AFTER_CHANGE_COLUMNS)
    return pd.DataFrame(records, columns=GAMES_AFTER_CHANGE_COLUMNS)


def oc_tenure_at_season_start(history: pd.DataFrame) -> pd.DataFrame:
    """Offensive-coordinator tenure (1, 2 or 3 meaning three-plus seasons)
    per team-season, from preseason observations only.

    Tenure is known only when every observation it needs exists: year 1
    needs this season's and last season's OC; year 2 and 3+ also need the
    season before that.  Unknown or ambiguous team-seasons are omitted, never
    guessed.  In-season rows, undated rows and observations after their own
    ``sampled_as_of`` cutoff are never read, so a later revision cannot
    change a season-start tenure.  Names are compared after stripping the
    parenthetical disambiguator, case-folded.
    """

    _require_columns(
        history,
        {
            "season",
            "team",
            "role",
            "person",
            "effective_observed_at",
            "sampled_as_of",
            "sample_mode",
            "observed_at_basis",
        },
        label="history",
    )
    rows = history.loc[
        history["sample_mode"].eq("preseason")
        & history["observed_at_basis"].eq("wikipedia_revision")
        & history["role"].astype(str).str.upper().eq("OC")
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=OC_TENURE_COLUMNS)
    rows["team"] = rows["team"].map(_team)
    rows["season"] = pd.to_numeric(rows["season"], errors="raise").astype(int)
    rows["observed"] = _utc_series(rows["effective_observed_at"], label="history observation")
    rows["cutoff"] = _utc_series(rows["sampled_as_of"], label="history cutoff")
    if rows["observed"].gt(rows["cutoff"]).any():
        raise DataContractError("history observation after requested cutoff")
    rows = rows.loc[rows["person"].notna() & rows["person"].astype(str).str.strip().ne("")]

    observed: dict[tuple[int, str], tuple[str, str, pd.Timestamp, pd.Timestamp]] = {}
    for (season, team), group in rows.groupby(["season", "team"]):
        names = group["person"].astype(str).map(_base_person_name).unique()
        if len(names) != 1:
            continue
        observed[(int(str(season)), str(team))] = (
            str(group["person"].iloc[0]).strip(),
            str(names[0]),
            pd.Timestamp(group["observed"].max()),
            pd.Timestamp(group["cutoff"].max()),
        )

    records: list[dict[str, Any]] = []
    for (season, team), (display, current, observed_at, cutoff) in sorted(observed.items()):
        prior = observed.get((season - 1, team))
        if prior is None:
            continue
        used: list[pd.Timestamp] = [observed_at, prior[2]]
        if current != prior[1]:
            tenure = 1
        else:
            earlier = observed.get((season - 2, team))
            if earlier is None:
                continue
            tenure = 2 if prior[1] != earlier[1] else 3
            used.append(earlier[2])
        records.append(
            {
                "season": season,
                "team": team,
                "oc_name": display,
                "oc_tenure_years": tenure,
                "observed_at": max(used),
                "cutoff": cutoff,
            }
        )
    if not records:
        return pd.DataFrame(columns=OC_TENURE_COLUMNS)
    return pd.DataFrame(records, columns=OC_TENURE_COLUMNS)
