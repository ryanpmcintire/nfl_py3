from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final

import numpy as np
import pandas as pd

from nfl_ats.availability import position_group, resolve_unavailability
from nfl_ats.players import attach_snap_player_ids

_ELIGIBLE_ROSTER_STATUS = "ACT"

NO_DESIGNATION_POSITION_PRIOR = 100.0
NO_DESIGNATION_ROLE_PRIOR = 20.0
NO_DESIGNATION_RATE_VERSION = "v2-recent-role"
_ALL = "__all__"

RECENT_ROLE_RETURNING_CONTRIBUTOR = "returning_contributor"
RECENT_ROLE_NO_RECENT_ROLE = "no_recent_role"
RECENT_ROLE_UNKNOWN_NO_HISTORY = "unknown_no_history"

PROBABILITY_SOURCE_BASE_MODEL_QB: Final = "base_model_qb"
PROBABILITY_SOURCE_AVAILABILITY_MODEL: Final = "availability_model"
PROBABILITY_SOURCE_UNAVAILABLE: Final = "unavailable"

_DEPTH_CHART_POSITION_ALIASES: dict[str, str] = {
    "LDE": "DE",
    "RDE": "DE",
    "LDT": "DT",
    "RDT": "DT",
    "WLB": "OLB",
    "SLB": "OLB",
    "LILB": "ILB",
    "RILB": "ILB",
    "MLB": "ILB",
    "LCB": "CB",
    "RCB": "CB",
    "NB": "CB",
    "SS": "S",
    "FS": "S",
    "LT": "T",
    "RT": "T",
    "LG": "G",
    "RG": "G",
    "PK": "K",
}


def depth_chart_position_group(position: object) -> str:

    normalized = str(position).strip().upper()
    generic = _DEPTH_CHART_POSITION_ALIASES.get(normalized, normalized)
    return position_group(generic)


def _active_roster_snap_timeline(rosters: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:

    active = rosters.loc[rosters["status"].eq(_ELIGIBLE_ROSTER_STATUS)].copy()
    active = active.drop_duplicates(["season", "week", "team", "gsis_id"])

    linked_snaps = attach_snap_player_ids(snaps, rosters)
    linked_snaps = linked_snaps.loc[linked_snaps["gsis_id"].notna()].copy()
    linked_snaps["total_snaps"] = sum(
        pd.to_numeric(linked_snaps[column], errors="coerce").fillna(0.0)
        for column in ("offense_snaps", "defense_snaps", "st_snaps")
    )
    played = (
        linked_snaps.groupby(["season", "week", "team", "gsis_id"], observed=True)["total_snaps"]
        .max()
        .gt(0)
        .rename("played")
        .reset_index()
    )
    timeline = active.merge(
        played,
        on=["season", "week", "team", "gsis_id"],
        how="left",
        validate="many_to_one",
    )
    timeline["played"] = timeline["played"].fillna(False).astype(bool)
    timeline = timeline.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    timeline["recent_role_played"] = timeline.groupby("gsis_id", sort=False)["played"].shift(1)
    return timeline


def _recent_role_label(value: Any) -> str:
    if value is True:
        return RECENT_ROLE_RETURNING_CONTRIBUTOR
    if value is False:
        return RECENT_ROLE_NO_RECENT_ROLE
    return RECENT_ROLE_UNKNOWN_NO_HISTORY


def build_no_designation_outcomes(
    injuries: pd.DataFrame, rosters: pd.DataFrame, snaps: pd.DataFrame
) -> pd.DataFrame:

    timeline = _active_roster_snap_timeline(rosters, snaps)
    listed = injuries.loc[:, ["season", "week", "team", "gsis_id"]].drop_duplicates().copy()
    listed["_listed"] = True
    merged = timeline.merge(listed, on=["season", "week", "team", "gsis_id"], how="left")
    not_listed = merged.loc[merged["_listed"].isna()].drop(columns="_listed").copy()
    not_listed["unavailable"] = (~not_listed["played"]).astype(float)
    not_listed["position_group"] = not_listed["position"].map(position_group)
    not_listed["recent_role"] = not_listed["recent_role_played"].map(_recent_role_label)
    return (
        not_listed[
            [
                "season",
                "week",
                "team",
                "gsis_id",
                "position",
                "position_group",
                "recent_role",
                "played",
                "unavailable",
            ]
        ]
        .sort_values(["season", "week", "team", "gsis_id"])
        .reset_index(drop=True)
    )


def latest_recent_roles(
    rosters: pd.DataFrame, snaps: pd.DataFrame, *, before_season: int
) -> dict[str, str]:

    timeline = _active_roster_snap_timeline(rosters, snaps)
    prior = timeline.loc[pd.to_numeric(timeline["season"]).lt(before_season)]
    if prior.empty:
        return {}
    latest = (
        prior.sort_values(["gsis_id", "season", "week"])
        .groupby("gsis_id", sort=False, as_index=False)
        .tail(1)
    )
    return {
        str(row.gsis_id): (
            RECENT_ROLE_RETURNING_CONTRIBUTOR if row.played else RECENT_ROLE_NO_RECENT_ROLE
        )
        for row in latest.itertuples(index=False)
    }


def build_no_designation_rates(
    outcomes: pd.DataFrame,
    *,
    target_seasons: Iterable[int],
    position_prior: float = NO_DESIGNATION_POSITION_PRIOR,
    role_prior: float = NO_DESIGNATION_ROLE_PRIOR,
) -> pd.DataFrame:

    if not np.isfinite(position_prior) or position_prior < 0:
        raise ValueError("position_prior must be finite and nonnegative")
    if not np.isfinite(role_prior) or role_prior < 0:
        raise ValueError("role_prior must be finite and nonnegative")
    targets = sorted({int(value) for value in target_seasons})
    if not targets:
        raise ValueError("At least one target season is required")
    rows: list[dict[str, Any]] = []
    for target_season in targets:
        training = outcomes.loc[pd.to_numeric(outcomes["season"]).lt(target_season)]
        if training.empty:
            continue
        source_start = int(training["season"].min())
        source_end = int(training["season"].max())
        total = len(training)
        unavailable = int(training["unavailable"].sum())
        global_rate = unavailable / total
        rows.append(
            {
                "target_season": target_season,
                "position_group": _ALL,
                "recent_role": _ALL,
                "unavailability_probability": global_rate,
                "observations": total,
                "unavailable": unavailable,
                "source_start_season": source_start,
                "source_end_season": source_end,
                "rate_version": NO_DESIGNATION_RATE_VERSION,
            }
        )
        for group_name, group in training.groupby("position_group", observed=True, sort=True):
            group_observations = len(group)
            group_missing = int(group["unavailable"].sum())
            group_rate = (group_missing + position_prior * global_rate) / (
                group_observations + position_prior
            )
            rows.append(
                {
                    "target_season": target_season,
                    "position_group": str(group_name),
                    "recent_role": _ALL,
                    "unavailability_probability": group_rate,
                    "observations": group_observations,
                    "unavailable": group_missing,
                    "source_start_season": source_start,
                    "source_end_season": source_end,
                    "rate_version": NO_DESIGNATION_RATE_VERSION,
                }
            )
            for role_name, role_group in group.groupby("recent_role", observed=True, sort=True):
                role_observations = len(role_group)
                role_missing = int(role_group["unavailable"].sum())
                role_rate = (role_missing + role_prior * group_rate) / (
                    role_observations + role_prior
                )
                rows.append(
                    {
                        "target_season": target_season,
                        "position_group": str(group_name),
                        "recent_role": str(role_name),
                        "unavailability_probability": role_rate,
                        "observations": role_observations,
                        "unavailable": role_missing,
                        "source_start_season": source_start,
                        "source_end_season": source_end,
                        "rate_version": NO_DESIGNATION_RATE_VERSION,
                    }
                )
    if not rows:
        raise ValueError("No no-designation rates could be estimated")
    return (
        pd.DataFrame(rows)
        .sort_values(["target_season", "position_group", "recent_role"])
        .reset_index(drop=True)
    )


def no_designation_rate_lookup(rates: pd.DataFrame) -> dict[tuple[int, str, str], float]:
    return {
        (int(str(row.target_season)), str(row.position_group), str(row.recent_role)): float(
            str(row.unavailability_probability)
        )
        for row in rates.itertuples(index=False)
    }


def no_designation_unavailability(
    lookup: dict[tuple[int, str, str], float],
    *,
    target_season: int,
    position: object,
    recent_role: str = RECENT_ROLE_UNKNOWN_NO_HISTORY,
) -> float | None:
    group = depth_chart_position_group(position)
    for key in (
        (target_season, group, recent_role),
        (target_season, group, _ALL),
        (target_season, _ALL, _ALL),
    ):
        if key in lookup:
            return lookup[key]
    return None


def resolve_play_probability(
    *,
    gsis_id: str | None,
    position: object,
    target_season: int,
    current_injury: Any | None,
    learned_lookup: dict[tuple[int, str, str, str], float] | None,
    no_designation_lookup: dict[tuple[int, str, str], float] | None,
    recent_role: str = RECENT_ROLE_UNKNOWN_NO_HISTORY,
) -> tuple[float | None, str, str]:

    if not gsis_id:
        return None, PROBABILITY_SOURCE_UNAVAILABLE, "no gsis_id on this depth-chart row"
    if current_injury is not None:
        report_status = current_injury.get("report_status")
        practice_status = current_injury.get("practice_status")
        row_position = current_injury.get("position") or position
        unavailable, basis = resolve_unavailability(
            learned_lookup,
            target_season=target_season,
            report_status=report_status,
            practice_status=practice_status,
            position=row_position,
        )
        return (
            1.0 - unavailable,
            PROBABILITY_SOURCE_AVAILABILITY_MODEL,
            f"listed on this week's injury report (report={report_status!r}, "
            f"practice={practice_status!r}); availability model basis={basis}",
        )
    if no_designation_lookup:
        rate = no_designation_unavailability(
            no_designation_lookup,
            target_season=target_season,
            position=position,
            recent_role=recent_role,
        )
        if rate is not None:
            return (
                1.0 - rate,
                PROBABILITY_SOURCE_AVAILABILITY_MODEL,
                "no injury designation this week; using the position's historical "
                f"no-designation base rate (recent role: {recent_role})",
            )
    return (
        None,
        PROBABILITY_SOURCE_UNAVAILABLE,
        "no injury designation this week and no no-designation base rate is available "
        "for this position",
    )


__all__ = [
    "NO_DESIGNATION_POSITION_PRIOR",
    "NO_DESIGNATION_RATE_VERSION",
    "NO_DESIGNATION_ROLE_PRIOR",
    "PROBABILITY_SOURCE_AVAILABILITY_MODEL",
    "PROBABILITY_SOURCE_BASE_MODEL_QB",
    "PROBABILITY_SOURCE_UNAVAILABLE",
    "RECENT_ROLE_NO_RECENT_ROLE",
    "RECENT_ROLE_RETURNING_CONTRIBUTOR",
    "RECENT_ROLE_UNKNOWN_NO_HISTORY",
    "build_no_designation_outcomes",
    "build_no_designation_rates",
    "depth_chart_position_group",
    "latest_recent_roles",
    "no_designation_rate_lookup",
    "no_designation_unavailability",
    "resolve_play_probability",
]
