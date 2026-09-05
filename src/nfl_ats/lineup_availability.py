"""Per-player play probabilities for the This Week lineup panel.

Owner complaint (2026-09-05): "why is that only quarterbacks have the lineup
percentage filled in?" Root cause, read in ``scripts/build_week_lineups.py``:
``probability = qb_probability if position == "QB" and gsis_id == model_qb_id
else None`` -- every other listed starter was ``model_role: "context_only"``
with no probability at all.

This module does not invent a new probability model. It reuses the EXACT
same learned-then-fixed availability machinery
(:func:`nfl_ats.availability.resolve_unavailability`) that already produces
``{side}_qb_start_probability`` in the production feature table
(``nfl_ats.players.enrich_with_player_features`` / ``_injury_unavailability``)
-- the one QB the model consumes is already scored this way; this module
applies the identical rule to every other player on the depth chart.

It adds exactly one genuinely new, honestly-labelled quantity that
production does not compute anywhere: the **no-designation base rate** --
the historical unavailability rate of active-roster players who do not
appear on that week's injury report AT ALL (a healthy scratch, a coach's
decision, a practice-squad elevation gone sideways, and so on). Without it,
every player with no current-week injury designation would have to be
either invented at 100% (dishonestly precise) or left blank (which is
exactly the bug this module exists to fix). It is built with the same
expanding-window, strictly-earlier-season discipline and the same
Bayesian-shrinkage style as
:func:`nfl_ats.availability.build_season_lagged_availability_rates`, over
the same local player snapshot -- no network fetch.

**Measured, not assumed: a position-only rate would be materially
misleading for starters.** A first cut of this table (grouping only by
position group) put every "front"/"skill"/etc. player's no-designation
unavailability near 30-38% -- because nflverse's weekly roster ``status``
column mixes a WR1 who plays nearly every healthy week with a WR5 who
rarely dresses, and a flat position-group average is dragged toward the
deep bench. (``status == "INA"`` was also, at first, folded into the
"active roster" population; it is nflverse's OWN weekly inactive-list tag,
so including it made "unavailable" tautological rather than empirically
observed -- fixed by restricting to ``status == "ACT"`` only.) So each
outcome row also carries ``recent_role``: whether that same player (any
team) recorded a snap in their own most recent earlier ACT-roster
appearance -- ``"returning_contributor"`` (yes), ``"no_recent_role"`` (no),
or ``"unknown_no_history"`` (no earlier ACT appearance at all, e.g. a
rookie). This is a simple, leakage-safe proxy for "currently a rotation
piece" that needs no historical depth-chart data (which the local player
snapshot does not carry): it only ever looks at a player's OWN strictly
earlier appearances, never the target week's own outcome.

**Documented simplification:** ``build_no_designation_outcomes`` determines
"listed" from the FINAL weekly injury record, not a decision-time-filtered
view (``nfl_ats.availability.build_availability_outcomes`` filters injuries
to revisions visible >=24h before kickoff for exactly this reason). A player
whose only injury-report appearance that week lands after that cutoff --
name added right before kickoff -- is therefore counted as "listed" here and
excluded from the not-listed pool, rather than (correctly, but at real
engineering cost for a training-set nuance) counted as "not listed as of
decision time". This only shrinks the historical training pool slightly; it
introduces no leakage into any live decision, because the pool is built
exclusively from seasons strictly earlier than the season being served.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final

import numpy as np
import pandas as pd

from nfl_ats.availability import position_group, resolve_unavailability
from nfl_ats.players import attach_snap_player_ids

#: Deliberately NOT ``nfl_ats.players._ACTIVE_ROSTER_STATUSES`` (``{"ACT",
#: "INA"}``): that constant is right for roster-continuity purposes but
#: wrong here. "INA" is nflverse's own weekly gameday-inactive tag, so
#: folding it into the "active roster" population makes "unavailable"
#: tautological (measured: 100% of not-listed "INA" rows have zero snaps)
#: rather than an empirically discovered rate. Only "ACT" is a genuine
#: "expected to be eligible to play" population.
_ELIGIBLE_ROSTER_STATUS = "ACT"

#: Shrinkage priors for the no-designation base rate: a position group's
#: rate is pulled toward the season's global not-listed rate by
#: ``NO_DESIGNATION_POSITION_PRIOR`` pseudo-observations (same magnitude as
#: ``nfl_ats.availability.AVAILABILITY_POSITION_PRIOR``), and within a
#: position group, a ``recent_role`` rate is further pulled toward its own
#: group's rate by ``NO_DESIGNATION_ROLE_PRIOR`` pseudo-observations.
NO_DESIGNATION_POSITION_PRIOR = 100.0
NO_DESIGNATION_ROLE_PRIOR = 20.0
NO_DESIGNATION_RATE_VERSION = "v2-recent-role"
_ALL = "__all__"

#: ``recent_role`` values `build_no_designation_outcomes` tags every row
#: with; see the module docstring's "measured, not assumed" note.
RECENT_ROLE_RETURNING_CONTRIBUTOR = "returning_contributor"
RECENT_ROLE_NO_RECENT_ROLE = "no_recent_role"
RECENT_ROLE_UNKNOWN_NO_HISTORY = "unknown_no_history"

#: Probability provenance tags carried per player in the lineup artifact.
PROBABILITY_SOURCE_BASE_MODEL_QB: Final = "base_model_qb"
PROBABILITY_SOURCE_AVAILABILITY_MODEL: Final = "availability_model"
PROBABILITY_SOURCE_UNAVAILABLE: Final = "unavailable"

#: nflverse depth charts (``scripts/build_week_lineups.py``'s own
#: ``position_order``) use side-specific tags -- LDE/RDE, LILB/RILB/MLB/
#: WLB/SLB, LCB/RCB/NB, SS/FS, LT/RT, LG/RG -- that
#: ``nfl_ats.availability.position_group`` does not recognize (it only knows
#: the generic tags injuries/rosters already use). Mapped to the closest
#: generic tag before grouping, so an unlisted starting DE and an unlisted
#: LDE1 land in the same "front" bucket.
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
    """``nfl_ats.availability.position_group``, tolerant of side-specific
    depth-chart position tags (see ``_DEPTH_CHART_POSITION_ALIASES``)."""

    normalized = str(position).strip().upper()
    generic = _DEPTH_CHART_POSITION_ALIASES.get(normalized, normalized)
    return position_group(generic)


def _active_roster_snap_timeline(rosters: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, gsis_id, position) genuinely
    eligible (``status == "ACT"``) roster appearance, tagging whether the
    player recorded any snap that week and, via ``recent_role_played``,
    whether they did so in their own most recent EARLIER ACT appearance
    (any team, any season) -- ``True``/``False``/``pd.NA`` (no earlier ACT
    appearance at all). Looking only at a player's own strictly earlier
    rows keeps this leakage-safe: it never uses the target row's own
    outcome.
    """

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
    """One row per (season, week, team, gsis_id) eligible-roster player who
    carries NO injury-report row that week, tagging whether they recorded a
    snap that week and their ``recent_role`` (see the module docstring's
    "measured, not assumed" note).

    ``injuries``/``rosters``/``snaps`` must already be canonicalized -- the
    same contract ``nfl_ats.players.enrich_with_player_features`` requires
    (``nfl_ats.players.canonicalize_injuries`` / ``canonicalize_rosters`` /
    ``canonicalize_snaps``, or a snapshot already written through them).
    """

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
    """Serving-time counterpart of ``recent_role``: for every ``gsis_id``
    with at least one ACT-roster appearance strictly before
    ``before_season``, whether they recorded a snap in their OWN latest such
    appearance. A player absent from the result has no known history and
    should be treated as ``RECENT_ROLE_UNKNOWN_NO_HISTORY``.
    """

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
    """Expanding, strictly-earlier-season no-designation base rates.

    Same discipline as
    ``nfl_ats.availability.build_season_lagged_availability_rates``: each
    ``target_season``'s rate is trained only on seasons strictly before it.
    Three levels, each shrunk toward its parent: global (``__all__``,
    ``__all__``) -> per-``position_group`` (shrunk toward global by
    ``position_prior``) -> per-(``position_group``, ``recent_role``)
    (shrunk toward its own group's rate by ``role_prior``).
    """

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
    """Play probability, source tag, and a human-readable reason for one
    player who is NOT the base-model QB (that path stays bit-identical and
    is handled entirely by the caller before this function is reached).

    ``current_injury`` is the player's own visible injury-report row for
    this week (must expose ``report_status``/``practice_status`` and
    optionally ``position`` via ``.get``/attribute access, e.g. a
    ``pandas.Series``), already filtered by the caller to
    observed-before-``generated_at``, or ``None`` when no such row is
    visible. Never invents a number: returns ``(None, "unavailable",
    reason)`` when there is no ``gsis_id`` to key a rate to, or when neither
    the learned/fixed model nor the no-designation lookup can produce one.
    """

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
