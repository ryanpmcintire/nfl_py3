"""Leak-safe identity crosswalk primitives for XLG-06 Stage 2.

The only permitted college-to-NFL identity path is
``recruiting.athleteId -> draft_picks.collegeAthleteId -> nfl_players.espn_id
-> nfl_players.gsis_id``.  Names and CFBD's ``nflAthleteId`` are deliberately
excluded: the latter is a separate CFBD-internal identifier space.

This module audits identity coverage only.  It does not construct a feature,
read an outcome, or make a model/registry decision.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nfl_ats.data import DataContractError, require_columns

RECRUITING_ID_COLUMN = "athleteId"
DRAFT_COLLEGE_ID_COLUMN = "collegeAthleteId"
NFL_ESPN_ID_COLUMN = "espn_id"
NFL_GSIS_ID_COLUMN = "gsis_id"


def _canonical_id(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = frame[column].astype("string").str.strip()
    invalid = values.isna() | values.eq("") | values.eq("<NA>") | values.eq("0")
    if invalid.any():
        raise DataContractError(f"{label} contains {int(invalid.sum())} null/empty/zero ids")
    return values


def _assert_unique_mapping(frame: pd.DataFrame, key: str, value: str, label: str) -> None:
    conflicting = frame.groupby(key, dropna=False, sort=False)[value].nunique(dropna=False)
    if conflicting.gt(1).any():
        examples = conflicting[conflicting.gt(1)].index.tolist()[:5]
        raise DataContractError(f"{label} has conflicting mappings for {key}: {examples}")


def build_recruit_to_nfl_crosswalk(
    recruiting: pd.DataFrame,
    draft_picks: pd.DataFrame,
    nfl_players: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join recruiting rows to stable NFL identities without a name join.

    The returned frame has one row per input recruiting row and nullable
    ``nfl_espn_id``/``gsis_id`` columns.  Ambiguous draft identities are
    excluded when they disagree; duplicate equivalent rows are collapsed
    before the join.  Coverage is reported separately for rows with a usable
    recruiting id, so missing ids are visible rather than silently discarded.
    """

    require_columns(recruiting, (RECRUITING_ID_COLUMN,), "XLG-06 recruiting")
    require_columns(draft_picks, (DRAFT_COLLEGE_ID_COLUMN, "nflAthleteId"), "XLG-06 draft picks")
    require_columns(nfl_players, (NFL_ESPN_ID_COLUMN, NFL_GSIS_ID_COLUMN), "XLG-06 NFL players")

    recruits = recruiting.copy()
    recruits["_college_espn_id"] = recruits[RECRUITING_ID_COLUMN].astype("string").str.strip()

    draft = draft_picks.loc[:, [DRAFT_COLLEGE_ID_COLUMN, "nflAthleteId"]].copy()
    draft["_college_espn_id"] = draft[DRAFT_COLLEGE_ID_COLUMN].astype("string").str.strip()
    draft_valid = (
        draft["_college_espn_id"].notna()
        & draft["_college_espn_id"].ne("")
        & draft["_college_espn_id"].ne("<NA>")
        & draft["_college_espn_id"].ne("0")
    )
    draft = draft.loc[draft_valid].copy()
    # CFBD's NFL id is never a join key here, but conflicting values for one
    # college identity still indicate an ambiguous source crosswalk.
    draft["_cfbd_nfl_id"] = draft["nflAthleteId"].astype("string").str.strip()
    draft_conflict_counts = draft.groupby("_college_espn_id", sort=False)["_cfbd_nfl_id"].nunique(
        dropna=False
    )
    ambiguous_draft_ids = set(draft_conflict_counts[draft_conflict_counts.gt(1)].index)
    # Historical CFBD rows reuse some collegeAthleteId values for different
    # people.  Exclude those keys rather than allowing a false identity join.
    draft = draft.loc[~draft["_college_espn_id"].isin(ambiguous_draft_ids)].copy()
    draft["nfl_espn_id"] = draft["_college_espn_id"]
    draft = draft.drop_duplicates(subset=["_college_espn_id", "nfl_espn_id"]).reset_index(drop=True)
    _assert_unique_mapping(draft, "_college_espn_id", "nfl_espn_id", "draft crosswalk")

    players = nfl_players.loc[:, [NFL_ESPN_ID_COLUMN, NFL_GSIS_ID_COLUMN]].copy()
    players["nfl_espn_id"] = players[NFL_ESPN_ID_COLUMN].astype("string").str.strip()
    players = players.loc[
        players["nfl_espn_id"].notna()
        & players["nfl_espn_id"].ne("")
        & players["nfl_espn_id"].ne("<NA>")
        & players["nfl_espn_id"].ne("0")
    ].copy()
    players["gsis_id"] = _canonical_id(players, NFL_GSIS_ID_COLUMN, "NFL gsis_id")
    players = players.drop_duplicates().reset_index(drop=True)
    _assert_unique_mapping(players, "nfl_espn_id", "gsis_id", "NFL player crosswalk")

    mapping = draft.merge(players, on="nfl_espn_id", how="left", validate="one_to_one")
    result = recruits.merge(
        mapping.loc[:, ["_college_espn_id", "nfl_espn_id", "gsis_id"]],
        on="_college_espn_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="_college_espn_id")
    result["nfl_espn_id"] = result["nfl_espn_id"].astype("string")
    result["gsis_id"] = result["gsis_id"].astype("string")

    usable_recruiting = recruits["_college_espn_id"].notna() & recruits["_college_espn_id"].ne("")
    draft_linked = result["nfl_espn_id"].notna()
    gsis_linked = result["gsis_id"].notna()
    audit = {
        "recruiting_rows": len(result),
        "recruiting_id_rows": int(usable_recruiting.sum()),
        "draft_college_ids": int(draft["_college_espn_id"].nunique()),
        "draft_rows": len(draft_picks),
        "draft_rows_with_college_id": len(draft),
        "ambiguous_draft_ids_excluded": len(ambiguous_draft_ids),
        "draft_to_nfl_espn_ids": int(mapping["nfl_espn_id"].notna().sum()),
        "nfl_player_rows": len(nfl_players),
        "nfl_player_rows_with_espn_id": len(players),
        "recruiting_to_draft_rate": _rate(usable_recruiting & draft_linked),
        "recruiting_to_gsis_rate": _rate(usable_recruiting & gsis_linked),
        "nfl_espn_without_gsis": int(mapping["gsis_id"].isna().sum()),
        "name_join_used": False,
        "cfbd_nflAthleteId_used": False,
    }
    return result, audit


def _rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def summarize_crosswalk_cohorts(crosswalk: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Report identity reach by recruiting cohort for eligibility decisions."""

    if "year" not in crosswalk.columns:
        return {}
    ids = crosswalk[RECRUITING_ID_COLUMN].astype("string").str.strip()
    usable = ids.notna() & ids.ne("") & ids.ne("<NA>") & ids.ne("0")
    linked = crosswalk[NFL_GSIS_ID_COLUMN].notna()
    cohorts: dict[str, dict[str, float | int]] = {}
    for cohort, positions in crosswalk.groupby("year", dropna=False, sort=True).groups.items():
        cohort_key = str(cohort)
        if cohort_key in {"nan", "<NA>", "None"}:
            cohort_key = "unknown"
        cohort_usable = usable.loc[positions]
        linked_count = int((cohort_usable & linked.loc[positions]).sum())
        usable_count = int(cohort_usable.sum())
        cohorts[cohort_key] = {
            "recruiting_rows": len(positions),
            "recruiting_id_rows": usable_count,
            "gsis_rows": linked_count,
            "recruiting_to_gsis_rate": linked_count / usable_count if usable_count else 0.0,
        }
    return cohorts
