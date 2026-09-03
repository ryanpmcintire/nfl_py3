"""Contract tests for the XLG-06 Stage-2 identity crosswalk."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.xlg06_crosswalk import (
    build_recruit_to_nfl_crosswalk,
    summarize_crosswalk_cohorts,
)


def _sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recruiting = pd.DataFrame(
        {"athleteId": ["101", "102", None], "name": ["A", "B", "C"], "year": [2024] * 3}
    )
    draft = pd.DataFrame(
        {
            "collegeAthleteId": [101, 102],
            "nflAthleteId": [9001, 9002],
        }
    )
    players = pd.DataFrame(
        {
            "espn_id": [101, 102],
            "gsis_id": ["00-AAA", "00-BBB"],
            "display_name": ["Wrong Name A", "Wrong Name B"],
        }
    )
    return recruiting, draft, players


def test_crosswalk_uses_ids_and_preserves_recruiting_rows() -> None:
    result, audit = build_recruit_to_nfl_crosswalk(*_sources())

    assert len(result) == 3
    assert result["gsis_id"].astype("string").tolist() == ["00-AAA", "00-BBB", pd.NA]
    assert audit["recruiting_rows"] == 3
    assert audit["recruiting_id_rows"] == 2
    assert audit["recruiting_to_gsis_rate"] == pytest.approx(2 / 3)
    assert audit["name_join_used"] is False
    assert audit["cfbd_nflAthleteId_used"] is False


def test_missing_nfl_identity_remains_auditable() -> None:
    recruiting, draft, players = _sources()
    players = players.iloc[[0]].copy()

    result, audit = build_recruit_to_nfl_crosswalk(recruiting, draft, players)

    assert result["gsis_id"].astype("string").tolist() == ["00-AAA", pd.NA, pd.NA]
    assert audit["recruiting_to_draft_rate"] == pytest.approx(2 / 3)
    assert audit["recruiting_to_gsis_rate"] == pytest.approx(1 / 3)


def test_conflicting_draft_identity_is_excluded_fail_closed() -> None:
    recruiting, draft, players = _sources()
    draft = pd.concat(
        [draft, pd.DataFrame({"collegeAthleteId": [101], "nflAthleteId": [9999]})],
        ignore_index=True,
    )

    result, audit = build_recruit_to_nfl_crosswalk(recruiting, draft, players)

    assert result["gsis_id"].astype("string").tolist() == [pd.NA, "00-BBB", pd.NA]
    assert audit["ambiguous_draft_ids_excluded"] == 1


def test_conflicting_gsis_identity_fails_closed() -> None:
    recruiting, draft, players = _sources()
    players = pd.concat(
        [players, pd.DataFrame({"espn_id": [101], "gsis_id": ["00-OTHER"]})],
        ignore_index=True,
    )

    with pytest.raises(DataContractError, match="conflicting mappings"):
        build_recruit_to_nfl_crosswalk(recruiting, draft, players)


def test_zero_source_ids_are_not_treated_as_matches() -> None:
    recruiting, draft, players = _sources()
    recruiting.loc[2, "athleteId"] = "0"
    draft = pd.concat(
        [draft, pd.DataFrame({"collegeAthleteId": [0], "nflAthleteId": [9003]})],
        ignore_index=True,
    )

    result, audit = build_recruit_to_nfl_crosswalk(recruiting, draft, players)

    assert result["gsis_id"].astype("string").tolist() == ["00-AAA", "00-BBB", pd.NA]
    assert audit["draft_rows"] == 3
    assert audit["draft_rows_with_college_id"] == 2


def test_missing_player_ids_are_excluded_from_mapping_but_counted() -> None:
    recruiting, draft, players = _sources()
    players = pd.concat(
        [
            players,
            pd.DataFrame({"espn_id": [None], "gsis_id": ["00-NO-ESPN"]}),
        ],
        ignore_index=True,
    )

    result, audit = build_recruit_to_nfl_crosswalk(recruiting, draft, players)

    assert len(result) == len(recruiting)
    assert audit["nfl_player_rows"] == 3
    assert audit["nfl_player_rows_with_espn_id"] == 2


def test_cohort_summary_is_explicit_about_usable_ids() -> None:
    recruiting, draft, players = _sources()
    result, _ = build_recruit_to_nfl_crosswalk(recruiting, draft, players)

    summary = summarize_crosswalk_cohorts(result)

    assert summary["2024"] == {
        "recruiting_rows": 3,
        "recruiting_id_rows": 2,
        "gsis_rows": 2,
        "recruiting_to_gsis_rate": 1.0,
    }


def test_cohort_summary_returns_empty_without_year() -> None:
    recruiting, draft, players = _sources()
    stripped = [
        frame.drop(columns="year", errors="ignore") for frame in (recruiting, draft, players)
    ]
    result, _ = build_recruit_to_nfl_crosswalk(*stripped)

    assert summarize_crosswalk_cohorts(result) == {}
