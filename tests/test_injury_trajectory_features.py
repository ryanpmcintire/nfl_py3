"""Leakage contracts for each CX14 feature family."""

import pandas as pd
import pytest

from nfl_ats.injury_trajectory_features import (
    LEADS,
    build_flags,
    prepare_revisions,
    split_half_reliability,
)


def row(player="qb", when="2024-09-06T19:00Z", practice="DNP", **kwargs):
    return {
        "season": 2024,
        "week": 1,
        "game_type": "REG",
        "team": "BUF",
        "gsis_id": player,
        "position": "QB",
        "report_status": "Questionable",
        "practice_status": practice,
        "date_modified": when,
        "report_primary_injury": "illness",
        **kwargs,
    }


def games(kickoff="2024-09-08T20:25Z"):
    return pd.DataFrame(
        [
            {
                "game_id": "game",
                "season": 2024,
                "week": 1,
                "home_team": "BUF",
                "away_team": "MIA",
                "kickoff": kickoff,
            }
        ]
    )


def inputs():
    return pd.DataFrame(
        [
            row(when="2024-09-04T19:00Z", practice="LP"),
            row(),
            row("p2"),
            row("p3"),
            row("rest", when="2024-09-04T18:00Z", report_primary_injury="rest"),
        ]
    )


def birthdays():
    return pd.DataFrame({"gsis_id": ["rest"], "birth_date": ["1994-09-08"]})


def test_four_constructs_and_exact_age_boundary():
    revisions, _ = prepare_revisions(inputs())
    flags = build_flags(games(), revisions, birth_dates=birthdays()).iloc[0]
    assert all(flags[lead] for lead in LEADS)
    assert flags["paired_trajectory_players"] == 1
    assert flags["illness_players"] == 3
    younger = birthdays().assign(birth_date="1994-09-09")
    assert not build_flags(games(), revisions, birth_dates=younger).iloc[0]["LEAD-11"]


@pytest.mark.parametrize("lead", LEADS)
def test_every_family_excludes_post_deadline_revisions(lead):
    revisions, _ = prepare_revisions(inputs())
    expected = build_flags(games(), revisions, birth_dates=birthdays())
    future = inputs().assign(
        date_modified="2024-09-08T20:00Z", practice_status="FP", report_status="Out"
    )
    combined, _ = prepare_revisions(pd.concat([inputs(), future], ignore_index=True))
    actual = build_flags(games(), combined, birth_dates=birthdays())
    pd.testing.assert_series_equal(actual[lead], expected[lead])
    assert actual.iloc[0].after_cutoff_excluded == 4


@pytest.mark.parametrize("lead", LEADS)
def test_proxy_rows_cannot_create_any_family_flag(lead):
    raw = inputs().assign(observed_at_is_proxy=True)
    revisions, coverage = prepare_revisions(raw)
    assert revisions.empty
    assert coverage["2024"]["proxy_excluded"] == len(raw)
    assert not build_flags(games(), revisions, birth_dates=birthdays())[lead].any()


def test_final_only_report_does_not_imply_trajectory_or_known_age():
    revisions, _ = prepare_revisions(pd.DataFrame([row()]))
    flags = build_flags(games(), revisions).iloc[0]
    assert not flags["LEAD-08_covered"]
    assert not flags["LEAD-11_covered"]
    assert flags["LEAD-09"]


def test_later_questionable_designation_does_not_leak_into_friday():
    raw = pd.DataFrame([row(report_status="Out"), row(when="2024-09-08T20:01Z")])
    revisions, _ = prepare_revisions(raw)
    assert not build_flags(games(), revisions).iloc[0]["LEAD-09"]


def test_proxy_basis_excluded_even_without_boolean_column():
    raw = pd.DataFrame([row(observed_at_basis="week_proxy")])
    revisions, coverage = prepare_revisions(raw)
    assert revisions.empty
    assert coverage["2024"]["proxy_excluded"] == 1


def test_missing_halves_are_not_zero_reliability():
    revisions, _ = prepare_revisions(pd.DataFrame([row()]))
    flags = build_flags(games(), revisions)
    assert split_half_reliability(flags, "LEAD-08")["correlation"] is None


@pytest.mark.parametrize("kickoff", ["2024-09-05T23:00Z", "2024-09-08T17:00Z", "2024-09-10T00:20Z"])
def test_cutoff_uses_own_kickoff_or_sunday_lock(kickoff):
    revisions, _ = prepare_revisions(inputs())
    flag = build_flags(games(kickoff), revisions).iloc[0]
    expected = min(pd.Timestamp(kickoff), pd.Timestamp("2024-09-08T20:00Z"))
    assert flag.cutoff == expected
