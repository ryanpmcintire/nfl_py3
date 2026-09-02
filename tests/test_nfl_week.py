from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nfl_ats.nfl_week import pool_decision_cutoff, week_cycle_sunday


@pytest.mark.parametrize(
    ("kickoff", "expected"),
    [
        ("2025-09-04T00:20:00+00:00", "2025-09-04T00:20:00+00:00"),
        ("2025-09-07T17:00:00+00:00", "2025-09-07T17:00:00+00:00"),
        ("2025-09-08T00:20:00+00:00", "2025-09-07T20:00:00+00:00"),
        ("2025-09-09T00:15:00+00:00", "2025-09-07T20:00:00+00:00"),
        # DST ends on this Sunday: 16:00 America/New_York is 21:00 UTC.
        ("2025-11-04T01:15:00+00:00", "2025-11-02T21:00:00+00:00"),
    ],
)
def test_pool_decision_cutoff_uses_each_nfl_weeks_sunday_lock(kickoff: str, expected: str) -> None:
    actual = pool_decision_cutoff(datetime.fromisoformat(kickoff))
    assert actual == datetime.fromisoformat(expected)
    assert actual.tzinfo is UTC


def test_pool_decision_cutoff_rejects_timezone_naive_kickoff() -> None:
    with pytest.raises(ValueError, match="explicit timezone"):
        pool_decision_cutoff(datetime(2025, 9, 7, 13))


def test_week_cycle_sunday_treats_monday_as_part_of_the_prior_nfl_week() -> None:
    assert week_cycle_sunday(datetime(2025, 9, 8).date()).isoformat() == "2025-09-07"
