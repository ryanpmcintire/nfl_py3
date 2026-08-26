"""Leakage regression test for the illness-designation battery
(``docs/illness_battery.md``): proves a report revision issued AFTER a
game's decision-cutoff (``nfl_ats.pick_refresh.pick_deadline``) can never
reach that game's as-of ``illness_count``/``active_illness_count`` features,
and that a team-week with no report before its cutoff resolves to MISSING,
never a leaked or defaulted-to-zero count.

Modeled on ``tests/test_respiratory_battery_leakage.py``'s canary pattern:
load the real script module (not a reimplementation) and feed it a synthetic
fixture with a deliberately extreme "future" value that must not leak.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_script("illness_battery_screen_test", "illness_battery_screen.py")


def _cutoffs(rows: list[tuple[int, int, str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["season", "week", "team", "cutoff_date"])
    frame["cutoff_date"] = pd.to_datetime(frame["cutoff_date"], utc=True)
    return frame


def test_add_is_illness_matches_any_of_the_four_reason_columns() -> None:
    frame = pd.DataFrame(
        {
            "report_primary_injury": ["Illness", "Knee", None, "Ankle"],
            "report_secondary_injury": [None, None, None, "Illness"],
            "practice_primary_injury": [None, None, "Illness (Non-COVID)", None],
            "practice_secondary_injury": [None, None, None, None],
        }
    )
    flagged = screen.add_is_illness(frame)["is_illness"]
    assert flagged.tolist() == [True, False, True, True]


def test_asof_resolution_never_sees_a_revision_issued_after_the_cutoff() -> None:
    """Canary: gsis A is NOT illness-flagged early in the week, then a much
    LATER revision (past the cutoff) flips it to illness. A cutoff strictly
    before that later revision must see the OLD (non-illness) state; a
    cutoff on/after it must see the flip."""

    early = pd.Timestamp("2023-10-04T12:00:00Z")  # Wednesday
    late = pd.Timestamp("2023-10-06T20:00:00Z")  # Friday, after the cutoff below

    injuries = screen.add_is_illness(
        pd.DataFrame(
            {
                "season": [2023, 2023],
                "week": [5, 5],
                "team": ["KC", "KC"],
                "gsis_id": ["A", "A"],
                "date_modified": [early, late],
                "report_primary_injury": [None, "Illness"],
                "report_secondary_injury": [None, None],
                "practice_primary_injury": [None, None],
                "practice_secondary_injury": [None, None],
                "report_status": ["Questionable", "Questionable"],
            }
        )
    )

    cutoff_before_late = late - pd.Timedelta(hours=1)
    cutoff_after_late = late + pd.Timedelta(hours=1)

    resolved_before = screen.resolve_asof_team_week(
        injuries, _cutoffs([(2023, 5, "KC", cutoff_before_late.isoformat())])
    )
    resolved_after = screen.resolve_asof_team_week(
        injuries, _cutoffs([(2023, 5, "KC", cutoff_after_late.isoformat())])
    )

    assert resolved_before.loc[0, "illness_count"] == 0, (
        "a revision issued AFTER the cutoff leaked into the as-of illness count"
    )
    assert resolved_after.loc[0, "illness_count"] == 1, (
        "the revision was not visible even once its own date_modified was on/before cutoff"
    )


def test_team_week_with_no_report_before_cutoff_is_missing_not_zero() -> None:
    """A team-week whose ONLY report row postdates the cutoff must be ABSENT
    from the resolved frame entirely -- never present with a defaulted-to-0
    illness_count. This is the property the screen script's
    ``attach_team_week_features`` relies on (``home_missing``/``away_missing``
    via ``.isna()`` after a left-merge) to exclude, not zero-fill, missing
    team-weeks from every cell."""

    only_revision = pd.Timestamp("2024-09-06T18:00:00Z")
    injuries = screen.add_is_illness(
        pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "team": ["SEA"],
                "gsis_id": ["Z"],
                "date_modified": [only_revision],
                "report_primary_injury": ["Illness"],
                "report_secondary_injury": [None],
                "practice_primary_injury": [None],
                "practice_secondary_injury": [None],
                "report_status": ["Questionable"],
            }
        )
    )
    cutoff_strictly_before = only_revision - pd.Timedelta(minutes=1)
    resolved = screen.resolve_asof_team_week(
        injuries, _cutoffs([(2024, 1, "SEA", cutoff_strictly_before.isoformat())])
    )
    assert resolved.empty, "a team-week with zero pre-cutoff reports must resolve to missing"


def test_a_null_date_modified_row_never_becomes_visible() -> None:
    """The measured 2025-season gap (date_modified entirely absent) and the
    2009 gap (99.6% absent): a NaT date_modified must never satisfy
    ``date_modified <= cutoff`` under any cutoff, however late."""

    injuries = screen.add_is_illness(
        pd.DataFrame(
            {
                "season": [2025],
                "week": [3],
                "team": ["DAL"],
                "gsis_id": ["Q"],
                # Matches nfl_ats.load_injuries()'s own conversion
                # (pd.to_datetime(..., errors="coerce", utc=True)) -- a
                # bare [pd.NaT] column would infer a tz-naive dtype instead,
                # which is not the real calling contract this function sees.
                "date_modified": pd.to_datetime([None], errors="coerce", utc=True),
                "report_primary_injury": ["Illness"],
                "report_secondary_injury": [None],
                "practice_primary_injury": [None],
                "practice_secondary_injury": [None],
                "report_status": ["Questionable"],
            }
        )
    )
    far_future_cutoff = pd.Timestamp("2030-01-01T00:00:00Z")
    resolved = screen.resolve_asof_team_week(
        injuries, _cutoffs([(2025, 3, "DAL", far_future_cutoff.isoformat())])
    )
    assert resolved.empty, "a NaT date_modified row leaked into an as-of feature"


def test_active_illness_excludes_out_and_doubtful_but_not_null_status() -> None:
    """``active_illness_count`` (docs/illness_battery.md section 3): illness
    AND report_status NOT IN {Out, Doubtful}. A null report_status does not
    confirm the player was ruled out, so it counts as active-eligible."""

    ts = pd.Timestamp("2022-11-04T18:00:00Z")
    injuries = screen.add_is_illness(
        pd.DataFrame(
            {
                "season": [2022, 2022, 2022, 2022],
                "week": [9, 9, 9, 9],
                "team": ["MIA", "MIA", "MIA", "MIA"],
                "gsis_id": ["P1", "P2", "P3", "P4"],
                "date_modified": [ts, ts, ts, ts],
                "report_primary_injury": ["Illness", "Illness", "Illness", "Illness"],
                "report_secondary_injury": [None, None, None, None],
                "practice_primary_injury": [None, None, None, None],
                "practice_secondary_injury": [None, None, None, None],
                "report_status": ["Out", "Doubtful", "Questionable", None],
            }
        )
    )
    cutoff = ts + pd.Timedelta(hours=1)
    resolved = screen.resolve_asof_team_week(
        injuries, _cutoffs([(2022, 9, "MIA", cutoff.isoformat())])
    )
    assert resolved.loc[0, "illness_count"] == 4
    assert resolved.loc[0, "active_illness_count"] == 2, (
        "expected Questionable + null-status players to count as active; "
        "Out and Doubtful must be excluded"
    )
