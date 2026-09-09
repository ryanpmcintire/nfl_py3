"""Tests for ``scripts/movement_leads_battery.py`` (LEAD-01/06/07).

Synthetic, deterministic fixtures only -- no real archive data is read here.
Covers: day-part cutoff bucketing from snapshot timestamps, the deadline
guard rejecting post-deadline/post-kickoff snapshots (leakage), the
rising-total-dog flag construction, and frozen-line grading in the
per-point-value metric.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts.movement_leads_battery import (  # noqa: E402
    SUNDAY_DAY_OFFSET,
    SUNDAY_HOUR_ET,
    WEDNESDAY_DAY_OFFSET,
    WEDNESDAY_HOUR_ET,
    _home_spread_at_weekday_cutoff,
    _weekday_cutoff_et_utc,
    per_point_value_diff_metric,
    rising_total_dog_pick,
)

_CONSENSUS_COLUMNS = [
    "nflverse_game_id",
    "season",
    "week",
    "decision_label",
    "capture_kind",
    "market",
    "outcome_side",
    "line",
    "price",
    "home_spread_line",
    "bookmaker_key",
    "observed_at_utc",
    "commence_time_utc",
    "snapshot_timestamp_utc",
]


def _spread_quote(
    *,
    game_id: str,
    season: int,
    week: int,
    observed_at: datetime,
    commence_at: datetime,
    home_spread: float,
) -> dict:
    return {
        "nflverse_game_id": game_id,
        "season": season,
        "week": week,
        "decision_label": "intraday_hourly",
        "capture_kind": "historical_backfill",
        "market": "spreads",
        "outcome_side": "HOME",
        "line": home_spread,
        "price": -110,
        "home_spread_line": home_spread,
        "bookmaker_key": "consensus_book",
        "observed_at_utc": pd.Timestamp(observed_at),
        "commence_time_utc": pd.Timestamp(commence_at),
        "snapshot_timestamp_utc": pd.Timestamp(observed_at),
    }


def test_weekday_cutoff_wed_noon_lands_on_the_correct_wednesday() -> None:
    kickoff = pd.Series([pd.Timestamp("2024-09-08T17:00:00Z")])
    cutoff = _weekday_cutoff_et_utc(kickoff, WEDNESDAY_DAY_OFFSET, WEDNESDAY_HOUR_ET)
    assert cutoff.iloc[0] == pd.Timestamp("2024-09-04T16:00:00Z")


def test_weekday_cutoff_sunday_1600_lands_on_the_same_sunday() -> None:
    kickoff = pd.Series([pd.Timestamp("2024-09-08T17:00:00Z")])
    cutoff = _weekday_cutoff_et_utc(kickoff, SUNDAY_DAY_OFFSET, SUNDAY_HOUR_ET)
    assert cutoff.iloc[0] == pd.Timestamp("2024-09-08T20:00:00Z")


def test_weekday_cutoff_anchors_off_a_thursday_kickoff_to_the_same_week() -> None:
    thursday_kickoff = pd.Series([pd.Timestamp("2024-09-05T23:15:00Z")])
    cutoff = _weekday_cutoff_et_utc(thursday_kickoff, WEDNESDAY_DAY_OFFSET, WEDNESDAY_HOUR_ET)
    assert cutoff.iloc[0] == pd.Timestamp("2024-09-04T16:00:00Z")


def test_deadline_guard_excludes_a_snapshot_observed_after_the_cutoff() -> None:
    game_id = "2024_01_AAA_BBB"
    commence = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
    before_cutoff = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 4, 15, 0, tzinfo=UTC),
        commence_at=commence,
        home_spread=-2.5,
    )
    after_cutoff = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 4, 17, 0, tzinfo=UTC),
        commence_at=commence,
        home_spread=-4.0,
    )
    corrected = pd.DataFrame([before_cutoff, after_cutoff])
    kickoff = pd.DataFrame(
        {
            "nflverse_game_id": [game_id],
            "season": [2024],
            "week": [1],
            "commence_time_utc": [pd.Timestamp(commence)],
            "week_first_commence_utc": [pd.Timestamp(commence)],
        }
    )

    spread, n_missing = _home_spread_at_weekday_cutoff(
        corrected,
        kickoff,
        day_offset_from_sunday=WEDNESDAY_DAY_OFFSET,
        hour=WEDNESDAY_HOUR_ET,
        column_name="wed_noon_home_spread",
    )

    assert n_missing == 0
    assert len(spread) == 1
    assert spread.iloc[0]["wed_noon_home_spread"] == pytest.approx(-2.5)


def test_deadline_guard_clips_to_kickoff_for_an_early_sunday_game() -> None:
    """A game kicking off BEFORE the nominal Sunday 16:00 ET cutoff must never
    use a quote observed between its own kickoff and that nominal cutoff --
    the per-game deadline is min(kickoff, Sunday 16:00 ET), not the nominal
    clock time alone.
    """

    game_id = "2024_01_CCC_DDD"
    commence = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
    pre_kickoff = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 8, 14, 0, tzinfo=UTC),
        commence_at=commence,
        home_spread=-1.0,
    )
    post_kickoff_pre_nominal = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 8, 18, 0, tzinfo=UTC),
        commence_at=commence,
        home_spread=-7.0,
    )
    corrected = pd.DataFrame([pre_kickoff, post_kickoff_pre_nominal])
    kickoff = pd.DataFrame(
        {
            "nflverse_game_id": [game_id],
            "season": [2024],
            "week": [1],
            "commence_time_utc": [pd.Timestamp(commence)],
            "week_first_commence_utc": [pd.Timestamp(commence)],
        }
    )

    spread, n_missing = _home_spread_at_weekday_cutoff(
        corrected,
        kickoff,
        day_offset_from_sunday=SUNDAY_DAY_OFFSET,
        hour=SUNDAY_HOUR_ET,
        column_name="sun_am_home_spread",
    )

    assert n_missing == 0
    assert len(spread) == 1
    assert spread.iloc[0]["sun_am_home_spread"] == pytest.approx(-1.0)


def test_deadline_guard_reports_missing_when_no_quote_precedes_the_cutoff() -> None:
    game_id = "2024_01_EEE_FFF"
    commence = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
    only_late_quote = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 5, 12, 0, tzinfo=UTC),
        commence_at=commence,
        home_spread=-3.0,
    )
    corrected = pd.DataFrame([only_late_quote])
    kickoff = pd.DataFrame(
        {
            "nflverse_game_id": [game_id],
            "season": [2024],
            "week": [1],
            "commence_time_utc": [pd.Timestamp(commence)],
            "week_first_commence_utc": [pd.Timestamp(commence)],
        }
    )

    spread, n_missing = _home_spread_at_weekday_cutoff(
        corrected,
        kickoff,
        day_offset_from_sunday=WEDNESDAY_DAY_OFFSET,
        hour=WEDNESDAY_HOUR_ET,
        column_name="wed_noon_home_spread",
    )

    assert n_missing == 1
    assert spread.empty


def test_rising_total_dog_flags_only_rising_total_with_stable_spread() -> None:
    tue_open_total = pd.Series([44.0, 44.0, 44.0, 44.0])
    latest_total = pd.Series([46.5, 46.5, 45.0, 46.5])
    tue_open_spread = pd.Series([3.0, 3.0, 3.0, -3.0])
    latest_spread = pd.Series([3.0, 4.0, 3.0, -3.0])
    production_home = pd.Series([False, False, False, False])

    pick, flagged = rising_total_dog_pick(
        tue_open_total, latest_total, tue_open_spread, latest_spread, production_home
    )

    assert flagged.iloc[0]
    assert pick.iloc[0]

    assert not flagged.iloc[1]
    assert not pick.iloc[1]

    assert not flagged.iloc[2]
    assert not pick.iloc[2]

    assert flagged.iloc[3]
    assert not pick.iloc[3]


def test_rising_total_dog_pick_em_tie_falls_back_to_production() -> None:
    tue_open_total = pd.Series([44.0])
    latest_total = pd.Series([47.0])
    tue_open_spread = pd.Series([0.0])
    latest_spread = pd.Series([0.0])
    production_home = pd.Series([True])

    pick, flagged = rising_total_dog_pick(
        tue_open_total, latest_total, tue_open_spread, latest_spread, production_home
    )
    assert flagged.iloc[0]
    assert pick.iloc[0]


def test_rising_total_dog_handles_missing_readings_without_flagging() -> None:
    tue_open_total = pd.Series([44.0])
    latest_total = pd.Series([np.nan])
    tue_open_spread = pd.Series([3.0])
    latest_spread = pd.Series([np.nan])
    production_home = pd.Series([False])

    pick, flagged = rising_total_dog_pick(
        tue_open_total, latest_total, tue_open_spread, latest_spread, production_home
    )
    assert not flagged.iloc[0]
    assert not pick.iloc[0]


def test_per_point_value_metric_grades_against_the_declared_margin_column() -> None:
    rows = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1],
            "_pick_a": [True, True, False, False],
            "_pick_c": [True, True, False, False],
            "production_pick": [False, False, True, True],
            "_move_a": [1.5, -1.5, 1.5, -1.5],
            "_move_c": [2.0, -2.0, 2.0, -2.0],
            "margin_a": [1.0, 1.0, -1.0, -1.0],
            "margin_b": [-1.0, -1.0, 1.0, 1.0],
        }
    )

    metric_correct_line = per_point_value_diff_metric(
        pick_a_col="_pick_a",
        move_a_col="_move_a",
        pick_c_col="_pick_c",
        move_c_col="_move_c",
        production_col="production_pick",
        margin_col="margin_a",
        threshold=1.0,
    )
    result_a = metric_correct_line(rows)
    assert result_a["per_point_value_wed"] > 0
    assert result_a["per_point_value_sun_am"] > 0

    metric_wrong_line = per_point_value_diff_metric(
        pick_a_col="_pick_a",
        move_a_col="_move_a",
        pick_c_col="_pick_c",
        move_c_col="_move_c",
        production_col="production_pick",
        margin_col="margin_b",
        threshold=1.0,
    )
    result_b = metric_wrong_line(rows)
    assert result_b["per_point_value_wed"] < 0
    assert result_b["per_point_value_sun_am"] < 0


def test_per_point_value_metric_respects_the_eligibility_threshold() -> None:
    rows = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "_pick_a": [True, True],
            "_pick_c": [True, True],
            "production_pick": [False, False],
            "_move_a": [0.5, 1.5],
            "_move_c": [0.5, 0.5],
            "margin": [1.0, 1.0],
        }
    )
    metric = per_point_value_diff_metric(
        pick_a_col="_pick_a",
        move_a_col="_move_a",
        pick_c_col="_pick_c",
        move_c_col="_move_c",
        production_col="production_pick",
        margin_col="margin",
        threshold=1.0,
    )
    result = metric(rows)
    assert result["per_point_value_wed"] > 0
    assert result["per_point_value_sun_am"] == 0.0
