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


# ---------------------------------------------------------------------------
# 1. Day-part bucketing from snapshot timestamps
# ---------------------------------------------------------------------------


def test_weekday_cutoff_wed_noon_lands_on_the_correct_wednesday() -> None:
    # A Sunday 13:00 ET kickoff (2024-09-08 is a real Sunday).
    kickoff = pd.Series([pd.Timestamp("2024-09-08T17:00:00Z")])  # 13:00 ET
    cutoff = _weekday_cutoff_et_utc(kickoff, WEDNESDAY_DAY_OFFSET, WEDNESDAY_HOUR_ET)
    # Wednesday before that Sunday is 2024-09-04; noon ET = 16:00 UTC (EDT, UTC-4).
    assert cutoff.iloc[0] == pd.Timestamp("2024-09-04T16:00:00Z")


def test_weekday_cutoff_sunday_1600_lands_on_the_same_sunday() -> None:
    kickoff = pd.Series([pd.Timestamp("2024-09-08T17:00:00Z")])  # Sunday 13:00 ET
    cutoff = _weekday_cutoff_et_utc(kickoff, SUNDAY_DAY_OFFSET, SUNDAY_HOUR_ET)
    # Same Sunday, 16:00 ET = 20:00 UTC (EDT, UTC-4).
    assert cutoff.iloc[0] == pd.Timestamp("2024-09-08T20:00:00Z")


def test_weekday_cutoff_anchors_off_a_thursday_kickoff_to_the_same_week() -> None:
    # A Thursday-night kickoff should still anchor to that week's Sunday, not
    # the following week's.
    thursday_kickoff = pd.Series([pd.Timestamp("2024-09-05T23:15:00Z")])  # Thu ~19:15 ET
    cutoff = _weekday_cutoff_et_utc(thursday_kickoff, WEDNESDAY_DAY_OFFSET, WEDNESDAY_HOUR_ET)
    assert cutoff.iloc[0] == pd.Timestamp("2024-09-04T16:00:00Z")


# ---------------------------------------------------------------------------
# 2. Deadline guard rejecting post-deadline / post-kickoff snapshots (leakage)
# ---------------------------------------------------------------------------


def test_deadline_guard_excludes_a_snapshot_observed_after_the_cutoff() -> None:
    game_id = "2024_01_AAA_BBB"
    commence = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)  # Sunday 13:00 ET
    before_cutoff = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 4, 15, 0, tzinfo=UTC),  # Wed 11:00 ET
        commence_at=commence,
        home_spread=-2.5,
    )
    after_cutoff = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 4, 17, 0, tzinfo=UTC),  # Wed 13:00 ET (after noon)
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
    commence = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)  # Sunday 13:00 ET kickoff
    pre_kickoff = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 8, 14, 0, tzinfo=UTC),  # Sunday 10:00 ET
        commence_at=commence,
        home_spread=-1.0,
    )
    post_kickoff_pre_nominal = _spread_quote(
        game_id=game_id,
        season=2024,
        week=1,
        observed_at=datetime(2024, 9, 8, 18, 0, tzinfo=UTC),  # Sunday 14:00 ET (after kickoff)
        commence_at=commence,
        home_spread=-7.0,  # a settlement-leaking value that must never surface
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
        observed_at=datetime(2024, 9, 5, 12, 0, tzinfo=UTC),  # Thu -- after Wed noon
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


# ---------------------------------------------------------------------------
# 3. Rising-total, stable-spread dog flag
# ---------------------------------------------------------------------------


def test_rising_total_dog_flags_only_rising_total_with_stable_spread() -> None:
    tue_open_total = pd.Series([44.0, 44.0, 44.0, 44.0])
    latest_total = pd.Series([46.5, 46.5, 45.0, 46.5])  # rises 2.5, 2.5, 1.0, 2.5
    tue_open_spread = pd.Series([3.0, 3.0, 3.0, -3.0])  # home dog, home dog, home dog, home fav
    latest_spread = pd.Series([3.0, 4.0, 3.0, -3.0])  # stable, moves 1.0, stable, stable
    production_home = pd.Series([False, False, False, False])

    pick, flagged = rising_total_dog_pick(
        tue_open_total, latest_total, tue_open_spread, latest_spread, production_home
    )

    # Row 0: total +2.5 (>=2.0), spread stable (0.0 < 0.5) -> flagged, home is
    # the dog (tue_open_spread > 0) -> pick home.
    assert flagged.iloc[0]
    assert pick.iloc[0]

    # Row 1: total +2.5 but spread moved 1.0 (>= 0.5) -> not flagged, falls
    # back to the production pick (False).
    assert not flagged.iloc[1]
    assert not pick.iloc[1]

    # Row 2: total only +1.0 (< 2.0) -> not flagged.
    assert not flagged.iloc[2]
    assert not pick.iloc[2]

    # Row 3: total +2.5, spread stable, but home is the FAVORITE
    # (tue_open_spread < 0) -> flagged, dog is away -> pick away (False).
    assert flagged.iloc[3]
    assert not pick.iloc[3]


def test_rising_total_dog_pick_em_tie_falls_back_to_production() -> None:
    tue_open_total = pd.Series([44.0])
    latest_total = pd.Series([47.0])
    tue_open_spread = pd.Series([0.0])  # pick 'em -- no underdog
    latest_spread = pd.Series([0.0])
    production_home = pd.Series([True])

    pick, flagged = rising_total_dog_pick(
        tue_open_total, latest_total, tue_open_spread, latest_spread, production_home
    )
    assert flagged.iloc[0]
    assert pick.iloc[0]  # falls back to the production pick (True)


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


# ---------------------------------------------------------------------------
# 4. Frozen-line grading (per-point-value metric grades on the declared
#    margin column, not any other candidate line)
# ---------------------------------------------------------------------------


def test_per_point_value_metric_grades_against_the_declared_margin_column() -> None:
    # 4 games, 1 week. Both candidate picks agree with each other and differ
    # from production on every game; move sizes are >= threshold for both
    # arms so every game is eligible in both.
    rows = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1],
            "_pick_a": [True, True, False, False],
            "_pick_c": [True, True, False, False],
            "production_pick": [False, False, True, True],
            "_move_a": [1.5, -1.5, 1.5, -1.5],
            "_move_c": [2.0, -2.0, 2.0, -2.0],
            # margin_a: matches _pick_a's own direction on every row, so the
            # candidate (which always disagrees with production) is always
            # CORRECT when graded on margin_a.
            "margin_a": [1.0, 1.0, -1.0, -1.0],
            # margin_b: the opposite sign pattern -- candidate is always
            # WRONG (production is correct) when graded on margin_b.
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
    # Candidate beats production on every game when graded on margin_a.
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
    # Same picks, different declared margin column -> opposite sign: proves
    # the metric grades on whichever margin_col it is told to, not a
    # hardcoded one (the frozen-line contract).
    assert result_b["per_point_value_wed"] < 0
    assert result_b["per_point_value_sun_am"] < 0


def test_per_point_value_metric_respects_the_eligibility_threshold() -> None:
    # Only games with |move| >= threshold count toward that arm's ratio.
    rows = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "_pick_a": [True, True],
            "_pick_c": [True, True],
            "production_pick": [False, False],
            "_move_a": [0.5, 1.5],  # only the second game clears 1.0
            "_move_c": [0.5, 0.5],  # neither game clears 1.0
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
    # wed arm has one eligible game and candidate beats production -> positive.
    assert result["per_point_value_wed"] > 0
    # sun_am arm has zero eligible games -> defined as neutral 0.0, not NaN.
    assert result["per_point_value_sun_am"] == 0.0
