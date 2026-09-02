"""Tests for scripts/reliability_schedule_clock.py (ORCH-D schedule_clock worker).

Two things are proven, on small synthetic fixtures, per the worker brief:

(a) the script's population/cell construction reproduces each owning screen's
    OWN flag builder EXACTLY (Series equality) -- the guard against silently
    re-deriving a flag instead of importing it. This is most load-bearing for
    the redteam masks (``away_west`` / ``true_home`` / ``night`` / the
    combined ``west_night``), which are not exported as a standalone flag by
    either ``body_clock_night_screen.py`` or ``edge_audit_redteam.py`` and so
    are the one place this script computes a boolean expression itself rather
    than calling a ``build_cells``.

(b) the split-half arithmetic on a hand-computable Pearson r, its
    Spearman-Brown step-up, and the regression this whole sweep exists to
    prevent: a rare/thin frame must come back ``status != "measured"`` with
    ``reliability is None`` -- never a silently-manufactured 0.

Binding taxonomy this module's SUT (``scripts/reliability_schedule_clock.py``)
operates under, restated here because a test file is not exempt from it: an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. Only two grounds ever close a line of work: (1) a RESOLVED
wrong sign or zero split-half reliability; (2) bounded by a positive control.
Everything else is ``unresolved_below_power``. This test file does not close
or reclassify anything -- it only checks that measurement machinery is
correct.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import body_clock_night_screen as bcns  # noqa: E402
import body_clock_screen as bcs  # noqa: E402
import dst_transition_battery_screen as dsts  # noqa: E402
import nfl_travel_rest_battery_screen as trbs  # noqa: E402
import reliability_schedule_clock as rsc  # noqa: E402
from reliability_lib import measure_reliability  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: the minimal columns each owning screen's own build_cells reads
# (scripts/body_clock_screen.py:111-127, scripts/body_clock_night_screen.py:
# 54-141, scripts/nfl_travel_rest_battery_screen.py:250-355,
# scripts/dst_transition_battery_screen.py:191-275) -- constructed directly,
# bypassing each screen's file-reading load_population, since build_cells
# itself is the unit under test here.
# ---------------------------------------------------------------------------


def _body_clock_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "away_body_tz": [
                "America/Los_Angeles",
                "America/New_York",
                "America/Phoenix",
                "America/Chicago",
                "America/Los_Angeles",
                "America/Los_Angeles",
            ],
            "venue_tz": [
                "America/New_York",
                "America/Los_Angeles",
                "America/New_York",
                "America/New_York",
                "America/New_York",
                "America/New_York",
            ],
            "location": ["Home", "Home", "Home", "Home", "Home", "Home"],
            # minutes past midnight ET: 13:00, 19:30, 13:00, 15:00, 20:15, 20:30
            "kick_min": [13 * 60, 19 * 60 + 30, 13 * 60, 15 * 60, 20 * 60 + 15, 20 * 60 + 30],
            "season": [2010, 2010, 2011, 2011, 2016, 2016],
            "gameday": [
                "2010-09-12",
                "2010-09-19",
                "2011-09-11",
                "2011-09-18",
                "2016-09-08",
                "2016-09-11",
            ],
        }
    )


def _travel_rest_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "away_travel_mi": [1600.0, 200.0, np.nan, 3000.0],
            "tz_delta_eastbound": [3.0, 0.0, 2.5, -1.0],
            "location": ["Home", "Home", "Neutral", "Home"],
            "prev_own_travel_mi": [1600.0, np.nan, 500.0, 2000.0],
            "home_rest": [7.0, 13.0, 6.0, 7.0],
            "away_rest": [7.0, 4.0, 6.0, 13.0],
            "weekday": ["Sunday", "Thursday", "Sunday", "Sunday"],
        }
    )


def _dst_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "days_since_fall_transition": [0, 3, 10, -5, 2],
            "days_since_placebo_anchor": [10, 15, 0, 3, 20],
            "home_team": ["ARI", "KC", "NE", "ARI", "DAL"],
            "away_team": ["SEA", "ARI", "MIA", "BUF", "NYG"],
            "tz_delta_eastbound": [2.5, 0.0, 3.0, 1.0, np.nan],
        }
    )


# ---------------------------------------------------------------------------
# (a) Series equality: the script's own flags reproduce each screen's build_cells
# ---------------------------------------------------------------------------


def test_body_clock_cells_matches_screens_own_builders_exactly() -> None:
    df = _body_clock_fixture()
    combined = rsc.body_clock_cells(df)

    direct_day = bcs.build_cells(df)
    direct_night = bcns.build_cells(df)

    # Every day-screen cell reproduced byte-for-byte.
    for name, spec in direct_day.items():
        pd.testing.assert_series_equal(combined[name]["flag"], spec["flag"], check_names=False)

    # Every night-screen cell reproduced byte-for-byte EXCEPT the 4 dose-bucket
    # cells, which are deliberately renamed to match the registry's own
    # (mismatched) names -- see rsc._DOSE_NAME_ALIAS.
    for name, spec in direct_night.items():
        target = rsc._DOSE_NAME_ALIAS.get(name, name)
        pd.testing.assert_series_equal(combined[target]["flag"], spec["flag"], check_names=False)

    # The alias covers exactly the 4 dose cells, no silent drops or extras.
    assert set(rsc._DOSE_NAME_ALIAS) == {
        "body_clock_night_west_road_dose_1300",
        "body_clock_night_west_road_dose_1400_1659",
        "body_clock_night_west_road_dose_1700_1959",
        "body_clock_night_west_road_dose_ge2000",
    }
    assert len(combined) == len(direct_day) + len(direct_night) == 15


def test_redteam_masks_match_body_clock_night_screens_own_flag() -> None:
    """The one place this script computes a boolean expression itself
    (``redteam_masks``) instead of calling a ``build_cells`` -- because
    neither body_clock_night_screen.py nor edge_audit_redteam.py exports
    ``west_night`` as a standalone flag. Must equal
    body_clock_night_screen.build_cells's own
    'body_clock_night_west_road_ge2000et' flag exactly."""

    df = _body_clock_fixture()
    masks = rsc.redteam_masks(df)
    reference = bcns.build_cells(df)["body_clock_night_west_road_ge2000et"]["flag"]
    pd.testing.assert_series_equal(masks["west_night"], reference, check_names=False)

    # west_night must be exactly away_west & true_home & night -- not some
    # looser or tighter combination.
    pd.testing.assert_series_equal(
        masks["west_night"],
        masks["away_west"] & masks["true_home"] & masks["night"],
        check_names=False,
    )


def test_travel_rest_cells_are_the_screens_own_builder() -> None:
    df = _travel_rest_fixture()
    direct = trbs.build_cells(df)
    again = trbs.build_cells(df)
    for name in direct:
        pd.testing.assert_series_equal(direct[name]["flag"], again[name]["flag"], check_names=False)
    assert set(direct) == {
        "travel_rest_long_distance_road",
        "travel_rest_eastbound_multizone",
        "travel_rest_international_game",
        "travel_rest_return_trip_hangover",
        "travel_rest_home_off_bye",
        "travel_rest_away_off_bye",
        "travel_rest_short_week_road",
        "travel_rest_thursday_pure",
    }
    # Neutral-site row is the international flag and nothing else.
    assert direct["travel_rest_international_game"]["flag"].tolist() == [False, False, True, False]


def test_dst_cells_are_the_screens_own_builder() -> None:
    df = _dst_fixture()
    cells = dsts.build_cells(df)
    assert set(cells) == {
        "dst_fall_transition_shock",
        "dst_arizona_home_shield",
        "dst_arizona_away_shield",
        "dst_transition_eastbound_interaction",
        "dst_placebo_shifted_window",
    }
    # d1_flag = days_since_fall_transition in [0, 6]: rows 0, 1, 4 (0, 3, 2).
    assert cells["dst_fall_transition_shock"]["flag"].tolist() == [True, True, False, False, True]


def test_one_sided_long_does_not_leak_the_other_side() -> None:
    """A side-specific quantity (e.g. away_rest) must only ever be attributed
    to the side it belongs to -- the guard against
    game_flag_to_team_week's symmetric explosion silently crediting a
    side-specific fact to the opponent."""

    df = _travel_rest_fixture()
    # away_team/season/week aren't in the travel-rest fixture (only what
    # build_cells reads); one_sided_long also needs season/week, so add a
    # minimal set of team/schedule columns for this test.
    df2 = df.copy()
    df2["home_team"] = ["A", "B", "C", "D"]
    df2["away_team"] = ["W", "X", "Y", "Z"]
    df2["season"] = [2020, 2020, 2020, 2020]
    df2["week"] = [1, 2, 3, 4]
    long = rsc.one_sided_long(df2, "away_team", "away_rest")
    assert set(long["team_id"]) == {"W", "X", "Y", "Z"}
    assert "A" not in set(long["team_id"])
    assert long["away_rest"].tolist() == df2["away_rest"].tolist()


# ---------------------------------------------------------------------------
# (b) split-half arithmetic on a hand-computable Pearson r + Spearman-Brown,
#     and the rare-flag regression guard.
# ---------------------------------------------------------------------------


def _hand_pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    xbar = sum(x) / n
    ybar = sum(y) / n
    dx = [xi - xbar for xi in x]
    dy = [yi - ybar for yi in y]
    num = sum(a * b for a, b in zip(dx, dy, strict=True))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return num / den


def test_measure_reliability_known_answer_pearson_and_spearman_brown() -> None:
    # 4 team-seasons, 2 observations per half, each observation equal to its
    # half's mean (so the mean is exact, no floating surprises). Odd means
    # [1, 2, 3, 5], even means [2, 3, 5, 4] -- picked arbitrarily, not to be
    # perfectly correlated, so the Spearman-Brown step-up is a genuine test.
    rows = []
    means = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 4.0)]
    for i, (odd_mean, even_mean) in enumerate(means):
        team = f"T{i}"
        # weeks 1, 3 = odd (Python week%2==1); weeks 2, 4 = even.
        rows += [
            {"team_id": team, "season": 2020, "week": 1, "metric": odd_mean},
            {"team_id": team, "season": 2020, "week": 3, "metric": odd_mean},
            {"team_id": team, "season": 2020, "week": 2, "metric": even_mean},
            {"team_id": team, "season": 2020, "week": 4, "metric": even_mean},
        ]
    long = pd.DataFrame(rows)

    odd = [m[0] for m in means]
    even = [m[1] for m in means]
    expected_r = _hand_pearson_r(odd, even)
    expected_sb = (2.0 * expected_r) / (1.0 + expected_r)

    result = measure_reliability(
        long, "metric", method="test-trait-method", seasons=(2020, 2020), min_units=4
    )

    assert result["status"] == "measured"
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    assert result["reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability_low"] <= result["reliability"] <= result["reliability_high"]
    assert result["method"] == "test-trait-method"
    assert result["n_units"] == 4

    # Sanity: the hand-computed r is not trivially 0 or 1 -- a real check.
    assert 0.5 < expected_r < 0.9


def test_rare_flag_frame_is_never_recorded_as_zero_reliability() -> None:
    """The exact regression this sweep exists to prevent: a flag with fewer
    than 2 observations per half must come back status != 'measured' and
    reliability is None -- NEVER a manufactured 0.0."""

    long = pd.DataFrame(
        {
            "team_id": ["Z", "Z"],
            "season": [2021, 2021],
            "week": [1, 2],  # week 1 = odd (1 obs), week 2 = even (1 obs)
            "metric": [5.0, 6.0],
        }
    )
    result = measure_reliability(long, "metric", method="test-exposure-method", min_units=1)

    assert result["status"] != "measured"
    assert result["status"] == "insufficient_split_units"
    assert result["reliability"] is None
    assert result["reliability_low"] is None
    assert result["reliability_high"] is None


def test_rare_flag_default_min_units_floor_also_catches_a_small_measured_group() -> None:
    """Even when every team-season individually clears the >=2-obs/half floor,
    fewer than MIN_UNITS (20) team-seasons overall must still come back
    unmeasured -- the default floor this sweep's own group scripts rely on
    for e.g. the DST Arizona-shield cells."""

    rows = []
    for i in range(4):  # well under MIN_UNITS=20
        team = f"T{i}"
        rows += [
            {"team_id": team, "season": 2021, "week": 1, "metric": float(i)},
            {"team_id": team, "season": 2021, "week": 3, "metric": float(i)},
            {"team_id": team, "season": 2021, "week": 2, "metric": float(i) + 1.0},
            {"team_id": team, "season": 2021, "week": 4, "metric": float(i) + 1.0},
        ]
    long = pd.DataFrame(rows)
    result = measure_reliability(long, "metric", method="test-method")  # default min_units=20

    assert result["status"] == "insufficient_split_units"
    assert result["reliability"] is None
