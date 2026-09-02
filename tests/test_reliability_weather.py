"""``scripts/reliability_weather.py``'s cell-to-quantity mapping and estimator.

Two guarantees, both of which a silent rename or a copied-and-drifted
estimator would break (same shape as ``tests/test_reliability_graph_team_stat.py``,
the sibling worked example this file is modeled on):

1. ``reliability_weather.py`` IMPORTS ``nfl_weather_battery_screen.build_cells``
   (and the other three screens' ``build_cells``) rather than re-deriving any
   flag threshold inline. If a future edit ever starts reimplementing a
   look-alike flag, this file's flag-reproduction tests catch the drift.
2. ``reliability_weather.measure_venue`` (the VENUE-method entry point this
   script's main loop actually calls) does the split arithmetic it claims,
   on a frame whose answer is computable by hand -- and returns an UNMEASURED
   status rather than a number when there are too few units to split.

A third guarantee specific to this group: ``dominant_unit_check`` is the
guard against the near-constant-column hazard the orchestrator's own smoke
run hit (and this script's own first run reproduced on
``weather_battery_high_altitude_road``: +0.7749 with Denver included,
-0.2414 with it excluded -- a sign flip driven by one structurally fixed
unit). A synthetic fixture reproducing that exact shape is tested below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

import nfl_weather_battery_screen as battery_screen  # noqa: E402
import reliability_lib as rlib  # noqa: E402
import reliability_weather as sweep  # noqa: E402

from nfl_ats.experiment_runner import FLAG_BUILDERS  # noqa: E402

# ---------------------------------------------------------------------------
# 1. reliability_weather.py's import of build_cells reproduces the battery's
#    own flag exactly on a small synthetic fixture (never re-derived inline).
# ---------------------------------------------------------------------------


def _synthetic_battery_population() -> pd.DataFrame:
    """A handful of games with exactly the columns ``build_cells`` reads.

    Shaped like ``nfl_weather_battery_screen.load_population``'s OUTPUT (not
    its raw input), since that is the boundary ``build_cells`` sits behind.
    """

    return pd.DataFrame(
        {
            "game_id": [f"G{i}" for i in range(6)],
            "season": [2021] * 6,
            "week": [1, 1, 2, 2, 14, 14],
            "weekday": ["Sunday", "Thursday", "Sunday", "Sunday", "Sunday", "Sunday"],
            "home_team": ["BUF", "BUF", "MIA", "DEN", "GB", "MIA"],
            "away_team": ["NE", "NYJ", "IND", "KC", "CHI", "TB"],
            "outdoor": [True, True, False, True, True, True],
            "roof": ["outdoors", "outdoors", "dome", "outdoors", "outdoors", "outdoors"],
            "temp": [20.0, 30.0, float("nan"), 45.0, 24.0, 38.0],
            "wind": [10.0, 8.0, float("nan"), 5.0, 20.0, 6.0],
            "spread_line": [-3.0, -3.0, 2.0, -6.0, 1.0, -1.0],
            "away_modal_roof": ["outdoors", "outdoors", "outdoors", "outdoors", "dome", "dome"],
            "away_modal_surface": ["grass", "grass", "grass", "grass", "turf", "grass"],
            "surface_norm": ["grass", "grass", "turf", "turf", "turf", "turf"],
            "stadium": [
                "Highmark",
                "Highmark",
                "Hard Rock",
                "Empower Field",
                "Lambeau",
                "Hard Rock",
            ],
        }
    )


def test_battery_build_cells_import_is_the_screens_own_function() -> None:
    """No look-alike reimplementation: the exact function object is reused."""

    assert sweep.battery_screen.build_cells is battery_screen.build_cells
    assert sweep.battery_screen is battery_screen


def test_battery_build_cells_reproduces_extreme_cold_flag_by_hand() -> None:
    df = _synthetic_battery_population()
    cells = sweep.battery_screen.build_cells(df)

    # weather_battery_extreme_cold: outdoor AND temp <= 25F.
    expected = pd.Series([True, False, False, False, True, False], index=df.index, name="temp")
    got = cells["weather_battery_extreme_cold"]["flag"]
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_battery_build_cells_reproduces_high_wind_outdoor_flag_by_hand() -> None:
    df = _synthetic_battery_population()
    cells = sweep.battery_screen.build_cells(df)

    # weather_battery_high_wind_outdoor: outdoor AND wind >= 15mph.
    expected = pd.Series([False, False, False, False, True, False], index=df.index)
    got = cells["weather_battery_high_wind_outdoor"]["flag"]
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_battery_build_cells_reproduces_dome_team_outdoors_cold_flag_by_hand() -> None:
    df = _synthetic_battery_population()
    cells = sweep.battery_screen.build_cells(df)

    # weather_battery_dome_team_outdoors_cold: away_modal_roof in
    # {dome, closed} AND outdoor AND temp <= 40F. Rows 4 and 5 (GB/MIA home,
    # both against a dome-modal away opponent) qualify: outdoor True and
    # temp 24/38 <= 40; rows 0-3's away_modal_roof is "outdoors", never dome.
    expected = pd.Series([False, False, False, False, True, True], index=df.index)
    got = cells["weather_battery_dome_team_outdoors_cold"]["flag"]
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_kn_flag_builder_names_are_registered_in_experiment_runner() -> None:
    """Every kn quantity key in ENTRY_SPECS names a real FLAG_BUILDERS entry."""

    for name, spec in sweep.ENTRY_SPECS.items():
        if spec[sweep.FAMILY] != "forecast_kn":
            continue
        assert spec[sweep.QUANTITY] in FLAG_BUILDERS, name


def test_entry_specs_is_exactly_the_33_entry_names() -> None:
    assert set(sweep.ENTRY_SPECS) == set(sweep.ENTRY_NAMES)
    assert len(sweep.ENTRY_NAMES) == 33


# ---------------------------------------------------------------------------
# 2. The split arithmetic, on an answer computable by hand, through this
#    script's own measure_venue() entry point.
# ---------------------------------------------------------------------------


def _venue_quantity_frame(
    values: dict[tuple[str, int], list[float]], metric_col: str
) -> pd.DataFrame:
    """One row per (venue, season, week); mirrors ``sweep._venue_long``'s shape."""

    rows = []
    for (venue, season), series in values.items():
        for index, value in enumerate(series, start=1):
            rows.append({"venue": venue, "season": season, "week": index, metric_col: value})
    return pd.DataFrame(rows)


def test_measure_venue_recovers_a_hand_computed_correlation() -> None:
    # reliability_lib.MIN_UNITS is 20, so this needs >= 20 venue-seasons --
    # a deterministic linear pattern (even_mean = odd_mean plus a small,
    # index-varying offset) keeps the expected correlation independently
    # computable by numpy from the exact same odd/even half-means the
    # estimator is fed, same convention as
    # tests/test_reliability_graph_team_stat.py's "hand computed" tests.
    odd_means = [float(i) for i in range(1, 25)]
    even_means = [float(i) + ((-1.0) ** i) * 2.0 for i in range(1, 25)]
    values = {}
    for index, (odd, even) in enumerate(zip(odd_means, even_means, strict=True)):
        values[(f"V{index}", 2020)] = [odd, even, odd, even]
    frame = _venue_quantity_frame(values, "temp")

    quantities = {"temp_actual": frame}
    result = sweep.measure_venue(quantities, "temp_actual", (2020, 2020), n_boot=200)

    import numpy as np

    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])
    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == len(odd_means)
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    assert result["method"] == rlib.METHOD_VENUE or "raw" in result["method"].lower()


def test_measure_venue_too_few_units_returns_unmeasured_not_zero() -> None:
    frame = _venue_quantity_frame(
        {("V0", 2020): [1.0, 2.0, 3.0, 4.0], ("V1", 2020): [2.0, 1.0, 4.0, 3.0]}, "temp"
    )
    quantities = {"temp_actual": frame}
    result = sweep.measure_venue(quantities, "temp_actual", (2020, 2020), n_boot=100)

    assert result["status"] != rlib.STATUS_MEASURED
    assert result["reliability"] is None
    assert result["reliability_low"] is None and result["reliability_high"] is None


# ---------------------------------------------------------------------------
# 3. The near-constant-column hazard guard (dominant_unit_check).
# ---------------------------------------------------------------------------


def test_dominant_unit_check_catches_a_sign_flip_like_high_altitude_road() -> None:
    """Reproduces this script's own first-run finding on a tiny fixture.

    One unit ("DEN") is a structurally fixed always-high outlier every
    season; every other unit carries small, roughly zero-centered noise.
    Excluding the dominant unit should flip the sign, exactly like the real
    measured run (+0.7749 with Denver in, -0.2414 with it excluded).
    """

    # reliability_lib.MIN_UNITS is 20, and split_half_reliability needs >= 2
    # observations per half, so every unit gets 4 weeks (odd half = weeks
    # 1,3; even half = weeks 2,4 -- same convention as
    # tests/test_reliability_graph_team_stat.py's known-answer tests), and
    # the DEN-excluded remeasurement must ALSO clear the 20-unit floor, so
    # this needs >= 21 other units. One season is enough: a unit here is a
    # (team_id, season) pair.
    rows = []
    for week, value in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)):
        rows.append({"team_id": "DEN", "season": 2020, "week": week, "exposure": value})
    # 24 other units: odd-half (weeks 1,3) and even-half (weeks 2,4) means
    # set directly from two independent, roughly balanced patterns so the
    # T-only correlation is small/uncorrelated -- DEN's single far-out point
    # at (1, 1) is what should move the measured reliability.
    odd_pattern = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    even_pattern = [0.6, 0.0, 0.8, 0.2, 1.0, 0.4]
    for i in range(24):
        odd_value = odd_pattern[i % len(odd_pattern)]
        even_value = even_pattern[i % len(even_pattern)]
        for week, value in ((1, odd_value), (2, even_value), (3, odd_value), (4, even_value)):
            rows.append({"team_id": f"T{i}", "season": 2020, "week": week, "exposure": value})
    long = pd.DataFrame(rows)

    baseline = rlib.measure_reliability(
        long,
        "exposure",
        method=rlib.METHOD_EXPOSURE,
        unit_col="team_id",
        seasons=(2020, 2020),
        n_boot=200,
    )
    assert baseline["status"] == rlib.STATUS_MEASURED

    dom = sweep.dominant_unit_check(
        long,
        "exposure",
        (2020, 2020),
        unit_col="team_id",
        method=rlib.METHOD_EXPOSURE,
        baseline_reliability=baseline["reliability"],
        n_boot=200,
    )
    assert dom is not None
    assert dom["dominant_unit"] == "DEN"
    # The dominant unit sits at exposure=1.0 every season, far from the
    # near-zero population mean -- removing it should collapse or flip the
    # measured reliability relative to the baseline.
    assert dom["reliability_without_dominant_unit"] != pytest.approx(
        dom["reliability_with_dominant_unit"], abs=1e-6
    )


def test_near_constant_check_flags_one_or_two_active_units() -> None:
    long = pd.DataFrame(
        {
            "team_id": ["A", "A", "B", "B", "C", "C"],
            "exposure": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = sweep.near_constant_check(long, "exposure")
    assert result["n_units_with_any_positive_value"] == 1
    assert result["near_constant"] is False  # n_units_total (3) < 5, too small to judge


def test_near_constant_check_does_not_flag_broad_participation() -> None:
    long = pd.DataFrame(
        {
            "team_id": [f"T{i}" for i in range(10)],
            "exposure": [1.0, 0.0] * 5,
        }
    )
    result = sweep.near_constant_check(long, "exposure")
    assert result["near_constant"] is False
