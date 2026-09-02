"""The reliability sweep's estimator and its cell-to-column mapping.

Two guarantees, both of which a silent rename or a copied-and-drifted
estimator would break:

1. ``scripts/reliability_graph_team_stat.py`` maps a registry cell to the SAME
   team-week column that ``scripts/graph_team_stat_screen.py`` builds that
   cell's graph-rating arm from. If those two ever disagree, the registry
   would carry a reliability belonging to a different construct than the
   effect recorded beside it.
2. ``scripts/reliability_lib.measure_reliability`` does the split arithmetic
   the sweep claims it does, on a frame whose answer is computable by hand --
   and, critically, returns an UNMEASURED status rather than a number when
   there are too few units to split. A NaN written through as a number would
   manufacture the appearance of a ``no_split_half_reliability`` closing
   ground out of nothing, which AGENTS.md's taxonomy forbids.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

import reliability_graph_team_stat as sweep  # noqa: E402
import reliability_lib as rlib  # noqa: E402
import reliability_map as relmap  # noqa: E402

# ---------------------------------------------------------------------------
# 1. The cell -> column mapping agrees with the screen that built the cells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected_column"),
    [
        ("graph_team_stat_def_sack_rate", "def_sack_rate"),
        ("graph_team_stat_off_cpoe", "off_cpoe"),
        ("graph_team_stat_injury_skill_epa_value_lost", "injury_skill_epa_value_lost"),
        ("graph_team_stat_pbp_off_pass_rate", "pbp_off_pass_rate"),
        ("graph_def_ypp_on_production", "def_yards_per_play"),
    ],
)
def test_cell_maps_to_the_column_the_screen_builds_its_arm_from(
    entry: str, expected_column: str
) -> None:
    assert sweep.family_for(entry) == expected_column


def test_every_mapped_column_is_a_family_the_screen_can_actually_build() -> None:
    """The screen builds an arm only for names ``discover_family_pairs`` returns.

    ``graph_team_stat_screen.build_arm_columns`` looks each family up in
    ``relmap.discover_family_pairs(...)`` and SKIPS anything missing, so a
    mapped column absent from that set would be a cell the screen never
    scored -- i.e. a mapping error, not a measurement.
    """

    columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        "result",
        "home_def_sack_rate",
        "away_def_sack_rate",
        "home_off_cpoe",
        "away_off_cpoe",
        "home_def_yards_per_play",
        "away_def_yards_per_play",
    ]
    dtypes = {
        column: (np.dtype(object) if "team" in column else np.dtype(float)) for column in columns
    }
    for name in ("game_id", "game_type"):
        dtypes[name] = np.dtype(object)
    families, _excluded = relmap.discover_family_pairs(columns, dtypes)

    for entry in (
        "graph_team_stat_def_sack_rate",
        "graph_team_stat_off_cpoe",
        "graph_def_ypp_on_production",
    ):
        assert sweep.family_for(entry) in families


def test_an_unrelated_name_maps_to_nothing_rather_than_guessing() -> None:
    assert sweep.family_for("weather_battery_extreme_cold") is None


# ---------------------------------------------------------------------------
# 2. The split arithmetic, on an answer computable by hand
# ---------------------------------------------------------------------------


def _long_frame(values: dict[tuple[str, int], list[float]]) -> pd.DataFrame:
    """One row per (team, season, week); weeks 1..n alternate odd/even."""

    rows = []
    for (team, season), series in values.items():
        for index, value in enumerate(series, start=1):
            rows.append({"team_id": team, "season": season, "week": index, "metric": value})
    return pd.DataFrame(rows)


def test_recovers_a_hand_computed_correlation_and_its_spearman_brown_step_up() -> None:
    # Four observations per team-season: weeks 1,3 (odd half) and 2,4 (even
    # half). Each team-season's half-means are set directly, so the Pearson r
    # between the odd and even half-means is computable by hand.
    odd_means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    even_means = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
    values = {}
    for index, (odd, even) in enumerate(zip(odd_means, even_means, strict=True)):
        # weeks 1,2,3,4 -> odd half sees weeks 1 and 3, even half weeks 2 and 4
        values[(f"T{index}", 2020)] = [odd, even, odd, even]
    long = _long_frame(values)

    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])
    expected_sb = 2.0 * expected_r / (1.0 + expected_r)

    result = rlib.measure_reliability(
        long, "metric", method=rlib.METHOD_TRAIT, n_boot=200, min_units=3
    )

    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == len(odd_means)
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    assert result["spearman_brown_full_length_reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability_low"] <= result["reliability"] <= result["reliability_high"]
    assert result["reliability_low"] >= -1.0 and result["reliability_high"] <= 1.0


def test_seasons_restriction_uses_only_the_cells_own_window() -> None:
    """A cell's reliability must come from the seasons the cell was scored on."""

    inside = {(f"T{i}", 2011): [float(i), float(i), float(i), float(i)] for i in range(6)}
    outside = {(f"T{i}", 2019): [0.0, 9.0, 0.0, 9.0] for i in range(6)}
    long = _long_frame({**inside, **outside})

    restricted = rlib.measure_reliability(
        long,
        "metric",
        method=rlib.METHOD_TRAIT,
        seasons=(2011, 2013),
        n_boot=200,
        min_units=3,
    )
    assert restricted["n_units"] == 6
    assert restricted["seasons"] == [2011, 2013]


# ---------------------------------------------------------------------------
# 3. An unmeasurable reliability is reported as unmeasured, never as a number
# ---------------------------------------------------------------------------


def test_too_few_units_returns_unmeasured_not_zero() -> None:
    long = _long_frame({("T0", 2020): [1.0, 2.0, 3.0, 4.0], ("T1", 2020): [2.0, 1.0, 4.0, 3.0]})
    result = rlib.measure_reliability(long, "metric", method=rlib.METHOD_TRAIT, n_boot=100)

    assert result["status"] == rlib.STATUS_INSUFFICIENT_UNITS
    assert result["reliability"] is None
    assert result["reliability_low"] is None and result["reliability_high"] is None


def test_a_constant_metric_returns_unmeasured_not_zero() -> None:
    long = _long_frame({(f"T{i}", 2020): [7.0] * 4 for i in range(30)})
    result = rlib.measure_reliability(
        long, "metric", method=rlib.METHOD_TRAIT, n_boot=100, min_units=3
    )

    assert result["status"] in (rlib.STATUS_CONSTANT, rlib.STATUS_INSUFFICIENT_UNITS)
    assert result["reliability"] is None


def test_a_strongly_negative_correlation_falls_back_to_the_raw_r_and_stays_on_scale() -> None:
    """Spearman-Brown is unbounded below; the recorded number never leaves [-1, 1].

    ``2r/(1+r)`` diverges as ``r -> -1``, so a strongly negative half-length
    correlation cannot be reported on the correlation scale after the step-up.
    The harness falls back to the raw Pearson r and says so in the method
    string, rather than emitting a value the registry validator would (rightly)
    refuse.
    """

    odd_means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    even_means = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    values = {}
    for index, (odd, even) in enumerate(zip(odd_means, even_means, strict=True)):
        values[(f"T{index}", 2020)] = [odd, even, odd, even]
    long = _long_frame(values)

    result = rlib.measure_reliability(
        long, "metric", method=rlib.METHOD_TRAIT, n_boot=200, min_units=3
    )

    assert result["status"] == rlib.STATUS_MEASURED
    assert result["pearson_r"] == pytest.approx(-1.0, abs=1e-9)
    assert math.isnan(result["spearman_brown_full_length_reliability"]) or (
        result["spearman_brown_full_length_reliability"] < -1.0
    )
    assert -1.0 <= result["reliability"] <= 1.0
    assert "raw" in result["method"].lower()


# ---------------------------------------------------------------------------
# 4. Flag exposure and the reported-only replication diagnostic
# ---------------------------------------------------------------------------


def test_game_flag_explodes_to_two_team_rows_per_game() -> None:
    games = pd.DataFrame(
        {
            "season": [2020, 2020],
            "week": [1, 2],
            "home_team": ["AAA", "BBB"],
            "away_team": ["CCC", "DDD"],
        }
    )
    flag = pd.Series([True, False])
    long = rlib.game_flag_to_team_week(games, flag)

    assert len(long) == 4
    assert set(long["team_id"]) == {"AAA", "BBB", "CCC", "DDD"}
    assert long.loc[long["team_id"] == "AAA", "exposure"].iloc[0] == 1.0
    assert long.loc[long["team_id"] == "DDD", "exposure"].iloc[0] == 0.0


def test_half_season_replication_reports_both_halves_and_never_a_reliability() -> None:
    games = pd.DataFrame(
        {
            "season": [2019] * 4 + [2020] * 4,
            "home_cover": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    flag = pd.Series([True, False, True, False, True, False, True, False])
    report = rlib.half_season_replication(games, flag, outcome_col="home_cover")

    assert set(report) >= {"odd_seasons", "even_seasons", "sign_agreement", "status", "note"}
    assert "reliability" not in report
    assert report["odd_seasons"]["gap_pts"] == pytest.approx(100.0)
    assert report["even_seasons"]["gap_pts"] == pytest.approx(100.0)
    assert report["sign_agreement"] is True
    # Only 2 flagged rows per half: honestly under-powered, and it says so.
    assert report["status"] == rlib.STATUS_INSUFFICIENT_UNITS
