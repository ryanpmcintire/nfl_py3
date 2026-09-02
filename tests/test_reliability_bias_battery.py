"""The bias_battery reliability sweep's cell-to-hypothesis mapping and its
split arithmetic (ORCH-D).

Two guarantees, both of which a silent rename or a copied-and-drifted
estimator would break:

1. ``scripts/reliability_bias_battery.py`` measures the reliability of the
   SAME flag (for an ``exposure``-kind cell) or the same continuous parent
   quantity (for a ``trait``-kind cell) that
   ``scripts/nfl_bias_battery_screen.build_hypotheses`` builds that cell's
   flag from -- never a silently re-derived copy. ``HYPOTHESIS_SPEC``'s keys
   must exactly match what ``build_hypotheses`` returns (any drift there
   would mean a registry cell is being measured under a construct the battery
   never scored), and ``_exposure_frame``'s output must equal the battery's
   own flag Series pointwise, not a hand-rederived approximation of it.
2. ``reliability_lib.measure_reliability`` does the split arithmetic the
   sweep claims it does, on a frame whose answer is computable by hand --
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

import nfl_bias_battery_screen as battery  # noqa: E402
import reliability_bias_battery as sweep  # noqa: E402
import reliability_lib as rlib  # noqa: E402

# ---------------------------------------------------------------------------
# A small, hand-constructed long frame carrying every column
# ``build_hypotheses`` reads, so it can be called on a fixture this test
# fully controls rather than the real (large, slow-to-load) feature table.
# ---------------------------------------------------------------------------


def _synthetic_long_frame() -> pd.DataFrame:
    """Six team-game rows, hand-picked to exercise several hypotheses'
    flags in both directions (True and False) so the mapping test has
    something real to check.
    """

    return pd.DataFrame(
        {
            "team": ["SEA", "DAL", "GB", "KC", "SF", "PIT"],
            "opponent": ["NE", "PHI", "CHI", "LV", "ARI", "CLE"],
            "season": [2021, 2021, 2021, 2021, 2021, 2021],
            "week": [12, 16, 3, 8, 6, 14],
            "is_home": [False, True, True, True, True, True],
            "neutral_site": [0, 0, 0, 0, 0, 0],
            "gametime_hour": [13, 20, 13, 13, 21, 13],
            "weekday": ["Sunday", "Sunday", "Sunday", "Thursday", "Monday", "Sunday"],
            "own_rest": [7, 7, 7, 4, 7, 14],
            "opp_rest": [7, 7, 7, 7, 7, 7],
            "prior_games": [10, 14, 2, 6, 4, 12],
            "prior_win_pct": [0.20, 0.85, 0.50, 0.60, 0.75, 0.30],
            "opp_prior_games": [9, 12, 2, 6, 4, 12],
            "opp_prior_win_pct": [0.25, 0.60, 0.50, 0.10, 0.75, 0.60],
            "backup_qb_flag": [1.0, 0.0, np.nan, 0.0, 0.0, 1.0],
            "prior_score_margin": [-20.0, 10.0, 0.0, 3.0, -3.0, 20.0],
            "three_plus_road_flag": [True, False, False, False, False, False],
            "sandwich_flag": [False, False, True, False, False, False],
            "revenge_flag": [False, True, False, False, False, False],
            "team_is_favorite": [False, True, True, True, False, True],
            "team_spread": [-3.0, 6.5, 4.0, 12.5, -7.0, 3.0],
            "spread_line": [3.0, 6.5, 4.0, 12.5, 7.0, 3.0],
            "roof": ["outdoors", "dome", "outdoors", "outdoors", "outdoors", "outdoors"],
            "temp": [45.0, 70.0, 20.0, 55.0, 60.0, 25.0],
        }
    )


# ---------------------------------------------------------------------------
# 1. The cell -> hypothesis -> flag/column mapping agrees with the battery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected_hypothesis"),
    [
        ("bias_battery_bad_team_late", "bad_team_late"),
        ("bias_battery_backup_qb_start", "backup_qb_start"),
        ("bias_battery_backup_qb_start_opener", "backup_qb_start"),
        ("bias_battery_extra_rest_edge_opener", "extra_rest_edge"),
        ("bias_battery_west_coast_early_kickoff_opener", "west_coast_early_kickoff"),
    ],
)
def test_entry_maps_to_the_hypothesis_the_battery_actually_builds(
    entry: str, expected_hypothesis: str
) -> None:
    assert sweep.hypothesis_for(entry) == expected_hypothesis


def test_hypothesis_spec_exactly_covers_what_build_hypotheses_returns() -> None:
    """No stale/missing mapping: every key HYPOTHESIS_SPEC declares must be a
    hypothesis the battery actually builds, and vice versa -- the same
    assertion ``build_frame()`` makes at runtime, exercised here on a fast
    synthetic fixture instead of the real feature table.
    """

    hyps = battery.build_hypotheses(_synthetic_long_frame())
    assert len(hyps) == 17
    assert set(sweep.HYPOTHESIS_SPEC) == set(hyps)


def test_exposure_frame_reproduces_the_battery_flag_exactly() -> None:
    """The guard against silently re-deriving a flag: for an exposure-kind
    cell, the ``exposure`` column ``_exposure_frame`` builds must equal the
    battery's OWN flag Series pointwise, not a hand-recomputed approximation.
    """

    long_df = _synthetic_long_frame()
    hyps = battery.build_hypotheses(long_df)

    for name in ("sandwich_spot", "division_revenge_game", "three_plus_road_games"):
        assert sweep.HYPOTHESIS_SPEC[name]["kind"] == "exposure"
        flag = hyps[name]["flag"]
        frame = sweep._exposure_frame(long_df, flag, eligible=None)
        assert frame["exposure"].tolist() == flag.astype(float).tolist()
        # And it disagrees with an intentionally wrong mask, proving the
        # comparison above is not vacuously true.
        wrong = ~flag
        assert frame["exposure"].tolist() != wrong.astype(float).tolist()


def test_exposure_frame_respects_the_battery_eligibility_mask() -> None:
    """backup_qb_start's eligible=notna() rows must become NaN exposure
    (excluded from the reliability split), matching
    ``nfl_bias_battery_screen.score_hypothesis``'s own population restriction
    -- not silently defaulted to 0/False.
    """

    long_df = _synthetic_long_frame()
    hyps = battery.build_hypotheses(long_df)
    spec = hyps["backup_qb_start"]
    frame = sweep._exposure_frame(long_df, spec["flag"], spec["eligible"])

    # Row 2 (GB) has backup_qb_flag = NaN -> ineligible -> exposure NaN.
    assert math.isnan(frame.loc[2, "exposure"])
    assert frame.loc[0, "exposure"] == 1.0  # SEA: backup_qb_flag == 1.0
    assert frame.loc[1, "exposure"] == 0.0  # DAL: backup_qb_flag == 0.0


def test_an_unrelated_entry_name_is_rejected_not_guessed() -> None:
    with pytest.raises(AssertionError):
        sweep.hypothesis_for("weather_battery_extreme_cold")


# ---------------------------------------------------------------------------
# 2. The split arithmetic, on an answer computable by hand
# ---------------------------------------------------------------------------


def _long_frame(values: dict[tuple[str, int], list[float]], *, id_col: str) -> pd.DataFrame:
    """One row per (unit, season, week); weeks 1..n alternate odd/even."""

    rows = []
    for (unit, season), series in values.items():
        for index, value in enumerate(series, start=1):
            rows.append({id_col: unit, "season": season, "week": index, "metric": value})
    return pd.DataFrame(rows)


def test_trait_column_recovers_a_hand_computed_correlation_and_its_spearman_brown_step_up() -> None:
    """Exercises the SAME arithmetic ``build_frame()`` uses for a trait cell
    (e.g. rest_diff = own_rest - opp_rest, measured with unit_col="team"),
    on a frame whose odd/even half-means -- and therefore whose Pearson r and
    Spearman-Brown step-up -- are computable by hand.
    """

    odd_means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    even_means = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
    values = {}
    for index, (odd, even) in enumerate(zip(odd_means, even_means, strict=True)):
        values[(f"T{index}", 2020)] = [odd, even, odd, even]
    long = _long_frame(values, id_col="team")

    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])
    expected_sb = 2.0 * expected_r / (1.0 + expected_r)

    result = rlib.measure_reliability(
        long, "metric", method=rlib.METHOD_TRAIT, unit_col="team", n_boot=200, min_units=3
    )

    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == len(odd_means)
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    assert result["spearman_brown_full_length_reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability_low"] <= result["reliability"] <= result["reliability_high"]
    assert result["reliability_low"] >= -1.0 and result["reliability_high"] <= 1.0


def test_exposure_column_recovers_a_hand_computed_correlation_via_exposure_frame() -> None:
    """Exercises ``_exposure_frame`` end to end: build the exposure column
    from a flag Series the same way an exposure-kind cell does, then check
    the resulting split-half correlation against a value computable by hand.
    """

    # 3 team-seasons, 4 weeks each; flag is True on weeks 1,3 (odd) with
    # varying rates and weeks 2,4 (even) with a DIFFERENT, hand-picked rate,
    # so the odd/even team-season MEAN exposures are known in advance.
    long_df = pd.DataFrame(
        {
            "team": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "season": [2020] * 12,
            "week": [1, 2, 3, 4] * 3,
        }
    )
    # odd-week (1,3) flags per team: A=[T,T] B=[T,F] C=[F,F]
    # even-week (2,4) flags per team: A=[F,F] B=[T,F] C=[T,T]
    flag = pd.Series([True, False, True, False, True, True, False, False, False, True, False, True])
    frame = sweep._exposure_frame(long_df, flag, eligible=None)
    assert frame["exposure"].tolist() == flag.astype(float).tolist()

    result = rlib.measure_reliability(
        frame, "exposure", method=rlib.METHOD_EXPOSURE, unit_col="team_id", n_boot=200, min_units=3
    )
    odd_means = np.array([1.0, 0.5, 0.0])
    even_means = np.array([0.0, 0.5, 1.0])
    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])

    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == 3
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)


def test_seasons_restriction_uses_only_the_cells_own_window() -> None:
    """A cell's reliability must come from the seasons the cell was scored
    on -- mirrors ``bias_battery_*_opener`` entries measuring only 2020-2025
    while the base entry measures 2009-2025."""

    inside = {(f"T{i}", 2011): [float(i), float(i), float(i), float(i)] for i in range(6)}
    outside = {(f"T{i}", 2019): [0.0, 9.0, 0.0, 9.0] for i in range(6)}
    long = _long_frame({**inside, **outside}, id_col="team")

    restricted = rlib.measure_reliability(
        long,
        "metric",
        method=rlib.METHOD_TRAIT,
        unit_col="team",
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
    long = _long_frame(
        {("T0", 2020): [1.0, 2.0, 3.0, 4.0], ("T1", 2020): [2.0, 1.0, 4.0, 3.0]}, id_col="team"
    )
    result = rlib.measure_reliability(long, "metric", method=rlib.METHOD_TRAIT, unit_col="team")

    assert result["status"] != rlib.STATUS_MEASURED
    assert result["status"] == rlib.STATUS_INSUFFICIENT_UNITS
    assert result["reliability"] is None
    assert result["reliability_low"] is None and result["reliability_high"] is None


def test_too_few_eligible_exposure_rows_after_masking_returns_unmeasured() -> None:
    """backup_qb_start's eligibility mask can shrink a cell's usable rows
    well below the >=2-per-half floor; that must surface as unmeasured, not
    as a spurious number computed on a handful of leftover rows."""

    long_df = pd.DataFrame(
        {"team": ["A", "A", "B", "B"], "season": [2020, 2020, 2020, 2020], "week": [1, 2, 1, 2]}
    )
    flag = pd.Series([True, False, True, False])
    eligible = pd.Series([True, False, False, False])  # only 1 row survives
    frame = sweep._exposure_frame(long_df, flag, eligible)

    result = rlib.measure_reliability(
        frame, "exposure", method=rlib.METHOD_EXPOSURE, unit_col="team_id"
    )
    assert result["status"] != rlib.STATUS_MEASURED
    assert result["reliability"] is None
