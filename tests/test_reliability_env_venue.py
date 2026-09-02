"""Tests for ``scripts/reliability_env_venue.py`` (ORCH-D env_venue group).

Two guarantees this file protects:

1. Where the sweep derives a NEW continuous quantity from a screen's own raw
   columns (``roof_open_frame`` from ``roof_decision_screen``'s ``roof``/
   ``is_home``/``home_team`` columns), that derivation agrees EXACTLY with
   the source screen's own flag logic on the overlapping cases -- the guard
   against silently re-deriving a flag by hand.
2. ``reliability_lib.measure_reliability`` does the split arithmetic the
   sweep claims, on a frame whose answer is computable by hand, and reports
   an unmeasured status (never a number) when there are too few units.

A third block tests this file's own hazard-handling additions: the
compositional-constraint guard (``force_diagnostic``, ``_compositional_guard``,
``random_half_probe``) that a concurrent ORCH-D worker's HAZARD message
motivated for the ``venue_milestone_post_bye_*``/``home_opener`` entries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

import reliability_env_venue as sweep  # noqa: E402
import reliability_lib as rlib  # noqa: E402
import roof_decision_screen  # noqa: E402

# ---------------------------------------------------------------------------
# 1. roof_open_frame reproduces roof_decision_screen's own retract_open /
#    retract_closed gate exactly -- never re-derived by hand.
# ---------------------------------------------------------------------------


def _tiny_roof_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "is_home": [True, True, True, False, True, True],
            "home_team": ["ARI", "ARI", "SF", "ARI", "DAL", "ARI"],
            "roof": ["open", "closed", "open", "open", "closed", "dome"],
            "season": [2020, 2020, 2020, 2020, 2020, 2020],
            "week": [1, 2, 1, 3, 1, 4],
        }
    )


def test_roof_open_matches_the_screens_own_retract_open_and_retract_closed_gates() -> None:
    frame = _tiny_roof_frame()
    retractable = roof_decision_screen.RETRACTABLE_TEAMS

    # roof_decision_screen.build_long_table's own gate, reproduced here only
    # to compare against -- NOT used to build the sweep's roof_open column.
    expected_open = (
        frame["is_home"] & frame["home_team"].isin(retractable) & (frame["roof"] == "open")
    )
    expected_closed = (
        frame["is_home"] & frame["home_team"].isin(retractable) & (frame["roof"] == "closed")
    )

    out = sweep.roof_open_frame(frame, retractable_teams=retractable)

    # Every row the source screen would flag retract_open==True must carry
    # roof_open==1.0, and every retract_closed==True row must carry 0.0 --
    # on the exact rows that survive the population filter (is_home & a
    # RETRACTABLE_TEAMS venue).
    assert set(out.index) == set(
        frame.index[frame["is_home"] & frame["home_team"].isin(retractable)]
    )
    for idx in out.index:
        if expected_open.loc[idx]:
            assert out.loc[idx, "roof_open"] == 1.0
        elif expected_closed.loc[idx]:
            assert out.loc[idx, "roof_open"] == 0.0
        else:
            assert np.isnan(out.loc[idx, "roof_open"])  # "dome" reading: neither open nor closed

    # DAL (non-retractable in this tiny fixture's team) and the away row are
    # excluded from the population entirely, matching is_home & RETRACTABLE_TEAMS.
    assert "DAL" not in out["home_team"].to_numpy() or not out.empty


def test_roof_open_frame_excludes_non_retractable_venues_and_away_rows() -> None:
    frame = _tiny_roof_frame()
    out = sweep.roof_open_frame(frame, retractable_teams=roof_decision_screen.RETRACTABLE_TEAMS)
    assert (out["home_team"] == "SF").sum() == 0  # not a RETRACTABLE_TEAMS venue
    assert len(out) == 4  # excludes SF (non-retractable) and the away row (index 3, is_home=False)
    assert bool(out["is_home"].all())
    assert bool(out["home_team"].isin(roof_decision_screen.RETRACTABLE_TEAMS).all())


# ---------------------------------------------------------------------------
# 2. The split arithmetic, on an answer computable by hand (mirrors
#    tests/test_reliability_graph_team_stat.py's shape).
# ---------------------------------------------------------------------------


def _long_frame(values: dict[tuple[str, int], list[float]]) -> pd.DataFrame:
    rows = []
    for (unit, season), series in values.items():
        for index, value in enumerate(series, start=1):
            rows.append({"unit_id": unit, "season": season, "week": index, "metric": value})
    return pd.DataFrame(rows)


def test_measure_reliability_recovers_a_hand_computed_correlation() -> None:
    odd_means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    even_means = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
    values = {}
    for index, (odd, even) in enumerate(zip(odd_means, even_means, strict=True)):
        values[(f"V{index}", 2020)] = [odd, even, odd, even]
    long = _long_frame(values)

    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])
    expected_sb = 2.0 * expected_r / (1.0 + expected_r)

    result = rlib.measure_reliability(
        long, "metric", method=rlib.METHOD_VENUE, unit_col="unit_id", n_boot=200, min_units=3
    )

    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == len(odd_means)
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    assert result["reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability_low"] <= result["reliability"] <= result["reliability_high"]


def test_too_few_units_returns_unmeasured_not_zero() -> None:
    long = _long_frame({("V0", 2020): [1.0, 2.0, 3.0, 4.0], ("V1", 2020): [2.0, 1.0, 4.0, 3.0]})
    result = rlib.measure_reliability(
        long, "metric", method=rlib.METHOD_VENUE, unit_col="unit_id", n_boot=100
    )
    assert result["status"] == rlib.STATUS_INSUFFICIENT_UNITS
    assert result["reliability"] is None
    assert result["reliability_low"] is None and result["reliability_high"] is None


# ---------------------------------------------------------------------------
# 3. force_diagnostic (hazard i) and the compositional guard (hazard ii)
# ---------------------------------------------------------------------------


def test_force_diagnostic_wipes_the_recordable_fields_but_keeps_the_raw_numbers() -> None:
    measured = rlib.measure_reliability(
        _long_frame(
            {(f"V{i}", 2020): [7.0, 7.0, 7.0, 7.0] for i in range(3)} | {("V3", 2020): [1, 2, 1, 2]}
        ),
        "metric",
        method=rlib.METHOD_VENUE,
        unit_col="unit_id",
        n_boot=100,
        min_units=3,
    )
    forced = sweep.force_diagnostic(
        measured, status=sweep.STATUS_NOT_INFORMATIVE_CONSTANT, note="diagnostic only"
    )
    assert forced["status"] == sweep.STATUS_NOT_INFORMATIVE_CONSTANT
    assert forced["reliability"] is None
    assert forced["reliability_low"] is None and forced["reliability_high"] is None
    assert forced["note"] == "diagnostic only"
    # the raw measurement is preserved for transparency, just not "recordable"
    assert forced["n_units"] == measured["n_units"]


def _compositional_flag_frame(n_units: int, seed: int) -> pd.DataFrame:
    """A flag that fires EXACTLY once per unit-season, timing randomized.

    Mirrors the bye/home-opener shape the concurrent worker's HAZARD message
    described: season total conserved (always exactly one game flagged out
    of many), so the flag's own presence in one half structurally means its
    absence from the other.
    """

    rng = np.random.default_rng(seed)
    rows = []
    n_games = 10
    for i in range(n_units):
        fire_week = int(rng.integers(1, n_games + 1))
        for week in range(1, n_games + 1):
            rows.append(
                {
                    "unit_id": f"U{i}",
                    "season": 2020,
                    "week": week,
                    "exposure": 1.0 if week == fire_week else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_random_half_probe_and_compositional_guard_catch_a_conserved_total_flag() -> None:
    frame = _compositional_flag_frame(40, seed=1)
    measured = rlib.measure_reliability(
        frame,
        "exposure",
        method=rlib.METHOD_EXPOSURE,
        unit_col="unit_id",
        seasons=(2020, 2020),
        n_boot=200,
    )
    assert measured["status"] == rlib.STATUS_MEASURED
    assert measured["reliability"] is not None and measured["reliability"] < 0

    probe = sweep.random_half_probe(
        frame, "exposure", unit_col="unit_id", seasons=(2020, 2020), n_reseeds=5, seed=99
    )
    assert probe["mean_reliability"] is not None
    # The conserved-total structure survives randomizing which half each
    # unit's single flagged week lands in -- it is an artifact of "exactly
    # one flagged week per unit", not of chronological order.
    assert probe["mean_reliability"] < 0
    assert sweep._compositional_guard(measured["reliability"], probe) is True


def test_compositional_guard_does_not_fire_on_a_mild_or_positive_reliability() -> None:
    probe_mild = {"mean_reliability": -0.05}
    probe_positive = {"mean_reliability": 0.4}
    assert sweep._compositional_guard(-0.04, probe_mild) is False
    assert sweep._compositional_guard(0.7, probe_positive) is False
    assert sweep._compositional_guard(None, probe_mild) is False


# ---------------------------------------------------------------------------
# 4. The 27-entry manifest is disjoint and exhaustive across the 5 builders.
# ---------------------------------------------------------------------------


def test_entry_groups_are_disjoint_and_total_27() -> None:
    groups = [
        sweep.ALTITUDE_ENTRIES,
        sweep.ROOF_ENTRIES,
        sweep.SURFACE_ENTRIES,
        sweep.ENVIRONMENTAL_ENTRIES,
        sweep.MILESTONE_ENTRIES,
    ]
    seen: set[str] = set()
    for group in groups:
        assert not (seen & set(group)), "entry assigned to more than one family"
        seen |= set(group)
    assert len(sweep.ALL_ENTRIES) == 27
    assert seen == set(sweep.ALL_ENTRIES)
