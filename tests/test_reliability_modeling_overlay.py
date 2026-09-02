"""The modeling_overlay sweep's parent mapping and its split arithmetic.

Two guarantees, both of which a silent rename or a drifted estimator breaks:

1. Where ``scripts/reliability_modeling_overlay.py`` says a registry cell's
   parent quantity is column X, X is the quantity the cell's own BUILDER uses.
   If those disagree, the registry would carry a reliability belonging to a
   different construct than the effect recorded beside it.
2. ``reliability_lib.measure_reliability`` does the split arithmetic the sweep
   claims -- on a frame whose answer is computable by hand -- and returns an
   UNMEASURED status rather than a number when too few units survive the
   split. A NaN written through as a number would manufacture the appearance
   of a ``no_split_half_reliability`` closing ground out of nothing, which
   AGENTS.md's binding taxonomy forbids.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src", REPO / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.append(str(_extra))

import reliability_lib as rlib  # noqa: E402
import reliability_map as relmap  # noqa: E402
import reliability_modeling_overlay as sweep  # noqa: E402


def _cell(entry: str) -> sweep.Cell:
    matches = [cell for cell in sweep.CELLS if cell.entry == entry]
    assert len(matches) == 1, f"{entry} appears {len(matches)} times in CELLS"
    return matches[0]


# ---------------------------------------------------------------------------
# 1. Parent mapping agrees with the builder that defines each cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        "overlay_loo_drop_spread_gap_zone_fade",
        "overlay_subset_production_plus_spread_gap_zone",
        "pick_conditioned_spread_gap_zone_pre2018",
    ],
)
def test_spread_gap_cells_map_to_the_quantity_the_overlay_actually_thresholds(
    entry: str,
) -> None:
    """The overlay flips on ``abs(spread_line)`` inside its own bounds."""

    from nfl_ats.spread_gap_zone_fade_overlay import (
        SPREAD_GAP_LOWER_BOUND,
        SPREAD_GAP_UPPER_BOUND,
        apply_spread_gap_zone_fade_overlay,
    )

    cell = _cell(entry)
    assert cell.disposition == sweep.DISPOSITION_TRAIT
    assert cell.metric == "abs_spread_line"

    inside = (SPREAD_GAP_LOWER_BOUND + SPREAD_GAP_UPPER_BOUND) / 2.0
    outside = SPREAD_GAP_UPPER_BOUND + 3.0
    predictions = pd.DataFrame(
        {
            "game_id": ["g_pos", "g_neg", "g_out"],
            "home_team": ["AAA", "BBB", "CCC"],
            "away_team": ["DDD", "EEE", "FFF"],
            "home_cover_probability": [0.62, 0.62, 0.62],
            # Same magnitude, opposite signs: a rule keyed on the ABSOLUTE
            # value must treat the first two identically.
            "spread_line": [inside, -inside, outside],
        }
    )
    result = apply_spread_gap_zone_fade_overlay(predictions, enabled=True)
    flipped = {flip.game_id for flip in result.flips}
    assert flipped == {"g_pos", "g_neg"}, (
        "the overlay's trigger is not a function of |spread_line| alone; the sweep's "
        f"parent mapping for {entry} would be measuring the wrong column"
    )


@pytest.mark.parametrize(
    ("entry", "expected_metric"),
    [
        ("best_pick_calibrated_probability_top1", "calibrated_probability"),
        ("best_pick_key_number_distance_top1", "key_number_distance"),
    ],
)
def test_best_pick_cells_map_to_a_signal_the_ranker_declares(
    entry: str, expected_metric: str
) -> None:
    import best_pick_ranker

    cell = _cell(entry)
    assert cell.disposition == sweep.DISPOSITION_TRAIT
    assert cell.metric == expected_metric
    assert expected_metric in best_pick_ranker.SIGNALS


def test_division_revenge_cells_map_to_the_v3_gap_column_the_map_discovers() -> None:
    """``gap_division_revenge`` is a real home/away pair, not a guessed name."""

    for entry in (
        "overlay_loo_drop_division_revenge_tilt",
        "overlay_subset_production_plus_division_revenge",
    ):
        cell = _cell(entry)
        assert cell.metric == "gap_division_revenge"
        assert cell.metric in relmap.V3_ONLY_PAIR_BASES


def test_movement_cells_map_to_the_screen_s_own_confidence_column() -> None:
    """``abs_predicted`` in the screen's artifact IS |predicted move|."""

    if not sweep.MOVEMENT_PER_GAME.is_file():
        pytest.skip("movement tilt screen artifact not present in this checkout")
    per_game = pd.read_csv(sweep.MOVEMENT_PER_GAME)
    assert {"abs_predicted", "predicted_close_minus_open"} <= set(per_game.columns)
    np.testing.assert_allclose(
        per_game["abs_predicted"].to_numpy(),
        per_game["predicted_close_minus_open"].abs().to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    )
    assert _cell("movement_direction_tilt_opener").metric == "abs_predicted"
    assert _cell("movement_direction_tilt_opener_variant2_top_quartile").metric == "abs_predicted"
    assert (
        _cell("movement_direction_tilt_opener_variant1_no_filter").metric
        == "predicted_move_team_signed"
    )


# ---------------------------------------------------------------------------
# 2. The split arithmetic, on an answer computable by hand
# ---------------------------------------------------------------------------


def _hand_frame(n_units: int = 24) -> tuple[pd.DataFrame, float]:
    """One frame whose odd/even unit means are known exactly.

    Each unit gets two odd-week observations equal to ``odd_i`` and two
    even-week observations equal to ``even_i``, so each half's mean IS that
    number and the split-half Pearson r is ``corrcoef(odd, even)``.
    """

    rng = np.random.default_rng(7)
    odd = rng.normal(size=n_units)
    even = 0.7 * odd + 0.3 * rng.normal(size=n_units)
    rows = []
    for index in range(n_units):
        for week, value in ((1, odd[index]), (3, odd[index]), (2, even[index]), (4, even[index])):
            rows.append(
                {
                    "team_id": f"T{index:02d}",
                    "season": 2020,
                    "week": week,
                    "planted": float(value),
                }
            )
    expected_r = float(np.corrcoef(odd, even)[0, 1])
    return pd.DataFrame(rows), expected_r


def test_measure_reliability_reproduces_a_hand_computed_pearson_r() -> None:
    frame, expected_r = _hand_frame()
    measured = rlib.measure_reliability(
        frame, "planted", method="unit test", seasons=(2020, 2020), n_boot=200
    )
    assert measured["status"] == rlib.STATUS_MEASURED
    assert measured["n_units"] == 24
    assert measured["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    # Spearman-Brown step-up: 2r / (1 + r), which is what gets recorded.
    assert measured["reliability"] == pytest.approx(
        (2.0 * expected_r) / (1.0 + expected_r), abs=1e-9
    )
    assert measured["reliability_low"] <= measured["reliability"] <= measured["reliability_high"]


def test_too_few_units_is_reported_unmeasured_and_never_as_a_number() -> None:
    frame, _expected = _hand_frame(n_units=5)
    measured = rlib.measure_reliability(
        frame, "planted", method="unit test", seasons=(2020, 2020), n_boot=200
    )
    assert measured["status"] != rlib.STATUS_MEASURED
    assert measured["status"] == rlib.STATUS_INSUFFICIENT_UNITS
    assert measured["reliability"] is None
    assert measured["reliability_low"] is None
    assert measured["reliability_high"] is None


def test_random_halves_check_separates_a_trait_from_a_conserved_total() -> None:
    """The conserved-quantity guard's discriminator actually discriminates."""

    # A real trait: a stable per-unit level plus per-observation noise. Enough
    # observations per unit that a RANDOM half-split still clears the
    # >=2-per-half floor the estimator enforces.
    n_units, n_obs = 30, 12
    rng = np.random.default_rng(3)
    levels = rng.normal(scale=1.0, size=n_units)
    rows = []
    for index in range(n_units):
        for week in range(1, n_obs + 1):
            rows.append(
                {
                    "team_id": f"T{index:02d}",
                    "season": 2020,
                    "week": week,
                    "planted": float(levels[index] + rng.normal(scale=0.5)),
                }
            )
    real = sweep.random_halves_check(
        pd.DataFrame(rows), "planted", unit_col="team_id", seasons=(2020, 2020), reseeds=3
    )
    assert real["mean"] is not None and real["mean"] > 0.0

    # A conserved total: each unit's observations sum to a fixed budget, so
    # more in one half mechanically forces less in the other.
    rng = np.random.default_rng(11)
    rows = []
    for index in range(n_units):
        parts = rng.dirichlet(np.ones(n_obs)) * 100.0
        for week, value in zip(range(1, n_obs + 1), parts, strict=True):
            rows.append(
                {
                    "team_id": f"T{index:02d}",
                    "season": 2020,
                    "week": int(week),
                    "planted": float(value),
                }
            )
    conserved = sweep.random_halves_check(
        pd.DataFrame(rows), "planted", unit_col="team_id", seasons=(2020, 2020), reseeds=3
    )
    assert conserved["mean"] is not None
    assert conserved["mean"] <= sweep.COMPOSITIONAL_RANDOM_HALVES_MAX


# ---------------------------------------------------------------------------
# 3. Group bookkeeping
# ---------------------------------------------------------------------------


def test_every_cell_is_listed_once_and_carries_a_reason() -> None:
    names = [cell.entry for cell in sweep.CELLS]
    assert len(names) == len(set(names))
    assert len(names) == 59
    for cell in sweep.CELLS:
        assert cell.parent.strip(), f"{cell.entry} has no parent/reason sentence"
        assert cell.provenance.strip(), f"{cell.entry} has no provenance"
        if cell.disposition == sweep.DISPOSITION_NO_TRAIT:
            assert cell.frame is None and cell.metric is None
        else:
            assert cell.frame and cell.metric and cell.method


def test_exposure_cells_carry_the_exposure_method_string() -> None:
    """An exposure number must never be labelled as a trait reliability."""

    for cell in sweep.CELLS:
        if cell.disposition == sweep.DISPOSITION_EXPOSURE:
            assert cell.method == rlib.METHOD_EXPOSURE
        elif cell.disposition == sweep.DISPOSITION_TRAIT:
            assert cell.method == rlib.METHOD_TRAIT
