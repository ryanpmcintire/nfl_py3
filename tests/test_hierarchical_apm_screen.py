"""Release-blocking tests for the hierarchical APM screen (no network)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats.data import DataContractError
from scripts.hierarchical_apm_screen import (
    apply_hierarchical_shrinkage,
    paired_mse_comparison,
)


def test_shrinkage_endpoints_and_passthrough() -> None:
    flat = {"intercept": 1.0, "offense_player::A": 4.0, "offense_player::B": 0.0}
    lookup = {("A", 2020): "OFF_SKILL", ("B", 2020): "OFF_SKILL"}
    counts = {"A": 100, "B": 100}
    # k=0 recovers the flat fit exactly.
    exact = apply_hierarchical_shrinkage(flat, lookup, counts, 2020, shrinkage_k=0.0)
    assert exact["offense_player::A"] == pytest.approx(4.0)
    # Huge k collapses everyone to the unit mean (2.0).
    pooled = apply_hierarchical_shrinkage(flat, lookup, counts, 2020, shrinkage_k=1e12)
    assert pooled["offense_player::A"] == pytest.approx(2.0)
    assert pooled["offense_player::B"] == pytest.approx(2.0)
    # Intercept and team effects pass through untouched.
    assert pooled["intercept"] == pytest.approx(1.0)
    # Players without a unit pass through untouched.
    assert apply_hierarchical_shrinkage(flat, {}, counts, 2020)[
        "offense_player::A"
    ] == pytest.approx(4.0)
    with pytest.raises(DataContractError):
        apply_hierarchical_shrinkage(flat, lookup, counts, 2020, shrinkage_k=-1.0)


def test_offense_and_defense_pool_separately() -> None:
    flat = {
        "offense_player::A": 4.0,
        "offense_player::B": 0.0,
        "defense_player::A": -4.0,
    }
    lookup = {("A", 2020): "OFF_SKILL", ("B", 2020): "OFF_SKILL"}
    shrunk = apply_hierarchical_shrinkage(
        flat, lookup, {"A": 100, "B": 100}, 2020, shrinkage_k=100.0
    )
    # Weight 0.5 each way against the per-side unit mean (offense mean 2.0,
    # defense mean -4.0 from its lone member): sides never average together.
    assert shrunk["offense_player::A"] == pytest.approx(3.0)
    assert shrunk["offense_player::B"] == pytest.approx(1.0)
    assert shrunk["defense_player::A"] == pytest.approx(-4.0)


def test_paired_mse_comparison_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    actual = rng.normal(size=60)
    flat_pred = actual + rng.normal(scale=1.0, size=60)
    hier_pred = actual + rng.normal(scale=0.9, size=60)
    first = paired_mse_comparison(actual, flat_pred, hier_pred, seed=5, samples=200)
    second = paired_mse_comparison(actual, flat_pred, hier_pred, seed=5, samples=200)
    assert first["mse_delta_flat_minus_hier"] == pytest.approx(second["mse_delta_flat_minus_hier"])
    assert first["mse_delta_ci95"] == pytest.approx(second["mse_delta_ci95"])
    assert 0.0 <= first["probability_hierarchical_better"] <= 1.0
