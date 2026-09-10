from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from nfl_ats.surgical_gating import (
    VALUE_LOST_DIFF_COLUMNS,
    VALUE_LOST_MAGNITUDE_THRESHOLD,
    derive_conditional_median_threshold,
    gate_active_fraction,
    gate_by_value_lost_magnitude,
    raw_value_magnitude,
    surgical_predeclaration_summary,
)


def test_threshold_is_pinned_to_its_derived_value() -> None:
    assert pytest.approx(2.247849687590416, abs=1e-12) == VALUE_LOST_MAGNITUDE_THRESHOLD


def test_diff_columns_are_pregame_injury_value_lost_only() -> None:

    assert VALUE_LOST_DIFF_COLUMNS == (
        "diff_injury_skill_epa_value_lost",
        "diff_injury_defense_disruption_value_lost",
    )


def test_conditional_median_ignores_zero_mass() -> None:
    magnitude = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    assert float(magnitude.median()) == 0.0
    assert derive_conditional_median_threshold(magnitude) == pytest.approx(2.0)


def test_conditional_median_matches_manual_computation_on_asymmetric_tail() -> None:
    magnitude = pd.Series([0.0] * 10 + [1.0, 2.0, 4.0, 8.0])
    assert derive_conditional_median_threshold(magnitude) == pytest.approx(3.0)


def test_conditional_median_raises_with_no_nonzero_values() -> None:
    with pytest.raises(ValueError, match="no nonzero values"):
        derive_conditional_median_threshold(pd.Series([0.0, 0.0, 0.0]))


def test_conditional_median_accepts_plain_arrays() -> None:
    assert derive_conditional_median_threshold(np.array([0.0, 0.0, 5.0, 7.0])) == pytest.approx(6.0)


def test_raw_value_magnitude_sums_absolute_diff_columns() -> None:
    features = pd.DataFrame(
        {
            "diff_injury_skill_epa_value_lost": [1.5, -2.0, 0.0],
            "diff_injury_defense_disruption_value_lost": [-0.5, 3.0, 0.0],
        }
    )
    result = raw_value_magnitude(features)
    np.testing.assert_allclose(result.to_numpy(), [2.0, 5.0, 0.0])


def test_raw_value_magnitude_requires_numeric_columns() -> None:
    features = pd.DataFrame(
        {
            "diff_injury_skill_epa_value_lost": ["not_a_number"],
            "diff_injury_defense_disruption_value_lost": [1.0],
        }
    )
    with pytest.raises((ValueError, TypeError)):
        raw_value_magnitude(features)


def test_gate_defers_to_baseline_below_threshold_and_uses_candidate_above() -> None:
    baseline = np.array([0.3, 0.3, 0.3, 0.3])
    candidate = np.array([0.9, 0.9, 0.9, 0.9])
    magnitude = np.array([0.0, 1.0, 2.0, 3.0])
    gated = gate_by_value_lost_magnitude(baseline, candidate, magnitude, threshold=2.0)
    np.testing.assert_allclose(gated, [0.3, 0.3, 0.9, 0.9])


def test_gate_at_threshold_is_active_ge_not_gt() -> None:
    baseline = np.array([0.1])
    candidate = np.array([0.9])
    magnitude = np.array([2.0])
    gated = gate_by_value_lost_magnitude(baseline, candidate, magnitude, threshold=2.0)
    np.testing.assert_allclose(gated, [0.9])


def test_gate_zero_threshold_recovers_candidate_untouched() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.uniform(0.0, 1.0, size=50)
    candidate = rng.uniform(0.0, 1.0, size=50)
    magnitude = np.abs(rng.normal(size=50))
    gated = gate_by_value_lost_magnitude(baseline, candidate, magnitude, threshold=0.0)
    np.testing.assert_allclose(gated, candidate)


def test_gate_large_threshold_recovers_baseline_untouched() -> None:
    rng = np.random.default_rng(11)
    baseline = rng.uniform(0.0, 1.0, size=50)
    candidate = rng.uniform(0.0, 1.0, size=50)
    magnitude = np.abs(rng.normal(size=50))
    gated = gate_by_value_lost_magnitude(baseline, candidate, magnitude, threshold=1e9)
    np.testing.assert_allclose(gated, baseline)


def test_gate_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        gate_by_value_lost_magnitude(
            np.array([0.1]), np.array([0.9]), np.array([1.0]), threshold=-0.1
        )


def test_gate_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        gate_by_value_lost_magnitude(
            np.array([0.1, 0.2]), np.array([0.9]), np.array([1.0, 2.0]), threshold=1.0
        )


def test_gate_defaults_to_the_frozen_threshold() -> None:
    baseline = np.array([0.1])
    candidate = np.array([0.9])
    just_below = np.array([VALUE_LOST_MAGNITUDE_THRESHOLD - 1e-9])
    just_at = np.array([VALUE_LOST_MAGNITUDE_THRESHOLD])
    np.testing.assert_allclose(
        gate_by_value_lost_magnitude(baseline, candidate, just_below), baseline
    )
    np.testing.assert_allclose(
        gate_by_value_lost_magnitude(baseline, candidate, just_at), candidate
    )


def test_gate_active_fraction_matches_manual_count() -> None:
    magnitude = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert gate_active_fraction(magnitude, threshold=2.0) == pytest.approx(0.6)


def test_gate_active_fraction_at_default_threshold_is_between_zero_and_one() -> None:
    rng = np.random.default_rng(5)
    magnitude = np.abs(rng.normal(scale=3.0, size=1000))
    fraction = gate_active_fraction(magnitude)
    assert 0.0 <= fraction <= 1.0


def test_predeclaration_summary_carries_the_frozen_threshold_and_columns() -> None:
    summary = surgical_predeclaration_summary()
    assert summary["threshold"] == VALUE_LOST_MAGNITUDE_THRESHOLD
    assert summary["diff_columns"] == list(VALUE_LOST_DIFF_COLUMNS)
    assert "mod07_weak_signal_stack" in summary["inherits"]


_FORBIDDEN_SUBSTRINGS = ("result", "correct", "cover", "outcome", "actual", "margin_vs")


@pytest.mark.parametrize(
    "func",
    [
        gate_by_value_lost_magnitude,
        gate_active_fraction,
        raw_value_magnitude,
        derive_conditional_median_threshold,
    ],
)
def test_gating_functions_take_no_postgame_parameters(func) -> None:
    parameter_names = " ".join(inspect.signature(func).parameters.keys()).lower()
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in parameter_names, (
            f"{func.__name__} has a parameter suggesting postgame information: {parameter_names}"
        )


def test_value_lost_diff_columns_do_not_reference_outcomes() -> None:
    for column in VALUE_LOST_DIFF_COLUMNS:
        lowered = column.lower()
        for forbidden in _FORBIDDEN_SUBSTRINGS:
            assert forbidden not in lowered
