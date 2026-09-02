"""``nfl-ats weak-signals set-reliability`` writes ONLY the reliability field.

Reliability is one of only two admissible closing grounds (AGENTS.md: "wrong
sign, or the trait has no split-half reliability"), yet most NFL entries carry
``reliability: null`` -- so the ground can be neither used nor ruled out. The
sweep that fills them in must not become a back door for rewriting recorded
measurements, so these tests pin the guarantee byte-for-byte: every other
field survives untouched, an out-of-range or non-finite value is refused, and
attaching a low reliability never reclassifies anything on its own.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from nfl_ats.weak_signals import (
    WEAK_SIGNAL_REGISTRY_VERSION,
    WeakSignalError,
    registry_from_payload,
    set_reliability,
)


def _signal(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "recorded_at": "2026-08-21",
        "description": "a small measured effect",
        "source": "artifacts/some_screen/20260821T000000Z/results.json",
        "effect": 0.10,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "classification_evidence": "predeclared battery cell, mined multiplicity",
        "closing_ground": None,
        "league": "nfl",
        "seasons": [2009, 2025],
        "standard_error": 0.30,
        "interval": [-0.44, 0.64],
        "probability_positive": 0.64,
        "sample_games": 4317,
        "sample_blocks": 294,
        "reliability": None,
        "family": None,
        "notes": "original recording note",
        "plain_summary": "A short sentence a fan can read on its own.",
        "category": "environment",
    }
    body.update(overrides)
    return body


def _payload(**signals: dict[str, Any]) -> dict[str, Any]:
    return {"version": WEAK_SIGNAL_REGISTRY_VERSION, "notes": ["test ledger"], "signals": signals}


_METHOD = "team-season odd/even-week split-half of wind_at_venue, Spearman-Brown corrected"
_SOURCE = "artifacts/reliability_sweep/weather/20260901T000000Z/results.json"


def test_sets_only_reliability_and_appends_one_audit_note() -> None:
    registry = registry_from_payload(_payload(weather_cell=_signal()))
    original = registry.signals["weather_cell"]

    updated_registry = set_reliability(
        registry,
        "weather_cell",
        reliability=0.6421,
        reliability_low=0.5012,
        reliability_high=0.7503,
        method=_METHOD,
        source=_SOURCE,
        reason="every cell in this battery thresholds the same venue-wind trait",
        changed_at="2026-09-01T12:00:00+00:00",
    )
    updated = updated_registry.signals["weather_cell"]

    assert updated.reliability == pytest.approx(0.6421)
    assert original.notes in updated.notes
    assert "reliability set: None -> 0.6421" in updated.notes
    assert "95% [0.5012, 0.7503]" in updated.notes
    assert _METHOD in updated.notes
    assert _SOURCE in updated.notes
    assert "2026-09-01T12:00:00+00:00" in updated.notes

    # Everything else is carried over byte-for-byte, including ``source``:
    # the artifact holding the reliability goes in the audit line, never over
    # the entry's own provenance.
    assert updated.effect == original.effect
    assert updated.effect_units == original.effect_units
    assert updated.interval == original.interval
    assert updated.standard_error == original.standard_error
    assert updated.probability_positive == original.probability_positive
    assert updated.sample_games == original.sample_games
    assert updated.sample_blocks == original.sample_blocks
    assert updated.classification == original.classification
    assert updated.classification_evidence == original.classification_evidence
    assert updated.closing_ground == original.closing_ground
    assert updated.family == original.family
    assert updated.plain_summary == original.plain_summary
    assert updated.category == original.category
    assert updated.league == original.league
    assert updated.seasons == original.seasons
    assert updated.source == original.source
    assert updated.description == original.description
    assert updated.recorded_at == original.recorded_at
    assert updated.name == original.name

    # The registry passed in is untouched (immutability).
    assert registry.signals["weather_cell"].reliability is None


def test_a_low_reliability_does_not_reclassify_or_close_the_entry() -> None:
    """AGENTS.md: recording is not closing.

    A measured reliability below ``NO_SPLIT_HALF_RELIABILITY_MAX`` makes an
    entry a *candidate* for the ``no_split_half_reliability`` ground. Acting on
    that stays a separate, explicit decision, so this command must leave the
    classification and closing_ground exactly as it found them.
    """

    registry = registry_from_payload(_payload(flat_cell=_signal()))
    updated = set_reliability(
        registry,
        "flat_cell",
        reliability=0.02,
        reliability_low=-0.19,
        reliability_high=0.24,
        method=_METHOD,
        source=_SOURCE,
        reason="measured on the cell's own seasons",
    ).signals["flat_cell"]

    assert updated.reliability == pytest.approx(0.02)
    assert updated.classification == "unresolved_below_power"
    assert updated.closing_ground is None


def test_appends_to_empty_notes_without_a_leading_blank_line() -> None:
    registry = registry_from_payload(_payload(alpha=_signal(notes="")))
    updated = set_reliability(
        registry,
        "alpha",
        reliability=0.5,
        reliability_low=0.3,
        reliability_high=0.7,
        method=_METHOD,
        source=_SOURCE,
        reason="fills a null",
        changed_at="2026-09-01T12:00:00+00:00",
    ).signals["alpha"]

    assert updated.notes.startswith("[2026-09-01T12:00:00+00:00] reliability set:")
    assert "\n\n" not in updated.notes


def test_overwriting_an_existing_reliability_records_the_previous_value() -> None:
    registry = registry_from_payload(_payload(alpha=_signal(reliability=0.31)))
    updated = set_reliability(
        registry,
        "alpha",
        reliability=0.44,
        reliability_low=0.20,
        reliability_high=0.68,
        method=_METHOD,
        source=_SOURCE,
        reason="remeasured on the cell's own seasons rather than the full archive",
    ).signals["alpha"]

    assert updated.reliability == pytest.approx(0.44)
    assert "reliability set: 0.31 -> 0.4400" in updated.notes


def test_refuses_an_unknown_entry() -> None:
    registry = registry_from_payload(_payload(alpha=_signal()))
    with pytest.raises(WeakSignalError, match="No recorded signal named 'beta'"):
        set_reliability(
            registry,
            "beta",
            reliability=0.5,
            reliability_low=0.3,
            reliability_high=0.7,
            method=_METHOD,
            source=_SOURCE,
            reason="typo in the name",
        )


@pytest.mark.parametrize(
    ("reliability", "low", "high"),
    [
        (1.4, 1.2, 1.6),
        (-1.4, -1.6, -1.2),
        (0.5, -1.2, 0.7),
        (0.5, 0.3, 1.7),
    ],
)
def test_refuses_values_outside_the_correlation_scale(
    reliability: float, low: float, high: float
) -> None:
    registry = registry_from_payload(_payload(alpha=_signal()))
    with pytest.raises(WeakSignalError, match=r"outside \[-1, 1\]"):
        set_reliability(
            registry,
            "alpha",
            reliability=reliability,
            reliability_low=low,
            reliability_high=high,
            method=_METHOD,
            source=_SOURCE,
            reason="out of range",
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_refuses_a_non_finite_reliability(bad: float) -> None:
    """An unmeasurable reliability is reported as unmeasured, never as a number.

    ``split_half_reliability`` returns NaN when a construct has too few
    usable units to split. Writing that through as if it were a measurement
    would manufacture the appearance of evidence where none exists, and --
    because reliability is a closing ground -- could later be cited to close
    a line of work on nothing at all.
    """

    registry = registry_from_payload(_payload(alpha=_signal()))
    with pytest.raises(WeakSignalError, match="must be a finite number"):
        set_reliability(
            registry,
            "alpha",
            reliability=bad,
            reliability_low=-0.2,
            reliability_high=0.2,
            method=_METHOD,
            source=_SOURCE,
            reason="not measurable",
        )


def test_refuses_an_interval_that_does_not_contain_the_point_estimate() -> None:
    registry = registry_from_payload(_payload(alpha=_signal()))
    with pytest.raises(WeakSignalError, match="does not contain the point estimate"):
        set_reliability(
            registry,
            "alpha",
            reliability=0.80,
            reliability_low=0.10,
            reliability_high=0.40,
            method=_METHOD,
            source=_SOURCE,
            reason="mismatched interval",
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [("method", "method is required"), ("source", "source is required"), ("reason", "reason is")],
)
def test_requires_method_source_and_reason(field: str, message: str) -> None:
    """Which quantity was measured is not optional metadata.

    A trait's split-half correlation and a per-game flag's exposure
    reliability are different quantities on the same [-1, 1] scale; without
    the method recorded alongside the number, a later reader cannot tell them
    apart and would compare them.
    """

    registry = registry_from_payload(_payload(alpha=_signal()))
    kwargs: dict[str, Any] = {
        "method": _METHOD,
        "source": _SOURCE,
        "reason": "fills a null",
    }
    kwargs[field] = "   "
    with pytest.raises(WeakSignalError, match=message):
        set_reliability(
            registry,
            "alpha",
            reliability=0.5,
            reliability_low=0.3,
            reliability_high=0.7,
            **kwargs,
        )
