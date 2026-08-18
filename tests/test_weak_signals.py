from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from nfl_ats.weak_signals import (
    WEAK_SIGNAL_REGISTRY_VERSION,
    Registry,
    WeakSignalError,
    combination_report,
    load_registry,
    overlap_warnings,
    poolable_signals,
    pooled_effect,
    record_signal,
    registry_from_payload,
    save_registry,
    sign_test,
    signal_from_payload,
)


def _signal(**overrides: Any) -> dict[str, Any]:
    body = {
        "recorded_at": "2026-08-17",
        "description": "a small measured effect",
        "source": "docs/example.md",
        "effect": 0.10,
        "effect_units": "ats_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2009, 2017],
        "standard_error": 0.30,
    }
    body.update(overrides)
    return body


def _payload(**signals: dict[str, Any]) -> dict[str, Any]:
    return {"version": WEAK_SIGNAL_REGISTRY_VERSION, "notes": ["test ledger"], "signals": signals}


def test_round_trips_and_rejects_unknown_fields(tmp_path: Path) -> None:
    registry = registry_from_payload(_payload(alpha=_signal()))
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    assert load_registry(destination) == registry

    with pytest.raises(WeakSignalError, match="unknown top-level fields"):
        registry_from_payload({**_payload(), "budget": 1})
    with pytest.raises(WeakSignalError, match="unknown fields"):
        registry_from_payload(_payload(alpha=_signal(owner="ryan")))


def test_missing_registry_file_loads_as_empty(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "absent.json").signals == {}


def test_validation_rejects_bad_taxonomy_units_and_seasons() -> None:
    with pytest.raises(WeakSignalError, match="unknown classification"):
        registry_from_payload(_payload(alpha=_signal(classification="probably_fine")))
    with pytest.raises(WeakSignalError, match="unknown effect_units"):
        registry_from_payload(_payload(alpha=_signal(effect_units="vibes")))
    with pytest.raises(WeakSignalError, match="seasons out of order"):
        registry_from_payload(_payload(alpha=_signal(seasons=[2020, 2011])))
    with pytest.raises(WeakSignalError, match="non-finite effect"):
        registry_from_payload(_payload(alpha=_signal(effect=float("nan"))))


def test_recording_the_same_signal_twice_needs_an_explicit_replace() -> None:
    # A silently overwritten effect would let a second look at one signal
    # masquerade as independent new evidence.
    registry = Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    first = signal_from_payload("alpha", _signal())
    registry = record_signal(registry, first)
    with pytest.raises(WeakSignalError, match="already recorded"):
        record_signal(registry, first)
    updated = record_signal(
        registry, signal_from_payload("alpha", _signal(effect=0.2)), replace=True
    )
    assert updated.signals["alpha"].effect == pytest.approx(0.2)


def test_standard_error_is_recovered_from_an_interval() -> None:
    signal = signal_from_payload("alpha", _signal(standard_error=None, interval=[-0.423, 0.417]))
    # A 95% interval spans 2 * 1.96 standard errors.
    assert signal.resolved_standard_error() == pytest.approx((0.417 + 0.423) / 3.9199, rel=1e-3)


def test_sign_test_counts_directions_not_magnitudes() -> None:
    # Ten of eleven leaning one way is a real binomial event even though every
    # individual effect here is far too small for its own test to resolve.
    signals = [
        signal_from_payload(f"s{i}", _signal(effect=0.05 if i < 10 else -0.05)) for i in range(11)
    ]
    result = sign_test(signals)
    assert result["signals"] == 11
    assert result["favouring_candidate"] == 10
    assert result["p_value"] < 0.05
    assert "further than chance" in result["interpretation"]


def test_sign_test_calls_a_coin_flip_a_coin_flip() -> None:
    signals = [
        signal_from_payload(f"s{i}", _signal(effect=0.05 if i % 2 == 0 else -0.05))
        for i in range(10)
    ]
    result = sign_test(signals)
    assert result["p_value"] > 0.10
    assert "coin flip" in result["interpretation"]


def test_sign_test_on_an_empty_pile_is_not_a_finding() -> None:
    assert sign_test([])["p_value"] == 1.0


def test_pooling_sharpens_the_standard_error_toward_sqrt_k() -> None:
    # The mechanism: K identical signals pool to sqrt(K) times the precision.
    signals = [
        signal_from_payload(f"s{i}", _signal(effect=0.20, standard_error=0.40)) for i in range(4)
    ]
    pooled = pooled_effect(signals, method="fixed")
    assert pooled["pooled_effect"] == pytest.approx(0.20)
    assert pooled["standard_error"] == pytest.approx(0.40 / math.sqrt(4))
    assert pooled["sharpening_vs_best_single"] == pytest.approx(2.0)
    # Four is NOT enough, and pretending otherwise is the trap this registry
    # exists to avoid: each signal sits at 0.5 sigma, so four of them reach
    # only 1.0 sigma and the pooled interval still spans zero.
    assert not pooled["excludes_zero"]


def test_pooling_needs_roughly_sixteen_half_sigma_signals_to_resolve() -> None:
    # The honest arithmetic behind "keep collecting them". To carry a 0.5-sigma
    # effect past 1.96 sigma you need sqrt(K) >= 3.92, i.e. about sixteen
    # independent signals -- which is the actual price of this strategy.
    def pooled_sigma(count: int) -> float:
        signals = [
            signal_from_payload(f"s{i}", _signal(effect=0.20, standard_error=0.40))
            for i in range(count)
        ]
        result = pooled_effect(signals, method="fixed")
        return abs(result["pooled_effect"]) / result["standard_error"]

    assert pooled_sigma(4) == pytest.approx(1.0, rel=1e-6)
    assert pooled_sigma(9) == pytest.approx(1.5, rel=1e-6)
    assert pooled_sigma(16) == pytest.approx(2.0, rel=1e-6)

    sixteen = [
        signal_from_payload(f"s{i}", _signal(effect=0.20, standard_error=0.40)) for i in range(16)
    ]
    assert pooled_effect(sixteen, method="fixed")["excludes_zero"]


def test_random_effects_widens_when_signals_disagree() -> None:
    disagreeing = [
        signal_from_payload("a", _signal(effect=2.0, standard_error=0.2)),
        signal_from_payload("b", _signal(effect=-2.0, standard_error=0.2)),
    ]
    fixed = pooled_effect(disagreeing, method="fixed")
    random = pooled_effect(disagreeing, method="random")
    assert random["heterogeneity_tau_squared"] > 0.0
    assert random["standard_error"] > fixed["standard_error"]


def test_pooling_refuses_to_mix_units() -> None:
    mixed = [
        signal_from_payload("a", _signal(effect_units="ats_points")),
        signal_from_payload("b", _signal(effect_units="brier")),
    ]
    with pytest.raises(WeakSignalError, match="mixed effect units"):
        pooled_effect(mixed)


def test_pooling_without_any_uncertainty_reports_rather_than_guesses() -> None:
    bare = [signal_from_payload("a", _signal(standard_error=None))]
    result = pooled_effect(bare)
    assert result["signals"] == 0
    assert result["pooled_effect"] is None


def test_overlapping_seasons_are_flagged_as_shared_noise() -> None:
    overlapping = [
        signal_from_payload("a", _signal(seasons=[2009, 2015])),
        signal_from_payload("b", _signal(seasons=[2014, 2020])),
    ]
    warnings = overlap_warnings(overlapping)
    assert len(warnings) == 1
    assert "2014-2015" in warnings[0]

    disjoint = [
        signal_from_payload("a", _signal(seasons=[2009, 2012])),
        signal_from_payload("b", _signal(seasons=[2013, 2016])),
    ]
    assert overlap_warnings(disjoint) == []

    # Different leagues cannot share football.
    cross = [
        signal_from_payload("a", _signal(seasons=[2009, 2015], league="nfl")),
        signal_from_payload("b", _signal(seasons=[2009, 2015], league="cfb")),
    ]
    assert overlap_warnings(cross) == []


def test_only_genuinely_unresolved_signals_are_poolable() -> None:
    # Folding a refuted mechanism or a control-bounded null into the pool would
    # launder a known failure into a fresh-looking positive.
    registry = registry_from_payload(
        _payload(
            live=_signal(classification="unresolved_below_power"),
            refuted=_signal(classification="refuted_mechanism"),
            bounded=_signal(classification="bounded_by_control"),
        )
    )
    assert [s.name for s in poolable_signals(registry)] == ["live"]
    report = combination_report(registry)
    assert report["eligible"] == ["live"]
    assert report["excluded_with_reason"] == {
        "bounded": "bounded_by_control",
        "refuted": "refuted_mechanism",
    }


def test_combination_report_filters_by_league_and_records_seasons() -> None:
    registry = registry_from_payload(
        _payload(
            nfl_one=_signal(league="nfl", seasons=[2009, 2011]),
            cfb_one=_signal(league="cfb", seasons=[2006, 2008]),
        )
    )
    report = combination_report(registry, league="nfl")
    assert report["eligible"] == ["nfl_one"]
    assert report["seasons_touched_by_inputs"] == [2009, 2010, 2011]
    assert "predeclared" in report["guidance"]


def test_pooling_refuses_to_mix_leagues_when_none_is_chosen() -> None:
    """NFL and CFB are different populations, so an unscoped pool is an error.

    Latent until the registry held its first CFB signal: before that every
    eligible entry was NFL, so the omitted filter silently did the right thing.
    """

    registry = registry_from_payload(
        _payload(
            nfl_one=_signal(league="nfl", seasons=[2009, 2011]),
            cfb_one=_signal(league="cfb", seasons=[2006, 2008]),
        )
    )
    with pytest.raises(ValueError, match="Refusing to pool across leagues"):
        combination_report(registry)

    # Naming one league is still fine, and still pools only that league.
    assert combination_report(registry, league="cfb")["eligible"] == ["cfb_one"]


def test_live_ledger_validates_if_present() -> None:
    """The shipped ledger must always satisfy the schema, whatever it holds."""

    path = Path(__file__).resolve().parents[1] / "registry" / "weak_signals.json"
    if not path.is_file():
        pytest.skip("no weak-signal ledger committed yet")
    registry = load_registry(path)
    for name, signal in registry.signals.items():
        assert signal.classification in {
            "unresolved_below_power",
            "refuted_mechanism",
            "bounded_by_control",
        }, name
        assert signal.source, name
        assert math.isfinite(signal.effect), name
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == WEAK_SIGNAL_REGISTRY_VERSION


def test_weak_signal_direction_is_the_recorded_field() -> None:
    assert signal_from_payload("a", _signal(effect=0.01)).favours_candidate
    assert not signal_from_payload("a", _signal(effect=-0.01)).favours_candidate
