from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from nfl_ats.weak_signals import (
    NO_SPLIT_HALF_RELIABILITY_MAX,
    WEAK_SIGNAL_REGISTRY_VERSION,
    Registry,
    WeakSignalError,
    coherence_problems,
    combination_report,
    family_overlap_warnings,
    load_registry,
    overlap_warnings,
    poolable_signals,
    pooled_effect,
    record_signal,
    registry_from_payload,
    save_registry,
    sign_test,
    signal_family,
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


def test_plain_summary_and_category_round_trip(tmp_path: Path) -> None:
    registry = registry_from_payload(
        _payload(
            alpha=_signal(
                plain_summary="A short sentence a fan can read on its own.",
                category="onfield",
            )
        )
    )
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    loaded = load_registry(destination)
    assert loaded == registry
    assert loaded.signals["alpha"].plain_summary == "A short sentence a fan can read on its own."
    assert loaded.signals["alpha"].category == "onfield"


def test_plain_summary_and_category_are_optional_and_round_trip_as_none(tmp_path: Path) -> None:
    # The pre-existing registry (480 rows as of this change) carries neither
    # field, so both must stay optional and survive a save/load cycle as None.
    registry = registry_from_payload(_payload(alpha=_signal()))
    signal = registry.signals["alpha"]
    assert signal.plain_summary is None
    assert signal.category is None
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    assert load_registry(destination) == registry


def test_category_rejects_an_unknown_value() -> None:
    with pytest.raises(WeakSignalError, match="unknown category"):
        registry_from_payload(_payload(alpha=_signal(category="vibes")))


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


def test_signal_family_collapses_decompositions_of_one_construct() -> None:
    """Grades, era splits, window splits and battery cells are ONE family."""

    def family(name: str, *, league: str = "nfl") -> str:
        return signal_family(signal_from_payload(name, _signal(league=league)))

    # The opener grade is the same construct as the close grade.
    assert family("bias_battery_short_week_opener") == family("bias_battery_short_week")
    # Era splits partition their parent.
    assert (
        family("altitude_deficit_4000ft_era_2018_2025")
        == family("altitude_deficit_4000ft_era_2009_2017")
        == family("altitude_deficit_4000ft")
    )
    # Bare-year and pre/post window splits too.
    assert (
        family("body_clock_west_road_early_2009_2016")
        == family("body_clock_west_road_early_2017_2025")
        == family("body_clock_west_road_early")
    )
    assert family("bye_overval_home_edge_pre2011") == family("bye_overval_home_edge_post2011")
    # A battery marker collapses every cell of the screening battery.
    assert family("bias_battery_home_underdog") == "bias_battery"
    assert family("cfb_bias_battery_home_underdog", league="cfb") == "cfb_bias_battery"
    # An explicit declaration always wins over inference.
    declared = signal_from_payload("odd_name", _signal(family="declared_family"))
    assert signal_family(declared) == "declared_family"


def test_family_overlap_warnings_report_families_not_pairs() -> None:
    """The per-family report replaces 55k+ pairwise strings with one row per
    correlated decomposition group (registry_correlation_audit risk #3)."""

    members = [
        signal_from_payload("bias_battery_cell_a", _signal(seasons=[2009, 2025])),
        signal_from_payload("bias_battery_cell_b", _signal(seasons=[2009, 2025])),
        signal_from_payload("lone_signal", _signal(seasons=[2009, 2025])),
    ]
    report = family_overlap_warnings(members)
    assert report["families"] == 2
    assert report["families_with_internal_overlap"] == 1
    assert report["pairwise_within_family_pairs"] == 1
    # The two families share seasons, so cross-family correlation is counted.
    assert report["cross_family_shared_window_pairs"] == 1
    assert report["pairwise_overlap_pairs"] == 3
    entry = report["within_family"][0]
    assert entry["family"] == "bias_battery"
    assert entry["members"] == 2
    assert entry["shared_seasons"] == [2009, 2025]
    assert "correlated decompositions" in entry["warning"]

    # Disjoint windows produce no warnings at all.
    disjoint = [
        signal_from_payload("x_a", _signal(seasons=[2009, 2010])),
        signal_from_payload("x_b", _signal(seasons=[2011, 2012])),
    ]
    empty = family_overlap_warnings(disjoint)
    assert empty["families_with_internal_overlap"] == 0
    assert empty["cross_family_shared_window_pairs"] == 0
    assert empty["within_family"] == []


def test_combination_report_carries_per_family_overlap_output() -> None:
    registry = registry_from_payload(
        _payload(
            widget=_signal(seasons=[2009, 2017]),
            widget_opener=_signal(seasons=[2009, 2017]),
            unrelated=_signal(seasons=[2009, 2017]),
        )
    )
    report = combination_report(registry, league="nfl")
    structured = report["overlap_warnings"]
    assert structured["families_with_internal_overlap"] == 1
    assert structured["within_family"][0]["family"] == "widget"
    assert report["overlap_pairwise_count"] >= 1


def test_effect_outside_interval_is_refused_at_record_time() -> None:
    """A point estimate outside its own interval is a recording contradiction.

    Enforced only at RECORD time: historical rows are never rewritten, so the
    pre-existing ledger entry predating this check must keep loading -- it is
    surfaced by ``coherence_problems`` instead.
    """

    contradictory = signal_from_payload("alpha", _signal(effect=0.05, interval=[-0.03, 0.03]))
    base = Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    with pytest.raises(WeakSignalError, match="outside its own interval"):
        record_signal(base, contradictory)
    # And the soft load-time counterpart reports it without raising.
    problems = coherence_problems([contradictory])
    assert problems and problems[0]["signal"] == "alpha"


def test_bounded_by_control_needs_quantitative_evidence() -> None:
    with pytest.raises(WeakSignalError, match="no quantitative evidence"):
        signal_from_payload(
            "alpha",
            _signal(
                classification="bounded_by_control",
                closing_ground="positive_control_bound",
                classification_evidence="the control saw nothing",
                standard_error=None,
            ),
        )
    # With a number attached it loads fine.
    ok = signal_from_payload(
        "alpha",
        _signal(
            classification="bounded_by_control",
            closing_ground="positive_control_bound",
            classification_evidence="the control saw nothing at P+ 0.984",
            probability_positive=0.984,
            standard_error=None,
        ),
    )
    assert ok.classification == "bounded_by_control"


def test_no_reliability_closure_cannot_cite_a_reliable_trait() -> None:
    reliable = _signal(
        classification="refuted_mechanism",
        closing_ground="no_split_half_reliability",
        classification_evidence="trait persists across halves",
        reliability=0.719,
    )
    with pytest.raises(WeakSignalError, match="ceiling this ground admits"):
        signal_from_payload("alpha", reliable)
    assert NO_SPLIT_HALF_RELIABILITY_MAX == 0.10


def test_only_genuinely_unresolved_signals_are_poolable() -> None:
    # Folding a refuted mechanism or a control-bounded null into the pool would
    # launder a known failure into a fresh-looking positive.
    registry = registry_from_payload(
        _payload(
            live=_signal(classification="unresolved_below_power"),
            refuted=_signal(
                classification="refuted_mechanism",
                closing_ground="wrong_sign_resolved",
                classification_evidence="whole interval below zero",
                effect=-0.6,
                standard_error=None,
                interval=[-0.9, -0.3],
            ),
            bounded=_signal(
                classification="bounded_by_control",
                closing_ground="positive_control_bound",
                classification_evidence="deliberate-leak control detected at P+ 0.984",
                probability_positive=0.984,
            ),
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


def test_terminal_verdict_requires_an_admissible_closing_ground() -> None:
    """AGENTS.md, binding: an interval containing zero never closes a line.

    A terminal classification with no stated ground is exactly how that
    violation was written in practice -- "failed, CI contains 0" -- so the
    ledger now refuses it with the rule quoted in the error.
    """

    with pytest.raises(WeakSignalError, match="no admissible closing_ground"):
        signal_from_payload(
            "alpha",
            _signal(
                classification="refuted_mechanism",
                classification_evidence="interval contains zero",
            ),
        )
    with pytest.raises(WeakSignalError, match="no admissible closing_ground"):
        signal_from_payload(
            "alpha",
            _signal(
                classification="bounded_by_control",
                closing_ground="wrong_sign_resolved",
                classification_evidence="mismatched ground for this classification",
            ),
        )


def test_wrong_sign_must_be_resolved_not_a_lean() -> None:
    crossing = _signal(
        classification="refuted_mechanism",
        closing_ground="wrong_sign_resolved",
        classification_evidence="negative point estimate",
        standard_error=None,
        interval=[-0.5, 0.2],
    )
    with pytest.raises(WeakSignalError, match="is not entirely"):
        signal_from_payload("alpha", crossing)

    resolved = signal_from_payload(
        "alpha",
        _signal(
            effect=-0.6,
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval on the wrong side of zero",
            standard_error=None,
            interval=[-0.9, -0.3],
        ),
    )
    assert resolved.closing_ground == "wrong_sign_resolved"


def test_no_reliability_ground_needs_the_measurement_it_cites() -> None:
    without_measurement = _signal(
        classification="refuted_mechanism",
        closing_ground="no_split_half_reliability",
        classification_evidence="trait does not persist",
    )
    with pytest.raises(WeakSignalError, match="no reliability measurement"):
        signal_from_payload("alpha", without_measurement)

    recorded = signal_from_payload("alpha", _signal(**{**without_measurement, "reliability": 0.02}))
    assert recorded.reliability == pytest.approx(0.02)


def test_control_bounded_closure_must_cite_its_evidence() -> None:
    with pytest.raises(WeakSignalError, match="classification_evidence is empty"):
        signal_from_payload(
            "alpha",
            _signal(
                classification="bounded_by_control",
                closing_ground="positive_control_bound",
                classification_evidence="   ",
            ),
        )


def test_unresolved_signals_cannot_carry_a_closing_ground() -> None:
    with pytest.raises(WeakSignalError, match="cannot carry a closing_ground"):
        signal_from_payload("alpha", _signal(closing_ground="wrong_sign_resolved"))


def test_record_signal_enforces_closure_grounds_directly() -> None:
    """The CLI builds WeakSignal directly, so record time must enforce too."""

    from nfl_ats.weak_signals import WeakSignal

    registry = Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    bad = WeakSignal(
        name="alpha",
        recorded_at="2026-08-18",
        description="terminal verdict with no admissible ground",
        source="docs/example.md",
        effect=-0.1,
        effect_units="ats_points",
        classification="refuted_mechanism",
        league="nfl",
        seasons=(2009, 2017),
        classification_evidence="interval contains zero",
    )
    with pytest.raises(WeakSignalError, match="no admissible closing_ground"):
        record_signal(registry, bad)


def test_closing_ground_round_trips(tmp_path: Path) -> None:
    payload = _payload(
        alpha=_signal(
            effect=-0.6,
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            classification_evidence="whole interval below zero",
            standard_error=None,
            interval=[-0.9, -0.3],
        )
    )
    registry = registry_from_payload(payload)
    destination = tmp_path / "weak_signals.json"
    save_registry(registry, destination)
    reloaded = load_registry(destination)
    assert reloaded.signals["alpha"].closing_ground == "wrong_sign_resolved"
